"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol,         │
  │               end_symbol, device) → torch.Tensor [1, out_len]       │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device,           │
  │               max_len) → float  (corpus-level BLEU, 0–100)          │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional

import wandb
from tqdm import tqdm

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    PAD positions are masked out and contribute zero to the loss.
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : [batch * tgt_len, vocab_size]
            target : [batch * tgt_len]
        Returns:
            Scalar loss.
        """
        log_probs = F.log_softmax(logits, dim=-1)

        # Smooth target distribution: eps / (V-1) everywhere …
        true_dist = logits.new_full(log_probs.shape, self.smoothing / (self.vocab_size - 1))
        # … then place (1-eps) on the correct token
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        # Zero out the PAD column and PAD rows
        true_dist[:, self.pad_idx] = 0.0
        pad_rows = (target == self.pad_idx)
        true_dist[pad_rows] = 0.0

        loss = -(true_dist * log_probs).sum(dim=-1)
        n_tokens = (~pad_rows).sum().clamp(min=1)
        return loss.sum() / n_tokens


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    One epoch of training or evaluation.

    Returns:
        avg_loss : Average loss over the epoch.
    """
    model.train() if is_train else model.eval()
    pad_idx     = 1   # <pad> index agreed upon in Vocabulary
    total_loss  = 0.0
    total_tokens = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(data_iter, desc=f"{'Train' if is_train else 'Val'} epoch {epoch_num}")
        for src, tgt in pbar:
            src = src.to(device)   # [batch, src_len]
            tgt = tgt.to(device)   # [batch, tgt_len]

            # Teacher forcing: decoder input excludes last token (<eos>),
            # targets exclude first token (<sos>)
            tgt_input  = tgt[:, :-1]   # [batch, tgt_len-1]
            tgt_output = tgt[:, 1:]    # [batch, tgt_len-1]

            src_mask = make_src_mask(src, pad_idx=pad_idx)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx=pad_idx)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            # logits: [batch, tgt_len-1, vocab_size]

            batch_size, tgt_len, vocab_size = logits.shape
            loss = loss_fn(
                logits.reshape(-1, vocab_size),
                tgt_output.reshape(-1),
            )

            n_tokens = (tgt_output != pad_idx).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(total_tokens, 1)
    if wandb.run is not None:
        tag = "train" if is_train else "val"
        wandb.log({f"{tag}/loss": avg_loss, "epoch": epoch_num})

    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Autoregressive greedy decoding.

    Args:
        model        : Trained Transformer (eval mode expected).
        src          : [1, src_len]
        src_mask     : [1, 1, 1, src_len]
        max_len      : Maximum tokens to generate.
        start_symbol : <sos> index.
        end_symbol   : <eos> index.
        device       : Device string.

    Returns:
        ys : [1, out_len]  — includes start_symbol; stops at end_symbol.
    """
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=1)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Corpus-level BLEU score on the test set.

    Returns:
        bleu_score : float in range [0, 100].
    """
    model.eval()
    pad_idx = tgt_vocab.stoi['<pad>']
    sos_idx = tgt_vocab.stoi['<sos>']
    eos_idx = tgt_vocab.stoi['<eos>']

    predictions = []
    references  = []

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU eval"):
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                src_i    = src[i:i+1]
                mask_i   = make_src_mask(src_i, pad_idx=1)
                output   = greedy_decode(
                    model, src_i, mask_i, max_len, sos_idx, eos_idx, device
                )

                pred_tokens = []
                for idx in output[0].tolist():
                    if idx == eos_idx:
                        break
                    if idx not in (sos_idx, pad_idx):
                        pred_tokens.append(tgt_vocab.lookup_token(idx))

                ref_tokens = []
                for idx in tgt[i].tolist():
                    if idx == eos_idx:
                        break
                    if idx not in (sos_idx, pad_idx):
                        ref_tokens.append(tgt_vocab.lookup_token(idx))

                predictions.append(' '.join(pred_tokens))
                references.append(' '.join(ref_tokens))

    # Compute corpus BLEU using sacrebleu (preferred) or nltk fallback
    try:
        import sacrebleu as sb
        bleu = sb.corpus_bleu(predictions, [references])
        return bleu.score
    except ImportError:
        pass

    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        tokenised_preds = [p.split() for p in predictions]
        tokenised_refs  = [[r.split()] for r in references]
        score = corpus_bleu(
            tokenised_refs,
            tokenised_preds,
            smoothing_function=SmoothingFunction().method1,
        )
        return score * 100.0
    except ImportError:
        pass

    # Last resort: simple unigram overlap
    total_correct = total_pred = 0
    for pred, ref in zip(predictions, references):
        p_toks = pred.split()
        r_toks = set(ref.split())
        total_correct += sum(1 for t in p_toks if t in r_toks)
        total_pred    += len(p_toks)
    return (total_correct / max(total_pred, 1)) * 100.0


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """Save model + optimizer + scheduler state."""
    torch.save(
        {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'model_config': {
                'src_vocab_size': model.src_embedding.num_embeddings,
                'tgt_vocab_size': model.tgt_embedding.num_embeddings,
                'd_model':        model.d_model,
                'N':              len(model.encoder.layers),
                'num_heads':      model.encoder.layers[0].self_attn.num_heads,
                'd_ff':           model.encoder.layers[0].ffn.linear1.out_features,
                'dropout':        model.encoder.layers[0].dropout.p,
            },
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """Restore model (and optionally optimizer/scheduler) state from disk."""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and ckpt.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt.get('epoch', 0)


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """Full training experiment with W&B logging."""
    import torch.optim as optim
    from dataset import build_datasets
    from lr_scheduler import NoamScheduler

    # ── Config ────────────────────────────────────────────────────────
    config = dict(
        d_model      = 256,
        N            = 3,
        num_heads    = 8,
        d_ff         = 512,
        dropout      = 0.1,
        batch_size   = 128,
        num_epochs   = 20,
        warmup_steps = 4000,
        smoothing    = 0.1,
        min_freq     = 2,
    )
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # ── W&B ───────────────────────────────────────────────────────────
    wandb.init(project="da6401-a3", config=config)
    cfg = wandb.config

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab, spacy_de = \
        build_datasets(batch_size=cfg.batch_size, min_freq=cfg.min_freq)

    # ── Model ─────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model        = cfg.d_model,
        N              = cfg.N,
        num_heads      = cfg.num_heads,
        d_ff           = cfg.d_ff,
        dropout        = cfg.dropout,
    ).to(device)
    model.set_vocabs(src_vocab, tgt_vocab, spacy_de)
    wandb.watch(model, log='all', log_freq=100)

    # ── Optimizer, scheduler, loss ────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)
    loss_fn   = LabelSmoothingLoss(len(tgt_vocab), pad_idx=1, smoothing=cfg.smoothing)

    # ── Training loop ─────────────────────────────────────────────────
    best_val_loss = float('inf')
    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
        )
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path="best_checkpoint.pt")
            print(f"  → Saved best checkpoint (val_loss={val_loss:.4f})")

    # ── Final BLEU on test set ─────────────────────────────────────────
    load_checkpoint("best_checkpoint.pt", model)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"Test BLEU: {bleu:.2f}")
    wandb.log({'test_bleu': bleu})
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
