"""
experiments.py — W&B Ablation Experiments for DA6401 Assignment 3

Runs all 5 report experiments sequentially:
  2.1  Noam vs Fixed LR
  2.2  Scaling factor ablation (with / without 1/√dk)
  2.3  Attention rollout & head specialisation (uses saved baseline)
  2.4  Sinusoidal PE vs Learned Embeddings
  2.5  Label smoothing ε=0.1 vs ε=0.0
"""

import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from dataset import build_datasets
from model import (
    Transformer, MultiHeadAttention, PositionalEncoding,
    make_src_mask, make_tgt_mask,
    scaled_dot_product_attention,
    PositionwiseFeedForward, EncoderLayer, DecoderLayer,
    Encoder, Decoder,
)
from lr_scheduler import NoamScheduler
from train import (
    LabelSmoothingLoss, run_epoch, evaluate_bleu,
    greedy_decode, save_checkpoint, load_checkpoint,
)

WANDB_PROJECT = "da6401-a3"
DEVICE = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
    else 'cpu'
)

# ── Shared small config for ablations (fast runs) ─────────────────────
BASE_CFG = dict(
    d_model=256, N=3, num_heads=8, d_ff=512,
    dropout=0.1, batch_size=128, num_epochs=15,
    warmup_steps=4000, smoothing=0.1, min_freq=2,
)


def _build_data():
    return build_datasets(batch_size=BASE_CFG['batch_size'], min_freq=BASE_CFG['min_freq'])


def _train(model, train_loader, val_loader, loss_fn, optimizer, scheduler,
           num_epochs, run_name):
    best_val = float('inf')
    for epoch in range(num_epochs):
        run_epoch(train_loader, model, loss_fn, optimizer, scheduler,
                  epoch_num=epoch, is_train=True,  device=DEVICE)
        val_loss = run_epoch(val_loader, model, loss_fn, None, None,
                             epoch_num=epoch, is_train=False, device=DEVICE)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch,
                            path=f"{run_name}_best.pt")


# ══════════════════════════════════════════════════════════════════════
# 2.1  Noam vs Fixed LR
# ══════════════════════════════════════════════════════════════════════

def exp_noam_vs_fixed():
    train_loader, val_loader, _, src_vocab, tgt_vocab, spacy_de = _build_data()

    for use_noam in [True, False]:
        lr_name = "noam" if use_noam else "fixed_lr_1e-4"
        wandb.init(project=WANDB_PROJECT, name=f"2.1_{lr_name}",
                   config={**BASE_CFG, 'lr_schedule': lr_name}, reinit=True)

        model = Transformer(len(src_vocab), len(tgt_vocab), **{
            k: BASE_CFG[k] for k in ('d_model','N','num_heads','d_ff','dropout')
        }).to(DEVICE)
        model.set_vocabs(src_vocab, tgt_vocab, spacy_de)

        if use_noam:
            optimizer = optim.Adam(model.parameters(), lr=1.0,
                                   betas=(0.9, 0.98), eps=1e-9)
            scheduler = NoamScheduler(optimizer, BASE_CFG['d_model'],
                                      BASE_CFG['warmup_steps'])
        else:
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            scheduler = None

        loss_fn = LabelSmoothingLoss(len(tgt_vocab), pad_idx=1,
                                     smoothing=BASE_CFG['smoothing'])
        _train(model, train_loader, val_loader, loss_fn, optimizer, scheduler,
               BASE_CFG['num_epochs'], f"exp21_{lr_name}")
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════
# 2.2  Scaling factor 1/√dk ablation
# ══════════════════════════════════════════════════════════════════════

class UnscaledDotProductAttention(nn.Module):
    """MHA without the 1/√dk scaling — for ablation 2.2."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        self.q_grad_norms = []
        self.k_grad_norms = []

    def _attn_unscaled(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1))   # no √dk
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        w = F.softmax(scores, dim=-1)
        w = torch.nan_to_num(w, nan=0.0)
        return torch.matmul(w, V), w

    def forward(self, query, key, value, mask=None):
        batch = query.size(0)
        Q = self.W_q(query).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key  ).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        x, _ = self._attn_unscaled(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        return self.W_o(x)


def _build_unscaled_transformer(src_vs, tgt_vs, cfg):
    """Transformer whose MHA uses unscaled dot-product attention."""
    model = Transformer(src_vs, tgt_vs,
                        d_model=cfg['d_model'], N=cfg['N'],
                        num_heads=cfg['num_heads'], d_ff=cfg['d_ff'],
                        dropout=cfg['dropout'])
    # Swap every MHA module to the unscaled version
    def replace_mha(mod):
        for name, child in list(mod.named_children()):
            if isinstance(child, MultiHeadAttention):
                new = UnscaledDotProductAttention(
                    child.d_model, child.num_heads, child.dropout.p)
                setattr(mod, name, new)
            else:
                replace_mha(child)
    replace_mha(model)
    return model


def exp_scaling_factor():
    train_loader, val_loader, _, src_vocab, tgt_vocab, spacy_de = _build_data()

    for scaled in [True, False]:
        run_name = "with_scaling" if scaled else "no_scaling"
        wandb.init(project=WANDB_PROJECT, name=f"2.2_{run_name}",
                   config={**BASE_CFG, 'scaled': scaled}, reinit=True)

        if scaled:
            model = Transformer(len(src_vocab), len(tgt_vocab), **{
                k: BASE_CFG[k] for k in ('d_model','N','num_heads','d_ff','dropout')
            })
        else:
            model = _build_unscaled_transformer(len(src_vocab), len(tgt_vocab), BASE_CFG)

        model = model.to(DEVICE)
        model.set_vocabs(src_vocab, tgt_vocab, spacy_de)

        optimizer = optim.Adam(model.parameters(), lr=1.0,
                               betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, BASE_CFG['d_model'],
                                  BASE_CFG['warmup_steps'])
        loss_fn = LabelSmoothingLoss(len(tgt_vocab), pad_idx=1,
                                     smoothing=BASE_CFG['smoothing'])

        # Log gradient norms of Q & K during first 1000 steps
        step = [0]

        def log_grad_hook(name):
            def hook(grad):
                if step[0] < 1000:
                    wandb.log({f"grad_norm/{name}": grad.norm().item(),
                               "step": step[0]})
                    step[0] += 1
            return hook

        for n, p in model.named_parameters():
            if 'W_q.weight' in n or 'W_k.weight' in n:
                p.register_hook(log_grad_hook(n))

        _train(model, train_loader, val_loader, loss_fn, optimizer, scheduler,
               BASE_CFG['num_epochs'], f"exp22_{run_name}")
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════
# 2.3  Attention Rollout & Head Specialisation
# ══════════════════════════════════════════════════════════════════════

def _get_attention_weights(model, src, src_mask):
    """Extract per-head attention weights from every encoder layer."""
    all_weights = []

    hooks = []
    for layer in model.encoder.layers:
        storage = {}

        # Patch the MHA forward to capture attention weights
        original_forward = layer.self_attn.forward

        def make_hook(store, orig):
            def patched_forward(query, key, value, mask=None):
                batch = query.size(0)
                d_k = layer.self_attn.d_k
                nh  = layer.self_attn.num_heads
                Q = layer.self_attn.W_q(query).view(batch,-1,nh,d_k).transpose(1,2)
                K = layer.self_attn.W_k(key  ).view(batch,-1,nh,d_k).transpose(1,2)
                V = layer.self_attn.W_v(value).view(batch,-1,nh,d_k).transpose(1,2)
                scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(d_k)
                if mask is not None:
                    scores = scores.masked_fill(mask, float('-inf'))
                w = F.softmax(scores, dim=-1)
                w = torch.nan_to_num(w, nan=0.0)
                store['weights'] = w.detach().cpu()
                out = torch.matmul(w, V)
                out = out.transpose(1,2).contiguous().view(batch,-1,layer.self_attn.d_model)
                return layer.self_attn.W_o(out)
            return patched_forward

        layer.self_attn.forward = make_hook(storage, original_forward)
        all_weights.append(storage)
        hooks.append((layer.self_attn, original_forward))

    model.eval()
    with torch.no_grad():
        model.encode(src, src_mask)

    # Restore original forwards
    for (mha, orig) in hooks:
        mha.forward = orig

    return [s['weights'] for s in all_weights]   # list[tensor(1,heads,seq,seq)]


def exp_attention_rollout(checkpoint_path="best_checkpoint.pt"):
    import matplotlib.pyplot as plt
    import matplotlib

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab, spacy_de = _build_data()

    model = Transformer(len(src_vocab), len(tgt_vocab), **{
        k: BASE_CFG[k] for k in ('d_model','N','num_heads','d_ff','dropout')
    }).to(DEVICE)
    model.set_vocabs(src_vocab, tgt_vocab, spacy_de)

    try:
        load_checkpoint(checkpoint_path, model)
        print(f"Loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"No checkpoint found at {checkpoint_path}, using random weights")

    wandb.init(project=WANDB_PROJECT, name="2.3_attention_rollout",
               config=BASE_CFG, reinit=True)

    # Pick a sample sentence from the test set
    for src_batch, tgt_batch in test_loader:
        sample_src = src_batch[0:1].to(DEVICE)
        sample_tgt = tgt_batch[0:1]
        break

    src_mask = make_src_mask(sample_src, pad_idx=1)
    # Decode source tokens for axis labels
    src_tokens = [src_vocab.lookup_token(i.item()) for i in sample_src[0]
                  if i.item() not in (1, 2, 3)]   # skip pad/sos/eos

    # Extract attention from last encoder layer
    all_layer_weights = _get_attention_weights(model, sample_src, src_mask)
    last_layer_weights = all_layer_weights[-1][0]   # [heads, seq, seq]

    num_heads  = last_layer_weights.shape[0]
    seq_len    = last_layer_weights.shape[1]
    labels     = [src_vocab.lookup_token(i.item()) for i in sample_src[0][:seq_len]]

    fig, axes = plt.subplots(2, num_heads // 2, figsize=(4 * num_heads // 2, 8))
    axes = axes.flatten()

    for h in range(num_heads):
        ax = axes[h]
        data = last_layer_weights[h].numpy()
        im = ax.imshow(data, cmap='viridis', vmin=0, vmax=data.max())
        ax.set_title(f"Head {h+1}")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        plt.colorbar(im, ax=ax)

    plt.suptitle("Last Encoder Layer — Per-Head Attention Maps", fontsize=12)
    plt.tight_layout()
    wandb.log({"attention_heatmaps": wandb.Image(fig)})
    plt.savefig("attention_heads.png", dpi=120, bbox_inches='tight')
    plt.close()

    print("Attention heatmaps logged to W&B")
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
# 2.4  Sinusoidal PE vs Learned Embeddings
# ══════════════════════════════════════════════════════════════════════

class LearnedPositionalEncoding(nn.Module):
    """Drop-in replacement for PositionalEncoding using learned embeddings."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.embedding(positions)
        return self.dropout(x)


def _transformer_with_learned_pe(src_vs, tgt_vs, cfg):
    model = Transformer(src_vs, tgt_vs,
                        d_model=cfg['d_model'], N=cfg['N'],
                        num_heads=cfg['num_heads'], d_ff=cfg['d_ff'],
                        dropout=cfg['dropout'])
    model.pos_enc = LearnedPositionalEncoding(cfg['d_model'], cfg['dropout'])
    return model


def exp_pe_vs_learned():
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab, spacy_de = _build_data()

    pad_idx = tgt_vocab.stoi['<pad>']

    for use_sinusoidal in [True, False]:
        pe_name = "sinusoidal" if use_sinusoidal else "learned"
        wandb.init(project=WANDB_PROJECT, name=f"2.4_{pe_name}_pe",
                   config={**BASE_CFG, 'pe_type': pe_name}, reinit=True)

        if use_sinusoidal:
            model = Transformer(len(src_vocab), len(tgt_vocab), **{
                k: BASE_CFG[k] for k in ('d_model','N','num_heads','d_ff','dropout')
            })
        else:
            model = _transformer_with_learned_pe(len(src_vocab), len(tgt_vocab), BASE_CFG)

        model = model.to(DEVICE)
        model.set_vocabs(src_vocab, tgt_vocab, spacy_de)

        optimizer = optim.Adam(model.parameters(), lr=1.0,
                               betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, BASE_CFG['d_model'],
                                  BASE_CFG['warmup_steps'])
        loss_fn = LabelSmoothingLoss(len(tgt_vocab), pad_idx=1,
                                     smoothing=BASE_CFG['smoothing'])

        for epoch in range(BASE_CFG['num_epochs']):
            run_epoch(train_loader, model, loss_fn, optimizer, scheduler,
                      epoch_num=epoch, is_train=True, device=DEVICE)
            run_epoch(val_loader, model, loss_fn, None, None,
                      epoch_num=epoch, is_train=False, device=DEVICE)

            # BLEU every 5 epochs
            if (epoch + 1) % 5 == 0:
                bleu = evaluate_bleu(model, val_loader, tgt_vocab, device=DEVICE)
                wandb.log({"val_bleu": bleu, "epoch": epoch})
                print(f"[{pe_name}] epoch {epoch} val BLEU: {bleu:.2f}")

        wandb.finish()


# ══════════════════════════════════════════════════════════════════════
# 2.5  Label Smoothing ε=0.1 vs ε=0.0
# ══════════════════════════════════════════════════════════════════════

def exp_label_smoothing():
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab, spacy_de = _build_data()

    for eps in [0.1, 0.0]:
        run_name = f"smoothing_{eps}"
        wandb.init(project=WANDB_PROJECT, name=f"2.5_{run_name}",
                   config={**BASE_CFG, 'smoothing': eps}, reinit=True)

        model = Transformer(len(src_vocab), len(tgt_vocab), **{
            k: BASE_CFG[k] for k in ('d_model','N','num_heads','d_ff','dropout')
        }).to(DEVICE)
        model.set_vocabs(src_vocab, tgt_vocab, spacy_de)

        optimizer = optim.Adam(model.parameters(), lr=1.0,
                               betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, BASE_CFG['d_model'],
                                  BASE_CFG['warmup_steps'])
        loss_fn = LabelSmoothingLoss(len(tgt_vocab), pad_idx=1, smoothing=eps)

        pad_idx = tgt_vocab.stoi['<pad>']

        for epoch in range(BASE_CFG['num_epochs']):
            model.train()
            for src, tgt in train_loader:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                sm = make_src_mask(src, pad_idx=1)
                tm = make_tgt_mask(tgt_in, pad_idx=1)
                logits = model(src, tgt_in, sm, tm)

                B, T, V = logits.shape
                loss = loss_fn(logits.reshape(-1, V), tgt_out.reshape(-1))

                # Log prediction confidence (softmax prob of correct token)
                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)
                    mask  = (tgt_out != pad_idx)
                    correct_probs = probs.gather(
                        2, tgt_out.unsqueeze(-1)).squeeze(-1)
                    confidence = correct_probs[mask].mean().item()
                wandb.log({"confidence": confidence, "train_loss": loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            val_loss = run_epoch(val_loader, model, loss_fn, None, None,
                                 epoch_num=epoch, is_train=False, device=DEVICE)
            print(f"[eps={eps}] epoch {epoch}  val_loss={val_loss:.4f}")

        wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='all',
        choices=['all', '2.1', '2.2', '2.3', '2.4', '2.5'])
    parser.add_argument('--checkpoint', type=str, default='best_checkpoint.pt')
    args = parser.parse_args()

    print(f"Running on: {DEVICE}")

    if args.exp in ('all', '2.1'):
        print("\n=== Experiment 2.1: Noam vs Fixed LR ===")
        exp_noam_vs_fixed()

    if args.exp in ('all', '2.2'):
        print("\n=== Experiment 2.2: Scaling Factor Ablation ===")
        exp_scaling_factor()

    if args.exp in ('all', '2.3'):
        print("\n=== Experiment 2.3: Attention Rollout ===")
        exp_attention_rollout(args.checkpoint)

    if args.exp in ('all', '2.4'):
        print("\n=== Experiment 2.4: Sinusoidal PE vs Learned ===")
        exp_pe_vs_learned()

    if args.exp in ('all', '2.5'):
        print("\n=== Experiment 2.5: Label Smoothing ===")
        exp_label_smoothing()

    print("\nAll experiments complete.")
