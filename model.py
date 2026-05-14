"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  STANDALONE ATTENTION FUNCTION
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V

    Args:
        Q    : (..., seq_q, d_k)
        K    : (..., seq_k, d_k)
        V    : (..., seq_k, d_v)
        mask : Optional bool tensor broadcastable to (..., seq_q, seq_k).
               True → position is masked out (set to -inf before softmax).

    Returns:
        output  : (..., seq_q, d_v)
        attn_w  : (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_w = F.softmax(scores, dim=-1)
    # Rows that were fully masked become NaN; replace with 0 (no attention)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
#  Exposed at module level so they can be tested independently and
#  reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Padding mask for the encoder.

    Returns: BoolTensor [batch, 1, 1, src_len]
             True  → PAD token (masked out)
             False → real token
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Combined padding + causal mask for the decoder.

    Returns: BoolTensor [batch, 1, tgt_len, tgt_len]
             True → masked out (PAD or future position)
    """
    tgt_len = tgt.size(1)
    # Padding: [batch, 1, 1, tgt_len]
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    # Causal (look-ahead): upper triangle is True → [1, 1, tgt_len, tgt_len]
    causal = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)
    return pad_mask | causal  # broadcasts to [batch, 1, tgt_len, tgt_len]


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (§3.2.2 of the paper).
    NOT using torch.nn.MultiheadAttention.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : [batch, seq_q, d_model]
            key   : [batch, seq_k, d_model]
            value : [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to [batch, heads, seq_q, seq_k]

        Returns:
            [batch, seq_q, d_model]
        """
        batch = query.size(0)

        # Project and reshape to [batch, heads, seq, d_k]
        Q = self.W_q(query).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key  ).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)

        x, _ = scaled_dot_product_attention(Q, K, V, mask)

        # Merge heads: [batch, seq_q, d_model]
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        return self.W_o(x)


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (§3.5).
    PE is registered as a buffer (not a trainable parameter).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # even dims → sin
        pe[:, 1::2] = torch.cos(position * div_term)   # odd  dims → cos
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]  (x + PE[:, :seq_len, :])
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂  (§3.3)"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout  = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  (Post-LayerNorm)
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    x → norm(x + Self-Attn(x)) → norm(x + FFN(x))
    Post-LN: LayerNorm applied after the residual connection,
    matching the original "Attention Is All You Need" paper.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER  (Post-LayerNorm)
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    x → norm(x + MaskedSelfAttn(x))
      → norm(x + CrossAttn(x, memory))
      → norm(x + FFN(x))
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """N identical EncoderLayers + final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """N identical DecoderLayers + final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.
    """

    def __init__(
        self,
        src_vocab_size: int   = 7853,   # Multi30k DE vocab (min_freq=2)
        tgt_vocab_size: int   = 5893,   # Multi30k EN vocab (min_freq=2)
        d_model:        int   = 256,
        N:              int   = 3,
        num_heads:      int   = 8,
        d_ff:           int   = 512,
        dropout:        float = 0.1,
        checkpoint_path: str  = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Embeddings + positional encoding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc       = PositionalEncoding(d_model, dropout)

        # Encoder and Decoder stacks
        enc_layer    = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer    = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        # Final linear projection to vocabulary
        self.projection = nn.Linear(d_model, tgt_vocab_size)

        # Vocabulary references for Transformer.infer (set via set_vocabs or auto-built)
        self.src_vocab = None
        self.tgt_vocab = None
        self.spacy_de  = None

        self._init_weights()

        # Resolve checkpoint path: explicit arg → best_checkpoint.pt (download if needed)
        _ckpt = checkpoint_path if checkpoint_path is not None else "best_checkpoint.pt"
        if not os.path.exists(_ckpt):
            DRIVE_ID = "1Ag7AfbEfcWUdaU6xa36sHi8v8T8vzXUb"
            gdown.download(id=DRIVE_ID, output=_ckpt, quiet=False)
        if os.path.exists(_ckpt):
            ckpt = torch.load(_ckpt, map_location='cpu')
            self.load_state_dict(ckpt['model_state_dict'])
            # Restore vocab dicts if they were saved in the checkpoint
            if 'src_vocab_stoi' in ckpt:
                self._restore_vocab_from_ckpt(ckpt)

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def set_vocabs(self, src_vocab, tgt_vocab, spacy_de=None) -> None:
        """Attach vocabulary objects needed by Transformer.infer."""
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.spacy_de  = spacy_de

    def _restore_vocab_from_ckpt(self, data: dict) -> None:
        """Reconstruct lightweight vocab objects from a stoi/itos dict."""
        class _Vocab:
            def __init__(self, stoi, itos):
                self.stoi = stoi
                self.itos = itos
            def lookup_token(self, idx):
                return self.itos[idx]
        sv = _Vocab(data['src_vocab_stoi'], data['src_vocab_itos'])
        tv = _Vocab(data['tgt_vocab_stoi'], data['tgt_vocab_itos'])
        self.set_vocabs(sv, tv, spacy_de=None)

    def _try_load_vocab(self) -> bool:
        """
        Try to load vocab from vocab.pt (shipped in submission zip) or from
        the checkpoint. Returns True if vocab was successfully loaded.
        """
        # 1. vocab.pt alongside the code (most reliable — no Drive/spaCy needed)
        for candidate in ['vocab.pt', os.path.join(os.path.dirname(__file__), 'vocab.pt')]:
            if os.path.exists(candidate):
                data = torch.load(candidate, map_location='cpu')
                self._restore_vocab_from_ckpt(data)
                return True
        # 2. vocab embedded in checkpoint
        for ckpt_path in ['best_checkpoint.pt',
                          os.path.join(os.path.dirname(__file__), 'best_checkpoint.pt')]:
            if os.path.exists(ckpt_path):
                data = torch.load(ckpt_path, map_location='cpu')
                if 'src_vocab_stoi' in data:
                    self._restore_vocab_from_ckpt(data)
                    return True
        return False

    # ── AUTOGRADER HOOKS ───────────────────────────────────────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src      : [batch, src_len]
            src_mask : [batch, 1, 1, src_len]
        Returns:
            memory   : [batch, src_len, d_model]
        """
        x = self.pos_enc(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            memory   : [batch, src_len, d_model]
            src_mask : [batch, 1, 1, src_len]
            tgt      : [batch, tgt_len]
            tgt_mask : [batch, 1, tgt_len, tgt_len]
        Returns:
            logits   : [batch, tgt_len, tgt_vocab_size]
        """
        x = self.pos_enc(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.projection(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            src      : [batch, src_len]
            tgt      : [batch, tgt_len]
            src_mask : [batch, 1, 1, src_len]
            tgt_mask : [batch, 1, tgt_len, tgt_len]
        Returns:
            logits   : [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str, beam_size: int = 5, max_len: int = 100) -> str:
        """
        Translate a raw German sentence to English using beam search decoding.

        Args:
            src_sentence : Raw German string.
            beam_size    : Number of beams (default 5). Higher = better BLEU, slower.
            max_len      : Maximum output tokens to generate.

        Vocabulary is loaded automatically from vocab.pt on first call if
        set_vocabs() has not been called explicitly.
        """
        if self.src_vocab is None or self.tgt_vocab is None:
            if not self._try_load_vocab():
                raise RuntimeError(
                    "vocab.pt not found and checkpoint does not contain vocab dicts."
                )

        device = next(self.parameters()).device

        # Tokenise — use spaCy if available, else whitespace split
        if self.spacy_de is not None:
            tokens = [tok.text.lower() for tok in self.spacy_de(src_sentence)]
        else:
            tokens = src_sentence.lower().split()

        unk_idx = self.src_vocab.stoi.get('<unk>', 0)
        sos_idx = self.src_vocab.stoi['<sos>']
        eos_idx = self.src_vocab.stoi['<eos>']
        pad_idx = self.src_vocab.stoi['<pad>']

        indices = (
            [sos_idx]
            + [self.src_vocab.stoi.get(t, unk_idx) for t in tokens]
            + [eos_idx]
        )
        src = torch.tensor([indices], dtype=torch.long, device=device)
        src_mask = make_src_mask(src, pad_idx=pad_idx)

        tgt_sos = self.tgt_vocab.stoi['<sos>']
        tgt_eos = self.tgt_vocab.stoi['<eos>']
        tgt_pad = self.tgt_vocab.stoi['<pad>']

        self.eval()
        with torch.no_grad():
            memory = self.encode(src, src_mask)

            # ── Beam search ────────────────────────────────────────────
            # Each beam: (cumulative_log_prob, token_id_list)
            beams: list = [(0.0, [tgt_sos])]
            completed: list = []

            for _ in range(max_len):
                all_candidates: list = []

                for cum_lp, seq in beams:
                    if seq[-1] == tgt_eos:
                        completed.append((cum_lp, seq))
                        continue

                    ys = torch.tensor([seq], dtype=torch.long, device=device)
                    tgt_mask = make_tgt_mask(ys, pad_idx=tgt_pad)
                    logits   = self.decode(memory, src_mask, ys, tgt_mask)
                    # Log-probs over vocab for the last position
                    log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
                    top_lp, top_idx = log_probs.topk(beam_size)

                    for lp, idx in zip(top_lp.tolist(), top_idx.tolist()):
                        all_candidates.append((cum_lp + lp, seq + [idx]))

                if not all_candidates:
                    break

                # Rank by length-normalised score to avoid bias toward short seqs
                all_candidates.sort(
                    key=lambda x: x[0] / max(len(x[1]) - 1, 1),
                    reverse=True,
                )
                beams = all_candidates[:beam_size]

                # Stop early if every active beam has emitted <eos>
                if all(s[-1] == tgt_eos for _, s in beams):
                    completed.extend(beams)
                    beams = []
                    break

            completed.extend(beams)
            if not completed:
                completed = [(0.0, [tgt_sos, tgt_eos])]

            # Pick the hypothesis with the best length-normalised score
            completed.sort(
                key=lambda x: x[0] / max(len(x[1]) - 1, 1),
                reverse=True,
            )
            best_seq = completed[0][1]

        out_tokens = []
        for idx in best_seq:
            if idx == tgt_eos:
                break
            if idx not in (tgt_sos, tgt_pad):
                out_tokens.append(self.tgt_vocab.lookup_token(idx))

        return ' '.join(out_tokens)
