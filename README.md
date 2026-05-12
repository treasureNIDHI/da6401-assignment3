# DA6401 Assignment 3 — Transformer for Machine Translation

Implementation of the Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017)  
for German→English Neural Machine Translation on the Multi30k dataset.

> **Code Repository:** https://github.com/treasureNIDHI/da6401-assignment3
> **GitHub Skeleton:** https://github.com/MiRL-IITM/da6401_assignment_3  
> **W&B Report (Public):** https://wandb.ai/nidhi-jagatpura-iit-madras/da6401-a3/reports/DA6401-Assignment-3-—-Transformer-for-Machine-Translation--VmlldzoxNjg1MzEwMQ==

---

## Results

| Metric | Value |
|--------|-------|
| **Test BLEU** | **37.60** |
| Best checkpoint epoch | 14 / 20 |
| Val loss at best checkpoint | 2.624 |
| Architecture | d_model=256, N=3, h=8, d_ff=512 |

---

## Project Structure

Follows the official DA6401 Assignment-3 GitHub skeleton structure:

```
├── model.py          # Transformer architecture (Encoder, Decoder, Multi-Head Attention, PE)
├── dataset.py        # Multi30k data loading, Vocabulary, spaCy tokenization
├── train.py          # Training loop, greedy decoding, BLEU evaluation, checkpointing
├── lr_scheduler.py   # Noam learning rate scheduler
├── experiments.py    # Ablation experiments (Sections 2.1–2.5)
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

---

## Training

### Baseline (reproduces 37.60 test BLEU)
```bash
export WANDB_API_KEY=<your_key>
python train.py
```

### Ablation experiments (Section 2)
```bash
# Run all five ablations — logs results to W&B
python experiments.py --exp all

# Or individually
python experiments.py --exp 2.1   # Noam vs Fixed LR
python experiments.py --exp 2.2   # Scaling factor 1/sqrt(d_k) ablation
python experiments.py --exp 2.3   # Attention rollout & head specialisation
python experiments.py --exp 2.4   # Sinusoidal vs Learned positional encoding
python experiments.py --exp 2.5   # Label smoothing ablation
```

---

## Design Decisions

### Layer Normalisation: Post-LayerNorm
We use **Post-LayerNorm** (i.e. `norm(x + sublayer(x))`), matching the original "Attention Is All You Need" paper exactly. In Post-LN, the residual stream is normalised *after* the skip connection is added, which keeps the gradient signal stable and produces better-calibrated representations at each layer. Pre-LN (normalising the input *before* the sublayer) would improve training stability for very deep networks but diverges from the paper's specification. Since we use only N=3 layers, Post-LN converges reliably with the Noam warmup schedule.

### Positional Encoding: Fixed Sinusoidal (Buffer)
The sinusoidal PE is registered as a `torch.Tensor` buffer (not an `nn.Parameter`), meaning it is saved with the model state but not updated during backprop — matching the autograder's contract.

---

## Autograder API

All contracted signatures are preserved exactly:

```python
# model.py
scaled_dot_product_attention(Q, K, V, mask=None) -> (output, attn_weights)
make_src_mask(src, pad_idx=1)                    -> BoolTensor [batch, 1, 1, src_len]
make_tgt_mask(tgt, pad_idx=1)                    -> BoolTensor [batch, 1, tgt_len, tgt_len]
Transformer.encode(src, src_mask)                -> Tensor
Transformer.decode(memory, src_mask, tgt, tgt_mask) -> Tensor
Transformer.infer(src_sentence: str)             -> str

# train.py
greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device) -> Tensor [1, out_len]
evaluate_bleu(model, test_dataloader, tgt_vocab, device, max_len)              -> float (0–100)
save_checkpoint(model, optimizer, scheduler, epoch, path)                      -> None
load_checkpoint(path, model, optimizer, scheduler)                             -> int
```

### Loading the best checkpoint

The best checkpoint (epoch 14, val loss 2.624) is hosted on Google Drive and downloaded automatically:

```python
model = Transformer(
    src_vocab_size=7853,
    tgt_vocab_size=5893,
    d_model=256, N=3, num_heads=8, d_ff=512, dropout=0.1,
    checkpoint_path="best_checkpoint.pt",   # downloads from Drive if not present
)
```

---

## Architecture

| Component | Configuration |
|---|---|
| Encoder layers (N) | 3 |
| Decoder layers (N) | 3 |
| Model dimension d_model | 256 |
| Attention heads h | 8 |
| Head dimension d_k = d_v | 32 |
| Feed-forward dimension d_ff | 512 |
| Dropout | 0.1 |
| Positional encoding | Fixed sinusoidal (buffer) |
| Layer normalisation | Post-LayerNorm |
| Optimizer | Adam (β₁=0.9, β₂=0.98, ε=1e-9) |
| LR schedule | Noam (warmup_steps=4000) |
| Loss | Label smoothing ε=0.1 |
| Dataset | Multi30k DE→EN via HuggingFace |

---

## Ablation Experiments (W&B Report Section 2)

| # | Experiment | Key Finding |
|---|---|---|
| 2.1 | Noam vs Fixed LR | Noam: 36.7 val BLEU vs Fixed 1e-4: 34.0 — warmup prevents early divergence in self-attention |
| 2.2 | Scaling factor 1/√dₖ | Without scaling: decoder Q/K grad norms ~6× larger — softmax saturation confirmed |
| 2.3 | Attention head specialisation | 8 heads show distinct local, long-range, positional, and boundary roles; head redundancy observed |
| 2.4 | Sinusoidal vs Learned PE | Sinusoidal: 38.1 BLEU vs Learned: 37.5 — sinusoidal converges faster and generalises to longer sequences |
| 2.5 | Label smoothing | ε=0.0 confidence 0.636 vs ε=0.1 confidence 0.579 — smoothing regularises over-confidence |

Full analysis and interactive plots: https://wandb.ai/nidhi-jagatpura-iit-madras/da6401-a3/reports/DA6401-Assignment-3-—-Transformer-for-Machine-Translation--VmlldzoxNjg1MzEwMQ==

---

## References

- Vaswani et al., "Attention Is All You Need", NeurIPS 2017. https://arxiv.org/abs/1706.03762
- Multi30k: https://huggingface.co/datasets/bentrevett/multi30k
- GitHub Skeleton: https://github.com/MiRL-IITM/da6401_assignment_3
