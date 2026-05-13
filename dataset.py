"""
dataset.py — Multi30k Dataset, Vocabulary, and DataLoader
DA6401 Assignment 3
"""

from collections import Counter
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import spacy
from datasets import load_dataset


# ══════════════════════════════════════════════════════════════════════
#  VOCABULARY
# ══════════════════════════════════════════════════════════════════════

class Vocabulary:
    """
    Simple token ↔ index mapping.

    Special tokens:
        <unk> = 0
        <pad> = 1
        <sos> = 2
        <eos> = 3
    """

    SPECIALS = ['<unk>', '<pad>', '<sos>', '<eos>']

    def __init__(self) -> None:
        self.stoi = {tok: i for i, tok in enumerate(self.SPECIALS)}
        self.itos = {i: tok for i, tok in enumerate(self.SPECIALS)}

    def build_from_token_lists(self, token_lists: List[List[str]], min_freq: int = 2) -> None:
        counter = Counter(tok for tokens in token_lists for tok in tokens)
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx]   = token

    def __len__(self) -> int:
        return len(self.stoi)

    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, '<unk>')

    def lookup_index(self, token: str) -> int:
        return self.stoi.get(token, self.stoi['<unk>'])

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.stoi['<unk>']
        return [self.stoi.get(t, unk) for t in tokens]


# ══════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════

class Multi30kDataset(Dataset):
    """
    Wraps the bentrevett/multi30k HuggingFace dataset.

    Call build_vocab() once on the training split, then reuse the
    returned vocabs when constructing val / test splits.
    """

    def __init__(
        self,
        split: str = 'train',
        src_vocab: Vocabulary = None,
        tgt_vocab: Vocabulary = None,
        min_freq: int = 2,
    ) -> None:
        self.split = split

        # Load spacy models (German → English), auto-download if missing
        try:
            self.spacy_de = spacy.load('de_core_news_sm')
        except OSError:
            import subprocess, sys
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "de_core_news_sm"],
                check=True
            )
            self.spacy_de = spacy.load('de_core_news_sm')
        try:
            self.spacy_en = spacy.load('en_core_web_sm')
        except OSError:
            import subprocess, sys
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True
            )
            self.spacy_en = spacy.load('en_core_web_sm')

        # Load raw dataset
        raw = load_dataset('bentrevett/multi30k', split=split)
        self.raw_src = [ex['de'] for ex in raw]
        self.raw_tgt = [ex['en'] for ex in raw]

        # Build or reuse vocabularies
        if src_vocab is None or tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self.build_vocab(min_freq=min_freq)
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        self.data = self.process_data()

    # ── Tokenisers ────────────────────────────────────────────────────

    def tokenize_de(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self.spacy_de(text)]

    def tokenize_en(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self.spacy_en(text)]

    # ── Vocabulary building ───────────────────────────────────────────

    def build_vocab(self, min_freq: int = 2) -> Tuple[Vocabulary, Vocabulary]:
        """
        Build src (de) and tgt (en) vocabularies from this split's data.
        Should be called only on the training split.
        """
        src_tokens = [self.tokenize_de(s) for s in self.raw_src]
        tgt_tokens = [self.tokenize_en(s) for s in self.raw_tgt]

        src_vocab = Vocabulary()
        tgt_vocab = Vocabulary()
        src_vocab.build_from_token_lists(src_tokens, min_freq=min_freq)
        tgt_vocab.build_from_token_lists(tgt_tokens, min_freq=min_freq)

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        return src_vocab, tgt_vocab

    # ── Data processing ───────────────────────────────────────────────

    def process_data(self) -> List[Tuple[List[int], List[int]]]:
        """
        Tokenise every sentence pair and convert to integer indices.
        Wraps each sequence with <sos> and <eos>.
        """
        sos_s, eos_s = self.src_vocab.stoi['<sos>'], self.src_vocab.stoi['<eos>']
        sos_t, eos_t = self.tgt_vocab.stoi['<sos>'], self.tgt_vocab.stoi['<eos>']

        processed = []
        for src_text, tgt_text in zip(self.raw_src, self.raw_tgt):
            src_ids = [sos_s] + self.src_vocab.encode(self.tokenize_de(src_text)) + [eos_s]
            tgt_ids = [sos_t] + self.tgt_vocab.encode(self.tokenize_en(tgt_text)) + [eos_t]
            processed.append((src_ids, tgt_ids))

        return processed

    # ── PyTorch Dataset interface ─────────────────────────────────────

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src, tgt = self.data[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════
#  DATALOADER HELPERS
# ══════════════════════════════════════════════════════════════════════

def collate_fn(batch, src_pad_idx: int = 1, tgt_pad_idx: int = 1):
    """Pad a batch of (src, tgt) tensor pairs to the same length."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_pad_idx)
    return src_padded, tgt_padded


def get_dataloader(
    dataset: Multi30kDataset,
    batch_size: int = 128,
    shuffle: bool = True,
) -> DataLoader:
    """Return a DataLoader with padding collation."""
    pad_idx = dataset.src_vocab.stoi['<pad>']
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, pad_idx, pad_idx),
    )


def build_datasets(batch_size: int = 128, min_freq: int = 2):
    """
    Convenience function: build train / val / test datasets and loaders.

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab, spacy_de
    """
    print("Loading training data and building vocabularies...")
    train_ds = Multi30kDataset(split='train', min_freq=min_freq)
    src_vocab = train_ds.src_vocab
    tgt_vocab  = train_ds.tgt_vocab
    spacy_de   = train_ds.spacy_de

    print(f"  src vocab size: {len(src_vocab)}")
    print(f"  tgt vocab size: {len(tgt_vocab)}")

    print("Loading validation data...")
    val_ds = Multi30kDataset(split='validation', src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    print("Loading test data...")
    test_ds = Multi30kDataset(split='test', src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = get_dataloader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = get_dataloader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab, spacy_de
