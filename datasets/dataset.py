"""
Dataset loading for the 800K transformer.

Supports two datasets:
  1. "custom"   — local text file (datasets/sample_corpus.txt)
  2. "wikitext2" — WikiText-2 via HuggingFace datasets library

Both produce raw text strings that are byte-tokenized by ByteTokenizer.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import ByteTokenizer

# ── Available datasets ───────────────────────────────────────────────────────

DATASETS = {
    "custom": {
        "description": "Custom proverbs corpus (local file)",
        "path": os.path.join(os.path.dirname(__file__), "sample_corpus.txt"),
    },
    "wikitext2": {
        "description": "WikiText-2 via HuggingFace datasets",
        "hf_name": "wikitext",
        "hf_config": "wikitext-2-raw-v1",
    },
}


# ── Text Dataset ─────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """Byte-level token dataset with fixed-length sliding windows."""

    def __init__(self, text: str, seq_len: int):
        self.tokenizer = ByteTokenizer()
        self.tokens = self.tokenizer.encode(text)
        self.seq_len = seq_len

    def __len__(self):
        return max(1, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for target
        chunk = self.tokens[start:end]

        # Pad if needed
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [0] * (self.seq_len + 1 - len(chunk))

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ── Infinite Data Iterator ───────────────────────────────────────────────────

class InfiniteDataLoader:
    """
    Wraps a DataLoader so we can iterate indefinitely.
    Used for step-based training (no epoch concept).
    """

    def __init__(self, dataset: TextDataset, batch_size: int):
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
        self.iter = iter(self.loader)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            return next(self.iter)


# ── Dataset Loaders ──────────────────────────────────────────────────────────

def load_custom_corpus(path: str = None) -> str:
    """Load custom text corpus from local file."""
    if path is None:
        path = DATASETS["custom"]["path"]
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_wikitext2(split: str = "train") -> str:
    """
    Load WikiText-2 (raw) via HuggingFace datasets library.

    Args:
        split: "train", "validation", or "test"
    Returns:
        Raw text string (all articles concatenated)
    """
    from datasets import load_dataset as hf_load_dataset

    hf_name = DATASETS["wikitext2"]["hf_name"]
    hf_config = DATASETS["wikitext2"]["hf_config"]

    print(f"Downloading WikiText-2 from HuggingFace ({hf_name}/{hf_config})...")
    ds = hf_load_dataset(hf_name, hf_config, split=split)
    print(f"  Loaded {len(ds)} examples from {split} split")

    # Concatenate all text entries
    text = "\n".join(example["text"] for example in ds)
    return text


def load_dataset(
    name: str,
    seq_len: int,
    split: str = "train",
    custom_path: str = None,
) -> TextDataset:
    """
    Load a dataset by name and return a TextDataset.

    Args:
        name:        "custom" or "wikitext2"
        seq_len:     sequence length for the dataset
        split:       "train", "validation", or "test" (wikitext2 only)
        custom_path: override path for custom dataset
    Returns:
        TextDataset ready for DataLoader
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")

    if name == "custom":
        text = load_custom_corpus(custom_path)
        print(f"Dataset [custom]: {len(text):,} chars, {len(text.encode('utf-8')):,} bytes")
    elif name == "wikitext2":
        text = load_wikitext2(split)
        print(f"Dataset [wikitext2/{split}]: {len(text):,} chars, {len(text.encode('utf-8')):,} bytes")
    else:
        raise ValueError(f"Dataset '{name}' not implemented")

    return TextDataset(text, seq_len)


def create_infinite_loader(
    name: str,
    seq_len: int,
    batch_size: int,
    split: str = "train",
    custom_path: str = None,
) -> InfiniteDataLoader:
    """Load dataset and wrap in an InfiniteDataLoader for step-based training."""
    dataset = load_dataset(name, seq_len, split, custom_path)
    return InfiniteDataLoader(dataset, batch_size)
