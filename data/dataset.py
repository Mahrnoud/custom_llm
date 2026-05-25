"""
data/dataset.py
─────────────────────────────────────────────────────────────────────────────
Dataset utilities shared across all three training stages.

Classes:
  TextDataset          – Memory-mapped tokenised pre-training corpus
  WeightedDataset      – Samples from multiple datasets by weight
  ToolUseDataset       – Chat-format dataset for Stage-3 tool-call training
  DataCollator         – Pads/truncates a batch & builds labels

Usage:
  See individual docstrings and the stage training scripts.
─────────────────────────────────────────────────────────────────────────────
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────────────────
# Memory-Mapped Pre-training Dataset
# ──────────────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    """
    Reads a pre-tokenised binary corpus stored as a uint16 NumPy memmap.

    Pre-tokenise your raw corpus once with:
        tokens = tokenizer.encode(text).ids          # list[int]
        arr    = np.array(tokens, dtype=np.uint16)
        arr.tofile("corpus.bin")

    Then pass corpus_path="corpus.bin" here.

    Each sample is a contiguous chunk of `seq_len` tokens drawn from a
    random position (no padding overhead during pre-training).
    """

    def __init__(
        self,
        corpus_path: str,
        seq_len: int = 2048,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        data = np.memmap(corpus_path, dtype=np.uint16, mode="r")
        self.data   = data
        self.n_full = len(data) - seq_len - 1
        assert self.n_full > 0, f"Corpus too small for seq_len={seq_len}"
        rng = np.random.default_rng(seed)
        self.indices = rng.permutation(self.n_full)

    def __len__(self) -> int:
        return self.n_full

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = int(self.indices[idx % len(self.indices)])
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return {"input_ids": x, "labels": y}


# ──────────────────────────────────────────────────────────────────────────
# JSONL Dataset (general text stored as JSON Lines)
# ──────────────────────────────────────────────────────────────────────────
class JsonlDataset(Dataset):
    """
    Reads text from a .jsonl file, tokenises on the fly.
    Each line: {"text": "..."}

    Suitable for medium-sized datasets that fit in memory.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,              # HuggingFace tokenizers.Tokenizer
        seq_len: int = 2048,
        bos_id: int = 1,
        eos_id: int = 2,
    ):
        self.tokenizer = tokenizer
        self.seq_len   = seq_len
        self.bos_id    = bos_id
        self.eos_id    = eos_id

        self.examples: List[str] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text", obj.get("content", ""))
                if text:
                    self.examples.append(text)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.examples[idx]
        ids  = [self.bos_id] + self.tokenizer.encode(text).ids + [self.eos_id]
        # Truncate or pad to seq_len
        if len(ids) > self.seq_len + 1:
            ids = ids[: self.seq_len + 1]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:],  dtype=torch.long)
        return {"input_ids": x, "labels": y}


# ──────────────────────────────────────────────────────────────────────────
# Weighted Multi-Dataset Sampler
# ──────────────────────────────────────────────────────────────────────────
class WeightedDataset(Dataset):
    """
    Combines multiple datasets with per-dataset sampling weights.

    At each step, a dataset is chosen proportionally to its weight; then
    a random sample is drawn from it. This allows mixing corpora of
    vastly different sizes without explicit concatenation.
    """

    def __init__(self, datasets: List[Dataset], weights: List[float]):
        assert len(datasets) == len(weights)
        total = sum(weights)
        self.datasets = datasets
        self.weights  = [w / total for w in weights]
        self._len     = sum(len(d) for d in datasets)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, _: int) -> Dict[str, torch.Tensor]:
        # Sample dataset by weight, then sample item uniformly
        ds = random.choices(self.datasets, weights=self.weights, k=1)[0]
        idx = random.randint(0, len(ds) - 1)
        return ds[idx]


# ──────────────────────────────────────────────────────────────────────────
# Tool-Use Dataset (Stage 3)
# ──────────────────────────────────────────────────────────────────────────
TOOL_CALL_FORMAT = """\
<|system|>
You are a helpful assistant. When you need external information, use tools.
<|end|>
<|user|>
{user_message}
<|end|>
<|assistant|>
{assistant_response}
<|end|>"""

class ToolUseDataset(Dataset):
    """
    Dataset for Stage-3 tool-use fine-tuning.

    Expected JSONL format (one JSON object per line):
    {
      "user":      "What is the speed of light in a vacuum?",
      "assistant": "<think>I need an accurate value.</think>
                   <tool_call>{\"name\": \"web_search\", \"query\": \"speed of light vacuum\"}</tool_call>
                   <tool_result>299,792,458 m/s</tool_result>
                   The speed of light in a vacuum is exactly 299,792,458 metres per second."
    }

    Labels are set to -100 for all user/system tokens so the loss is
    computed ONLY on assistant-generated tokens.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        seq_len: int = 2048,
        bos_id: int = 1,
        eos_id: int = 2,
        mask_user_tokens: bool = True,
    ):
        self.tokenizer        = tokenizer
        self.seq_len          = seq_len
        self.bos_id           = bos_id
        self.eos_id           = eos_id
        self.mask_user_tokens = mask_user_tokens

        self.examples: List[Dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "user" in obj and "assistant" in obj:
                    self.examples.append(obj)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        full_text = TOOL_CALL_FORMAT.format(
            user_message=ex["user"],
            assistant_response=ex["assistant"],
        )

        # Tokenise the whole conversation
        all_ids = [self.bos_id] + self.tokenizer.encode(full_text).ids + [self.eos_id]

        if len(all_ids) > self.seq_len + 1:
            all_ids = all_ids[: self.seq_len + 1]

        input_ids = torch.tensor(all_ids[:-1], dtype=torch.long)
        labels    = torch.tensor(all_ids[1:],  dtype=torch.long)

        if self.mask_user_tokens:
            # Mask everything up to (and including) <|assistant|> token
            assistant_marker = self.tokenizer.encode("<|assistant|>").ids
            labels = self._mask_before_assistant(labels, assistant_marker)

        return {"input_ids": input_ids, "labels": labels}

    def _mask_before_assistant(
        self,
        labels: torch.Tensor,
        marker_ids: List[int],
    ) -> torch.Tensor:
        """Set labels to -100 for tokens before the assistant turn."""
        ids = labels.tolist()
        m = len(marker_ids)
        cut = 0
        for i in range(len(ids) - m + 1):
            if ids[i : i + m] == marker_ids:
                cut = i + m
                break
        if cut > 0:
            labels[:cut] = -100
        return labels


# ──────────────────────────────────────────────────────────────────────────
# Data Collator
# ──────────────────────────────────────────────────────────────────────────
class DataCollator:
    """
    Pads a batch of variable-length sequences to the longest in the batch.
    Also builds an attention mask (1 = real token, 0 = pad).
    """

    def __init__(self, pad_id: int = 0, seq_len: int = 2048):
        self.pad_id  = pad_id
        self.seq_len = seq_len

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in batch)
        max_len = min(max_len, self.seq_len)

        input_ids_list, labels_list, mask_list = [], [], []
        for item in batch:
            ids  = item["input_ids"][:max_len]
            labs = item["labels"][:max_len]
            pad_len = max_len - ids.shape[0]

            ids  = torch.cat([ids,  torch.full((pad_len,), self.pad_id,  dtype=torch.long)])
            labs = torch.cat([labs, torch.full((pad_len,), -100,          dtype=torch.long)])
            mask = torch.cat([torch.ones(max_len - pad_len, dtype=torch.long),
                              torch.zeros(pad_len,          dtype=torch.long)])

            input_ids_list.append(ids)
            labels_list.append(labs)
            mask_list.append(mask)

        return {
            "input_ids":      torch.stack(input_ids_list),
            "labels":         torch.stack(labels_list),
            "attention_mask": torch.stack(mask_list),
        }


# ──────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────
def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pad_id: int = 0,
    seq_len: int = 2048,
) -> DataLoader:
    collator = DataCollator(pad_id=pad_id, seq_len=seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )
