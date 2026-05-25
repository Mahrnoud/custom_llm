"""
training/stage1_pretrain.py
─────────────────────────────────────────────────────────────────────────────
Stage 1: General English Language Modeling Pre-Training

Trains the model from scratch on a large-scale English corpus to learn:
  • Basic grammar, syntax, and semantics
  • General world knowledge and factual associations
  • Multi-sentence coherence and text generation
  • Foundational reasoning patterns

Data mix (configurable in Stage1Config):
  60% Web crawl (Common Crawl / C4 style)
  20% Books (Project Gutenberg, OpenLibrary, etc.)
  15% Wikipedia / encyclopaedic text
   5% News articles

Run:
    python training/stage1_pretrain.py \
        --output_dir  checkpoints/stage1 \
        --data_dir    data/stage1 \
        --tokenizer   tokenizer/ \
        [--resume     checkpoints/stage1/checkpoint_XXXXXXX.pt]
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# ── Make project root importable ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.model_config import ModelConfig, Stage1Config, TokenizerConfig
from data.dataset import JsonlDataset, TextDataset, WeightedDataset, make_dataloader
from model.architecture import build_model
from training.trainer import Trainer


# ──────────────────────────────────────────────────────────────────────────
# Data preparation helpers
# ──────────────────────────────────────────────────────────────────────────
def load_tokenizer(tokenizer_dir: str):
    """Load the trained BPE tokenizer."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
    return tok


def build_stage1_datasets(data_dir: str, tokenizer, cfg: Stage1Config, model_cfg: ModelConfig):
    """
    Build train + eval datasets from data_dir subdirectories.

    Expected layout:
        data/stage1/
            english_web_crawl/  ← .jsonl files  OR  corpus.bin memmap
            books/
            wikipedia_en/
            news/

    Falls back gracefully if a directory doesn't exist.
    """
    datasets, weights = [], []
    subdirs = [
        "english_web_crawl",
        "books",
        "wikipedia_en",
        "news",
    ]
    default_weights = cfg.data_weights

    for subdir, w in zip(subdirs, default_weights):
        path = Path(data_dir) / subdir
        if not path.exists():
            print(f"[Stage1] Warning: {path} not found – skipping.")
            continue

        # Prefer pre-tokenised binary
        bin_file = path / "corpus.bin"
        if bin_file.exists():
            ds = TextDataset(str(bin_file), seq_len=cfg.seq_len)
        else:
            jsonl_files = list(path.glob("*.jsonl"))
            if not jsonl_files:
                print(f"[Stage1] Warning: no .bin or .jsonl in {path} – skipping.")
                continue
            # Combine all .jsonl files into one dataset list then merge
            sub_datasets = [
                JsonlDataset(
                    str(f), tokenizer, seq_len=cfg.seq_len,
                    bos_id=model_cfg.bos_token_id,
                    eos_id=model_cfg.eos_token_id,
                )
                for f in jsonl_files
            ]
            sub_datasets = [ds for ds in sub_datasets if len(ds) > 0]
            if not sub_datasets:
                print(f"[Stage1] Warning: all .jsonl files in {path} are empty – skipping.")
                continue
            if len(sub_datasets) == 1:
                ds = sub_datasets[0]
            else:
                ds = WeightedDataset(sub_datasets, [1.0] * len(sub_datasets))

        if len(ds) == 0:
            print(f"[Stage1] Warning: {path} produced 0 samples – skipping.")
            continue

        datasets.append(ds)
        weights.append(w)
        print(f"[Stage1] Loaded {subdir:25s}  size={len(ds):>10,}  weight={w}")

    if not datasets:
        raise RuntimeError(f"No data found in {data_dir}. Check your data layout.")

    train_ds = WeightedDataset(datasets, weights)

    # Tiny eval split: last 1024 samples from the smallest dataset
    eval_ds = datasets[0]  # use first dataset for eval
    return train_ds, eval_ds


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main(args):
    model_cfg = ModelConfig()
    stage_cfg = Stage1Config()
    tok_cfg   = TokenizerConfig()

    print(f"\n{'═'*60}")
    print(f"  STAGE 1 – General Pre-Training")
    print(f"  Parameters: ~{model_cfg.approximate_params/1e6:.0f}M")
    print(f"  Max steps : {stage_cfg.max_steps:,}")
    print(f"  Sequence  : {stage_cfg.seq_len}")
    print(f"{'═'*60}\n")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = load_tokenizer(args.tokenizer)
    print(f"[Stage1] Tokenizer loaded  vocab_size={tokenizer.get_vocab_size():,}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(model_cfg)
    print(f"[Stage1] Model created with {model.num_parameters():,} parameters")

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds, eval_ds = build_stage1_datasets(
        args.data_dir, tokenizer, stage_cfg, model_cfg
    )

    train_loader = make_dataloader(
        train_ds,
        batch_size=stage_cfg.batch_size,
        pad_id=model_cfg.pad_token_id,
        seq_len=stage_cfg.seq_len,
    )
    eval_loader = make_dataloader(
        eval_ds,
        batch_size=stage_cfg.batch_size,
        shuffle=False,
        pad_id=model_cfg.pad_token_id,
        seq_len=stage_cfg.seq_len,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        stage_cfg=stage_cfg,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
        use_tb=args.tensorboard,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: General LM Pre-Training")
    p.add_argument("--output_dir",   default="checkpoints/stage1")
    p.add_argument("--data_dir",     default="data/stage1")
    p.add_argument("--tokenizer",    default="tokenizer/")
    p.add_argument("--resume",       default=None, help="Path to checkpoint to resume from")
    p.add_argument("--wandb",        action="store_true", help="Enable WandB logging")
    p.add_argument("--tensorboard",  action="store_true", help="Enable TensorBoard logging")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
