"""
training/stage2_stem.py
─────────────────────────────────────────────────────────────────────────────
Stage 2: STEM Specialisation

Continues training from the best Stage-1 checkpoint on a curated STEM corpus
to build deep expertise in:
  • Mathematics    (proofs, symbolic computation, problem solving)
  • Physics        (mechanics, electromagnetism, quantum theory, cosmology)
  • Chemistry      (organic, inorganic, reaction mechanisms, stoichiometry)
  • Programming    (Python, C++, algorithms, data structures, system design)

A 10 % replay of general English data is included to prevent catastrophic
forgetting of language fundamentals.

Data mix:
  30% Mathematics  (ArXiv math, textbooks, problem sets)
  20% Physics      (ArXiv physics, lecture notes)
  15% Chemistry    (journals, textbooks, reaction databases)
  25% Code         (GitHub, StackOverflow, programming books)
  10% General replay (Stage-1 data)

Run:
    python training/stage2_stem.py \
        --output_dir   checkpoints/stage2 \
        --data_dir     data/stage2 \
        --tokenizer    tokenizer/ \
        --stage1_ckpt  checkpoints/stage1/checkpoint_best.pt
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.model_config import ModelConfig, Stage2Config
from data.dataset import JsonlDataset, TextDataset, WeightedDataset, make_dataloader
from model.architecture import build_model
from training.trainer import Trainer


# ──────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────
STEM_SUBDIRS = {
    "mathematics":    0.30,
    "physics":        0.20,
    "chemistry":      0.15,
    "code":           0.25,
    "general_replay": 0.10,   # anti-forgetting replay from Stage 1 data
}


def load_tokenizer(tokenizer_dir: str):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))


def build_stage2_datasets(data_dir: str, tokenizer, cfg: Stage2Config, model_cfg: ModelConfig):
    """
    Build train / eval datasets from data/stage2 subdirectories.

    Expected layout (same convention as Stage 1):
        data/stage2/
            mathematics/      ← .jsonl or corpus.bin
            physics/
            chemistry/
            code/
            general_replay/   ← copy of some Stage-1 data
    """
    datasets, weights = [], []

    for subdir, default_weight in STEM_SUBDIRS.items():
        path = Path(data_dir) / subdir
        if not path.exists():
            print(f"[Stage2] Warning: {path} not found – skipping.")
            continue

        bin_file = path / "corpus.bin"
        if bin_file.exists():
            ds = TextDataset(str(bin_file), seq_len=cfg.seq_len)
        else:
            jsonl_files = sorted(path.glob("*.jsonl"))
            if not jsonl_files:
                print(f"[Stage2] Warning: no data in {path} – skipping.")
                continue
            sub_ds = [
                JsonlDataset(str(f), tokenizer, seq_len=cfg.seq_len,
                             bos_id=model_cfg.bos_token_id,
                             eos_id=model_cfg.eos_token_id)
                for f in jsonl_files
            ]
            ds = sub_ds[0] if len(sub_ds) == 1 else WeightedDataset(sub_ds, [1.0]*len(sub_ds))

        datasets.append(ds)
        weights.append(default_weight)
        print(f"[Stage2] Loaded {subdir:20s}  size={len(ds):>10,}  weight={default_weight}")

    if not datasets:
        raise RuntimeError(f"No STEM data found in {data_dir}.")

    train_ds = WeightedDataset(datasets, weights)
    eval_ds  = datasets[0]   # mathematics for eval perplexity
    return train_ds, eval_ds


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main(args):
    model_cfg = ModelConfig()
    stage_cfg = Stage2Config()

    print(f"\n{'═'*60}")
    print(f"  STAGE 2 – STEM Specialisation")
    print(f"  Continuing from: {args.stage1_ckpt}")
    print(f"  Max steps: {stage_cfg.max_steps:,}  |  LR: {stage_cfg.learning_rate}")
    print(f"{'═'*60}\n")

    tokenizer = load_tokenizer(args.tokenizer)

    # ── Build model (same architecture as Stage 1) ─────────────────────
    model = build_model(model_cfg)

    # ── Load Stage-1 weights ───────────────────────────────────────────
    if args.stage1_ckpt and Path(args.stage1_ckpt).exists():
        ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        print(f"[Stage2] Loaded Stage-1 weights from {args.stage1_ckpt}")
    else:
        print("[Stage2] WARNING: no Stage-1 checkpoint found. Training from scratch.")

    # ── Datasets ──────────────────────────────────────────────────────
    train_ds, eval_ds = build_stage2_datasets(args.data_dir, tokenizer, stage_cfg, model_cfg)

    train_loader = make_dataloader(
        train_ds, batch_size=stage_cfg.batch_size,
        pad_id=model_cfg.pad_token_id, seq_len=stage_cfg.seq_len,
    )
    eval_loader = make_dataloader(
        eval_ds, batch_size=stage_cfg.batch_size, shuffle=False,
        pad_id=model_cfg.pad_token_id, seq_len=stage_cfg.seq_len,
    )

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
    p = argparse.ArgumentParser(description="Stage 2: STEM Specialisation")
    p.add_argument("--output_dir",   default="checkpoints/stage2")
    p.add_argument("--data_dir",     default="data/stage2")
    p.add_argument("--tokenizer",    default="tokenizer/")
    p.add_argument("--stage1_ckpt",  default="checkpoints/stage1/checkpoint_best.pt")
    p.add_argument("--resume",       default=None)
    p.add_argument("--wandb",        action="store_true")
    p.add_argument("--tensorboard",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
