"""
training/stage3_tools.py
─────────────────────────────────────────────────────────────────────────────
Stage 3: Tool-Use Training

Teaches the model to:
  1. Recognise when it needs external information
  2. Emit a structured <tool_call> block
  3. Read <tool_result> content
  4. Synthesise retrieved data into a complete, accurate answer

Training format:
  <|system|>  … tool descriptions …  <|end|>
  <|user|>    … user question …      <|end|>
  <|assistant|>
    <think>… reasoning …</think>
    <tool_call>{"name": "web_search", "query": "…"}</tool_call>
    <tool_result>… retrieved data …</tool_result>
    … final answer …
  <|end|>

Loss is masked on user/system tokens; only the assistant turn is supervised.

Run:
    python training/stage3_tools.py \
        --output_dir  checkpoints/stage3 \
        --data_dir    data/stage3 \
        --tokenizer   tokenizer/ \
        --stage2_ckpt checkpoints/stage2/checkpoint_best.pt
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.model_config import ModelConfig, Stage3Config
from data.dataset import ToolUseDataset, WeightedDataset, JsonlDataset, make_dataloader
from model.architecture import build_model
from tools.tool_registry import (
    TOOL_DESCRIPTIONS,
    ToolCall,
    execute_tool,
    build_system_prompt,
)
from training.trainer import Trainer


# ──────────────────────────────────────────────────────────────────────────
# Synthetic Data Generator
# ──────────────────────────────────────────────────────────────────────────
SEED_QUESTIONS = [
    # Search-required questions
    ("What is the current world population?",
     "web_search", {"query": "current world population 2025"}),
    ("What is the melting point of tungsten in Celsius?",
     "web_search", {"query": "melting point of tungsten Celsius"}),
    ("Solve: integral of x² dx from 0 to 3",
     "calculator", {"expression": "3**3 / 3 - 0**3 / 3"}),
    ("What is 15% of 8,240?",
     "calculator", {"expression": "8240 * 0.15"}),
    ("Write a Python function to compute Fibonacci numbers iteratively.",
     "python_exec", {"code": "def fib(n):\n    a,b=0,1\n    for _ in range(n): a,b=b,a+b\n    return a\nprint([fib(i) for i in range(10)])"}),
    ("How many seconds are in a year?",
     "calculator", {"expression": "365.25 * 24 * 3600"}),
    ("What is the chemical formula for caffeine?",
     "web_search", {"query": "chemical formula caffeine molecular structure"}),
    ("Compute the determinant of [[3,1],[2,4]]",
     "calculator", {"expression": "3*4 - 1*2"}),
    ("What is Avogadro's number?",
     "web_search", {"query": "Avogadro number exact value"}),
    ("Explain the Heisenberg uncertainty principle.",
     "web_search", {"query": "Heisenberg uncertainty principle explanation"}),
]


def generate_synthetic_tool_example(
    question: str,
    tool_name: str,
    tool_params: Dict,
    tokenizer=None,
) -> Dict[str, str]:
    """
    Generate one synthetic training example:
      user: <question>
      assistant: <think>…</think><tool_call>…</tool_call><tool_result>…</tool_result><answer>
    """
    call   = ToolCall(name=tool_name, params=tool_params)
    result = execute_tool(call)

    think_block = f"<think>I should use {tool_name} to answer this accurately.</think>"
    call_block  = f"<tool_call>{call.to_json()}</tool_call>"
    result_block = f"<tool_result>{result.format_for_model()[:500]}</tool_result>"

    if result.success and result.content.strip():
        answer = f"Based on the retrieved information: {result.content[:300].strip()}"
    else:
        answer = "I was unable to retrieve the information. Based on my knowledge: [answer here]"

    return {
        "user": question,
        "assistant": f"{think_block}\n{call_block}\n{result_block}\n{answer}",
    }


def create_synthetic_dataset(
    output_path: str,
    n_examples: int = 500,
    seed: int = 42,
) -> None:
    """
    Generate a synthetic tool-use JSONL file for Stage-3 training.
    Run this once before training if you don't have real tool-use data.
    """
    random.seed(seed)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    examples = []
    while len(examples) < n_examples:
        q, tool, params = random.choice(SEED_QUESTIONS)
        ex = generate_synthetic_tool_example(q, tool, params)
        examples.append(ex)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[Stage3] Generated {len(examples)} synthetic examples → {output_path}")


# ──────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────
def load_tokenizer(tokenizer_dir: str):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))


def build_stage3_datasets(data_dir: str, tokenizer, cfg: Stage3Config, model_cfg: ModelConfig):
    """
    Expected layout:
        data/stage3/
            tool_use_synthetic/   ← .jsonl  (generated or curated tool-call traces)
            search_qa/            ← .jsonl  (question → search → answer)
            retrieval_augmented/  ← .jsonl  (RAG-style examples)
            general_replay/       ← .jsonl  (Stage-1 data replay)
    """
    datasets, weights = [], []

    subdirs = {
        "tool_use_synthetic":  (ToolUseDataset, 0.40, True),
        "search_qa":           (ToolUseDataset, 0.25, True),
        "retrieval_augmented": (ToolUseDataset, 0.25, True),
        "general_replay":      (JsonlDataset,   0.10, False),
    }

    for subdir, (ds_cls, weight, is_tool) in subdirs.items():
        path = Path(data_dir) / subdir

        # Auto-generate synthetic data if not present
        if subdir == "tool_use_synthetic" and not path.exists():
            print(f"[Stage3] Auto-generating synthetic tool-use data …")
            path.mkdir(parents=True, exist_ok=True)
            create_synthetic_dataset(str(path / "synthetic.jsonl"), n_examples=1000)

        if not path.exists():
            print(f"[Stage3] Warning: {path} not found – skipping.")
            continue

        jsonl_files = sorted(path.glob("*.jsonl"))
        if not jsonl_files:
            continue

        if is_tool:
            sub_ds = [
                ToolUseDataset(
                    str(f), tokenizer, seq_len=cfg.seq_len,
                    bos_id=model_cfg.bos_token_id,
                    eos_id=model_cfg.eos_token_id,
                    mask_user_tokens=cfg.mask_user_tokens,
                )
                for f in jsonl_files
            ]
        else:
            sub_ds = [
                JsonlDataset(str(f), tokenizer, seq_len=cfg.seq_len,
                             bos_id=model_cfg.bos_token_id, eos_id=model_cfg.eos_token_id)
                for f in jsonl_files
            ]

        ds = sub_ds[0] if len(sub_ds) == 1 else WeightedDataset(sub_ds, [1.0]*len(sub_ds))
        datasets.append(ds)
        weights.append(weight)
        print(f"[Stage3] Loaded {subdir:25s}  size={len(ds):>8,}  weight={weight}")

    if not datasets:
        raise RuntimeError(f"No Stage-3 data found in {data_dir}.")

    train_ds = WeightedDataset(datasets, weights)
    eval_ds  = datasets[0]
    return train_ds, eval_ds


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main(args):
    model_cfg = ModelConfig()
    stage_cfg = Stage3Config()

    print(f"\n{'═'*60}")
    print(f"  STAGE 3 – Tool-Use Training")
    print(f"  Continuing from: {args.stage2_ckpt}")
    print(f"  Max steps: {stage_cfg.max_steps:,}  |  LR: {stage_cfg.learning_rate}")
    print(f"{'═'*60}\n")

    tokenizer = load_tokenizer(args.tokenizer)

    # ── Build model ────────────────────────────────────────────────────
    model = build_model(model_cfg)

    # ── Load Stage-2 weights ───────────────────────────────────────────
    if args.stage2_ckpt and Path(args.stage2_ckpt).exists():
        ckpt = torch.load(args.stage2_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        print(f"[Stage3] Loaded Stage-2 weights from {args.stage2_ckpt}")
    else:
        print("[Stage3] WARNING: no Stage-2 checkpoint found.")

    # ── Datasets ──────────────────────────────────────────────────────
    train_ds, eval_ds = build_stage3_datasets(args.data_dir, tokenizer, stage_cfg, model_cfg)

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

    # Print the system prompt that was used during training (for reference)
    print("\n[Stage3] System prompt used during training:")
    print(build_system_prompt())


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3: Tool-Use Training")
    p.add_argument("--output_dir",   default="checkpoints/stage3")
    p.add_argument("--data_dir",     default="data/stage3")
    p.add_argument("--tokenizer",    default="tokenizer/")
    p.add_argument("--stage2_ckpt",  default="checkpoints/stage2/checkpoint_best.pt")
    p.add_argument("--resume",       default=None)
    p.add_argument("--wandb",        action="store_true")
    p.add_argument("--tensorboard",  action="store_true")
    p.add_argument("--gen_data",     action="store_true",
                   help="Only generate synthetic data and exit")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.gen_data:
        create_synthetic_dataset("data/stage3/tool_use_synthetic/synthetic.jsonl", n_examples=2000)
    else:
        main(args)
