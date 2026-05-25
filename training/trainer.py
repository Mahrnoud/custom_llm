"""
training/trainer.py
─────────────────────────────────────────────────────────────────────────────
Base Trainer class used by all three training stages.

Features:
  • Mixed-precision  (bfloat16 via torch.amp)
  • Gradient checkpointing
  • Gradient accumulation
  • Cosine LR schedule with linear warm-up
  • Gradient clipping
  • WandB / TensorBoard logging (optional)
  • Checkpoint save / resume
  • Multi-GPU via PyTorch DDP (single-process fallback)
─────────────────────────────────────────────────────────────────────────────
"""

import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────
# LR Schedule
# ──────────────────────────────────────────────────────────────────────────
def cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Cosine annealing with a linear warm-up phase."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────────────────
# Base Trainer
# ──────────────────────────────────────────────────────────────────────────
class Trainer:

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        stage_cfg,            # Stage1Config / Stage2Config / Stage3Config
        output_dir: str,
        device: Optional[str] = None,
        use_wandb: bool = False,
        use_tb: bool = False,
        project_name: str = "moe-llm",
    ):
        self.model       = model
        self.train_loader= train_loader
        self.eval_loader = eval_loader
        self.cfg         = stage_cfg
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        # Gradient checkpointing
        if getattr(stage_cfg, "gradient_checkpointing", False):
            enable_checkpointing = getattr(self.model, "gradient_checkpointing_enable", None)
            if callable(enable_checkpointing):
                enable_checkpointing()
            else:
                print(
                    f"[{stage_cfg.stage_name}] Warning: gradient checkpointing requested "
                    "but model does not support gradient_checkpointing_enable()."
                )

        # Optimiser
        self.optimizer = AdamW(
            self._get_param_groups(),
            lr=stage_cfg.learning_rate,
            betas=(stage_cfg.beta1, stage_cfg.beta2),
            eps=stage_cfg.epsilon,
            weight_decay=stage_cfg.weight_decay,
        )

        # LR scheduler
        self.scheduler = cosine_with_warmup(
            self.optimizer,
            warmup_steps=stage_cfg.warmup_steps,
            total_steps=stage_cfg.max_steps,
        )

        # Mixed precision scaler (for bf16 on CUDA)
        self.scaler = None  # bf16 doesn't need scaling
        self.use_bf16 = getattr(stage_cfg, "bf16", True) and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float32

        # Counters
        self.global_step    = 0
        self.best_eval_loss = float("inf")

        # Logging
        self._init_logging(use_wandb, use_tb, project_name)

    # ── Parameter groups: no weight-decay on norms / biases ──────────────
    def _get_param_groups(self):
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or "norm" in name or "bias" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay,    "weight_decay": self.cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    # ── Logging setup ─────────────────────────────────────────────────────
    def _init_logging(self, use_wandb: bool, use_tb: bool, project_name: str):
        self.writer = None
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(project=project_name, name=self.cfg.stage_name, config=vars(self.cfg))
        if use_tb and TB_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb_logs"))

    def _log(self, metrics: Dict[str, float]):
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(metrics, step=self.global_step)
        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, self.global_step)

    # ── Checkpoint helpers ────────────────────────────────────────────────
    def save_checkpoint(self, tag: str = ""):
        fname = f"checkpoint_{tag}.pt" if tag else f"checkpoint_{self.global_step:07d}.pt"
        path  = self.output_dir / fname
        torch.save({
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "sched_state": self.scheduler.state_dict(),
            "best_eval_loss": self.best_eval_loss,
        }, path)
        print(f"[{self.cfg.stage_name}] Saved checkpoint → {path}")
        return path

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.scheduler.load_state_dict(ckpt["sched_state"])
        self.global_step    = ckpt["global_step"]
        self.best_eval_loss = ckpt.get("best_eval_loss", float("inf"))
        print(f"[{self.cfg.stage_name}] Resumed from {path}  (step={self.global_step})")

    # ── Single training step ──────────────────────────────────────────────
    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids  = batch["input_ids"].to(self.device)
        labels     = batch["labels"].to(self.device)
        attn_mask  = batch.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)

        amp_context = (
            torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
            if self.use_bf16
            else nullcontext()
        )
        with amp_context:
            out  = self.model(input_ids, attention_mask=attn_mask, labels=labels)
            loss = out["loss"]

        return loss

    # ── Evaluation ────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self) -> float:
        if self.eval_loader is None:
            return float("nan")
        self.model.eval()
        total_loss, n = 0.0, 0
        for batch in self.eval_loader:
            loss = self._forward_step(batch)
            total_loss += loss.item()
            n += 1
        self.model.train()
        return total_loss / max(n, 1)

    # ── Main training loop ────────────────────────────────────────────────
    def train(self):
        cfg = self.cfg
        self.model.train()
        loader_iter = iter(self.train_loader)

        accum_loss  = 0.0
        t0          = time.time()

        print(f"\n{'─'*60}")
        print(f"  Stage : {cfg.stage_name}")
        print(f"  Steps : {cfg.max_steps:,}  (accum={cfg.gradient_accumulation_steps})")
        print(f"  Device: {self.device}  |  bf16={self.use_bf16}")
        print(f"{'─'*60}\n")

        while self.global_step < cfg.max_steps:
            # ── Accumulation micro-steps ───────────────────────────────
            self.optimizer.zero_grad()
            for micro in range(cfg.gradient_accumulation_steps):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.train_loader)
                    batch = next(loader_iter)

                loss = self._forward_step(batch) / cfg.gradient_accumulation_steps
                loss.backward()
                accum_loss += loss.item()

            # ── Gradient clip & optimiser step ─────────────────────────
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            # ── Logging ────────────────────────────────────────────────
            if self.global_step % cfg.log_every == 0:
                lr   = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                tokens_per_sec = (
                    cfg.log_every
                    * cfg.gradient_accumulation_steps
                    * cfg.batch_size
                    * cfg.seq_len
                    / elapsed
                )
                print(
                    f"[{cfg.stage_name}] step={self.global_step:6d} | "
                    f"loss={accum_loss:.4f} | lr={lr:.2e} | "
                    f"{tokens_per_sec/1e3:.1f}k tok/s"
                )
                self._log({
                    "train/loss": accum_loss,
                    "train/lr":   lr,
                    "train/tok_per_sec": tokens_per_sec,
                })
                accum_loss = 0.0
                t0 = time.time()

            # ── Evaluation ─────────────────────────────────────────────
            if self.global_step % cfg.eval_every == 0:
                eval_loss = self.evaluate()
                perplexity = math.exp(min(eval_loss, 20))
                print(
                    f"  [eval] step={self.global_step} | "
                    f"loss={eval_loss:.4f} | ppl={perplexity:.2f}"
                )
                self._log({"eval/loss": eval_loss, "eval/ppl": perplexity})
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_checkpoint(tag="best")
                self.model.train()

            # ── Periodic checkpoint ─────────────────────────────────────
            if self.global_step % cfg.save_every == 0:
                self.save_checkpoint()

        # Final save
        self.save_checkpoint(tag="final")
        print(f"\n[{cfg.stage_name}] Training complete. Best eval loss: {self.best_eval_loss:.4f}")
