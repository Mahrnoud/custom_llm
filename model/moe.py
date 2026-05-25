"""
model/moe.py
─────────────────────────────────────────────────────────────────────────────
Sparse Mixture-of-Experts (MoE) Feed-Forward Network.

Design:
  • Top-k token routing (default k=2) via a noisy top-k router
  • SwiGLU activation inside each expert
  • Auxiliary load-balancing loss  (prevents expert collapse)
  • Router z-loss                  (improves training stability)
  • Expert-parallel aware layout for future multi-GPU scaling

References:
  • "Mixtral of Experts"    – Jiang et al. 2024
  • "Switch Transformers"   – Fedus et al. 2021 (load-balance loss)
  • "ST-MoE"               – Zoph et al. 2022  (z-loss)
─────────────────────────────────────────────────────────────────────────────
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────
# Single Expert (SwiGLU FFN)
# ──────────────────────────────────────────────────────────────────────────
class Expert(nn.Module):
    """
    A single FFN expert using the SwiGLU activation:
        out = (W_gate · x  ⊙  swish(W_up · x)) · W_down
    This avoids an explicit bias and matches LLaMA / Mistral design.
    """

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up   = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ──────────────────────────────────────────────────────────────────────────
# Noisy Top-k Router
# ──────────────────────────────────────────────────────────────────────────
class NoisyTopKRouter(nn.Module):
    """
    Computes per-token routing probabilities over `n_experts`.

    During training, multiplicative Gaussian noise is added to logits
    before the top-k selection to encourage exploration.

    Returns:
      dispatch_weights  : (B*T, k)        – softmax weights for top-k experts
      expert_indices    : (B*T, k) int64  – which expert each slot uses
      router_logits     : (B*T, n_experts)– raw logits (for aux losses)
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        k: int = 2,
        noise_std: float = 1e-2,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.noise_std = noise_std
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,          # (B*T, d_model) – flattened tokens
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.gate(x)      # (N, n_experts)

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits_noisy = logits + noise
        else:
            logits_noisy = logits

        # Top-k selection
        top_vals, top_idx = torch.topk(logits_noisy, self.k, dim=-1)
        weights = F.softmax(top_vals, dim=-1)   # normalise selected experts

        return weights, top_idx, logits


# ──────────────────────────────────────────────────────────────────────────
# Auxiliary Losses
# ──────────────────────────────────────────────────────────────────────────
def load_balance_loss(
    router_logits: torch.Tensor,   # (N, n_experts)
    expert_indices: torch.Tensor,  # (N, k)
    n_experts: int,
    top_k: int,
) -> torch.Tensor:
    """
    Switch Transformer auxiliary loss that penalises uneven expert utilisation.
    L_aux = n_experts * sum_i(f_i * P_i)
    where f_i = fraction of tokens dispatched to expert i,
          P_i = mean router probability for expert i.
    """
    N = router_logits.shape[0]
    probs = F.softmax(router_logits, dim=-1)          # (N, n_experts)

    # One-hot mask of which experts were selected
    one_hot = torch.zeros_like(probs)                  # (N, n_experts)
    one_hot.scatter_(1, expert_indices, 1.0 / top_k)  # uniform weight across top-k

    # fraction of tokens per expert
    f = one_hot.mean(dim=0)    # (n_experts,)
    # mean prob per expert
    p = probs.mean(dim=0)      # (n_experts,)

    return n_experts * (f * p).sum()


def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """
    ST-MoE z-loss: penalises large router logit magnitudes.
    L_z = (1/N) * sum_n log²(sum_e exp(logit_{n,e}))
    Encourages the router to stay well-calibrated.
    """
    log_z = torch.logsumexp(router_logits, dim=-1)   # (N,)
    return (log_z ** 2).mean()


# ──────────────────────────────────────────────────────────────────────────
# Sparse MoE Layer
# ──────────────────────────────────────────────────────────────────────────
class SparseMoELayer(nn.Module):
    """
    Replaces the dense FFN in each transformer block.

    Forward returns (output, aux_loss) where aux_loss should be added to
    the main cross-entropy loss during training.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        expert_hidden_dim: int,
        top_k: int = 2,
        aux_loss_coeff: float = 1e-2,
        z_loss_coeff: float = 1e-3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_experts       = n_experts
        self.top_k           = top_k
        self.aux_loss_coeff  = aux_loss_coeff
        self.z_loss_coeff    = z_loss_coeff

        self.router  = NoisyTopKRouter(d_model, n_experts, k=top_k)
        self.experts = nn.ModuleList(
            [Expert(d_model, expert_hidden_dim) for _ in range(n_experts)]
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,     # (B, T, d_model)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        x_flat = x.view(B * T, D)            # (N, D) where N = B*T

        weights, expert_idx, router_logits = self.router(x_flat)
        # weights   : (N, k)
        # expert_idx: (N, k)

        # ── Dispatch tokens to experts ────────────────────────────────────
        output = torch.zeros_like(x_flat)    # (N, D)

        # Iterate over the k slots; for each slot dispatch tokens to
        # their assigned expert.  This loop is over k (usually 2) so it
        # is cheap; the inner expert calls process variable-length batches.
        for slot in range(self.top_k):
            slot_expert = expert_idx[:, slot]   # (N,)
            slot_weight = weights[:, slot]      # (N,)

            for e_id in range(self.n_experts):
                token_mask = (slot_expert == e_id)  # which tokens go here
                if not token_mask.any():
                    continue
                expert_input  = x_flat[token_mask]          # (n_e, D)
                expert_output = self.experts[e_id](expert_input)  # (n_e, D)
                # Accumulate weighted expert output
                output[token_mask] += (
                    slot_weight[token_mask].unsqueeze(-1) * expert_output
                )

        output = self.dropout(output)
        output = output.view(B, T, D)

        # ── Auxiliary losses ──────────────────────────────────────────────
        if self.training:
            lb_loss = load_balance_loss(router_logits, expert_idx, self.n_experts, self.top_k)
            z_loss  = router_z_loss(router_logits)
            aux     = self.aux_loss_coeff * lb_loss + self.z_loss_coeff * z_loss
        else:
            aux = torch.tensor(0.0, device=x.device)

        return output, aux

    # ── Convenience: expert utilisation stats for logging ─────────────────
    @torch.no_grad()
    def expert_utilisation(self, x: torch.Tensor) -> torch.Tensor:
        """Returns fraction of tokens dispatched to each expert. (n_experts,)"""
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        _, expert_idx, _ = self.router(x_flat)
        counts = torch.bincount(expert_idx.flatten(), minlength=self.n_experts).float()
        return counts / (B * T * self.top_k)
