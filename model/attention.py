"""
model/attention.py
─────────────────────────────────────────────────────────────────────────────
Grouped-Query Attention (GQA) with:
  • Rotary Position Embeddings (RoPE) – extended-base (YaRN-style)
  • FlashAttention-2 when available, graceful fallback to SDPA
  • KV-cache support for autoregressive inference
─────────────────────────────────────────────────────────────────────────────
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Try to import FlashAttention-2 ────────────────────────────────────────
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────
# Rotary Position Embeddings
# ──────────────────────────────────────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    """
    RoPE with an extended base frequency for long-context generalisation.
    The large `theta` (500_000) follows the Llama-3 / GPT-4o approach.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 500_000.0):
        super().__init__()
        # Precompute inverse frequencies (half the head_dim)
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)         # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)        # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,          # (B, n_heads, T, head_dim)
        k: torch.Tensor,          # (B, n_kv_heads, T, head_dim)
        offset: int = 0,          # KV-cache offset during inference
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2] + offset
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)

        cos = self.cos_cached[:, :, offset : offset + q.shape[2], :]
        sin = self.sin_cached[:, :, offset : offset + q.shape[2], :]

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot.to(q.dtype), k_rot.to(k.dtype)


# ──────────────────────────────────────────────────────────────────────────
# KV-Cache Container
# ──────────────────────────────────────────────────────────────────────────
class KVCache(nn.Module):
    """Growable key-value cache for fast autoregressive decoding."""

    def __init__(self):
        super().__init__()
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            self.k, self.v = new_k, new_v
        else:
            self.k = torch.cat([self.k, new_k], dim=2)   # concat along seq dim
            self.v = torch.cat([self.v, new_v], dim=2)
        return self.k, self.v

    def reset(self):
        self.k = self.v = None

    @property
    def seq_len(self) -> int:
        return 0 if self.k is None else self.k.shape[2]


# ──────────────────────────────────────────────────────────────────────────
# Grouped-Query Attention
# ──────────────────────────────────────────────────────────────────────────
class GroupedQueryAttention(nn.Module):
    """
    GQA: n_heads query heads share n_kv_heads key/value heads.
    Setting n_kv_heads == n_heads gives Multi-Head Attention (MHA).
    Setting n_kv_heads == 1        gives Multi-Query Attention (MQA).

    Attention backend priority:
      1. FlashAttention-2  (fastest; requires CUDA + float16/bfloat16)
      2. torch.nn.functional.scaled_dot_product_attention  (SDPA)
      3. Manual softmax attention  (fallback)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        rope_theta: float = 500_000.0,
        attn_dropout: float = 0.0,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        assert d_model == n_heads * head_dim
        assert n_heads % n_kv_heads == 0

        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = head_dim
        self.n_rep      = n_heads // n_kv_heads   # repetitions for GQA expansion
        self.scale      = head_dim ** -0.5
        self.attn_drop  = attn_dropout
        self.use_flash  = use_flash_attn and FLASH_ATTN_AVAILABLE

        # Projections
        self.q_proj = nn.Linear(d_model, n_heads    * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model,    bias=False)

        self.rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len, theta=rope_theta)

    # ── GQA helper: repeat KV heads to match Q heads ─────────────────────
    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_kv_heads, T, D) → (B, n_heads, T, D)"""
        if self.n_rep == 1:
            return x
        return x.repeat_interleave(self.n_rep, dim=1)

    # ── FlashAttention-2 path ─────────────────────────────────────────────
    def _flash_attn(
        self,
        q: torch.Tensor,      # (B, n_heads, T, D)
        k: torch.Tensor,      # (B, n_kv_heads, T, D)
        v: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        # flash_attn expects (B, T, H, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = flash_attn_func(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
            causal=is_causal,
        )
        return out.transpose(1, 2)   # (B, H, T, D)

    # ── SDPA / manual fallback path ───────────────────────────────────────
    def _sdpa_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        k_exp = self._expand_kv(k)
        v_exp = self._expand_kv(v)
        dropout_p = self.attn_drop if self.training else 0.0
        if attention_mask is not None and is_causal:
            q_len, k_len = q.shape[-2], k_exp.shape[-2]
            causal_mask = torch.ones(
                q_len,
                k_len,
                device=q.device,
                dtype=torch.bool,
            ).triu(diagonal=1)
            causal_mask = q.new_zeros(q_len, k_len).masked_fill(
                causal_mask,
                torch.finfo(q.dtype).min,
            )
            attention_mask = attention_mask.to(device=q.device, dtype=q.dtype) + causal_mask
            is_causal = False

        return F.scaled_dot_product_attention(
            q, k_exp, v_exp,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,                            # (B, T, d_model)
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        cache_offset = kv_cache.seq_len if kv_cache is not None else 0

        # Project
        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k, offset=cache_offset)

        # Update KV cache if present (inference mode)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
            is_causal = False   # already encoded in cache; no causal mask needed

        # Attention computation
        if self.use_flash and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            if kv_cache is not None:
                # FlashAttention doesn't support arbitrary KV; fall through
                attn_out = self._sdpa_attn(q, k, v, attention_mask, is_causal)
            else:
                attn_out = self._flash_attn(q, k, v, is_causal=is_causal)
        else:
            attn_out = self._sdpa_attn(q, k, v, attention_mask, is_causal)

        # Reshape & project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn_out)
