"""
model/architecture.py
─────────────────────────────────────────────────────────────────────────────
Full MoE-Transformer architecture (~250 M parameters).

Stack:
  Embedding
    └── Dropout
  × n_layers:
    PreNorm → GQA-RoPE → residual
    PreNorm → SparseMoE  → residual
  Final RMSNorm
  LM Head  (weight-tied to embedding)

Design choices:
  • RMSNorm          – lighter than LayerNorm, no mean subtraction
  • SwiGLU experts   – no bias projections
  • Pre-norm         – more stable gradient flow
  • Weight tying     – embedding ↔ LM head (saves ~25 M params)
─────────────────────────────────────────────────────────────────────────────
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.attention import GroupedQueryAttention, KVCache
from model.moe import SparseMoELayer


# ──────────────────────────────────────────────────────────────────────────
# RMSNorm
# ──────────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalisation (no mean centering)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


# ──────────────────────────────────────────────────────────────────────────
# Transformer Block
# ──────────────────────────────────────────────────────────────────────────
class MoETransformerBlock(nn.Module):
    """
    Single transformer block:
      x = x + Attention(RMSNorm(x))
      x = x + MoE-FFN(RMSNorm(x))
    Returns (x, aux_loss).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        n_experts: int,
        n_experts_active: int,
        expert_hidden_dim: int,
        max_seq_len: int,
        rope_theta: float,
        attn_dropout: float,
        moe_dropout: float,
        rms_eps: float,
        aux_loss_coeff: float,
        z_loss_coeff: float,
        use_flash_attn: bool,
    ):
        super().__init__()

        self.attn_norm = RMSNorm(d_model, eps=rms_eps)
        self.attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            attn_dropout=attn_dropout,
            use_flash_attn=use_flash_attn,
        )

        self.ffn_norm = RMSNorm(d_model, eps=rms_eps)
        self.moe = SparseMoELayer(
            d_model=d_model,
            n_experts=n_experts,
            expert_hidden_dim=expert_hidden_dim,
            top_k=n_experts_active,
            aux_loss_coeff=aux_loss_coeff,
            z_loss_coeff=z_loss_coeff,
            dropout=moe_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ── Attention sub-layer ───────────────────────────────────────────
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, attention_mask=attention_mask, kv_cache=kv_cache, is_causal=is_causal)
        x = residual + x

        # ── MoE FFN sub-layer ─────────────────────────────────────────────
        residual = x
        x, aux_loss = self.moe(self.ffn_norm(x))
        x = residual + x

        return x, aux_loss


# ──────────────────────────────────────────────────────────────────────────
# Full Model
# ──────────────────────────────────────────────────────────────────────────
class MoELanguageModel(nn.Module):
    """
    Causal language model: ~250 M total parameters.

    Forward signature:
      input_ids       : (B, T)
      attention_mask  : (B, T)  optional padding mask
      labels          : (B, T)  optional; if given, returns loss
      kv_caches       : list of KVCache, one per layer (inference only)

    Returns:
      logits  : (B, T, vocab_size)
      loss    : scalar cross-entropy + aux_loss  (None if no labels)
      aux_loss: scalar MoE routing loss
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        # ── Token Embedding ───────────────────────────────────────────────
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.embed_drop   = nn.Dropout(config.embed_dropout)

        # ── Transformer Blocks ────────────────────────────────────────────
        self.layers = nn.ModuleList([
            MoETransformerBlock(
                d_model          = config.d_model,
                n_heads          = config.n_heads,
                n_kv_heads       = config.n_kv_heads,
                head_dim         = config.head_dim,
                n_experts        = config.n_experts,
                n_experts_active = config.n_experts_active,
                expert_hidden_dim= config.expert_hidden_dim,
                max_seq_len      = config.max_seq_len,
                rope_theta       = config.rope_theta,
                attn_dropout     = config.attention_dropout,
                moe_dropout      = config.moe_dropout,
                rms_eps          = config.rms_norm_eps,
                aux_loss_coeff   = config.router_aux_loss_coeff,
                z_loss_coeff     = config.router_z_loss_coeff,
                use_flash_attn   = config.use_flash_attn,
            )
            for _ in range(config.n_layers)
        ])

        # ── Final Norm + LM Head ──────────────────────────────────────────
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # ── Initialise weights ────────────────────────────────────────────
        self.apply(self._init_weights)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # ── Parameter count helper ────────────────────────────────────────────
    def num_parameters(self, only_trainable: bool = False) -> int:
        params = self.parameters() if not only_trainable else filter(lambda p: p.requires_grad, self.parameters())
        return sum(p.numel() for p in params)

    # ── KV-Cache allocation for inference ─────────────────────────────────
    def allocate_kv_caches(self) -> List[KVCache]:
        return [KVCache() for _ in self.layers]

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,                   # (B, T)
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[KVCache]] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:

        B, T = input_ids.shape
        is_causal = (kv_caches is None)

        # Build padding mask for SDPA (additive, 0 = attend, -inf = ignore)
        if attention_mask is not None and kv_caches is None:
            # Convert boolean mask → additive mask
            attn_mask_4d = (
                (1.0 - attention_mask[:, None, None, :].float()) * -1e9
            ).to(input_ids.device)
        else:
            attn_mask_4d = None

        # Embedding
        x = self.embed_drop(self.embed_tokens(input_ids))

        # Transformer layers
        total_aux = torch.zeros(1, device=x.device, dtype=x.dtype)
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            if self.gradient_checkpointing and self.training and kv_caches is None:
                def layer_forward(hidden_states, checkpointed_layer=layer):
                    return checkpointed_layer(
                        hidden_states,
                        attention_mask=attn_mask_4d,
                        kv_cache=None,
                        is_causal=is_causal,
                    )

                x, aux = checkpoint(layer_forward, x, use_reentrant=False)
            else:
                x, aux = layer(x, attention_mask=attn_mask_4d, kv_cache=cache, is_causal=is_causal)
            total_aux = total_aux + aux

        x = self.norm(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        # ── Loss ──────────────────────────────────────────────────────────
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = ce_loss + total_aux

        return {
            "logits":   logits,
            "loss":     loss,
            "aux_loss": total_aux,
        }

    # ── Autoregressive generation ─────────────────────────────────────────
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Simple top-p + top-k sampling with KV-cache."""
        eos = eos_token_id or self.config.eos_token_id
        caches = self.allocate_kv_caches()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # On first pass use the full prompt; afterward just the new token
            cur_input = generated if caches[0].seq_len == 0 else generated[:, -1:]
            out = self.forward(cur_input, kv_caches=caches)
            next_logits = out["logits"][:, -1, :] / temperature  # (B, vocab)

            # Top-k filtering
            if top_k > 0:
                vals, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < vals[:, -1:]] = float("-inf")

            # Top-p (nucleus) filtering
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove] = float("-inf")
            next_logits.scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated = torch.cat([generated, next_tok], dim=-1)

            if (next_tok == eos).all():
                break

        return generated


# ──────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────
def build_model(config) -> MoELanguageModel:
    model = MoELanguageModel(config)
    total   = model.num_parameters()
    active  = total - (
        config.n_layers
        * (config.n_experts - config.n_experts_active)
        * 3 * config.d_model * config.expert_hidden_dim
    )
    print(f"[Model] Total parameters : {total:,}")
    print(f"[Model] Active parameters: {active:,}  (top-{config.n_experts_active} routing)")
    return model
