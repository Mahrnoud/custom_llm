"""
model_config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration for the ~250M-parameter English-only MoE-LLM.

Architecture highlights (2026 best-practices):
  • Grouped-Query Attention  (GQA) with RoPE
  • Mixture-of-Experts FFN   (top-2 sparse routing)
  • RMSNorm + SwiGLU
  • Flash-Attention 2 (when available)
  • Three-stage curriculum training
─────────────────────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    # ── Tokenizer ──────────────────────────────────────────────────────────
    vocab_size: int = 32_838          # BPE vocabulary size
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # ── Dimensions ─────────────────────────────────────────────────────────
    d_model: int = 768                # Hidden dimension
    n_layers: int = 12                # Transformer blocks
    max_seq_len: int = 2_048          # Maximum context window

    # ── Attention (GQA) ────────────────────────────────────────────────────
    n_heads: int = 12                 # Query heads
    n_kv_heads: int = 3               # Key/Value heads (GQA; must divide n_heads)
    head_dim: int = 64                # d_model // n_heads
    attention_dropout: float = 0.0    # Typically 0 for large LLMs
    use_flash_attn: bool = True       # Use FlashAttention-2 when available

    # ── RoPE ───────────────────────────────────────────────────────────────
    rope_theta: float = 500_000.0     # YaRN-style extended base for long context

    # ── Mixture-of-Experts FFN ─────────────────────────────────────────────
    n_experts: int = 8                # Total experts per layer
    n_experts_active: int = 2         # Active experts per token (top-k)
    expert_hidden_dim: int = 1_536    # FFN inner dimension per expert
    moe_dropout: float = 0.0
    router_z_loss_coeff: float = 1e-3 # Router z-loss for load balancing
    router_aux_loss_coeff: float = 1e-2

    # ── Normalisation ──────────────────────────────────────────────────────
    rms_norm_eps: float = 1e-5
    use_pre_norm: bool = True         # Pre-LN (more stable)

    # ── Regularisation ─────────────────────────────────────────────────────
    hidden_dropout: float = 0.0
    embed_dropout: float = 0.0

    # ── Misc ───────────────────────────────────────────────────────────────
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.head_dim = self.d_model // self.n_heads

    @property
    def approximate_params(self) -> int:
        """Rough parameter count estimate (total, not active)."""
        embed   = self.vocab_size * self.d_model
        attn    = self.n_layers * (
            self.d_model * self.d_model           # Q proj
            + 2 * self.d_model * (self.n_kv_heads * self.head_dim)  # K, V proj
            + self.d_model * self.d_model         # O proj
        )
        moe_ffn = self.n_layers * self.n_experts * (
            2 * self.d_model * self.expert_hidden_dim   # gate + up
            + self.expert_hidden_dim * self.d_model      # down
        )
        router  = self.n_layers * self.d_model * self.n_experts
        norms   = self.n_layers * 2 * self.d_model + self.d_model  # per-block + final
        return embed + attn + moe_ffn + router + norms


# ──────────────────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class TokenizerConfig:
    vocab_size: int = 32_838
    min_frequency: int = 2
    # Special tokens
    special_tokens: list = field(default_factory=lambda: [
        "<pad>", "<bos>", "<eos>", "<unk>",
        # Tool-call tokens (Stage 3)
        "<tool_call>", "</tool_call>",
        "<tool_result>", "</tool_result>",
        "<search>", "</search>",
        "<think>", "</think>",
    ])
    # Domain-aware pre-tokeniser patterns (regex)
    additional_patterns: list = field(default_factory=lambda: [
        r"\d+\.?\d*",                  # Numbers / floats
        r"[A-Za-z]+\d+|[A-Za-z]+_\d+", # Variable names like x1, var_2
        r"[+\-*/=<>!&|^~%]+",         # Operator clusters
        r"[{}()\[\],.;:\'\"]",         # Punctuation
    ])


# ──────────────────────────────────────────────────────────────────────────
# Stage 1 – General Pre-training
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Stage1Config:
    stage_name: str = "stage1_pretrain"

    # Data (fill in your actual paths)
    data_paths: list = field(default_factory=lambda: [
        "data/stage1/english_web_crawl",
        "data/stage1/books",
        "data/stage1/wikipedia_en",
        "data/stage1/news",
    ])
    data_weights: list = field(default_factory=lambda: [0.60, 0.20, 0.15, 0.05])

    # Optimiser
    learning_rate: float = 3e-4
    lr_scheduler: str = "cosine_with_warmup"
    warmup_steps: int = 2_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    grad_clip: float = 1.0

    # Training
    batch_size: int = 32              # per-device
    seq_len: int = 2_048
    gradient_accumulation_steps: int = 8
    max_steps: int = 100_000
    save_every: int = 2_000
    eval_every: int = 1_000
    log_every: int = 50

    # Precision / memory
    bf16: bool = True
    gradient_checkpointing: bool = True


# ──────────────────────────────────────────────────────────────────────────
# Stage 2 – STEM Specialisation
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Stage2Config:
    stage_name: str = "stage2_stem"

    data_paths: list = field(default_factory=lambda: [
        "data/stage2/mathematics",     # e.g. ArXiv math, textbooks
        "data/stage2/physics",         # Physics papers, textbooks
        "data/stage2/chemistry",       # Chemistry literature
        "data/stage2/code",            # GitHub, StackOverflow, coding books
        "data/stage2/general_replay",  # ~10 % Stage-1 data to avoid forgetting
    ])
    data_weights: list = field(default_factory=lambda: [0.30, 0.20, 0.15, 0.25, 0.10])

    # Smaller LR for fine-tuning
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine_with_warmup"
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    grad_clip: float = 1.0

    batch_size: int = 16
    seq_len: int = 2_048
    gradient_accumulation_steps: int = 8
    max_steps: int = 30_000
    save_every: int = 1_000
    eval_every: int = 500
    log_every: int = 25

    bf16: bool = True
    gradient_checkpointing: bool = True

    # Load the best Stage-1 checkpoint before starting
    resume_from_stage1: bool = True


# ──────────────────────────────────────────────────────────────────────────
# Stage 3 – Tool-Use Training
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Stage3Config:
    stage_name: str = "stage3_tools"

    data_paths: list = field(default_factory=lambda: [
        "data/stage3/tool_use_synthetic",  # Synthetically generated tool-call traces
        "data/stage3/search_qa",           # Search → answer pairs
        "data/stage3/retrieval_augmented", # RAG-style examples
        "data/stage3/general_replay",      # Replay to avoid forgetting
    ])
    data_weights: list = field(default_factory=lambda: [0.40, 0.25, 0.25, 0.10])

    learning_rate: float = 5e-5
    lr_scheduler: str = "cosine_with_warmup"
    warmup_steps: int = 200
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    grad_clip: float = 1.0

    batch_size: int = 8
    seq_len: int = 2_048
    gradient_accumulation_steps: int = 8
    max_steps: int = 10_000
    save_every: int = 500
    eval_every: int = 200
    log_every: int = 10

    bf16: bool = True
    gradient_checkpointing: bool = True
    resume_from_stage2: bool = True

    # Tool-call masking: only compute loss on assistant tokens
    mask_user_tokens: bool = True


# ──────────────────────────────────────────────────────────────────────────
# Quick instantiation helpers
# ──────────────────────────────────────────────────────────────────────────
def get_default_model_config() -> ModelConfig:
    cfg = ModelConfig()
    print(f"[Config] Approximate parameter count: {cfg.approximate_params:,}")
    return cfg


def get_all_configs():
    return (
        get_default_model_config(),
        TokenizerConfig(),
        Stage1Config(),
        Stage2Config(),
        Stage3Config(),
    )


if __name__ == "__main__":
    model_cfg, tok_cfg, s1, s2, s3 = get_all_configs()
    print(f"  d_model={model_cfg.d_model}, n_layers={model_cfg.n_layers}")
    print(f"  n_heads={model_cfg.n_heads}, n_kv_heads={model_cfg.n_kv_heads}")
    print(f"  n_experts={model_cfg.n_experts}, active={model_cfg.n_experts_active}")
    print(f"  vocab_size={model_cfg.vocab_size}")
    print(f"  Approx total params: {model_cfg.approximate_params:,}")
