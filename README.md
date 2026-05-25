# English MoE-LLM — Training Pipeline (2026)

A **~250M-parameter** English-only language model trained from scratch, featuring:

- **Mixture-of-Experts (MoE)** FFN with sparse top-2 routing
- **Grouped-Query Attention (GQA)** with RoPE (extended base)
- **FlashAttention-2** support (graceful fallback to SDPA)
- **Three-stage curriculum training**
- **Custom BPE tokenizer** with STEM + code domain support

---

## Architecture at a Glance

| Component | Setting |
|---|---|
| Hidden dim (`d_model`) | 768 |
| Transformer layers | 12 |
| Attention heads (Q) | 12 |
| KV heads (GQA) | 3 |
| Head dim | 64 |
| MoE experts (total) | 8 |
| MoE experts (active) | 2 |
| Expert hidden dim | 1 536 |
| Context length | 2 048 |
| Vocabulary size | 32 768 |
| **Total params** | **~250 M** |
| **Active params/token** | **~85 M** |
| Positional encoding | RoPE θ=500 000 |
| Activation | SwiGLU |
| Normalisation | RMSNorm (pre-norm) |

---

## Project Structure

```
llm_project/
├── config/
│   └── model_config.py        ← All hyperparameters (model + all 3 stages)
├── tokenizer/
│   └── train_tokenizer.py     ← BPE tokenizer training script
├── model/
│   ├── attention.py           ← GQA + RoPE + FlashAttention-2
│   ├── moe.py                 ← Sparse MoE layer + aux losses
│   └── architecture.py        ← Full transformer + generation
├── data/
│   └── dataset.py             ← TextDataset, ToolUseDataset, collator
├── training/
│   ├── trainer.py             ← Base trainer (AMP, grad accum, checkpointing)
│   ├── stage1_pretrain.py     ← Stage 1: General pre-training
│   ├── stage2_stem.py         ← Stage 2: STEM specialisation
│   └── stage3_tools.py        ← Stage 3: Tool-use training
└── tools/
    └── tool_registry.py       ← Tool definitions + execution engine
```

---

## Installation

```bash
# Create environment
conda create -n moellm python=3.11 -y
conda activate moellm

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tokenizers transformers datasets
pip install numpy tqdm

# Optional: FlashAttention-2 (CUDA 11.8+ required)
pip install flash-attn --no-build-isolation

# Optional: Logging
pip install wandb tensorboard
```

---

## Step 0 — Train the Tokenizer

```bash
python tokenizer/train_tokenizer.py \
    --data_dir  data/tokenizer_corpus \
    --output    tokenizer/ \
    --vocab_size 32768 \
    --verify
```

**Data layout for tokenizer training:**
```
data/tokenizer_corpus/
    english_web.jsonl         ← {"text": "..."}  one document per line
    books.jsonl
    wikipedia.jsonl
    math_papers.jsonl
    code.jsonl
```

The tokenizer will be saved as:
```
tokenizer/
    vocab.json
    merges.txt
    tokenizer.json            ← main HuggingFace tokenizers file
    tokenizer_config.json
    special_tokens_map.json
```

---

## Step 1 — General Pre-Training

```bash
python training/stage1_pretrain.py \
    --output_dir  checkpoints/stage1 \
    --data_dir    data/stage1 \
    --tokenizer   tokenizer/
```

**Recommended data layout:**
```
data/stage1/
    english_web_crawl/        ← 60% — C4, Common Crawl (English filtered)
    books/                    ← 20% — Project Gutenberg, OpenLibrary
    wikipedia_en/             ← 15% — Wikipedia (English)
    news/                     ←  5% — CC-News, RealNews
```

Each subdirectory can contain:
- `corpus.bin` — pre-tokenised uint16 binary (fastest)
- `*.jsonl` — `{"text": "..."}` one document per line

**Pre-tokenise a JSONL file to binary (recommended for large corpora):**
```python
import numpy as np
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer/tokenizer.json")
tokens = []
with open("data/stage1/english_web_crawl/data.jsonl") as f:
    for line in f:
        import json; obj = json.loads(line)
        tokens.extend([1] + tok.encode(obj["text"]).ids + [2])

np.array(tokens, dtype=np.uint16).tofile("data/stage1/english_web_crawl/corpus.bin")
```

**Stage 1 hyperparameters (from `config/model_config.py`):**
- Steps: 100 000
- Batch size: 32 × 8 accumulation = effective 256
- Learning rate: 3e-4 (cosine decay, 2 000 warmup steps)
- Context: 2 048 tokens

---

## Step 2 — STEM Specialisation

```bash
python training/stage2_stem.py \
    --output_dir   checkpoints/stage2 \
    --data_dir     data/stage2 \
    --tokenizer    tokenizer/ \
    --stage1_ckpt  checkpoints/stage1/checkpoint_best.pt
```

**Data layout:**
```
data/stage2/
    mathematics/     ← 30% — ArXiv math, Lean proofs, textbooks
    physics/         ← 20% — ArXiv physics, lecture notes
    chemistry/       ← 15% — Chemical journals, textbooks
    code/            ← 25% — GitHub (Python/C++/Rust), StackOverflow
    general_replay/  ← 10% — Subset of Stage-1 data (anti-forgetting)
```

**Stage 2 hyperparameters:**
- Steps: 30 000
- Learning rate: 1e-4 (reduced for fine-tuning)
- Batch size: 16 × 8 accumulation = effective 128

---

## Step 3 — Tool-Use Training

```bash
# Optional: Generate synthetic tool-use data first
python training/stage3_tools.py --gen_data

# Then train
python training/stage3_tools.py \
    --output_dir   checkpoints/stage3 \
    --data_dir     data/stage3 \
    --tokenizer    tokenizer/ \
    --stage2_ckpt  checkpoints/stage2/checkpoint_best.pt
```

**Data layout:**
```
data/stage3/
    tool_use_synthetic/   ← 40% — Auto-generated tool-call traces
    search_qa/            ← 25% — Search → answer pairs
    retrieval_augmented/  ← 25% — RAG-style (context + question + answer)
    general_replay/       ← 10% — Replay to prevent forgetting
```

**Tool-use JSONL format:**
```jsonl
{"user": "What is the boiling point of ethanol?",
 "assistant": "<think>I should verify this precisely.</think>\n<tool_call>{\"name\": \"web_search\", \"query\": \"boiling point of ethanol Celsius\"}</tool_call>\n<tool_result>Ethanol boils at 78.37 °C (173.1 °F) at standard pressure.</tool_result>\nEthanol has a boiling point of 78.37 °C at 1 atm."}
```

**Available tools at inference time:**
| Tool | Description |
|---|---|
| `web_search` | Search the web for current information |
| `fetch_url` | Retrieve and read a web page |
| `calculator` | Evaluate mathematical expressions |
| `python_exec` | Run sandboxed Python code |
| `summarise` | Extractive text summarisation |

---

## Inference Example

```python
import torch
from tokenizers import Tokenizer
from model.architecture import MoELanguageModel
from config.model_config import ModelConfig

# Load
cfg   = ModelConfig()
model = MoELanguageModel(cfg)
ckpt  = torch.load("checkpoints/stage3/checkpoint_best.pt")
model.load_state_dict(ckpt["model_state"])
model.eval().cuda()

tok = Tokenizer.from_file("tokenizer/tokenizer.json")

# Generate
prompt = "<|user|>\nWhat is the integral of sin(x) from 0 to π?\n<|end|>\n<|assistant|>\n"
ids    = torch.tensor([tok.encode(prompt).ids]).cuda()
out    = model.generate(ids, max_new_tokens=200, temperature=0.7)
print(tok.decode(out[0].tolist()))
```

---

## Hardware Requirements

| Stage | Min GPU Memory | Recommended |
|---|---|---|
| Tokenizer training | CPU only | — |
| Stage 1 (100k steps) | 24 GB | 2 × A100 80 GB |
| Stage 2 (30k steps) | 16 GB | 1 × A100 40 GB |
| Stage 3 (10k steps) | 16 GB | 1 × A100 40 GB |
| Inference (fp16) | 4 GB | 8 GB |

Enable bfloat16 (`bf16=True` in all stage configs) for ~50% memory reduction.

---

## Key Design Decisions

### Why MoE?
Sparse MoE allows the total parameter count (~250M) to far exceed the
parameters active per token (~85M), giving better capacity per FLOP.

### Why GQA?
Grouped-Query Attention (n_heads=12, n_kv_heads=3) reduces KV-cache memory
by 4× vs standard MHA with minimal quality loss.

### Why RoPE θ=500,000?
The extended base frequency enables better length generalisation out-of-distribution,
following the approach used in Llama-3 and similar 2024-2025 models.

### Why three stages?
- **Stage 1** builds broad language understanding from diverse English text.
- **Stage 2** injects deep STEM knowledge without overwriting general ability
  (10% replay prevents catastrophic forgetting).
- **Stage 3** teaches tool use on a small but focused dataset so it doesn't
  degrade STEM or general performance.