"""
tokenizer/train_tokenizer.py
─────────────────────────────────────────────────────────────────────────────
Trains a Byte-Pair Encoding (BPE) tokenizer from scratch using the
HuggingFace `tokenizers` library.

Supported domains:
  • General English text
  • Mathematical symbols  (∑ ∫ ∂ √ ∞ ≤ ≥ ≠ ∈ ∉ ∀ ∃ …)
  • Physics / Chemistry   (subscripts, superscripts, element symbols, SI units)
  • Programming syntax    (operators, brackets, keywords, indentation tokens)

Usage:
  python tokenizer/train_tokenizer.py \
      --data_dir  data/tokenizer_corpus \
      --output    tokenizer/ \
      --vocab_size 32768
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterator

from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from tokenizers.implementations import ByteLevelBPETokenizer


# ──────────────────────────────────────────────────────────────────────────
# Special Tokens
# ──────────────────────────────────────────────────────────────────────────
SPECIAL_TOKENS = [
    "<pad>",           # id=0  – padding
    "<bos>",           # id=1  – beginning of sequence
    "<eos>",           # id=2  – end of sequence
    "<unk>",           # id=3  – unknown
    # Tool-use (Stage 3)
    "<tool_call>",     # open  a tool invocation block
    "</tool_call>",    # close a tool invocation block
    "<tool_result>",   # open  a tool-result block
    "</tool_result>",  # close a tool-result block
    "<search>",        # web-search query block
    "</search>",       # end search query
    "<think>",         # chain-of-thought scratch-pad
    "</think>",        # end chain-of-thought
    # Role markers (instruction fine-tuning ready)
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|end|>",
]

# ──────────────────────────────────────────────────────────────────────────
# Domain-Specific Unicode Characters to Force-Add to the Vocabulary
# ──────────────────────────────────────────────────────────────────────────
MATH_CHARS = list(
    "∑∫∂√∞≤≥≠≈∈∉∀∃⊂⊃⊆⊇∩∪∧∨¬→←↔⇒⇔⟨⟩"
    "αβγδεζηθικλμνξπρστυφχψω"
    "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ"
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    "·×÷±∓ℕℤℚℝℂ∇△□■▷◁"
)

CHEM_CHARS = list(
    "⇌⇀↽⟶⟵⟷⊕⊖→←↑↓"  # reaction arrows
)

# SI unit strings we want kept as single tokens
SI_UNIT_TOKENS = [
    "kg", "mol", "km/h", "m/s", "m/s²", "kJ/mol",
    "°C", "°F", "°K", "µm", "nm", "pm",
    "MHz", "GHz", "THz", "kPa", "MPa", "GPa",
]

# Common programming token fragments
CODE_TOKENS = [
    "def ", "class ", "import ", "from ", "return ", "yield ",
    "lambda ", "async ", "await ", "with ", "while ", "for ",
    "if ", "elif ", "else:", "try:", "except ", "finally:",
    "True", "False", "None", "self.", "cls.",
    "->", "::", "!=", "==", "<=", ">=", "**", "//",
    "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
    "<<", ">>", "&&", "||",
    "print(", "len(", "range(", "enumerate(",
    "np.", "pd.", "torch.", "tf.", "plt.",
    "```python", "```cpp", "```js", "```bash", "```",
]


# ──────────────────────────────────────────────────────────────────────────
# Text Iterator over a Directory of `.txt` / `.jsonl` Files
# ──────────────────────────────────────────────────────────────────────────
def text_iterator(data_dir: str, batch_size: int = 1000) -> Iterator[list[str]]:
    """
    Yields batches of strings from:
      • Plain text (.txt) files — whole lines
      • JSON-Lines  (.jsonl)   — reads the "text" field
    """
    data_path = Path(data_dir)
    files = sorted(
        list(data_path.rglob("*.txt")) + list(data_path.rglob("*.jsonl"))
    )
    if not files:
        raise FileNotFoundError(f"No .txt or .jsonl files found in {data_dir}")

    print(f"[Tokenizer] Found {len(files)} files in {data_dir}")
    batch: list[str] = []

    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="replace") as fh:
            if fp.suffix == ".jsonl":
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", obj.get("content", ""))
                    except json.JSONDecodeError:
                        continue
                    if text:
                        batch.append(text)
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
            else:
                for line in fh:
                    line = line.strip()
                    if line:
                        batch.append(line)
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []

    if batch:
        yield batch


# ──────────────────────────────────────────────────────────────────────────
# Build & Train Tokenizer
# ──────────────────────────────────────────────────────────────────────────
def build_tokenizer(vocab_size: int = 32_768) -> ByteLevelBPETokenizer:
    """
    Constructs a ByteLevelBPE tokenizer with:
      • Byte-level fallback (never produces <unk> for any UTF-8 input)
      • Whitespace normalisation (NFD → strip accents is intentionally OFF
        to preserve math/chemistry Unicode)
      • Split-on-whitespace-and-punctuation pre-tokeniser so that math
        operators and code symbols become their own merge candidates
    """
    tokenizer = ByteLevelBPETokenizer(
        add_prefix_space=False,
        lowercase=False,           # Preserve case (important for code + formulas)
        unicode_normalizer="nfc",  # NFC keeps composed Unicode (e.g. Å stays Å)
    )
    return tokenizer


def train_tokenizer(
    data_dir: str,
    output_dir: str,
    vocab_size: int = 32_768,
    min_frequency: int = 2,
    batch_size: int = 1000,
) -> None:
    """
    Train the BPE tokenizer, then post-process it:
      1. Add all special tokens.
      2. Inject domain-specific characters/tokens into the vocabulary.
      3. Save in HuggingFace-compatible format.
    """
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = build_tokenizer(vocab_size)

    # Collect all mandatory vocabulary items
    initial_alphabet = list(
        set(MATH_CHARS + CHEM_CHARS)
    )

    print(f"[Tokenizer] Training BPE on {data_dir}")
    print(f"            vocab_size={vocab_size}, min_frequency={min_frequency}")

    # The ByteLevelBPETokenizer.train_from_iterator takes an iterable of
    # strings (not batches), so we flatten.
    def flat_iter():
        for batch in text_iterator(data_dir, batch_size=batch_size):
            yield from batch

    tokenizer.train_from_iterator(
        flat_iter(),
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # ── Post-training: Add domain tokens that may have been missed ────────
    # We manually add them; the tokenizer will assign them the next
    # available IDs if they are not already present.
    domain_additions = SI_UNIT_TOKENS + CODE_TOKENS
    tokenizer.add_tokens(domain_additions)

    # ── Save ──────────────────────────────────────────────────────────────
    tokenizer.save_model(output_dir)

    # Also save vocab + merges in a single JSON for HuggingFace `tokenizers`
    wrapped = tokenizer._tokenizer  # access underlying tokenizers.Tokenizer
    wrapped.save(os.path.join(output_dir, "tokenizer.json"))

    # Save special tokens map for transformers compatibility
    special_map = {
        "pad_token": "<pad>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "additional_special_tokens": SPECIAL_TOKENS[4:],
    }
    with open(os.path.join(output_dir, "special_tokens_map.json"), "w") as f:
        json.dump(special_map, f, indent=2, ensure_ascii=False)

    # Save tokenizer config
    tokenizer_config = {
        "model_type": "byte_level_bpe",
        "vocab_size": tokenizer.get_vocab_size(),
        "add_prefix_space": False,
        "lowercase": False,
        "unicode_normalizer": "nfc",
        "special_tokens": SPECIAL_TOKENS,
    }
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

    print(f"\n[Tokenizer] ✓ Saved to {output_dir}")
    print(f"            Final vocab size: {tokenizer.get_vocab_size():,}")


# ──────────────────────────────────────────────────────────────────────────
# Verification Utility
# ──────────────────────────────────────────────────────────────────────────
def verify_tokenizer(tokenizer_dir: str) -> None:
    """Run a quick smoke-test on the trained tokenizer."""
    from tokenizers import Tokenizer as HFTokenizer

    tok = HFTokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))

    test_cases = [
        # General English
        "The quick brown fox jumps over the lazy dog.",
        # Math
        "Let f(x) = ∑_{n=0}^{∞} xⁿ/n! = eˣ, where x ∈ ℝ.",
        # Physics
        "F = ma, and E = mc², where c ≈ 3×10⁸ m/s.",
        # Chemistry
        "2H₂ + O₂ ⇌ 2H₂O   ΔH° = −483.6 kJ/mol",
        # Python code
        "def fibonacci(n: int) -> int:\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        # C++ code
        "std::vector<int> v = {1, 2, 3};\nfor (auto& x : v) { x *= 2; }",
        # Tool-call token
        "<tool_call>{\"name\": \"web_search\", \"query\": \"speed of light\"}</tool_call>",
    ]

    print("\n[Tokenizer Verification]")
    print("─" * 60)
    for text in test_cases:
        enc = tok.encode(text)
        decoded = tok.decode(enc.ids, skip_special_tokens=False)
        n_tok = len(enc.ids)
        ok = "✓" if decoded == text else "✗"
        print(f"{ok} tokens={n_tok:4d}  {repr(text[:55])}")

    # Check special token IDs
    vocab = tok.get_vocab()
    print("\nSpecial token IDs:")
    for st in SPECIAL_TOKENS[:8]:
        print(f"  {st!r:25s} → {vocab.get(st, 'MISSING')}")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train BPE tokenizer for English-STEM LLM")
    p.add_argument("--data_dir",  required=True, help="Directory with .txt/.jsonl training files")
    p.add_argument("--output",    default="tokenizer/", help="Where to save the tokenizer")
    p.add_argument("--vocab_size",type=int, default=32_768)
    p.add_argument("--min_freq",  type=int, default=2)
    p.add_argument("--batch_size",type=int, default=1000, help="Streaming batch size")
    p.add_argument("--verify",    action="store_true", help="Run verification after training")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_tokenizer(
        data_dir=args.data_dir,
        output_dir=args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq,
        batch_size=args.batch_size,
    )
    if args.verify:
        verify_tokenizer(args.output)
