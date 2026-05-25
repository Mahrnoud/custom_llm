"""
scripts/prepare_data.py
──────────────────────────────────────────────────────────────────────────────
Downloads and prepares datasets for tokenizer training and Stage-1 pre-training.

Dataset mix:
  Web crawl  (60%)  → FineWeb CC-MAIN-2025-26 + FineWeb-Edu CC-MAIN-2025-26
  Books      (20%)  → Project Gutenberg (HF)
  Wikipedia  (15%)  → Wikimedia/Wikipedia (English)
  News       ( 5%)  → CC-News via HuggingFaceFW/fineweb with news filter
                      (cc_news has no HF namespace — using reliable proxy)

Usage:
  pip install datasets huggingface_hub tqdm

  Laptop test run (10 000 balanced records):
      python scripts/prepare_data.py --mode test --total 10000

  Full training run:
      python scripts/prepare_data.py --mode full --total 5000000
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("[ERROR] Run: pip install datasets huggingface_hub tqdm")

from tqdm import tqdm


# ── Stage-1 data mix ──────────────────────────────────────────────────────
STAGE1_MIX = {
    "english_web_crawl": 0.60,
    "books":             0.20,
    "wikipedia_en":      0.15,
    "news":              0.05,
}

MIN_DOC_CHARS = 200


# ── Dataset source registry ───────────────────────────────────────────────
@dataclass
class DatasetSource:
    hf_path:    str
    hf_name:    Optional[str]
    hf_split:   str
    text_field: str


SOURCES = {
    # Web
    "fineweb": DatasetSource(
        hf_path="HuggingFaceFW/fineweb",
        hf_name="CC-MAIN-2025-26",
        hf_split="train",
        text_field="text",
    ),
    "fineweb_edu": DatasetSource(
        hf_path="HuggingFaceFW/fineweb-edu",
        hf_name="CC-MAIN-2025-26",
        hf_split="train",
        text_field="text",
    ),
    # Books — Project Gutenberg
    "gutenberg": DatasetSource(
        hf_path="sedthh/gutenberg_english",
        hf_name=None,
        hf_split="train",
        text_field="TEXT",
    ),
    # Wikipedia
    "wikipedia": DatasetSource(
        hf_path="wikimedia/wikipedia",
        hf_name="20231101.en",
        hf_split="train",
        text_field="text",
    ),
    # News — cc_news has no HF namespace (legacy dataset, broken in datasets>=2.x)
    # Using these in priority order:
    #   1. cc_news (old datasets lib compatible)
    #   2. mdsiber/cc-news-2024 (community mirror)
    #   3. FineWeb filtered to news-like URLs as last resort
    "news_primary": DatasetSource(
        hf_path="cc_news",
        hf_name=None,
        hf_split="train",
        text_field="text",
    ),
    "news_fallback": DatasetSource(
        hf_path="mdsiber/cc-news-2024",
        hf_name=None,
        hf_split="train",
        text_field="text",
    ),
}


# ── Streaming helper ──────────────────────────────────────────────────────
def stream_texts(source: DatasetSource, n: int, seed: int = 42) -> Iterator[str]:
    """Stream up to n clean documents. No full download."""
    label = source.hf_path + (f" [{source.hf_name}]" if source.hf_name else "")
    print(f"    ↳ streaming {n:,} docs from {label}")
    try:
        kwargs = dict(streaming=True, split=source.hf_split, trust_remote_code=False)
        if source.hf_name:
            kwargs["name"] = source.hf_name
        ds = load_dataset(source.hf_path, **kwargs).shuffle(seed=seed, buffer_size=10_000)
    except Exception as e:
        print(f"    [WARN] Could not load {source.hf_path}: {e}")
        return

    collected = 0
    for row in ds:
        text = row.get(source.text_field, "")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if len(text) < MIN_DOC_CHARS:
            continue
        yield text
        collected += 1
        if collected >= n:
            break

    if collected < n:
        print(f"    [WARN] Only got {collected:,}/{n:,} from {source.hf_path}")


def stream_news(n: int, seed: int) -> list[str]:
    """
    Try cc_news first; fall back to mdsiber/cc-news-2024 if that fails.
    If both fail, fills from FineWeb (web data is better than empty news).
    """
    texts = list(tqdm(stream_texts(SOURCES["news_primary"], n, seed),
                      total=n, desc="CC-News (primary)"))
    if len(texts) >= n // 2:
        return texts[:n]

    print(f"    [INFO] Primary news failed or partial ({len(texts)} docs). Trying fallback …")
    more = list(tqdm(stream_texts(SOURCES["news_fallback"], n - len(texts), seed),
                     total=n - len(texts), desc="CC-News (fallback)"))
    texts += more

    if len(texts) < n // 2:
        print(f"    [WARN] Both news sources failed ({len(texts)} docs).")
        print(f"    [INFO] Filling news bucket from FineWeb to avoid an empty split.")
        needed = n - len(texts)
        fill = list(tqdm(stream_texts(SOURCES["fineweb"], needed, seed + 1),
                         total=needed, desc="FineWeb fill for news"))
        texts += fill

    random.seed(seed)
    random.shuffle(texts)
    return texts[:n]


# ── Bucket collectors ─────────────────────────────────────────────────────
def collect_web(n: int, seed: int) -> list[str]:
    n_fw  = int(n * 0.50)
    n_edu = n - n_fw
    texts  = list(tqdm(stream_texts(SOURCES["fineweb"],     n_fw,  seed), total=n_fw,  desc="FineWeb"))
    texts += list(tqdm(stream_texts(SOURCES["fineweb_edu"], n_edu, seed), total=n_edu, desc="FineWeb-Edu"))
    random.seed(seed)
    random.shuffle(texts)
    return texts[:n]


def collect_books(n: int, seed: int) -> list[str]:
    """100% Gutenberg. (pile-uncopyrighted uses zstd compression — broken in datasets>=2.x)"""
    texts = list(tqdm(stream_texts(SOURCES["gutenberg"], n, seed), total=n, desc="Gutenberg"))
    return texts[:n]


def collect_wikipedia(n: int, seed: int) -> list[str]:
    return list(tqdm(stream_texts(SOURCES["wikipedia"], n, seed), total=n, desc="Wikipedia"))


# ── File writers ──────────────────────────────────────────────────────────
def write_jsonl(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    size_kb = path.stat().st_size / 1024
    print(f"    ✓ Wrote {len(texts):,} docs  ({size_kb:,.0f} KB) → {path}")


# ── Main orchestrator ─────────────────────────────────────────────────────
def build_and_save(out_root: str, total: int, seed: int) -> dict[str, list[str]]:
    counts = {k: max(1, int(total * v)) for k, v in STAGE1_MIX.items()}
    counts["english_web_crawl"] += total - sum(counts.values())  # fix rounding

    print(f"\n{'─'*60}")
    print(f"  Stage-1 dataset plan  (total={total:,})")
    for bucket, n in counts.items():
        print(f"    {bucket:25s}  {n:6,}  ({n/total*100:.1f}%)")
    print(f"{'─'*60}\n")

    bucket_texts: dict[str, list[str]] = {}

    print("[1/4] Web crawl …")
    bucket_texts["english_web_crawl"] = collect_web(counts["english_web_crawl"], seed)

    print("\n[2/4] Books …")
    bucket_texts["books"] = collect_books(counts["books"], seed)

    print("\n[3/4] Wikipedia …")
    bucket_texts["wikipedia_en"] = collect_wikipedia(counts["wikipedia_en"], seed)

    print("\n[4/4] News …")
    bucket_texts["news"] = stream_news(counts["news"], seed)

    stage1_dir = Path(out_root) / "stage1"
    tok_dir    = Path(out_root) / "tokenizer_corpus"

    print(f"\n{'─'*60}  Writing Stage-1 …")
    for bucket, texts in bucket_texts.items():
        write_jsonl(stage1_dir / bucket / "data.jsonl", texts)

    print(f"\n{'─'*60}  Writing tokenizer corpus …")
    for bucket, texts in bucket_texts.items():
        share = max(1, int(total * STAGE1_MIX[bucket]))
        sample = texts[:share]
        random.seed(seed)
        random.shuffle(sample)
        write_jsonl(tok_dir / f"{bucket}.jsonl", sample)

    return bucket_texts


def print_summary(bucket_texts: dict, out_root: str):
    stage1_dir = Path(out_root) / "stage1"
    tok_dir    = Path(out_root) / "tokenizer_corpus"

    total_docs  = sum(len(v) for v in bucket_texts.values())
    total_chars = sum(sum(len(t) for t in v) for v in bucket_texts.values())

    print(f"\n{'═'*60}")
    print(f"  SUMMARY")
    print(f"  {total_docs:,} docs  |  {total_chars/1e6:.1f}M chars")
    print(f"\n  Stage-1  ({stage1_dir}):")
    for bucket in bucket_texts:
        p = stage1_dir / bucket / "data.jsonl"
        n = len(bucket_texts[bucket])
        kb = p.stat().st_size / 1024 if p.exists() else 0
        print(f"    {bucket:25s}  {n:6,} docs  {kb:8,.0f} KB")
    print(f"\n  Tokenizer corpus  ({tok_dir}):")
    for fp in sorted(tok_dir.glob("*.jsonl")):
        lines = sum(1 for _ in open(fp))
        kb = fp.stat().st_size / 1024
        print(f"    {fp.name:35s}  {lines:6,} docs  {kb:8,.0f} KB")
    print(f"\n  Next steps:")
    print(f"    python tokenizer/train_tokenizer.py \\")
    print(f"        --data_dir {tok_dir} --output tokenizer/ --verify")
    print(f"    python training/stage1_pretrain.py \\")
    print(f"        --data_dir {stage1_dir} --tokenizer tokenizer/")
    print(f"{'═'*60}\n")


def parse_args():
    p = argparse.ArgumentParser(description="Download & prepare datasets")
    p.add_argument("--mode",  choices=["test","full"], default="test")
    p.add_argument("--total", type=int, default=None,
                   help="Total records (default: 10000 for test, 5000000 for full)")
    p.add_argument("--out_dir", default="data")
    p.add_argument("--seed",  type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.total is None:
        args.total = 10_000 if args.mode == "test" else 5_000_000

    print(f"\n{'═'*60}")
    print(f"  prepare_data.py   mode={args.mode}   total={args.total:,}")
    print(f"{'═'*60}")

    random.seed(args.seed)
    bucket_texts = build_and_save(args.out_dir, args.total, args.seed)
    print_summary(bucket_texts, args.out_dir)
