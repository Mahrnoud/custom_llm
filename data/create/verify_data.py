"""
scripts/verify_data.py
──────────────────────────────────────────────────────────────────────────────
Checks your data directories after running prepare_data.py.
Prints doc counts, average length, and a sample text from each bucket.

Usage:
    python scripts/verify_data.py
    python scripts/verify_data.py --data_root data/
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
from pathlib import Path


def check_dir(label: str, path: Path, sample_chars: int = 200) -> dict:
    if not path.exists():
        print(f"  ✗  {label:30s}  [MISSING]  {path}")
        return {}

    jsonl_files = sorted(path.glob("*.jsonl"))
    if not jsonl_files:
        print(f"  ✗  {label:30s}  [NO .jsonl FILES]  {path}")
        return {}

    total_docs = 0
    total_chars = 0
    sample = ""

    for fp in jsonl_files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text", "")
                total_docs  += 1
                total_chars += len(text)
                if not sample:
                    sample = text[:sample_chars].replace("\n", " ")

    avg = total_chars / total_docs if total_docs else 0
    size_mb = sum(fp.stat().st_size for fp in jsonl_files) / 1e6

    print(
        f"  ✓  {label:30s}  "
        f"{total_docs:6,} docs  "
        f"avg {avg:6.0f} chars  "
        f"{size_mb:6.2f} MB"
    )
    if sample:
        print(f"       sample: "{sample[:100]}…"")

    return {"docs": total_docs, "chars": total_chars, "mb": size_mb}


def main(data_root: str):
    root = Path(data_root)
    print(f"\n{'═'*70}")
    print(f"  Data verification  →  {root.resolve()}")
    print(f"{'═'*70}")

    # ── Stage-1 ───────────────────────────────────────────────────────────
    print("\nStage-1 buckets:")
    stage1_buckets = [
        "english_web_crawl",
        "books",
        "wikipedia_en",
        "news",
    ]
    s1_stats = {}
    for bucket in stage1_buckets:
        s1_stats[bucket] = check_dir(bucket, root / "stage1" / bucket)

    # ── Tokenizer corpus ──────────────────────────────────────────────────
    print("\nTokenizer corpus:")
    tok_dir = root / "tokenizer_corpus"
    if tok_dir.exists():
        for fp in sorted(tok_dir.glob("*.jsonl")):
            check_dir(fp.stem, tok_dir, sample_chars=120)
    else:
        print(f"  ✗  {str(tok_dir)} not found")

    # ── Totals ────────────────────────────────────────────────────────────
    total_docs  = sum(v.get("docs", 0) for v in s1_stats.values())
    total_chars = sum(v.get("chars", 0) for v in s1_stats.values())
    total_mb    = sum(v.get("mb", 0) for v in s1_stats.values())
    print(f"\n{'─'*70}")
    print(f"  Stage-1 totals: {total_docs:,} docs  |  {total_chars/1e6:.1f}M chars  |  {total_mb:.1f} MB")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    args = p.parse_args()
    main(args.data_root)
