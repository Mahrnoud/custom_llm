"""
scripts/clean_data.py
──────────────────────────────────────────────────────────────────────────────
Cleans JSONL files produced by prepare_data.py.

Bug fixes vs previous version:
  1. Now COUNTS and REPORTS actual replacements made per document
     (before: applied fixes silently so it looked like nothing happened)
  2. Handles LITERAL "NBSP" string from Gutenberg OCR artefacts
     (before: only handled unicode U+00A0 — Gutenberg has the 4-letter word)
  3. Handles literal "\\r\\n" escape sequences left un-decoded in some sources

Usage:
    python scripts/clean_data.py                     # in-place (overwrites)
    python scripts/clean_data.py --out_dir data_clean/  # separate output dir
    python scripts/clean_data.py --dry_run           # stats only, no writes
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import re
from pathlib import Path


MIN_CHARS = 150

# Project Gutenberg boundary markers
PG_START_RE = re.compile(r"\*{3}\s*START OF (THE |THIS )?PROJECT GUTENBERG", re.IGNORECASE)
PG_END_RE   = re.compile(r"\*{3}\s*END OF (THE |THIS )?PROJECT GUTENBERG",   re.IGNORECASE)

# HTML entities found in web/news crawl data
HTML_ENTITIES = {
    "&amp;":   "&",
    "&lt;":    "<",
    "&gt;":    ">",
    "&quot;":  '"',
    "&#39;":   "'",
    "&nbsp;":  " ",
    "&mdash;": "—",
    "&ndash;": "–",
    "&hellip;":"…",
}


# ── Fix functions — each returns (fixed_text, n_replacements) ─────────────
def fix_line_endings(text: str):
    """\\r\\n and stray \\r → \\n"""
    orig = text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Also handle literal escape sequences left un-decoded in some sources
    text = text.replace("\\r\\n", "\n").replace("\\r", "\n").replace("\\n", "\n")
    return text, (orig != text)


def fix_nbsp(text: str):
    """Both unicode U+00A0 AND the 4-letter OCR artefact 'NBSP'."""
    orig = text
    text = text.replace("\u00a0", " ")                     # real unicode NBSP
    text = re.sub(r"(?m)^\s*NBSP\s*$", "", text)           # Gutenberg literal "NBSP" on its own line
    text = re.sub(r"(?<!\w)NBSP(?!\w)", " ", text)         # "NBSP" mid-sentence
    return text, (orig != text)


def fix_zero_width(text: str):
    """Strip invisible formatting characters."""
    orig = text
    ZW   = "\u200b\u200c\u200d\ufeff\u2060"
    text = text.translate(str.maketrans("", "", ZW))
    return text, (orig != text)


def collapse_blank_lines(text: str):
    """3+ consecutive blank lines → 2."""
    orig = text
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, (orig != text)


def collapse_spaces(text: str):
    """4+ consecutive spaces → 1  (HTML table/pre artefact)."""
    orig = text
    text = re.sub(r" {4,}", " ", text)
    return text, (orig != text)


def fix_html_entities(text: str):
    orig = text
    for entity, char in HTML_ENTITIES.items():
        text = text.replace(entity, char)
    text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
    return text, (orig != text)


def strip_gutenberg_boilerplate(text: str):
    orig = text
    start_m = PG_START_RE.search(text)
    if start_m:
        body_start = text.find("\n", start_m.end())
        if body_start != -1:
            text = text[body_start + 1:]

    end_m = PG_END_RE.search(text)
    if end_m:
        text = text[: end_m.start()]

    text = text.strip()
    return text, (orig != text)


# ── Per-bucket cleaners ───────────────────────────────────────────────────
def clean_text(text: str, bucket: str) -> tuple[str, dict]:
    """Clean text and return (cleaned_text, counts_of_each_fix_applied)."""
    fixes = {}

    if "books" in bucket:
        text, f = strip_gutenberg_boilerplate(text);  fixes["pg_boilerplate"] = int(f)

    text, f = fix_line_endings(text);   fixes["line_endings"] = int(f)
    text, f = fix_nbsp(text);           fixes["nbsp"]         = int(f)
    text, f = fix_zero_width(text);     fixes["zero_width"]   = int(f)
    text, f = collapse_blank_lines(text); fixes["blank_lines"] = int(f)

    if "books" not in bucket:
        text, f = fix_html_entities(text);  fixes["html_entities"] = int(f)
        text, f = collapse_spaces(text);    fixes["space_collapse"] = int(f)

    return text.strip(), fixes


# ── File processor ────────────────────────────────────────────────────────
def process_file(src: Path, dst: Path, bucket: str, dry_run: bool) -> dict:
    stats = {
        "in_docs":       0,
        "out_docs":      0,
        "docs_modified": 0,
        "dropped_short": 0,
        "dropped_empty": 0,
        "fix_counts":    {},
    }

    cleaned_lines = []

    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            stats["in_docs"] += 1
            text = obj.get("text", "")

            if not text:
                stats["dropped_empty"] += 1
                continue

            cleaned, fixes = clean_text(text, bucket)

            # Track which fixes fired
            if any(fixes.values()):
                stats["docs_modified"] += 1
                for fix_name, fired in fixes.items():
                    stats["fix_counts"][fix_name] = stats["fix_counts"].get(fix_name, 0) + fired

            if len(cleaned) < MIN_CHARS:
                stats["dropped_short"] += 1
                continue

            obj["text"] = cleaned
            cleaned_lines.append(json.dumps(obj, ensure_ascii=False))

    stats["out_docs"] = len(cleaned_lines)

    if not dry_run and cleaned_lines:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            f.write("\n".join(cleaned_lines) + "\n")

    return stats


# ── Main ──────────────────────────────────────────────────────────────────
def main(in_dir: str, out_dir: str, dry_run: bool):
    in_root  = Path(in_dir)
    out_root = Path(out_dir)

    print(f"\n{'═'*65}")
    print(f"  clean_data.py   dry_run={dry_run}")
    print(f"  in  → {in_root}    out → {out_root}")
    print(f"{'═'*65}\n")

    targets = [in_root / "stage1", in_root / "tokenizer_corpus"]

    grand = {"in": 0, "out": 0, "modified": 0, "dropped": 0}

    for target in targets:
        if not target.exists():
            continue

        jsonl_files = sorted(target.rglob("*.jsonl"))
        if not jsonl_files:
            continue

        print(f"[{target.relative_to(in_root)}]")

        for src in jsonl_files:
            rel    = src.relative_to(in_root)
            dst    = out_root / rel
            # For stage1/books/data.jsonl  → parent name "books" is the bucket.
            # For tokenizer_corpus/books.jsonl → parent is "tokenizer_corpus",
            # so fall back to the file stem ("books") to pick the right cleaner.
            bucket = src.parent.name if src.parent.name != "tokenizer_corpus" else src.stem

            s = process_file(src, dst, bucket, dry_run)

            dropped   = s["in_docs"] - s["out_docs"]
            pct_kept  = s["out_docs"] / max(s["in_docs"], 1) * 100
            pct_fixed = s["docs_modified"] / max(s["in_docs"], 1) * 100
            tag       = "[DRY] " if dry_run else "  ✓   "

            print(
                f"  {tag} {str(rel):45s}"
                f"  {s['in_docs']:6,} → {s['out_docs']:6,} docs"
                f"  ({pct_kept:.0f}% kept)"
                f"  {s['docs_modified']:5,} docs fixed  ({pct_fixed:.1f}%)"
            )

            # Show which fixes fired
            if s["fix_counts"]:
                detail = "  │  fixes: " + ", ".join(
                    f"{k}={v}" for k, v in s["fix_counts"].items() if v
                )
                print(detail)

            grand["in"]      += s["in_docs"]
            grand["out"]     += s["out_docs"]
            grand["modified"]+= s["docs_modified"]
            grand["dropped"] += dropped

        print()

    drop_pct = grand["dropped"] / max(grand["in"], 1) * 100
    mod_pct  = grand["modified"] / max(grand["in"], 1) * 100
    print(f"{'─'*65}")
    print(f"  Total in      : {grand['in']:,}")
    print(f"  Total out     : {grand['out']:,}  ({100-drop_pct:.1f}% kept)")
    print(f"  Docs modified : {grand['modified']:,}  ({mod_pct:.1f}% of input had at least one fix)")
    print(f"  Docs dropped  : {grand['dropped']:,}  ({drop_pct:.1f}% — too short after cleaning)")
    if not dry_run:
        print(f"  Output        : {out_root}/")
    print()


def parse_args():
    p = argparse.ArgumentParser(description="Clean Stage-1 and tokenizer JSONL files")
    p.add_argument("--in_dir",  default="data")
    p.add_argument("--out_dir", default="data",    help="Same as in_dir = in-place overwrite")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.in_dir, args.out_dir, args.dry_run)
