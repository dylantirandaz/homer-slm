#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvp.io import read_jsonl, word_count, write_jsonl
from mvp.text import normalize_whitespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build corpus.jsonl from downloaded raw texts.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/processed/corpus.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = ROOT / args.raw_dir
    metadata_path = raw_dir / "metadata.jsonl"
    if not metadata_path.exists():
        print(f"Missing {metadata_path}; run scripts/download_corpus.py first.", file=sys.stderr)
        return 2

    records = []
    for meta in read_jsonl(metadata_path):
        source_path = ROOT / meta["source_path"]
        if not source_path.exists():
            print(f"[missing] {source_path}")
            continue
        text = normalize_whitespace(source_path.read_text(encoding="utf-8"))
        if not text:
            continue
        records.append({**meta, "text": text, "word_count": word_count(text)})

    count = write_jsonl(ROOT / args.out, records)
    total_words = sum(record["word_count"] for record in records)
    print(f"wrote={count} total_words={total_words} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

