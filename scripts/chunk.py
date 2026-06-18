#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvp.io import read_jsonl, word_count, write_jsonl
from mvp.text import chunk_words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk corpus records into generation-0 units.")
    parser.add_argument("--input", default="data/processed/corpus.jsonl")
    parser.add_argument("--out", default="data/processed/chunks.jsonl")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = list(read_jsonl(ROOT / args.input))
    chunks = []
    for record in records:
        for index, text in enumerate(chunk_words(record["text"], args.chunk_size, args.overlap)):
            chunks.append(
                {
                    "chunk_id": f"{record['text_id']}_{index:05d}",
                    "text_id": record["text_id"],
                    "title": record["title"],
                    "author": record["author"],
                    "tradition": record["tradition"],
                    "school": record["school"],
                    "generation": 0,
                    "chunk_index": index,
                    "word_count": word_count(text),
                    "text": text,
                }
            )

    count = write_jsonl(ROOT / args.out, chunks)
    print(f"wrote={count} chunks out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

