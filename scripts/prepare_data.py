#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odyssey.io import write_jsonl
from odyssey.prompts import SYSTEM_PROMPT, training_user_prompt
from odyssey.text import chunk_words, extractive_summary, normalize_whitespace, word_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Odyssey chunks and MLX training data.")
    parser.add_argument("--input", default="data/raw/odyssey.txt")
    parser.add_argument("--chunks-out", default="data/processed/chunks.jsonl")
    parser.add_argument("--training-dir", default="data/training/mlx")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=90)
    parser.add_argument("--target-words", type=int, default=180)
    parser.add_argument("--max-examples", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = ROOT / args.input
    if not source.exists():
        print(f"Missing {source}; run scripts/download_odyssey.py first.", file=sys.stderr)
        return 2
    text = normalize_whitespace(source.read_text(encoding="utf-8"))
    chunks = []
    for index, chunk in enumerate(chunk_words(text, args.chunk_size, args.overlap)):
        chunks.append(
            {
                "chunk_id": f"odyssey_{index:05d}",
                "text_id": "odyssey",
                "title": "The Odyssey",
                "author": "Homer",
                "translation": "Samuel Butler",
                "generation": 0,
                "chunk_index": index,
                "word_count": word_count(chunk),
                "text": chunk,
            }
        )
    write_jsonl(ROOT / args.chunks_out, chunks)

    examples = []
    for record in chunks:
        target = extractive_summary(record["text"], args.target_words)
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": training_user_prompt(record["text"], args.target_words)},
                    {"role": "assistant", "content": target},
                ],
                "source_id": record["chunk_id"],
                "target_word_count": args.target_words,
            }
        )
    random.Random(args.seed).shuffle(examples)
    if args.max_examples:
        examples = examples[: args.max_examples]
    valid_count = max(1, int(len(examples) * 0.08))
    test_count = max(1, int(len(examples) * 0.08))
    test = examples[:test_count]
    valid = examples[test_count : test_count + valid_count]
    train = examples[test_count + valid_count :]
    out_dir = ROOT / args.training_dir
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    write_jsonl(out_dir / "test.jsonl", test)
    print(f"chunks={len(chunks)} train={len(train)} valid={len(valid)} test={len(test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

