#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvp.io import read_jsonl, word_count, write_jsonl
from mvp.prompts import TRAINING_SYSTEM_PROMPT
from mvp.text import extractive_philosophy_digest, truncate_words


def make_example(record: dict, input_word_limit: int, target_word_count: int) -> dict | None:
    input_text = truncate_words(record["text"], input_word_limit)
    output_text = extractive_philosophy_digest(input_text, target_word_count)
    if word_count(output_text) < 30:
        return None
    user_prompt = (
        "Compress this philosophical passage while preserving its argument, concepts, "
        "and unresolved tensions. Remove examples and ornament.\n\n"
        f"Target length: {target_word_count} words\n\n"
        f"Passage:\n{input_text}"
    )
    return {
        "messages": [
            {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": output_text},
        ],
        "source_id": record.get("chunk_id"),
        "source_text_id": record.get("text_id"),
        "target_word_count": target_word_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MLX chat-format LoRA training data.")
    parser.add_argument("--input", default="data/processed/chunks.jsonl")
    parser.add_argument("--out-dir", default="data/training/mlx")
    parser.add_argument("--input-word-limit", type=int, default=900)
    parser.add_argument("--target-word-count", type=int, default=140)
    parser.add_argument("--max-examples", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--valid-size", type=float, default=0.08)
    parser.add_argument("--test-size", type=float, default=0.08)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = list(read_jsonl(ROOT / args.input))
    examples = []
    for record in records:
        example = make_example(record, args.input_word_limit, args.target_word_count)
        if example:
            examples.append(example)

    random.Random(args.seed).shuffle(examples)
    if args.max_examples:
        examples = examples[: args.max_examples]

    total = len(examples)
    test_count = max(1, int(total * args.test_size))
    valid_count = max(1, int(total * args.valid_size))
    test = examples[:test_count]
    valid = examples[test_count : test_count + valid_count]
    train = examples[test_count + valid_count :]

    out_dir = ROOT / args.out_dir
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    write_jsonl(out_dir / "test.jsonl", test)
    print(f"train={len(train)} valid={len(valid)} test={len(test)} out={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

