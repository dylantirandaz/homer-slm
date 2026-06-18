#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odyssey.io import write_jsonl
from odyssey.text import chunk_words, extract_odyssey_body, word_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare plain-text Odyssey data for continued pretraining.")
    parser.add_argument("--input", default="data/raw/odyssey.txt")
    parser.add_argument("--training-dir", default="data/training/pretrain")
    parser.add_argument("--sample-words", type=int, default=384)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--max-examples", type=int, default=0, help="0 keeps every generated sample.")
    parser.add_argument("--valid-fraction", type=float, default=0.08)
    parser.add_argument("--test-fraction", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


def split_records(records: list[dict], valid_fraction: float, test_fraction: float) -> tuple[list[dict], list[dict], list[dict]]:
    test_count = max(1, int(len(records) * test_fraction))
    valid_count = max(1, int(len(records) * valid_fraction))
    test = records[:test_count]
    valid = records[test_count : test_count + valid_count]
    train = records[test_count + valid_count :]
    return train, valid, test


def main() -> int:
    args = parse_args()
    source = ROOT / args.input
    if not source.exists():
        print(f"Missing {source}; run scripts/download_odyssey.py first.", file=sys.stderr)
        return 2
    text = extract_odyssey_body(source.read_text(encoding="utf-8"))
    if args.overlap >= args.sample_words:
        print("--overlap must be smaller than --sample-words", file=sys.stderr)
        return 2

    records = []
    for index, passage in enumerate(chunk_words(text, args.sample_words, args.overlap)):
        records.append(
            {
                "text": passage,
                "source_id": f"odyssey_pretrain_{index:05d}",
                "word_count": word_count(passage),
            }
        )

    rng = random.Random(args.seed)
    rng.shuffle(records)
    if args.max_examples > 0:
        records = records[: args.max_examples]

    train, valid, test = split_records(records, args.valid_fraction, args.test_fraction)
    out_dir = ROOT / args.training_dir
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    write_jsonl(out_dir / "test.jsonl", test)
    metadata = {
        "input": str(source.relative_to(ROOT)),
        "training_dir": str(out_dir.relative_to(ROOT)),
        "objective": "plain_text_continued_pretraining",
        "sample_words": args.sample_words,
        "overlap": args.overlap,
        "source_word_count": word_count(text),
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "source_words={source_word_count} train={train} valid={valid} test={test}".format(
            **metadata
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
