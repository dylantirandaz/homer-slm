#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvp.io import read_jsonl, word_count


def main() -> int:
    generations_dir = ROOT / "data/generations"
    print("Minimum Viable Philosophy")
    print("=" * 27)

    if generations_dir.exists():
        for gen_dir in sorted(generations_dir.glob("gen_*")):
            path = gen_dir / "units.jsonl"
            if not path.exists():
                continue
            records = list(read_jsonl(path))
            words = sum(record.get("output_word_count", word_count(record.get("text", ""))) for record in records)
            print(f"{gen_dir.name}: units={len(records):4d} words={words:7d}")
            if records:
                sample = records[-1].get("text", "").replace("\n", " ")
                print(f"  latest: {sample[:220]}")

    paragraph = ROOT / "outputs/final_paragraph.txt"
    sentence = ROOT / "outputs/final_sentence.txt"
    if paragraph.exists():
        print("\nFinal paragraph")
        print("-" * 15)
        print(paragraph.read_text(encoding="utf-8").strip())
    if sentence.exists():
        print("\nFinal sentence")
        print("-" * 14)
        print(sentence.read_text(encoding="utf-8").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

