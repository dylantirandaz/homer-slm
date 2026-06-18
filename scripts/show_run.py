#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvp.io import read_jsonl, word_count


def latest_essay_run(runs_dir: Path) -> Path | None:
    if not runs_dir.exists():
        return None
    candidates = [
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and (path / "README.md").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


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

    latest_run = latest_essay_run(ROOT / "outputs/runs")
    if latest_run:
        print("\nRun essays")
        print("-" * 10)
        print(str(latest_run.relative_to(ROOT) / "README.md"))
        essay_paths = sorted((latest_run / "essays").glob("gen_*.md"))
        if essay_paths:
            final_essay = essay_paths[-1]
            excerpt = final_essay.read_text(encoding="utf-8").strip().replace("\n", " ")
            print(f"latest essay: {final_essay.relative_to(ROOT)}")
            print(excerpt[:700])
        return 0

    paragraph = ROOT / "outputs/final_paragraph.txt"
    sentence = ROOT / "outputs/final_sentence.txt"
    if paragraph.exists():
        print("\nLegacy final paragraph")
        print("-" * 22)
        print(paragraph.read_text(encoding="utf-8").strip())
    if sentence.exists():
        print("\nLegacy final sentence")
        print("-" * 21)
        print(sentence.read_text(encoding="utf-8").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
