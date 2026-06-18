#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvp.compression import compress_generation, write_final_outputs
from mvp.io import read_jsonl, write_jsonl


def parse_schedule(raw: str, generations: int) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) < generations:
        raise ValueError("schedule must have at least as many entries as --generations")
    return values[:generations]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recursive philosophy compression.")
    parser.add_argument("--input", default="data/processed/chunks.jsonl")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--schedule", default="500,250,120,60,25")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--adapter-path", default="outputs/adapters/philosophy-compressor")
    parser.add_argument("--generations-dir", default="data/generations")
    parser.add_argument("--max-initial-records", type=int, default=None)
    parser.add_argument("--per-text-chunks", type=int, default=None)
    parser.add_argument("--max-input-words", type=int, default=1100)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def select_initial_records(input_path: Path, args: argparse.Namespace) -> Path:
    if not args.max_initial_records and not args.per_text_chunks:
        return input_path

    records = list(read_jsonl(input_path))
    selected = []
    per_text_counts: dict[str, int] = {}
    for record in records:
        if args.per_text_chunks:
            text_id = record.get("text_id", "unknown")
            count = per_text_counts.get(text_id, 0)
            if count >= args.per_text_chunks:
                continue
            per_text_counts[text_id] = count + 1
        selected.append(record)
        if args.max_initial_records and len(selected) >= args.max_initial_records:
            break

    out_path = ROOT / "data/processed/pipeline_input.jsonl"
    write_jsonl(out_path, selected)
    print(f"selected_initial_records={len(selected)} path={out_path.relative_to(ROOT)}")
    return out_path


def main() -> int:
    args = parse_args()
    schedule = parse_schedule(args.schedule, args.generations)
    adapter = str(ROOT / args.adapter_path) if args.adapter_path else None
    input_path = select_initial_records(ROOT / args.input, args)

    for generation in range(1, args.generations + 1):
        target = schedule[generation - 1]
        out_dir = ROOT / args.generations_dir / f"gen_{generation:03d}"
        output_path = out_dir / "units.jsonl"
        print(f"\n=== Generation {generation} target_words={target} ===")
        summary = compress_generation(
            input_path=input_path,
            output_path=output_path,
            generation=generation,
            target_word_count=target,
            batch_size=args.batch_size,
            model_name=args.model,
            adapter_path=adapter,
            force=args.force,
            max_input_words=args.max_input_words,
        )
        print(summary)
        input_path = output_path

        remaining = len(list(read_jsonl(output_path)))
        print(f"[gen {generation}] units={remaining} path={output_path.relative_to(ROOT)}")

    write_final_outputs(
        final_generation_path=input_path,
        model_name=args.model,
        adapter_path=adapter,
        paragraph_path=ROOT / "outputs/final_paragraph.txt",
        sentence_path=ROOT / "outputs/final_sentence.txt",
    )
    print("\nFinal paragraph: outputs/final_paragraph.txt")
    print("Final sentence: outputs/final_sentence.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
