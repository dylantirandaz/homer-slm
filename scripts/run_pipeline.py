#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvp.compression import MlxGenerator, compress_generation, write_final_outputs
from mvp.essays import write_generation_essay, write_essay_index
from mvp.io import read_jsonl, write_jsonl


def parse_schedule(raw: str, generations: int) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) < generations:
        raise ValueError("schedule must have at least as many entries as --generations")
    return values[:generations]


def default_run_id() -> str:
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recursive philosophy compression.")
    parser.add_argument("--input", default="data/processed/chunks.jsonl")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--schedule", default="500,250,120,60,25")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--adapter-path", default="outputs/adapters/philosophy-compressor")
    parser.add_argument("--generations-dir", default="data/generations")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-essays", action="store_true")
    parser.add_argument("--essay-adapter-path", default="")
    parser.add_argument("--essay-word-count", type=int, default=750)
    parser.add_argument("--essay-max-input-words", type=int, default=2400)
    parser.add_argument("--essay-max-tokens", type=int, default=None)
    parser.add_argument("--write-legacy-final", action="store_true")
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
    run_id = args.run_id or default_run_id()
    run_dir = ROOT / args.runs_dir / run_id
    essays_dir = run_dir / "essays"
    essay_records = []
    essay_generator = None
    essay_adapter = str(ROOT / args.essay_adapter_path) if args.essay_adapter_path else None
    if not args.skip_essays:
        essay_generator = MlxGenerator(
            model_name=args.model,
            adapter_path=essay_adapter,
            max_tokens=args.essay_max_tokens or max(420, int(args.essay_word_count * 1.35)),
            repetition_penalty=1.25,
            repetition_context_size=256,
        )
        print(f"run_id={run_id}")
        print(f"essays_dir={essays_dir.relative_to(ROOT)}")

    for generation in range(1, args.generations + 1):
        target = schedule[generation - 1]
        out_dir = ROOT / args.generations_dir / f"gen_{generation:03d}"
        output_path = out_dir / "units.jsonl"
        parent_path = input_path
        print(f"\n=== Generation {generation} target_words={target} ===")
        summary = compress_generation(
            input_path=parent_path,
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

        if essay_generator:
            essay_record = write_generation_essay(
                parent_path=parent_path,
                generation_path=output_path,
                essay_path=essays_dir / f"gen_{generation:03d}.md",
                model_name=args.model,
                compression_adapter_path=args.adapter_path,
                essay_adapter_path=essay_adapter,
                generator=essay_generator,
                target_word_count=args.essay_word_count,
                max_input_words=args.essay_max_input_words,
                max_tokens=args.essay_max_tokens,
            )
            essay_records.append(essay_record)
            print(
                f"[gen {generation}] essay={Path(essay_record['essay_path']).relative_to(ROOT)} "
                f"retention={essay_record['retention_ratio']:.2f}"
            )

    if essay_records:
        index_path = run_dir / "README.md"
        write_essay_index(
            records=essay_records,
            index_path=index_path,
            run_id=run_id,
            model_name=args.model,
            compression_adapter_path=args.adapter_path,
            essay_adapter_path=essay_adapter,
        )
        print(f"\nRun essays: {index_path.relative_to(ROOT)}")

    if args.write_legacy_final:
        write_final_outputs(
            final_generation_path=input_path,
            model_name=args.model,
            adapter_path=adapter,
            paragraph_path=ROOT / "outputs/final_paragraph.txt",
            sentence_path=ROOT / "outputs/final_sentence.txt",
        )
        print("\nLegacy final paragraph: outputs/final_paragraph.txt")
        print("Legacy final sentence: outputs/final_sentence.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
