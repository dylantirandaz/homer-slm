#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvp.analysis import concept_coverage, understanding_prompt
from mvp.compression import MlxGenerator, get_record_id
from mvp.io import read_jsonl, utc_now, word_count, write_jsonl
from mvp.text import truncate_words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit concept absorption and loss across generations.")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--generations-dir", default="data/generations")
    parser.add_argument("--out-jsonl", default="outputs/runs/absorption_trace.jsonl")
    parser.add_argument("--out-md", default="outputs/runs/absorption_trace.md")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument(
        "--adapter-path",
        default="outputs/adapters/philosophy-compressor",
        help="Compression adapter path to record in the report.",
    )
    parser.add_argument(
        "--commentary-adapter-path",
        default="",
        help="Optional adapter for audit commentary. Defaults to the base instruct model.",
    )
    parser.add_argument("--max-source-words", type=int, default=900)
    parser.add_argument("--max-commentary-tokens", type=int, default=180)
    parser.add_argument("--max-units-per-generation", type=int, default=None)
    parser.add_argument("--no-model-commentary", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(read_jsonl(path))


def build_record_index(chunks_path: Path, generations_dir: Path) -> dict[str, dict]:
    index = {}
    for record in load_records(chunks_path):
        index[get_record_id(record)] = record

    for gen_dir in sorted(generations_dir.glob("gen_*")):
        for record in load_records(gen_dir / "units.jsonl"):
            index[get_record_id(record)] = record
    return index


def generation_paths(generations_dir: Path) -> list[Path]:
    return [
        gen_dir / "units.jsonl"
        for gen_dir in sorted(generations_dir.glob("gen_*"))
        if (gen_dir / "units.jsonl").exists()
    ]


def parent_text(record: dict, index: dict[str, dict]) -> tuple[str, list[str]]:
    parent_ids = record.get("parent_ids") or []
    missing = []
    texts = []
    for parent_id in parent_ids:
        parent = index.get(parent_id)
        if not parent:
            missing.append(parent_id)
            continue
        text = parent.get("text", "")
        if text:
            texts.append(text)
    return "\n\n---\n\n".join(texts), missing


def markdown_report(records: list[dict], args: argparse.Namespace) -> str:
    lines = [
        "# Absorption Trace",
        "",
        f"Generated: {utc_now()}",
        "",
        f"Model: `{args.model}`",
        f"Compression adapter: `{args.adapter_path}`",
        f"Commentary adapter: `{args.commentary_adapter_path or 'none; base model'}`",
        "",
        "This report compares each compressed unit with its parent text. The concept metrics are lexical and deterministic; the model-understanding notes are deterministic base-model output using greedy decoding unless a commentary adapter is provided.",
        "",
        "## Generation Summary",
        "",
        "| Generation | Units | Avg Concept Retention | Avg Source Words | Avg Output Words |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    by_generation: dict[int, list[dict]] = {}
    for record in records:
        by_generation.setdefault(record["generation"], []).append(record)

    for generation, generation_records in sorted(by_generation.items()):
        count = len(generation_records)
        avg_retention = sum(record["coverage"]["retention_ratio"] for record in generation_records) / count
        avg_source_words = sum(record["coverage"]["source_word_count"] for record in generation_records) / count
        avg_output_words = sum(record["coverage"]["output_word_count"] for record in generation_records) / count
        lines.append(
            f"| {generation} | {count} | {avg_retention:.2f} | {avg_source_words:.0f} | {avg_output_words:.0f} |"
        )

    for record in records:
        coverage = record["coverage"]
        retained = ", ".join(coverage["retained_concepts"][:20]) or "none detected"
        lost = ", ".join(coverage["lost_concepts"][:20]) or "none detected"
        introduced = ", ".join(coverage["introduced_concepts"][:12]) or "none detected"
        lines.extend(
            [
                "",
                f"## gen_{record['generation']:03d} / {record['unit_id']}",
                "",
                f"Parents: `{', '.join(record['parent_ids'])}`",
                "",
                f"Concept retention: `{coverage['retention_ratio']:.2f}`",
                "",
                f"Retained: {retained}",
                "",
                f"Lost or muted: {lost}",
                "",
                f"Introduced: {introduced}",
                "",
                "Model understanding of compressed child:",
                "",
                record.get("model_understanding", "").strip() or "_Skipped._",
                "",
                "Compressed child:",
                "",
                record["compressed_text"].strip(),
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    chunks_path = ROOT / args.chunks
    generations_dir = ROOT / args.generations_dir
    paths = generation_paths(generations_dir)
    if not paths:
        print(f"No generation files found in {generations_dir}. Run scripts/run_pipeline.py first.", file=sys.stderr)
        return 2

    index = build_record_index(chunks_path, generations_dir)
    generator = None
    if not args.no_model_commentary:
        adapter = str(ROOT / args.commentary_adapter_path) if args.commentary_adapter_path else None
        generator = MlxGenerator(args.model, adapter_path=adapter, max_tokens=args.max_commentary_tokens)

    trace_records = []
    for path in paths:
        unit_records = load_records(path)
        if args.max_units_per_generation:
            unit_records = unit_records[: args.max_units_per_generation]

        for record in unit_records:
            source_text, missing_parent_ids = parent_text(record, index)
            compressed_text = record.get("text", "")
            coverage = concept_coverage(source_text, compressed_text)
            commentary = ""
            if generator:
                prompt = understanding_prompt(output_text=compressed_text, coverage=coverage)
                commentary = generator.generate(prompt, max_tokens=args.max_commentary_tokens)

            trace_record = {
                "timestamp": utc_now(),
                "generation": record.get("generation"),
                "unit_id": record.get("unit_id"),
                "parent_ids": record.get("parent_ids", []),
                "missing_parent_ids": missing_parent_ids,
                "source_excerpt": truncate_words(source_text, args.max_source_words),
                "compressed_text": compressed_text,
                "source_word_count": word_count(source_text),
                "compressed_word_count": word_count(compressed_text),
                "coverage": coverage,
                "model": args.model,
                "compression_adapter_path": args.adapter_path,
                "commentary_adapter_path": args.commentary_adapter_path or None,
                "model_understanding": commentary,
            }
            trace_records.append(trace_record)
            print(
                f"[trace] gen={trace_record['generation']} unit={trace_record['unit_id']} "
                f"retention={coverage['retention_ratio']:.2f}"
            )

    out_jsonl = ROOT / args.out_jsonl
    out_md = ROOT / args.out_md
    write_jsonl(out_jsonl, trace_records)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown_report(trace_records, args), encoding="utf-8")
    print(f"wrote={len(trace_records)} jsonl={out_jsonl.relative_to(ROOT)} md={out_md.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
