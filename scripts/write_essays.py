#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvp.essays import write_generation_essays


def default_run_id() -> str:
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write per-generation research essays.")
    parser.add_argument("--initial-input", default="data/processed/pipeline_input.jsonl")
    parser.add_argument("--fallback-input", default="data/processed/chunks.jsonl")
    parser.add_argument("--generations-dir", default="data/generations")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--compression-adapter-path", default="outputs/adapters/philosophy-compressor")
    parser.add_argument("--essay-adapter-path", default="")
    parser.add_argument("--essay-word-count", type=int, default=750)
    parser.add_argument("--essay-max-input-words", type=int, default=2400)
    parser.add_argument("--essay-max-tokens", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or default_run_id()
    initial_path = ROOT / args.initial_input
    if not initial_path.exists():
        initial_path = ROOT / args.fallback_input

    run_dir = ROOT / args.runs_dir / run_id
    essays_dir = run_dir / "essays"
    index_path = run_dir / "README.md"
    essay_adapter = str(ROOT / args.essay_adapter_path) if args.essay_adapter_path else None

    records = write_generation_essays(
        initial_path=initial_path,
        generations_dir=ROOT / args.generations_dir,
        essays_dir=essays_dir,
        index_path=index_path,
        run_id=run_id,
        model_name=args.model,
        compression_adapter_path=args.compression_adapter_path,
        essay_adapter_path=essay_adapter,
        target_word_count=args.essay_word_count,
        max_input_words=args.essay_max_input_words,
        max_tokens=args.essay_max_tokens,
    )
    for record in records:
        print(
            f"[essay] gen={record['generation']} retention={record['retention_ratio']:.2f} "
            f"path={Path(record['essay_path']).relative_to(ROOT)}"
        )
    print(f"wrote={len(records)} index={index_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
