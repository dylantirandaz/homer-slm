#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvp.compression import compress_generation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress one generation with an MLX model.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--target-word-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--adapter-path", default="outputs/adapters/philosophy-compressor")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-input-words", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapter = str(ROOT / args.adapter_path) if args.adapter_path else None
    summary = compress_generation(
        input_path=ROOT / args.input,
        output_path=ROOT / args.output,
        generation=args.generation,
        target_word_count=args.target_word_count,
        batch_size=args.batch_size,
        model_name=args.model,
        adapter_path=adapter,
        force=args.force,
        max_tokens=args.max_tokens,
        max_input_words=args.max_input_words,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
