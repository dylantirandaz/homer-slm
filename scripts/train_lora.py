#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLX LoRA training for Odyssey summarization.")
    parser.add_argument("--model-id", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--data", default="data/training/mlx")
    parser.add_argument("--adapter-path", default="outputs/adapters/odyssey-qwen25-0.5b")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--learning-rate", default="1e-5")
    parser.add_argument("--grad-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if "GLM-5.2" in args.model_id:
        print(
            "Note: GLM-5.2 is a 753B-class checkpoint. This script only works if your "
            "environment can load the model with mlx_lm.lora.",
            file=sys.stderr,
        )
    lora_cmd = shutil.which("mlx_lm.lora")
    if not lora_cmd:
        candidate = ROOT / ".venv" / "bin" / "mlx_lm.lora"
        if candidate.exists():
            lora_cmd = str(candidate)
    if not lora_cmd:
        print("Missing mlx_lm.lora. Install requirements-mlx.txt in the active environment.", file=sys.stderr)
        return 2
    data_dir = ROOT / args.data
    if not (data_dir / "train.jsonl").exists():
        print(f"Missing {data_dir / 'train.jsonl'}; run scripts/prepare_data.py first.", file=sys.stderr)
        return 2
    adapter_path = ROOT / args.adapter_path
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        lora_cmd,
        "--model",
        args.model_id,
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapter_path),
        "--iters",
        str(args.iters),
        "--batch-size",
        str(args.batch_size),
        "--num-layers",
        str(args.num_layers),
        "--learning-rate",
        str(args.learning_rate),
        "--mask-prompt",
    ]
    if args.grad_checkpoint:
        command.append("--grad-checkpoint")
    print("running:", " ".join(command))
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
