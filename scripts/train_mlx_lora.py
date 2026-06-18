#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real MLX LoRA fine-tune.")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--data", default="data/training/mlx")
    parser.add_argument("--adapter-path", default="outputs/adapters/philosophy-compressor")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--learning-rate", default="1e-5")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--no-mask-prompt", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lora_cmd = shutil.which("mlx_lm.lora")
    if not lora_cmd:
        for candidate in (
            Path(sys.prefix) / "bin" / "mlx_lm.lora",
            Path(sys.executable).parent / "mlx_lm.lora",
            ROOT / ".venv" / "bin" / "mlx_lm.lora",
        ):
            if candidate.exists():
                lora_cmd = str(candidate)
                break
    if not lora_cmd:
        print("Missing mlx_lm.lora. Install requirements.txt inside the active virtualenv.", file=sys.stderr)
        return 2

    data_dir = ROOT / args.data
    if not (data_dir / "train.jsonl").exists():
        print(f"Missing {data_dir / 'train.jsonl'}; run scripts/prepare_training_data.py first.", file=sys.stderr)
        return 2

    adapter_path = ROOT / args.adapter_path
    adapter_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        lora_cmd,
        "--model",
        args.model,
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
    ]
    if not args.no_mask_prompt:
        command.append("--mask-prompt")
    if args.grad_checkpoint:
        command.append("--grad-checkpoint")

    print("running:", " ".join(command))
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
