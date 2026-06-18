#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUANTIZED_MODEL_MARKERS = ("2bit", "4bit", "8bit", "q2", "q4", "q8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLX fine-tuning for the Odyssey SLM.")
    parser.add_argument("--model-id", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--data", default="data/training/mlx")
    parser.add_argument("--adapter-path", default="outputs/adapters/odyssey-qwen25-0.5b")
    parser.add_argument("--fine-tune-type", choices=("lora", "dora", "full"), default="lora")
    parser.add_argument("--optimizer", choices=("adam", "adamw", "muon", "sgd", "adafactor"), default="adam")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--learning-rate", default="1e-5")
    parser.add_argument("--val-batches", type=int)
    parser.add_argument("--test-batches", type=int)
    parser.add_argument("--steps-per-report", type=int)
    parser.add_argument("--steps-per-eval", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mask-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-checkpoint", action="store_true")
    return parser.parse_args()


def plain_text_dataset(data_dir: Path) -> bool:
    with (data_dir / "train.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            return "text" in sample and "messages" not in sample and "prompt" not in sample
    return False


def main() -> int:
    args = parse_args()
    if "GLM-5.2" in args.model_id:
        print(
            "Note: GLM-5.2 is a 753B-class checkpoint. This script only works if your "
            "environment can load the model with mlx_lm.lora.",
            file=sys.stderr,
        )
    if args.fine_tune_type == "full" and any(marker in args.model_id.lower() for marker in QUANTIZED_MODEL_MARKERS):
        print(
            "Full-parameter fine-tuning requires a non-quantized model. "
            "Use mlx-community/Qwen2.5-0.5B-Instruct-bf16 for the default small Qwen run.",
            file=sys.stderr,
        )
        return 2
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
    if args.mask_prompt and plain_text_dataset(data_dir):
        print("Plain-text dataset detected; disabling prompt masking.", file=sys.stderr)
        args.mask_prompt = False
    adapter_path = ROOT / args.adapter_path
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        lora_cmd,
        "--model",
        args.model_id,
        "--train",
        "--data",
        str(data_dir),
        "--fine-tune-type",
        args.fine_tune_type,
        "--optimizer",
        args.optimizer,
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
    if args.mask_prompt:
        command.append("--mask-prompt")
    if args.grad_checkpoint:
        command.append("--grad-checkpoint")
    optional_ints = (
        ("--val-batches", args.val_batches),
        ("--test-batches", args.test_batches),
        ("--steps-per-report", args.steps_per_report),
        ("--steps-per-eval", args.steps_per_eval),
        ("--save-every", args.save_every),
        ("--max-seq-length", args.max_seq_length),
        ("--seed", args.seed),
    )
    for option, value in optional_ints:
        if value is not None:
            command.extend([option, str(value)])
    print("running:", " ".join(command))
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
