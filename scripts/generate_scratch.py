#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odyssey.scratch_model import ByteGPT, ScratchConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a scratch-trained Odyssey byte GPT.")
    parser.add_argument("--checkpoint", default="outputs/scratch/odyssey-byte-gpt")
    parser.add_argument("--prompt", default="Tell me, O Muse,")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def sample_next(logits: np.ndarray, rng: np.random.Generator, temperature: float, top_k: int) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits.astype(np.float64) / temperature
    if top_k > 0 and top_k < logits.shape[0]:
        keep = np.argpartition(logits, -top_k)[-top_k:]
        filtered = np.full_like(logits, -np.inf)
        filtered[keep] = logits[keep]
        logits = filtered
    probs = softmax(logits)
    return int(rng.choice(np.arange(logits.shape[0]), p=probs))


def main() -> int:
    args = parse_args()
    checkpoint = ROOT / args.checkpoint
    metadata_path = checkpoint / "metadata.json"
    weights_path = checkpoint / "weights.safetensors"
    if not metadata_path.exists() or not weights_path.exists():
        print(f"Missing scratch checkpoint at {checkpoint}", file=sys.stderr)
        return 2

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = ScratchConfig.from_dict(metadata["config"])
    model = ByteGPT(config)
    model.load_weights(str(weights_path))
    model.eval()

    rng = np.random.default_rng(args.seed)
    output = list(args.prompt.encode("utf-8", errors="replace"))
    if not output:
        output = [ord("T")]

    for _ in range(args.tokens):
        context = np.array(output[-config.context_size :], dtype=np.int32)[None, :]
        logits = model(mx.array(context))[0, -1]
        mx.eval(logits)
        next_byte = sample_next(np.array(logits), rng, args.temperature, args.top_k)
        output.append(next_byte)

    print(bytes(output).decode("utf-8", errors="replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
