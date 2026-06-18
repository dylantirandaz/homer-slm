#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odyssey.scratch_model import ByteGPT, ScratchConfig, parameter_count
from odyssey.text import extract_odyssey_body


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a byte-level GPT from scratch on the Odyssey.")
    parser.add_argument("--input", default="data/raw/odyssey.txt")
    parser.add_argument("--out", default="outputs/scratch/odyssey-byte-gpt")
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-size", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--mlp-dim", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=300)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def load_bytes(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/download_odyssey.py first.")
    text = extract_odyssey_body(path.read_text(encoding="utf-8"))
    data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    if len(data) < 10_000:
        raise ValueError("Training text is unexpectedly short.")
    return data


def split_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(len(data) * 0.90)
    valid_end = int(len(data) * 0.95)
    return data[:train_end], data[train_end:valid_end], data[valid_end:]


def get_batch(data: np.ndarray, rng: np.random.Generator, batch_size: int, context_size: int) -> tuple[mx.array, mx.array]:
    starts = rng.integers(0, len(data) - context_size - 1, size=batch_size)
    x = np.stack([data[start : start + context_size] for start in starts])
    y = np.stack([data[start + 1 : start + context_size + 1] for start in starts])
    return mx.array(x), mx.array(y)


def loss_fn(model: ByteGPT, x: mx.array, y: mx.array) -> mx.array:
    logits = model(x)
    return nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1),
        reduction="mean",
    )


def evaluate(
    model: ByteGPT,
    valid_data: np.ndarray,
    rng: np.random.Generator,
    batch_size: int,
    context_size: int,
    batches: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = get_batch(valid_data, rng, batch_size, context_size)
        loss = loss_fn(model, x, y)
        mx.eval(loss)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(model: ByteGPT, out_dir: Path, config: ScratchConfig, metadata: dict, step: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = out_dir / "weights.safetensors"
    step_weights = out_dir / f"{step:06d}_weights.safetensors"
    model.save_weights(str(weights))
    model.save_weights(str(step_weights))
    payload = {"config": config.to_dict(), **metadata, "step": step}
    (out_dir / "metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved={weights.relative_to(ROOT)} step_weights={step_weights.relative_to(ROOT)}")


def main() -> int:
    args = parse_args()
    if args.d_model % args.heads:
        print("--d-model must be divisible by --heads", file=sys.stderr)
        return 2

    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = load_bytes(ROOT / args.input)
    train_data, valid_data, test_data = split_data(data)
    config = ScratchConfig(
        context_size=args.context_size,
        n_layers=args.layers,
        n_heads=args.heads,
        d_model=args.d_model,
        mlp_dim=args.mlp_dim,
    )
    model = ByteGPT(config)
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    loss_and_grad = nn.value_and_grad(model, lambda x, y: loss_fn(model, x, y))

    metadata = {
        "objective": "from_scratch_byte_language_modeling",
        "source": args.input,
        "source_bytes": int(len(data)),
        "train_bytes": int(len(train_data)),
        "valid_bytes": int(len(valid_data)),
        "test_bytes": int(len(test_data)),
        "parameters": parameter_count(model),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    }
    print(
        "scratch_pretrain source_bytes={source_bytes} train={train_bytes} valid={valid_bytes} "
        "params={parameters}".format(**metadata)
    )

    start = time.time()
    last_loss = None
    for step in range(1, args.iters + 1):
        x, y = get_batch(train_data, rng, args.batch_size, args.context_size)
        loss, grads = loss_and_grad(x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        last_loss = float(loss.item())

        if step == 1 or step % 20 == 0:
            elapsed = max(time.time() - start, 1e-6)
            tokens = step * args.batch_size * args.context_size
            print(
                f"iter={step} train_loss={last_loss:.3f} "
                f"bytes_per_sec={tokens / elapsed:.0f}"
            )

        if step % args.eval_every == 0 or step == args.iters:
            val_loss = evaluate(
                model,
                valid_data,
                rng,
                args.batch_size,
                args.context_size,
                args.eval_batches,
            )
            print(f"iter={step} val_loss={val_loss:.3f}")
            metadata["last_train_loss"] = last_loss
            metadata["last_val_loss"] = val_loss

        if step % args.save_every == 0:
            save_checkpoint(model, ROOT / args.out, config, metadata, step)

    metadata["elapsed_seconds"] = round(time.time() - start, 3)
    save_checkpoint(model, ROOT / args.out, config, metadata, args.iters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
