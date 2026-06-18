#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odyssey.scratch_model import ByteGPT, ScratchConfig
from odyssey.text import ODYSSEY_TERMS, extract_odyssey_body


LOSS_CURVE = [
    {"step": 1000, "val_loss": 1.683},
    {"step": 2000, "val_loss": 1.358},
    {"step": 3000, "val_loss": 1.265},
    {"step": 4000, "val_loss": 1.236},
    {"step": 5000, "val_loss": 1.261},
    {"step": 6000, "val_loss": 1.330},
    {"step": 7000, "val_loss": 1.411},
    {"step": 8000, "val_loss": 1.553},
    {"step": 9000, "val_loss": 1.736},
    {"step": 10000, "val_loss": 1.894},
]


CHECKPOINTS = [
    ("2.5k", "002500_weights.safetensors"),
    ("5k", "005000_weights.safetensors"),
    ("7.5k", "007500_weights.safetensors"),
    ("10k", "010000_weights.safetensors"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write blog artifacts for the Odyssey scratch pretraining experiment.")
    parser.add_argument("--checkpoint", default="outputs/scratch/odyssey-byte-gpt-10k")
    parser.add_argument("--source", default="data/raw/odyssey.txt")
    parser.add_argument("--out", default="outputs/blog")
    parser.add_argument("--prompt", default="Tell me, O Muse,")
    parser.add_argument("--tokens", type=int, default=650)
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--top-k", type=int, default=20)
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
    return int(rng.choice(np.arange(logits.shape[0]), p=softmax(logits)))


def generate(model: ByteGPT, config: ScratchConfig, prompt: str, tokens: int, temperature: float, top_k: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    output = list(prompt.encode("utf-8", errors="replace")) or [ord("T")]
    for _ in range(tokens):
        context = np.array(output[-config.context_size :], dtype=np.int32)[None, :]
        logits = model(mx.array(context))[0, -1]
        mx.eval(logits)
        output.append(sample_next(np.array(logits), rng, temperature, top_k))
    return bytes(output).decode("utf-8", errors="replace")


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z'-]*", text.lower())


def ngrams(tokens: list[str], size: int) -> set[tuple[str, ...]]:
    if len(tokens) < size:
        return set()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def score_sample(sample: str, source_ngrams: set[tuple[str, ...]]) -> dict:
    tokens = tokenize_words(sample)
    eight_grams = ngrams(tokens, 8)
    overlap = eight_grams & source_ngrams
    term_counts = Counter(token for token in tokens if token in ODYSSEY_TERMS)
    repeated_bigram_count = sum(count - 1 for count in Counter(zip(tokens, tokens[1:])).values() if count > 1)
    return {
        "words": len(tokens),
        "unique_words": len(set(tokens)),
        "unique_ratio": round(len(set(tokens)) / max(len(tokens), 1), 3),
        "odyssey_terms": dict(term_counts.most_common()),
        "odyssey_term_hits": sum(term_counts.values()),
        "eight_gram_count": len(eight_grams),
        "training_eight_gram_overlaps": len(overlap),
        "repeated_bigram_count": repeated_bigram_count,
    }


def write_loss_svg(path: Path) -> None:
    width = 760
    height = 380
    left = 70
    right = 30
    top = 30
    bottom = 55
    xs = [point["step"] for point in LOSS_CURVE]
    ys = [point["val_loss"] for point in LOSS_CURVE]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 1.15, 1.95

    def px(step: int) -> float:
        return left + (step - x_min) / (x_max - x_min) * (width - left - right)

    def py(loss: float) -> float:
        return top + (y_max - loss) / (y_max - y_min) * (height - top - bottom)

    points = " ".join(f"{px(point['step']):.1f},{py(point['val_loss']):.1f}" for point in LOSS_CURVE)
    labels = []
    for point in LOSS_CURVE:
        x = px(point["step"])
        y = py(point["val_loss"])
        labels.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#9b4d2e" />')
        if point["step"] in {4000, 5000, 10000}:
            labels.append(
                f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" font-size="13" fill="#42382f">'
                f'{point["step"]}: {point["val_loss"]:.3f}</text>'
            )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbf8f1" />
  <text x="{left}" y="22" font-family="Georgia, serif" font-size="18" fill="#2f2924">Validation loss by training step</text>
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#6f6258" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#6f6258" />
  <polyline fill="none" stroke="#9b4d2e" stroke-width="3" points="{points}" />
  {''.join(labels)}
  <text x="{width / 2 - 45:.1f}" y="{height - 15}" font-family="Arial, sans-serif" font-size="14" fill="#42382f">training step</text>
  <text transform="translate(18 {height / 2 + 45:.1f}) rotate(-90)" font-family="Arial, sans-serif" font-size="14" fill="#42382f">validation loss</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def main() -> int:
    args = parse_args()
    checkpoint = ROOT / args.checkpoint
    metadata_path = checkpoint / "metadata.json"
    if not metadata_path.exists():
        print(f"Missing {metadata_path}", file=sys.stderr)
        return 2

    source_path = ROOT / args.source
    source_text = extract_odyssey_body(source_path.read_text(encoding="utf-8"))
    source_ngrams = ngrams(tokenize_words(source_text), 8)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = ScratchConfig.from_dict(metadata["config"])

    out_dir = ROOT / args.out
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    write_loss_svg(out_dir / "loss_curve.svg")

    results = []
    for label, weights_name in CHECKPOINTS:
        weights_path = checkpoint / weights_name
        if not weights_path.exists():
            print(f"Missing {weights_path}", file=sys.stderr)
            return 2
        model = ByteGPT(config)
        model.load_weights(str(weights_path))
        model.eval()
        sample = generate(model, config, args.prompt, args.tokens, args.temperature, args.top_k, args.seed)
        sample_path = samples_dir / f"{label.replace('.', '_')}.txt"
        sample_path.write_text(sample + "\n", encoding="utf-8")
        results.append(
            {
                "checkpoint": label,
                "weights": weights_name,
                "sample_path": str(sample_path.relative_to(ROOT)),
                "sample_excerpt": sample[:650],
                "metrics": score_sample(sample, source_ngrams),
            }
        )

    payload = {
        "checkpoint_dir": args.checkpoint,
        "prompt": args.prompt,
        "tokens": args.tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
        "model_metadata": metadata,
        "loss_curve": LOSS_CURVE,
        "results": results,
    }
    (out_dir / "scratch_blog_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = [
        "| Checkpoint | Val loss | Words | Odyssey term hits | Training 8-gram overlaps | Repeated bigrams |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    loss_by_label = {"2.5k": None, "5k": 1.261, "7.5k": None, "10k": 1.894}
    for result in results:
        metrics = result["metrics"]
        rows.append(
            f"| {result['checkpoint']} | {loss_by_label[result['checkpoint']] or 'n/a'} | "
            f"{metrics['words']} | {metrics['odyssey_term_hits']} | "
            f"{metrics['training_eight_gram_overlaps']} | {metrics['repeated_bigram_count']} |"
        )

    report = [
        "# Odyssey Scratch Checkpoint Comparison",
        "",
        f"Prompt: `{args.prompt}`",
        f"Decoding: temperature `{args.temperature}`, top-k `{args.top_k}`, generated bytes `{args.tokens}`.",
        "",
        *rows,
        "",
        "## Samples",
        "",
    ]
    for result in results:
        escaped = html.escape(result["sample_excerpt"].strip())
        report.extend([f"### {result['checkpoint']}", "", f"```text\n{escaped}\n```", ""])
    (out_dir / "scratch_checkpoint_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"wrote={out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
