# Odyssey SLM

Small language model fine-tuning on Homer's *Odyssey*.

This repo downloads a public-domain English *Odyssey*, prepares summarization-style training examples from the full text, and fine-tunes a small MLX LoRA adapter.

## Model Target

Default model:

```text
mlx-community/Qwen2.5-0.5B-Instruct-4bit
```

This is small enough to fine-tune locally on Apple Silicon while still being a real SLM. Larger Qwen models can be tried by overriding `MODEL_ID`, but the 0.5B model is the practical default.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-mlx.txt
```

## Workflow

```bash
make data
make train
make chat
```

The adapter is written to:

```text
outputs/adapters/odyssey-qwen25-0.5b/
```

Training data is written to:

```text
data/training/mlx/train.jsonl
data/training/mlx/valid.jsonl
data/training/mlx/test.jsonl
```

## Chat UI

Start the local Greco-deco chat interface:

```bash
make chat
```

Then open:

```text
http://127.0.0.1:8765
```

The chat server uses the fine-tuned adapter by default:

```text
outputs/adapters/odyssey-qwen25-0.5b/
```
