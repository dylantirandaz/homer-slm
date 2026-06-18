# Odyssey SLM

Small language model fine-tuning on Homer's *Odyssey*.

This repo downloads a public-domain English *Odyssey*, prepares Odyssey-only training data, fine-tunes a small MLX LoRA adapter, and can run a full-parameter continued-training pass on plain Odyssey text.

## Model Target

Default model:

```text
mlx-community/Qwen2.5-0.5B-Instruct-4bit
```

This is small enough to LoRA-tune locally on Apple Silicon while still being a real SLM. Larger Qwen models can be tried by overriding `MODEL_ID`, but the 0.5B model is the practical default.

Full-parameter training uses the non-quantized bf16 checkpoint:

```text
mlx-community/Qwen2.5-0.5B-Instruct-bf16
```

Override it with `FULL_MODEL_ID`. Full-parameter training cannot use the 4-bit checkpoint because quantized weights do not support full backpropagation.

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

For full-parameter continued training on the Odyssey text:

```bash
make pretrain-data
make train-full
```

For from-scratch pretraining, use the byte-level GPT path:

```bash
make train-scratch
make generate-scratch
```

This initializes a small transformer randomly and trains it only on the Odyssey body text. It does not use Qwen weights or a pretrained tokenizer. The default scratch run is a 10,000-iteration, 6-layer byte GPT.
The scratch model is a generation model, not an instruction/chat model; use it to inspect what the randomly initialized model absorbs from the Odyssey text.

To compare an intermediate checkpoint:

```bash
make generate-scratch SCRATCH_WEIGHTS=002500_weights.safetensors
```

For a write-up of the scratch experiment:

```text
docs/teaching-a-tiny-model-to-dream-in-homer.md
```

Regenerate its local supporting artifacts with:

```bash
make blog-artifacts
```

The adapter is written to:

```text
outputs/adapters/odyssey-qwen25-0.5b/
```

The full-parameter continued-training output is written to:

```text
outputs/adapters/odyssey-qwen25-0.5b-full/
```

The from-scratch checkpoint is written to:

```text
outputs/scratch/odyssey-byte-gpt/
outputs/scratch/odyssey-byte-gpt-10k/
```

Training data is written to:

```text
data/training/mlx/train.jsonl
data/training/mlx/valid.jsonl
data/training/mlx/test.jsonl
```

Plain-text continued-training data is written to:

```text
data/training/pretrain/train.jsonl
data/training/pretrain/valid.jsonl
data/training/pretrain/test.jsonl
```

The chat server uses the fine-tuned adapter by default:

```text
outputs/adapters/odyssey-qwen25-0.5b/
```

To chat with the full-trained bf16 model instead:

```bash
make chat-full
```

`chat-full` is useful for inspecting the continued-pretraining checkpoint. For normal chat, `make chat` is currently the better surface because it uses the Odyssey LoRA chat adapter plus retrieval grounding.

Chat decoding is probabilistic by default:

```text
temperature=0.65 top_p=0.9 top_k=40
```

Override those settings with `CHAT_TEMPERATURE`, `CHAT_TOP_P`, and `CHAT_TOP_K` when running `make chat`.
