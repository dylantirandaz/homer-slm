# Minimum Viable Philosophy

Terminal-first experiment in philosophical compression.

This repo downloads a public-domain philosophy corpus, prepares compression training examples, fine-tunes a small local model with MLX LoRA, then recursively compresses the corpus until only a small final text remains.

No dashboard is required. The terminal is the interface.

## Setup

Apple Silicon is expected for MLX training.

```bash
cd minimum-viable-philosophy
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or use the Makefile:

```bash
make setup
```

If the system Python is too old for the current MLX package, install a newer Python with Homebrew:

```bash
brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Corpus

Download public-domain texts:

```bash
python scripts/download_corpus.py --min-success 20
python scripts/ingest.py
python scripts/chunk.py --chunk-size 900 --overlap 80
```

Equivalent Makefile shortcut:

```bash
make data
```

## Training Data

Prepare bootstrapped compression examples for LoRA fine-tuning:

```bash
python scripts/prepare_training_data.py \
  --input data/processed/chunks.jsonl \
  --out-dir data/training/mlx \
  --target-word-count 140 \
  --max-examples 1600
```

The training labels are extractive philosophical digests generated from the public-domain corpus. They are not used as a runtime compressor; they provide a compression curriculum for the LoRA adapter.

Equivalent Makefile shortcut:

```bash
make training-data
```

## Train

Default model:

```text
mlx-community/Qwen2.5-0.5B-Instruct-4bit
```

Run a real MLX LoRA fine-tune:

```bash
python scripts/train_mlx_lora.py \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --data data/training/mlx \
  --adapter-path outputs/adapters/philosophy-compressor \
  --iters 120 \
  --batch-size 1 \
  --num-layers 8 \
  --grad-checkpoint
```

Increase `--iters` after the first successful run.

## Recursive Compression

Run five generations with the trained adapter. For a fast first run that still
samples every downloaded text, use two chunks per text:

```bash
python scripts/run_pipeline.py \
  --input data/processed/chunks.jsonl \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --adapter-path outputs/adapters/philosophy-compressor \
  --generations 5 \
  --batch-size 2 \
  --schedule 260,160,90,45,20 \
  --per-text-chunks 2 \
  --max-input-words 1100
```

For the full corpus, expect a long run:

```bash
python scripts/run_pipeline.py \
  --input data/processed/chunks.jsonl \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --adapter-path outputs/adapters/philosophy-compressor \
  --generations 5 \
  --batch-size 1 \
  --schedule 260,160,90,45,20 \
  --max-input-words 900
```

Show the terminal view:

```bash
python scripts/show_run.py
```

Equivalent Makefile shortcuts:

```bash
make train
make run
make show
```

Final files are written to:

```text
outputs/final_paragraph.txt
outputs/final_sentence.txt
data/generations/
```

The first local run is summarized in `results/first-run.md`.
