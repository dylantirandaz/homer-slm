PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
MODEL ?= mlx-community/Qwen2.5-0.5B-Instruct-4bit
ADAPTER ?= outputs/adapters/philosophy-compressor
COMMENTARY_ADAPTER ?=
TRACE_COMMENTARY_ARG := $(if $(COMMENTARY_ADAPTER),--commentary-adapter-path $(COMMENTARY_ADAPTER),)

.PHONY: help setup download ingest chunk data training-data train run trace show test output-dirs

help:
	@printf "Minimum Viable Philosophy\n"
	@printf "\n"
	@printf "Targets:\n"
	@printf "  setup           Create .venv and install requirements\n"
	@printf "  data            Download, ingest, and chunk the corpus\n"
	@printf "  training-data   Build MLX LoRA training data\n"
	@printf "  train           Fine-tune the local MLX LoRA adapter\n"
	@printf "  run             Run recursive compression\n"
	@printf "  trace           Analyze concept absorption/loss across generations\n"
	@printf "  show            Print the latest run summary\n"
	@printf "  test            Run smoke tests\n"
	@printf "  output-dirs     Ensure output directories exist\n"

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

download: output-dirs
	$(PY) scripts/download_corpus.py --min-success 20

ingest: output-dirs
	$(PY) scripts/ingest.py

chunk: output-dirs
	$(PY) scripts/chunk.py --chunk-size 900 --overlap 80

data: download ingest chunk

training-data: output-dirs
	$(PY) scripts/prepare_training_data.py \
		--input data/processed/chunks.jsonl \
		--out-dir data/training/mlx \
		--target-word-count 140 \
		--max-examples 1600

train: output-dirs
	$(PY) scripts/train_mlx_lora.py \
		--model $(MODEL) \
		--data data/training/mlx \
		--adapter-path $(ADAPTER) \
		--iters 120 \
		--batch-size 1 \
		--num-layers 8 \
		--grad-checkpoint

run: output-dirs
	$(PY) scripts/run_pipeline.py \
		--input data/processed/chunks.jsonl \
		--model $(MODEL) \
		--adapter-path $(ADAPTER) \
		--generations 5 \
		--batch-size 2 \
		--schedule 260,160,90,45,20 \
		--per-text-chunks 2 \
		--max-input-words 1100

trace: output-dirs
	$(PY) scripts/analyze_absorption.py --model $(MODEL) --adapter-path $(ADAPTER) $(TRACE_COMMENTARY_ARG)

show:
	$(PY) scripts/show_run.py

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests

output-dirs:
	mkdir -p outputs/logs outputs/runs
