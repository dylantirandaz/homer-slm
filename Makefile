PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
MODEL_ID ?= mlx-community/Qwen2.5-0.5B-Instruct-4bit
FULL_MODEL_ID ?= mlx-community/Qwen2.5-0.5B-Instruct-bf16
ADAPTER ?= outputs/adapters/odyssey-qwen25-0.5b
FULL_ADAPTER ?= outputs/adapters/odyssey-qwen25-0.5b-full
PORT ?= 8765
CHAT_TEMPERATURE ?= 0.75
CHAT_TOP_P ?= 0.9
CHAT_TOP_K ?= 40

.PHONY: help setup data pretrain-data train train-full chat output-dirs

help:
	@printf "Minimum Viable Odyssey\n\n"
	@printf "Targets:\n"
	@printf "  setup   Create .venv and install base requirements\n"
	@printf "  data    Download and prepare Odyssey chunks/training data\n"
	@printf "  pretrain-data  Prepare plain-text Odyssey continuation data\n"
	@printf "  train   Run MLX LoRA chat/summarization tuning for MODEL_ID\n"
	@printf "  train-full  Run full-parameter continued training on Odyssey text\n"
	@printf "  chat    Serve the local Odyssey SLM chat UI with probabilistic decoding\n"

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -r requirements-mlx.txt

output-dirs:
	mkdir -p data/raw data/processed data/training/mlx data/training/pretrain outputs/adapters

data: output-dirs
	$(PYTHON) scripts/download_odyssey.py
	$(PYTHON) scripts/prepare_data.py

pretrain-data: output-dirs
	$(PYTHON) scripts/prepare_pretrain_data.py

train: output-dirs
	$(PY) scripts/train_lora.py --model-id $(MODEL_ID) --data data/training/mlx --adapter-path $(ADAPTER)

train-full: output-dirs
	$(PY) scripts/train_lora.py --fine-tune-type full --model-id $(FULL_MODEL_ID) --data data/training/pretrain --adapter-path $(FULL_ADAPTER) --iters 50 --batch-size 1 --num-layers -1 --learning-rate 5e-6 --no-mask-prompt --grad-checkpoint --steps-per-report 5 --steps-per-eval 25 --save-every 50 --max-seq-length 1024

chat:
	$(PY) scripts/chat_server.py --model-id $(MODEL_ID) --adapter-path $(ADAPTER) --port $(PORT) --temperature $(CHAT_TEMPERATURE) --top-p $(CHAT_TOP_P) --top-k $(CHAT_TOP_K)
