PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
MODEL_ID ?= mlx-community/Qwen2.5-0.5B-Instruct-4bit
ADAPTER ?= outputs/adapters/odyssey-qwen25-0.5b

.PHONY: help setup data train output-dirs

help:
	@printf "Minimum Viable Odyssey\n\n"
	@printf "Targets:\n"
	@printf "  setup   Create .venv and install base requirements\n"
	@printf "  data    Download and prepare Odyssey chunks/training data\n"
	@printf "  train   Run MLX LoRA training for MODEL_ID\n"

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -r requirements-mlx.txt

output-dirs:
	mkdir -p data/raw data/processed data/training/mlx outputs/adapters

data: output-dirs
	$(PYTHON) scripts/download_odyssey.py
	$(PYTHON) scripts/prepare_data.py

train: output-dirs
	$(PY) scripts/train_lora.py --model-id $(MODEL_ID) --data data/training/mlx --adapter-path $(ADAPTER)
