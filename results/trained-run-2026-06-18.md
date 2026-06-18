# Trained Deterministic Run

Generated locally on 2026-06-18 after a 200-iteration MLX LoRA training run.

## Training

- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
- Adapter: `outputs/adapters/philosophy-compressor-20260618-003538`
- Training split: 1,344 train / 128 valid / 128 test examples
- Iterations: 200
- Final train loss: 0.143
- Final validation loss: 0.154
- Peak memory: about 2.9 GB

MLX warned that some training examples exceeded 2,048 tokens and were truncated. The next data-quality pass should shorten or pre-split those examples.

## Compression Run

The trained adapter was used with deterministic greedy decoding plus a deterministic repetition penalty. The run sampled two chunks per text and compressed for five generations.

| Generation | Units | Words |
| --- | ---: | ---: |
| `gen_001` | 32 | 1,563 |
| `gen_002` | 16 | 815 |
| `gen_003` | 8 | 693 |
| `gen_004` | 4 | 328 |
| `gen_005` | 2 | 209 |

## Absorption Trace

The absorption trace was written locally to:

```text
outputs/runs/absorption_trace.jsonl
outputs/runs/absorption_trace.md
```

Average deterministic concept retention:

| Generation | Avg Concept Retention |
| --- | ---: |
| 1 | 0.11 |
| 2 | 0.53 |
| 3 | 0.85 |
| 4 | 0.63 |
| 5 | 0.55 |

Interpretation: the current adapter is too extractive and loses conceptual breadth early. Later generations retain more of the surviving concept vocabulary, but only after much of the original philosophical range has already disappeared.
