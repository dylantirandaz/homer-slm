from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


@dataclass(frozen=True)
class ScratchConfig:
    vocab_size: int = 256
    context_size: int = 192
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 192
    mlp_dim: int = 768

    @classmethod
    def from_dict(cls, data: dict) -> "ScratchConfig":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})

    def to_dict(self) -> dict:
        return asdict(self)


class CausalBlock(nn.Module):
    def __init__(self, config: ScratchConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiHeadAttention(config.d_model, config.n_heads)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.d_model)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        h = self.ln1(x)
        x = x + self.attn(h, h, h, mask)
        h = self.ln2(x)
        return x + self.fc2(nn.gelu(self.fc1(h)))


class ByteGPT(nn.Module):
    def __init__(self, config: ScratchConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_size, config.d_model)
        self.blocks = [CausalBlock(config) for _ in range(config.n_layers)]
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        _, seq_len = tokens.shape
        if seq_len > self.config.context_size:
            tokens = tokens[:, -self.config.context_size :]
            seq_len = self.config.context_size

        positions = mx.arange(seq_len)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len, dtype=x.dtype)
        for block in self.blocks:
            x = block(x, mask)
        return self.output(self.final_norm(x))


def parameter_count(model: nn.Module) -> int:
    return int(sum(value.size for _, value in tree_flatten(model.parameters())))
