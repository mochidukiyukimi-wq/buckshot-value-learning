import torch
from torch import nn
from torch.nn import functional as F

from .encoding import FeatureEncoder


class EncoderBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_width: int):
        super().__init__()
        self.heads = heads
        self.attention_norm = nn.LayerNorm(width)
        self.qkv_projection = nn.Linear(width, 3 * width)
        self.attention_output = nn.Linear(width, width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_width), nn.GELU(), nn.Linear(ffn_width, width)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, width = tokens.shape
        projected = self.qkv_projection(self.attention_norm(tokens))
        query, key, value = projected.reshape(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        tokens = tokens + self.attention_output(attended)
        return tokens + self.ffn(self.ffn_norm(tokens))


class ValueTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = FeatureEncoder(config.width)
        self.blocks = nn.ModuleList(
            EncoderBlock(config.width, config.heads, config.ffn_width)
            for _ in range(config.layers)
        )
        self.output_norm = nn.LayerNorm(config.width)
        self.value_head = nn.Linear(config.width, config.support_points)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return self.value_head(self.output_norm(tokens[:, 0]).float()).float()

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self(self.encoder(features))


def build_model(config) -> ValueTransformer:
    return ValueTransformer(config)
