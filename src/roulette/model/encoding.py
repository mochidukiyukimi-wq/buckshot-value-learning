import numpy as np
import torch
from torch import nn

from .. import _native as native


def encode_states(states) -> np.ndarray:
    return native.encode_states(states)


def validate_features(features: np.ndarray) -> None:
    if (
        features.dtype != np.float32
        or features.ndim != 2
        or features.shape[1] != native.FEATURE_COUNT
    ):
        raise ValueError("Expected float32 [B,43] actor-normalized features")
    if not features.flags.c_contiguous or not np.isfinite(features).all():
        raise ValueError("Features must be contiguous and finite")
    if ((features[:, :29] < 0) | (features[:, :29] > 1)).any():
        raise ValueError("Numeric features must be in [0,1]")
    item_ids = features[:, 29:]
    if ((item_ids < 0) | (item_ids > 8) | (item_ids != np.floor(item_ids))).any():
        raise ValueError("Item token IDs must be integers in [0,8]")


class FeatureEncoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.global_projection = nn.Linear(19, width)
        self.player_projection = nn.Linear(5, width)
        self.round_embedding = nn.Embedding(9, width)
        self.hp_embedding = nn.Embedding(5, width)
        self.count_embedding = nn.Embedding(8, width)
        self.item_embedding = nn.Embedding(9, width)
        self.type_embedding = nn.Embedding(3, width)
        self.role_embedding = nn.Embedding(3, width)
        self.register_buffer("token_types", torch.tensor([0, 1, 1] + [2] * 14))
        self.register_buffer("owner_roles", torch.tensor([0, 1, 2] + [1] * 7 + [2] * 7))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        global_token = self.global_projection(features[:, :19])
        global_token = global_token + self.round_embedding(
            (features[:, 0] * 8).round().long()
        )
        player_features = features[:, 19:29].reshape(-1, 2, 5)
        player_tokens = self.player_projection(player_features)
        player_tokens = player_tokens + self.hp_embedding(
            (player_features[:, :, 0] * 4).round().long()
        )
        player_tokens = player_tokens + self.count_embedding(
            (player_features[:, :, 1] * 7).round().long()
        )
        item_tokens = self.item_embedding(features[:, 29:].long())
        tokens = torch.cat(
            (global_token[:, None, :], player_tokens, item_tokens), dim=1
        )
        # No slot-position embedding: inventory order cannot change the value.
        return (
            tokens
            + self.type_embedding(self.token_types)
            + self.role_embedding(self.owner_roles)
        )
