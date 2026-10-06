import numpy as np
import torch
from torch import nn

from .. import _native as native


FEATURE_SCHEMA = native.FEATURE_SCHEMA
GLOBAL_COLUMNS = FEATURE_SCHEMA.global_features.columns
PLAYER_COLUMNS = FEATURE_SCHEMA.player_features.columns
ITEM_COLUMNS = FEATURE_SCHEMA.item_features.columns
NUMERIC_COLUMNS = FEATURE_SCHEMA.numeric_features.columns


def encode_states(states) -> np.ndarray:
    return native.encode_states(states)


def validate_features(features: np.ndarray) -> None:
    if (
        features.dtype != np.float32
        or features.ndim != 2
        or features.shape[1] != FEATURE_SCHEMA.feature_count
    ):
        raise ValueError(
            f"Expected float32 [B,{FEATURE_SCHEMA.feature_count}] actor-normalized features"
        )
    if not features.flags.c_contiguous or not np.isfinite(features).all():
        raise ValueError("Features must be contiguous and finite")
    numeric_features = features[:, NUMERIC_COLUMNS]
    if ((numeric_features < 0) | (numeric_features > 1)).any():
        raise ValueError("Numeric features must be in [0,1]")
    item_ids = features[:, ITEM_COLUMNS]
    if (
        (item_ids < FEATURE_SCHEMA.empty_item_id)
        | (item_ids >= FEATURE_SCHEMA.item_vocabulary_size)
        | (item_ids != np.floor(item_ids))
    ).any():
        raise ValueError(
            f"Item token IDs must be integers in "
            f"[{FEATURE_SCHEMA.empty_item_id},{FEATURE_SCHEMA.item_vocabulary_size - 1}]"
        )


class FeatureEncoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.global_projection = nn.Linear(FEATURE_SCHEMA.global_features.count, width)
        self.player_projection = nn.Linear(FEATURE_SCHEMA.player_feature_count, width)
        self.round_embedding = nn.Embedding(FEATURE_SCHEMA.round_count_limit + 1, width)
        self.hp_embedding = nn.Embedding(FEATURE_SCHEMA.hp_limit + 1, width)
        self.count_embedding = nn.Embedding(FEATURE_SCHEMA.inventory_limit + 1, width)
        self.item_embedding = nn.Embedding(FEATURE_SCHEMA.item_vocabulary_size, width)
        self.type_embedding = nn.Embedding(FEATURE_SCHEMA.token_type_count, width)
        self.role_embedding = nn.Embedding(FEATURE_SCHEMA.owner_role_count, width)
        self.register_buffer("token_types", torch.tensor(FEATURE_SCHEMA.token_types))
        self.register_buffer(
            "owner_roles", torch.tensor(FEATURE_SCHEMA.token_owner_roles)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        global_token = self.global_projection(features[:, GLOBAL_COLUMNS])
        round_counts = (
            (
                features[:, FEATURE_SCHEMA.round_count_column]
                * FEATURE_SCHEMA.round_count_limit
            )
            .round()
            .long()
        )
        global_token = global_token + self.round_embedding(round_counts)
        player_features = features[:, PLAYER_COLUMNS].reshape(
            -1, FEATURE_SCHEMA.player_count, FEATURE_SCHEMA.player_feature_count
        )
        player_hp = (
            (player_features[:, :, FEATURE_SCHEMA.hp_field] * FEATURE_SCHEMA.hp_limit)
            .round()
            .long()
        )
        inventory_counts = (
            (
                player_features[:, :, FEATURE_SCHEMA.inventory_count_field]
                * FEATURE_SCHEMA.inventory_limit
            )
            .round()
            .long()
        )
        player_tokens = self.player_projection(player_features)
        player_tokens = player_tokens + self.hp_embedding(player_hp)
        player_tokens = player_tokens + self.count_embedding(inventory_counts)
        item_tokens = self.item_embedding(features[:, ITEM_COLUMNS].long())
        tokens = torch.cat(
            (global_token[:, None, :], player_tokens, item_tokens), dim=1
        )
        # No slot-position embedding: inventory order cannot change the value.
        return (
            tokens
            + self.type_embedding(self.token_types)
            + self.role_embedding(self.owner_roles)
        )
