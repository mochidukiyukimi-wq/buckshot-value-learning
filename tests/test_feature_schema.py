"""Check the native/Python feature contract and compatibility with saved v1 models."""

import numpy as np
import pytest
import torch

from roulette import _native as native
from roulette.model.encoding import FeatureEncoder, encode_states, validate_features


SCHEMA = native.FEATURE_SCHEMA


def representative_state(actor):
    state = native.State()
    state.actor = actor
    state.ammo = native.make_reload_belief(2, 4)
    state.player(actor).hp = 2
    state.player(actor).inventory = [1, 1, 0, 0, 0, 0, 0, 2]
    state.player(actor).knife = True
    state.player(actor).skip_opponent = True
    state.player(1 - actor).hp = 3
    state.player(1 - actor).inventory = [0, 0, 1, 0, 0, 0, 0, 0]
    state.player(1 - actor).cuff_blocked = True
    return state


# These fixtures deliberately pin the existing serialized v1 layout independently
# of the shared schema: changing a schema constant must not silently move old inputs.
@pytest.mark.parametrize("actor", [0, 1])
def test_native_encoding_preserves_v1_columns_and_actor_perspective(actor):
    expected = np.array(
        [0.75]
        + [0.0] * 4
        + [2 / 3, 1 / 3]
        + [0.0] * 12
        + [0.5, 4 / 7, 1, 1, 0, 0.75, 1 / 7, 0, 0, 1]
        + [1, 2, 8, 8, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
        dtype=np.float32,
    )
    features = encode_states([representative_state(actor)])
    np.testing.assert_array_equal(features[0], expected)
    validate_features(features)


def test_shared_schema_preserves_v1_token_types_and_owner_roles():
    encoder = FeatureEncoder(width=1)
    assert native.INPUT_SCHEMA == "actor-normalized-43f-17tokens-v1"
    assert SCHEMA.feature_count == native.FEATURE_COUNT == 43
    assert SCHEMA.token_count == 17
    assert encoder.token_types.tolist() == [0, 1, 1] + [2] * 14
    assert encoder.owner_roles.tolist() == [0, 1, 2] + [1] * 7 + [2] * 7


@pytest.mark.parametrize("actor", [0, 1])
def test_python_encoder_decodes_native_counts_and_item_ids(actor):
    encoder = FeatureEncoder(width=1)
    # Isolate discrete embeddings so a wrong slice or normalization coefficient
    # changes an observable token value instead of hiding inside random weights.
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.zero_()
        for embedding, scale in (
            (encoder.round_embedding, 1),
            (encoder.hp_embedding, 1),
            (encoder.count_embedding, 10),
            (encoder.item_embedding, 1),
        ):
            embedding.weight[:, 0] = scale * torch.arange(embedding.num_embeddings)
    features = torch.from_numpy(encode_states([representative_state(actor)]))
    tokens = encoder(features)
    expected = torch.tensor(
        [6, 42, 13, 1, 2, 8, 8, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
        dtype=torch.float32,
    )
    torch.testing.assert_close(tokens[0, :, 0], expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "section,value,message",
    [
        ("numeric", 1.1, "Numeric features"),
        ("numeric", np.nan, "finite"),
        ("items", -1, "Item token IDs"),
        ("items", 1.5, "Item token IDs"),
        ("items", SCHEMA.item_vocabulary_size, "Item token IDs"),
    ],
)
def test_validation_uses_shared_numeric_and_item_ranges(section, value, message):
    features = encode_states([representative_state(0)])
    columns = SCHEMA.numeric_features if section == "numeric" else SCHEMA.item_features
    features[0, columns.start] = value
    with pytest.raises(ValueError, match=message):
        validate_features(features)


@pytest.mark.parametrize("relative_player", [-1, SCHEMA.player_count])
def test_schema_rejects_invalid_relative_player(relative_player):
    with pytest.raises(ValueError, match="player"):
        SCHEMA.player_columns(relative_player)
    with pytest.raises(ValueError, match="player"):
        SCHEMA.inventory_columns(relative_player)
