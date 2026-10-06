import pytest
import torch

from roulette import _native as n


@pytest.fixture(autouse=True, scope="session")
def limit_test_threads():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def state_with(*, live=1, blank=1, actor=0, hp=(4, 4), inventory0=(), inventory1=()):
    state = n.State()
    state.actor = actor
    state.ammo = n.make_reload_belief(live, blank)
    for player, items in enumerate((inventory0, inventory1)):
        state.player(player).hp = hp[player]
        counts = [0] * (n.FEATURE_SCHEMA.item_vocabulary_size - n.FEATURE_SCHEMA.first_item_id)
        for item in items:
            counts[int(item)] += 1
        state.player(player).inventory = counts
    return state
