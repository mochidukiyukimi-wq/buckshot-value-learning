import math
import random

import pytest

from roulette import _native as n
from conftest import state_with


def use(state, item):
    return n.collect_successors(state, n.make_item_use(item))


def test_initial_distribution_and_first_real_p2_turn():
    state = n.sample_initial_state(12)
    assert state.actor == 0
    assert n.inventory_size(state.player(0).inventory) == 2
    assert n.inventory_size(state.player(1).inventory) == 1
    outcomes = n.collect_successors(state, n.make_shot(n.ShotTarget.OPPONENT))
    for outcome in outcomes:
        assert outcome.state.actor == 1
        assert n.inventory_size(outcome.state.player(1).inventory) == 3


@pytest.mark.parametrize("occupied,expected", [(0, 36), (5, 36), (6, 8), (7, 1)])
def test_item_draw_distribution_preserves_capacity_and_mass(occupied, expected):
    inventory = [occupied] + [0] * 7
    outcomes = n.item_draw_outcomes(inventory)
    assert len(outcomes) == expected
    assert sum(outcome.probability for outcome in outcomes) == pytest.approx(1)
    assert all(
        n.inventory_size(outcome.inventory) == min(7, occupied + 2)
        for outcome in outcomes
    )


def test_two_draws_have_unequal_count_vector_probabilities():
    outcomes = n.item_draw_outcomes([0] * 8)
    for outcome in outcomes:
        same_kind = max(outcome.inventory) == 2
        assert outcome.probability == (1 / 64 if same_kind else 2 / 64)


def test_reload_is_sixteen_independent_pairs():
    outcomes = n.reload_outcomes()
    assert len(outcomes) == 16
    assert all(outcome.probability == 1 / 16 for outcome in outcomes)
    assert (
        len(
            {
                tuple(outcome.ammo.weights) + (outcome.ammo.rounds,)
                for outcome in outcomes
            }
        )
        == 16
    )


@pytest.mark.parametrize(
    "item", [n.Item.CIGARETTE, n.Item.KNIFE, n.Item.HANDCUFFS, n.Item.INVERTER]
)
def test_deterministic_items_consume_one_without_turn_change(item):
    state = state_with(inventory0=(item,))
    (outcome,) = use(state, item)
    assert outcome.probability == 1
    assert outcome.state.actor == 0
    assert n.inventory_size(outcome.state.player(0).inventory) == 0
    assert n.inventory_size(state.player(0).inventory) == 1


def test_cigarette_is_legal_at_full_hp():
    state = state_with(inventory0=(n.Item.CIGARETTE,))
    assert n.make_item_use(n.Item.CIGARETTE) in n.legal_actions(state)
    (outcome,) = use(state, n.Item.CIGARETTE)
    assert outcome.state.player(0).hp == 4


@pytest.mark.parametrize("hp,heal,harm", [(1, 3, 0), (3, 4, 2), (4, 4, 3)])
def test_medicine_observable_result_and_terminal_priority(hp, heal, harm):
    state = state_with(hp=(hp, 4), inventory0=(n.Item.MEDICINE,))
    healed, harmed = use(state, n.Item.MEDICINE)
    assert (healed.probability, harmed.probability) == (0.6, 0.4)
    assert healed.state.player(0).hp == heal
    assert harmed.state.player(0).hp == harm
    if harm == 0:
        assert harmed.kind == n.SuccessorKind.TERMINAL
        assert harmed.state.ammo == state.ammo


def test_self_blank_keeps_turn_and_cuff_effect_but_consumes_knife():
    state = state_with(live=0, blank=2)
    state.player(0).knife = True
    state.player(0).skip_opponent = True
    state.player(1).cuff_blocked = True
    (outcome,) = n.collect_successors(state, n.make_shot(n.ShotTarget.SELF))
    assert outcome.kind == n.SuccessorKind.INTERNAL
    assert outcome.state.actor == 0
    assert outcome.state.player(0).skip_opponent
    assert not outcome.state.player(0).knife


def test_self_live_changes_turn_even_with_cuffs():
    state = state_with(live=2, blank=0)
    state.player(0).skip_opponent = True
    state.player(1).cuff_blocked = True
    for outcome in n.collect_successors(state, n.make_shot(n.ShotTarget.SELF)):
        assert outcome.state.actor == 1
        assert outcome.state.player(0).hp == 3
        assert not outcome.state.player(0).skip_opponent
        assert not outcome.state.player(1).cuff_blocked
        assert outcome.kind == n.SuccessorKind.REPLENISHMENT


def test_opponent_shot_skip_consumed_without_replenishment():
    state = state_with(live=2, blank=0)
    state.player(0).skip_opponent = True
    state.player(1).cuff_blocked = True
    (outcome,) = n.collect_successors(state, n.make_shot(n.ShotTarget.OPPONENT))
    assert outcome.state.actor == 0
    assert not outcome.state.player(0).skip_opponent
    assert outcome.state.player(1).cuff_blocked
    assert outcome.kind == n.SuccessorKind.INTERNAL


def test_blocked_cuffs_are_still_consumed():
    state = state_with(inventory0=(n.Item.HANDCUFFS,))
    state.player(1).cuff_blocked = True
    (outcome,) = use(state, n.Item.HANDCUFFS)
    assert not outcome.state.player(0).skip_opponent
    assert outcome.state.player(1).cuff_blocked
    assert n.inventory_size(outcome.state.player(0).inventory) == 0


def test_beer_last_round_preserves_knife_and_does_not_replenish():
    state = state_with(live=1, blank=0, inventory0=(n.Item.BEER,))
    state.player(0).knife = True
    outcomes = use(state, n.Item.BEER)
    assert len(outcomes) == 16
    assert all(outcome.kind == n.SuccessorKind.RELOAD for outcome in outcomes)
    assert all(
        outcome.state.actor == 0 and outcome.state.player(0).knife
        for outcome in outcomes
    )
    assert all(
        n.inventory_size(outcome.state.player(0).inventory) == 0 for outcome in outcomes
    )


def test_last_round_and_turn_change_resolve_full_cartesian_product():
    state = state_with(live=1, blank=0)
    outcomes = n.collect_successors(state, n.make_shot(n.ShotTarget.OPPONENT))
    assert len(outcomes) == 576
    assert math.fsum(outcome.probability for outcome in outcomes) == pytest.approx(1)
    assert all(outcome.kind == n.SuccessorKind.BOTH for outcome in outcomes)
    assert all(
        outcome.state.actor == 1
        and n.inventory_size(outcome.state.player(1).inventory) == 2
        for outcome in outcomes
    )


def test_terminal_shot_does_not_reload_or_draw_items():
    state = state_with(live=1, blank=0, hp=(4, 1))
    (outcome,) = n.collect_successors(state, n.make_shot(n.ShotTarget.OPPONENT))
    assert outcome.kind == n.SuccessorKind.TERMINAL
    assert outcome.state.ammo.rounds == 0
    assert n.inventory_size(outcome.state.player(1).inventory) == 0


def test_turn_change_is_boundary_even_with_full_inventory():
    state = state_with(live=2, blank=0, inventory1=(n.Item.KNIFE,) * 7)
    (outcome,) = n.collect_successors(state, n.make_shot(n.ShotTarget.OPPONENT))
    assert outcome.kind == n.SuccessorKind.REPLENISHMENT
    assert n.inventory_size(outcome.state.player(1).inventory) == 7


def test_drug_targets_and_full_capacity_after_consumption():
    state = state_with(
        inventory0=(n.Item.DRUG,) + (n.Item.BEER,) * 6,
        inventory1=(n.Item.CIGARETTE, n.Item.DRUG),
    )
    actions = n.legal_actions(state)
    assert n.make_steal(-1) in actions
    assert n.make_steal(int(n.Item.CIGARETTE)) in actions
    with pytest.raises(ValueError):
        n.make_steal(int(n.Item.DRUG))
    (outcome,) = n.collect_successors(state, n.make_steal(int(n.Item.CIGARETTE)))
    assert outcome.state.player(0).inventory[int(n.Item.CIGARETTE)] == 1
    assert n.inventory_size(outcome.state.player(0).inventory) == 7
    assert outcome.state.player(0).hp == 4
    assert outcome.state.player(1).inventory[int(n.Item.CIGARETTE)] == 0
    (discarded,) = n.collect_successors(state, n.make_steal(-1))
    assert n.inventory_size(discarded.state.player(0).inventory) == 6


def test_knife_does_not_stack():
    state = state_with(live=2, blank=0, inventory0=(n.Item.KNIFE,))
    state.player(0).knife = True
    (armed,) = use(state, n.Item.KNIFE)
    for outcome in n.collect_successors(
        armed.state, n.make_shot(n.ShotTarget.OPPONENT)
    ):
        assert outcome.state.player(1).hp == 2


def test_sampled_and_enumerated_shots_share_distribution():
    state = state_with(live=1, blank=3)
    action = n.make_shot(n.ShotTarget.OPPONENT)
    live_samples = sum(
        n.sample_transition(state, action, seed).state.player(1).hp == 3
        for seed in range(3000)
    )
    assert live_samples / 3000 == pytest.approx(0.25, abs=0.03)


def test_random_legal_trajectories_preserve_invariants():
    rng = random.Random(23)
    state = n.sample_initial_state(8)
    for _ in range(1000):
        action = rng.choice(n.legal_actions(state))
        outcomes = n.collect_successors(state, action)
        assert math.fsum(outcome.probability for outcome in outcomes) == pytest.approx(
            1
        )
        for outcome in outcomes:
            n.validate_state(outcome.state)
        state = n.sample_transition(state, action, rng.getrandbits(64)).state
        if n.is_terminal(state):
            state = n.sample_initial_state(rng.getrandbits(64))
