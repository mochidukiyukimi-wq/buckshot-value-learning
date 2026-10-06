import numpy as np
import pytest

from roulette import _native as n
from conftest import state_with


@pytest.mark.parametrize("live,blank", [(1, 1), (1, 4), (4, 1), (4, 4), (0, 3), (3, 0)])
def test_inversion_preserves_information_and_is_involutive(live, blank):
    prior = n.make_reload_belief(live, blank)
    inverted = n.invert_current_round(prior)
    assert inverted.rounds == prior.rounds
    assert n.invert_current_round(inverted) == prior
    live_probability, _ = n.observe_current_round(inverted, n.Round.LIVE)
    assert live_probability == pytest.approx(blank / (live + blank))
    if live and blank:
        remaining_live_hypotheses = {
            index // 2 for index, weight in enumerate(inverted.weights) if weight
        }
        assert remaining_live_hypotheses == {live - 1, live + 1}


def test_observation_then_consumption_updates_next_round_distribution():
    prior = n.make_reload_belief(2, 3)
    probability, posterior = n.observe_current_round(prior, n.Round.LIVE)
    assert probability == pytest.approx(2 / 5)
    assert n.observe_current_round(posterior, n.Round.LIVE)[0] == 1
    next_round = n.consume_current_round(posterior)
    assert next_round.rounds == 4
    assert n.observe_current_round(next_round, n.Round.LIVE)[0] == pytest.approx(1 / 4)


@pytest.mark.parametrize(
    "round_type,observation_probability,next_live_probability",
    [(n.Round.BLANK, 0.4, 0.25), (n.Round.LIVE, 0.6, 0.5)],
)
def test_unknown_inversion_is_conditioned_before_next_round(
    round_type, observation_probability, next_live_probability
):
    inverted = n.invert_current_round(n.make_reload_belief(2, 3))
    probability, posterior = n.observe_current_round(inverted, round_type)
    assert probability == pytest.approx(observation_probability)
    consumed = n.consume_current_round(posterior)
    assert n.observe_current_round(consumed, n.Round.LIVE)[0] == pytest.approx(
        next_live_probability
    )


def test_unknown_inversion_does_not_encode_realized_count():
    state = state_with(live=4, blank=4, inventory0=(n.Item.INVERTER,))
    (outcome,) = n.collect_successors(state, n.make_item_use(n.Item.INVERTER))
    features = n.encode_states([outcome.state])[0]
    belief_columns = n.FEATURE_SCHEMA.belief_probabilities.columns
    assert np.count_nonzero(features[belief_columns]) == 2
    assert np.isclose(features[belief_columns].sum(), 1)
    assert any(
        weight and index // 2 == 5
        for index, weight in enumerate(outcome.state.ammo.weights)
    )


def test_magnifier_conditions_without_consumption_or_turn_change():
    state = state_with(live=1, blank=3, inventory0=(n.Item.MAGNIFIER,))
    outcomes = n.collect_successors(state, n.make_item_use(n.Item.MAGNIFIER))
    assert len(outcomes) == 2
    assert [outcome.probability for outcome in outcomes] == [0.75, 0.25]
    assert all(
        outcome.state.actor == 0 and outcome.state.ammo.rounds == 4
        for outcome in outcomes
    )
    assert all(
        np.count_nonzero(outcome.state.ammo.weights) == 1 for outcome in outcomes
    )


def test_exact_key_normalizes_integer_weights_without_rounding():
    state = state_with()
    scaled = state.copy()
    ammo = scaled.ammo
    ammo.weights = [weight * 11 for weight in ammo.weights]
    scaled.ammo = ammo
    assert n.make_state_key(state) == n.make_state_key(scaled)
    changed = state.copy()
    changed.ammo = n.invert_current_round(state.ammo)
    assert n.make_state_key(state) != n.make_state_key(changed)


@pytest.mark.parametrize(
    "field", ["hp", "knife", "skip_opponent", "cuff_blocked", "inventory"]
)
def test_key_contains_future_relevant_player_state(field):
    state = state_with()
    changed = state.copy()
    if field == "hp":
        changed.player(0).hp = 3
    elif field == "knife":
        changed.player(0).knife = True
    elif field == "skip_opponent":
        changed.player(0).skip_opponent = True
        changed.player(1).cuff_blocked = True
    elif field == "cuff_blocked":
        changed.player(1).cuff_blocked = True
    else:
        changed.player(0).inventory = [1] + [0] * 7
    assert n.make_state_key(state) != n.make_state_key(changed)


def test_actor_normalization_respects_player_swap():
    state = state_with(hp=(2, 3), inventory0=(n.Item.BEER,), inventory1=(n.Item.DRUG,))
    swapped = n.swap_players(state)
    np.testing.assert_array_equal(n.encode_states([state]), n.encode_states([swapped]))
    assert n.to_actor_value(0.3, 0) == 0.3
    assert n.to_actor_value(0.3, 1) == pytest.approx(0.7)


def test_integer_overflow_is_explicit():
    with pytest.raises(OverflowError):
        n.checked_multiply(2**63, 2)
    with pytest.raises(OverflowError):
        n.checked_add(2**64 - 1, 1)


def test_invalid_belief_is_rejected():
    belief = n.AmmoBelief()
    belief.rounds = 2
    belief.weights = [1] + [0] * 17
    assert n.canonicalize_belief(belief).rounds == 2
    belief.weights = [0] * 16 + [1, 0]
    with pytest.raises(ValueError):
        n.canonicalize_belief(belief)


def test_sampling_domain_reproducible_and_covers_boundary_types():
    domain = n.SamplingDomain()
    domain.max_items_per_player = 2
    domain.include_effects = True
    roots = n.sample_roots(domain, 400, 9)
    assert [n.make_state_key(root) for root in roots] == [
        n.make_state_key(root) for root in n.sample_roots(domain, 400, 9)
    ]
    for root in roots:
        n.validate_state(root)
        assert n.inventory_size(root.player(0).inventory) <= 2
        assert n.inventory_size(root.player(1).inventory) <= 2
    assert any(root.player(root.actor).knife for root in roots)
    assert any(
        n.observe_current_round(root.ammo, n.Round.LIVE)[0] in (0, 1) for root in roots
    )
    assert len(n.sample_trajectory_roots(3, 7)) == 3
