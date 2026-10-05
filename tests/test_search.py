import gc

import numpy as np
import pytest

from roulette import _native as n
from conftest import state_with


def solve(root, *, work_budget=128, memoize=True, label_buffer=4096):
    config = n.SearchConfig()
    config.memoize = memoize
    config.value_batch_size = 17
    config.label_buffer_size = label_buffer
    search = n.SearchSession(root, config, 12)
    labels = []
    frontier_keys = set()
    while True:
        progress = search.advance(work_budget)
        if progress.status == n.SearchStatus.NEEDS_VALUES:
            assert progress.request.evaluator_version == 12
            frontier_keys.update(progress.request.keys)
            search.submit_values(
                progress.request.request_id,
                np.full(len(progress.request.features), 0.5, np.float32),
            )
        elif progress.status in (
            n.SearchStatus.NEEDS_LABEL_DRAIN,
            n.SearchStatus.COMPLETE,
        ):
            while True:
                drained = search.take_labels(label_buffer)
                if not len(drained["values"]):
                    break
                labels.extend(zip(drained["keys"], drained["values"]))
            if progress.status == n.SearchStatus.COMPLETE:
                return search.result(), labels, frontier_keys
        else:
            assert progress.status == n.SearchStatus.YIELDED


@pytest.mark.parametrize("actor", [0, 1])
def test_hand_computed_unknown_round_case(actor):
    state = state_with(hp=(1, 1), actor=actor)
    result, labels, _ = solve(state)
    assert result.win_probability_actor == pytest.approx(0.75)
    actor_values = [
        value if actor == 0 else 1 - value for value in result.action_values_p0
    ]
    assert actor_values == pytest.approx([0.5, 0.75])
    assert len(result.actions) == len(n.legal_actions(state))
    assert dict(labels)[n.make_state_key(state)] == pytest.approx(0.75)


def test_chunk_size_cache_and_label_buffer_do_not_change_search():
    state = state_with(hp=(1, 1), inventory0=(n.Item.MAGNIFIER, n.Item.INVERTER))
    baseline, labels, frontier = solve(state)
    for work, memo, buffer in ((1, True, 1), (7, True, 3), (32, False, 2)):
        candidate, candidate_labels, _ = solve(
            state, work_budget=work, memoize=memo, label_buffer=buffer
        )
        assert candidate.action_values_p0 == pytest.approx(baseline.action_values_p0)
        assert dict(candidate_labels) == pytest.approx(dict(labels))
        assert len(candidate_labels) == len(dict(candidate_labels))
    assert len(labels) == baseline.stats.internal_nodes
    assert not (set(dict(labels)) & frontier)


def test_proven_win_does_not_remove_unselected_internal_teachers():
    state = state_with(live=1, blank=0, hp=(1, 1), inventory0=(n.Item.CIGARETTE,))
    assert n.prove_immediate_win(state)
    result, labels, _ = solve(state)
    assert result.win_probability_actor == 1
    assert len(labels) > 1
    assert len(result.actions) == 3


def test_resource_limit_never_returns_a_complete_result():
    state = state_with(inventory0=(n.Item.MAGNIFIER,))
    config = n.SearchConfig()
    config.max_internal_nodes = 1
    session = n.SearchSession(state, config, 0)
    assert session.advance(100).status == n.SearchStatus.RESOURCE_LIMIT
    assert not len(session.take_labels(100)["values"])
    with pytest.raises(RuntimeError, match="incomplete"):
        session.result()


def test_complete_result_is_unavailable_before_backup():
    session = n.SearchSession(state_with(), n.SearchConfig(), 0)
    with pytest.raises(RuntimeError, match="incomplete"):
        session.result()


def request_session():
    session = n.SearchSession(state_with(), n.SearchConfig(), 0)
    while True:
        progress = session.advance(100)
        if progress.status == n.SearchStatus.NEEDS_VALUES:
            return session, progress.request


@pytest.mark.parametrize(
    "invalid_kind",
    ["int", "shape", "noncontiguous", "nan", "infinity", "range", "count"],
)
def test_value_boundary_rejects_invalid_arrays(invalid_kind):
    session, request = request_session()
    count = len(request.features)
    array = np.full(count, 0.5, np.float32)
    if invalid_kind == "int":
        array = np.ones(count, np.int32)
    elif invalid_kind == "shape":
        array = array[:, None]
    elif invalid_kind == "noncontiguous":
        array = np.zeros(count * 2, np.float32)[::2]
    elif invalid_kind == "nan":
        array[0] = np.nan
    elif invalid_kind == "infinity":
        array[0] = np.inf
    elif invalid_kind == "range":
        array[0] = 1.1
    else:
        array = array[:-1]
    with pytest.raises(ValueError):
        session.submit_values(request.request_id, array)
    session.submit_values(request.request_id, np.full(count, 0.5, np.float64))
    with pytest.raises(ValueError, match="stale"):
        session.submit_values(request.request_id, np.full(count, 0.5, np.float32))


def test_request_and_label_arrays_own_memory_after_session_destruction():
    session, request = request_session()
    features = request.features
    copied = features.copy()
    assert features.flags.owndata
    del session, request
    gc.collect()
    np.testing.assert_array_equal(features, copied)
    roots = [state_with(live=1, blank=0, hp=(1, 1))]
    session = n.SearchSession(roots[0], n.SearchConfig(), 0)
    assert session.advance(100).status == n.SearchStatus.COMPLETE
    batch = session.take_labels(10)
    del session
    gc.collect()
    assert batch["features"].flags.owndata
    assert batch["values"].flags.owndata
    assert len(batch["values"]) == 1


def test_root_sessions_do_not_share_cached_values():
    root = state_with()
    results = []
    for value in (0.1, 0.9):
        session = n.SearchSession(root, n.SearchConfig(), 0)
        while True:
            progress = session.advance(100)
            if progress.status == n.SearchStatus.NEEDS_VALUES:
                session.submit_values(
                    progress.request.request_id,
                    np.full(len(progress.request.features), value, np.float32),
                )
            elif progress.status == n.SearchStatus.COMPLETE:
                results.append(session.result().win_probability_actor)
                break
    assert abs(results[0] - results[1]) > 0.3


def test_exact_tie_keeps_first_action():
    assert n.choose_best_action(0, [0.5, 0.5]) == 0
    assert n.choose_best_action(1, [0.5, 0.5]) == 0
