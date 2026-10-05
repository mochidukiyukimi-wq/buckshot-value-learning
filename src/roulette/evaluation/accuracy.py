import numpy as np

from .. import _native as native
from ..model.encoding import encode_states
from ..training.ema import freeze_teacher
from ..training.teacher import evaluate_frontier, make_training_batch


def error_summary(errors) -> dict:
    errors = np.asarray(errors, dtype=np.float64)
    return {
        "mae": float(np.abs(errors).mean()),
        "p99": float(np.quantile(np.abs(errors), 0.99)),
        "maximum_error": float(np.abs(errors).max()),
        "mse": float(np.square(errors).mean()),
    }


def evaluate_root_residuals(
    model,
    fixed_roots,
    support,
    search_config,
    version=0,
    work_budget=128,
    heartbeat=None,
) -> tuple[dict, set[bytes]]:
    model.eval()
    with freeze_teacher(model, version) as teacher:
        predictions = evaluate_frontier(
            teacher, encode_states(fixed_roots), support, search_config.value_batch_size
        )
        searched = make_training_batch(
            fixed_roots, teacher, support, search_config, work_budget, heartbeat
        )
    references = np.array(
        [result.win_probability_actor for result in searched.results], dtype=np.float64
    )
    return {
        **error_summary(predictions - references),
        "root_count": len(fixed_roots),
        "model_version": version,
        "metric_kind": "latest_model_bootstrap_residual",
        "predictions": predictions.tolist(),
        "search_values": references.tolist(),
    }, set(searched.keys)


def exact_reference_cases():
    cases = []
    for actor in (0, 1):
        for knife in (False, True):
            state = native.State()
            state.actor = actor
            state.ammo = native.make_reload_belief(1, 0)
            state.player(actor).hp = 1
            state.player(1 - actor).hp = 2 if knife else 1
            state.player(actor).knife = knife
            # With known live ammunition, self shot kills the actor and opponent shot kills the opponent.
            cases.append(
                (state, 1.0, [0.0 if actor == 0 else 1.0, 1.0 if actor == 0 else 0.0])
            )
    return cases


def evaluate_value_error(model, reference_cases, support, search_config) -> dict:
    with freeze_teacher(model, 0) as teacher:
        predictions = evaluate_frontier(
            teacher,
            encode_states([case[0] for case in reference_cases]),
            support,
            search_config.value_batch_size,
        )
    expected = np.array([case[1] for case in reference_cases])
    return {
        **error_summary(predictions - expected),
        "case_count": len(reference_cases),
        "metric_kind": "exact_terminal_tactical_value_error",
    }


def evaluate_action_regret(model, solved_cases, support, search_config) -> dict:
    with freeze_teacher(model, 0) as teacher:
        searched = make_training_batch(
            [case[0] for case in solved_cases], teacher, support, search_config
        )
    regrets = []
    for case, result in zip(solved_cases, searched.results, strict=True):
        actor = case[0].actor
        action_values_actor = [value if actor == 0 else 1 - value for value in case[2]]
        regrets.append(
            max(action_values_actor) - action_values_actor[result.best_action_index]
        )
    return {
        "mean_regret": float(np.mean(regrets)),
        "maximum_regret": float(np.max(regrets)),
        "case_count": len(regrets),
        "metric_kind": "exact_terminal_tactical_action_regret",
    }
