import torch

from roulette import _native as n
from roulette.evaluation.accuracy import exact_reference_cases, evaluate_action_regret
from roulette.evaluation.matches import (
    RandomAgent,
    ShotOnlyAgent,
    SearchAgent,
    evaluate_matches,
)
from roulette.model.transformer import build_model
from roulette.config import ModelConfig


def test_paired_full_games_finish_and_include_seat_swap():
    cases = [n.sample_initial_state(100 + index) for index in range(10)]
    seeds = list(range(10))
    first = evaluate_matches((ShotOnlyAgent(), RandomAgent()), cases, seeds)
    second = evaluate_matches((ShotOnlyAgent(), RandomAgent()), cases, seeds)
    assert first == second
    assert first["completed_games"] == 20
    assert first["incomplete_games"] == 0
    assert sum(game["seat_swapped"] for game in first["games"]) == 10


def test_search_agent_solves_exact_cases_and_play_matches():
    model = build_model(
        ModelConfig(width=16, layers=1, heads=2, ffn_width=32, support_points=11)
    ).eval()
    support = torch.linspace(0, 1, 11)
    search_config = n.SearchConfig()
    search_config.value_batch_size = 32
    cases = exact_reference_cases()
    assert (
        evaluate_action_regret(model, cases, support, search_config)["mean_regret"] == 0
    )
    agent = SearchAgent(model, support, search_config)
    result = evaluate_matches((agent, RandomAgent()), [cases[0][0]], [42])
    assert result["completed_games"] == 2
    assert result["agent_a_wins"] == 2
