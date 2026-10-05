import random

from .. import _native as native
from ..training.ema import freeze_teacher
from ..training.teacher import generate_root_labels


class SearchAgent:
    def __init__(self, model, support, search_config, work_budget=128):
        self.model = model
        self.support = support
        self.search_config = search_config
        self.work_budget = work_budget

    def choose(self, state, rng):
        with freeze_teacher(self.model, 0) as teacher:
            batch = generate_root_labels(
                state, teacher, self.support, self.search_config, self.work_budget
            )
        result = batch.results[0]
        return result.actions[result.best_action_index]


class RandomAgent:
    def choose(self, state, rng):
        return rng.choice(native.legal_actions(state))


class ShotOnlyAgent:
    """Simple measurable baseline, always shoots the opponent."""

    def choose(self, state, rng):
        return native.make_shot(native.ShotTarget.OPPONENT)


def play_game(agent_a, agent_b, initial_state, seed: int, max_actions=500) -> dict:
    rng = random.Random(seed)
    state = initial_state.copy()
    agents = (agent_a, agent_b)
    for action_index in range(max_actions):
        action = agents[state.actor].choose(state, rng)
        state = native.sample_transition(state, action, rng.getrandbits(64)).state
        if native.is_terminal(state):
            return {
                "winner": 0 if state.player(1).hp == 0 else 1,
                "actions": action_index + 1,
                "seed": seed,
                "complete": True,
            }
    return {"winner": None, "actions": max_actions, "seed": seed, "complete": False}


def evaluate_matches(agents, cases, seeds) -> dict:
    games = []
    wins = 0
    for state, seed in zip(cases, seeds, strict=True):
        for swap in (False, True):
            ordered = agents[::-1] if swap else agents
            result = play_game(
                *ordered, native.swap_players(state) if swap else state, seed
            )
            result["seat_swapped"] = swap
            if result["complete"] and result["winner"] == int(swap):
                wins += 1
            games.append(result)
    complete = sum(game["complete"] for game in games)
    return {
        "games": games,
        "completed_games": complete,
        "incomplete_games": len(games) - complete,
        "agent_a_wins": wins,
        "agent_a_win_rate": wins / complete if complete else None,
        "metric_kind": "paired_seed_seat_swapped_matches",
    }
