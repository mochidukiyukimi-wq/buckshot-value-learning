import argparse
import json
from pathlib import Path


from . import _native as native
from .config import load_config, native_sampling, native_search
from .evaluation.accuracy import (
    evaluate_root_residuals,
    evaluate_value_error,
    evaluate_action_regret,
    exact_reference_cases,
)
from .evaluation.benchmark import benchmark_pipeline
from .evaluation.matches import RandomAgent, SearchAgent, evaluate_matches
from .model.transformer import build_model
from .training.checkpoint import load_checkpoint
from .training.train import configure_runtime, run_training


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fixed-rule Buckshot boundary value learning"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "benchmark"):
        command = subcommands.add_parser(name)
        command.add_argument("--config", default="configs/cpu_pilot.json")
        command.add_argument("--run-dir")
        if name == "train":
            command.add_argument("--resume", action="store_true")
            transition = command.add_mutually_exclusive_group()
            transition.add_argument(
                "--stage-transition",
                action="store_true",
                help="Explicitly change sampling/runtime while preserving model, optimizer, EMA and step",
            )
            transition.add_argument(
                "--loss-transition",
                action="store_true",
                help="Migrate CE to squared Cramér, permitting a configured LR adjustment while preserving learned state",
            )
            transition.add_argument(
                "--ema-decay-transition",
                action="store_true",
                help="Explicitly change only EMA decay while preserving learned state and validation history",
            )
        else:
            command.add_argument("--checkpoint")
            command.add_argument("--output")
            if name == "evaluate":
                command.add_argument(
                    "--match-pairs",
                    type=int,
                    default=0,
                    help="Paired, seat-swapped full-game evaluations against random",
                )
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    overrides = {"run_dir": arguments.run_dir} if arguments.run_dir else None
    config = load_config(arguments.config, overrides)
    if arguments.command == "train":
        result = run_training(
            config,
            resume=arguments.resume,
            stage_transition=arguments.stage_transition,
            ema_decay_transition=arguments.ema_decay_transition,
            loss_transition=arguments.loss_transition,
        )
    else:
        configure_runtime(config)
        checkpoint_path = arguments.checkpoint or str(
            Path(config.run_dir) / "latest.pt"
        )
        saved = load_checkpoint(checkpoint_path, config, for_inference=True)
        model = build_model(config.model).to(config.device)
        model.load_state_dict(saved["model"])
        model.eval()
        if arguments.command == "benchmark":
            result = benchmark_pipeline(
                config, model, saved["support"].to(config.device)
            )
        else:
            support = saved["support"].to(config.device)
            search_config = native_search(config)
            fixed_roots = native.sample_roots(
                native_sampling(config),
                config.training.validation_roots,
                config.seed + 1000003,
            )
            residuals, _ = evaluate_root_residuals(
                model,
                fixed_roots,
                support,
                search_config,
                saved["step"],
                config.search.work_budget,
            )
            cases = exact_reference_cases()
            result = {
                "step": saved["step"],
                "root_residuals": residuals,
                "exact_value_error": evaluate_value_error(
                    model, cases, support, search_config
                ),
                "exact_action_regret": evaluate_action_regret(
                    model, cases, support, search_config
                ),
            }
            if arguments.match_pairs < 0:
                raise ValueError("match-pairs cannot be negative")
            if arguments.match_pairs:
                agent = SearchAgent(
                    model, support, search_config, config.search.work_budget
                )
                initial_states = [
                    native.sample_initial_state(config.seed + 2000003 + index)
                    for index in range(arguments.match_pairs)
                ]
                seeds = [
                    config.seed + 3000003 + index
                    for index in range(arguments.match_pairs)
                ]
                result["matches"] = evaluate_matches(
                    (agent, RandomAgent()), initial_states, seeds
                )
        output = Path(
            arguments.output or Path(config.run_dir) / f"{arguments.command}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
