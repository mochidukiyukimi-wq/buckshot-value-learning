"""Inspect one saved run without modifying its model or keeping extra checkpoint copies."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from roulette import _native as native
from roulette.config import load_config, native_search
from roulette.evaluation.accuracy import exact_reference_cases
from roulette.evaluation.matches import (
    RandomAgent,
    SearchAgent,
    ShotOnlyAgent,
    evaluate_matches,
)
from roulette.model.transformer import build_model
from roulette.training.checkpoint import load_checkpoint
from roulette.training.train import configure_runtime


def verify_run(config_path):
    config = load_config(config_path)
    configure_runtime(config)
    directory = Path(config.run_dir)
    checkpoint_path = directory / "latest.pt"
    saved = load_checkpoint(checkpoint_path, config)
    if list(directory.glob("*.pt")) != [checkpoint_path]:
        raise RuntimeError(
            "Run directory contains checkpoint history instead of just latest.pt"
        )
    if list(directory.glob(".latest-*.tmp")):
        raise RuntimeError("Checkpoint temporary files remain")
    for group in ("model", "ema"):
        if not all(torch.isfinite(tensor).all() for tensor in saved[group].values()):
            raise RuntimeError(f"Non-finite tensor in {group}")
    events = [
        json.loads(line)
        for line in (directory / "metrics.jsonl").read_text().splitlines()
    ]
    training_events = [event for event in events if event["event"] == "training"]
    tensorboard = EventAccumulator(str(directory / "tensorboard")).Reload()
    scalar_tags = tensorboard.Tags()["scalars"]
    if (
        "validation/mae" not in scalar_tags
        or "training/cross_entropy" not in scalar_tags
    ):
        raise RuntimeError("Required TensorBoard series are missing")
    model = build_model(config.model).to(config.device)
    model.load_state_dict(saved["model"])
    model.eval()
    support = saved["support"].to(config.device)
    cases = [case[0] for case in exact_reference_cases()]
    tactical_matches = evaluate_matches(
        (SearchAgent(model, support, native_search(config)), RandomAgent()),
        cases,
        [config.seed + 4000003 + index for index in range(len(cases))],
    )
    baseline_cases = [
        native.sample_initial_state(config.seed + 5000003 + index)
        for index in range(10)
    ]
    baseline_matches = evaluate_matches(
        (ShotOnlyAgent(), RandomAgent()),
        baseline_cases,
        [config.seed + 6000003 + index for index in range(10)],
    )
    with checkpoint_path.open("rb") as checkpoint_file:
        checkpoint_digest = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "step": saved["step"],
        "completed_units": saved["completed_units"],
        "save_reason": saved["save_reason"],
        "model_and_ema_finite": True,
        "training_roots": sum(event["root_count"] for event in training_events),
        "training_teacher_rows": sum(
            event["teacher_rows"] for event in training_events
        ),
        "training_inference_rows": sum(
            event["inference_rows"] for event in training_events
        ),
        "training_peak_rss_bytes": max(
            event["process_peak_rss_bytes"] for event in training_events
        ),
        "max_observed_internal_key_overlap": max(
            event["validation_internal_key_overlap"] for event in training_events
        ),
        "checkpoint_events": [
            {"reason": event["reason"], "step": event["step"]}
            for event in events
            if event["event"] == "checkpoint"
        ],
        "tensorboard_scalar_tags": scalar_tags,
        "tensorboard_validation_count": len(tensorboard.Scalars("validation/mae")),
        "tensorboard_cross_entropy_count": len(
            tensorboard.Scalars("training/cross_entropy")
        ),
        "learned_model_tactical_matches": tactical_matches,
        "simulator_baseline_full_games": baseline_matches,
        "strength_limit": "Tactical-case wins and baseline full games do not establish learned model strength from normal starts",
    }
    (directory / "verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cpu_pilot.json")
    print(json.dumps(verify_run(parser.parse_args().config), indent=2))
