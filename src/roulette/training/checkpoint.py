"""Persist and restore the current training state in one atomic latest.pt file."""

from copy import deepcopy
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .. import _native as native
from ..config import Config, resolved_config
from .learner import TrainingState


CHECKPOINT_VERSION = 1
TRAINING_OBJECTIVE = "squared_cramer"


def capture_rng() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA RNG state cannot resume on this CPU-only environment"
            )
        torch.cuda.set_rng_state_all(state["cuda"])


def metadata_for(config: Config) -> dict:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "input_schema": native.INPUT_SCHEMA,
        "rules_version": native.RULES_VERSION,
        "training_objective": TRAINING_OBJECTIVE,
        "config": resolved_config(config),
    }


def validate_checkpoint_metadata(metadata: dict, config: Config) -> None:
    expected = metadata_for(config)
    for key in (
        "checkpoint_version",
        "input_schema",
        "rules_version",
        "training_objective",
    ):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"Incompatible checkpoint {key}")
    actual_config = deepcopy(metadata["config"])
    expected_config = expected["config"]
    # Run duration and reporting schedules can change without changing learned state.
    run_settings = {
        "generation_units",
        "max_runtime_seconds",
        "checkpoint_every_units",
        "checkpoint_every_seconds",
        "evaluate_every_units",
    }
    for values in (actual_config, expected_config):
        values.pop("run_dir", None)
        for key in run_settings:
            values["training"].pop(key, None)
    if actual_config != expected_config:
        raise ValueError(
            "Checkpoint configuration differs from model, sampling, seed or learning settings"
        )


def save_checkpoint(path: str | Path, state: dict) -> None:
    path = Path(path)
    if path.name != "latest.pt":
        raise ValueError(
            "Checkpoints must overwrite latest.pt; historical models are not retained"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".latest-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            torch.save(state, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_checkpoint(path: str | Path, config: Config) -> dict:
    # Locally produced checkpoints contain optimizer and RNG state requiring pickle.
    state = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(state["metadata"], config)
    saved_support = state["support"]
    expected_support = (
        torch.arange(config.model.support_points, dtype=torch.float64)
        / (config.model.support_points - 1)
    ).float()
    # Keep the saved FP32 support; uniformly spaced values may differ by one rounding unit.
    if (
        not isinstance(saved_support, torch.Tensor)
        or saved_support.dtype != torch.float32
        or saved_support.shape != expected_support.shape
        or saved_support[0].item() != 0.0
        or saved_support[-1].item() != 1.0
        or not torch.allclose(
            saved_support.cpu(),
            expected_support,
            rtol=0,
            atol=torch.finfo(torch.float32).eps,
        )
    ):
        raise ValueError("Incompatible categorical support")
    return state


def restore_training_state(
    path: str | Path, config: Config, state: TrainingState
) -> None:
    saved = load_checkpoint(path, config)
    state.model.load_state_dict(saved["model"])
    state.ema.load_state_dict(saved["ema"])
    state.optimizer.load_state_dict(saved["optimizer"])
    state.support = saved["support"].to(config.device)
    state.step = saved["step"]
    state.completed_units = saved["completed_units"]
    state.validation_history = saved["validation_history"]
    restore_rng(saved["rng"])


def save_training_state(
    path: str | Path,
    config: Config,
    state: TrainingState,
    environment: dict,
    reason: str,
) -> None:
    save_checkpoint(
        path,
        {
            "metadata": metadata_for(config),
            "model": state.model.state_dict(),
            "ema": state.ema.state_dict(),
            "optimizer": state.optimizer.state_dict(),
            "rng": capture_rng(),
            "step": state.step,
            "completed_units": state.completed_units,
            "support": state.support.detach().cpu(),
            "validation_history": state.validation_history,
            "environment": environment,
            "save_reason": reason,
        },
    )
