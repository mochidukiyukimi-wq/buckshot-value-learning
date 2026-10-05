"""Only latest.pt is retained. Atomic replacement preserves the previous valid checkpoint."""

import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .. import _native as native
from ..config import resolved_config


CHECKPOINT_VERSION = 1
TRAINING_OBJECTIVE = "squared_cramer"
LEGACY_OBJECTIVE = "cross_entropy"


def checkpoint_objective(metadata: dict) -> str:
    return metadata.get("training_objective", LEGACY_OBJECTIVE)


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


def metadata_for(config) -> dict:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "input_schema": native.INPUT_SCHEMA,
        "rules_version": native.RULES_VERSION,
        "training_objective": TRAINING_OBJECTIVE,
        "config": resolved_config(config),
    }


def validate_checkpoint_metadata(
    metadata: dict,
    config,
    *,
    allow_ema_decay_change=False,
    allow_loss_change=False,
    allow_loss_learning_rate_change=False,
) -> None:
    expected = metadata_for(config)
    for key in ("checkpoint_version", "input_schema", "rules_version"):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"Incompatible checkpoint {key}")
    validate_training_objective(metadata, allow_loss_change=allow_loss_change)
    actual_config = metadata.get("config", {})
    # These limits control how far a continuation runs, not the training algorithm.
    allowed_run_changes = {
        "generation_units",
        "max_runtime_seconds",
        "checkpoint_every_units",
        "checkpoint_every_seconds",
        "evaluate_every_units",
    }
    if allow_ema_decay_change:
        # Explicit EMA tuning keeps the model, data domain and optimizer settings fixed.
        allowed_run_changes.add("ema_decay")
    if (
        allow_loss_change
        and allow_loss_learning_rate_change
        and checkpoint_objective(metadata) == LEGACY_OBJECTIVE
    ):
        # CE's AdamW moments have a different scale. A measured LR adjustment is
        # permitted only at the explicit CE->Cramér migration, never silently.
        allowed_run_changes.add("learning_rate")
    old = json.loads(json.dumps(actual_config))
    new = resolved_config(config)
    for values in (old, new):
        values.pop("run_dir", None)
        for key in allowed_run_changes:
            values.get("training", {}).pop(key, None)
    if old != new:
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


def validate_stage_transition(metadata: dict, config) -> None:
    """Permit an explicit new data/runtime stage while preserving learned parameters and optimizer."""
    expected = metadata_for(config)
    for key in ("checkpoint_version", "input_schema", "rules_version"):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"Incompatible checkpoint {key}")
    validate_training_objective(metadata)
    previous = metadata.get("config", {})
    if previous.get("model") != expected["config"]["model"]:
        raise ValueError("A stage transition cannot change the model or support")
    for key in ("learning_rate", "ema_decay"):
        if previous.get("training", {}).get(key) != expected["config"]["training"][key]:
            raise ValueError(f"A stage transition cannot change {key}")


def validate_training_objective(metadata: dict, *, allow_loss_change=False) -> None:
    objective = checkpoint_objective(metadata)
    if objective == TRAINING_OBJECTIVE:
        return
    if allow_loss_change and objective == LEGACY_OBJECTIVE:
        return
    raise ValueError(
        "Incompatible training objective; CE continuation requires --loss-transition"
    )


def load_checkpoint(
    path: str | Path,
    config,
    *,
    stage_transition=False,
    ema_decay_transition=False,
    loss_transition=False,
    for_inference=False,
) -> dict:
    if sum((stage_transition, ema_decay_transition, loss_transition)) > 1:
        raise ValueError("Training transitions must be separate")
    if for_inference and any((stage_transition, ema_decay_transition, loss_transition)):
        raise ValueError("Inference loading cannot apply a training transition")
    # Only load this task's locally produced files: optimizer/RNG state requires pickle.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if stage_transition:
        validate_stage_transition(state["metadata"], config)
    else:
        validate_checkpoint_metadata(
            state["metadata"],
            config,
            allow_ema_decay_change=ema_decay_transition,
            allow_loss_change=loss_transition or for_inference,
            allow_loss_learning_rate_change=loss_transition,
        )
    saved_support = state["support"]
    expected_support = (
        torch.arange(config.model.support_points, dtype=torch.float64)
        / (config.model.support_points - 1)
    ).float()
    # linspace differs by one float32 rounding unit between PyTorch versions and devices.
    # Retain the stored values while requiring the same uniformly spaced probability support.
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
