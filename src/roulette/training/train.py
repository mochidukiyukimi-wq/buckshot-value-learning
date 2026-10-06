"""Coordinate target generation, learner updates, validation, and run completion."""

import json
from pathlib import Path
import random

import numpy as np
import torch

from .. import _native as native
from ..config import Config, native_sampling, native_search, resolved_config
from ..evaluation.accuracy import evaluate_root_residuals
from ..observability.logging import RunLogger
from ..observability.metrics import execution_environment
from ..runtime import configure_runtime
from .checkpoint import metadata_for, restore_training_state, save_training_state
from .ema import freeze_teacher
from .learner import create_training_state, learn_from_batch
from .run_control import RunControl, TrainingStopped
from .teacher import SearchResourceLimit, make_training_batch


def run_training(config: Config, resume: bool = False) -> dict:
    configure_runtime(config)
    random.seed(config.seed)
    np.random.seed(config.seed % (2**32))
    torch.manual_seed(config.seed)
    directory = Path(config.run_dir)
    checkpoint_path = directory / "latest.pt"
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(
            f"{checkpoint_path} already exists; use --resume or a new run directory"
        )
    state = create_training_state(config)
    if resume:
        restore_training_state(checkpoint_path, config, state)
    initial_step = state.step
    fixed_roots = native.sample_roots(
        native_sampling(config), config.training.validation_roots, config.seed + 1000003
    )
    search_config = native_search(config)
    environment = execution_environment(config)
    status, failure = "completed", None

    with RunControl(config.training) as control, RunLogger(directory) as logger:

        def checkpoint(reason):
            save_training_state(checkpoint_path, config, state, environment, reason)
            control.checkpoint_saved()
            logger.write_jsonl(
                {
                    "event": "checkpoint",
                    "reason": reason,
                    "step": state.step,
                    "completed_units": state.completed_units,
                    "path": str(checkpoint_path),
                }
            )

        def heartbeat():
            control.poll()
            if control.checkpoint_due:
                checkpoint("periodic_time")
            control.raise_if_stopped()

        def evaluate_current_model():
            return evaluate_root_residuals(
                state.model,
                fixed_roots,
                state.support,
                search_config,
                state.step,
                config.search.work_budget,
                heartbeat,
            )

        try:
            (directory / "resolved_config.json").write_text(
                json.dumps(resolved_config(config), indent=2), encoding="utf-8"
            )
            logger.record_start(metadata_for(config), state, environment, resume)
            checkpoint("resume" if resume else "initial")
            validation, validation_keys = evaluate_current_model()
            if state.validation_history:
                # Recompute keys on resume to measure overlap with newly generated targets.
                logger.write_jsonl(
                    {"event": "resume_validation", "step": state.step, **validation}
                )
            else:
                state.validation_history.append({"step": state.step, **validation})
                logger.write_jsonl(
                    {
                        "event": "validation",
                        "generation_unit": state.completed_units,
                        "step": state.step,
                        **validation,
                    }
                )
                logger.write_tensorboard(validation, state.step, "validation")
                print(
                    f"initial validation MAE={validation['mae']:.6f} P99={validation['p99']:.6f}",
                    flush=True,
                )
            while (
                config.training.generation_units is None
                or state.completed_units < config.training.generation_units
            ):
                heartbeat()
                unit = state.completed_units + 1
                roots = native.sample_roots(
                    native_sampling(config),
                    config.training.roots_per_unit,
                    config.seed + unit,
                )
                with freeze_teacher(state.ema, state.completed_units) as teacher:
                    batch = make_training_batch(
                        roots,
                        teacher,
                        state.support,
                        search_config,
                        config.search.work_budget,
                        heartbeat,
                    )
                mean_updates, epochs_started = learn_from_batch(
                    state, batch, config, control
                )
                state.completed_units = unit
                validation = None
                if (
                    unit % config.training.evaluate_every_units == 0
                    and not control.stop_requested
                ):
                    validation, validation_keys = evaluate_current_model()
                    state.validation_history.append({"step": state.step, **validation})
                    logger.write_tensorboard(validation, state.step, "validation")
                logger.record_generation(
                    state,
                    batch,
                    mean_updates,
                    validation_keys,
                    control.elapsed_seconds,
                    epochs_started,
                )
                del batch
                if (
                    unit % config.training.checkpoint_every_units == 0
                    or control.checkpoint_due
                ):
                    checkpoint("periodic_unit")
                control.raise_if_stopped()
        except TrainingStopped:
            status = control.stop_reason
            logger.write_jsonl(
                {"event": "stopped", "reason": status, "step": state.step}
            )
        except BaseException as error:
            status, failure = "failed", error
            logger.write_jsonl(
                {
                    "event": "failure",
                    "type": type(error).__name__,
                    "message": str(error),
                    "step": state.step,
                    "incomplete_roots": config.training.roots_per_unit,
                    "resource_limit": int(isinstance(error, SearchResourceLimit)),
                    "nan": int(isinstance(error, FloatingPointError)),
                }
            )
        finally:
            checkpoint("exit_" + status)
        summary = {
            "status": status,
            "step": state.step,
            "initial_step": initial_step,
            "completed_units": state.completed_units,
            "elapsed_seconds": control.elapsed_seconds,
            "checkpoint": str(checkpoint_path),
            "initial_validation": state.validation_history[0]
            if state.validation_history
            else None,
            "final_validation": state.validation_history[-1]
            if state.validation_history
            else None,
            "environment": environment,
        }
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    if failure is not None:
        raise failure
    return summary
