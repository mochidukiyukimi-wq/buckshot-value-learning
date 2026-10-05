"""Own the training lifecycle; inference, evaluation, saving and rendering have separate modules."""

import json
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from .. import _native as native
from ..config import native_sampling, native_search, resolved_config
from ..evaluation.accuracy import evaluate_root_residuals
from ..model.support import make_support, value_loss, value_to_two_hot, logits_to_value
from ..model.transformer import build_model
from ..observability.logging import RunLogger, render_cli
from ..observability.metrics import execution_environment
from .checkpoint import (
    capture_rng,
    load_checkpoint,
    metadata_for,
    restore_rng,
    save_checkpoint,
)
from .ema import initialize_ema, freeze_teacher, update_ema
from .teacher import make_training_batch


class TrainingStopped(Exception):
    pass


def configure_runtime(config) -> None:
    torch.set_num_threads(config.training.intra_threads)
    if torch.get_num_interop_threads() != config.training.inter_threads:
        torch.set_num_interop_threads(config.training.inter_threads)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")


def train_step(model, features, values, optimizer, support) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    targets = value_to_two_hot(values, support)
    logits = model.predict_logits(features)
    loss = value_loss(logits, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training loss")
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("Non-finite gradient; optimizer was not updated")
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    with torch.no_grad():
        predictions = logits_to_value(logits, support)
        entropy = -(targets * targets.clamp_min(1e-30).log()).sum(-1).mean()
        return {
            "cross_entropy": loss.item(),
            "value_mse": torch.square(predictions - values).mean().item(),
            "teacher_entropy": entropy.item(),
            "excess_cross_entropy": (loss - entropy).item(),
            "gradient_norm": gradient_norm.item(),
        }


def run_training(config, resume=False, *, stage_transition=False) -> dict:
    if stage_transition and not resume:
        raise ValueError("A stage transition requires --resume")
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
    directory.mkdir(parents=True, exist_ok=True)
    model = build_model(config.model).to(config.device)
    ema = initialize_ema(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.learning_rate, weight_decay=0
    )
    support = make_support(config.model, config.device)
    step, completed_units = 0, 0
    validation_history = []
    stage_lineage = []
    if resume:
        saved = load_checkpoint(
            checkpoint_path, config, stage_transition=stage_transition
        )
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        support = saved["support"].to(config.device)
        restore_rng(saved["rng"])
        step, completed_units = saved["step"], saved["completed_units"]
        validation_history = saved["validation_history"]
        stage_lineage = saved.get("stage_lineage", [])
        if stage_transition:
            stage_lineage.append(
                {
                    "previous_metadata": saved["metadata"],
                    "step": step,
                    "completed_units": completed_units,
                    "previous_validation_history": validation_history,
                }
            )
            # A wider sampling domain needs its own baseline; old-domain residuals stay in lineage.
            validation_history = []
    initial_step = step
    fixed_roots = native.sample_roots(
        native_sampling(config), config.training.validation_roots, config.seed + 1000003
    )
    search_config = native_search(config)
    started = time.monotonic()
    last_checkpoint = started
    stop_requested = False
    stop_reason = "completed"
    logger = RunLogger(directory)
    environment = execution_environment(config)
    previous_handlers = {}

    def request_stop(signum, frame):
        nonlocal stop_requested, stop_reason
        stop_requested = True
        stop_reason = "interrupted"

    for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        signal_number = getattr(signal, signal_name, None)
        if signal_number is not None:
            previous_handlers[signal_number] = signal.signal(
                signal_number, request_stop
            )

    def save(reason):
        nonlocal last_checkpoint
        save_checkpoint(
            checkpoint_path,
            {
                "metadata": metadata_for(config),
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": capture_rng(),
                "step": step,
                "completed_units": completed_units,
                "support": support.detach().cpu(),
                "validation_history": validation_history,
                "environment": environment,
                "save_reason": reason,
                "stage_lineage": stage_lineage,
            },
        )
        last_checkpoint = time.monotonic()
        logger.write_jsonl(
            {
                "event": "checkpoint",
                "reason": reason,
                "step": step,
                "completed_units": completed_units,
                "path": str(checkpoint_path),
            }
        )

    def heartbeat():
        nonlocal stop_requested, stop_reason
        if time.monotonic() - started >= config.training.max_runtime_seconds:
            stop_requested = True
            stop_reason = "runtime_limit"
        if (
            time.monotonic() - last_checkpoint
            >= config.training.checkpoint_every_seconds
        ):
            save("periodic_time")
        if stop_requested:
            raise TrainingStopped()

    status = "completed"
    failure = None
    validation_keys = set()
    try:
        (directory / "resolved_config.json").write_text(
            json.dumps(resolved_config(config), indent=2), encoding="utf-8"
        )
        logger.write_jsonl(
            {
                "event": "start",
                "resume": resume,
                "stage_transition": stage_transition,
                "step": step,
                "rules_version": native.RULES_VERSION,
                "input_schema": native.INPUT_SCHEMA,
                "environment": environment,
                "config": resolved_config(config),
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
            }
        )
        save("initial" if not resume else "resume")
        if not validation_history:
            validation, validation_keys = evaluate_root_residuals(
                model,
                fixed_roots,
                support,
                search_config,
                step,
                config.search.work_budget,
                heartbeat,
            )
            validation_history.append({"step": step, **validation})
            logger.write_jsonl(
                {
                    "event": "validation",
                    "generation_unit": completed_units,
                    "step": step,
                    **validation,
                }
            )
            logger.write_tensorboard(validation, step, "validation")
            print(
                f"initial validation MAE={validation['mae']:.6f} P99={validation['p99']:.6f}",
                flush=True,
            )
        else:
            # Regenerate validation keys on resume so overlap counts are measured, not assumed zero.
            validation, validation_keys = evaluate_root_residuals(
                model,
                fixed_roots,
                support,
                search_config,
                step,
                config.search.work_budget,
                heartbeat,
            )
            logger.write_jsonl(
                {"event": "resume_validation", "step": step, **validation}
            )
        while (
            config.training.generation_units is None
            or completed_units < config.training.generation_units
        ):
            heartbeat()
            unit = completed_units + 1
            roots = native.sample_roots(
                native_sampling(config),
                config.training.roots_per_unit,
                config.seed + unit,
            )
            with freeze_teacher(ema, completed_units) as teacher:
                batch = make_training_batch(
                    roots,
                    teacher,
                    support,
                    search_config,
                    config.search.work_budget,
                    heartbeat,
                )
            update_metrics = []
            update_count = 0
            # Each generated row is used at most max_epochs_per_unit times, never retained as replay.
            for epoch in range(config.training.max_epochs_per_unit):
                order = torch.randperm(len(batch.values)).numpy()
                for start in range(0, len(order), config.training.batch_size):
                    indices = order[start : start + config.training.batch_size]
                    update_start = time.perf_counter()
                    features = torch.from_numpy(batch.features[indices]).to(
                        config.device
                    )
                    values = torch.from_numpy(batch.values[indices]).to(config.device)
                    update_metrics.append(
                        train_step(model, features, values, optimizer, support)
                    )
                    step += 1
                    update_count += 1
                    update_ema(ema, model, config.training.ema_decay)
                    batch.metrics.record_update(
                        len(indices), time.perf_counter() - update_start
                    )
                    if (
                        time.monotonic() - started
                        >= config.training.max_runtime_seconds
                    ):
                        stop_requested = True
                        stop_reason = "runtime_limit"
                    if (
                        update_count >= config.training.updates_per_unit
                        or stop_requested
                    ):
                        break
                if update_count >= config.training.updates_per_unit or stop_requested:
                    break
            completed_units = unit
            validation = None
            if unit % config.training.evaluate_every_units == 0 and not stop_requested:
                validation, validation_keys = evaluate_root_residuals(
                    model,
                    fixed_roots,
                    support,
                    search_config,
                    step,
                    config.search.work_budget,
                    heartbeat,
                )
                validation_history.append({"step": step, **validation})
                logger.write_tensorboard(validation, step, "validation")
            mean_updates = {
                key: float(np.mean([row[key] for row in update_metrics]))
                for key in update_metrics[0]
            }
            event = {
                "event": "training",
                "generation_unit": unit,
                "step": step,
                "elapsed_seconds": time.monotonic() - started,
                **mean_updates,
                **batch.metrics.summarize_interval(),
                "validation": validation or validation_history[-1],
                "validation_internal_key_overlap": len(
                    set(batch.keys) & validation_keys
                ),
                "epochs_started": epoch + 1,
            }
            logger.write_jsonl(event)
            logger.write_tensorboard(mean_updates, step)
            render_cli(event)
            del batch
            if (
                unit % config.training.checkpoint_every_units == 0
                or time.monotonic() - last_checkpoint
                >= config.training.checkpoint_every_seconds
            ):
                save("periodic_unit")
            if stop_requested:
                raise TrainingStopped()
    except TrainingStopped:
        status = stop_reason
        logger.write_jsonl({"event": "stopped", "reason": stop_reason, "step": step})
    except BaseException as error:
        status = "failed"
        failure = error
        logger.write_jsonl(
            {
                "event": "failure",
                "type": type(error).__name__,
                "message": str(error),
                "step": step,
                "incomplete_roots": config.training.roots_per_unit,
                "resource_limit": int(type(error).__name__ == "SearchResourceLimit"),
                "nan": int(isinstance(error, FloatingPointError)),
            }
        )
    finally:
        try:
            save("exit_" + status)
        finally:
            logger.close()
            for signal_number, previous_handler in previous_handlers.items():
                signal.signal(signal_number, previous_handler)
    summary = {
        "status": status,
        "step": step,
        "initial_step": initial_step,
        "completed_units": completed_units,
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": str(checkpoint_path),
        "initial_validation": validation_history[0] if validation_history else None,
        "final_validation": validation_history[-1] if validation_history else None,
        "environment": environment,
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if failure is not None:
        raise failure
    return summary
