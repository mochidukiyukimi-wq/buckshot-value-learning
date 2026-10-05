"""Compare CE and squared Cramér at the saved LR without modifying learned state.

Both diagnostic branches start from identical model/AdamW state and see identical
two-hot labels generated once by the saved frozen EMA. CE is a reference branch
only; the production trainer has exactly one objective, squared Cramér.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from roulette import _native as native
from roulette.config import load_config, native_sampling, native_search
from roulette.model.squared_cramer import squared_cramer_loss
from roulette.model.support import value_to_two_hot
from roulette.model.transformer import build_model
from roulette.training.checkpoint import load_checkpoint, capture_rng, restore_rng
from roulette.training.ema import freeze_teacher
from roulette.training.teacher import make_training_batch
from roulette.training.train import configure_runtime
from roulette.training.update_metrics import parameter_update_metrics


def measure(config_path, output_path, unit_count=24, learning_rates=()):
    if any(not np.isfinite(rate) or rate <= 0 for rate in learning_rates):
        raise ValueError("Diagnostic learning rates must be finite and positive")
    config = load_config(config_path)
    configure_runtime(config)
    checkpoint_path = Path(config.run_dir) / "latest.pt"
    checkpoint_signature = checkpoint_path.stat()
    saved = load_checkpoint(checkpoint_path, config, loss_transition=True)
    support = saved["support"].to(config.device)
    original_rng = capture_rng()
    teacher_model = build_model(config.model).to(config.device)
    teacher_model.load_state_dict(saved["ema"])
    started = time.monotonic()
    try:
        batches = []
        with freeze_teacher(teacher_model, saved["completed_units"]) as teacher:
            for index in range(unit_count + 4):
                roots = native.sample_roots(
                    native_sampling(config),
                    config.training.roots_per_unit,
                    config.seed + saved["completed_units"] + index + 1,
                )
                batches.append(
                    make_training_batch(
                        roots,
                        teacher,
                        support,
                        native_search(config),
                        config.search.work_budget,
                    )
                )
        # Separate diagnostic generator avoids changing the saved/global shuffle RNG.
        generator = torch.Generator().manual_seed(config.seed + saved["step"])
        minibatches = []
        for batch in batches[:unit_count]:
            updates = 0
            for _ in range(config.training.max_epochs_per_unit):
                order = torch.randperm(len(batch.values), generator=generator).numpy()
                for offset in range(0, len(order), config.training.batch_size):
                    indices = order[offset : offset + config.training.batch_size]
                    minibatches.append(
                        (
                            torch.from_numpy(batch.features[indices]),
                            torch.from_numpy(batch.values[indices]),
                        )
                    )
                    updates += 1
                    if updates >= config.training.updates_per_unit:
                        break
                if updates >= config.training.updates_per_unit:
                    break

        def heldout_loss(model):
            total_loss, rows = 0.0, 0
            with torch.no_grad():
                for batch in batches[unit_count:]:
                    for offset in range(
                        0, len(batch.values), config.training.batch_size
                    ):
                        features = torch.from_numpy(
                            batch.features[offset : offset + config.training.batch_size]
                        ).to(config.device)
                        values = torch.from_numpy(
                            batch.values[offset : offset + config.training.batch_size]
                        ).to(config.device)
                        targets = value_to_two_hot(values, support)
                        total_loss += squared_cramer_loss(
                            model.predict_logits(features), targets
                        ).item() * len(values)
                        rows += len(values)
            return total_loss / rows

        branches = {}
        branch_settings = [
            ("cross_entropy_reference", config.training.learning_rate),
            ("squared_cramer", config.training.learning_rate),
        ]
        branch_settings.extend(
            (f"squared_cramer_lr_{rate:g}", rate)
            for rate in dict.fromkeys(learning_rates)
            if rate != config.training.learning_rate
        )
        for objective, learning_rate in branch_settings:
            model = build_model(config.model).to(config.device)
            model.load_state_dict(saved["model"])
            model.train()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.training.learning_rate, weight_decay=0
            )
            optimizer.load_state_dict(deepcopy(saved["optimizer"]))
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            parameters = list(model.parameters())
            initial_parameters = [
                parameter.detach().clone() for parameter in parameters
            ]
            initial_heldout = heldout_loss(model)
            updates = []
            for features, values in minibatches:
                features, values = features.to(config.device), values.to(config.device)
                targets = value_to_two_hot(values, support)
                optimizer.zero_grad(set_to_none=True)
                logits = model.predict_logits(features)
                cramer = squared_cramer_loss(logits, targets)
                if objective == "cross_entropy_reference":
                    loss = -(targets * logits.float().log_softmax(-1)).sum(-1).mean()
                else:
                    loss = cramer
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 10.0)
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("Non-finite diagnostic gradient")
                previous = [parameter.detach().clone() for parameter in parameters]
                optimizer.step()
                updates.append(
                    {
                        "loss": loss.item(),
                        "squared_cramer": cramer.item(),
                        "gradient_norm": gradient_norm.item(),
                        **parameter_update_metrics(parameters, previous),
                    }
                )
            branches[objective] = {
                "learning_rate": optimizer.param_groups[0]["lr"],
                "updates": len(updates),
                "initial_heldout_squared_cramer": initial_heldout,
                "final_heldout_squared_cramer": heldout_loss(model),
                "first_update": updates[0],
                "last_update": updates[-1],
                "mean_gradient_norm": float(
                    np.mean([row["gradient_norm"] for row in updates])
                ),
                "mean_update_norm": float(
                    np.mean([row["parameter_update_norm"] for row in updates])
                ),
                "mean_relative_update": float(
                    np.mean([row["relative_parameter_update"] for row in updates])
                ),
                "total_parameter_movement": parameter_update_metrics(
                    parameters, initial_parameters
                ),
                "all_updates": updates,
            }
            del model, optimizer, parameters, initial_parameters, previous
        latest_signature = checkpoint_path.stat()
        if (
            latest_signature.st_ino,
            latest_signature.st_size,
            latest_signature.st_mtime_ns,
        ) != (
            checkpoint_signature.st_ino,
            checkpoint_signature.st_size,
            checkpoint_signature.st_mtime_ns,
        ):
            raise RuntimeError(
                "Checkpoint changed during the diagnostic; stop training before measuring"
            )
        result = {
            "saved_step": saved["step"],
            "completed_units": saved["completed_units"],
            "device": config.device,
            "ema_decay": config.training.ema_decay,
            "sampling_units": unit_count,
            "heldout_units": 4,
            "checkpoint_unchanged": True,
            "elapsed_seconds": time.monotonic() - started,
            "branches": branches,
            "interpretation_limit": "Frozen-teacher diagnostic; does not establish long-run LR optimality or true win probability.",
        }
        Path(output_path).write_text(json.dumps(result, indent=2, allow_nan=False))
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "branches"}
            ),
            flush=True,
        )
        for name, branch in branches.items():
            print(
                name,
                json.dumps(
                    {
                        key: value
                        for key, value in branch.items()
                        if key != "all_updates"
                    }
                ),
                flush=True,
            )
        return result
    finally:
        restore_rng(original_rng)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tsubame.json")
    parser.add_argument(
        "--output", default="runs/tsubame/loss_transition_measurement.json"
    )
    parser.add_argument("--units", type=int, default=24)
    parser.add_argument("--learning-rates", type=float, nargs="*", default=[])
    arguments = parser.parse_args()
    if arguments.units < 1:
        parser.error("units must be positive")
    measure(
        arguments.config, arguments.output, arguments.units, arguments.learning_rates
    )
