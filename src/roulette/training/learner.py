"""Own learned state and update it from one completed batch of search targets."""

from dataclasses import dataclass, field
import time

import numpy as np
import torch

from ..config import Config
from ..model.support import make_support, value_to_two_hot, logits_to_value
from ..model.squared_cramer import squared_cramer_loss
from ..model.transformer import ValueTransformer, build_model
from .ema import initialize_ema, update_ema
from .run_control import RunControl
from .teacher import TrainingBatch
from .update_metrics import parameter_update_metrics


MAX_GRADIENT_NORM = 10.0


@dataclass
class TrainingState:
    model: ValueTransformer
    ema: ValueTransformer
    optimizer: torch.optim.Optimizer
    support: torch.Tensor
    step: int = 0
    completed_units: int = 0
    validation_history: list[dict] = field(default_factory=list)


def create_training_state(config: Config) -> TrainingState:
    model = build_model(config.model).to(config.device)
    return TrainingState(
        model=model,
        ema=initialize_ema(model),
        optimizer=torch.optim.AdamW(
            model.parameters(), lr=config.training.learning_rate, weight_decay=0
        ),
        support=make_support(config.model, config.device),
    )


def update_model(model, features, values, optimizer, support) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    targets = value_to_two_hot(values, support)
    logits = model.predict_logits(features)
    loss = squared_cramer_loss(logits, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training loss")
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("Non-finite gradient; optimizer was not updated")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), MAX_GRADIENT_NORM
    )
    parameters = list(model.parameters())
    previous_parameters = [parameter.detach().clone() for parameter in parameters]
    optimizer.step()
    with torch.no_grad():
        predictions = logits_to_value(logits, support)
        return {
            "squared_cramer": loss.item(),
            "value_mse": torch.square(predictions - values).mean().item(),
            "gradient_norm": gradient_norm.item(),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **parameter_update_metrics(parameters, previous_parameters),
        }


def learn_from_batch(
    state: TrainingState, batch: TrainingBatch, config: Config, control: RunControl
) -> tuple[dict, int]:
    update_metrics = []
    settings = config.training
    # Rows live for this generation unit only; no long-term replay is retained.
    for epoch in range(settings.max_epochs_per_unit):
        order = torch.randperm(len(batch.values)).numpy()
        for start in range(0, len(order), settings.batch_size):
            indices = order[start : start + settings.batch_size]
            update_start = time.perf_counter()
            features = torch.from_numpy(batch.features[indices]).to(config.device)
            values = torch.from_numpy(batch.values[indices]).to(config.device)
            update_metrics.append(
                update_model(
                    state.model, features, values, state.optimizer, state.support
                )
            )
            state.step += 1
            update_ema(state.ema, state.model, settings.ema_decay)
            batch.metrics.record_update(
                len(indices), time.perf_counter() - update_start
            )
            control.poll()
            if (
                len(update_metrics) >= settings.updates_per_unit
                or control.stop_requested
            ):
                break
        if len(update_metrics) >= settings.updates_per_unit or control.stop_requested:
            break
    mean_metrics = {
        key: float(np.mean([row[key] for row in update_metrics]))
        for key in update_metrics[0]
    }
    return mean_metrics, epoch + 1
