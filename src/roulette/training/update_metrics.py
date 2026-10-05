"""Measure actual AdamW parameter movement, including retained optimizer momentum."""

import torch


@torch.no_grad()
def parameter_update_metrics(parameters, previous_parameters) -> dict:
    parameter_squared_norm = torch.stack(
        [previous.float().square().sum() for previous in previous_parameters]
    ).sum()
    update_squared_norm = torch.stack(
        [
            (current.detach().float() - previous.float()).square().sum()
            for current, previous in zip(parameters, previous_parameters, strict=True)
        ]
    ).sum()
    parameter_norm = parameter_squared_norm.sqrt()
    update_norm = update_squared_norm.sqrt()
    return {
        "parameter_norm": parameter_norm.item(),
        "parameter_update_norm": update_norm.item(),
        "relative_parameter_update": (
            update_norm / parameter_norm.clamp_min(1e-30)
        ).item(),
    }
