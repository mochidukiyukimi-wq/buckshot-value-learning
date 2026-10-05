"""Squared Cramér objective on the uniform categorical support spanning [0, 1]."""

import torch


def squared_cramer_from_probabilities(
    probabilities: torch.Tensor, target_probabilities: torch.Tensor
) -> torch.Tensor:
    if probabilities.shape != target_probabilities.shape or probabilities.ndim < 2:
        raise ValueError("Predictions and teachers must have matching batch/bin shapes")
    bin_count = probabilities.shape[-1]
    if bin_count < 2:
        raise ValueError("At least two probability bins are required")
    prediction_cdf = probabilities.float().cumsum(dim=-1)[..., :-1]
    teacher_cdf = target_probabilities.detach().float().cumsum(dim=-1)[..., :-1]
    # Production has 101 points: spacing=0.01 and CDF indices 0..99.
    # Tiny test models retain the same integral on their coarser uniform support.
    support_spacing = 1.0 / (bin_count - 1)
    state_losses = support_spacing * (prediction_cdf - teacher_cdf).square().sum(-1)
    return state_losses.mean()


def squared_cramer_loss(
    logits: torch.Tensor, target_probabilities: torch.Tensor
) -> torch.Tensor:
    probabilities = torch.softmax(logits.float(), dim=-1)
    return squared_cramer_from_probabilities(probabilities, target_probabilities)
