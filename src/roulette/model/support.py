import torch
from torch.nn import functional as F


def make_support(config, device="cpu") -> torch.Tensor:
    return torch.linspace(
        0, 1, config.support_points, dtype=torch.float32, device=device
    )


def value_to_two_hot(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    values = values.detach().float()
    if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Teacher values must be finite probabilities")
    if support.ndim != 1 or len(support) < 2 or support[0] != 0 or support[-1] != 1:
        raise ValueError("Support must span [0,1]")
    if not (support[1:] > support[:-1]).all():
        raise ValueError("Support must increase strictly")
    right = torch.searchsorted(support, values).clamp(1, len(support) - 1)
    left = right - 1
    right_weight = (values - support[left]) / (support[right] - support[left])
    targets = torch.zeros(
        (*values.shape, len(support)), device=values.device, dtype=torch.float32
    )
    targets.scatter_add_(-1, left.unsqueeze(-1), (1 - right_weight).unsqueeze(-1))
    targets.scatter_add_(-1, right.unsqueeze(-1), right_weight.unsqueeze(-1))
    return targets


def logits_to_value(logits: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    return (torch.softmax(logits.float(), dim=-1) * support.float()).sum(dim=-1)


def value_loss(
    logits: torch.Tensor, target_probabilities: torch.Tensor
) -> torch.Tensor:
    return (
        -(target_probabilities.detach() * F.log_softmax(logits.float(), dim=-1))
        .sum(dim=-1)
        .mean()
    )
