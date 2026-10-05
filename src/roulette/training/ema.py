from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass

import torch


def initialize_ema(model):
    ema = deepcopy(model).eval()
    ema.requires_grad_(False)
    ema._active_teacher_leases = 0
    return ema


@torch.no_grad()
def update_ema(ema, model, decay: float) -> None:
    if not 0 <= decay < 1:
        raise ValueError("EMA decay must be [0,1)")
    if getattr(ema, "_active_teacher_leases", 0):
        raise RuntimeError("Cannot update EMA while a teacher generation is active")
    for ema_parameter, model_parameter in zip(
        ema.parameters(), model.parameters(), strict=True
    ):
        ema_parameter.mul_(decay).add_(model_parameter.detach(), alpha=1 - decay)
    for ema_buffer, model_buffer in zip(ema.buffers(), model.buffers(), strict=True):
        ema_buffer.copy_(model_buffer)


@dataclass(frozen=True)
class FrozenTeacher:
    model: object
    version: int
    parameter_versions: tuple[int, ...]

    def verify_unchanged(self) -> None:
        if self.parameter_versions != tuple(
            parameter._version for parameter in self.model.parameters()
        ):
            raise RuntimeError("Teacher weights changed during search")


@contextmanager
def freeze_teacher(ema, version: int):
    ema.eval()
    ema._active_teacher_leases = getattr(ema, "_active_teacher_leases", 0) + 1
    teacher = FrozenTeacher(
        ema, version, tuple(parameter._version for parameter in ema.parameters())
    )
    try:
        yield teacher
        teacher.verify_unchanged()
    finally:
        ema._active_teacher_leases -= 1
