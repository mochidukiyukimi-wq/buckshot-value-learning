"""Configure PyTorch execution for training, evaluation, and diagnostics."""

import torch

from .config import Config


def configure_runtime(config: Config) -> None:
    torch.set_num_threads(config.training.intra_threads)
    if torch.get_num_interop_threads() != config.training.inter_threads:
        torch.set_num_interop_threads(config.training.inter_threads)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
