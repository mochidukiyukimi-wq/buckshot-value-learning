"""Configuration of computation and the declared finite sampling domain; game rules are fixed."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from . import _native as native


@dataclass
class ModelConfig:
    width: int = 256
    layers: int = 8
    heads: int = 8
    ffn_width: int = 1024
    support_points: int = 101


@dataclass
class SamplingConfig:
    max_items_per_player: int = 2
    max_initial_shell_type_count: int = 4
    include_effects: bool = True
    include_replenishment_boundaries: bool = True


@dataclass
class SearchConfig:
    max_internal_nodes: int = 20000
    max_frontier_nodes: int = 200000
    value_batch_size: int = 256
    label_buffer_size: int = 4096
    work_budget: int = 128
    memoize: bool = True


@dataclass
class TrainingConfig:
    generation_units: int | None = 20
    roots_per_unit: int = 2
    batch_size: int = 64
    updates_per_unit: int = 4
    max_epochs_per_unit: int = 2
    learning_rate: float = 0.0001
    ema_decay: float = 0.95
    validation_roots: int = 4
    evaluate_every_units: int = 5
    checkpoint_every_units: int = 5
    checkpoint_every_seconds: float = 120
    max_runtime_seconds: float = 600
    intra_threads: int = 4
    inter_threads: int = 1


@dataclass
class Config:
    seed: int = 20261005
    device: str = "cpu"
    run_dir: str = "runs/cpu_pilot"
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(path: str | Path | None, overrides: dict | None = None) -> Config:
    values = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    values.update(overrides or {})
    for name, section_type in (
        ("model", ModelConfig),
        ("sampling", SamplingConfig),
        ("search", SearchConfig),
        ("training", TrainingConfig),
    ):
        values[name] = section_type(**values.get(name, {}))
    config = Config(**values)
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    model = config.model
    if (
        min(model.width, model.layers, model.heads, model.ffn_width) <= 0
        or model.width % model.heads
    ):
        raise ValueError(
            "Model dimensions must be positive; width must be divisible by heads"
        )
    if model.support_points < 2:
        raise ValueError("At least two categorical support points are required")
    if config.device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    native.validate_sampling_domain(native_sampling(config))
    search = config.search
    if (
        min(
            search.max_internal_nodes,
            search.max_frontier_nodes,
            search.value_batch_size,
            search.label_buffer_size,
            search.work_budget,
        )
        <= 0
    ):
        raise ValueError("Search capacities must be positive")
    training = config.training
    if training.generation_units is not None and (
        type(training.generation_units) is not int or training.generation_units <= 0
    ):
        raise ValueError(
            "generation_units must be a positive integer or null for no count limit"
        )
    integer_limits = (
        training.roots_per_unit,
        training.batch_size,
        training.updates_per_unit,
        training.max_epochs_per_unit,
        training.validation_roots,
        training.evaluate_every_units,
        training.checkpoint_every_units,
        training.intra_threads,
        training.inter_threads,
    )
    if min(integer_limits) <= 0:
        raise ValueError("Training counts and thread counts must be positive")
    if not 0 <= training.ema_decay < 1 or training.learning_rate <= 0:
        raise ValueError("EMA decay must be [0,1); learning rate must be positive")
    if training.checkpoint_every_seconds <= 0 or training.max_runtime_seconds <= 0:
        raise ValueError("Checkpoint and runtime intervals must be positive")


def resolved_config(config: Config) -> dict:
    return asdict(config)


def native_sampling(config: Config) -> native.SamplingDomain:
    domain = native.SamplingDomain()
    for key, value in asdict(config.sampling).items():
        setattr(domain, key, value)
    return domain


def native_search(config: Config) -> native.SearchConfig:
    search = native.SearchConfig()
    for key, value in asdict(config.search).items():
        if key != "work_budget":
            setattr(search, key, value)
    return search
