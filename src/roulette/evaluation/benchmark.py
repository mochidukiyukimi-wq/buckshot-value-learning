import time

import numpy as np

from .. import _native as native
from ..config import native_sampling, native_search
from ..model.encoding import encode_states
from ..model.support import make_support
from ..observability.metrics import execution_environment
from ..training.ema import freeze_teacher
from ..training.teacher import evaluate_frontier, make_training_batch


def benchmark_search(cases, search_config, work_budget=128) -> dict:
    start = time.perf_counter()
    statistics = []
    for state in cases:
        session = native.SearchSession(state, search_config, 0)
        while True:
            progress = session.advance(work_budget)
            if progress.status == native.SearchStatus.NEEDS_VALUES:
                session.submit_values(
                    progress.request.request_id,
                    np.full(len(progress.request.features), 0.5, np.float32),
                )
            elif progress.status == native.SearchStatus.NEEDS_LABEL_DRAIN:
                session.take_labels(search_config.label_buffer_size)
            elif progress.status == native.SearchStatus.COMPLETE:
                statistics.append(session.result().stats)
                break
            elif progress.status == native.SearchStatus.RESOURCE_LIMIT:
                raise RuntimeError("Benchmark search exceeded capacity")
    return {
        "seconds": time.perf_counter() - start,
        "internal_nodes": sum(stat.internal_nodes for stat in statistics),
        "frontier_nodes": sum(stat.frontier_nodes for stat in statistics),
        "transitions": sum(stat.transitions for stat in statistics),
    }


def benchmark_inference(model, batch_sizes, device, features, support) -> list[dict]:
    measurements = []
    with freeze_teacher(model, 0) as teacher:
        for batch_size in batch_sizes:
            batch = np.ascontiguousarray(
                np.resize(features, (batch_size, native.FEATURE_COUNT))
            )
            warmup_start = time.perf_counter()
            evaluate_frontier(teacher, batch, support, batch_size)
            warmup_seconds = time.perf_counter() - warmup_start
            start = time.perf_counter()
            for _ in range(3):
                evaluate_frontier(teacher, batch, support, batch_size)
            seconds = (time.perf_counter() - start) / 3
            measurements.append(
                {
                    "batch_size": batch_size,
                    "warmup_seconds": warmup_seconds,
                    "mean_seconds": seconds,
                    "rows_per_second": batch_size / seconds,
                    "device": str(device),
                    "dtype": "float32",
                    "compile": False,
                }
            )
    return measurements


def benchmark_pipeline(config, model, support=None) -> dict:
    roots = native.sample_roots(native_sampling(config), 2, config.seed + 900001)
    if support is None:
        support = make_support(config.model, config.device)
    search_config = native_search(config)
    inference = benchmark_inference(
        model,
        [1, 32, config.search.value_batch_size],
        config.device,
        encode_states(roots),
        support,
    )
    pure_search = benchmark_search(roots, search_config, config.search.work_budget)
    start = time.perf_counter()
    with freeze_teacher(model, 0) as teacher:
        batch = make_training_batch(
            roots, teacher, support, search_config, config.search.work_budget
        )
    return {
        "environment": execution_environment(config),
        "inference": inference,
        "search": pure_search,
        "pipeline_seconds": time.perf_counter() - start,
        "pipeline": batch.metrics.summarize_interval(),
    }
