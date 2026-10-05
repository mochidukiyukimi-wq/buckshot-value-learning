"""Drive native sessions; only completed root label collections become training data."""

from dataclasses import dataclass
import time

import numpy as np
import torch

from .. import _native as native
from ..model.encoding import validate_features
from ..model.support import logits_to_value
from ..observability.metrics import Metrics
from .ema import FrozenTeacher, freeze_teacher as freeze_teacher


class SearchResourceLimit(RuntimeError):
    pass


@dataclass
class TrainingBatch:
    features: np.ndarray
    values: np.ndarray
    keys: list[bytes]
    results: list
    metrics: Metrics


@torch.inference_mode()
def evaluate_frontier(
    teacher: FrozenTeacher,
    features: np.ndarray,
    support: torch.Tensor,
    batch_size: int,
    metrics: Metrics | None = None,
) -> np.ndarray:
    teacher.verify_unchanged()
    validate_features(features)
    outputs = []
    device = next(teacher.model.parameters()).device
    for start in range(0, len(features), batch_size):
        actual = features[start : start + batch_size]
        # One fixed inference shape; duplicate padding is discarded after inference.
        padded = np.empty((batch_size, native.FEATURE_COUNT), dtype=np.float32)
        padded[: len(actual)] = actual
        padded[len(actual) :] = actual[-1]
        transfer_start = time.perf_counter()
        tensor = torch.from_numpy(padded).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        transfer_seconds = time.perf_counter() - transfer_start
        inference_start = time.perf_counter()
        logits = teacher.model.predict_logits(tensor)
        values = logits_to_value(logits, support).clamp(0, 1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - inference_start
        transfer_start = time.perf_counter()
        outputs.append(values[: len(actual)].cpu().numpy().copy())
        transfer_seconds += time.perf_counter() - transfer_start
        if metrics:
            metrics.record_inference(len(actual), inference_seconds, transfer_seconds)
    teacher.verify_unchanged()
    return np.concatenate(outputs) if outputs else np.empty(0, np.float32)


def make_training_batch(
    roots,
    teacher: FrozenTeacher,
    support,
    search_config,
    work_budget: int = 128,
    heartbeat=None,
) -> TrainingBatch:
    sessions = [
        native.SearchSession(root, search_config, teacher.version) for root in roots
    ]
    features_by_root = [[] for _ in roots]
    values_by_root = [[] for _ in roots]
    keys_by_root = [[] for _ in roots]
    results = [None for _ in roots]
    metrics = Metrics(root_count=len(roots))
    while any(result is None for result in results):
        requests = []
        for root_index, session in enumerate(sessions):
            if results[root_index] is not None:
                continue
            search_start = time.perf_counter()
            progress = session.advance(work_budget)
            metrics.search_seconds += time.perf_counter() - search_start
            if progress.status == native.SearchStatus.RESOURCE_LIMIT:
                metrics.resource_limit += 1
                metrics.incomplete_roots = sum(result is None for result in results)
                raise SearchResourceLimit(
                    f"Root {root_index} exceeded search capacity; no partial labels are accepted"
                )
            if progress.status == native.SearchStatus.NEEDS_VALUES:
                if progress.request.evaluator_version != teacher.version:
                    raise RuntimeError("Evaluation version mismatch")
                requests.append(
                    (root_index, progress.request, progress.request.features)
                )
            if progress.status in (
                native.SearchStatus.NEEDS_LABEL_DRAIN,
                native.SearchStatus.COMPLETE,
            ):
                drained = session.take_labels(search_config.label_buffer_size)
                while len(drained["values"]):
                    features_by_root[root_index].append(drained["features"])
                    values_by_root[root_index].append(drained["values"])
                    keys_by_root[root_index].extend(drained["keys"])
                    drained = session.take_labels(search_config.label_buffer_size)
            if progress.status == native.SearchStatus.COMPLETE:
                results[root_index] = session.result()
                metrics.record_search(results[root_index].stats)
        if requests:
            concatenated = np.concatenate([request[2] for request in requests])
            predictions = evaluate_frontier(
                teacher, concatenated, support, search_config.value_batch_size, metrics
            )
            offset = 0
            for root_index, request, features in requests:
                rows = len(features)
                sessions[root_index].submit_values(
                    request.request_id,
                    np.ascontiguousarray(predictions[offset : offset + rows]),
                )
                offset += rows
        if heartbeat:
            heartbeat()
    teacher.verify_unchanged()
    all_features = [batch for batches in features_by_root for batch in batches]
    all_values = [batch for batches in values_by_root for batch in batches]
    keys = [key for root_keys in keys_by_root for key in root_keys]
    metrics.teacher_rows = sum(len(values) for values in all_values)
    return TrainingBatch(
        np.concatenate(all_features), np.concatenate(all_values), keys, results, metrics
    )


def generate_root_labels(root, teacher, support, search_config, work_budget=128):
    return make_training_batch([root], teacher, support, search_config, work_budget)
