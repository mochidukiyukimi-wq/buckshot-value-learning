from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from roulette import _native as n
from roulette.config import load_config, ModelConfig, resolved_config
from roulette.model.encoding import encode_states
from roulette.model.support import (
    value_to_two_hot,
    logits_to_value,
)
from roulette.model.squared_cramer import squared_cramer_loss
from roulette.model.transformer import build_model
from roulette.training.ema import initialize_ema, update_ema, freeze_teacher
from roulette.training.teacher import evaluate_frontier, make_training_batch
from roulette.training.train import run_training
from roulette.training.learner import update_model
from roulette.training.checkpoint import (
    capture_rng,
    restore_rng,
    load_checkpoint,
    save_checkpoint,
)
from conftest import state_with


def tiny_model():
    return build_model(
        ModelConfig(width=16, layers=1, heads=2, ffn_width=32, support_points=11)
    )


def tiny_config(directory, units=2):
    config = load_config(None)
    config.run_dir = str(directory)
    config.model = ModelConfig(
        width=16, layers=1, heads=2, ffn_width=32, support_points=11
    )
    config.sampling.max_items_per_player = 0
    config.sampling.max_initial_shell_type_count = 1
    config.sampling.include_effects = False
    config.training.generation_units = units
    config.training.roots_per_unit = 2
    config.training.validation_roots = 2
    config.training.intra_threads = 1
    config.training.inter_threads = 1
    config.training.evaluate_every_units = 1
    config.training.checkpoint_every_units = 1
    config.training.updates_per_unit = 1
    config.search.value_batch_size = 32
    return config


def test_two_hot_preserves_mean_endpoints_and_support_points():
    support = torch.linspace(0, 1, 101)
    values = torch.cat((torch.rand(1000), support)).requires_grad_(True)
    targets = value_to_two_hot(values, support)
    torch.testing.assert_close(targets.sum(-1), torch.ones(len(values)))
    torch.testing.assert_close(
        (targets * support).sum(-1), values.detach(), atol=1e-7, rtol=1e-6
    )
    assert not targets.requires_grad
    assert targets[1000, 0] == 1
    assert targets[-1, -1] == 1
    assert (targets > 0).sum(-1).max() <= 2


def test_finite_logits_endpoints_with_tolerance_and_squared_cramer():
    support = torch.linspace(0, 1, 11)
    targets = value_to_two_hot(torch.tensor([0.0, 0.37, 1.0]), support)
    logits = targets.clamp_min(1e-15).log()
    predictions = logits_to_value(logits, support)
    torch.testing.assert_close(
        predictions, torch.tensor([0.0, 0.37, 1.0]), atol=1e-6, rtol=1e-6
    )
    assert squared_cramer_loss(logits, targets).item() < 1e-12


def test_item_slot_permutation_and_batch_padding_do_not_change_prediction():
    model = tiny_model().eval()
    support = torch.linspace(0, 1, 11)
    state = state_with(
        inventory0=(n.Item.BEER, n.Item.MEDICINE), inventory1=(n.Item.KNIFE,)
    )
    features = encode_states([state])
    permuted = features.copy()
    actor_item_columns = n.FEATURE_SCHEMA.inventory_columns(
        n.FEATURE_SCHEMA.actor_player_index
    ).columns
    permuted[:, actor_item_columns] = permuted[:, actor_item_columns][:, ::-1]
    with freeze_teacher(model, 0) as teacher:
        single = evaluate_frontier(teacher, features, support, 1)
        padded = evaluate_frontier(teacher, features, support, 32)
        reordered = evaluate_frontier(teacher, permuted, support, 1)
    np.testing.assert_allclose(single, padded, atol=1e-6)
    np.testing.assert_allclose(single, reordered, atol=1e-6)
    assert model.encoder(torch.from_numpy(features)).shape == (
        1,
        n.FEATURE_SCHEMA.token_count,
        model.encoder.global_projection.out_features,
    )


def test_ema_formula_and_active_teacher_lease():
    model = tiny_model()
    ema = initialize_ema(model)
    old = [parameter.clone() for parameter in ema.parameters()]
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(2)
    update_ema(ema, model, 0.75)
    for before, after in zip(old, ema.parameters(), strict=True):
        torch.testing.assert_close(after, before + 0.5)
    with freeze_teacher(ema, 7):
        with pytest.raises(RuntimeError, match="active"):
            update_ema(ema, model, 0.9)
    assert all(not parameter.requires_grad for parameter in ema.parameters())


def test_teacher_detects_external_weight_mutation():
    ema = initialize_ema(tiny_model())
    with pytest.raises(RuntimeError, match="changed"):
        with freeze_teacher(ema, 0):
            with torch.no_grad():
                next(ema.parameters()).add_(1)


def test_fixed_teacher_labels_can_be_fitted_with_finite_gradients():
    torch.manual_seed(7)
    model = tiny_model()
    support = torch.linspace(0, 1, 11)
    ema = initialize_ema(model)
    roots = [state_with(live=1, blank=0, hp=(1, 1)), state_with(hp=(1, 1))]
    with freeze_teacher(ema, 0) as teacher:
        batch = make_training_batch(roots, teacher, support, n.SearchConfig())
    features = torch.from_numpy(batch.features)
    values = torch.from_numpy(batch.values)
    targets = value_to_two_hot(values, support)
    initial_loss = squared_cramer_loss(model.predict_logits(features), targets).item()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(50):
        metrics = update_model(model, features, values, optimizer, support)
        assert all(np.isfinite(value) for value in metrics.values())
    final_loss = squared_cramer_loss(model.predict_logits(features), targets).item()
    assert final_loss < initial_loss * 0.5


def test_save_resume_reproduces_training_exactly(tmp_path):
    continuous_config = tiny_config(tmp_path / "continuous", units=2)
    continuous = run_training(continuous_config)
    split_config = tiny_config(tmp_path / "split", units=1)
    run_training(split_config)
    split_config.training.generation_units = 2
    resumed = run_training(split_config, resume=True)
    assert continuous["step"] == resumed["step"] == 2
    first = load_checkpoint(tmp_path / "continuous/latest.pt", continuous_config)
    second = load_checkpoint(tmp_path / "split/latest.pt", split_config)
    for name in ("model", "ema"):
        for key in first[name]:
            assert torch.equal(first[name][key], second[name][key]), key
    for key, value in first["optimizer"]["state"].items():
        for field, tensor in value.items():
            assert torch.equal(tensor, second["optimizer"]["state"][key][field])
    assert torch.equal(first["rng"]["torch"], second["rng"]["torch"])
    assert len(list((tmp_path / "split").glob("*.pt"))) == 1
    assert not list((tmp_path / "split").glob("*.tmp"))
    events = [
        json.loads(line)
        for line in (tmp_path / "split/metrics.jsonl").read_text().splitlines()
    ]
    assert any(event.get("reason") == "periodic_unit" for event in events)
    assert events[-1]["reason"] == "exit_completed"


def test_failure_still_saves_latest_and_logs_failure(tmp_path):
    import signal

    previous_handlers = {
        number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)
    }
    config = tiny_config(tmp_path)
    config.search.max_frontier_nodes = 1
    with pytest.raises(RuntimeError, match="capacity"):
        run_training(config)
    saved = load_checkpoint(tmp_path / "latest.pt", config)
    assert saved["save_reason"] == "exit_failed"
    events = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert any(
        event["event"] == "failure" and event["resource_limit"] == 1 for event in events
    )
    assert all(
        signal.getsignal(number) == handler
        for number, handler in previous_handlers.items()
    )


@pytest.mark.parametrize("generation_units", [2, None])
def test_runtime_limit_saves_at_exit(tmp_path, generation_units):
    config = tiny_config(tmp_path, units=generation_units)
    config.training.max_runtime_seconds = 0.001
    summary = run_training(config)
    assert summary["status"] == "runtime_limit"
    assert (
        load_checkpoint(tmp_path / "latest.pt", config)["save_reason"]
        == "exit_runtime_limit"
    )


def test_checkpoint_rejects_incompatible_schema_and_model(tmp_path):
    config = tiny_config(tmp_path)
    run_training(config)
    changed = deepcopy(config)
    changed.model.width = 32
    with pytest.raises(ValueError, match="differs"):
        load_checkpoint(tmp_path / "latest.pt", changed)
    saved = load_checkpoint(tmp_path / "latest.pt", config)
    saved["metadata"]["input_schema"] = "wrong"
    save_checkpoint(tmp_path / "latest.pt", saved)
    with pytest.raises(ValueError, match="input_schema"):
        load_checkpoint(tmp_path / "latest.pt", config)
    with pytest.raises(ValueError, match="latest.pt"):
        save_checkpoint(tmp_path / "old-model.pt", saved)


def test_rng_capture_and_restore():
    state = capture_rng()
    draws = torch.rand(5)
    restore_rng(state)
    assert torch.equal(draws, torch.rand(5))


def test_periodic_time_checkpoint_is_latest_only(tmp_path):
    config = tiny_config(tmp_path, units=1)
    config.training.checkpoint_every_units = 10
    config.training.checkpoint_every_seconds = 0.001
    run_training(config)
    events = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert any(event.get("reason") == "periodic_time" for event in events)
    assert len(list(tmp_path.glob("*.pt"))) == 1
    assert (
        load_checkpoint(tmp_path / "latest.pt", config)["save_reason"]
        == "exit_completed"
    )


def test_sigint_stops_safely_and_saves_at_exit(tmp_path, monkeypatch):
    import signal
    from roulette.training import train as train_module

    original = train_module.make_training_batch
    previous_handler = signal.getsignal(signal.SIGINT)

    def interrupt_generation(*args, **kwargs):
        signal.raise_signal(signal.SIGINT)
        return original(*args, **kwargs)

    monkeypatch.setattr(train_module, "make_training_batch", interrupt_generation)
    config = tiny_config(tmp_path)
    summary = run_training(config)
    assert summary["status"] == "interrupted"
    assert (
        load_checkpoint(tmp_path / "latest.pt", config)["save_reason"]
        == "exit_interrupted"
    )
    assert signal.getsignal(signal.SIGINT) == previous_handler


def test_resume_preserves_support_rounding_across_runtimes(tmp_path):
    config = tiny_config(tmp_path, units=1)
    run_training(config)
    saved = load_checkpoint(tmp_path / "latest.pt", config)
    rounded_support = saved["support"].clone()
    rounded_support[1] = torch.nextafter(rounded_support[1], torch.tensor(1.0))
    saved["support"] = rounded_support
    save_checkpoint(tmp_path / "latest.pt", saved)
    run_training(config, resume=True)
    resumed = load_checkpoint(tmp_path / "latest.pt", config)
    assert torch.equal(resumed["support"], rounded_support)
    resumed["support"][1] += 0.0001
    save_checkpoint(tmp_path / "latest.pt", resumed)
    with pytest.raises(ValueError, match="categorical support"):
        load_checkpoint(tmp_path / "latest.pt", config)


def test_resume_without_count_limit_passes_previous_limit_and_saves_on_stop(
    tmp_path, monkeypatch
):
    import signal

    config = tiny_config(tmp_path, units=1)
    run_training(config)
    config.training.generation_units = None
    config_path = tmp_path / "configuration.json"
    config_path.write_text(json.dumps(resolved_config(config)))
    config = load_config(config_path)
    assert config.training.generation_units is None

    from roulette.training import learner as learner_module

    original_update_model = learner_module.update_model
    continued_updates = 0

    def stop_after_two_continued_updates(*args, **kwargs):
        nonlocal continued_updates
        metrics = original_update_model(*args, **kwargs)
        continued_updates += 1
        if continued_updates == 2:
            signal.raise_signal(signal.SIGTERM)
        return metrics

    monkeypatch.setattr(
        learner_module, "update_model", stop_after_two_continued_updates
    )
    summary = run_training(config, resume=True)
    saved = load_checkpoint(tmp_path / "latest.pt", config)
    assert summary["initial_step"] == 1
    assert summary["step"] == saved["step"] == 3
    assert saved["completed_units"] == 3
    assert summary["status"] == "interrupted"
    assert saved["save_reason"] == "exit_interrupted"
    assert saved["metadata"]["config"]["training"]["generation_units"] is None


@pytest.mark.parametrize("objective", ["cross_entropy", None])
def test_checkpoint_requires_current_training_objective(tmp_path, objective):
    config = tiny_config(tmp_path, units=1)
    run_training(config)
    saved = load_checkpoint(tmp_path / "latest.pt", config)
    saved["metadata"]["training_objective"] = objective
    save_checkpoint(tmp_path / "latest.pt", saved)
    with pytest.raises(ValueError, match="training_objective"):
        load_checkpoint(tmp_path / "latest.pt", config)


def test_checkpoint_failure_closes_logs_and_restores_signal_handlers(
    tmp_path, monkeypatch
):
    import signal
    from roulette.training import train as train_module

    previous_handler = signal.getsignal(signal.SIGINT)
    closed_logs = []
    original_close = train_module.RunLogger.close

    def record_close(logger):
        original_close(logger)
        closed_logs.append(logger)

    def fail_checkpoint(*args, **kwargs):
        raise OSError("checkpoint write failed")

    monkeypatch.setattr(train_module.RunLogger, "close", record_close)
    monkeypatch.setattr(train_module, "save_training_state", fail_checkpoint)
    with pytest.raises(OSError, match="checkpoint write failed"):
        run_training(tiny_config(tmp_path))
    assert len(closed_logs) == 1
    assert closed_logs[0].jsonl.closed
    assert signal.getsignal(signal.SIGINT) == previous_handler


def test_nonfinite_gradient_does_not_update_model_or_optimizer():
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters())
    before = [parameter.detach().clone() for parameter in model.parameters()]
    parameter = next(model.parameters())
    hook = parameter.register_hook(
        lambda gradient: torch.full_like(gradient, float("nan"))
    )
    try:
        features = torch.from_numpy(encode_states([state_with()]))
        with pytest.raises(FloatingPointError, match="Non-finite gradient"):
            update_model(
                model,
                features,
                torch.tensor([0.5]),
                optimizer,
                torch.linspace(0, 1, 11),
            )
    finally:
        hook.remove()
    assert all(
        torch.equal(saved, current)
        for saved, current in zip(before, model.parameters(), strict=True)
    )
    assert not optimizer.state
