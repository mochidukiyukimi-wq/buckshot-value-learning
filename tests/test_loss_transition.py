"""Loss migration is explicit and retains model, EMA, AdamW, RNG and history."""

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

from roulette.config import resolved_config
from roulette.training.checkpoint import capture_rng
import json

import pytest

from roulette.training.checkpoint import load_checkpoint, save_checkpoint
from roulette.training.train import run_training
from test_model_training import tiny_config, assert_checkpoint_values_equal


def legacy_checkpoint(tmp_path):
    config = tiny_config(tmp_path, units=1)
    run_training(config)
    saved = load_checkpoint(tmp_path / "latest.pt", config)
    saved["metadata"].pop("training_objective")
    save_checkpoint(tmp_path / "latest.pt", saved)
    return config, saved


def test_legacy_ce_requires_explicit_transition_and_preserves_state(tmp_path):
    config, before = legacy_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="training objective"):
        run_training(config, resume=True)
    # Read-only inference can still inspect historical CE models.
    load_checkpoint(tmp_path / "latest.pt", config, for_inference=True)
    summary = run_training(config, resume=True, loss_transition=True)
    after = load_checkpoint(tmp_path / "latest.pt", config)
    assert summary["initial_step"] == summary["step"] == before["step"]
    for field in (
        "model",
        "ema",
        "optimizer",
        "rng",
        "support",
        "validation_history",
        "stage_lineage",
    ):
        assert_checkpoint_values_equal(before[field], after[field])
    assert after["metadata"]["training_objective"] == "squared_cramer"
    adjustment = after["training_adjustments"][-1]
    assert adjustment["previous_value"] == "cross_entropy"
    assert adjustment["new_value"] == "squared_cramer"
    assert adjustment["step"] == before["step"]
    config.training.generation_units = 2
    run_training(config, resume=True)
    latest = load_checkpoint(tmp_path / "latest.pt", config)
    assert latest["step"] > before["step"]
    assert latest["training_adjustments"] == after["training_adjustments"]


@pytest.mark.parametrize(
    "section,field,new_value",
    [
        ("training", "ema_decay", 0.99),
        ("sampling", "max_initial_shell_type_count", 2),
        ("search", "memoize", False),
        ("model", "support_points", 101),
    ],
)
def test_loss_transition_cannot_change_other_settings(
    tmp_path, section, field, new_value
):
    config, _ = legacy_checkpoint(tmp_path)
    changed = deepcopy(config)
    setattr(getattr(changed, section), field, new_value)
    with pytest.raises(ValueError, match="differs"):
        load_checkpoint(tmp_path / "latest.pt", changed, loss_transition=True)


def test_loss_transition_requires_resume_and_is_separate(tmp_path):
    config = tiny_config(tmp_path)
    with pytest.raises(ValueError, match="requires --resume"):
        run_training(config, loss_transition=True)
    with pytest.raises(ValueError, match="must be separate"):
        run_training(
            config, resume=True, ema_decay_transition=True, loss_transition=True
        )
    from roulette.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["train", "--resume", "--ema-decay-transition", "--loss-transition"]
        )


def test_loss_transition_changes_only_configured_lr_and_retains_adamw_moments(tmp_path):
    config, before = legacy_checkpoint(tmp_path)
    previous_learning_rate = config.training.learning_rate
    config.training.learning_rate = 0.00001
    with pytest.raises(ValueError, match="differs"):
        load_checkpoint(tmp_path / "latest.pt", config, for_inference=True)
    run_training(config, resume=True, loss_transition=True)
    after = load_checkpoint(tmp_path / "latest.pt", config)
    for field in (
        "model",
        "ema",
        "rng",
        "support",
        "step",
        "completed_units",
        "validation_history",
    ):
        assert_checkpoint_values_equal(before[field], after[field])
    assert_checkpoint_values_equal(
        before["optimizer"]["state"], after["optimizer"]["state"]
    )
    for previous_group, resumed_group in zip(
        before["optimizer"]["param_groups"],
        after["optimizer"]["param_groups"],
        strict=True,
    ):
        assert resumed_group["lr"] == 0.00001
        assert {key: value for key, value in resumed_group.items() if key != "lr"} == {
            key: value for key, value in previous_group.items() if key != "lr"
        }
    assert after["training_adjustments"][-1]["learning_rate"] == {
        "previous_value": previous_learning_rate,
        "new_value": 0.00001,
    }
    config.training.generation_units = 2
    run_training(config, resume=True)
    assert (
        load_checkpoint(tmp_path / "latest.pt", config)["optimizer"]["param_groups"][0][
            "lr"
        ]
        == 0.00001
    )


def test_loss_transition_cannot_retune_an_already_cramer_checkpoint(tmp_path):
    config = tiny_config(tmp_path, units=1)
    run_training(config)
    config.training.learning_rate = 0.00001
    with pytest.raises(ValueError, match="differs"):
        load_checkpoint(tmp_path / "latest.pt", config, loss_transition=True)


def test_transition_diagnostic_preserves_checkpoint_and_rng(tmp_path):
    config, _ = legacy_checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(resolved_config(config)))
    checkpoint_path = tmp_path / "latest.pt"
    before_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    before_rng = capture_rng()
    module_spec = importlib.util.spec_from_file_location(
        "measure_loss_transition",
        Path(__file__).parents[1] / "tools/measure_loss_transition.py",
    )
    diagnostic_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(diagnostic_module)
    result = diagnostic_module.measure(
        config_path,
        tmp_path / "measurement.json",
        unit_count=2,
        learning_rates=(0.00001, 0.001),
    )
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == before_digest
    assert_checkpoint_values_equal(before_rng, capture_rng())
    branches = result["branches"]
    assert (
        branches["squared_cramer"]["initial_heldout_squared_cramer"]
        == branches["cross_entropy_reference"]["initial_heldout_squared_cramer"]
    )
    assert (
        branches["squared_cramer"]["updates"]
        == branches["cross_entropy_reference"]["updates"]
        == 2
    )
    assert branches["squared_cramer_lr_1e-05"]["learning_rate"] == 0.00001
    assert branches["squared_cramer_lr_0.001"]["learning_rate"] == 0.001
    for branch in branches.values():
        assert branch["mean_gradient_norm"] > 0 and branch["mean_update_norm"] > 0
