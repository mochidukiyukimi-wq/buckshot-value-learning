"""Analytical CDF integrals, precision and gradient boundaries for the value loss."""

import pytest
import torch

from roulette.model.squared_cramer import (
    squared_cramer_from_probabilities,
    squared_cramer_loss,
)
from roulette.model.support import value_to_two_hot, logits_to_value
from roulette.training.update_metrics import parameter_update_metrics


def test_identical_distributions_have_exactly_zero_loss():
    probabilities = torch.softmax(torch.randn(8, 101), -1)
    assert squared_cramer_from_probabilities(probabilities, probabilities).item() == 0


def test_endpoint_teachers_and_batch_mean_use_bin_sum():
    support = torch.linspace(0, 1, 101)
    predictions = value_to_two_hot(torch.tensor([0.0, 0.25]), support)
    teachers = value_to_two_hot(torch.tensor([1.0, 0.5]), support)
    # 100 intervals differ by 1 in row 0; 25 intervals differ by 1 in row 1.
    torch.testing.assert_close(
        squared_cramer_from_probabilities(predictions, teachers), torch.tensor(0.625)
    )
    assert squared_cramer_from_probabilities(teachers, predictions).item() == 0.625


def test_same_mean_different_shapes_have_positive_distribution_loss():
    support = torch.linspace(0, 1, 101)
    center = value_to_two_hot(torch.tensor([0.5]), support)
    split = torch.zeros_like(center)
    split[0, 25] = split[0, 75] = 0.5
    assert (center * support).sum() == (split * support).sum() == 0.5
    # Fifty intervals have a CDF difference of magnitude 0.5.
    torch.testing.assert_close(
        squared_cramer_from_probabilities(center, split), torch.tensor(0.125)
    )


def test_nonnegativity_and_independent_formula_for_random_distributions():
    probabilities = torch.softmax(torch.randn(32, 101), -1)
    teachers = torch.softmax(torch.randn(32, 101), -1)
    loss = squared_cramer_from_probabilities(probabilities, teachers)
    expected_rows = []
    for prediction, teacher in zip(probabilities, teachers, strict=True):
        expected_rows.append(
            sum(
                (prediction[: k + 1].sum() - teacher[: k + 1].sum()) ** 2
                for k in range(100)
            )
            * 0.01
        )
    torch.testing.assert_close(loss, torch.stack(expected_rows).mean())
    assert loss.item() > 0
    assert (
        squared_cramer_from_probabilities(probabilities.flip(0), teachers).item() >= 0
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_fp32_calculation_and_no_teacher_gradient_under_autocast(dtype):
    logits = torch.randn(4, 101, dtype=dtype, requires_grad=True)
    teachers = (
        value_to_two_hot(torch.tensor([0.0, 0.37, 0.5, 1.0]), torch.linspace(0, 1, 101))
        .to(dtype)
        .requires_grad_()
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = squared_cramer_loss(logits, teachers)
    assert loss.dtype == torch.float32
    expected = (
        0.01
        * (
            torch.softmax(logits.float(), -1).cumsum(-1)[:, :-1]
            - teachers.detach().float().cumsum(-1)[:, :-1]
        )
        .square()
        .sum(-1)
    ).mean()
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    loss.backward()
    assert teachers.grad is None
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_inference_still_returns_softmax_expectation():
    logits = torch.randn(3, 101)
    support = torch.linspace(0, 1, 101)
    torch.testing.assert_close(
        logits_to_value(logits, support), (logits.softmax(-1) * support).sum(-1)
    )


def test_parameter_update_metrics_measure_actual_movement():
    previous = [torch.tensor([3.0, 4.0]), torch.tensor([0.0])]
    current = [torch.tensor([3.0, 4.3]), torch.tensor([0.4])]
    metrics = parameter_update_metrics(current, previous)
    assert metrics["parameter_norm"] == 5.0
    assert metrics["parameter_update_norm"] == pytest.approx(0.5)
    assert metrics["relative_parameter_update"] == pytest.approx(0.1)
