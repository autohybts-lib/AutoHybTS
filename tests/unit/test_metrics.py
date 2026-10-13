"""Metric values against hand computations, conventions for edge cases, and T18."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from autohybts.evaluation.metrics import METRICS, compute_metrics, resolve_metrics
from autohybts.exceptions import ConfigurationError

ACTUAL = np.array([10.0, 12.0, 8.0])
PRED = np.array([11.0, 10.0, 8.0])
TRAIN = np.array([1.0, 3.0, 2.0, 6.0])


def test_hand_computed_values() -> None:
    values = compute_metrics(resolve_metrics(METRICS), ACTUAL, PRED, TRAIN, seasonal_period=1)
    assert values["MAE"] == pytest.approx(1.0)
    assert values["RMSE"] == pytest.approx(np.sqrt(5 / 3))
    assert values["sMAPE"] == pytest.approx((2 / 21 + 4 / 22 + 0) / 3)
    # Naive scale on training: mean(|2|, |-1|, |4|) = 7/3.
    assert values["MASE"] == pytest.approx(1.0 / (7 / 3))
    assert values["WAPE"] == pytest.approx(3 / 30)
    assert values["RRMSE"] == pytest.approx(np.sqrt(5 / 3) / 10)


def test_seasonal_mase_scale() -> None:
    train = np.array([1.0, 2.0, 4.0, 2.0, 5.0])  # m = 2: |4-1|, |2-2|, |5-4| -> mean 4/3
    assert METRICS["MASE"](ACTUAL, PRED, train, 2) == pytest.approx(1.0 / (4 / 3))


def test_smape_counts_zero_over_zero_as_perfect() -> None:
    assert METRICS["sMAPE"]([0.0, 2.0], [0.0, 2.0], TRAIN) == 0.0


@pytest.mark.parametrize(
    ("name", "train", "actual"),
    [
        ("MASE", np.full(10, 5.0), ACTUAL),  # flat training: zero scale
        ("MASE", np.array([1.0, 2.0]), ACTUAL),  # training not longer than m
        ("WAPE", TRAIN, np.zeros(3)),
        ("RRMSE", TRAIN, np.array([1.0, -1.0, 0.0])),  # mean of actuals is zero
    ],
)
def test_undefined_metrics_are_nan_not_zero_or_inf(
    name: str, train: np.ndarray, actual: np.ndarray
) -> None:
    period = 2 if name == "MASE" and train.size == 2 else 1
    assert np.isnan(METRICS[name](actual, PRED, train, period))


def test_mase_scale_comes_only_from_training() -> None:
    # T18: changing the targets moves the numerator only; the scale is a training statistic.
    shocked = ACTUAL + 100.0
    ratio = METRICS["MASE"](shocked, PRED, TRAIN) / METRICS["MAE"](shocked, PRED, TRAIN)
    assert ratio == pytest.approx(
        METRICS["MASE"](ACTUAL, PRED, TRAIN) / METRICS["MAE"](ACTUAL, PRED, TRAIN)
    )


@given(
    actual=st.lists(st.floats(1.0, 1e4), min_size=1, max_size=30),
    noise=st.floats(-50, 50),
)
def test_rrmse_matches_the_legacy_definition(actual: list[float], noise: float) -> None:
    # Legacy: RMSE of steps 1..H divided by the mean of the actuals over the same steps.
    a = np.asarray(actual)
    p = a + noise
    legacy = np.sqrt(np.mean((a - p) ** 2)) / np.mean(a)
    assert METRICS["RRMSE"](a, p, TRAIN) == pytest.approx(legacy)


def test_every_metric_documents_itself() -> None:
    for name, metric in METRICS.items():
        assert metric.name == name
        assert metric.version >= 1
        assert metric.formula


@pytest.mark.parametrize("names", [["MSLE"], [], ["MAE", "MAE"]])
def test_invalid_metric_lists(names: list[str]) -> None:
    with pytest.raises(ConfigurationError):
        resolve_metrics(names)


def test_shape_mismatch() -> None:
    with pytest.raises(ConfigurationError, match="mesmo tamanho"):
        METRICS["MAE"]([1.0, 2.0], [1.0], TRAIN)
