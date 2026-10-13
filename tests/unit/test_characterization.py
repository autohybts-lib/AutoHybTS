"""Characterization with the replication project's two rules: ADF stationarity, STL strength."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from autohybts.data.characterization import (
    boxcox_transform,
    characterize,
    stl_boxcox_strength,
)

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def noise(n: int = 400, seed: int = 0) -> np.ndarray:
    return 10 + np.random.default_rng(seed).normal(size=n)


def seasonal_walk(n: int = 400, seed: int = 0) -> np.ndarray:
    t = np.arange(n)
    walk = np.cumsum(np.random.default_rng(seed).normal(0, 0.3, n))
    return 50 + walk + 6 * np.sin(2 * np.pi * t / 7)


def plain_walk(n: int = 400, seed: int = 0) -> np.ndarray:
    return 50 + np.cumsum(np.random.default_rng(seed).normal(size=n))


def test_white_noise_is_stationary_and_skips_the_seasonal_rule() -> None:
    record = characterize(noise(), periods=[7])
    assert record["stationary"]
    assert record["diff_order"] == 0
    assert record["category"] == "stationary"
    assert np.isnan(record["fs_7"])


def test_a_weekly_cycle_on_a_random_walk_is_seasonal() -> None:
    record = characterize(seasonal_walk(), periods=[7, 30])
    assert not record["stationary"]
    assert record["diff_order"] == 1
    assert record["dominant_period"] == 7
    assert record["fs"] > 0.5
    assert record["category"] == "seasonal"


def test_a_plain_random_walk_is_nonseasonal() -> None:
    record = characterize(plain_walk(), periods=[7])
    assert record["diff_order"] == 1
    assert record["category"] == "nonseasonal"


def test_every_order_reports_adf_and_kpss() -> None:
    record = characterize(plain_walk(), periods=[7], max_d=2)
    for order in range(3):
        for key in ("adf_stat", "adf_pvalue", "adf_crit5", "kpss_stat", "kpss_pvalue"):
            assert f"{key}_d{order}" in record


def test_boxcox_shifts_non_positive_series_and_respects_bounds() -> None:
    transformed, fitted = boxcox_transform(np.linspace(-5, 5, 50) ** 2 - 3)
    assert np.all(np.isfinite(transformed))
    assert -1.0 <= fitted <= 2.0


def test_strength_needs_two_cycles() -> None:
    strength, fitted = stl_boxcox_strength(seasonal_walk(13), 7)
    assert np.isnan(strength)
    assert np.isnan(fitted)
