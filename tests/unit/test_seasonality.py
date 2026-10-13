"""Seasonality diagnostics: classical decomposition, strength and the rule by position."""

from __future__ import annotations

import numpy as np
import pytest

from autohybts.data.seasonality import (
    MIN_CYCLES,
    classical_decomposition,
    detect_seasonality,
    seasonal_strength,
)


def seasonal(n: int = 140, period: int = 7, amplitude: float = 3.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 20 + 0.05 * t + amplitude * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.5, n)


@pytest.mark.parametrize("period", [7, 12])
def test_decomposition_adds_up_and_leaves_the_edges_undefined(period: int) -> None:
    y = seasonal(n=10 * period, period=period)
    trend, season, remainder = classical_decomposition(y, period)
    valid = np.isfinite(trend)
    assert np.allclose(trend[valid] + season[valid] + remainder[valid], y[valid])
    assert (~valid).sum() == 2 * (period // 2)
    assert abs(season[:period].sum()) < 1e-9


def test_strength_is_high_with_a_clear_cycle_and_low_on_noise() -> None:
    assert seasonal_strength(seasonal(), 7) > 0.9
    noise = np.random.default_rng(1).normal(size=700)
    assert seasonal_strength(noise, 7) < 0.1


def test_strength_is_undefined_without_two_cycles() -> None:
    assert np.isnan(seasonal_strength(seasonal(n=10), 7))


def test_rule_detects_a_cycle_and_suggests_a_seasonal_difference() -> None:
    result = detect_seasonality(seasonal(), 7)
    assert result.seasonal
    assert result.p_value < 0.05
    assert result.seasonal_differences == 1
    assert result.autocorrelation > 0.5


def test_false_detection_on_white_noise_stays_near_alpha() -> None:
    detections = [
        detect_seasonality(np.random.default_rng(seed).normal(size=365), 7).seasonal
        for seed in range(100)
    ]
    assert np.mean(detections) <= 0.12


def test_rule_needs_enough_cycles_and_a_period() -> None:
    short = detect_seasonality(seasonal(n=MIN_CYCLES * 7 - 1), 7)
    assert not short.seasonal
    assert "ciclos" in short.reason
    annual = detect_seasonality(seasonal(n=50), 1)
    assert not annual.seasonal
    assert annual.reason == "período < 3"


def test_rule_on_a_constant_series() -> None:
    result = detect_seasonality(np.full(70, 3.0), 7)
    assert not result.seasonal
    assert result.reason == "série sem variação"
