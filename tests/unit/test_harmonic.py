"""Dynamic harmonic regression, the multiple-seasonality representative (plano L06)."""

from __future__ import annotations

import numpy as np
import pytest

from autohybts.adapters.harmonic import HarmonicRegressionForecaster
from autohybts.exceptions import ConfigurationError
from autohybts.registry import default_registry
from autohybts.residuals.generation import local_context
from autohybts.search.spaces import DEFAULT_BASES


def two_seasons(n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    return (
        20
        + 0.01 * t
        + 3 * np.sin(2 * np.pi * t / 7)
        + 5 * np.cos(2 * np.pi * t / 30.5)
        + np.sin(4 * np.pi * t / 30.5)
    )


def exact() -> HarmonicRegressionForecaster:
    return HarmonicRegressionForecaster(periods=(7, 30.5), harmonics=(3, 2), ar_lags=0, ridge=0)


def test_recovers_a_noise_free_signal_with_two_periods() -> None:
    full = two_seasons(260)
    model = exact().fit(full[:200])
    forecast = model.predict(30, local_context(full, 200))
    np.testing.assert_allclose(forecast, full[200:230], atol=1e-6)


def test_later_contexts_keep_the_phase_without_refitting() -> None:
    full = two_seasons(260)
    model = exact().fit(full[:200])
    coef = model.coef_.copy()
    # 17 steps later (not a multiple of either period): still on the true curve.
    np.testing.assert_allclose(model.predict(5, local_context(full, 217)), full[217:222], atol=1e-6)
    np.testing.assert_array_equal(model.coef_, coef)


def test_one_period_is_not_enough_for_two_seasons() -> None:
    full = two_seasons(260)
    weekly = HarmonicRegressionForecaster(periods=(7,), harmonics=(3,), ar_lags=0).fit(full[:200])
    error = np.abs(weekly.predict(30, local_context(full, 200)) - full[200:230]).mean()
    assert error > 1.0


def test_autoregressive_lags_feed_predicted_values() -> None:
    rng = np.random.default_rng(0)
    y = two_seasons(200) + rng.normal(0, 0.3, 200)
    model = HarmonicRegressionForecaster(periods=(7,), harmonics=(2,), ar_lags=2).fit(y)
    forecast = model.predict(3, local_context(y, 200))
    deterministic = model._deterministic(np.arange(200, 203, dtype=np.float64))
    base = deterministic @ model.coef_[:-2]
    phi = model.coef_[-2:]
    path = [y[198], y[199]]
    for step in range(3):
        expected = base[step] + phi[0] * path[-1] + phi[1] * path[-2]
        assert forecast[step] == pytest.approx(expected)
        path.append(expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"periods": (7, 365), "harmonics": (3,)},
        {"periods": (), "harmonics": ()},
        {"periods": (7,), "harmonics": (4,)},  # 2 * 4 > 7
        {"periods": (2,), "harmonics": (1,)},
        {"periods": (7,), "harmonics": (0,)},
        {"ar_lags": -1},
        {"ridge": -1.0},
    ],
)
def test_invalid_settings(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        HarmonicRegressionForecaster(**kwargs)  # type: ignore[arg-type]


def test_registered_out_of_the_default_search() -> None:
    registry = default_registry()
    assert registry.entry("harmonic_regression").family == "statistical"
    assert "harmonic_regression" not in DEFAULT_BASES
    model = registry.make_base("harmonic_regression", {"periods": (7, 30.5), "harmonics": (3, 2)})
    assert model.min_train_size == 1 + 2 * model.n_features
