"""Extra baselines (plano de ensembles F1a) and the RMAE metric."""

from __future__ import annotations

import numpy as np
import pytest

from autohybts.adapters.naive import (
    HistoricAverageForecaster,
    RandomWalkWithDriftForecaster,
    SeasonalWindowAverageForecaster,
    WindowAverageForecaster,
)
from autohybts.evaluation.metrics import METRICS
from autohybts.exceptions import ConfigurationError
from autohybts.registry import default_registry
from autohybts.residuals.generation import local_context
from autohybts.search.spaces import DEFAULT_BASES

Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


def test_historic_average_is_a_frozen_parameter() -> None:
    model = HistoricAverageForecaster().fit(Y)
    np.testing.assert_allclose(model.predict(3, local_context(Y, 8)), [4.5] * 3)
    longer = np.append(Y, [100.0, 100.0])
    np.testing.assert_allclose(model.predict(1, local_context(longer, 10)), [4.5])


def test_window_average_reads_the_context() -> None:
    model = WindowAverageForecaster(window=3).fit(Y)
    np.testing.assert_allclose(model.predict(2, local_context(Y, 8)), [7.0, 7.0])
    np.testing.assert_allclose(model.predict(1, local_context(Y, 5)), [4.0])


def test_seasonal_window_average_averages_each_position() -> None:
    model = SeasonalWindowAverageForecaster(season_length=2, window=2).fit(Y)
    # last 4 values 5,6,7,8 -> positions (5+7)/2 = 6 and (6+8)/2 = 7, repeated
    np.testing.assert_allclose(model.predict(5, local_context(Y, 8)), [6, 7, 6, 7, 6])
    assert model.min_history == 4


def test_drift_extends_the_average_change() -> None:
    model = RandomWalkWithDriftForecaster().fit(Y)
    assert model.drift_ == pytest.approx(1.0)
    np.testing.assert_allclose(model.predict(3, local_context(Y, 6)), [7.0, 8.0, 9.0])


@pytest.mark.parametrize(
    "build",
    [
        lambda: WindowAverageForecaster(window=0),
        lambda: SeasonalWindowAverageForecaster(window=True),
        lambda: SeasonalWindowAverageForecaster(season_length=0),
    ],
)
def test_invalid_baseline_parameters(build) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ConfigurationError):
        build()


def test_baselines_are_registered_but_not_in_the_default_search() -> None:
    names = set(default_registry().names("base"))
    extra = {"historic_average", "window_average", "seasonal_window_average", "random_walk_drift"}
    assert extra <= names
    assert not extra & set(DEFAULT_BASES)


def test_rmae_matches_the_replication_definition() -> None:
    # projeto_st_replicacao validation: actual [2, 4], forecast [1, 5] -> 1/2 at h=1, 1/3 at h=2
    rmae = METRICS["RMAE"]
    assert rmae([2.0], [1.0], Y) == pytest.approx(0.5)
    assert rmae([2.0, 4.0], [1.0, 5.0], Y) == pytest.approx(1 / 3)
    assert np.isnan(rmae([0.0, 0.0], [1.0, 1.0], Y))


@pytest.mark.parametrize(
    ("trend", "damped", "season_length"),
    [(True, False, 7), (True, True, 7), (False, False, 7), (True, False, 1), (False, False, 1)],
)
def test_exponential_smoothing_reproduces_statsmodels_with_frozen_parameters(
    trend: bool, damped: bool, season_length: int
) -> None:
    statsmodels = pytest.importorskip("statsmodels.tsa.holtwinters")
    rng = np.random.default_rng(0)
    t = np.arange(200)
    y = 20 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.5, 200)
    from autohybts.adapters.statsmodels_ets import ExponentialSmoothingForecaster

    model = ExponentialSmoothingForecaster(trend, damped, season_length).fit(y)
    reference = statsmodels.ExponentialSmoothing(
        y,
        trend="add" if trend else None,
        damped_trend=damped,
        seasonal="add" if season_length > 1 else None,
        seasonal_periods=season_length if season_length > 1 else None,
        initialization_method="estimated",
    ).fit()
    np.testing.assert_allclose(
        model.predict(10, local_context(y, 200)), reference.forecast(10), atol=1e-8
    )
    # Frozen: a longer context moves the states, never the parameters.
    longer = np.append(y, y[-20:])
    alpha = model.alpha_
    assert model.predict(3, local_context(longer, 220)).shape == (3,)
    assert model.alpha_ == alpha


def test_exponential_smoothing_validates_its_settings() -> None:
    from autohybts.adapters.statsmodels_ets import ExponentialSmoothingForecaster

    with pytest.raises(ConfigurationError, match="damped"):
        ExponentialSmoothingForecaster(trend=False, damped=True)
    with pytest.raises(ConfigurationError):
        ExponentialSmoothingForecaster(season_length=0)
