"""Seasonal ARIMA: automatic orders on the training vector, frozen forecasts, cache."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from autohybts.adapters.statsmodels_sarima import (
    _SARIMA_CACHE,
    MAX_SEASON_LENGTH,
    SarimaForecaster,
)
from autohybts.domain.fit_count import count_fits
from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError
from autohybts.residuals.generation import local_context
from autohybts.search.spaces import SearchSpace


def monthly(n: int = 120, seed: int = 0) -> np.ndarray:
    """Trend, a strong yearly cycle and AR(1) noise."""

    rng = np.random.default_rng(seed)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.5 * noise[i - 1] + rng.normal(0, 1.0)
    t = np.arange(n)
    return 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + noise


def test_auto_takes_a_seasonal_difference_on_a_strong_cycle() -> None:
    model = SarimaForecaster(season_length=12).fit(monthly())
    (_, d, _), (_, big_d, _, m) = model.orders_
    assert m == 12
    assert big_d == 1
    assert d in (0, 1)
    forecast = model.predict(12, local_context(model.train_, model.train_.size))
    assert forecast.shape == (12,)
    assert np.all(np.isfinite(forecast))


def test_season_length_one_is_a_non_seasonal_arima() -> None:
    model = SarimaForecaster().fit(monthly())
    assert model.seasonal_order_ == (0, 0, 0, 0)


def test_explicit_orders_are_used_as_given() -> None:
    model = SarimaForecaster(season_length=12, order=(1, 0, 0), seasonal_order=(0, 1, 1)).fit(
        monthly()
    )
    assert model.orders_ == ((1, 0, 0), (0, 1, 1, 12))


def test_a_longer_context_moves_the_state_without_refitting() -> None:
    y = monthly(132)
    model = SarimaForecaster(season_length=12, order=(1, 0, 0), seasonal_order=(0, 1, 1))
    model.fit(y[:120])
    params = np.array(model.result_.params, copy=True)
    with count_fits() as fits:
        early = model.predict(6, local_context(y, 120))
        late = model.predict(6, local_context(y, y.size))
    assert fits.total == 0
    assert np.array_equal(params, model.result_.params)
    assert not np.allclose(early, late)


def test_cache_reuses_orders_and_parameters_without_sharing_results() -> None:
    y = monthly()
    _SARIMA_CACHE.clear()
    fresh = SarimaForecaster(season_length=12).fit(y)
    hits = _SARIMA_CACHE.hits
    cached = SarimaForecaster(season_length=12).fit(y)
    assert _SARIMA_CACHE.hits == hits + 1
    assert cached.orders_ == fresh.orders_
    assert cached.result_ is not fresh.result_
    assert np.allclose(cached.result_.params, fresh.result_.params)


@pytest.mark.parametrize(
    "params",
    [
        {"season_length": 0},
        {"season_length": True},
        {"seasonal_order": (1, 1, 1)},
        {"order": (1, 0, 0), "seasonal_order": (0, 2, 0)},
        {"order": (1, 0, 0), "seasonal_order": (0, 1)},
        {"max_P": -1},
    ],
)
def test_invalid_settings_are_refused(params: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        SarimaForecaster(**params)  # type: ignore[arg-type]


def test_long_cycles_are_refused_with_the_alternative() -> None:
    model = SarimaForecaster(season_length=MAX_SEASON_LENGTH + 1)
    with pytest.raises(UnsupportedCapabilityError, match="harmonic_regression"):
        model.fit(monthly(200))


def test_search_skips_sarima_without_seasonality_and_passes_the_period() -> None:
    from autohybts.registry import default_registry

    registry = default_registry()
    plain = SearchSpace(base_models=("naive", "sarima"), seasonal_period=1)
    skipped = {p.candidate.name: p.skip_reason for p in plain.base_proposals(registry)}
    assert skipped["sarima"] == "sem sazonalidade declarada: equivale ao arima"
    seasonal = SearchSpace(base_models=("sarima",), seasonal_period=12)
    assert seasonal.default_params("sarima") == {"season_length": 12}
