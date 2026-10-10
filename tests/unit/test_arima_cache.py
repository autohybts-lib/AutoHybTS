"""The ARIMA fit cache gives the same model as a fresh fit and shares no fitted state."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from autohybts.adapters.statsmodels_arima import _FIT_CACHE, ArimaForecaster
from autohybts.residuals.generation import local_context


def series(n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(0)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.6 * noise[i - 1] + rng.normal()
    return 50 + 0.1 * np.arange(n) + noise


def test_cache_hit_reproduces_a_fresh_fit_without_sharing_state() -> None:
    y = series()
    _FIT_CACHE.clear()
    fresh = ArimaForecaster().fit(y)
    hits = _FIT_CACHE.hits
    cached = ArimaForecaster().fit(y)
    assert _FIT_CACHE.hits == hits + 1
    assert cached.order_ == fresh.order_
    assert cached.result_ is not fresh.result_
    for end in (180, 200):
        np.testing.assert_allclose(
            cached.predict(7, local_context(y, end)), fresh.predict(7, local_context(y, end))
        )


def test_cache_key_depends_on_data_and_settings() -> None:
    y = series()
    _FIT_CACHE.clear()
    ArimaForecaster().fit(y)
    ArimaForecaster(max_p=1).fit(y)  # different settings: no hit
    changed = y.copy()
    changed[-1] += 1.0
    ArimaForecaster().fit(changed)  # different data: no hit
    assert _FIT_CACHE.hits == 0


def test_cache_is_bounded() -> None:
    _FIT_CACHE.clear()
    size = _FIT_CACHE.size
    try:
        _FIT_CACHE.size = 2
        for shift in range(4):
            ArimaForecaster(order=(1, 0, 0)).fit(series(60) + shift)
        assert len(_FIT_CACHE._items) == 2
    finally:
        _FIT_CACHE.size = size
        _FIT_CACHE.clear()
