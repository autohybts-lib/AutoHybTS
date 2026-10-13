"""Opt-in Box-Cox and outlier treatment, fitted on the training values only (ADR-0009)."""

from __future__ import annotations

import numpy as np
import pytest

from autohybts.adapters.sklearn_regression import LagRegressionForecaster
from autohybts.data.target_prep import clean_outliers, fit_target_transform
from autohybts.domain.fit_count import count_fits
from autohybts.exceptions import ConfigurationError
from autohybts.residuals.generation import local_context
from autohybts.search.spaces import SearchSpace


def growing(n: int = 200, seed: int = 0) -> np.ndarray:
    t = np.arange(n)
    noise = np.random.default_rng(seed).normal(0, 0.05, n)
    return np.exp(0.01 * t + 0.3 * np.sin(2 * np.pi * t / 7) + noise) * 10


@pytest.mark.parametrize("kind", ["boxcox", "log"])
def test_transform_round_trip(kind: str) -> None:
    y = growing()
    fitted = fit_target_transform(y, kind)
    assert np.allclose(fitted.invert(fitted.apply(y)), y)


def test_non_positive_series_are_shifted() -> None:
    y = np.linspace(-5, 5, 50)
    fitted = fit_target_transform(y, "boxcox")
    assert fitted.shift == pytest.approx(6.0)
    assert np.allclose(fitted.invert(fitted.apply(y)), y)


@pytest.mark.parametrize("method", ["winsorize", "interpolate"])
def test_outliers_are_pulled_in_and_the_rest_kept(method: str) -> None:
    y = 10 + np.random.default_rng(1).normal(0, 0.5, 100)
    y[40] = 60.0
    cleaned = clean_outliers(y, method)
    assert cleaned[40] < 20
    keep = np.arange(100) != 40
    assert np.array_equal(cleaned[keep], y[keep])


def test_lag_model_with_boxcox_forecasts_on_the_original_scale() -> None:
    y = growing(220)
    model = LagRegressionForecaster(lags=7, transform="boxcox").fit(y[:200])
    with count_fits() as fits:
        forecast = model.predict(14, local_context(y, 200))
    assert fits.total == 0
    assert forecast.shape == (14,)
    assert np.all(forecast > 0)
    assert abs(np.median(forecast) - np.median(y[200:214])) < 0.5 * np.median(y[200:214])


def test_outliers_change_the_training_values_but_not_the_context() -> None:
    y = 10 + np.random.default_rng(2).normal(0, 0.5, 200)
    y[150] = 80.0
    plain = LagRegressionForecaster(lags=7).fit(y)
    treated = LagRegressionForecaster(lags=7, outliers="interpolate").fit(y)
    assert not np.allclose(plain.model_.coef_, treated.model_.coef_)
    context = local_context(y, 200)
    assert treated.predict(1, context).shape == (1,)


def test_settings_are_validated_and_reach_the_search() -> None:
    with pytest.raises(ConfigurationError):
        LagRegressionForecaster(transform="sqrt")
    with pytest.raises(ConfigurationError):
        LagRegressionForecaster(outliers="drop")
    space = SearchSpace(seasonal_period=7, transform="boxcox", outliers="winsorize")
    assert space.default_params("knn") == {
        "lags": 7,
        "transform": "boxcox",
        "outliers": "winsorize",
    }
    assert space.describe()["transform"] == "boxcox"
    assert "transform" not in SearchSpace(seasonal_period=7).describe()
