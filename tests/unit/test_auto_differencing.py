"""Automatic regular and seasonal differences, chosen on the training vector and frozen."""

from __future__ import annotations

import numpy as np
import pytest

from autohybts.adapters.sklearn_regression import LagRegressionForecaster
from autohybts.data.preprocessing import Differencer
from autohybts.domain.fit_count import count_fits
from autohybts.exceptions import ConfigurationError
from autohybts.residuals.generation import local_context
from autohybts.search.spaces import SearchSpace


def walk(n: int = 300, seed: int = 0) -> np.ndarray:
    return 50 + np.cumsum(np.random.default_rng(seed).normal(size=n))


def weekly(n: int = 280, seed: int = 0) -> np.ndarray:
    t = np.arange(n)
    noise = np.random.default_rng(seed).normal(0, 0.3, n)
    return 20 + 0.2 * t + 5 * np.sin(2 * np.pi * t / 7) + noise


@pytest.mark.parametrize(("d", "seasonal"), [(1, 0), (2, 0), (0, 1), (1, 1)])
def test_inverting_the_true_differences_recovers_the_future(d: int, seasonal: int) -> None:
    y = weekly(120)
    differencer = Differencer(d, seasonal, 7)
    fitted, _ = differencer.fit_transform(y[:100])
    _, future_level = differencer.fit_transform(y)
    recovered = fitted.inverse_forecast(future_level[-20:])
    assert np.allclose(recovered, y[100:])
    assert differencer.lost == d + 7 * seasonal


def test_explicit_settings_are_validated() -> None:
    for params in (
        {"differencing": "yes"},
        {"differencing": -1},
        {"seasonal_differencing": 2, "season_length": 7},
        {"seasonal_differencing": 1},
        {"differencing_test": "pp"},
        {"season_length": 0},
    ):
        with pytest.raises(ConfigurationError):
            LagRegressionForecaster(**params)  # type: ignore[arg-type]


def test_explicit_seasonal_difference_without_statsmodels_tests() -> None:
    model = LagRegressionForecaster(lags=7, seasonal_differencing=1, season_length=7)
    model.fit(weekly())
    assert model.differencer_ == Differencer(0, 1, 7)
    assert model.min_history == 7 + 7


def test_search_passes_automatic_differencing_only_when_asked() -> None:
    plain = SearchSpace(seasonal_period=7)
    assert plain.default_params("knn") == {"lags": 7}
    assert "differencing" not in plain.describe()
    auto = SearchSpace(seasonal_period=7, differencing="auto", differencing_test="adf")
    assert auto.default_params("knn") == {
        "lags": 7,
        "differencing": "auto",
        "season_length": 7,
        "differencing_test": "adf",
    }
    assert auto.describe()["differencing"] == "auto"
    assert auto.default_params("arima") == {}
    with pytest.raises(ConfigurationError):
        SearchSpace(differencing="always")


@pytest.mark.filterwarnings("ignore:adfuller currently returns:FutureWarning")
class TestWithStatsmodels:
    @pytest.fixture(autouse=True)
    def _needs_statsmodels(self) -> None:
        pytest.importorskip("statsmodels")

    @pytest.mark.parametrize("test", ["kpss", "adf"])
    def test_order_follows_the_integration_of_the_series(self, test: str) -> None:
        from autohybts.data.stationarity import choose_d

        noise = np.random.default_rng(1).normal(size=300)
        assert choose_d(noise, test=test) == 0
        assert choose_d(walk(), test=test) == 1
        assert choose_d(np.cumsum(walk()), test=test) == 2

    def test_adf_rule_matches_the_replication_project(self) -> None:
        from statsmodels.tsa.stattools import adfuller

        from autohybts.data.stationarity import choose_d

        for seed in range(5):
            y = np.cumsum(walk(150, seed)) if seed % 2 else walk(150, seed)
            expected = 2
            for order in range(3):
                values = np.diff(y, n=order) if order else y
                stat, p_value, _, _, critical, _ = adfuller(values, autolag="AIC", regression="c")
                if p_value <= 0.05 and stat < critical["5%"]:
                    expected = order
                    break
            assert choose_d(y, test="adf") == expected

    def test_strong_weekly_cycle_takes_a_seasonal_difference_first(self) -> None:
        from autohybts.data.stationarity import choose_differences

        differencer = choose_differences(weekly(), season_length=7)
        assert differencer.seasonal == 1
        assert differencer.period == 7

    def test_lag_model_chooses_on_the_training_vector_and_keeps_it_frozen(self) -> None:
        y = walk(320)
        model = LagRegressionForecaster(lags=7, differencing="auto").fit(y[:300])
        assert model.differencer_.d == 1
        assert model.min_history == 7 + 1
        chosen = model.differencer_
        with count_fits() as fits:
            early = model.predict(5, local_context(y, 300))
            late = model.predict(5, local_context(y, 320))
        assert fits.total == 0
        assert model.differencer_ is chosen
        assert np.all(np.isfinite(early))
        assert not np.allclose(early, late)

    def test_experiment_option_reaches_every_lag_model_of_the_search(self) -> None:
        from autohybts import AutoHybridExperiment
        from autohybts.datasets import get_data

        exp = AutoHybridExperiment(fh=14, fold=2, differencing="auto").fit(
            get_data("synthetic_daily")
        )
        space = exp._space(with_residuals=False)
        for name in ("linear", "knn", "random_forest"):
            assert space.default_params(name)["differencing"] == "auto"
            assert space.default_params(name)["season_length"] == 7
        result = exp.create_model("linear", differencing="auto", season_length=7)
        assert np.isfinite(result.metrics["MASE"])
