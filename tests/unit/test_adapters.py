from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from autohybts.adapters.base import FloatArray, ForecasterBase
from autohybts.adapters.naive import NaiveForecaster, SeasonalNaiveForecaster
from autohybts.adapters.sklearn_regression import LagRegressionForecaster
from autohybts.adapters.statsmodels_arima import ArimaForecaster
from autohybts.domain.context import TemporalContext
from autohybts.exceptions import (
    CandidateFitError,
    ConfigurationError,
    DataValidationError,
    InsufficientHistoryError,
    UnsupportedCapabilityError,
)


def context(values: np.ndarray) -> TemporalContext:
    return TemporalContext(
        series_id="s",
        frequency="D",
        origin_index=len(values) - 1,
        origin_time=pd.Timestamp("2024-01-01") + pd.Timedelta(days=len(values) - 1),
        observed=values,
    )


def test_naive_is_horizontal_at_the_last_value_even_with_trend() -> None:
    y = np.arange(30.0) * 2
    assert NaiveForecaster().fit(y).predict(5, context(y)).tolist() == [58.0] * 5


def test_seasonal_naive_repeats_the_last_season() -> None:
    y = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    forecast = SeasonalNaiveForecaster(season_length=3).fit(y).predict(7, context(y))
    assert forecast.tolist() == [10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0]


@pytest.mark.parametrize("season", [0, -2, 1.5, True])
def test_seasonal_naive_rejects_invalid_season(season: object) -> None:
    with pytest.raises(ConfigurationError, match="season_length"):
        SeasonalNaiveForecaster(season_length=season)


def test_naive_conditions_on_the_context_not_on_training() -> None:
    model = NaiveForecaster().fit([1.0, 2.0, 3.0])
    assert model.predict(2, context(np.array([5.0, 9.0]))).tolist() == [9.0, 9.0]


def test_linear_lags_extrapolate_an_exact_line() -> None:
    y = 3.0 + 0.5 * np.arange(40)
    model = LagRegressionForecaster("linear", lags=2).fit(y)
    expected = 3.0 + 0.5 * np.arange(40, 46)
    assert np.allclose(model.predict(6, context(y)), expected)


def test_differencing_lets_a_tree_follow_a_trend() -> None:
    y = np.arange(100.0)
    flat = LagRegressionForecaster("tree", lags=3).fit(y).predict(10, context(y))
    trending = (
        LagRegressionForecaster("tree", lags=3, differencing=1).fit(y).predict(10, context(y))
    )
    assert flat.max() <= y.max()  # trees cannot extrapolate levels
    assert np.allclose(trending, np.arange(100.0, 110.0))


def test_scaling_does_not_change_a_linear_fit() -> None:
    y = np.sin(np.arange(80) / 3) * 100 + 500
    a = LagRegressionForecaster("linear", lags=5, scale=True).fit(y).predict(4, context(y))
    b = LagRegressionForecaster("linear", lags=5, scale=False).fit(y).predict(4, context(y))
    assert np.allclose(a, b)


def test_constant_series_does_not_divide_by_zero() -> None:
    y = np.full(30, 7.0)
    assert np.allclose(LagRegressionForecaster("svm", lags=3).fit(y).predict(3, context(y)), 7.0)


def test_estimator_params_and_seed_reach_the_estimator() -> None:
    model = LagRegressionForecaster(
        "random_forest", lags=3, estimator_params={"n_estimators": 5}, random_state=11
    ).fit(np.arange(30.0))
    assert model.model_.n_estimators == 5
    assert model.model_.random_state == 11


def test_unknown_estimator_and_bad_params() -> None:
    with pytest.raises(ConfigurationError, match="Estimador desconhecido"):
        LagRegressionForecaster("xgboost")
    with pytest.raises(ConfigurationError, match="differencing"):
        LagRegressionForecaster("linear", differencing=-1)
    with pytest.raises(ConfigurationError, match="Parâmetro inválido"):
        LagRegressionForecaster("linear", estimator_params={"banana": 1}).fit(np.arange(20.0))


def test_estimator_failure_becomes_a_candidate_error() -> None:
    # 5 neighbours requested, only 3 training rows available.
    y = np.arange(5.0)
    model = LagRegressionForecaster("knn", lags=2, estimator_params={"n_neighbors": 5}).fit(y)
    with pytest.raises(CandidateFitError, match="LagRegressionForecaster falhou na previsão"):
        model.predict(2, context(y))


def test_training_with_missing_values_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="ausentes"):
        NaiveForecaster().fit([1.0, np.nan, 3.0])


def test_repr_shows_parameters() -> None:
    assert repr(SeasonalNaiveForecaster(season_length=12)) == (
        "SeasonalNaiveForecaster(season_length=12)"
    )


class TestArima:
    @pytest.fixture(autouse=True)
    def _needs_statsmodels(self) -> None:
        pytest.importorskip("statsmodels")

    def test_fixed_order(self) -> None:
        y = np.cumsum(np.random.default_rng(0).normal(0.2, 1, 120))
        model = ArimaForecaster(order=(1, 1, 0)).fit(y)
        assert model.order_ == (1, 1, 0)
        assert model.fitted_values().shape == (120,)
        assert model.predict(5, context(y)).shape == (5,)

    @pytest.mark.slow
    def test_auto_order_differences_a_random_walk(self) -> None:
        y = np.cumsum(np.random.default_rng(1).normal(0, 1, 200))
        model = ArimaForecaster().fit(y)
        assert model.order_[1] >= 1

    def test_new_context_updates_state_without_refit(self) -> None:
        rng = np.random.default_rng(2)
        y = np.cumsum(rng.normal(0, 1, 150))
        model = ArimaForecaster(order=(1, 1, 0)).fit(y[:100])
        params = model.result_.params.copy()
        later = model.predict(3, context(y))
        assert not np.allclose(later, model.predict(3, context(y[:100])))
        assert np.array_equal(model.result_.params, params)

    @pytest.mark.parametrize("order", [(1, 3, 0), (1, 1), "manual", (1, -1, 0)])
    def test_invalid_order(self, order: object) -> None:
        with pytest.raises(ConfigurationError):
            ArimaForecaster(order=order)


def test_arima_without_statsmodels_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules makes the import fail as if the package were absent.
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.arima.model", None)
    with pytest.raises(UnsupportedCapabilityError, match=r"autohybts\[stats\]"):
        ArimaForecaster(order=(1, 0, 0)).fit(np.arange(30.0))


class BrokenForecaster(ForecasterBase):
    def __init__(self, output: str = "nan") -> None:
        super().__init__()
        self.output = output

    def _fit(self, y: FloatArray) -> None:
        return None

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        if self.output == "nan":
            return np.full(horizon, np.nan)
        return np.zeros(horizon + 1)


@pytest.mark.parametrize("output", ["nan", "wrong-length"])
def test_invalid_forecasts_are_candidate_errors(output: str) -> None:
    model = BrokenForecaster(output).fit([1.0, 2.0])
    with pytest.raises(CandidateFitError, match="previsão inválida"):
        model.predict(3, context(np.arange(3.0)))


def test_predict_needs_a_temporal_context() -> None:
    model = NaiveForecaster().fit([1.0, 2.0])
    with pytest.raises(ConfigurationError, match="TemporalContext"):
        model.predict(2, np.arange(3.0))


def test_context_shorter_than_the_lags_is_reported() -> None:
    model = LagRegressionForecaster("linear", lags=5).fit(np.arange(30.0))
    with pytest.raises(InsufficientHistoryError, match="contexto"):
        model.predict(2, context(np.arange(4.0)))


@pytest.mark.parametrize(("train", "message"), [(["a", "b"], "numérico"), ([[1.0, 2.0]], "1-D")])
def test_invalid_training_vectors(train: object, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        NaiveForecaster().fit(train)


@pytest.mark.parametrize("kwargs", [{"max_p": -1}, {"max_d": True}, {"max_q": 1.5}])
def test_arima_rejects_invalid_search_bounds(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        ArimaForecaster(**kwargs)


def test_arima_requires_fit_before_fitted_values() -> None:
    model = ArimaForecaster(order=(1, 0, 0))
    assert model.min_history == 1
    with pytest.raises(ConfigurationError, match="fit antes"):
        model.fitted_values()


def test_arima_reports_when_no_order_can_be_fitted(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("statsmodels")

    def failing(*args: object, **kwargs: object) -> object:
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(ArimaForecaster, "_fit_one", failing)
    with pytest.raises(CandidateFitError, match="Nenhuma ordem ARIMA"):
        ArimaForecaster(order=(1, 0, 0)).fit(np.arange(30.0))


def test_arima_auto_order_on_a_short_series_skips_kpss() -> None:
    pytest.importorskip("statsmodels")
    model = ArimaForecaster(max_p=0, max_q=0, max_d=1)
    assert model._choose_d(np.arange(6.0)) == 0


def test_lag_regression_records_fit_warnings() -> None:
    y = np.sin(np.arange(60) / 3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = LagRegressionForecaster("mlp", lags=3, estimator_params={"max_iter": 1}).fit(y)
    assert any(w.startswith("ConvergenceWarning") for w in model.fit_warnings_)
