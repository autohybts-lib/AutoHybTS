"""T05 with covariates (ADR-0011): the future of an origin never changes its forecast.

``past`` features are lagged by the horizon, so values of ``x`` after the origin cannot
reach it. A ``future`` feature is known in advance by declaration: changing it inside the
horizon may change the forecast, and that is the point of declaring it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from autohybts.data.exog import ExogFrame
from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalSplitter
from autohybts.domain.specs import CandidateSpec, ExogFeature, ModelSpec
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.registry import default_registry

REGISTRY = default_registry()
HORIZON = 5
N = 140
FAST_PARAMS: dict[str, dict[str, Any]] = {
    "arima": {"order": (1, 0, 0)},
    "sarima": {"season_length": 7, "order": (1, 0, 0), "seasonal_order": (1, 0, 0)},
    "random_forest": {"estimator_params": {"n_estimators": 10}},
    "extra_trees_regressor": {"estimator_params": {"n_estimators": 10}},
    "mlp": {"estimator_params": {"hidden_layer_sizes": (8,), "max_iter": 300}},
    "hist_gradient_boosting": {"estimator_params": {"max_iter": 20}},
    "adaboost_r2": {"estimator_params": {"n_estimators": 10}},
    "lightgbm_regressor": {"estimator_params": {"n_estimators": 20}},
    "xgboost_regressor": {"estimator_params": {"n_estimators": 20}},
    "catboost_regressor": {"estimator_params": {"iterations": 20}},
}
EXTRA_MODULES = {
    "arima": "statsmodels",
    "sarima": "statsmodels",
    "lightgbm_regressor": "lightgbm",
    "xgboost_regressor": "xgboost",
    "catboost_regressor": "catboost",
}
EXOG_BASES = [n for n in REGISTRY.names("base") if REGISTRY.entry(n).capabilities.accepts_exog]
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def inputs(seed: int = 3) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=N, freq="D")
    x = rng.normal(0, 1, N)
    holiday = (np.arange(N) % 7 == 0).astype(float)
    y = 10 + np.cumsum(rng.normal(0, 0.5, N)) + 2 * np.roll(x, HORIZON) + 3 * holiday
    return y, pd.DataFrame({"x": x, "holiday": holiday}, index=index)


def forecasts(name: str, y: np.ndarray, X: pd.DataFrame, variant: str = "A") -> dict[str, Any]:
    series = SeriesFrame.from_arrays(X.index, y)
    plan = TemporalSplitter(horizon=HORIZON, n_search_folds=2, n_select_folds=2, step=15).plan(
        series
    )
    evaluator = TemporalEvaluator(REGISTRY, exog=ExogFrame.from_frame(X, future=["holiday"]))
    lag = HORIZON if variant == "A" else 1
    candidate = CandidateSpec(
        "exogenous",
        "standalone",
        ModelSpec.of(name, FAST_PARAMS.get(name)),
        variant=variant,
        exog=(ExogFeature("x", lag), ExogFeature("holiday", 0)),
    )
    result = evaluator.evaluate(candidate, plan)
    return {t.fold_id: (t.origin_index, t.status, t.y_pred) for t in result.trials}


@pytest.mark.parametrize("name", EXOG_BASES)
def test_future_of_series_and_past_covariates_does_not_change_forecasts(name: str) -> None:
    if name in EXTRA_MODULES:
        pytest.importorskip(EXTRA_MODULES[name])
    y, X = inputs()
    reference = forecasts(name, y, X)
    assert {status for _, status, _ in reference.values()} == {"SUCCEEDED"}
    for cut in (80, 110):
        later_y = y.copy()
        later_y[cut:] += 50.0
        later_x = X.copy()
        later_x.iloc[cut:, 0] += 50.0  # column "x" (past)
        for changed_y, changed_x in ((later_y, X), (y, later_x)):
            changed = forecasts(name, changed_y, changed_x)
            for fold, (origin, _, forecast) in reference.items():
                if origin < cut:
                    assert np.allclose(changed[fold][2], forecast), (name, cut, fold)


@pytest.mark.parametrize("name", ["linear", "arima"])
def test_variant_b_sees_past_covariates_one_step_late(name: str) -> None:
    pytest.importorskip("statsmodels")
    y, X = inputs()
    reference = forecasts(name, y, X, variant="B")
    for fold, (origin, status, forecast) in reference.items():
        assert status == "SUCCEEDED"
        later_x = X.copy()
        later_x.iloc[origin + HORIZON :, 0] += 50.0  # after the last forecast date of the fold
        assert np.allclose(forecasts(name, y, later_x, variant="B")[fold][2], forecast)


def test_declared_future_covariates_do_reach_the_horizon() -> None:
    y, X = inputs()
    reference = forecasts("linear", y, X)
    fold, (origin, _, forecast) = next(iter(reference.items()))
    later = X.copy()
    later.iloc[origin + 1 : origin + 1 + HORIZON, 1] += 10.0  # column "holiday" (future)
    assert not np.allclose(forecasts("linear", y, later)[fold][2], forecast)
