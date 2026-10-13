"""T05 for base models: changing the future of an origin never changes its forecast."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalSplitter
from autohybts.registry import default_registry

REGISTRY = default_registry()
FAST_PARAMS: dict[str, dict[str, Any]] = {
    "arima": {"order": (1, 1, 1)},
    "random_forest": {"estimator_params": {"n_estimators": 10}},
    "extra_trees_regressor": {"estimator_params": {"n_estimators": 10}},
    "mlp": {"estimator_params": {"hidden_layer_sizes": (8,), "max_iter": 300}},
    "hist_gradient_boosting": {"estimator_params": {"max_iter": 20}},
    "adaboost_r2": {"estimator_params": {"n_estimators": 10}},
    "lightgbm_regressor": {"estimator_params": {"n_estimators": 20}},
    "xgboost_regressor": {"estimator_params": {"n_estimators": 20}},
    "catboost_regressor": {"estimator_params": {"iterations": 20}},
}
# Models from optional extras are skipped when their library is missing.
EXTRA_MODULES: dict[str, str] = {
    "arima": "statsmodels",
    "exponential_smoothing": "statsmodels",
    "lightgbm_regressor": "lightgbm",
    "sf_auto_arima": "statsforecast",
    "sf_auto_ets": "statsforecast",
    "sf_auto_theta": "statsforecast",
    "sf_auto_ces": "statsforecast",
    "xgboost_regressor": "xgboost",
    "catboost_regressor": "catboost",
}
HORIZON = 7
# The tiny MLP used for speed stops before converging; irrelevant to the contract.
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def forecasts(name: str, values: np.ndarray) -> dict[str, tuple[int, np.ndarray]]:
    """Origin index and forecast per fold, each fitted on its own training slice only."""

    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=len(values)), values)
    splitter = TemporalSplitter(horizon=HORIZON, n_search_folds=3, n_select_folds=3, step=15)
    plan = splitter.plan(frame)
    out = {}
    for origin in plan:
        model = REGISTRY.make_base(name, FAST_PARAMS.get(name))
        model.fit(plan.train_values(origin))
        out[origin.fold_id] = (origin.origin_index, model.predict(HORIZON, plan.context(origin)))
    return out


@pytest.mark.parametrize("name", REGISTRY.names("base"))
def test_future_perturbation_does_not_change_forecasts(name: str) -> None:
    if name in EXTRA_MODULES:
        pytest.importorskip(EXTRA_MODULES[name])
    rng = np.random.default_rng(3)
    base = 10 + np.cumsum(rng.normal(0, 1, 140))
    reference = forecasts(name, base)

    # Origins sit at 65, 80, 95 (search) and 102, 117, 132 (select).
    for cut in (70, 100, 125):
        perturbed = base.copy()
        perturbed[cut:] += 50.0 + 10.0 * np.abs(perturbed[cut:])
        changed = forecasts(name, perturbed)
        for fold_id, (index, forecast) in changed.items():
            if index < cut:
                assert np.array_equal(forecast, reference[fold_id][1]), (
                    f"{name}: forecast at {fold_id} changed with data after {cut}"
                )
        # The perturbation is real: some origin after the cut must see it.
        assert any(
            not np.array_equal(forecast, reference[fold_id][1])
            for fold_id, (index, forecast) in changed.items()
            if index >= cut
        )
