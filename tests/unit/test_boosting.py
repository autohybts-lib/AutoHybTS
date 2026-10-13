"""Boosting on lags (plano de ensembles F1c) and the opt-in correctors (F1b)."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from autohybts.adapters.sklearn_regression import build_estimator
from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalSplitter
from autohybts.domain.specs import CandidateSpec, ModelSpec
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.exceptions import UnsupportedCapabilityError
from autohybts.registry import default_registry

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
FAST = {
    "hist_gradient_boosting": {"estimator_params": {"max_iter": 20}},
    "adaboost_r2": {"estimator_params": {"n_estimators": 10}},
    "lightgbm_regressor": {"estimator_params": {"n_estimators": 20}},
    "xgboost_regressor": {"estimator_params": {"n_estimators": 20}},
    "catboost_regressor": {"estimator_params": {"iterations": 20}},
    "random_forest": {"estimator_params": {"n_estimators": 10}},
}

# Estimators of the `boosting` extra -> module to import.
OPTIONAL = {
    "lightgbm_regressor": "lightgbm",
    "xgboost_regressor": "xgboost",
    "catboost_regressor": "catboost",
}


def plan():  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(2)
    noise = np.zeros(200)
    for i in range(1, 200):
        noise[i] = 0.8 * noise[i - 1] + rng.normal()
    values = 10 + 2 * np.sin(2 * np.pi * np.arange(200) / 7) + noise
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=200), values)
    return TemporalSplitter(horizon=5, n_search_folds=1, n_select_folds=1).plan(frame)


@pytest.mark.parametrize(
    "name",
    [
        "hist_gradient_boosting",
        "adaboost_r2",
        "lightgbm_regressor",
        "xgboost_regressor",
        "catboost_regressor",
    ],
)
def test_boosting_bases_forecast_and_are_deterministic_with_a_seed(name: str) -> None:
    if name in OPTIONAL:
        pytest.importorskip(OPTIONAL[name])
    spec = CandidateSpec.standalone(name, {"lags": 7, **FAST[name]}, family="tree")
    first = TemporalEvaluator(default_registry(), seasonal_period=7).evaluate(spec, plan())
    second = TemporalEvaluator(default_registry(), seasonal_period=7).evaluate(spec, plan())
    assert all(t.succeeded for t in first.trials), first.failures
    assert [t.y_pred for t in first.trials] == [t.y_pred for t in second.trials]


@pytest.mark.parametrize(
    "corrector",
    [
        "svm",
        "tree",
        "random_forest",
        "hist_gradient_boosting",
        "adaboost_r2",
        "lightgbm_regressor",
        "xgboost_regressor",
        "catboost_regressor",
    ],
)
def test_opt_in_correctors_work_inside_a_hybrid(corrector: str) -> None:
    if corrector in OPTIONAL:
        pytest.importorskip(OPTIONAL[corrector])
    hybrid = CandidateSpec(
        "residual",
        "residual_additive",
        ModelSpec.of("seasonal_naive", {"season_length": 7}),
        ModelSpec.of(corrector, FAST.get(corrector, {})),
        7,
        1.0,
        "historical_oof_1step",
    )
    result = TemporalEvaluator(default_registry(), seasonal_period=7).evaluate(hybrid, plan())
    assert all(t.succeeded for t in result.trials), result.failures
    assert np.isfinite(result.trials[0].diagnostics["correction_std_ratio"])


def test_missing_lightgbm_is_a_capability_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    with pytest.raises(UnsupportedCapabilityError, match=r"autohybts\[boosting\]"):
        build_estimator("lightgbm_regressor", None, 0)
    spec = CandidateSpec.standalone("lightgbm_regressor", {"lags": 7}, family="tree")
    result = TemporalEvaluator(default_registry()).evaluate(spec, plan())
    assert {t.status for t in result.trials} == {"SKIPPED"}


def test_lightgbm_runs_single_threaded_by_default() -> None:
    pytest.importorskip("lightgbm")
    assert build_estimator("lightgbm_regressor", None, 0).get_params()["n_jobs"] == 1


def test_xgboost_and_catboost_run_single_threaded_and_quietly() -> None:
    pytest.importorskip("xgboost")
    pytest.importorskip("catboost")
    assert build_estimator("xgboost_regressor", None, 0).get_params()["n_jobs"] == 1
    params = build_estimator("catboost_regressor", None, 0).get_params()
    assert params["thread_count"] == 1
    assert params["allow_writing_files"] is False  # no catboost_info/ directory
    assert params["random_state"] == 0


@pytest.mark.parametrize("module", ["xgboost", "catboost"])
def test_missing_xgboost_or_catboost_is_a_capability_error(
    module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(UnsupportedCapabilityError, match=r"autohybts\[boosting\]"):
        build_estimator(f"{module}_regressor", None, 0)
