"""TemporalEvaluator: same units for every candidate, failures audited, errors not swallowed."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pytest

from autohybts.adapters.base import FloatArray, ForecasterBase
from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalPlan, TemporalSplitter, split_sealed_test
from autohybts.domain.specs import CandidateSpec
from autohybts.evaluation.coverage import coverage_report
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.exceptions import (
    ConfigurationError,
    SealedTestAccessError,
    TemporalLeakageError,
)
from autohybts.registry import ModelRegistry, default_registry
from autohybts.reporting.leaderboard import leaderboard

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def synthetic(n: int = 160, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i - 1] + rng.normal()
    t = np.arange(n)
    return 50 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 7) + noise


def make_plan(values: np.ndarray) -> TemporalPlan:
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=len(values)), values)
    return TemporalSplitter(horizon=7, n_search_folds=3, n_select_folds=2).plan(frame)


def spec(
    registry: ModelRegistry, name: str, params: Mapping[str, Any] | None = None
) -> CandidateSpec:
    return CandidateSpec.standalone(name, params, family=registry.entry(name).family)


class Exploding(ForecasterBase):
    """Fails at one specific origin length, to check that the evaluator carries on."""

    def __init__(self, fail_at: int = -1, error: str = "candidate") -> None:
        super().__init__()
        self.fail_at = fail_at
        self.error = error

    def _fit(self, y: FloatArray) -> None:
        if y.size == self.fail_at:
            if self.error == "leak":
                raise TemporalLeakageError("forbidden look-ahead")
            raise RuntimeError("numerical blow-up")
        self.last_ = float(y[-1])

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        return np.full(horizon, history[-1])


@pytest.fixture
def registry() -> ModelRegistry:
    registry = default_registry()
    registry.register_base("exploding", lambda p: Exploding(**p), family="statistical")
    return registry


def test_leaderboard_for_naive_and_linear(registry: ModelRegistry) -> None:
    # I4 acceptance: a leaderboard built from EvaluationResult on a synthetic series.
    plan = make_plan(synthetic())
    evaluator = TemporalEvaluator(registry, seasonal_period=7)
    results = [
        evaluator.evaluate(spec(registry, "naive"), plan, "search"),
        evaluator.evaluate(spec(registry, "linear", {"lags": 14}), plan, "search"),
    ]
    table = leaderboard(results)
    assert list(table["model"]) == ["linear(lags=14)", "naive"]  # linear wins on this series
    assert (table["coverage"] == 1.0).all()
    assert (table["n_units"] == 3).all()
    assert {"MASE", "MAE", "RMSE", "sMAPE", "fit_seconds", "family"} <= set(table.columns)


def test_every_candidate_is_scored_on_the_same_units(registry: ModelRegistry) -> None:
    plan = make_plan(synthetic())
    evaluator = TemporalEvaluator(registry)
    folds = {
        name: [t.fold_id for t in evaluator.evaluate(spec(registry, name), plan).trials]
        for name in ("naive", "linear", "tree")
    }
    assert folds["naive"] == folds["linear"] == folds["tree"]
    assert folds["naive"] == [o.fold_id for o in plan]


def test_repeated_evaluation_is_identical(registry: ModelRegistry) -> None:
    # T12 (I4 part): same candidate, plan and seed give the same scores and seeds.
    plan = make_plan(synthetic())
    evaluator = TemporalEvaluator(registry, session_seed=7)
    candidate = spec(registry, "mlp", {"lags": 7, "estimator_params": {"max_iter": 200}})
    first, second = (evaluator.evaluate(candidate, plan) for _ in range(2))
    for a, b in zip(first.trials, second.trials, strict=True):
        assert (a.y_pred, a.seed, a.trial_key) == (b.y_pred, b.seed, b.trial_key)
    assert len({t.seed for t in first.trials}) == len(first.trials)  # one seed per fold


def test_fixed_random_state_is_respected(registry: ModelRegistry) -> None:
    plan = make_plan(synthetic())
    result = TemporalEvaluator(registry).evaluate(
        spec(
            registry, "random_forest", {"random_state": 5, "estimator_params": {"n_estimators": 5}}
        ),
        plan,
    )
    assert {t.seed for t in result.trials} == {5}


def test_deterministic_models_get_no_seed(registry: ModelRegistry) -> None:
    result = TemporalEvaluator(registry).evaluate(spec(registry, "naive"), make_plan(synthetic()))
    assert {t.seed for t in result.trials} == {None}


def test_failing_origin_is_audited_and_evaluation_continues(registry: ModelRegistry) -> None:
    # T14 (I4 part).
    plan = make_plan(synthetic())
    target = plan.search_origins[1]
    result = TemporalEvaluator(registry).evaluate(
        spec(registry, "exploding", {"fail_at": target.n_train}), plan
    )
    statuses = {t.fold_id: t.status for t in result.trials}
    assert statuses[target.fold_id] == "FAILED"
    assert sum(s == "SUCCEEDED" for s in statuses.values()) == len(statuses) - 1
    assert "numerical blow-up" in result.failures[target.fold_id]
    assert result.coverage() == pytest.approx((len(statuses) - 1) / len(statuses))
    report = coverage_report(result, "MASE")
    assert report.missing == (target.fold_id,)
    assert not report.complete
    assert np.isnan(result.scores("MASE")[target.fold_id])


def test_leakage_inside_a_model_stops_the_run(registry: ModelRegistry) -> None:
    plan = make_plan(synthetic())
    candidate = spec(
        registry, "exploding", {"fail_at": plan.search_origins[0].n_train, "error": "leak"}
    )
    with pytest.raises(TemporalLeakageError):
        TemporalEvaluator(registry).evaluate(candidate, plan)


def test_invalid_hyperparameters_stop_the_run(registry: ModelRegistry) -> None:
    with pytest.raises(ConfigurationError):
        TemporalEvaluator(registry).evaluate(
            spec(registry, "linear", {"estimator_params": {"banana": 1}}), make_plan(synthetic())
        )


def test_missing_optional_dependency_skips(
    registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.arima.model", None)
    result = TemporalEvaluator(registry).evaluate(
        spec(registry, "arima", {"order": (1, 0, 0)}), make_plan(synthetic())
    )
    assert {t.status for t in result.trials} == {"SKIPPED"}
    assert "autohybts[stats]" in next(iter(result.failures.values()))


def test_too_short_training_for_the_model_fails_the_unit(registry: ModelRegistry) -> None:
    plan = make_plan(synthetic())
    result = TemporalEvaluator(registry).evaluate(spec(registry, "linear", {"lags": 155}), plan)
    assert {t.status for t in result.trials} == {"FAILED"}
    assert "InsufficientHistoryError" in result.failures["search-0"]


def test_mase_scale_uses_each_origins_training_only(registry: ModelRegistry) -> None:
    # T18 at evaluator level: MASE / MAE equals 1 / scale of that origin's training slice.
    values = synthetic()
    plan = make_plan(values)
    result = TemporalEvaluator(registry, seasonal_period=7).evaluate(spec(registry, "naive"), plan)
    for trial, origin in zip(result.trials, plan, strict=True):
        train = values[origin.train_slice]
        scale = np.mean(np.abs(train[7:] - train[:-7]))
        assert trial.metrics["MASE"] == pytest.approx(trial.metrics["MAE"] / scale)


def test_degenerate_scale_makes_the_unit_invalid_for_that_metric(registry: ModelRegistry) -> None:
    values = np.concatenate([np.full(130, 5.0), np.arange(30.0)])
    plan = make_plan(values)
    result = TemporalEvaluator(registry).evaluate(spec(registry, "naive"), plan)
    report = coverage_report(result, "MASE")
    assert report.missing  # flat training windows: MASE undefined there
    assert coverage_report(result, "MAE").complete
    assert result.coverage("MAE") == 1.0 > result.coverage("MASE")


def test_layers_can_be_evaluated_separately(registry: ModelRegistry) -> None:
    plan = make_plan(synthetic())
    evaluator = TemporalEvaluator(registry)
    search = evaluator.evaluate(spec(registry, "naive"), plan, "search")
    select = evaluator.evaluate(spec(registry, "naive"), plan, "select")
    assert {t.layer for t in search.trials} == {"search"}
    assert [t.fold_id for t in select.trials] == ["select-0", "select-1"]


def test_forecast_frame_is_indexed_and_carries_the_truth(registry: ModelRegistry) -> None:
    values = synthetic()
    plan = make_plan(values)
    result = TemporalEvaluator(registry).evaluate(spec(registry, "naive"), plan, "select")
    frame = result.forecasts().to_pandas()
    last = plan.select_origins[-1]
    rows = frame[frame["fold_id"] == last.fold_id]
    assert rows["y_true"].tolist() == values[last.target_slice].tolist()
    assert rows["timestamp"].iloc[-1] == plan.history.end


def test_sealed_test_is_refused(registry: ModelRegistry) -> None:
    _, sealed = split_sealed_test(
        SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=60), synthetic(60)), 10
    )
    with pytest.raises(SealedTestAccessError):
        TemporalEvaluator(registry).evaluate(spec(registry, "naive"), sealed)


def test_residual_candidates_are_evaluated(registry: ModelRegistry) -> None:
    from autohybts.domain.specs import ModelSpec

    candidate = CandidateSpec(
        "residual",
        "residual_additive",
        ModelSpec.of("naive"),
        ModelSpec.of("linear"),
        7,
        1.0,
        "historical_oof_1step",
    )
    result = TemporalEvaluator(registry).evaluate(candidate, make_plan(synthetic()))
    assert {t.status for t in result.trials} == {"SUCCEEDED"}
    with pytest.raises(ConfigurationError, match="não registrado"):
        TemporalEvaluator(registry).evaluate(
            CandidateSpec(
                "residual",
                "residual_additive",
                ModelSpec.of("naive"),
                ModelSpec.of("svm"),
                7,
                1.0,
                "historical_oof_1step",
            ),
            make_plan(synthetic()),
        )


def test_protocol_settings_change_trial_keys(registry: ModelRegistry) -> None:
    from autohybts.residuals.generation import ResidualSettings

    plan = make_plan(synthetic())
    candidate = spec(registry, "naive")
    keys = {
        TemporalEvaluator(registry, **kwargs).evaluate(candidate, plan).trials[0].trial_key
        for kwargs in (
            {},
            {"seasonal_period": 7},
            {"metrics": ("MAE",)},
            {"residual_settings": ResidualSettings(tail_fraction=0.3)},
        )
    }
    assert len(keys) == 4


def test_unknown_model_and_invalid_settings(registry: ModelRegistry) -> None:
    with pytest.raises(ConfigurationError, match="não registrado"):
        TemporalEvaluator(registry).evaluate(
            CandidateSpec.standalone("prophet", family="x"), make_plan(synthetic())
        )
    with pytest.raises(ConfigurationError):
        TemporalEvaluator(registry, seasonal_period=0)
    with pytest.raises(ConfigurationError):
        TemporalEvaluator(registry, metrics=("MAPE",))
    with pytest.raises(ConfigurationError, match="plano temporal"):
        TemporalEvaluator(registry).evaluate(spec(registry, "naive"), "plan")


def test_leaderboard_refuses_incomparable_results(registry: ModelRegistry) -> None:
    evaluator = TemporalEvaluator(registry)
    a = evaluator.evaluate(spec(registry, "naive"), make_plan(synthetic()))
    b = evaluator.evaluate(spec(registry, "naive"), make_plan(synthetic(seed=1)))
    c = TemporalEvaluator(registry, metrics=("MAE",)).evaluate(
        spec(registry, "naive"), make_plan(synthetic())
    )
    with pytest.raises(ConfigurationError, match="planos"):
        leaderboard([a, b])
    with pytest.raises(ConfigurationError, match="métricas"):
        leaderboard([a, c])
    with pytest.raises(ConfigurationError, match="ordenação"):
        leaderboard([a], sort="WAPE")
    with pytest.raises(ConfigurationError):
        leaderboard([])


def test_trial_table(registry: ModelRegistry) -> None:
    result = TemporalEvaluator(registry).evaluate(spec(registry, "naive"), make_plan(synthetic()))
    table = result.to_frame()
    assert list(table.index) == [o.fold_id for o in make_plan(synthetic())]
    assert {"status", "MASE", "fit_seconds", "seed", "reason"} <= set(table.columns)


class ThirdParty:
    """Implements the protocol directly, without ForecasterBase's error wrapping."""

    def __init__(self, crash: bool = True) -> None:
        self.crash = crash

    def fit(self, y: object, X: None = None) -> ThirdParty:
        if self.crash:
            raise ZeroDivisionError("division by zero in vendor code")
        return self

    def predict(self, horizon: int, context: object) -> np.ndarray:
        return np.zeros(horizon)

    def clone(self) -> ThirdParty:
        return ThirdParty(self.crash)

    def get_params(self) -> dict[str, Any]:
        return {"crash": self.crash}


def test_third_party_exceptions_become_failed_trials(registry: ModelRegistry) -> None:
    registry.register_base("vendor", lambda p: ThirdParty(**p), family="statistical")
    result = TemporalEvaluator(registry).evaluate(spec(registry, "vendor"), make_plan(synthetic()))
    assert {t.status for t in result.trials} == {"FAILED"}
    assert result.failures["search-0"].startswith("ZeroDivisionError")
    assert np.isnan(result.mean("MASE"))
    assert result.coverage() == 0.0
    assert len(result.forecasts()) == 0


def test_forecasts_skip_failed_origins(registry: ModelRegistry) -> None:
    plan = make_plan(synthetic())
    target = plan.search_origins[0]
    result = TemporalEvaluator(registry).evaluate(
        spec(registry, "exploding", {"fail_at": target.n_train}), plan
    )
    frame = result.forecasts().to_pandas()
    assert target.fold_id not in set(frame["fold_id"])
    assert len(frame) == (len(plan.origins()) - 1) * 7
    assert repr(result.forecasts()).startswith("ForecastFrame(rows=")
