"""Temporal cross-validation, model comparison, statistical tests and nested CV."""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pandas as pd
import pytest

from autohybts import AutoHybridExperiment
from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import SealedTest, TemporalCrossValidation, split_sealed_test
from autohybts.datasets import get_data
from autohybts.exceptions import (
    ConfigurationError,
    InsufficientHistoryError,
    SealedTestAccessError,
)
from autohybts.reporting.comparison import (
    diebold_mariano,
    friedman_nemenyi,
    holm,
    win_rate_matrix,
)
from autohybts.reporting.plots import COMPARISON_PLOTS

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
needs_matplotlib = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None, reason="extra `plot` not installed"
)

FAST = {
    "fh": 7,
    "fold": 2,
    "budget": "smoke",
    "base_models": ("naive", "seasonal_naive", "linear"),
    "correctors": ("linear",),
    "residual_windows": (7,),
    "alphas": (1.0,),
}


def frame(n: int = 200) -> SeriesFrame:
    return SeriesFrame.from_series(get_data("synthetic_daily", n=n))


@pytest.fixture(scope="module")
def experiment() -> AutoHybridExperiment:
    exp = AutoHybridExperiment(**FAST).fit(get_data("synthetic_daily", n=260))
    exp.compare_models()
    return exp


# ------------------------------------------------------------------ the CV plan
def test_cv_origins_are_anchored_at_the_end_and_step_back() -> None:
    plan = TemporalCrossValidation(horizon=7, n_folds=4, step=5).split(frame(100))
    assert [o.origin_index for o in plan] == [77, 82, 87, 92]
    assert plan.cv_origins[-1].last_target_index == 99
    assert {o.layer for o in plan} == {"cv"}
    assert list(plan.to_frame().index) == ["cv-0", "cv-1", "cv-2", "cv-3"]


def test_cv_required_length_and_rolling_windows() -> None:
    splitter = TemporalCrossValidation(
        horizon=7, n_folds=3, fold_strategy="rolling", window_size=30
    )
    assert splitter.required_length == 30 + 7 + 2 * 7
    plan = splitter.split(frame(51))
    assert {o.n_train for o in plan} == {30}
    with pytest.raises(InsufficientHistoryError, match="exigem ao menos 51"):
        splitter.split(frame(50))


@pytest.mark.parametrize(
    "kwargs",
    [{"horizon": 0}, {"horizon": 7, "n_folds": 0}, {"horizon": 7, "fold_strategy": "rolling"}],
)
def test_invalid_cv_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        TemporalCrossValidation(**kwargs)


def test_cv_plan_refuses_the_sealed_test_and_other_layers() -> None:
    _, sealed = split_sealed_test(frame(100), 10)
    with pytest.raises(SealedTestAccessError):
        TemporalCrossValidation(horizon=7).split(sealed)
    plan = TemporalCrossValidation(horizon=7, n_folds=2).split(frame(100))
    with pytest.raises(ConfigurationError, match="cv"):
        plan.origins("search")


def test_cv_fingerprint_changes_with_folds_and_data() -> None:
    data = frame(120)
    prints = {
        TemporalCrossValidation(horizon=7, n_folds=3).split(data).fingerprint,
        TemporalCrossValidation(horizon=7, n_folds=4).split(data).fingerprint,
        TemporalCrossValidation(horizon=7, n_folds=3, step=3).split(data).fingerprint,
        TemporalCrossValidation(horizon=7, n_folds=3).split(frame(121)).fingerprint,
    }
    assert len(prints) == 4


# ------------------------------------------------------------------ statistics
def test_holm_adjustment() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    adjusted = holm([0.2, math.nan])
    assert adjusted[0] == pytest.approx(0.2)
    assert math.isnan(adjusted[1])


def test_diebold_mariano() -> None:
    rng = np.random.default_rng(0)
    worse = rng.normal(2.0, 0.3, 40)
    better = worse - 0.5 + rng.normal(0, 0.05, 40)
    statistic, p_value = diebold_mariano(worse, better)
    assert statistic > 0
    assert p_value < 1e-6
    assert all(math.isnan(v) for v in diebold_mariano(worse, worse))
    assert all(math.isnan(v) for v in diebold_mariano(worse[:2], better[:2]))
    _, p_lagged = diebold_mariano(worse, better, max_lag=3)
    assert 0 <= p_lagged <= 1


def test_friedman_critical_difference_matches_nemenyi(experiment: AutoHybridExperiment) -> None:
    cv = experiment.cross_validate(models=["naive", "seasonal_naive", "linear"], folds=10, step=3)
    result = friedman_nemenyi(cv.results, cv.metric)
    # Nemenyi q(0.05, k=3) = 2.343; CD = q * sqrt(k(k+1) / 6N).
    assert result["critical_difference"] == pytest.approx(2.343 * math.sqrt(12 / 60), abs=1e-3)
    assert 0 <= result["p_value"] <= 1
    assert result["mean_ranks"].sum() == pytest.approx(6.0)  # 1 + 2 + 3


def test_win_rates_are_complementary_without_ties(experiment: AutoHybridExperiment) -> None:
    cv = experiment.cross_validate(models=["naive", "linear", "seasonal_naive"], folds=6)
    matrix = win_rate_matrix(cv.results, cv.metric)
    for a in matrix.index:
        for b in matrix.columns:
            if a != b:
                assert matrix.loc[a, b] + matrix.loc[b, a] == pytest.approx(1.0)


# ------------------------------------------------------------------ experiment API
def test_cross_validate_compares_on_shared_folds(experiment: AutoHybridExperiment) -> None:
    cv = experiment.cross_validate(models=["naive", "linear"], folds=6, step=7, reference="naive")
    assert list(cv.splits.index) == [f"cv-{i}" for i in range(6)]
    assert cv.units.shape == (6, 2)
    assert cv.labels[cv.reference] == "naive"
    summary = cv.summary
    assert {"mean", "std", "median", "n_valid", "mean_rank"} <= set(summary.columns)
    tests = cv.tests()
    assert list(tests["reference"]) == ["naive"]
    assert {"wilcoxon_p", "dm_p", "wilcoxon_p_holm", "dm_p_holm"} <= set(tests.columns)
    assert cv.splits["last_target"].max() == experiment.history_.end


def test_cross_validate_defaults_to_finalists(experiment: AutoHybridExperiment) -> None:
    cv = experiment.cross_validate(folds=4)
    finalists = experiment.compare_.comparison.results
    assert [r.candidate_id for r in cv.results] == [r.candidate_id for r in finalists]
    assert cv.results[0].candidate_id == experiment.compare_.best.candidate_id


def test_cross_validation_options_and_reuse(experiment: AutoHybridExperiment, monkeypatch) -> None:
    first = experiment.cross_validate(
        models=["linear"], folds=3, fold_strategy="rolling", window_size=60
    )
    assert {o for o in first.splits["n_train"]} == {60}
    from autohybts.evaluation.temporal import TemporalEvaluator

    def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError("re-evaluated")

    monkeypatch.setattr(TemporalEvaluator, "evaluate", refuse)
    again = experiment.cross_validate(
        models=["linear"], folds=3, fold_strategy="rolling", window_size=60
    )
    assert again.units.equals(first.units)


def test_cross_validation_never_reads_the_sealed_test(monkeypatch: pytest.MonkeyPatch) -> None:
    exp = AutoHybridExperiment(**FAST).fit(get_data("synthetic_daily", n=260))

    def refuse(self: SealedTest) -> object:
        raise AssertionError("sealed test read during cross-validation")

    monkeypatch.setattr(SealedTest, "_unseal", refuse)
    exp.cross_validate(models=["naive", "linear"], folds=4)


def test_compare_result_carries_the_finalists_comparison(experiment: AutoHybridExperiment) -> None:
    comparison = experiment.compare_.comparison
    assert comparison.reference == 0
    assert comparison.results[0].candidate_id == experiment.compare_.best.candidate_id
    assert {t.layer for r in comparison.results for t in r.trials} == {"select"}


def test_compare_external_evaluates_each_model_once() -> None:
    exp = AutoHybridExperiment(**FAST).fit(get_data("synthetic_daily", n=260))
    exp.compare_models()
    first = exp.compare_external(models=["naive", "seasonal_naive", "linear"])
    assert list(first.table.index) == ["naive", "seasonal_naive(season_length=7)", "linear(lags=7)"]
    assert first.table["ganho_MASE_vs_ref"].iloc[0] == 0.0
    assert first.steps.shape == (7, 3)
    assert len(first.tests) == 2
    again = exp.compare_external(models=["naive"])  # reuses the stored evaluation
    assert again.table.loc["naive", "MASE"] == first.table.loc["naive", "MASE"]
    with pytest.raises(ConfigurationError, match="selado"):
        AutoHybridExperiment(**{**FAST, "test_size": 0}).fit(
            get_data("synthetic_daily", n=260)
        ).compare_external(models=["naive"])


def test_nested_cv_runs_the_procedure_on_past_data_only(monkeypatch: pytest.MonkeyPatch) -> None:
    exp = AutoHybridExperiment(**FAST).fit(get_data("synthetic_daily", n=320))
    seen_ends: list[pd.Timestamp] = []
    original = AutoHybridExperiment.fit

    def spy(self: AutoHybridExperiment, y: object) -> AutoHybridExperiment:
        seen_ends.append(y.end if isinstance(y, SeriesFrame) else y.index[-1])
        return original(self, y)

    monkeypatch.setattr(AutoHybridExperiment, "fit", spy)
    result = exp.nested_cross_validate(outer_folds=2, baselines=("naive",))
    folds = result.folds
    assert list(folds.index) == ["cv-0", "cv-1"]
    for end, origin in zip(seen_ends, folds["origin"], strict=True):
        assert end == origin  # inner experiment sees history up to the outer origin only
    assert {"autohybts_MASE", "naive_MASE", "champion", "inner_selection_gap"} <= set(folds.columns)
    assert set(result.summary.index) == {"autohybts_MASE", "naive_MASE"}


def test_invalid_model_references(experiment: AutoHybridExperiment) -> None:
    with pytest.raises(ConfigurationError, match="Referência"):
        experiment.cross_validate(models=["naive"], reference="arima")
    with pytest.raises(ConfigurationError, match="Referência"):
        experiment.cross_validate(models=["naive"], reference=3)
    with pytest.raises(ConfigurationError, match="não reconhecido"):
        experiment.cross_validate(models=[42])


@needs_matplotlib
@pytest.mark.parametrize("kind", COMPARISON_PLOTS)
def test_comparison_plots(experiment: AutoHybridExperiment, kind: str) -> None:
    cv = experiment.cross_validate(models=["naive", "seasonal_naive", "linear"], folds=5)
    artifact = cv.plot(kind)
    assert artifact.kind == kind
    assert artifact.provenance["units"] == 5
    assert artifact.origin_range[1] < experiment.history_.end


@needs_matplotlib
def test_unknown_comparison_plot(experiment: AutoHybridExperiment) -> None:
    with pytest.raises(ConfigurationError, match="desconhecido"):
        experiment.cross_validate(models=["naive"], folds=3).plot("radar")
