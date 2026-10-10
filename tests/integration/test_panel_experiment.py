"""Panels end to end: oracle 5 of §19.2 (two series with conflicting gains) and the API."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from autohybts import AutoHybridExperiment, load_model, save_model
from autohybts.data.panel import PanelFrame, PanelPlan, plan_panel
from autohybts.data.splitter import TemporalSplitter
from autohybts.datasets import get_data
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.exceptions import (
    ConfigurationError,
    SealedTestAccessError,
)
from autohybts.forecasting.finalized import PanelForecaster
from autohybts.registry import default_registry
from autohybts.search.budgets import SearchGuardConfig
from autohybts.search.ledger import SearchResult
from autohybts.search.runner import SearchEngine
from autohybts.search.spaces import SearchSpace
from autohybts.selection.incremental import IncrementalSelector, SelectionResult
from autohybts.selection.policy import SelectionPolicy, paired_comparison
from autohybts.selection.rank import RANKING_CONVENTION

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

FAST = {
    "fh": 7,
    "fold": 2,
    "budget": "smoke",
    "base_models": ("naive", "seasonal_naive", "linear"),
    "correctors": ("linear",),
    "residual_windows": (7,),
    "alphas": (1.0,),
}


# ------------------------------------------------------------------------------ oracle 5
SEASONAL_ONLY = SearchSpace(
    base_models=("seasonal_naive",),
    correctors=("linear",),
    residual_windows=(7,),
    alphas=(1.0,),
    refine_sources=(),
    seasonal_period=7,
)


def ar_series(n: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + rng.normal(0, 1.0)
    t = np.arange(n)
    return 50 + 0.02 * t + 3 * np.sin(2 * np.pi * t / 7) + noise


def conflicting_panel() -> PanelPlan:
    """``a``: AR(0.9) residuals the corrector can learn; ``b``: white noise, nothing to learn."""

    index = pd.date_range("2024-01-01", periods=400, freq="D")
    panel = PanelFrame.from_dict(
        {
            "a": pd.Series(ar_series(400, 0.9, seed=0), index=index),
            "b": pd.Series(ar_series(400, 0.0, seed=3), index=index),
        }
    )
    splitter = TemporalSplitter(horizon=1, n_search_folds=10, n_select_folds=10, step=3)
    return plan_panel(splitter, panel)


def run(plan: PanelPlan) -> tuple[SearchResult, SelectionResult]:
    evaluator = TemporalEvaluator(default_registry(), seasonal_period=7)
    search = SearchEngine(evaluator, SEASONAL_ONLY, SearchGuardConfig.preset("small")).run(plan)
    return search, IncrementalSelector(evaluator, SelectionPolicy()).select(search, plan)


@pytest.fixture(scope="module")
def oracle5() -> tuple[PanelPlan, SearchResult, SelectionResult]:
    plan = conflicting_panel()
    return (plan, *run(plan))


def test_oracle5_gains_conflict_and_the_pooled_decision_is_auditable(
    oracle5: tuple[PanelPlan, SearchResult, SelectionResult],
) -> None:
    plan, _, selection = oracle5
    decision = selection.decisions[-1]
    base = next(f for f in selection.finalists if f.candidate.strategy == "standalone")
    hybrid = next(f for f in selection.finalists if f.candidate.strategy != "standalone")

    per_series = {
        sid: paired_comparison(base.for_series(sid), hybrid.for_series(sid), "MASE", 1e-12)
        for sid in plan.series_ids
    }
    # The conflict: the hybrid wins most units of ``a`` and loses most units of ``b``.
    assert per_series["a"].median_gain > 0 > per_series["b"].median_gain
    assert per_series["a"].win_rate >= 0.6 > per_series["b"].win_rate

    # The decision pools the units of both series and is reproducible from them.
    assert set(decision.unit_ids) == {
        f"{sid}:{fold}" for sid in plan.series_ids for fold in (f"select-{i}" for i in range(10))
    }
    gains = [g for comparison in per_series.values() for g in comparison.gains]
    wins = [
        new < old
        for comparison in per_series.values()
        for old, new in zip(comparison.champion_errors, comparison.challenger_errors, strict=True)
    ]
    assert decision.median_gain == pytest.approx(float(np.median(gains)))
    assert decision.win_rate == pytest.approx(float(np.mean(wins)))
    # Pooled, the win rate falls below 60 %: the simple model stays champion.
    assert not decision.accepted
    assert "win_rate_below_threshold" in decision.reason_codes
    assert selection.champion.strategy == "standalone"
    assert set(selection.champion_per_series) == {"a", "b"}


def test_oracle5_mean_rank_is_the_average_of_per_series_ranks(
    oracle5: tuple[PanelPlan, SearchResult, SelectionResult],
) -> None:
    plan, search, _ = oracle5
    ranked = RANKING_CONVENTION.rank(list(search.results), "MASE")
    per_series = {
        sid: {
            s.key: s.mean_rank
            for s in RANKING_CONVENTION.rank([r.for_series(sid) for r in search.results], "MASE")
        }
        for sid in plan.series_ids
    }
    for structure in ranked:
        expected = np.mean([per_series[sid][structure.key] for sid in plan.series_ids])
        assert structure.mean_rank == pytest.approx(expected)
        assert set(structure.per_series) == set(plan.series_ids)
    assert [s.mean_rank for s in ranked] == sorted(s.mean_rank for s in ranked)


def test_oracle5_is_reproducible(
    oracle5: tuple[PanelPlan, SearchResult, SelectionResult],
) -> None:
    plan, _, first = oracle5
    _, second = run(plan)
    assert first.champion == second.champion
    assert first.decisions == second.decisions
    assert first.audit == second.audit
    assert first.champion_per_series == second.champion_per_series


# ------------------------------------------------------------------------------ API
def panel_data() -> dict[str, pd.Series]:
    return {
        "a": get_data("synthetic_daily", n=200, seed=1),
        "b": get_data("synthetic_daily", n=200, seed=2),
    }


@pytest.fixture(scope="module")
def fitted() -> tuple[AutoHybridExperiment, object, PanelForecaster]:
    exp = AutoHybridExperiment(**FAST).fit(panel_data())
    result = exp.compare_models()
    model = exp.finalize_model(result.best)
    assert isinstance(model, PanelForecaster)
    return exp, result, model


def test_panel_inputs_are_equivalent() -> None:
    data = panel_data()
    from_dict = AutoHybridExperiment(**FAST).fit(data)
    long = PanelFrame.from_dict(data).to_long()
    from_long = AutoHybridExperiment(**FAST).fit(
        long, series_col="series_id", date_col="timestamp", target="value"
    )
    from_panel = AutoHybridExperiment(**FAST).fit(PanelFrame.from_dict(data))
    assert from_dict.is_panel
    assert from_dict.splits.equals(from_long.splits)
    assert from_dict.work_plan_.fingerprint == from_panel.work_plan_.fingerprint
    assert from_dict.splits.index.names == ["series_id", "fold_id"]
    with pytest.raises(ConfigurationError, match="series_col"):
        AutoHybridExperiment(**FAST).fit(long)


def test_single_series_path_is_unchanged() -> None:
    y = get_data("synthetic_daily", n=200, seed=1)
    exp = AutoHybridExperiment(**FAST).fit(y)
    assert not exp.is_panel
    assert exp.work_plan_ is exp.plan_
    assert list(exp.splits.index) == ["search-0", "search-1", "select-0", "select-1"]


def test_panel_compare_finalize_predict(
    fitted: tuple[AutoHybridExperiment, object, PanelForecaster],
) -> None:
    exp, result, model = fitted
    assert model.series_ids == ("a", "b")
    # Each series is refitted with the configuration chosen for it.
    for sid, spec in result.selection.champion_per_series.items():  # type: ignore[attr-defined]
        assert model[sid].spec == spec
        assert model[sid].training_cutoff == exp.panel_[sid].end
    forecast = exp.predict_model(model).predictions
    assert forecast.groupby("series_id").size().to_dict() == {"a": 7, "b": 7}
    with pytest.raises(ConfigurationError, match="'z'"):
        model["z"]


def test_panel_external_evaluation_is_per_series_and_once(
    fitted: tuple[AutoHybridExperiment, object, PanelForecaster],
) -> None:
    exp, _, model = fitted
    external = exp.evaluate_external(model)
    table = external.metrics
    assert list(table.index.get_level_values("series_id")) == ["a", "b", "média"]
    assert table["MASE"].iloc[-1] == pytest.approx(table["MASE"].iloc[:2].mean())
    with pytest.raises(SealedTestAccessError):
        exp.evaluate_external(model)


def test_panel_model_round_trip(
    fitted: tuple[AutoHybridExperiment, object, PanelForecaster], tmp_path: object
) -> None:
    _, _, model = fitted
    path = save_model(model, tmp_path)  # type: ignore[arg-type]
    loaded = load_model(path, trusted=True)
    assert isinstance(loaded, PanelForecaster)
    assert loaded.model_id == model.model_id
    pd.testing.assert_frame_equal(loaded.predict(5), model.predict(5))


def test_panel_reporting_and_limits(
    fitted: tuple[AutoHybridExperiment, object, PanelForecaster],
) -> None:
    exp, _, _ = fitted
    profile = exp.data_profile()
    assert list(profile.columns) == ["a", "b"]
    with pytest.raises(ConfigurationError, match="'z'"):
        exp.plot_data("series", series_id="z")
    cv = exp.cross_validate(models=["naive", "seasonal_naive"], folds=2)
    assert len(cv.units) == 4
    assert sorted({unit.split(":")[0] for unit in cv.units.index}) == ["a", "b"]


def test_nested_cv_runs_on_the_panel(
    fitted: tuple[AutoHybridExperiment, object, PanelForecaster],
) -> None:
    exp, _, _ = fitted
    nested = exp.nested_cross_validate(outer_folds=2, baselines=("naive",), budget="smoke")
    folds = nested.folds
    assert folds.index.names == ["fold_id", "series_id"]
    assert len(folds) == 4  # 2 outer folds x 2 series
    assert set(folds.index.get_level_values("series_id")) == {"a", "b"}
    assert set(folds["status"]) == {"SUCCEEDED"}
    assert nested.summary.loc[f"autohybts_{nested.metric}", "n_folds"] == 4
    # Each series is cut at its own origin; the outer block is never in the inner history.
    for (_, sid), row in folds.iterrows():
        assert row["origin"] < exp.panel_[sid].end


@pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None, reason="extra `plot` not installed"
)
def test_panel_plots_one_series(
    fitted: tuple[AutoHybridExperiment, object, PanelForecaster],
) -> None:
    exp, result, _ = fitted
    exp.plot_data("series", series_id="b")
    exp.plot_model(result.best, "cv_forecasts", series_id="b")  # type: ignore[attr-defined]


def test_a_member_of_an_evaluated_panel_cannot_be_reevaluated(
    fitted: tuple[AutoHybridExperiment, object, PanelForecaster],
) -> None:
    exp, _, model = fitted
    if model.model_id not in exp._externally_evaluated:
        exp.evaluate_external(model)
    with pytest.raises(SealedTestAccessError):
        exp.evaluate_external(model["a"])


def test_panel_variant_b_is_evaluated_one_step_at_a_time() -> None:
    exp = AutoHybridExperiment(**{**FAST, "variant": "B"}).fit(panel_data())
    model = exp.finalize_model(exp.compare_models().best)
    assert isinstance(model, PanelForecaster)
    predictions = exp.evaluate_external(model).predictions
    # Rolling one step: every forecast is one step ahead of an observed origin.
    assert set(predictions["horizon"]) == {1}
    assert predictions.groupby("series_id").size().to_dict() == {"a": 7, "b": 7}
