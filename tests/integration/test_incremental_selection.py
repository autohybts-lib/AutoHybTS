"""Search + incremental selection on real models: T12, oracle 2 of §19.2, audit.

Oracle 1 (simple model kept when residuals are unpredictable) is asserted at rule level in
tests/unit/test_selection_rules.py (T10, T24). End to end it does not hold as a guarantee:
measured on white noise with the correct base (linear on 7 lags, horizon 1), the protocol
rule (median gain >= 1 %, wins >= 60 %, variability >= 0.05) accepted a spurious hybrid in
5/12 seeds with 10 select units and 3/12 with 30. The thresholds are protocol rules, not
error control (§30.2); see the I7 notes in docs/implementation-plan.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalPlan, TemporalSplitter
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.exceptions import ConfigurationError
from autohybts.registry import default_registry
from autohybts.search.budgets import SearchGuardConfig
from autohybts.search.ledger import SearchResult
from autohybts.search.runner import SearchEngine
from autohybts.search.spaces import SearchSpace
from autohybts.selection.incremental import IncrementalSelector
from autohybts.selection.policy import SelectionPolicy

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def ar_series(n: int, phi: float, seed: int, noise_sd: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + rng.normal(0, noise_sd)
    t = np.arange(n)
    return 50 + 0.02 * t + 3 * np.sin(2 * np.pi * t / 7) + noise


def plan_for(values: np.ndarray, horizon: int, folds: int = 5) -> TemporalPlan:
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=len(values)), values)
    return TemporalSplitter(
        horizon=horizon, n_search_folds=folds, n_select_folds=folds, step=max(horizon, 3)
    ).plan(frame)


def pipeline(
    plan: TemporalPlan, space: SearchSpace, policy: SelectionPolicy
) -> tuple[SearchResult, object]:
    evaluator = TemporalEvaluator(default_registry(), seasonal_period=7)
    search = SearchEngine(evaluator, space, SearchGuardConfig.preset("small")).run(plan)
    return search, IncrementalSelector(evaluator, policy).select(search, plan)


SEASONAL_ONLY = SearchSpace(
    base_models=("seasonal_naive",),
    correctors=("linear",),
    residual_windows=(7,),
    alphas=(1.0,),
    refine_sources=(),
    seasonal_period=7,
)


def test_predictable_residuals_lead_to_an_accepted_hybrid() -> None:
    # Oracle 2: one step ahead, strong AR(0.9) residuals left by the seasonal naive.
    plan = plan_for(ar_series(400, 0.9, seed=0), horizon=1, folds=10)
    _, selection = pipeline(plan, SEASONAL_ONLY, SelectionPolicy())
    assert selection.champion.strategy == "residual_additive"
    decision = selection.decisions[-1]
    assert decision.accepted
    assert decision.variability_status == "passed"


def test_selection_is_reproducible() -> None:
    # T12.
    plan = plan_for(ar_series(300, 0.8, seed=1), horizon=7, folds=3)
    space = SearchSpace(
        base_models=("naive", "seasonal_naive", "linear", "knn"),
        seasonal_period=7,
        correctors=("linear",),
        residual_windows=(7,),
    )
    runs = [pipeline(plan, space, SelectionPolicy())[1] for _ in range(2)]
    first, second = runs
    assert first.champion == second.champion
    assert first.decisions == second.decisions
    assert first.audit == second.audit


def test_selection_only_scores_on_v_select_and_audits_the_search() -> None:
    plan = plan_for(ar_series(300, 0.8, seed=2), horizon=7, folds=3)
    space = SearchSpace(
        base_models=("naive", "seasonal_naive", "linear"),
        seasonal_period=7,
        correctors=("linear",),
        residual_windows=(7,),
    )
    search, selection = pipeline(plan, space, SelectionPolicy())
    assert {t.layer for r in selection.finalists for t in r.trials} == {"select"}
    audit = selection.audit
    assert audit.total_trials == search.trial_count
    assert audit.winner_id == selection.champion.candidate_id
    assert audit.comparison_origin_ids == tuple(o.fold_id for o in plan.select_origins)
    assert audit.search_protocol_hash == search.protocol["protocol_hash"]
    assert np.isfinite(audit.selection_gap)
    assert len(audit.decision_reasons) == len(selection.decisions)
    table = selection.decisions_frame()
    assert list(table["family"]) == [d.family for d in selection.decisions]


def test_exploratory_profile_is_flagged() -> None:
    plan = plan_for(ar_series(300, 0.8, seed=2), horizon=7, folds=3)
    space = SearchSpace(base_models=("naive", "linear"), correctors=(), seasonal_period=7)
    _, selection = pipeline(plan, space, SelectionPolicy(profile="exploratory"))
    assert selection.audit.exploratory


def test_search_and_selection_must_share_plan_and_metric() -> None:
    plan = plan_for(ar_series(300, 0.8, seed=2), horizon=7, folds=3)
    other = plan_for(ar_series(300, 0.8, seed=5), horizon=7, folds=3)
    space = SearchSpace(base_models=("naive", "linear"), correctors=(), seasonal_period=7)
    search, _ = pipeline(plan, space, SelectionPolicy())
    evaluator = TemporalEvaluator(default_registry(), seasonal_period=7)
    with pytest.raises(ConfigurationError, match="outro plano"):
        IncrementalSelector(evaluator, SelectionPolicy()).select(search, other)
    mae_first = TemporalEvaluator(default_registry(), metrics=("MAE", "MASE"), seasonal_period=7)
    with pytest.raises(ConfigurationError, match="incomparáveis"):
        IncrementalSelector(mae_first, SelectionPolicy()).select(search, plan)
