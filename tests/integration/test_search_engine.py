"""SearchEngine: budgets (T19), audit (T26), failures (T14), isolation (T22), determinism."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from autohybts.adapters.base import FloatArray, ForecasterBase
from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalPlan, TemporalSplitter, split_sealed_test
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.exceptions import CoverageError, SealedTestAccessError
from autohybts.registry import ModelRegistry, default_registry
from autohybts.search.budgets import SearchGuardConfig
from autohybts.search.cache import MemoryTrialCache
from autohybts.search.ledger import SearchResult
from autohybts.search.runner import SearchEngine
from autohybts.search.spaces import SearchSpace

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

FAST_SPACE = SearchSpace(
    base_models=("naive", "seasonal_naive", "linear", "knn", "tree", "mlp"),
    base_params={
        "mlp": [{"lags": 7, "estimator_params": {"hidden_layer_sizes": (8,), "max_iter": 200}}]
    },
    correctors=("linear", "knn"),
    residual_windows=(7,),
    alphas=(0.5, 1.0),
    refine_sources=("historical_oof_multistep",),
    seasonal_period=7,
)


def series(n: int = 220, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + rng.normal()
    t = np.arange(n)
    return 50 + 0.02 * t + 3 * np.sin(2 * np.pi * t / 7) + noise


def frame(values: np.ndarray) -> SeriesFrame:
    return SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=len(values)), values)


def make_plan(values: np.ndarray | None = None) -> TemporalPlan:
    return TemporalSplitter(horizon=7, n_search_folds=2, n_select_folds=2).plan(
        frame(series() if values is None else values)
    )


def engine(
    guard: SearchGuardConfig, registry: ModelRegistry | None = None, **kwargs: object
) -> SearchEngine:
    evaluator = TemporalEvaluator(registry or default_registry(), seasonal_period=7)
    return SearchEngine(evaluator, kwargs.pop("space", FAST_SPACE), guard, **kwargs)


def comparable(result: SearchResult) -> list[tuple[object, ...]]:
    """Ledger without timings or cache flags."""

    return [
        (e.candidate_id, e.stage, e.status, e.parent, e.reason, e.score, e.coverage)
        for e in result.ledger
    ]


class Failing(ForecasterBase):
    def _fit(self, y: FloatArray) -> None:
        raise RuntimeError("always fails")

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        return np.zeros(horizon)


@pytest.mark.parametrize("total", [3, 8, 15, 40])
def test_trials_never_exceed_the_budgets(total: int) -> None:
    # T19.
    guard = SearchGuardConfig(
        max_trials_total=total, max_trials_per_family=5, max_candidates_per_stage=6
    )
    result = engine(guard).run(make_plan())
    assert result.trial_count <= total
    assert all(count <= 5 for count in result.trials_by_family.values())
    stages = [e.stage for e in result.ledger if e.status == "EVALUATED"]
    assert all(stages.count(stage) <= 6 for stage in set(stages))


def test_ledger_accounts_for_every_proposal() -> None:
    # T26: evaluated, pruned and skipped proposals are all recorded, failures included.
    registry = default_registry()
    registry.register_base("failing", lambda p: Failing(), family="statistical")
    space = replace(FAST_SPACE, base_models=("failing", *FAST_SPACE.base_models))
    guard = SearchGuardConfig(
        max_trials_total=12, max_trials_per_family=8, max_candidates_per_stage=8
    )
    result = engine(guard, registry, space=space).run(make_plan())

    counts = result.counts()
    assert counts["EVALUATED"] == result.trial_count == len(result.results)
    assert counts["PRUNED_BUDGET"] > 0
    failing = next(e for e in result.ledger if e.label == "failing")
    assert failing.status == "EVALUATED"
    assert failing.coverage == 0.0
    assert failing.n_failed_units == 2
    assert result.result(failing.candidate_id).failures  # reasons kept
    assert failing.candidate_id not in result.retained
    table = result.ledger_frame()
    assert {"stage", "status", "parent", "reason", "score"} <= set(table.columns)
    assert result.stop_reason in {"max_trials_total", "budget_limited"}


def test_search_uses_only_the_search_layer() -> None:
    result = engine(SearchGuardConfig.preset("smoke")).run(make_plan())
    assert {t.layer for r in result.results for t in r.trials} == {"search"}


def test_stages_and_genealogy() -> None:
    result = engine(SearchGuardConfig.preset("small")).run(make_plan())
    evaluated = [e for e in result.ledger if e.status == "EVALUATED"]
    assert {e.stage for e in evaluated} >= {"S0", "S2"}
    ids = {e.candidate_id for e in evaluated}
    for entry in evaluated:
        if entry.stage in {"S2", "S3"}:
            assert entry.parent in ids  # every hybrid descends from something evaluated
    assert set(result.retained) <= ids
    assert set(result.residual_structures) <= ids


def test_refinement_runs_within_budget() -> None:
    result = engine(SearchGuardConfig.preset("small")).run(make_plan())
    assert any(e.stage == "S3" and e.status == "EVALUATED" for e in result.ledger)


def test_doubling_the_budget_changes_neither_plan_nor_sealed_test() -> None:
    # T22: the plan comes from development data only; more search does not touch it.
    values = series(260)
    dev, sealed = split_sealed_test(frame(values), 30)
    assert sealed is not None
    splitter = TemporalSplitter(horizon=7, n_search_folds=2, n_select_folds=2)
    plan = splitter.plan(dev)
    small = engine(SearchGuardConfig(max_trials_total=6)).run(plan)
    large = engine(SearchGuardConfig(max_trials_total=12)).run(plan)
    assert small.plan_fingerprint == large.plan_fingerprint == splitter.plan(dev).fingerprint
    assert sealed.fingerprint == frame(values).tail(30).fingerprint
    common = set(small.retained) & {r.candidate_id for r in large.results}
    for candidate_id in {r.candidate_id for r in small.results} & {
        r.candidate_id for r in large.results
    }:
        a, b = small.result(candidate_id), large.result(candidate_id)
        assert [t.y_pred for t in a.trials] == [t.y_pred for t in b.trials]
    assert common


def test_parallel_and_sequential_runs_are_identical() -> None:
    plan = make_plan()
    guard = SearchGuardConfig.preset("small")
    sequential = engine(guard).run(plan)
    parallel = engine(guard, max_workers=4).run(plan)
    assert comparable(sequential) == comparable(parallel)
    assert sequential.retained == parallel.retained
    assert sequential.protocol["protocol_hash"] == parallel.protocol["protocol_hash"]


def test_cache_reuses_evaluations_with_the_same_key() -> None:
    plan = make_plan()
    cache = MemoryTrialCache()
    guard = SearchGuardConfig.preset("smoke")
    first = engine(guard, cache=cache).run(plan)
    second = engine(guard, cache=cache).run(plan)
    assert comparable(first) == comparable(second)
    assert all(e.cached for e in second.ledger if e.status == "EVALUATED")
    other_plan = make_plan(series(seed=1))
    third = engine(guard, cache=cache).run(other_plan)
    assert not any(e.cached for e in third.ledger)


def test_wall_time_limit_stops_new_stages() -> None:
    ticks: Iterator[float] = iter([0.0, 0.0, 100.0, 100.0, 100.0, 100.0])
    guard = SearchGuardConfig(max_wall_time_s=10.0)
    result = engine(guard, clock=lambda: next(ticks)).run(make_plan())
    assert result.stop_reason == "max_wall_time"
    assert any(e.status == "STOPPED" for e in result.ledger)
    assert all(e.stage == "S0" for e in result.ledger if e.status == "EVALUATED")


def test_total_failure_is_explicit() -> None:
    registry = ModelRegistry()
    registry.register_base("failing", lambda p: Failing(), family="statistical")
    space = SearchSpace(base_models=("failing",), correctors=())
    with pytest.raises(CoverageError, match="Nenhum modelo-base"):
        engine(SearchGuardConfig(), registry, space=space).run(make_plan())


def test_sealed_test_is_refused() -> None:
    _, sealed = split_sealed_test(frame(series()), 20)
    with pytest.raises(SealedTestAccessError):
        engine(SearchGuardConfig()).run(sealed)


def test_smoke_search_is_fast_and_fully_audited() -> None:
    # I6 acceptance: a smoke search on a synthetic series in < 60 s with a complete ledger.
    plan = make_plan(series(300))
    started = time.perf_counter()
    result = engine(
        SearchGuardConfig.preset("smoke"),
        space=SearchSpace(seasonal_period=7, base_params={"arima": [{"order": (1, 1, 1)}]}),
    ).run(plan)
    assert time.perf_counter() - started < 60
    assert result.trial_count == 10
    statuses = set(result.counts())
    assert statuses <= {"EVALUATED", "PRUNED_BUDGET", "SKIPPED_CAPABILITY", "DUPLICATE", "STOPPED"}
    assert len(result.protocol["protocol_hash"]) == 64
