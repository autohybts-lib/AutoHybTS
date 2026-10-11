"""SearchEngine: budgets (T19), audit (T26), failures (T14), isolation (T22), determinism."""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from autohybts.adapters.base import FloatArray, ForecasterBase
from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalPlan, TemporalSplitter, split_sealed_test
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.exceptions import (
    ConfigurationError,
    CoverageError,
    SealedTestAccessError,
    TemporalLeakageError,
)
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
    # Cost accounting: cache hits are counted apart and refit nothing.
    assert first.cached_count == 0
    assert first.unit_count == sum(len(r.trials) for r in first.results)
    assert second.cached_count == second.trial_count > 0
    assert second.unit_count == 0
    other_plan = make_plan(series(seed=1))
    third = engine(guard, cache=cache).run(other_plan)
    assert not any(e.cached for e in third.ledger)


class FakeClock:
    """Manual clock shared by the engine and the evaluator (thread-safe)."""

    def __init__(self) -> None:
        self.now = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds


class TickingEvaluator:
    """Evaluator proxy: each candidate takes ``seconds`` on the fake clock; starts recorded."""

    def __init__(self, clock: FakeClock, seconds: float) -> None:
        self.inner = TemporalEvaluator(default_registry(), seasonal_period=7)
        self.clock = clock
        self.seconds = seconds
        self.starts: list[float] = []

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    def evaluate(self, candidate: object, plan: object, layer: str = "search") -> object:
        self.starts.append(self.clock())
        result = self.inner.evaluate(candidate, plan, layer)  # type: ignore[arg-type]
        self.clock.advance(self.seconds)
        return result


def ticking_engine(
    limit: float | None, seconds: float, max_workers: int = 1
) -> tuple[SearchEngine, TickingEvaluator, FakeClock]:
    clock = FakeClock()
    evaluator = TickingEvaluator(clock, seconds)
    guard = SearchGuardConfig(max_wall_time_s=limit)
    engine = SearchEngine(
        evaluator,  # type: ignore[arg-type]
        FAST_SPACE,
        guard,
        max_workers=max_workers,
        clock=clock,
    )
    return engine, evaluator, clock


def reconciled(result: SearchResult) -> None:
    """Every evaluated entry has a result; every result has exactly one evaluated entry."""

    evaluated = [e.candidate_id for e in result.ledger if e.status == "EVALUATED"]
    assert len(evaluated) == len(set(evaluated))
    assert set(evaluated) == {r.candidate_id for r in result.results}


def test_wall_time_stops_new_candidates_inside_a_stage() -> None:
    # Each base takes 4 s and the deadline is 10 s: bases start at 0, 4 and 8; none after.
    search, evaluator, clock = ticking_engine(limit=10.0, seconds=4.0)
    result = search.run(make_plan())
    assert result.stop_reason == "max_wall_time"
    assert evaluator.starts == [0.0, 4.0, 8.0]
    assert result.trial_count == 3
    stopped = [e for e in result.ledger if e.status == "STOPPED"]
    assert any(e.reason == "max_wall_time_s atingido antes de iniciar" for e in stopped)
    assert all(e.stage == "S0" for e in result.ledger if e.status == "EVALUATED")
    # Timings come from the injected clock; the running candidate finished all origins.
    assert all(e.seconds == 4.0 for e in result.ledger if e.status == "EVALUATED")
    assert all(r.coverage("MASE") == 1.0 for r in result.results)
    assert result.elapsed_s == clock.now == 12.0
    reconciled(result)


def test_a_running_candidate_finishes_after_the_deadline_but_nothing_else_starts() -> None:
    search, evaluator, _ = ticking_engine(limit=10.0, seconds=50.0)
    result = search.run(make_plan())
    assert evaluator.starts == [0.0]
    assert result.trial_count == 1
    assert result.stop_reason == "max_wall_time"
    reconciled(result)


def test_deadline_before_the_first_evaluation_is_explicit() -> None:
    clock = FakeClock()
    evaluator = TickingEvaluator(clock, 1.0)
    guard = SearchGuardConfig(max_wall_time_s=5.0)

    def late() -> float:  # the start is read at 0, every later reading is past the deadline
        value = clock()
        clock.advance(100.0)
        return value

    with pytest.raises(CoverageError, match="max_wall_time_s"):
        SearchEngine(evaluator, FAST_SPACE, guard, clock=late).run(make_plan())  # type: ignore[arg-type]
    assert evaluator.starts == []


def test_parallel_queue_does_not_start_candidates_after_the_deadline() -> None:
    search, evaluator, _ = ticking_engine(limit=10.0, seconds=4.0, max_workers=2)
    result = search.run(make_plan())
    assert result.stop_reason == "max_wall_time"
    assert evaluator.starts
    assert max(evaluator.starts) < 10.0
    assert any(e.status == "STOPPED" for e in result.ledger)
    assert result.trial_count == len(evaluator.starts)
    reconciled(result)


def test_deadline_inside_the_last_stage_is_reported() -> None:
    # Without a deadline, N candidates start at 0, 1, ..., N-1 (1 s each). A deadline at
    # N-1 keeps out only the last one, which belongs to the last stage.
    free, _, _ = ticking_engine(limit=None, seconds=1.0)
    complete = free.run(make_plan())
    evaluated = [e for e in complete.ledger if e.status == "EVALUATED"]
    search, _, _ = ticking_engine(limit=float(len(evaluated) - 1), seconds=1.0)
    result = search.run(make_plan())
    assert result.trial_count == len(evaluated) - 1
    assert result.stop_reason == "max_wall_time"
    (stopped,) = [e for e in result.ledger if e.status == "STOPPED"]
    assert stopped.candidate_id == evaluated[-1].candidate_id
    assert stopped.stage == evaluated[-1].stage != "S0"
    reconciled(result)


def test_without_a_deadline_every_stage_runs() -> None:
    search, _, _ = ticking_engine(limit=None, seconds=1000.0)
    result = search.run(make_plan())
    assert result.stop_reason != "max_wall_time"
    assert not any(e.status == "STOPPED" for e in result.ledger)
    assert {e.stage for e in result.ledger if e.status == "EVALUATED"} > {"S0"}
    reconciled(result)


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


class Leaking:
    """Evaluator proxy whose evaluation raises a protocol error."""

    def __init__(self, error: Exception) -> None:
        self.inner = TemporalEvaluator(default_registry(), seasonal_period=7)
        self.error = error

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    def evaluate(self, candidate: object, plan: object, layer: str = "search") -> object:
        raise self.error


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize(
    "error", [TemporalLeakageError("resíduo posterior"), ConfigurationError("parâmetro inválido")]
)
def test_a_deadline_never_hides_configuration_or_leakage_errors(
    workers: int, error: Exception
) -> None:
    guard = SearchGuardConfig(max_wall_time_s=3600.0)
    engine = SearchEngine(Leaking(error), FAST_SPACE, guard, max_workers=workers)  # type: ignore[arg-type]
    with pytest.raises(type(error)):
        engine.run(make_plan())
