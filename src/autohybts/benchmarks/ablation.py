"""Ablation of the selection procedure.

Four procedures, the same search budget and the same external origins (each series'
sealed test):

* ``lowest_error``: the finalist with the lowest mean error on ``V_select``;
* ``incremental``: the guarded incremental chain with the protocol thresholds, all
  families;
* ``incremental_simple``: as above, only structures with at most two fitted models
  (standalone and additive residual hybrids);
* ``guarded_nested``: as ``incremental``, but if nested cross-validation on the
  development history says the procedure (all families, as in ``incremental``) does not
  beat the procedure "best standalone model of the inner selection", the best standalone
  model of ``V_select`` is kept. Both sides of that comparison are chosen inside each
  outer fold, without its block.

Each row (series x seed x procedure) records the external score of the chosen model,
those of the best standalone model of ``V_select`` and of the ``naive`` and
``seasonal_naive`` references, and the full cost of the procedure: candidates and units
(origin x series) of every search, selection and nested inner search, the configured
budget, the stop reason, the proposals cut by each budget limit and the seconds of each
phase. ``lowest_error``,
``incremental`` and ``guarded_nested`` share one search (column ``search``): their search
cost is the same work, not three measurements. Budgets limit by count only (no wall
clock), so a run is reproducible on another machine. Refitting the chosen model on the
pre-test history (``seconds_refit``) and reading the external test
(``seconds_external``) are reporting, not part of the procedure's cost.

``ablation_report`` gives, per procedure: median gain against the best standalone model
on the test, wins/ties/losses against it, share of hybrids and mean complexity,
validation–test gap, mean rank, spread of the median gain across seeds, coverage and
cost. The external test only reports: never pick the "winning" procedure on it.

From Python::

    from autohybts.benchmarks import ablation_report, run_ablation, synthetic_cases

    table = run_ablation(synthetic_cases(12), seeds=(0, 1, 2))
    ablation_report(table)

From the command line (``--out`` also writes the resolved protocol next to the CSV)::

    python -m autohybts.benchmarks.ablation --dataset synthetic --series 12 --seeds 0 1 2
    python -m autohybts.benchmarks.ablation --dataset m4-daily --series 20 --out ablation.csv
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from autohybts.benchmarks.checkpoint import (
    Checkpoint,
    series_fingerprint,
)
from autohybts.datasets import get_data, load_benchmark
from autohybts.datasets.cache import CacheManager
from autohybts.datasets.descriptor import BenchmarkDataset
from autohybts.datasets.providers import M4Provider
from autohybts.domain.identity import stable_hash
from autohybts.domain.specs import CandidateSpec
from autohybts.evaluation.metrics import resolve_metrics
from autohybts.exceptions import (
    ConfigurationError,
    DataValidationError,
    TemporalLeakageError,
    UnsupportedCapabilityError,
)
from autohybts.experiment import AutoHybridExperiment
from autohybts.persistence.manifest import environment, json_safe, source_fingerprint, utc_now
from autohybts.reporting.strategies import complexity
from autohybts.results import CompareResult
from autohybts.search.budgets import PRESETS, BudgetPreset

__all__ = [
    "ABLATION_SCHEMA",
    "PROCEDURES",
    "AblationCase",
    "ablation_protocol",
    "ablation_report",
    "ablation_uncertainty",
    "benchmark_cases",
    "run_ablation",
    "run_case",
    "synthetic_cases",
]

# Version of the table layout; 2 added the cost breakdown and the naive references, and
# ``candidates``/``seconds`` of ``guarded_nested`` now include the nested CV's own work.
ABLATION_SCHEMA: Final = "ablation/3"
# Version 3 aligns the nested guard's reference with its lowest-select-error fallback.
PROCEDURES: Final = ("lowest_error", "incremental", "incremental_simple", "guarded_nested")
ALL_FAMILIES: Final = ("standalone", "residual", "multistage", "decomposition", "combination")
SIMPLE_FAMILIES: Final = ("standalone", "residual")
REFERENCES: Final = ("naive", "seasonal_naive")
DEFAULT_SPACE: Final[dict[str, Any]] = {
    "fold": 2,
    "base_models": ("naive", "seasonal_naive", "linear", "knn", "mlp"),
    "correctors": ("linear", "knn"),
    "residual_windows": (7,),
    "alphas": (1.0,),
}


@dataclass(frozen=True)
class AblationCase:
    """One series: development history with the external test at the end."""

    series_id: str
    values: pd.Series
    horizon: int
    test_size: int
    seasonal_period: int | None = None
    # Group for stratified reports: residual predictability, benchmark subset ...
    stratum: str = ""
    source: dict[str, Any] = field(default_factory=dict)


def synthetic_cases(n_series: int, n: int = 400, horizon: int = 14) -> Iterator[AblationCase]:
    """Daily series whose residual predictability cycles through AR(0), AR(0.5), AR(0.8)."""

    for index in range(n_series):
        phi = (0.0, 0.5, 0.8)[index % 3]
        y = get_data("synthetic_daily", n=n, seed=100 + index, phi=phi)
        yield AblationCase(f"syn{index}-phi{phi}", y, horizon, horizon, 7, f"phi={phi}")


def benchmark_cases(
    name: str,
    subset: str,
    n_series: int,
    seed: int = 0,
    *,
    series_ids: Sequence[str] | None = None,
    development_only: bool = False,
    development_tail: int | None = None,
) -> Iterator[AblationCase]:
    """A random sample of an official benchmark (downloads on first use).

    The sample is reproducible: ``seed`` draws ``n_series`` ids without replacement from
    the benchmark's id list; the official test of each series is its sealed test.
    """

    if development_only:
        if name != "m4" or subset != "daily":
            raise ConfigurationError("development_only está disponível somente para M4 diário.")
        yield from _m4_development_cases(subset, n_series, seed, series_ids, development_tail)
        return
    if development_tail is not None:
        raise ConfigurationError("development_tail exige development_only=True.")
    bench = load_benchmark(name, subset)
    if not isinstance(bench, BenchmarkDataset):  # pragma: no cover - guaranteed without series_id
        raise ConfigurationError("load_benchmark devolveu uma série; esperado um conjunto.")
    ids = list(bench.series_ids)
    if series_ids is None and n_series > len(ids):
        raise ConfigurationError("Mais séries solicitadas do que disponíveis no benchmark.")
    rng = np.random.default_rng(seed)
    chosen = (
        list(series_ids)
        if series_ids is not None
        else rng.choice(ids, size=min(n_series, len(ids)), replace=False).tolist()
    )
    if len(set(chosen)) != len(chosen) or any(sid not in ids for sid in chosen):
        raise ConfigurationError("IDs de séries duplicados ou ausentes no benchmark.")
    for sid in chosen:
        series = bench.series(str(sid))
        assert series.test is not None
        yield AblationCase(
            str(sid),
            series.full(),
            len(series.test),
            len(series.test),
            seasonal_period=series.descriptor.extra.get("seasonal_period"),
            stratum=f"{name}-{subset} h={len(series.test)}",
            source={**series.descriptor.to_dict(), "sampling_seed": seed},
        )


def _m4_development_cases(
    subset: str,
    n_series: int,
    seed: int,
    series_ids: Sequence[str] | None,
    tail: int | None,
) -> Iterator[AblationCase]:
    """Cost pilot: fetch only official training data and reserve a development tail."""
    if tail is not None and (isinstance(tail, bool) or not isinstance(tail, int) or tail < 1):
        raise ConfigurationError("development_tail deve ser inteiro positivo.")
    provider = M4Provider()
    cache = CacheManager()
    available = provider.list(cache, subset)["series_id"].tolist()
    if series_ids is None and n_series > len(available):
        raise ConfigurationError("Mais séries solicitadas do que disponíveis no benchmark.")
    chosen = (
        list(series_ids)
        if series_ids is not None
        else np.random.default_rng(seed).choice(available, size=n_series, replace=False).tolist()
    )
    if not chosen or len(set(chosen)) != len(chosen) or any(sid not in available for sid in chosen):
        raise ConfigurationError("IDs de séries duplicados ou ausentes no benchmark.")
    for sid in chosen:
        series = provider.load(cache, subset, series_id=sid, with_test=False)
        if isinstance(series, BenchmarkDataset):
            raise ConfigurationError("Esperada uma série de treinamento do M4.")
        original_length = len(series.train)
        values = series.train if tail is None else series.train.iloc[-tail:]
        horizon = series.descriptor.horizon
        if horizon is None or len(values) <= horizon:
            raise DataValidationError("Treino insuficiente para o bloco de desenvolvimento.")
        audit = _development_audit(values)
        source = {
            **series.descriptor.to_dict(),
            "sampling_seed": seed,
            "official_test_accessed": False,
            "evaluation_partition": "development_tail",
            "original_train_length": original_length,
            "development_tail": tail,
            "data_audit": audit,
            "source_files": {
                name: cache.read_manifest("m4", name)
                for name in ("M4-info.csv", f"{subset.title()}-train.csv")
            },
        }
        yield AblationCase(
            str(sid),
            values,
            horizon,
            horizon,
            seasonal_period=series.descriptor.extra.get("seasonal_period"),
            stratum=f"m4-{subset}-development h={horizon}",
            source=source,
        )


def _development_audit(values: pd.Series) -> dict[str, Any]:
    """Actual finite-value and chronological checks before any pilot fitting."""
    numbers = values.to_numpy(dtype=float)
    index = pd.DatetimeIndex(values.index)
    deltas = np.diff(index.to_numpy(dtype="datetime64[ns]").astype(np.int64))
    regular = len(deltas) > 0 and bool(np.all(deltas == deltas[0])) and bool(deltas[0] > 0)
    audit = {
        "status": "READY",
        "observations": len(values),
        "missing": int(values.isna().sum()),
        "nonfinite": int((~np.isfinite(numbers)).sum()),
        "duplicate_timestamps": int(index.duplicated().sum()),
        "chronological": bool(index.is_monotonic_increasing),
        "regular": regular,
        "start": str(index.min()),
        "end": str(index.max()),
        "constant": bool(np.ptp(numbers) == 0),
    }
    if (
        audit["nonfinite"]
        or audit["duplicate_timestamps"]
        or not regular
        or not audit["chronological"]
    ):
        raise DataValidationError(f"Auditoria dos dados de desenvolvimento bloqueada: {audit}.")
    if audit["constant"]:
        audit["status"] = "READY_WITH_WARNINGS"
    return audit


@dataclass(frozen=True)
class _Search:
    """One experiment's search and selection, with its cost."""

    name: str
    experiment: AutoHybridExperiment
    compare: CompareResult
    seconds: float

    @property
    def candidates(self) -> int:
        return self.compare.search.trial_count

    @property
    def units(self) -> int:
        finalists = self.compare.selection.finalists
        return self.compare.search.unit_count + sum(len(r.trials) for r in finalists)

    @property
    def fits(self) -> int:
        finalists = self.compare.selection.finalists
        selected = sum(t.n_fits or 0 for r in finalists for t in r.trials)
        return self.compare.search.fit_count + selected

    @property
    def seconds_search(self) -> float:
        return self.compare.search.elapsed_s

    @property
    def seconds_selection(self) -> float:
        # Validation of the series, selection on V_select and bookkeeping.
        return max(0.0, self.seconds - self.compare.search.elapsed_s)


def _search(
    name: str, common: dict[str, Any], values: pd.Series, include: Sequence[str]
) -> _Search:
    started = time.perf_counter()
    experiment = AutoHybridExperiment(**common).fit(values)
    compare = experiment.compare_models(include=include)
    return _Search(name, experiment, compare, time.perf_counter() - started)


def run_case(
    case: AblationCase,
    seed: int,
    budget: BudgetPreset = "smoke",
    space: dict[str, Any] | None = None,
    outer_folds: int = 2,
    *,
    snapshot_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, object]]:
    """The four procedures on one series and seed; one row per procedure."""

    common: dict[str, Any] = {
        **DEFAULT_SPACE,
        **(space or {}),
        "fh": case.horizon,
        "test_size": case.test_size,
        "session_id": seed,
        "budget": budget,
        "seasonal_period": case.seasonal_period,
    }
    full = _search("full", common, case.values, ALL_FAMILIES)
    if snapshot_callback is not None:
        snapshot_callback(
            json_safe(
                {
                    "phase": "full",
                    "snapshot": full.compare.snapshot,
                    "splits": full.experiment.splits.reset_index().to_dict(orient="records"),
                }
            )
        )
    exp = full.experiment
    metric = full.compare.search.metric
    finalists = list(full.compare.selection.finalists)
    best_base = exp.get_standalone_reference("lowest_select_error")
    initial = exp.get_standalone_reference("initial_champion")
    lowest = min(finalists, key=lambda r: (r.mean(metric), r.candidate_id)).candidate

    simple = _search("simple", common, case.values, SIMPLE_FAMILIES)
    if snapshot_callback is not None:
        snapshot_callback(
            json_safe(
                {
                    "phase": "simple",
                    "snapshot": simple.compare.snapshot,
                    "splits": simple.experiment.splits.reset_index().to_dict(orient="records"),
                }
            )
        )

    started = time.perf_counter()
    nested = exp.nested_cross_validate(
        outer_folds=outer_folds,
        baselines=(),
        include=ALL_FAMILIES,
        standalone_reference="lowest_select_error",
    )
    nested_seconds = time.perf_counter() - started
    if snapshot_callback is not None:
        snapshot_callback(
            json_safe(
                {
                    "phase": "nested",
                    "snapshots": nested.snapshots,
                    "folds": nested.folds.reset_index().to_dict(orient="records"),
                    "cost": nested.cost.reset_index().to_dict(orient="records"),
                }
            )
        )
    means = nested.summary["mean"]
    procedure_score = float(means.get(f"autohybts_{metric}", np.inf))
    reference_score = float(means.get(f"best_standalone_{metric}", np.inf))
    guarded = full.compare.best if procedure_score < reference_score else best_base
    nested_units = int(
        nested.cost.get("search_units", pd.Series(dtype=float)).fillna(0).sum()
        + nested.cost.get("select_units", pd.Series(dtype=float)).fillna(0).sum()
    )

    # V_select results of every finalist of both searches: the validation side of the gap.
    validated = {r.candidate_id: r for r in (*finalists, *simple.compare.selection.finalists)}
    tested: dict[str, tuple[float, float, float]] = {}

    def test_score(
        candidate: CandidateSpec, experiment: AutoHybridExperiment
    ) -> tuple[float, float, float]:
        """External score, refit and external-evaluation seconds; read once per candidate."""

        if candidate.candidate_id not in tested:
            started = time.perf_counter()
            final = experiment.finalize_model(candidate)
            refitted = time.perf_counter()
            score = float(experiment.evaluate_external(final).metrics[metric].iloc[0])
            tested[candidate.candidate_id] = (
                score,
                refitted - started,
                time.perf_counter() - refitted,
            )
        return tested[candidate.candidate_id]

    base_test = test_score(best_base, exp)[0]
    references = {
        f"{name}_test": test_score(exp.create_model(name).candidate, exp)[0] for name in REFERENCES
    }
    choices: dict[str, tuple[CandidateSpec, _Search, bool]] = {
        "lowest_error": (lowest, full, False),
        "incremental": (full.compare.best, full, False),
        "incremental_simple": (simple.compare.best, simple, False),
        "guarded_nested": (guarded, full, True),
    }
    rows: list[dict[str, object]] = []
    for procedure, (candidate, search, with_nested) in choices.items():
        score, refit_seconds, external_seconds = test_score(candidate, search.experiment)
        ledger = search.compare.search.ledger
        pruned = Counter(e.reason for e in ledger if e.status == "PRUNED_BUDGET")
        validation = validated[candidate.candidate_id]
        seconds_nested = nested_seconds if with_nested else 0.0
        rows.append(
            {
                "schema": ABLATION_SCHEMA,
                "series_id": case.series_id,
                "stratum": case.stratum,
                "seed": seed,
                "procedure": procedure,
                "champion": candidate.name,
                "is_hybrid": candidate.strategy != "standalone",
                "complexity": complexity(candidate),
                "test_score": score,
                "best_base": best_base.name,
                "best_base_id": best_base.candidate_id,
                "initial_champion": initial.name,
                "initial_champion_id": initial.candidate_id,
                "standalone_reference_rule": "lowest_select_error",
                "best_base_test": base_test,
                **references,
                "gain_vs_base": (base_test - score) / abs(base_test) if base_test else np.nan,
                "validation_score": validation.mean(metric),
                "gap": score - validation.mean(metric),
                "coverage": validation.coverage(metric),
                # --- cost of the procedure (the external test and the refit excluded)
                "search": search.name,
                "candidates": search.candidates + (nested.total_candidates if with_nested else 0),
                "search_candidates": search.candidates,
                "nested_candidates": nested.total_candidates if with_nested else 0,
                "units": search.units + (nested_units if with_nested else 0),
                "fits": search.fits + (nested.total_fits if with_nested else 0),
                "max_trials_total": search.compare.snapshot["guard"]["max_trials_total"],
                "stop_reason": search.compare.search.stop_reason,
                # proposals cut by each budget limit (the search's own, not the nested CV's)
                "pruned_candidates": sum(pruned.values()),
                "pruned_by": ";".join(f"{k}:{v}" for k, v in sorted(pruned.items())),
                "seconds": search.seconds + seconds_nested,
                "seconds_search": search.seconds_search,
                "seconds_selection": search.seconds_selection,
                "seconds_nested": seconds_nested,
                "seconds_refit": refit_seconds,
                "seconds_external": external_seconds,
            }
        )
    return rows


def run_ablation(
    cases: Iterable[AblationCase],
    seeds: Sequence[int] = (0,),
    budget: BudgetPreset = "smoke",
    space: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
    outer_folds: int = 2,
    *,
    checkpoint: str | Path | None = None,
    resume: bool = False,
    retry_failed: bool = False,
    on_error: Literal["raise", "record"] = "raise",
    configuration: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Run a fixed case/seed plan, optionally checkpointing complete cases atomically.

    ``resume`` requires the same protocol, data hashes, environment and source code.
    Successful cases are never rerun. Interrupted cases resume automatically; failed
    or skipped cases require ``retry_failed=True``. ``on_error="record"`` continues
    after case failures, but configuration and temporal leakage errors always stop.
    The returned table's attrs contain ``protocol``, ``cases`` and study ``coverage``.
    """

    cases = list(cases)
    seeds = tuple(seeds)
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ConfigurationError("Sementes devem ser inteiros.")
    if isinstance(outer_folds, bool) or not isinstance(outer_folds, int) or outer_folds < 1:
        raise ConfigurationError("outer_folds deve ser inteiro positivo.")
    if not cases or not seeds:
        raise ConfigurationError("Informe ao menos uma série e uma semente.")
    if len({case.series_id for case in cases}) != len(cases) or len(set(seeds)) != len(seeds):
        raise ConfigurationError("Séries e sementes não podem conter duplicatas.")
    if on_error not in ("raise", "record"):
        raise ConfigurationError("on_error deve ser 'raise' ou 'record'.")
    if (resume and checkpoint is None) or (retry_failed and not resume):
        raise ConfigurationError("Retomada exige checkpoint; retry_failed exige resume=True.")
    if space and space.get("registry") is not None and checkpoint is not None:
        raise ConfigurationError(
            "Checkpoint não admite registry customizado sem identidade de código."
        )
    protocol = json_safe(ablation_protocol(cases, seeds, budget, space, outer_folds))
    if configuration is not None:
        protocol["configuration"] = json_safe(configuration)
    protocol["source_fingerprint"] = source_fingerprint()
    identity = stable_hash({k: v for k, v in protocol.items() if k != "created_at"})
    protocol["study_id"] = identity
    manager = (
        Checkpoint(checkpoint, identity, protocol, resume) if checkpoint else nullcontext(None)
    )
    with manager as store:
        if store:
            protocol = store.protocol
        states = store.read() if store else {}
        plan = {
            (case.series_id, seed): stable_hash([case.series_id, seed])
            for case in cases
            for seed in seeds
        }
        if set(states) - set(plan.values()):
            raise ConfigurationError("Checkpoint contém casos fora do plano.")
        for (sid, seed), key in plan.items():
            if key not in states:
                states[key] = {
                    "series_id": sid,
                    "seed": seed,
                    "status": "PENDING",
                    "history": [],
                    "rows": [],
                    "snapshots": [],
                    "columns": [],
                }
                if store:
                    store.write(key, states[key])
            state = states[key]
            if (
                state["series_id"] != sid
                or state["seed"] != seed
                or state["status"]
                not in ("PENDING", "RUNNING", "SUCCEEDED", "FAILED", "SKIPPED", "INTERRUPTED")
            ):
                raise ConfigurationError("Identidade ou estado de caso inválido no checkpoint.")
            if state["status"] == "SUCCEEDED" and (
                len(state["rows"]) != len(PROCEDURES)
                or {row["procedure"] for row in state["rows"]} != set(PROCEDURES)
                or any(row["series_id"] != sid or row["seed"] != seed for row in state["rows"])
            ):
                raise ConfigurationError("Resultado concluído incompleto no checkpoint.")
        for case in cases:
            for seed in seeds:
                key = plan[case.series_id, seed]
                state = states[key]
                status = state["status"]
                if status == "SUCCEEDED" or (status in ("FAILED", "SKIPPED") and not retry_failed):
                    continue
                if status == "RUNNING":
                    state["history"][-1].update(
                        status="INTERRUPTED", reason="process exited", seconds=None
                    )
                if state["history"] and state["snapshots"]:
                    state["history"][-1]["snapshots"] = state["snapshots"]
                state.update(status="RUNNING", snapshots=[], rows=[])
                state["history"].append({"status": "RUNNING", "started_at": utc_now()})
                if store:
                    store.write(key, state)
                started = time.perf_counter()

                def collect(
                    snapshot: dict[str, Any],
                    unit_state: dict[str, Any] = state,
                    unit_key: str = key,
                ) -> None:
                    unit_state["snapshots"].append(snapshot)
                    if store:
                        store.write(unit_key, unit_state)

                try:
                    rows = run_case(
                        case, seed, budget, space, outer_folds, snapshot_callback=collect
                    )
                    if {row["procedure"] for row in rows} != set(PROCEDURES) or len(rows) != 4:
                        raise ConfigurationError("Caso não produziu os quatro procedimentos.")
                    state.update(status="SUCCEEDED", rows=json_safe(rows), columns=list(rows[0]))
                except BaseException as error:
                    interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
                    skipped = isinstance(error, UnsupportedCapabilityError)
                    state["status"] = (
                        "INTERRUPTED" if interrupted else "SKIPPED" if skipped else "FAILED"
                    )
                    state["history"][-1]["reason"] = f"{type(error).__name__}: {error}"
                    fatal = isinstance(
                        error, (ConfigurationError, TemporalLeakageError)
                    ) and not isinstance(error, DataValidationError)
                    if (
                        interrupted
                        or fatal
                        or on_error == "raise"
                        or not isinstance(error, Exception)
                    ):
                        raise
                finally:
                    state["history"][-1].update(
                        status=state["status"],
                        finished_at=utc_now(),
                        seconds=time.perf_counter() - started,
                    )
                    if store:
                        store.write(key, state)
                if progress is not None:
                    progress(f"{case.series_id} seed={seed}: {state['status']}")
        ordered = [states[key] for key in plan.values()]
        rows = [row for state in ordered for row in state["rows"]]
        columns = [
            "series_id",
            "seed",
            "procedure",
            "test_score",
            "gain_vs_base",
            "is_hybrid",
            "complexity",
            "gap",
            "coverage",
            "seconds",
            "candidates",
            "units",
        ]
        column_order = next((state["columns"] for state in ordered if state["rows"]), columns)
        table = pd.DataFrame(rows, columns=column_order)
        numeric = {
            "test_score",
            "gain_vs_base",
            "validation_score",
            "best_base_test",
            "naive_test",
            "seasonal_naive_test",
            "gap",
            "coverage",
            "complexity",
            "candidates",
            "search_candidates",
            "nested_candidates",
            "units",
            "fits",
            "max_trials_total",
            "pruned_candidates",
        }
        for column in table.columns:
            if column in numeric or column.startswith("seconds"):
                table[column] = pd.to_numeric(table[column], errors="raise")
        per_procedure = {}
        for procedure in PROCEDURES:
            completed = table.loc[table["procedure"] == procedure]
            valid = int(np.isfinite(completed["test_score"]).sum())
            paired = int(np.isfinite(completed["gain_vs_base"]).sum())
            per_procedure[procedure] = {
                "n_planned": len(plan),
                "n_succeeded": len(completed),
                "n_metric_valid": valid,
                "n_paired": paired,
                "metric_coverage": valid / len(plan),
                "paired_coverage": paired / len(plan),
            }
        counts = Counter(state["status"] for state in states.values())
        table.attrs.update(
            protocol=protocol,
            cases=ordered,
            coverage={
                "n_planned_cases": len(plan),
                "n_planned_procedures": len(plan) * len(PROCEDURES),
                "n_succeeded_cases": counts["SUCCEEDED"],
                "n_failed_cases": counts["FAILED"],
                "n_skipped_cases": counts["SKIPPED"],
                "n_result_rows": len(rows),
                "n_interrupted_cases": counts["INTERRUPTED"],
                "case_coverage": counts["SUCCEEDED"] / len(plan),
                "procedures": per_procedure,
                "n_attempts": sum(len(state["history"]) for state in ordered),
                "known_seconds_total": sum(
                    attempt.get("seconds") or 0.0
                    for state in ordered
                    for attempt in state["history"]
                ),
                "n_unknown_duration_attempts": sum(
                    attempt.get("seconds") is None
                    for state in ordered
                    for attempt in state["history"]
                ),
            },
        )
        return table


def ablation_protocol(
    cases: Sequence[AblationCase],
    seeds: Sequence[int],
    budget: BudgetPreset = "smoke",
    space: dict[str, Any] | None = None,
    outer_folds: int = 2,
) -> dict[str, Any]:
    """Configured settings of a run, to save next to its table.

    Metrics, initial selection policy and budget overrides follow the experiment's
    configuration. Per-series resolved folds and family order remain in the comparison
    snapshots; this metadata does not perform a search or read the external test.

    Local changes not committed cannot be detected without running ``git``; record them
    by hand when the working tree is not clean.
    """

    if budget not in PRESETS:
        raise ConfigurationError(f"Preset desconhecido {budget!r}; opções: {sorted(PRESETS)}.")
    settings = {**DEFAULT_SPACE, **(space or {}), "budget": budget}
    experiment = AutoHybridExperiment(**settings)
    metrics = [metric.name for metric in resolve_metrics(experiment.metrics)]
    guard = experiment._guard().describe()
    if guard["max_wall_time_s"] is None:
        del guard["max_wall_time_s"]
    return {
        "schema": ABLATION_SCHEMA,
        "created_at": utc_now(),
        "environment": environment(),
        "procedures": list(PROCEDURES),
        "families": {"full": list(ALL_FAMILIES), "simple": list(SIMPLE_FAMILIES)},
        "references": list(REFERENCES),
        "standalone_reference_rule": "lowest_select_error",
        "budget": {"preset": budget, "guard": guard},
        "metrics": metrics,
        "ranking_metric": metrics[0],
        "selection_policy": experiment._selection_policy().describe(),
        "cache": "sem cache de tentativas entre buscas; ajustes ARIMA memoizados por vetor de "
        "treino (só o tempo muda, nunca o resultado)",
        "space": {k: _jsonable(v) for k, v in {**DEFAULT_SPACE, **(space or {})}.items()},
        "outer_folds": outer_folds,
        "seeds": list(seeds),
        "series": [
            {
                "series_id": case.series_id,
                "length": len(case.values),
                "horizon": case.horizon,
                "test_size": case.test_size,
                "seasonal_period": case.seasonal_period,
                "stratum": case.stratum,
                "data_fingerprint": series_fingerprint(case.values),
                "source": case.source,
            }
            for case in cases
        ],
    }


def _jsonable(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def _finite_gains(gains: pd.Series) -> pd.Series:
    return gains.where(np.isfinite(gains))


def _gain_share(gains: pd.Series, outcome: str) -> float:
    valid = gains.dropna()
    if valid.empty:
        return float("nan")
    outcomes = {"better": valid > 0, "tie": valid == 0, "worse": valid < 0}
    return float(outcomes[outcome].mean())


def ablation_report(table: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Per procedure: gain vs best base, wins/ties/losses, hybrids, gap, rank, seed spread, cost.

    Gain statistics use finite paired gains only. ``n_rows`` counts supplied case/seed
    rows, not the study's planned cases; ``n_paired`` and ``paired_coverage`` identify
    the denominator of wins, ties and losses. Without valid gains their shares are NaN.

    ``by`` (e.g. ``"stratum"``) adds an outer level: one block per group.
    """

    if by is not None:
        if by not in table.columns:
            raise ConfigurationError(f"Coluna {by!r} ausente na tabela da ablação.")
        blocks = {key: ablation_report(group) for key, group in table.groupby(by, sort=True)}
        return pd.concat(blocks, names=[by])
    table = table.copy()
    table["gain_vs_base"] = _finite_gains(table["gain_vs_base"])
    table["test_score"] = table["test_score"].where(np.isfinite(table["test_score"]))
    table["rank"] = table.groupby(["series_id", "seed"])["test_score"].rank(method="average")
    per_seed = table.groupby(["procedure", "seed"])["gain_vs_base"].median()
    report = (
        table.groupby("procedure")
        .agg(
            median_gain_vs_base=("gain_vs_base", "median"),
            beats_base_share=("gain_vs_base", lambda g: _gain_share(g, "better")),
            ties_base_share=("gain_vs_base", lambda g: _gain_share(g, "tie")),
            worse_than_base_share=("gain_vs_base", lambda g: _gain_share(g, "worse")),
            n_rows=("gain_vs_base", "size"),
            n_paired=("gain_vs_base", "count"),
            hybrid_share=("is_hybrid", "mean"),
            mean_complexity=("complexity", "mean"),
            mean_gap=("gap", "mean"),
            mean_rank=("rank", "mean"),
            coverage=("coverage", "mean"),
            mean_seconds=("seconds", "mean"),
            candidates=("candidates", "mean"),
            units=("units", "mean"),
        )
        .join(per_seed.groupby("procedure").std().rename("seed_std_of_median_gain"))
    )
    report["n_invalid_gain"] = report["n_rows"] - report["n_paired"]
    report["paired_coverage"] = report["n_paired"] / report["n_rows"]
    return report.loc[[p for p in PROCEDURES if p in report.index]]


def ablation_uncertainty(
    table: pd.DataFrame, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> pd.DataFrame:
    """Median gain vs the best base per procedure, with a series-level bootstrap interval.

    Seeds of one series are not independent: each series' gain is first summarized (median
    over its seeds), then series are resampled with replacement. Also reports the share
    of series where the procedure is better/worse than the base. Non-finite gains are
    excluded before aggregation; series without any finite gain remain in ``n_series``
    but not in ``n_paired_series`` or the bootstrap. Their absence is reported explicitly.
    """

    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ConfigurationError("n_boot deve ser inteiro positivo.")
    if not 0 < level < 1:
        raise ConfigurationError("level deve estar em (0, 1).")
    table = table.copy()
    table["gain_vs_base"] = _finite_gains(table["gain_vs_base"])
    per_series = table.groupby(["procedure", "series_id"])["gain_vs_base"].median()
    rng = np.random.default_rng(seed)
    alpha = (1 - level) / 2
    rows = []
    for procedure in [p for p in PROCEDURES if p in per_series.index.get_level_values(0)]:
        series_gains = per_series.loc[procedure]
        gains = series_gains.dropna().to_numpy(dtype=float)
        if gains.size:
            draws = gains[rng.integers(0, gains.size, (n_boot, gains.size))]
            medians = np.median(draws, axis=1)
            median = float(np.median(gains))
            ci_low = float(np.quantile(medians, alpha))
            ci_high = float(np.quantile(medians, 1 - alpha))
        else:
            median = ci_low = ci_high = float("nan")
        rows.append(
            {
                "procedure": procedure,
                "n_series": int(series_gains.size),
                "n_paired_series": int(gains.size),
                "n_invalid_series": int(series_gains.size - gains.size),
                "paired_series_coverage": float(gains.size / series_gains.size),
                "median_gain_vs_base": median,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "better_share": float(np.mean(gains > 0)) if gains.size else float("nan"),
                "worse_share": float(np.mean(gains < 0)) if gains.size else float("nan"),
            }
        )
    columns = [
        "procedure",
        "n_series",
        "n_paired_series",
        "n_invalid_series",
        "paired_series_coverage",
        "median_gain_vs_base",
        "ci_low",
        "ci_high",
        "better_share",
        "worse_share",
    ]
    return pd.DataFrame(rows, columns=columns).set_index("procedure")


CONFIG_SCHEMA: Final = "autohybts-ablation-config/1"


def _study_config(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "schema",
        "preset",
        "dataset",
        "series",
        "series_ids",
        "sampling_seed",
        "seeds",
        "budget",
        "outer_folds",
        "space",
        "seasonal_period",
        "development_only",
        "development_tail",
    }
    if set(raw) - allowed:
        raise ConfigurationError(
            f"Campos desconhecidos no protocolo: {sorted(set(raw) - allowed)}."
        )
    if raw.get("schema", CONFIG_SCHEMA) != CONFIG_SCHEMA:
        raise ConfigurationError("Esquema de configuração de ablação desconhecido.")
    preset = raw.get("preset", "demo")
    if preset not in ("demo", "research"):
        raise ConfigurationError("preset deve ser 'demo' ou 'research'.")
    space = raw.get("space", {})
    if not isinstance(space, dict):
        raise ConfigurationError("space deve ser um objeto JSON.")
    reserved = {
        "fh",
        "test_size",
        "session_id",
        "budget",
        "seasonal_period",
        "registry",
        "progress",
    }
    accepted = set(inspect.signature(AutoHybridExperiment).parameters) - reserved
    if set(space) - accepted:
        raise ConfigurationError(f"Parâmetros inválidos em space: {sorted(set(space) - accepted)}.")
    AutoHybridExperiment(**space)._selection_policy()
    config = {
        "schema": CONFIG_SCHEMA,
        "preset": preset,
        "dataset": "synthetic",
        "series": 6,
        "sampling_seed": 0,
        "seeds": [0],
        "outer_folds": 2,
        "budget": "small" if preset == "research" else "smoke",
        **raw,
        "space": {**DEFAULT_SPACE, "fold": "auto" if preset == "research" else 2, **space},
    }
    fold = config["space"]["fold"]
    if not (fold == "auto" or (isinstance(fold, int) and not isinstance(fold, bool) and fold > 0)):
        raise ConfigurationError("fold deve ser auto ou inteiro positivo.")
    if config["dataset"] not in ("synthetic", "m4-daily", "m3-monthly"):
        raise ConfigurationError("dataset deve ser synthetic, m4-daily ou m3-monthly.")
    if "development_only" in config and not isinstance(config["development_only"], bool):
        raise ConfigurationError("development_only deve ser booleano.")
    if config.get("development_only") and config["dataset"] != "m4-daily":
        raise ConfigurationError("O piloto development_only requer dataset m4-daily.")
    if "development_tail" in config:
        tail = config["development_tail"]
        if (
            not config.get("development_only")
            or isinstance(tail, bool)
            or not isinstance(tail, int)
            or tail < 1
        ):
            raise ConfigurationError("development_tail exige development_only e inteiro positivo.")
    for name in ("series", "outer_folds"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigurationError(f"{name} deve ser inteiro positivo.")
    seeds = config["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ConfigurationError("seeds deve conter inteiros distintos.")
    sample_seed = config["sampling_seed"]
    if isinstance(sample_seed, bool) or not isinstance(sample_seed, int) or sample_seed < 0:
        raise ConfigurationError("sampling_seed deve ser inteiro não negativo.")
    ids = config.get("series_ids")
    if ids is not None:
        if config["dataset"] == "synthetic" or "series" in raw:
            raise ConfigurationError(
                "series_ids requer benchmark e não pode ser combinado com series."
            )
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(sid, str) for sid in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ConfigurationError("series_ids deve conter IDs distintos.")
        config["series"] = len(ids)
    if "seasonal_period" in config:
        period = config["seasonal_period"]
        if period is not None and (
            isinstance(period, bool) or not isinstance(period, int) or period < 1
        ):
            raise ConfigurationError("seasonal_period deve ser positivo ou null para inferir.")
    ablation_protocol([], seeds, config["budget"], config["space"], config["outer_folds"])
    return config


def _configured_cases(config: dict[str, Any]) -> list[AblationCase]:
    if config["dataset"] == "synthetic":
        cases = list(synthetic_cases(config["series"]))
    else:
        name, subset = config["dataset"].split("-")
        cases = list(
            benchmark_cases(
                name,
                subset,
                config["series"],
                config["sampling_seed"],
                series_ids=config.get("series_ids"),
                development_only=config.get("development_only", False),
                development_tail=config.get("development_tail"),
            )
        )
    if "seasonal_period" in config:
        cases = [replace(case, seasonal_period=config["seasonal_period"]) for case in cases]
    return cases


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ablação do procedimento de seleção.")
    parser.add_argument("--config", type=Path, help="Protocolo JSON versionado")
    parser.add_argument("--preset", choices=("demo", "research"))
    parser.add_argument("--dataset", help="synthetic, m4-daily, m3-monthly")
    parser.add_argument("--series", type=int)
    parser.add_argument("--series-ids", nargs="+")
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--budget")
    parser.add_argument("--outer-folds", type=int)
    parser.add_argument("--fold", help="auto ou número de origens por camada")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--on-error", choices=("raise", "record"), default="raise")
    parser.add_argument(
        "--out",
        default=None,
        help="CSV com uma linha por série, semente e procedimento; o protocolo vai em "
        "<out>.protocol.json",
    )
    args = parser.parse_args(argv)
    if (args.resume and args.checkpoint is None) or (args.retry_failed and not args.resume):
        raise ConfigurationError("--resume exige --checkpoint; --retry-failed exige --resume.")
    names = (
        "preset",
        "dataset",
        "series",
        "series_ids",
        "sampling_seed",
        "seeds",
        "budget",
        "outer_folds",
    )
    raw = {name: getattr(args, name) for name in names if getattr(args, name) is not None}
    if args.config is not None:
        if raw or args.fold is not None:
            raise ConfigurationError("Não combine --config com alterações de protocolo na CLI.")
        raw = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConfigurationError("O protocolo deve ser um objeto JSON.")
    else:
        if args.fold is not None:
            try:
                fold = "auto" if args.fold == "auto" else int(args.fold)
            except ValueError as error:
                raise ConfigurationError("--fold deve ser auto ou inteiro.") from error
            raw["space"] = {"fold": fold}
    config = _study_config(raw)
    cases = _configured_cases(config)
    if config.get("development_only"):
        print(
            "Piloto de desenvolvimento: teste oficial não carregado; "
            "métricas não são benchmark oficial.",
            flush=True,
        )
        audits = pd.DataFrame(
            [{"series_id": case.series_id, **case.source["data_audit"]} for case in cases]
        )
        print(audits.to_string(index=False), flush=True)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            audits.to_csv(f"{args.out}.audit.csv", index=False)
    table = run_ablation(
        cases,
        config["seeds"],
        config["budget"],
        config["space"],
        progress=lambda m: print(m, flush=True),
        outer_folds=config["outer_folds"],
        checkpoint=args.checkpoint,
        resume=args.resume,
        retry_failed=args.retry_failed,
        on_error=args.on_error,
        configuration=config,
    )
    if args.out:
        table.to_csv(args.out, index=False)
        protocol = {
            **table.attrs["protocol"],
            "dataset": config["dataset"],
            "configuration": config,
            "resolved_cases": table.attrs["cases"],
            "coverage": table.attrs["coverage"],
        }
        Path(f"{args.out}.protocol.json").write_text(
            json.dumps(json_safe(protocol), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
    print(json.dumps(table.attrs["coverage"], ensure_ascii=False))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        if not table.empty:
            print(ablation_report(table).round(4))


if __name__ == "__main__":
    main()
