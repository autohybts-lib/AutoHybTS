"""Ablation of the selection procedure (architecture §30.5).

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
import json
import time
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from autohybts.datasets import get_data, load_benchmark
from autohybts.datasets.descriptor import BenchmarkDataset
from autohybts.domain.specs import CandidateSpec
from autohybts.evaluation.temporal import DEFAULT_METRICS
from autohybts.exceptions import ConfigurationError
from autohybts.experiment import AutoHybridExperiment
from autohybts.persistence.manifest import environment, utc_now
from autohybts.reporting.strategies import complexity
from autohybts.results import CompareResult
from autohybts.search.budgets import PRESETS, BudgetPreset
from autohybts.selection.policy import SelectionPolicy

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
ABLATION_SCHEMA: Final = "ablation/2"
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


def synthetic_cases(n_series: int, n: int = 400, horizon: int = 14) -> Iterator[AblationCase]:
    """Daily series whose residual predictability cycles through AR(0), AR(0.5), AR(0.8)."""

    for index in range(n_series):
        phi = (0.0, 0.5, 0.8)[index % 3]
        y = get_data("synthetic_daily", n=n, seed=100 + index, phi=phi)
        yield AblationCase(f"syn{index}-phi{phi}", y, horizon, horizon, 7, f"phi={phi}")


def benchmark_cases(name: str, subset: str, n_series: int, seed: int = 0) -> Iterator[AblationCase]:
    """A random sample of an official benchmark (downloads on first use).

    The sample is reproducible: ``seed`` draws ``n_series`` ids without replacement from
    the benchmark's id list; the official test of each series is its sealed test.
    """

    bench = load_benchmark(name, subset)
    if not isinstance(bench, BenchmarkDataset):  # pragma: no cover - guaranteed without series_id
        raise ConfigurationError("load_benchmark devolveu uma série; esperado um conjunto.")
    ids = list(bench.series_ids)
    rng = np.random.default_rng(seed)
    for sid in rng.choice(ids, size=min(n_series, len(ids)), replace=False):
        series = bench.series(str(sid))
        assert series.test is not None
        yield AblationCase(
            str(sid),
            series.full(),
            len(series.test),
            len(series.test),
            stratum=f"{name}-{subset} h={len(series.test)}",
        )


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
    exp = full.experiment
    metric = full.compare.search.metric
    finalists = list(full.compare.selection.finalists)
    standalone = [r for r in finalists if r.candidate.strategy == "standalone"]
    best_base = min(standalone, key=lambda r: (r.mean(metric), r.candidate_id)).candidate
    lowest = min(finalists, key=lambda r: (r.mean(metric), r.candidate_id)).candidate

    simple = _search("simple", common, case.values, SIMPLE_FAMILIES)

    started = time.perf_counter()
    nested = exp.nested_cross_validate(outer_folds=outer_folds, baselines=(), include=ALL_FAMILIES)
    nested_seconds = time.perf_counter() - started
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
) -> pd.DataFrame:
    """Every case × seed × procedure, one row each (see ``run_case``)."""

    rows: list[dict[str, object]] = []
    for case in cases:
        for seed in seeds:
            rows.extend(run_case(case, seed, budget, space, outer_folds))
            if progress is not None:
                progress(f"{case.series_id} seed={seed}")
    return pd.DataFrame(rows)


def ablation_protocol(
    cases: Sequence[AblationCase],
    seeds: Sequence[int],
    budget: BudgetPreset = "smoke",
    space: dict[str, Any] | None = None,
    outer_folds: int = 2,
) -> dict[str, Any]:
    """Resolved settings of a run, to save next to its table (§27).

    Local changes not committed cannot be detected without running ``git``; record them
    by hand when the working tree is not clean.
    """

    if budget not in PRESETS:
        raise ConfigurationError(f"Preset desconhecido {budget!r}; opções: {sorted(PRESETS)}.")
    return {
        "schema": ABLATION_SCHEMA,
        "created_at": utc_now(),
        "environment": environment(),
        "procedures": list(PROCEDURES),
        "families": {"full": list(ALL_FAMILIES), "simple": list(SIMPLE_FAMILIES)},
        "references": list(REFERENCES),
        "budget": {"preset": budget, "guard": dict(PRESETS[budget]["guard"])},
        "metrics": list(DEFAULT_METRICS),
        "ranking_metric": DEFAULT_METRICS[0],
        "selection_policy": SelectionPolicy().describe(),
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
            }
            for case in cases
        ],
    }


def _jsonable(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def ablation_report(table: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Per procedure: gain vs best base, wins/ties/losses, hybrids, gap, rank, seed spread, cost.

    ``by`` (e.g. ``"stratum"``) adds an outer level: one block per group.
    """

    if by is not None:
        if by not in table.columns:
            raise ConfigurationError(f"Coluna {by!r} ausente na tabela da ablação.")
        blocks = {key: ablation_report(group) for key, group in table.groupby(by, sort=True)}
        return pd.concat(blocks, names=[by])
    table = table.copy()
    table["rank"] = table.groupby(["series_id", "seed"])["test_score"].rank(method="average")
    per_seed = table.groupby(["procedure", "seed"])["gain_vs_base"].median()
    report = (
        table.groupby("procedure")
        .agg(
            median_gain_vs_base=("gain_vs_base", "median"),
            beats_base_share=("gain_vs_base", lambda g: float((g > 0).mean())),
            ties_base_share=("gain_vs_base", lambda g: float((g == 0).mean())),
            worse_than_base_share=("gain_vs_base", lambda g: float((g < 0).mean())),
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
    return report.loc[[p for p in PROCEDURES if p in report.index]]


def ablation_uncertainty(
    table: pd.DataFrame, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> pd.DataFrame:
    """Median gain vs the best base per procedure, with a series-level bootstrap interval.

    Seeds of one series are not independent: each series' gain is first summarized (median
    over its seeds), then series are resampled with replacement. Also reports the share
    of series where the procedure is better/worse than the base, with the same resampling.
    """

    per_series = table.groupby(["procedure", "series_id"])["gain_vs_base"].median()
    rng = np.random.default_rng(seed)
    alpha = (1 - level) / 2
    rows = []
    for procedure in [p for p in PROCEDURES if p in per_series.index.get_level_values(0)]:
        gains = per_series.loc[procedure].to_numpy(dtype=float)
        draws = gains[rng.integers(0, gains.size, (n_boot, gains.size))]
        medians = np.median(draws, axis=1)
        rows.append(
            {
                "procedure": procedure,
                "n_series": int(gains.size),
                "median_gain_vs_base": float(np.median(gains)),
                "ci_low": float(np.quantile(medians, alpha)),
                "ci_high": float(np.quantile(medians, 1 - alpha)),
                "better_share": float(np.mean(gains > 0)),
                "worse_share": float(np.mean(gains < 0)),
            }
        )
    return pd.DataFrame(rows).set_index("procedure")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ablação do procedimento de seleção (§30.5).")
    parser.add_argument("--dataset", default="synthetic", help="synthetic, m4-daily, m3-monthly")
    parser.add_argument("--series", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--budget", default="smoke")
    parser.add_argument("--outer-folds", type=int, default=2)
    parser.add_argument(
        "--out",
        default=None,
        help="CSV com uma linha por série, semente e procedimento; o protocolo vai em "
        "<out>.protocol.json",
    )
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    if args.dataset == "synthetic":
        cases = list(synthetic_cases(args.series))
    else:
        name, subset = args.dataset.split("-")
        cases = list(benchmark_cases(name, subset, args.series))
    table = run_ablation(
        cases,
        args.seeds,
        args.budget,
        progress=lambda m: print(m, flush=True),
        outer_folds=args.outer_folds,
    )
    if args.out:
        table.to_csv(args.out, index=False)
        protocol = ablation_protocol(cases, args.seeds, args.budget, outer_folds=args.outer_folds)
        protocol["dataset"] = args.dataset
        Path(f"{args.out}.protocol.json").write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(ablation_report(table).round(4))


if __name__ == "__main__":
    main()
