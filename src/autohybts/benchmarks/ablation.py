"""Ablation of the selection procedure (architecture §30.5).

Four procedures, the same search budget and the same external origins (each series'
sealed test):

* ``lowest_error``: the finalist with the lowest mean error on ``V_select``;
* ``incremental``: the guarded incremental chain with the protocol thresholds, all
  families;
* ``incremental_simple``: as above, only structures with at most two fitted models
  (standalone and additive residual hybrids);
* ``guarded_nested``: as ``incremental``, but if nested cross-validation on the
  development history says the procedure does not beat the best standalone model, that
  model is kept.

``ablation_report`` gives, per procedure: median gain against the best standalone model
on the test, share of series where a hybrid was chosen and where the champion beat the
best standalone model, validation–test gap, mean rank, spread of the median gain across
seeds, coverage, time and number of candidates. The external test only reports: never
pick the "winning" procedure on it.

From Python::

    from autohybts.benchmarks import ablation_report, run_ablation, synthetic_cases

    table = run_ablation(synthetic_cases(12), seeds=(0, 1, 2))
    ablation_report(table)

From the command line::

    python -m autohybts.benchmarks.ablation --dataset synthetic --series 12 --seeds 0 1 2
    python -m autohybts.benchmarks.ablation --dataset m4-daily --series 20 --out ablation.csv
"""

from __future__ import annotations

import argparse
import time
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from autohybts.datasets import get_data, load_benchmark
from autohybts.datasets.descriptor import BenchmarkDataset
from autohybts.domain.specs import CandidateSpec
from autohybts.exceptions import ConfigurationError
from autohybts.experiment import AutoHybridExperiment
from autohybts.search.budgets import BudgetPreset

__all__ = [
    "PROCEDURES",
    "AblationCase",
    "ablation_report",
    "benchmark_cases",
    "run_ablation",
    "run_case",
    "synthetic_cases",
]

PROCEDURES: Final = ("lowest_error", "incremental", "incremental_simple", "guarded_nested")
ALL_FAMILIES: Final = ("standalone", "residual", "multistage", "decomposition", "combination")
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


def synthetic_cases(n_series: int, n: int = 400, horizon: int = 14) -> Iterator[AblationCase]:
    """Daily series whose residual predictability cycles through AR(0), AR(0.5), AR(0.8)."""

    for index in range(n_series):
        phi = (0.0, 0.5, 0.8)[index % 3]
        y = get_data("synthetic_daily", n=n, seed=100 + index, phi=phi)
        yield AblationCase(f"syn{index}-phi{phi}", y, horizon, horizon, 7)


def benchmark_cases(name: str, subset: str, n_series: int, seed: int = 0) -> Iterator[AblationCase]:
    """A random sample of an official benchmark (downloads on first use)."""

    bench = load_benchmark(name, subset)
    if not isinstance(bench, BenchmarkDataset):  # pragma: no cover - guaranteed without series_id
        raise ConfigurationError("load_benchmark devolveu uma série; esperado um conjunto.")
    ids = list(bench.series_ids)
    rng = np.random.default_rng(seed)
    for sid in rng.choice(ids, size=min(n_series, len(ids)), replace=False):
        series = bench.series(str(sid))
        assert series.test is not None
        yield AblationCase(str(sid), series.full(), len(series.test), len(series.test))


def run_case(
    case: AblationCase,
    seed: int,
    budget: BudgetPreset = "smoke",
    space: dict[str, Any] | None = None,
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
    started = time.perf_counter()
    exp = AutoHybridExperiment(**common).fit(case.values)
    full = exp.compare_models(include=ALL_FAMILIES)
    search_seconds = time.perf_counter() - started
    metric = full.search.metric
    finalists = list(full.selection.finalists)
    standalone = [r for r in finalists if r.candidate.strategy == "standalone"]
    best_base = min(standalone, key=lambda r: (r.mean(metric), r.candidate_id)).candidate
    lowest = min(finalists, key=lambda r: (r.mean(metric), r.candidate_id)).candidate

    started = time.perf_counter()
    simple_exp = AutoHybridExperiment(**common).fit(case.values)
    simple = simple_exp.compare_models(include=("standalone", "residual"))
    simple_seconds = time.perf_counter() - started

    started = time.perf_counter()
    nested = exp.nested_cross_validate(outer_folds=2, baselines=(best_base.base.name,))
    nested_seconds = time.perf_counter() - started
    means = nested.summary["mean"]
    procedure_score = float(means.get(f"autohybts_{metric}", np.inf))
    baseline_score = float(means.get(f"{best_base.base.name}_{metric}", np.inf))
    guarded = full.best if procedure_score < baseline_score else best_base

    # V_select results of every finalist of both searches: the validation side of the gap.
    validated = {r.candidate_id: r for r in (*finalists, *simple.selection.finalists)}
    tested: dict[str, float] = {}

    def test_score(candidate: CandidateSpec, experiment: AutoHybridExperiment) -> float:
        """External score, one evaluation per candidate (the sealed test is read once)."""

        if candidate.candidate_id not in tested:
            final = experiment.finalize_model(candidate)
            tested[candidate.candidate_id] = float(
                experiment.evaluate_external(final).metrics[metric].iloc[0]
            )
        return tested[candidate.candidate_id]

    base_test = test_score(best_base, exp)
    choices: dict[str, tuple[CandidateSpec, AutoHybridExperiment, float, int]] = {
        "lowest_error": (lowest, exp, search_seconds, full.search.trial_count),
        "incremental": (full.best, exp, search_seconds, full.search.trial_count),
        "incremental_simple": (simple.best, simple_exp, simple_seconds, simple.search.trial_count),
        "guarded_nested": (guarded, exp, search_seconds + nested_seconds, full.search.trial_count),
    }
    rows: list[dict[str, object]] = []
    for procedure, (candidate, experiment, seconds, trials) in choices.items():
        score = test_score(candidate, experiment)
        validation = validated[candidate.candidate_id]
        rows.append(
            {
                "series_id": case.series_id,
                "seed": seed,
                "procedure": procedure,
                "champion": candidate.name,
                "is_hybrid": candidate.strategy != "standalone",
                "test_score": score,
                "best_base_test": base_test,
                "gain_vs_base": (base_test - score) / abs(base_test) if base_test else np.nan,
                "validation_score": validation.mean(metric),
                "gap": score - validation.mean(metric),
                "coverage": validation.coverage(metric),
                "seconds": seconds,
                "candidates": trials,
            }
        )
    return rows


def run_ablation(
    cases: Iterable[AblationCase],
    seeds: Sequence[int] = (0,),
    budget: BudgetPreset = "smoke",
    space: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Every case × seed × procedure, one row each (see ``run_case``)."""

    rows: list[dict[str, object]] = []
    for case in cases:
        for seed in seeds:
            rows.extend(run_case(case, seed, budget, space))
            if progress is not None:
                progress(f"{case.series_id} seed={seed}")
    return pd.DataFrame(rows)


def ablation_report(table: pd.DataFrame) -> pd.DataFrame:
    """Per procedure: gain vs best base, hybrid share, gap, rank, seed spread, cost."""

    table = table.copy()
    table["rank"] = table.groupby(["series_id", "seed"])["test_score"].rank(method="average")
    per_seed = table.groupby(["procedure", "seed"])["gain_vs_base"].median()
    report = (
        table.groupby("procedure")
        .agg(
            median_gain_vs_base=("gain_vs_base", "median"),
            hybrid_share=("is_hybrid", "mean"),
            beats_base_share=("gain_vs_base", lambda g: float((g > 0).mean())),
            mean_gap=("gap", "mean"),
            mean_rank=("rank", "mean"),
            coverage=("coverage", "mean"),
            mean_seconds=("seconds", "mean"),
            candidates=("candidates", "mean"),
        )
        .join(per_seed.groupby("procedure").std().rename("seed_std_of_median_gain"))
    )
    return report.loc[[p for p in PROCEDURES if p in report.index]]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ablação do procedimento de seleção (§30.5).")
    parser.add_argument("--dataset", default="synthetic", help="synthetic, m4-daily, m3-monthly")
    parser.add_argument("--series", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--budget", default="smoke")
    parser.add_argument(
        "--out", default=None, help="CSV com uma linha por série, semente e procedimento"
    )
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    if args.dataset == "synthetic":
        cases: Iterator[AblationCase] = synthetic_cases(args.series)
    else:
        name, subset = args.dataset.split("-")
        cases = benchmark_cases(name, subset, args.series)
    table = run_ablation(cases, args.seeds, args.budget, progress=lambda m: print(m, flush=True))
    if args.out:
        table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(ablation_report(table).round(4))


if __name__ == "__main__":
    main()
