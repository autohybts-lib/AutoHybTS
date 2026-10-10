"""Ablation of the selection procedure (architecture §30.5).

Four procedures, same search budget and the same external origins (each series' sealed
test):

  (i)   lowest error: the finalist with the lowest mean error on V_select;
  (ii)  incremental: the guarded incremental chain with the protocol thresholds;
  (iii) incremental + complexity limit: as (ii), only structures with at most two
        fitted models (standalone and additive residual hybrids);
  (iv)  guarded + nested: as (ii), but if nested cross-validation on the development
        history says the procedure does not beat the best standalone model, the best
        standalone model is kept.

Reported per procedure: gain against the best standalone model on the test, share of
series where a hybrid was chosen and where the champion beat the best standalone model
(on the test), validation–test gap, mean rank, spread across seeds,
coverage, time and number of candidates. The external test is used only to report:
do not pick the "winning" procedure on it.

Usage::

    python benchmarks/ablation.py --dataset synthetic --series 6 --seeds 0 1
    python benchmarks/ablation.py --dataset m4-daily --series 20 --seeds 0 1 2 --out ablation.csv
"""

from __future__ import annotations

import argparse
import time
import warnings
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from autohybts import AutoHybridExperiment
from autohybts.datasets import get_data, load_benchmark
from autohybts.domain.specs import CandidateSpec

PROCEDURES = ("lowest_error", "incremental", "incremental_simple", "guarded_nested")
ALL_FAMILIES = ("standalone", "residual", "multistage", "decomposition", "combination")


@dataclass
class Case:
    series_id: str
    values: pd.Series  # development history + external test at the end
    horizon: int
    test_size: int
    seasonal_period: int | None


def cases(dataset: str, n_series: int) -> Iterator[Case]:
    if dataset == "synthetic":
        for seed in range(n_series):
            phi = [0.0, 0.5, 0.8][seed % 3]
            y = get_data("synthetic_daily", n=400, seed=100 + seed, phi=phi)
            yield Case(f"syn{seed}-phi{phi}", y, 14, 14, 7)
        return
    name, subset = dataset.split("-")
    bench = load_benchmark(name, subset)
    ids = list(bench.series_ids)  # type: ignore[union-attr]
    rng = np.random.default_rng(0)
    for sid in rng.choice(ids, size=min(n_series, len(ids)), replace=False):
        series = bench.series(str(sid))  # type: ignore[union-attr]
        assert series.test is not None
        yield Case(str(sid), series.full(), len(series.test), len(series.test), None)


def run_case(case: Case, seed: int, budget: str) -> list[dict[str, object]]:
    common = {
        "fh": case.horizon,
        "test_size": case.test_size,
        "session_id": seed,
        "budget": budget,
        "fold": 2,
        "seasonal_period": case.seasonal_period,
        "base_models": ("naive", "seasonal_naive", "linear", "knn", "mlp"),
        "correctors": ("linear", "knn"),
        "residual_windows": (7,),
        "alphas": (1.0,),
    }
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    exp = AutoHybridExperiment(**common).fit(case.values)  # type: ignore[arg-type]
    full = exp.compare_models(include=ALL_FAMILIES)
    search_seconds = time.perf_counter() - started
    finalists = {r.candidate.candidate_id: r for r in full.selection.finalists}
    standalone = [r for r in full.selection.finalists if r.candidate.strategy == "standalone"]
    metric = full.search.metric
    best_base = min(standalone, key=lambda r: (r.mean(metric), r.candidate_id)).candidate
    lowest = min(finalists.values(), key=lambda r: (r.mean(metric), r.candidate_id)).candidate

    started = time.perf_counter()
    simple_exp = AutoHybridExperiment(**common).fit(case.values)  # type: ignore[arg-type]
    simple = simple_exp.compare_models(include=("standalone", "residual"))
    simple_best = simple.best  # standalone and additive residual only: complexity <= 2
    simple_seconds = time.perf_counter() - started

    started = time.perf_counter()
    nested = exp.nested_cross_validate(outer_folds=2, baselines=(best_base.base.name,))
    nested_seconds = time.perf_counter() - started
    means = nested.summary["mean"]
    procedure_score = float(means.get(f"autohybts_{metric}", np.inf))
    baseline_score = float(means.get(f"{best_base.base.name}_{metric}", np.inf))
    guarded = full.best if procedure_score < baseline_score else best_base

    # V_select results of every finalist, from both searches (validation side of the gap).
    validated = {
        r.candidate.candidate_id: r
        for r in (*full.selection.finalists, *simple.selection.finalists)
    }
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
    choices = {
        "lowest_error": (lowest, exp, search_seconds, full.search.trial_count),
        "incremental": (full.best, exp, search_seconds, full.search.trial_count),
        "incremental_simple": (simple_best, simple_exp, simple_seconds, simple.search.trial_count),
        "guarded_nested": (guarded, exp, search_seconds + nested_seconds, full.search.trial_count),
    }
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


def report(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["rank"] = table.groupby(["series_id", "seed"])["test_score"].rank(method="average")
    per_seed = table.groupby(["procedure", "seed"])["gain_vs_base"].median()
    return (
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
        .loc[list(PROCEDURES)]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--dataset", default="synthetic", help="synthetic, m4-daily, m3-monthly ..."
    )
    parser.add_argument("--series", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--budget", default="smoke")
    parser.add_argument("--out", default=None, help="CSV with one row per series, seed, procedure")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    rows = []
    for case in cases(args.dataset, args.series):
        for seed in args.seeds:
            rows.extend(run_case(case, seed, args.budget))
            print(f"{case.series_id} seed={seed} ok", flush=True)
    table = pd.DataFrame(rows)
    if args.out:
        table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(report(table).round(4))


if __name__ == "__main__":
    main()
