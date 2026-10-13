"""Calibration of the acceptance rule on controls with a known answer (plan M02, §30.2).

Two controls, both daily, with the sealed test held out and never read:

* ``white_noise`` (negative control): level plus white noise. Nothing beyond the mean is
  predictable, so a hybrid champion is a **false acceptance** of complexity.
* ``ar_residual`` (positive control): trend, weekly seasonality and AR(0.8) noise, with
  bases that cannot model the AR part (``naive``, ``seasonal_naive``) and a linear
  corrector. The residuals are predictable at short horizons, so a standalone champion
  is a **rejected useful gain**.

For each series and each number of select origins (``fold``), one experiment runs the
search and the selection once; every policy is then applied to the *same* finalists with
the pure ``run_chain``, so policies differ only in the decision rule. Rates are reported
per chain (the final champion: what a user gets) and per comparison (each hybrid
challenger), with percentile bootstrap intervals that resample **series**, the
independent units (origins of one series are not independent). Calibration and
confirmation sets use disjoint seeds: tune a policy on the first, check it once on the
second.

From Python::

    from autohybts.benchmarks.calibration import (
        calibration_report, control_cases, run_calibration,
    )

    table = run_calibration(control_cases(20), folds=(2, 3, 5))
    calibration_report(table)

From the command line (``--out`` also writes the resolved protocol)::

    python -m autohybts.benchmarks.calibration --series 20 --folds 2 3 5 --out cal.csv
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from autohybts.datasets import get_data
from autohybts.domain.results import EvaluationResult
from autohybts.evaluation.temporal import DEFAULT_METRICS
from autohybts.exceptions import ConfigurationError
from autohybts.experiment import AutoHybridExperiment
from autohybts.persistence.manifest import environment, utc_now
from autohybts.search.budgets import BudgetPreset
from autohybts.selection.incremental import run_chain
from autohybts.selection.policy import SelectionPolicy

__all__ = [
    "CALIBRATION_SCHEMA",
    "CONTROLS",
    "DEFAULT_POLICIES",
    "ControlCase",
    "calibration_protocol",
    "calibration_report",
    "control_cases",
    "run_calibration",
]

CALIBRATION_SCHEMA: Final = "calibration/1"
# Control -> what a hybrid champion means there.
CONTROLS: Final = {"white_noise": "negative", "ar_residual": "positive"}
SPLITS: Final = {"calibration": 10_000, "confirmation": 20_000}
CONTROL_SPACE: Final[dict[str, dict[str, Any]]] = {
    "white_noise": {
        "base_models": ("naive", "seasonal_naive", "linear"),
        "correctors": ("linear", "knn"),
        "residual_windows": (7,),
        "alphas": (1.0,),
    },
    "ar_residual": {
        "base_models": ("naive", "seasonal_naive"),
        "correctors": ("linear",),
        "residual_windows": (7,),
        "alphas": (1.0,),
    },
}
# Policy name -> overrides of the experiment's SelectionPolicy.
DEFAULT_POLICIES: Final[dict[str, dict[str, Any]]] = {
    "rule": {},
    "rule+wilcoxon_0.05": {"significance_alpha": 0.05},
}


@dataclass(frozen=True)
class ControlCase:
    series_id: str
    control: str
    values: pd.Series


def control_cases(n_series: int, split: str = "calibration", n: int = 300) -> Iterator[ControlCase]:
    """``n_series`` series of each control; ``split`` picks disjoint seeds."""

    if split not in SPLITS:
        raise ConfigurationError(f"split deve ser um de {sorted(SPLITS)}; recebido {split!r}.")
    offset = SPLITS[split]
    for index in range(n_series):
        seed = offset + index
        yield ControlCase(f"wn{seed}", "white_noise", get_data("white_noise_daily", n=n, seed=seed))
        yield ControlCase(
            f"ar{seed}", "ar_residual", get_data("synthetic_daily", n=n, seed=seed, phi=0.8)
        )


def _is_hybrid(result: EvaluationResult) -> bool:
    return result.candidate.strategy != "standalone"


def run_calibration(
    cases: Iterable[ControlCase],
    folds: Sequence[int] = (2, 3, 5),
    policies: Mapping[str, Mapping[str, Any]] = DEFAULT_POLICIES,
    fh: int = 7,
    budget: BudgetPreset = "smoke",
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
    include: Sequence[str] = ("standalone", "residual"),
) -> pd.DataFrame:
    """One row per series, number of select origins and policy (see the module docstring).

    ``include`` are the families of the search; with ``"ensemble"`` the rates measure the
    calibration gate of the composed families (plano de ensembles §6): any champion that
    is not a single model counts as a composed champion.
    """

    rows: list[dict[str, object]] = []
    for case in cases:
        if case.control not in CONTROLS:
            raise ConfigurationError(f"Controle desconhecido {case.control!r}.")
        for fold in folds:
            exp = AutoHybridExperiment(
                fh=fh, fold=fold, budget=budget, session_id=seed, **CONTROL_SPACE[case.control]
            ).fit(case.values)
            compare = exp.compare_models(include=include)
            selection = compare.selection
            finalists = {r.candidate.family: r for r in selection.finalists}
            initial_id = (
                selection.decisions[0].champion_before
                if selection.decisions
                else selection.champion.candidate_id
            )
            initial = next(r for r in selection.finalists if r.candidate_id == initial_id)
            metric = compare.search.metric
            for name, overrides in policies.items():
                policy = replace(exp.policy_, **dict(overrides))
                champion, decisions = run_chain(initial, finalists, policy, metric)
                by_id = {r.candidate_id: r for r in selection.finalists}
                hybrid = [d for d in decisions if _is_hybrid(by_id[d.challenger])]
                rows.append(
                    {
                        "schema": CALIBRATION_SCHEMA,
                        "series_id": case.series_id,
                        "control": case.control,
                        "kind": CONTROLS[case.control],
                        "fold": fold,
                        "policy": name,
                        "select_units": len(initial.unit_ids),
                        "champion": champion.candidate.name,
                        "champion_is_hybrid": _is_hybrid(champion),
                        "champion_strategy": champion.candidate.strategy,
                        "hybrid_comparisons": len(hybrid),
                        "hybrid_accepts": sum(d.accepted for d in hybrid),
                        "accepted_families": "+".join(d.family for d in decisions if d.accepted),
                    }
                )
            if progress is not None:
                progress(f"{case.series_id} fold={fold}")
    return pd.DataFrame(rows)


def _bootstrap(
    groups: pd.DataFrame, statistic: Callable[[pd.DataFrame], float], n_boot: int, seed: int
) -> tuple[float, float]:
    """Percentile interval resampling rows (one row per series) with replacement."""

    rng = np.random.default_rng(seed)
    values = [
        statistic(groups.iloc[rng.integers(0, len(groups), len(groups))]) for _ in range(n_boot)
    ]
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def _comparison_rate(frame: pd.DataFrame) -> float:
    comparisons = frame["hybrid_comparisons"].sum()
    return float(frame["hybrid_accepts"].sum() / comparisons) if comparisons else float("nan")


def calibration_report(table: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per control, number of select origins and policy: rates with 95 % intervals.

    ``hybrid_champion_rate`` is the share of series whose final champion is a hybrid: on
    the negative control the false acceptance per chain, on the positive control one
    minus the rejected-gain rate (``rejected_gain_rate``). ``comparison_accept_rate`` is
    the share of hybrid challengers accepted. Intervals resample series.
    """

    rows = []
    for (control, fold, policy), group in table.groupby(["control", "fold", "policy"], sort=True):
        rate = float(group["champion_is_hybrid"].mean())
        low, high = _bootstrap(group, lambda g: float(g["champion_is_hybrid"].mean()), n_boot, seed)
        c_low, c_high = _bootstrap(group, _comparison_rate, n_boot, seed)
        units = int(group["select_units"].median())
        rows.append(
            {
                "control": control,
                "kind": CONTROLS.get(str(control), "?"),
                "fold": fold,
                "policy": policy,
                "n_series": int(group["series_id"].nunique()),
                "select_units": units,
                "min_one_sided_p": 1 / 2**units,
                "hybrid_champion_rate": rate,
                "rate_ci_low": low,
                "rate_ci_high": high,
                "rejected_gain_rate": 1 - rate
                if CONTROLS.get(str(control)) == "positive"
                else float("nan"),
                "comparison_accept_rate": _comparison_rate(group),
                "comparison_ci_low": c_low,
                "comparison_ci_high": c_high,
            }
        )
    return pd.DataFrame(rows).set_index(["control", "fold", "policy"])


def calibration_protocol(
    n_series: int,
    split: str,
    folds: Sequence[int],
    policies: Mapping[str, Mapping[str, Any]] = DEFAULT_POLICIES,
    fh: int = 7,
    budget: BudgetPreset = "smoke",
    seed: int = 0,
    include: Sequence[str] = ("standalone", "residual"),
) -> dict[str, Any]:
    """Resolved settings of a calibration run, to save next to its table."""

    return {
        "schema": CALIBRATION_SCHEMA,
        "created_at": utc_now(),
        "environment": environment(),
        "split": split,
        "seed_offset": SPLITS[split],
        "n_series_per_control": n_series,
        "controls": dict(CONTROLS),
        "spaces": {c: {k: list(v) for k, v in s.items()} for c, s in CONTROL_SPACE.items()},
        "folds": list(folds),
        # Base policy of every experiment (thresholds, variability, family order); the
        # named policies below override it. Family order is the default one: the controls
        # register no model of their own.
        "base_policy": SelectionPolicy().describe(),
        "metric": DEFAULT_METRICS[0],
        "policies": {k: dict(v) for k, v in policies.items()},
        "fh": fh,
        "include": list(include),
        "budget": budget,
        "session_seed": seed,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibração da regra de aceite (M02).")
    parser.add_argument("--series", type=int, default=10, help="séries por controle")
    parser.add_argument("--split", default="calibration", choices=sorted(SPLITS))
    parser.add_argument("--folds", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--budget", default="smoke")
    parser.add_argument(
        "--include", nargs="+", default=["standalone", "residual"], help="famílias da busca"
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    cases = list(control_cases(args.series, args.split))
    table = run_calibration(
        cases,
        args.folds,
        budget=args.budget,
        progress=lambda m: print(m, flush=True),
        include=args.include,
    )
    report = calibration_report(table)
    if args.out:
        table.to_csv(args.out, index=False)
        report.to_csv(f"{args.out}.report.csv")
        protocol = calibration_protocol(
            args.series, args.split, args.folds, budget=args.budget, include=args.include
        )
        Path(f"{args.out}.protocol.json").write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(report.round(3))


if __name__ == "__main__":
    main()
