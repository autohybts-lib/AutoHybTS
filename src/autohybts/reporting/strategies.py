"""How the strategies of building a forecaster compare: error against cost (Pareto).

Every candidate the search evaluated is a point: error on ``V_search`` against fit
time or structural complexity. A point is on the Pareto front when no other point is
at least as good on both axes and better on one. The front shows what each extra unit
of complexity bought; it is a report, never an input of the selection rule.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

from autohybts.domain.results import EvaluationResult
from autohybts.domain.specs import CandidateSpec
from autohybts.exceptions import ConfigurationError
from autohybts.reporting.plots import PlotArtifact, _figure

__all__ = ["STRATEGY_PLOTS", "complexity", "pareto_front", "strategy_plot", "strategy_table"]

Axis: TypeAlias = Literal["cost", "complexity"]
STRATEGY_PLOTS: Final = ("pareto", "strategies")
_COLORS: Final = {
    "standalone": "#4C72B0",
    "residual_additive": "#DD8452",
    "residual_multistage": "#C44E52",
    "decomposition": "#55A868",
    "fixed_combination": "#8172B3",
    "dynamic_selection": "#937860",
}


def complexity(candidate: CandidateSpec) -> int:
    """Number of fitted models in the structure (a base is 1, each corrector adds 1)."""

    if candidate.strategy == "fixed_combination":
        return 1 + len(candidate.members)
    if candidate.corrector is None:
        return 1
    return 2 + len(candidate.members)


def pareto_front(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Mask of points not dominated when both ``x`` and ``y`` are minimized."""

    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    front = np.zeros(xs.size, dtype=bool)
    for i in range(xs.size):
        if not (math.isfinite(xs[i]) and math.isfinite(ys[i])):
            continue
        dominated = (xs <= xs[i]) & (ys <= ys[i]) & ((xs < xs[i]) | (ys < ys[i]))
        front[i] = not dominated.any()
    return front


def strategy_table(results: Sequence[EvaluationResult], metric: str) -> pd.DataFrame:
    """One row per evaluated candidate: score, coverage, cost, complexity, Pareto flags."""

    rows = []
    for result in results:
        candidate = result.candidate
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "name": candidate.name,
                "model": candidate.label,
                "strategy": candidate.strategy,
                "family": candidate.family,
                "score": result.mean(metric),
                "coverage": result.coverage(metric),
                "cost_seconds": sum(t.fit_seconds + t.predict_seconds for t in result.trials),
                "complexity": complexity(candidate),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    valid = table["coverage"] == 1.0
    score = table["score"].where(valid, math.inf)
    table["pareto_cost"] = pareto_front(table["cost_seconds"], score)
    table["pareto_complexity"] = pareto_front(table["complexity"], score)
    return table.sort_values(["score", "candidate_id"]).reset_index(drop=True)


def strategy_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per strategy: candidates, best and median score, best model, mean cost."""

    if table.empty:
        return table
    best = table.loc[table.groupby("strategy")["score"].idxmin()].set_index("strategy")
    summary = table.groupby("strategy").agg(
        candidates=("candidate_id", "size"),
        best_score=("score", "min"),
        median_score=("score", "median"),
        mean_cost_seconds=("cost_seconds", "mean"),
    )
    summary["best_model"] = best["name"]
    return summary.sort_values("best_score")


def strategy_plot(
    kind: str, table: pd.DataFrame, metric: str, x: Axis = "cost", source: str = "V_search"
) -> PlotArtifact:
    """``pareto`` (error vs cost or complexity, front highlighted) or ``strategies``."""

    if kind not in STRATEGY_PLOTS:
        raise ConfigurationError(f"Gráfico desconhecido {kind!r}; opções: {STRATEGY_PLOTS}.")
    if x not in ("cost", "complexity"):
        raise ConfigurationError("x deve ser 'cost' ou 'complexity'.")
    data = table[table["coverage"] == 1.0]
    if data.empty:
        raise ConfigurationError("Nenhuma candidata com cobertura completa para desenhar.")
    if kind == "pareto":
        figure = _pareto(data, metric, x, source)
    else:
        figure = _strategies(data, metric, source)
    return PlotArtifact(
        kind, figure, source, provenance={"metric": metric, "x": x, "n_candidates": len(data)}
    )


def _pareto(data: pd.DataFrame, metric: str, x: Axis, source: str) -> object:
    figure, (ax,) = _figure(height=4.8)
    column = "cost_seconds" if x == "cost" else "complexity"
    flag = "pareto_cost" if x == "cost" else "pareto_complexity"
    rng = np.random.default_rng(0)
    for strategy, group in data.groupby("strategy"):
        xs = group[column].to_numpy(dtype=float)
        if x == "complexity":  # small jitter so equal complexities do not overlap
            xs = xs + rng.uniform(-0.12, 0.12, xs.size)
        ax.scatter(
            xs,
            group["score"],
            s=36,
            alpha=0.75,
            color=_COLORS.get(str(strategy), "#777777"),
            label=str(strategy),
        )
    front = data[data[flag]].sort_values(column)
    ax.step(front[column], front["score"], where="post", color="black", linewidth=1.0)
    for _, row in front.iterrows():
        ax.annotate(
            row["name"],
            (row[column], row["score"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    if x == "cost":
        ax.set_xscale("log")
        ax.set_xlabel("custo de ajuste + previsão (s, escala log)")
    else:
        ax.set_xlabel("complexidade (modelos ajustados na estrutura)")
        ax.set_xticks(sorted(data["complexity"].unique()))
    ax.set_ylabel(f"{metric} médio ({source})")
    ax.set_title(
        f"Fronteira de Pareto: erro × {'custo' if x == 'cost' else 'complexidade'}", fontsize=10
    )
    ax.legend(fontsize=7, loc="best")
    return figure


def _strategies(data: pd.DataFrame, metric: str, source: str) -> object:
    order = data.groupby("strategy")["score"].min().sort_values().index.tolist()
    figure, (ax,) = _figure(height=max(3.0, 0.6 * len(order) + 1.2))
    rng = np.random.default_rng(0)
    for position, strategy in enumerate(order):
        group = data[data["strategy"] == strategy]
        ys = position + rng.uniform(-0.15, 0.15, len(group))
        ax.scatter(group["score"], ys, s=30, alpha=0.75, color=_COLORS.get(strategy, "#777777"))
        best = group.loc[group["score"].idxmin()]
        ax.annotate(
            best["name"],
            (best["score"], position),
            fontsize=7,
            xytext=(4, 6),
            textcoords="offset points",
        )
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric} médio ({source})")
    ax.set_title("Candidatas por estratégia (melhor de cada uma anotada)", fontsize=10)
    return figure
