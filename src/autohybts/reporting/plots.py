"""Plots of data and models (§34.4-§34.6), with matplotlib as an optional backend.

Figures are built with matplotlib's object API (no pyplot, no global state), and
matplotlib is imported only when a plot is requested. Every plot uses development data
or out-of-sample validation forecasts; the sealed test never appears. Each artifact
records where its numbers come from.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from autohybts.data.schema import SeriesFrame
from autohybts.domain.results import EvaluationResult
from autohybts.domain.specs import CandidateSpec
from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError
from autohybts.reporting.comparison import ComparisonResult
from autohybts.reporting.diagnostics import error_by_horizon, forecast_errors
from autohybts.selection.incremental import SelectionResult

__all__ = [
    "COMPARISON_PLOTS",
    "DATA_PLOTS",
    "MODEL_PLOTS",
    "PlotArtifact",
    "comparison_plot",
    "data_plot",
    "model_plot",
]

DATA_PLOTS: Final = ("series", "acf", "seasonality", "distribution")
MODEL_PLOTS: Final = (
    "cv_forecasts",
    "residuals",
    "metric_by_horizon",
    "selection_trace",
    "search_overfitting",
    "model_structure",
)
COMPARISON_PLOTS: Final = (
    "leaderboard",
    "boxplot",
    "metric_by_fold",
    "win_rate",
    "critical_difference",
    "forecasts",
)
_LAYER_COLORS: Final = {"search": "#4C72B0", "select": "#DD8452", "cv": "#55A868"}


@dataclass(frozen=True)
class PlotArtifact:
    kind: str
    figure: Any
    source_split: str
    candidate_id: str | None = None
    origin_range: tuple[pd.Timestamp, pd.Timestamp] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    backend: str = "matplotlib"

    def save(self, path: str | Path) -> Path:
        """Write the figure (format from the extension, e.g. ``.png`` or ``.svg``)."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(target, bbox_inches="tight", dpi=120)
        return target

    def _repr_png_(self) -> bytes:
        import io

        buffer = io.BytesIO()
        self.figure.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
        return buffer.getvalue()


def _figure(width: float = 10.0, height: float = 3.6, rows: int = 1) -> tuple[Any, Any]:
    try:
        from matplotlib.figure import Figure
    except ModuleNotFoundError as error:
        raise UnsupportedCapabilityError(
            'Gráficos exigem o extra `plot`: pip install "autohybts[plot]".'
        ) from error
    figure = Figure(figsize=(width, height * rows), layout="constrained")
    axes = figure.subplots(rows, 1, squeeze=False)[:, 0]
    return figure, axes


# --------------------------------------------------------------------------- data plots
def data_plot(
    kind: str, history: SeriesFrame, seasonal_period: int, sealed_size: int
) -> PlotArtifact:
    if kind not in DATA_PLOTS:
        raise ConfigurationError(f"Gráfico de dados desconhecido {kind!r}; opções: {DATA_PLOTS}.")
    figure, (ax,) = _figure()
    values = history.values
    provenance = {"observations": len(history), "sealed_test_size": sealed_size}
    if kind == "series":
        ax.plot(history.timestamps, values, color="#333333", linewidth=1)
        ax.set_title(f"Histórico de desenvolvimento ({len(history)} pontos; teste selado omitido)")
        ax.set_xlabel("data")
    elif kind == "acf":
        lags = max(1, min(3 * seasonal_period, len(values) // 4, 60))
        acf = _acf(values, lags)
        bound = 1.96 / math.sqrt(len(values))
        ax.bar(range(1, lags + 1), acf, color="#4C72B0", width=0.6)
        ax.axhline(bound, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(-bound, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"Autocorrelação (n={len(values)}, {lags} lags, faixa ±1,96/√n)")
        ax.set_xlabel("lag")
        provenance["lags"] = lags
    elif kind == "seasonality":
        if seasonal_period < 2:
            raise ConfigurationError("Sem período sazonal (seasonal_period=1).")
        position = np.arange(len(values)) % seasonal_period
        frame = pd.DataFrame({"position": position, "value": values})
        stats = frame.groupby("position")["value"].agg(["mean", "std", "count"])
        ax.errorbar(
            stats.index, stats["mean"], yerr=stats["std"], fmt="o-", color="#4C72B0", capsize=3
        )
        ax.set_title(
            f"Perfil sazonal (período {seasonal_period}; média ± desvio; "
            f"{int(stats['count'].min())}–{int(stats['count'].max())} pontos por posição)"
        )
        ax.set_xlabel("posição no ciclo")
    else:
        ax.hist(values, bins=min(40, max(10, len(values) // 20)), color="#4C72B0")
        ax.set_title(f"Distribuição dos valores (n={len(values)})")
    return PlotArtifact(
        kind,
        figure,
        "development",
        origin_range=(history.start, history.end),
        provenance=provenance,
    )


def _acf(values: np.ndarray, lags: int) -> np.ndarray:
    centered = values - values.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        return np.zeros(lags)
    return np.array([np.dot(centered[:-k], centered[k:]) / denominator for k in range(1, lags + 1)])


# --------------------------------------------------------------------------- model plots
def model_plot(
    kind: str,
    candidate: CandidateSpec,
    evaluation: EvaluationResult,
    history: SeriesFrame,
    selection: SelectionResult | None,
    search_scores: dict[str, float] | None = None,
) -> PlotArtifact:
    if kind not in MODEL_PLOTS:
        raise ConfigurationError(f"Gráfico de modelo desconhecido {kind!r}; opções: {MODEL_PLOTS}.")
    metric = evaluation.metric_names[0]
    origins = [t.origin_time for t in evaluation.trials]
    origin_range = (min(origins), max(origins)) if origins else None
    provenance: dict[str, Any] = {
        "metric": metric,
        "units": len(evaluation.trials),
        "coverage": evaluation.coverage(metric),
        "layers": sorted({t.layer for t in evaluation.trials}),
        "series": 1,
    }
    if kind in ("selection_trace", "search_overfitting") and selection is None:
        raise ConfigurationError(f"O gráfico {kind!r} exige compare_models antes.")

    if kind == "cv_forecasts":
        figure = _cv_forecasts(candidate, evaluation, history, metric)
    elif kind == "residuals":
        figure = _residuals(candidate, evaluation)
    elif kind == "metric_by_horizon":
        figure = _metric_by_horizon(candidate, evaluation)
    elif kind == "selection_trace":
        assert selection is not None
        figure = _selection_trace(candidate, selection)
        provenance["decisions"] = len(selection.decisions)
        provenance["trials"] = selection.audit.total_trials
    elif kind == "search_overfitting":
        assert selection is not None
        figure = _search_overfitting(selection, search_scores or {}, metric)
    else:
        figure = _model_structure(candidate)
    return PlotArtifact(
        kind,
        figure,
        "validation",
        candidate_id=candidate.candidate_id,
        origin_range=origin_range,
        provenance=provenance,
    )


def _cv_forecasts(
    candidate: CandidateSpec, evaluation: EvaluationResult, history: SeriesFrame, metric: str
) -> Any:
    figure, (ax,) = _figure()
    first = min(t.origin_index for t in evaluation.trials)
    start = max(0, first - 3 * max(t.horizon for t in evaluation.trials))
    ax.plot(
        history.timestamps[start:],
        history.values[start:],
        color="#333333",
        linewidth=1,
        label="observado (desenvolvimento)",
    )
    errors = forecast_errors(evaluation)
    for (layer, fold), group in errors.groupby(["layer", "unit"], sort=False):
        ax.plot(
            group["timestamp"],
            group["y_pred"],
            color=_LAYER_COLORS.get(str(layer), "C2"),
            linewidth=1.5,
            label=f"previsões {layer}" if str(fold).endswith("-0") else None,
        )
    ax.set_title(
        f"{candidate.label}\nprevisões fora da amostra por origem ({len(evaluation.trials)} "
        f"origens; {metric} médio {evaluation.mean(metric):.3f}; cobertura "
        f"{evaluation.coverage(metric):.0%})",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8)
    return figure


def _residuals(candidate: CandidateSpec, evaluation: EvaluationResult) -> Any:
    figure, axes = _figure(rows=2, height=2.8)
    errors = forecast_errors(evaluation)
    for layer, group in errors.groupby("layer"):
        axes[0].scatter(
            group["timestamp"],
            group["error"],
            s=10,
            color=_LAYER_COLORS.get(str(layer), "C2"),
            label=str(layer),
        )
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].set_title(
        f"{candidate.label}\nerros fora da amostra (y − ŷ) nas origens de validação", fontsize=10
    )
    axes[0].legend(fontsize=8)
    axes[1].hist(errors["error"], bins=30, color="#4C72B0")
    axes[1].set_title(f"distribuição dos erros (n={len(errors)} passos)", fontsize=10)
    return figure


def _metric_by_horizon(candidate: CandidateSpec, evaluation: EvaluationResult) -> Any:
    figure, (ax,) = _figure()
    table = error_by_horizon(evaluation)
    for layer, group in table.groupby("layer"):
        ax.plot(
            group["horizon"],
            group["mae"],
            marker="o",
            color=_LAYER_COLORS.get(str(layer), "C2"),
            label=f"{layer} (n={int(group['n_origins'].max())} origens)",
        )
    ax.set_xlabel("passo à frente")
    ax.set_ylabel("MAE")
    ax.set_title(f"{candidate.label}\nerro absoluto médio por horizonte", fontsize=10)
    ax.legend(fontsize=8)
    return figure


def _selection_trace(candidate: CandidateSpec, selection: SelectionResult) -> Any:
    figure, (ax,) = _figure(height=4.2)
    decisions = selection.decisions
    if not decisions:
        ax.text(0.5, 0.5, "sem desafiantes: o campeão inicial foi mantido", ha="center")
        ax.set_axis_off()
        return figure
    thresholds = decisions[0].thresholds
    gains = [d.median_gain if math.isfinite(d.median_gain) else 0.0 for d in decisions]
    colors = ["#55A868" if d.accepted else "#C44E52" for d in decisions]
    labels = [f"{d.family}\n{d.challenger_name or _short(d.challenger_label)}" for d in decisions]
    x = np.arange(len(decisions))
    ax.bar(x, gains, color=colors)
    ax.axhline(
        cast(float, thresholds["median_gain_threshold"]),
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label="limiar de ganho mediano",
    )
    for i, decision in enumerate(decisions):
        ax.annotate(
            f"vitórias {decision.win_rate:.0%}",
            (i, gains[i]),
            ha="center",
            va="bottom" if gains[i] >= 0 else "top",
            fontsize=7,
        )
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("ganho mediano vs campeão atual")
    audit = selection.audit
    ax.set_title(
        "Seleção incremental em V_select (verde = aceito); campeão final: "
        f"{_short(candidate.label)}"
        f"\n{audit.total_trials} tentativas na busca, {audit.pruned_trials} podadas; perfil "
        f"{thresholds['profile']}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    return figure


def _search_overfitting(
    selection: SelectionResult, search_scores: dict[str, float], metric: str
) -> Any:
    figure, (ax,) = _figure(width=6, height=5)
    xs, ys = [], []
    for finalist in selection.finalists:
        search = search_scores.get(finalist.candidate_id)
        if search is None:
            continue
        xs.append(search)
        ys.append(finalist.mean(metric))
        ax.annotate(finalist.candidate.name, (search, ys[-1]), fontsize=7)
    ax.scatter(xs, ys, color="#4C72B0")
    if xs:
        low, high = min(xs + ys), max(xs + ys)
        ax.plot([low, high], [low, high], color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(f"{metric} em V_search")
    ax.set_ylabel(f"{metric} em V_select")
    gap = selection.audit.selection_gap
    ax.set_title(f"Busca × seleção das finalistas (gap do vencedor: {gap:+.3f})", fontsize=10)
    return figure


def _model_structure(candidate: CandidateSpec) -> Any:
    figure, (ax,) = _figure(width=8, height=2.4)
    ax.set_axis_off()
    boxes: Sequence[tuple[float, str]]
    if candidate.corrector is None:
        boxes = [(0.5, f"base\n{candidate.base.label}")]
    else:
        boxes = [
            (0.15, f"base\n{candidate.base.label}"),
            (0.5, f"resíduos\n{candidate.residual_source}\njanela {candidate.window_size}"),
            (0.85, f"corretor\n{candidate.corrector.label}\nα = {candidate.alpha}"),
        ]
    for x, text in boxes:
        ax.text(
            x,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "#EEF2F8", "edgecolor": "#4C72B0"},
        )
    for (x0, _), (x1, _) in itertools.pairwise(boxes):
        ax.annotate(
            "",
            xy=(x1 - 0.12, 0.5),
            xytext=(x0 + 0.12, 0.5),
            arrowprops={"arrowstyle": "->", "color": "gray"},
        )
    formula = "ŷ = b" if candidate.corrector is None else f"ŷ = b + {candidate.alpha} · r̂"
    ax.set_title(
        f"Estrutura ({candidate.strategy}, variante {candidate.variant}): {formula}", fontsize=10
    )
    return figure


def _short(label: str, width: int = 28) -> str:
    return label if len(label) <= width else label[: width - 1] + "…"


# --------------------------------------------------------------------------- comparison plots
def comparison_plot(kind: str, comparison: ComparisonResult) -> PlotArtifact:
    """Several models on the same units: ``leaderboard``, ``boxplot``, ``metric_by_fold``,
    ``win_rate``, ``critical_difference`` or ``forecasts``."""

    if kind not in COMPARISON_PLOTS:
        raise ConfigurationError(
            f"Gráfico de comparação desconhecido {kind!r}; opções: {COMPARISON_PLOTS}."
        )
    metric = comparison.metric
    units = comparison.units
    note = f"{len(units)} unidades ({comparison.source}); {metric}"
    if kind == "leaderboard":
        figure = _cmp_leaderboard(comparison, note)
    elif kind == "boxplot":
        figure, (ax,) = _figure(height=max(3.0, 0.45 * units.shape[1] + 1))
        order = comparison.summary.index
        ax.boxplot(
            [units[c].dropna() for c in order], orientation="horizontal", tick_labels=list(order)
        )
        ax.set_xlabel(metric)
        ax.set_title(f"Distribuição por unidade — {note}", fontsize=10)
    elif kind == "metric_by_fold":
        figure, (ax,) = _figure()
        for column in comparison.summary.index:
            ax.plot(units.index, units[column], marker="o", label=_short(column, 40))
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} por dobra — {note}", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    elif kind == "win_rate":
        figure = _cmp_win_rate(comparison, note)
    elif kind == "critical_difference":
        figure = _cmp_critical_difference(comparison)
    else:
        figure = _cmp_forecasts(comparison, note)
    origins = [t.origin_time for t in comparison.results[0].trials]
    return PlotArtifact(
        kind,
        figure,
        comparison.source,
        origin_range=(min(origins), max(origins)) if origins else None,
        provenance={
            "metric": metric,
            "units": len(units),
            "models": comparison.labels,
            "source": comparison.source,
        },
    )


def _cmp_leaderboard(comparison: ComparisonResult, note: str) -> Any:
    summary = comparison.summary
    figure, (ax,) = _figure(height=max(3.0, 0.45 * len(summary) + 1))
    y = np.arange(len(summary))[::-1]
    reference = comparison.labels[comparison.reference]
    colors = ["#DD8452" if label == reference else "#4C72B0" for label in summary.index]
    ax.barh(y, summary["mean"], xerr=summary["std"], color=colors, capsize=3)
    ax.set_yticks(y, [_short(label, 40) for label in summary.index], fontsize=8)
    for yi, (_, row) in zip(y, summary.iterrows(), strict=True):
        end = row["mean"] + (row["std"] if math.isfinite(row["std"]) else 0.0)
        ax.annotate(f"  posto {row['mean_rank']:.2f}", (end, yi), va="center", fontsize=7)
    ax.set_xlabel(f"{comparison.metric} médio ± desvio entre unidades")
    ax.set_title(f"Ranking (laranja = referência) — {note}", fontsize=10)
    return figure


def _cmp_win_rate(comparison: ComparisonResult, note: str) -> Any:
    matrix = comparison.win_rates
    labels = [_short(label, 24) for label in matrix.index]
    size = max(4.0, 0.7 * len(labels) + 2)
    figure, (ax,) = _figure(width=size + 1, height=size)
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, vmin=0, vmax=1, cmap="RdBu")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = float(values[i, j])
            if math.isfinite(value):
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title(f"Vitórias da linha sobre a coluna — {note}", fontsize=10)
    return figure


def _cmp_critical_difference(comparison: ComparisonResult) -> Any:
    result = comparison.friedman()
    ranks = result["mean_ranks"].sort_values()
    k = len(ranks)
    figure, (ax,) = _figure(height=max(2.6, 0.35 * k + 1.2))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-k - 1, 2)
    ax.axhline(0, color="black", linewidth=1)
    for position in range(1, k + 1):
        ax.plot([position, position], [-0.1, 0.1], color="black", linewidth=1)
        ax.text(position, 0.3, str(position), ha="center", fontsize=8)
    for i, (label, rank) in enumerate(ranks.items()):
        level = -(i + 1)
        ax.plot([rank, rank], [0, level], color="#4C72B0", linewidth=0.8)
        ax.text(rank, level, f" {_short(str(label), 34)} ({rank:.2f})", va="center", fontsize=8)
    cd = result["critical_difference"]
    if math.isfinite(cd):
        ax.plot([1, 1 + cd], [1.2, 1.2], color="#C44E52", linewidth=3)
        ax.text(1 + cd / 2, 1.45, f"CD = {cd:.2f}", ha="center", fontsize=8, color="#C44E52")
        # Groups whose mean ranks differ by less than CD are not distinguishable.
        values = ranks.to_numpy()
        for start in range(k):
            end = max(j for j in range(start, k) if values[j] - values[start] < cd)
            if end > start:
                ax.plot(
                    [values[start], values[end]],
                    [-0.5 - 0.15 * start] * 2,
                    color="gray",
                    linewidth=3,
                    alpha=0.6,
                )
    ax.set_axis_off()
    p_value = result["p_value"]
    p_text = f"p = {p_value:.3g}" if math.isfinite(p_value) else "p indisponível (< 3 modelos)"
    ax.set_title(
        f"Postos médios (menor = melhor) — Friedman {p_text}; "
        f"{result['n_units']} unidades completas, Nemenyi α = {result['alpha']}",
        fontsize=9,
    )
    return figure


def _cmp_forecasts(comparison: ComparisonResult, note: str) -> Any:
    figure, (ax,) = _figure()
    history = comparison.history
    first = min(t.origin_index for t in comparison.results[0].trials)
    horizon = max(t.horizon for t in comparison.results[0].trials)
    start = max(0, first - 2 * horizon)
    ax.plot(
        history.timestamps[start:],
        history.values[start:],
        color="#333333",
        linewidth=1.2,
        label="observado",
    )
    for index, (label, result) in enumerate(
        zip(comparison.labels, comparison.results, strict=True)
    ):
        errors = forecast_errors(result)
        for fold, group in errors.groupby("unit", sort=False):
            ax.plot(
                group["timestamp"],
                group["y_pred"],
                color=f"C{index % 10}",
                linewidth=1,
                alpha=0.9,
                label=_short(label, 40) if fold == errors["unit"].iloc[0] else None,
            )
    ax.set_title(f"Previsões fora da amostra dos modelos comparados — {note}", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    return figure
