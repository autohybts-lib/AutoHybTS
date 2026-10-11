"""Typed, immutable results returned by the experiment verbs (ADR-0001)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from autohybts.domain.results import EvaluationResult
from autohybts.domain.specs import CandidateSpec
from autohybts.exceptions import ConfigurationError
from autohybts.reporting.comparison import ComparisonResult
from autohybts.reporting.plots import PlotArtifact
from autohybts.reporting.strategies import (
    STRATEGY_PLOTS,
    strategy_plot,
    strategy_summary,
    strategy_table,
)
from autohybts.search.ledger import SearchResult
from autohybts.selection.incremental import SelectionAudit, SelectionResult

__all__ = [
    "CompareResult",
    "Event",
    "ExternalComparison",
    "ExternalResult",
    "ModelResult",
    "NestedCVResult",
    "PredictResult",
]


@dataclass(frozen=True, slots=True)
class Event:
    kind: str
    message: str
    at: datetime
    duration_ms: float | None = None


@dataclass(frozen=True, slots=True)
class ModelResult:
    """One candidate evaluated on every origin (search and select layers)."""

    candidate: CandidateSpec
    evaluation: EvaluationResult

    @property
    def cv_results(self) -> pd.DataFrame:
        """One row per origin, plus ``Mean`` and ``Std`` rows (like PyCaret)."""

        table = self.evaluation.to_frame()
        metrics = list(self.evaluation.metric_names)
        summary = table[metrics].agg(["mean", "std"]).rename(index={"mean": "Mean", "std": "Std"})
        return pd.concat([table[["layer", "status", *metrics]], summary])

    @property
    def metrics(self) -> dict[str, float]:
        return {m: self.evaluation.mean(m) for m in self.evaluation.metric_names}

    def __repr__(self) -> str:
        return f"ModelResult({self.candidate.label!r})"


@dataclass(frozen=True)
class CompareResult:
    """Search on V_search and incremental selection on V_select."""

    best: CandidateSpec
    leaderboard: pd.DataFrame
    selection_leaderboard: pd.DataFrame
    decisions: pd.DataFrame
    audit: SelectionAudit
    search: SearchResult
    selection: SelectionResult
    snapshot: dict[str, Any]
    comparison: ComparisonResult | None = None
    events: tuple[Event, ...] = field(default_factory=tuple)

    def strategy_table(self) -> pd.DataFrame:
        """Every candidate of the search: score on ``V_search``, cost, complexity, Pareto."""

        return strategy_table(self.search.results, self.search.metric)

    @property
    def strategies(self) -> pd.DataFrame:
        """Per strategy (standalone, residual, multistage ...): best and median score, cost."""

        return strategy_summary(self.strategy_table())

    def plot(self, kind: str = "pareto", x: str = "cost") -> PlotArtifact:
        """``pareto`` (error × cost, or × complexity with ``x="complexity"``) and
        ``strategies`` over every searched candidate; the comparison plots
        (``leaderboard``, ``boxplot``, ``win_rate`` ...) over the finalists on ``V_select``.
        """

        if kind in STRATEGY_PLOTS:
            return strategy_plot(kind, self.strategy_table(), self.search.metric, x)  # type: ignore[arg-type]
        if self.comparison is None:
            raise ConfigurationError("Sem comparação das finalistas neste resultado.")
        artifact: PlotArtifact = self.comparison.plot(kind)
        return artifact

    def __repr__(self) -> str:
        return f"CompareResult(best={self.best.label!r}, trials={self.audit.total_trials})"


@dataclass(frozen=True)
class PredictResult:
    predictions: pd.DataFrame
    model_label: str

    def __repr__(self) -> str:
        return f"PredictResult({self.model_label!r}, steps={len(self.predictions)})"


@dataclass(frozen=True)
class ExternalResult:
    """Terminal evaluation on the sealed (or an external) test; never fed back to the search."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    model_label: str


@dataclass(frozen=True)
class ExternalComparison:
    """Terminal comparison on the sealed test: reporting only, never fed back (§30.1).

    ``tests`` compares each model with the reference on per-step absolute errors of the
    single test block (Diebold-Mariano with ``fh - 1`` Newey-West lags, Holm-adjusted).
    """

    table: pd.DataFrame
    steps: pd.DataFrame
    tests: pd.DataFrame
    reference: str

    def __repr__(self) -> str:
        return f"ExternalComparison({len(self.table)} modelos, referência {self.reference!r})"


@dataclass(frozen=True)
class NestedCVResult:
    """Outer rolling-origin evaluation of the whole automated procedure (§30.1, T27).

    In each outer fold the search and the selection ran again on the history before the
    outer origin only; the outer block was never seen by them. ``include`` are the
    families of the inner searches; ``cost`` has one row per outer fold with the work
    done there: candidates and units (origin x series) of the inner search and
    selection, configured budget, stop reason and seconds (search, selection, refit,
    total).
    """

    folds: pd.DataFrame
    summary: pd.DataFrame
    metric: str
    include: tuple[str, ...] = ()
    cost: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def total_candidates(self) -> int:
        """Candidates evaluated by every inner search together."""

        if "search_candidates" not in self.cost:
            return 0
        return int(self.cost["search_candidates"].fillna(0).sum())

    @property
    def total_seconds(self) -> float:
        if "seconds_total" not in self.cost:
            return 0.0
        return float(self.cost["seconds_total"].fillna(0.0).sum())

    def __repr__(self) -> str:
        return f"NestedCVResult({len(self.folds)} dobras externas, {self.metric})"
