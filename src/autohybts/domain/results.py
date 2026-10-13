"""Trial and evaluation results (architecture §5.3, §11.1)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import numpy as np
import pandas as pd

from autohybts.domain.specs import CandidateSpec
from autohybts.forecasting.forecast_frame import FORECAST_COLUMNS, ForecastFrame
from autohybts.types import TrialStatus

__all__ = ["EvaluationResult", "Layer", "TrialResult", "combine_results", "unit_key"]

Layer: TypeAlias = Literal["search", "select", "cv"]
_SINGLE: Final = "__single__"


def unit_key(series_id: str, fold_id: str) -> str:
    """Evaluation unit: the fold for a single series, ``series:fold`` in a panel."""

    return fold_id if series_id == _SINGLE else f"{series_id}:{fold_id}"


@dataclass(frozen=True, slots=True)
class TrialResult:
    """One candidate at one origin: forecast, metrics, timings and outcome.

    ``metrics`` may hold ``NaN`` for a metric undefined at this unit; a failed or skipped
    trial has no forecast and no metrics, only a ``reason``.
    """

    candidate_id: str
    series_id: str
    fold_id: str
    layer: Layer
    origin_index: int
    origin_time: pd.Timestamp
    horizon: int
    status: TrialStatus
    trial_key: str
    seed: int | None = None
    y_pred: tuple[float, ...] = ()
    y_true: tuple[float, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0
    reason: str | None = None
    diagnostics: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    # Leaf model fits made by this trial (domain.fit_count); None for trials built
    # outside the evaluator.
    n_fits: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def succeeded(self) -> bool:
        return self.status == "SUCCEEDED"

    @property
    def unit_id(self) -> str:
        return unit_key(self.series_id, self.fold_id)

    def is_valid(self, metric: str | None = None) -> bool:
        """Succeeded and, if ``metric`` is given, that metric is finite."""

        if not self.succeeded:
            return False
        return metric is None or bool(np.isfinite(self.metrics.get(metric, np.nan)))


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """All trials of one candidate on one layer of a temporal plan."""

    candidate: CandidateSpec
    plan_fingerprint: str
    frequency: str
    seasonal_period: int
    metric_names: tuple[str, ...]
    trials: tuple[TrialResult, ...]

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    @property
    def series_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(trial.series_id for trial in self.trials))

    @property
    def unit_ids(self) -> tuple[str, ...]:
        return tuple(trial.unit_id for trial in self.trials)

    def for_series(self, series_id: str) -> EvaluationResult:
        """The trials of one series, as a result of their own."""

        return EvaluationResult(
            self.candidate,
            self.plan_fingerprint,
            self.frequency,
            self.seasonal_period,
            self.metric_names,
            tuple(t for t in self.trials if t.series_id == series_id),
        )

    @property
    def n_expected(self) -> int:
        return len(self.trials)

    def coverage(self, metric: str | None = None) -> float:
        """Share of expected units with a valid result (for ``metric``, if given)."""

        if not self.trials:
            return 0.0
        return sum(trial.is_valid(metric) for trial in self.trials) / len(self.trials)

    def scores(self, metric: str) -> pd.Series:
        """Metric per unit (``unit_id``); invalid units are ``NaN``, never zero."""

        return pd.Series(
            {
                trial.unit_id: trial.metrics[metric] if trial.is_valid(metric) else np.nan
                for trial in self.trials
            },
            name=metric,
            dtype="float64",
        )

    def mean(self, metric: str) -> float:
        """Mean over valid units; ``NaN`` if none is valid."""

        values = self.scores(metric).dropna()
        return float(values.mean()) if len(values) else float("nan")

    @property
    def failures(self) -> dict[str, str]:
        return {
            trial.unit_id: trial.reason or trial.status
            for trial in self.trials
            if not trial.succeeded
        }

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "candidate_id": trial.candidate_id,
                "unit": trial.unit_id,
                "series_id": trial.series_id,
                "fold_id": trial.fold_id,
                "layer": trial.layer,
                "origin": trial.origin_time,
                "status": trial.status,
                **{name: trial.metrics.get(name, np.nan) for name in self.metric_names},
                "fit_seconds": trial.fit_seconds,
                "predict_seconds": trial.predict_seconds,
                "seed": trial.seed,
                "reason": trial.reason,
            }
            for trial in self.trials
        ]
        return pd.DataFrame(rows).set_index("unit")

    def forecasts(self) -> ForecastFrame:
        """Forecasts of the succeeded trials, indexed by date."""

        rows = []
        for trial in self.trials:
            if not trial.succeeded:
                continue
            stamps = pd.date_range(
                trial.origin_time, periods=trial.horizon + 1, freq=self.frequency
            )
            for step in range(1, trial.horizon + 1):
                rows.append(
                    {
                        "series_id": trial.series_id,
                        "candidate_id": trial.candidate_id,
                        "fold_id": trial.fold_id,
                        "origin": trial.origin_time,
                        "timestamp": stamps[step],
                        "horizon": step,
                        "y_pred": trial.y_pred[step - 1],
                        "y_true": trial.y_true[step - 1],
                    }
                )
        return ForecastFrame(pd.DataFrame(rows, columns=list(FORECAST_COLUMNS)), self.frequency)


def combine_results(
    parts: Sequence[EvaluationResult], candidate: CandidateSpec, plan_fingerprint: str
) -> EvaluationResult:
    """One result from per-series parts (e.g. each series at its own best configuration)."""

    if not parts:
        raise ValueError("combine_results needs at least one part.")
    first = parts[0]
    return EvaluationResult(
        candidate,
        plan_fingerprint,
        first.frequency,
        first.seasonal_period,
        first.metric_names,
        tuple(trial for part in parts for trial in part.trials),
    )
