"""Descriptive diagnostics built from evaluation and selection results (§17.2).

Every table carries its denominators. Nothing here tests significance or feeds back
into the search.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from autohybts.domain.results import EvaluationResult
from autohybts.exceptions import ConfigurationError
from autohybts.selection.incremental import SelectionDecision

__all__ = ["champion_evolution", "error_by_horizon", "forecast_errors", "residual_provenance"]


def forecast_errors(result: EvaluationResult) -> pd.DataFrame:
    """Out-of-sample error of every step of every succeeded origin (``y_true - y_pred``)."""

    rows = []
    for trial in result.trials:
        if not trial.succeeded:
            continue
        stamps = pd.date_range(trial.origin_time, periods=trial.horizon + 1, freq=result.frequency)
        for step, (actual, predicted) in enumerate(zip(trial.y_true, trial.y_pred, strict=True), 1):
            rows.append(
                {
                    "unit": trial.unit_id,
                    "series_id": trial.series_id,
                    "fold_id": trial.fold_id,
                    "layer": trial.layer,
                    "origin": trial.origin_time,
                    "timestamp": stamps[step],
                    "horizon": step,
                    "y_true": actual,
                    "y_pred": predicted,
                    "error": actual - predicted,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "unit",
            "series_id",
            "fold_id",
            "layer",
            "origin",
            "timestamp",
            "horizon",
            "y_true",
            "y_pred",
            "error",
        ],
    )


def error_by_horizon(result: EvaluationResult) -> pd.DataFrame:
    """Mean absolute error and number of origins per step and layer."""

    errors = forecast_errors(result)
    if errors.empty:
        return pd.DataFrame(columns=["layer", "horizon", "mae", "n_origins"])
    errors["abs_error"] = errors["error"].abs()
    table = (
        errors.groupby(["layer", "horizon"])
        .agg(mae=("abs_error", "mean"), n_origins=("unit", "nunique"))
        .reset_index()
    )
    return table


def champion_evolution(decisions: Sequence[SelectionDecision], initial_label: str) -> pd.DataFrame:
    """Champion after each decision, starting from the initial standalone champion."""

    rows = [
        {"step": 0, "family": "", "challenger": "", "accepted": True, "champion": initial_label}
    ]
    champion = initial_label
    for step, decision in enumerate(decisions, 1):
        if decision.accepted:
            champion = decision.challenger_label
        rows.append(
            {
                "step": step,
                "family": decision.family,
                "challenger": decision.challenger_label,
                "accepted": decision.accepted,
                "champion": champion,
            }
        )
    return pd.DataFrame(rows)


def residual_provenance(model: object) -> pd.DataFrame:
    """Provenance of the residuals a fitted hybrid learned from (origin, target, availability)."""

    residuals = getattr(model, "residuals_", None)
    if residuals is None:
        raise ConfigurationError("O modelo não é um híbrido residual ajustado.")
    table: pd.DataFrame = residuals.to_frame()
    if not table.empty and not np.all(table["available_at"] == table["target"]):
        raise ConfigurationError("Resíduo com disponibilidade incoerente.")
    return table
