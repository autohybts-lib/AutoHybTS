"""Optional exploratory intervention screening; never changes observations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autohybts.data.schema import SeriesFrame
from autohybts.exceptions import ConfigurationError, InsufficientHistoryError


def intervention_report(
    frame: SeriesFrame,
    *,
    types: tuple[str, ...] = ("AO", "LS", "TC"),
    threshold: float = 4.0,
    decay: float = 0.7,
    period: int = 1,
    max_events: int = 10,
) -> pd.DataFrame:
    """Screen pulse, step and decaying-pulse regressors against trend/seasonal controls.

    Scores are exploratory OLS scores, not ARIMA-adjusted significance tests.
    Greedy selection refits controls and accepted interventions on this history only.
    """
    if not types or any(kind not in {"AO", "LS", "TC"} for kind in types):
        raise ConfigurationError("types must contain AO, LS and/or TC.")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ConfigurationError("threshold must be finite and positive.")
    if not np.isfinite(decay) or not 0 < decay < 1:
        raise ConfigurationError("decay must be between 0 and 1.")
    for name, value in (("period", period), ("max_events", max_events)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigurationError(f"{name} must be a positive integer.")
    y = frame.values
    n = len(y)
    if n < max(20, 3 * period):
        raise InsufficientHistoryError("Intervention screening needs 20 points and 3 cycles.")
    t = np.arange(n)
    controls = np.column_stack([np.ones(n), t / n])
    if period > 1:
        controls = np.column_stack([controls, *((t % period == k) for k in range(1, period))])
    rows: list[dict[str, object]] = []
    used: set[int] = set()
    # Keep five observations on either side of a step to avoid unsupported edge shifts.
    for _ in range(min(max_events, n - controls.shape[1] - 2)):
        q, _ = np.linalg.qr(controls, mode="reduced")
        residual = y - q @ (q.T @ y)
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        if scale <= np.finfo(float).eps * max(1.0, float(np.max(np.abs(y)))):
            break
        best = None
        for position in range(n):
            if position in used:
                continue
            for kind in dict.fromkeys(types):
                if kind == "LS" and not 5 <= position <= n - 5:
                    continue
                x = np.zeros(n)
                if kind == "AO":
                    x[position] = 1
                elif kind == "LS":
                    x[position:] = 1
                else:
                    x[position:] = decay ** np.arange(n - position)
                adjusted = x - q @ (q.T @ x)
                norm = float(adjusted @ adjusted)
                if norm < 1e-10:
                    continue
                effect = float(adjusted @ residual) / norm
                score = abs(effect) * np.sqrt(norm) / scale
                if best is None or score > best[0]:
                    best = (score, position, kind, effect, x)
        if best is None or best[0] < threshold:
            break
        score, position, kind, effect, x = best
        used.add(position)
        controls = np.column_stack([controls, x])
        rows.append(
            {
                "series_id": frame.series_id,
                "position": position,
                "timestamp": frame.timestamps[position],
                "type": kind,
                "effect": effect,
                "score": score,
                "recommendation": "review_event_before_treatment",
            }
        )
    columns = ["series_id", "position", "timestamp", "type", "effect", "score", "recommendation"]
    report = pd.DataFrame(rows, columns=columns)
    report.attrs.update(
        method="exploratory_ols", threshold=threshold, decay=decay, treatment_applied=False
    )
    return report
