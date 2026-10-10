"""Corrector inputs: windows of residuals, each available before the residual it predicts.

A training row predicts ``r[t]`` from the ``L`` residuals of targets ``t-L .. t-1``. Its
forecast origin is ``t - 1``; every feature must have ``available_at <= t - 1``. Rows whose
window is not contiguous are skipped (policy ``skip``); nothing is padded or imputed.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from autohybts.exceptions import (
    ConfigurationError,
    InsufficientHistoryError,
    TemporalLeakageError,
)
from autohybts.residuals.provenance import ResidualSeries

__all__ = ["latest_window", "training_rows"]


def training_rows(
    series: ResidualSeries, window: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """``(X, y, origins)`` for the corrector; ``origins[i]`` is the forecast origin of row ``i``."""

    _check_window(window)
    targets, residuals, available = series.targets, series.residuals, series.available_at
    rows, labels, origins = [], [], []
    for i in range(window, len(series)):
        span = targets[i - window : i + 1]
        if span[-1] - span[0] != window:  # gap somewhere in the window: skip the row
            continue
        origin = int(targets[i]) - 1
        if int(np.max(available[i - window : i])) > origin:
            raise TemporalLeakageError(
                f"Janela do alvo {int(targets[i])} usa resíduo disponível só depois da origem."
            )
        rows.append(residuals[i - window : i])
        labels.append(residuals[i])
        origins.append(origin)
    if not rows:
        raise InsufficientHistoryError(
            f"Nenhuma janela contígua de {window} resíduos entre {len(series)} resíduos."
        )
    return np.vstack(rows), np.asarray(labels), np.asarray(origins, dtype=np.int64)


def latest_window(series: ResidualSeries, window: int, origin: int) -> npt.NDArray[np.float64]:
    """The ``window`` residuals ending at ``origin``, all available at ``origin``."""

    _check_window(window)
    if len(series) < window:
        raise InsufficientHistoryError(
            f"Janela de {window} resíduos pedida com apenas {len(series)} disponíveis."
        )
    targets = series.targets[-window:]
    if int(np.max(series.available_at[-window:])) > origin:
        raise TemporalLeakageError(f"Janela da origem {origin} contém resíduo posterior a ela.")
    if int(targets[-1]) != origin or int(targets[-1] - targets[0]) != window - 1:
        raise InsufficientHistoryError(
            f"Os últimos {window} resíduos não formam uma janela contígua terminando na origem "
            f"{origin}."
        )
    return np.array(series.residuals[-window:], dtype=np.float64)


def _check_window(window: int) -> None:
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ConfigurationError(f"window_size deve ser inteiro >= 1; recebido {window!r}.")
