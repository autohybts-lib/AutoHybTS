"""Forecast error metrics (architecture §17).

Every metric is lower-is-better and is computed on one origin's forecast over steps
``1..H``. A metric that is undefined for a unit (for example MASE with a zero training
scale) returns ``NaN``; the unit then counts as invalid for that metric and lowers the
coverage, instead of entering the ranking with an arbitrary value.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from autohybts.exceptions import ConfigurationError

__all__ = ["METRICS", "MIN_SCALE", "Metric", "compute_metrics", "resolve_metrics"]

FloatArray = npt.NDArray[np.float64]
# Scales at or below this are treated as zero (same threshold as the legacy project).
MIN_SCALE: Final = 1e-12


@dataclass(frozen=True, slots=True)
class Metric:
    name: str
    version: int
    formula: str
    scale_free: bool
    function: Callable[[FloatArray, FloatArray, FloatArray, int], float]

    def __call__(
        self,
        y_true: npt.ArrayLike,
        y_pred: npt.ArrayLike,
        y_train: npt.ArrayLike,
        seasonal_period: int = 1,
    ) -> float:
        actual, predicted, train = _arrays(y_true, y_pred, y_train)
        return float(self.function(actual, predicted, train, seasonal_period))


def _mae(a: FloatArray, p: FloatArray, train: FloatArray, m: int) -> float:
    return float(np.mean(np.abs(a - p)))


def _rmse(a: FloatArray, p: FloatArray, train: FloatArray, m: int) -> float:
    return float(np.sqrt(np.mean((a - p) ** 2)))


def _smape(a: FloatArray, p: FloatArray, train: FloatArray, m: int) -> float:
    denominator = np.abs(a) + np.abs(p)
    # A step where actual and forecast are both zero is a perfect forecast: error 0.
    steps = np.divide(2.0 * np.abs(a - p), denominator, out=np.zeros_like(a), where=denominator > 0)
    return float(np.mean(steps))


def _mase(a: FloatArray, p: FloatArray, train: FloatArray, m: int) -> float:
    if train.size <= m:
        return float("nan")
    scale = float(np.mean(np.abs(train[m:] - train[:-m])))
    return _ratio(_mae(a, p, train, m), scale)


def _wape(a: FloatArray, p: FloatArray, train: FloatArray, m: int) -> float:
    return _ratio(float(np.sum(np.abs(a - p))), float(np.sum(np.abs(a))))


def _rrmse(a: FloatArray, p: FloatArray, train: FloatArray, m: int) -> float:
    return _ratio(_rmse(a, p, train, m), float(np.mean(a)))


def _ratio(numerator: float, scale: float) -> float:
    return numerator / scale if scale > MIN_SCALE else float("nan")


METRICS: Final[dict[str, Metric]] = {
    metric.name: metric
    for metric in (
        Metric("MAE", 1, "mean(|y - ŷ|)", False, _mae),
        Metric("RMSE", 1, "sqrt(mean((y - ŷ)^2))", False, _rmse),
        Metric(
            "sMAPE",
            1,
            "mean(2|y - ŷ| / (|y| + |ŷ|)), fraction in [0, 2]; 0/0 counts as 0",
            True,
            _smape,
        ),
        Metric(
            "MASE",
            1,
            "MAE / mean(|y_t - y_{t-m}|) over the origin's training window, m = seasonal "
            "period; NaN if the scale is <= 1e-12 or training has <= m points",
            True,
            _mase,
        ),
        Metric("WAPE", 1, "sum(|y - ŷ|) / sum(|y|); NaN if sum(|y|) <= 1e-12", True, _wape),
        Metric(
            "RRMSE",
            1,
            "RMSE / mean(y) over steps 1..H; NaN if mean(y) <= 1e-12 (legacy parity)",
            True,
            _rrmse,
        ),
    )
}


def resolve_metrics(names: Iterable[str]) -> tuple[Metric, ...]:
    metrics = []
    for name in names:
        if name not in METRICS:
            raise ConfigurationError(f"Métrica desconhecida {name!r}; opções: {sorted(METRICS)}.")
        metrics.append(METRICS[name])
    if not metrics:
        raise ConfigurationError("Informe ao menos uma métrica.")
    if len({metric.name for metric in metrics}) != len(metrics):
        raise ConfigurationError("Métricas repetidas.")
    return tuple(metrics)


def compute_metrics(
    metrics: Iterable[Metric],
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    y_train: npt.ArrayLike,
    seasonal_period: int = 1,
) -> dict[str, float]:
    return {metric.name: metric(y_true, y_pred, y_train, seasonal_period) for metric in metrics}


def _arrays(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, y_train: npt.ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    train = np.asarray(y_train, dtype=np.float64)
    if actual.ndim != 1 or actual.shape != predicted.shape or actual.size == 0:
        raise ConfigurationError(
            f"Observado e previsto devem ser vetores do mesmo tamanho; recebidos "
            f"{actual.shape} e {predicted.shape}."
        )
    return actual, predicted, train
