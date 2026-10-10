"""Decomposition hybrid: one model for the smooth part, another for the volatile part (§16).

``y = s + v``. The base forecasts ``s``; a corrector forecasts ``v`` from windows of its
past values, recursively in variant A (only its own predictions enter the window). The
forecast is the sum of the two paths.

* ``ma``: causal moving average, ``s[t]`` uses ``y`` up to ``t`` only (the window expands
  at the start). A centred average would use future values and is not offered. The
  window comes from the training vector alone: the one whose volatile part has kurtosis
  closest to 3 (Babu and Reddy, 2014, as in the legacy project). Being causal, it is
  recomputed on any context, so new observations reach the frozen models (variant B).
* ``stl``: robust STL fitted on the training vector only. It is not recomputed on later
  observations, so it serves variant A only (forecasting from the end of the training).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from autohybts.adapters.base import BaseForecaster, FloatArray, ForecasterBase
from autohybts.exceptions import (
    ConfigurationError,
    InsufficientHistoryError,
    UnsupportedCapabilityError,
)
from autohybts.residuals.correctors import ResidualCorrector
from autohybts.residuals.generation import local_context
from autohybts.residuals.recursive import recursive_correction

__all__ = [
    "DecompositionForecaster",
    "causal_moving_average",
    "kurtosis_window",
    "stl_smooth",
]


def causal_moving_average(y: FloatArray, window: int) -> FloatArray:
    """``out[t] = mean(y[max(0, t - window + 1) : t + 1])``: no value after ``t`` is used."""

    if isinstance(window, bool) or not isinstance(window, int | np.integer) or window < 1:
        raise ConfigurationError(
            f"Janela da média móvel deve ser inteiro >= 1; recebido {window!r}."
        )
    cumulative = np.concatenate(([0.0], np.cumsum(y, dtype=np.float64)))
    index = np.arange(1, y.size + 1)
    start = np.maximum(index - int(window), 0)
    return np.asarray((cumulative[index] - cumulative[start]) / (index - start), dtype=np.float64)


def kurtosis_window(y: FloatArray, max_window: int) -> int:
    """Window in ``2..min(max_window, len(y) // 2)`` whose ``y - ma`` has kurtosis closest to 3."""

    limit = max(2, min(int(max_window), y.size // 2))
    gaps = np.array(
        [abs(_kurtosis(y - causal_moving_average(y, w)) - 3.0) for w in range(2, limit + 1)]
    )
    if not np.isfinite(gaps).any():  # constant volatile parts: any window is as good
        return 2
    return int(np.nanargmin(gaps)) + 2


def stl_smooth(y: FloatArray, period: int) -> FloatArray:
    """Trend plus seasonal part of a robust STL fitted on ``y`` only."""

    try:
        from statsmodels.tsa.seasonal import STL
    except ModuleNotFoundError as error:
        raise UnsupportedCapabilityError(
            'A decomposição STL requer o extra `stats`: pip install "autohybts[stats]".'
        ) from error
    result = STL(y, period=int(period), robust=True).fit()
    return np.asarray(result.trend + result.seasonal, dtype=np.float64)


class DecompositionForecaster(ForecasterBase):
    def __init__(
        self,
        smooth: BaseForecaster,
        volatile: ResidualCorrector,
        window_size: int = 7,
        method: str = "ma",
        ma_window: int | None = None,
        max_window: int = 365,
        period: int = 7,
        min_corrector_rows: int = 10,
    ) -> None:
        super().__init__()
        if not isinstance(smooth, BaseForecaster):
            raise ConfigurationError("smooth deve cumprir o contrato BaseForecaster.")
        if not isinstance(volatile, ResidualCorrector):
            raise ConfigurationError("volatile deve cumprir o contrato ResidualCorrector.")
        if method not in ("ma", "stl"):
            raise ConfigurationError(f"method deve ser 'ma' ou 'stl'; recebido {method!r}.")
        for name, value, minimum in (
            ("window_size", window_size, 1),
            ("max_window", max_window, 2),
            ("period", period, 2),
            ("min_corrector_rows", min_corrector_rows, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ConfigurationError(
                    f"{name} deve ser inteiro >= {minimum}; recebido {value!r}."
                )
        if ma_window is not None and (
            isinstance(ma_window, bool) or not isinstance(ma_window, int) or ma_window < 1
        ):
            raise ConfigurationError("ma_window deve ser inteiro >= 1 ou None.")
        self.smooth = smooth
        self.volatile = volatile
        self.window_size = window_size
        self.method = method
        self.ma_window = ma_window
        self.max_window = max_window
        self.period = period
        self.min_corrector_rows = min_corrector_rows

    @property
    def min_train_size(self) -> int:
        rows = self.window_size + self.min_corrector_rows
        decomposition = 2 * self.period + 1 if self.method == "stl" else 4
        return max(int(getattr(self.smooth, "min_train_size", 1)), rows, decomposition)

    @property
    def min_history(self) -> int:
        smooth = self.smooth_ if self.is_fitted else self.smooth
        return max(int(getattr(smooth, "min_history", 1)), self.window_size)

    def _fit(self, y: FloatArray) -> None:
        if self.method == "ma":
            self.ma_window_ = self.ma_window or kurtosis_window(y, self.max_window)
            smooth = causal_moving_average(y, self.ma_window_)
        else:
            smooth = stl_smooth(y, self.period)
        volatile = y - smooth
        rows = sliding_window_view(volatile, self.window_size + 1)
        if rows.shape[0] < self.min_corrector_rows:
            raise InsufficientHistoryError(
                f"Apenas {rows.shape[0]} linhas para o modelo da parte volátil (mínimo "
                f"{self.min_corrector_rows})."
            )
        windows, targets = rows[:, :-1], rows[:, -1]
        self.smooth_ = self.smooth.clone().fit(smooth)
        self.volatile_ = self.volatile.clone().fit(windows, targets)
        self.train_ = y.copy()
        self.smooth_train_ = smooth
        scale = float(np.std(targets))
        fitted = np.array([self.volatile_.predict_next(row) for row in windows])
        self.correction_std_ratio_ = float(np.std(fitted)) / scale if scale > 1e-12 else np.nan

    def diagnostics(self) -> dict[str, float]:
        values = {"correction_std_ratio": float(getattr(self, "correction_std_ratio_", np.nan))}
        if self.method == "ma" and hasattr(self, "ma_window_"):
            values["ma_window"] = float(self.ma_window_)
        return values

    def parts(self, history: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Smooth and volatile parts of ``history`` as known at its last observation."""

        if self.method == "ma":
            smooth = causal_moving_average(history, self.ma_window_)
            return smooth, history - smooth
        n = self.train_.size
        if history.size < n or not np.array_equal(history[-n:], self.train_):
            raise UnsupportedCapabilityError(
                "A STL foi ajustada só no treino e não incorpora observações novas: prevê "
                "apenas a partir do fim do treino (variante A)."
            )
        return self.smooth_train_, self.train_ - self.smooth_train_

    def predict_components(
        self, horizon: int, history: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Paths of the smooth and the volatile part; the forecast is their sum."""

        smooth, volatile = self.parts(history)
        smooth_path = self.smooth_.predict(horizon, local_context(smooth, smooth.size))
        window = np.array(volatile[-self.window_size :], dtype=np.float64)
        return smooth_path, recursive_correction(self.volatile_, window, horizon)

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        smooth_path, volatile_path = self.predict_components(horizon, history)
        return smooth_path + volatile_path


def _kurtosis(values: FloatArray) -> float:
    """Pearson kurtosis (3 for a normal distribution); NaN for a constant vector."""

    centred = values - float(np.mean(values))
    second = float(np.mean(centred**2))
    if second <= 1e-24:
        return float("nan")
    return float(np.mean(centred**4)) / second**2
