"""Intermittent-demand methods: Croston-SBA, TSB, ADIDA and IMAPA (plano de ensembles L08).

They forecast series with many zeros (sparse demand) and form their own family,
``intermittent``: they are base forecasters, not a combination of other models (IMAPA
averages its own aggregation levels, it is not an ensemble of candidates).

* **Croston-SBA** smooths the non-zero demand sizes ``z`` and the intervals between them
  ``p`` separately and forecasts ``(1 − α/2) · z / p`` (the Syntetos-Boylan correction of
  Croston's bias).
* **TSB** (Teunter-Syntetos-Babai) smooths the demand probability every period and the
  size only when there is demand; it forecasts ``p · z`` and decays towards zero when
  demand stops (Croston keeps the last rate forever).
* **ADIDA** sums the series in non-overlapping buckets of ``k`` periods (``k`` = mean
  interval between demands in the training vector), smooths the buckets and spreads the
  bucket forecast evenly: ``ℓ / k``.
* **IMAPA** averages the ADIDA forecasts of every level ``1..k``.

Smoothing constants are fixed parameters (0.1 by default, as in the literature) and the
aggregation level is learned on the training vector and then frozen; the recursions run
over the context, so a fitted model forecasts from any later origin without refitting
(variant B, out-of-fold residuals). Forecasts are flat over the horizon and never negative.
Series with negative values are not demand: ``fit`` refuses them (the trial is SKIPPED).
"""

from __future__ import annotations

import math

import numpy as np

from autohybts.adapters.base import FloatArray, ForecasterBase
from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError

__all__ = [
    "AdidaForecaster",
    "CrostonSBAForecaster",
    "ImapaForecaster",
    "TSBForecaster",
    "demand_profile",
]


def _smoothing(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{name} deve ser numérico; recebido {value!r}.")
    if not 0.0 < float(value) <= 1.0:
        raise ConfigurationError(f"{name} deve estar em (0, 1]; recebido {value!r}.")
    return float(value)


def _ses_level(values: FloatArray, alpha: float) -> float:
    """Level of simple exponential smoothing after the last value (starts at the first)."""

    level = float(values[0])
    for value in values[1:]:
        level += alpha * (float(value) - level)
    return level


def _intervals(history: FloatArray) -> FloatArray:
    """Periods between consecutive demands; the first counts from the start of the series."""

    positions = np.flatnonzero(history > 0)
    return np.diff(np.concatenate(([-1], positions))).astype(np.float64)


def demand_profile(y: FloatArray) -> dict[str, float]:
    """Share of zeros, mean demand interval (ADI) and squared CV of non-zero sizes.

    ADI > 1.32 or CV² > 0.49 are the Syntetos-Boylan cut-offs between smooth, erratic,
    intermittent and lumpy demand; they describe the series, they do not gate the models.
    """

    values = np.asarray(y, dtype=np.float64)
    sizes = values[values > 0]
    adi = float(np.mean(_intervals(values))) if sizes.size else math.nan
    cv2 = float(np.var(sizes) / np.mean(sizes) ** 2) if sizes.size > 1 else math.nan
    return {"zero_share": float(np.mean(values == 0)), "adi": adi, "cv2": cv2}


class _DemandForecaster(ForecasterBase):
    """Refuses negative values and records the demand profile of the training vector."""

    def _check_demand(self, y: FloatArray) -> None:
        if (y < 0).any():
            raise UnsupportedCapabilityError(
                f"{type(self).__name__} é para demanda (valores >= 0); o treino tem negativos."
            )
        self.profile_ = demand_profile(y)

    def diagnostics(self) -> dict[str, float]:
        profile: dict[str, float] = getattr(self, "profile_", {})
        return {name: value for name, value in profile.items() if math.isfinite(value)}


class CrostonSBAForecaster(_DemandForecaster):
    """Croston with the Syntetos-Boylan approximation: ``(1 − α/2) · z / p``."""

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha = _smoothing("alpha", alpha)

    def _fit(self, y: FloatArray) -> None:
        self._check_demand(y)

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        sizes = history[history > 0]
        if sizes.size == 0:
            return np.zeros(horizon)
        size = _ses_level(sizes, self.alpha)
        interval = _ses_level(_intervals(history), self.alpha)
        rate = (1.0 - self.alpha / 2.0) * size / interval
        return np.full(horizon, rate, dtype=np.float64)


class TSBForecaster(_DemandForecaster):
    """Teunter-Syntetos-Babai: demand probability smoothed every period, size on demand."""

    def __init__(self, alpha_d: float = 0.1, alpha_p: float = 0.1) -> None:
        super().__init__()
        self.alpha_d = _smoothing("alpha_d", alpha_d)
        self.alpha_p = _smoothing("alpha_p", alpha_p)

    def _fit(self, y: FloatArray) -> None:
        self._check_demand(y)

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        sizes = history[history > 0]
        if sizes.size == 0:
            return np.zeros(horizon)
        probability = float(history[0] > 0)
        size = float(sizes[0])
        for value in history[1:]:
            occurred = value > 0
            probability += self.alpha_p * (float(occurred) - probability)
            if occurred:
                size += self.alpha_d * (float(value) - size)
        return np.full(horizon, probability * size, dtype=np.float64)


class _AggregatingForecaster(_DemandForecaster):
    """Learns the aggregation level ``k`` (mean demand interval, rounded) on the training."""

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha = _smoothing("alpha", alpha)

    def _fit(self, y: FloatArray) -> None:
        self._check_demand(y)
        adi = self.profile_["adi"]
        self.aggregation_ = 1 if math.isnan(adi) else max(1, round(adi))

    def _level(self, history: FloatArray, k: int) -> float:
        """ADIDA forecast per period at level ``k``; buckets end at the last observation."""

        usable = (history.size // k) * k
        if usable == 0:
            return float(np.mean(history))
        buckets = history[history.size - usable :].reshape(-1, k).sum(axis=1)
        return _ses_level(buckets, self.alpha) / k


class AdidaForecaster(_AggregatingForecaster):
    """Aggregate–disaggregate: smoothing on buckets of ``k`` periods, spread evenly."""

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        return np.full(horizon, self._level(history, self.aggregation_), dtype=np.float64)


class ImapaForecaster(_AggregatingForecaster):
    """Mean of the ADIDA forecasts over aggregation levels ``1..k``."""

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        levels = [self._level(history, k) for k in range(1, self.aggregation_ + 1)]
        return np.full(horizon, float(np.mean(levels)), dtype=np.float64)
