"""Additive exponential smoothing (Holt, Holt-Winters, damped) with frozen parameters.

Parameters (smoothing constants, damping and initial states) are estimated with
``statsmodels.tsa.holtwinters.ExponentialSmoothing`` on the training vector only. Forecasts
run the same error-correction recursion over the context with those parameters frozen, so
a fitted model forecasts from any later origin without refitting (variant B, out-of-fold
residuals); on the training vector the recursion reproduces statsmodels' fitted values and
forecasts exactly (tested).

Recursion (additive trend and season, period ``m``, damping ``φ``)::

    ℓ_t = α (y_t − s_{t−m}) + (1 − α)(ℓ_{t−1} + φ b_{t−1})
    b_t = β (ℓ_t − ℓ_{t−1}) + (1 − β) φ b_{t−1}
    s_t = γ (y_t − ℓ_{t−1} − φ b_{t−1}) + (1 − γ) s_{t−m}
    ŷ_{t+h} = ℓ_t + (φ + … + φ^h) b_t + s_{t+h−m(⌊(h−1)/m⌋+1)}
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from autohybts.adapters.base import FloatArray, ForecasterBase
from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError

__all__ = ["ExponentialSmoothingForecaster"]


def _value(params: dict[str, object], name: str, default: float) -> float:
    value = params.get(name)
    if value is None:
        return default
    number = float(value)  # type: ignore[arg-type]
    return default if math.isnan(number) else number


class ExponentialSmoothingForecaster(ForecasterBase):
    """Holt (``trend=True``), Holt-Winters (``season_length > 1``) or simple smoothing."""

    def __init__(self, trend: bool = True, damped: bool = False, season_length: int = 1) -> None:
        super().__init__()
        if isinstance(season_length, bool) or not isinstance(season_length, int):
            raise ConfigurationError(f"season_length deve ser inteiro; recebido {season_length!r}.")
        if season_length < 1:
            raise ConfigurationError("season_length deve ser >= 1.")
        if damped and not trend:
            raise ConfigurationError("damped exige trend=True.")
        self.trend = trend
        self.damped = damped
        self.season_length = season_length

    @property
    def min_train_size(self) -> int:
        return max(10, 2 * self.season_length + 2)

    def _fit(self, y: FloatArray) -> None:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ModuleNotFoundError as error:
            raise UnsupportedCapabilityError(
                'exponential_smoothing requer o extra `stats`: pip install "autohybts[stats]".'
            ) from error
        seasonal = self.season_length > 1
        model = ExponentialSmoothing(
            y,
            trend="add" if self.trend else None,
            damped_trend=self.damped,
            seasonal="add" if seasonal else None,
            seasonal_periods=self.season_length if seasonal else None,
            initialization_method="estimated",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = dict(model.fit().params)
        self.alpha_ = _value(params, "smoothing_level", 0.0)
        self.beta_ = _value(params, "smoothing_trend", 0.0) if self.trend else 0.0
        self.gamma_ = _value(params, "smoothing_seasonal", 0.0) if seasonal else 0.0
        self.phi_ = _value(params, "damping_trend", 1.0) if self.damped else 1.0
        self.level0_ = _value(params, "initial_level", float(y[0]))
        self.trend0_ = _value(params, "initial_trend", 0.0) if self.trend else 0.0
        seasons = params.get("initial_seasons")
        self.seasons0_ = np.asarray(seasons, dtype=np.float64).copy() if seasonal else np.zeros(1)

    def _states(self, history: FloatArray) -> tuple[float, float, FloatArray]:
        """Level, trend and seasonal states after the last value of ``history``."""

        m = max(self.season_length, 1)
        level, slope, seasons = self.level0_, self.trend0_, self.seasons0_.copy()
        for index, value in enumerate(history):
            season = seasons[index % m] if m > 1 else 0.0
            damped = self.phi_ * slope
            new_level = self.alpha_ * (value - season) + (1 - self.alpha_) * (level + damped)
            if self.trend:
                slope = self.beta_ * (new_level - level) + (1 - self.beta_) * damped
            if m > 1:
                seasons[index % m] = (
                    self.gamma_ * (value - level - damped) + (1 - self.gamma_) * season
                )
            level = new_level
        return level, slope, seasons

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        level, slope, seasons = self._states(history)
        m = max(self.season_length, 1)
        damping = np.cumsum(self.phi_ ** np.arange(1, horizon + 1))
        n = history.size
        season = (
            np.array([seasons[(n + k) % m] for k in range(horizon)]) if m > 1 else np.zeros(horizon)
        )
        return np.asarray(level + damping * slope + season, dtype=np.float64)
