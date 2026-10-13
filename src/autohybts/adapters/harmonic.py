"""Dynamic harmonic regression: several seasonal periods at once (plano de ensembles L06).

The representative of multiple seasonality. One least-squares fit on the training vector::

    y_t = a + b·t + Σ_m Σ_{k≤K_m} [c_mk sin(2πkt/m) + d_mk cos(2πkt/m)] + Σ_{i≤p} φ_i y_{t−i} + ε_t

``periods`` may be non-integer (365.25 for yearly seasonality in daily data) and each one
gets ``K_m`` Fourier pairs, so a long period costs ``2·K_m`` coefficients instead of ``m``
seasonal states. Coefficients are frozen after ``fit``; forecasts run the regression from
the context, feeding predicted values to the autoregressive lags. The phase ``t`` is the
position in the context, which starts where the training vector started (the contract of
``TemporalContext``), so a fitted model forecasts from any later origin without refitting.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from autohybts.adapters.base import FloatArray, ForecasterBase
from autohybts.exceptions import ConfigurationError

__all__ = ["HarmonicRegressionForecaster"]


class HarmonicRegressionForecaster(ForecasterBase):
    """Trend + Fourier terms for each period in ``periods`` + ``ar_lags`` autoregressive lags."""

    def __init__(
        self,
        periods: Sequence[float] = (7.0, 365.25),
        harmonics: Sequence[int] = (3, 5),
        trend: bool = True,
        ar_lags: int = 1,
        ridge: float = 1e-6,
    ) -> None:
        super().__init__()
        periods = tuple(float(p) for p in periods)
        harmonics = tuple(harmonics)
        if not periods or len(periods) != len(harmonics):
            raise ConfigurationError("periods e harmonics devem ter o mesmo tamanho (>= 1).")
        for period, count in zip(periods, harmonics, strict=True):
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ConfigurationError(f"harmonics deve ter inteiros >= 1; recebido {count!r}.")
            if period <= 2 or 2 * count > period:
                raise ConfigurationError(
                    f"Período {period} com {count} harmônicos: exige período > 2 e "
                    "2·harmônicos <= período."
                )
        if isinstance(ar_lags, bool) or not isinstance(ar_lags, int) or ar_lags < 0:
            raise ConfigurationError(f"ar_lags deve ser inteiro >= 0; recebido {ar_lags!r}.")
        if ridge < 0:
            raise ConfigurationError("ridge deve ser >= 0.")
        self.periods = periods
        self.harmonics = harmonics
        self.trend = trend
        self.ar_lags = ar_lags
        self.ridge = ridge

    @property
    def n_features(self) -> int:
        return 1 + int(self.trend) + 2 * sum(self.harmonics) + self.ar_lags

    @property
    def min_train_size(self) -> int:
        return self.ar_lags + 2 * self.n_features

    @property
    def min_history(self) -> int:
        return max(1, self.ar_lags)

    def _deterministic(self, t: FloatArray) -> FloatArray:
        """Intercept, scaled trend and Fourier columns at positions ``t``."""

        columns = [np.ones_like(t)]
        if self.trend:
            columns.append(t / self.scale_)
        for period, count in zip(self.periods, self.harmonics, strict=True):
            for k in range(1, count + 1):
                angle = 2 * np.pi * k * t / period
                columns.extend([np.sin(angle), np.cos(angle)])
        return np.column_stack(columns)

    def _fit(self, y: FloatArray) -> None:
        self.scale_ = float(y.size)  # frozen: the trend keeps the training's unit
        p = self.ar_lags
        t = np.arange(p, y.size, dtype=np.float64)
        lagged = [y[p - i : y.size - i] for i in range(1, p + 1)]
        design = np.column_stack([self._deterministic(t), *lagged]) if p else self._deterministic(t)
        # Small ridge on everything but the intercept keeps near-collinear terms stable.
        penalty = np.full(design.shape[1], self.ridge)
        penalty[0] = 0.0
        gram = design.T @ design + np.diag(penalty * len(t))
        self.coef_ = np.linalg.solve(gram, design.T @ y[p:])

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        n = history.size
        t = np.arange(n, n + horizon, dtype=np.float64)
        base = self._deterministic(t) @ self.coef_[: self.n_features - self.ar_lags]
        if not self.ar_lags:
            return np.asarray(base, dtype=np.float64)
        phi = self.coef_[self.n_features - self.ar_lags :]
        path = list(history[-self.ar_lags :])
        forecast = np.empty(horizon)
        for step in range(horizon):
            recent = np.asarray(path[-1 : -self.ar_lags - 1 : -1])
            forecast[step] = base[step] + float(phi @ recent)
            path.append(forecast[step])
        return forecast
