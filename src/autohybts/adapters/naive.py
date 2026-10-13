"""Simple baselines: last value, last season, averages and drift (plano de ensembles F1a).

These are the controls every composed structure has to beat (§7.2, C2). Parameters
learned in ``fit`` (a mean, a drift) are frozen; everything else comes from the context.
"""

from __future__ import annotations

import numpy as np

from autohybts.adapters.base import FloatArray, ForecasterBase
from autohybts.exceptions import ConfigurationError

__all__ = [
    "HistoricAverageForecaster",
    "NaiveForecaster",
    "RandomWalkWithDriftForecaster",
    "SeasonalNaiveForecaster",
    "SeasonalWindowAverageForecaster",
    "WindowAverageForecaster",
]


class NaiveForecaster(ForecasterBase):
    """Every step equals the last observed value."""

    def _fit(self, y: FloatArray) -> None:
        # Nothing to learn: the forecast depends only on the context.
        return None

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        return np.full(horizon, history[-1], dtype=np.float64)


class SeasonalNaiveForecaster(ForecasterBase):
    """Repeats the last ``season_length`` observations cyclically."""

    def __init__(self, season_length: int = 7) -> None:
        super().__init__()
        if isinstance(season_length, bool) or not isinstance(season_length, int):
            raise ConfigurationError(f"season_length deve ser inteiro; recebido {season_length!r}.")
        if season_length < 1:
            raise ConfigurationError(f"season_length deve ser >= 1; recebido {season_length}.")
        self.season_length = season_length

    @property
    def min_train_size(self) -> int:
        return self.season_length

    @property
    def min_history(self) -> int:
        return self.season_length

    def _fit(self, y: FloatArray) -> None:
        return None

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        return np.resize(history[-self.season_length :], horizon).astype(np.float64)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{name} deve ser inteiro >= 1; recebido {value!r}.")
    return value


class HistoricAverageForecaster(ForecasterBase):
    """Every step equals the mean of the training vector (a fitted, frozen parameter)."""

    def _fit(self, y: FloatArray) -> None:
        self.mean_ = float(np.mean(y))

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        return np.full(horizon, self.mean_, dtype=np.float64)


class WindowAverageForecaster(ForecasterBase):
    """Every step equals the mean of the last ``window`` observations of the context."""

    def __init__(self, window: int = 7) -> None:
        super().__init__()
        self.window = _positive_int("window", window)

    @property
    def min_train_size(self) -> int:
        return self.window

    @property
    def min_history(self) -> int:
        return self.window

    def _fit(self, y: FloatArray) -> None:
        return None

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        return np.full(horizon, float(np.mean(history[-self.window :])), dtype=np.float64)


class SeasonalWindowAverageForecaster(ForecasterBase):
    """Each seasonal position is the mean of that position over the last ``window`` seasons."""

    def __init__(self, season_length: int = 7, window: int = 2) -> None:
        super().__init__()
        self.season_length = _positive_int("season_length", season_length)
        self.window = _positive_int("window", window)

    @property
    def min_train_size(self) -> int:
        return self.season_length * self.window

    @property
    def min_history(self) -> int:
        return self.season_length * self.window

    def _fit(self, y: FloatArray) -> None:
        return None

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        span = self.season_length * self.window
        pattern = history[-span:].reshape(self.window, self.season_length).mean(axis=0)
        return np.resize(pattern, horizon).astype(np.float64)


class RandomWalkWithDriftForecaster(ForecasterBase):
    """Last observed value plus ``h`` times the average change of the training vector.

    The drift ``(y[-1] - y[0]) / (n - 1)`` is a fitted parameter, frozen afterwards; the
    starting point is the last value of the context.
    """

    @property
    def min_train_size(self) -> int:
        return 2

    def _fit(self, y: FloatArray) -> None:
        self.drift_ = float((y[-1] - y[0]) / (y.size - 1))

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        steps = np.arange(1, horizon + 1, dtype=np.float64)
        return np.asarray(history[-1] + self.drift_ * steps, dtype=np.float64)
