"""Naive baselines: repeat the last value, or the last season."""

from __future__ import annotations

import numpy as np

from autohybts.adapters.base import FloatArray, ForecasterBase
from autohybts.exceptions import ConfigurationError

__all__ = ["NaiveForecaster", "SeasonalNaiveForecaster"]


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
