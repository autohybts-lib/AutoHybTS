"""Nixtla StatsForecast models: AutoARIMA, AutoETS, AutoTheta and AutoCES (plano F1d).

Each model is fitted on the training vector with ``statsforecast.models.<Model>.fit`` and
then frozen: forecasts call ``forward(y=context, h)``, which applies the fitted model
(orders, coefficients, smoothing constants) to the observed context without re-estimating
it. On the training vector ``forward`` reproduces ``predict`` exactly; on a longer
context it updates the states only (tested), so a fitted model forecasts from later
origins without refitting (variant B, out-of-fold residuals).

StatsForecast is an optional extra (``nixtla-stats``) and is imported only when a model
is fitted. Its current releases require ``pandas<3``: installing the extra brings
pandas 2.x, which AutoHybTS supports (floor 2.1). The ``sf_`` names keep these models
apart from ``arima`` (statsmodels), whose parity proofs must not change.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, Final

import numpy as np

from autohybts.adapters.base import FloatArray, ForecasterBase
from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError

__all__ = ["STATSFORECAST_MODELS", "StatsForecastForecaster"]

# registry name -> class in statsforecast.models
STATSFORECAST_MODELS: Final[dict[str, str]] = {
    "sf_auto_arima": "AutoARIMA",
    "sf_auto_ets": "AutoETS",
    "sf_auto_theta": "AutoTheta",
    "sf_auto_ces": "AutoCES",
}


class StatsForecastForecaster(ForecasterBase):
    """One StatsForecast model with frozen parameters; ``model`` is a registry name."""

    def __init__(
        self,
        model: str = "sf_auto_ets",
        season_length: int = 1,
        model_params: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if model not in STATSFORECAST_MODELS:
            options = sorted(STATSFORECAST_MODELS)
            raise ConfigurationError(
                f"Modelo StatsForecast desconhecido {model!r}; opções: {options}."
            )
        if isinstance(season_length, bool) or not isinstance(season_length, int):
            raise ConfigurationError(f"season_length deve ser inteiro; recebido {season_length!r}.")
        if season_length < 1:
            raise ConfigurationError("season_length deve ser >= 1.")
        self.model = model
        self.season_length = season_length
        self.model_params = model_params

    @property
    def min_train_size(self) -> int:
        return max(10, 2 * self.season_length + 2)

    def _fit(self, y: FloatArray) -> None:
        try:
            module = importlib.import_module("statsforecast.models")
        except ModuleNotFoundError as error:
            raise UnsupportedCapabilityError(
                f"{self.model} requer o StatsForecast: "
                'pip install "autohybts[nixtla-stats]" (traz pandas 2.x).'
            ) from error
        cls = getattr(module, STATSFORECAST_MODELS[self.model])
        params = {"season_length": self.season_length, **dict(self.model_params or {})}
        try:
            estimator = cls(**params)
        except TypeError as error:
            raise ConfigurationError(f"Parâmetro inválido para {self.model}: {error}") from error
        self.model_ = estimator.fit(np.asarray(y, dtype=np.float64))

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        forecast = self.model_.forward(y=np.asarray(history, dtype=np.float64), h=horizon)
        return np.asarray(forecast["mean"], dtype=np.float64)
