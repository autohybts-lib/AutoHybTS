"""Bring your own model without subclassing: two functions, or any scikit-learn-like regressor.

``CustomForecaster(fit_fn, predict_fn)``: ``fit_fn(y, **params)`` receives the training
vector of an origin (numpy, oldest first) and returns any state; ``predict_fn(state,
horizon, history, **params)`` returns ``horizon`` forecasts after the last value of
``history``. ``history`` starts with the training vector and may be longer (new
observations in variant B, earlier contexts for out-of-fold residuals): use it to update
the inputs, never to refit.

``RegressorForecaster(estimator)`` puts any object with ``fit(X, y)``/``predict(X)``
(LightGBM, XGBoost, CatBoost, scikit-learn) on lag windows, with the same scaling,
differencing and recursive forecasting as the built-in lag models.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from autohybts.adapters.base import FloatArray, ForecasterBase
from autohybts.exceptions import ConfigurationError
from autohybts.models.lagged import LaggedModelForecaster
from autohybts.models.requirements import require

__all__ = ["CustomForecaster", "RegressorForecaster"]


class CustomForecaster(ForecasterBase):
    def __init__(
        self,
        fit_fn: Callable[..., Any],
        predict_fn: Callable[..., Any],
        params: Mapping[str, Any] | None = None,
        min_train_size: int = 2,
        min_history: int = 1,
        requires: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        if not callable(fit_fn) or not callable(predict_fn):
            raise ConfigurationError("fit_fn e predict_fn devem ser funções.")
        self.fit_fn = fit_fn
        self.predict_fn = predict_fn
        self.params = params
        self._min_train_size = int(min_train_size)
        self._min_history = int(min_history)
        self.requires = requires

    # Exposed under the same names as the arguments, as get_params/clone expect.
    @property
    def min_train_size(self) -> int:
        return self._min_train_size

    @property
    def min_history(self) -> int:
        return self._min_history

    def _fit(self, y: FloatArray) -> None:
        require(self.requires)
        self.state_ = self.fit_fn(y.copy(), **dict(self.params or {}))

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        out = self.predict_fn(self.state_, horizon, history.copy(), **dict(self.params or {}))
        return np.asarray(out, dtype=np.float64).reshape(-1)


class RegressorForecaster(LaggedModelForecaster):
    """Any regressor with ``fit(X, y)``/``predict(X)`` on lag windows (oldest lag first)."""

    def __init__(
        self,
        estimator: Any,
        lags: int | tuple[int, ...] = 14,
        differencing: int | str = 0,
        scale: bool = True,
        requires: Mapping[str, str] | None = None,
        seasonal_differencing: int = 0,
        season_length: int = 1,
        differencing_test: str = "kpss",
    ) -> None:
        if not (
            callable(getattr(estimator, "fit", None))
            and callable(getattr(estimator, "predict", None))
        ):
            raise ConfigurationError("estimator precisa de fit(X, y) e predict(X).")
        super().__init__(
            lags=lags,
            differencing=differencing,
            seasonal_differencing=seasonal_differencing,
            season_length=season_length,
            differencing_test=differencing_test,
            scale=scale,
            validation_fraction=0.0,
            requires=requires,
        )
        self.estimator = estimator

    def _train(
        self,
        x_train: FloatArray,
        y_train: FloatArray,
        x_val: FloatArray | None,
        y_val: FloatArray | None,
    ) -> dict[str, float]:
        self.model_ = copy.deepcopy(self.estimator)
        self.model_.fit(x_train, y_train)
        return {}

    def _infer(self, x: FloatArray) -> FloatArray:
        return np.asarray(self.model_.predict(x), dtype=np.float64).reshape(-1)
