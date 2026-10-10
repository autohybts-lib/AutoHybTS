"""scikit-learn regressors on causal lag windows, forecasting multi-step recursively."""

from __future__ import annotations

import copy
import importlib
import warnings
from collections.abc import Mapping
from typing import Any, Final

import numpy as np

from autohybts.adapters.base import FloatArray, ForecasterBase
from autohybts.data.preprocessing import DifferenceTransform
from autohybts.data.windows import lag_matrix, latest_lags, normalize_lags
from autohybts.exceptions import ConfigurationError

__all__ = ["ESTIMATORS", "LagRegressionForecaster", "build_estimator", "fit_estimator"]

# name -> (module, class); imported only when a model is fitted.
ESTIMATORS: Final[dict[str, tuple[str, str]]] = {
    "linear": ("sklearn.linear_model", "LinearRegression"),
    "knn": ("sklearn.neighbors", "KNeighborsRegressor"),
    "svm": ("sklearn.svm", "SVR"),
    "tree": ("sklearn.tree", "DecisionTreeRegressor"),
    "random_forest": ("sklearn.ensemble", "RandomForestRegressor"),
    "mlp": ("sklearn.neural_network", "MLPRegressor"),
}
DEFAULT_ESTIMATOR_PARAMS: Final[dict[str, dict[str, Any]]] = {
    "mlp": {"hidden_layer_sizes": (32,), "max_iter": 1000},
}
SEEDED: Final = frozenset({"tree", "random_forest", "mlp"})


class LagRegressionForecaster(ForecasterBase):
    """Regressor trained on ``[y[t - lag] for lag in lags] -> y[t]``.

    Optional differencing and standardization are fitted on the training vector only.
    At prediction time the differencing anchors come from the context (observed values),
    while the learned scale and estimator stay frozen. Multi-step forecasts are recursive:
    each prediction becomes a lag for the next step.
    """

    def __init__(
        self,
        estimator: str = "linear",
        lags: int | tuple[int, ...] = 7,
        differencing: int = 0,
        scale: bool = True,
        estimator_params: Mapping[str, Any] | None = None,
        random_state: int | None = 0,
    ) -> None:
        super().__init__()
        if estimator not in ESTIMATORS:
            raise ConfigurationError(
                f"Estimador desconhecido {estimator!r}; opções: {sorted(ESTIMATORS)}."
            )
        if isinstance(differencing, bool) or not isinstance(differencing, int) or differencing < 0:
            raise ConfigurationError(
                f"differencing deve ser inteiro >= 0; recebido {differencing!r}."
            )
        self._lags = normalize_lags(lags)
        self.estimator = estimator
        self.lags = lags
        self.differencing = differencing
        self.scale = scale
        self.estimator_params = estimator_params
        self.random_state = random_state

    @property
    def min_history(self) -> int:
        return max(self._lags) + self.differencing

    @property
    def min_train_size(self) -> int:
        return self.min_history + 1

    def _fit(self, y: FloatArray) -> None:
        level = self._differenced(y)
        self.center_ = float(np.mean(level)) if self.scale else 0.0
        spread = float(np.std(level)) if self.scale else 1.0
        self.spread_ = spread if spread > 0 else 1.0
        features, targets = lag_matrix(self._standardized(level), self._lags)
        self.model_ = build_estimator(self.estimator, self.estimator_params, self.random_state)
        self.fit_warnings_ = fit_estimator(self.model_, features, targets)

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        anchors = None
        level = history
        if self.differencing:
            anchors, level = DifferenceTransform(self.differencing).fit_transform(history)
        window = list(self._standardized(level))
        steps = []
        for _ in range(horizon):
            row = latest_lags(np.asarray(window), self._lags)
            step = float(self.model_.predict(row.reshape(1, -1))[0])
            window.append(step)
            steps.append(step)
        forecast = np.asarray(steps) * self.spread_ + self.center_
        return forecast if anchors is None else anchors.inverse_forecast(forecast)

    def _differenced(self, y: FloatArray) -> FloatArray:
        if not self.differencing:
            return y
        return DifferenceTransform(self.differencing).fit_transform(y)[1]

    def _standardized(self, values: FloatArray) -> FloatArray:
        return (values - self.center_) / self.spread_


def build_estimator(
    name: str, estimator_params: Mapping[str, Any] | None, random_state: int | None
) -> Any:
    """Unfitted scikit-learn estimator; seeded estimators get ``random_state`` unless overridden."""

    if name not in ESTIMATORS:
        raise ConfigurationError(f"Estimador desconhecido {name!r}; opções: {sorted(ESTIMATORS)}.")
    module_name, class_name = ESTIMATORS[name]
    cls = getattr(importlib.import_module(module_name), class_name)
    params = {
        **copy.deepcopy(DEFAULT_ESTIMATOR_PARAMS.get(name, {})),
        **dict(estimator_params or {}),
    }
    if name in SEEDED:
        params.setdefault("random_state", random_state)
    try:
        return cls(**params)
    except TypeError as error:
        raise ConfigurationError(f"Parâmetro inválido para {name}: {error}") from error


def fit_estimator(model: Any, features: FloatArray, targets: FloatArray) -> list[str]:
    """Fit ``model`` and return its warnings (e.g. MLP non-convergence) instead of emitting them.

    The warnings are kept on the fitted object for auditing; printing one per trial and
    origin would flood the notebook without changing the outcome.
    """

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(features, targets)
    return [f"{w.category.__name__}: {w.message}" for w in caught]
