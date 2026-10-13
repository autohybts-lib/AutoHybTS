"""Residual corrector contract (§5.2) and scikit-learn correctors (§8.1)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt

from autohybts.adapters.base import FloatArray, Parametrized, candidate_errors
from autohybts.adapters.sklearn_regression import (
    PER_COLUMN,
    build_estimator,
    check_scale,
    column_scale,
    fit_estimator,
    scalar_scale,
    scale_values,
    single_threaded,
    unscale_values,
)
from autohybts.domain.fit_count import record_fit
from autohybts.exceptions import ConfigurationError, DataValidationError

__all__ = ["CORRECTOR_ESTIMATORS", "ResidualCorrector", "SklearnCorrector"]

FEATURE_ORDERS: Final = ("chronological", "lag")
# linear, knn and mlp are the protocol presets (§8.1); the others are opt-in correctors
# (plano de ensembles F1b/F1c).
CORRECTOR_ESTIMATORS: Final = (
    "linear",
    "knn",
    "mlp",
    "svm",
    "tree",
    "random_forest",
    "hist_gradient_boosting",
    "adaboost_r2",
    "lightgbm_regressor",
    "xgboost_regressor",
    "catboost_regressor",
)


@runtime_checkable
class ResidualCorrector(Protocol):
    """Learns the next residual from a window of past residuals."""

    def fit(self, windows: npt.ArrayLike, targets: npt.ArrayLike) -> Self: ...

    def predict_next(self, window: npt.ArrayLike) -> float: ...

    def clone(self) -> Self: ...

    def get_params(self) -> dict[str, Any]: ...


class SklearnCorrector(Parametrized):
    """scikit-learn regressor on residual windows, standardized by the training residuals.

    Regularization is set through ``estimator_params`` (e.g. ``{"alpha": 1e-3}`` for the
    MLP's L2 penalty); keep correctors small on short series (§30.2).

    ``scale=True`` uses one mean and spread (those of the target residuals) for windows and
    targets; ``scale="columns"`` standardizes each window position and the target
    separately, like scikit-learn's ``StandardScaler`` on X and y, and ``"minmax"`` maps
    each of them to [0, 1] like ``MinMaxScaler``.

    ``feature_order`` is the column order the estimator sees: ``"chronological"`` (oldest
    residual first, the default) or ``"lag"`` (most recent first, lag 1, 2, ...). Linear
    models and KNN do not depend on it; estimators with random initialization (MLP) do,
    so ``"lag"`` reproduces correctors trained on lag-ordered features.
    """

    def __init__(
        self,
        estimator: str = "linear",
        scale: bool | str = True,
        estimator_params: Mapping[str, Any] | None = None,
        random_state: int | None = 0,
        feature_order: str = "chronological",
    ) -> None:
        if estimator not in CORRECTOR_ESTIMATORS:
            raise ConfigurationError(
                f"Corretor desconhecido {estimator!r}; opções: {CORRECTOR_ESTIMATORS}."
            )
        self.estimator = estimator
        self.scale = check_scale(scale)
        self.estimator_params = estimator_params
        self.random_state = random_state
        if feature_order not in FEATURE_ORDERS:
            raise ConfigurationError(
                f"feature_order deve ser um de {FEATURE_ORDERS}; recebido {feature_order!r}."
            )
        self.feature_order = feature_order

    def fit(self, windows: npt.ArrayLike, targets: npt.ArrayLike) -> Self:
        features = np.array(windows, dtype=np.float64)
        labels = np.array(targets, dtype=np.float64)
        if features.ndim != 2 or labels.shape != (features.shape[0],) or not features.size:
            raise DataValidationError(
                f"Janelas {features.shape} e alvos {labels.shape} incompatíveis."
            )
        if not (np.isfinite(features).all() and np.isfinite(labels).all()):
            raise DataValidationError("Resíduos não finitos no treino do corretor.")
        features = self._ordered(features)
        self.center_, self.spread_ = scalar_scale(labels, self.scale)
        if self.scale in PER_COLUMN:
            self.x_center_, self.x_spread_ = column_scale(features, self.scale)
        else:  # windows share the target's mean and spread
            self.x_center_ = np.full(features.shape[1], self.center_)
            self.x_spread_ = np.full(features.shape[1], self.spread_)
        self.window_size_ = features.shape[1]
        self.model_ = build_estimator(self.estimator, self.estimator_params, self.random_state)
        with candidate_errors(self, "ajuste"), single_threaded(self.estimator):
            self.fit_warnings_ = fit_estimator(
                self.model_, self._scaled_windows(features), self._standardized(labels)
            )
        record_fit()
        return self

    def predict_next(self, window: npt.ArrayLike) -> float:
        if not hasattr(self, "model_"):
            raise ConfigurationError("SklearnCorrector precisa de fit antes de predict_next.")
        values = np.array(window, dtype=np.float64)
        if values.shape != (self.window_size_,):
            raise ConfigurationError(
                f"Janela com shape {values.shape}; o corretor foi ajustado com {self.window_size_}."
            )
        row = self._scaled_windows(self._ordered(values.reshape(1, -1)))
        with candidate_errors(self, "previsão"), single_threaded(self.estimator):
            step = float(self.model_.predict(row)[0])
        return float(unscale_values(step, self.center_, self.spread_, self.scale))

    def _ordered(self, windows: FloatArray) -> FloatArray:
        if self.feature_order == "lag":
            return np.ascontiguousarray(windows[:, ::-1])
        return windows

    def _standardized(self, values: FloatArray) -> FloatArray:
        return scale_values(values, self.center_, self.spread_, self.scale)  # type: ignore[no-any-return]

    def _scaled_windows(self, windows: FloatArray) -> FloatArray:
        return scale_values(windows, self.x_center_, self.x_spread_, self.scale)  # type: ignore[no-any-return]
