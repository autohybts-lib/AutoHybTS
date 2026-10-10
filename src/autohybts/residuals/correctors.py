"""Residual corrector contract (§5.2) and scikit-learn correctors (§8.1)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt

from autohybts.adapters.base import FloatArray, Parametrized, candidate_errors
from autohybts.adapters.sklearn_regression import build_estimator, fit_estimator
from autohybts.exceptions import ConfigurationError, DataValidationError

__all__ = ["CORRECTOR_ESTIMATORS", "ResidualCorrector", "SklearnCorrector"]

CORRECTOR_ESTIMATORS: Final = ("linear", "knn", "mlp")


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
    """

    def __init__(
        self,
        estimator: str = "linear",
        scale: bool = True,
        estimator_params: Mapping[str, Any] | None = None,
        random_state: int | None = 0,
    ) -> None:
        if estimator not in CORRECTOR_ESTIMATORS:
            raise ConfigurationError(
                f"Corretor desconhecido {estimator!r}; opções: {CORRECTOR_ESTIMATORS}."
            )
        self.estimator = estimator
        self.scale = scale
        self.estimator_params = estimator_params
        self.random_state = random_state

    def fit(self, windows: npt.ArrayLike, targets: npt.ArrayLike) -> Self:
        features = np.array(windows, dtype=np.float64)
        labels = np.array(targets, dtype=np.float64)
        if features.ndim != 2 or labels.shape != (features.shape[0],) or not features.size:
            raise DataValidationError(
                f"Janelas {features.shape} e alvos {labels.shape} incompatíveis."
            )
        if not (np.isfinite(features).all() and np.isfinite(labels).all()):
            raise DataValidationError("Resíduos não finitos no treino do corretor.")
        self.center_ = float(np.mean(labels)) if self.scale else 0.0
        spread = float(np.std(labels)) if self.scale else 1.0
        self.spread_ = spread if spread > 0 else 1.0
        self.window_size_ = features.shape[1]
        self.model_ = build_estimator(self.estimator, self.estimator_params, self.random_state)
        with candidate_errors(self, "ajuste"):
            self.fit_warnings_ = fit_estimator(
                self.model_, self._standardized(features), self._standardized(labels)
            )
        return self

    def predict_next(self, window: npt.ArrayLike) -> float:
        if not hasattr(self, "model_"):
            raise ConfigurationError("SklearnCorrector precisa de fit antes de predict_next.")
        values = np.array(window, dtype=np.float64)
        if values.shape != (self.window_size_,):
            raise ConfigurationError(
                f"Janela com shape {values.shape}; o corretor foi ajustado com {self.window_size_}."
            )
        with candidate_errors(self, "previsão"):
            step = float(self.model_.predict(self._standardized(values).reshape(1, -1))[0])
        return step * self.spread_ + self.center_

    def _standardized(self, values: FloatArray) -> FloatArray:
        return (values - self.center_) / self.spread_
