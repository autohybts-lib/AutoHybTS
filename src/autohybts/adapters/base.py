"""Base forecaster contract (architecture §5.2) and shared adapter plumbing.

Semantics of ``predict(horizon, context)``: forecast the ``horizon`` steps after
``context.origin_index`` with the parameters learned in ``fit``, conditioning only on
``context.observed``. A model fitted once can therefore forecast from later origins
without being refitted; that is how variant B and out-of-fold residuals get new
observations into a frozen model (it covers the ``update_state`` of §13.3).
"""

from __future__ import annotations

import copy
import inspect
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt

from autohybts.domain.context import TemporalContext
from autohybts.exceptions import (
    AutoHybTSError,
    CandidateFitError,
    ConfigurationError,
    DataValidationError,
    InsufficientHistoryError,
    UnsupportedCapabilityError,
)

__all__ = ["BaseForecaster", "FloatArray", "ForecasterBase", "Parametrized", "candidate_errors"]

FloatArray = npt.NDArray[np.float64]


@runtime_checkable
class BaseForecaster(Protocol):
    """Anything that can be fitted on a training vector and forecast from a context."""

    def fit(self, y: npt.ArrayLike, X: None = None) -> Self: ...

    def predict(self, horizon: int, context: TemporalContext) -> FloatArray: ...

    def clone(self) -> Self: ...

    def get_params(self) -> dict[str, Any]: ...


class Parametrized:
    """scikit-learn style ``get_params``/``clone``/``repr`` from the ``__init__`` signature.

    Subclasses store every ``__init__`` argument unchanged under the same attribute name;
    fitted state goes in attributes ending in ``_``.
    """

    def get_params(self) -> dict[str, Any]:
        names = [
            name
            for name, parameter in inspect.signature(type(self).__init__).parameters.items()
            if name != "self" and parameter.kind is parameter.POSITIONAL_OR_KEYWORD
        ]
        return {name: copy.deepcopy(getattr(self, name)) for name in names}

    def clone(self) -> Self:
        """Unfitted copy with the same parameters and no shared state."""

        return type(self)(**self.get_params())

    def __repr__(self) -> str:
        params = ", ".join(f"{name}={value!r}" for name, value in self.get_params().items())
        return f"{type(self).__name__}({params})"


class ForecasterBase(Parametrized, ABC):
    """Validation shared by adapters; subclasses implement ``_fit`` and ``_predict``."""

    def __init__(self) -> None:
        self._is_fitted = False

    @property
    def min_train_size(self) -> int:
        """Smallest training vector ``fit`` accepts."""

        return 1

    @property
    def min_history(self) -> int:
        """Smallest ``context.observed`` that ``predict`` accepts."""

        return 1

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def diagnostics(self) -> dict[str, float]:
        """Numbers about the last fit that selection may use (empty by default)."""

        return {}

    def fit(self, y: npt.ArrayLike, X: None = None) -> Self:
        if X is not None:
            raise UnsupportedCapabilityError("Covariáveis exógenas ainda não são suportadas.")
        values = _training_vector(y)
        if values.size < self.min_train_size:
            raise InsufficientHistoryError(
                f"{type(self).__name__} exige ao menos {self.min_train_size} observações de "
                f"treino; recebidas {values.size}."
            )
        self._is_fitted = False
        with candidate_errors(self, "ajuste"):
            self._fit(values)
        self._is_fitted = True
        return self

    def predict(self, horizon: int, context: TemporalContext) -> FloatArray:
        if not self._is_fitted:
            raise ConfigurationError(f"{type(self).__name__} precisa de fit antes de predict.")
        if isinstance(horizon, bool) or not isinstance(horizon, int | np.integer) or horizon < 1:
            raise ConfigurationError(f"horizon deve ser inteiro >= 1; recebido {horizon!r}.")
        if not isinstance(context, TemporalContext):
            raise ConfigurationError("predict exige um TemporalContext.")
        if context.observed.size < self.min_history:
            raise InsufficientHistoryError(
                f"{type(self).__name__} exige ao menos {self.min_history} observações no "
                f"contexto; recebidas {context.observed.size}."
            )
        with candidate_errors(self, "previsão"):
            raw = self._predict(int(horizon), context.observed)
        forecast = np.asarray(raw, dtype=np.float64)
        if forecast.shape != (horizon,) or not np.isfinite(forecast).all():
            raise CandidateFitError(
                f"{type(self).__name__} produziu previsão inválida (shape {forecast.shape}, "
                "ou valores não finitos)."
            )
        return forecast

    @abstractmethod
    def _fit(self, y: FloatArray) -> None:
        """Learn parameters from the training vector ``y`` (already validated)."""

    @abstractmethod
    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        """Forecast ``horizon`` steps after the last value of ``history``."""


def _training_vector(y: npt.ArrayLike) -> FloatArray:
    try:
        values = np.array(y, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DataValidationError("O treino deve ser numérico.") from error
    if values.ndim != 1:
        raise DataValidationError(f"O treino deve ser 1-D; recebido shape {values.shape}.")
    if not np.isfinite(values).all():
        raise DataValidationError("O treino contém valores ausentes ou não finitos.")
    return values


@contextmanager
def candidate_errors(model: object, stage: str) -> Iterator[None]:
    """Library errors pass through; anything else from an adapter becomes CandidateFitError."""

    try:
        yield
    except AutoHybTSError:
        raise
    except Exception as error:
        raise CandidateFitError(
            f"{type(model).__name__} falhou na {stage}: {type(error).__name__}: {error}"
        ) from error
