"""Residual sequence correctors (plano de ensembles §4.2, F1e).

A window corrector (``ResidualCorrector``) regresses the next residual on a window of past
residuals and is iterated step by step. A *sequence* corrector is a time-series model of
the residual series itself: it is fitted once on the contiguous out-of-fold residuals of
the training vector and forecasts the whole residual vector of the horizon from the
residual history available at the origin (its own recursion feeds only its predictions,
as variant A requires). State-space models (ARIMA, exponential smoothing) need the whole
residual history for their states, which a fixed window cannot give them.

The wrapped model is a built-in base forecaster with frozen parameters: a longer residual
history (new observations in variant B) updates its state without refitting.

Sequence correctors work in the additive residual hybrid (and therefore in multistage
hybrids). Strategies that need a window corrector (decomposition, residual combinations,
dynamic selection) get ``UnsupportedCapabilityError`` and the trial is SKIPPED.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt

from autohybts.adapters.base import BaseForecaster, FloatArray, Parametrized, candidate_errors
from autohybts.exceptions import (
    ConfigurationError,
    DataValidationError,
    UnsupportedCapabilityError,
)

__all__ = [
    "SEQUENCE_MODELS",
    "ResidualSequenceForecaster",
    "SequenceCorrector",
    "fitted_correction_ratio",
]

# Built-in bases a sequence corrector may wrap, with the settings that suit residuals
# (no trend: residuals of a decent base have no drift).
SEQUENCE_MODELS: dict[str, dict[str, Any]] = {
    "arima": {},
    "exponential_smoothing": {"trend": False},
}
# How many recent training positions estimate the spread of the fitted corrections.
_RATIO_POSITIONS = 100


@runtime_checkable
class ResidualSequenceForecaster(Protocol):
    """Fitted on a residual series; forecasts the next ``horizon`` residuals."""

    def fit_sequence(self, residuals: npt.ArrayLike) -> Self: ...

    def forecast(self, horizon: int, residuals: npt.ArrayLike) -> FloatArray: ...

    def clone(self) -> Self: ...

    def get_params(self) -> dict[str, Any]: ...


class SequenceCorrector(Parametrized):
    """A built-in base model (``model``) applied to the residual series."""

    def __init__(self, model: str = "arima", model_params: Mapping[str, Any] | None = None) -> None:
        if model not in SEQUENCE_MODELS:
            raise ConfigurationError(
                f"Modelo de sequência desconhecido {model!r}; opções: {sorted(SEQUENCE_MODELS)}."
            )
        self.model = model
        self.model_params = model_params

    def _make(self) -> BaseForecaster:
        from autohybts.registry import default_registry

        params = {**SEQUENCE_MODELS[self.model], **dict(self.model_params or {})}
        return default_registry().make_base(self.model, params)

    @property
    def min_residuals(self) -> int:
        return int(getattr(self._make(), "min_train_size", 1))

    def fit_sequence(self, residuals: npt.ArrayLike) -> Self:
        self.model_ = self._make().fit(_residual_vector(residuals))
        return self

    def forecast(self, horizon: int, residuals: npt.ArrayLike) -> FloatArray:
        from autohybts.residuals.generation import local_context

        values = _residual_vector(residuals)
        return self.model_.predict(horizon, local_context(values, values.size))

    # A window corrector's methods: refused, so window-only strategies SKIP the trial.
    def fit(self, windows: npt.ArrayLike, targets: npt.ArrayLike) -> Self:
        raise UnsupportedCapabilityError(_WINDOW_ONLY)

    def predict_next(self, window: npt.ArrayLike) -> float:
        raise UnsupportedCapabilityError(_WINDOW_ONLY)


_WINDOW_ONLY = (
    "Corretor de sequência residual: só funciona no híbrido aditivo (e em estágios), não "
    "como corretor de janelas."
)


def fitted_correction_ratio(corrector: SequenceCorrector, residuals: FloatArray) -> float:
    """``correction_std_ratio`` of a sequence corrector (ADR-0003).

    std of its one-step forecasts over std of the residuals they forecast, on the last
    ``_RATIO_POSITIONS`` positions of its training residuals (each forecast uses only the
    residuals before its position, with the frozen fitted model).
    """

    start = max(int(getattr(corrector.model_, "min_history", 1)), residuals.size - _RATIO_POSITIONS)
    targets = residuals[start:]
    scale = float(np.std(targets)) if targets.size else 0.0
    if scale <= 1e-12:
        return float("nan")
    batch = getattr(corrector.model_, "one_step_path", None)
    with candidate_errors(corrector, "diagnóstico"):
        if callable(batch):  # same forecasts in one pass (ARIMA's Kalman filter)
            fitted = np.asarray(batch(residuals, start), dtype=np.float64)
        else:
            fitted = np.array(
                [corrector.forecast(1, residuals[:end])[0] for end in range(start, residuals.size)]
            )
    return float(np.std(fitted)) / scale


def _residual_vector(residuals: npt.ArrayLike) -> FloatArray:
    values = np.array(residuals, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise DataValidationError("A série de resíduos deve ser 1-D, não vazia e finita.")
    return values
