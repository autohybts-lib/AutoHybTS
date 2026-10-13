"""Ensembles of complete forecasters.

``ŷ = agg(ŷ_1, ..., ŷ_K)`` over components that may themselves be hybrids or other
compositions:

* ``equal``: mean, weights ``1/K``.
* ``median``: per-step median of the component forecasts.
* ``trimmed_mean``: per step, drop the ``max(1, round(trim · K))`` lowest and highest
  forecasts and average the rest.
* ``inverse_mse``: ``w_k ∝ 1 / MSE_k`` of one-step out-of-fold errors.
* ``nnls``: non-negative least squares of the target on the out-of-fold forecasts,
  normalized to sum 1 and shrunk towards equal weights,
  ``w = (1 − shrinkage) · w_nnls + shrinkage / K`` (few out-of-fold rows make free
  weights unstable: the forecast combination puzzle).

Out-of-fold forecasts come from ``historical_oof_1step`` inside the training vector: each
component is refitted on the prefix before the tail and forecasts the tail one step at a
time, so the weights never see data after the origin. Every component is then fitted on
the whole training vector; a frozen ensemble takes new observations like any base
(variant B).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import numpy as np

from autohybts.adapters.base import BaseForecaster, FloatArray, ForecasterBase
from autohybts.exceptions import CandidateFitError, ConfigurationError
from autohybts.residuals.generation import ResidualSettings, build_residuals, local_context

__all__ = ["WEIGHTED_METHODS", "EnsembleForecaster"]

WEIGHTED_METHODS: Final = ("equal", "inverse_mse", "nnls")
_METHODS: Final = (*WEIGHTED_METHODS, "median", "trimmed_mean")


class EnsembleForecaster(ForecasterBase):
    composite = True

    def __init__(
        self,
        components: Sequence[BaseForecaster],
        method: str = "equal",
        trim: float = 0.2,
        shrinkage: float = 0.5,
        settings: ResidualSettings | None = None,
    ) -> None:
        super().__init__()
        if len(components) < 2:
            raise ConfigurationError("Um ensemble precisa de ao menos dois componentes.")
        if not all(isinstance(c, BaseForecaster) for c in components):
            raise ConfigurationError("Componentes devem cumprir o contrato BaseForecaster.")
        if method not in _METHODS:
            raise ConfigurationError(f"method deve ser um de {_METHODS}; recebido {method!r}.")
        self.components = tuple(components)
        self.method = method
        self.trim = trim
        self.shrinkage = shrinkage
        self.settings = settings

    @property
    def needs_oof(self) -> bool:
        return self.method in ("inverse_mse", "nnls")

    @property
    def min_train_size(self) -> int:
        largest = max(int(getattr(c, "min_train_size", 1)) for c in self.components)
        if not self.needs_oof:
            return largest
        share = (self.settings or ResidualSettings()).tail_fraction
        return math.ceil(largest / (1.0 - share)) + 1

    @property
    def min_history(self) -> int:
        components = self.components_ if self.is_fitted else self.components
        return max(int(getattr(c, "min_history", 1)) for c in components)

    def _fit(self, y: FloatArray) -> None:
        self.components_ = tuple(c.clone().fit(y) for c in self.components)
        k = len(self.components)
        weights: FloatArray | None = None
        if self.method == "equal":
            weights = np.full(k, 1.0 / k)
        elif self.needs_oof:
            predictions, targets = self._out_of_fold(y)
            if self.method == "inverse_mse":
                mse = np.mean((targets[:, None] - predictions) ** 2, axis=0)
                perfect = mse <= 1e-12
                inverse = perfect.astype(np.float64) if perfect.any() else 1.0 / mse
                weights = inverse / inverse.sum()
            else:
                weights = _nnls_weights(predictions, targets, self.shrinkage)
        self.weights_: FloatArray | None = weights

    def _out_of_fold(self, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        """One-step out-of-fold forecasts of every component on the same training tail."""

        settings = self.settings or ResidualSettings()
        series = [
            build_residuals(template, fitted, y, "historical_oof_1step", settings)
            for template, fitted in zip(self.components, self.components_, strict=True)
        ]
        targets = series[0].targets
        if any(not np.array_equal(s.targets, targets) for s in series[1:]):
            raise CandidateFitError("Componentes com caudas fora da amostra desalinhadas.")
        actual = y[targets]
        predictions = np.column_stack([actual - s.residuals for s in series])
        return predictions, actual

    def diagnostics(self) -> dict[str, float]:
        weights = getattr(self, "weights_", None)
        if weights is None:
            return {}
        return {f"weight_{i}": float(w) for i, w in enumerate(weights)}

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        context = local_context(history, history.size)
        paths = np.vstack([c.predict(horizon, context) for c in self.components_])
        if self.weights_ is not None:
            return np.asarray(self.weights_ @ paths, dtype=np.float64)
        if self.method == "median":
            return np.asarray(np.median(paths, axis=0), dtype=np.float64)
        cut = max(1, round(self.trim * len(self.components_)))
        cut = min(cut, (len(self.components_) - 1) // 2)
        ordered = np.sort(paths, axis=0)
        return np.asarray(ordered[cut : len(ordered) - cut].mean(axis=0), dtype=np.float64)


def _nnls_weights(predictions: FloatArray, targets: FloatArray, shrinkage: float) -> FloatArray:
    from scipy.optimize import nnls

    k = predictions.shape[1]
    raw, _ = nnls(predictions, targets)
    total = float(raw.sum())
    free = raw / total if total > 1e-12 else np.full(k, 1.0 / k)
    return np.asarray((1.0 - shrinkage) * free + shrinkage / k, dtype=np.float64)
