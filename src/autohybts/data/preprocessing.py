"""Target transforms fitted per fold on training data only, with causal inversion (§34.5).

A transform is fitted on the training segment of one origin and returns an immutable
fitted object. Inverting a forecast uses only anchors stored at fit time, so values
after the origin can never influence the transform or its inverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from autohybts.exceptions import ConfigurationError, InsufficientHistoryError

__all__ = [
    "DifferenceTransform",
    "Differencer",
    "FittedDifference",
    "FittedDifferences",
    "FittedIdentity",
    "FittedSeasonalDifference",
    "FittedTransform",
    "IdentityTransform",
    "SeasonalDifferenceTransform",
    "TargetTransform",
]

FloatArray = npt.NDArray[np.float64]


class FittedTransform(Protocol):
    def inverse_forecast(self, prediction: npt.ArrayLike) -> FloatArray:
        """Map a multi-step forecast in transformed space back to the original scale."""
        ...


class TargetTransform(Protocol):
    def fit_transform(self, y_train: npt.ArrayLike) -> tuple[FittedTransform, FloatArray]:
        """Fit on training values only; return the fitted transform and transformed values."""
        ...


@dataclass(frozen=True, slots=True)
class FittedIdentity:
    def inverse_forecast(self, prediction: npt.ArrayLike) -> FloatArray:
        return np.array(prediction, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class IdentityTransform:
    def fit_transform(self, y_train: npt.ArrayLike) -> tuple[FittedIdentity, FloatArray]:
        return FittedIdentity(), np.array(y_train, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class FittedDifference:
    order: int
    # Last value of the series differenced k times, for k = 0..order-1.
    anchors: tuple[float, ...]

    def inverse_forecast(self, prediction: npt.ArrayLike) -> FloatArray:
        level = np.array(prediction, dtype=np.float64)
        for anchor in reversed(self.anchors):
            level = anchor + np.cumsum(level)
        return level


@dataclass(frozen=True, slots=True)
class DifferenceTransform:
    """Differencing of order ``d``; the inverse integrates from the last training values."""

    order: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 1:
            raise ConfigurationError(f"Ordem de diferenciação inválida: {self.order!r}.")

    def fit_transform(self, y_train: npt.ArrayLike) -> tuple[FittedDifference, FloatArray]:
        level = np.array(y_train, dtype=np.float64)
        if level.ndim != 1 or level.size <= self.order:
            raise InsufficientHistoryError(
                f"Diferenciação de ordem {self.order} exige mais de {self.order} observações."
            )
        anchors = []
        for _ in range(self.order):
            anchors.append(float(level[-1]))
            level = np.diff(level)
        return FittedDifference(self.order, tuple(anchors)), level


@dataclass(frozen=True, slots=True)
class FittedSeasonalDifference:
    period: int
    # The last ``period`` values before differencing: the base of the first cycle ahead.
    anchors: tuple[float, ...]

    def inverse_forecast(self, prediction: npt.ArrayLike) -> FloatArray:
        steps = np.array(prediction, dtype=np.float64)
        level = np.concatenate([np.asarray(self.anchors), np.empty(steps.size)])
        for i, step in enumerate(steps):
            level[self.period + i] = step + level[i]
        return level[self.period :]


@dataclass(frozen=True, slots=True)
class SeasonalDifferenceTransform:
    """``y[t] - y[t - period]``; the inverse adds back the last observed cycle."""

    period: int

    def __post_init__(self) -> None:
        if isinstance(self.period, bool) or not isinstance(self.period, int) or self.period < 2:
            raise ConfigurationError(f"Período da diferença sazonal inválido: {self.period!r}.")

    def fit_transform(self, y_train: npt.ArrayLike) -> tuple[FittedSeasonalDifference, FloatArray]:
        level = np.array(y_train, dtype=np.float64)
        if level.ndim != 1 or level.size <= self.period:
            raise InsufficientHistoryError(
                f"A diferença sazonal de período {self.period} exige mais de {self.period} "
                "observações."
            )
        anchors = tuple(float(v) for v in level[-self.period :])
        return FittedSeasonalDifference(self.period, anchors), level[self.period :] - level[
            : -self.period
        ]


@dataclass(frozen=True, slots=True)
class FittedDifferences:
    """Seasonal difference (if any) followed by ``d`` regular differences, inverted in reverse."""

    steps: tuple[FittedSeasonalDifference | FittedDifference, ...]

    def inverse_forecast(self, prediction: npt.ArrayLike) -> FloatArray:
        level = np.array(prediction, dtype=np.float64)
        for step in reversed(self.steps):
            level = step.inverse_forecast(level)
        return level


@dataclass(frozen=True, slots=True)
class Differencer:
    """``D`` seasonal differences of ``period`` (0 or 1), then ``d`` regular differences."""

    d: int = 0
    seasonal: int = 0
    period: int = 1

    def fit_transform(self, y_train: npt.ArrayLike) -> tuple[FittedDifferences, FloatArray]:
        level = np.array(y_train, dtype=np.float64)
        steps: list[FittedSeasonalDifference | FittedDifference] = []
        if self.seasonal:
            fitted_seasonal, level = SeasonalDifferenceTransform(self.period).fit_transform(level)
            steps.append(fitted_seasonal)
        if self.d:
            fitted_regular, level = DifferenceTransform(self.d).fit_transform(level)
            steps.append(fitted_regular)
        return FittedDifferences(tuple(steps)), level

    @property
    def lost(self) -> int:
        """Observations consumed before the first differenced value."""

        return self.d + self.seasonal * self.period
