"""Variant A recursion of the corrector (§7.3, §13.2)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

__all__ = ["recursive_correction", "roll_forward"]

FloatArray = npt.NDArray[np.float64]


class _NextStep(Protocol):
    def predict_next(self, window: FloatArray) -> float: ...


def roll_forward(window: FloatArray, value: float) -> FloatArray:
    """Drop the oldest residual and append ``value``; returns a new array."""

    return np.append(window[1:], value)


def recursive_correction(corrector: _NextStep, window: FloatArray, horizon: int) -> FloatArray:
    """Corrections for steps ``1..horizon``.

    After the first step the window is fed with the corrector's own *predicted* residuals:
    in variant A no observation after the origin exists, so an observed residual can never
    enter here.
    """

    current = np.array(window, dtype=np.float64)
    corrections = np.empty(horizon, dtype=np.float64)
    for step in range(horizon):
        corrections[step] = float(corrector.predict_next(current.copy()))
        current = roll_forward(current, corrections[step])
    return corrections
