"""Optional, fold-local preparation of a training vector: outliers and Box-Cox (ADR-0009).

Both are off by default and fitted on the training values each model receives, never on
anything after its origin. Outliers change only the values the model learns from: the
context of a forecast and the targets of every metric stay as observed. The Box-Cox
transform is frozen at fit time and inverted on the forecasts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from autohybts.exceptions import ConfigurationError

__all__ = [
    "OUTLIER_METHODS",
    "TRANSFORMS",
    "FittedTarget",
    "check_preprocessing",
    "clean_outliers",
    "fit_target_transform",
]

FloatArray = npt.NDArray[np.float64]

TRANSFORMS: Final = ("boxcox", "log")
OUTLIER_METHODS: Final = ("winsorize", "interpolate")
OUTLIER_THRESHOLD: Final = 4.0
_FLOOR: Final = 1e-8


def check_preprocessing(transform: object, outliers: object) -> None:
    if transform is not None and transform not in TRANSFORMS:
        raise ConfigurationError(
            f"transform deve ser None ou um de {list(TRANSFORMS)}; recebido {transform!r}."
        )
    if outliers is not None and outliers not in OUTLIER_METHODS:
        raise ConfigurationError(
            f"outliers deve ser None ou um de {list(OUTLIER_METHODS)}; recebido {outliers!r}."
        )


def clean_outliers(values: npt.ArrayLike, method: str, window: int = 7) -> FloatArray:
    """Points with robust z > 4 against a centred rolling median, winsorized or interpolated."""

    y = np.asarray(values, dtype=np.float64).copy()
    if y.size < 3:
        return y
    half = max(1, window // 2)
    padded = np.pad(y, half, mode="edge")
    median = np.array([np.median(padded[i : i + 2 * half + 1]) for i in range(y.size)])
    residual = y - median
    mad = float(np.median(np.abs(residual - np.median(residual)))) * 1.4826
    if mad <= 0:
        return y
    z = (residual - np.median(residual)) / mad
    flagged = np.abs(z) > OUTLIER_THRESHOLD
    if not flagged.any():
        return y
    if method == "winsorize":
        limit = OUTLIER_THRESHOLD * mad
        center = np.median(residual)
        y[flagged] = median[flagged] + np.clip(residual[flagged] - center, -limit, limit) + center
        return y
    good = np.flatnonzero(~flagged)
    y[flagged] = np.interp(np.flatnonzero(flagged), good, y[good])
    return y


@dataclass(frozen=True, slots=True)
class FittedTarget:
    """Box-Cox (or log, ``lam = 0``) with a shift that keeps the training values positive."""

    lam: float
    shift: float

    def apply(self, values: npt.ArrayLike) -> FloatArray:
        z = np.maximum(np.asarray(values, dtype=np.float64) + self.shift, _FLOOR)
        if abs(self.lam) < 1e-8:
            return np.log(z)
        return (np.power(z, self.lam) - 1.0) / self.lam

    def invert(self, values: npt.ArrayLike) -> FloatArray:
        v = np.asarray(values, dtype=np.float64)
        if abs(self.lam) < 1e-8:
            return np.exp(v) - self.shift
        base = np.maximum(self.lam * v + 1.0, _FLOOR)
        return np.power(base, 1.0 / self.lam) - self.shift


def fit_target_transform(values: npt.ArrayLike, kind: str) -> FittedTarget:
    """``lambda`` by maximum likelihood in [-1, 2] (``"boxcox"``) or 0 (``"log"``)."""

    y = np.asarray(values, dtype=np.float64)
    lowest = float(np.min(y))
    shift = 1.0 - lowest if lowest <= 0 else 0.0
    if kind == "log":
        return FittedTarget(0.0, shift)
    from scipy.optimize import minimize_scalar
    from scipy.stats import boxcox_llf

    positive = y + shift
    optimum = minimize_scalar(
        lambda value: -boxcox_llf(value, positive), bounds=(-1.0, 2.0), method="bounded"
    )
    lam = float(optimum.x)
    return FittedTarget(0.0 if math.isclose(lam, 0.0, abs_tol=1e-8) else lam, shift)
