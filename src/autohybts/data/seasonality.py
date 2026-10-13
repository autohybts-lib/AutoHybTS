"""Seasonality diagnostics: strength, autocorrelation and a test by position in the cycle.

Pure numpy, plus scipy (installed with scikit-learn) for the Friedman test, so the rule
behind ``seasonal_period="auto"`` needs no optional extra. The decomposition is the
classical one: a centred moving average of one cycle for the trend and the mean of the
detrended values by position for the seasonal indices. The strength follows Wang, Smith
and Hyndman (2006), ``F_s = max(0, 1 - Var(R) / Var(S + R))``, and the suggested number
of seasonal differences uses the threshold of ``nsdiffs`` in the R package ``forecast``
(``F_s >= 0.64``).

The functions see only the values they receive: callers pass a training vector (or the
development history, for an exploratory report), never anything after it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

__all__ = [
    "MIN_CYCLES",
    "SEASONAL_DIFFERENCE_STRENGTH",
    "SeasonalityTest",
    "classical_decomposition",
    "detect_seasonality",
    "seasonal_strength",
]

FloatArray = npt.NDArray[np.float64]

# Cycles needed before testing: the centred average loses one cycle, and two complete
# cycles (blocks of the Friedman test) must remain.
MIN_CYCLES: Final = 3
SEASONAL_DIFFERENCE_STRENGTH: Final = 0.64


@dataclass(frozen=True, slots=True)
class SeasonalityTest:
    """Evidence about one candidate period; ``seasonal`` is the decision of the rule."""

    period: int
    n_cycles: float
    strength: float
    autocorrelation: float
    p_value: float
    seasonal: bool
    seasonal_differences: int
    reason: str


def classical_decomposition(
    values: npt.ArrayLike, period: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Trend, seasonal and remainder; the trend is ``NaN`` where the average is incomplete."""

    y = np.asarray(values, dtype=np.float64)
    if period < 2 or y.size < 2 * period:
        raise ValueError(f"A decomposição exige período >= 2 e 2 ciclos; período {period}.")
    if period % 2:
        weights = np.full(period, 1.0 / period)
    else:
        weights = np.r_[0.5, np.ones(period - 1), 0.5] / period
    half = weights.size // 2
    trend = np.full(y.size, np.nan)
    trend[half : y.size - half] = np.convolve(y, weights, mode="valid")
    detrended = y - trend
    positions = np.arange(y.size) % period
    indices = np.array([np.nanmean(detrended[positions == k]) for k in range(period)])
    indices -= indices.mean()
    seasonal = indices[positions]
    return trend, seasonal, detrended - seasonal


def seasonal_strength(values: npt.ArrayLike, period: int) -> float:
    """``max(0, 1 - Var(R) / Var(S + R))`` of the classical decomposition; ``NaN`` if undefined."""

    try:
        trend, _, remainder = classical_decomposition(values, period)
    except ValueError:
        return float("nan")
    valid = np.isfinite(trend)
    detrended = np.asarray(values, dtype=np.float64)[valid] - trend[valid]
    total = float(np.var(detrended))
    if total == 0.0:
        return float("nan")
    return max(0.0, 1.0 - float(np.var(remainder[valid])) / total)


def detect_seasonality(
    values: npt.ArrayLike,
    period: int,
    alpha: float = 0.05,
    min_cycles: int = MIN_CYCLES,
) -> SeasonalityTest:
    """Is there a cycle of length ``period``? Friedman test of the detrended values.

    Each run of ``period`` consecutive detrended values is a block and each position in
    the cycle a treatment, so slow changes between cycles do not count as seasonality.
    The rule declares seasonality when the series has at least ``min_cycles`` cycles and
    the positions differ at level ``alpha``. On simulated series without a cycle the false
    detection stayed near ``alpha`` with independent noise and reached about 12 % with
    persistent noise (AR(0.8), random walk). Strength and the lag autocorrelation are
    reported, not used by the rule.
    """

    y = np.asarray(values, dtype=np.float64)
    n_cycles = y.size / period if period >= 1 else 0.0
    if period < 3:
        return SeasonalityTest(period, n_cycles, np.nan, np.nan, np.nan, False, 0, "período < 3")
    if y.size < min_cycles * period:
        return SeasonalityTest(
            period,
            n_cycles,
            np.nan,
            np.nan,
            np.nan,
            False,
            0,
            f"menos de {min_cycles} ciclos completos",
        )
    from scipy.stats import friedmanchisquare

    trend, _, _ = classical_decomposition(y, period)
    valid = np.flatnonzero(np.isfinite(trend))
    first = int(valid[0])
    detrended = y[valid] - trend[valid]
    if float(np.ptp(detrended)) == 0.0:
        return SeasonalityTest(
            period, n_cycles, np.nan, np.nan, np.nan, False, 0, "série sem variação"
        )
    blocks = detrended[: detrended.size // period * period].reshape(-1, period)
    # Columns by position in the cycle: the first valid value sits at position first % m.
    blocks = blocks[:, np.argsort((first + np.arange(period)) % period)]
    p_value = float(friedmanchisquare(*blocks.T).pvalue)
    strength = seasonal_strength(y, period)
    seasonal = p_value < alpha
    differences = int(seasonal and strength >= SEASONAL_DIFFERENCE_STRENGTH)
    reason = f"Friedman p={p_value:.3g} {'<' if seasonal else '>='} {alpha}"
    return SeasonalityTest(
        period,
        n_cycles,
        strength,
        _autocorrelation(detrended, period),
        p_value,
        seasonal,
        differences,
        reason,
    )


def _autocorrelation(values: FloatArray, lag: int) -> float:
    centred = values - values.mean()
    denominator = float(np.dot(centred, centred))
    if lag >= values.size or denominator == 0.0:
        return float("nan")
    return float(np.dot(centred[lag:], centred[:-lag]) / denominator)
