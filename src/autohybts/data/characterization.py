"""Series characterization with the two rules of the replication project.

1. **Stationarity** (ADF on the values received, constant, lags by AIC): stationary when
   ``p <= alpha`` and the statistic is below the 5 % critical value. When the series is
   not stationary, ``d`` is the smallest order up to ``max_d`` that passes; if none
   passes, ``max_d``. KPSS is computed for every order too, as information.
2. **Seasonality**, only for non-stationary series: strength ``F_S = 1 - Var(R) /
   Var(S + R)`` of a robust STL fitted on the Box-Cox transform of the series (``lambda``
   by maximum likelihood within ``boxcox_bounds``, after shifting a non-positive series
   to start at 1), for each candidate period with at least two cycles; the dominant
   period is the one with the largest strength, and the series is seasonal when that
   strength exceeds ``seasonal_threshold`` (0.5).

The category is ``"stationary"``, ``"nonseasonal"`` or ``"seasonal"``. Everything is
computed on the values received: callers pass a training vector or the development
history, never the sealed test. statsmodels (extra ``stats``) and scipy are imported
only here, when a characterization is requested.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from autohybts.exceptions import UnsupportedCapabilityError

__all__ = [
    "BOXCOX_BOUNDS",
    "SEASONAL_THRESHOLD",
    "adf_profile",
    "boxcox_transform",
    "characterize",
    "kpss_profile",
    "stl_boxcox_strength",
    "stl_class",
]

FloatArray = npt.NDArray[np.float64]

BOXCOX_BOUNDS: Final = (-1.0, 2.0)
SEASONAL_THRESHOLD: Final = 0.5


def _statsmodels() -> tuple[Any, Any]:
    try:
        from statsmodels.tsa.seasonal import STL
        from statsmodels.tsa.stattools import adfuller, kpss
    except ModuleNotFoundError as error:
        raise UnsupportedCapabilityError(
            'A caracterização requer o extra `stats`: pip install "autohybts[stats]".'
        ) from error
    return STL, (adfuller, kpss)


def stl_class() -> Any:
    """statsmodels' ``STL`` (extra ``stats``)."""

    stl, _ = _statsmodels()
    return stl


def adf_profile(values: npt.ArrayLike, alpha: float = 0.05) -> dict[str, float | bool]:
    """ADF statistic, p-value, 5 % critical value and the stationarity decision."""

    _, (adfuller, _) = _statsmodels()
    y = np.asarray(values, dtype=np.float64)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = adfuller(y, autolag="AIC", regression="c")
    except (ValueError, np.linalg.LinAlgError):
        return {"stat": np.nan, "pvalue": np.nan, "crit_5": np.nan, "stationary": False}
    if isinstance(result, tuple):
        statistic, p_value, critical = result[0], result[1], result[4]
    else:  # statsmodels >= 0.16 returns an ADFullerResult
        statistic, p_value, critical = result.statistic, result.pvalue, result.critical_values
    return {
        "stat": float(statistic),
        "pvalue": float(p_value),
        "crit_5": float(critical["5%"]),
        "stationary": bool(p_value <= alpha and statistic < critical["5%"]),
    }


def kpss_profile(values: npt.ArrayLike) -> dict[str, float]:
    """KPSS (constant, automatic lags) statistic and p-value; ``NaN`` when it fails."""

    _, (_, kpss) = _statsmodels()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic, p_value, _, _ = kpss(np.asarray(values, dtype=np.float64), "c", nlags="auto")
    except (ValueError, OverflowError, np.linalg.LinAlgError):
        return {"stat": np.nan, "pvalue": np.nan}
    return {"stat": float(statistic), "pvalue": float(p_value)}


def boxcox_transform(
    values: npt.ArrayLike, bounds: tuple[float, float] = BOXCOX_BOUNDS
) -> tuple[FloatArray, float]:
    """Box-Cox transform with ``lambda`` by maximum likelihood within ``bounds``."""

    from scipy.optimize import minimize_scalar
    from scipy.stats import boxcox, boxcox_llf

    y = np.asarray(values, dtype=np.float64)
    lowest = float(np.min(y))
    positive = y - lowest + 1.0 if lowest <= 0 else y
    optimum = minimize_scalar(
        lambda value: -boxcox_llf(value, positive), bounds=bounds, method="bounded"
    )
    fitted = float(optimum.x)
    return np.asarray(boxcox(positive, lmbda=fitted), dtype=np.float64), fitted


def stl_boxcox_strength(
    values: npt.ArrayLike, period: int, bounds: tuple[float, float] = BOXCOX_BOUNDS
) -> tuple[float, float]:
    """``(F_S, lambda)`` of a robust STL on the Box-Cox series; ``NaN`` without two cycles."""

    stl, _ = _statsmodels()
    y = np.asarray(values, dtype=np.float64)
    if period < 2 or y.size < 2 * period or np.isclose(np.var(y), 0.0):
        return float("nan"), float("nan")
    transformed, fitted = boxcox_transform(y, bounds)
    decomposition = stl(transformed, period=int(period), robust=True).fit()
    denominator = float(np.var(decomposition.seasonal + decomposition.resid))
    if np.isclose(denominator, 0.0):
        return 0.0, fitted
    return 1.0 - float(np.var(decomposition.resid)) / denominator, fitted


def characterize(
    values: npt.ArrayLike,
    periods: Sequence[int],
    max_d: int = 2,
    alpha: float = 0.05,
    seasonal_threshold: float = SEASONAL_THRESHOLD,
    bounds: tuple[float, float] = BOXCOX_BOUNDS,
) -> dict[str, Any]:
    """Both rules on ``values``: per-order tests, ``d``, strengths and the category."""

    y = np.asarray(values, dtype=np.float64)
    record: dict[str, Any] = {"n": int(y.size)}
    selected: int | None = None
    for order in range(max_d + 1):
        level = np.diff(y, n=order) if order else y
        adf = adf_profile(level, alpha)
        kpss = kpss_profile(level)
        record.update(
            {
                f"adf_stat_d{order}": adf["stat"],
                f"adf_pvalue_d{order}": adf["pvalue"],
                f"adf_crit5_d{order}": adf["crit_5"],
                f"adf_stationary_d{order}": adf["stationary"],
                f"kpss_stat_d{order}": kpss["stat"],
                f"kpss_pvalue_d{order}": kpss["pvalue"],
            }
        )
        if selected is None and adf["stationary"]:
            selected = order
    record["stationary"] = bool(record["adf_stationary_d0"])
    record["diff_order"] = max_d if selected is None else selected
    strengths: dict[int, float] = {}
    for period in periods:
        strength, fitted = (
            (float("nan"), float("nan"))
            if record["stationary"]
            else stl_boxcox_strength(y, int(period), bounds)
        )
        strengths[int(period)] = strength
        record[f"fs_{int(period)}"] = strength
        record[f"boxcox_lambda_{int(period)}"] = fitted
    finite = {p: s for p, s in strengths.items() if np.isfinite(s)}
    best = max(finite, key=lambda p: finite[p]) if finite else None
    record["fs"] = max(0.0, finite[best]) if best is not None else float("nan")
    record["dominant_period"] = best if best is not None and record["fs"] > 0 else float("nan")
    if record["stationary"]:
        record["category"] = "stationary"
    elif best is not None and record["fs"] > seasonal_threshold:
        record["category"] = "seasonal"
    else:
        record["category"] = "nonseasonal"
    return record
