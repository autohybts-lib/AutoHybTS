"""Stationarity tests and the automatic choice of regular and seasonal differences.

``d`` is the smallest order in ``0..max_d`` whose differenced series passes the test on
the values received (a training vector):

* ``"kpss"``: KPSS does not reject level stationarity at ``alpha`` (the rule of
  ``ndiffs`` in the R package ``forecast`` and of the ``arima`` adapter);
* ``"adf"``: ADF rejects a unit root, ``p <= alpha`` with the statistic below the 5 %
  critical value (the rule of the replication project).

When no order passes, ``max_d`` is used. ``D`` is one seasonal difference when the
seasonality test of :mod:`autohybts.data.seasonality` finds a significant cycle with
strength ``>= 0.64``; it is chosen first and ``d`` is tested on the seasonally
differenced series, as in ``auto.arima``. The tests come from statsmodels (extra
``stats``), imported only when a choice is made.
"""

from __future__ import annotations

import warnings
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from autohybts.data.preprocessing import Differencer
from autohybts.data.seasonality import detect_seasonality
from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError

__all__ = [
    "DIFFERENCING_TESTS",
    "adf_stationary",
    "check_differencing",
    "choose_d",
    "choose_differences",
    "differencer_for",
    "kpss_stationary",
    "max_lost",
]

FloatArray = npt.NDArray[np.float64]

DIFFERENCING_TESTS: Final = ("kpss", "adf")
MAX_D: Final = 2
# Below this many values a test says little; the current order is kept.
MIN_TEST_SIZE: Final = 8


def _stattools() -> Any:
    try:
        from statsmodels.tsa import stattools
    except ModuleNotFoundError as error:
        raise UnsupportedCapabilityError(
            'differencing="auto" requer o extra `stats`: pip install "autohybts[stats]".'
        ) from error
    return stattools


def kpss_stationary(values: npt.ArrayLike, alpha: float = 0.05) -> bool:
    """KPSS (constant, automatic lags) does not reject level stationarity at ``alpha``."""

    stattools = _stattools()
    with warnings.catch_warnings():
        # KPSS warns when the p-value is outside its lookup table; the bound is fine.
        warnings.simplefilter("ignore")
        p_value = stattools.kpss(np.asarray(values, dtype=np.float64), "c", nlags="auto")[1]
    return bool(p_value >= alpha)


def adf_stationary(values: npt.ArrayLike, alpha: float = 0.05) -> bool:
    """ADF (constant, lags by AIC) rejects a unit root: ``p <= alpha`` and stat < 5 % critical."""

    stattools = _stattools()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = stattools.adfuller(
                np.asarray(values, dtype=np.float64), autolag="AIC", regression="c"
            )
    except (ValueError, np.linalg.LinAlgError):
        return False
    if isinstance(result, tuple):
        statistic, p_value, critical = result[0], result[1], result[4]
    else:  # statsmodels >= 0.16 returns an ADFullerResult
        statistic, p_value, critical = result.statistic, result.pvalue, result.critical_values
    return bool(p_value <= alpha and statistic < critical["5%"])


def choose_d(
    values: npt.ArrayLike, max_d: int = MAX_D, test: str = "kpss", alpha: float = 0.05
) -> int:
    """Smallest number of regular differences that makes the series pass ``test``."""

    stationary = kpss_stationary if test == "kpss" else adf_stationary
    series = np.asarray(values, dtype=np.float64)
    for d in range(max_d + 1):
        if d == max_d or series.size < MIN_TEST_SIZE or stationary(series, alpha):
            return d
        series = np.diff(series)
    raise AssertionError("unreachable: the loop returns at d == max_d")  # pragma: no cover


def choose_differences(
    values: npt.ArrayLike, season_length: int = 1, max_d: int = MAX_D, test: str = "kpss"
) -> Differencer:
    """``D`` from the seasonality test, then ``d`` on the seasonally differenced series."""

    y = np.asarray(values, dtype=np.float64)
    seasonal = (
        detect_seasonality(y, season_length).seasonal_differences if season_length >= 3 else 0
    )
    level = y[season_length:] - y[:-season_length] if seasonal else y
    return Differencer(choose_d(level, max_d, test), seasonal, season_length)


def check_differencing(
    differencing: object, seasonal_differencing: object, season_length: object, test: object
) -> None:
    """Validate the differencing settings shared by the lag models."""

    if differencing != "auto" and (
        isinstance(differencing, bool) or not isinstance(differencing, int) or differencing < 0
    ):
        raise ConfigurationError(
            f"differencing deve ser inteiro >= 0 ou 'auto'; recebido {differencing!r}."
        )
    if seasonal_differencing not in (0, 1) or isinstance(seasonal_differencing, bool):
        raise ConfigurationError(
            f"seasonal_differencing deve ser 0 ou 1; recebido {seasonal_differencing!r}."
        )
    if isinstance(season_length, bool) or not isinstance(season_length, int) or season_length < 1:
        raise ConfigurationError(
            f"season_length deve ser inteiro >= 1; recebido {season_length!r}."
        )
    if seasonal_differencing and season_length < 2:
        raise ConfigurationError("seasonal_differencing=1 exige season_length >= 2.")
    if test not in DIFFERENCING_TESTS:
        raise ConfigurationError(
            f"differencing_test deve ser um de {list(DIFFERENCING_TESTS)}; recebido {test!r}."
        )


def differencer_for(
    y: npt.ArrayLike,
    differencing: int | str,
    seasonal_differencing: int,
    season_length: int,
    test: str,
) -> Differencer:
    """The differences a lag model uses: chosen on ``y`` when ``"auto"``, else as given."""

    if differencing == "auto":
        return choose_differences(y, season_length, MAX_D, test)
    return Differencer(int(differencing), seasonal_differencing, season_length)


def max_lost(differencing: int | str, seasonal_differencing: int, season_length: int) -> int:
    """Most observations the differences may consume (before ``"auto"`` has chosen)."""

    if differencing == "auto":
        return MAX_D + (season_length if season_length >= 3 else 0)
    return int(differencing) + seasonal_differencing * season_length
