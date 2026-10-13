"""Seasonal ARIMA through statsmodels (optional extra ``stats``).

``order="auto"`` follows the steps of Hyndman and Khandakar (2008) on the training
vector only: ``D`` from the seasonality test of :mod:`autohybts.data.seasonality` (one
seasonal difference when the cycle is significant and ``F_s >= 0.64``, the threshold of
``nsdiffs``), ``d`` by successive KPSS tests on the seasonally differenced series, then a
stepwise search of ``(p, q, P, Q)`` by AICc from four starting models. ``season_length=1``
gives a non-seasonal ARIMA chosen the same way.

As in :mod:`autohybts.adapters.statsmodels_arima`, the chosen orders and parameters are
cached by training vector, and forecasting from a longer context re-runs the state-space
filter with the parameters fixed (``apply(refit=False)``): variant B and out-of-fold
residuals never re-estimate the model.
"""

from __future__ import annotations

import hashlib
import warnings
from typing import Any, Final

import numpy as np

from autohybts.adapters.base import FloatArray
from autohybts.adapters.statsmodels_arima import (
    TREND_BY_D,
    ArimaForecaster,
    _arima_class,
    _BoundedCache,
)
from autohybts.data.seasonality import detect_seasonality
from autohybts.exceptions import (
    CandidateFitError,
    ConfigurationError,
    UnsupportedCapabilityError,
)

__all__ = ["MAX_SEASON_LENGTH", "SarimaForecaster"]

# Seasonal state-space models grow with the period: beyond a day of hours the fit is too
# slow for a search; long cycles are better served by Fourier terms.
MAX_SEASON_LENGTH: Final = 24
# Most models the stepwise search may fit on one training vector.
MAX_FITS: Final = 40

Orders = tuple[tuple[int, int, int], tuple[int, int, int, int]]


class SarimaForecaster(ArimaForecaster):
    def __init__(
        self,
        season_length: int = 1,
        order: tuple[int, int, int] | str = "auto",
        seasonal_order: tuple[int, int, int] | None = None,
        max_p: int = 2,
        max_d: int = 2,
        max_q: int = 2,
        max_P: int = 1,
        max_D: int = 1,
        max_Q: int = 1,
    ) -> None:
        super().__init__(order=order, max_p=max_p, max_d=max_d, max_q=max_q)
        if isinstance(season_length, bool) or not isinstance(season_length, int):
            raise ConfigurationError(
                f"season_length deve ser inteiro >= 1; recebido {season_length!r}."
            )
        if season_length < 1:
            raise ConfigurationError(
                f"season_length deve ser inteiro >= 1; recebido {season_length!r}."
            )
        if seasonal_order is not None:
            if order == "auto":
                raise ConfigurationError("seasonal_order exige uma order explícita (p, d, q).")
            if (
                not isinstance(seasonal_order, tuple | list)
                or len(seasonal_order) != 3
                or any(
                    isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in seasonal_order
                )
            ):
                raise ConfigurationError(
                    f"seasonal_order deve ser (P, D, Q); recebido {seasonal_order!r}."
                )
            if seasonal_order[1] > 1:
                raise ConfigurationError("Mais de uma diferença sazonal não é suportada.")
        for name, value in (("max_P", max_P), ("max_D", max_D), ("max_Q", max_Q)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigurationError(f"{name} deve ser inteiro >= 0; recebido {value!r}.")
        self.season_length = season_length
        self.seasonal_order = seasonal_order
        self.max_P = max_P
        self.max_D = min(max_D, 1)
        self.max_Q = max_Q

    @property
    def min_train_size(self) -> int:
        if self.season_length == 1:
            return 10
        return max(10, 3 * self.season_length)

    @property
    def min_history(self) -> int:
        if not self.is_fitted:
            return 1
        (p, d, _), (big_p, big_d, _, m) = self.orders_
        return p + d + (big_p + big_d) * m + 1

    def _fit(self, y: FloatArray) -> None:
        if self.season_length > MAX_SEASON_LENGTH:
            raise UnsupportedCapabilityError(
                f"SARIMA com season_length={self.season_length} é lento demais para a busca "
                f"(máximo {MAX_SEASON_LENGTH}); use harmonic_regression para ciclos longos."
            )
        arima = _arima_class()
        self.fit_warnings_: list[str] = []
        key = self._cache_key(y)
        cached = _SARIMA_CACHE.get(key)
        if cached is not None:
            orders, params, fit_warnings = cached
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.result_ = self._model(arima, y, orders).filter(params)
            self._store(orders, y)
            self.fit_warnings_ = list(fit_warnings)
            return
        m = self.season_length
        if self.order == "auto":
            big_d = self._choose_big_d(y) if m > 1 else 0
            d = self._choose_d(_seasonal_difference(y, m, big_d))
            orders, result = self._stepwise(arima, y, d, big_d)
        else:
            p, d, q = (int(v) for v in self.order)
            big_p, big_d, big_q = (int(v) for v in (self.seasonal_order or (0, 0, 0)))
            orders = ((p, d, q), (big_p, big_d, big_q, m if m > 1 else 0))
            try:
                result = self._fit_orders(arima, y, orders)
            except (ValueError, np.linalg.LinAlgError) as error:
                raise CandidateFitError(
                    f"SARIMA {orders} não pôde ser ajustado: {error}"
                ) from error
        self.result_ = result
        self._store(orders, y)
        _SARIMA_CACHE.put(
            key, (orders, np.array(result.params, copy=True), tuple(self.fit_warnings_))
        )

    def _store(self, orders: Orders, y: FloatArray) -> None:
        self.orders_ = orders
        self.order_ = orders[0]
        self.seasonal_order_ = orders[1]
        self.train_ = y.copy()

    def _cache_key(self, y: FloatArray) -> str:
        digest = hashlib.sha256(np.ascontiguousarray(y, dtype=np.float64).tobytes()).hexdigest()
        settings = (
            self.season_length,
            self.order,
            self.seasonal_order,
            self.max_p,
            self.max_d,
            self.max_q,
            self.max_P,
            self.max_D,
            self.max_Q,
        )
        return f"{digest}|{settings}"

    def _choose_big_d(self, y: FloatArray) -> int:
        if self.max_D == 0:
            return 0
        # Strength alone is inflated on short vectors; require the seasonality test too.
        return detect_seasonality(y, self.season_length).seasonal_differences

    def _stepwise(self, arima: Any, y: FloatArray, d: int, big_d: int) -> tuple[Orders, Any]:
        seasonal = self.season_length > 1
        bounds = (
            self.max_p,
            self.max_q,
            self.max_P if seasonal else 0,
            self.max_Q if seasonal else 0,
        )
        scores: dict[tuple[int, int, int, int], tuple[float, Any]] = {}
        errors: list[str] = []

        def clip(c: tuple[int, ...]) -> tuple[int, int, int, int]:
            p, q, big_p, big_q = (min(max(v, 0), b) for v, b in zip(c, bounds, strict=True))
            return p, q, big_p, big_q

        def score(c: tuple[int, int, int, int]) -> float:
            if c not in scores:
                orders = self._orders(c, d, big_d)
                try:
                    result = self._fit_orders(arima, y, orders)
                    value = float(result.aicc)
                except (ValueError, np.linalg.LinAlgError) as error:
                    errors.append(f"{orders}: {error}")
                    result, value = None, float("inf")
                scores[c] = (value if np.isfinite(value) else float("inf"), result)
            return scores[c][0]

        starts = [clip(c) for c in ((2, 2, 1, 1), (0, 0, 0, 0), (1, 0, 1, 0), (0, 1, 0, 1))]
        for c in dict.fromkeys(starts):
            score(c)
        best = min(scores, key=lambda c: (scores[c][0], c))
        while len(scores) < MAX_FITS:
            neighbours = [
                clip(tuple(v + step * (i == k) for k, v in enumerate(best)))
                for i in range(4)
                for step in (-1, 1)
            ]
            neighbours += [clip((best[0] + s, best[1] + s, best[2], best[3])) for s in (-1, 1)] + [
                clip((best[0], best[1], best[2] + s, best[3] + s)) for s in (-1, 1)
            ]
            fresh = [c for c in dict.fromkeys(neighbours) if c not in scores]
            if not fresh:
                break
            for c in fresh[: MAX_FITS - len(scores)]:
                score(c)
            candidate = min(scores, key=lambda c: (scores[c][0], c))
            if candidate == best:
                break
            best = candidate
        value, result = scores[best]
        if result is None or not np.isfinite(value):
            raise CandidateFitError(f"Nenhuma ordem SARIMA pôde ser ajustada: {errors[:3]}")
        return self._orders(best, d, big_d), result

    def _orders(self, c: tuple[int, int, int, int], d: int, big_d: int) -> Orders:
        p, q, big_p, big_q = c
        m = self.season_length if self.season_length > 1 else 0
        return (p, d, q), (big_p, big_d, big_q, m)

    def _model(self, arima: Any, y: FloatArray, orders: Orders) -> Any:
        order, seasonal = orders
        trend = TREND_BY_D[min(order[1] + seasonal[1], 2)]
        return arima(y, order=order, seasonal_order=seasonal, trend=trend)

    def _fit_orders(self, arima: Any, y: FloatArray, orders: Orders) -> Any:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = self._model(arima, y, orders).fit()
        self.fit_warnings_.extend(f"{orders}: {w.message}" for w in caught)
        return result


def _seasonal_difference(y: FloatArray, period: int, times: int) -> FloatArray:
    for _ in range(times):
        y = y[period:] - y[:-period]
    return y


_SARIMA_CACHE: _BoundedCache[tuple[Orders, Any, tuple[str, ...]]] = _BoundedCache()
