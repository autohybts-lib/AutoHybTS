"""ARIMA through statsmodels (optional extra ``stats``).

``order="auto"`` picks ``d`` by successive KPSS tests and ``(p, q)`` by AIC, using the
training vector only. The chosen order and parameters are cached by training vector, so
candidates that share an origin (standalone, hybrids, out-of-fold prefixes) estimate
the ARIMA once. Forecasting from a different context re-runs the state-space
filter on the observed values with the fitted parameters fixed (``apply(refit=False)``),
so new observations update the state without re-estimation.
"""

from __future__ import annotations

import hashlib
import itertools
import threading
import warnings
from collections import OrderedDict
from typing import Any

import numpy as np

from autohybts.adapters.base import FloatArray, ForecasterBase
from autohybts.exceptions import (
    CandidateFitError,
    ConfigurationError,
    UnsupportedCapabilityError,
)

__all__ = ["ArimaForecaster"]

# statsmodels only accepts trend terms of order >= d: constant at d=0, drift at d=1.
TREND_BY_D = {0: "c", 1: "t", 2: "n"}


class ArimaForecaster(ForecasterBase):
    def __init__(
        self,
        order: tuple[int, int, int] | str = "auto",
        max_p: int = 2,
        max_d: int = 2,
        max_q: int = 2,
    ) -> None:
        super().__init__()
        if order != "auto":
            if (
                not isinstance(order, tuple | list)
                or len(order) != 3
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in order)
            ):
                raise ConfigurationError(f"order deve ser 'auto' ou (p, d, q); recebido {order!r}.")
            if order[1] > 2:
                raise ConfigurationError("Diferenciação acima de 2 não é suportada.")
        for name, value in (("max_p", max_p), ("max_d", max_d), ("max_q", max_q)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigurationError(f"{name} deve ser inteiro >= 0; recebido {value!r}.")
        self.order = order
        self.max_p = max_p
        self.max_d = min(max_d, 2)
        self.max_q = max_q

    @property
    def min_train_size(self) -> int:
        return 10

    @property
    def min_history(self) -> int:
        if not self.is_fitted:
            return 1
        p, d, _ = self.order_
        return p + d + 1

    def fitted_values(self) -> FloatArray:
        """In-sample one-step fitted values; the first ``d`` points are filter start-up."""

        if not self.is_fitted:
            raise ConfigurationError("ArimaForecaster precisa de fit antes de fitted_values.")
        return np.asarray(self.result_.fittedvalues, dtype=np.float64)

    def one_step_path(self, y: FloatArray, start: int) -> FloatArray:
        """One-step forecasts of ``y[start:]``, each conditioned on ``y[:t]``, parameters fixed.

        Equivalent to ``predict(1, context(y[:t]))`` for every ``t`` (the Kalman filter at
        ``t - 1`` only uses ``y[:t]``), in a single filter pass.
        """

        if not self.is_fitted:
            raise ConfigurationError("ArimaForecaster precisa de fit antes de one_step_path.")
        values = np.asarray(y, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.result_.apply(values, refit=False)
            path = result.predict(start=start, end=values.size - 1)
        return np.asarray(path, dtype=np.float64)

    def _fit(self, y: FloatArray) -> None:
        arima = _arima_class()
        self.fit_warnings_: list[str] = []
        key = self._cache_key(y)
        cached = _FIT_CACHE.get(key)
        if cached is not None:
            # Same data and settings: the same order and parameters, without re-estimating.
            order, params, fit_warnings = cached
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = arima(y, order=order, trend=TREND_BY_D[order[1]])
                self.result_ = model.filter(params)
            self.order_ = order
            self.fit_warnings_ = list(fit_warnings)
            self.train_ = y.copy()
            return
        if isinstance(self.order, str):
            d = self._choose_d(y)
            grid = itertools.product(range(self.max_p + 1), range(self.max_q + 1))
            orders = [(p, d, q) for p, q in grid]
        else:
            p, d, q = self.order
            orders = [(int(p), int(d), int(q))]
        best: tuple[float, tuple[int, int, int], Any] | None = None
        errors = []
        for order in orders:
            try:
                result = self._fit_one(arima, y, order)
            except (ValueError, np.linalg.LinAlgError) as error:
                errors.append(f"{order}: {error}")
                continue
            if np.isfinite(result.aic) and (best is None or result.aic < best[0]):
                best = (float(result.aic), order, result)
        if best is None:
            raise CandidateFitError(f"Nenhuma ordem ARIMA pôde ser ajustada: {errors[:3]}")
        _, self.order_, self.result_ = best
        self.train_ = y.copy()
        _FIT_CACHE.put(
            key, (self.order_, np.array(self.result_.params, copy=True), tuple(self.fit_warnings_))
        )

    def _cache_key(self, y: FloatArray) -> str:
        digest = hashlib.sha256(np.ascontiguousarray(y, dtype=np.float64).tobytes()).hexdigest()
        return f"{digest}|{self.order}|{self.max_p}|{self.max_d}|{self.max_q}"

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        result = self.result_
        if not (history.shape == self.train_.shape and np.array_equal(history, self.train_)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = result.apply(history, refit=False)
        return np.asarray(result.forecast(horizon), dtype=np.float64)

    def _fit_one(self, arima: Any, y: FloatArray, order: tuple[int, int, int]) -> Any:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = arima(y, order=order, trend=TREND_BY_D[order[1]]).fit()
        self.fit_warnings_.extend(f"{order}: {w.message}" for w in caught)
        return result

    def _choose_d(self, y: FloatArray) -> int:
        from statsmodels.tsa.stattools import kpss

        series = y
        for d in range(self.max_d + 1):
            if d == self.max_d or series.size < 8:
                return d
            with warnings.catch_warnings():
                # KPSS warns when the p-value is outside its lookup table; the bound is fine.
                warnings.simplefilter("ignore")
                p_value = kpss(series, regression="c", nlags="auto")[1]
            if p_value >= 0.05:
                return d
            series = np.diff(series)
        raise AssertionError("unreachable: the loop returns at d == max_d")  # pragma: no cover


class _BoundedCache:
    """Thread-safe LRU of fitted ``(order, params, warnings)`` by training vector and settings.

    Only numbers are shared: every model still builds its own statsmodels result, so no
    fitted state passes between trials. Order selection and estimation are deterministic
    for a given vector, so a hit gives the same model as a fresh fit.
    """

    def __init__(self, size: int = 1024) -> None:
        self.size = size
        self._items: OrderedDict[str, tuple[tuple[int, int, int], Any, tuple[str, ...]]] = (
            OrderedDict()
        )
        self._lock = threading.Lock()
        self.hits = 0

    def get(self, key: str) -> tuple[tuple[int, int, int], Any, tuple[str, ...]] | None:
        with self._lock:
            item = self._items.get(key)
            if item is not None:
                self._items.move_to_end(key)
                self.hits += 1
            return item

    def put(self, key: str, item: tuple[tuple[int, int, int], Any, tuple[str, ...]]) -> None:
        with self._lock:
            self._items[key] = item
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self.hits = 0


_FIT_CACHE = _BoundedCache()


def _arima_class() -> Any:
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ModuleNotFoundError as error:
        raise UnsupportedCapabilityError(
            'ARIMA requer o extra `stats`: pip install "autohybts[stats]".'
        ) from error
    return ARIMA
