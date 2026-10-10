"""Causal lag windows: every feature row uses only values strictly before its target."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from autohybts.exceptions import ConfigurationError, InsufficientHistoryError

__all__ = ["lag_matrix", "latest_lags", "normalize_lags"]


def normalize_lags(lags: int | Sequence[int]) -> tuple[int, ...]:
    """``L`` means lags ``1..L``; a sequence is kept in the given order."""

    if isinstance(lags, bool):
        raise ConfigurationError("`lags` deve ser inteiro ou sequência de inteiros.")
    if isinstance(lags, int):
        if lags <= 0:
            raise ConfigurationError(f"Número de lags deve ser positivo; recebido {lags}.")
        return tuple(range(1, lags + 1))
    normalized = tuple(lags)
    if not normalized:
        raise ConfigurationError("A lista de lags está vazia.")
    if any(isinstance(lag, bool) or not isinstance(lag, int) or lag <= 0 for lag in normalized):
        raise ConfigurationError(f"Lags devem ser inteiros positivos; recebido {list(normalized)}.")
    if len(set(normalized)) != len(normalized):
        raise ConfigurationError(f"Lags duplicados: {list(normalized)}.")
    return normalized


def lag_matrix(
    values: npt.ArrayLike, lags: int | Sequence[int]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Rows ``X[i] = [v[t - lag] for lag in lags]`` with target ``v[t]``, for ``t >= max(lags)``."""

    array = _as_vector(values)
    lag_values = normalize_lags(lags)
    max_lag = max(lag_values)
    if array.size <= max_lag:
        raise InsufficientHistoryError(
            f"Série com {array.size} pontos é curta demais para lags até {max_lag}."
        )
    targets = np.arange(max_lag, array.size)
    features = np.column_stack([array[targets - lag] for lag in lag_values])
    return features, array[targets].copy()


def latest_lags(values: npt.ArrayLike, lags: int | Sequence[int]) -> npt.NDArray[np.float64]:
    """Feature row for the step right after the last value: ``[v[n - lag] for lag in lags]``."""

    array = _as_vector(values)
    lag_values = normalize_lags(lags)
    if array.size < max(lag_values):
        raise InsufficientHistoryError(
            f"Histórico com {array.size} pontos é curto demais para lags até {max(lag_values)}."
        )
    return np.array([array[array.size - lag] for lag in lag_values], dtype=np.float64)


def _as_vector(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ConfigurationError(f"Esperado vetor 1-D; recebido shape {array.shape}.")
    return array
