"""Validation of timestamps and target values (architecture §11.2, §34.5).

Every check fails loudly. Nothing here reorders, resamples, deduplicates or imputes:
silently repairing a series would change what the model is allowed to see.
"""

from __future__ import annotations

from typing import Final, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.tseries.frequencies import to_offset

from autohybts.exceptions import (
    ConfigurationError,
    DataValidationError,
    UnsupportedCapabilityError,
)

__all__ = [
    "MISSING_POLICIES",
    "MissingPolicy",
    "canonical_frequency",
    "check_timestamps",
    "coerce_values",
    "epoch_ns",
    "resolve_frequency",
]

# Fold-local imputation arrives with preprocessing; until then missing values are rejected.
MissingPolicy: TypeAlias = Literal["reject"]
MISSING_POLICIES: Final[tuple[MissingPolicy, ...]] = ("reject",)


def canonical_frequency(frequency: str) -> str:
    """Return pandas' canonical alias for ``frequency`` or raise ``ConfigurationError``."""

    try:
        return str(to_offset(frequency).freqstr)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"Frequência inválida: {frequency!r}.") from error


def epoch_ns(timestamps: pd.DatetimeIndex) -> npt.NDArray[np.int64]:
    """Nanoseconds since the epoch (UTC for tz-aware indexes), whatever the index unit."""

    naive = timestamps if timestamps.tz is None else timestamps.tz_convert("UTC").tz_localize(None)
    return naive.as_unit("ns").to_numpy(dtype="datetime64[ns]").view(np.int64)


def resolve_frequency(timestamps: pd.DatetimeIndex, frequency: str | None) -> str:
    """Declared frequency, else the index's own, else inferred; returned as canonical alias."""

    if frequency is not None:
        return canonical_frequency(frequency)
    if timestamps.freq is not None:
        return canonical_frequency(str(timestamps.freqstr))
    if len(timestamps) < 3:
        raise DataValidationError(
            "São necessárias ao menos 3 observações para inferir a frequência; informe `frequency`."
        )
    inferred = pd.infer_freq(timestamps)
    if inferred is None:
        raise DataValidationError(
            "Não foi possível inferir uma frequência regular. Informe `frequency` ou corrija "
            "lacunas e irregularidades nas datas."
        )
    return canonical_frequency(inferred)


def check_timestamps(timestamps: pd.DatetimeIndex, frequency: str) -> None:
    """Require non-empty, strictly increasing timestamps on a regular grid of ``frequency``."""

    if len(timestamps) == 0:
        raise DataValidationError("A série está vazia.")
    if timestamps.hasnans:
        position = int(np.flatnonzero(timestamps.isna())[0])
        raise DataValidationError(f"Data ausente (NaT) na posição {position}.")

    stamps = epoch_ns(timestamps)
    steps = np.diff(stamps)
    if (steps == 0).any():
        position = int(np.flatnonzero(steps == 0)[0]) + 1
        raise DataValidationError(
            f"Data duplicada {timestamps[position]} (posições {position - 1} e {position}). "
            "Duplicatas não são agregadas nem descartadas automaticamente."
        )
    if (steps < 0).any():
        position = int(np.flatnonzero(steps < 0)[0]) + 1
        raise DataValidationError(
            f"Datas fora de ordem: {timestamps[position]} aparece depois de "
            f"{timestamps[position - 1]} (posição {position}). Ordene a série antes; "
            "a AutoHybTS não reordena dados."
        )

    expected = pd.date_range(start=timestamps[0], periods=len(timestamps), freq=frequency)
    mismatch = np.flatnonzero(epoch_ns(expected) != stamps)
    if mismatch.size:
        position = int(mismatch[0])
        raise DataValidationError(
            f"A série não é regular na frequência {frequency!r}: esperado {expected[position]}, "
            f"encontrado {timestamps[position]} (posição {position}). Lacunas não são "
            "preenchidas automaticamente."
        )


def coerce_values(values: object, *, expected_length: int, missing: str) -> npt.NDArray[np.float64]:
    """Return a read-only float64 copy of ``values``, enforcing the missing-value policy."""

    if missing not in MISSING_POLICIES:
        raise UnsupportedCapabilityError(
            f"Política de faltantes {missing!r} ainda não implementada; use 'reject'."
        )
    if isinstance(values, pd.Series | pd.Index):
        values = values.to_numpy(dtype="float64", na_value=np.nan)
    try:
        array = np.array(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DataValidationError("Os valores do alvo devem ser numéricos.") from error
    if array.ndim != 1:
        raise DataValidationError(f"O alvo deve ser unidimensional; recebido shape {array.shape}.")
    if array.size != expected_length:
        raise DataValidationError(
            f"{array.size} valores para {expected_length} datas; os tamanhos devem coincidir."
        )
    invalid = ~np.isfinite(array)
    if invalid.any():
        position = int(np.flatnonzero(invalid)[0])
        raise DataValidationError(
            f"{int(invalid.sum())} valor(es) ausente(s) ou não finito(s); o primeiro na posição "
            f"{position}. Com missing='reject', trate os faltantes antes de entregar a série."
        )
    array.flags.writeable = False
    return array
