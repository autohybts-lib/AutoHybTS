"""Validated, immutable univariate series (architecture §11.1-§11.2)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from autohybts.data.validation import (
    MissingPolicy,
    check_timestamps,
    coerce_values,
    epoch_ns,
    resolve_frequency,
)
from autohybts.domain.identity import canonical_json
from autohybts.exceptions import ConfigurationError, DataValidationError

__all__ = ["SINGLE_SERIES_ID", "SeriesFrame"]

# Reserved id for univariate experiments; panels (several series) come later (§11.2).
SINGLE_SERIES_ID: Final = "__single__"


@dataclass(frozen=True, slots=True, eq=False)
class SeriesFrame:
    """One regular, strictly increasing, fully observed series.

    The constructor validates everything and stores read-only copies, so an instance
    can be shared between trials without defensive copying. Compare instances by
    ``fingerprint`` (SHA-256 of identity, timestamps and values), not by ``==``.
    """

    timestamps: pd.DatetimeIndex
    values: npt.NDArray[np.float64]
    frequency: str
    series_id: str = SINGLE_SERIES_ID
    fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.series_id, str) or not self.series_id:
            raise ConfigurationError("`series_id` deve ser um texto não vazio.")
        timestamps = pd.DatetimeIndex(self.timestamps)
        frequency = resolve_frequency(timestamps, self.frequency)
        check_timestamps(timestamps, frequency)
        values = coerce_values(self.values, expected_length=len(timestamps), missing="reject")
        object.__setattr__(self, "timestamps", pd.DatetimeIndex(timestamps, freq=frequency))
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(
            self, "fingerprint", _fingerprint(self.series_id, frequency, timestamps, values)
        )

    @classmethod
    def from_arrays(
        cls,
        timestamps: Any,
        values: object,
        *,
        frequency: str | None = None,
        series_id: str = SINGLE_SERIES_ID,
        missing: MissingPolicy = "reject",
    ) -> SeriesFrame:
        """Build from any timestamp sequence and values; ``frequency`` is inferred if omitted."""

        try:
            index = pd.DatetimeIndex(timestamps)
        except (TypeError, ValueError) as error:
            raise DataValidationError("As datas não puderam ser interpretadas.") from error
        array = coerce_values(values, expected_length=len(index), missing=missing)
        return cls(index, array, resolve_frequency(index, frequency), series_id)

    @classmethod
    def from_series(
        cls,
        series: pd.Series,
        *,
        frequency: str | None = None,
        series_id: str = SINGLE_SERIES_ID,
        missing: MissingPolicy = "reject",
    ) -> SeriesFrame:
        """Build from a ``pd.Series`` indexed by ``DatetimeIndex`` or ``PeriodIndex``."""

        index = series.index
        if isinstance(index, pd.PeriodIndex):
            # Period aliases are not timestamp offsets in pandas 3 ("M" vs "MS"); the
            # converted index carries the right timestamp frequency itself.
            index = index.to_timestamp(how="start")
        if not isinstance(index, pd.DatetimeIndex):
            raise DataValidationError(
                "A série precisa de índice temporal (DatetimeIndex ou PeriodIndex); recebido "
                f"{type(index).__name__}."
            )
        return cls.from_arrays(
            index, series, frequency=frequency, series_id=series_id, missing=missing
        )

    @classmethod
    def from_pandas(
        cls,
        data: pd.DataFrame,
        *,
        target: str,
        date_col: str | None = None,
        frequency: str | None = None,
        series_id: str = SINGLE_SERIES_ID,
        missing: MissingPolicy = "reject",
    ) -> SeriesFrame:
        """Build from a ``DataFrame``; dates come from ``date_col`` or from the index."""

        missing_columns = [name for name in (target, date_col) if name and name not in data.columns]
        if missing_columns:
            raise ConfigurationError(
                f"Coluna(s) {missing_columns} não encontrada(s). Disponíveis: {list(data.columns)}."
            )
        if date_col is None:
            if not isinstance(data.index, pd.DatetimeIndex | pd.PeriodIndex):
                raise DataValidationError(
                    "Informe `date_col` ou use um DataFrame com índice temporal."
                )
            return cls.from_series(
                data[target], frequency=frequency, series_id=series_id, missing=missing
            )
        try:
            dates = pd.to_datetime(data[date_col], errors="raise")
        except (TypeError, ValueError) as error:
            raise DataValidationError(
                f"A coluna de datas {date_col!r} contém valores que não são datas."
            ) from error
        return cls.from_arrays(
            dates, data[target], frequency=frequency, series_id=series_id, missing=missing
        )

    def __len__(self) -> int:
        return len(self.values)

    @property
    def start(self) -> pd.Timestamp:
        return self.timestamps[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.timestamps[-1]

    def head(self, n: int) -> SeriesFrame:
        """First ``n`` observations as a new, revalidated frame."""

        return self._slice(0, n)

    def tail(self, n: int) -> SeriesFrame:
        """Last ``n`` observations as a new, revalidated frame."""

        return self._slice(len(self) - n, len(self))

    def to_series(self) -> pd.Series:
        """A writable copy as ``pd.Series`` named ``value``."""

        return pd.Series(self.values.copy(), index=self.timestamps.copy(), name="value")

    def _slice(self, start: int, stop: int) -> SeriesFrame:
        if not 0 <= start < stop <= len(self):
            raise ConfigurationError(f"Recorte [{start}, {stop}) inválido para {len(self)} pontos.")
        return SeriesFrame(
            self.timestamps[start:stop], self.values[start:stop], self.frequency, self.series_id
        )


def _fingerprint(
    series_id: str,
    frequency: str,
    timestamps: pd.DatetimeIndex,
    values: npt.NDArray[np.float64],
) -> str:
    digest = hashlib.sha256()
    meta = {
        "schema": "series-frame/1",
        "series_id": series_id,
        "frequency": frequency,
        "tz": None if timestamps.tz is None else str(timestamps.tz),
        "n": len(values),
    }
    digest.update(canonical_json(meta).encode("utf-8"))
    digest.update(np.ascontiguousarray(epoch_ns(timestamps), dtype="<i8").tobytes())
    # + 0.0 maps -0.0 to 0.0 so equal values always hash equally.
    digest.update(np.ascontiguousarray(values + 0.0, dtype="<f8").tobytes())
    return digest.hexdigest()
