"""Indexed forecasts (architecture §11.1, §34.7): never a bare array without dates."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from autohybts.exceptions import ConfigurationError

__all__ = ["FORECAST_COLUMNS", "ForecastFrame"]

FORECAST_COLUMNS: Final = (
    "series_id",
    "candidate_id",
    "fold_id",
    "origin",
    "timestamp",
    "horizon",
    "y_pred",
    "y_true",
)
_GROUP: Final = ["series_id", "candidate_id", "origin"]


class ForecastFrame:
    """Long table of forecasts, one row per (series, candidate, origin, step).

    Invariants checked at construction: steps ``1..H`` appear exactly once per group,
    every ``timestamp`` is ``origin`` plus ``horizon`` periods of ``frequency``, and
    ``y_pred`` is finite. ``y_true`` is ``NaN`` for forecasts of the unknown future.
    """

    __slots__ = ("_frame", "frequency")

    def __init__(self, frame: pd.DataFrame, frequency: str) -> None:
        missing = [column for column in FORECAST_COLUMNS if column not in frame.columns]
        if missing:
            raise ConfigurationError(f"ForecastFrame sem as colunas {missing}.")
        data = frame.loc[:, list(FORECAST_COLUMNS)].reset_index(drop=True).copy()
        data["horizon"] = data["horizon"].astype("int64")
        data["y_pred"] = data["y_pred"].astype("float64")
        data["y_true"] = data["y_true"].astype("float64")
        _validate(data, frequency)
        self._frame = data
        self.frequency = frequency

    @classmethod
    def concat(cls, frames: list[ForecastFrame]) -> ForecastFrame:
        if not frames:
            raise ConfigurationError("Nenhuma previsão para concatenar.")
        frequencies = {frame.frequency for frame in frames}
        if len(frequencies) != 1:
            raise ConfigurationError(f"Frequências diferentes: {sorted(frequencies)}.")
        return cls(
            pd.concat([frame._frame for frame in frames], ignore_index=True), frames[0].frequency
        )

    def to_pandas(self) -> pd.DataFrame:
        """A copy of the table."""

        return self._frame.copy()

    def __len__(self) -> int:
        return len(self._frame)

    def __repr__(self) -> str:
        return f"ForecastFrame(rows={len(self)}, frequency={self.frequency!r})"


def _validate(data: pd.DataFrame, frequency: str) -> None:
    if not np.isfinite(data["y_pred"].to_numpy()).all():
        raise ConfigurationError("ForecastFrame com previsões não finitas.")
    for key, group in data.groupby(_GROUP, sort=False):
        steps = group["horizon"].to_numpy()
        if not np.array_equal(np.sort(steps), np.arange(1, len(steps) + 1)):
            raise ConfigurationError(
                f"Passos {sorted(steps.tolist())} em {key}; esperado 1..{len(steps)} sem repetição."
            )
        origin = pd.Timestamp(group["origin"].iloc[0])
        expected = pd.date_range(origin, periods=len(steps) + 1, freq=frequency)[steps]
        if not np.array_equal(
            pd.DatetimeIndex(group["timestamp"]).as_unit("ns").to_numpy(),
            expected.as_unit("ns").to_numpy(),
        ):
            raise ConfigurationError(f"Datas incoerentes com a origem e a frequência em {key}.")
