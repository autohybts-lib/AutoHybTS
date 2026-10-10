"""Reading user files into a validated ``SeriesFrame`` with explicit column mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from autohybts.data.schema import SINGLE_SERIES_ID, SeriesFrame
from autohybts.data.validation import MissingPolicy

__all__ = ["read_csv"]


def read_csv(
    path: str | Path,
    *,
    target: str,
    date_col: str,
    frequency: str | None = None,
    series_id: str = SINGLE_SERIES_ID,
    missing: MissingPolicy = "reject",
    **read_csv_kwargs: Any,
) -> SeriesFrame:
    """Read a CSV and validate it; extra keyword arguments go to ``pandas.read_csv``.

    Columns are never guessed: ``target`` and ``date_col`` must be named explicitly.
    """

    data = pd.read_csv(path, **read_csv_kwargs)
    return SeriesFrame.from_pandas(
        data,
        target=target,
        date_col=date_col,
        frequency=frequency,
        series_id=series_id,
        missing=missing,
    )
