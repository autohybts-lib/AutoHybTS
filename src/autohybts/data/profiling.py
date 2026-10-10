"""Readable data quality report of the development history (§34.5).

Missing values, duplicates and irregular timestamps are already refused when a
``SeriesFrame`` is built, so the profile reports what is left to look at: extent,
level, spread, flat stretches and extreme values.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from autohybts.data.schema import SeriesFrame

__all__ = ["profile_series"]


def profile_series(frame: SeriesFrame) -> pd.DataFrame:
    """One-column table (``value``) indexed by indicator name."""

    values = frame.values
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826
    robust_z = np.abs(values - median) / mad if mad > 0 else np.zeros_like(values)
    rows = {
        "série": frame.series_id,
        "início": frame.start,
        "fim": frame.end,
        "observações": len(frame),
        "frequência": frame.frequency,
        "faltantes": 0,
        "datas duplicadas": 0,
        "média": float(np.mean(values)),
        "desvio-padrão": float(np.std(values)),
        "mínimo": float(np.min(values)),
        "máximo": float(np.max(values)),
        "constante": bool(np.all(values == values[0])),
        "maior trecho constante": _longest_flat_run(values),
        "extremos (|z robusto| > 4)": int(np.sum(robust_z > 4)),
    }
    return pd.DataFrame({"value": pd.Series(rows, dtype=object)})


def _longest_flat_run(values: np.ndarray) -> int:
    longest = current = 1
    for previous, value in itertools.pairwise(values):
        current = current + 1 if value == previous else 1
        longest = max(longest, current)
    return longest
