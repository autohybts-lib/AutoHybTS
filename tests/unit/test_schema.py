"""T01: a minimal regular series is accepted with correct schema, dtype and identity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autohybts.data.schema import SINGLE_SERIES_ID, SeriesFrame
from autohybts.exceptions import ConfigurationError, DataValidationError


def daily(n: int = 10, start: str = "2024-01-01") -> pd.Series:
    return pd.Series(
        np.arange(n, dtype=float), index=pd.date_range(start, periods=n, freq="D"), name="y"
    )


def test_regular_series_is_accepted() -> None:
    frame = SeriesFrame.from_series(daily())
    assert len(frame) == 10
    assert frame.frequency == "D"
    assert frame.series_id == SINGLE_SERIES_ID
    assert frame.values.dtype == np.float64
    assert frame.start == pd.Timestamp("2024-01-01")
    assert frame.end == pd.Timestamp("2024-01-10")


def test_values_are_read_only_copies() -> None:
    source = np.arange(5, dtype=float)
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=5, freq="D"), source)
    source[0] = 99.0
    assert frame.values[0] == 0.0
    with pytest.raises(ValueError, match="read-only"):
        frame.values[0] = 1.0


def test_integers_are_coerced_to_float64() -> None:
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=3, freq="D"), [1, 2, 3])
    assert frame.values.dtype == np.float64


def test_to_series_round_trip_is_a_writable_copy() -> None:
    frame = SeriesFrame.from_series(daily())
    series = frame.to_series()
    series.iloc[0] = 50.0
    assert frame.values[0] == 0.0
    assert SeriesFrame.from_series(frame.to_series()).fingerprint == frame.fingerprint


def test_from_pandas_with_date_column() -> None:
    data = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=6, freq="MS"), "sales": 1.0})
    frame = SeriesFrame.from_pandas(data, target="sales", date_col="date")
    assert frame.frequency == "MS"


def test_from_pandas_with_datetime_index() -> None:
    data = daily().to_frame("sales")
    assert len(SeriesFrame.from_pandas(data, target="sales")) == 10


def test_from_pandas_reports_missing_columns() -> None:
    data = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, freq="D"), "v": 1.0})
    with pytest.raises(ConfigurationError, match=r"'sales'.*Disponíveis"):
        SeriesFrame.from_pandas(data, target="sales", date_col="date")


def test_period_index_is_converted() -> None:
    series = pd.Series([1.0, 2.0, 3.0], index=pd.period_range("2024-01", periods=3, freq="M"))
    frame = SeriesFrame.from_series(series)
    assert frame.frequency == "MS"
    assert frame.start == pd.Timestamp("2024-01-01")


def test_non_temporal_index_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="índice temporal"):
        SeriesFrame.from_series(pd.Series([1.0, 2.0, 3.0]))


def test_declared_frequency_must_match() -> None:
    with pytest.raises(DataValidationError, match="não é regular"):
        SeriesFrame.from_series(daily(), frequency="W")


def test_head_and_tail_keep_identity_and_frequency() -> None:
    frame = SeriesFrame.from_series(daily(), series_id="loja-1")
    head, tail = frame.head(7), frame.tail(3)
    assert (len(head), len(tail)) == (7, 3)
    assert head.series_id == tail.series_id == "loja-1"
    assert tail.start == pd.Timestamp("2024-01-08")
    with pytest.raises(ConfigurationError):
        frame.head(0)


def test_fingerprint_is_stable_and_sensitive() -> None:
    base = SeriesFrame.from_series(daily())
    assert SeriesFrame.from_series(daily()).fingerprint == base.fingerprint

    changed_value = daily()
    changed_value.iloc[3] += 1e-9
    variants = [
        SeriesFrame.from_series(changed_value),
        SeriesFrame.from_series(daily(start="2024-01-02")),
        SeriesFrame.from_series(daily(), series_id="other"),
        SeriesFrame.from_series(daily(n=11)),
    ]
    assert len({base.fingerprint, *(frame.fingerprint for frame in variants)}) == 5


def test_fingerprint_ignores_negative_zero_and_index_unit() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    a = SeriesFrame.from_arrays(index, [0.0, 1.0, 2.0])
    b = SeriesFrame.from_arrays(index.as_unit("s"), [-0.0, 1.0, 2.0])
    assert a.fingerprint == b.fingerprint


def test_empty_series_id_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="series_id"):
        SeriesFrame.from_series(daily(), series_id="")


def test_from_pandas_without_dates_needs_a_temporal_index() -> None:
    with pytest.raises(DataValidationError, match="date_col"):
        SeriesFrame.from_pandas(pd.DataFrame({"v": [1.0, 2.0, 3.0]}), target="v")


def test_two_dimensional_values_are_rejected() -> None:
    with pytest.raises(DataValidationError, match="unidimensional"):
        SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=2), [[1.0], [2.0]])
