"""T02 and §34.5: invalid series fail loudly and are never repaired silently."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autohybts.data.schema import SeriesFrame
from autohybts.data.validation import canonical_frequency, epoch_ns
from autohybts.exceptions import (
    ConfigurationError,
    DataValidationError,
    UnsupportedCapabilityError,
)


def build(dates: list[str], values: list[float] | None = None, **kwargs: object) -> SeriesFrame:
    values = values if values is not None else [float(i) for i in range(len(dates))]
    return SeriesFrame.from_arrays(pd.to_datetime(dates), values, **kwargs)


def test_unordered_timestamps_are_rejected_not_sorted() -> None:
    with pytest.raises(DataValidationError, match=r"fora de ordem.*posição 2"):
        build(["2024-01-01", "2024-01-03", "2024-01-02"], frequency="D")


def test_duplicated_timestamps_are_rejected() -> None:
    with pytest.raises(DataValidationError, match=r"duplicada.*posições 1 e 2"):
        build(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"], frequency="D")


def test_gaps_are_rejected_not_filled() -> None:
    with pytest.raises(DataValidationError, match=r"esperado 2024-01-03.*posição 2"):
        build(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"], frequency="D")


def test_irregular_series_cannot_be_inferred() -> None:
    with pytest.raises(DataValidationError, match="inferir uma frequência regular"):
        build(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"])


def test_inference_needs_three_points() -> None:
    with pytest.raises(DataValidationError, match="ao menos 3"):
        build(["2024-01-01", "2024-01-02"])
    assert len(build(["2024-01-01", "2024-01-02"], frequency="D")) == 2


def test_start_off_the_frequency_grid_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="esperado 2024-02-01"):
        build(["2024-01-15", "2024-02-15", "2024-03-15"], frequency="MS")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_missing_or_non_finite_values_are_rejected(bad: float) -> None:
    with pytest.raises(DataValidationError, match="primeiro na posição 1"):
        build(["2024-01-01", "2024-01-02", "2024-01-03"], [1.0, bad, 3.0])


def test_nullable_pandas_missing_values_are_rejected() -> None:
    series = pd.Series(
        pd.array([1, None, 3], dtype="Int64"), index=pd.date_range("2024-01-01", periods=3)
    )
    with pytest.raises(DataValidationError, match="ausente"):
        SeriesFrame.from_series(series)


def test_non_numeric_values_are_rejected() -> None:
    with pytest.raises(DataValidationError, match="numéricos"):
        build(["2024-01-01", "2024-01-02", "2024-01-03"], ["a", "b", "c"])


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="tamanhos"):
        build(["2024-01-01", "2024-01-02", "2024-01-03"], [1.0, 2.0])


def test_nat_is_rejected() -> None:
    index = pd.DatetimeIndex(["2024-01-01", None, "2024-01-03"])
    with pytest.raises(DataValidationError, match="NaT"):
        SeriesFrame.from_arrays(index, [1.0, 2.0, 3.0], frequency="D")


def test_empty_series_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="vazia"):
        SeriesFrame.from_arrays(pd.DatetimeIndex([]), [], frequency="D")


def test_unparseable_dates_are_rejected() -> None:
    with pytest.raises(DataValidationError, match="datas"):
        SeriesFrame.from_arrays(["ontem", "hoje"], [1.0, 2.0], frequency="D")


def test_unknown_missing_policy_is_not_silently_accepted() -> None:
    with pytest.raises(UnsupportedCapabilityError, match="reject"):
        SeriesFrame.from_arrays(
            pd.date_range("2024-01-01", periods=3),
            [1.0, 2.0, 3.0],
            missing="ffill",
        )


# Aliases whose spelling is the same in pandas 2.x and 3.x (e.g. "h"/"H" changed).
@pytest.mark.parametrize(("alias", "expected"), [("D", "D"), ("MS", "MS"), ("W", "W-SUN")])
def test_canonical_frequency(alias: str, expected: str) -> None:
    assert canonical_frequency(alias) == expected


@pytest.mark.parametrize("alias", ["banana", "", "3Q-XYZ"])
def test_invalid_frequency_is_a_configuration_error(alias: str) -> None:
    with pytest.raises(ConfigurationError, match="Frequência inválida"):
        canonical_frequency(alias)


def test_timezone_aware_series_hash_by_utc_instant() -> None:
    utc = pd.date_range("2024-01-01 03:00", periods=3, freq="h", tz="UTC")
    local = utc.tz_convert("America/Recife")
    assert (epoch_ns(utc) == epoch_ns(local)).all()
    frame = SeriesFrame.from_arrays(local, [1.0, 2.0, 3.0])
    assert str(frame.timestamps.tz) == "America/Recife"
