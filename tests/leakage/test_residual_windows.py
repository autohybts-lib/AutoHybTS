"""T06: every residual feature consumed by the corrector is available at the row's origin."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from autohybts.exceptions import InsufficientHistoryError
from autohybts.residuals.features import latest_window, training_rows
from autohybts.residuals.provenance import ResidualRecord, ResidualSeries


def series_for(targets: list[int]) -> ResidualSeries:
    return ResidualSeries(
        tuple(
            ResidualRecord(t - 1, t, 1, float(t), 0.0, float(t), t, "historical_oof_1step")
            for t in targets
        )
    )


def test_rows_and_origins() -> None:
    X, y, origins = training_rows(series_for(list(range(10, 15))), 2)
    assert X.tolist() == [[10.0, 11.0], [11.0, 12.0], [12.0, 13.0]]
    assert y.tolist() == [12.0, 13.0, 14.0]
    assert origins.tolist() == [11, 12, 13]


def test_rows_across_gaps_are_skipped_not_padded() -> None:
    _, y, _ = training_rows(series_for([1, 2, 3, 7, 8, 9, 10]), 2)
    assert y.tolist() == [3.0, 9.0, 10.0]  # rows for 7 and 8 would straddle the gap


@given(
    targets=st.lists(st.integers(0, 200), min_size=3, max_size=80, unique=True),
    window=st.integers(1, 10),
)
def test_every_feature_is_available_at_its_row_origin(targets: list[int], window: int) -> None:
    series = series_for(sorted(targets))
    try:
        X, y, origins = training_rows(series, window)
    except InsufficientHistoryError:
        return
    # Residual values equal their target positions here, so features are positions.
    assert (origins[:, None] >= X).all()
    assert (X.max(axis=1) == origins).all()  # the window ends right at the origin
    assert (y == origins + 1).all()


def test_latest_window_ends_at_the_origin() -> None:
    series = series_for(list(range(20, 30)))
    assert latest_window(series, 3, 29).tolist() == [27.0, 28.0, 29.0]


def test_latest_window_refuses_stale_or_broken_windows() -> None:
    with pytest.raises(InsufficientHistoryError):
        latest_window(series_for(list(range(20, 30))), 3, 30)  # last residual is not at the origin
    with pytest.raises(InsufficientHistoryError):
        latest_window(series_for([20, 21, 25, 26]), 3, 26)  # gap inside the window
    with pytest.raises(InsufficientHistoryError):
        latest_window(series_for([20, 21]), 3, 21)


def test_window_later_than_origin_is_leakage() -> None:
    from autohybts.exceptions import TemporalLeakageError

    with pytest.raises(TemporalLeakageError):
        latest_window(series_for(list(range(20, 30))), 3, 28)


def test_no_contiguous_window() -> None:
    with pytest.raises(InsufficientHistoryError, match="contígua"):
        training_rows(series_for([1, 3, 5, 7]), 2)
