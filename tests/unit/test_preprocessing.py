from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from autohybts.data.preprocessing import DifferenceTransform, IdentityTransform
from autohybts.exceptions import ConfigurationError, InsufficientHistoryError


def test_identity() -> None:
    fitted, transformed = IdentityTransform().fit_transform([1.0, 2.0])
    assert transformed.tolist() == [1.0, 2.0]
    assert fitted.inverse_forecast([3.0]).tolist() == [3.0]


def test_first_difference_inverts_from_last_training_value() -> None:
    fitted, transformed = DifferenceTransform(1).fit_transform([10.0, 12.0, 15.0])
    assert transformed.tolist() == [2.0, 3.0]
    assert fitted.inverse_forecast([1.0, 1.0, -2.0]).tolist() == [16.0, 17.0, 15.0]


def test_second_difference_continues_a_quadratic() -> None:
    y = np.arange(8.0) ** 2
    fitted, transformed = DifferenceTransform(2).fit_transform(y[:5])
    assert np.allclose(transformed, 2.0)
    assert np.allclose(fitted.inverse_forecast([2.0, 2.0, 2.0]), y[5:])


@given(
    order=st.integers(min_value=1, max_value=3),
    y=st.lists(st.floats(-1e3, 1e3), min_size=5, max_size=40),
    h=st.integers(min_value=1, max_value=10),
)
def test_inverting_the_true_future_differences_recovers_the_future(
    order: int, y: list[float], h: int
) -> None:
    full = np.asarray(y + y[:h])  # any continuation works
    train = full[: len(y)]
    fitted, _ = DifferenceTransform(order).fit_transform(train)
    future_diff = np.diff(full, n=order)[len(y) - order :]
    assert np.allclose(fitted.inverse_forecast(future_diff), full[len(y) :], atol=1e-6)


@pytest.mark.parametrize("order", [0, -1, True, 1.5])
def test_invalid_order(order: object) -> None:
    with pytest.raises(ConfigurationError):
        DifferenceTransform(order)


def test_short_training() -> None:
    with pytest.raises(InsufficientHistoryError):
        DifferenceTransform(2).fit_transform([1.0, 2.0])
