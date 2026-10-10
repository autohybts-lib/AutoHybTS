from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from autohybts.data.windows import lag_matrix, latest_lags, normalize_lags
from autohybts.exceptions import ConfigurationError, InsufficientHistoryError


def test_integer_means_consecutive_lags() -> None:
    assert normalize_lags(3) == (1, 2, 3)
    assert normalize_lags([7, 1]) == (7, 1)


@pytest.mark.parametrize("bad", [0, -1, [], [0, 1], [1, 1], [1.5], True, [True]])
def test_invalid_lags(bad: object) -> None:
    with pytest.raises(ConfigurationError):
        normalize_lags(bad)


def test_lag_matrix_values() -> None:
    features, targets = lag_matrix(np.arange(6.0), [1, 3])
    assert targets.tolist() == [3.0, 4.0, 5.0]
    assert features.tolist() == [[2.0, 0.0], [3.0, 1.0], [4.0, 2.0]]


def test_latest_lags_is_the_next_step_row() -> None:
    assert latest_lags(np.arange(6.0), [1, 3]).tolist() == [5.0, 3.0]


def test_short_history() -> None:
    with pytest.raises(InsufficientHistoryError):
        lag_matrix(np.arange(3.0), 3)
    with pytest.raises(InsufficientHistoryError):
        latest_lags(np.arange(2.0), 3)


@given(
    n=st.integers(min_value=2, max_value=60),
    lags=st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=4, unique=True),
)
def test_features_only_use_strictly_earlier_values(n: int, lags: list[int]) -> None:
    if n <= max(lags):
        return
    # With values equal to their positions, every feature must be smaller than its target.
    features, targets = lag_matrix(np.arange(float(n)), lags)
    assert (features < targets[:, None]).all()
    assert np.array_equal(targets[:, None] - features, np.broadcast_to(lags, features.shape))


def test_matrix_input_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="1-D"):
        lag_matrix(np.ones((4, 2)), 1)
