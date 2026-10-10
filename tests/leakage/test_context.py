"""TemporalContext refuses information that is not available at its origin (§11.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autohybts.domain.context import TemporalContext
from autohybts.exceptions import ConfigurationError, TemporalLeakageError

ORIGIN_TIME = pd.Timestamp("2024-01-10")


def context(**overrides: object) -> TemporalContext:
    fields: dict[str, object] = {
        "series_id": "s",
        "frequency": "D",
        "origin_index": 9,
        "origin_time": ORIGIN_TIME,
        "observed": np.arange(10.0),
    }
    fields.update(overrides)
    return TemporalContext(**fields)


def test_valid_context_is_read_only() -> None:
    ctx = context(residuals=[0.1, 0.2], residuals_available_at=[8, 9])
    for array in (ctx.observed, ctx.residuals, ctx.residuals_available_at):
        assert not array.flags.writeable


def test_observations_after_the_origin_are_leakage() -> None:
    with pytest.raises(TemporalLeakageError, match="até a posição 10"):
        context(observed=np.arange(11.0))


def test_observations_ending_before_the_origin_are_inconsistent() -> None:
    with pytest.raises(ConfigurationError, match="termina na posição 8"):
        context(observed=np.arange(9.0))


def test_offset_window_must_still_end_at_the_origin() -> None:
    assert context(observed=np.arange(4.0), observed_start=6).origin_index == 9
    with pytest.raises(TemporalLeakageError):
        context(observed=np.arange(5.0), observed_start=6)


def test_residual_available_after_the_origin_is_leakage() -> None:
    with pytest.raises(TemporalLeakageError, match="posição 10"):
        context(residuals=[0.1, 0.2], residuals_available_at=[9, 10])


def test_each_residual_needs_an_availability_time() -> None:
    with pytest.raises(ConfigurationError, match="available_at"):
        context(residuals=[0.1, 0.2], residuals_available_at=[9])


def test_empty_observations_are_rejected() -> None:
    with pytest.raises(ConfigurationError):
        context(observed=np.empty(0))
