"""Residual records, series and generation for the three sources (§7.2, ADR-0002)."""

from __future__ import annotations

import numpy as np
import pytest

from autohybts.adapters.naive import NaiveForecaster, SeasonalNaiveForecaster
from autohybts.adapters.statsmodels_arima import ArimaForecaster
from autohybts.exceptions import (
    ConfigurationError,
    InsufficientHistoryError,
    UnsupportedCapabilityError,
)
from autohybts.residuals.generation import ResidualSettings, build_residuals
from autohybts.residuals.provenance import ResidualRecord, ResidualSeries

Y = np.sin(np.arange(100) / 5) * 10 + np.arange(100) * 0.1


def record(target: int, origin: int | None = None, residual: float = 1.0) -> ResidualRecord:
    origin = target - 1 if origin is None else origin
    return ResidualRecord(
        origin, target, target - origin, residual, 0.0, residual, target, "in_sample"
    )


def test_record_invariants() -> None:
    assert record(5).available_at == 5
    with pytest.raises(ConfigurationError, match="incoerente"):
        ResidualRecord(4, 5, 2, 1.0, 0.0, 1.0, 5, "in_sample")
    with pytest.raises(ConfigurationError, match="available_at"):
        ResidualRecord(4, 5, 1, 1.0, 0.0, 1.0, 4, "in_sample")
    with pytest.raises(ConfigurationError, match="y_true - y_base"):
        ResidualRecord(4, 5, 1, 1.0, 0.0, 2.0, 5, "in_sample")
    with pytest.raises(ConfigurationError, match="Fonte"):
        ResidualRecord(4, 5, 1, 1.0, 0.0, 1.0, 5, "future")


def test_series_rejects_unordered_or_duplicate_targets() -> None:
    with pytest.raises(ConfigurationError, match="ordem"):
        ResidualSeries((record(5), record(4)))
    with pytest.raises(ConfigurationError, match="repetido"):
        ResidualSeries((record(5), record(5, origin=3)))


def test_series_arrays_and_audit_table() -> None:
    series = ResidualSeries((record(3, residual=0.5), record(4, residual=-1.0)))
    assert series.targets.tolist() == [3, 4]
    assert series.residuals.tolist() == [0.5, -1.0]
    assert not series.residuals.flags.writeable
    table = series.to_frame()
    assert list(table.columns) == [
        "origin",
        "target",
        "horizon",
        "y_true",
        "y_base",
        "residual",
        "available_at",
        "source",
    ]
    assert series.extend((record(5),)).targets.tolist() == [3, 4, 5]
    assert series.sources == ("in_sample",)


def test_oof_one_step_uses_a_base_fitted_before_the_tail() -> None:
    template = SeasonalNaiveForecaster(season_length=7)
    series = build_residuals(
        template, template.clone().fit(Y), Y, "historical_oof_1step", ResidualSettings()
    )
    assert series.targets.tolist() == list(range(50, 100))
    assert {r.horizon for r in series.records} == {1}
    # Seasonal naive one step ahead: y[t] - y[t - 7].
    assert np.allclose(series.residuals, Y[50:] - Y[43:93])


def test_oof_multistep_blocks_carry_horizons() -> None:
    template = NaiveForecaster()
    settings = ResidualSettings(block_size=10)
    series = build_residuals(
        template, template.clone().fit(Y), Y, "historical_oof_multistep", settings
    )
    assert series.targets.tolist() == list(range(50, 100))
    assert [r.horizon for r in series.records[:12]] == [*range(1, 11), 1, 2]
    first = series.records[0]
    assert first.origin == 49
    assert first.y_base == Y[49]  # naive repeats y[49] over the block


def test_default_block_splits_the_tail_in_two() -> None:
    template = NaiveForecaster()
    series = build_residuals(
        template, template.clone().fit(Y), Y, "historical_oof_multistep", ResidualSettings()
    )
    assert max(r.horizon for r in series.records) == 25


def test_tail_is_capped() -> None:
    template = NaiveForecaster()
    settings = ResidualSettings(max_tail=20)
    series = build_residuals(template, template.clone().fit(Y), Y, "historical_oof_1step", settings)
    assert len(series) == 20


def test_in_sample_needs_fitted_values() -> None:
    template = NaiveForecaster()
    with pytest.raises(UnsupportedCapabilityError, match="in_sample"):
        build_residuals(template, template.clone().fit(Y), Y, "in_sample", ResidualSettings())


def test_in_sample_drops_the_burn_in() -> None:
    pytest.importorskip("statsmodels")
    template = ArimaForecaster(order=(1, 1, 0))
    fitted = template.clone().fit(Y)
    settings = ResidualSettings(tail_fraction=0.9, min_burn_in=15)
    series = build_residuals(template, fitted, Y, "in_sample", settings)
    assert series.targets[0] == 15
    assert np.allclose(series.residuals, Y[15:] - fitted.fitted_values()[15:])


def test_prefix_must_fit_the_base() -> None:
    template = SeasonalNaiveForecaster(season_length=30)
    y = Y[:50]
    with pytest.raises(InsufficientHistoryError, match="prefixo"):
        build_residuals(
            template,
            template.clone().fit(y),
            y,
            "historical_oof_1step",
            ResidualSettings(tail_fraction=0.9),
        )


def test_unknown_source() -> None:
    template = NaiveForecaster()
    with pytest.raises(ConfigurationError):
        build_residuals(template, template, Y, "future", ResidualSettings())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tail_fraction": 0.0},
        {"tail_fraction": 1.0},
        {"max_tail": 0},
        {"block_size": True},
        {"min_burn_in": -1},
        {"min_corrector_rows": 1.5},
        {"on_failure": "ignore"},
    ],
)
def test_invalid_settings(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        ResidualSettings(**kwargs)


def test_generation_never_reads_past_the_vector() -> None:
    # Perturbing y after position k leaves every residual of a target before k unchanged
    # (the prefix model is fitted before the tail, which starts after k here).
    template = NaiveForecaster()
    base = build_residuals(
        template, template.clone().fit(Y), Y, "historical_oof_1step", ResidualSettings()
    )
    changed = Y.copy()
    changed[80:] += 100.0
    other = build_residuals(
        template, template.clone().fit(changed), changed, "historical_oof_1step", ResidualSettings()
    )
    before = series_before(base, 80)
    assert np.array_equal(before, series_before(other, 80))


def series_before(series: ResidualSeries, k: int) -> np.ndarray:
    return series.residuals[series.targets < k]
