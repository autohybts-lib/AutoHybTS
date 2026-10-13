"""``seasonal_period="auto"``: the search period comes from data every origin has seen."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autohybts import AutoHybridExperiment
from autohybts.datasets import get_data
from autohybts.exceptions import ConfigurationError


def test_white_noise_searches_without_a_period_but_keeps_the_metric_scale() -> None:
    exp = AutoHybridExperiment(fh=14, seasonal_period="auto").fit(get_data("white_noise_daily"))
    assert exp.seasonal_period_ == 7
    assert exp.search_period_ == 1
    assert exp.seasonality_ is not None
    assert not exp.seasonality_["seasonal"].iloc[0]
    assert exp._space(with_residuals=False).seasonal_period == 1


def test_a_weekly_cycle_keeps_the_period() -> None:
    exp = AutoHybridExperiment(fh=14, seasonal_period="auto").fit(get_data("synthetic_daily"))
    assert exp.search_period_ == 7
    assert bool(exp.seasonality_["seasonal"].iloc[0])


def test_few_cycles_keep_the_period_because_the_test_has_no_power() -> None:
    index = pd.date_range("2016-01-01", periods=60, freq="MS")
    noise = pd.Series(np.random.default_rng(3).normal(size=60) + 10, index=index)
    exp = AutoHybridExperiment(fh=6, fold=2, seasonal_period="auto").fit(noise)
    assert exp.seasonality_ is not None
    assert exp.seasonality_["n_cycles"].iloc[0] < 6
    assert bool(exp.seasonality_["keep_period"].iloc[0])
    assert exp.search_period_ == 12


def test_default_and_explicit_periods_do_not_test() -> None:
    y = get_data("synthetic_daily")
    default = AutoHybridExperiment(fh=14).fit(y)
    assert default.search_period_ == default.seasonal_period_ == 7
    assert default.seasonality_ is None
    explicit = AutoHybridExperiment(fh=14, seasonal_period=14).fit(y)
    assert explicit.search_period_ == explicit.seasonal_period_ == 14


@pytest.mark.parametrize("value", ["weekly", 0, True, 2.5])
def test_invalid_periods_are_refused(value: object) -> None:
    with pytest.raises(ConfigurationError, match="seasonal_period"):
        AutoHybridExperiment(fh=14, seasonal_period=value).fit(  # type: ignore[arg-type]
            get_data("synthetic_daily")
        )


def test_the_decision_ignores_everything_after_the_first_origin() -> None:
    y = get_data("synthetic_daily")
    exp = AutoHybridExperiment(fh=14, seasonal_period="auto").fit(y)
    first = min(exp.plan_.search_origins, key=lambda o: o.origin_index).origin_index
    changed = y.copy()
    changed.iloc[first + 1 :] = np.random.default_rng(0).normal(size=len(y) - first - 1)
    other = AutoHybridExperiment(fh=14, seasonal_period="auto").fit(changed)
    pd.testing.assert_frame_equal(exp.seasonality_, other.seasonality_)


def test_report_has_one_row_per_series_and_period() -> None:
    y = get_data("synthetic_daily")
    exp = AutoHybridExperiment(fh=14).fit({"a": y, "b": y * 2})
    report = exp.seasonality_report(periods=[7, 30])
    assert list(report.index) == [("a", 7), ("a", 30), ("b", 7), ("b", 30)]
    assert {"strength", "p_value", "seasonal", "seasonal_differences"} <= set(report.columns)
    assert bool(report.loc[("a", 7), "seasonal"])
