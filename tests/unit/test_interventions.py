"""Intervention identification and isolation from sealed observations."""

import numpy as np
import pandas as pd
import pytest

from autohybts import AutoHybridExperiment
from autohybts.data.interventions import intervention_report
from autohybts.data.schema import SeriesFrame
from autohybts.exceptions import ConfigurationError


@pytest.mark.parametrize("kind", ["AO", "LS", "TC"])
def test_identifies_known_intervention(kind):
    y = np.random.default_rng(42).normal(0, 0.1, 100)
    if kind == "AO":
        y[50] += 10
    elif kind == "LS":
        y[50:] += 10
    else:
        y[50:] += 10 * 0.7 ** np.arange(50)
    frame = SeriesFrame.from_series(pd.Series(y, index=pd.date_range("2020", periods=100)))
    original = frame.values.copy()
    report = intervention_report(frame, max_events=1)
    assert report.iloc[0]["type"] == kind
    assert report.iloc[0]["position"] == 50
    np.testing.assert_array_equal(frame.values, original)


def test_sealed_test_cannot_change_report():
    y = pd.Series(
        np.random.default_rng(0).normal(size=140), index=pd.date_range("2020", periods=140)
    )
    first = AutoHybridExperiment(fh=7, seasonal_period=1).fit(y)
    changed = y.copy()
    changed.iloc[-7:] = 1e9
    second = AutoHybridExperiment(fh=7, seasonal_period=1).fit(changed)
    pd.testing.assert_frame_equal(first.outlier_report(), second.outlier_report())


def test_constant_history_has_no_events():
    frame = SeriesFrame.from_series(pd.Series(np.ones(50), index=pd.date_range("2020", periods=50)))
    assert intervention_report(frame).empty


@pytest.mark.parametrize("options", [{"types": ("IO",)}, {"decay": 1}, {"threshold": 0}])
def test_invalid_options(options):
    frame = SeriesFrame.from_series(pd.Series(np.ones(50), index=pd.date_range("2020", periods=50)))
    with pytest.raises(ConfigurationError):
        intervention_report(frame, **options)
