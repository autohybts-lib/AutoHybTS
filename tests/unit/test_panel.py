"""Panels of series: validation, per-series sealed tests and plans, unit ids."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autohybts.data.panel import PanelFrame, PanelPlan, plan_panel, split_panel_sealed_test
from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalCrossValidation, TemporalSplitter
from autohybts.domain.results import unit_key
from autohybts.exceptions import (
    ConfigurationError,
    DataValidationError,
    InsufficientHistoryError,
)


def daily(n: int, seed: int, start: str = "2024-01-01") -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(10, 1, n), index=pd.date_range(start, periods=n, freq="D"))


def test_from_dict_and_from_long_agree() -> None:
    panel = PanelFrame.from_dict({"a": daily(60, 0), "b": daily(80, 1)})
    assert panel.series_ids == ("a", "b")
    assert panel.frequency == "D"
    assert not panel.is_single
    assert [len(f) for f in panel] == [60, 80]
    long = panel.to_long()
    again = PanelFrame.from_long(long, series_col="series_id", date_col="timestamp", target="value")
    assert again.fingerprint == panel.fingerprint


def test_panel_rejects_mixed_frequencies_and_reserved_id() -> None:
    weekly = pd.Series(range(30), index=pd.date_range("2024-01-07", periods=30, freq="W"))
    with pytest.raises(DataValidationError, match="frequências"):
        PanelFrame.from_dict({"a": daily(60, 0), "b": weekly.astype(float)})
    with pytest.raises(DataValidationError, match="vazio"):
        PanelFrame({})
    single = SeriesFrame.from_series(daily(60, 0))
    with pytest.raises(DataValidationError, match="reservado"):
        PanelFrame(
            {single.series_id: single, "b": SeriesFrame.from_series(daily(60, 1), series_id="b")}
        )
    with pytest.raises(ConfigurationError, match="disponíveis"):
        PanelFrame.from_dict({"a": daily(60, 0)})["z"]


def test_sealed_test_is_split_per_series() -> None:
    panel = PanelFrame.from_dict({"a": daily(60, 0), "b": daily(80, 1, start="2024-02-01")})
    development, sealed = split_panel_sealed_test(panel, 7)
    assert [len(f) for f in development] == [53, 73]
    assert set(sealed) == {"a", "b"}
    assert all(test.size == 7 for test in sealed.values())
    assert development["b"].end == panel["b"].end - pd.Timedelta(days=7)


def test_panel_plan_keys_folds_by_series() -> None:
    panel = PanelFrame.from_dict({"a": daily(120, 0), "b": daily(150, 1)})
    plan = plan_panel(TemporalSplitter(horizon=7, n_search_folds=2, n_select_folds=2), panel)
    assert isinstance(plan, PanelPlan)
    frame = plan.to_frame()
    assert frame.index.names == ["series_id", "fold_id"]
    assert len(frame) == 8
    # Each series keeps its own origins: the longer one ends later.
    assert plan["b"].history.end > plan["a"].history.end
    cv = plan_panel(TemporalCrossValidation(horizon=7, n_folds=3), panel)
    assert len(cv.to_frame()) == 6
    assert (
        plan.fingerprint
        != plan_panel(
            TemporalSplitter(horizon=7, n_search_folds=2, n_select_folds=3), panel
        ).fingerprint
    )


def test_short_series_is_named_in_the_error() -> None:
    panel = PanelFrame.from_dict({"long": daily(200, 0), "short": daily(20, 1)})
    with pytest.raises(InsufficientHistoryError, match="'short'"):
        plan_panel(TemporalSplitter(horizon=7, n_search_folds=3, n_select_folds=3), panel)


def test_unit_ids_keep_single_series_fold_ids() -> None:
    assert unit_key("__single__", "select-0") == "select-0"
    assert unit_key("a", "select-0") == "a:select-0"
