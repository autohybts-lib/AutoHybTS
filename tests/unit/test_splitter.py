from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalSplitter, split_sealed_test
from autohybts.exceptions import ConfigurationError, InsufficientHistoryError


def frame(n: int, series_id: str = "s1") -> SeriesFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="D")
    return SeriesFrame.from_arrays(index, np.arange(float(n)), series_id=series_id)


def test_sealed_split_sizes() -> None:
    dev, sealed = split_sealed_test(frame(50), 14)
    assert sealed is not None
    assert (len(dev), sealed.size, sealed.cutoff) == (36, 14, 36)
    assert sealed.start == dev.end + pd.Timedelta(days=1)
    assert sealed.series_id == "s1"


def test_zero_test_size_means_no_sealed_test() -> None:
    dev, sealed = split_sealed_test(frame(20), 0)
    assert sealed is None
    assert len(dev) == 20


@pytest.mark.parametrize("size", [20, 25])
def test_test_size_must_leave_history(size: int) -> None:
    with pytest.raises(InsufficientHistoryError):
        split_sealed_test(frame(20), size)


@pytest.mark.parametrize("size", [-1, 1.5, True])
def test_invalid_test_size(size: object) -> None:
    with pytest.raises(ConfigurationError):
        split_sealed_test(frame(20), size)


def test_origins_are_anchored_at_the_end_of_development() -> None:
    plan = TemporalSplitter(horizon=7, n_search_folds=2, n_select_folds=2, embargo=1).plan(
        frame(86)
    )
    assert [o.origin_index for o in plan.search_origins] == [56, 63]
    assert [o.origin_index for o in plan.select_origins] == [71, 78]
    assert plan.select_origins[-1].last_target_index == plan.dev_end == 85
    assert [o.fold_id for o in plan] == ["search-0", "search-1", "select-0", "select-1"]


def test_step_defaults_to_horizon_and_can_overlap() -> None:
    plan = TemporalSplitter(horizon=5, n_search_folds=1, n_select_folds=3, step=2).plan(frame(40))
    assert [o.origin_index for o in plan.select_origins] == [30, 32, 34]
    assert TemporalSplitter(horizon=5).effective_step == 5


def test_rolling_windows_have_fixed_length() -> None:
    plan = TemporalSplitter(
        horizon=3, n_search_folds=2, n_select_folds=2, fold_strategy="rolling", window_size=10
    ).plan(frame(40))
    assert {o.n_train for o in plan} == {10}
    assert all(o.train_start == o.origin_index - 9 for o in plan)


def test_expanding_windows_start_at_zero() -> None:
    plan = TemporalSplitter(horizon=3).plan(frame(40))
    assert {o.train_start for o in plan} == {0}


def test_required_length_is_exact() -> None:
    splitter = TemporalSplitter(horizon=7, n_search_folds=2, n_select_folds=3, embargo=2)
    # 14 (2 * horizon train) + 2 * 7 + 3 * 7 + 2
    assert splitter.required_length == 51
    assert len(splitter.plan(frame(51)).search_origins) == 2
    with pytest.raises(InsufficientHistoryError, match="exige ao menos 51"):
        splitter.plan(frame(50))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 0},
        {"horizon": 3, "n_search_folds": 0},
        {"horizon": 3, "n_select_folds": 0},
        {"horizon": 3, "step": 0},
        {"horizon": 3, "embargo": -1},
        {"horizon": 3, "min_train_size": 0},
        {"horizon": 3, "fold_strategy": "sliding"},
        {"horizon": 3, "window_size": 10},
        {"horizon": 3, "fold_strategy": "rolling"},
        {"horizon": 3, "fold_strategy": "rolling", "window_size": 4, "min_train_size": 5},
        {"horizon": True},
    ],
)
def test_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        TemporalSplitter(**kwargs)


def test_plan_accessors() -> None:
    plan = TemporalSplitter(horizon=4, n_search_folds=1, n_select_folds=1).plan(frame(20))
    origin = plan.select_origins[0]
    assert plan.train_values(origin).tolist() == list(np.arange(16.0))
    assert plan.target_values(origin).tolist() == [16.0, 17.0, 18.0, 19.0]
    context = plan.context(origin)
    assert context.origin_index == 15
    assert context.observed[-1] == 15.0


def test_foreign_origin_is_rejected() -> None:
    splitter = TemporalSplitter(horizon=4, n_search_folds=1, n_select_folds=1)
    plan_a, plan_b = splitter.plan(frame(20)), splitter.plan(frame(25))
    with pytest.raises(ConfigurationError, match="não pertence"):
        plan_a.target_values(plan_b.select_origins[0])


def test_to_frame_matches_origins() -> None:
    plan = TemporalSplitter(horizon=7, n_search_folds=2, n_select_folds=2).plan(frame(60))
    table = plan.to_frame()
    assert list(table.index) == ["search-0", "search-1", "select-0", "select-1"]
    assert list(table.columns) == [
        "layer",
        "train_start",
        "origin",
        "first_target",
        "last_target",
        "n_train",
        "horizon",
    ]
    last = table.loc["select-1"]
    assert last["last_target"] == pd.Timestamp("2024-02-29")
    assert last["first_target"] == last["origin"] + pd.Timedelta(days=1)


def test_fingerprint_changes_with_any_parameter_or_data() -> None:
    base = TemporalSplitter(horizon=5, n_search_folds=2, n_select_folds=2)
    data = frame(80)
    variants = [
        base.plan(data),
        TemporalSplitter(horizon=6, n_search_folds=2, n_select_folds=2).plan(data),
        TemporalSplitter(horizon=5, n_search_folds=3, n_select_folds=2).plan(data),
        TemporalSplitter(horizon=5, n_search_folds=2, n_select_folds=3).plan(data),
        TemporalSplitter(horizon=5, n_search_folds=2, n_select_folds=2, step=4).plan(data),
        TemporalSplitter(horizon=5, n_search_folds=2, n_select_folds=2, embargo=1).plan(data),
        TemporalSplitter(horizon=5, n_search_folds=2, n_select_folds=2, min_train_size=11).plan(
            data
        ),
        TemporalSplitter(
            horizon=5, n_search_folds=2, n_select_folds=2, fold_strategy="rolling", window_size=20
        ).plan(data),
        base.plan(frame(81)),
        base.plan(frame(80, series_id="s2")),
    ]
    assert len({plan.fingerprint for plan in variants}) == len(variants)


def test_fingerprint_is_reproducible_and_ignores_default_spelling() -> None:
    data = frame(80)
    implicit = TemporalSplitter(horizon=5).plan(data)
    explicit = TemporalSplitter(horizon=5, step=5, min_train_size=10).plan(data)
    assert (
        implicit.fingerprint
        == explicit.fingerprint
        == TemporalSplitter(horizon=5).plan(frame(80)).fingerprint
    )


def test_origins_by_layer() -> None:
    plan = TemporalSplitter(horizon=3, n_search_folds=2, n_select_folds=1).plan(frame(40))
    assert plan.origins("search") == plan.search_origins
    assert plan.origins("select") == plan.select_origins
    assert plan.origins() == plan.search_origins + plan.select_origins
    with pytest.raises(ConfigurationError, match="Camada"):
        plan.origins("test")


def test_plan_requires_a_series_frame() -> None:
    with pytest.raises(ConfigurationError, match="SeriesFrame"):
        TemporalSplitter(horizon=2).plan(frame(40).to_series())
