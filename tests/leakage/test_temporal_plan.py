"""Temporal plan invariants: T03, T21, and the I2 parts of T04, T23 and UX03."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from autohybts.data.preprocessing import DifferenceTransform
from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import (
    SealedTest,
    TemporalPlan,
    TemporalSplitter,
    ensure_not_sealed,
    split_sealed_test,
)
from autohybts.exceptions import (
    ConfigurationError,
    InsufficientHistoryError,
    SealedTestAccessError,
    TemporalLeakageError,
)


def frame(values: np.ndarray) -> SeriesFrame:
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return SeriesFrame.from_arrays(index, values)


splitters = st.builds(
    lambda horizon, n_search, n_select, step, embargo, min_train, rolling, extra: TemporalSplitter(
        horizon=horizon,
        n_search_folds=n_search,
        n_select_folds=n_select,
        step=step,
        embargo=embargo,
        min_train_size=min_train,
        fold_strategy="rolling" if rolling else "expanding",
        window_size=min_train + extra if rolling else None,
    ),
    horizon=st.integers(1, 15),
    n_search=st.integers(1, 4),
    n_select=st.integers(1, 4),
    step=st.integers(1, 15),
    embargo=st.integers(0, 5),
    min_train=st.integers(1, 40),
    rolling=st.booleans(),
    extra=st.integers(0, 10),
)


@settings(max_examples=300, deadline=None)
@given(splitter=splitters, n=st.integers(5, 250))
def test_plan_exists_exactly_when_history_suffices(splitter: TemporalSplitter, n: int) -> None:
    data = frame(np.arange(float(n)))
    if n < splitter.required_length:
        with pytest.raises(InsufficientHistoryError):
            splitter.plan(data)
    else:
        splitter.plan(data)


@settings(max_examples=300, deadline=None)
@given(splitter=splitters, n=st.integers(5, 250))
def test_every_origin_is_causal_and_layers_do_not_overlap(
    splitter: TemporalSplitter, n: int
) -> None:
    assume(n >= splitter.required_length)
    plan = splitter.plan(frame(np.arange(float(n))))

    for origin in plan:
        # T03: training ends at the origin; targets start right after and stay in development.
        assert 0 <= origin.train_start <= origin.origin_index
        assert origin.n_train >= splitter.train_length_required
        assert origin.train_slice.stop == origin.target_slice.start == origin.origin_index + 1
        assert origin.last_target_index <= plan.dev_end
        assert len(plan.target_values(origin)) == splitter.horizon
        if splitter.fold_strategy == "rolling":
            assert origin.n_train == splitter.window_size

    # T21: selection never scores labels the search optimised on.
    search_targets = {t for o in plan.search_origins for t in range(*o.target_slice.indices(n))}
    select_targets = {t for o in plan.select_origins for t in range(*o.target_slice.indices(n))}
    assert search_targets.isdisjoint(select_targets)
    first_select = plan.select_origins[0].origin_index
    assert max(search_targets) + splitter.embargo <= first_select
    assert plan.select_origins[-1].last_target_index == plan.dev_end


def test_mutating_the_sealed_test_does_not_change_the_plan() -> None:
    # T04 (I2 part): the plan depends only on the development history.
    values = np.sin(np.arange(120.0))
    shocked = values.copy()
    shocked[-20:] += 1000.0
    splitter = TemporalSplitter(horizon=7, n_search_folds=3, n_select_folds=2)

    plans = []
    for series in (values, shocked):
        dev, sealed = split_sealed_test(frame(series), 20)
        assert sealed is not None
        plans.append(splitter.plan(dev))
    assert plans[0].fingerprint == plans[1].fingerprint
    assert plans[0].to_frame().equals(plans[1].to_frame())


def test_sealed_test_is_opaque() -> None:
    # T23 (I2 part): the sealed test cannot be read as data.
    _, sealed = split_sealed_test(frame(np.arange(40.0) + 0.125), 10)
    assert sealed is not None
    for read in (np.asarray, list, pd.Series, lambda s: s[0], lambda s: next(iter(s))):
        with pytest.raises(SealedTestAccessError):
            read(sealed)
    assert ".125" not in repr(sealed)  # metadata only, never values
    assert len(sealed) == 10


def test_sealed_test_is_refused_where_history_is_expected() -> None:
    _, sealed = split_sealed_test(frame(np.arange(40.0)), 10)
    with pytest.raises(SealedTestAccessError):
        TemporalSplitter(horizon=2).plan(sealed)
    for container in (sealed, [sealed], {"y": sealed}, {sealed: 1}, (1, sealed)):
        with pytest.raises(SealedTestAccessError):
            ensure_not_sealed(container)
    ensure_not_sealed(1, "a", [1, 2], {"k": "v"})


def test_sealed_type_is_distinct_from_series_frame() -> None:
    _, sealed = split_sealed_test(frame(np.arange(40.0)), 10)
    assert isinstance(sealed, SealedTest)
    assert not isinstance(sealed, SeriesFrame)


def test_hand_built_plan_with_overlapping_layers_is_rejected() -> None:
    plan = TemporalSplitter(horizon=5, n_search_folds=1, n_select_folds=1).plan(
        frame(np.arange(40.0))
    )
    late_search = dataclasses.replace(
        plan.search_origins[0],
        origin_index=plan.select_origins[0].origin_index - 2,
        origin_time=plan.history.timestamps[plan.select_origins[0].origin_index - 2],
    )
    with pytest.raises(TemporalLeakageError, match="invade a seleção"):
        TemporalPlan(plan.splitter, plan.history, (late_search,), plan.select_origins)


def test_hand_built_plan_with_targets_past_development_is_rejected() -> None:
    plan = TemporalSplitter(horizon=5, n_search_folds=1, n_select_folds=1).plan(
        frame(np.arange(40.0))
    )
    too_late = dataclasses.replace(
        plan.select_origins[0],
        origin_index=plan.dev_end - 1,
        origin_time=plan.history.timestamps[plan.dev_end - 1],
    )
    with pytest.raises(TemporalLeakageError, match="fim do desenvolvimento"):
        TemporalPlan(plan.splitter, plan.history, plan.search_origins, (too_late,))


def test_context_and_fold_transform_ignore_values_after_the_origin() -> None:
    # T05 groundwork and UX03 (I2 part): fold-level objects are blind to the future.
    values = np.cumsum(np.ones(60))
    splitter = TemporalSplitter(horizon=5, n_search_folds=2, n_select_folds=2)
    plan = splitter.plan(frame(values))
    for origin in plan:
        future_changed = values.copy()
        future_changed[origin.origin_index + 1 :] *= -50.0
        other = splitter.plan(frame(future_changed))
        twin = next(o for o in other if o.fold_id == origin.fold_id)

        assert np.array_equal(plan.context(origin).observed, other.context(twin).observed)
        fitted_a, _ = DifferenceTransform(1).fit_transform(plan.train_values(origin))
        fitted_b, _ = DifferenceTransform(1).fit_transform(other.train_values(twin))
        assert fitted_a == fitted_b
        assert not np.array_equal(plan.target_values(origin), other.target_values(twin))


def _corruptions(plan: TemporalPlan) -> dict[str, tuple[tuple[object, ...], type[Exception]]]:
    search, select = plan.search_origins, plan.select_origins
    first = search[0]
    return {
        "no search origins": (((), select), ConfigurationError),
        "out of order": ((tuple(reversed(search)), select), ConfigurationError),
        "wrong fold label": (
            ((dataclasses.replace(first, fold=5), *search[1:]), select),
            ConfigurationError,
        ),
        "other series": (
            ((dataclasses.replace(first, series_id="x"), *search[1:]), select),
            ConfigurationError,
        ),
        "inconsistent time": (
            (
                (dataclasses.replace(first, origin_time=pd.Timestamp("1999-01-01")), *search[1:]),
                select,
            ),
            ConfigurationError,
        ),
        "train after origin": (
            ((dataclasses.replace(first, train_start=first.origin_index + 1), *search[1:]), select),
            TemporalLeakageError,
        ),
    }


@pytest.mark.parametrize(
    "case",
    [
        "no search origins",
        "out of order",
        "wrong fold label",
        "other series",
        "inconsistent time",
        "train after origin",
    ],
)
def test_hand_built_plan_with_broken_origins_is_rejected(case: str) -> None:
    plan = TemporalSplitter(horizon=4, n_search_folds=2, n_select_folds=2).plan(
        frame(np.arange(60.0))
    )
    (search, select), error = _corruptions(plan)[case]
    with pytest.raises(error):
        TemporalPlan(plan.splitter, plan.history, search, select)


def test_unseal_is_reserved_for_external_evaluation() -> None:
    full = frame(np.arange(40.0))
    _, sealed = split_sealed_test(full, 10)
    assert sealed is not None
    revealed = sealed._unseal()
    assert revealed.fingerprint == full.tail(10).fingerprint == sealed.fingerprint
