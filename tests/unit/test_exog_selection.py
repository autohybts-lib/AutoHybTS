"""Automatic choice of exogenous features and with × without comparison (ADR-0011)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autohybts import AutoHybridExperiment
from autohybts.data.exog_selection import (
    BASE_METHODS,
    SELECTION_METHODS,
    ExogSelection,
    select_exog,
    vote_selection,
)
from autohybts.domain.specs import ExogFeature
from autohybts.exceptions import ConfigurationError, InsufficientHistoryError

KINDS = {"x": "past", "feriado": "future", "z0": "past", "z1": "past", "z2": "past"}
STEP = 5


def series(seed: int, relevant: bool = True, n: int = 300) -> tuple[np.ndarray, pd.DataFrame]:
    """AR(1) noise plus 3·x seven steps earlier and a holiday effect; z columns are noise."""

    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    holiday = (np.arange(n) % 7 == 0).astype(float)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.5 * noise[i - 1] + rng.normal(0, 1)
    y = 20 + noise + ((3 * np.roll(x, 7) + 2 * holiday) if relevant else 0)
    columns = {"x": x, "feriado": holiday, **{f"z{k}": rng.normal(0, 1, n) for k in range(3)}}
    return y, pd.DataFrame(columns)


@pytest.mark.parametrize("method", ["ccf", "granger", "lasso", "stepwise_bic", "vote"])
def test_methods_find_the_lagged_driver(method: str) -> None:
    y, X = series(0)
    chosen = select_exog(y, X, KINDS, STEP, method).features
    assert ExogFeature("x", 7) in chosen


@pytest.mark.parametrize("method", ["granger", "lasso", "stepwise_bic", "vote"])
def test_methods_find_the_known_calendar_effect(method: str) -> None:
    y, X = series(0)
    assert ExogFeature("feriado", 0) in select_exog(y, X, KINDS, STEP, method).features


def test_conservative_methods_rarely_pick_noise() -> None:
    picked = {m: 0 for m in ("ccf", "granger", "lasso", "vote")}
    for seed in range(5):
        y, X = series(100 + seed, relevant=False)
        for method in picked:
            picked[method] += len(select_exog(y, X, KINDS, STEP, method).features)
    assert all(count <= 1 for count in picked.values()), picked


def test_mutual_information_controls_false_picks_across_lags() -> None:
    picked = 0
    for seed in range(3):
        y, X = series(100 + seed, relevant=False)
        picked += len(select_exog(y, X, KINDS, STEP, "mutual_info").features)
    assert picked <= 1
    y, X = series(0)
    assert ExogFeature("x", 7) in select_exog(y, X, KINDS, STEP, "mutual_info").features


def test_candidate_lags_respect_the_kinds() -> None:
    y, X = series(0)
    table = select_exog(y, X, KINDS, STEP, "granger", max_lag=9).table
    assert set(table.loc[table.column == "feriado", "lag"]) == {0}
    assert set(table.loc[table.column == "x", "lag"]) == set(range(STEP, 10))
    assert set(table.columns) == {
        "method",
        "column",
        "lag",
        "feature",
        "score",
        "p_value",
        "selected",
    }


def test_vote_keeps_majority_columns_at_their_most_chosen_lag() -> None:
    def part(*features: ExogFeature) -> ExogSelection:
        return ExogSelection("m", tuple(features), pd.DataFrame())

    votes = vote_selection(
        [
            part(ExogFeature("a", 7), ExogFeature("b", 5)),
            part(ExogFeature("a", 8)),
            part(ExogFeature("a", 8), ExogFeature("c", 5)),
            part(),
            part(ExogFeature("b", 5)),
        ]
    )
    assert votes.features == (ExogFeature("a", 8),)
    assert len(BASE_METHODS) == 5


def test_selection_validates_its_input() -> None:
    y, X = series(0)
    with pytest.raises(ConfigurationError, match="desconhecido"):
        select_exog(y, X, KINDS, STEP, "pca")
    with pytest.raises(ConfigurationError, match="uma linha"):
        select_exog(y[:-1], X, KINDS, STEP, "ccf")
    with pytest.raises(ConfigurationError, match="max_lag"):
        select_exog(y, X, KINDS, STEP, "ccf", max_lag=2)
    with pytest.raises(InsufficientHistoryError):
        select_exog(y[:40], X.iloc[:40], KINDS, STEP, "ccf")
    assert "vote" in SELECTION_METHODS


# ---------------------------------------------------------------- experiment
N, FH = 260, 5


def experiment_data() -> tuple[pd.Series, pd.DataFrame]:
    y, X = series(7, n=N + 20)
    index = pd.date_range("2024-01-01", periods=N + 20, freq="D")
    X.index = index
    return pd.Series(y[:N], index=index[:N]), X


SETTINGS = {
    "fh": FH,
    "fold": 2,
    "budget": "smoke",
    "base_models": ["naive", "linear"],
    "correctors": ["linear"],
}


def test_auto_selection_offers_one_set_per_method() -> None:
    y, X = experiment_data()
    exp = AutoHybridExperiment(**SETTINGS, exog_future=["feriado"], exog_selection="auto").fit(y, X)  # type: ignore[arg-type]
    report = exp.exog_report()
    assert set(report.method) == set(SELECTION_METHODS)
    assert len(exp.exog_sets_) >= 2
    assert exp.exog_features_ == exp.exog_sets_[0]
    result = exp.compare_models()
    assert result.snapshot["space"]["exog_sets"]


def test_selection_never_sees_v_select_or_later_data() -> None:
    y, X = experiment_data()
    settings = {**SETTINGS, "exog_future": ["feriado"], "exog_selection": ["granger", "lasso"]}
    reference = AutoHybridExperiment(**settings).fit(y, X)  # type: ignore[arg-type]
    first = min(o.origin_index for o in reference.plan_.select_origins)
    changed_y, changed_X = y.copy(), X.copy()
    changed_y.iloc[first + 1 :] += 100.0
    changed_X.iloc[first + 1 :, 0] += 100.0
    changed = AutoHybridExperiment(**settings).fit(changed_y, changed_X)  # type: ignore[arg-type]
    pd.testing.assert_frame_equal(changed.exog_report(), reference.exog_report())
    assert changed.exog_sets_ == reference.exog_sets_


def test_selection_options_are_validated() -> None:
    y, X = experiment_data()
    with pytest.raises(ConfigurationError, match="exog_selection"):
        AutoHybridExperiment(**SETTINGS, exog_selection="pca").fit(y, X)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="exigem X"):
        AutoHybridExperiment(**SETTINGS, exog_selection="auto").fit(y)  # type: ignore[arg-type]
    plain = AutoHybridExperiment(**SETTINGS).fit(y, X)  # type: ignore[arg-type]
    assert set(plain.exog_report().method) == {"all"}


def test_covariates_never_start_as_champion_and_compare_exog_reports_tests() -> None:
    y, X = experiment_data()
    exp = AutoHybridExperiment(**SETTINGS, exog_future=["feriado"]).fit(y, X)  # type: ignore[arg-type]
    result = exp.compare_models()
    decisions = result.decisions
    assert "exogenous" in set(decisions.family)
    first = decisions.iloc[0]
    assert "exog" not in str(first.champion_before)
    comparison = exp.compare_exog()
    assert [r.candidate.exog != () for r in comparison.results] == [False, True]
    tests = comparison.tests()
    assert {"wilcoxon_p", "dm_p", "wilcoxon_p_holm"} <= set(tests.columns)
    no_exog = AutoHybridExperiment(**SETTINGS).fit(y)  # type: ignore[arg-type]
    no_exog.compare_models()
    with pytest.raises(ConfigurationError, match="com e sem"):
        no_exog.compare_exog()
