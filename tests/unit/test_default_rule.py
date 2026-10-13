"""The calibrated default rule (ADR-0008): ``fold="auto"`` and the 5 % gain threshold."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from autohybts import AutoHybridExperiment
from autohybts.exceptions import ConfigurationError, InsufficientHistoryError
from autohybts.experiment import AUTO_FOLD
from autohybts.selection.policy import SelectionPolicy


def series(n: int, freq: str = "D") -> pd.Series:
    values = 100 + np.random.default_rng(0).normal(size=n).cumsum()
    return pd.Series(values, index=pd.date_range("2000-01-01", periods=n, freq=freq))


def test_defaults_are_the_calibrated_rule() -> None:
    assert SelectionPolicy().median_gain_threshold == 0.05
    assert SelectionPolicy().win_rate_threshold == 0.60
    assert AutoHybridExperiment().fold == "auto"
    assert AUTO_FOLD == 7


def test_auto_uses_seven_origins_when_they_fit() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning when the calibrated number fits
        exp = AutoHybridExperiment(fh=14).fit(series(730))
    assert exp.fold_ == 7
    assert len(exp.plan_.select_origins) == 7


def test_auto_falls_back_with_a_warning_on_short_series() -> None:
    with pytest.warns(UserWarning, match="calibração da regra de aceite"):
        exp = AutoHybridExperiment(fh=12).fit(series(144, "MS"))
    assert exp.fold_ == 4
    assert len(exp.plan_.select_origins) == 4


def test_explicit_fold_is_used_as_given() -> None:
    exp = AutoHybridExperiment(fh=12, fold=2).fit(series(144, "MS"))
    assert exp.fold_ == 2
    with pytest.raises(ConfigurationError, match="fold"):
        AutoHybridExperiment(fh=12, fold=0).fit(series(144, "MS"))
    with pytest.raises(ConfigurationError, match="fold"):
        AutoHybridExperiment(fh=12, fold="many").fit(series(144, "MS"))


def test_auto_still_fails_when_nothing_fits() -> None:
    with pytest.raises(InsufficientHistoryError):
        AutoHybridExperiment(fh=14).fit(series(30))


def test_search_grids_reach_the_search_and_the_snapshot() -> None:
    exp = AutoHybridExperiment(
        fh=7,
        fold=2,
        budget="smoke",
        base_models=["knn", "naive"],
        base_params={"knn": [{"lags": 7}, {"lags": 14}]},
        correctors=["linear"],
        corrector_params={"linear": {"scale": "columns"}},
    ).fit(series(300))
    result = exp.compare_models()
    labels = set(result.search.ledger_frame()["label"])
    assert {"knn(lags=7)", "knn(lags=14)"} <= labels
    snapshot = result.snapshot["experiment"]
    assert snapshot["base_params"] == {"knn": [{"lags": 7}, {"lags": 14}]}
    assert snapshot["corrector_params"] == {"linear": [{"scale": "columns"}]}
    with pytest.raises(ConfigurationError, match="grade"):
        AutoHybridExperiment(fh=7, fold=2, base_params={"knn": [3]}).fit(
            series(300)
        ).compare_models()
