"""Ablation report and protocol on a hand-built table (the full run is a slow test)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from autohybts.benchmarks import (
    ABLATION_SCHEMA,
    PROCEDURES,
    ablation_protocol,
    ablation_report,
    ablation_uncertainty,
    synthetic_cases,
)
from autohybts.exceptions import ConfigurationError


def table() -> pd.DataFrame:
    rows = []
    gains = {
        "lowest_error": [0.1, -0.2, 0.0, 0.3],
        "incremental": [0.0, 0.0, 0.0, 0.1],
        "incremental_simple": [0.0, -0.1, 0.0, 0.0],
        "guarded_nested": [0.0, 0.0, 0.0, 0.0],
    }
    for procedure, values in gains.items():
        for i, gain in enumerate(values):
            rows.append(
                {
                    "series_id": f"s{i // 2}",
                    "seed": i % 2,
                    "procedure": procedure,
                    "is_hybrid": gain != 0,
                    "complexity": 2 if gain != 0 else 1,
                    "test_score": 1.0 - gain,
                    "gain_vs_base": gain,
                    "gap": 0.1,
                    "coverage": 1.0,
                    "seconds": 10.0,
                    "candidates": 10 + (5 if procedure == "guarded_nested" else 0),
                    "units": 40,
                }
            )
    return pd.DataFrame(rows)


def test_report_counts_wins_ties_and_losses_against_the_base() -> None:
    report = ablation_report(table())
    assert list(report.index) == list(PROCEDURES)
    lowest = report.loc["lowest_error"]
    assert lowest["beats_base_share"] == pytest.approx(0.5)
    assert lowest["ties_base_share"] == pytest.approx(0.25)
    assert lowest["worse_than_base_share"] == pytest.approx(0.25)
    shares = report[["beats_base_share", "ties_base_share", "worse_than_base_share"]]
    assert shares.sum(axis=1).tolist() == pytest.approx([1.0] * len(PROCEDURES))
    assert report.loc["guarded_nested", "candidates"] == 15
    assert report.loc["incremental", "mean_complexity"] == pytest.approx(1.25)


def test_protocol_records_the_resolved_settings() -> None:
    cases = list(synthetic_cases(2, n=120))
    protocol = ablation_protocol(cases, seeds=(0, 1), budget="small", outer_folds=3)
    assert protocol["schema"] == ABLATION_SCHEMA
    assert protocol["seeds"] == [0, 1]
    assert [s["series_id"] for s in protocol["series"]] == [c.series_id for c in cases]
    assert protocol["budget"]["guard"]["max_trials_total"] == 50
    assert "max_wall_time_s" not in protocol["budget"]["guard"]
    assert protocol["environment"]["autohybts"]
    assert protocol["metrics"][0] == protocol["ranking_metric"] == "MASE"
    assert protocol["selection_policy"]["win_rate_threshold"] == 0.6
    assert "cache" in protocol
    json.dumps(protocol, default=str)  # serializable
    with pytest.raises(ConfigurationError, match="Preset"):
        ablation_protocol(cases, seeds=(0,), budget="huge")  # type: ignore[arg-type]


def test_report_by_stratum_and_series_level_uncertainty() -> None:
    frame = table()
    frame["stratum"] = ["phi=0.0" if s == "s0" else "phi=0.8" for s in frame["series_id"]]
    stratified = ablation_report(frame, by="stratum")
    assert stratified.index.names == ["stratum", "procedure"]
    assert set(stratified.index.get_level_values("stratum")) == {"phi=0.0", "phi=0.8"}
    with pytest.raises(ConfigurationError, match="ausente"):
        ablation_report(frame, by="nothing")

    uncertainty = ablation_uncertainty(frame, n_boot=500)
    lowest = uncertainty.loc["lowest_error"]
    # Seeds are summarized per series first: s0 -> median(0.1, -0.2) = -0.05,
    # s1 -> median(0.0, 0.3) = 0.15; across series: 0.05.
    assert lowest["n_series"] == 2
    assert lowest["median_gain_vs_base"] == pytest.approx(0.05)
    assert (lowest["better_share"], lowest["worse_share"]) == (0.5, 0.5)
    assert lowest["ci_low"] <= lowest["median_gain_vs_base"] <= lowest["ci_high"]
    assert (
        uncertainty.loc["guarded_nested", "ci_low"]
        == 0.0
        == uncertainty.loc["guarded_nested", "ci_high"]
    )


def test_protocol_uses_configured_metrics_policy_and_budget() -> None:
    from autohybts import AutoHybridExperiment
    from autohybts.benchmarks.ablation import DEFAULT_SPACE

    cases = list(synthetic_cases(1, n=120))
    space = {
        "metrics": ("RMSE", "MAE"),
        "profile": "quickstart",
        "significance_alpha": 0.1,
        "max_trials": 3,
        "max_wall_time_s": 60.0,
    }
    protocol = ablation_protocol(cases, (0,), space=space)
    experiment = AutoHybridExperiment(**{**DEFAULT_SPACE, **space}).fit(cases[0].values)
    assert protocol["metrics"] == ["RMSE", "MAE"]
    assert protocol["ranking_metric"] == "RMSE"
    assert protocol["selection_policy"] == experiment.policy_.describe()
    assert protocol["selection_policy"]["variability"] is None
    assert protocol["selection_policy"]["significance"]["alpha"] == 0.1
    assert protocol["budget"]["guard"]["max_trials_total"] == 3
    assert protocol["budget"]["guard"]["max_wall_time_s"] == 60.0
    json.dumps(protocol)


@pytest.mark.parametrize("metrics", [(), ("unknown",), ("MAE", "MAE")])
def test_protocol_refuses_invalid_metrics(metrics: tuple[str, ...]) -> None:
    with pytest.raises(ConfigurationError):
        ablation_protocol([], (0,), space={"metrics": metrics})


def test_report_excludes_nonfinite_gains_and_records_the_denominator() -> None:
    frame = table()
    selected = frame["procedure"] == "lowest_error"
    frame.loc[selected, "gain_vs_base"] = [0.1, np.nan, np.inf, -0.2]
    original = frame.copy(deep=True)
    lowest = ablation_report(frame).loc["lowest_error"]
    assert lowest["n_rows"] == 4
    assert lowest["n_paired"] == 2
    assert lowest["n_invalid_gain"] == 2
    assert lowest["paired_coverage"] == 0.5
    assert lowest["median_gain_vs_base"] == pytest.approx(-0.05)
    assert lowest["beats_base_share"] == 0.5
    assert lowest["ties_base_share"] == 0.0
    assert lowest["worse_than_base_share"] == 0.5
    pd.testing.assert_frame_equal(frame, original)


def test_report_without_paired_gains_keeps_statistics_undefined() -> None:
    frame = table()
    frame["gain_vs_base"] = np.nan
    report = ablation_report(frame)
    assert (report["n_paired"] == 0).all()
    assert (report["paired_coverage"] == 0).all()
    assert (
        report[
            ["median_gain_vs_base", "beats_base_share", "ties_base_share", "worse_than_base_share"]
        ]
        .isna()
        .all()
        .all()
    )


def test_bootstrap_aggregates_finite_seeds_and_counts_unpaired_series() -> None:
    frame = table()
    frame = frame.loc[frame["procedure"] == "lowest_error"].copy()
    frame["gain_vs_base"] = [0.1, np.nan, np.inf, -np.inf]
    original = frame.copy(deep=True)
    result = ablation_uncertainty(frame, n_boot=100).loc["lowest_error"]
    assert result["n_series"] == 2
    assert result["n_paired_series"] == 1
    assert result["n_invalid_series"] == 1
    assert result["paired_series_coverage"] == 0.5
    assert result["median_gain_vs_base"] == pytest.approx(0.1)
    assert result["ci_low"] == pytest.approx(0.1)
    assert result["ci_high"] == pytest.approx(0.1)
    assert result["better_share"] == 1.0
    assert result["worse_share"] == 0.0
    pd.testing.assert_frame_equal(frame, original)


def test_bootstrap_without_valid_series_returns_undefined_estimates() -> None:
    frame = table()
    frame["gain_vs_base"] = np.nan
    result = ablation_uncertainty(frame, n_boot=100)
    assert (result["n_series"] == 2).all()
    assert (result["n_paired_series"] == 0).all()
    assert (result["paired_series_coverage"] == 0).all()
    assert (
        result[["median_gain_vs_base", "ci_low", "ci_high", "better_share", "worse_share"]]
        .isna()
        .all()
        .all()
    )


@pytest.mark.parametrize("n_boot", [0, -1, True, 1.5])
def test_bootstrap_refuses_invalid_sample_counts(n_boot: object) -> None:
    with pytest.raises(ConfigurationError, match="n_boot"):
        ablation_uncertainty(table(), n_boot=n_boot)  # type: ignore[arg-type]


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, np.nan])
def test_bootstrap_refuses_invalid_confidence_levels(level: float) -> None:
    with pytest.raises(ConfigurationError, match="level"):
        ablation_uncertainty(table(), level=level)
