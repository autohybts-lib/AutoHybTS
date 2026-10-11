"""Ablation report and protocol on a hand-built table (the full run is a slow test)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from autohybts.benchmarks import (
    ABLATION_SCHEMA,
    PROCEDURES,
    ablation_protocol,
    ablation_report,
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
    from autohybts.benchmarks import ablation_uncertainty

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
