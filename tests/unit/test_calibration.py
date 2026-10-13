"""Calibration of the acceptance rule (M02): report on a hand-built table, tiny real run."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from autohybts.benchmarks.calibration import (
    CALIBRATION_SCHEMA,
    calibration_protocol,
    calibration_report,
    control_cases,
    main,
    run_calibration,
)
from autohybts.exceptions import ConfigurationError

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def table() -> pd.DataFrame:
    rows = []
    for i in range(10):
        rows.append(
            {
                "series_id": f"wn{i}",
                "control": "white_noise",
                "fold": 2,
                "policy": "rule",
                "select_units": 2,
                "champion_is_hybrid": i < 3,  # 3 false acceptances in 10 chains
                "hybrid_comparisons": 2,
                "hybrid_accepts": 1 if i < 3 else 0,
            }
        )
        rows.append(
            {
                "series_id": f"ar{i}",
                "control": "ar_residual",
                "fold": 2,
                "policy": "rule",
                "select_units": 2,
                "champion_is_hybrid": i < 8,  # 2 rejected useful gains in 10
                "hybrid_comparisons": 1,
                "hybrid_accepts": 1 if i < 8 else 0,
            }
        )
    return pd.DataFrame(rows)


def test_report_rates_and_intervals_resample_series() -> None:
    report = calibration_report(table(), n_boot=500)
    negative = report.loc[("white_noise", 2, "rule")]
    assert negative["kind"] == "negative"
    assert negative["hybrid_champion_rate"] == pytest.approx(0.3)
    assert negative["comparison_accept_rate"] == pytest.approx(3 / 20)
    assert negative["rate_ci_low"] <= 0.3 <= negative["rate_ci_high"]
    assert pd.isna(negative["rejected_gain_rate"])
    positive = report.loc[("ar_residual", 2, "rule")]
    assert positive["rejected_gain_rate"] == pytest.approx(0.2)
    assert positive["min_one_sided_p"] == pytest.approx(0.25)
    assert positive["n_series"] == 10


def test_splits_use_disjoint_seeds() -> None:
    calibration = {c.series_id for c in control_cases(3, "calibration", n=60)}
    confirmation = {c.series_id for c in control_cases(3, "confirmation", n=60)}
    assert len(calibration) == len(confirmation) == 6
    assert not calibration & confirmation
    with pytest.raises(ConfigurationError, match="split"):
        list(control_cases(1, "test"))


def test_tiny_run_applies_every_policy_to_the_same_finalists(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    result = run_calibration(control_cases(1), folds=(2,))
    assert set(result["schema"]) == {CALIBRATION_SCHEMA}
    assert len(result) == 4  # 2 controls x 1 fold x 2 policies
    # With 2 select units the Wilcoxon predicate can never pass: no hybrid accepted.
    wilcoxon = result[result["policy"] == "rule+wilcoxon_0.05"]
    assert (wilcoxon["hybrid_accepts"] == 0).all()
    assert (result["select_units"] == 2).all()
    protocol = calibration_protocol(1, "calibration", (2,))
    assert protocol["seed_offset"] == 10_000
    json.dumps(protocol, default=str)
    out = tmp_path / "cal.csv"
    main(["--series", "1", "--folds", "2", "--out", str(out)])
    assert (tmp_path / "cal.csv.protocol.json").is_file()
    assert (tmp_path / "cal.csv.report.csv").is_file()
    assert "white_noise" in capsys.readouterr().out


def test_calibration_never_reads_the_sealed_test(monkeypatch: pytest.MonkeyPatch) -> None:
    from autohybts.data.splitter import SealedTest

    def refuse(self: SealedTest) -> object:
        raise AssertionError("a calibração leu o teste selado")

    monkeypatch.setattr(SealedTest, "_unseal", refuse)
    table = run_calibration(control_cases(1), folds=(2,), policies={"rule": {}})
    assert len(table) == 2
    protocol = calibration_protocol(1, "calibration", (2,))
    assert protocol["base_policy"]["median_gain_threshold"] == 0.01
    assert protocol["base_policy"]["variability"]["version"] == 2
    assert protocol["metric"] == "MASE"


def test_calibration_can_include_the_composed_families() -> None:
    table = run_calibration(
        control_cases(1), folds=(2,), policies={"rule": {}}, include=("standalone", "ensemble")
    )
    assert set(table["control"]) == {"white_noise", "ar_residual"}
    assert "champion_strategy" in table.columns
    protocol = calibration_protocol(1, "calibration", (2,), include=("standalone", "ensemble"))
    assert protocol["include"] == ["standalone", "ensemble"]
