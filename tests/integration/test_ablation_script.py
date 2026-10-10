"""The §30.5 ablation runs end to end on a tiny synthetic case."""

from __future__ import annotations

import pytest

from autohybts.benchmarks import PROCEDURES, ablation_report, run_ablation, synthetic_cases

pytestmark = [
    pytest.mark.slow,
    pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning"),
]


def test_ablation_reports_every_procedure() -> None:
    table = run_ablation(synthetic_cases(1), seeds=(0,))
    assert set(table["procedure"]) == set(PROCEDURES)
    report = ablation_report(table)
    assert list(report.index) == list(PROCEDURES)
    assert {"median_gain_vs_base", "mean_gap", "mean_rank", "candidates"} <= set(report.columns)
    # The external test is read once per candidate, whatever the procedure.
    assert table.groupby("champion")["test_score"].nunique().max() == 1
