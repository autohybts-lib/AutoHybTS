"""The §30.5 ablation script runs end to end on a tiny synthetic case."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "benchmarks" / "ablation.py"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning"),
]


def test_ablation_reports_every_procedure() -> None:
    spec = importlib.util.spec_from_file_location("ablation", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ablation"] = module  # dataclasses look the module up
    spec.loader.exec_module(module)
    case = next(module.cases("synthetic", 1))
    rows = module.run_case(case, seed=0, budget="smoke")
    import pandas as pd

    table = pd.DataFrame(rows)
    assert set(table["procedure"]) == set(module.PROCEDURES)
    report = module.report(table)
    assert list(report.index) == list(module.PROCEDURES)
    assert {"median_gain_vs_base", "mean_gap", "mean_rank", "candidates"} <= set(report.columns)
    # The external test is read once per candidate, whatever the procedure.
    assert table.groupby("champion")["test_score"].nunique().max() == 1
