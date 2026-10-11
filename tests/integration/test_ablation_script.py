"""The §30.5 ablation runs end to end on a tiny synthetic case."""

from __future__ import annotations

import json

import pytest

from autohybts.benchmarks import (
    ABLATION_SCHEMA,
    PROCEDURES,
    ablation_report,
    run_ablation,
    synthetic_cases,
)
from autohybts.benchmarks.ablation import main

pytestmark = [
    pytest.mark.slow,
    pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning"),
]


def test_ablation_reports_every_procedure_and_its_full_cost() -> None:
    table = run_ablation(synthetic_cases(1), seeds=(0,))
    assert set(table["procedure"]) == set(PROCEDURES)
    assert set(table["schema"]) == {ABLATION_SCHEMA}
    report = ablation_report(table)
    assert list(report.index) == list(PROCEDURES)
    assert {"median_gain_vs_base", "mean_gap", "mean_rank", "candidates", "units"} <= set(
        report.columns
    )
    # The external test is read once per candidate, whatever the procedure.
    assert table.groupby("champion")["test_score"].nunique().max() == 1
    rows = table.set_index("procedure")
    # Three procedures share the full search; only guarded_nested adds the nested CV.
    shared = rows.loc[["lowest_error", "incremental", "guarded_nested"], "search_candidates"]
    assert shared.nunique() == 1
    assert rows.loc["incremental_simple", "search"] == "simple"
    nested = rows.loc["guarded_nested"]
    assert nested["nested_candidates"] > 0
    assert nested["candidates"] == nested["search_candidates"] + nested["nested_candidates"]
    assert nested["seconds"] == pytest.approx(
        rows.loc["incremental", "seconds"] + nested["seconds_nested"]
    )
    assert (rows.loc[["lowest_error", "incremental"], "nested_candidates"] == 0).all()
    assert (table["search_candidates"] <= table["max_trials_total"]).all()
    assert (table["stop_reason"] != "max_wall_time").all()  # presets have no wall clock
    assert table[["naive_test", "seasonal_naive_test"]].notna().all().all()
    # Proposals cut by each limit add up; external evaluation is timed apart from the cost.
    for _, row in table.iterrows():
        parts = [int(p.split(":")[1]) for p in str(row["pruned_by"]).split(";") if ":" in p]
        assert sum(parts) == row["pruned_candidates"]
    assert (table["seconds_external"] > 0).all()
    # The procedure's cost reconciles with its phases (refit and external read excluded).
    phases = table[["seconds_search", "seconds_selection", "seconds_nested"]].sum(axis=1)
    assert phases.tolist() == pytest.approx(table["seconds"].tolist())


def test_command_line_writes_the_table_and_the_protocol(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "ablation.csv"
    main(["--dataset", "synthetic", "--series", "1", "--out", str(out)])
    protocol = json.loads((tmp_path / "ablation.csv.protocol.json").read_text(encoding="utf-8"))
    assert protocol["schema"] == ABLATION_SCHEMA
    assert protocol["series"][0]["series_id"] == "syn0-phi0.0"
    assert protocol["budget"]["preset"] == "smoke"
    assert out.read_text(encoding="utf-8").count("\n") == len(PROCEDURES) + 1
    assert "lowest_error" in capsys.readouterr().out
