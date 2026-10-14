"""SQLite experiment store (ADR-0010)."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autohybts import AutoHybridExperiment, ExperimentStore
from autohybts.exceptions import ArtifactCompatibilityError, ConfigurationError
from autohybts.tracking import STORE_SCHEMA
from autohybts.tracking.store import TABLES, _value

SETTINGS = {
    "fh": 3,
    "fold": 2,
    "budget": "smoke",
    "base_models": ["naive", "linear"],
    "correctors": ["linear"],
}


def series(n: int = 90, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = 50 + 5 * np.sin(np.arange(n) / 3) + rng.normal(0, 1, n)
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=n, freq="D"))


@pytest.fixture(scope="module")
def finished() -> tuple[AutoHybridExperiment, object]:
    exp = AutoHybridExperiment(**SETTINGS).fit(series())  # type: ignore[arg-type]
    model = exp.finalize_model(exp.compare_models().best)
    exp.evaluate_external(model)
    return exp, model


def counts(store: ExperimentStore) -> dict[str, int]:
    return {t: int(store.query(f"SELECT COUNT(*) AS n FROM {t}").n[0]) for t in TABLES}


def test_log_writes_every_table_of_a_compared_experiment(tmp_path: Path, finished) -> None:
    exp, model = finished
    store = ExperimentStore(tmp_path / "runs.db")
    run_id = store.log(exp, name="demo", tags={"owner": "test"}, models=[model])
    found = counts(store)
    assert found["runs"] == 1
    assert found["series"] == 1
    assert found["splits"] == len(exp.splits)
    assert found["ledger"] == len(exp.compare_.search.ledger_frame())
    assert found["decisions"] == len(exp.compare_.decisions)
    assert found["models"] == 1
    assert found["external"] == len(exp.evaluator_.metrics)
    assert found["trials"] > 0
    assert found["trial_metrics"] > 0
    assert found["events"] > 0
    run = store.runs().iloc[0]
    assert run.run_id == run_id
    assert run.champion_id == exp.compare_.best.candidate_id
    assert run.ranking_metric == exp.evaluator_.metrics[0]
    assert run.state == "SELECTED"
    assert store.query("SELECT value FROM meta WHERE key = 'schema'").value[0] == STORE_SCHEMA


def test_layer_metrics_match_the_experiment_tables(tmp_path: Path, finished) -> None:
    exp, _ = finished
    store = ExperimentStore(tmp_path / "runs.db")
    store.log(exp)
    stored = store.query(
        "SELECT candidate_id, layer, value FROM metrics WHERE metric = ? ORDER BY 1, 2",
        ("MASE",),
    )
    expected = exp.get_metrics()[["candidate_id", "layer", "MASE"]].sort_values(
        ["candidate_id", "layer"]
    )
    assert stored.candidate_id.tolist() == expected.candidate_id.tolist()
    assert np.allclose(stored.value, expected.MASE)


def test_logging_again_replaces_the_run_and_a_new_name_adds_one(tmp_path: Path, finished) -> None:
    exp, model = finished
    store = ExperimentStore(tmp_path / "runs.db")
    first = store.log(exp, name="a", models=[model])
    before = counts(store)
    assert store.log(exp, name="a", models=[model]) == first
    assert counts(store) == before
    assert store.log(exp, name="b") != first
    assert len(store.runs()) == 2
    store.delete_run(first)
    assert store.runs().name.tolist() == ["b"]


def test_series_values_are_written_only_on_request(tmp_path: Path, finished) -> None:
    exp, model = finished
    store = ExperimentStore(tmp_path / "runs.db")
    store.log(exp, models=[model])
    assert counts(store)["predictions"] == 0
    store.log(exp, models=[model], values=True)
    stored = store.query("SELECT y_true, y_pred FROM predictions ORDER BY timestamp")
    assert len(stored) == exp.fh
    assert stored.y_true.notna().all()


def test_logging_never_changes_the_experiment(tmp_path: Path, finished) -> None:
    exp, _ = finished
    before = exp.get_metrics()
    ExperimentStore(tmp_path / "runs.db").log(exp)
    pd.testing.assert_frame_equal(exp.get_metrics(), before)


def test_undefined_values_become_null() -> None:
    assert _value(float("nan")) is None
    assert _value(np.float64("nan")) is None
    assert _value(pd.NaT) is None
    assert _value(None) is None
    assert _value(np.int64(3)) == 3
    assert _value(np.bool_(True)) == 1
    assert _value(("a", "b")) == '["a", "b"]'
    assert _value(pd.Timestamp("2024-01-01")) == "2024-01-01T00:00:00"


def test_queries_are_read_only(tmp_path: Path, finished) -> None:
    store = ExperimentStore(tmp_path / "runs.db")
    with pytest.raises(ConfigurationError, match="ainda não existe"):
        store.runs()
    store.log(finished[0])
    with pytest.raises(pd.errors.DatabaseError, match="readonly"):
        store.query("DELETE FROM runs")
    assert len(store.runs()) == 1


def test_foreign_files_and_schemas_are_refused(tmp_path: Path) -> None:
    other = tmp_path / "other.db"
    with sqlite3.connect(other) as con:
        con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        con.execute("INSERT INTO meta VALUES ('schema', 'someone-else/9')")
    con.close()
    with pytest.raises(ArtifactCompatibilityError, match="someone-else/9"):
        ExperimentStore(other).runs()
    text = tmp_path / "notes.db"
    text.write_text("not a database " * 100, encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match="não é um banco"):
        ExperimentStore(text).runs()


def test_log_needs_a_fitted_experiment_and_a_name(tmp_path: Path, finished) -> None:
    store = ExperimentStore(tmp_path / "runs.db")
    with pytest.raises(ConfigurationError, match="fit"):
        store.log(AutoHybridExperiment(**SETTINGS))  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="name"):
        store.log(finished[0], name=" ")
    with pytest.raises(ConfigurationError, match="finalizados"):
        store.log(finished[0], models=["model"])  # type: ignore[list-item]


def test_a_run_without_compare_models_keeps_its_evaluations(tmp_path: Path) -> None:
    exp = AutoHybridExperiment(**SETTINGS).fit(series())  # type: ignore[arg-type]
    exp.create_model("naive")
    store = ExperimentStore(tmp_path / "runs.db")
    store.log(exp, name="manual")
    run = store.runs().iloc[0]
    assert run.state == "VALIDATED"
    assert run.champion_id is None
    layers = store.query("SELECT layer FROM evaluations ORDER BY layer").layer.tolist()
    assert layers == ["search", "select"]
    assert counts(store)["decisions"] == 0


def test_panel_external_metrics_are_stored_per_series_and_as_a_mean(tmp_path: Path) -> None:
    exp = AutoHybridExperiment(**SETTINGS).fit({"a": series(), "b": series(seed=1)})  # type: ignore[arg-type]
    model = exp.finalize_model(exp.compare_models().best)
    exp.evaluate_external(model)
    store = ExperimentStore(tmp_path / "runs.db")
    store.log(exp, models=[model], values=True)
    rows = store.query("SELECT series_id, value FROM external WHERE metric = 'MASE'")
    assert sorted(rows.series_id.fillna("mean")) == ["a", "b", "mean"]
    assert set(store.query("SELECT DISTINCT series_id FROM splits").series_id) == {"a", "b"}
    assert counts(store)["predictions"] == 2 * exp.fh


def test_importing_the_package_does_not_load_sqlite() -> None:
    code = "import sys, autohybts; print('sqlite3' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
