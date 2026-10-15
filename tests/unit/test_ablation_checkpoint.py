"""Ablation interruption, identity, failure accounting and JSON configuration contracts."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autohybts.benchmarks import ablation
from autohybts.benchmarks.ablation import PROCEDURES, AblationCase, run_ablation
from autohybts.benchmarks.checkpoint import Checkpoint, series_fingerprint
from autohybts.exceptions import ConfigurationError, InsufficientHistoryError, TemporalLeakageError


def cases() -> list[AblationCase]:
    index = pd.date_range("2020-01-01", periods=40, freq="D")
    return [
        AblationCase(sid, pd.Series(np.arange(40, dtype=float), index=index), 2, 2, 1)
        for sid in ("a", "b")
    ]


def rows(case: AblationCase, seed: int) -> list[dict[str, object]]:
    return [
        {
            "series_id": case.series_id,
            "seed": seed,
            "procedure": procedure,
            "test_score": 1.0,
            "gain_vs_base": 0.0,
            "is_hybrid": False,
            "complexity": 1,
            "gap": 0.0,
            "coverage": 1.0,
            "seconds": 1.0,
            "candidates": 1,
            "units": 1,
        }
        for procedure in PROCEDURES
    ]


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    observed: list[tuple[str, int]] = []

    def successful(case, seed, *args, snapshot_callback=None):  # type: ignore[no-untyped-def]
        observed.append((case.series_id, seed))
        if snapshot_callback:
            snapshot_callback({"phase": "full", "snapshot": {"resolved_fold": 2}})
        return rows(case, seed)

    monkeypatch.setattr(ablation, "run_case", successful)
    monkeypatch.setattr(ablation, "source_fingerprint", lambda: "source-a")
    return observed


def test_resume_reuses_successful_cases_without_duplicates(tmp_path: Path, calls) -> None:  # type: ignore[no-untyped-def]
    checkpoint = tmp_path / "study.sqlite"
    original = run_ablation(cases(), seeds=(0, 1), checkpoint=checkpoint)
    resumed = run_ablation(cases(), seeds=(0, 1), checkpoint=checkpoint, resume=True)
    pd.testing.assert_frame_equal(original, resumed)
    assert calls == [("a", 0), ("a", 1), ("b", 0), ("b", 1)]
    assert resumed.attrs["coverage"]["n_planned_cases"] == 4
    assert resumed.attrs["coverage"]["n_planned_procedures"] == 16
    assert resumed.attrs["coverage"]["case_coverage"] == 1
    assert all(unit["snapshots"] for unit in resumed.attrs["cases"])
    assert all(len(unit["history"]) == 1 for unit in resumed.attrs["cases"])


def test_interrupted_case_resumes_and_keeps_completed_cases(
    tmp_path: Path, calls, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    checkpoint = tmp_path / "study.sqlite"
    original = ablation.run_case

    def interrupted(case, seed, *args, **kwargs):  # type: ignore[no-untyped-def]
        if case.series_id == "b":
            raise KeyboardInterrupt
        return original(case, seed, *args, **kwargs)

    monkeypatch.setattr(ablation, "run_case", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_ablation(cases(), checkpoint=checkpoint)
    monkeypatch.setattr(ablation, "run_case", original)
    result = run_ablation(cases(), checkpoint=checkpoint, resume=True)
    assert calls == [("a", 0), ("b", 0)]
    history = result.attrs["cases"][1]["history"]
    assert [attempt["status"] for attempt in history] == ["INTERRUPTED", "SUCCEEDED"]
    assert len(result) == 8


def test_failures_are_counted_and_only_retried_explicitly(
    tmp_path: Path, calls, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    checkpoint = tmp_path / "study.sqlite"
    original = ablation.run_case

    def failing(case, seed, *args, **kwargs):  # type: ignore[no-untyped-def]
        if case.series_id == "b":
            raise InsufficientHistoryError("short history")
        return original(case, seed, *args, **kwargs)

    monkeypatch.setattr(ablation, "run_case", failing)
    failed = run_ablation(cases(), checkpoint=checkpoint, on_error="record")
    assert failed.attrs["coverage"]["n_failed_cases"] == 1
    assert failed.attrs["coverage"]["case_coverage"] == 0.5
    assert len(failed) == 4
    monkeypatch.setattr(ablation, "run_case", original)
    kept = run_ablation(cases(), checkpoint=checkpoint, resume=True)
    assert len(kept) == 4
    assert calls == [("a", 0)]
    retried = run_ablation(cases(), checkpoint=checkpoint, resume=True, retry_failed=True)
    assert len(retried) == 8
    assert calls == [("a", 0), ("b", 0)]
    assert "InsufficientHistoryError" in retried.attrs["cases"][1]["history"][0]["reason"]


@pytest.mark.parametrize("change", ["data", "time", "seed", "space", "source", "configuration"])
def test_resume_refuses_a_changed_study(
    tmp_path: Path, calls, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:  # type: ignore[no-untyped-def]
    checkpoint = tmp_path / "study.sqlite"
    planned = cases()
    run_ablation(planned, checkpoint=checkpoint, configuration={"sampling_seed": 0})
    kwargs = {"configuration": {"sampling_seed": 0}}
    if change == "data":
        planned[0] = replace(planned[0], values=planned[0].values + 1)
    elif change == "time":
        y = planned[0].values.copy()
        y.index = y.index + pd.Timedelta(days=1)
        planned[0] = replace(planned[0], values=y)
    elif change == "seed":
        kwargs["seeds"] = (1,)  # type: ignore[assignment]
    elif change == "space":
        kwargs["space"] = {"metrics": ("RMSE",)}  # type: ignore[dict-item]
    elif change == "source":
        monkeypatch.setattr(ablation, "source_fingerprint", lambda: "source-b")
    else:
        kwargs["configuration"] = {"sampling_seed": 1}
    with pytest.raises(ConfigurationError, match="incompatível"):
        run_ablation(planned, checkpoint=checkpoint, resume=True, **kwargs)
    assert calls == [("a", 0), ("b", 0)]


def test_configuration_and_leakage_errors_are_never_suppressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for error in (ConfigurationError("bad configuration"), TemporalLeakageError("future")):

        def fatal(*args, current=error, **kwargs):  # type: ignore[no-untyped-def]
            raise current

        monkeypatch.setattr(ablation, "run_case", fatal)
        with pytest.raises(type(error)):
            run_ablation(
                cases(), checkpoint=tmp_path / f"{type(error).__name__}.sqlite", on_error="record"
            )


def test_checkpoint_has_one_writer_and_checks_payload_integrity(tmp_path: Path) -> None:
    path = tmp_path / "study.sqlite"
    with Checkpoint(path, "identity", {}, False) as first:
        first.write("case", {"status": "PENDING"})
        with (
            pytest.raises(ConfigurationError, match="em uso"),
            Checkpoint(path, "identity", {}, True),
        ):
            pytest.fail("second writer was admitted")
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE cases SET payload = '{}' ")
    with (
        Checkpoint(path, "identity", {}, True) as store,
        pytest.raises(ConfigurationError, match="corrompido"),
    ):
        store.read()


def test_running_record_from_a_crash_is_recovered(tmp_path: Path, calls) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "study.sqlite"
    run_ablation(cases(), checkpoint=path)
    with sqlite3.connect(path) as connection:
        protocol = json.loads(connection.execute("SELECT protocol FROM metadata").fetchone()[0])
    with Checkpoint(path, protocol["study_id"], protocol, True) as store:
        records = store.read()
        key = next(key for key, value in records.items() if value["series_id"] == "b")
        record = records[key]
        record.update(status="RUNNING", rows=[])
        record["history"][-1] = {"status": "RUNNING", "started_at": "2020-01-01T00:00:00Z"}
        store.write(key, record)
    result = run_ablation(cases(), checkpoint=path, resume=True)
    assert calls == [("a", 0), ("b", 0), ("b", 0)]
    assert result.attrs["cases"][1]["history"][0]["seconds"] is None


def test_all_failed_study_still_has_its_planned_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise InsufficientHistoryError("short")

    monkeypatch.setattr(ablation, "run_case", failing)
    result = run_ablation(cases(), on_error="record")
    assert result.empty
    assert result.attrs["coverage"]["n_planned_cases"] == 2
    assert result.attrs["coverage"]["n_failed_cases"] == 2
    assert result.attrs["coverage"]["case_coverage"] == 0
    assert ablation.ablation_report(result).empty
    assert ablation.ablation_uncertainty(result).empty


def test_undefined_metrics_survive_checkpoint_roundtrip(
    tmp_path: Path,
    calls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    def undefined(case, seed, *args, **kwargs):  # type: ignore[no-untyped-def]
        return [{**row, "test_score": np.nan, "gain_vs_base": np.nan} for row in rows(case, seed)]

    monkeypatch.setattr(ablation, "run_case", undefined)
    path = tmp_path / "study.sqlite"
    first = run_ablation(cases(), checkpoint=path)
    resumed = run_ablation(cases(), checkpoint=path, resume=True)
    pd.testing.assert_frame_equal(first, resumed)
    assert ablation.ablation_report(resumed)["n_paired"].eq(0).all()
    assert ablation.ablation_uncertainty(resumed)["n_paired_series"].eq(0).all()
    coverage = resumed.attrs["coverage"]["procedures"]["incremental"]
    assert coverage["n_planned"] == coverage["n_succeeded"] == 2
    assert coverage["n_metric_valid"] == coverage["n_paired"] == 0


def test_retry_preserves_partial_snapshots_and_total_attempt_cost(
    tmp_path: Path,
    calls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    original = ablation.run_case

    def partial(case, seed, *args, snapshot_callback=None):  # type: ignore[no-untyped-def]
        snapshot_callback({"phase": "full", "snapshot": {"partial": True}})
        raise InsufficientHistoryError("outer fold too short")

    monkeypatch.setattr(ablation, "run_case", partial)
    path = tmp_path / "study.sqlite"
    run_ablation(cases(), checkpoint=path, on_error="record")
    monkeypatch.setattr(ablation, "run_case", original)
    result = run_ablation(cases(), checkpoint=path, resume=True, retry_failed=True)
    assert result.attrs["coverage"]["n_attempts"] == 4
    assert result.attrs["coverage"]["known_seconds_total"] > 0
    assert result.attrs["coverage"]["n_unknown_duration_attempts"] == 0
    for state in result.attrs["cases"]:
        assert state["history"][0]["snapshots"][0]["snapshot"]["partial"]


def test_benchmark_ids_and_seasonality_are_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    from autohybts.datasets.descriptor import BenchmarkDataset, DatasetDescriptor

    frames = [
        pd.DataFrame(
            {
                "series_id": case.series_id,
                "timestamp": case.values.index,
                "value": case.values.to_numpy(),
            }
        )
        for case in cases()
    ]
    descriptor = DatasetDescriptor(
        "fixture",
        "fixture/daily",
        "fixture",
        "1",
        None,
        "daily",
        2,
        "value",
        "timestamp",
        "series_id",
        "official",
    )
    benchmark = BenchmarkDataset(
        pd.concat([frame.iloc[:-2] for frame in frames]),
        pd.concat([frame.iloc[-2:] for frame in frames]),
        pd.DataFrame({"seasonal_period": [1, 7]}, index=["a", "b"]),
        descriptor,
    )
    monkeypatch.setattr(ablation, "load_benchmark", lambda *args: benchmark)
    selected = list(ablation.benchmark_cases("fixture", "daily", 2, series_ids=["b", "a"]))
    assert [case.series_id for case in selected] == ["b", "a"]
    assert [case.seasonal_period for case in selected] == [7, 1]
    assert all(case.test_size == 2 for case in selected)
    with pytest.raises(ConfigurationError, match="disponíveis"):
        list(ablation.benchmark_cases("fixture", "daily", 3))


def test_config_separates_sampling_and_model_seeds() -> None:
    config = ablation._study_config(
        {
            "preset": "research",
            "dataset": "m4-daily",
            "series_ids": ["D1", "D2"],
            "sampling_seed": 17,
            "seeds": [3, 5],
            "space": {"metrics": ["RMSE"]},
        }
    )
    assert config["space"]["fold"] == "auto"
    assert config["sampling_seed"] == 17
    assert config["seeds"] == [3, 5]
    assert config["budget"] == "small"
    assert config["series"] == 2


@pytest.mark.parametrize(
    "raw",
    [
        {"unknown": 1},
        {"schema": "unknown"},
        {"series": 0},
        {"space": {"fold": 0}},
        {"seeds": [0, 0]},
        {"space": {"test_size": 365}},
        {"space": {"registry": {}}},
        {"series_ids": ["D1"]},
        {"preset": "unknown"},
        {"sampling_seed": -1},
    ],
)
def test_invalid_config_is_rejected_before_loading_data(raw: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        ablation._study_config(raw)


def test_series_identity_includes_values_and_dates() -> None:
    y = cases()[0].values
    assert series_fingerprint(y) != series_fingerprint(y + 1)
    shifted = y.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    assert series_fingerprint(y) != series_fingerprint(shifted)
