"""Profiling, plots, metrics tables and evidence: UX04, UX06, UX07, UX14, CSV journey."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autohybts import AutoHybridExperiment
from autohybts.data.ingest import read_csv
from autohybts.datasets import get_data
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError
from autohybts.reporting.diagnostics import (
    champion_evolution,
    error_by_horizon,
    forecast_errors,
    residual_provenance,
)
from autohybts.reporting.plots import DATA_PLOTS, MODEL_PLOTS

needs_matplotlib = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None, reason="extra `plot` not installed"
)
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

FAST = {
    "fh": 7,
    "fold": 2,
    "budget": "smoke",
    "base_models": ("naive", "seasonal_naive", "linear"),
    "correctors": ("linear",),
    "residual_windows": (7,),
    "alphas": (1.0,),
}


@pytest.fixture(scope="module")
def experiment():
    exp = AutoHybridExperiment(**FAST).fit(get_data("synthetic_daily", n=240))
    result = exp.compare_models()
    return exp, result


def test_data_profile_covers_only_development() -> None:
    y = get_data("synthetic_daily", n=240)
    exp = AutoHybridExperiment(**FAST).fit(y)
    profile = exp.data_profile()["value"]
    assert profile["observações"] == 233
    assert profile["fim"] == y.index[-8]
    assert profile["faltantes"] == 0


@needs_matplotlib
@pytest.mark.parametrize("kind", DATA_PLOTS)
def test_data_plots_never_show_the_sealed_test(kind: str) -> None:
    # UX04.
    y = get_data("synthetic_daily", n=240)
    exp = AutoHybridExperiment(**FAST).fit(y)
    artifact = exp.plot_data(kind)
    assert artifact.source_split == "development"
    assert artifact.origin_range == (y.index[0], y.index[-8])
    assert artifact.provenance["sealed_test_size"] == 7
    if kind == "series":
        (line,) = artifact.figure.axes[0].get_lines()
        assert pd.Timestamp(max(line.get_xdata())) == y.index[-8]
        assert len(line.get_ydata()) == 233


@needs_matplotlib
def test_invalid_data_plots() -> None:
    exp = AutoHybridExperiment(**{**FAST, "seasonal_period": 1}).fit(
        get_data("synthetic_daily", n=240)
    )
    with pytest.raises(ConfigurationError, match="sazonal"):
        exp.plot_data("seasonality")
    with pytest.raises(ConfigurationError, match="desconhecido"):
        exp.plot_data("violin")


@needs_matplotlib
@pytest.mark.parametrize("kind", MODEL_PLOTS)
def test_model_plots_use_validation_only(experiment, kind: str) -> None:
    # UX06.
    exp, result = experiment
    artifact = exp.plot_model(result.best, kind)
    assert artifact.source_split == "validation"
    assert artifact.candidate_id == result.best.candidate_id
    _, last = artifact.origin_range
    assert last < exp.history_.end
    assert artifact.provenance["layers"] == ["search", "select"]
    assert artifact.provenance["units"] == 4


@needs_matplotlib
def test_plots_reuse_evaluations(experiment, monkeypatch: pytest.MonkeyPatch) -> None:
    exp, result = experiment
    exp.plot_model(result.best, "cv_forecasts")  # warm: combines search + select layers

    def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError("re-evaluated a candidate already evaluated")

    monkeypatch.setattr(TemporalEvaluator, "evaluate", refuse)
    exp.plot_model(result.best, "residuals")
    exp.get_metrics(result.best)


@needs_matplotlib
def test_selection_plots_need_compare() -> None:
    exp = AutoHybridExperiment(**FAST).fit(get_data("synthetic_daily", n=240))
    model = exp.create_model("naive")
    assert exp.plot_model(model, "cv_forecasts").kind == "cv_forecasts"
    with pytest.raises(ConfigurationError, match="compare_models"):
        exp.plot_model(model, "selection_trace")
    with pytest.raises(ConfigurationError, match="compare_models"):
        exp.export_evidence("unused")


def test_forecast_errors_are_aligned_with_the_targets() -> None:
    y = get_data("synthetic_daily", n=240)
    exp = AutoHybridExperiment(**FAST).fit(y)
    errors = forecast_errors(exp.create_model("naive").evaluation)
    history = exp.history_.to_series()
    for _, row in errors.iterrows():
        assert row["timestamp"] == row["origin"] + pd.Timedelta(days=int(row["horizon"]))
        assert row["y_true"] == history[row["timestamp"]]
        assert row["y_pred"] == history[row["origin"]]  # naive repeats the origin value


def test_diagnostic_tables(experiment) -> None:
    exp, result = experiment
    evaluation = exp.create_model("linear", lags=7).evaluation
    by_horizon = error_by_horizon(evaluation)
    assert set(by_horizon["layer"]) == {"search", "select"}
    assert (by_horizon["n_origins"] == 2).all()
    assert list(by_horizon[by_horizon["layer"] == "search"]["horizon"]) == list(range(1, 8))
    evolution = champion_evolution(result.selection.decisions, "start")
    assert len(evolution) == len(result.selection.decisions) + 1
    assert evolution["champion"].iloc[-1] in {"start", result.best.label}


def test_residual_provenance_of_a_finalized_hybrid() -> None:
    exp = AutoHybridExperiment(**FAST).fit(get_data("synthetic_daily", n=240))
    hybrid = exp.create_hybrid(
        base="seasonal_naive", corrector="linear", base_params={"season_length": 7}
    )
    final = exp.finalize_model(hybrid)
    table = residual_provenance(final.model)
    assert (table["available_at"] == table["target"]).all()
    assert table["target"].max() == len(exp.history_) - 1
    with pytest.raises(ConfigurationError, match="híbrido"):
        residual_provenance(exp.finalize_model(exp.create_model("naive")).model)


def test_metrics_tables(experiment) -> None:
    exp, result = experiment
    summary = exp.get_metrics()
    assert set(summary["layer"]) == {"search", "select"}
    assert {"MASE", "n_units", "coverage"} <= set(summary.columns)
    per_origin = exp.get_metrics(result.best)
    assert len(per_origin) == 4


@needs_matplotlib
def test_evidence_bundle_without_raw_values(experiment, tmp_path: Path) -> None:
    # UX07.
    exp, result = experiment
    bundle = exp.export_evidence(tmp_path / "evidence", plots=("selection_trace",))
    expected = {
        "audit.json",
        "leaderboard.csv",
        "manifest.json",
        "metrics.csv",
        "plot_selection_trace.png",
        "selection_history.json",
        "selection_leaderboard.csv",
        "snapshot.json",
        "splits.json",
        "trials.csv",
    }
    assert set(bundle.files) == expected
    manifest = json.loads((bundle.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == result.audit.protocol_hash == bundle.experiment_id
    assert manifest["raw_series_values_included"] is False
    text = "".join(
        (bundle.path / name).read_text(encoding="utf-8")
        for name in expected
        if not name.endswith(".png")
    )
    for value in np.random.default_rng(0).choice(exp.history_.values, 25, replace=False):
        assert f"{value:.8f}"[:10] not in text
    decisions = json.loads((bundle.path / "selection_history.json").read_text(encoding="utf-8"))
    assert len(decisions) == len(result.selection.decisions)


def test_forecasting_works_without_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    # UX14.
    monkeypatch.setitem(sys.modules, "matplotlib.figure", None)
    exp = AutoHybridExperiment(**FAST).fit(get_data("synthetic_daily", n=240))
    final = exp.finalize_model(exp.create_model("naive").candidate)
    assert len(exp.predict_model(final).predictions) == 7
    with pytest.raises(UnsupportedCapabilityError, match=r"autohybts\[plot\]"):
        exp.plot_data("series")


def test_csv_journey(tmp_path: Path) -> None:
    # Acceptance: CSV in, forecast out, with explicit column mapping.
    y = get_data("synthetic_daily", n=240)
    path = tmp_path / "vendas.csv"
    pd.DataFrame({"dia": y.index.strftime("%Y-%m-%d"), "vendas": y.to_numpy()}).to_csv(
        path, index=False, sep=";"
    )
    frame = read_csv(path, target="vendas", date_col="dia", sep=";")
    exp = AutoHybridExperiment(**FAST).fit(frame)
    final = exp.finalize_model(exp.compare_models().best)
    forecast = exp.predict_model(final).predictions
    assert forecast.index[0] == y.index[-7]  # first sealed date
    assert np.isfinite(exp.evaluate_external(final).metrics["MASE"].iloc[0])
