"""Public API end to end: T04, T13, T15, T25, UX05, UX08-UX10, UX12, UX13, oracle 3."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autohybts import (
    AutoHybridExperiment,
    AutoHybridForecaster,
    FinalizedForecaster,
    load_model,
    read_manifest,
    save_model,
)
from autohybts.data.splitter import SealedTest
from autohybts.datasets import get_data
from autohybts.exceptions import (
    ArtifactCompatibilityError,
    ConfigurationError,
    SealedTestAccessError,
)
from autohybts.experiment import infer_seasonal_period

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

FAST = {
    "fh": 7,
    "fold": 2,
    "budget": "smoke",
    "base_models": ("naive", "seasonal_naive", "linear", "knn"),
    "correctors": ("linear",),
    "residual_windows": (7,),
    "alphas": (1.0,),
}


def series(n: int = 240, seed: int = 0) -> pd.Series:
    return get_data("synthetic_daily", n=n, seed=seed)


@pytest.fixture(scope="module")
def fitted() -> tuple[AutoHybridExperiment, object, FinalizedForecaster]:
    exp = AutoHybridExperiment(**FAST).fit(series())
    result = exp.compare_models()
    return exp, result, exp.finalize_model(result.best)


def test_fit_seals_the_test_and_freezes_the_plan() -> None:
    y = series()
    exp = AutoHybridExperiment(**FAST).fit(y)
    assert len(exp.history_) == len(y) - 7
    assert exp.history_.end == y.index[-8]
    assert list(exp.splits.index) == ["search-0", "search-1", "select-0", "select-1"]
    assert exp.splits["last_target"].max() == exp.history_.end  # nothing reaches the test
    assert exp.seasonal_period_ == 7


def test_compare_returns_leaderboards_decisions_and_audit(fitted) -> None:
    # UX05.
    _, result, _ = fitted
    assert result.leaderboard["is_champion"].sum() == 1
    assert {"mean_rank", "MASE", "coverage", "family", "strategy"} <= set(
        result.leaderboard.columns
    )
    assert len(result.decisions) == len(result.selection.decisions)
    assert result.audit.total_trials <= 10  # smoke budget
    assert set(result.selection_leaderboard.index) == {
        r.candidate_id for r in result.selection.finalists
    }
    assert [e.kind for e in result.events][:3] == [
        "experiment.fitted",
        "search.finished",
        "selection.finished",
    ]


def test_sealed_test_is_only_read_by_external_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    # UX12: compare and finalize never unseal; evaluate_external does, once per model.
    exp = AutoHybridExperiment(**FAST).fit(series())

    def refuse(self: SealedTest) -> object:
        raise AssertionError("sealed test read before external evaluation")

    monkeypatch.setattr(SealedTest, "_unseal", refuse)
    final = exp.finalize_model(exp.compare_models().best)
    monkeypatch.undo()
    exp.evaluate_external(final)
    with pytest.raises(SealedTestAccessError, match="já foi avaliado"):
        exp.evaluate_external(final)


def test_changing_the_sealed_test_changes_nothing_before_evaluation() -> None:
    # T04 and oracle 3: a shock inside the sealed test leaves selection identical.
    y = series()
    shocked = y.copy()
    shocked.iloc[-7:] += 500.0
    runs = []
    for values in (y, shocked):
        exp = AutoHybridExperiment(**FAST).fit(values)
        result = exp.compare_models()
        final = exp.finalize_model(result.best)
        runs.append((exp, result, final))
    (exp_a, a, final_a), (exp_b, b, final_b) = runs
    assert a.best == b.best
    pd.testing.assert_frame_equal(
        a.leaderboard.drop(columns=["fit_seconds", "predict_seconds"]),
        b.leaderboard.drop(columns=["fit_seconds", "predict_seconds"]),
    )
    assert a.decisions.equals(b.decisions)
    assert a.snapshot == b.snapshot
    pd.testing.assert_frame_equal(final_a.predict(7), final_b.predict(7))
    ext_a, ext_b = exp_a.evaluate_external(final_a), exp_b.evaluate_external(final_b)
    assert ext_b.metrics["MAE"].iloc[0] > ext_a.metrics["MAE"].iloc[0]  # only the final score moves


def test_refit_uses_only_the_pre_test_history(fitted) -> None:
    # T13.
    exp, _, final = fitted
    assert final.history.fingerprint == exp.history_.fingerprint
    assert final.training_cutoff == exp.history_.end
    assert final.manifest["training_cutoff"] == str(exp.history_.end)
    assert final.manifest["sealed_test"]["size"] == 7
    predictions = final.predict(3)
    assert predictions.index[0] == exp.history_.end + pd.Timedelta(days=1)


def test_predictions_are_dated_and_refit_free(fitted) -> None:
    exp, _, final = fitted
    result = exp.predict_model(final)
    assert len(result.predictions) == 7
    assert list(result.predictions.columns) == ["series_id", "origin", "horizon", "y_pred"]
    assert list(result.predictions["horizon"]) == list(range(1, 8))
    assert len(exp.predict_model(final, fh=20).predictions) == 20


def test_save_load_in_another_process(fitted, tmp_path: Path) -> None:
    # T15 / UX08.
    _, _, final = fitted
    path = save_model(final, tmp_path / "champion")
    expected = final.predict(7)["y_pred"].tolist()
    code = (
        "import json, sys; from autohybts import load_model; "
        "m = load_model(sys.argv[1], trusted=True); "
        "print(json.dumps(m.predict(7)['y_pred'].tolist()))"
    )
    output = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, text=True, check=True
    ).stdout
    assert np.allclose(json.loads(output), expected, rtol=0, atol=1e-12)


def test_untrusted_or_tampered_artifacts_are_refused(fitted, tmp_path: Path) -> None:
    # UX10.
    _, _, final = fitted
    path = save_model(final, tmp_path / "m")
    manifest = read_manifest(path)  # plain JSON, no code runs
    assert manifest["spec"]["label"] == final.spec.label
    assert manifest["environment"]["autohybts"]
    with pytest.raises(ArtifactCompatibilityError, match="trusted=True"):
        load_model(path)
    payload = path / "model.pkl"
    payload.write_bytes(payload.read_bytes() + b"tampered")
    with pytest.raises(ArtifactCompatibilityError, match="Checksum"):
        load_model(path, trusted=True)
    with pytest.raises(ArtifactCompatibilityError, match="Manifesto"):
        read_manifest(tmp_path / "missing")


def test_signed_and_pinned_artifacts_resist_tampering(fitted, tmp_path: Path) -> None:
    # The manifest checksum alone can be rewritten together with the payload; a pinned hash
    # or an HMAC key kept outside the artifact cannot.
    _, _, final = fitted
    key = b"k" * 32
    path = save_model(final, tmp_path / "signed", signing_key=key)
    manifest = read_manifest(path)
    digest = manifest["payload"]["sha256"]
    assert manifest["payload"]["hmac_sha256"]
    assert load_model(path, trusted=True, signing_key=key, sha256=digest).spec == final.spec
    with pytest.raises(ArtifactCompatibilityError, match="HMAC"):
        load_model(path, trusted=True, signing_key=b"x" * 32)
    with pytest.raises(ArtifactCompatibilityError, match="sha256="):
        load_model(path, trusted=True, sha256="0" * 64)
    with pytest.raises(ArtifactCompatibilityError, match="16 bytes"):
        load_model(path, trusted=True, signing_key=b"short")
    # Payload and manifest rewritten consistently: the plain checksum passes, the key does not.
    unsigned = save_model(final, tmp_path / "unsigned")
    for name in ("model.pkl", "manifest.json"):
        (path / name).write_bytes((unsigned / name).read_bytes())
    assert load_model(path, trusted=True).spec == final.spec
    with pytest.raises(ArtifactCompatibilityError, match="HMAC"):
        load_model(path, trusted=True, signing_key=key)


def test_manifest_cannot_point_the_payload_elsewhere(fitted, tmp_path: Path) -> None:
    _, _, final = fitted
    path = save_model(final, tmp_path / "m")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest["payload"]["file"] = "../other.pkl"
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match=r"model\.pkl"):
        load_model(path, trusted=True)


def test_invalid_calls_fail_explicitly(fitted) -> None:
    # UX09.
    exp, result, _ = fitted
    with pytest.raises(ConfigurationError, match="finalizado"):
        exp.predict_model(result.best)
    misaligned = series().iloc[-5:]
    fresh = exp.finalize_model(result.best)
    with pytest.raises(ConfigurationError, match="começar em"):
        exp.evaluate_external(fresh, misaligned)
    with pytest.raises(ConfigurationError, match="fit"):
        AutoHybridExperiment().compare_models()
    with pytest.raises(ConfigurationError, match="include"):
        exp.compare_models(include=("residual",))
    with pytest.raises(ConfigurationError, match=r"pd.Series"):
        AutoHybridExperiment().fit(series().to_frame())


def test_rerun_and_snapshot_reproduce_the_selection(fitted) -> None:
    # UX13 and T25.
    _, result, _ = fitted
    again = AutoHybridExperiment(**FAST).fit(series()).compare_models()
    assert again.best == result.best
    assert again.decisions.equals(result.decisions)
    rebuilt = AutoHybridExperiment.from_snapshot(json.loads(json.dumps(result.snapshot)))
    assert rebuilt.fit(series()).compare_models().best == result.best
    with pytest.raises(ConfigurationError, match="esquema"):
        AutoHybridExperiment.from_snapshot({"schema": "other"})


def test_operational_mode_without_sealed_test() -> None:
    y = series()
    exp = AutoHybridExperiment(**{**FAST, "test_size": 0}).fit(y)
    final = exp.finalize_model(exp.compare_models().best)
    assert final.training_cutoff == y.index[-1]
    assert final.predict(3).index[0] == y.index[-1] + pd.Timedelta(days=1)
    with pytest.raises(ConfigurationError, match="y_test"):
        exp.evaluate_external(final)


def test_external_test_can_be_supplied(fitted) -> None:
    y = series(260)
    exp = AutoHybridExperiment(**{**FAST, "test_size": 0}).fit(y.iloc[:240])
    final = exp.finalize_model(exp.compare_models().best)
    external = exp.evaluate_external(final, y.iloc[240:250])
    assert len(external.predictions) == 10
    assert {"MASE", "MAE"} <= set(external.metrics.columns)


def test_create_model_and_hybrid() -> None:
    exp = AutoHybridExperiment(**FAST).fit(series())
    model = exp.create_model("linear", lags=7)
    assert list(model.cv_results.index[-2:]) == ["Mean", "Std"]
    assert len(model.cv_results) == 4 + 2
    hybrid = exp.create_hybrid(
        base="seasonal_naive", corrector="linear", base_params={"season_length": 7}
    )
    assert hybrid.candidate.strategy == "residual_additive"
    assert set(hybrid.metrics) == {"MASE", "MAE", "RMSE", "sMAPE"}
    assert exp.finalize_model(hybrid).spec == hybrid.candidate
    assert not exp.models().empty


def test_one_call_estimator() -> None:
    forecaster = AutoHybridForecaster(**FAST).fit(series())
    assert len(forecaster.predict()) == 7
    assert forecaster.leaderboard()["is_champion"].sum() == 1
    assert isinstance(forecaster.selection_history_, pd.DataFrame)
    assert forecaster.get_best_model().training_cutoff == series().index[-1]
    with pytest.raises(ConfigurationError):
        AutoHybridForecaster().predict()


@pytest.mark.parametrize(
    ("frequency", "period"),
    [("D", 7), ("MS", 12), ("W-SUN", 52), ("h", 24), ("YS-JAN", 1), ("15min", 1)],
)
def test_seasonal_period_inference(frequency: str, period: int) -> None:
    assert infer_seasonal_period(frequency) == period


def test_get_data() -> None:
    y = get_data("synthetic_daily", n=50)
    assert len(y) == 50
    assert y.index.freqstr == "D"
    with pytest.raises(ConfigurationError, match="synthetic_daily"):
        get_data("airline")
