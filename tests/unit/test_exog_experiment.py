"""Exogenous covariates through the public experiment API (ADR-0011)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autohybts import AutoHybridExperiment, load_model, save_model
from autohybts.domain.specs import ExogFeature
from autohybts.exceptions import (
    ConfigurationError,
    DataValidationError,
    UnsupportedCapabilityError,
)

N, FH, EXTRA = 200, 5, 20
SETTINGS = {
    "fold": 2,
    "budget": "smoke",
    "base_models": ["naive", "linear"],
    "correctors": ["linear"],
}


def data(seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    """y depends on x five days earlier and on a holiday known in advance."""

    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=N + EXTRA, freq="D")
    x = rng.normal(0, 1, N + EXTRA)
    holiday = (np.arange(N + EXTRA) % 7 == 0).astype(float)
    y = 30 + 4 * np.roll(x, FH) + 6 * holiday + np.cumsum(rng.normal(0, 0.2, N + EXTRA))
    return pd.Series(y[:N], index=index[:N]), pd.DataFrame(
        {"x": x, "feriado": holiday}, index=index
    )


def experiment(variant: str = "A", **extra: object) -> AutoHybridExperiment:
    y, X = data()
    params = {**SETTINGS, "fh": FH if variant == "A" else 1, "test_size": FH, "variant": variant}
    return AutoHybridExperiment(**{**params, "exog_future": ["feriado"], **extra}).fit(y, X)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def compared() -> tuple[AutoHybridExperiment, object]:
    exp = experiment()
    return exp, exp.compare_models()


def test_without_covariates_nothing_changes() -> None:
    y, _ = data()
    exp = AutoHybridExperiment(fh=FH, **SETTINGS).fit(y)  # type: ignore[arg-type]
    assert exp.exog_ is None
    assert exp.exog_features_ == ()
    assert "exog" not in exp.evaluator_.protocol()
    result = exp.compare_models()
    assert "exogenous" not in result.snapshot["include"]
    assert "exog_sets" not in result.snapshot["space"]
    with pytest.raises(ConfigurationError, match="exige X"):
        exp.compare_models(include=["standalone", "exogenous"])
    with pytest.raises(ConfigurationError, match="exige X"):
        exp.create_model("linear", exog=True)


def test_features_follow_kinds_lags_and_the_test_size() -> None:
    exp = experiment(exog_lags={"x": [7]})
    assert exp.exog_.kinds == ("past", "future")
    assert exp.exog_features_ == (
        ExogFeature("feriado", 0),
        ExogFeature("x", FH),
        ExogFeature("x", 7),
    )
    y, X = data()
    longer_test = AutoHybridExperiment(fh=3, test_size=FH, exog_future=["feriado"], **SETTINGS)
    assert ExogFeature("x", FH) in longer_test.fit(y, X).exog_features_  # type: ignore[arg-type]
    assert ExogFeature("x", 1) in experiment("B").exog_features_


def test_fit_validates_the_covariates() -> None:
    y, X = data()
    with pytest.raises(ConfigurationError, match="lag >= 5"):
        experiment(exog_lags={"x": [2]})
    with pytest.raises(ConfigurationError, match="inexistentes"):
        AutoHybridExperiment(fh=FH, exog_future=["chuva"], **SETTINGS).fit(y, X)  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="não cobre"):
        AutoHybridExperiment(fh=FH, **SETTINGS).fit(y, X.iloc[10:])  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="exigem X"):
        AutoHybridExperiment(fh=FH, exog_future=["feriado"], **SETTINGS).fit(y)  # type: ignore[arg-type]
    with pytest.raises(UnsupportedCapabilityError, match="painel"):
        AutoHybridExperiment(fh=FH, **SETTINGS).fit({"a": y, "b": y + 1}, X)  # type: ignore[arg-type]


def test_compare_models_tries_and_selects_the_exogenous_family(compared) -> None:
    exp, result = compared
    assert "exogenous" in result.snapshot["include"]
    assert result.snapshot["space"]["exog_sets"] == [[["feriado", 0], ["x", FH]]]
    ledger = result.search.ledger_frame()
    assert (ledger.stage == "SX").any()
    assert set(ledger.loc[ledger.stage == "SX", "family"]) == {"exogenous"}
    decision = result.decisions.set_index("family").loc["exogenous"]
    assert bool(decision.accepted)
    assert result.best.exog
    explicit = exp.compare_models(include=["standalone", "residual"])
    assert not (explicit.search.ledger_frame().family == "exogenous").any()


def test_finalized_model_forecasts_with_the_experiment_covariates(compared) -> None:
    exp, result = compared
    final = exp.finalize_model(result.best)
    assert final.exog is not None
    assert final.manifest["exog"]["kinds"] == {"feriado": "future", "x": "past"}
    automatic = exp.predict_model(final).predictions
    _, X = data()
    explicit = exp.predict_model(final, X_future=X).predictions
    pd.testing.assert_frame_equal(automatic, explicit)
    with pytest.raises(ConfigurationError, match="defasagem"):
        exp.predict_model(final, fh=FH + 1)
    external = exp.evaluate_external(final)
    assert external.metrics["MASE"].iloc[0] < 1.0


def test_future_columns_are_needed_for_the_horizon(compared) -> None:
    exp, result = compared
    final = exp.finalize_model(result.best)
    with pytest.raises(DataValidationError, match="future"):
        final.predict(FH)
    _, X = data()
    assert final.predict(FH, X[["feriado"]]).shape[0] == FH


def test_persisted_model_keeps_its_covariates(compared, tmp_path: Path) -> None:
    exp, result = compared
    final = exp.finalize_model(result.best)
    _, X = data()
    loaded = load_model(save_model(final, tmp_path / "model"), trusted=True)
    pd.testing.assert_frame_equal(loaded.predict(FH, X), final.predict(FH, X))


def test_variant_b_streams_with_the_covariates_of_each_date() -> None:
    exp = experiment("B", exog_lags={"x": [FH]})
    final = exp.finalize_model(exp.create_model("linear", exog=True).candidate)
    external = exp.evaluate_external(final)
    assert external.metrics["MASE"].iloc[0] < 1.0
    y, X = data()
    full = pd.Series(np.r_[y.to_numpy(), np.zeros(EXTRA)], index=X.index)
    later = final.update_state(full.iloc[N - FH : N], X_new=X.iloc[N - FH : N])
    stream = later.stream()
    ticket = stream.predict_next(X.loc[X.index[N]])
    stream.observe(1.0, X.loc[X.index[N]])
    assert ticket.timestamp == X.index[N]
    assert stream.model.last_observation == X.index[N]
    plain = exp.finalize_model(exp.create_model("linear").candidate)
    with pytest.raises(ConfigurationError, match="não usa exógenas"):
        plain.update_state(full.iloc[N - FH : N], X_new=X.iloc[N - FH : N])
    with pytest.raises(ConfigurationError, match="não usa exógenas"):
        exp.predict_model(plain, X_future=X)


def test_nested_cross_validation_carries_the_covariates() -> None:
    exp = experiment()
    exp.compare_models()
    nested = exp.nested_cross_validate(outer_folds=2, baselines=(), budget="smoke")
    assert len(nested.folds) == 2
    assert nested.cost["status"].eq("SUCCEEDED").all()
    assert not nested.summary.empty
