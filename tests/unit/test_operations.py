"""Model registry, production forecasts and monitoring in the SQLite store (ADR-0010)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autohybts import AutoHybridExperiment, ExperimentStore, FinalizedForecaster, save_model
from autohybts.datasets import get_data
from autohybts.exceptions import ArtifactCompatibilityError, ConfigurationError

SETTINGS = {"fh": 3, "fold": 2, "budget": "smoke", "base_models": ["naive", "linear"]}


@pytest.fixture(scope="module")
def models() -> tuple[FinalizedForecaster, FinalizedForecaster]:
    y = get_data("synthetic_daily", n=120)
    exp = AutoHybridExperiment(**SETTINGS).fit(y)  # type: ignore[arg-type]
    first = exp.finalize_model(exp.create_model("naive"))
    second = exp.finalize_model(exp.create_model("linear"))
    assert isinstance(first, FinalizedForecaster)
    assert isinstance(second, FinalizedForecaster)
    return first, second


def registered(tmp_path: Path, model: FinalizedForecaster, **kwargs: object) -> ExperimentStore:
    store = ExperimentStore(tmp_path / "ops.db")
    path = save_model(model, tmp_path / model.model_id[:12])
    store.register_model(model, path, name="price", **kwargs)  # type: ignore[arg-type]
    return store


def test_registering_keeps_path_and_hash_and_starts_as_candidate(tmp_path: Path, models) -> None:
    model = models[0]
    store = registered(tmp_path, model, reference_mae=1.5)
    row = store.registry("price").iloc[0]
    manifest = (tmp_path / model.model_id[:12] / "manifest.json").read_text(encoding="utf-8")
    assert row.stage == "candidate"
    assert row.sha256 in manifest
    assert row.reference_mae == 1.5
    store.promote(model.model_id)
    store.register_model(model, row.path, name="price", reference_mae=2.0)
    again = store.registry("price").iloc[0]
    assert again.stage == "production"
    assert again.reference_mae == 2.0


def test_a_new_production_model_archives_the_old_one(tmp_path: Path, models) -> None:
    first, second = models
    store = registered(tmp_path, first)
    store.register_model(second, save_model(second, tmp_path / "second"), name="price")
    store.promote(first.model_id, reason="first")
    store.promote(second.model_id, reason="better")
    stages = dict(store.query("SELECT model_id, stage FROM registry").itertuples(index=False))
    assert stages == {first.model_id: "archived", second.model_id: "production"}
    history = store.query("SELECT model_id, stage FROM stage_history ORDER BY at, stage")
    assert history.stage.tolist().count("candidate") == 2
    assert (history.model_id == first.model_id).sum() == 3


def test_promote_refuses_unknown_models_and_stages(tmp_path: Path, models) -> None:
    store = registered(tmp_path, models[0])
    with pytest.raises(ConfigurationError, match="stage"):
        store.promote(models[0].model_id, "staging")
    with pytest.raises(ConfigurationError, match="não está registrado"):
        store.promote("0" * 64)


def test_load_checks_trust_and_the_registered_hash(tmp_path: Path, models) -> None:
    model = models[0]
    store = registered(tmp_path, model)
    with pytest.raises(ConfigurationError, match="production"):
        store.load("price")
    store.promote(model.model_id)
    with pytest.raises(ArtifactCompatibilityError, match="trusted"):
        store.load("price")
    assert store.load("price", trusted=True).model_id == model.model_id
    payload = tmp_path / model.model_id[:12] / "model.pkl"
    payload.write_bytes(payload.read_bytes() + b"tampered")
    with pytest.raises(ArtifactCompatibilityError):
        store.load("price", trusted=True)


def test_register_refuses_another_models_artifact(tmp_path: Path, models) -> None:
    first, second = models
    store = ExperimentStore(tmp_path / "ops.db")
    path = save_model(first, tmp_path / "first")
    with pytest.raises(ArtifactCompatibilityError, match="não é deste modelo"):
        store.register_model(second, path, name="price")
    with pytest.raises(ConfigurationError, match="name"):
        store.register_model(first, path, name="")


def test_monitor_alerts_only_on_a_full_window_above_the_threshold(tmp_path: Path, models) -> None:
    model = models[0]
    store = registered(tmp_path, model, reference_mae=1.0)
    index = pd.date_range("2030-01-01", periods=5, freq="D")
    store.log_forecasts(model.model_id, pd.DataFrame({"y_pred": np.zeros(5)}, index=index))
    errors = pd.Series([1.0, 1.0, 3.0, 3.0, 3.0], index=index)
    store.log_actuals(model.history.series_id, errors)
    status = store.monitor(model.model_id, window=3, threshold=1.5)
    assert np.allclose(status.rolling_mae, [1, 1, 5 / 3, 7 / 3, 3])
    assert status.alert.tolist() == [False, False, True, True, True]
    store.log_actuals(model.history.series_id, pd.Series([1.0], index=index[2:3]))
    corrected = store.monitor(model.model_id, window=3, threshold=1.5)
    assert corrected.alert.tolist() == [False, False, False, True, True]


def test_monitor_without_a_reference_never_alerts(tmp_path: Path, models) -> None:
    model = models[0]
    store = registered(tmp_path, model)
    table = model.predict(3)
    assert store.log_forecasts(model.model_id, table) == 3
    store.log_actuals(model.history.series_id, table["y_pred"] + 10)
    status = store.monitor(model.model_id, window=2)
    assert status.ratio.isna().all()
    assert not status.alert.any()


def test_operation_inputs_are_validated(tmp_path: Path, models) -> None:
    model = models[0]
    store = registered(tmp_path, model)
    with pytest.raises(ConfigurationError, match="y_pred"):
        store.log_forecasts(model.model_id, pd.DataFrame({"timestamp": []}))
    with pytest.raises(ConfigurationError, match="não está registrado"):
        store.log_forecasts("0" * 64, model.predict(1))
    with pytest.raises(ConfigurationError, match="DatetimeIndex"):
        store.log_actuals("s", pd.Series([1.0]))
    with pytest.raises(ConfigurationError, match="window"):
        store.monitor(model.model_id, window=0)
    with pytest.raises(ConfigurationError, match="threshold"):
        store.monitor(model.model_id, threshold=float("nan"))


def test_runs_come_and_go_without_touching_operations(tmp_path: Path, models) -> None:
    model = models[0]
    store = registered(tmp_path, model)
    exp = AutoHybridExperiment(**SETTINGS).fit(get_data("synthetic_daily", n=120))  # type: ignore[arg-type]
    exp.create_model("naive")
    store.delete_run(store.log(exp))
    assert len(store.registry()) == 1
    assert "registry" in store.tables()


def test_commodity_price_has_a_supply_shock() -> None:
    price = get_data("commodity_price_daily")
    assert len(price) == 1095
    assert (price > 0).all()
    pd.testing.assert_series_equal(price, get_data("commodity_price_daily"))
    k = int(1095 * 0.75)
    before, after = price.iloc[:k].pct_change().std(), price.iloc[k:].pct_change().std()
    assert after > 1.5 * before
    assert price.iloc[k] > 1.15 * price.iloc[k - 1]
    calm = get_data("commodity_price_daily", shock_at=None)
    assert calm.pct_change().std() < 1.2 * before
    with pytest.raises(ConfigurationError, match="shock_at"):
        get_data("commodity_price_daily", shock_at=1.5)


def test_covariates_round_trip_and_later_values_replace_earlier_ones(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "ops.db")
    index = pd.date_range("2030-01-01", periods=3, freq="D")
    frame = pd.DataFrame({"temp": [20.0, np.nan, 22.0], "feriado": [0.0, 1.0, 0.0]}, index=index)
    assert store.log_covariates("s", frame) == 5
    table = store.covariates("s")
    assert table.loc[index[0], "temp"] == 20.0
    assert np.isnan(table.loc[index[1], "temp"])
    store.log_covariates("s", pd.DataFrame({"temp": [25.0]}, index=index[:1]))
    assert store.covariates("s").loc[index[0], "temp"] == 25.0
    assert list(store.covariates("s", start=index[1]).index) == list(index[1:])
    assert list(store.covariates("s", end=index[0]).index) == [index[0]]
    assert store.covariates("other").empty
    with pytest.raises(ConfigurationError, match="DatetimeIndex"):
        store.log_covariates("s", pd.DataFrame({"temp": [1.0]}))
