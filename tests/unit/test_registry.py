from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from autohybts.adapters.base import FloatArray, ForecasterBase
from autohybts.exceptions import ConfigurationError
from autohybts.registry import Capabilities, ModelRegistry, default_registry


class MeanForecaster(ForecasterBase):
    """Custom model used to exercise registration (§12.3)."""

    def __init__(self, shrink: float = 1.0) -> None:
        super().__init__()
        self.shrink = shrink

    def _fit(self, y: FloatArray) -> None:
        self.mean_ = float(np.mean(y)) * self.shrink

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        return np.full(horizon, self.mean_)


def mean_factory(params: Mapping[str, Any]) -> MeanForecaster:
    return MeanForecaster(**params)


def test_default_registry_contents() -> None:
    registry = default_registry()
    assert set(registry.names()) == {
        "naive",
        "seasonal_naive",
        "arima",
        "linear",
        "knn",
        "svm",
        "tree",
        "random_forest",
        "mlp",
        # opt-in additions of the ensemble plan (F1a, F1c): registered, not searched by default
        "historic_average",
        "window_average",
        "seasonal_window_average",
        "random_walk_drift",
        "exponential_smoothing",
        "harmonic_regression",
        "sf_auto_arima",
        "sf_auto_ets",
        "sf_auto_theta",
        "sf_auto_ces",
        "ridge_regression",
        "lasso_regression",
        "elastic_net_regression",
        "extra_trees_regressor",
        "hist_gradient_boosting",
        "adaboost_r2",
        "lightgbm_regressor",
        "xgboost_regressor",
        "catboost_regressor",
        # intermittent demand (L08), a family of its own
        "croston_sba",
        "tsb",
        "adida",
        "imapa",
    }
    assert set(registry.names("corrector")) == {
        "linear",
        "knn",
        "mlp",
        "svm",
        "tree",
        "random_forest",
        "hist_gradient_boosting",
        "adaboost_r2",
        "lightgbm_regressor",
        "xgboost_regressor",
        "catboost_regressor",
        # residual sequence correctors (F1e)
        "arima_sequence",
        "exponential_smoothing_sequence",
    }
    # Known families first, in the default order; the others (intermittent) after them.
    assert registry.families() == (
        "statistical",
        "linear",
        "kernel",
        "tree",
        "neural",
        "intermittent",
    )
    assert registry.names(family="tree") == (
        "adaboost_r2",
        "catboost_regressor",
        "extra_trees_regressor",
        "hist_gradient_boosting",
        "lightgbm_regressor",
        "random_forest",
        "tree",
        "xgboost_regressor",
    )
    assert registry.entry("arima").capabilities.requires_extra == "stats"


def test_default_registry_is_a_new_object_each_time() -> None:
    first = default_registry()
    first.register_base("mean", mean_factory, family="statistical")
    assert "mean" in first
    assert "mean" not in default_registry()


def test_custom_model_registration_and_defaults() -> None:
    registry = ModelRegistry()
    registry.register_base("mean", mean_factory, family="baseline", default_params={"shrink": 0.5})
    model = registry.make_base("mean")
    assert model.get_params() == {"shrink": 0.5}
    assert registry.make_base("mean", {"shrink": 2.0}).get_params() == {"shrink": 2.0}
    assert registry.families() == ("baseline",)


def test_duplicates_need_explicit_replace() -> None:
    registry = ModelRegistry()
    registry.register_base("mean", mean_factory, family="statistical")
    with pytest.raises(ConfigurationError, match="replace=True"):
        registry.register_base("mean", mean_factory, family="statistical")
    registry.register_base("mean", mean_factory, family="linear", replace=True)
    assert registry.entry("mean").family == "linear"


@pytest.mark.parametrize("name", ["Mean", "1model", "with-dash", ""])
def test_invalid_names(name: str) -> None:
    with pytest.raises(ConfigurationError, match="Nome de modelo inválido"):
        ModelRegistry().register_base(name, mean_factory, family="statistical")


def test_factory_must_build_a_forecaster() -> None:
    with pytest.raises(ConfigurationError, match="contrato"):
        ModelRegistry().register_base("bad", lambda p: object(), family="statistical")
    with pytest.raises(ConfigurationError, match="chamável"):
        ModelRegistry().register_base("bad", "not callable", family="statistical")


def test_declared_in_sample_capability_is_checked() -> None:
    with pytest.raises(ConfigurationError, match="fitted_values"):
        ModelRegistry().register_base(
            "mean",
            mean_factory,
            family="statistical",
            capabilities=Capabilities(in_sample_fitted=True),
        )


def test_unknown_model_lists_alternatives() -> None:
    with pytest.raises(ConfigurationError, match="Disponíveis: adaboost_r2, adida, arima"):
        default_registry().make_base("prophet")


def test_copy_is_independent() -> None:
    registry = ModelRegistry()
    snapshot = registry.copy()
    registry.register_base("mean", mean_factory, family="statistical")
    assert "mean" not in snapshot


def test_catalog_frame() -> None:
    table = default_registry().to_frame()
    assert table.loc[("base", "arima"), "in_sample_fitted"]
    assert not table.loc[("base", "mlp"), "deterministic"]
    assert table.loc[("corrector", "linear"), "family"] == "corrector"
    assert table.index.names == ["kind", "name"]
    assert {"family", "multi_step", "online_state_update", "description"} <= set(table.columns)


def test_lag_model_params_round_trip_and_estimator_mismatch() -> None:
    registry = default_registry()
    params = registry.make_base("svm", {"lags": 3}).get_params()
    assert params["estimator"] == "svm"
    assert registry.make_base("svm", params).get_params() == params
    with pytest.raises(ConfigurationError, match="estimator='knn'"):
        registry.make_base("svm", {"estimator": "knn"})


def test_factory_output_is_checked_on_every_build() -> None:
    calls = {"n": 0}

    def flaky(params: Mapping[str, Any]) -> object:
        calls["n"] += 1
        return MeanForecaster() if calls["n"] == 1 else object()

    registry = ModelRegistry()
    registry.register_base("flaky", flaky, family="statistical")
    with pytest.raises(ConfigurationError, match="não devolveu"):
        registry.make_base("flaky")


def test_default_correctors() -> None:
    registry = default_registry()
    assert {"knn", "linear", "mlp"} <= set(registry.names("corrector"))
    corrector = registry.make_corrector("knn", {"estimator_params": {"n_neighbors": 3}})
    assert corrector.get_params()["estimator"] == "knn"
    with pytest.raises(ConfigurationError, match="estimator='mlp'"):
        registry.make_corrector("knn", {"estimator": "mlp"})
    with pytest.raises(ConfigurationError, match="não registrado"):
        registry.make_corrector("arima")


def test_corrector_contract_is_checked() -> None:
    registry = ModelRegistry()
    with pytest.raises(ConfigurationError, match="ResidualCorrector"):
        registry.register_corrector("bad", mean_factory)
    calls = {"n": 0}

    def flaky(params: Mapping[str, Any]) -> object:
        from autohybts.residuals.correctors import SklearnCorrector

        calls["n"] += 1
        return SklearnCorrector() if calls["n"] == 1 else object()

    registry.register_corrector("flaky", flaky)
    with pytest.raises(ConfigurationError, match="não devolveu"):
        registry.make_corrector("flaky")
