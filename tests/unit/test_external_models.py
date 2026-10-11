"""User models (I14): version requirements, function/regressor adapters and registration.

Everything here runs without PyTorch or TensorFlow; the framework wrappers are covered in
tests/leakage/test_external_frameworks.py.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from autohybts import AutoHybridExperiment
from autohybts.datasets import get_data
from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError
from autohybts.models import (
    CustomForecaster,
    RegressorForecaster,
    check_specifier,
    installed_version,
    require,
    torch_model,
)
from autohybts.registry import default_registry
from autohybts.residuals.generation import local_context

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


# ------------------------------------------------------------------ requirements
@pytest.mark.parametrize(
    ("version", "specifier", "ok"),
    [
        ("2.5.1+cpu", ">=2.2", True),
        ("2.5.1+cpu", ">=2.2,<2.5", False),
        ("2.16.0", ">=2.16", True),
        ("2.15.1", ">=2.16", False),
        ("1.4.2", "~=1.4", True),
        ("2.0.0", "~=1.4", False),
        ("3.0", "==3", True),
        ("3.0.1", "!=3.0.1", False),
        ("2.0.0rc1", ">=2", True),
    ],
)
def test_version_specifiers(version: str, specifier: str, ok: bool) -> None:
    assert check_specifier(version, specifier) is ok


def test_missing_or_old_library_is_a_capability_error() -> None:
    assert installed_version("numpy") is not None
    require({"numpy": ">=1"})
    with pytest.raises(UnsupportedCapabilityError, match="não está instalado"):
        require({"surely_not_a_package_xyz": ">=1"})
    with pytest.raises(UnsupportedCapabilityError, match="pip install"):
        require({"numpy": ">=999"})
    with pytest.raises(ConfigurationError, match="inválida"):
        check_specifier("1.0", "at least 1")


# ------------------------------------------------------------------ adapters
def drift_fit(y: np.ndarray, damping: float = 1.0) -> dict[str, float]:
    return {"slope": damping * float(y[-1] - y[0]) / (len(y) - 1)}


def drift_predict(
    state: dict[str, float], horizon: int, history: np.ndarray, damping: float = 1.0
) -> np.ndarray:
    return history[-1] + state["slope"] * np.arange(1, horizon + 1)


def test_custom_forecaster_from_two_functions() -> None:
    y = np.arange(50, dtype=float)
    model = CustomForecaster(drift_fit, drift_predict, params={"damping": 0.5}).fit(y)
    np.testing.assert_allclose(model.predict(3, local_context(y, 50)), [49.5, 50.0, 50.5])
    # A longer context moves the inputs, never the fitted state.
    longer = np.arange(60, dtype=float)
    assert model.predict(1, local_context(longer, 60))[0] == pytest.approx(59.5)
    clone = model.clone()
    assert clone.get_params()["params"] == {"damping": 0.5}
    assert not clone.is_fitted


def test_regressor_forecaster_wraps_any_fit_predict_estimator() -> None:
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(size=120)) + 50
    model = RegressorForecaster(Ridge(alpha=1.0), lags=7, differencing=1).fit(y)
    path = model.predict(5, local_context(y, 120))
    assert path.shape == (5,)
    assert np.isfinite(path).all()
    with pytest.raises(ConfigurationError, match="fit"):
        RegressorForecaster(object())


# ------------------------------------------------------------------ registration
def test_registered_model_is_used_everywhere_by_name() -> None:
    y = get_data("synthetic_daily", n=260)
    exp = AutoHybridExperiment(
        fh=7,
        fold=2,
        budget="smoke",
        base_models=("naive", "drift"),
        correctors=("linear",),
        residual_windows=(7,),
        alphas=(1.0,),
    )
    exp.register_model(
        "drift",
        lambda p: CustomForecaster(drift_fit, drift_predict, params=dict(p)),
        family="custom",
        version="1",
        tuning_space={"damping": (0.1, 1.0)},
    )
    exp.fit(y)
    assert "drift" in exp.models().loc["base"].index
    assert exp.create_model("drift", damping=0.8).evaluation.coverage("MASE") == 1.0
    hybrid = exp.create_hybrid("drift", "linear")
    assert hybrid.evaluation.coverage("MASE") == 1.0
    result = exp.compare_models()
    assert "drift" in {e.label.split("(")[0] for e in result.search.ledger}
    # The new family enters the chain right before the residual hybrids.
    order = exp.policy_.family_order
    assert order.index("custom") == order.index("residual") - 1
    # tune_model uses the space declared at registration.
    _, tuner = exp.tune_model("drift", n_trials=3, sampler="random", return_tuner=True)
    assert "damping" in tuner.trials.columns
    # The registration enters the evaluator protocol, hence every trial key.
    assert exp.evaluator_.protocol()["external_models"]["base:drift"]["version"] == "1"
    assert "external_models" not in AutoHybridExperiment().fit(y).evaluator_.protocol()


def test_unmet_requirement_skips_the_trial_and_explains() -> None:
    y = get_data("synthetic_daily", n=200)
    exp = AutoHybridExperiment(fh=7, fold=1).fit(y)
    exp.register_model(
        "future",
        lambda p: CustomForecaster(drift_fit, drift_predict),
        requires={"numpy": ">=999"},
    )
    result = exp.create_model("future")
    assert {t.status for t in result.evaluation.trials} == {"SKIPPED"}
    assert "numpy>=999" in (result.evaluation.trials[0].reason or "")


def test_register_an_instance_and_a_corrector() -> None:
    exp = AutoHybridExperiment(fh=7, fold=1)
    exp.register_model("ridge", RegressorForecaster(Ridge(), lags=7), family="linear")
    model = exp.registry.make_base("ridge", {"lags": 14})
    assert model.get_params()["lags"] == 14
    exp.register_model(
        "torch_lstm", torch_model(object, kind="corrector"), kind="corrector", family="x"
    )  # factories do not import the framework: registering never needs it
    assert "torch_lstm" in exp.registry
    with pytest.raises(ConfigurationError, match="kind"):
        exp.register_model("bad", lambda p: None, kind="model")


def test_registry_deep_copy_is_independent() -> None:
    registry = default_registry()
    clone = copy.deepcopy(registry)
    clone.register_base(
        "drift", lambda p: CustomForecaster(drift_fit, drift_predict), family="custom"
    )
    assert "drift" in clone
    assert "drift" not in registry
