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
    LaggedModelForecaster,
    RegressorForecaster,
    WindowCorrector,
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
        # stable releases, compound bounds, compatible release, equality with padding
        ("2.16.0", ">=2.16", True),
        ("2.15.1", ">=2.16", False),
        ("1.4.2", "~=1.4", True),
        ("2.0.0", "~=1.4", False),
        ("3.0", "==3", True),
        ("3.0.1", "!=3.0.1", False),
        # local versions: ignored by a specifier without a local label
        ("2.5.1+cpu", ">=2.2", True),
        ("2.5.1+cpu", ">=2.2,<2.5", False),
        ("2.5.1+cu130", "==2.5.1", True),
        ("2.5.1+cpu", "==2.5.1+cu130", False),
        # pre-releases come before the release (PEP 440); inside the range they count
        ("2.0.0rc1", ">=2", False),
        ("2.2.0rc1", ">=2.2", False),
        ("2.2.0rc1", "==2.2", False),
        ("2.3.0rc1", ">=2.2", True),
        ("2.6.0.dev20241001", ">=2.2,<3", True),
        ("2.0.0rc1", ">=2.0.0rc1", True),
        # post-releases: not equal to the release, matched by a wildcard
        ("2.2.0.post1", "==2.2", False),
        ("2.2.0.post1", "==2.2.*", True),
        ("2.2.0.post1", ">=2.2", True),
        # wildcard exclusion
        ("2.2.3", "!=2.2.*", False),
        ("2.3.0", "!=2.2.*", True),
    ],
)
def test_version_specifiers_follow_pep_440(version: str, specifier: str, ok: bool) -> None:
    assert check_specifier(version, specifier) is ok


@pytest.mark.parametrize("specifier", ["at least 1", "=>2", "==2.*.1", ">>1"])
def test_invalid_specifiers_are_configuration_errors(specifier: str) -> None:
    with pytest.raises(ConfigurationError, match="inválida"):
        check_specifier("1.0", specifier)


def test_installed_version_outside_pep_440_cannot_be_checked() -> None:
    with pytest.raises(UnsupportedCapabilityError, match="PEP 440"):
        check_specifier("1.0-beta-custom_build", ">=1")


def test_missing_or_old_library_is_a_capability_error() -> None:
    assert installed_version("numpy") is not None
    require({"numpy": ">=1"})
    require({"numpy": ""})  # any version
    with pytest.raises(UnsupportedCapabilityError, match="não está instalado"):
        require({"surely_not_a_package_xyz": ">=1"})
    with pytest.raises(UnsupportedCapabilityError, match="pip install"):
        require({"numpy": ">=999"})
    with pytest.raises(ConfigurationError, match="inválida"):
        require({"numpy": "at least 1"})


def test_malformed_requirements_are_refused_at_registration() -> None:
    registry = default_registry()
    factory = lambda p: CustomForecaster(drift_fit, drift_predict)  # noqa: E731
    with pytest.raises(ConfigurationError, match="inválida"):
        registry.register_base("drift", factory, family="custom", requires={"numpy": "=>1"})
    with pytest.raises(ConfigurationError, match="requires"):
        registry.register_base("drift", factory, family="custom", requires={"numpy": 1})  # type: ignore[dict-item]
    with pytest.raises(ConfigurationError, match="inválida"):
        registry.register_corrector(
            "ridge_c",
            lambda p: copy.deepcopy(default_registry().make_corrector("linear")),
            requires={"sklearn": "1.4+"},
        )
    # A valid requirement that is not met is only checked when the model is built.
    registry.register_base("drift", factory, family="custom", requires={"numpy": ">=999"})
    with pytest.raises(UnsupportedCapabilityError):
        registry.make_base("drift")


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


# ------------------------------------------------------------------ internal validation (M01)
class RecordingForecaster(LaggedModelForecaster):
    """Framework-free wrapper: records what would be handed to the library."""

    def _train(self, x_train, y_train, x_val, y_val):  # type: ignore[no-untyped-def]
        self.seen_ = (x_train.copy(), y_train.copy(), x_val, y_val)
        return {"epochs_run": 1.0}

    def _infer(self, x):  # type: ignore[no-untyped-def]
        return np.zeros(len(x))


class RecordingCorrector(WindowCorrector):
    def _train(self, x_train, y_train, x_val, y_val):  # type: ignore[no-untyped-def]
        self.seen_ = (x_train.copy(), y_train.copy(), x_val, y_val)
        return {}

    def _infer(self, x):  # type: ignore[no-untyped-def]
        return np.zeros(len(x))


def trend(n: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 10 + 0.5 * np.arange(n) + rng.normal(0, 1, n)


@pytest.mark.parametrize("differencing", [0, 1])
@pytest.mark.parametrize("scale", [True, False])
def test_validation_tail_does_not_shape_internal_training(differencing: int, scale: bool) -> None:
    y = trend()
    lags = 7
    model = RecordingForecaster(lags=lags, differencing=differencing, scale=scale).fit(y)
    rows = len(y) - differencing - lags
    cut = rows - round(rows * 0.2)
    assert (model.train_rows_, model.validation_rows_) == (cut, rows - cut)
    # Values that only validation targets use: level[cut + lags:] -> y[cut + lags + d:].
    perturbed = y.copy()
    perturbed[cut + lags + differencing :] += 1000.0
    other = RecordingForecaster(lags=lags, differencing=differencing, scale=scale).fit(perturbed)
    assert (other.center_, other.spread_) == (model.center_, model.spread_)
    np.testing.assert_array_equal(other.seen_[0], model.seen_[0])
    np.testing.assert_array_equal(other.seen_[1], model.seen_[1])
    assert not np.array_equal(other.seen_[3], model.seen_[3])  # the tail itself changed
    if not scale:
        assert (model.center_, model.spread_) == (0.0, 1.0)
    diagnostics = model.diagnostics()
    assert diagnostics["train_rows"] == cut
    assert diagnostics["validation_rows"] == rows - cut


def test_without_validation_the_whole_training_vector_is_used() -> None:
    y = trend()
    model = RecordingForecaster(lags=7, validation_fraction=0.0).fit(y)
    assert model.center_ == pytest.approx(float(np.mean(y)))
    assert model.validation_rows_ == 0
    assert model.seen_[2] is None
    # Short history: 11 rows, 10 % would hold out 1 row (< 2), so nothing is held out.
    short_y = trend(14)
    short = RecordingForecaster(lags=3, validation_fraction=0.1).fit(short_y)
    assert short.validation_rows_ == 0
    assert short.center_ == pytest.approx(float(np.mean(short_y)))


def test_constant_series_and_forecasts_stay_finite() -> None:
    model = RecordingForecaster(lags=7).fit(np.full(80, 3.0))
    assert model.spread_ == 1.0
    path = model.predict(5, local_context(np.full(90, 3.0), 85))
    assert path.shape == (5,)
    assert np.isfinite(path).all()
    np.testing.assert_allclose(path, 3.0)  # zero in standardized scale -> the mean


def test_corrector_scales_with_internal_training_residuals_only() -> None:
    rng = np.random.default_rng(3)
    residuals = rng.normal(0, 1, 80)
    windows = np.column_stack([np.roll(residuals, k) for k in range(5, 0, -1)])[5:]
    targets = residuals[5:]
    corrector = RecordingCorrector().fit(windows, targets)
    cut = len(targets) - round(len(targets) * 0.2)
    assert corrector.center_ == pytest.approx(float(np.mean(targets[:cut])))
    shifted = targets.copy()
    shifted[cut:] += 50.0
    other = RecordingCorrector().fit(windows, shifted)
    assert (other.center_, other.spread_) == (corrector.center_, corrector.spread_)
    np.testing.assert_array_equal(other.seen_[1], corrector.seen_[1])
    assert np.isfinite(corrector.predict_next(windows[-1]))


# ------------------------------------------------------------------ optional libraries (M07b)
def test_missing_framework_is_skipped_with_the_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import autohybts.models.requirements as requirements

    monkeypatch.setattr(requirements, "installed_version", lambda package: None)
    model = torch_model(object)({})  # the module is never built: the check comes first
    with pytest.raises(UnsupportedCapabilityError, match=r'pip install "torch>=2.2"'):
        model.fit(np.arange(60.0))


def test_version_check_without_packaging_explains_how_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "packaging.specifiers", None)
    with pytest.raises(UnsupportedCapabilityError, match="packaging"):
        require({"numpy": ">=1"})
    require({"numpy": ""})  # no specifier: nothing to compare, packaging not needed


def test_registered_model_reaches_the_nested_inner_searches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp = AutoHybridExperiment(
        fh=7,
        fold=1,
        budget="smoke",
        base_models=("naive", "drift"),
        correctors=("linear",),
        residual_windows=(7,),
        alphas=(1.0,),
    )
    exp.register_model(
        "drift", lambda p: CustomForecaster(drift_fit, drift_predict), family="custom", version="7"
    )
    exp.fit(get_data("synthetic_daily", n=240))
    searched: list[set[str]] = []
    original = AutoHybridExperiment.compare_models

    def spy(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = original(self, *args, **kwargs)
        searched.append({e.label.split("(")[0] for e in result.search.ledger})
        assert self.evaluator_.protocol()["external_models"]["base:drift"]["version"] == "7"
        return result

    monkeypatch.setattr(AutoHybridExperiment, "compare_models", spy)
    nested = exp.nested_cross_validate(outer_folds=2, baselines=())
    assert len(searched) == 2
    assert all("drift" in labels for labels in searched)
    assert (nested.cost["status"] == "SUCCEEDED").all()


def test_differenced_forecasts_return_to_the_original_scale() -> None:
    # A network predicting 0 (standardized) adds the training mean of the differences.
    y = trend()
    model = RecordingForecaster(lags=7, differencing=1).fit(y)
    rows = len(y) - 1 - 7
    cut = rows - round(rows * 0.2)
    assert model.center_ == pytest.approx(float(np.mean(np.diff(y)[: cut + 7])))
    path = model.predict(5, local_context(y, len(y)))
    np.testing.assert_allclose(path, y[-1] + model.center_ * np.arange(1, 6))


def test_corrector_without_internal_validation_uses_every_row() -> None:
    rng = np.random.default_rng(4)
    windows = rng.normal(size=(40, 5))
    targets = rng.normal(size=40)
    corrector = RecordingCorrector(validation_fraction=0.0).fit(windows, targets)
    assert corrector.center_ == pytest.approx(float(np.mean(targets)))
    assert corrector.seen_[2] is None
    assert len(corrector.seen_[1]) == 40


class NewestLag(RecordingForecaster):
    def _infer(self, x):  # type: ignore[no-untyped-def]
        return x[:, -1]  # windows are oldest first: the last column is the newest lag


class OldestLag(RecordingForecaster):
    def _infer(self, x):  # type: ignore[no-untyped-def]
        return x[:, 0]


def test_recursive_forecasts_feed_lags_in_time_order() -> None:
    y = trend(80)
    context = local_context(y, len(y))
    # Newest lag repeated: a naive forecast.
    np.testing.assert_allclose(NewestLag(lags=3).fit(y).predict(4, context), [y[-1]] * 4)
    # Oldest of 3 lags, each prediction appended as the newest: y[-3], y[-2], y[-1], y[-3].
    np.testing.assert_allclose(
        OldestLag(lags=3).fit(y).predict(5, context), [y[-3], y[-2], y[-1], y[-3], y[-2]]
    )
