"""Every registered base model honours the BaseForecaster contract (§5.2, §12.3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from autohybts.adapters.base import BaseForecaster
from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalPlan, TemporalSplitter
from autohybts.exceptions import (
    ConfigurationError,
    InsufficientHistoryError,
    UnsupportedCapabilityError,
)
from autohybts.registry import default_registry

REGISTRY = default_registry()
# Small settings keep the suite fast; behaviour is the same as with the defaults.
FAST_PARAMS: dict[str, dict[str, Any]] = {
    "arima": {"order": (1, 1, 1)},
    "random_forest": {"estimator_params": {"n_estimators": 10}},
    "mlp": {"estimator_params": {"hidden_layer_sizes": (8,), "max_iter": 300}},
}
BASES = REGISTRY.names("base")
HORIZON = 7
# The tiny MLP used for speed stops before converging; irrelevant to the contract.
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def series(n: int = 160, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 20 + 0.05 * t + 2 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.5, n)


def plan_for(values: np.ndarray) -> TemporalPlan:
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=len(values)), values)
    return TemporalSplitter(horizon=HORIZON, n_search_folds=1, n_select_folds=2).plan(frame)


def make(name: str) -> BaseForecaster:
    if name == "arima":
        pytest.importorskip("statsmodels")
    return REGISTRY.make_base(name, FAST_PARAMS.get(name))


@pytest.fixture(scope="module")
def plan() -> TemporalPlan:
    return plan_for(series())


@pytest.mark.parametrize("name", BASES)
def test_registered_model_satisfies_protocol(name: str) -> None:
    model = REGISTRY.make_base(name)
    assert isinstance(model, BaseForecaster)
    assert REGISTRY.make_base(name) is not model  # fresh instance each time
    if REGISTRY.entry(name).capabilities.in_sample_fitted:
        assert callable(getattr(model, "fitted_values", None))


@pytest.mark.parametrize("name", BASES)
def test_forecast_shape_dtype_and_finiteness(name: str, plan: TemporalPlan) -> None:
    origin = plan.select_origins[0]
    forecast = make(name).fit(plan.train_values(origin)).predict(HORIZON, plan.context(origin))
    assert forecast.shape == (HORIZON,)
    assert forecast.dtype == np.float64
    assert np.isfinite(forecast).all()


@pytest.mark.parametrize("name", BASES)
def test_predict_before_fit_is_an_error(name: str, plan: TemporalPlan) -> None:
    with pytest.raises(ConfigurationError, match="fit antes"):
        make(name).predict(HORIZON, plan.context(plan.select_origins[0]))


@pytest.mark.parametrize("name", BASES)
def test_refitting_is_deterministic(name: str, plan: TemporalPlan) -> None:
    origin = plan.select_origins[0]
    runs = [
        make(name).fit(plan.train_values(origin)).predict(HORIZON, plan.context(origin))
        for _ in range(2)
    ]
    assert np.array_equal(runs[0], runs[1])


@pytest.mark.parametrize("name", BASES)
def test_clone_shares_no_state(name: str, plan: TemporalPlan) -> None:
    origin = plan.select_origins[0]
    context = plan.context(origin)
    model = make(name).fit(plan.train_values(origin))
    before = model.predict(HORIZON, context)

    twin = model.clone()
    assert twin is not model
    assert twin.get_params() == model.get_params()
    with pytest.raises(ConfigurationError, match="fit antes"):
        twin.predict(HORIZON, context)  # clones start unfitted
    twin.fit(series(seed=1)[: origin.origin_index + 1] * 3.0)

    assert np.array_equal(model.predict(HORIZON, context), before)


@pytest.mark.parametrize("name", BASES)
def test_get_params_rebuilds_an_equivalent_model(name: str, plan: TemporalPlan) -> None:
    origin = plan.select_origins[0]
    model = make(name)
    rebuilt = REGISTRY.make_base(name, model.get_params())
    a = model.fit(plan.train_values(origin)).predict(HORIZON, plan.context(origin))
    b = rebuilt.fit(plan.train_values(origin)).predict(HORIZON, plan.context(origin))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", BASES)
def test_later_context_needs_no_refit(name: str, plan: TemporalPlan) -> None:
    # Frozen parameters + new observations: what variant B and OOF residuals rely on.
    first, later = plan.select_origins
    model = make(name).fit(plan.train_values(first))
    model._fit = _refuse_fit
    forecast = model.predict(HORIZON, plan.context(later))
    assert forecast.shape == (HORIZON,)
    assert REGISTRY.entry(name).capabilities.online_state_update


@pytest.mark.parametrize("name", BASES)
def test_inputs_are_not_mutated(name: str, plan: TemporalPlan) -> None:
    origin = plan.select_origins[0]
    train = np.array(plan.train_values(origin))
    snapshot = train.copy()
    make(name).fit(train).predict(HORIZON, plan.context(origin))
    assert np.array_equal(train, snapshot)


@pytest.mark.parametrize("name", BASES)
def test_too_short_training_is_reported(name: str) -> None:
    model = make(name)
    with pytest.raises(InsufficientHistoryError):
        model.fit(np.arange(float(model.min_train_size - 1)))


@pytest.mark.parametrize("name", BASES)
def test_exogenous_inputs_are_not_supported_yet(name: str, plan: TemporalPlan) -> None:
    with pytest.raises(UnsupportedCapabilityError):
        make(name).fit(plan.train_values(plan.select_origins[0]), X=np.ones(3))


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_invalid_horizon(horizon: object, plan: TemporalPlan) -> None:
    model = make("naive").fit(plan.train_values(plan.select_origins[0]))
    with pytest.raises(ConfigurationError, match="horizon"):
        model.predict(horizon, plan.context(plan.select_origins[0]))


def _refuse_fit(y: object) -> None:
    raise AssertionError("predict must not refit")


@pytest.mark.parametrize("name", BASES)
def test_one_step_path_matches_the_step_by_step_loop(name: str, plan: TemporalPlan) -> None:
    model = make(name)
    if not hasattr(model, "one_step_path"):
        pytest.skip("model has no batch one-step path")
    from autohybts.residuals.generation import local_context

    values = np.asarray(plan.history.values)
    model.fit(values[:100])
    fast = model.one_step_path(values, 100)
    slow = [model.predict(1, local_context(values, t))[0] for t in range(100, values.size)]
    assert np.allclose(fast, slow, rtol=0, atol=1e-9)
