"""Tuning distributions and how sampled values reach the base parameters."""

from __future__ import annotations

import numpy as np
import pytest

from autohybts.domain.specs import CandidateSpec
from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError
from autohybts.registry import default_registry
from autohybts.search.tuning import (
    LAG_SPACES,
    PLAIN_SPACES,
    TUNING_SPACES,
    Distribution,
    Tuner,
    _lookup,
    _with_base_params,
    default_space,
)


def test_distributions_sample_inside_their_bounds() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert 2 <= Distribution("int", 2, 5).sample(rng) <= 5
        assert 0.5 <= Distribution("float", 0.5, 1.5).sample(rng) <= 1.5
        assert 1e-3 <= Distribution("log_float", 1e-3, 1e2).sample(rng) <= 1e2
        assert Distribution("categorical", choices=("a", "b")).sample(rng) in ("a", "b")
    with pytest.raises(ConfigurationError):
        Distribution("int", 5, 2)
    with pytest.raises(ConfigurationError):
        Distribution("log_float", 0.0, 1.0)
    with pytest.raises(ConfigurationError):
        Distribution("categorical")


def test_default_spaces_cover_the_lag_models() -> None:
    registry = default_registry()
    for name in LAG_SPACES:
        space = default_space(name, seasonal_period=7)
        assert space["lags"].high == 28
        registry.make_base(name, {"lags": 7})
    assert default_space("linear", 30)["lags"].high == 60
    with pytest.raises(UnsupportedCapabilityError):
        default_space("arima", 7)


def test_models_without_lags_get_no_lags() -> None:
    for name in PLAIN_SPACES:
        assert "lags" not in default_space(name, seasonal_period=7)
    assert set(TUNING_SPACES) == set(LAG_SPACES) | set(PLAIN_SPACES)


OPTIONAL = {"lightgbm_regressor": "lightgbm", "xgboost_regressor": "xgboost"}
OPTIONAL["catboost_regressor"] = "catboost"
OPTIONAL["exponential_smoothing"] = "statsmodels"


@pytest.mark.parametrize("name", sorted(TUNING_SPACES))
def test_every_sampled_configuration_fits(name: str) -> None:
    # Every parameter name and range of a built-in space is accepted by the model.
    if name in OPTIONAL:
        pytest.importorskip(OPTIONAL[name])
    rng = np.random.default_rng(0)
    y = 20 + np.sin(np.arange(160) / 2) * 3 + rng.normal(0, 0.3, 160) + (np.arange(160) % 4 == 0)
    space = default_space(name, seasonal_period=7)
    base = CandidateSpec.standalone(name, family="any")
    for _ in range(2):
        sampled = {key: dist.sample(rng) for key, dist in space.items()}
        tuned = _with_base_params(base, sampled)
        model = default_registry().make_base(name, dict(tuned.base.params))
        model.fit(y)


def test_sampled_values_set_nested_parameters() -> None:
    mlp = CandidateSpec.standalone("mlp", {"lags": 7}, family="neural")
    tuned = _with_base_params(mlp, {"lags": 14, "estimator_params.hidden_layer_sizes": "32,16"})
    assert tuned.base.params == {"lags": 14, "estimator_params": {"hidden_layer_sizes": [32, 16]}}
    assert _lookup(tuned.base.params, "estimator_params.hidden_layer_sizes") == "32,16"
    assert _lookup(tuned.base.params, "estimator_params.alpha") is None
    explicit_none = CandidateSpec.standalone("knn", {"estimator_params": None}, family="kernel")
    knn = _with_base_params(explicit_none, {"estimator_params.n_neighbors": 3})
    assert knn.base.params["estimator_params"] == {"n_neighbors": 3}
    broken = CandidateSpec.standalone("knn", {"estimator_params": 5}, family="kernel")
    with pytest.raises(ConfigurationError):
        _with_base_params(broken, {"estimator_params.n_neighbors": 3})


def test_tuner_arguments_are_checked() -> None:
    evaluator = object()
    with pytest.raises(ConfigurationError):
        Tuner(evaluator, n_trials=0)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        Tuner(evaluator, sampler="grid")  # type: ignore[arg-type]
