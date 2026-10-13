from __future__ import annotations

import pytest

from autohybts.domain.identity import derive_seed, stable_hash
from autohybts.domain.specs import CandidateSpec, ModelSpec, trial_key
from autohybts.exceptions import ConfigurationError


def residual(**overrides: object) -> CandidateSpec:
    fields: dict[str, object] = {
        "family": "residual",
        "strategy": "residual_additive",
        "base": ModelSpec.of("arima"),
        "corrector": ModelSpec.of("mlp", {"lags": 7}),
        "window_size": 7,
        "alpha": 0.5,
        "residual_source": "historical_oof_1step",
    }
    fields.update(overrides)
    return CandidateSpec(**fields)


def test_model_spec_is_canonical_and_immutable() -> None:
    a = ModelSpec.of("mlp", {"lags": 7, "estimator_params": {"hidden_layer_sizes": (8,)}})
    b = ModelSpec.of("mlp", {"estimator_params": {"hidden_layer_sizes": [8]}, "lags": 7})
    assert a == b
    params = a.params
    params["lags"] = 99
    assert a.params["lags"] == 7


def test_candidate_id_is_pinned() -> None:
    # If this changes, every stored candidate_id and trial cache is invalidated.
    spec = CandidateSpec.standalone("linear", {"lags": 14}, family="linear")
    expected = {
        "schema": "candidate/1",
        "family": "linear",
        "strategy": "standalone",
        "base": {"name": "linear", "params": {"lags": 14}},
        "corrector": None,
        "window_size": None,
        "alpha": None,
        "residual_source": None,
        "variant": "A",
    }
    assert spec.candidate_id == stable_hash(expected)
    assert spec.candidate_id == "8214c4056db223af43c96ec3bdd2265ce0cd2cadaab05f11f0786d03b537b522"


@pytest.mark.parametrize(
    "change",
    [
        {"alpha": 1.0},
        {"window_size": 28},
        {"residual_source": "historical_oof_multistep"},
        {"corrector": ModelSpec.of("knn")},
        {"base": ModelSpec.of("linear")},
        {"variant": "B"},
        {"family": "other"},
    ],
)
def test_candidate_id_changes_with_any_field(change: dict[str, object]) -> None:
    assert residual(**change).candidate_id != residual().candidate_id


def test_standalone_rejects_residual_fields() -> None:
    with pytest.raises(ConfigurationError, match="standalone"):
        CandidateSpec("linear", "standalone", ModelSpec.of("linear"), alpha=1.0)


@pytest.mark.parametrize(
    "change",
    [{"alpha": None}, {"residual_source": "future"}, {"strategy": "magic"}, {"variant": "C"}],
)
def test_invalid_residual_candidates(change: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        residual(**change)


def test_labels() -> None:
    assert CandidateSpec.standalone("naive", family="statistical").label == "naive"
    assert residual().label == "arima + mlp(lags=7) [w=7, alpha=0.5, historical_oof_1step]"


def test_model_spec_rejects_non_object_params() -> None:
    with pytest.raises(ConfigurationError):
        ModelSpec("x", "[1, 2]")
    with pytest.raises(ConfigurationError):
        ModelSpec("", "{}")


def test_trial_key_depends_on_every_input() -> None:
    spec = CandidateSpec.standalone("naive", family="statistical")
    base = {"unit_id": "s/select-0@10", "seed": 1, "data_fingerprint": "d", "code_version": "0.1"}
    keys = {trial_key(spec, **base)}
    for name, value in [
        ("unit_id", "s/select-1@20"),
        ("seed", 2),
        ("data_fingerprint", "e"),
        ("code_version", "0.2"),
    ]:
        keys.add(trial_key(spec, **{**base, name: value}))
    keys.add(trial_key(CandidateSpec.standalone("linear", family="linear"), **base))
    assert len(keys) == 6


def test_derived_seeds_are_stable_and_distinct() -> None:
    first = derive_seed(42, "cand", "search-0")
    assert first == derive_seed(42, "cand", "search-0")
    seeds = {derive_seed(42, "cand", fold) for fold in ("search-0", "search-1", "select-0")}
    seeds.add(derive_seed(43, "cand", "search-0"))
    assert len(seeds) == 4
    assert all(0 <= seed < 2**31 for seed in seeds)
