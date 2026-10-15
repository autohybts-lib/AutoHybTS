"""Public thresholds, profile inheritance and reproducible configuration."""

from __future__ import annotations

import json

import numpy as np
import pytest

from autohybts import AutoHybridExperiment
from autohybts.benchmarks.ablation import _study_config
from autohybts.datasets import get_data
from autohybts.exceptions import ConfigurationError, ProtocolConfigurationError
from autohybts.selection.policy import SelectionPolicy


@pytest.mark.parametrize("profile", ["standard", "research_strict", "quickstart", "exploratory"])
def test_default_policy_content_is_unchanged(profile):
    expected = (
        SelectionPolicy.quickstart()
        if profile == "quickstart"
        else SelectionPolicy(profile=profile)
    )
    assert (
        AutoHybridExperiment(profile=profile)._selection_policy().describe() == expected.describe()
    )


@pytest.mark.parametrize("profile", ["standard", "quickstart"])
def test_explicit_variability_threshold_enables_the_criterion(profile):
    policy = AutoHybridExperiment(profile=profile, variability_threshold=0.2)._selection_policy()
    assert policy.variability.threshold == 0.2


@pytest.mark.parametrize(
    "name",
    [
        "median_gain_threshold",
        "win_rate_threshold",
        "variability_threshold",
        "min_coverage",
        "epsilon",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, "0.1", -0.1])
def test_invalid_public_thresholds_fail_during_config_validation(name, value):
    with pytest.raises(ConfigurationError, match=name):
        _study_config({"space": {name: value}})


@pytest.mark.parametrize(
    ("name", "value"),
    [("win_rate_threshold", 1.1), ("min_coverage", 1.1), ("min_coverage", 0), ("epsilon", 0)],
)
def test_threshold_ranges(name, value):
    with pytest.raises(ConfigurationError, match=name):
        AutoHybridExperiment(**{name: value})._selection_policy()


def test_research_strict_cannot_reduce_coverage():
    with pytest.raises(ProtocolConfigurationError, match="cobertura"):
        AutoHybridExperiment(profile="research_strict", min_coverage=0.9)._selection_policy()


def test_custom_policy_reaches_snapshot_clone_and_json_without_reading_test():
    settings = dict(
        fh=7,
        fold=2,
        budget="smoke",
        base_models=["naive", "linear"],
        correctors=["linear"],
        median_gain_threshold=0.1,
        win_rate_threshold=0.8,
        variability_threshold=0.2,
        min_coverage=1.0,
        epsilon=1e-8,
    )
    exp = AutoHybridExperiment(**settings).fit(get_data("synthetic_daily", n=240))
    result = exp.compare_models()
    policy = result.snapshot["policy"]
    assert policy["median_gain_threshold"] == 0.1
    assert policy["win_rate_threshold"] == 0.8
    assert policy["variability"]["threshold"] == 0.2
    assert policy["epsilon"] == 1e-8
    assert not exp._externally_evaluated
    nested = exp.nested_cross_validate(outer_folds=1)
    assert len(nested.snapshots) == 1
    assert nested.snapshots[0]["snapshot"]["policy"]["median_gain_threshold"] == 0.1
    assert nested.snapshots[0]["snapshot"]["policy"]["variability"]["threshold"] == 0.2
    assert not exp._externally_evaluated
    restored = AutoHybridExperiment.from_snapshot(json.loads(json.dumps(result.snapshot)))
    assert json.loads(json.dumps(restored.get_params())) == json.loads(
        json.dumps(exp.clone().get_params())
    )
    assert restored._selection_policy().describe() == policy
    config = _study_config(
        {
            "space": {
                key: settings[key]
                for key in (
                    "median_gain_threshold",
                    "win_rate_threshold",
                    "variability_threshold",
                    "min_coverage",
                    "epsilon",
                )
            }
        }
    )
    assert config["space"]["median_gain_threshold"] == 0.1


def test_gain_threshold_changes_a_real_decision_and_protocol_identity():
    settings = dict(
        fh=7, fold=2, budget="smoke", base_models=["naive", "linear"], correctors=["linear"]
    )
    y = get_data("synthetic_daily", n=240)
    low = AutoHybridExperiment(**settings, median_gain_threshold=0).fit(y).compare_models()
    high = AutoHybridExperiment(**settings, median_gain_threshold=2).fit(y).compare_models()
    assert high.decisions["accepted"].eq(False).all()
    assert high.audit.protocol_hash != low.audit.protocol_hash
    assert high.audit.search_protocol_hash == low.audit.search_protocol_hash
    assert np.isfinite(high.selection_leaderboard["MASE"]).all()
