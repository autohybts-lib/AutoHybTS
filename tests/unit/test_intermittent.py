"""Intermittent-demand methods (plano de ensembles L08) and the RMSSE metric."""

from __future__ import annotations

import numpy as np
import pytest

from autohybts import AutoHybridExperiment
from autohybts.adapters.intermittent import (
    AdidaForecaster,
    CrostonSBAForecaster,
    ImapaForecaster,
    TSBForecaster,
    demand_profile,
)
from autohybts.datasets import get_data
from autohybts.evaluation.metrics import METRICS
from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError
from autohybts.registry import default_registry
from autohybts.residuals.generation import local_context
from autohybts.search.spaces import DEFAULT_BASES

# Demands at positions 2, 4 and 8: sizes 3, 5, 2; intervals 3, 2, 4 (mean 3).
Y = np.array([0.0, 0.0, 3.0, 0.0, 5.0, 0.0, 0.0, 0.0, 2.0])
METHODS = ("croston_sba", "tsb", "adida", "imapa")


def forecast(model, values: np.ndarray, horizon: int = 3) -> np.ndarray:  # type: ignore[no-untyped-def]
    return model.predict(horizon, local_context(values, values.size))


def test_croston_sba_smooths_sizes_and_intervals() -> None:
    # sizes 3 -> 4 -> 3, intervals 3 -> 2.5 -> 3.25 (alpha 0.5); SBA factor 1 - 0.5 / 2
    model = CrostonSBAForecaster(alpha=0.5).fit(Y)
    np.testing.assert_allclose(forecast(model, Y), [0.75 * 3 / 3.25] * 3)


def test_tsb_decays_when_demand_stops() -> None:
    model = TSBForecaster(alpha_d=0.5, alpha_p=0.5).fit(Y)
    # probability 0.5390625 after the last demand, size 3 (worked by hand)
    np.testing.assert_allclose(forecast(model, Y), [0.5390625 * 3] * 3)
    quiet = np.append(Y, np.zeros(5))
    later = forecast(model, quiet)[0]
    assert 0 < later < forecast(model, Y)[0]
    # Croston keeps the last rate: it only changes when a demand happens.
    croston = CrostonSBAForecaster(alpha=0.5).fit(Y)
    np.testing.assert_allclose(forecast(croston, quiet), forecast(croston, Y))


def test_adida_and_imapa_aggregate_at_the_learned_level() -> None:
    adida = AdidaForecaster(alpha=0.5).fit(Y)
    assert adida.aggregation_ == 3  # mean interval of the training vector
    # buckets of 3 ending at the last value: 3, 5, 2 -> level 3 -> 3 / 3 per period
    np.testing.assert_allclose(forecast(adida, Y), [1.0] * 3)
    imapa = ImapaForecaster(alpha=0.5).fit(Y)
    # levels 1, 2, 3: 1.1796875, 2 / 2, 3 / 3
    np.testing.assert_allclose(forecast(imapa, Y), [(1.1796875 + 1 + 1) / 3] * 3)


def test_aggregation_level_is_frozen_after_fit() -> None:
    model = AdidaForecaster().fit(Y)
    dense = np.append(Y, np.full(30, 4.0))  # demand every day afterwards
    forecast(model, dense)
    assert model.aggregation_ == 3


@pytest.mark.parametrize("name", METHODS)
def test_forecasts_are_flat_and_never_negative(name: str) -> None:
    model = default_registry().make_base(name).fit(Y)
    values = forecast(model, Y, horizon=5)
    assert (values >= 0).all()
    assert np.ptp(values) == 0
    np.testing.assert_array_equal(forecast(model, np.zeros(9)), np.zeros(3))


@pytest.mark.parametrize("name", METHODS)
def test_negative_values_are_not_demand(name: str) -> None:
    with pytest.raises(UnsupportedCapabilityError, match="demanda"):
        default_registry().make_base(name).fit(np.array([1.0, -1.0, 0.0, 2.0]))


def test_negative_series_are_skipped_by_the_evaluator() -> None:
    y = get_data("synthetic_daily", n=200) - 60  # mostly negative
    exp = AutoHybridExperiment(fh=7, fold=1).fit(y)
    trial = exp.create_model("croston_sba").evaluation.trials[0]
    assert trial.status == "SKIPPED"


@pytest.mark.parametrize(
    "build",
    [
        lambda: CrostonSBAForecaster(alpha=0.0),
        lambda: CrostonSBAForecaster(alpha=1.5),
        lambda: TSBForecaster(alpha_p=True),
        lambda: AdidaForecaster(alpha="0.1"),
    ],
)
def test_invalid_smoothing_constants(build) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ConfigurationError):
        build()


def test_demand_profile() -> None:
    profile = demand_profile(Y)
    assert profile["zero_share"] == pytest.approx(6 / 9)
    assert profile["adi"] == pytest.approx(3.0)
    assert profile["cv2"] == pytest.approx(np.var([3, 5, 2]) / np.mean([3, 5, 2]) ** 2)
    assert np.isnan(demand_profile(np.zeros(5))["adi"])
    model = CrostonSBAForecaster().fit(np.zeros(5))
    assert "adi" not in model.diagnostics()  # undefined values are left out, not NaN


def test_rmsse_definition() -> None:
    train = np.array([1.0, 3.0, 2.0, 4.0])
    actual, predicted = np.array([2.0, 4.0]), np.array([3.0, 3.0])
    scale = np.mean(np.diff(train) ** 2)  # m = 1
    assert METRICS["RMSSE"](actual, predicted, train) == pytest.approx(np.sqrt(1.0 / scale))
    assert np.isnan(METRICS["RMSSE"](actual, predicted, np.ones(4)))
    assert np.isnan(METRICS["RMSSE"](actual, predicted, train, seasonal_period=4))


def test_family_joins_the_chain_only_when_searched() -> None:
    registry = default_registry()
    assert set(METHODS) == set(registry.names("base", "intermittent"))
    assert not set(METHODS) & set(DEFAULT_BASES)
    y = get_data("intermittent_daily", n=200)
    plain = AutoHybridExperiment(fh=7, fold=1, budget="smoke").fit(y)
    plain.compare_models()
    assert "intermittent" not in plain.policy_.family_order  # default protocol unchanged
    searched = AutoHybridExperiment(fh=7, fold=1, budget="smoke", base_models=["naive", "tsb"]).fit(
        y
    )
    result = searched.compare_models()
    order = searched.policy_.family_order
    assert order.index("intermittent") == order.index("residual") - 1
    assert "intermittent" in set(result.decisions["family"])
