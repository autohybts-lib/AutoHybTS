"""Dynamic selection of the corrector (variant B only): DSEL, dref, local (I13, T16)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from autohybts import AutoHybridExperiment
from autohybts.adapters.naive import SeasonalNaiveForecaster
from autohybts.datasets import get_data
from autohybts.domain.specs import CandidateSpec, ModelSpec
from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError
from autohybts.registry import default_registry
from autohybts.residuals.correctors import SklearnCorrector
from autohybts.residuals.generation import local_context
from autohybts.search.spaces import SearchSpace
from autohybts.strategies.dynamic import DynamicSelectionForecaster

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

SEASONAL = ModelSpec.of("seasonal_naive", {"season_length": 7})
LINEAR = ModelSpec.of("linear")


def values(n: int = 300, phi: float = 0.8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + rng.normal()
    return 20 + 3 * np.sin(2 * np.pi * np.arange(n) / 7) + noise


def model(method: str = "dref", **kwargs: object) -> DynamicSelectionForecaster:
    pool = [SklearnCorrector(), SklearnCorrector("knn")]
    return DynamicSelectionForecaster(
        SeasonalNaiveForecaster(season_length=7),
        pool,
        method=method,
        **kwargs,  # type: ignore[arg-type]
    )


def spec(variant: str = "B", **options: object) -> CandidateSpec:
    return CandidateSpec(
        "dynamic",
        "dynamic_selection",
        SEASONAL,
        LINEAR,
        7,
        1.0,
        "historical_oof_1step",
        variant,  # type: ignore[arg-type]
        members=(ModelSpec.of("knn"),),
        options_json=json.dumps({"method": "dref", **options}),
    )


def test_dynamic_selection_exists_only_in_variant_b() -> None:
    # T16: a configuration error in variant A, at every entry point.
    with pytest.raises(ConfigurationError, match="variante B"):
        spec("A")
    with pytest.raises(ConfigurationError, match="variante B"):
        SearchSpace(dynamic_methods=("dref",), variant="A")
    exp = AutoHybridExperiment(fh=7, fold=1, budget="smoke").fit(get_data("synthetic_daily", n=200))
    with pytest.raises(ConfigurationError, match="variante B"):
        exp.compare_models(include=("standalone", "residual", "dynamic"))
    y = values()
    fitted = model().fit(y)
    with pytest.raises(UnsupportedCapabilityError, match="um passo"):
        fitted.predict(2, local_context(y, y.size))


def test_spec_names_and_options() -> None:
    dynamic = spec()
    assert dynamic.name == "dref(seasonal_naive-linear,knn)"
    assert dynamic.label.startswith("dref: seasonal_naive(season_length=7) + {linear | knn}")
    with pytest.raises(ConfigurationError, match="dref"):
        spec(method="vote")
    with pytest.raises(ConfigurationError, match="Opções"):
        spec(gate=True)


def test_dsel_comes_from_the_training_tail_only() -> None:
    y = values()
    fitted = model(dsel_size=40).fit(y[:250])
    assert fitted.dsel_errors_.shape == (40, 3)  # none + two correctors
    assert fitted.library_.shape == (40, 7)
    # The DSEL targets are inside the training vector, before its end.
    positions = fitted.residuals_.targets[-40:]
    np.testing.assert_allclose(fitted.dsel_target_, y[positions])
    assert positions.max() == 249
    # Column 0 is "no correction": its error is the residual itself.
    np.testing.assert_allclose(fitted.dsel_errors_[:, 0], np.abs(fitted.residuals_.residuals[-40:]))
    assert set(fitted.diagnostics()) == {"correction_std_ratio", "best_static", "dsel_size"}


@pytest.mark.parametrize("method", ["dref", "local"])
def test_components_add_up_and_choices_are_in_the_pool(method: str) -> None:
    y = values()
    fitted = model(method).fit(y[:250])
    for end in (250, 260, 275):
        context = local_context(y, end)
        prediction = fitted.predict(1, context)[0]
        base, correction = fitted.predict_components(1, y[:end])
        assert prediction == pytest.approx(base[0] + correction[0])
        assert 0 <= fitted.choice_at(y[:end]) <= 2
    # Predicting twice from the same context gives the same answer (no hidden state).
    assert (
        fitted.predict(1, local_context(y, 260))[0] == fitted.predict(1, local_context(y, 260))[0]
    )


def test_dref_can_decide_not_to_correct() -> None:
    # White noise around a seasonal pattern: residuals are unpredictable, so "none" wins
    # on part of the neighbourhoods.
    y = values(phi=0.0, seed=3)
    fitted = model(neighbors=5).fit(y[:250])
    choices = [fitted.choice_at(y[:end]) for end in range(250, 300)]
    assert 0 in choices


def test_search_proposes_dynamic_selection_in_s4() -> None:
    space = SearchSpace(
        base_models=("seasonal_naive",),
        correctors=("linear", "knn"),
        residual_windows=(7,),
        seasonal_period=7,
        variant="B",
        dynamic_methods=("dref", "local"),
    )
    best = CandidateSpec(
        "residual", "residual_additive", SEASONAL, LINEAR, 7, 1.0, "historical_oof_1step", "B"
    )
    proposals = space.dynamic_proposals([best], default_registry())
    assert [p.candidate.name for p in proposals] == [
        "dref(seasonal_naive-linear,knn)",
        "local(seasonal_naive-linear)",
    ]
    assert {p.stage for p in proposals} == {"S4"}
    assert space.describe()["dynamic_methods"] == ["dref", "local"]


def test_variant_b_experiment_with_dynamic_family_and_stream() -> None:
    y = get_data("synthetic_daily", n=300)
    exp = AutoHybridExperiment(
        fh=7,
        fold=2,
        variant="B",
        base_models=("seasonal_naive", "linear"),
        correctors=("linear", "knn"),
        residual_windows=(7,),
        alphas=(1.0,),
    ).fit(y)
    result = exp.compare_models(include=("standalone", "residual", "dynamic"))
    evaluated = {e.strategy for e in result.search.ledger if e.status == "EVALUATED"}
    assert "dynamic_selection" in evaluated
    assert "dynamic" in set(result.decisions["family"])
    dynamic = next(
        r.candidate for r in result.search.results if r.candidate.strategy == "dynamic_selection"
    )
    final = exp.finalize_model(dynamic)
    stream = final.stream()
    ticket = stream.predict_next()
    assert ticket.prediction == pytest.approx(ticket.base_prediction + ticket.correction)
    stream.observe(float(y.iloc[-1]))
    assert len(stream.log()) == 1
    assert isinstance(stream.log(), pd.DataFrame)
