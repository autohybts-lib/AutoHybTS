"""Decomposition, fixed combination and multistage: operators, specs and identities (I12b)."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from autohybts.adapters.naive import NaiveForecaster, SeasonalNaiveForecaster
from autohybts.adapters.sklearn_regression import LagRegressionForecaster
from autohybts.domain.specs import MAX_STAGES, CandidateSpec, ModelSpec
from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError
from autohybts.reporting.strategies import complexity as complexity_of
from autohybts.residuals.correctors import SklearnCorrector
from autohybts.residuals.generation import local_context
from autohybts.selection.rank import structure_key
from autohybts.strategies.additive_residual import ResidualHybridForecaster
from autohybts.strategies.combination import FixedCombinationForecaster
from autohybts.strategies.decomposition import (
    DecompositionForecaster,
    causal_moving_average,
    kurtosis_window,
)
from autohybts.strategies.multistage import stack_stages

needs_statsmodels = pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None, reason="extra `stats` not installed"
)

LINEAR = ModelSpec.of("linear", {"lags": 7})
SEASONAL = ModelSpec.of("seasonal_naive", {"season_length": 7})
CORRECTOR = ModelSpec.of("linear")
MLP_SPEC = ModelSpec.of("mlp")


def series(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 30 + 0.05 * t + 4 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1, n)


# ------------------------------------------------------------------ moving average
@given(
    arrays(np.float64, st.integers(1, 60), elements=st.floats(-1e3, 1e3)),
    st.integers(1, 30),
    st.integers(0, 59),
)
def test_moving_average_is_causal(values: np.ndarray, window: int, cut: int) -> None:
    # Prefix unchanged: the average up to t never depends on values after t.
    cut = min(cut, values.size - 1) + 1
    full = causal_moving_average(values, window)
    np.testing.assert_allclose(full[:cut], causal_moving_average(values[:cut], window))


def test_moving_average_values_and_expanding_start() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(causal_moving_average(y, 2), [1.0, 1.5, 2.5, 3.5, 4.5])
    np.testing.assert_allclose(causal_moving_average(y, 10), [1.0, 1.5, 2.0, 2.5, 3.0])
    with pytest.raises(ConfigurationError):
        causal_moving_average(y, 0)


def test_kurtosis_window_stays_in_range_and_uses_only_its_input() -> None:
    y = series(120)
    window = kurtosis_window(y, 365)
    assert 2 <= window <= 60
    assert kurtosis_window(y, 5) <= 5
    assert kurtosis_window(np.full(30, 4.0), 365) == 2  # constant: every window ties


# ------------------------------------------------------------------ decomposition
@pytest.mark.parametrize("method", ["ma", pytest.param("stl", marks=needs_statsmodels)])
def test_parts_sum_to_the_series(method: str) -> None:
    y = series()
    model = DecompositionForecaster(
        LagRegressionForecaster(lags=7), SklearnCorrector(), method=method
    ).fit(y)
    smooth, volatile = model.parts(y)
    np.testing.assert_allclose(smooth + volatile, y)
    path = model.predict(14, local_context(y, y.size))
    np.testing.assert_allclose(path, np.sum(model.predict_components(14, y), axis=0))


def test_ma_window_is_chosen_on_the_training_vector_only() -> None:
    y = series(220)
    first = DecompositionForecaster(LagRegressionForecaster(lags=7), SklearnCorrector()).fit(
        y[:150]
    )
    assert first.ma_window_ == kurtosis_window(y[:150], 365)
    assert first.diagnostics()["ma_window"] == first.ma_window_
    fixed = DecompositionForecaster(
        LagRegressionForecaster(lags=7), SklearnCorrector(), ma_window=5
    ).fit(y)
    assert fixed.ma_window_ == 5
    assert fixed.diagnostics()["correction_std_ratio"] >= 0


def test_ma_decomposition_takes_new_observations_without_refit() -> None:
    y = series(220)
    model = DecompositionForecaster(LagRegressionForecaster(lags=7), SklearnCorrector()).fit(
        y[:200]
    )
    later = model.predict(1, local_context(y, 210))
    smooth, volatile = model.parts(y[:210])
    expected = model.smooth_.predict(1, local_context(smooth, 210))[0] + float(
        model.volatile_.predict_next(volatile[-7:])
    )
    assert later[0] == pytest.approx(expected)


@needs_statsmodels
def test_stl_only_forecasts_from_the_end_of_its_training() -> None:
    y = series(220)
    model = DecompositionForecaster(
        LagRegressionForecaster(lags=7), SklearnCorrector(), method="stl"
    ).fit(y[:200])
    model.predict(7, local_context(y, 200))
    with pytest.raises(UnsupportedCapabilityError, match="STL"):
        model.predict(1, local_context(y, 205))


def test_decomposition_rejects_bad_arguments() -> None:
    smooth, volatile = NaiveForecaster(), SklearnCorrector()
    with pytest.raises(ConfigurationError):
        DecompositionForecaster(smooth, volatile, method="x13")
    with pytest.raises(ConfigurationError):
        DecompositionForecaster(smooth, volatile, ma_window=0)
    with pytest.raises(ConfigurationError):
        DecompositionForecaster(volatile, volatile)  # type: ignore[arg-type]


# ------------------------------------------------------------------ combination
def test_equal_weights_average_the_members() -> None:
    y = series()
    members = (NaiveForecaster(), SeasonalNaiveForecaster(season_length=7))
    model = FixedCombinationForecaster(members).fit(y)
    np.testing.assert_allclose(model.weights_, [0.5, 0.5])
    context = local_context(y, y.size)
    expected = np.mean([m.clone().fit(y).predict(7, context) for m in members], axis=0)
    np.testing.assert_allclose(model.predict(7, context), expected)


def test_inverse_mse_weights_favour_the_better_member() -> None:
    y = series(300)
    model = FixedCombinationForecaster(
        (NaiveForecaster(), SeasonalNaiveForecaster(season_length=7)), weights="inverse_mse"
    ).fit(y)
    assert model.weights_.sum() == pytest.approx(1.0)
    # The series is mostly weekly seasonality: the seasonal naive errs less one step ahead.
    assert model.weights_[1] > model.weights_[0]
    assert model.diagnostics() == {"weight_0": model.weights_[0], "weight_1": model.weights_[1]}


def test_combination_needs_two_members_and_known_weights() -> None:
    with pytest.raises(ConfigurationError):
        FixedCombinationForecaster((NaiveForecaster(),))
    with pytest.raises(ConfigurationError):
        FixedCombinationForecaster((NaiveForecaster(), NaiveForecaster()), weights="stacked")


# ------------------------------------------------------------------ multistage
def test_stages_nest_residual_hybrids() -> None:
    model = stack_stages(
        SeasonalNaiveForecaster(season_length=7),
        [SklearnCorrector(), SklearnCorrector()],
        window_size=7,
        alpha=1.0,
        residual_source="historical_oof_1step",
    )
    assert isinstance(model.base, ResidualHybridForecaster)
    assert isinstance(model.base.base, SeasonalNaiveForecaster)
    with pytest.raises(ConfigurationError):
        stack_stages(
            NaiveForecaster(),
            [SklearnCorrector()] * (MAX_STAGES + 1),
            window_size=7,
            alpha=1.0,
            residual_source="historical_oof_1step",
        )


def test_second_stage_learns_the_errors_of_the_first() -> None:
    y = series(300)
    model = stack_stages(
        SeasonalNaiveForecaster(season_length=7),
        [SklearnCorrector(), SklearnCorrector()],
        window_size=7,
        alpha=1.0,
        residual_source="historical_oof_1step",
    ).fit(y)
    # The residuals of stage 2 are errors of the whole stage-1 hybrid, never after their target.
    assert all(r.available_at == r.target for r in model.residuals_.records)
    # Out of fold: a stage-1 hybrid fitted on the prefix before the tail made each forecast.
    first = model.residuals_.records[0]
    prefix_model = model.base.clone().fit(y[: first.target])
    for record in model.residuals_.records[:5]:
        forecast = prefix_model.predict(1, local_context(y, record.target))[0]
        assert record.y_base == pytest.approx(forecast)


# ------------------------------------------------------------------ specs
def decomposition(method: str = "ma", variant: str = "A", **options: object) -> CandidateSpec:
    return CandidateSpec(
        "decomposition",
        "decomposition",
        LINEAR,
        CORRECTOR,
        7,
        variant=variant,  # type: ignore[arg-type]
        options_json=json.dumps({"method": method, **options}),
    )


def test_spec_rules_of_the_new_strategies() -> None:
    assert decomposition().label == "ma(curtose): linear(lags=7) | linear [w=7]"
    assert decomposition(ma_window=5).label.startswith("ma(5)")
    assert decomposition("stl", period=7).label.startswith("stl:")
    with pytest.raises(ConfigurationError, match="variante A"):
        decomposition("stl", variant="B", period=7)
    with pytest.raises(ConfigurationError, match="period"):
        decomposition("stl")
    with pytest.raises(ConfigurationError, match="Opções desconhecidas"):
        decomposition(smoothing=3)
    with pytest.raises(ConfigurationError, match="não usa"):
        CandidateSpec(
            "decomposition",
            "decomposition",
            LINEAR,
            CORRECTOR,
            7,
            1.0,
            options_json='{"method": "ma"}',
        )
    with pytest.raises(ConfigurationError, match="estágios"):
        CandidateSpec(
            "residual",
            "residual_multistage",
            SEASONAL,
            CORRECTOR,
            7,
            1.0,
            "historical_oof_1step",
            members=(CORRECTOR,) * MAX_STAGES,
        )
    with pytest.raises(ConfigurationError, match="repetidos"):
        CandidateSpec(
            "combination",
            "fixed_combination",
            LINEAR,
            members=(LINEAR,),
            options_json='{"weights": "equal"}',
        )
    with pytest.raises(ConfigurationError, match="members"):
        CandidateSpec(
            "residual",
            "residual_additive",
            SEASONAL,
            CORRECTOR,
            7,
            1.0,
            "historical_oof_1step",
            members=(CORRECTOR,),
        )


def test_new_fields_enter_identity_only_when_used() -> None:
    plain = CandidateSpec(
        "residual", "residual_additive", SEASONAL, CORRECTOR, 7, 1.0, "historical_oof_1step"
    )
    assert "members" not in plain.describe()
    assert "options" not in plain.describe()
    staged = CandidateSpec(
        "residual",
        "residual_multistage",
        SEASONAL,
        CORRECTOR,
        7,
        1.0,
        "historical_oof_1step",
        members=(CORRECTOR,),
    )
    assert staged.candidate_id != plain.candidate_id
    assert staged.stages == 2
    assert plain.stages == 1
    assert staged.label.count("linear") == 2
    equal = CandidateSpec(
        "combination",
        "fixed_combination",
        LINEAR,
        members=(SEASONAL,),
        options_json='{"weights": "equal"}',
    )
    inverse = CandidateSpec(
        "combination",
        "fixed_combination",
        LINEAR,
        members=(SEASONAL,),
        options_json='{"weights":"inverse_mse"}',
    )
    assert equal.candidate_id != inverse.candidate_id
    assert inverse.options == {"weights": "inverse_mse"}
    # Structures: weighting and method are part of the structure, parameters are not.
    assert structure_key(equal) != structure_key(inverse)
    assert structure_key(decomposition()) == structure_key(decomposition(ma_window=5))
    assert structure_key(decomposition()) != structure_key(decomposition("stl", period=7))


# ------------------------------------------------------------------ corrector combination
def test_corrector_combination_weights_and_prediction() -> None:
    from autohybts.strategies.combination import CorrectorCombination

    rng = np.random.default_rng(0)
    rows = rng.normal(size=(120, 7))
    targets = rows @ np.linspace(0.1, 0.7, 7) + rng.normal(0, 0.05, 120)
    linear, knn = SklearnCorrector(), SklearnCorrector("knn")
    combined = CorrectorCombination([linear, knn], weights="inverse_mse").fit(rows, targets)
    # A linear target: the linear corrector errs less on the held-out rows.
    assert combined.weights_.sum() == pytest.approx(1.0)
    assert combined.weights_[0] > combined.weights_[1]
    window = rows[-1]
    expected = sum(
        w * m.predict_next(window)
        for w, m in zip(combined.weights_, combined.members_, strict=True)
    )
    assert combined.predict_next(window) == pytest.approx(expected)
    equal = CorrectorCombination([linear, knn]).fit(rows, targets)
    np.testing.assert_allclose(equal.weights_, [0.5, 0.5])
    with pytest.raises(ConfigurationError):
        CorrectorCombination([linear])


def test_residual_combination_spec() -> None:
    spec = CandidateSpec(
        "combination",
        "residual_combination",
        SEASONAL,
        CORRECTOR,
        7,
        1.0,
        "historical_oof_1step",
        members=(MLP_SPEC,),
        options_json='{"weights": "equal"}',
    )
    assert spec.name == "seasonal_naive-comb(linear,mlp)"
    assert "comb[" in spec.label
    assert complexity_of(spec) == 3
    with pytest.raises(ConfigurationError, match="repetidos"):
        CandidateSpec(
            "combination",
            "residual_combination",
            SEASONAL,
            CORRECTOR,
            7,
            1.0,
            "historical_oof_1step",
            members=(CORRECTOR,),
            options_json='{"weights": "equal"}',
        )
    with pytest.raises(ConfigurationError, match="members"):
        CandidateSpec(
            "combination",
            "residual_combination",
            SEASONAL,
            CORRECTOR,
            7,
            1.0,
            "historical_oof_1step",
            options_json='{"weights": "equal"}',
        )


# ------------------------------------------------------------------ per-stage settings
def test_extra_stages_can_have_their_own_settings() -> None:
    staged = CandidateSpec(
        "residual",
        "residual_multistage",
        SEASONAL,
        CORRECTOR,
        7,
        1.0,
        "historical_oof_1step",
        members=(CORRECTOR,),
        options_json='{"stages": [{"window_size": 14, "alpha": 0.5}]}',
    )
    assert "estágios" in staged.label
    model = stack_stages(
        SeasonalNaiveForecaster(season_length=7),
        [SklearnCorrector(), SklearnCorrector()],
        window_size=7,
        alpha=1.0,
        residual_source="historical_oof_1step",
        stage_params=staged.options["stages"],
    )
    assert (model.window_size, model.alpha) == (14, 0.5)
    assert (model.base.window_size, model.base.alpha) == (7, 1.0)
    for bad in ('{"stages": [{"window_size": 0}]}', '{"stages": [{"lags": 3}]}', '{"stages": []}'):
        with pytest.raises(ConfigurationError):
            CandidateSpec(
                "residual",
                "residual_multistage",
                SEASONAL,
                CORRECTOR,
                7,
                1.0,
                "historical_oof_1step",
                members=(CORRECTOR,),
                options_json=bad,
            )


def test_per_stage_search_axis() -> None:
    from autohybts.registry import default_registry
    from autohybts.search.spaces import SearchSpace

    best = CandidateSpec(
        "residual", "residual_additive", SEASONAL, CORRECTOR, 7, 1.0, "historical_oof_1step"
    )
    shared = SearchSpace(
        correctors=("linear",), residual_windows=(7, 28), alphas=(0.5, 1.0), max_stages=2
    )
    per_stage = SearchSpace(
        correctors=("linear",),
        residual_windows=(7, 28),
        alphas=(0.5, 1.0),
        max_stages=2,
        stage_params="per_stage",
    )
    registry = default_registry()
    assert len(shared._stage_proposals(best, registry)) == 1
    proposals = per_stage._stage_proposals(best, registry)
    stages = [p.candidate.options.get("stages") for p in proposals]
    assert stages == [None, [{"window_size": 28}], [{"alpha": 0.5}]]
    assert per_stage.describe()["stage_params"] == "per_stage"
    with pytest.raises(ConfigurationError):
        SearchSpace(stage_params="mixed")
