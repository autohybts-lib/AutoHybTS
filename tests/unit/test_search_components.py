"""Budgets, search space generation and stage retention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalSplitter
from autohybts.domain.specs import CandidateSpec, ModelSpec
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.exceptions import ConfigurationError
from autohybts.registry import default_registry
from autohybts.search.budgets import PRESETS, SearchGuardConfig
from autohybts.search.pruning import best_per_structure, retain_bases, structure_key
from autohybts.search.spaces import SearchSpace

REGISTRY = default_registry()


def test_presets_are_valid_and_ordered() -> None:
    totals = [
        SearchGuardConfig.preset(name).max_trials_total
        for name in ("smoke", "small", "medium", "large")
    ]
    assert totals == sorted(totals)
    assert SearchGuardConfig.preset("small", max_trials_total=7).max_trials_total == 7


def test_presets_are_reproducible_budgets_by_count() -> None:
    # No wall clock by default (opt-in, cooperative) and no dead keys: every preset key
    # is a guard setting, and the number of origins is the experiment's ``fold``.
    for name, preset in PRESETS.items():
        assert set(preset) == {"guard"}
        assert SearchGuardConfig.preset(name).max_wall_time_s is None  # type: ignore[arg-type]
    assert SearchGuardConfig.preset("smoke", max_wall_time_s=30.0).max_wall_time_s == 30.0


def test_test_feedback_to_search_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="teste selado"):
        SearchGuardConfig(allow_test_feedback_to_search=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_trials_total": 0},
        {"max_refinements": -1},
        {"top_k_per_family": True},
        {"max_wall_time_s": 0.0},
        {"mode": "yolo"},
    ],
)
def test_invalid_guards(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        SearchGuardConfig(**kwargs)


def test_unknown_preset() -> None:
    with pytest.raises(ConfigurationError, match="Preset"):
        SearchGuardConfig.preset("huge")


def test_base_proposals_inject_seasonality_and_lags() -> None:
    space = SearchSpace(base_models=("seasonal_naive", "linear", "naive"), seasonal_period=12)
    proposals = space.base_proposals(REGISTRY)
    params = {p.candidate.base.name: p.candidate.base.params for p in proposals}
    assert params["seasonal_naive"] == {"season_length": 12}
    assert params["linear"] == {"lags": 12}
    assert params["naive"] == {}
    assert all(p.stage == "S0" and p.skip_reason is None for p in proposals)


def test_seasonal_naive_is_skipped_without_seasonality() -> None:
    proposals = SearchSpace(base_models=("seasonal_naive",)).base_proposals(REGISTRY)
    assert proposals[0].skip_reason is not None


def test_base_grids_expand_into_candidates() -> None:
    space = SearchSpace(
        base_models=("linear",), base_params={"linear": [{"lags": 3}, {"lags": 14}]}
    )
    assert [p.candidate.base.params["lags"] for p in space.base_proposals(REGISTRY)] == [3, 14]


def test_residual_grid_is_breadth_first_over_bases() -> None:
    bases = [
        CandidateSpec.standalone("naive", family="statistical"),
        CandidateSpec.standalone("linear", {"lags": 7}, family="linear"),
    ]
    space = SearchSpace(correctors=("linear",), residual_windows=(7, 28), alphas=(1.0,))
    proposals = space.residual_proposals(bases, REGISTRY)
    assert [(p.candidate.base.name, p.candidate.window_size) for p in proposals] == [
        ("naive", 7),
        ("linear", 7),
        ("naive", 28),
        ("linear", 28),
    ]
    assert proposals[1].parent == bases[1].candidate_id
    assert {p.stage for p in proposals} == {"S2"}


def test_in_sample_needs_the_capability() -> None:
    bases = [
        CandidateSpec.standalone("naive", family="statistical"),
        CandidateSpec.standalone("arima", {"order": (1, 0, 0)}, family="statistical"),
    ]
    space = SearchSpace(
        correctors=("linear",),
        residual_windows=(7,),
        alphas=(1.0,),
        residual_sources=("in_sample",),
    )
    reasons = [p.skip_reason for p in space.residual_proposals(bases, REGISTRY)]
    assert reasons[0] is not None
    assert reasons[1] is None


def test_refinement_changes_one_axis_at_a_time() -> None:
    best = CandidateSpec(
        "residual",
        "residual_additive",
        ModelSpec.of("arima"),
        ModelSpec.of("knn"),
        7,
        0.5,
        "historical_oof_1step",
    )
    space = SearchSpace(
        refine_sources=("historical_oof_multistep", "in_sample"),
        corrector_params={
            "knn": [
                {"estimator_params": {"n_neighbors": 3}},
                {"estimator_params": {"n_neighbors": 9}},
            ]
        },
    )
    proposals = space.refinement_proposals([best], REGISTRY, max_refinements=1)
    assert len(proposals) == 2  # one per axis
    by_source, by_params = proposals
    assert by_source.candidate.residual_source == "historical_oof_multistep"
    assert by_source.candidate.corrector == best.corrector
    assert by_params.candidate.residual_source == best.residual_source
    assert by_params.candidate.corrector.params == {"estimator_params": {"n_neighbors": 3}}
    assert {p.parent for p in proposals} == {best.candidate_id}
    assert {p.stage for p in proposals} == {"S3"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_models": ()},
        {"residual_sources": ("future",)},
        {"residual_windows": (0,)},
        {"seasonal_period": 0},
    ],
)
def test_invalid_spaces(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        SearchSpace(**kwargs)


def evaluated(
    names_params: list[tuple[str, dict[str, object] | None]], values: np.ndarray | None = None
):
    values = (
        values if values is not None else 10 + np.cumsum(np.random.default_rng(0).normal(0, 1, 120))
    )
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=len(values)), values)
    plan = TemporalSplitter(horizon=7, n_search_folds=2, n_select_folds=1).plan(frame)
    evaluator = TemporalEvaluator(REGISTRY)
    return [
        evaluator.evaluate(
            CandidateSpec.standalone(name, params, family=REGISTRY.entry(name).family),
            plan,
            "search",
        )
        for name, params in names_params
    ]


def test_structure_key_ignores_hyperparameters() -> None:
    a = CandidateSpec.standalone("linear", {"lags": 3}, family="linear")
    b = CandidateSpec.standalone("linear", {"lags": 9}, family="linear")
    assert (
        structure_key(a)
        == structure_key(b)
        != structure_key(CandidateSpec.standalone("knn", family="kernel"))
    )


def test_best_configuration_per_structure() -> None:
    results = evaluated([("linear", {"lags": 2}), ("linear", {"lags": 30}), ("naive", None)])
    best = best_per_structure(results, "MASE")
    assert len(best) == 2
    linear = next(r for r in best if r.candidate.base.name == "linear")
    assert linear.mean("MASE") == min(r.mean("MASE") for r in results[:2])


def test_partial_coverage_is_not_retained() -> None:
    results = evaluated([("naive", None), ("linear", {"lags": 100})])  # too long for search-0 only
    assert results[1].coverage("MASE") < 1.0
    assert [r.candidate.base.name for r in best_per_structure(results, "MASE")] == ["naive"]
    assert len(best_per_structure(results, "MASE", full_pairing=False)) == 2


def test_retention_top_k_plus_arima_and_best_without_duplicates() -> None:
    pytest.importorskip("statsmodels")
    results = evaluated(
        [
            ("naive", None),
            ("arima", {"order": (1, 1, 0)}),
            ("linear", None),
            ("tree", None),
            ("random_forest", {"estimator_params": {"n_estimators": 5}}),
            ("knn", None),
        ]
    )
    retained = retain_bases(results, "MASE", top_k=1)
    names = [r.candidate.base.name for r in retained]
    assert len(names) == len(set(names))  # T17: ARIMA never appears twice
    assert "arima" in names
    families = [r.candidate.family for r in retained if r.candidate.base.name != "arima"]
    assert len(families) == len(set(families))  # one per family (top_k=1)
    assert retained[0].mean("MASE") == min(r.mean("MASE") for r in results)
