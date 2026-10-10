"""Advanced families in the search and the API, and ``tune_model`` (I12b)."""

from __future__ import annotations

import importlib.util
from collections import Counter

import pandas as pd
import pytest

from autohybts import AutoHybridExperiment
from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalSplitter
from autohybts.datasets import get_data
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError
from autohybts.registry import default_registry
from autohybts.search.budgets import SearchGuardConfig
from autohybts.search.runner import SearchEngine
from autohybts.search.spaces import SearchSpace
from autohybts.search.tuning import Distribution, Tuner

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

ALL = ("standalone", "residual", "multistage", "decomposition", "combination")
FAST = {
    "fh": 7,
    "fold": 2,
    "base_models": ("naive", "seasonal_naive", "linear"),
    "correctors": ("linear",),
    "residual_windows": (7,),
    "alphas": (1.0,),
}


def series(n: int = 300, seed: int = 0) -> pd.Series:
    return get_data("synthetic_daily", n=n, seed=seed)


def plan() -> object:
    return TemporalSplitter(horizon=7, n_search_folds=2, n_select_folds=2).plan(
        SeriesFrame.from_series(series())
    )


def advanced_space(**overrides: object) -> SearchSpace:
    settings: dict[str, object] = {
        "base_models": ("naive", "seasonal_naive", "linear"),
        "correctors": ("linear",),
        "residual_windows": (7,),
        "alphas": (1.0,),
        "seasonal_period": 7,
        "max_stages": 2,
        "decomposition_methods": ("ma",),
        "combination_weights": ("equal", "inverse_mse"),
    }
    return SearchSpace(**{**settings, **overrides})  # type: ignore[arg-type]


# ------------------------------------------------------------------ search
@pytest.mark.parametrize("budget", ["smoke", "small"])
def test_every_family_is_represented_and_budgets_hold(budget: str) -> None:
    evaluator = TemporalEvaluator(default_registry(), seasonal_period=7)
    guard = SearchGuardConfig.preset(budget)  # type: ignore[arg-type]
    result = SearchEngine(evaluator, advanced_space(), guard).run(plan())  # type: ignore[arg-type]
    evaluated = Counter(e.strategy for e in result.ledger if e.status == "EVALUATED")
    # S4 runs on what is left, but does not starve the residual stages before it.
    assert evaluated["residual_additive"] >= 1
    assert evaluated["residual_multistage"] >= 1
    assert evaluated["decomposition"] >= 1
    assert evaluated["fixed_combination"] >= 1
    assert sum(evaluated.values()) <= guard.max_trials_total
    stages = {e.strategy: e.stage for e in result.ledger}
    assert stages["residual_multistage"] == "S3"
    assert stages["decomposition"] == stages["fixed_combination"] == "S4"


def test_advanced_settings_are_validated_and_described_only_when_used() -> None:
    assert "max_stages" not in SearchSpace().describe()
    described = advanced_space().describe()
    assert described["max_stages"] == 2
    assert described["decomposition_methods"] == ["ma"]
    with pytest.raises(ConfigurationError, match="variante A"):
        advanced_space(decomposition_methods=("stl",), variant="B")
    with pytest.raises(ConfigurationError, match="max_stages"):
        advanced_space(max_stages=4)
    with pytest.raises(ConfigurationError, match="Pesos"):
        advanced_space(combination_weights=("median",))


def test_stl_without_seasonality_is_skipped_with_a_reason() -> None:
    space = advanced_space(decomposition_methods=("stl",), seasonal_period=1)
    registry = default_registry()
    bases = [p.candidate for p in space.base_proposals(registry)]
    proposals = space.advanced_proposals(bases, registry)
    stl = [p for p in proposals if p.candidate.strategy == "decomposition"]
    assert stl
    assert all("sazonalidade" in (p.skip_reason or "") for p in stl)


def test_combination_joins_the_two_best_bases_of_different_families() -> None:
    space = advanced_space(decomposition_methods=())
    registry = default_registry()
    bases = [p.candidate for p in space.base_proposals(registry)]  # naive, seasonal, linear
    proposals = space.advanced_proposals(bases, registry)
    combination = proposals[0].candidate
    families = {bases[0].family, registry.entry(combination.members[0].name).family}
    assert len(families) == 2  # naive and seasonal_naive share a family: linear is chosen
    assert combination.members[0].name == "linear"


# ------------------------------------------------------------------ API
@pytest.fixture(scope="module")
def experiment() -> AutoHybridExperiment:
    return AutoHybridExperiment(**FAST).fit(series())


def test_compare_models_with_every_family(experiment: AutoHybridExperiment) -> None:
    result = experiment.compare_models(include=ALL)
    families = set(result.decisions["family"])
    assert {"decomposition", "combination"} <= families
    strategies = {e.strategy for e in result.search.ledger if e.status == "EVALUATED"}
    assert {"residual_multistage", "decomposition", "fixed_combination"} <= strategies
    with pytest.raises(ConfigurationError, match="inclua 'residual'"):
        experiment.compare_models(include=("standalone", "multistage"))
    with pytest.raises(ConfigurationError, match="include"):
        experiment.compare_models(include=("standalone", "ensemble"))


def test_create_the_new_structures_by_hand(experiment: AutoHybridExperiment) -> None:
    staged = experiment.create_hybrid("seasonal_naive", "linear", extra_stages=["linear"])
    assert staged.candidate.strategy == "residual_multistage"
    decomposition = experiment.create_decomposition("linear", "linear", ma_window=7)
    assert decomposition.candidate.options == {"method": "ma", "ma_window": 7}
    combination = experiment.create_combination(["seasonal_naive", "linear"], weights="equal")
    assert combination.candidate.family == "combination"
    for result in (staged, decomposition, combination):
        assert result.evaluation.coverage("MASE") == 1.0
    with pytest.raises(ConfigurationError):
        experiment.create_combination(["linear"])


@pytest.mark.skipif(importlib.util.find_spec("statsmodels") is None, reason="needs stats")
def test_stl_decomposition_by_hand_uses_the_seasonal_period(
    experiment: AutoHybridExperiment,
) -> None:
    result = experiment.create_decomposition("linear", "linear", method="stl")
    assert result.candidate.options == {"method": "stl", "period": 7}
    assert result.evaluation.coverage("MASE") == 1.0


def test_variant_b_experiment_creates_variant_b_candidates() -> None:
    exp = AutoHybridExperiment(**{**FAST, "variant": "B"}).fit(series())
    assert exp.create_model("linear").candidate.variant == "B"
    assert exp.create_hybrid("seasonal_naive", "linear").candidate.variant == "B"
    assert exp.create_decomposition("linear", "linear").candidate.variant == "B"


# ------------------------------------------------------------------ tune_model
class LayerSpy:
    """Delegates to a real evaluator and records the layer of every evaluation."""

    def __init__(self, evaluator: TemporalEvaluator) -> None:
        self.evaluator = evaluator
        self.layers: list[object] = []

    @property
    def metrics(self) -> tuple[str, ...]:
        return self.evaluator.metrics

    def evaluate(self, candidate, plan, layer=None):  # type: ignore[no-untyped-def]
        self.layers.append(layer)
        return self.evaluator.evaluate(candidate, plan, layer)


def test_tuning_sees_only_v_search_and_never_loses_to_the_original() -> None:
    spy = LayerSpy(TemporalEvaluator(default_registry(), seasonal_period=7))
    exp_plan = plan()
    original = AutoHybridExperiment(**FAST).fit(series())._candidates(["linear"])[0]
    space = {"lags": Distribution("int", 2, 21)}
    result = Tuner(spy, n_trials=6, sampler="random", session_seed=3).tune(
        original,
        exp_plan,  # type: ignore[arg-type]
        space,
    )
    assert set(spy.layers) == {"search"}
    assert result.trials.loc[0, "candidate_id"] == original.candidate_id
    scores = result.trials.set_index("candidate_id")["score"]
    assert scores[result.best.candidate_id] <= scores[original.candidate_id]
    assert result.improvement >= 0
    again = Tuner(spy, n_trials=6, sampler="random", session_seed=3).tune(
        original,
        exp_plan,  # type: ignore[arg-type]
        space,
    )
    pd.testing.assert_frame_equal(result.trials, again.trials)


def test_tune_model_api(experiment: AutoHybridExperiment) -> None:
    tuned, tuner = experiment.tune_model(
        "knn",
        n_trials=4,
        search_space={"lags": (3, 14), "estimator_params.n_neighbors": [3, 5, 9]},
        sampler="random",
        return_tuner=True,
    )
    assert tuned.candidate == tuner.best
    assert len(tuner.trials) <= 4
    assert {t.layer for t in tuned.evaluation.trials} == {"search", "select"}
    with pytest.raises(UnsupportedCapabilityError, match="espaço de ajuste"):
        experiment.tune_model("naive")
    with pytest.raises(ConfigurationError, match="Distribution"):
        experiment.tune_model("knn", search_space={"lags": 7})  # type: ignore[dict-item]


@pytest.mark.skipif(importlib.util.find_spec("optuna") is not None, reason="optuna installed")
def test_optuna_sampler_needs_the_extra(experiment: AutoHybridExperiment) -> None:
    with pytest.raises(UnsupportedCapabilityError, match="optuna"):
        experiment.tune_model("knn", n_trials=2, sampler="optuna")


@pytest.mark.skipif(importlib.util.find_spec("optuna") is None, reason="extra `optuna` missing")
def test_optuna_sampler_is_seeded(experiment: AutoHybridExperiment) -> None:
    runs = [
        experiment.tune_model("knn", n_trials=4, sampler="optuna", return_tuner=True)[1]
        for _ in range(2)
    ]
    assert runs[0].sampler == "optuna"
    pd.testing.assert_frame_equal(runs[0].trials, runs[1].trials)


def test_corrector_combination_and_stage_settings_through_the_api() -> None:
    exp = AutoHybridExperiment(**{**FAST, "correctors": ("linear", "knn")}).fit(series())
    combined = exp.create_hybrid(
        "seasonal_naive", "linear", combine_correctors=["knn"], combination_weights="equal"
    )
    assert combined.candidate.strategy == "residual_combination"
    staged = exp.create_hybrid(
        "seasonal_naive", "linear", extra_stages=["linear"], stage_params=[{"window_size": 5}]
    )
    assert staged.candidate.options == {"stages": [{"window_size": 5}]}
    for result in (combined, staged):
        assert result.evaluation.coverage("MASE") == 1.0
    with pytest.raises(ConfigurationError, match="não os dois"):
        exp.create_hybrid("naive", "linear", extra_stages=["linear"], combine_correctors=["knn"])
    with pytest.raises(ConfigurationError, match="stage_params"):
        exp.create_hybrid("naive", "linear", stage_params=[{"alpha": 0.5}])
    result = exp.compare_models(include=("standalone", "residual", "combination"))
    strategies = {e.strategy for e in result.search.ledger}
    assert "residual_combination" in strategies
