"""Compositions of candidates (ADR-0007): identity, validation, building and fit counting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalSplitter
from autohybts.domain.fit_count import count_fits, record_fit
from autohybts.domain.specs import COMPOSITE_BASE, CandidateSpec, ModelSpec
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.exceptions import ConfigurationError
from autohybts.registry import default_registry
from autohybts.registry.capabilities import Capabilities
from autohybts.reporting.strategies import complexity
from autohybts.residuals.generation import local_context
from autohybts.selection.rank import structure_key
from autohybts.strategies.base import build_model

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

LINEAR = CandidateSpec.standalone("linear", {"lags": 7}, family="linear")
NAIVE = CandidateSpec.standalone("naive", family="statistical")
KNN = CandidateSpec.standalone("knn", {"lags": 7}, family="kernel")
SEASONAL = CandidateSpec.standalone("seasonal_naive", {"season_length": 7}, family="statistical")
HYBRID = CandidateSpec(
    "residual",
    "residual_additive",
    ModelSpec.of("naive"),
    ModelSpec.of("linear"),
    7,
    1.0,
    "historical_oof_1step",
)


def ensemble(*components: CandidateSpec, method: str = "equal", **options: object) -> CandidateSpec:
    return CandidateSpec.ensemble(components, method=method, **options)


def series(n: int = 160, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.6 * noise[i - 1] + rng.normal()
    t = np.arange(n)
    return 20 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 7) + noise


# ------------------------------------------------------------------ identity and validation
def test_ensemble_identity_is_canonical_and_depends_on_subtrees() -> None:
    first = ensemble(LINEAR, NAIVE)
    assert ensemble(NAIVE, LINEAR) == first  # commutative: canonical order
    assert ensemble(LINEAR, NAIVE, method="inverse_mse") != first
    other = ensemble(CandidateSpec.standalone("linear", {"lags": 14}, family="linear"), NAIVE)
    assert other.candidate_id != first.candidate_id
    # Defaults are written into the options, so explicit and implicit defaults coincide.
    assert ensemble(LINEAR, NAIVE, KNN, method="trimmed_mean") == ensemble(
        LINEAR, NAIVE, KNN, method="trimmed_mean", trim=0.2
    )
    assert ensemble(LINEAR, NAIVE, method="nnls").options == {"method": "nnls", "shrinkage": 0.5}
    # Legacy strategies keep their identities: no "components" key when empty.
    assert "components" not in LINEAR.describe()


def test_nested_compositions_and_their_names() -> None:
    hybrids = ensemble(HYBRID, LINEAR)
    corrected = CandidateSpec(
        "ensemble",
        "residual_composite",
        COMPOSITE_BASE,
        ModelSpec.of("knn"),
        7,
        1.0,
        "historical_oof_1step",
        components=(ensemble(LINEAR, NAIVE),),
    )
    assert (LINEAR.depth, HYBRID.depth) == (0, 0)  # single structures
    assert corrected.depth == 2  # residual composition over an ensemble
    assert ensemble(corrected, KNN).depth == 3  # the plan's example: maximum allowed
    assert set(corrected.model_names()) == {
        ("base", "linear"),
        ("base", "naive"),
        ("corrector", "knn"),
    }
    assert set(hybrids.model_names()) == {
        ("base", "linear"),
        ("base", "naive"),
        ("corrector", "linear"),
    }
    assert corrected.name == "(equal[linear,naive])-knn"
    assert "ensemble[equal" in corrected.label
    assert complexity(hybrids) == 3
    assert complexity(corrected) == 3
    assert structure_key(ensemble(LINEAR, NAIVE)) != structure_key(ensemble(LINEAR, KNN))
    assert structure_key(ensemble(LINEAR, NAIVE)) == structure_key(
        ensemble(CandidateSpec.standalone("linear", {"lags": 14}, family="linear"), NAIVE)
    )


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: ensemble(LINEAR), "de 2 a"),
        (lambda: ensemble(LINEAR, LINEAR), "repetidos"),
        (lambda: ensemble(LINEAR, NAIVE, method="median"), "mediana"),
        (lambda: ensemble(LINEAR, NAIVE, method="trimmed_mean"), "aparada"),
        (lambda: ensemble(LINEAR, NAIVE, KNN, method="trimmed_mean", trim=0.6), "trim"),
        (lambda: ensemble(LINEAR, NAIVE, method="nnls", shrinkage=2), "shrinkage"),
        (lambda: ensemble(LINEAR, NAIVE, method="equal", shrinkage=0.1), "shrinkage"),
        (lambda: ensemble(LINEAR, NAIVE, method="vote"), "method"),
        (
            lambda: ensemble(
                LINEAR, CandidateSpec.standalone("naive", family="statistical", variant="B")
            ),
            "variante",
        ),
        (
            lambda: CandidateSpec(
                "ensemble",
                "ensemble",
                ModelSpec.of("linear"),
                components=(LINEAR, NAIVE),
                options_json='{"method": "equal"}',
            ),
            "COMPOSITE_BASE",
        ),
        (
            lambda: CandidateSpec(
                "ensemble",
                "residual_composite",
                COMPOSITE_BASE,
                ModelSpec.of("knn"),
                7,
                1.0,
                "historical_oof_1step",
                components=(LINEAR, NAIVE),
            ),
            "exatamente um",
        ),
        (
            lambda: CandidateSpec(
                "linear", "standalone", ModelSpec.of("linear"), components=(NAIVE,)
            ),
            "não aceita components",
        ),
        (
            lambda: ensemble(ensemble(ensemble(ensemble(LINEAR, NAIVE), KNN), SEASONAL), HYBRID),
            "níveis",
        ),
    ],
)
def test_invalid_compositions_are_refused_before_fitting(build, match: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ConfigurationError, match=match):
        build()


# ------------------------------------------------------------------ building and forecasting
def fitted(candidate: CandidateSpec, y: np.ndarray) -> object:
    model, _ = build_model(candidate, default_registry(), unit_seed=1)
    return model.fit(y)


def test_aggregations_match_their_definitions() -> None:
    y = series()
    context = local_context(y, len(y))
    members = (
        LINEAR,
        NAIVE,
        KNN,
        SEASONAL,
        CandidateSpec.standalone("svm", {"lags": 7}, family="kernel"),
    )
    paths = np.vstack(
        [fitted(m, y).predict(5, context) for m in sorted(members, key=lambda c: c.candidate_id)]
    )
    np.testing.assert_allclose(
        fitted(ensemble(*members[:2]), y).predict(5, context),
        np.vstack(
            [
                fitted(m, y).predict(5, context)
                for m in sorted(members[:2], key=lambda c: c.candidate_id)
            ]
        ).mean(axis=0),
    )
    np.testing.assert_allclose(
        fitted(ensemble(*members, method="median"), y).predict(5, context), np.median(paths, axis=0)
    )
    trimmed = np.sort(paths, axis=0)[1:4].mean(axis=0)  # 5 members, trim 0.2: one off each end
    np.testing.assert_allclose(
        fitted(ensemble(*members, method="trimmed_mean"), y).predict(5, context), trimmed
    )


@pytest.mark.parametrize("method", ["inverse_mse", "nnls"])
def test_learned_weights_are_a_convex_combination(method: str) -> None:
    model = fitted(ensemble(LINEAR, NAIVE, KNN, method=method), series())
    weights = np.array([v for k, v in model.diagnostics().items() if k.startswith("weight_")])
    assert weights.shape == (3,)
    assert (weights >= 0).all()
    assert weights.sum() == pytest.approx(1.0)
    if method == "nnls":
        assert (weights >= 0.5 / 3 - 1e-12).all()  # shrinkage 0.5 keeps every member


def test_full_shrinkage_gives_equal_weights() -> None:
    model = fitted(ensemble(LINEAR, NAIVE, method="nnls", shrinkage=1.0), series())
    assert model.diagnostics() == pytest.approx({"weight_0": 0.5, "weight_1": 0.5})


def test_ensemble_of_hybrids_and_corrected_ensemble_forecast() -> None:
    y = series()
    context = local_context(y, len(y))
    for candidate in (
        ensemble(HYBRID, LINEAR, method="nnls"),
        CandidateSpec(
            "ensemble",
            "residual_composite",
            COMPOSITE_BASE,
            ModelSpec.of("linear"),
            7,
            1.0,
            "historical_oof_1step",
            components=(ensemble(LINEAR, NAIVE),),
        ),
    ):
        path = fitted(candidate, y).predict(7, context)
        assert path.shape == (7,)
        assert np.isfinite(path).all()


# ------------------------------------------------------------------ fit counting
def plan():  # type: ignore[no-untyped-def]
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=200), series(200))
    return TemporalSplitter(horizon=7, n_search_folds=1, n_select_folds=1).plan(frame)


@pytest.mark.parametrize(
    ("candidate", "fits"),
    [
        (LINEAR, 1),
        (HYBRID, 3),  # base, base on the prefix for out-of-fold residuals, corrector
        (ensemble(LINEAR, NAIVE), 2),
        (ensemble(LINEAR, NAIVE, method="nnls"), 4),  # + one prefix fit per member
    ],
)
def test_each_trial_counts_its_leaf_fits(candidate: CandidateSpec, fits: int) -> None:
    result = TemporalEvaluator(default_registry(), seasonal_period=7).evaluate(candidate, plan())
    assert {t.n_fits for t in result.trials} == {fits}


def test_rolling_variant_b_adds_no_fit_after_the_origin() -> None:
    a = TemporalEvaluator(default_registry(), seasonal_period=7).evaluate(HYBRID, plan())
    hybrid_b = CandidateSpec(
        "residual",
        "residual_additive",
        ModelSpec.of("naive"),
        ModelSpec.of("linear"),
        7,
        1.0,
        "historical_oof_1step",
        variant="B",
    )
    b = TemporalEvaluator(default_registry(), seasonal_period=7).evaluate(hybrid_b, plan())
    assert [t.n_fits for t in b.trials] == [t.n_fits for t in a.trials]


def test_counter_is_scoped_and_silent_outside() -> None:
    record_fit()  # no counter open: nothing happens
    with count_fits() as outer:
        record_fit(2)
        with count_fits() as inner:
            record_fit()
        assert (outer.total, inner.total) == (2, 1)


# ------------------------------------------------------------------ capabilities
@pytest.mark.parametrize(
    "kwargs",
    [
        {"training_scope": "panel"},
        {"output_type": "interval"},
        {"horizon_strategy": "magic"},
        {"max_horizon": 0},
        {"max_horizon": True},
        {"target_mode": "panel"},
    ],
)
def test_invalid_capabilities(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        Capabilities(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("capabilities", "reason"),
    [
        (Capabilities(training_scope="global"), "globalmente"),
        (Capabilities(output_type="quantile"), "quantile"),
        (Capabilities(max_horizon=3), "no máximo 3"),
        (Capabilities(target_mode="exogenous"), "univariada"),
        (Capabilities(target_mode="multivariate"), "multivariate"),
    ],
)
def test_ineligible_models_are_skipped_not_fitted(capabilities: Capabilities, reason: str) -> None:
    from autohybts.adapters.naive import NaiveForecaster

    registry = default_registry()
    registry.register_base(
        "special", lambda p: NaiveForecaster(), family="statistical", capabilities=capabilities
    )
    result = TemporalEvaluator(registry).evaluate(
        CandidateSpec.standalone("special", family="statistical"), plan()
    )
    assert {t.status for t in result.trials} == {"SKIPPED"}
    assert reason in (result.trials[0].reason or "")
    assert {t.n_fits for t in result.trials} == {0}
    # Inside an ensemble the whole composition is skipped, never half fitted.
    mixed = TemporalEvaluator(registry).evaluate(
        ensemble(CandidateSpec.standalone("special", family="statistical"), LINEAR), plan()
    )
    assert {t.status for t in mixed.trials} == {"SKIPPED"}
