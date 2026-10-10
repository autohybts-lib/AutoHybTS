"""Phase 1 acceptance (§22): baseline and residual hybrid, compared causally, horizon 14."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autohybts.adapters.base import BaseForecaster
from autohybts.adapters.naive import SeasonalNaiveForecaster
from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalSplitter
from autohybts.data.windows import lag_matrix
from autohybts.domain.specs import CandidateSpec, ModelSpec
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.exceptions import (
    ConfigurationError,
    InsufficientHistoryError,
    UnsupportedCapabilityError,
)
from autohybts.forecasting.multi_step_a import MultiStepEngineA
from autohybts.registry import default_registry
from autohybts.reporting.leaderboard import leaderboard
from autohybts.residuals.correctors import SklearnCorrector
from autohybts.residuals.generation import ResidualSettings, local_context
from autohybts.strategies.additive_residual import ResidualHybridForecaster

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
REGISTRY = default_registry()


def ar_series(n: int = 300, phi: float = 0.8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + rng.normal()
    t = np.arange(n)
    return 50 + 0.02 * t + 3 * np.sin(2 * np.pi * t / 7) + noise


def residual_spec(
    window: int = 7, alpha: float = 1.0, source: str = "historical_oof_1step"
) -> CandidateSpec:
    return CandidateSpec(
        "residual",
        "residual_additive",
        ModelSpec.of("seasonal_naive"),
        ModelSpec.of("linear"),
        window,
        alpha,
        source,
    )


def test_baseline_and_hybrid_are_compared_on_the_same_origins() -> None:
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=300), ar_series())
    plan = TemporalSplitter(horizon=14, n_search_folds=3, n_select_folds=2).plan(frame)
    evaluator = TemporalEvaluator(REGISTRY, seasonal_period=7)
    baseline = CandidateSpec.standalone("seasonal_naive", family="statistical")
    runs = [[evaluator.evaluate(c, plan) for c in (baseline, residual_spec())] for _ in range(2)]
    first, second = runs
    for a, b in zip(first, second, strict=True):  # reproducible
        assert [t.y_pred for t in a.trials] == [t.y_pred for t in b.trials]
    table = leaderboard(first)
    assert (table["coverage"] == 1.0).all()
    assert set(table["strategy"]) == {"standalone", "residual_additive"}
    assert first[0].scores("MASE").index.equals(first[1].scores("MASE").index)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_corrector_reduces_one_step_error_on_ar_residuals(seed: int) -> None:
    # Oracle 2 of §19.2: with AR(0.8) noise left by the seasonal naive, the corrector
    # predicts the next residual. Measured: step-1 MAE ratio 0.41-0.92 over 6 seeds, while
    # the 14-step MASE ratio stays around 1 (0.96-1.10) because the correction decays.
    # Only the robust part is asserted; winning the full horizon is not imposed.
    frame = SeriesFrame.from_arrays(
        pd.date_range("2024-01-01", periods=400), ar_series(400, seed=seed)
    )
    plan = TemporalSplitter(horizon=14, n_search_folds=10, n_select_folds=10, step=5).plan(frame)
    evaluator = TemporalEvaluator(REGISTRY, seasonal_period=7)
    base = evaluator.evaluate(
        CandidateSpec.standalone("seasonal_naive", family="statistical"), plan
    )
    hybrid = evaluator.evaluate(residual_spec(), plan)

    def step_one_mae(result: object) -> float:
        return float(np.mean([abs(t.y_true[0] - t.y_pred[0]) for t in result.trials]))

    assert step_one_mae(hybrid) < step_one_mae(base)


def test_alpha_zero_reproduces_the_base() -> None:
    y = ar_series()
    context = local_context(y, y.size)
    base = SeasonalNaiveForecaster(7).fit(y).predict(14, context)
    model = ResidualHybridForecaster(SeasonalNaiveForecaster(7), SklearnCorrector(), alpha=0.0).fit(
        y
    )
    assert np.array_equal(model.predict(14, context), base)


def test_hybrid_equals_a_manual_zhang_style_computation() -> None:
    # Base + linear corrector on OOF one-step residuals, recursion written out by hand.
    y = ar_series()
    model = ResidualHybridForecaster(
        SeasonalNaiveForecaster(7), SklearnCorrector(scale=False), window_size=3
    ).fit(y)
    tail = y[150:]
    residuals = tail - y[143:293]  # seasonal naive, one step ahead, fitted before the tail
    X, target = lag_matrix(residuals, [3, 2, 1])  # oldest first, like the residual window
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), target, rcond=None)
    window = list(residuals[-3:])
    corrections = []
    for _ in range(5):
        step = beta[0] + np.dot(beta[1:], window[-3:])
        corrections.append(step)
        window.append(step)
    base = np.resize(y[-7:], 5)
    assert np.allclose(model.predict(5, local_context(y, y.size)), base + np.array(corrections))


def test_short_series_with_window_28_is_unavailable_not_padded() -> None:
    # Oracle 4 of §19.2.
    y = ar_series(n=70)
    model = ResidualHybridForecaster(SeasonalNaiveForecaster(7), SklearnCorrector(), window_size=28)
    with pytest.raises(InsufficientHistoryError):
        model.fit(y)


def test_operational_fallback_serves_the_base_and_says_why() -> None:
    y = ar_series(n=70)
    settings = ResidualSettings(on_failure="fallback_to_base")
    model = ResidualHybridForecaster(
        SeasonalNaiveForecaster(7), SklearnCorrector(), window_size=28, settings=settings
    ).fit(y)
    context = local_context(y, y.size)
    assert model.fallback_reason_ is not None
    assert np.array_equal(
        model.predict(5, context), SeasonalNaiveForecaster(7).fit(y).predict(5, context)
    )


def test_short_series_is_a_failed_unit_in_the_evaluator() -> None:
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=60), ar_series(n=60))
    plan = TemporalSplitter(horizon=7, n_search_folds=1, n_select_folds=1).plan(frame)
    result = TemporalEvaluator(REGISTRY).evaluate(residual_spec(window=28), plan)
    assert {t.status for t in result.trials} == {"FAILED"}
    assert result.coverage() == 0.0


def test_later_context_extends_residuals_without_refit() -> None:
    y = ar_series()
    model = ResidualHybridForecaster(SeasonalNaiveForecaster(7), SklearnCorrector()).fit(y[:250])
    fitted_corrector = model.corrector_
    later = model.predict(5, local_context(y, 260))
    assert model.corrector_ is fitted_corrector
    assert later.shape == (5,)
    with pytest.raises(UnsupportedCapabilityError, match="estendem"):
        model.predict(5, local_context(y[1:], 255))  # not an extension of the training vector


def test_hybrid_honours_the_forecaster_contract() -> None:
    y = ar_series()
    model = ResidualHybridForecaster(
        SeasonalNaiveForecaster(7), SklearnCorrector(), window_size=5, alpha=0.5
    )
    assert isinstance(model, BaseForecaster)
    twin = model.clone()
    assert twin.get_params()["window_size"] == 5
    assert twin.base is not model.base
    context = local_context(y, y.size)
    assert np.array_equal(model.fit(y).predict(7, context), twin.fit(y).predict(7, context))


@pytest.mark.parametrize(
    "kwargs",
    [{"window_size": 0}, {"alpha": float("nan")}, {"residual_source": "future"}, {"alpha": True}],
)
def test_invalid_hybrid_parameters(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        ResidualHybridForecaster(SeasonalNaiveForecaster(7), SklearnCorrector(), **kwargs)


def test_components_must_honour_their_contracts() -> None:
    with pytest.raises(ConfigurationError, match="BaseForecaster"):
        ResidualHybridForecaster(object(), SklearnCorrector())
    with pytest.raises(ConfigurationError, match="ResidualCorrector"):
        ResidualHybridForecaster(SeasonalNaiveForecaster(7), object())


def test_engine_a_returns_a_dated_frame() -> None:
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=300), ar_series())
    plan = TemporalSplitter(horizon=14, n_search_folds=1, n_select_folds=1).plan(frame)
    origin = plan.select_origins[0]
    model = ResidualHybridForecaster(SeasonalNaiveForecaster(7), SklearnCorrector()).fit(
        plan.train_values(origin)
    )
    out = MultiStepEngineA().forecast(model, plan.context(origin), 14, candidate_id="h").to_pandas()
    assert len(out) == 14
    assert out["timestamp"].iloc[0] == origin.origin_time + pd.Timedelta(days=1)
    with pytest.raises(ConfigurationError, match="data de origem"):
        MultiStepEngineA().forecast(
            model, local_context(plan.train_values(origin), origin.n_train), 14
        )
