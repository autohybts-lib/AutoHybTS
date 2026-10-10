"""T05 and T07 for residual hybrids."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from autohybts.adapters.base import Parametrized
from autohybts.adapters.naive import SeasonalNaiveForecaster
from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalSplitter
from autohybts.domain.specs import CandidateSpec, ModelSpec
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.registry import default_registry
from autohybts.residuals.generation import local_context
from autohybts.strategies.additive_residual import ResidualHybridForecaster

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

REGISTRY = default_registry()
SOURCES = ("historical_oof_1step", "historical_oof_multistep", "in_sample")


def hybrid(source: str, corrector: str = "linear", window: int = 7) -> CandidateSpec:
    base = (
        ModelSpec.of("arima", {"order": (1, 0, 0)})
        if source == "in_sample"
        else ModelSpec.of("seasonal_naive")
    )
    params = {"estimator_params": {"max_iter": 300}} if corrector == "mlp" else None
    return CandidateSpec(
        "residual", "residual_additive", base, ModelSpec.of(corrector, params), window, 1.0, source
    )


def forecasts(
    candidate: CandidateSpec, values: np.ndarray
) -> dict[str, tuple[int, tuple[float, ...]]]:
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=len(values)), values)
    plan = TemporalSplitter(horizon=7, n_search_folds=2, n_select_folds=2, step=15).plan(frame)
    result = TemporalEvaluator(REGISTRY).evaluate(candidate, plan)
    assert all(t.succeeded for t in result.trials), result.failures
    return {t.fold_id: (t.origin_index, t.y_pred) for t in result.trials}


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("corrector", ["linear", "knn", "mlp"])
def test_future_perturbation_does_not_change_hybrid_forecasts(source: str, corrector: str) -> None:
    if source == "in_sample":
        pytest.importorskip("statsmodels")
    rng = np.random.default_rng(5)
    noise = np.zeros(160)
    for i in range(1, 160):
        noise[i] = 0.7 * noise[i - 1] + rng.normal()
    values = 20 + 3 * np.sin(2 * np.pi * np.arange(160) / 7) + noise
    candidate = hybrid(source, corrector)
    reference = forecasts(candidate, values)

    for cut in (125, 145):
        perturbed = values.copy()
        perturbed[cut:] += 40.0
        changed = forecasts(candidate, perturbed)
        for fold_id, (index, path) in changed.items():
            if index < cut:
                assert path == reference[fold_id][1], f"{fold_id} saw data after {cut}"
        assert any(path != reference[f][1] for f, (index, path) in changed.items() if index >= cut)


class SpyCorrector(Parametrized):
    """Records every window it is asked about; predicts mean(window) + 1."""

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def fit(self, windows: npt.ArrayLike, targets: npt.ArrayLike) -> SpyCorrector:
        return self

    def predict_next(self, window: npt.ArrayLike) -> float:
        values = np.asarray(window, dtype=float)
        self.calls.append(values.copy())
        return float(values.mean() + 1.0)

    def get_params(self) -> dict[str, Any]:
        return {}

    def clone(self) -> SpyCorrector:
        return self  # keep the same spy so the test can read its calls


def test_variant_a_recursion_feeds_only_predicted_residuals() -> None:
    # T07: after step 1 the window holds the initial residuals shifted left plus the
    # corrector's own outputs, never observed residuals of future steps.
    y = 10 + np.sin(np.arange(120) / 3) * 4
    spy = SpyCorrector()
    model = ResidualHybridForecaster(SeasonalNaiveForecaster(7), spy, window_size=4).fit(y)
    spy.calls.clear()
    model.predict(3, local_context(y, y.size))

    initial = model.residuals_.residuals[-4:]
    first, second, third = spy.calls
    assert np.array_equal(first, initial)
    r1 = first.mean() + 1
    assert np.allclose(second, [*initial[1:], r1])
    r2 = second.mean() + 1
    assert np.allclose(third, [*initial[2:], r1, r2])


def test_initial_window_is_available_at_the_origin() -> None:
    # T06 at model level: the residuals that seed the forecast all end at the origin.
    y = 10 + np.sin(np.arange(120) / 3) * 4
    model = ResidualHybridForecaster(SeasonalNaiveForecaster(7), SpyCorrector(), window_size=4).fit(
        y
    )
    assert model.residuals_.available_at.max() == y.size - 1
    assert (model.residuals_.targets < y.size).all()
    assert (model.row_origins_ < model.residuals_.targets.max()).all()
