"""ET01/ET02/ET03 for compositions: nothing after the origin shapes weights or forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalSplitter
from autohybts.domain.specs import CandidateSpec, ModelSpec
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.registry import default_registry

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

LINEAR = CandidateSpec.standalone("linear", {"lags": 7}, family="linear")
NAIVE = CandidateSpec.standalone("naive", family="statistical")
KNN = CandidateSpec.standalone("knn", {"lags": 7}, family="kernel")
HYBRID = CandidateSpec(
    "residual",
    "residual_additive",
    ModelSpec.of("naive"),
    ModelSpec.of("linear"),
    7,
    1.0,
    "historical_oof_1step",
)
COMPOSITIONS = {
    "equal": CandidateSpec.ensemble((LINEAR, NAIVE)),
    "median": CandidateSpec.ensemble((LINEAR, NAIVE, KNN), method="median"),
    "inverse_mse": CandidateSpec.ensemble((LINEAR, NAIVE), method="inverse_mse"),
    "nnls": CandidateSpec.ensemble((LINEAR, NAIVE, KNN), method="nnls"),
    "ensemble_of_hybrids": CandidateSpec.ensemble((HYBRID, LINEAR), method="nnls"),
    "corrected_ensemble": CandidateSpec.corrected(
        CandidateSpec.ensemble((LINEAR, NAIVE), method="inverse_mse"),
        ModelSpec.of("linear"),
        7,
        1.0,
        "historical_oof_1step",
    ),
}


def values(n: int = 220) -> np.ndarray:
    rng = np.random.default_rng(3)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i - 1] + rng.normal()
    t = np.arange(n)
    return 30 + 0.04 * t + 2 * np.sin(2 * np.pi * t / 7) + noise


def run(candidate: CandidateSpec, data: np.ndarray, variant: str = "A") -> dict[str, object]:
    if variant == "B":
        candidate = _as_b(candidate)
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=len(data)), data)
    plan = TemporalSplitter(horizon=5, n_search_folds=2, n_select_folds=2, step=6).plan(frame)
    result = TemporalEvaluator(default_registry(), seasonal_period=7).evaluate(candidate, plan)
    assert all(t.succeeded for t in result.trials), result.failures
    return {t.fold_id: (t.origin_index, t.y_pred, dict(t.diagnostics)) for t in result.trials}


def _as_b(candidate: CandidateSpec) -> CandidateSpec:
    from dataclasses import replace

    components = tuple(_as_b(c) for c in candidate.components)
    if candidate.strategy in ("ensemble", "residual_composite"):
        return replace(candidate, variant="B", components=components)
    return replace(candidate, variant="B")


@pytest.mark.parametrize("variant", ["A", "B"])
@pytest.mark.parametrize("name", sorted(COMPOSITIONS))
def test_future_values_do_not_change_weights_or_forecasts(name: str, variant: str) -> None:
    data = values()
    reference = run(COMPOSITIONS[name], data, variant)
    # Perturb right after the targets of the second origin: two origins are unaffected.
    origins = sorted(origin for origin, _, _ in reference.values())
    cut = origins[1] + 5 + 1
    perturbed = data.copy()
    perturbed[cut:] += 50.0
    changed = run(COMPOSITIONS[name], perturbed, variant)
    compared = 0
    for fold_id, (origin, path, diagnostics) in changed.items():
        if origin + 5 < cut:  # every target of this origin is before the perturbation
            compared += 1
            np.testing.assert_allclose(path, reference[fold_id][1], rtol=1e-9, atol=1e-9)
            assert diagnostics == pytest.approx(reference[fold_id][2])
    assert compared >= 2
