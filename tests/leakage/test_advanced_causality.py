"""Causality of decomposition, combination, multistage and dynamic selection (T05, I12b/I13)."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalSplitter
from autohybts.domain.specs import CandidateSpec, ModelSpec
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.registry import default_registry

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

REGISTRY = default_registry()
LINEAR = ModelSpec.of("linear", {"lags": 7})
SEASONAL = ModelSpec.of("seasonal_naive", {"season_length": 7})
CORRECTOR = ModelSpec.of("linear")


def candidates(variant: str) -> dict[str, CandidateSpec]:
    found = {
        "multistage": CandidateSpec(
            "residual",
            "residual_multistage",
            SEASONAL,
            CORRECTOR,
            7,
            1.0,
            "historical_oof_1step",
            variant,  # type: ignore[arg-type]
            members=(CORRECTOR,),
        ),
        "decomposition_ma": CandidateSpec(
            "decomposition",
            "decomposition",
            LINEAR,
            CORRECTOR,
            7,
            variant=variant,  # type: ignore[arg-type]
            options_json=json.dumps({"method": "ma"}),
        ),
        "combination": CandidateSpec(
            "combination",
            "fixed_combination",
            LINEAR,
            variant=variant,  # type: ignore[arg-type]
            members=(SEASONAL,),
            options_json=json.dumps({"weights": "inverse_mse"}),
        ),
    }
    found["residual_combination"] = CandidateSpec(
        "combination",
        "residual_combination",
        SEASONAL,
        CORRECTOR,
        7,
        1.0,
        "historical_oof_1step",
        variant,  # type: ignore[arg-type]
        members=(ModelSpec.of("knn"),),
        options_json=json.dumps({"weights": "inverse_mse"}),
    )
    found["multistage_per_stage"] = CandidateSpec(
        "residual",
        "residual_multistage",
        SEASONAL,
        CORRECTOR,
        7,
        1.0,
        "historical_oof_1step",
        variant,  # type: ignore[arg-type]
        members=(CORRECTOR,),
        options_json=json.dumps({"stages": [{"window_size": 5, "alpha": 0.5}]}),
    )
    if variant == "B":
        for method, members in (("dref", (ModelSpec.of("knn"),)), ("local", ())):
            found[f"dynamic_{method}"] = CandidateSpec(
                "dynamic",
                "dynamic_selection",
                SEASONAL,
                CORRECTOR,
                7,
                1.0,
                "historical_oof_1step",
                "B",
                members=members,
                options_json=json.dumps({"method": method, "neighbors": 5}),
            )
    if variant == "A" and importlib.util.find_spec("statsmodels") is not None:
        found["decomposition_stl"] = CandidateSpec(
            "decomposition",
            "decomposition",
            LINEAR,
            CORRECTOR,
            7,
            options_json=json.dumps({"method": "stl", "period": 7}),
        )
    return found


CASES = [(variant, name) for variant in ("A", "B") for name in candidates(variant)]


def forecasts(
    candidate: CandidateSpec, values: np.ndarray
) -> dict[str, tuple[int, tuple[float, ...]]]:
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=len(values)), values)
    plan = TemporalSplitter(horizon=7, n_search_folds=2, n_select_folds=2, step=15).plan(frame)
    result = TemporalEvaluator(REGISTRY, seasonal_period=7).evaluate(candidate, plan)
    assert all(t.succeeded for t in result.trials), result.failures
    return {t.fold_id: (t.origin_index, t.y_pred) for t in result.trials}


@pytest.mark.parametrize(("variant", "name"), CASES)
def test_future_perturbation_does_not_change_earlier_forecasts(variant: str, name: str) -> None:
    rng = np.random.default_rng(11)
    noise = np.zeros(220)
    for i in range(1, 220):
        noise[i] = 0.7 * noise[i - 1] + rng.normal()
    values = 20 + 0.03 * np.arange(220) + 3 * np.sin(2 * np.pi * np.arange(220) / 7) + noise
    candidate = candidates(variant)[name]
    reference = forecasts(candidate, values)

    for cut in (185, 205):
        perturbed = values.copy()
        perturbed[cut:] += 40.0
        changed = forecasts(candidate, perturbed)
        for fold_id, (index, path) in changed.items():
            if index < cut:
                if variant == "B":
                    # Rolling: the forecast of target t uses observations up to t - 1 only,
                    # so every target up to the cut must be unchanged.
                    path = path[: max(0, cut - index)]
                    reference_path = reference[fold_id][1][: len(path)]
                else:
                    reference_path = reference[fold_id][1]
                assert path == reference_path, f"{fold_id} saw data after {cut}"
        assert any(path != reference[f][1] for f, (index, path) in changed.items() if index >= cut)
