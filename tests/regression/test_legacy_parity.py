"""Parity with the legacy project on captured outputs (fixtures from capture_legacy*.py).

Covered: the causal moving average and the kurtosis window of the decomposition, the
mean-rank convention (ADR-0004), the paired gain/win share of the selection rule (I1) and
the forecasts of the seven residual hybrids of the replication in variants A and B
(plano de ensembles F1e). The three statistical-ANN hybrids require the optional
StatsForecast environment (pandas 2.x), exercised by the nixtla CI job.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autohybts.domain.results import EvaluationResult, TrialResult
from autohybts.domain.specs import CandidateSpec, ModelSpec
from autohybts.selection.policy import paired_comparison
from autohybts.selection.rank import RankingConvention, structure_key
from autohybts.strategies.decomposition import causal_moving_average, kurtosis_window

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "legacy_composed.json").read_text(encoding="utf-8")
)

STRUCTURES = {
    "arima": CandidateSpec.standalone("arima", family="statistical"),
    "naive": CandidateSpec.standalone("naive", family="statistical"),
    "arima+mlp": CandidateSpec(
        "residual",
        "residual_additive",
        ModelSpec.of("arima"),
        ModelSpec.of("mlp"),
        7,
        1.0,
        "historical_oof_1step",
    ),
}


@pytest.mark.parametrize("case", FIXTURE["decomposition"], ids=lambda c: f"seed{c['seed']}")
def test_moving_average_and_kurtosis_window_match_the_legacy(case: dict[str, object]) -> None:
    y = np.asarray(case["y"], dtype=float)
    np.testing.assert_allclose(
        causal_moving_average(y, int(case["window"])),  # type: ignore[arg-type]
        np.asarray(case["moving_average"]),
        rtol=0,
        atol=1e-12,
    )
    assert kurtosis_window(y, int(case["max_window"])) == case["kurtosis_window"]  # type: ignore[arg-type]


def result(structure: str, scores: dict[str, float], series_id: str = "s") -> EvaluationResult:
    candidate = STRUCTURES[structure]
    trials = tuple(
        TrialResult(
            candidate_id=candidate.candidate_id,
            series_id=series_id,
            fold_id=unit,
            layer="select",
            origin_index=10,
            origin_time=pd.Timestamp("2024-01-01"),
            horizon=1,
            status="SUCCEEDED",
            trial_key=f"{structure}-{unit}",
            metrics={"MASE": score},
        )
        for unit, score in scores.items()
    )
    return EvaluationResult(candidate, "plan", "D", 1, ("MASE",), trials)


def test_mean_ranks_match_the_legacy_convention() -> None:
    rows = pd.DataFrame(FIXTURE["ranking"]["rows"])
    per_series = {
        sid: [
            result(s, {"select-0": float(v)}, sid)
            for s, v in zip(g["structure"], g["score"], strict=True)
        ]
        for sid, g in rows.groupby("m4_id")
    }
    ranked = RankingConvention().rank(per_series, "MASE")
    names = {structure_key(spec): name for name, spec in STRUCTURES.items()}
    ours = {names[r.key]: r.mean_rank for r in ranked}
    assert ours == pytest.approx(FIXTURE["ranking"]["mean_ranks"])


def test_paired_gain_and_win_share_match_the_legacy() -> None:
    rows = pd.DataFrame(FIXTURE["ranking"]["rows"])
    pivot = rows.pivot(index="m4_id", columns="structure", values="score")
    champion = result("arima", pivot["arima"].to_dict())
    challenger = result("arima+mlp", pivot["arima+mlp"].to_dict())
    paired = paired_comparison(champion, challenger, "MASE", 1e-12)
    legacy = FIXTURE["ranking"]["gain_arima_mlp_over_arima"]
    assert paired.median_gain == pytest.approx(legacy["median_gain"])
    assert paired.win_rate == pytest.approx(legacy["win_share"])


# --------------------------------------------------------------------------------------
# The four KNN residual hybrids of the replication (plano de ensembles F1e), built only
# from candidate specs: same lags, same KNN, the legacy's 120-point residual tail and its
# per-column scaling (scale="columns") where the estimator depends on scale.
HYBRIDS = json.loads(
    (Path(__file__).parent / "fixtures" / "legacy_hybrids.json").read_text(encoding="utf-8")
)
KNN_CORRECTOR = ModelSpec.of(
    "knn", {"scale": "columns", "estimator_params": {"n_neighbors": 5, "weights": "distance"}}
)
LEGACY_BASES = {
    "linear_knn_residual": ModelSpec.of("linear", {"lags": 7, "scale": False}),
    "tree_knn_residual": ModelSpec.of(
        "tree",
        {
            "lags": 7,
            "scale": False,
            "random_state": 42,
            "estimator_params": {"min_samples_leaf": 5},
        },
    ),
    "svm_knn_residual": ModelSpec.of(
        "svm",
        {
            "lags": 7,
            "scale": "columns",
            "estimator_params": {"kernel": "rbf", "gamma": "scale", "epsilon": 0.1, "C": 10},
        },
    ),
    "knn_knn_residual": ModelSpec.of(
        "knn",
        {
            "lags": 7,
            "scale": "columns",
            "estimator_params": {"n_neighbors": 5, "weights": "distance"},
        },
    ),
}


# The three statistical-ANN hybrids: StatsForecast bases with the replication's settings
# and the MLP corrector it used (5 logistic units, L-BFGS, min-max scaling per column,
# features ordered by lag).
ANN_CORRECTOR = ModelSpec.of(
    "mlp",
    {
        "scale": "minmax",
        "feature_order": "lag",
        "random_state": 42,
        "estimator_params": {
            "hidden_layer_sizes": [5],
            "activation": "logistic",
            "solver": "lbfgs",
            "max_iter": 2000,
            "alpha": 0.0001,
            "tol": 0.0001,
        },
    },
)
STATISTICAL_BASES = {
    name: ModelSpec.of(
        model,
        {
            "season_length": HYBRIDS["hybrids"][name]["params"]["base_params"]["season_length"],
            "model_params": {
                key: value
                for key, value in HYBRIDS["hybrids"][name]["params"]["base_params"].items()
                if key != "season_length"
            },
        },
    )
    for name, model in (
        ("ets_ann_residual", "sf_auto_ets"),
        ("arima_ann_residual", "sf_auto_arima"),
        ("sarima_ann_residual", "sf_auto_arima"),
    )
    if name in HYBRIDS["hybrids"]
}
CASES = {
    **{name: (base, KNN_CORRECTOR) for name, base in LEGACY_BASES.items()},
    **{name: (base, ANN_CORRECTOR) for name, base in STATISTICAL_BASES.items()},
}


def test_the_fixture_has_the_seven_hybrids() -> None:
    assert set(CASES) == {
        "linear_knn_residual",
        "tree_knn_residual",
        "svm_knn_residual",
        "knn_knn_residual",
        "ets_ann_residual",
        "arima_ann_residual",
        "sarima_ann_residual",
    }


@pytest.mark.parametrize("name", sorted(CASES))
def test_residual_hybrids_match_the_legacy_in_both_variants(name: str) -> None:
    from autohybts.registry import default_registry
    from autohybts.residuals.generation import ResidualSettings, local_context
    from autohybts.strategies.base import build_model

    base, corrector = CASES[name]
    if base.name.startswith("sf_"):
        pytest.importorskip("statsforecast")  # extra nixtla-stats (pandas 2.x)
    y = np.asarray(HYBRIDS["y"], dtype=float)
    train, horizon = int(HYBRIDS["train"]), int(HYBRIDS["horizon"])
    legacy = HYBRIDS["hybrids"][name]
    spec = CandidateSpec(
        "residual",
        "residual_additive",
        base,
        corrector,
        int(legacy["params"]["residual_lags"]),
        1.0,
        "historical_oof_1step",
    )
    settings = ResidualSettings(tail_fraction=0.99, max_tail=int(legacy["params"]["residual_tail"]))
    model, _ = build_model(spec, default_registry(), unit_seed=0, residual_settings=settings)
    model.fit(y[:train])
    variant_a = model.predict(horizon, local_context(y, train))
    variant_b = [model.predict(1, local_context(y, train + k))[0] for k in range(horizon)]
    # Statistical bases + L-BFGS MLP fits differ slightly across numerical backends:
    # Linux CI differed from the captured forecasts by up to 4.24e-7 at a level of 60.
    # Compare these iterative fits to seven relative decimal digits; preserve the
    # stricter absolute-only check for the four KNN hybrids.
    rtol = 1e-7 if base.name.startswith("sf_") else 0
    np.testing.assert_allclose(variant_a, legacy["variant_a"], rtol=rtol, atol=1e-9)
    np.testing.assert_allclose(variant_b, legacy["variant_b"], rtol=rtol, atol=1e-9)
