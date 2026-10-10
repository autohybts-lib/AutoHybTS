"""Parity with the legacy project on captured outputs (I1; fixture from capture_legacy.py).

Covered: the causal moving average and the kurtosis window of the decomposition, the
mean-rank convention (ADR-0004) and the paired gain/win share of the selection rule.
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
