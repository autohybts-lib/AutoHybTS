"""Strategy comparison report: complexity, Pareto front and short model names."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from autohybts.domain.results import EvaluationResult, TrialResult
from autohybts.domain.specs import CandidateSpec, ModelSpec
from autohybts.exceptions import ConfigurationError
from autohybts.reporting.strategies import (
    complexity,
    pareto_front,
    strategy_plot,
    strategy_summary,
    strategy_table,
)

ARIMA = ModelSpec.of("arima")
MLP = ModelSpec.of("mlp")
LINEAR = ModelSpec.of("linear")
STANDALONE = CandidateSpec("statistical", "standalone", ARIMA)
HYBRID = CandidateSpec("residual", "residual_additive", ARIMA, MLP, 7, 1.0, "historical_oof_1step")
STAGED = CandidateSpec(
    "residual",
    "residual_multistage",
    ARIMA,
    MLP,
    7,
    1.0,
    "historical_oof_1step",
    members=(LINEAR,),
)
DECOMPOSITION = CandidateSpec(
    "decomposition", "decomposition", LINEAR, MLP, 7, options_json='{"method": "ma"}'
)
COMBINATION = CandidateSpec(
    "combination",
    "fixed_combination",
    ARIMA,
    members=(LINEAR,),
    options_json=json.dumps({"weights": "equal"}),
)


def test_short_names_read_like_the_structure() -> None:
    assert STANDALONE.name == "arima"
    assert HYBRID.name == "arima-mlp"
    assert STAGED.name == "arima-mlp-linear"
    assert DECOMPOSITION.name == "ma(linear-mlp)"
    assert COMBINATION.name == "comb(arima,linear)"


def test_complexity_counts_fitted_models() -> None:
    assert [complexity(c) for c in (STANDALONE, HYBRID, STAGED, DECOMPOSITION, COMBINATION)] == [
        1,
        2,
        3,
        2,
        2,
    ]


def test_pareto_front_keeps_only_non_dominated_points() -> None:
    cost = [1.0, 2.0, 3.0, 2.0, 0.5]
    error = [3.0, 2.0, 1.0, 2.5, np.inf]
    assert pareto_front(cost, error).tolist() == [True, True, True, False, False]
    # Ties on both axes do not dominate each other.
    assert pareto_front([1.0, 1.0], [2.0, 2.0]).tolist() == [True, True]


def result(candidate: CandidateSpec, error: float, seconds: float) -> EvaluationResult:
    trial = TrialResult(
        candidate_id=candidate.candidate_id,
        series_id="s",
        fold_id="search-0",
        layer="search",
        origin_index=10,
        origin_time=pd.Timestamp("2024-01-01"),
        horizon=1,
        status="SUCCEEDED",
        trial_key="k",
        metrics={"MASE": error},
        fit_seconds=seconds,
    )
    return EvaluationResult(candidate, "plan", "D", 1, ("MASE",), (trial,))


def test_table_summary_and_plots() -> None:
    results = [
        result(STANDALONE, 1.0, 0.1),
        result(HYBRID, 0.8, 1.0),
        result(STAGED, 0.9, 5.0),
        result(COMBINATION, 0.7, 2.0),
    ]
    table = strategy_table(results, "MASE")
    assert table["name"].tolist() == [
        "comb(arima,linear)",
        "arima-mlp",
        "arima-mlp-linear",
        "arima",
    ]
    flags = dict(zip(table["name"], table["pareto_cost"], strict=True))
    assert flags == {
        "comb(arima,linear)": True,
        "arima-mlp": True,
        "arima-mlp-linear": False,  # dearer and worse than arima-mlp
        "arima": True,
    }
    summary = strategy_summary(table)
    assert summary.index[0] == "fixed_combination"
    assert summary.loc["residual_additive", "best_model"] == "arima-mlp"
    with pytest.raises(ConfigurationError):
        strategy_plot("bubbles", table, "MASE")
    if importlib.util.find_spec("matplotlib") is not None:
        for kind, x in (("pareto", "cost"), ("pareto", "complexity"), ("strategies", "cost")):
            artifact = strategy_plot(kind, table, "MASE", x)  # type: ignore[arg-type]
            assert artifact.provenance["n_candidates"] == 4
