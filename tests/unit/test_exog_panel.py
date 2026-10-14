"""Panels of series with exogenous covariates (ADR-0011)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autohybts import AutoHybridExperiment, PanelForecaster
from autohybts.data.exog import ExogPanel
from autohybts.data.panel import PanelFrame, plan_panel
from autohybts.data.splitter import TemporalSplitter
from autohybts.domain.specs import CandidateSpec, ExogFeature, ModelSpec
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.exceptions import ConfigurationError, DataValidationError
from autohybts.registry import default_registry

N, FH, EXTRA = 200, 5, 10
SETTINGS = {
    "fh": FH,
    "fold": 2,
    "budget": "small",
    "base_models": ["naive", "linear"],
    "correctors": ["linear"],
    "exog_future": ["feriado"],
}


def panel(seed: int = 0) -> tuple[dict[str, pd.Series], dict[str, pd.DataFrame]]:
    """Three series driven by their own x five days earlier and their own holiday."""

    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=N + EXTRA, freq="D")
    ys, Xs = {}, {}
    for k, sid in enumerate(["a", "b", "c"]):
        x = rng.normal(0, 1, N + EXTRA)
        holiday = (np.arange(N + EXTRA) % 7 == k).astype(float)
        y = 30 + 4 * np.roll(x, FH) + 6 * holiday + np.cumsum(rng.normal(0, 0.2, N + EXTRA))
        ys[sid] = pd.Series(y[:N], index=index[:N])
        Xs[sid] = pd.DataFrame({"x": x, "feriado": holiday}, index=index)
    return ys, Xs


def long_format(
    ys: dict[str, pd.Series], Xs: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = pd.concat([s.to_frame("v").assign(serie=k, data=s.index) for k, s in ys.items()])
    X = pd.concat([f.assign(serie=k, data=f.index) for k, f in Xs.items()])
    return y.reset_index(drop=True), X.reset_index(drop=True)


@pytest.fixture(scope="module")
def compared() -> tuple[AutoHybridExperiment, object]:
    ys, Xs = panel()
    exp = AutoHybridExperiment(**SETTINGS).fit(ys, Xs)  # type: ignore[arg-type]
    return exp, exp.compare_models()


def test_mapping_and_long_inputs_give_the_same_covariates() -> None:
    ys, Xs = panel()
    from_mapping = AutoHybridExperiment(**SETTINGS).fit(ys, Xs)  # type: ignore[arg-type]
    y_long, X_long = long_format(ys, Xs)
    from_long = AutoHybridExperiment(**SETTINGS).fit(  # type: ignore[arg-type]
        y_long, X_long, series_col="serie", date_col="data", target="v"
    )
    assert isinstance(from_mapping.exog_, ExogPanel)
    assert from_long.exog_.fingerprint == from_mapping.exog_.fingerprint
    assert from_mapping.evaluator_.protocol()["exog"]["series"] == 3


def test_panel_covariates_are_validated() -> None:
    ys, Xs = panel()
    with pytest.raises(DataValidationError, match=r"\['c'\]"):
        AutoHybridExperiment(**SETTINGS).fit(ys, {k: Xs[k] for k in ("a", "b")})  # type: ignore[arg-type]
    different = {**Xs, "c": Xs["c"].rename(columns={"x": "z"})}
    with pytest.raises(DataValidationError, match="mesmas"):
        AutoHybridExperiment(**SETTINGS).fit(ys, different)  # type: ignore[arg-type]
    short = {**Xs, "b": Xs["b"].iloc[20:]}
    with pytest.raises(DataValidationError, match="'b'"):
        AutoHybridExperiment(**SETTINGS).fit(ys, short)  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="dicionário"):
        AutoHybridExperiment(**SETTINGS).fit(ys, [Xs["a"]])  # type: ignore[arg-type]


def test_panel_selects_finalizes_and_forecasts_with_covariates(compared) -> None:
    exp, result = compared
    decision = result.decisions.set_index("family").loc["exogenous"]
    assert bool(decision.accepted)
    final = exp.finalize_model(result.best)
    assert isinstance(final, PanelForecaster)
    assert all(member.exog is not None for member in final.models.values())
    automatic = exp.predict_model(final).predictions
    _, Xs = panel()
    explicit = exp.predict_model(final, X_future=Xs).predictions
    pd.testing.assert_frame_equal(automatic, explicit)
    assert set(automatic.series_id) == {"a", "b", "c"}
    with pytest.raises(ConfigurationError, match="dicionário"):
        exp.predict_model(final, X_future=Xs["a"])
    external = exp.evaluate_external(final).metrics
    assert {"a", "b", "c"} <= set(external.index.get_level_values("series_id"))


def test_selection_votes_across_series() -> None:
    ys, Xs = panel()
    exp = AutoHybridExperiment(
        **{**SETTINGS, "budget": "smoke"}, exog_selection=["granger", "lasso"]
    ).fit(ys, Xs)  # type: ignore[arg-type]
    report = exp.exog_report()
    assert set(report.series_id) == {"a", "b", "c"}
    assert ExogFeature("x", FH) in exp.exog_sets_[0]


def test_each_series_reads_only_its_own_covariates_and_past() -> None:
    ys, Xs = panel()
    splitter = TemporalSplitter(horizon=FH, n_search_folds=2, n_select_folds=2, step=10)
    plan = plan_panel(splitter, PanelFrame.from_dict(ys))
    candidate = CandidateSpec(
        "exogenous",
        "standalone",
        ModelSpec.of("linear"),
        exog=(ExogFeature("x", FH), ExogFeature("feriado", 0)),
    )

    def forecasts(X: dict[str, pd.DataFrame]) -> dict[str, tuple[int, tuple[float, ...]]]:
        evaluator = TemporalEvaluator(
            default_registry(), exog=ExogPanel.from_input(X, list(ys), future=["feriado"])
        )
        result = evaluator.evaluate(candidate, plan)
        return {t.unit_id: (t.origin_index, t.y_pred) for t in result.trials}

    reference = forecasts(Xs)
    other = {**Xs, "a": Xs["a"] + 100.0}  # another series' covariates
    changed = forecasts(other)
    others = [unit for unit in reference if unit.startswith(("b:", "c:"))]
    assert others
    for unit in others:
        assert np.allclose(changed[unit][1], reference[unit][1])
    cut = min(origin for u, (origin, _) in reference.items() if u.startswith("b:")) + 1
    later = {**Xs, "b": Xs["b"].copy()}
    later["b"].iloc[cut:, 0] += 100.0  # column "x" (past) of series b after the cut
    after = forecasts(later)
    early = [u for u, (origin, _) in reference.items() if u.startswith("b:") and origin < cut]
    assert early
    for unit in early:
        assert np.allclose(after[unit][1], reference[unit][1])
