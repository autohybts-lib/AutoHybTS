"""Variant B: T08 (no refit while streaming), residual timing, causality, T16, UX11."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from autohybts import AutoHybridExperiment
from autohybts.adapters.base import FloatArray, ForecasterBase, Parametrized
from autohybts.adapters.naive import SeasonalNaiveForecaster
from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalPlan, TemporalSplitter
from autohybts.datasets import get_data
from autohybts.domain.specs import CandidateSpec, ModelSpec
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.exceptions import ConfigurationError
from autohybts.forecasting.finalized import FinalizedForecaster
from autohybts.forecasting.rolling_one_step_b import rolling_one_step
from autohybts.registry import Capabilities, default_registry
from autohybts.residuals.generation import local_context
from autohybts.strategies.additive_residual import ResidualHybridForecaster

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


class CountingNaive(ForecasterBase):
    """Naive forecaster that counts every fit, across instances."""

    fits: ClassVar[int] = 0

    def _fit(self, y: FloatArray) -> None:
        type(self).fits += 1

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        return np.full(horizon, history[-1])


class SpyCorrector(Parametrized):
    def __init__(self) -> None:
        self.windows: list[np.ndarray] = []

    def fit(self, windows: npt.ArrayLike, targets: npt.ArrayLike) -> SpyCorrector:
        return self

    def predict_next(self, window: npt.ArrayLike) -> float:
        self.windows.append(np.asarray(window, dtype=float).copy())
        return 0.0

    def clone(self) -> SpyCorrector:
        return self

    def get_params(self) -> dict[str, Any]:
        return {}


def values(n: int = 200, seed: int = 0) -> np.ndarray:
    return get_data("synthetic_daily", n=n, seed=seed).to_numpy()


def plan_for(data: np.ndarray, horizon: int = 7) -> TemporalPlan:
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=len(data)), data)
    return TemporalSplitter(horizon=horizon, n_search_folds=2, n_select_folds=2).plan(frame)


def finalized(model: Any, data: np.ndarray, spec: CandidateSpec) -> FinalizedForecaster:
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=len(data)), data)
    return FinalizedForecaster(spec, model.fit(data), frame, None, {"model_id": "m"}, len(data))


def test_streaming_never_refits() -> None:
    # T08: observe() appends the value and changes the state; fit is never called.
    data = values()
    spec = CandidateSpec.standalone("naive", family="statistical", variant="B")
    final = finalized(CountingNaive(), data[:150], spec)
    CountingNaive.fits = 0
    stream = final.stream()
    for value in data[150:170]:
        ticket = stream.predict_next()
        assert ticket.prediction == stream.model.history.values[-1]  # naive: last known value
        stream.observe(value)
    assert CountingNaive.fits == 0
    assert stream.model.training_cutoff == final.training_cutoff
    assert stream.model.last_observation == final.training_cutoff + pd.Timedelta(days=20)
    assert len(stream.log()) == 20


def test_evaluation_in_b_fits_once_per_origin() -> None:
    registry = default_registry()
    registry.register_base("counting", lambda p: CountingNaive(), family="statistical")
    plan = plan_for(values())
    CountingNaive.fits = 0
    candidate = CandidateSpec.standalone("counting", family="statistical", variant="B")
    result = TemporalEvaluator(registry).evaluate(candidate, plan)
    assert CountingNaive.fits == len(plan.origins())
    # Rolling naive one step ahead: each forecast is the previous actual value.
    for trial, origin in zip(result.trials, plan, strict=True):
        actual = plan.target_values(origin)
        previous = np.concatenate([[plan.history.values[origin.origin_index]], actual[:-1]])
        assert np.allclose(trial.y_pred, previous)


def test_observed_residual_enters_only_at_the_next_step() -> None:
    data = values()
    spy = SpyCorrector()
    hybrid = ResidualHybridForecaster(SeasonalNaiveForecaster(7), spy, window_size=3)
    spec = CandidateSpec(
        "residual",
        "residual_additive",
        ModelSpec.of("seasonal_naive"),
        ModelSpec.of("linear"),
        3,
        1.0,
        "historical_oof_1step",
        variant="B",
    )
    final = finalized(hybrid, data[:150], spec)
    stream = final.stream()
    first = stream.predict_next()
    window_before = spy.windows[-1].copy()
    stream.observe(data[150])
    stream.predict_next()
    window_after = spy.windows[-1]
    # The window shifted by one and its newest residual is y - (base forecast emitted).
    assert np.allclose(window_after[:-1], window_before[1:])
    assert window_after[-1] == pytest.approx(data[150] - first.base_prediction)


def test_rolling_forecasts_ignore_values_not_yet_revealed() -> None:
    data = values()
    model = SeasonalNaiveForecaster(7).fit(data[:150])
    reference = rolling_one_step(model, data[:150], data[150:170])
    changed = data.copy()
    changed[160:] += 100.0
    other = rolling_one_step(model, changed[:150], changed[150:170])
    assert np.array_equal(reference[:11], other[:11])  # forecast of 160 uses values up to 159
    assert not np.array_equal(reference[11:], other[11:])


def test_b_and_a_differ_beyond_the_first_step() -> None:
    # Oracle 6: same candidate, different protocols; at step 1 they coincide.
    plan = plan_for(values(), horizon=7)
    evaluator = TemporalEvaluator(default_registry())
    a = evaluator.evaluate(CandidateSpec.standalone("linear", family="linear"), plan)
    b = evaluator.evaluate(CandidateSpec.standalone("linear", family="linear", variant="B"), plan)
    for ta, tb in zip(a.trials, b.trials, strict=True):
        assert ta.y_pred[0] == pytest.approx(tb.y_pred[0])
        assert ta.y_pred[1:] != tb.y_pred[1:]
    assert b.mean("MAE") < a.mean("MAE")  # revealing actuals helps on this AR series


def test_dynamic_selection_is_refused_in_variant_a() -> None:
    # T16.
    fields = (
        "residual",
        "dynamic_selection",
        ModelSpec.of("naive"),
        ModelSpec.of("linear"),
        7,
        1.0,
        "historical_oof_1step",
    )
    with pytest.raises(ConfigurationError, match="variante B"):
        CandidateSpec(*fields, variant="A")
    candidate = CandidateSpec(*fields, variant="B")
    result = TemporalEvaluator(default_registry()).evaluate(candidate, plan_for(values()))
    assert {t.status for t in result.trials} == {"SKIPPED"}
    assert "I13" in next(iter(result.failures.values()))


def test_models_without_state_update_are_skipped_in_b() -> None:
    registry = default_registry()
    registry.register_base(
        "frozen",
        lambda p: CountingNaive(),
        family="statistical",
        capabilities=Capabilities(online_state_update=False),
    )
    candidate = CandidateSpec.standalone("frozen", family="statistical", variant="B")
    result = TemporalEvaluator(registry).evaluate(candidate, plan_for(values()))
    assert {t.status for t in result.trials} == {"SKIPPED"}


def test_update_state_extends_history_without_refit() -> None:
    # UX11.
    data = values()
    spec = CandidateSpec.standalone("naive", family="statistical")
    final = finalized(CountingNaive(), data[:150], spec)
    later = pd.Series(data[150:155], index=pd.date_range(final.training_cutoff, periods=6)[1:])
    updated = final.update_state(later)
    assert updated.model is final.model
    assert updated.training_cutoff == final.training_cutoff
    assert updated.last_observation == later.index[-1]
    assert updated.predict(1)["y_pred"].iloc[0] == data[154]
    with pytest.raises(ConfigurationError, match="nunca reajusta"):
        final.update_state(later, refit=True)
    with pytest.raises(ConfigurationError, match="devem começar"):
        final.update_state(later.iloc[1:])


def test_stream_enforces_predict_then_observe() -> None:
    data = values()
    final = finalized(
        CountingNaive(), data[:150], CandidateSpec.standalone("naive", family="statistical")
    )
    stream = final.stream()
    with pytest.raises(ConfigurationError, match="predict_next"):
        stream.observe(1.0)
    stream.predict_next()
    with pytest.raises(ConfigurationError, match="Observe"):
        stream.predict_next()


def test_experiment_in_variant_b() -> None:
    y = get_data("synthetic_daily", n=260)
    exp = AutoHybridExperiment(
        fh=7,
        variant="B",
        test_size=20,
        fold=2,
        budget="smoke",
        base_models=["naive", "seasonal_naive", "linear"],
        correctors=["linear"],
        residual_windows=[7],
        alphas=[1.0],
    ).fit(y)
    result = exp.compare_models()
    assert result.best.variant == "B"
    final = exp.finalize_model(result.best)
    external = exp.evaluate_external(final)
    assert list(external.predictions["horizon"].unique()) == [1]
    stream = final.stream()
    for value in y.iloc[-20:]:
        stream.predict_next()
        stream.observe(value)
    assert np.allclose(stream.log()["y_pred"], external.predictions["y_pred"])
    assert len(exp.predict_model(final, fh=1).predictions) == 1
    with pytest.raises(ConfigurationError, match="stream"):
        exp.predict_model(final, fh=3)
    with pytest.raises(ConfigurationError, match="variant"):
        AutoHybridExperiment(variant="C").fit(y)


def test_hybrid_context_extension_matches_stream() -> None:
    data = values()
    model = ResidualHybridForecaster(SeasonalNaiveForecaster(7), SpyCorrector(), window_size=3)
    model.fit(data[:150])
    direct = rolling_one_step(model, data[:150], data[150:160])
    stepwise = [model.predict(1, local_context(data, t))[0] for t in range(150, 160)]
    assert np.allclose(direct, stepwise)
