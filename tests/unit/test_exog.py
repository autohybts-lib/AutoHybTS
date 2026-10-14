"""Exogenous covariates: data layer, contract, models and evaluator (ADR-0011)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autohybts.adapters.naive import NaiveForecaster
from autohybts.adapters.sklearn_regression import LagRegressionForecaster
from autohybts.data.exog import ExogFrame, features_from, minimum_lag
from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalSplitter
from autohybts.domain.context import TemporalContext
from autohybts.domain.specs import CandidateSpec, ExogFeature, ModelSpec
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.exceptions import (
    ConfigurationError,
    DataValidationError,
    TemporalLeakageError,
    UnsupportedCapabilityError,
)
from autohybts.registry import default_registry
from autohybts.residuals.correctors import SklearnCorrector
from autohybts.residuals.generation import ResidualSettings, build_residuals, local_context
from autohybts.strategies.additive_residual import ResidualHybridForecaster

N, H = 160, 5


def data(seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    """y depends on x (lagged by H) and on a calendar dummy known in advance."""

    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=N + H, freq="D")
    x = rng.normal(0, 1, N + H)
    holiday = (np.arange(N + H) % 7 == 0).astype(float)
    y = 20 + 3 * np.roll(x, H) + 4 * holiday + rng.normal(0, 0.3, N + H)
    X = pd.DataFrame({"x": x, "holiday": holiday, "noise": rng.normal(0, 1, N + H)}, index=index)
    return pd.Series(y, index=index), X


def frame(future: tuple[str, ...] = ("holiday",)) -> ExogFrame:
    return ExogFrame.from_frame(data()[1], future=future)


# ---------------------------------------------------------------- data layer
def test_frame_validates_its_input() -> None:
    _, X = data()
    with pytest.raises(DataValidationError, match="DatetimeIndex"):
        ExogFrame.from_frame(X.reset_index(drop=True))
    holes = X.copy()
    holes.iloc[3, 0] = np.nan
    with pytest.raises(DataValidationError, match="ausentes"):
        ExogFrame.from_frame(holes)
    with pytest.raises(DataValidationError, match="numéricas"):
        ExogFrame.from_frame(X.assign(x="a"))
    with pytest.raises(ConfigurationError, match="inexistentes"):
        ExogFrame.from_frame(X, future=["temperature"])
    with pytest.raises(DataValidationError, match="crescentes"):
        ExogFrame.from_frame(X.iloc[::-1])
    flags = ExogFrame.from_frame(X.assign(flag=X.holiday > 0))
    assert flags.kind("flag") == "past"


def test_kinds_decide_the_smallest_causal_lag() -> None:
    exog = frame()
    assert exog.kinds == ("past", "future", "past")
    assert exog.default_features(H) == (
        ExogFeature("x", H),
        ExogFeature("holiday", 0),
        ExogFeature("noise", H),
    )
    exog.check([ExogFeature("holiday", 0), ExogFeature("x", H)], H)
    with pytest.raises(ConfigurationError, match="past"):
        exog.check([ExogFeature("x", H - 1)], H)
    assert minimum_lag("A", 14) == 14
    assert minimum_lag("B", 14) == 1


def test_design_shifts_each_feature_by_its_lag() -> None:
    exog = frame()
    timeline = exog.timestamps[:N]
    design = exog.design([ExogFeature("x", 2), ExogFeature("holiday", 0)], timeline, 2)
    raw = data()[1]
    assert np.isnan(design.values[:2, 0]).all()
    assert np.allclose(design.values[2:, 0], raw.x.to_numpy()[: N - 2])
    assert np.allclose(design.values[:, 1], raw.holiday.to_numpy()[:N])
    assert design.max_ahead == 2
    assert design.first_complete() == 2
    assert design.names == ("x_lag2", "holiday")
    with pytest.raises(DataValidationError, match="ausentes"):
        design.rows(0, 5)
    beyond = exog.design([ExogFeature("holiday", 0)], pd.date_range(timeline[0], periods=N + 20), 1)
    assert beyond.max_ahead is None
    assert np.isnan(beyond.values[-1]).all()


def test_fingerprint_follows_values_and_kinds() -> None:
    _, X = data()
    assert frame().fingerprint == frame().fingerprint
    assert frame().fingerprint != frame(future=()).fingerprint
    assert frame().fingerprint != ExogFrame.from_frame(X + 1, future=["holiday"]).fingerprint


def test_features_from_accepts_several_spellings() -> None:
    expected = (ExogFeature("x", 3), ExogFeature("holiday", 0))
    assert features_from({"x": 3, "holiday": 0}) == expected
    assert features_from([("x", 3), ("holiday", 0)]) == expected
    assert features_from(list(expected)) == expected


# ---------------------------------------------------------------- identity
def test_exog_enters_the_candidate_id_only_when_present() -> None:
    plain = CandidateSpec("statistical", "standalone", ModelSpec.of("arima"))
    empty = CandidateSpec("statistical", "standalone", ModelSpec.of("arima"), exog=())
    first = CandidateSpec(
        "exogenous", "standalone", ModelSpec.of("arima"), exog=(ExogFeature("x", 5),)
    )
    swapped = CandidateSpec(
        "exogenous",
        "standalone",
        ModelSpec.of("arima"),
        exog=(ExogFeature("x", 5), ExogFeature("holiday", 0)),
    )
    reordered = CandidateSpec(
        "exogenous",
        "standalone",
        ModelSpec.of("arima"),
        exog=(ExogFeature("holiday", 0), ExogFeature("x", 5)),
    )
    assert plain.candidate_id == empty.candidate_id
    assert "exog" not in plain.describe()
    assert first.candidate_id != plain.candidate_id
    assert swapped.candidate_id == reordered.candidate_id
    assert swapped.name == "arima+exog"
    assert swapped.label.endswith("| exog: holiday, x_lag5")
    with pytest.raises(ConfigurationError, match="não recebe exógenas"):
        CandidateSpec(
            "decomposition",
            "decomposition",
            ModelSpec.of("naive"),
            ModelSpec.of("linear"),
            7,
            options_json='{"method": "ma"}',
            exog=(ExogFeature("x", 5),),
        )


# ---------------------------------------------------------------- contract
def test_models_without_the_capability_refuse_covariates() -> None:
    y = np.arange(30.0)
    with pytest.raises(UnsupportedCapabilityError, match="não aceita"):
        NaiveForecaster().fit(y, np.ones((30, 1)))
    with pytest.raises(DataValidationError, match="30 linhas"):
        LagRegressionForecaster().fit(y, np.ones((29, 1)))


def test_a_model_fitted_with_covariates_needs_them_to_forecast() -> None:
    y, X = data()
    values, rows = y.to_numpy()[:N], X[["holiday"]].to_numpy()
    model = LagRegressionForecaster().fit(values, rows[:N])
    with pytest.raises(ConfigurationError, match="mesmas colunas"):
        model.predict(H, local_context(values, N))
    with pytest.raises(ConfigurationError, match="só 2"):
        model.predict(H, local_context(values, N, rows, ahead=2))
    assert model.predict(H, local_context(values, N, rows, ahead=H)).shape == (H,)


def test_context_checks_rows_and_the_lag_limit() -> None:
    values = np.arange(10.0)
    common = {"series_id": "s", "frequency": "D", "origin_index": 9, "origin_time": None}
    with pytest.raises(ConfigurationError, match="12 linhas"):
        TemporalContext(observed=values, exog=np.ones((10, 1)), exog_ahead=2, **common)
    with pytest.raises(TemporalLeakageError, match="past"):
        TemporalContext(
            observed=values, exog=np.ones((13, 1)), exog_ahead=3, exog_max_ahead=2, **common
        )
    with pytest.raises(ConfigurationError, match="ausentes"):
        TemporalContext(observed=values, exog=np.full((10, 1), np.nan), **common)


def test_multi_step_residual_sources_refuse_covariates() -> None:
    y, X = data()
    values, rows = y.to_numpy()[:N], X[["holiday"]].to_numpy()[:N]
    base = LagRegressionForecaster()
    fitted = base.clone().fit(values, rows)
    with pytest.raises(UnsupportedCapabilityError, match="historical_oof_1step"):
        build_residuals(base, fitted, values, "historical_oof_multistep", ResidualSettings(), rows)
    series = build_residuals(base, fitted, values, "historical_oof_1step", ResidualSettings(), rows)
    assert len(series) > 0


@pytest.mark.parametrize("name", ["arima", "sarima", "linear", "knn", "tree"])
def test_exog_capable_models_use_the_covariates(name: str) -> None:
    if name in ("arima", "sarima"):
        pytest.importorskip("statsmodels")
    params = {"sarima": {"season_length": 7, "order": (1, 0, 0), "seasonal_order": (0, 0, 0)}}
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, N + H)
    y = 10 + 3 * x + rng.normal(0, 0.3, N + H)
    rows = x.reshape(-1, 1)
    registry = default_registry()
    assert registry.entry(name).capabilities.accepts_exog
    plain = registry.make_base(name, params.get(name)).fit(y[:N])
    covariate = registry.make_base(name, params.get(name)).fit(y[:N], rows[:N])
    error_plain = np.abs(plain.predict(H, local_context(y[:N], N)) - y[N:]).mean()
    error_exog = np.abs(covariate.predict(H, local_context(y[:N], N, rows, ahead=H)) - y[N:]).mean()
    # KNN mixes the covariate with 7 lags in its distance: a smaller, still clear gain.
    assert error_exog < (0.8 if name == "knn" else 0.5) * error_plain


def test_hybrid_needs_contexts_that_extend_its_training_covariates() -> None:
    y, X = data()
    values, rows = y.to_numpy(), X[["holiday"]].to_numpy()
    hybrid = ResidualHybridForecaster(
        LagRegressionForecaster(), SklearnCorrector("linear"), window_size=5
    ).fit(values[:N], rows[:N])
    assert hybrid.predict(H, local_context(values[:N], N, rows, ahead=H)).shape == (H,)
    extended = hybrid.predict(1, local_context(values[: N + 2], N + 2, rows, ahead=1))
    assert extended.shape == (1,)
    changed = rows.copy()
    changed[0] += 1
    with pytest.raises(UnsupportedCapabilityError, match="estendem"):
        hybrid.predict(1, local_context(values[: N + 2], N + 2, changed, ahead=1))


# ---------------------------------------------------------------- evaluator
def plan_and_evaluator(future: tuple[str, ...] = ("holiday",)):
    y, X = data()
    series = SeriesFrame.from_arrays(y.index[:N], y.to_numpy()[:N])
    plan = TemporalSplitter(horizon=H, n_search_folds=2, n_select_folds=2, step=10).plan(series)
    evaluator = TemporalEvaluator(default_registry(), exog=ExogFrame.from_frame(X, future))
    return plan, evaluator


def spec(base: str = "linear", variant: str = "A", **extra) -> CandidateSpec:
    features = (ExogFeature("x", H if variant == "A" else 1), ExogFeature("holiday", 0))
    return CandidateSpec(
        "exogenous", "standalone", ModelSpec.of(base), variant=variant, exog=features, **extra
    )


@pytest.mark.parametrize("variant", ["A", "B"])
def test_evaluator_scores_candidates_with_covariates(variant: str) -> None:
    plan, evaluator = plan_and_evaluator()
    plain = evaluator.evaluate(
        CandidateSpec("linear", "standalone", ModelSpec.of("linear"), variant=variant), plan
    )
    covariate = evaluator.evaluate(spec(variant=variant), plan)
    assert all(t.status == "SUCCEEDED" for t in covariate.trials)
    assert covariate.mean("MAE") < plain.mean("MAE")


def test_metrics_keep_the_scale_of_the_full_training_vector() -> None:
    plan, evaluator = plan_and_evaluator()
    origin = plan.origins("search")[0]
    trial = evaluator.evaluate(spec(), plan, "search").trials[0]
    actual, forecast = np.array(trial.y_true), np.array(trial.y_pred)
    train = plan.train_values(origin)
    scale = np.mean(np.abs(np.diff(train)))
    assert trial.metrics["MASE"] == pytest.approx(np.mean(np.abs(actual - forecast)) / scale)


def test_evaluator_skips_bases_without_the_capability_and_refuses_missing_covariates() -> None:
    plan, evaluator = plan_and_evaluator()
    skipped = evaluator.evaluate(spec("naive"), plan)
    assert {t.status for t in skipped.trials} == {"SKIPPED"}
    with pytest.raises(ConfigurationError, match="não tem X"):
        TemporalEvaluator(default_registry()).evaluate(spec(), plan)


def test_protocol_changes_only_when_there_are_covariates() -> None:
    _, evaluator = plan_and_evaluator()
    assert "exog" not in TemporalEvaluator(default_registry()).protocol()
    assert evaluator.protocol()["exog"]["kinds"] == {
        "x": "past",
        "holiday": "future",
        "noise": "past",
    }


@pytest.mark.parametrize("variant", ["A", "B"])
def test_residual_hybrids_pass_the_covariates_to_their_base(variant: str) -> None:
    plan, evaluator = plan_and_evaluator()
    lag = H if variant == "A" else 1
    hybrid = CandidateSpec(
        "exogenous",
        "residual_additive",
        ModelSpec.of("linear"),
        ModelSpec.of("linear"),
        5,
        1.0,
        "historical_oof_1step",
        variant=variant,
        exog=(ExogFeature("x", lag), ExogFeature("holiday", 0)),
    )
    result = evaluator.evaluate(hybrid, plan)
    assert {t.status for t in result.trials} == {"SUCCEEDED"}
    multistep = CandidateSpec(
        "exogenous",
        "residual_additive",
        ModelSpec.of("linear"),
        ModelSpec.of("linear"),
        5,
        1.0,
        "historical_oof_multistep",
        variant=variant,
        exog=(ExogFeature("x", lag),),
    )
    assert {t.status for t in evaluator.evaluate(multistep, plan).trials} == {"SKIPPED"}
