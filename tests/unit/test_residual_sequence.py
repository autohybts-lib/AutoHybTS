"""Residual sequence correctors (plano de ensembles §4.2, F1e)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autohybts.adapters.naive import SeasonalNaiveForecaster
from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalSplitter
from autohybts.domain.fit_count import count_fits
from autohybts.domain.specs import CandidateSpec, ModelSpec
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError
from autohybts.registry import default_registry
from autohybts.residuals.generation import local_context
from autohybts.residuals.sequence import ResidualSequenceForecaster, SequenceCorrector
from autohybts.search.spaces import SearchSpace
from autohybts.strategies.additive_residual import ResidualHybridForecaster

pytest.importorskip("statsmodels")
SEQUENCES = ("arima_sequence", "exponential_smoothing_sequence")


def series(n: int = 220, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + rng.normal()
    return 30 + 3 * np.sin(2 * np.pi * np.arange(n) / 7) + noise


def fitted_hybrid(y: np.ndarray) -> ResidualHybridForecaster:
    corrector = SequenceCorrector("exponential_smoothing")
    return ResidualHybridForecaster(SeasonalNaiveForecaster(7), corrector, window_size=7).fit(y)


def test_correction_is_the_wrapped_model_on_the_residual_series() -> None:
    from autohybts.adapters.statsmodels_ets import ExponentialSmoothingForecaster

    y = series()
    model = fitted_hybrid(y)
    assert model.is_sequence
    residuals = model.residuals_.residuals[model.residuals_.targets >= model.sequence_start_]
    wrapped = ExponentialSmoothingForecaster(trend=False).fit(residuals)
    expected = wrapped.predict(5, local_context(np.asarray(residuals), residuals.size))
    base_path, correction = model.predict_components(5, y)
    np.testing.assert_allclose(correction, expected)
    np.testing.assert_allclose(model.predict(5, local_context(y, y.size)), base_path + correction)


def test_later_context_extends_the_residuals_without_refitting() -> None:
    full = series(240)
    model = fitted_hybrid(full[:220])
    with count_fits() as fits:
        later = model.predict(3, local_context(full, 235))
    assert fits.total == 0
    assert later.shape == (3,)
    assert not np.allclose(later, model.predict(3, local_context(full, 220)))


@pytest.mark.parametrize("name", SEQUENCES)
def test_registered_sequence_correctors(name: str) -> None:
    corrector = default_registry().make_corrector(name)
    assert isinstance(corrector, ResidualSequenceForecaster)
    entry = default_registry().entry(name, "corrector")
    assert entry.capabilities.multi_step
    assert entry.capabilities.requires_extra == "stats"


def test_window_methods_are_refused() -> None:
    corrector = SequenceCorrector("arima")
    with pytest.raises(UnsupportedCapabilityError, match="sequência"):
        corrector.fit(np.ones((3, 2)), np.ones(3))
    with pytest.raises(UnsupportedCapabilityError, match="sequência"):
        corrector.predict_next(np.ones(2))
    with pytest.raises(ConfigurationError):
        SequenceCorrector("knn")


def plan():  # type: ignore[no-untyped-def]
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=220), series())
    return TemporalSplitter(horizon=7, n_search_folds=1, n_select_folds=1).plan(frame)


def test_window_only_strategies_skip_the_trial() -> None:
    decomposition = CandidateSpec(
        "decomposition",
        "decomposition",
        ModelSpec.of("linear"),
        ModelSpec.of("exponential_smoothing_sequence"),
        7,
        options_json='{"method":"ma"}',
    )
    result = TemporalEvaluator(default_registry(), seasonal_period=7).evaluate(
        decomposition, plan()
    )
    assert {t.status for t in result.trials} == {"SKIPPED"}


def test_search_proposes_one_window_and_no_window_only_structures() -> None:
    registry = default_registry()
    space = SearchSpace(
        correctors=("linear", "exponential_smoothing_sequence"),
        residual_windows=(7, 14),
        alphas=(1.0,),
        decomposition_methods=("ma",),
    )
    base = CandidateSpec.standalone("seasonal_naive", family="statistical")
    hybrids = space.residual_proposals([base], registry)
    windows = {
        p.candidate.corrector.name: set() for p in hybrids if p.candidate.corrector is not None
    }
    for proposal in hybrids:
        assert proposal.candidate.corrector is not None
        windows[proposal.candidate.corrector.name].add(proposal.candidate.window_size)
    assert windows == {"linear": {7, 14}, "exponential_smoothing_sequence": {7}}
    decompositions = space.advanced_proposals([base], registry)
    names = {p.candidate.corrector.name for p in decompositions if p.candidate.corrector}
    assert "exponential_smoothing_sequence" not in names
