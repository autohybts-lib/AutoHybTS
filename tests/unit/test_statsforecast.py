"""Nixtla StatsForecast adapters (plano de ensembles F1d).

StatsForecast requires pandas<3, so these tests run in an environment with the
``nixtla-stats`` extra (pandas 2.x) and are skipped elsewhere, except the ones about a
missing installation.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from autohybts import AutoHybridExperiment
from autohybts.adapters.statsforecast_models import STATSFORECAST_MODELS, StatsForecastForecaster
from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalSplitter
from autohybts.datasets import get_data
from autohybts.domain.fit_count import count_fits
from autohybts.domain.specs import CandidateSpec
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError
from autohybts.registry import default_registry
from autohybts.residuals.generation import local_context
from autohybts.search.spaces import DEFAULT_BASES, SearchSpace

CLASSES = {"sf_auto_arima": "AutoARIMA", "sf_auto_ets": "AutoETS", "sf_auto_theta": "AutoTheta"}
CLASSES["sf_auto_ces"] = "AutoCES"


def series(n: int = 220) -> np.ndarray:
    rng = np.random.default_rng(0)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.6 * noise[i - 1] + rng.normal()
    t = np.arange(n)
    return 40 + 0.03 * t + 3 * np.sin(2 * np.pi * t / 7) + noise


def test_registered_out_of_the_default_search() -> None:
    registry = default_registry()
    assert set(STATSFORECAST_MODELS) == set(CLASSES)
    for name in STATSFORECAST_MODELS:
        entry = registry.entry(name)
        assert entry.family == "statistical"
        assert entry.capabilities.requires_extra == "nixtla-stats"
        assert name not in DEFAULT_BASES
        assert SearchSpace(seasonal_period=12).default_params(name) == {"season_length": 12}


def test_invalid_settings() -> None:
    with pytest.raises(ConfigurationError):
        StatsForecastForecaster("sf_auto_prophet")
    with pytest.raises(ConfigurationError):
        StatsForecastForecaster("sf_auto_ets", season_length=0)
    with pytest.raises(ConfigurationError, match="não aceita model"):
        default_registry().make_base("sf_auto_ets", {"model": "sf_auto_arima"})


def test_missing_statsforecast_skips_the_trial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "statsforecast.models", None)
    with pytest.raises(UnsupportedCapabilityError, match="nixtla-stats"):
        StatsForecastForecaster("sf_auto_ets").fit(series())
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=220), series())
    plan = TemporalSplitter(horizon=7, n_search_folds=1, n_select_folds=1).plan(frame)
    spec = CandidateSpec.standalone("sf_auto_ets", family="statistical")
    result = TemporalEvaluator(default_registry()).evaluate(spec, plan)
    assert {t.status for t in result.trials} == {"SKIPPED"}


@pytest.mark.parametrize("name", sorted(CLASSES))
def test_forecasts_match_statsforecast_and_stay_frozen(name: str) -> None:
    models = pytest.importorskip("statsforecast.models")
    y = series()
    model = StatsForecastForecaster(name, season_length=7).fit(y[:200])
    reference = getattr(models, CLASSES[name])(season_length=7).fit(y[:200]).predict(7)["mean"]
    np.testing.assert_allclose(model.predict(7, local_context(y, 200)), reference)
    with count_fits() as fits:
        later = model.predict(7, local_context(y, 215))
    assert fits.total == 0
    refit = getattr(models, CLASSES[name])(season_length=7).fit(y[:215]).predict(7)["mean"]
    assert np.abs(later - refit).max() > 0  # frozen parameters, updated states


def test_experiment_and_hybrid_with_statsforecast_bases() -> None:
    pytest.importorskip("statsforecast")
    exp = AutoHybridExperiment(fh=7, fold=1, session_id=0).fit(get_data("synthetic_daily", n=200))
    standalone = exp.create_model("sf_auto_ets", season_length=7)
    assert standalone.evaluation.coverage("MASE") == 1.0
    hybrid = exp.create_hybrid("sf_auto_theta", "mlp", window=7, base_params={"season_length": 7})
    assert hybrid.evaluation.coverage("MASE") == 1.0
