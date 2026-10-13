"""Characterization through the experiment: table, plots and the replication's seasonality rule."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from autohybts import AutoHybridExperiment
from autohybts.datasets import get_data
from autohybts.exceptions import ConfigurationError
from autohybts.reporting.plots import _pacf

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(scope="module")
def experiment() -> AutoHybridExperiment:
    return AutoHybridExperiment(fh=14, fold=2).fit(get_data("synthetic_daily"))


def test_table_has_both_rules_and_the_category(experiment: AutoHybridExperiment) -> None:
    table = experiment.characterize(periods=[7, 30])
    row = table.iloc[0]
    assert {"diff_order", "stationary", "fs_7", "fs_30", "dominant_period", "category"} <= set(
        table.columns
    )
    assert row["category"] in ("stationary", "nonseasonal", "seasonal")
    if not row["stationary"]:
        assert row["dominant_period"] == 7


def test_panel_has_one_row_per_series() -> None:
    y = get_data("synthetic_daily")
    exp = AutoHybridExperiment(fh=14, fold=2).fit({"a": y, "b": get_data("white_noise_daily")})
    table = exp.characterize()
    assert list(table.index) == ["a", "b"]
    assert table.loc["b", "category"] == "stationary"


@pytest.mark.parametrize("kind", ["stationarity", "acf_pacf", "decomposition"])
def test_development_plots(experiment: AutoHybridExperiment, kind: str) -> None:
    pytest.importorskip("matplotlib")
    artifact = experiment.plot_data(kind)
    assert artifact.kind == kind
    assert artifact.source_split == "development"
    assert artifact.provenance["observations"] == len(experiment.history_)


def test_characterization_plot(experiment: AutoHybridExperiment) -> None:
    pytest.importorskip("matplotlib")
    artifact = experiment.plot_characterization()
    assert artifact.kind == "characterization"
    assert artifact.provenance == {"series": 1}


def test_pacf_matches_statsmodels_ywm() -> None:
    from statsmodels.tsa.stattools import pacf

    rng = np.random.default_rng(0)
    values = np.zeros(300)
    for i in range(2, 300):
        values[i] = 0.6 * values[i - 1] - 0.3 * values[i - 2] + rng.normal()
    expected = pacf(values, nlags=20, method="ywm")[1:]
    assert np.allclose(_pacf(values, 20), expected, atol=1e-10)


def test_strength_rule_drops_the_period_of_a_stationary_series() -> None:
    exp = AutoHybridExperiment(fh=14, seasonal_period="auto", seasonality_test="strength").fit(
        get_data("white_noise_daily")
    )
    assert exp.seasonality_ is not None
    assert exp.seasonality_["category"].iloc[0] == "stationary"
    assert exp.search_period_ == 1


def test_unknown_seasonality_test_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="seasonality_test"):
        AutoHybridExperiment(fh=14, seasonality_test="qs").fit(get_data("synthetic_daily"))
