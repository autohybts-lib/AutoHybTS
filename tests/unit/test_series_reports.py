"""Stationarity report, decomposition, Ljung-Box, MAPE/MdAPE and Kruskal-Wallis + Dunn."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from autohybts import AutoHybridExperiment
from autohybts.data.characterization import stationarity_table
from autohybts.datasets import get_data
from autohybts.evaluation.metrics import METRICS

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def walk(n: int = 300, seed: int = 0) -> np.ndarray:
    return 50 + np.cumsum(np.random.default_rng(seed).normal(size=n))


@pytest.fixture(scope="module")
def experiment() -> AutoHybridExperiment:
    return AutoHybridExperiment(fh=14, fold=3).fit(get_data("synthetic_daily"))


def test_stationarity_table_reports_both_rules_and_their_consensus() -> None:
    table = stationarity_table(walk())
    assert list(table.index) == [0, 1, 2]
    assert not table.loc[0, "consensus"]
    assert table.loc[1, "consensus"]
    assert table.attrs == {"d_adf": 1, "d_kpss": 1, "d_consensus": 1}
    noise = stationarity_table(np.random.default_rng(1).normal(size=300))
    assert noise.attrs["d_consensus"] == 0


def test_stationarity_report_on_the_development_history(experiment: AutoHybridExperiment) -> None:
    report = experiment.stationarity_report()
    assert report.index.names == ["series_id", "d"]
    assert set(report.attrs) == {"d_adf", "d_kpss", "d_consensus"}
    assert report["n"].iloc[0] == len(experiment.history_)


@pytest.mark.parametrize("method", ["stl", "classical"])
def test_decompose_adds_up(experiment: AutoHybridExperiment, method: str) -> None:
    parts = experiment.decompose(method)
    assert list(parts.columns) == ["observed", "trend", "seasonal", "resid"]
    assert len(parts) == len(experiment.history_)
    valid = parts.dropna()
    total = valid["trend"] + valid["seasonal"] + valid["resid"]
    assert np.allclose(total, valid["observed"])


def test_decompose_refuses_an_unknown_method(experiment: AutoHybridExperiment) -> None:
    with pytest.raises(ValueError, match="Decomposição desconhecida"):
        experiment.decompose("x11")


def test_ljung_box_finds_structure_left_by_a_naive_base(experiment: AutoHybridExperiment) -> None:
    table = experiment.residual_diagnostics("naive")
    assert list(table.index) == [10, 14]
    assert (table["lb_pvalue"] < 0.05).all()
    assert table.attrs["base"] == "naive"
    assert table.attrs["n_residuals"] == len(experiment.history_) // 2


def test_mape_and_mdape_are_undefined_with_zeros() -> None:
    actual, forecast, train = [2.0, 4.0, 5.0], [1.0, 4.0, 6.0], [1.0, 2.0, 3.0]
    assert METRICS["MAPE"](actual, forecast, train) == pytest.approx((0.5 + 0 + 0.2) / 3)
    assert METRICS["MdAPE"](actual, forecast, train) == pytest.approx(0.2)
    assert np.isnan(METRICS["MAPE"]([0.0, 1.0], [1.0, 1.0], train))
    assert np.isnan(METRICS["MdAPE"]([0.0, 1.0], [1.0, 1.0], train))


def test_kruskal_dunn_compares_every_pair(experiment: AutoHybridExperiment) -> None:
    comparison = experiment.cross_validate(["naive", "seasonal_naive", "linear"], folds=6, step=7)
    out = comparison.kruskal()
    assert out["n_models"] == 3
    assert 0 <= out["p_value"] <= 1
    assert len(out["dunn"]) == 3
    assert (out["dunn"]["p_holm"] >= out["dunn"]["p_value"]).all()
