"""Ensembles through the public API (plano de ensembles F2): create, search, select, persist."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from autohybts import AutoHybridExperiment, load_model, save_model
from autohybts.datasets import get_data
from autohybts.exceptions import ConfigurationError

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

FAST = {
    "fh": 7,
    "fold": 2,
    "budget": "small",
    "base_models": ("naive", "seasonal_naive", "linear", "knn"),
    "correctors": ("linear",),
    "residual_windows": (7,),
    "alphas": (1.0,),
}


@pytest.fixture(scope="module")
def experiment() -> AutoHybridExperiment:
    return AutoHybridExperiment(**FAST).fit(get_data("synthetic_daily", n=260))


def test_create_ensemble_and_hybrid_on_it(experiment: AutoHybridExperiment) -> None:
    linear = experiment.create_model("linear")
    ensemble = experiment.create_ensemble(["seasonal_naive", linear], method="nnls")
    assert ensemble.candidate.strategy == "ensemble"
    assert ensemble.evaluation.coverage("MASE") == 1.0
    corrected = experiment.create_hybrid(ensemble, "linear")
    assert corrected.candidate.strategy == "residual_composite"
    assert corrected.evaluation.coverage("MASE") == 1.0
    plain = experiment.create_hybrid(linear, "knn")  # a single model: the usual hybrid
    assert plain.candidate.strategy == "residual_additive"
    with pytest.raises(ConfigurationError, match="base composta"):
        experiment.create_hybrid(ensemble, "linear", extra_stages=("knn",))


@pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None, reason="extra `plot` not installed"
)
def test_structure_plot_names_the_components(experiment: AutoHybridExperiment) -> None:
    ensemble = experiment.create_ensemble(["seasonal_naive", "linear"], method="nnls")
    corrected = experiment.create_hybrid(ensemble, "linear")
    for result in (ensemble, corrected):
        figure = experiment.plot_model(result, "model_structure").figure
        texts = " ".join(text.get_text() for text in figure.axes[0].texts)
        assert "seasonal_naive" in texts
        assert "linear" in texts
        assert "composite" not in texts  # the placeholder base is never drawn
    title = experiment.plot_model(ensemble, "model_structure").figure.axes[0].get_title()
    assert "nnls(ŷ₁, …, ŷ₂)" in title


def test_search_with_ensembles_and_their_place_in_the_chain() -> None:
    exp = AutoHybridExperiment(**FAST).fit(get_data("synthetic_daily", n=260))
    plain = exp.compare_models()
    assert "ensemble" not in exp.policy_.family_order  # protocol unchanged without them
    assert "ensemble_methods" not in plain.snapshot["space"]
    result = exp.compare_models(include=("standalone", "residual", "ensemble"))
    order = exp.policy_.family_order
    assert order.index("ensemble") == order.index("dynamic") - 1
    ledger = result.search.ledger_frame()
    kinds = set(ledger.loc[ledger["family"] == "ensemble", "strategy"])
    assert kinds == {"ensemble", "residual_composite"}
    assert "ensemble" in set(result.decisions["family"])  # the family challenged the champion
    assert result.search.fit_count > result.search.unit_count  # hybrids fit more than once


def test_ensemble_champion_finalizes_predicts_and_persists(
    experiment: AutoHybridExperiment, tmp_path: Path
) -> None:
    ensemble = experiment.create_ensemble(["linear", "knn", "seasonal_naive"], method="median")
    final = experiment.finalize_model(ensemble)
    expected = final.predict(7)["y_pred"].to_numpy()
    assert np.isfinite(expected).all()
    path = save_model(final, tmp_path / "ensemble")
    np.testing.assert_allclose(load_model(path, trusted=True).predict(7)["y_pred"], expected)
    code = (
        "import json, sys; from autohybts import load_model; "
        "m = load_model(sys.argv[1], trusted=True); "
        "print(json.dumps(m.predict(7)['y_pred'].tolist()))"
    )
    output = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, text=True, check=True
    ).stdout
    np.testing.assert_allclose(json.loads(output), expected, rtol=0, atol=1e-12)
    assert experiment.evaluate_external(final).metrics["MASE"].notna().all()


def test_ensembles_in_variant_b_and_in_a_panel() -> None:
    b = AutoHybridExperiment(**FAST, variant="B").fit(get_data("synthetic_daily", n=260))
    ensemble = b.create_ensemble(["linear", "seasonal_naive"], method="inverse_mse")
    assert ensemble.evaluation.coverage("MASE") == 1.0
    panel = {
        "a": get_data("synthetic_daily", n=220, seed=1),
        "b": get_data("synthetic_daily", n=220, seed=2),
    }
    exp = AutoHybridExperiment(**FAST).fit(panel)
    result = exp.compare_models(include=("standalone", "ensemble"))
    assert set(result.selection.champion_per_series) == {"a", "b"}
    nested = exp.nested_cross_validate(outer_folds=2, baselines=(), budget="smoke")
    assert nested.include == ("standalone", "ensemble")
    assert nested.total_fits > 0
