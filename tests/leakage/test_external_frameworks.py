"""PyTorch and Keras wrappers (I14): causality, seeds, persistence, corrector role.

Skipped when the framework is not installed. Networks are tiny and trained briefly: the
tests check contracts, not accuracy.
"""

from __future__ import annotations

import importlib
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
import pytest

from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import TemporalSplitter
from autohybts.domain.specs import CandidateSpec, ModelSpec
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.registry import default_registry
from autohybts.residuals.generation import local_context

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def values(n: int = 150) -> np.ndarray:
    rng = np.random.default_rng(1)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.6 * noise[i - 1] + rng.normal()
    return 20 + 3 * np.sin(2 * np.pi * np.arange(n) / 7) + noise


# CI's frameworks job sets this so a missing framework fails instead of being skipped.
REQUIRED = os.environ.get("AUTOHYBTS_REQUIRE_FRAMEWORKS") == "1"


def need(module: str) -> Any:
    """Import a framework, skipping the test without it (failing under REQUIRED)."""

    if REQUIRED:
        return importlib.import_module(module)
    return pytest.importorskip(module)


def frameworks() -> list[object]:
    cases = []
    try:
        import torch  # noqa: F401

        from autohybts.models import torch_model
        from autohybts.models.torch_zoo import LSTMNet, MambaNet, TransformerNet

        cases += [
            pytest.param(torch_model(LSTMNet, hidden=4, lags=7, epochs=3), id="torch-lstm"),
            pytest.param(
                torch_model(TransformerNet, d_model=8, heads=2, layers=1, lags=7, epochs=3),
                id="torch-transformer",
            ),
            pytest.param(
                torch_model(MambaNet, d_model=8, d_state=4, layers=1, lags=7, epochs=3),
                id="torch-mamba",
            ),
        ]
    except ModuleNotFoundError:
        if REQUIRED:
            raise
        cases.append(pytest.param(None, id="torch", marks=pytest.mark.skip("torch missing")))
    try:
        import keras  # noqa: F401

        from autohybts.models import keras_model, keras_zoo

        cases.append(
            pytest.param(keras_model(keras_zoo.mlp, hidden=[8], lags=7, epochs=3), id="keras-mlp")
        )
    except ModuleNotFoundError:
        if REQUIRED:
            raise
        cases.append(pytest.param(None, id="keras", marks=pytest.mark.skip("keras missing")))
    return cases


def forecasts(factory: object, data: np.ndarray) -> dict[str, tuple[int, tuple[float, ...]]]:
    registry = default_registry()
    registry.register_base("net", factory, family="deep", version="1")  # type: ignore[arg-type]
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=len(data)), data)
    plan = TemporalSplitter(horizon=5, n_search_folds=1, n_select_folds=1, step=10).plan(frame)
    result = TemporalEvaluator(registry).evaluate(
        CandidateSpec.standalone("net", family="deep"), plan
    )
    assert all(t.succeeded for t in result.trials), result.failures
    return {t.fold_id: (t.origin_index, t.y_pred) for t in result.trials}


@pytest.mark.parametrize("factory", frameworks())
def test_future_perturbation_does_not_change_earlier_forecasts(factory: object) -> None:
    data = values()
    reference = forecasts(factory, data)
    cut = 145
    perturbed = data.copy()
    perturbed[cut:] += 40.0
    changed = forecasts(factory, perturbed)
    for fold_id, (index, path) in changed.items():
        if index < cut:
            np.testing.assert_allclose(path, reference[fold_id][1], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("factory", frameworks())
def test_seeded_fit_is_reproducible_and_survives_pickle(factory: object) -> None:
    y = values(120)
    first = factory({}).fit(y)  # type: ignore[operator]
    second = factory({}).fit(y)  # type: ignore[operator]
    context = local_context(y, 120)
    np.testing.assert_allclose(first.predict(7, context), second.predict(7, context), atol=1e-5)
    restored = pickle.loads(pickle.dumps(first))
    np.testing.assert_allclose(restored.predict(7, context), first.predict(7, context), atol=1e-6)
    # A frozen model forecasts from a later context without refitting.
    later = local_context(values(130), 130)
    assert restored.predict(1, later).shape == (1,)
    assert first.diagnostics()["epochs_run"] >= 1


def test_torch_corrector_inside_a_hybrid() -> None:
    need("torch")
    from autohybts.models import torch_model
    from autohybts.models.torch_zoo import LSTMNet

    registry = default_registry()
    registry.register_corrector(
        "net", torch_model(LSTMNet, kind="corrector", hidden=4, epochs=3), version="1"
    )
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=200), values(200))
    plan = TemporalSplitter(horizon=5, n_search_folds=1, n_select_folds=1).plan(frame)
    hybrid = CandidateSpec(
        "residual",
        "residual_additive",
        ModelSpec.of("seasonal_naive", {"season_length": 7}),
        ModelSpec.of("net"),
        7,
        1.0,
        "historical_oof_1step",
    )
    result = TemporalEvaluator(registry, seasonal_period=7).evaluate(hybrid, plan)
    assert all(t.succeeded for t in result.trials), result.failures
    assert np.isfinite(result.trials[0].diagnostics["correction_std_ratio"])


def test_vectorized_mamba_scan_matches_the_step_by_step_recursion() -> None:
    torch = need("torch")
    from torch import nn

    from autohybts.models.torch_zoo import _SelectiveSSM

    torch.manual_seed(0)
    block = _SelectiveSSM(8, 4, 2, 4).double()
    h = torch.randn(3, 20, 8, dtype=torch.float64)
    with torch.no_grad():
        out = block(h)
        length = h.shape[1]
        x, z = block.in_proj(block.norm(h)).chunk(2, dim=-1)
        x = nn.functional.silu(block.conv(x.transpose(1, 2))[..., :length].transpose(1, 2))
        dt, b, c = torch.split(block.x_proj(x), [1, block.d_state, block.d_state], dim=-1)
        delta = nn.functional.softplus(block.dt_proj(dt))
        a = -torch.exp(block.a_log)
        state = h.new_zeros(h.shape[0], block.inner, block.d_state)
        outputs = []
        for t in range(length):
            decay = torch.exp(delta[:, t, :, None] * a)
            state = decay * state + delta[:, t, :, None] * b[:, t, None, :] * x[:, t, :, None]
            outputs.append((state * c[:, t, None, :]).sum(-1) + block.d * x[:, t])
        reference = h + block.out_proj(torch.stack(outputs, 1) * nn.functional.silu(z))
    assert float((out - reference).abs().max()) < 1e-12
    # Causality: changing the last step leaves every earlier output unchanged.
    changed = h.clone()
    changed[:, -1] += 5.0
    with torch.no_grad():
        assert torch.allclose(block(changed)[:, :-1], out[:, :-1])


# ------------------------------------------------------------------ contracts (M07b)
@pytest.mark.parametrize("factory", frameworks())
def test_clone_keeps_parameters_and_drops_the_fitted_network(factory: object) -> None:
    fitted = factory({}).fit(values(120))  # type: ignore[operator]
    clone = fitted.clone()
    assert clone.get_params() == fitted.get_params()
    assert not clone.is_fitted
    assert not hasattr(clone, "net_")
    assert not hasattr(clone, "model_")
    assert not hasattr(clone, "center_")


@pytest.mark.parametrize("factory", frameworks())
def test_later_contexts_and_streaming_never_retrain(
    factory: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = factory({})  # type: ignore[operator]
    calls: list[int] = []
    original = type(model)._train

    def counting(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(model), "_train", counting)
    data = values(160)
    model.fit(data[:120])
    paths = [model.predict(3, local_context(data, end)) for end in range(120, 161, 8)]
    assert len(calls) == 1  # one training; every later context only moves the inputs
    assert all(p.shape == (3,) and np.isfinite(p).all() for p in paths)
    # Variant B: rolling one step at a time from a frozen finalized model.
    from autohybts.forecasting.finalized import FinalizedForecaster

    spec = CandidateSpec.standalone("net", family="deep")
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=120), data[:120])
    finalized = FinalizedForecaster(spec, model, frame, None, {}, len(frame))
    for step in range(120, 126):
        when = pd.DatetimeIndex([finalized.last_observation + pd.Timedelta(days=1)])
        finalized = finalized.update_state(pd.Series([data[step]], index=when))
        assert np.isfinite(finalized.predict(1)["y_pred"]).all()
    assert len(calls) == 1


def test_threads_do_not_mix_the_seeds_of_concurrent_trainings() -> None:
    need("torch")
    from concurrent.futures import ThreadPoolExecutor

    from autohybts.models import torch_model
    from autohybts.models.torch_zoo import MLPNet

    y = values(140)
    context = local_context(y, 140)

    def forecast(seed: int) -> np.ndarray:
        factory = torch_model(MLPNet, hidden=[8], dropout=0.2, lags=7, epochs=5)
        return factory({"random_state": seed}).fit(y).predict(5, context)

    sequential = [forecast(seed) for seed in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(forecast, range(4)))
    for one, other in zip(sequential, parallel, strict=True):
        np.testing.assert_allclose(one, other, rtol=0, atol=1e-6)
    assert not np.allclose(sequential[0], sequential[1])  # the seeds do matter


def test_registered_torch_model_through_the_api_and_another_process(tmp_path) -> None:  # type: ignore[no-untyped-def]
    need("torch")
    import json
    import subprocess
    import sys

    from autohybts import AutoHybridExperiment, load_model, save_model
    from autohybts.models import torch_model
    from autohybts.models.torch_zoo import MLPNet

    y = pd.Series(values(220), index=pd.date_range("2024-01-01", periods=220))
    exp = AutoHybridExperiment(
        fh=5,
        fold=1,
        budget="smoke",
        base_models=("naive", "net"),
        correctors=("linear",),
        residual_windows=(7,),
        alphas=(1.0,),
    )
    exp.register_model(
        "net",
        torch_model(MLPNet, hidden=[8], lags=7, epochs=3),
        family="deep",
        requires={"torch": ">=2.2"},
        version="1",
    )
    exp.fit(y)
    result = exp.compare_models()
    assert "net" in {e.label.split("(")[0] for e in result.search.ledger}
    assert exp.evaluator_.protocol()["external_models"]["base:net"]["version"] == "1"
    final = exp.finalize_model(exp.create_model("net").candidate)
    expected = final.predict(5)["y_pred"].to_numpy()
    path = save_model(final, tmp_path / "net")
    assert np.allclose(load_model(path, trusted=True).predict(5)["y_pred"], expected)
    code = (
        "import json, sys; from autohybts import load_model; "
        "m = load_model(sys.argv[1], trusted=True); "
        "print(json.dumps(m.predict(5)['y_pred'].tolist()))"
    )
    output = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, text=True, check=True
    ).stdout
    np.testing.assert_allclose(json.loads(output), expected, rtol=0, atol=1e-6)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    import torch

    assert manifest["frameworks"] == {"torch": torch.__version__}  # the installed version


def test_keras_corrector_inside_a_hybrid() -> None:
    need("keras")
    from autohybts.models import keras_model, keras_zoo

    registry = default_registry()
    registry.register_corrector(
        "knet", keras_model(keras_zoo.mlp, kind="corrector", hidden=[4], epochs=3), version="1"
    )
    frame = SeriesFrame.from_arrays(pd.date_range("2024-01-01", periods=200), values(200))
    plan = TemporalSplitter(horizon=5, n_search_folds=1, n_select_folds=1).plan(frame)
    hybrid = CandidateSpec(
        "residual",
        "residual_additive",
        ModelSpec.of("seasonal_naive", {"season_length": 7}),
        ModelSpec.of("knet"),
        7,
        1.0,
        "historical_oof_1step",
    )
    result = TemporalEvaluator(registry, seasonal_period=7).evaluate(hybrid, plan)
    assert all(t.succeeded for t in result.trials), result.failures
    assert np.isfinite(result.trials[0].diagnostics["correction_std_ratio"])


def test_torch_corrector_survives_save_and_load_in_another_process(tmp_path) -> None:  # type: ignore[no-untyped-def]
    need("torch")
    import json
    import subprocess
    import sys

    from autohybts import AutoHybridExperiment, load_model, save_model
    from autohybts.models import torch_model
    from autohybts.models.torch_zoo import LSTMNet

    y = pd.Series(values(220), index=pd.date_range("2024-01-01", periods=220))
    exp = AutoHybridExperiment(fh=5, fold=1, budget="smoke", seasonal_period=7)
    exp.register_model(
        "tnet",
        torch_model(LSTMNet, kind="corrector", hidden=4, epochs=3),
        kind="corrector",
        requires={"torch": ">=2.2"},
        version="1",
    )
    exp.fit(y)
    hybrid = exp.create_hybrid("seasonal_naive", "tnet")
    final = exp.finalize_model(hybrid)
    expected = final.predict(5)["y_pred"].to_numpy()
    path = save_model(final, tmp_path / "hybrid")
    np.testing.assert_allclose(load_model(path, trusted=True).predict(5)["y_pred"], expected)
    code = (
        "import json, sys; from autohybts import load_model; "
        "m = load_model(sys.argv[1], trusted=True); "
        "print(json.dumps(m.predict(5)['y_pred'].tolist()))"
    )
    output = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, text=True, check=True
    ).stdout
    np.testing.assert_allclose(json.loads(output), expected, rtol=0, atol=1e-6)
