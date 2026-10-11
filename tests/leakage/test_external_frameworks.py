"""PyTorch and Keras wrappers (I14): causality, seeds, persistence, corrector role.

Skipped when the framework is not installed. Networks are tiny and trained briefly: the
tests check contracts, not accuracy.
"""

from __future__ import annotations

import os
import pickle

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
        cases.append(pytest.param(None, id="torch", marks=pytest.mark.skip("torch missing")))
    try:
        import keras  # noqa: F401

        from autohybts.models import keras_model, keras_zoo

        cases.append(
            pytest.param(keras_model(keras_zoo.mlp, hidden=[8], lags=7, epochs=3), id="keras-mlp")
        )
    except ModuleNotFoundError:
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
    pytest.importorskip("torch")
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
    torch = pytest.importorskip("torch")
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
