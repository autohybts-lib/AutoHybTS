"""PyTorch models as AutoHybTS bases and residual correctors (extra ``torch``).

The user writes an ``nn.Module`` class (or any callable returning one) whose constructor
takes ``n_inputs`` (the number of lags) and its own keyword arguments, and whose
``forward`` maps a float32 tensor ``(batch, n_inputs)`` of lag values, oldest first, to
``(batch,)`` or ``(batch, 1)``. The wrapper does the rest: windows, scaling fitted on the
origin's training vector, Adam with mean squared error, early stopping on the last rows
of the training windows, seeds, recursive forecasting with frozen weights.

PyTorch is imported only when a model is fitted or loaded. Weights are pickled through
``state_dict``; the module class must be importable (defined in a ``.py`` module, not in
a notebook cell) to load a saved model in another process.
"""

from __future__ import annotations

import copy
import inspect
import io
import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from autohybts.adapters.base import FloatArray
from autohybts.exceptions import ConfigurationError
from autohybts.models.lagged import LaggedModelForecaster, WindowCorrector

__all__ = ["DEFAULT_TORCH_REQUIREMENT", "TorchCorrector", "TorchForecaster", "torch_model"]

DEFAULT_TORCH_REQUIREMENT = {"torch": ">=2.2"}
ModuleBuilder = Callable[..., Any]


def _device(name: str) -> Any:
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _build(module: ModuleBuilder, n_inputs: int, params: Mapping[str, Any] | None) -> Any:
    import torch

    net = module(n_inputs=n_inputs, **dict(params or {}))
    if not isinstance(net, torch.nn.Module):
        raise ConfigurationError(
            f"{getattr(module, '__name__', module)!r} não devolveu um torch.nn.Module."
        )
    return net


def train_torch(
    net: Any,
    x_train: FloatArray,
    y_train: FloatArray,
    x_val: FloatArray | None,
    y_val: FloatArray | None,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    seed: int | None,
    device: Any,
) -> dict[str, float]:
    """Adam + MSE with mini-batches; early stopping on the held-out tail if there is one."""

    import torch

    net.to(device)
    x = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    xv = None if x_val is None else torch.as_tensor(x_val, dtype=torch.float32, device=device)
    yv = None if y_val is None else torch.as_tensor(y_val, dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(0 if seed is None else int(seed))
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    best, best_state, waited, train_loss, epochs_run = math.inf, None, 0, math.nan, 0
    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        net.train()
        order = torch.randperm(len(x), generator=generator).to(device)
        total = 0.0
        for start in range(0, len(x), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(net(x[batch]).reshape(-1), y[batch])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        train_loss = total / len(x)
        if xv is None or yv is None:
            continue
        net.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(net(xv).reshape(-1), yv))
        if val_loss < best - 1e-9:
            best, best_state, waited = val_loss, copy.deepcopy(net.state_dict()), 0
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    summary = {"epochs_run": float(epochs_run), "train_loss": float(train_loss)}
    if math.isfinite(best):
        summary["val_loss"] = best
    return summary


def infer_torch(net: Any, x: FloatArray, device: Any) -> FloatArray:
    import torch

    with torch.no_grad():
        out = net(torch.as_tensor(x, dtype=torch.float32, device=device))
    return np.asarray(out.reshape(-1).cpu().numpy(), dtype=np.float64)


class _TorchState:
    """Pickle the network as ``state_dict`` bytes and rebuild it on load."""

    net_: Any
    module: ModuleBuilder
    module_params: Mapping[str, Any] | None
    device: str

    def _n_inputs(self) -> int:
        raise NotImplementedError

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        net = state.pop("net_", None)
        if net is not None:
            import torch

            buffer = io.BytesIO()
            torch.save({k: v.cpu() for k, v in net.state_dict().items()}, buffer)
            state["_net_weights"] = buffer.getvalue()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        weights = state.pop("_net_weights", None)
        self.__dict__.update(state)
        if weights is not None:
            import torch

            net = _build(self.module, self._n_inputs(), self.module_params)
            net.load_state_dict(torch.load(io.BytesIO(weights), weights_only=True))
            net.to(_device(self.device)).eval()
            self.net_ = net


class TorchForecaster(_TorchState, LaggedModelForecaster):
    """Base forecaster from a PyTorch module (see the module docstring for the contract)."""

    def __init__(
        self,
        module: ModuleBuilder,
        module_params: Mapping[str, Any] | None = None,
        lags: int | tuple[int, ...] = 14,
        differencing: int | str = 0,
        scale: bool = True,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_fraction: float = 0.2,
        patience: int = 10,
        random_state: int | None = 0,
        device: str = "cpu",
        requires: Mapping[str, str] | None = None,
        seasonal_differencing: int = 0,
        season_length: int = 1,
        differencing_test: str = "kpss",
        transform: str | None = None,
        outliers: str | None = None,
    ) -> None:
        if not callable(module):
            raise ConfigurationError("module deve ser uma classe ou função que cria a rede.")
        super().__init__(
            lags=lags,
            differencing=differencing,
            seasonal_differencing=seasonal_differencing,
            season_length=season_length,
            differencing_test=differencing_test,
            transform=transform,
            outliers=outliers,
            scale=scale,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_fraction=validation_fraction,
            patience=patience,
            random_state=random_state,
            requires=DEFAULT_TORCH_REQUIREMENT if requires is None else requires,
        )
        self.module = module
        self.module_params = module_params
        self.device = device

    def _n_inputs(self) -> int:
        return self.n_lags

    def _train(
        self,
        x_train: FloatArray,
        y_train: FloatArray,
        x_val: FloatArray | None,
        y_val: FloatArray | None,
    ) -> dict[str, float]:
        import torch

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        self.net_ = _build(self.module, self.n_lags, self.module_params)
        return train_torch(
            self.net_,
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            patience=self.patience,
            seed=self.random_state,
            device=_device(self.device),
        )

    def _infer(self, x: FloatArray) -> FloatArray:
        return infer_torch(self.net_, x, _device(self.device))


class TorchCorrector(_TorchState, WindowCorrector):
    """Residual corrector from a PyTorch module (input: residual window, oldest first)."""

    def __init__(
        self,
        module: ModuleBuilder,
        module_params: Mapping[str, Any] | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_fraction: float = 0.2,
        patience: int = 10,
        random_state: int | None = 0,
        device: str = "cpu",
        requires: Mapping[str, str] | None = None,
    ) -> None:
        if not callable(module):
            raise ConfigurationError("module deve ser uma classe ou função que cria a rede.")
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_fraction=validation_fraction,
            patience=patience,
            random_state=random_state,
            requires=DEFAULT_TORCH_REQUIREMENT if requires is None else requires,
        )
        self.module = module
        self.module_params = module_params
        self.device = device

    def _n_inputs(self) -> int:
        return int(self.n_lags_)

    def _train(
        self,
        x_train: FloatArray,
        y_train: FloatArray,
        x_val: FloatArray | None,
        y_val: FloatArray | None,
    ) -> dict[str, float]:
        import torch

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        self.net_ = _build(self.module, self.n_lags_, self.module_params)
        return train_torch(
            self.net_,
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            patience=self.patience,
            seed=self.random_state,
            device=_device(self.device),
        )

    def _infer(self, x: FloatArray) -> FloatArray:
        return infer_torch(self.net_, x, _device(self.device))


def torch_model(
    module: ModuleBuilder,
    kind: str = "base",
    requires: Mapping[str, str] | None = None,
    **defaults: Any,
) -> Callable[[Mapping[str, Any]], Any]:
    """Registry factory: wrapper settings go to the wrapper, everything else to ``module``.

    ``exp.register_model("lstm", torch_model(LSTMNet, hidden=32, lags=28))`` makes
    ``hidden`` a module argument and ``lags`` a wrapper argument; both can be changed per
    candidate (``create_model("lstm", hidden=64)``) and tuned.
    """

    cls = TorchForecaster if kind == "base" else TorchCorrector
    if kind not in ("base", "corrector"):
        raise ConfigurationError("kind deve ser 'base' ou 'corrector'.")
    wrapper_keys = set(inspect.signature(cls.__init__).parameters) - {
        "self",
        "module",
        "module_params",
        "requires",
    }

    def factory(params: Mapping[str, Any]) -> Any:
        merged = {**defaults, **dict(params)}
        wrapper = {k: v for k, v in merged.items() if k in wrapper_keys}
        module_params = {k: v for k, v in merged.items() if k not in wrapper_keys}
        if "lags" in wrapper and isinstance(wrapper["lags"], list):
            wrapper["lags"] = tuple(wrapper["lags"])
        return cls(module=module, module_params=module_params, requires=requires, **wrapper)

    factory.wrapper_keys = frozenset(wrapper_keys)  # type: ignore[attr-defined]
    return factory
