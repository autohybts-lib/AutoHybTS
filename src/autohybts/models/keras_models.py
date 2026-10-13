"""TensorFlow/Keras models as AutoHybTS bases and residual correctors (extra ``tensorflow``).

The user writes a builder: a function (or class) called as ``builder(n_inputs=...,
**params)`` that returns an *uncompiled* ``keras.Model`` mapping ``(batch, n_inputs)``
lag values, oldest first, to ``(batch, 1)``. The wrapper compiles it (Adam, mean squared
error), trains with early stopping on the last rows of the training windows and
forecasts recursively with frozen weights. Keras 3 (shipped with TensorFlow 2.16+) is
expected.

TensorFlow is imported only when a model is fitted or loaded. Weights are pickled as
arrays and the model is rebuilt from the builder, which must be importable to load a
saved model in another process.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from autohybts.adapters.base import FloatArray
from autohybts.exceptions import ConfigurationError
from autohybts.models.lagged import LaggedModelForecaster, WindowCorrector

__all__ = ["DEFAULT_KERAS_REQUIREMENT", "KerasCorrector", "KerasForecaster", "keras_model"]

DEFAULT_KERAS_REQUIREMENT = {"tensorflow": ">=2.16"}
Builder = Callable[..., Any]


def _build(builder: Builder, n_inputs: int, params: Mapping[str, Any] | None) -> Any:
    import keras

    model = builder(n_inputs=n_inputs, **dict(params or {}))
    if not isinstance(model, keras.Model):
        raise ConfigurationError(
            f"{getattr(builder, '__name__', builder)!r} não devolveu um keras.Model."
        )
    return model


def train_keras(
    model: Any,
    x_train: FloatArray,
    y_train: FloatArray,
    x_val: FloatArray | None,
    y_val: FloatArray | None,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
) -> dict[str, float]:
    import keras

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    callbacks = []
    validation = None
    if x_val is not None and y_val is not None:
        validation = (x_val.astype("float32"), y_val.astype("float32"))
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            )
        )
    history = model.fit(
        x_train.astype("float32"),
        y_train.astype("float32"),
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation,
        callbacks=callbacks,
        shuffle=True,
        verbose=0,
    )
    losses = history.history
    summary = {"epochs_run": float(len(losses["loss"])), "train_loss": float(losses["loss"][-1])}
    if "val_loss" in losses:
        summary["val_loss"] = float(min(losses["val_loss"]))
    return summary


def infer_keras(model: Any, x: FloatArray) -> FloatArray:
    out = model(x.astype("float32"), training=False)
    return np.asarray(np.asarray(out).reshape(-1), dtype=np.float64)


def _seed(seed: int | None) -> None:
    if seed is not None:
        import keras

        keras.utils.set_random_seed(int(seed))


class _KerasState:
    """Pickle weights as arrays and rebuild the model from its builder on load."""

    model_: Any
    builder: Builder
    builder_params: Mapping[str, Any] | None

    def _n_inputs(self) -> int:
        raise NotImplementedError

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        model = state.pop("model_", None)
        if model is not None:
            state["_model_weights"] = [np.asarray(w) for w in model.get_weights()]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        weights = state.pop("_model_weights", None)
        self.__dict__.update(state)
        if weights is not None:
            model = _build(self.builder, self._n_inputs(), self.builder_params)
            model.set_weights(weights)
            self.model_ = model


class KerasForecaster(_KerasState, LaggedModelForecaster):
    """Base forecaster from a Keras model builder (see the module docstring)."""

    def __init__(
        self,
        builder: Builder,
        builder_params: Mapping[str, Any] | None = None,
        lags: int | tuple[int, ...] = 14,
        differencing: int | str = 0,
        scale: bool = True,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_fraction: float = 0.2,
        patience: int = 10,
        random_state: int | None = 0,
        requires: Mapping[str, str] | None = None,
        seasonal_differencing: int = 0,
        season_length: int = 1,
        differencing_test: str = "kpss",
        transform: str | None = None,
        outliers: str | None = None,
    ) -> None:
        if not callable(builder):
            raise ConfigurationError("builder deve ser uma função ou classe que cria o modelo.")
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
            requires=DEFAULT_KERAS_REQUIREMENT if requires is None else requires,
        )
        self.builder = builder
        self.builder_params = builder_params

    def _n_inputs(self) -> int:
        return self.n_lags

    def _train(
        self,
        x_train: FloatArray,
        y_train: FloatArray,
        x_val: FloatArray | None,
        y_val: FloatArray | None,
    ) -> dict[str, float]:
        _seed(self.random_state)
        self.model_ = _build(self.builder, self.n_lags, self.builder_params)
        return train_keras(
            self.model_,
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            patience=self.patience,
        )

    def _infer(self, x: FloatArray) -> FloatArray:
        return infer_keras(self.model_, x)


class KerasCorrector(_KerasState, WindowCorrector):
    """Residual corrector from a Keras model builder (input: residual window, oldest first)."""

    def __init__(
        self,
        builder: Builder,
        builder_params: Mapping[str, Any] | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_fraction: float = 0.2,
        patience: int = 10,
        random_state: int | None = 0,
        requires: Mapping[str, str] | None = None,
    ) -> None:
        if not callable(builder):
            raise ConfigurationError("builder deve ser uma função ou classe que cria o modelo.")
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_fraction=validation_fraction,
            patience=patience,
            random_state=random_state,
            requires=DEFAULT_KERAS_REQUIREMENT if requires is None else requires,
        )
        self.builder = builder
        self.builder_params = builder_params

    def _n_inputs(self) -> int:
        return int(self.n_lags_)

    def _train(
        self,
        x_train: FloatArray,
        y_train: FloatArray,
        x_val: FloatArray | None,
        y_val: FloatArray | None,
    ) -> dict[str, float]:
        _seed(self.random_state)
        self.model_ = _build(self.builder, self.n_lags_, self.builder_params)
        return train_keras(
            self.model_,
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            patience=self.patience,
        )

    def _infer(self, x: FloatArray) -> FloatArray:
        return infer_keras(self.model_, x)


def keras_model(
    builder: Builder,
    kind: str = "base",
    requires: Mapping[str, str] | None = None,
    **defaults: Any,
) -> Callable[[Mapping[str, Any]], Any]:
    """Registry factory: wrapper settings go to the wrapper, everything else to ``builder``."""

    if kind not in ("base", "corrector"):
        raise ConfigurationError("kind deve ser 'base' ou 'corrector'.")
    cls = KerasForecaster if kind == "base" else KerasCorrector
    wrapper_keys = set(inspect.signature(cls.__init__).parameters) - {
        "self",
        "builder",
        "builder_params",
        "requires",
    }

    def factory(params: Mapping[str, Any]) -> Any:
        merged = {**defaults, **dict(params)}
        wrapper = {k: v for k, v in merged.items() if k in wrapper_keys}
        builder_params = {k: v for k, v in merged.items() if k not in wrapper_keys}
        if "lags" in wrapper and isinstance(wrapper["lags"], list):
            wrapper["lags"] = tuple(wrapper["lags"])
        return cls(builder=builder, builder_params=builder_params, requires=requires, **wrapper)

    factory.wrapper_keys = frozenset(wrapper_keys)  # type: ignore[attr-defined]
    return factory
