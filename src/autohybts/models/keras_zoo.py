"""Example Keras builders for ``KerasForecaster``/``KerasCorrector`` (extra ``tensorflow``).

Each builder is called as ``builder(n_inputs=..., **params)`` and returns an uncompiled
``keras.Model`` from ``(batch, n_inputs)`` lag values, oldest first, to ``(batch, 1)``.
Keras is imported inside the builders, so importing this module is cheap.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["lstm", "mlp", "transformer"]


def mlp(n_inputs: int, hidden: Sequence[int] = (64, 32), dropout: float = 0.0) -> Any:
    import keras

    inputs = keras.Input(shape=(n_inputs,))
    h = inputs
    for size in hidden:
        h = keras.layers.Dense(int(size), activation="relu")(h)
        if dropout:
            h = keras.layers.Dropout(dropout)(h)
    return keras.Model(inputs, keras.layers.Dense(1)(h))


def lstm(n_inputs: int, units: int = 32, layers: int = 1) -> Any:
    import keras

    inputs = keras.Input(shape=(n_inputs,))
    h = keras.layers.Reshape((n_inputs, 1))(inputs)
    for index in range(layers):
        h = keras.layers.LSTM(units, return_sequences=index < layers - 1)(h)
    return keras.Model(inputs, keras.layers.Dense(1)(h))


def transformer(
    n_inputs: int, d_model: int = 32, heads: int = 4, layers: int = 2, feedforward: int = 64
) -> Any:
    import keras

    class AddPosition(keras.layers.Layer):  # type: ignore[misc]  # learned position per lag
        def build(self, input_shape: Any) -> None:
            self.position = self.add_weight(
                shape=(input_shape[1], input_shape[2]), initializer="zeros", name="position"
            )

        def call(self, x: Any) -> Any:
            return x + self.position

    inputs = keras.Input(shape=(n_inputs,))
    h = keras.layers.Dense(d_model)(keras.layers.Reshape((n_inputs, 1))(inputs))
    h = AddPosition()(h)
    for _ in range(layers):
        attended = keras.layers.MultiHeadAttention(heads, d_model // heads)(
            h, h, use_causal_mask=True
        )
        h = keras.layers.LayerNormalization()(h + attended)
        ff = keras.layers.Dense(feedforward, activation="relu")(h)
        h = keras.layers.LayerNormalization()(h + keras.layers.Dense(d_model)(ff))
    last = keras.layers.Lambda(lambda t: t[:, -1])(h)
    return keras.Model(inputs, keras.layers.Dense(1)(last))
