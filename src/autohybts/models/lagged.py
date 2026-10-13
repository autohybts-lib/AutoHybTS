"""Shared plumbing of external models that learn from lag windows.

Every origin gets a fresh model: the training vector of that origin is (optionally)
differenced, cut into lag windows and handed to the library. When
``validation_fraction > 0`` the *last* rows of those windows (in time order) are held
out inside the training vector for early stopping; nothing after the origin is ever
used. The standardization (mean and standard deviation) is fitted on the values the
*internal training* rows use, before the held-out tail: the tail is scaled with those
statistics, so it does not influence what the network is trained on and early stopping
measures how the network extrapolates to it (M01, ADR-0006). The validation windows
still read the history before each of their targets, as a forecast would. Without a
held-out tail the statistics come from the whole training vector.

Forecasts are recursive: each prediction becomes the newest lag of the next step, with
the learned weights and statistics frozen, so a frozen model also forecasts from later
contexts (variant B, out-of-fold residuals).

Windows are given to the library in time order, oldest first: an array of shape
``(rows, n_lags)``. Sequence models read it as ``n_lags`` steps of one feature.

Training holds ``TRAINING_LOCK``: PyTorch and Keras seed process-wide generators
(``torch.manual_seed``, ``keras.utils.set_random_seed``), so two networks trained at
the same time in threads (``n_jobs > 1``) would draw from each other's seeds. With the
lock a seeded network gives the same weights whatever runs beside it; other candidates
still run in parallel, and inference does not take the lock.
"""

from __future__ import annotations

import threading
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np

from autohybts.adapters.base import FloatArray, ForecasterBase, Parametrized
from autohybts.data.preprocessing import DifferenceTransform
from autohybts.data.windows import lag_matrix, latest_lags, normalize_lags
from autohybts.domain.fit_count import record_fit
from autohybts.exceptions import ConfigurationError, InsufficientHistoryError
from autohybts.models.requirements import require

__all__ = [
    "TRAINING_LOCK",
    "LaggedModelForecaster",
    "WindowCorrector",
    "chronological",
    "split_tail",
    "tail_cut",
]

# Serializes seeding + training of framework networks across threads (see above).
TRAINING_LOCK = threading.RLock()


def chronological(features: FloatArray) -> FloatArray:
    """Lag rows ``[v[t-1], v[t-2], ...]`` reordered oldest first."""

    return np.ascontiguousarray(features[:, ::-1])


def tail_cut(n_rows: int, fraction: float, minimum: int = 8) -> int | None:
    """First held-out row when the last ``fraction`` of ``n_rows`` is held out, or ``None``.

    ``None`` when nothing is held out: ``fraction`` is 0, fewer than 2 rows would be held
    out or fewer than ``minimum`` would be left to train.
    """

    held = round(n_rows * fraction)
    if fraction <= 0 or held < 2 or n_rows - held < minimum:
        return None
    return n_rows - held


def split_tail(
    features: FloatArray, targets: FloatArray, fraction: float, minimum: int = 8
) -> tuple[FloatArray, FloatArray, FloatArray | None, FloatArray | None]:
    """Hold out the last rows (in time order) for early stopping; ``None`` if too few."""

    cut = tail_cut(len(targets), fraction, minimum)
    if cut is None:
        return features, targets, None, None
    return features[:cut], targets[:cut], features[cut:], targets[cut:]


def _check_training(epochs: int, batch_size: int, learning_rate: float, fraction: float) -> None:
    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigurationError(f"{name} deve ser inteiro >= 1; recebido {value!r}.")
    if not learning_rate > 0:
        raise ConfigurationError("learning_rate deve ser positivo.")
    if not 0.0 <= fraction < 0.5:
        raise ConfigurationError("validation_fraction deve estar em [0, 0.5).")


class LaggedModelForecaster(ForecasterBase):
    """Base of the PyTorch/Keras wrappers: subclasses implement ``_train`` and ``_infer``.

    ``requires`` maps import names to version specifiers and is checked at fit time.
    """

    requires: Mapping[str, str] | None

    def __init__(
        self,
        lags: int | tuple[int, ...] = 14,
        differencing: int = 0,
        scale: bool = True,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_fraction: float = 0.2,
        patience: int = 10,
        random_state: int | None = 0,
        requires: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(differencing, bool) or not isinstance(differencing, int) or differencing < 0:
            raise ConfigurationError(
                f"differencing deve ser inteiro >= 0; recebido {differencing!r}."
            )
        _check_training(epochs, batch_size, learning_rate, validation_fraction)
        self._lags = normalize_lags(lags)
        self.lags = lags
        self.differencing = differencing
        self.scale = scale
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state
        self.requires = requires

    @property
    def n_lags(self) -> int:
        return len(self._lags)

    @property
    def min_history(self) -> int:
        return max(self._lags) + self.differencing

    @property
    def min_train_size(self) -> int:
        return self.min_history + 10

    def _fit(self, y: FloatArray) -> None:
        require(self.requires)
        level = self._differenced(y)
        max_lag = max(self._lags)
        rows = max(0, len(level) - max_lag)
        cut = tail_cut(rows, self.validation_fraction)
        # Row i reads level[i : i + max_lag] and predicts level[i + max_lag]: the internal
        # training rows (i < cut) only touch level[: cut + max_lag].
        seen = level if cut is None else level[: cut + max_lag]
        self.center_ = float(np.mean(seen)) if self.scale else 0.0
        spread = float(np.std(seen)) if self.scale else 1.0
        self.spread_ = spread if spread > 0 else 1.0
        self.train_rows_ = rows if cut is None else cut
        self.validation_rows_ = 0 if cut is None else rows - cut
        features, targets = lag_matrix(self._standardized(level), self._lags)
        if len(targets) < 4:
            raise InsufficientHistoryError(f"Apenas {len(targets)} janelas para treinar a rede.")
        x_train, y_train, x_val, y_val = split_tail(
            chronological(features), targets, self.validation_fraction
        )
        with TRAINING_LOCK:
            self.history_ = self._train(x_train, y_train, x_val, y_val)

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        anchors = None
        level = history
        if self.differencing:
            anchors, level = DifferenceTransform(self.differencing).fit_transform(history)
        window = list(self._standardized(level))
        steps = []
        for _ in range(horizon):
            row = np.ascontiguousarray(latest_lags(np.asarray(window), self._lags)[::-1])
            step = float(self._infer(row.reshape(1, -1))[0])
            window.append(step)
            steps.append(step)
        forecast = np.asarray(steps) * self.spread_ + self.center_
        return forecast if anchors is None else anchors.inverse_forecast(forecast)

    def diagnostics(self) -> dict[str, float]:
        """Training summary plus the internal split (rows, never values)."""

        history = getattr(self, "history_", None) or {}
        summary = {key: float(value) for key, value in history.items()}
        if hasattr(self, "train_rows_"):
            summary["train_rows"] = float(self.train_rows_)
            summary["validation_rows"] = float(self.validation_rows_)
        return summary

    @abstractmethod
    def _train(
        self,
        x_train: FloatArray,
        y_train: FloatArray,
        x_val: FloatArray | None,
        y_val: FloatArray | None,
    ) -> dict[str, float]:
        """Fit the network on chronological windows; return a small training summary."""

    @abstractmethod
    def _infer(self, x: FloatArray) -> FloatArray:
        """Predictions (standardized scale) for rows of chronological windows."""

    def _differenced(self, y: FloatArray) -> FloatArray:
        if not self.differencing:
            return y
        return DifferenceTransform(self.differencing).fit_transform(y)[1]

    def _standardized(self, values: FloatArray) -> FloatArray:
        return (values - self.center_) / self.spread_


class WindowCorrector(Parametrized):
    """Base of the PyTorch/Keras residual correctors (``ResidualCorrector`` contract).

    Residual windows arrive oldest first; targets and windows are standardized with the
    corrector's training residuals of the internal training rows only (the held-out tail
    for early stopping is scaled with them, as in ``LaggedModelForecaster``).
    """

    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_fraction: float = 0.2,
        patience: int = 10,
        random_state: int | None = 0,
        requires: Mapping[str, str] | None = None,
    ) -> None:
        _check_training(epochs, batch_size, learning_rate, validation_fraction)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state
        self.requires = requires

    def fit(self, windows: Any, targets: Any) -> Any:
        require(self.requires)
        features = np.asarray(windows, dtype=np.float64)
        labels = np.asarray(targets, dtype=np.float64)
        if features.ndim != 2 or labels.shape != (features.shape[0],):
            raise ConfigurationError(
                f"Janelas {features.shape} e alvos {labels.shape} incompatíveis."
            )
        cut = tail_cut(len(labels), self.validation_fraction)
        seen = labels if cut is None else labels[:cut]
        self.center_ = float(np.mean(seen))
        spread = float(np.std(seen))
        self.spread_ = spread if spread > 0 else 1.0
        self.n_lags_ = features.shape[1]
        x_train, y_train, x_val, y_val = split_tail(
            (features - self.center_) / self.spread_,
            (labels - self.center_) / self.spread_,
            self.validation_fraction,
        )
        with TRAINING_LOCK:
            self.history_ = self._train(x_train, y_train, x_val, y_val)
        record_fit()
        return self

    def predict_next(self, window: Any) -> float:
        if not hasattr(self, "center_"):
            raise ConfigurationError(f"{type(self).__name__} precisa de fit antes de predict_next.")
        row = (np.asarray(window, dtype=np.float64) - self.center_) / self.spread_
        return float(self._infer(row.reshape(1, -1))[0]) * self.spread_ + self.center_

    @abstractmethod
    def _train(
        self,
        x_train: FloatArray,
        y_train: FloatArray,
        x_val: FloatArray | None,
        y_val: FloatArray | None,
    ) -> dict[str, float]: ...

    @abstractmethod
    def _infer(self, x: FloatArray) -> FloatArray: ...
