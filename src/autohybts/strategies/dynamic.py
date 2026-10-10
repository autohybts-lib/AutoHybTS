"""Dynamic selection of the corrector, variant B only (§16, R07; DReF and local combination).

A base forecaster and a pool of residual correctors. At each step, the residuals just
before the origin are compared with a library of past situations (the *DSEL*); the
``k`` most similar ones decide what to do:

* ``dref``: use the corrector with the smallest mean absolute error on those neighbours,
  or no correction at all when that is better ("none" is always in the pool).
* ``local``: fit a small kernel ridge on the neighbours, from (base forecast, correction
  of the best static corrector) to the observed value, and use it for this step only.

Causality. The DSEL is built from out-of-fold one-step errors on the *tail of the
training vector*: each pool member is fitted on the residuals before the tail and
scores the tail one step at a time. The window that scores the candidate (the
validation origin) is never part of it. At prediction time the query uses residuals up
to the origin only; the error of the current step enters after it is observed (variant
B). Multi-step forecasts are refused: without an observation between steps there is
nothing to select on (that is why the strategy does not exist in variant A).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from autohybts.adapters.base import BaseForecaster, FloatArray, ForecasterBase
from autohybts.exceptions import (
    ConfigurationError,
    InsufficientHistoryError,
    UnsupportedCapabilityError,
)
from autohybts.residuals.correctors import ResidualCorrector
from autohybts.residuals.features import latest_window, training_rows
from autohybts.residuals.generation import (
    ResidualSettings,
    build_residuals,
    local_context,
    one_step_residuals,
)
from autohybts.residuals.provenance import ONLINE_SOURCE, ResidualSeries

__all__ = ["DynamicSelectionForecaster"]


class DynamicSelectionForecaster(ForecasterBase):
    def __init__(
        self,
        base: BaseForecaster,
        pool: Sequence[ResidualCorrector],
        window_size: int = 7,
        method: str = "dref",
        neighbors: int = 10,
        lag_size: int = 7,
        dsel_size: int | None = None,
        local_alpha: float = 1.0,
        residual_source: str = "historical_oof_1step",
        settings: ResidualSettings | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base, BaseForecaster):
            raise ConfigurationError("base deve cumprir o contrato BaseForecaster.")
        if not pool or not all(isinstance(c, ResidualCorrector) for c in pool):
            raise ConfigurationError("pool deve ter ao menos um ResidualCorrector.")
        if method not in ("dref", "local"):
            raise ConfigurationError(f"method deve ser 'dref' ou 'local'; recebido {method!r}.")
        for name, value in (
            ("window_size", window_size),
            ("neighbors", neighbors),
            ("lag_size", lag_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigurationError(f"{name} deve ser inteiro >= 1; recebido {value!r}.")
        if dsel_size is not None and (isinstance(dsel_size, bool) or dsel_size < 2):
            raise ConfigurationError("dsel_size deve ser inteiro >= 2 ou None.")
        if not local_alpha > 0:
            raise ConfigurationError("local_alpha deve ser positivo.")
        self.base = base
        self.pool = tuple(pool)
        self.window_size = window_size
        self.method = method
        self.neighbors = neighbors
        self.lag_size = lag_size
        self.dsel_size = dsel_size
        self.local_alpha = local_alpha
        self.residual_source = residual_source
        self.settings = settings

    @property
    def min_train_size(self) -> int:
        settings = self.settings or ResidualSettings()
        rows = max(self.window_size, self.lag_size) + 2 * settings.min_corrector_rows
        base = int(getattr(self.base, "min_train_size", 1))
        return int(np.ceil((base + rows) / settings.tail_fraction))

    @property
    def min_history(self) -> int:
        return int(getattr(self.base_ if self.is_fitted else self.base, "min_history", 1))

    def _fit(self, y: FloatArray) -> None:
        settings = self.settings or ResidualSettings()
        self.base_ = self.base.clone().fit(y)
        self.train_ = y.copy()
        self.residuals_ = build_residuals(self.base, self.base_, y, self.residual_source, settings)
        values = self.residuals_.residuals
        memory = max(self.window_size, self.lag_size)
        dsel = self.dsel_size or len(values) // 2
        dsel = min(dsel, len(values) - memory - settings.min_corrector_rows)
        if dsel < 2:
            raise InsufficientHistoryError(
                f"{len(values)} resíduos não comportam um DSEL e o treino dos corretores."
            )
        split = len(values) - dsel
        head = ResidualSeries(self.residuals_.records[:split])
        windows, targets, _ = training_rows(head, self.window_size)
        if len(targets) < settings.min_corrector_rows:
            raise InsufficientHistoryError(
                f"Apenas {len(targets)} linhas para os corretores antes do DSEL."
            )
        # Out-of-fold one-step corrections on the DSEL: fitted before it, frozen over it.
        positions = np.arange(split, len(values))
        corrections = self._dsel_corrections(windows, targets, values, positions)
        actual = values[positions]
        # Column 0 is "no correction": its error is the residual itself.
        self.dsel_errors_ = np.abs(np.column_stack([actual, actual[:, None] - corrections]))
        self.dsel_corrections_ = corrections
        self.scale_ = float(np.std(values[:split])) or 1.0
        self.library_ = np.vstack([values[p - self.lag_size : p] for p in positions]) / self.scale_
        targets_index = self.residuals_.targets[positions]
        self.dsel_target_ = y[targets_index]
        self.dsel_level_ = self.dsel_target_ - actual
        self.best_ = int(np.argmin(self.dsel_errors_[:, 1:].mean(axis=0)))
        # Pool refitted on every residual for forecasting.
        all_windows, all_targets, _ = training_rows(self.residuals_, self.window_size)
        self.pool_ = tuple(c.clone().fit(all_windows, all_targets) for c in self.pool)
        best = self.pool_[self.best_]
        fitted = np.array([best.predict_next(row) for row in all_windows])
        scale = float(np.std(all_targets))
        self.correction_std_ratio_ = float(np.std(fitted)) / scale if scale > 1e-12 else np.nan

    def _dsel_corrections(
        self,
        windows: FloatArray,
        targets: FloatArray,
        values: FloatArray,
        positions: np.ndarray,
    ) -> FloatArray:
        columns = []
        for corrector in self.pool:
            fitted = corrector.clone().fit(windows, targets)
            columns.append(
                [fitted.predict_next(values[p - self.window_size : p]) for p in positions]
            )
        return np.asarray(columns, dtype=np.float64).T

    def diagnostics(self) -> dict[str, float]:
        if not hasattr(self, "best_"):
            return {}
        return {
            "correction_std_ratio": float(self.correction_std_ratio_),
            "best_static": float(self.best_),
            "dsel_size": float(len(self.dsel_target_)),
        }

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        level, correction, _ = self._decide(horizon, history)
        return np.array([level + correction])

    def predict_components(
        self, horizon: int, history: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Base forecast and the correction chosen (or combined) for this step."""

        level, correction, _ = self._decide(horizon, history)
        return np.array([level]), np.array([correction])

    def choice_at(self, history: FloatArray) -> int:
        """Pool index used at the origin ``history[-1]``: 0 is no correction (``dref``)."""

        return self._decide(1, history)[2]

    def _decide(self, horizon: int, history: FloatArray) -> tuple[float, float, int]:
        if horizon != 1:
            raise UnsupportedCapabilityError(
                "A seleção dinâmica prevê um passo por vez (variante B): sem observação entre "
                "os passos não há o que selecionar."
            )
        level = float(self.base_.predict(1, local_context(history, history.size))[0])
        residuals = self._residuals_at(history)
        window = latest_window(residuals, self.window_size, history.size - 1)
        query = latest_window(residuals, self.lag_size, history.size - 1) / self.scale_
        distances = np.linalg.norm(self.library_ - query, axis=1)
        nearest = np.argsort(distances, kind="stable")[: min(self.neighbors, distances.size)]
        if self.method == "dref":
            choice = int(np.argmin(self.dsel_errors_[nearest].mean(axis=0)))
            if choice == 0:
                return level, 0.0, 0
            return level, float(self.pool_[choice - 1].predict_next(window)), choice
        correction = float(self.pool_[self.best_].predict_next(window))
        combined = self._local(nearest, level, correction)
        return level, combined - level, self.best_ + 1

    def _local(self, nearest: np.ndarray, level: float, correction: float) -> float:
        from sklearn.kernel_ridge import KernelRidge

        features = np.column_stack(
            (self.dsel_level_[nearest], self.dsel_corrections_[nearest, self.best_])
        )
        center, spread = features.mean(axis=0), features.std(axis=0)
        spread[spread == 0] = 1.0
        target = self.dsel_target_[nearest]
        target_center = float(target.mean())
        target_spread = float(target.std()) or 1.0
        model = KernelRidge(alpha=self.local_alpha, kernel="rbf").fit(
            (features - center) / spread, (target - target_center) / target_spread
        )
        query = (np.array([[level, correction]]) - center) / spread
        return float(model.predict(query)[0]) * target_spread + target_center

    def _residuals_at(self, history: FloatArray) -> ResidualSeries:
        n = self.train_.size
        if history.size < n or not np.array_equal(history[:n], self.train_):
            raise UnsupportedCapabilityError(
                "A seleção dinâmica só prevê de contextos que estendem o seu treino."
            )
        if history.size == n:
            return self.residuals_
        return self.residuals_.extend(one_step_residuals(self.base_, history, n, ONLINE_SOURCE))
