"""Additive residual hybrid ``ŷ = b + α · r̂`` (architecture §7.3, §10, §13.2).

The hybrid is itself a ``BaseForecaster``. ``fit`` fits the base on the training vector,
generates residuals from that same vector only, and fits the corrector on residual
windows. ``predict`` forecasts the base path, then corrects it recursively: the residual
window starts with residuals available at the origin and is then fed only with predicted
residuals (variant A).

A context that extends the training vector (new observations after the fit, as in
variant B) appends the frozen base's one-step errors on those observations to the
residual history; nothing is refitted.
"""

from __future__ import annotations

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
from autohybts.residuals.recursive import recursive_correction
from autohybts.types import RESIDUAL_SOURCES

__all__ = ["ResidualHybridForecaster"]


class ResidualHybridForecaster(ForecasterBase):
    def __init__(
        self,
        base: BaseForecaster,
        corrector: ResidualCorrector,
        window_size: int = 7,
        alpha: float = 1.0,
        residual_source: str = "historical_oof_1step",
        settings: ResidualSettings | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base, BaseForecaster):
            raise ConfigurationError("base deve cumprir o contrato BaseForecaster.")
        if not isinstance(corrector, ResidualCorrector):
            raise ConfigurationError("corrector deve cumprir o contrato ResidualCorrector.")
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
            raise ConfigurationError(
                f"window_size deve ser inteiro >= 1; recebido {window_size!r}."
            )
        if isinstance(alpha, bool) or not isinstance(alpha, int | float) or not np.isfinite(alpha):
            raise ConfigurationError(f"alpha deve ser um número finito; recebido {alpha!r}.")
        if residual_source not in RESIDUAL_SOURCES:
            raise ConfigurationError(f"Fonte residual desconhecida {residual_source!r}.")
        self.base = base
        self.corrector = corrector
        self.window_size = window_size
        self.alpha = alpha
        self.residual_source = residual_source
        self.settings = settings

    @property
    def min_train_size(self) -> int:
        return int(getattr(self.base, "min_train_size", 1)) + self.window_size + 1

    @property
    def min_history(self) -> int:
        return int(getattr(self.base_ if self.is_fitted else self.base, "min_history", 1))

    @property
    def residual_settings(self) -> ResidualSettings:
        return self.settings or ResidualSettings()

    def _fit(self, y: FloatArray) -> None:
        self.base_ = self.base.clone().fit(y)
        self.train_ = y.copy()
        self.fallback_reason_: str | None = None
        try:
            self.residuals_ = build_residuals(
                self.base, self.base_, y, self.residual_source, self.residual_settings
            )
            windows, targets, self.row_origins_ = training_rows(self.residuals_, self.window_size)
            if len(targets) < self.residual_settings.min_corrector_rows:
                raise InsufficientHistoryError(
                    f"Apenas {len(targets)} linhas para o corretor (mínimo "
                    f"{self.residual_settings.min_corrector_rows})."
                )
            self.corrector_: ResidualCorrector | None = self.corrector.clone().fit(windows, targets)
            self.correction_std_ratio_ = _correction_std_ratio(self.corrector_, windows, targets)
        except (InsufficientHistoryError, UnsupportedCapabilityError) as error:
            if self.residual_settings.on_failure != "fallback_to_base":
                raise
            # Operational fallback (§16): serve the base forecast and say why.
            self.corrector_ = None
            self.correction_std_ratio_ = float("nan")
            self.fallback_reason_ = f"{type(error).__name__}: {error}"

    def diagnostics(self) -> dict[str, float]:
        """``correction_std_ratio`` of the last fit (ADR-0003); NaN if undefined."""

        return {"correction_std_ratio": float(getattr(self, "correction_std_ratio_", float("nan")))}

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        base_path, correction = self.predict_components(horizon, history)
        return base_path + self.alpha * correction

    def predict_components(
        self, horizon: int, history: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Base path ``b`` and correction ``r̂`` (zeros under fallback); ``ŷ = b + α · r̂``."""

        base_path = self.base_.predict(horizon, local_context(history, history.size))
        if self.corrector_ is None:
            return base_path, np.zeros(horizon)
        window = latest_window(self._residuals_at(history), self.window_size, history.size - 1)
        return base_path, recursive_correction(self.corrector_, window, horizon)

    def _residuals_at(self, history: FloatArray) -> ResidualSeries:
        n = self.train_.size
        if history.size < n or not np.array_equal(history[:n], self.train_):
            raise UnsupportedCapabilityError(
                "O híbrido só prevê de contextos que estendem o seu treino."
            )
        if history.size == n:
            return self.residuals_
        # New observations after the fit: frozen base, one-step errors, no refit.
        return self.residuals_.extend(one_step_residuals(self.base_, history, n, ONLINE_SOURCE))


def _correction_std_ratio(
    corrector: ResidualCorrector, windows: FloatArray, targets: FloatArray
) -> float:
    """std of the corrector's fitted corrections over std of the residuals it was fitted on.

    Computed on the corrector's own training rows (ADR-0003). A ratio near zero means the
    corrector barely moves the base forecast: the correction is inert.
    """

    scale = float(np.std(targets))
    if scale <= 1e-12:
        return float("nan")
    fitted = np.array([corrector.predict_next(row) for row in windows], dtype=np.float64)
    return float(np.std(fitted)) / scale
