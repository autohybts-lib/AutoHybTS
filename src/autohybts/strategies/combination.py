"""Fixed combinations (§16): of base forecasters, and of residual correctors.

``ŷ = Σ w_i · b_i`` with weights fixed at fit time:

* ``equal``: ``w_i = 1 / k``.
* ``inverse_mse``: ``w_i ∝ 1 / MSE_i``, where ``MSE_i`` is the mean squared one-step
  out-of-fold error of member ``i`` on the tail of the training vector (a separate fit on
  the prefix, as in ``historical_oof_1step``). Weights never see data after the origin.

Every member is then fitted on the whole training vector and forecasts from the same
context, so a frozen combination takes new observations like any base (variant B).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from autohybts.adapters.base import BaseForecaster, FloatArray, ForecasterBase, Parametrized
from autohybts.exceptions import ConfigurationError, InsufficientHistoryError
from autohybts.residuals.correctors import ResidualCorrector
from autohybts.residuals.generation import ResidualSettings, build_residuals, local_context

__all__ = ["CorrectorCombination", "FixedCombinationForecaster"]


class FixedCombinationForecaster(ForecasterBase):
    def __init__(
        self,
        members: Sequence[BaseForecaster],
        weights: str = "equal",
        settings: ResidualSettings | None = None,
    ) -> None:
        super().__init__()
        if len(members) < 2:
            raise ConfigurationError("Uma combinação precisa de ao menos dois membros.")
        if not all(isinstance(member, BaseForecaster) for member in members):
            raise ConfigurationError("Membros devem cumprir o contrato BaseForecaster.")
        if weights not in ("equal", "inverse_mse"):
            raise ConfigurationError(
                f"weights deve ser 'equal' ou 'inverse_mse'; recebido {weights!r}."
            )
        self.members = tuple(members)
        self.weights = weights
        self.settings = settings

    @property
    def min_train_size(self) -> int:
        largest = max(int(getattr(m, "min_train_size", 1)) for m in self.members)
        if self.weights == "equal":
            return largest
        # The out-of-fold tail needs a prefix of ``largest`` points before it.
        share = (self.settings or ResidualSettings()).tail_fraction
        return math.ceil(largest / (1.0 - share)) + 1

    @property
    def min_history(self) -> int:
        members = self.members_ if self.is_fitted else self.members
        return max(int(getattr(m, "min_history", 1)) for m in members)

    def _fit(self, y: FloatArray) -> None:
        self.members_ = tuple(member.clone().fit(y) for member in self.members)
        if self.weights == "equal":
            weights = np.full(len(self.members), 1.0 / len(self.members))
        else:
            settings = self.settings or ResidualSettings()
            errors = np.array(
                [
                    float(
                        np.mean(
                            build_residuals(
                                template, fitted, y, "historical_oof_1step", settings
                            ).residuals
                            ** 2
                        )
                    )
                    for template, fitted in zip(self.members, self.members_, strict=True)
                ]
            )
            perfect = errors <= 1e-12
            inverse = perfect.astype(np.float64) if perfect.any() else 1.0 / errors
            weights = inverse / inverse.sum()
        self.weights_: FloatArray = weights

    def diagnostics(self) -> dict[str, float]:
        if not hasattr(self, "weights_"):
            return {}
        return {f"weight_{i}": float(w) for i, w in enumerate(self.weights_)}

    def _predict(self, horizon: int, history: FloatArray) -> FloatArray:
        context = local_context(history, history.size)
        paths = np.vstack([member.predict(horizon, context) for member in self.members_])
        return np.asarray(self.weights_ @ paths, dtype=np.float64)


class CorrectorCombination(Parametrized):
    """Several residual correctors averaged with fixed weights; itself a ``ResidualCorrector``.

    Used inside a residual hybrid, ``ŷ = b + α · Σ w_j · r̂_j``, so it inherits every
    causality rule of the residual pipeline. ``inverse_mse`` weights come from the last
    ``holdout`` share of the corrector's own training rows (in time order): each member is
    fitted on the rows before them and scored on them; then every member is refitted on
    all rows.
    """

    def __init__(
        self,
        members: Sequence[ResidualCorrector],
        weights: str = "equal",
        holdout: float = 0.3,
    ) -> None:
        if len(members) < 2:
            raise ConfigurationError("Uma combinação de corretores precisa de ao menos dois.")
        if not all(isinstance(member, ResidualCorrector) for member in members):
            raise ConfigurationError("Membros devem cumprir o contrato ResidualCorrector.")
        if weights not in ("equal", "inverse_mse"):
            raise ConfigurationError(
                f"weights deve ser 'equal' ou 'inverse_mse'; recebido {weights!r}."
            )
        if not 0.0 < holdout < 1.0:
            raise ConfigurationError("holdout deve estar em (0, 1).")
        self.members = tuple(members)
        self.weights = weights
        self.holdout = holdout

    def fit(self, windows: npt.ArrayLike, targets: npt.ArrayLike) -> CorrectorCombination:
        features = np.asarray(windows, dtype=np.float64)
        labels = np.asarray(targets, dtype=np.float64)
        if self.weights == "equal":
            weights = np.full(len(self.members), 1.0 / len(self.members))
        else:
            split = len(labels) - max(1, round(len(labels) * self.holdout))
            if split < 2:
                raise InsufficientHistoryError(
                    f"{len(labels)} linhas não comportam os pesos fora da amostra dos corretores."
                )
            errors = []
            for member in self.members:
                fitted = member.clone().fit(features[:split], labels[:split])
                predictions = np.array([fitted.predict_next(row) for row in features[split:]])
                errors.append(float(np.mean((labels[split:] - predictions) ** 2)))
            mse = np.asarray(errors)
            perfect = mse <= 1e-12
            inverse = perfect.astype(np.float64) if perfect.any() else 1.0 / mse
            weights = inverse / inverse.sum()
        self.weights_: FloatArray = weights
        self.members_ = tuple(member.clone().fit(features, labels) for member in self.members)
        return self

    def predict_next(self, window: npt.ArrayLike) -> float:
        if not hasattr(self, "members_"):
            raise ConfigurationError("CorrectorCombination precisa de fit antes de predict_next.")
        values = np.asarray(window, dtype=np.float64)
        return float(
            sum(
                w * m.predict_next(values)
                for w, m in zip(self.weights_, self.members_, strict=True)
            )
        )
