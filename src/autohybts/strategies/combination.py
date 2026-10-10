"""Fixed combination of base forecasters (§16).

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

from autohybts.adapters.base import BaseForecaster, FloatArray, ForecasterBase
from autohybts.exceptions import ConfigurationError
from autohybts.residuals.generation import ResidualSettings, build_residuals, local_context

__all__ = ["FixedCombinationForecaster"]


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
