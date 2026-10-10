"""Residual multistage: corrections of corrections, up to ``MAX_STAGES`` (§16).

Formal targets. Stage 0 is the base ``b``. For ``k >= 1`` the stage-``k`` model is
``ŷ_k = ŷ_{k-1} + α · r̂_k`` and its corrector learns ``e_k(t) = y(t) - ŷ_{k-1}(t)``,
the error of the *whole* stage ``k - 1`` model, produced inside the training vector by
the configured residual source (out-of-fold by default). Because a residual hybrid is
itself a base forecaster, stage ``k`` is a residual hybrid whose base is stage ``k - 1``:
every stage inherits the provenance rules of the residual pipeline (``available_at`` equal
to the target, windows ending at the origin, variant A recursion fed only with predicted
residuals), so no observation error acts as a feature before it exists.
"""

from __future__ import annotations

from collections.abc import Sequence

from autohybts.adapters.base import BaseForecaster
from autohybts.domain.specs import MAX_STAGES
from autohybts.exceptions import ConfigurationError
from autohybts.residuals.correctors import ResidualCorrector
from autohybts.residuals.generation import ResidualSettings
from autohybts.strategies.additive_residual import ResidualHybridForecaster

__all__ = ["stack_stages"]


def stack_stages(
    base: BaseForecaster,
    correctors: Sequence[ResidualCorrector],
    *,
    window_size: int,
    alpha: float,
    residual_source: str,
    settings: ResidualSettings | None = None,
) -> ResidualHybridForecaster:
    """Nest one residual hybrid per corrector, the first corrector closest to the base."""

    if not 1 <= len(correctors) <= MAX_STAGES:
        raise ConfigurationError(f"De 1 a {MAX_STAGES} estágios; recebidos {len(correctors)}.")
    model: BaseForecaster = base
    for corrector in correctors:
        model = ResidualHybridForecaster(
            model,
            corrector,
            window_size=window_size,
            alpha=alpha,
            residual_source=residual_source,
            settings=settings,
        )
    assert isinstance(model, ResidualHybridForecaster)
    return model
