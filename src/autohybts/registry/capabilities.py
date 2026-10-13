"""What a model can do; the search only generates candidates its capabilities allow (§15)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final

from autohybts.exceptions import ConfigurationError

__all__ = [
    "HORIZON_STRATEGIES",
    "OUTPUT_TYPES",
    "TARGET_MODES",
    "TRAINING_SCOPES",
    "Capabilities",
]

TRAINING_SCOPES: Final = ("local", "global", "pretrained")
OUTPUT_TYPES: Final = ("point", "quantile", "samples", "distribution")
HORIZON_STRATEGIES: Final = ("recursive", "direct", "mimo", "dirrec", "fixed")
TARGET_MODES: Final = ("univariate", "multivariate", "exogenous")


@dataclass(frozen=True, slots=True)
class Capabilities:
    """Declared capabilities of a registered model.

    Attributes:
        multi_step: forecasts several steps from one origin (needed by variant A).
        online_state_update: forecasts from a later context with frozen parameters,
            i.e. new observations update its state without refitting (needed by variant B
            and by out-of-fold residuals).
        in_sample_fitted: exposes ``fitted_values()`` (allows the ``in_sample`` residual source).
        deterministic: same data and parameters give the same forecast whatever the seed;
            ``False`` means the output depends on ``random_state``.
        requires_extra: optional dependency group needed to fit, e.g. ``"stats"``.
        training_scope: ``"local"`` (one series), ``"global"`` (trained on a panel; not
            evaluated by the univariate search) or ``"pretrained"`` (a foundation model
            used as is on each series).
        output_type: ``"point"``; ``"quantile"``, ``"samples"`` and ``"distribution"``
            need a probabilistic contract and are skipped by the search.
        horizon_strategy: how multi-step forecasts are produced: ``"recursive"``,
            ``"direct"``, ``"mimo"``, ``"dirrec"`` or ``"fixed"`` (a network trained for
            one horizon). It describes the model, not the A/B protocol.
        max_horizon: longest horizon the model forecasts from one origin (``None``: any);
            longer requests are skipped, not truncated.
        target_mode: ``"univariate"`` (one series from its own history), ``"multivariate"``
            (several targets forecast jointly) or ``"exogenous"`` (needs covariates, past or
            known in advance). The search is univariate: the other two are skipped until
            they get their own contract (plano de ensembles L03, F7).
    """

    multi_step: bool = True
    online_state_update: bool = True
    in_sample_fitted: bool = False
    deterministic: bool = True
    requires_extra: str | None = None
    training_scope: str = "local"
    output_type: str = "point"
    horizon_strategy: str = "recursive"
    max_horizon: int | None = None
    target_mode: str = "univariate"

    def __post_init__(self) -> None:
        for name, allowed in (
            ("training_scope", TRAINING_SCOPES),
            ("output_type", OUTPUT_TYPES),
            ("horizon_strategy", HORIZON_STRATEGIES),
            ("target_mode", TARGET_MODES),
        ):
            if getattr(self, name) not in allowed:
                raise ConfigurationError(
                    f"{name} deve ser um de {allowed}; recebido {getattr(self, name)!r}."
                )
        if self.max_horizon is not None and (
            isinstance(self.max_horizon, bool)
            or not isinstance(self.max_horizon, int)
            or self.max_horizon < 1
        ):
            raise ConfigurationError("max_horizon deve ser inteiro >= 1 ou None.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
