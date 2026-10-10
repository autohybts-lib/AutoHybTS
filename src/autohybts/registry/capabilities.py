"""What a model can do; the search only generates candidates its capabilities allow (§15)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["Capabilities"]


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
    """

    multi_step: bool = True
    online_state_update: bool = True
    in_sample_fitted: bool = False
    deterministic: bool = True
    requires_extra: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
