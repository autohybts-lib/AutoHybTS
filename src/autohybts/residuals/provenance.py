"""Residual records with explicit provenance and availability (ADR-0002).

Positions are indices into the vector the residuals were generated from (the training
slice of one origin). A residual ``r = y[target] - b`` comes from a base forecast made
with data up to ``origin`` for ``horizon = target - origin`` steps ahead, and only becomes
known once ``y[target]`` is observed, so ``available_at == target`` for every source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from autohybts.exceptions import ConfigurationError
from autohybts.types import RESIDUAL_SOURCES

__all__ = ["ONLINE_SOURCE", "ResidualRecord", "ResidualSeries"]

# Residuals appended after the fit, from observations that arrived later (variant B).
ONLINE_SOURCE: Final = "online_1step"
_SOURCES: Final = (*RESIDUAL_SOURCES, ONLINE_SOURCE)


@dataclass(frozen=True, slots=True)
class ResidualRecord:
    origin: int
    target: int
    horizon: int
    y_true: float
    y_base: float
    residual: float
    available_at: int
    source: str

    def __post_init__(self) -> None:
        if self.source not in _SOURCES:
            raise ConfigurationError(f"Fonte de resíduo desconhecida {self.source!r}.")
        if self.horizon < 1 or self.target - self.origin != self.horizon:
            raise ConfigurationError(
                f"Resíduo incoerente: origem {self.origin}, alvo {self.target}, horizonte "
                f"{self.horizon}."
            )
        if self.available_at != self.target:
            raise ConfigurationError(
                "Um resíduo só existe quando o alvo é observado: available_at deve ser o alvo."
            )
        if not np.isclose(self.residual, self.y_true - self.y_base, rtol=0, atol=1e-9):
            raise ConfigurationError("residual deve ser y_true - y_base.")


@dataclass(frozen=True, slots=True, eq=False)
class ResidualSeries:
    """Residual records ordered by target, at most one per target.

    ``policy`` documents how multiple forecasts of the same target are handled; every
    generator here produces one forecast per target, so the policy is
    ``"one_forecast_per_target"`` and duplicates are rejected.
    """

    records: tuple[ResidualRecord, ...]
    policy: str = "one_forecast_per_target"
    targets: npt.NDArray[np.int64] = field(init=False, repr=False)
    residuals: npt.NDArray[np.float64] = field(init=False, repr=False)
    available_at: npt.NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        targets = np.array([r.target for r in self.records], dtype=np.int64)
        if targets.size and (np.diff(targets) <= 0).any():
            raise ConfigurationError("Resíduos fora de ordem ou com alvo repetido.")
        for name, values in (
            ("targets", targets),
            ("residuals", np.array([r.residual for r in self.records], dtype=np.float64)),
            ("available_at", np.array([r.available_at for r in self.records], dtype=np.int64)),
        ):
            values.flags.writeable = False
            object.__setattr__(self, name, values)

    def __len__(self) -> int:
        return len(self.records)

    def extend(self, records: tuple[ResidualRecord, ...]) -> ResidualSeries:
        return ResidualSeries(self.records + records, self.policy)

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted({record.source for record in self.records}))

    def to_frame(self) -> pd.DataFrame:
        """Provenance audit table (§17.2)."""

        return pd.DataFrame(
            [
                {
                    "origin": r.origin,
                    "target": r.target,
                    "horizon": r.horizon,
                    "y_true": r.y_true,
                    "y_base": r.y_base,
                    "residual": r.residual,
                    "available_at": r.available_at,
                    "source": r.source,
                }
                for r in self.records
            ]
        )
