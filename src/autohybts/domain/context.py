"""What a forecaster may see at an origin (architecture §11.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd

from autohybts.exceptions import ConfigurationError, TemporalLeakageError

__all__ = ["TemporalContext"]


def _empty_float() -> npt.NDArray[np.float64]:
    return np.empty(0, dtype=np.float64)


def _empty_int() -> npt.NDArray[np.int64]:
    return np.empty(0, dtype=np.int64)


@dataclass(frozen=True, slots=True, eq=False)
class TemporalContext:
    """Information available at ``origin_index``; positions refer to the development history.

    ``observed`` holds ``y[observed_start : origin_index + 1]`` and must end exactly at
    the origin. Each residual carries the position at which it became observable, and
    none may be later than the origin. Violations raise ``TemporalLeakageError`` here,
    at construction, instead of surfacing as optimistic scores later.
    """

    series_id: str
    frequency: str
    origin_index: int
    # None for contexts built inside a fit (e.g. residual generation on a bare vector).
    origin_time: pd.Timestamp | None
    observed: npt.NDArray[np.float64]
    observed_start: int = 0
    residuals: npt.NDArray[np.float64] = field(default_factory=_empty_float)
    residuals_available_at: npt.NDArray[np.int64] = field(default_factory=_empty_int)

    def __post_init__(self) -> None:
        observed = _read_only(np.array(self.observed, dtype=np.float64))
        residuals = _read_only(np.array(self.residuals, dtype=np.float64))
        available_at = _read_only(np.array(self.residuals_available_at, dtype=np.int64))
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "residuals_available_at", available_at)

        if self.observed_start < 0 or observed.ndim != 1 or observed.size == 0:
            raise ConfigurationError("Contexto sem observações válidas.")
        last_observed = self.observed_start + observed.size - 1
        if last_observed > self.origin_index:
            raise TemporalLeakageError(
                f"O contexto da origem {self.origin_index} contém observações até a posição "
                f"{last_observed}."
            )
        if last_observed < self.origin_index:
            raise ConfigurationError(
                f"O contexto da origem {self.origin_index} termina na posição {last_observed}."
            )
        if residuals.shape != available_at.shape or residuals.ndim != 1:
            raise ConfigurationError("Cada resíduo precisa de exatamente um `available_at`.")
        if available_at.size and int(np.max(available_at)) > self.origin_index:
            raise TemporalLeakageError(
                f"Resíduo disponível só na posição {int(np.max(available_at))} usado na origem "
                f"{self.origin_index}."
            )


_ArrayT = TypeVar("_ArrayT", bound=np.ndarray[Any, Any])


def _read_only(array: _ArrayT) -> _ArrayT:
    array.flags.writeable = False
    return array
