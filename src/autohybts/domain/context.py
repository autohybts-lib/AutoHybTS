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

    ``exog`` (optional, ADR-0011) holds exogenous features for the observed positions and
    ``exog_ahead`` rows after the origin. Row ``t`` holds what is usable at date ``t``;
    rows after the origin are allowed only up to ``exog_max_ahead`` (the smallest lag of a
    ``past`` feature; ``None`` when every feature is known in advance).
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
    exog: npt.NDArray[np.float64] | None = None
    exog_ahead: int = 0
    exog_max_ahead: int | None = None

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
        if self.exog is not None:
            self._check_exog(observed.size)

    def _check_exog(self, n_observed: int) -> None:
        exog = _read_only(np.array(self.exog, dtype=np.float64))
        object.__setattr__(self, "exog", exog)
        ahead = self.exog_ahead
        if isinstance(ahead, bool) or not isinstance(ahead, int) or ahead < 0:
            raise ConfigurationError(f"exog_ahead deve ser inteiro >= 0; recebido {ahead!r}.")
        if exog.ndim != 2 or exog.shape[0] != n_observed + ahead:
            raise ConfigurationError(
                f"As exógenas devem ter {n_observed + ahead} linhas (observações e "
                f"{ahead} à frente); recebidas {exog.shape}."
            )
        if self.exog_max_ahead is not None and ahead > self.exog_max_ahead:
            raise TemporalLeakageError(
                f"Exógenas {ahead} passos à frente da origem {self.origin_index}, mas uma "
                f"coluna past só é conhecida {self.exog_max_ahead} passos à frente."
            )
        if not np.isfinite(exog).all():
            raise ConfigurationError("As exógenas do contexto têm valores ausentes.")

    @property
    def exog_history(self) -> npt.NDArray[np.float64]:
        """Exogenous rows of the observed positions."""

        if self.exog is None:
            raise ConfigurationError("Contexto sem exógenas.")
        return self.exog[: self.observed.size]

    def exog_future(self, horizon: int) -> npt.NDArray[np.float64]:
        """Exogenous rows of the next ``horizon`` dates; refuses more than ``exog_ahead``."""

        if self.exog is None:
            raise ConfigurationError("Contexto sem exógenas.")
        if horizon > self.exog_ahead:
            raise ConfigurationError(
                f"Previsão de {horizon} passos com exógenas para só {self.exog_ahead}: forneça "
                "os valores do horizonte."
            )
        start = self.observed.size
        return self.exog[start : start + horizon]


_ArrayT = TypeVar("_ArrayT", bound=np.ndarray[Any, Any])


def _read_only(array: _ArrayT) -> _ArrayT:
    array.flags.writeable = False
    return array
