"""Residual generation inside one training vector (architecture §7.2).

Every source works on the training slice ``y`` of a single origin and never sees data
after it. The tail of ``y`` holds the residuals the corrector learns from:

* ``in_sample``: ``y[t]`` minus the fitted value of the base fitted on all of ``y``.
  Explicitly experimental: fitted values have seen ``y[t]`` and can be optimistic.
* ``historical_oof_1step``: a separate base is fitted on the prefix before the tail, then
  forecasts each tail point one step ahead from the observations before it, with its
  parameters frozen.
* ``historical_oof_multistep``: the tail is cut into non-overlapping blocks; for each block
  a base is refitted on everything before it and forecasts the whole block, so errors
  carry horizons ``1..block_size`` like the multi-step errors of variant A.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np

from autohybts.adapters.base import BaseForecaster, FloatArray
from autohybts.domain.context import TemporalContext
from autohybts.exceptions import (
    ConfigurationError,
    InsufficientHistoryError,
    UnsupportedCapabilityError,
)
from autohybts.residuals.provenance import ResidualRecord, ResidualSeries
from autohybts.types import RESIDUAL_SOURCES

__all__ = ["OnFailure", "ResidualSettings", "build_residuals", "one_step_residuals"]

OnFailure: TypeAlias = Literal["raise", "fallback_to_base"]
ON_FAILURE: Final = ("raise", "fallback_to_base")


@dataclass(frozen=True, slots=True)
class ResidualSettings:
    """Experiment-wide settings of residual generation (part of every trial key).

    Args:
        tail_fraction: share of the training vector used as the residual tail.
        max_tail: cap on the tail length (``None`` for no cap).
        block_size: block length of ``historical_oof_multistep``; ``None`` splits the
            tail in two blocks.
        min_burn_in: points dropped at the start of ``in_sample`` fitted values.
        min_corrector_rows: fewest training rows the corrector may be fitted on.
        on_failure: ``"raise"`` makes a hybrid without enough residuals fail (scientific
            default: the candidate is ineligible); ``"fallback_to_base"`` serves the base
            forecast and records it (§16).
    """

    tail_fraction: float = 0.5
    max_tail: int | None = None
    block_size: int | None = None
    min_burn_in: int = 10
    min_corrector_rows: int = 10
    on_failure: OnFailure = "raise"

    def __post_init__(self) -> None:
        if not 0.0 < self.tail_fraction < 1.0:
            raise ConfigurationError("tail_fraction deve estar em (0, 1).")
        for name in ("max_tail", "block_size"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ConfigurationError(f"{name} deve ser inteiro >= 1 ou None.")
        for name in ("min_burn_in", "min_corrector_rows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigurationError(f"{name} deve ser inteiro >= 0.")
        if self.on_failure not in ON_FAILURE:
            raise ConfigurationError(f"on_failure deve ser um de {ON_FAILURE}.")

    def tail_length(self, n: int) -> int:
        tail = int(n * self.tail_fraction)
        return tail if self.max_tail is None else min(tail, self.max_tail)

    def describe(self) -> dict[str, object]:
        return {
            "tail_fraction": self.tail_fraction,
            "max_tail": self.max_tail,
            "block_size": self.block_size,
            "min_burn_in": self.min_burn_in,
            "min_corrector_rows": self.min_corrector_rows,
            "on_failure": self.on_failure,
        }


def build_residuals(
    template: BaseForecaster,
    fitted: BaseForecaster,
    y: FloatArray,
    source: str,
    settings: ResidualSettings,
    X: FloatArray | None = None,
) -> ResidualSeries:
    """Residuals of ``y`` for ``source``; ``fitted`` is ``template`` already fitted on ``y``.

    ``X`` holds the exogenous rows of ``y`` (ADR-0011). With exogenous features only
    one-step sources apply: a multi-step forecast inside the training vector would read
    ``past`` features further ahead than their lag allows.
    """

    if source not in RESIDUAL_SOURCES:
        raise ConfigurationError(f"Fonte residual desconhecida {source!r}.")
    if source == "in_sample":
        return _in_sample(fitted, y, settings)
    if X is not None and source != "historical_oof_1step":
        raise UnsupportedCapabilityError(
            f"A fonte {source} faz previsões de vários passos dentro do treino; com exógenas "
            "use historical_oof_1step."
        )
    tail = settings.tail_length(y.size)
    start = y.size - tail
    if tail < 1 or start < _min_train(template):
        raise InsufficientHistoryError(
            f"Treino com {y.size} pontos não comporta uma cauda de resíduos ({tail}) e um "
            f"prefixo com o mínimo de treino da base ({_min_train(template)})."
        )
    if source == "historical_oof_1step":
        prefix_model = template.clone().fit(y[:start], None if X is None else X[:start])
        return ResidualSeries(one_step_residuals(prefix_model, y, start, source, X))
    block = settings.block_size or max(1, -(-tail // 2))
    records: list[ResidualRecord] = []
    for block_start in range(start, y.size, block):
        size = min(block, y.size - block_start)
        model = template.clone().fit(y[:block_start])
        forecast = model.predict(size, local_context(y, block_start))
        origin = block_start - 1
        records.extend(
            _record(origin, origin + h, y, float(forecast[h - 1]), source)
            for h in range(1, size + 1)
        )
    return ResidualSeries(tuple(records))


def one_step_residuals(
    model: BaseForecaster,
    y: FloatArray,
    start: int,
    source: str,
    X: FloatArray | None = None,
) -> tuple[ResidualRecord, ...]:
    """One-step errors of a fitted, frozen ``model`` for targets ``start..len(y)-1``.

    Models may offer ``one_step_path(y, start)`` computing the same forecasts in one pass;
    the contract tests require it to match the step-by-step loop below. ``X`` holds the
    exogenous rows of ``y``; the forecast of target ``t`` sees rows up to ``t``.
    """

    fast = None if X is not None else getattr(model, "one_step_path", None)
    if callable(fast):
        path = np.asarray(fast(y, start), dtype=np.float64)
        if path.shape != (y.size - start,) or not np.isfinite(path).all():
            raise ConfigurationError(f"one_step_path de {type(model).__name__} inválido.")
        return tuple(
            _record(target - 1, target, y, float(path[target - start]), source)
            for target in range(start, y.size)
        )
    records = []
    for target in range(start, y.size):
        forecast = float(model.predict(1, local_context(y, target, X, ahead=1))[0])
        records.append(_record(target - 1, target, y, forecast, source))
    return tuple(records)


def local_context(
    y: FloatArray, end: int, X: FloatArray | None = None, ahead: int = 0
) -> TemporalContext:
    """Context whose observations are ``y[:end]`` (origin ``end - 1``).

    With ``X``, the context carries its rows up to ``end + ahead`` (``ahead`` dates after
    the origin, already checked against the features' lags by whoever built ``X``).
    """

    return TemporalContext(
        series_id="__residuals__",
        frequency="",
        origin_index=end - 1,
        origin_time=None,
        observed=y[:end],
        exog=None if X is None else X[: end + ahead],
        exog_ahead=ahead if X is not None else 0,
    )


def _in_sample(fitted: BaseForecaster, y: FloatArray, settings: ResidualSettings) -> ResidualSeries:
    fitted_values = getattr(fitted, "fitted_values", None)
    if not callable(fitted_values):
        raise UnsupportedCapabilityError(
            f"{type(fitted).__name__} não expõe valores ajustados; a fonte in_sample não se aplica."
        )
    values = np.asarray(fitted_values(), dtype=np.float64)
    memory = int(getattr(fitted, "min_history", 1))
    burn_in = max(settings.min_burn_in, memory)
    if values.shape != y.shape or burn_in >= y.size:
        raise InsufficientHistoryError("Valores ajustados insuficientes após o burn-in.")
    tail = settings.tail_length(y.size)
    start = max(burn_in, y.size - tail)
    return ResidualSeries(
        tuple(_record(t - 1, t, y, float(values[t]), "in_sample") for t in range(start, y.size))
    )


def _record(
    origin: int, target: int, y: FloatArray, forecast: float, source: str
) -> ResidualRecord:
    actual = float(y[target])
    return ResidualRecord(
        origin=origin,
        target=target,
        horizon=target - origin,
        y_true=actual,
        y_base=forecast,
        residual=actual - forecast,
        available_at=target,
        source=source,
    )


def _min_train(model: BaseForecaster) -> int:
    return int(getattr(model, "min_train_size", 1))
