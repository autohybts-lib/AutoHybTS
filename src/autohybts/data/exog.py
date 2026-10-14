"""Exogenous covariates and the availability rule that keeps them causal (ADR-0011).

Each column is ``past`` (known only up to the origin; the default) or ``future`` (known in
advance: calendar, holidays, planned prices). A model never receives a raw column: it
receives *features* ``(column, lag)`` whose value at the target time ``t`` is
``x[t - lag]``. A ``past`` feature needs ``lag >= step``, where ``step`` is the protocol's
horizon (``fh`` in variant A, 1 in variant B), so every value a model sees at a target date
was already observed at the origin. ``future`` features may use ``lag = 0``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from autohybts.data.validation import epoch_ns
from autohybts.domain.identity import canonical_json
from autohybts.domain.specs import ExogFeature
from autohybts.exceptions import ConfigurationError, DataValidationError

__all__ = [
    "EXOG_KINDS",
    "ExogDesign",
    "ExogFeature",
    "ExogFrame",
    "ExogState",
    "as_rows",
    "design_from",
    "features_from",
    "minimum_lag",
]

EXOG_KINDS: Final = ("past", "future")


def minimum_lag(variant: str, fh: int) -> int:
    """Smallest lag of a ``past`` feature: the steps forecast from one origin."""

    return 1 if variant == "B" else int(fh)


@dataclass(frozen=True, slots=True, eq=False)
class ExogDesign:
    """Feature matrix aligned with a timeline; row ``t`` is what a model may use at ``t``.

    Attributes:
        values: ``(len(timeline), k)`` floats; NaN where a feature is not defined (the
            first ``lag`` rows, or dates the covariates do not cover).
        features: the features, in column order.
        max_ahead: rows after an origin whose values were known at that origin: the
            smallest lag of a ``past`` feature (``None`` when every feature is ``future``).
    """

    values: npt.NDArray[np.float64]
    features: tuple[ExogFeature, ...]
    max_ahead: int | None

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.features):
            raise ConfigurationError("A matriz de exógenas deve ter uma coluna por feature.")
        values.flags.writeable = False
        object.__setattr__(self, "values", values)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    def first_complete(self, start: int = 0) -> int:
        """First row ``>= start`` from which every row up to the end of the run is defined."""

        defined = np.isfinite(self.values).all(axis=1)
        missing = np.flatnonzero(~defined[start:])
        leading = 0
        while leading < missing.size and missing[leading] == leading:
            leading += 1
        return start + leading

    def rows(self, start: int, stop: int) -> npt.NDArray[np.float64]:
        """Rows ``[start, stop)``; every value must be defined."""

        if not 0 <= start <= stop <= self.values.shape[0]:
            raise DataValidationError(
                f"As exógenas cobrem {self.values.shape[0]} posições; pedido [{start}, {stop})."
            )
        block = self.values[start:stop]
        if not np.isfinite(block).all():
            raise DataValidationError(
                f"Exógenas ausentes entre as posições {start} e {stop - 1}: forneça os valores "
                "(colunas future precisam de valores no horizonte da previsão)."
            )
        return block


@dataclass(frozen=True, slots=True, eq=False)
class ExogFrame:
    """Validated covariates of one series, with the kind of each column.

    Build it with ``ExogFrame.from_frame(X, future=[...])``. The index may extend past the
    series (values of ``future`` columns for the forecast horizon).
    """

    timestamps: pd.DatetimeIndex
    values: npt.NDArray[np.float64]
    columns: tuple[str, ...]
    kinds: tuple[str, ...]
    fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        timestamps = pd.DatetimeIndex(self.timestamps)
        values = np.array(self.values, dtype=np.float64)
        if values.ndim != 2 or values.shape != (len(timestamps), len(self.columns)):
            raise DataValidationError(
                "As exógenas devem ter uma linha por data e uma coluna por nome."
            )
        if len(set(self.columns)) != len(self.columns) or not all(self.columns):
            raise DataValidationError(
                "Os nomes das colunas exógenas devem ser únicos e não vazios."
            )
        if len(self.kinds) != len(self.columns) or set(self.kinds) - set(EXOG_KINDS):
            raise ConfigurationError(f"Cada coluna precisa de um tipo em {EXOG_KINDS}.")
        if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
            raise DataValidationError("As datas das exógenas devem ser crescentes e únicas.")
        if not np.isfinite(values).all():
            raise DataValidationError(
                "As exógenas contêm valores ausentes ou não finitos; trate-os antes "
                "(uma lacuna seria preenchida com informação de outra data)."
            )
        values.flags.writeable = False
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "fingerprint", _fingerprint(self))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame | pd.Series, future: Iterable[str] = ()) -> ExogFrame:
        """Covariates from a ``DataFrame`` indexed by date; ``future`` names known columns."""

        if isinstance(frame, pd.Series):
            frame = frame.to_frame()
        if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
            raise DataValidationError("X deve ser um DataFrame indexado por DatetimeIndex.")
        if frame.shape[1] == 0:
            raise DataValidationError("X não tem colunas.")
        future = tuple(future)
        unknown = set(future) - {str(c) for c in frame.columns}
        if unknown:
            raise ConfigurationError(f"Colunas future inexistentes em X: {sorted(unknown)}.")
        try:
            values = frame.astype(np.float64).to_numpy()
        except (TypeError, ValueError) as error:
            raise DataValidationError("As exógenas devem ser numéricas (ou booleanas).") from error
        columns = tuple(str(c) for c in frame.columns)
        kinds = tuple("future" if c in future else "past" for c in columns)
        return cls(frame.index, values, columns, kinds)

    def kind(self, column: str) -> str:
        try:
            return self.kinds[self.columns.index(column)]
        except ValueError as error:
            raise ConfigurationError(f"Coluna exógena {column!r} não existe.") from error

    def default_features(
        self, step: int, columns: Sequence[str] | None = None
    ) -> tuple[ExogFeature, ...]:
        """Each column at its smallest causal lag: 0 for ``future``, ``step`` for ``past``."""

        chosen = self.columns if columns is None else tuple(columns)
        return tuple(ExogFeature(c, 0 if self.kind(c) == "future" else step) for c in chosen)

    def check(self, features: Sequence[ExogFeature], step: int) -> None:
        """Refuse a ``past`` feature lagged by less than ``step`` (it would leak)."""

        for feature in features:
            if self.kind(feature.column) == "past" and feature.lag < step:
                raise ConfigurationError(
                    f"{feature.column!r} é past: precisa de lag >= {step} (passos previstos de "
                    f"uma origem); recebido {feature.lag}. Declare-a em future se for "
                    "conhecida antes."
                )

    def design(
        self, features: Sequence[ExogFeature], timeline: pd.DatetimeIndex, step: int
    ) -> ExogDesign:
        """Feature matrix on ``timeline`` (the series dates, optionally extended)."""

        features = tuple(features)
        if not features:
            raise ConfigurationError("Informe ao menos uma feature exógena.")
        self.check(features, step)
        return design_from(
            self.to_frame(), dict(zip(self.columns, self.kinds, strict=True)), features, timeline
        )

    def to_frame(self) -> pd.DataFrame:
        """The covariates as a ``DataFrame`` indexed by date (a copy)."""

        return pd.DataFrame(
            np.array(self.values), index=self.timestamps, columns=list(self.columns)
        )

    def describe(self) -> dict[str, object]:
        return {
            "columns": list(self.columns),
            "kinds": dict(zip(self.columns, self.kinds, strict=True)),
            "fingerprint": self.fingerprint,
        }


def design_from(
    covariates: pd.DataFrame,
    kinds: Mapping[str, str],
    features: Sequence[ExogFeature],
    timeline: pd.DatetimeIndex,
) -> ExogDesign:
    """Feature matrix on ``timeline`` from raw covariates that may have gaps.

    A gap only matters if a row that uses it is requested (``ExogDesign.rows`` refuses it).
    """

    missing = {f.column for f in features} - set(covariates.columns)
    if missing:
        raise ConfigurationError(f"Colunas exógenas ausentes: {sorted(missing)}.")
    aligned = covariates.reindex(pd.DatetimeIndex(timeline))
    columns = [aligned[f.column].shift(f.lag).to_numpy(dtype=np.float64) for f in features]
    lags = [f.lag for f in features if kinds.get(f.column, "past") == "past"]
    return ExogDesign(np.column_stack(columns), tuple(features), min(lags) if lags else None)


@dataclass(frozen=True, eq=False)
class ExogState:
    """Covariates of a finalized model: raw values up to its last observation.

    ``start`` is the first position of the model's training vector (earlier rows lack a
    lagged feature); forecasting contexts start there too. Values for later dates come
    with each forecast (``future`` columns) or with new observations.
    """

    covariates: pd.DataFrame
    kinds: Mapping[str, str]
    features: tuple[ExogFeature, ...]
    start: int

    @classmethod
    def build(
        cls,
        covariates: pd.DataFrame,
        kinds: Mapping[str, str],
        features: Sequence[ExogFeature],
        timeline: pd.DatetimeIndex,
    ) -> ExogState:
        """State for a model trained on ``timeline``; covariates after it are dropped."""

        kept = covariates.loc[covariates.index <= timeline[-1]].copy()
        design = design_from(kept, kinds, features, timeline)
        return cls(kept, dict(kinds), tuple(features), design.first_complete())

    @property
    def max_ahead(self) -> int | None:
        lags = [f.lag for f in self.features if self.kinds.get(f.column, "past") == "past"]
        return min(lags) if lags else None

    @property
    def future_columns(self) -> tuple[str, ...]:
        return tuple(
            sorted({f.column for f in self.features if self.kinds.get(f.column) == "future"})
        )

    def extended(
        self,
        rows: pd.DataFrame | pd.Series | Mapping[str, float] | None,
        at: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Covariates plus ``rows`` (new dates appended, repeated dates replaced)."""

        if rows is None:
            return self.covariates
        frame = as_rows(rows, at)
        unknown = set(frame.columns) - set(self.covariates.columns)
        if unknown:
            raise ConfigurationError(f"Colunas exógenas desconhecidas: {sorted(unknown)}.")
        combined = frame.combine_first(self.covariates)
        return combined[list(self.covariates.columns)].sort_index()

    def design(self, timeline: pd.DatetimeIndex, rows: pd.DataFrame | None = None) -> ExogDesign:
        return design_from(self.extended(rows), self.kinds, self.features, timeline)

    def describe(self) -> dict[str, object]:
        return {
            "features": [f.describe() for f in self.features],
            "kinds": {c: self.kinds[c] for c in sorted({f.column for f in self.features})},
        }


def as_rows(
    rows: pd.DataFrame | pd.Series | Mapping[str, float], at: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Covariate rows as a numeric ``DataFrame`` indexed by date.

    A ``Series`` or a mapping is one row of values by column, dated ``at``.
    """

    if isinstance(rows, pd.DataFrame):
        frame = rows
    else:
        if at is None:
            raise ConfigurationError("Informe a data da linha de exógenas.")
        frame = pd.DataFrame([dict(rows)], index=pd.DatetimeIndex([at]))
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataValidationError("As exógenas devem ser indexadas por data.")
    try:
        return frame.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise DataValidationError("As exógenas devem ser numéricas (ou booleanas).") from error


def features_from(
    spec: Mapping[str, int] | Sequence[ExogFeature] | Sequence[Sequence[object]],
) -> tuple[ExogFeature, ...]:
    """Features from ``{column: lag}``, ``[(column, lag), ...]`` or ``ExogFeature`` objects."""

    if isinstance(spec, Mapping):
        return tuple(ExogFeature(str(c), int(lag)) for c, lag in spec.items())
    out = []
    for item in spec:
        if isinstance(item, ExogFeature):
            out.append(item)
        else:
            column, lag = item
            out.append(ExogFeature(str(column), int(lag)))  # type: ignore[call-overload]
    return tuple(out)


def _fingerprint(frame: ExogFrame) -> str:
    digest = hashlib.sha256()
    meta = {"schema": "exog-frame/1", "columns": list(frame.columns), "kinds": list(frame.kinds)}
    digest.update(canonical_json(meta).encode("utf-8"))
    digest.update(np.ascontiguousarray(epoch_ns(frame.timestamps), dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(frame.values + 0.0, dtype="<f8").tobytes())
    return digest.hexdigest()
