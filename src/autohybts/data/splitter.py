"""Sealed test, forecast origins and the frozen temporal plan (architecture §7.1, §30.1).

Layout of a series of length ``T`` with ``test_size`` sealed points::

    |<------------------------ development history ------------------------>|<- sealed test ->|
    | train ... | V_search origins + targets | embargo | V_select origins + targets |

* An origin ``o`` sees ``y[train_start : o + 1]`` and is scored on ``y[o + 1 : o + 1 + H]``.
* ``V_select`` holds the most recent origins; its last targets end at the last development
  point. ``V_search`` origins come earlier, and every search target lies at or before the
  first select origin minus ``embargo``, so selection never re-scores labels the search
  already optimised on.
* The sealed test is split off before any of this and is not part of the plan.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Final, Literal, NoReturn, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

from autohybts.data.schema import SeriesFrame
from autohybts.domain.context import TemporalContext
from autohybts.domain.identity import stable_hash
from autohybts.exceptions import (
    ConfigurationError,
    InsufficientHistoryError,
    SealedTestAccessError,
    TemporalLeakageError,
)

__all__ = [
    "CrossValidationPlan",
    "FoldStrategy",
    "Layer",
    "Origin",
    "SealedTest",
    "TemporalCrossValidation",
    "TemporalPlan",
    "TemporalSplitter",
    "ensure_not_sealed",
    "split_sealed_test",
]

# "cv": origins of a plain rolling-origin cross-validation (no search/select split).
Layer: TypeAlias = Literal["search", "select", "cv"]
FoldStrategy: TypeAlias = Literal["expanding", "rolling"]
FOLD_STRATEGIES: Final[tuple[FoldStrategy, ...]] = ("expanding", "rolling")


class SealedTest:
    """Final hold-out segment, opaque to everything except terminal external evaluation.

    It cannot be iterated, indexed or converted to an array, and its ``repr`` shows no
    values. Only metadata is public. ``_unseal`` is reserved for the external evaluation
    of a frozen model (stage I8).
    """

    __slots__ = ("_cutoff", "_frame")

    def __init__(self, frame: SeriesFrame, cutoff: int) -> None:
        self._frame = frame
        self._cutoff = cutoff

    @property
    def series_id(self) -> str:
        return self._frame.series_id

    @property
    def size(self) -> int:
        return len(self._frame)

    @property
    def start(self) -> pd.Timestamp:
        return self._frame.start

    @property
    def end(self) -> pd.Timestamp:
        return self._frame.end

    @property
    def cutoff(self) -> int:
        """Position of the first sealed observation in the full series."""

        return self._cutoff

    @property
    def fingerprint(self) -> str:
        return self._frame.fingerprint

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"SealedTest(series_id={self.series_id!r}, size={self.size}, "
            f"start={self.start}, end={self.end})"
        )

    def __iter__(self) -> NoReturn:
        _refuse()

    def __getitem__(self, key: object) -> NoReturn:
        _refuse()

    def __array__(self, *args: object, **kwargs: object) -> NoReturn:
        _refuse()

    def _unseal(self) -> SeriesFrame:
        return self._frame


def _refuse() -> NoReturn:
    raise SealedTestAccessError(
        "O teste selado não pode ser lido durante busca, seleção ou diagnóstico; ele só é "
        "usado na avaliação externa de um modelo já finalizado."
    )


def ensure_not_sealed(*objects: object) -> None:
    """Raise ``SealedTestAccessError`` if any argument (or a direct container item) is sealed."""

    for item in objects:
        members: tuple[object, ...]
        if isinstance(item, dict):
            members = (*item.keys(), *item.values())
        elif isinstance(item, list | tuple | set | frozenset):
            members = tuple(item)
        else:
            members = (item,)
        if any(isinstance(member, SealedTest) for member in members):
            _refuse()


def split_sealed_test(frame: SeriesFrame, test_size: int) -> tuple[SeriesFrame, SealedTest | None]:
    """Split the last ``test_size`` points off as the sealed test; ``0`` means no sealed test."""

    _require_int("test_size", test_size, minimum=0)
    if test_size == 0:
        return frame, None
    if test_size >= len(frame):
        raise InsufficientHistoryError(
            f"test_size={test_size} não deixa histórico de desenvolvimento em {len(frame)} pontos."
        )
    cutoff = len(frame) - test_size
    return frame.head(cutoff), SealedTest(frame.tail(test_size), cutoff)


@dataclass(frozen=True, slots=True)
class Origin:
    """Forecast origin: fit on ``train_slice``, forecast ``horizon`` steps, score on targets."""

    series_id: str
    layer: Layer
    fold: int
    train_start: int
    origin_index: int
    origin_time: pd.Timestamp
    horizon: int

    @property
    def fold_id(self) -> str:
        return f"{self.layer}-{self.fold}"

    @property
    def origin_id(self) -> str:
        return f"{self.series_id}/{self.fold_id}@{self.origin_index}"

    @property
    def n_train(self) -> int:
        return self.origin_index - self.train_start + 1

    @property
    def train_slice(self) -> slice:
        return slice(self.train_start, self.origin_index + 1)

    @property
    def target_slice(self) -> slice:
        return slice(self.origin_index + 1, self.origin_index + 1 + self.horizon)

    @property
    def last_target_index(self) -> int:
        return self.origin_index + self.horizon


@dataclass(frozen=True, slots=True)
class TemporalSplitter:
    """Configuration of search and selection origins inside the development history.

    Origins are anchored at the end of the history: the last select origin forecasts the
    final ``horizon`` points, earlier origins step back by ``step``.

    Args:
        horizon: steps forecast from each origin.
        n_search_folds: origins in ``V_search`` (screening and tuning).
        n_select_folds: origins in ``V_select`` (finalist comparison).
        step: distance between consecutive origins of a layer; defaults to ``horizon``.
        fold_strategy: ``"expanding"`` trains on all history up to the origin; ``"rolling"``
            on the last ``window_size`` points.
        min_train_size: minimum training length at the earliest origin; defaults to
            ``2 * horizon`` (an engineering default, not a scientific recommendation).
        window_size: training length for ``"rolling"``; not allowed for ``"expanding"``.
        embargo: extra observations between the last search target and the first
            select origin.
    """

    horizon: int
    n_search_folds: int = 3
    n_select_folds: int = 3
    step: int | None = None
    fold_strategy: FoldStrategy = "expanding"
    min_train_size: int | None = None
    window_size: int | None = None
    embargo: int = 0

    def __post_init__(self) -> None:
        _require_int("horizon", self.horizon, minimum=1)
        _require_int("n_search_folds", self.n_search_folds, minimum=1)
        _require_int("n_select_folds", self.n_select_folds, minimum=1)
        _require_int("embargo", self.embargo, minimum=0)
        if self.step is not None:
            _require_int("step", self.step, minimum=1)
        if self.min_train_size is not None:
            _require_int("min_train_size", self.min_train_size, minimum=1)
        if self.fold_strategy not in FOLD_STRATEGIES:
            raise ConfigurationError(
                f"fold_strategy deve ser um de {FOLD_STRATEGIES}; recebido {self.fold_strategy!r}."
            )
        if self.fold_strategy == "expanding" and self.window_size is not None:
            raise ConfigurationError("window_size só vale para fold_strategy='rolling'.")
        if self.fold_strategy == "rolling":
            if self.window_size is None:
                raise ConfigurationError("fold_strategy='rolling' exige window_size.")
            _require_int("window_size", self.window_size, minimum=1)
            if self.window_size < self.effective_min_train_size:
                raise ConfigurationError(
                    f"window_size={self.window_size} é menor que min_train_size="
                    f"{self.effective_min_train_size}."
                )

    @property
    def effective_step(self) -> int:
        return self.step if self.step is not None else self.horizon

    @property
    def effective_min_train_size(self) -> int:
        return self.min_train_size if self.min_train_size is not None else 2 * self.horizon

    @property
    def train_length_required(self) -> int:
        """Observations needed before the earliest origin (inclusive)."""

        if self.fold_strategy == "rolling" and self.window_size is not None:
            return self.window_size
        return self.effective_min_train_size

    @property
    def required_length(self) -> int:
        """Smallest development history that fits every origin of the plan."""

        folds = self.n_search_folds + self.n_select_folds - 2
        return (
            self.train_length_required
            + 2 * self.horizon
            + folds * self.effective_step
            + self.embargo
        )

    def describe(self) -> dict[str, Any]:
        """Effective configuration (defaults resolved); input to the plan fingerprint."""

        return {
            "horizon": self.horizon,
            "n_search_folds": self.n_search_folds,
            "n_select_folds": self.n_select_folds,
            "step": self.effective_step,
            "fold_strategy": self.fold_strategy,
            "min_train_size": self.effective_min_train_size,
            "window_size": self.window_size,
            "embargo": self.embargo,
        }

    def plan(self, history: SeriesFrame) -> TemporalPlan:
        """Freeze the origins of ``history``, which must not include the sealed test."""

        ensure_not_sealed(history)
        if not isinstance(history, SeriesFrame):
            raise ConfigurationError(
                f"O plano exige um SeriesFrame; recebido {type(history).__name__}."
            )
        if len(history) < self.required_length:
            raise InsufficientHistoryError(
                f"Histórico de desenvolvimento com {len(history)} pontos; este plano exige ao "
                f"menos {self.required_length} (treino mínimo {self.train_length_required}, "
                f"{self.n_search_folds} origem(ns) de busca e {self.n_select_folds} de seleção, "
                f"horizonte {self.horizon}, passo {self.effective_step}, embargo {self.embargo}). "
                "Reduza dobras, passo ou horizonte, ou forneça mais histórico."
            )
        step = self.effective_step
        last_select = len(history) - 1 - self.horizon
        select = [last_select - k * step for k in reversed(range(self.n_select_folds))]
        last_search = select[0] - self.embargo - self.horizon
        search = [last_search - k * step for k in reversed(range(self.n_search_folds))]
        return TemporalPlan(
            splitter=self,
            history=history,
            search_origins=self._origins(history, "search", search),
            select_origins=self._origins(history, "select", select),
        )

    def _origins(
        self, history: SeriesFrame, layer: Layer, indices: list[int]
    ) -> tuple[Origin, ...]:
        rolling = self.fold_strategy == "rolling" and self.window_size is not None
        return tuple(
            Origin(
                series_id=history.series_id,
                layer=layer,
                fold=fold,
                train_start=index - self.window_size + 1 if rolling and self.window_size else 0,
                origin_index=index,
                origin_time=history.timestamps[index],
                horizon=self.horizon,
            )
            for fold, index in enumerate(indices)
        )


@dataclass(frozen=True, slots=True, eq=False)
class TemporalPlan:
    """Frozen origins over the development history; the unit every candidate is scored on.

    The constructor re-checks every temporal invariant, so a plan that exists is valid
    even if it was not built by ``TemporalSplitter.plan``.
    """

    splitter: TemporalSplitter
    history: SeriesFrame
    search_origins: tuple[Origin, ...]
    select_origins: tuple[Origin, ...]
    fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ensure_not_sealed(self.history)
        _check_invariants(self)
        object.__setattr__(
            self,
            "fingerprint",
            stable_hash(
                {
                    "schema": "temporal-plan/1",
                    "splitter": self.splitter.describe(),
                    "history": self.history.fingerprint,
                    "origins": [origin.origin_id for origin in self.origins()],
                }
            ),
        )

    @property
    def dev_end(self) -> int:
        """Position of the last development observation."""

        return len(self.history) - 1

    def origins(self, layer: Layer | None = None) -> tuple[Origin, ...]:
        if layer is None:
            return self.search_origins + self.select_origins
        if layer == "search":
            return self.search_origins
        if layer == "select":
            return self.select_origins
        raise ConfigurationError(f"Camada desconhecida: {layer!r}.")

    def __iter__(self) -> Iterator[Origin]:
        return iter(self.origins())

    def train_values(self, origin: Origin) -> npt.NDArray[np.float64]:
        """Read-only training values of ``origin``."""

        self._own(origin)
        return self.history.values[origin.train_slice]

    def target_values(self, origin: Origin) -> npt.NDArray[np.float64]:
        """Read-only values to score ``origin`` against; never features of the same forecast."""

        self._own(origin)
        return self.history.values[origin.target_slice]

    def context(self, origin: Origin) -> TemporalContext:
        """What a forecaster may see at ``origin``: observations up to and including it."""

        self._own(origin)
        return TemporalContext(
            series_id=origin.series_id,
            frequency=self.history.frequency,
            origin_index=origin.origin_index,
            origin_time=origin.origin_time,
            observed=self.history.values[origin.train_slice],
            observed_start=origin.train_start,
        )

    def to_frame(self) -> pd.DataFrame:
        """One row per origin, in chronological order (shown as ``exp.splits``)."""

        stamps = self.history.timestamps
        rows = [
            {
                "fold_id": origin.fold_id,
                "layer": origin.layer,
                "train_start": stamps[origin.train_start],
                "origin": origin.origin_time,
                "first_target": stamps[origin.origin_index + 1],
                "last_target": stamps[origin.last_target_index],
                "n_train": origin.n_train,
                "horizon": origin.horizon,
            }
            for origin in self.origins()
        ]
        return pd.DataFrame(rows).set_index("fold_id")

    def _own(self, origin: Origin) -> None:
        if origin not in self.origins():
            raise ConfigurationError(f"A origem {origin.origin_id} não pertence a este plano.")


def _check_invariants(plan: TemporalPlan) -> None:
    if not plan.search_origins or not plan.select_origins:
        raise ConfigurationError("O plano precisa de origens de busca e de seleção.")
    for layer, origins in (("search", plan.search_origins), ("select", plan.select_origins)):
        indices = [origin.origin_index for origin in origins]
        if indices != sorted(set(indices)):
            raise ConfigurationError(f"Origens de {layer} fora de ordem ou repetidas.")
        for fold, origin in enumerate(origins):
            if origin.layer != layer or origin.fold != fold:
                raise ConfigurationError(f"Origem {origin.origin_id} rotulada incorretamente.")
            if origin.series_id != plan.history.series_id:
                raise ConfigurationError(f"Origem {origin.origin_id} de outra série.")
            if origin.origin_time != plan.history.timestamps[origin.origin_index]:
                raise ConfigurationError(f"Origem {origin.origin_id} com data inconsistente.")
            if not 0 <= origin.train_start <= origin.origin_index:
                raise TemporalLeakageError(f"Treino de {origin.origin_id} começa depois da origem.")
            if origin.last_target_index > plan.dev_end:
                raise TemporalLeakageError(
                    f"Alvos de {origin.origin_id} passam do fim do desenvolvimento "
                    f"({plan.dev_end})."
                )
    first_select = plan.select_origins[0].origin_index
    last_search_target = max(origin.last_target_index for origin in plan.search_origins)
    if last_search_target + plan.splitter.embargo > first_select:
        raise TemporalLeakageError(
            f"Alvo de busca na posição {last_search_target} invade a seleção, que começa na "
            f"origem {first_select} (embargo {plan.splitter.embargo})."
        )


def _require_int(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        raise ConfigurationError(f"{name} deve ser inteiro >= {minimum}; recebido {value!r}.")


@dataclass(frozen=True, slots=True)
class TemporalCrossValidation:
    """Rolling-origin cross-validation over the development history (§7.1).

    Unlike ``TemporalSplitter`` there is a single layer: every origin is used to score and
    compare models on the same units. Origins are anchored at the end of the history and
    step back by ``step`` (default ``horizon``, i.e. non-overlapping target blocks).
    """

    horizon: int
    n_folds: int = 5
    step: int | None = None
    fold_strategy: FoldStrategy = "expanding"
    min_train_size: int | None = None
    window_size: int | None = None

    def __post_init__(self) -> None:
        # Same validation rules as the two-layer splitter.
        TemporalSplitter(
            horizon=self.horizon,
            n_search_folds=1,
            n_select_folds=self.n_folds,
            step=self.step,
            fold_strategy=self.fold_strategy,
            min_train_size=self.min_train_size,
            window_size=self.window_size,
        )

    @property
    def effective_step(self) -> int:
        return self.step if self.step is not None else self.horizon

    @property
    def train_length_required(self) -> int:
        if self.fold_strategy == "rolling" and self.window_size is not None:
            return self.window_size
        return self.min_train_size if self.min_train_size is not None else 2 * self.horizon

    @property
    def required_length(self) -> int:
        return self.train_length_required + self.horizon + (self.n_folds - 1) * self.effective_step

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "rolling_origin_cv",
            "horizon": self.horizon,
            "n_folds": self.n_folds,
            "step": self.effective_step,
            "fold_strategy": self.fold_strategy,
            "min_train_size": self.train_length_required,
            "window_size": self.window_size,
        }

    def split(self, history: SeriesFrame) -> CrossValidationPlan:
        ensure_not_sealed(history)
        if not isinstance(history, SeriesFrame):
            raise ConfigurationError(
                f"A validação cruzada exige um SeriesFrame; recebido {type(history).__name__}."
            )
        if len(history) < self.required_length:
            raise InsufficientHistoryError(
                f"Histórico com {len(history)} pontos; {self.n_folds} dobras de horizonte "
                f"{self.horizon} e passo {self.effective_step} exigem ao menos "
                f"{self.required_length}."
            )
        last = len(history) - 1 - self.horizon
        indices = [last - k * self.effective_step for k in reversed(range(self.n_folds))]
        rolling = self.fold_strategy == "rolling" and self.window_size is not None
        origins = tuple(
            Origin(
                series_id=history.series_id,
                layer="cv",
                fold=fold,
                train_start=index - self.window_size + 1 if rolling and self.window_size else 0,
                origin_index=index,
                origin_time=history.timestamps[index],
                horizon=self.horizon,
            )
            for fold, index in enumerate(indices)
        )
        return CrossValidationPlan(self, history, origins)


@dataclass(frozen=True, slots=True, eq=False)
class CrossValidationPlan:
    """Frozen rolling-origin folds; same accessors as ``TemporalPlan``."""

    splitter: TemporalCrossValidation
    history: SeriesFrame
    cv_origins: tuple[Origin, ...]
    fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ensure_not_sealed(self.history)
        indices = [o.origin_index for o in self.cv_origins]
        if not indices or indices != sorted(set(indices)):
            raise ConfigurationError(
                "Dobras da validação cruzada vazias, fora de ordem ou repetidas."
            )
        for origin in self.cv_origins:
            if not 0 <= origin.train_start <= origin.origin_index:
                raise TemporalLeakageError(f"Treino de {origin.origin_id} começa depois da origem.")
            if origin.last_target_index > len(self.history) - 1:
                raise TemporalLeakageError(f"Alvos de {origin.origin_id} passam do histórico.")
        object.__setattr__(
            self,
            "fingerprint",
            stable_hash(
                {
                    "schema": "cv-plan/1",
                    "splitter": self.splitter.describe(),
                    "history": self.history.fingerprint,
                    "origins": [origin.origin_id for origin in self.cv_origins],
                }
            ),
        )

    def origins(self, layer: Layer | None = None) -> tuple[Origin, ...]:
        if layer not in (None, "cv"):
            raise ConfigurationError(f"A validação cruzada só tem a camada 'cv'; pedido {layer!r}.")
        return self.cv_origins

    def __iter__(self) -> Iterator[Origin]:
        return iter(self.cv_origins)

    def train_values(self, origin: Origin) -> npt.NDArray[np.float64]:
        self._own(origin)
        return self.history.values[origin.train_slice]

    def target_values(self, origin: Origin) -> npt.NDArray[np.float64]:
        self._own(origin)
        return self.history.values[origin.target_slice]

    def context(self, origin: Origin) -> TemporalContext:
        self._own(origin)
        return TemporalContext(
            series_id=origin.series_id,
            frequency=self.history.frequency,
            origin_index=origin.origin_index,
            origin_time=origin.origin_time,
            observed=self.history.values[origin.train_slice],
            observed_start=origin.train_start,
        )

    def to_frame(self) -> pd.DataFrame:
        stamps = self.history.timestamps
        return pd.DataFrame(
            [
                {
                    "fold_id": origin.fold_id,
                    "layer": origin.layer,
                    "train_start": stamps[origin.train_start],
                    "origin": origin.origin_time,
                    "first_target": stamps[origin.origin_index + 1],
                    "last_target": stamps[origin.last_target_index],
                    "n_train": origin.n_train,
                    "horizon": origin.horizon,
                }
                for origin in self.cv_origins
            ]
        ).set_index("fold_id")

    def _own(self, origin: Origin) -> None:
        if origin not in self.cv_origins:
            raise ConfigurationError(f"A origem {origin.origin_id} não pertence a este plano.")
