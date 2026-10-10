"""Panels: several regular series with the same frequency, evaluated side by side (I12).

Each series keeps its own sealed test, its own origins and its own MASE scale; the
panel only groups them so candidates are compared on every series at once (units are
``series:fold``). Series may have different lengths and start dates; the same splitter
configuration is applied to each.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from autohybts.data.schema import SINGLE_SERIES_ID, SeriesFrame
from autohybts.data.splitter import (
    CrossValidationPlan,
    SealedTest,
    TemporalCrossValidation,
    TemporalPlan,
    TemporalSplitter,
    ensure_not_sealed,
    split_sealed_test,
)
from autohybts.domain.identity import stable_hash
from autohybts.exceptions import (
    ConfigurationError,
    DataValidationError,
    InsufficientHistoryError,
)

__all__ = ["PanelFrame", "PanelPlan", "plan_panel", "split_panel_sealed_test"]


@dataclass(frozen=True, eq=False)
class PanelFrame:
    """Ordered collection of ``SeriesFrame`` with unique ids and one frequency."""

    series: Mapping[str, SeriesFrame]
    fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.series:
            raise DataValidationError("O painel está vazio.")
        for series_id, frame in self.series.items():
            if not isinstance(frame, SeriesFrame) or frame.series_id != series_id:
                raise DataValidationError(f"Série {series_id!r} inconsistente no painel.")
        frequencies = {frame.frequency for frame in self.series.values()}
        if len(frequencies) != 1:
            raise DataValidationError(f"Séries com frequências diferentes: {sorted(frequencies)}.")
        if len(self.series) > 1 and SINGLE_SERIES_ID in self.series:
            raise DataValidationError(f"{SINGLE_SERIES_ID!r} é reservado para séries únicas.")
        object.__setattr__(self, "series", dict(self.series))
        object.__setattr__(
            self,
            "fingerprint",
            stable_hash(
                {
                    "schema": "panel/1",
                    "series": {sid: frame.fingerprint for sid, frame in self.series.items()},
                }
            ),
        )

    @classmethod
    def single(cls, frame: SeriesFrame) -> PanelFrame:
        return cls({frame.series_id: frame})

    @classmethod
    def from_dict(cls, data: Mapping[str, pd.Series], frequency: str | None = None) -> PanelFrame:
        """One ``pd.Series`` (indexed by date) per series id."""

        return cls(
            {
                str(sid): SeriesFrame.from_series(values, frequency=frequency, series_id=str(sid))
                for sid, values in data.items()
            }
        )

    @classmethod
    def from_long(
        cls,
        data: pd.DataFrame,
        *,
        series_col: str,
        date_col: str,
        target: str,
        frequency: str | None = None,
    ) -> PanelFrame:
        """Long table with one row per (series, date); columns are named explicitly."""

        missing = [c for c in (series_col, date_col, target) if c not in data.columns]
        if missing:
            raise ConfigurationError(
                f"Coluna(s) {missing} não encontrada(s). Disponíveis: {list(data.columns)}."
            )
        series = {}
        for sid, group in data.groupby(series_col, sort=False):
            series[str(sid)] = SeriesFrame.from_pandas(
                group, target=target, date_col=date_col, frequency=frequency, series_id=str(sid)
            )
        return cls(series)

    @property
    def series_ids(self) -> tuple[str, ...]:
        return tuple(self.series)

    @property
    def frequency(self) -> str:
        return next(iter(self.series.values())).frequency

    @property
    def is_single(self) -> bool:
        return len(self.series) == 1

    def __getitem__(self, series_id: str) -> SeriesFrame:
        try:
            return self.series[series_id]
        except KeyError:
            raise ConfigurationError(
                f"Série {series_id!r} não está no painel; disponíveis: {list(self.series)}."
            ) from None

    def __iter__(self) -> Iterator[SeriesFrame]:
        return iter(self.series.values())

    def __len__(self) -> int:
        return len(self.series)

    def to_long(self) -> pd.DataFrame:
        return pd.concat(
            [
                pd.DataFrame(
                    {"series_id": sid, "timestamp": frame.timestamps, "value": frame.values}
                )
                for sid, frame in self.series.items()
            ],
            ignore_index=True,
        )


def split_panel_sealed_test(
    panel: PanelFrame, test_size: int
) -> tuple[PanelFrame, dict[str, SealedTest]]:
    """The last ``test_size`` points of every series are sealed."""

    history, sealed = {}, {}
    for sid, frame in panel.series.items():
        dev, test = split_sealed_test(frame, test_size)
        history[sid] = dev
        if test is not None:
            sealed[sid] = test
    return PanelFrame(history), sealed


@dataclass(frozen=True, eq=False)
class PanelPlan:
    """One temporal plan (or CV plan) per series, built with the same configuration."""

    plans: Mapping[str, TemporalPlan | CrossValidationPlan]
    fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.plans:
            raise ConfigurationError("Plano de painel vazio.")
        ensure_not_sealed(*self.plans.values())
        object.__setattr__(self, "plans", dict(self.plans))
        object.__setattr__(
            self,
            "fingerprint",
            stable_hash(
                {
                    "schema": "panel-plan/1",
                    "plans": {s: p.fingerprint for s, p in self.plans.items()},
                }
            ),
        )

    @property
    def series_ids(self) -> tuple[str, ...]:
        return tuple(self.plans)

    @property
    def frequency(self) -> str:
        return next(iter(self.plans.values())).history.frequency

    def __getitem__(self, series_id: str) -> TemporalPlan | CrossValidationPlan:
        return self.plans[series_id]

    def to_frame(self) -> pd.DataFrame:
        """Origins of every series; indexed by ``series_id`` and ``fold_id``."""

        frames = []
        for sid, plan in self.plans.items():
            frame = plan.to_frame().reset_index()
            frame.insert(0, "series_id", sid)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True).set_index(["series_id", "fold_id"])

    def describe(self) -> dict[str, Any]:
        first = next(iter(self.plans.values()))
        return {"series": list(self.plans), "splitter": first.splitter.describe()}


def plan_panel(
    splitter: TemporalSplitter | TemporalCrossValidation, panel: PanelFrame
) -> PanelPlan:
    """Apply ``splitter`` to every series; a series too short raises with its id."""

    plans: dict[str, TemporalPlan | CrossValidationPlan] = {}
    for sid, frame in panel.series.items():
        try:
            plans[sid] = (
                splitter.plan(frame)
                if isinstance(splitter, TemporalSplitter)
                else splitter.split(frame)
            )
        except (ConfigurationError, InsufficientHistoryError) as error:
            raise type(error)(f"Série {sid!r}: {error}") from error
    return PanelPlan(plans)
