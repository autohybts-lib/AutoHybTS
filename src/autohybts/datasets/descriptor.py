"""Common data model of downloadable datasets (§32.3)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import pandas as pd

from autohybts.data.panel import PanelFrame
from autohybts.exceptions import ConfigurationError

__all__ = ["BenchmarkDataset", "BenchmarkSeries", "DatasetDescriptor", "SplitKind"]

SplitKind: TypeAlias = Literal["official", "chronological", "custom", "none"]


@dataclass(frozen=True)
class DatasetDescriptor:
    """Where a dataset came from and how it is split; kept in manifests (D07)."""

    provider: str
    dataset_id: str
    source_url: str
    version: str | None
    license: str | None
    frequency: str | None
    horizon: int | None
    target_name: str
    time_col: str | None
    series_id_col: str | None
    split_kind: SplitKind
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "dataset_id": self.dataset_id,
            "source_url": self.source_url,
            "version": self.version,
            "license": self.license,
            "frequency": self.frequency,
            "horizon": self.horizon,
            "target_name": self.target_name,
            "time_col": self.time_col,
            "series_id_col": self.series_id_col,
            "split_kind": self.split_kind,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class BenchmarkSeries:
    """One series with its official split: ``train`` for the search, ``test`` sealed.

    ``full()`` gives train and test back to back, so
    ``AutoHybridExperiment(fh=h, test_size=len(test)).fit(series.full())`` seals exactly
    the official test.
    """

    train: pd.Series
    test: pd.Series | None
    descriptor: DatasetDescriptor

    @property
    def series_id(self) -> str:
        return str(self.train.name)

    def full(self) -> pd.Series:
        if self.test is None:
            return self.train.copy()
        joined: pd.Series = pd.concat([self.train, self.test])
        joined.name = self.train.name
        return joined

    def __repr__(self) -> str:
        test = 0 if self.test is None else len(self.test)
        return (
            f"BenchmarkSeries({self.descriptor.provider}:{self.series_id}, "
            f"treino={len(self.train)}, teste={test}, freq={self.descriptor.frequency})"
        )


@dataclass(frozen=True)
class BenchmarkDataset:
    """Many series in long format (``series_id``, ``timestamp``, ``value``)."""

    train: pd.DataFrame
    test: pd.DataFrame | None
    metadata: pd.DataFrame | None
    descriptor: DatasetDescriptor

    @property
    def series_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.train["series_id"].astype(str)))

    def series(self, series_id: str) -> BenchmarkSeries:
        if series_id not in set(self.series_ids):
            raise ConfigurationError(
                f"Série {series_id!r} não está em {self.descriptor.dataset_id}."
            )
        train = _one(self.train, series_id)
        test = None if self.test is None else _one(self.test, series_id)
        extra = dict(self.descriptor.extra)
        if self.metadata is not None and series_id in self.metadata.index:
            extra.update({str(k): v for k, v in self.metadata.loc[series_id].to_dict().items()})
        horizon = self.descriptor.horizon if test is None else len(test)
        descriptor = DatasetDescriptor(
            **{
                **self.descriptor.to_dict(),
                "dataset_id": f"{self.descriptor.dataset_id}/{series_id}",
                "horizon": horizon,
                "extra": extra,
            }
        )
        return BenchmarkSeries(train, test, descriptor)

    def panel(
        self, series_ids: Sequence[str] | None = None, *, with_test: bool = True
    ) -> PanelFrame:
        """Series as a panel; ``with_test`` appends each official test (to be sealed)."""

        chosen = list(series_ids or self.series_ids)
        data = {}
        for sid in chosen:
            item = self.series(sid)
            data[sid] = item.full() if with_test else item.train
        return PanelFrame.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"BenchmarkDataset({self.descriptor.dataset_id}, séries={len(self.series_ids)}, "
            f"freq={self.descriptor.frequency}, horizonte={self.descriptor.horizon})"
        )


def _one(frame: pd.DataFrame, series_id: str) -> pd.Series:
    rows = frame[frame["series_id"].astype(str) == series_id]
    return pd.Series(
        rows["value"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(rows["timestamp"]),
        name=series_id,
    )
