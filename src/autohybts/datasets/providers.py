"""Dataset providers: M4, M3 and Rdatasets (§32.1, phases D2–D3).

Each provider downloads through a ``CacheManager`` from its own approved domains and
turns the source format into ``BenchmarkDataset``/``BenchmarkSeries``. Official test
sets are kept apart from training data so that experiments can seal them.

Calendar dates: M4 gives a starting date per series (two-digit years, known to be
unreliable) and M3 gives decimal years. When a source date cannot produce a valid
regular index (or the series has no calendar, like M3 "OTHER"), a synthetic index is
used and ``index_source="synthetic"`` is recorded; the values and their order are what
matters for forecasting.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterator, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from autohybts.datasets.cache import CacheManager, RemoteFile, safe_extract_member
from autohybts.datasets.descriptor import BenchmarkDataset, BenchmarkSeries, DatasetDescriptor
from autohybts.exceptions import (
    ConfigurationError,
    DataValidationError,
    UnsupportedCapabilityError,
)

__all__ = [
    "PROVIDERS",
    "M3Provider",
    "M4Provider",
    "RdatasetsProvider",
    "regular_index",
]

# kind -> (period alias, timestamp alias tried in order for pandas 2 and 3)
_KINDS: Final = {
    "yearly": "Y",
    "quarterly": "Q",
    "monthly": "M",
    "weekly": "W",
    "daily": "D",
    "hourly": "h",
}
_SYNTHETIC_START: Final = pd.Timestamp("2000-01-01")


def regular_index(
    kind: str, start: pd.Timestamp | None, periods: int
) -> tuple[pd.DatetimeIndex, str]:
    """Regular timestamps of ``kind`` from ``start``; synthetic start if impossible."""

    if kind not in _KINDS:
        raise ConfigurationError(f"Tipo de frequência desconhecido {kind!r}.")
    if start is not None:
        try:
            return _index(kind, start, periods), "source"
        except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime):
            pass
    first = _SYNTHETIC_START
    if kind == "yearly" and periods > 200:  # keep long yearly series inside pandas bounds
        first = pd.Timestamp(year=max(1678, 2200 - periods), month=1, day=1)
    try:
        return _index(kind, first, periods), "synthetic"
    except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime) as error:
        raise DataValidationError(
            f"Série {kind} com {periods} pontos não cabe no intervalo de datas do pandas."
        ) from error


def _index(kind: str, start: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    if kind == "hourly":
        for alias in ("h", "H"):
            try:
                return pd.date_range(start.floor("h"), periods=periods, freq=alias)
            except ValueError:
                continue
        raise ValueError("frequência horária indisponível")
    period = pd.period_range(start=start, periods=periods, freq=_KINDS[kind])
    index = period.to_timestamp(how="start")
    # Keep inside the nanosecond range of pandas 2 (pandas 3 accepts more, CSV does not).
    if index[0].year < 1678 or index[-1].year > 2261:
        raise OverflowError("fora do intervalo")
    return pd.DatetimeIndex(index)


def _long(series_id: str, index: pd.DatetimeIndex, values: Sequence[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"series_id": series_id, "timestamp": index, "value": np.asarray(values, dtype=float)}
    )


# ------------------------------------------------------------------------------------ M4
_M4_BASE: Final = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"
_M4_SUBSETS: Final = {
    "yearly": "Yearly",
    "quarterly": "Quarterly",
    "monthly": "Monthly",
    "weekly": "Weekly",
    "daily": "Daily",
    "hourly": "Hourly",
}
# SHA-256 of files checked when this provider was written; others are pinned on first use.
_M4_PINNED: Final = {
    "M4-info.csv": "26a75cc9b9f65e351b39b0bdbb7e3464ac837df0baf541ab55abdd2ba1a14c88",
    "Daily-train.csv": "78e94591c60c06309f1e544fd7b2ccba28f3616b01913dd336a1d1d98483a1ec",
    "Daily-test.csv": "633b0815626ae266d36872c4d53e95eb93ee8d43e79587fceec519eba3e6b05e",
}


class M4Provider:
    """M4 competition: official train/test per frequency and ``M4-info.csv`` (D2)."""

    name: Final = "m4"
    domains: Final = frozenset({"raw.githubusercontent.com"})
    license: Final = "Dados da competição M4 (Makridakis et al.); ver o repositório M4-methods"
    version: Final = "Mcompetitions/M4-methods@master"
    subsets: Final = tuple(_M4_SUBSETS)

    def __init__(self, pinned: Mapping[str, str] | None = None) -> None:
        self.pinned = dict(_M4_PINNED if pinned is None else pinned)

    def remote(self, filename: str, folder: str = "") -> RemoteFile:
        url = f"{_M4_BASE}/{folder}{filename}"
        return RemoteFile(url, filename, self.pinned.get(filename))

    def _fetch(self, cache: CacheManager, filename: str, folder: str = "") -> Path:
        return cache.fetch(
            self.name,
            self.remote(filename, folder),
            allowed_domains=self.domains,
            license=self.license,
            version=self.version,
        )

    def info(self, cache: CacheManager) -> pd.DataFrame:
        table = pd.read_csv(self._fetch(cache, "M4-info.csv"))
        table = table.rename(
            columns={
                "M4id": "series_id",
                "Frequency": "seasonal_period",
                "Horizon": "horizon",
                "SP": "frequency",
                "StartingDate": "starting_date",
            }
        )
        table["subset"] = table["frequency"].str.lower()
        return table.set_index("series_id")

    def list(self, cache: CacheManager, subset: str | None = None) -> pd.DataFrame:
        info = self.info(cache)
        if subset is not None:
            info = info[info["subset"] == self._subset(subset)]
        return info.reset_index()[
            ["series_id", "subset", "category", "horizon", "seasonal_period", "starting_date"]
        ]

    def load(
        self, cache: CacheManager, subset: str, series_id: str | None = None
    ) -> BenchmarkDataset | BenchmarkSeries:
        subset = self._subset(subset)
        info = self.info(cache)
        name = _M4_SUBSETS[subset]
        train_path = self._fetch(cache, f"{name}-train.csv", "Train/")
        test_path = self._fetch(cache, f"{name}-test.csv", "Test/")
        wanted = None if series_id is None else {series_id}
        train_rows = dict(_read_wide(train_path, wanted))
        test_rows = dict(_read_wide(test_path, wanted))
        if series_id is not None and series_id not in train_rows:
            raise ConfigurationError(f"Série {series_id!r} não está no M4 {subset}.")
        frames_train, frames_test, meta = [], [], []
        for sid, train_values in train_rows.items():
            test_values = test_rows[sid]
            row = info.loc[sid]
            start = _m4_start(str(row["starting_date"]), subset)
            index, source = regular_index(subset, start, len(train_values) + len(test_values))
            frames_train.append(_long(sid, index[: len(train_values)], train_values))
            frames_test.append(_long(sid, index[len(train_values) :], test_values))
            meta.append(
                {
                    "series_id": sid,
                    "category": row["category"],
                    "seasonal_period": int(str(row["seasonal_period"])),
                    "index_source": source,
                }
            )
        descriptor = DatasetDescriptor(
            provider=self.name,
            dataset_id=f"m4/{subset}",
            source_url=_M4_BASE,
            version=self.version,
            license=self.license,
            frequency=subset,
            horizon=int(info[info["subset"] == subset]["horizon"].iloc[0]),
            target_name="value",
            time_col="timestamp",
            series_id_col="series_id",
            split_kind="official",
            extra={
                "seasonal_period": int(info[info["subset"] == subset]["seasonal_period"].iloc[0])
            },
        )
        dataset = BenchmarkDataset(
            pd.concat(frames_train, ignore_index=True),
            pd.concat(frames_test, ignore_index=True),
            pd.DataFrame(meta).set_index("series_id"),
            descriptor,
        )
        return dataset if series_id is None else dataset.series(series_id)

    @staticmethod
    def _subset(subset: str) -> str:
        key = subset.lower()
        if key not in _M4_SUBSETS:
            raise ConfigurationError(
                f"Subconjunto M4 desconhecido {subset!r}; opções: {list(_M4_SUBSETS)}."
            )
        return key


def _read_wide(path: Path, wanted: set[str] | None) -> Iterator[tuple[str, list[float]]]:
    """Rows ``id, v1, v2, ...`` of an M4 file; trailing empty cells are padding."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header V1, V2 ...
        for row in reader:
            if not row:
                continue
            sid = row[0]
            if wanted is not None and sid not in wanted:
                continue
            values = [float(cell) for cell in row[1:] if cell not in ("", "NA")]
            yield sid, values
            if wanted is not None:
                wanted = wanted - {sid}
                if not wanted:
                    return


def _m4_start(text: str, subset: str) -> pd.Timestamp | None:
    try:
        stamp = pd.to_datetime(text, format="%d-%m-%y %H:%M")
    except (ValueError, TypeError):
        return None
    return stamp if subset == "hourly" else stamp.normalize()


# ------------------------------------------------------------------------------------ M3
_MCOMP_URL: Final = "https://cran.r-project.org/src/contrib/Mcomp_2.8.tar.gz"
_MCOMP_SHA: Final = "c0e873054f91b4345f963d5e03df067dfcd512fe55d4be31d0c476d84676e5f3"
_M3_PERIODS: Final = {
    "YEARLY": "yearly",
    "QUARTERLY": "quarterly",
    "MONTHLY": "monthly",
    "OTHER": "other",
}


class M3Provider:
    """M3 competition from the R package ``Mcomp`` (CRAN); parsing needs the extra ``datasets``."""

    name: Final = "m3"
    domains: Final = frozenset({"cran.r-project.org"})
    license: Final = "GPL-3 (pacote R Mcomp); dados da competição M3 (IIF)"
    version: Final = "Mcomp 2.8"
    subsets: Final = ("yearly", "quarterly", "monthly", "other")

    def __init__(self, sha256: str | None = _MCOMP_SHA) -> None:
        self.sha256 = sha256

    def _frames(self, cache: CacheManager) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        folder = cache.path(self.name, "parsed")
        files = [folder / f"{part}.csv" for part in ("train", "test", "info")]
        archive = cache.fetch(
            self.name,
            RemoteFile(_MCOMP_URL, "Mcomp_2.8.tar.gz", self.sha256),
            allowed_domains=self.domains,
            license=self.license,
            version=self.version,
        )
        if not all(path.exists() for path in files):
            rda = safe_extract_member(archive, "Mcomp/data/M3.rda", cache.path(self.name, "raw"))
            train, test, info = m3_frames(_read_rda(rda))
            folder.mkdir(parents=True, exist_ok=True)
            for frame, path in zip((train, test, info), files, strict=True):
                frame.to_csv(path, index=False)
        train, test, info = (pd.read_csv(path) for path in files)
        for frame in (train, test):
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="ISO8601")
            frame["series_id"] = frame["series_id"].astype(str)
        return train, test, info.set_index("series_id")

    def list(self, cache: CacheManager, subset: str | None = None) -> pd.DataFrame:
        _, _, info = self._frames(cache)
        if subset is not None:
            info = info[info["subset"] == self._subset(subset)]
        return info.reset_index()

    def load(
        self, cache: CacheManager, subset: str, series_id: str | None = None
    ) -> BenchmarkDataset | BenchmarkSeries:
        subset = self._subset(subset)
        train, test, info = self._frames(cache)
        ids = set(info.index[info["subset"] == subset])
        if series_id is not None and series_id not in ids:
            raise ConfigurationError(f"Série {series_id!r} não está no M3 {subset}.")
        chosen = {series_id} if series_id is not None else ids
        meta = info.loc[sorted(chosen)]
        descriptor = DatasetDescriptor(
            provider=self.name,
            dataset_id=f"m3/{subset}",
            source_url=_MCOMP_URL,
            version=self.version,
            license=self.license,
            frequency=subset,
            horizon=int(meta["horizon"].iloc[0]),
            target_name="value",
            time_col="timestamp",
            series_id_col="series_id",
            split_kind="official",
            extra={},
        )
        dataset = BenchmarkDataset(
            train[train["series_id"].isin(chosen)].reset_index(drop=True),
            test[test["series_id"].isin(chosen)].reset_index(drop=True),
            meta,
            descriptor,
        )
        return dataset if series_id is None else dataset.series(series_id)

    def _subset(self, subset: str) -> str:
        key = subset.lower()
        if key not in self.subsets:
            raise ConfigurationError(
                f"Subconjunto M3 desconhecido {subset!r}; opções: {self.subsets}."
            )
        return key


def _read_rda(path: Path) -> Mapping[str, Mapping[str, Any]]:
    try:
        import rdata
    except ModuleNotFoundError as error:
        raise UnsupportedCapabilityError(
            'Ler o M3 (formato R) requer o extra `datasets`: pip install "autohybts[datasets]".'
        ) from error
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "missing constructor for R class Mdata/Mcomp"
        converted = rdata.conversion.convert(rdata.parser.parse_file(path), default_encoding="utf8")
    records: Mapping[str, Mapping[str, Any]] = converted["M3"]
    return records


def m3_frames(
    records: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Long train/test frames and per-series info from ``Mcomp``'s M3 list.

    Each record holds ``sn``, ``period``, ``type``, ``h``, ``x`` (train) and ``xx``
    (official test); ``x`` is indexed by (fractional) years.
    """

    trains, tests, info = [], [], []
    for record in records.values():
        sid = str(_scalar(record["sn"]))
        period = _M3_PERIODS[str(_scalar(record["period"]))]
        x, xx = record["x"], record["xx"]
        start = None if period == "other" else _decimal_year(x.index[0], period)
        kind = "daily" if period == "other" else period
        index, source = regular_index(kind, start, len(x) + len(xx))
        trains.append(_long(sid, index[: len(x)], x.to_numpy(dtype=float)))
        tests.append(_long(sid, index[len(x) :], xx.to_numpy(dtype=float)))
        info.append(
            {
                "series_id": sid,
                "subset": period,
                "category": str(_scalar(record["type"])),
                "horizon": int(_scalar(record["h"])),
                "n_train": len(x),
                "description": str(_scalar(record.get("description", ""))),
                "index_source": source,
            }
        )
    return (
        pd.concat(trains, ignore_index=True),
        pd.concat(tests, ignore_index=True),
        pd.DataFrame(info),
    )


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray | list | tuple):
        return value[0]
    return value


def _decimal_year(value: Any, period: str) -> pd.Timestamp:
    """``1990 + 3/12`` → April 1990; exact for fractions, rounded for floats."""

    point = (
        Fraction(value) if not isinstance(value, float) else Fraction(value).limit_denominator(12)
    )
    year = math.floor(point)
    steps = {"yearly": 1, "quarterly": 4, "monthly": 12}[period]
    offset = round((point - year) * steps)
    month = 1 + offset * (12 // steps)
    return pd.Timestamp(year=int(year), month=int(month), day=1)


# ----------------------------------------------------------------------------- Rdatasets
_RD_HOST: Final = "vincentarelbundock.github.io"
_RD_INDEX: Final = f"https://{_RD_HOST}/Rdatasets/datasets.csv"
_RD_KINDS: Final = {"YS": "yearly", "QS": "quarterly", "MS": "monthly", "W": "weekly", "D": "daily"}


class RdatasetsProvider:
    """Rdatasets catalogue (D3): heterogeneous tables, so time and target are explicit (D06)."""

    name: Final = "rdatasets"
    domains: Final = frozenset({_RD_HOST})
    license: Final = "Varia por pacote R de origem; consulte a documentação de cada dataset"
    version: Final = "Rdatasets (índice atual)"

    def index(self, cache: CacheManager, refresh: bool = False) -> pd.DataFrame:
        path = cache.fetch(
            self.name,
            RemoteFile(_RD_INDEX, "datasets.csv"),
            allowed_domains=self.domains,
            license=self.license,
            version=self.version,
            refresh=refresh,
        )
        return pd.read_csv(path)

    def list(self, cache: CacheManager, search: str | None = None) -> pd.DataFrame:
        table = self.index(cache)
        if search:
            text = table[["Package", "Item", "Title"]].astype(str).agg(" ".join, axis=1)
            table = table[text.str.contains(search, case=False, regex=False)]
        return table.reset_index(drop=True)

    def raw(self, cache: CacheManager, package: str, name: str) -> tuple[pd.DataFrame, str]:
        """The CSV as published (no interpretation) and its URL."""

        index = self.index(cache)
        rows = index[(index["Package"] == package) & (index["Item"] == name)]
        if rows.empty:
            raise ConfigurationError(f"{package}/{name} não está no índice do Rdatasets.")
        url = str(rows["CSV"].iloc[0])
        path = cache.fetch(
            self.name,
            RemoteFile(url, f"{package}__{name}.csv"),
            allowed_domains=self.domains,
            license=self.license,
            version=self.version,
        )
        return pd.read_csv(path), url

    def load(
        self,
        cache: CacheManager,
        *,
        package: str,
        name: str,
        target: str | None = None,
        time_col: str | None = None,
        frequency: str | None = None,
    ) -> BenchmarkSeries:
        """One series; ``target``, ``time_col`` and ``frequency`` must be given (D06)."""

        if target is None or time_col is None or frequency is None:
            raise ConfigurationError(
                "Rdatasets é um catálogo heterogêneo: informe target, time_col e frequency "
                "(ex.: target='value', time_col='time', frequency='MS'). Use "
                "raw_dataset('rdatasets', ...) para ver as colunas."
            )
        table, url = self.raw(cache, package, name)
        missing = [c for c in (target, time_col) if c not in table.columns]
        if missing:
            raise ConfigurationError(
                f"Coluna(s) {missing} não existem em {package}/{name}; disponíveis: "
                f"{list(table.columns)}."
            )
        index = _rd_index(table[time_col], frequency)
        series = pd.Series(table[target].to_numpy(dtype=float), index=index, name=name)
        if not series.index.is_monotonic_increasing or series.index.has_duplicates:
            raise DataValidationError(
                f"{time_col} de {package}/{name} não está em ordem estrita; ordene antes ou "
                "escolha outra coluna de tempo."
            )
        descriptor = DatasetDescriptor(
            provider=self.name,
            dataset_id=f"{package}/{name}",
            source_url=url,
            version=self.version,
            license=self.license,
            frequency=frequency,
            horizon=None,
            target_name=target,
            time_col=time_col,
            series_id_col=None,
            split_kind="none",
            extra={"package": package},
        )
        return BenchmarkSeries(series, None, descriptor)


def _rd_index(column: pd.Series, frequency: str) -> pd.DatetimeIndex:
    """Timestamps from a time column: decimal years (R ``ts``) or parseable dates."""

    if pd.api.types.is_numeric_dtype(column):
        kind = _RD_KINDS.get(frequency)
        if kind not in ("yearly", "quarterly", "monthly"):
            raise ConfigurationError(
                "Tempo numérico (anos decimais) só é convertido para frequency 'YS', 'QS' ou "
                f"'MS'; recebido {frequency!r}."
            )
        start = _decimal_year(float(column.iloc[0]), kind)
        index, _ = regular_index(kind, start, len(column))
        expected = np.array([float(_year_fraction(stamp, kind)) for stamp in index], dtype=float)
        if not np.allclose(expected, column.to_numpy(dtype=float), atol=1e-6):
            raise DataValidationError(
                "A coluna de tempo não é uma sequência regular de anos decimais na frequência "
                f"{frequency!r}."
            )
        return index
    try:
        return pd.DatetimeIndex(pd.to_datetime(column))
    except (ValueError, TypeError) as error:
        raise DataValidationError(
            f"Coluna de tempo não interpretável como data: {error}"
        ) from error


def _year_fraction(stamp: pd.Timestamp, kind: str) -> Fraction:
    if kind == "yearly":
        return Fraction(stamp.year)
    return Fraction(stamp.year) + Fraction(stamp.month - 1, 12)


PROVIDERS: Final[dict[str, M4Provider | M3Provider | RdatasetsProvider]] = {
    "m4": M4Provider(),
    "m3": M3Provider(),
    "rdatasets": RdatasetsProvider(),
}
