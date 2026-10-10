"""Public catalogue API: ``list_datasets``, ``describe_dataset``, ``load_dataset``,
``load_benchmark`` and ``raw_dataset`` (§32.2).

Every call that needs remote data goes through a ``CacheManager``: pass
``offline=True`` to use only what is already cached, or ``cache_dir`` to choose where
files live (default ``AUTOHYBTS_DATA_DIR`` or ``~/.cache/autohybts/datasets``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from autohybts.datasets.cache import CacheManager
from autohybts.datasets.descriptor import BenchmarkDataset, BenchmarkSeries, DatasetDescriptor
from autohybts.datasets.providers import PROVIDERS, M3Provider, M4Provider, RdatasetsProvider
from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError

__all__ = [
    "describe_dataset",
    "list_datasets",
    "load_benchmark",
    "load_dataset",
    "raw_dataset",
]

_BENCHMARKS = ("m4", "m3")


def _cache(cache: CacheManager | None, cache_dir: str | Path | None, offline: bool) -> CacheManager:
    if cache is not None:
        return cache
    return CacheManager(cache_dir, offline=offline)


def list_datasets(
    provider: str | None = None,
    *,
    subset: str | None = None,
    search: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
    cache: CacheManager | None = None,
) -> pd.DataFrame:
    """Catalogue of a provider (``m4``, ``m3``, ``rdatasets``), or of providers if ``None``.

    Without a provider nothing is downloaded. ``subset`` filters M3/M4 by frequency;
    ``search`` filters Rdatasets by package, name and title.
    """

    if provider is None:
        return pd.DataFrame(
            [
                {"provider": "synthetic", "kind": "séries sintéticas offline", "license": "MIT"},
                *(
                    {"provider": name, "kind": _kind(item), "license": item.license}
                    for name, item in PROVIDERS.items()
                ),
            ]
        )
    item = _provider(provider)
    manager = _cache(cache, cache_dir, offline)
    if isinstance(item, RdatasetsProvider):
        if subset is not None:
            raise ConfigurationError("Rdatasets não tem subconjuntos; use search=.")
        return item.list(manager, search)
    if search is not None:
        raise ConfigurationError(f"search= vale para rdatasets; para {provider} use subset=.")
    return item.list(manager, subset)


def describe_dataset(
    provider: str,
    subset: str | None = None,
    *,
    offline: bool = False,
    cache_dir: str | Path | None = None,
    cache: CacheManager | None = None,
) -> DatasetDescriptor:
    """Descriptor of a benchmark subset (downloads only the catalogue files)."""

    item = _provider(provider)
    if not isinstance(item, M4Provider | M3Provider):
        raise ConfigurationError(
            "describe_dataset vale para m4 e m3; no Rdatasets use list_datasets."
        )
    if subset is None:
        raise ConfigurationError(f"Informe subset: {list(item.subsets)}.")
    table = item.list(_cache(cache, cache_dir, offline), subset)
    return DatasetDescriptor(
        provider=item.name,
        dataset_id=f"{item.name}/{subset.lower()}",
        source_url=f"https://{sorted(item.domains)[0]}",
        version=item.version,
        license=item.license,
        frequency=subset.lower(),
        horizon=int(table["horizon"].iloc[0]),
        target_name="value",
        time_col="timestamp",
        series_id_col="series_id",
        split_kind="official",
        extra={"n_series": len(table)},
    )


def load_benchmark(
    name: str,
    subset: str,
    *,
    split: str = "official",
    series_id: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
    cache: CacheManager | None = None,
) -> BenchmarkDataset | BenchmarkSeries:
    """M4 or M3 with the official split; one ``BenchmarkSeries`` when ``series_id`` is given.

    The official test stays apart (``.test``); seal it with
    ``AutoHybridExperiment(test_size=len(series.test)).fit(series.full())`` so it never
    reaches the search or the selection.
    """

    if name not in _BENCHMARKS:
        raise ConfigurationError(f"Benchmark desconhecido {name!r}; opções: {_BENCHMARKS}.")
    if split != "official":
        raise UnsupportedCapabilityError("Só o split oficial dos benchmarks é oferecido.")
    item = PROVIDERS[name]
    assert isinstance(item, M4Provider | M3Provider)
    return item.load(_cache(cache, cache_dir, offline), subset, series_id)


def load_dataset(
    provider: str,
    *,
    package: str | None = None,
    name: str | None = None,
    target: str | None = None,
    time_col: str | None = None,
    frequency: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
    cache: CacheManager | None = None,
    **options: Any,
) -> BenchmarkDataset | BenchmarkSeries:
    """One dataset of any provider; Rdatasets needs ``target``, ``time_col`` and ``frequency``."""

    item = _provider(provider)
    manager = _cache(cache, cache_dir, offline)
    if isinstance(item, RdatasetsProvider):
        if package is None or name is None:
            raise ConfigurationError("Informe package e name (ex.: 'datasets', 'AirPassengers').")
        return item.load(
            manager,
            package=package,
            name=name,
            target=target,
            time_col=time_col,
            frequency=frequency,
        )
    return item.load(manager, options.get("subset", ""), options.get("series_id"))


def raw_dataset(
    provider: str,
    *,
    package: str,
    name: str,
    offline: bool = False,
    cache_dir: str | Path | None = None,
    cache: CacheManager | None = None,
) -> pd.DataFrame:
    """A Rdatasets table exactly as published, to choose its columns."""

    item = _provider(provider)
    if not isinstance(item, RdatasetsProvider):
        raise ConfigurationError("raw_dataset vale para rdatasets.")
    table, _ = item.raw(_cache(cache, cache_dir, offline), package, name)
    return table


def _provider(name: str) -> M4Provider | M3Provider | RdatasetsProvider:
    try:
        return PROVIDERS[name.lower()]
    except KeyError:
        raise ConfigurationError(
            f"Provedor desconhecido {name!r}; opções: {sorted(PROVIDERS)}."
        ) from None


def _kind(item: object) -> str:
    if isinstance(item, M4Provider):
        return "benchmark M4 (treino/teste oficiais)"
    if isinstance(item, M3Provider):
        return "benchmark M3 (treino/teste oficiais)"
    return "catálogo Rdatasets (colunas e tempo explícitos)"
