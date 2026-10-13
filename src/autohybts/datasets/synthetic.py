"""Offline synthetic series with known structure, for tutorials and tests (oracles of §19.2)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

import numpy as np
import pandas as pd

from autohybts.datasets.named import NAMED_SERIES
from autohybts.exceptions import ConfigurationError

__all__ = ["SYNTHETIC", "get_data"]


def _synthetic_daily(n: int = 730, seed: int = 42, phi: float = 0.7) -> pd.Series:
    """Linear trend + weekly seasonality + AR(1) noise: residual structure a corrector can use."""

    rng = np.random.default_rng(seed)
    noise = np.zeros(n)
    shocks = rng.normal(0.0, 1.0, n)
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + shocks[i]
    t = np.arange(n)
    values = 50 + 0.03 * t + 3 * np.sin(2 * np.pi * t / 7) + noise
    index = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(values, index=index, name="value")


def _white_noise_daily(n: int = 730, seed: int = 42) -> pd.Series:
    """Level plus white noise: nothing beyond the mean is predictable."""

    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(50 + rng.normal(0.0, 1.0, n), index=index, name="value")


def _intermittent_daily(
    n: int = 730, seed: int = 42, probability: float = 0.25, mean_size: float = 4.0
) -> pd.Series:
    """Sparse demand: a sale happens with ``probability``; sizes are 1 + Poisson."""

    rng = np.random.default_rng(seed)
    occurs = rng.random(n) < probability
    sizes = 1 + rng.poisson(mean_size - 1, n)
    index = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(np.where(occurs, sizes, 0).astype(float), index=index, name="value")


SYNTHETIC: Final[dict[str, Callable[..., pd.Series]]] = {
    "synthetic_daily": _synthetic_daily,
    "white_noise_daily": _white_noise_daily,
    "intermittent_daily": _intermittent_daily,
}


def get_data(name: str, **kwargs: Any) -> pd.Series:
    """A named series as ``pd.Series`` with a regular ``DatetimeIndex``.

    * Offline examples: ``"synthetic_daily"``, ``"white_noise_daily"``,
      ``"intermittent_daily"``; keyword arguments go to the generator (``n``, ``seed`` ...).
    * Benchmark series: ``"m4:D1"`` or ``"m3:N1402"`` — downloaded once to the cache
      (keywords ``offline``, ``cache_dir``). The official test is at the end; seal it
      with ``test_size`` equal to the benchmark horizon.
    * Well-known Rdatasets series by short name (``"ozone"``, ``"air_passengers"`` ...,
      see ``NAMED_SERIES``; case is ignored), downloaded the same way.
    """

    if name in SYNTHETIC:
        return SYNTHETIC[name](**kwargs)
    named = NAMED_SERIES.get(name.lower())
    if named is not None:
        from autohybts.datasets.catalog import load_dataset

        loaded = load_dataset(
            "rdatasets",
            package=named.package,
            name=named.name,
            target=named.target,
            time_col=named.time_col,
            frequency=named.frequency,
            **kwargs,
        )
        return loaded.full()  # type: ignore[union-attr]
    provider, _, series_id = name.partition(":")
    if provider in ("m4", "m3") and series_id:
        from autohybts.datasets.catalog import list_datasets, load_benchmark

        if provider == "m4":
            subset = _M4_PREFIX.get(series_id[:1].upper())
        else:
            listing = list_datasets("m3", **kwargs)
            match = listing[listing["series_id"] == series_id]
            subset = None if match.empty else str(match["subset"].iloc[0])
        if subset is None:
            raise ConfigurationError(f"Série {series_id!r} não encontrada no {provider.upper()}.")
        loaded = load_benchmark(provider, subset, series_id=series_id, **kwargs)
        return loaded.full()  # type: ignore[union-attr]
    raise ConfigurationError(
        f"Dataset {name!r} não disponível; offline: {sorted(SYNTHETIC)}; nomes prontos: "
        f"{sorted(NAMED_SERIES)}; benchmarks: 'm4:<id>' (ex.: 'm4:D1') ou 'm3:<id>' "
        "(ex.: 'm3:N1402'). Outras séries do Rdatasets: load_dataset('rdatasets', ...)."
    )


_M4_PREFIX: Final = {
    "Y": "yearly",
    "Q": "quarterly",
    "M": "monthly",
    "W": "weekly",
    "D": "daily",
    "H": "hourly",
}
