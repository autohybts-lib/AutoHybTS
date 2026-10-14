"""Benchmark series with exogenous covariates (ADR-0011), downloaded on request from Rdatasets.

``load_exog_dataset(name)`` returns ``(y, X)``: the target as a ``pd.Series`` and the
covariates as a ``DataFrame`` on the same dates, ready for ``fit(y, X)``. ``future`` lists
the columns known in advance (calendar, legislation, planned spending); pass it as
``exog_future``. Every other column is ``past``. Squared temperatures (``temperatura2``,
``temperatura_max2``) are derived columns: electricity demand rises both in cold and in heat.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

import pandas as pd

from autohybts.exceptions import ConfigurationError

__all__ = ["EXOG_DATASETS", "ExogDataset", "list_exog_datasets", "load_exog_dataset"]


@dataclass(frozen=True)
class ExogDataset:
    """Where a series with covariates lives and how to read it."""

    package: str
    name: str
    target: str
    covariates: dict[str, str]  # source column -> name in X
    start: str
    frequency: str
    future: tuple[str, ...]
    description: str
    reader: Callable[..., tuple[pd.Series, pd.DataFrame]] | None = field(default=None, repr=False)
    squared: tuple[str, ...] = ()


def _table(spec: ExogDataset, **kwargs: Any) -> tuple[pd.Series, pd.DataFrame]:
    from autohybts.datasets.catalog import raw_dataset

    raw = raw_dataset("rdatasets", package=spec.package, name=spec.name, **kwargs)
    index = pd.date_range(spec.start, periods=len(raw), freq=spec.frequency)
    y = pd.Series(raw[spec.target].to_numpy(dtype=float), index=index, name=spec.target)
    X = pd.DataFrame(
        {new: raw[old].to_numpy(dtype=float) for old, new in spec.covariates.items()}, index=index
    )
    for column in spec.squared:
        # Demand rises with cold and with heat: the square lets linear models see the U shape.
        X[f"{column}2"] = X[column] ** 2
    return y, X


def _bjsales(spec: ExogDataset, **kwargs: Any) -> tuple[pd.Series, pd.DataFrame]:
    from autohybts.datasets.catalog import raw_dataset

    sales = raw_dataset("rdatasets", package="datasets", name="BJsales", **kwargs)
    lead = raw_dataset("rdatasets", package="datasets", name="BJsales.lead", **kwargs)
    index = pd.date_range(spec.start, periods=len(sales), freq=spec.frequency)
    y = pd.Series(sales["value"].to_numpy(dtype=float), index=index, name="sales")
    return y, pd.DataFrame({"indicador": lead["value"].to_numpy(dtype=float)}, index=index)


def _vic_elec(spec: ExogDataset, **kwargs: Any) -> tuple[pd.Series, pd.DataFrame]:
    from autohybts.datasets.catalog import raw_dataset

    raw = raw_dataset("rdatasets", package="tsibbledata", name="vic_elec", **kwargs)
    daily = raw.groupby("Date").agg(
        demand=("Demand", "sum"), temperatura_max=("Temperature", "max"), feriado=("Holiday", "max")
    )
    index = pd.DatetimeIndex(pd.to_datetime(daily.index), freq="D")
    # Half-hourly MW summed over the day, times 0.5 h, in GWh.
    y = pd.Series(
        daily["demand"].to_numpy(dtype=float) * 0.5 / 1000, index=index, name="demanda_gwh"
    )
    X = pd.DataFrame(
        {
            "temperatura_max": daily["temperatura_max"].to_numpy(dtype=float),
            "temperatura_max2": daily["temperatura_max"].to_numpy(dtype=float) ** 2,
            "feriado": daily["feriado"].to_numpy(dtype=float),
            "fim_de_semana": (index.dayofweek >= 5).astype(float),
        },
        index=index,
    )
    return y, X


EXOG_DATASETS: Final[dict[str, ExogDataset]] = {
    "seatbelts": ExogDataset(
        "datasets",
        "Seatbelts",
        "drivers",
        {"kms": "km_rodados", "PetrolPrice": "preco_gasolina", "law": "lei_cinto"},
        "1969-01-01",
        "MS",
        ("lei_cinto",),
        "motoristas mortos ou feridos no Reino Unido, 1969–1984 (lei do cinto em 1983)",
    ),
    "bjsales": ExogDataset(
        "datasets",
        "BJsales",
        "value",
        {},
        "2000-01-01",
        "D",
        (),
        "vendas com indicador antecedente (Box–Jenkins); índice diário sintético",
        _bjsales,
    ),
    "uschange": ExogDataset(
        "fpp2",
        "uschange",
        "Consumption",
        {
            "Income": "renda",
            "Production": "producao",
            "Savings": "poupanca",
            "Unemployment": "desemprego",
        },
        "1970-01-01",
        "QS",
        (),
        "variação do consumo nos EUA ~ renda, produção, poupança e desemprego, 1970–2016",
    ),
    "insurance": ExogDataset(
        "fpp2",
        "insurance",
        "Quotes",
        {"TV.advert": "publicidade_tv"},
        "2002-01-01",
        "MS",
        ("publicidade_tv",),
        "cotações de seguro ~ gasto planejado com publicidade na TV, 2002–2005",
    ),
    "elecdaily": ExogDataset(
        "fpp2",
        "elecdaily",
        "Demand",
        {"Temperature": "temperatura", "WorkDay": "dia_util"},
        "2014-01-01",
        "D",
        ("dia_util",),
        "demanda elétrica diária de Victoria (Austrália) ~ temperatura e dia útil, 2014",
        squared=("temperatura",),
    ),
    "vic_elec_daily": ExogDataset(
        "tsibbledata",
        "vic_elec",
        "Demand",
        {},
        "2012-01-01",
        "D",
        ("feriado", "fim_de_semana"),
        "demanda elétrica diária de Victoria em GWh ~ temperatura máxima, feriado e fim de "
        "semana, 2012–2014",
        _vic_elec,
    ),
}


def list_exog_datasets() -> pd.DataFrame:
    """The series with covariates: target, columns, frequency and which columns are ``future``."""

    return pd.DataFrame(
        [
            {
                "name": name,
                "description": spec.description,
                "frequency": spec.frequency,
                "future": list(spec.future),
                "source": f"{spec.package}/{spec.name}",
            }
            for name, spec in EXOG_DATASETS.items()
        ]
    ).set_index("name")


def load_exog_dataset(name: str, **kwargs: Any) -> tuple[pd.Series, pd.DataFrame]:
    """``(y, X)`` of a series with covariates; downloaded once to the cache.

    Keywords go to the download (``offline``, ``cache_dir``, ``cache``). Use the dataset's
    ``future`` columns (``list_exog_datasets()``) as ``exog_future``.
    """

    spec = EXOG_DATASETS.get(name)
    if spec is None:
        raise ConfigurationError(
            f"Conjunto com exógenas {name!r} não existe; opções: {sorted(EXOG_DATASETS)}."
        )
    reader = spec.reader or _table
    return reader(spec, **kwargs)
