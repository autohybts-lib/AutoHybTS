"""Short names for well-known Rdatasets series, so ``get_data("ozone")`` works directly.

Each entry fixes what Rdatasets cannot infer (target, time column, frequency); the file is
still downloaded on request through the cache, never on import.
"""

from __future__ import annotations

from typing import Final, NamedTuple

__all__ = ["NAMED_SERIES", "NamedSeries"]


class NamedSeries(NamedTuple):
    package: str
    name: str
    target: str
    time_col: str
    frequency: str
    description: str


NAMED_SERIES: Final[dict[str, NamedSeries]] = {
    "air_passengers": NamedSeries(
        "datasets", "AirPassengers", "value", "time", "MS", "passageiros aéreos, 1949–1960"
    ),
    "johnson_johnson": NamedSeries(
        "datasets", "JohnsonJohnson", "value", "time", "QS", "lucro por ação J&J, 1960–1980"
    ),
    "lake_huron": NamedSeries(
        "datasets", "LakeHuron", "value", "time", "YS", "nível do lago Huron, 1875–1972"
    ),
    "lynx": NamedSeries("datasets", "lynx", "value", "time", "YS", "linces capturados, 1821–1934"),
    "nile": NamedSeries("datasets", "Nile", "value", "time", "YS", "vazão do rio Nilo, 1871–1970"),
    "nottem": NamedSeries(
        "datasets", "nottem", "value", "time", "MS", "temperatura em Nottingham, 1920–1939"
    ),
    "ozone": NamedSeries(
        "DAAG", "ozone", "Annual", "Year", "YS", "ozônio total em Halley Bay, 1956–2000"
    ),
    "sunspots": NamedSeries(
        "datasets", "sunspot.year", "value", "time", "YS", "manchas solares, 1700–1988"
    ),
    "sunspots_monthly": NamedSeries(
        "datasets", "sunspots", "value", "time", "MS", "manchas solares, 1749–1983"
    ),
    "uk_gas": NamedSeries(
        "datasets", "UKgas", "value", "time", "QS", "consumo de gás no Reino Unido, 1960–1986"
    ),
    "us_acc_deaths": NamedSeries(
        "datasets", "USAccDeaths", "value", "time", "MS", "mortes acidentais EUA, 1973–1978"
    ),
}
