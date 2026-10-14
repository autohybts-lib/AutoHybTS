"""Catalogue of series with covariates (no download: Rdatasets tables are stubbed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autohybts.datasets import catalog
from autohybts.datasets.exogenous import EXOG_DATASETS, list_exog_datasets, load_exog_dataset
from autohybts.exceptions import ConfigurationError


def fake_tables(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    rng = np.random.default_rng(0)

    def raw(provider: str, *, package: str, name: str, **_: object) -> pd.DataFrame:
        calls.append((package, name))
        if name == "vic_elec":
            stamps = pd.date_range("2012-01-01", periods=96, freq="30min")
            return pd.DataFrame(
                {
                    "Demand": np.full(96, 4000.0),
                    "Temperature": np.linspace(10, 30, 96),
                    "Date": stamps.strftime("%Y-%m-%d"),
                    "Holiday": [True] * 48 + [False] * 48,
                }
            )
        spec = next((s for s in EXOG_DATASETS.values() if s.name == name), None)
        columns = ["value"] if spec is None else [spec.target, *spec.covariates]
        return pd.DataFrame({c: rng.normal(size=12) for c in columns})

    monkeypatch.setattr(catalog, "raw_dataset", raw)
    return calls


def test_every_dataset_returns_aligned_series_and_covariates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tables(monkeypatch)
    for name, spec in EXOG_DATASETS.items():
        y, X = load_exog_dataset(name)
        assert isinstance(y.index, pd.DatetimeIndex)
        assert X.index.equals(y.index)
        assert set(spec.future) <= set(X.columns)
        assert not X.isna().any().any()


def test_vic_elec_is_aggregated_by_day(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_tables(monkeypatch)
    y, X = load_exog_dataset("vic_elec_daily")
    assert len(y) == 2
    assert y.iloc[0] == pytest.approx(48 * 4000 * 0.5 / 1000)
    assert X["feriado"].tolist() == [1.0, 0.0]
    assert X["temperatura_max"].iloc[1] == pytest.approx(30.0)


def test_bjsales_joins_the_leading_indicator(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_tables(monkeypatch)
    y, X = load_exog_dataset("bjsales")
    assert y.name == "sales"
    assert ("datasets", "BJsales.lead") in calls
    assert list(X.columns) == ["indicador"]


def test_listing_and_unknown_names() -> None:
    listing = list_exog_datasets()
    assert set(listing.index) == set(EXOG_DATASETS)
    assert listing.loc["elecdaily", "future"] == ["dia_util"]
    with pytest.raises(ConfigurationError, match="não existe"):
        load_exog_dataset("airline")
