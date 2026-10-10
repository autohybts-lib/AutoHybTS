"""UX02: a user CSV is read with explicit column mapping and validated."""

from __future__ import annotations

from pathlib import Path

import pytest

from autohybts.data.ingest import read_csv
from autohybts.exceptions import ConfigurationError, DataValidationError


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_csv_with_explicit_columns(tmp_path: Path) -> None:
    path = write(tmp_path / "s.csv", "dia;vendas\n2024-01-01;10\n2024-01-02;12\n2024-01-03;11\n")
    frame = read_csv(path, target="vendas", date_col="dia", sep=";")
    assert frame.frequency == "D"
    assert frame.values.tolist() == [10.0, 12.0, 11.0]


def test_csv_with_unknown_column(tmp_path: Path) -> None:
    path = write(tmp_path / "s.csv", "date,value\n2024-01-01,1\n2024-01-02,2\n2024-01-03,3\n")
    with pytest.raises(ConfigurationError, match="'sales'"):
        read_csv(path, target="sales", date_col="date")


def test_csv_with_bad_dates(tmp_path: Path) -> None:
    path = write(tmp_path / "s.csv", "date,value\n2024-01-01,1\nxx,2\n2024-01-03,3\n")
    with pytest.raises(DataValidationError, match="não são datas"):
        read_csv(path, target="value", date_col="date")


def test_csv_with_unsorted_rows(tmp_path: Path) -> None:
    path = write(tmp_path / "s.csv", "date,value\n2024-01-02,1\n2024-01-01,2\n2024-01-03,3\n")
    with pytest.raises(DataValidationError, match="fora de ordem"):
        read_csv(path, target="value", date_col="date", frequency="D")
