"""Every constructor parameter of the public estimators is documented in its docstring."""

from __future__ import annotations

import inspect
import re

import pytest

from autohybts import AutoHybridExperiment, AutoHybridForecaster

# Google style after ``inspect.getdoc``: ``    name: text`` or ``    a, b, c: text``.
ENTRY = re.compile(r"^ {4}(\*{0,2}[A-Za-z_][\w, *]*):", re.MULTILINE)


def documented(cls: type) -> set[str]:
    doc = inspect.getdoc(cls) or ""
    args = doc.split("Args:", 1)[1] if "Args:" in doc else ""
    names: set[str] = set()
    for match in ENTRY.finditer(args):
        names.update(part.strip().lstrip("*") for part in match.group(1).split(","))
    return names


@pytest.mark.parametrize("cls", [AutoHybridExperiment, AutoHybridForecaster])
def test_every_constructor_parameter_is_documented(cls: type) -> None:
    parameters = [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
    missing = [p for p in parameters if p not in documented(cls)]
    assert not missing, f"{cls.__name__}: parâmetros sem documentação em Args: {missing}"
