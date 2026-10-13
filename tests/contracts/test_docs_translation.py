"""The English documentation mirrors the Portuguese one page by page and notebook by notebook.

``documentation/`` (pt-BR) and ``documentation/en/`` are separate Sphinx projects published
as translations on Read the Docs; ``notebooks/en/`` holds the English copies of the
tutorials. A page or notebook added to one language without the other would leave a hole
in the language switcher, so the file sets and the notebook metadata must match.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "documentation"
NOTEBOOKS = ROOT / "notebooks"
# Build output and files copied or generated at build time.
SKIP = ("_build", "_generated", "_tools", "tutorials/notebooks")


def pages(tree: Path, exclude: tuple[str, ...] = ()) -> set[str]:
    names = set()
    for path in tree.rglob("*.rst"):
        name = path.relative_to(tree).as_posix()
        if not any(part in name for part in SKIP) and not name.startswith(exclude):
            names.add(name)
    return names


def test_both_trees_have_the_same_pages() -> None:
    portuguese = pages(DOCS, exclude=("en/",))
    english = pages(DOCS / "en")
    assert portuguese, "documentação em português vazia"
    assert portuguese - english == set(), "páginas sem versão em inglês"
    assert english - portuguese == set(), "páginas só em inglês"


def test_both_trees_include_the_generated_tables() -> None:
    for tree in (DOCS, DOCS / "en"):
        guide = tree / "user_guide"
        for page, include in (
            ("model_reference.rst", "_generated/base_models.rst"),
            ("model_reference.rst", "_generated/correctors.rst"),
            ("search_space.rst", "_generated/budgets.rst"),
            ("search_space.rst", "_generated/search_defaults.rst"),
        ):
            assert f".. include:: {include}" in (guide / page).read_text(encoding="utf-8")


def test_every_tutorial_has_an_english_copy() -> None:
    portuguese = {p.name for p in NOTEBOOKS.glob("*.ipynb")}
    english = {p.name for p in (NOTEBOOKS / "en").glob("*.ipynb")}
    assert portuguese == english


@pytest.mark.parametrize("name", sorted(p.name for p in NOTEBOOKS.glob("*.ipynb")))
def test_english_copy_keeps_the_metadata(name: str) -> None:
    def meta(path: Path) -> object:
        return json.loads(path.read_text(encoding="utf-8"))["metadata"]["autohybts"]

    english = NOTEBOOKS / "en" / name
    if not english.is_file():
        pytest.fail(f"sem versão em inglês: {name}")
    assert meta(english) == meta(NOTEBOOKS / name)


def test_english_tables_are_in_english() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "model_tables", DOCS / "_tools" / "model_tables.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.TEXT["pt"]) == set(module.TEXT["en"])
    assert "Default parameters" in module.base_tables("en")
    assert "Parâmetros padrão" in module.base_tables("pt")
