"""Notebook metadata follow one schema (plano de ensembles §10.3, documented in contributing.rst).

``metadata.autohybts`` holds what a reader and the test harness need: the public names
the notebook calls (``requires``), whether it downloads (``network``) or runs for long
(``heavy``), the optional extras it needs (``extras``), the seeds that make it
reproducible, the guide pages it illustrates and whether it follows the standard lesson
model (``template``: opens with a title and ends with a "Leitura" section, "Reading" in
the English copies under ``notebooks/en/``, whose guide pages live in
``documentation/en/``).
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb")) + sorted(
    (ROOT / "notebooks").glob("en/*.ipynb")
)
IDS = [p.relative_to(ROOT / "notebooks").with_suffix("").as_posix() for p in NOTEBOOKS]
FIELDS = ("requires", "network", "heavy", "extras", "seeds", "guide", "template")
EXTRAS = set(
    tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
)


def notebook(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def english(path: Path) -> bool:
    return path.parent.name == "en"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=IDS)
def test_metadata_schema(path: Path) -> None:
    meta = notebook(path)["metadata"]["autohybts"]  # type: ignore[index]
    assert set(meta) == set(FIELDS), f"campos fora do esquema: {sorted(set(meta) ^ set(FIELDS))}"
    assert meta["requires"]
    assert all(isinstance(name, str) and name.startswith("autohybts.") for name in meta["requires"])
    for flag in ("network", "heavy", "template"):
        assert isinstance(meta[flag], bool), flag
    assert set(meta["extras"]) <= EXTRAS, f"extras inexistentes: {set(meta['extras']) - EXTRAS}"
    assert all(isinstance(seed, int) and not isinstance(seed, bool) for seed in meta["seeds"])
    assert meta["guide"]
    docs = ROOT / "documentation" / ("en" if english(path) else "")
    for page in meta["guide"]:
        assert (docs / f"{page}.rst").is_file(), f"página inexistente: {page}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=IDS)
def test_template_notebooks_open_with_a_title_and_end_with_a_reading(path: Path) -> None:
    content = notebook(path)
    if not content["metadata"]["autohybts"]["template"]:  # type: ignore[index]
        pytest.skip("não declara o modelo de aula")
    cells = content["cells"]
    assert isinstance(cells, list)
    first = "".join(cells[0]["source"])
    assert cells[0]["cell_type"] == "markdown", "abre sem título"
    assert first.startswith("# "), "abre sem título"
    headings = [
        line
        for cell in cells
        if cell["cell_type"] == "markdown"
        for line in "".join(cell["source"]).splitlines()
        if line.startswith("## ")
    ]
    assert headings, "sem seções"
    reading = "Reading" if english(path) else "Leitura"
    assert reading in headings[-1], f"última seção não é a leitura: {headings[-1]!r}"


def test_new_notebook_skeleton_follows_the_schema(tmp_path: Path) -> None:
    import importlib.util

    nbformat = pytest.importorskip("nbformat")
    spec = importlib.util.spec_from_file_location(
        "new_notebook", ROOT / "notebooks" / "new_notebook.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for lang, folder in (("pt", tmp_path), ("en", tmp_path / "en")):
        skeleton = module.skeleton(
            "Teste",
            ["autohybts.AutoHybridExperiment.compare_models"],
            ["user_guide/ensembles"],
            ["plot"],
            False,
            lang,
        )
        folder.mkdir(exist_ok=True)
        path = folder / "99_skeleton.ipynb"
        nbformat.write(skeleton, path)
        test_metadata_schema(path)
        test_template_notebooks_open_with_a_title_and_end_with_a_reading(path)
