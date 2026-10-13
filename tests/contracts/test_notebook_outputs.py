"""Stored notebook outputs are published with the docs: no local paths, at least one figure."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

DIR = Path(__file__).resolve().parents[2] / "notebooks"
NOTEBOOKS = sorted(DIR.glob("*.ipynb")) + sorted(DIR.glob("en/*.ipynb"))
# Windows drive paths, user homes and temp folders of the machine that ran the notebook.
LOCAL_PATH = re.compile(
    r"[A-Za-z]:[\\/]{1,2}(?:Users|Artigos|Documents)|/home/[^/\s]+/|/Users/[^/\s]+/|AppData|"
    r"site-packages"
)


def _output_text(output: dict[str, object]) -> str:
    parts = [output.get("text", ""), output.get("traceback", "")]
    data = output.get("data", {})
    if isinstance(data, dict):
        parts += [value for key, value in data.items() if key.startswith("text/")]
    return json.dumps(parts)


@pytest.mark.parametrize(
    "notebook", NOTEBOOKS, ids=[path.relative_to(DIR).as_posix() for path in NOTEBOOKS]
)
def test_outputs_have_no_local_paths(notebook: Path) -> None:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    leaks = [
        (index, match.group(0))
        for index, cell in enumerate(cells)
        for output in cell.get("outputs", [])
        for match in [LOCAL_PATH.search(_output_text(output))]
        if match
    ]
    assert not leaks, f"saídas com caminhos locais (célula, trecho): {leaks}"


@pytest.mark.parametrize(
    "notebook", NOTEBOOKS, ids=[path.relative_to(DIR).as_posix() for path in NOTEBOOKS]
)
def test_every_tutorial_shows_at_least_one_figure(notebook: Path) -> None:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    figures = sum(
        1
        for cell in cells
        for output in cell.get("outputs", [])
        if isinstance(output.get("data"), dict)
        and ("image/png" in output["data"] or "image/svg+xml" in output["data"])
    )
    assert figures > 0, "o tutorial não tem nenhuma figura salva"
