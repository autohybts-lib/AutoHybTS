"""Stored notebook outputs are published with the docs: they must not leak local paths."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[2] / "notebooks").glob("*.ipynb"))
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


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=[path.name for path in NOTEBOOKS])
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
