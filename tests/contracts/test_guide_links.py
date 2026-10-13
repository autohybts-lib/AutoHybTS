"""Every guide page a tutorial illustrates links back to it, in both languages."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAIRS = [
    (notebook.stem, guide)
    for notebook in sorted((ROOT / "notebooks").glob("[0-9]*.ipynb"))
    for guide in json.loads(notebook.read_text(encoding="utf-8"))["metadata"]["autohybts"]["guide"]
]


@pytest.mark.parametrize("language", ["", "en/"])
@pytest.mark.parametrize(("notebook", "guide"), PAIRS)
def test_guide_page_links_to_its_tutorial(notebook: str, guide: str, language: str) -> None:
    page = (ROOT / "documentation" / f"{language}{guide}.rst").read_text(encoding="utf-8")
    assert f"tutorials/notebooks/{notebook}" in page, f"{language}{guide} não cita {notebook}"
