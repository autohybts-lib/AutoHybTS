"""Tutorial notebooks are the executable contract of the public API (ADR-0001).

Each notebook lists in ``metadata.autohybts.requires`` the public names it calls. It is
skipped while any of them is missing and runs end to end once they all exist, so a
stage that ships the API automatically switches its notebook on. Notebooks marked
``heavy`` run only with ``AUTOHYBTS_RUN_HEAVY=1``; one whose ``extras`` are not installed
is skipped and says which (a skip is never evidence that it runs). The metadata schema
is checked in ``tests/contracts/test_notebook_metadata.py``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import re
import tomllib
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).parents[2] / "notebooks"
NOTEBOOKS = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
EXTRAS: dict[str, list[str]] = tomllib.loads(
    (NOTEBOOKS_DIR.parent / "pyproject.toml").read_text(encoding="utf-8")
)["project"]["optional-dependencies"]


def _metadata(path: Path) -> dict[str, object]:
    meta = json.loads(path.read_text(encoding="utf-8"))["metadata"].get("autohybts", {})
    assert isinstance(meta, dict)
    return meta


def _resolves(dotted: str) -> bool:
    """True if ``pkg.module.Attr.attr`` can be imported and walked attribute by attribute."""

    parts = dotted.split(".")
    for split in range(len(parts), 0, -1):
        candidate = ".".join(parts[:split])
        try:
            target: object = importlib.import_module(candidate)
        except ModuleNotFoundError as error:
            # Only "this name is not a module" means not implemented; a missing
            # dependency inside an existing module must fail loudly, not skip.
            if error.name is None or not (candidate + ".").startswith(error.name + "."):
                raise
            continue
        for attribute in parts[split:]:
            if not hasattr(target, attribute):
                return False
            target = getattr(target, attribute)
        return True
    return False


def _missing_extras(extras: object) -> list[str]:
    """Distributions of the declared extras that are not installed, as ``extra (dist)``."""

    assert isinstance(extras, list)
    missing = []
    for extra in extras:
        for requirement in EXTRAS[extra]:
            match = re.match(r"[A-Za-z0-9_.\-]+", requirement)
            assert match is not None, requirement
            try:
                importlib.metadata.version(match.group(0))
            except importlib.metadata.PackageNotFoundError:
                missing.append(f"{extra} ({match.group(0)})")
    return missing


def _params() -> list[object]:
    params = []
    for path in NOTEBOOKS:
        marks = [pytest.mark.slow]
        if _metadata(path).get("network"):
            marks.append(pytest.mark.network)
        params.append(pytest.param(path, id=path.stem, marks=marks))
    return params


def test_every_notebook_declares_its_requirements() -> None:
    assert NOTEBOOKS
    for path in NOTEBOOKS:
        requires = _metadata(path).get("requires")
        assert isinstance(requires, list), path.name
        assert requires, path.name


def test_missing_extras_names_the_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _missing_extras([]) == []
    monkeypatch.setitem(EXTRAS, "fake_extra", ["not-a-real-distribution>=1"])
    assert _missing_extras(["fake_extra"]) == ["fake_extra (not-a-real-distribution)"]


def test_resolver() -> None:
    assert _resolves("autohybts.domain.identity.stable_hash")
    assert not _resolves("autohybts.domain.identity.missing_name")
    assert not _resolves("autohybts.not_a_module.Thing")


def test_resolver_does_not_hide_broken_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "broken_pkg").mkdir()
    (tmp_path / "broken_pkg" / "__init__.py").write_text(
        "import dependency_that_is_not_installed\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError, match="dependency_that_is_not_installed"):
        _resolves("broken_pkg.Thing")


@pytest.mark.parametrize("path", _params())
# Windows-only notice from pyzmq about the asyncio event loop; harmless for the kernel.
@pytest.mark.filterwarnings("ignore:Proactor event loop:RuntimeWarning")
def test_notebook_executes(path: Path, tmp_path: Path) -> None:
    requires = _metadata(path)["requires"]
    assert isinstance(requires, list)
    missing = [name for name in requires if not _resolves(name)]
    if missing:
        pytest.skip(f"API not implemented yet: {', '.join(missing)}")
    meta = _metadata(path)
    if meta.get("heavy") and os.environ.get("AUTOHYBTS_RUN_HEAVY") != "1":
        pytest.skip("heavy notebook: set AUTOHYBTS_RUN_HEAVY=1 to run it")
    absent = _missing_extras(meta.get("extras", []))
    if absent:
        pytest.skip(f"extras not installed: {', '.join(absent)}")

    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")

    notebook = nbformat.read(path, as_version=4)
    # Run inside tmp_path so saved artifacts never land in the repository.
    nbclient.NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    ).execute()
