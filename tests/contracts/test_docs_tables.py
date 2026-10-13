"""The user guide's model and search tables are generated from the code (documentation/_tools)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from autohybts.registry import default_registry
from autohybts.search.budgets import PRESETS

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "documentation"


def tables():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "model_tables", DOCS / "_tools" / "model_tables.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_registered_model_has_a_row() -> None:
    module = tables()
    registry = default_registry()
    bases, correctors = module.base_tables(), module.corrector_table()
    for name in registry.names("base"):
        assert f"``{name}``" in bases, name
    for name in registry.names("corrector"):
        assert f"``{name}``" in correctors, name
    for preset, settings in PRESETS.items():
        assert f"``{preset}``" in module.budget_table()
        assert str(settings["guard"]["max_trials_total"]) in module.budget_table()


def test_pages_include_the_generated_tables(tmp_path: Path) -> None:
    tables().write(tmp_path)
    for page, include in (
        ("model_reference.rst", "_generated/base_models.rst"),
        ("model_reference.rst", "_generated/correctors.rst"),
        ("search_space.rst", "_generated/budgets.rst"),
        ("search_space.rst", "_generated/search_defaults.rst"),
    ):
        assert f".. include:: {include}" in (DOCS / "user_guide" / page).read_text(encoding="utf-8")
        assert (tmp_path / Path(include).name).is_file()
