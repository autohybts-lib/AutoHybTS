"""Sphinx configuration of the AutoHybTS documentation.

Build: ``sphinx-build documentation documentation/_build/html``. The tutorial notebooks
live in ``notebooks/``; they are copied next to this file at build time and rendered
without execution (run them locally to see the outputs).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

import autohybts  # noqa: E402

project = "AutoHybTS"
author = "João Ataíde"
copyright = "2026, João Ataíde"
release = autohybts.__version__
version = release
language = "pt_BR"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_nb",
]
source_suffix = {".rst": "restructuredtext", ".ipynb": "myst-nb"}
exclude_patterns = ["_build", "**/.ipynb_checkpoints"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": False}
autosummary_generate = False
napoleon_google_docstring = True
napoleon_numpy_docstring = False
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

# Notebooks are documentation, not tests: render them as they are, outputs included if
# the author saved them.
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath", "colon_fence"]

html_theme = "furo"
html_title = f"AutoHybTS {release}"
html_logo = str(ROOT / "figures" / "AutoHybTS.png")
html_favicon = str(ROOT / "figures" / "AutoHybTS.png")
html_static_path: list[str] = []
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "light_css_variables": {"color-brand-primary": "#1F5F99", "color-brand-content": "#1F5F99"},
    "dark_css_variables": {"color-brand-primary": "#7FB3E6", "color-brand-content": "#7FB3E6"},
}


def _copy_notebooks() -> None:
    target = HERE / "tutorials" / "notebooks"
    target.mkdir(parents=True, exist_ok=True)
    for notebook in sorted((ROOT / "notebooks").glob("*.ipynb")):
        shutil.copy2(notebook, target / notebook.name)


_copy_notebooks()
