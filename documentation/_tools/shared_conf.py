"""Sphinx settings shared by the Portuguese (``documentation/``) and English
(``documentation/en/``) trees.

Each ``conf.py`` calls :func:`settings` for the Sphinx variables and :func:`prepare` for
the build-time steps: copy the tutorial notebooks of its language next to the pages and
generate the model and search tables from the code.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
DOCS = TOOLS.parent
ROOT = DOCS.parent

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def settings(language: str) -> dict[str, Any]:
    """Sphinx configuration values for one language (``"pt_BR"`` or ``"en"``)."""
    import autohybts

    release = autohybts.__version__
    return {
        "project": "AutoHybTS",
        "author": "João Ataíde",
        "copyright": "2026, João Ataíde",
        "release": release,
        "version": release,
        "language": language,
        "extensions": [
            "sphinx.ext.autodoc",
            "sphinx.ext.autosummary",
            "sphinx.ext.napoleon",
            "sphinx.ext.viewcode",
            "sphinx.ext.intersphinx",
            "sphinx_copybutton",
            "myst_nb",
        ],
        "source_suffix": {".rst": "restructuredtext", ".ipynb": "myst-nb"},
        "exclude_patterns": ["_build", "**/.ipynb_checkpoints", "**/_generated", "_tools"],
        "autodoc_member_order": "bysource",
        "autodoc_typehints": "description",
        "autodoc_default_options": {
            "members": True,
            "undoc-members": False,
            "show-inheritance": False,
        },
        "autosummary_generate": False,
        "napoleon_google_docstring": True,
        "napoleon_numpy_docstring": False,
        "intersphinx_mapping": {
            "python": ("https://docs.python.org/3", None),
            "pandas": ("https://pandas.pydata.org/docs", None),
        },
        # Notebooks are documentation, not tests: render them as they are, outputs
        # included if the author saved them.
        "nb_execution_mode": "off",
        "myst_enable_extensions": ["dollarmath", "colon_fence"],
        "html_theme": "furo",
        "html_title": f"AutoHybTS {release}",
        "html_logo": str(ROOT / "figures" / "AutoHybTS.png"),
        "html_favicon": str(ROOT / "figures" / "AutoHybTS.png"),
        "html_static_path": [],
        "html_theme_options": {
            "sidebar_hide_name": True,
            "navigation_with_keys": True,
            "light_css_variables": {
                "color-brand-primary": "#1F5F99",
                "color-brand-content": "#1F5F99",
            },
            "dark_css_variables": {
                "color-brand-primary": "#7FB3E6",
                "color-brand-content": "#7FB3E6",
            },
        },
    }


def prepare(source_dir: Path, notebooks: Path, language: str) -> None:
    """Copy ``notebooks/*.ipynb`` into ``tutorials/notebooks`` and write the tables.

    ``language`` is the table language (``"pt"`` or ``"en"``).
    """
    import model_tables

    target = source_dir / "tutorials" / "notebooks"
    target.mkdir(parents=True, exist_ok=True)
    for notebook in sorted(notebooks.glob("*.ipynb")):
        shutil.copy2(notebook, target / notebook.name)
    model_tables.write(source_dir / "user_guide" / "_generated", language)
