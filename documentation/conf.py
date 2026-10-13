"""Sphinx configuration of the AutoHybTS documentation (Portuguese).

Build: ``sphinx-build documentation documentation/_build/html``. The English version
lives in ``documentation/en`` (``sphinx-build documentation/en
documentation/_build/html-en``); both share ``_tools/shared_conf.py``. The tutorial
notebooks live in ``notebooks/`` (English: ``notebooks/en/``); they are copied next to
this file at build time and rendered without execution (run them locally to see the
outputs).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_tools"))

import shared_conf  # noqa: E402

globals().update(shared_conf.settings(language="pt_BR"))
# The English tree is a separate Sphinx project.
exclude_patterns = [*shared_conf.settings("pt_BR")["exclude_patterns"], "en"]

shared_conf.prepare(HERE, shared_conf.ROOT / "notebooks", language="pt")
