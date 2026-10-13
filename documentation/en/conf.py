"""Sphinx configuration of the AutoHybTS documentation (English).

Build: ``sphinx-build documentation/en documentation/_build/html-en``. The pages mirror
the Portuguese tree in ``documentation/`` page by page (same file names, checked by
``tests/contracts/test_docs_translation.py``); the settings come from
``documentation/_tools/shared_conf.py``. The tutorial notebooks are the English copies in
``notebooks/en/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_tools"))

import shared_conf  # noqa: E402

globals().update(shared_conf.settings(language="en"))

shared_conf.prepare(HERE, shared_conf.ROOT / "notebooks" / "en", language="en")
