"""Progress of long steps: tqdm bars when asked, nothing otherwise.

``progress=True`` in ``AutoHybridExperiment`` shows a bar per search stage, for the
finalists on ``V_select``, the choice of exogenous features and the outer folds of the
nested cross-validation (``tqdm.auto``: a widget in notebooks, text in a terminal). tqdm
is optional (extra ``progress``) and imported only when a bar is created, so
``import autohybts`` stays light. Progress is reporting only: it never changes what is
evaluated, the order or any identity.

Independently of bars, every experiment event is also sent to the standard ``logging``
logger ``"autohybts"`` at level ``INFO``.
"""

from __future__ import annotations

import importlib
import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol

from autohybts.exceptions import ConfigurationError, UnsupportedCapabilityError

__all__ = ["LOGGER", "NullProgress", "Progress", "ProgressBar", "TqdmProgress", "make_progress"]

LOGGER = logging.getLogger("autohybts")


class ProgressBar(Protocol):
    def update(self, n: int = 1) -> None: ...

    def set_postfix_str(self, text: str) -> None: ...


class Progress(Protocol):
    """Creates bars for steps with a known number of items."""

    def bar(self, total: int, description: str) -> Any: ...


class _NullBar:
    def update(self, n: int = 1) -> None:
        del n

    def set_postfix_str(self, text: str) -> None:
        del text


class NullProgress:
    """No bars (the default)."""

    @contextmanager
    def bar(self, total: int, description: str) -> Iterator[ProgressBar]:
        del total, description
        yield _NullBar()


@contextmanager
def _without_ipywidgets_warning() -> Iterator[None]:
    # Without ipywidgets, tqdm.auto falls back to text bars in a notebook and warns (on import
    # and per bar) with the absolute path of the user's environment; text bars are enough.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="IProgress not found")
        yield


class TqdmProgress:
    """tqdm bars (``tqdm.auto``); ``leave`` keeps finished bars on screen."""

    def __init__(self, leave: bool = True) -> None:
        try:
            with _without_ipywidgets_warning():
                tqdm = importlib.import_module("tqdm.auto").tqdm
        except ModuleNotFoundError as error:
            raise UnsupportedCapabilityError(
                'progress=True requer o tqdm: pip install "autohybts[progress]".'
            ) from error
        self._tqdm = tqdm
        self.leave = leave

    @contextmanager
    def bar(self, total: int, description: str) -> Iterator[ProgressBar]:
        with _without_ipywidgets_warning():
            bar = self._tqdm(total=total, desc=description, leave=self.leave, unit="cand")
        try:
            yield bar
        finally:
            bar.close()


def make_progress(progress: object) -> NullProgress | TqdmProgress:
    """``False``/``None``: no bars; ``True``: tqdm bars."""

    if progress is None or progress is False:
        return NullProgress()
    if progress is True:
        return TqdmProgress()
    raise ConfigurationError(f"progress deve ser True ou False; recebido {progress!r}.")
