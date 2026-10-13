"""Count model fits per trial (cost accounting of the search).

A *fit* is the training of one leaf model: a base forecaster that is not a composition
(naive, ARIMA, a lag regressor, a user model), a residual corrector or a fitted
decomposition. Compositions (hybrids, combinations, ensembles) do not count themselves:
their members do, so a residual hybrid at one origin counts the base, the extra fit of
the base that produces its out-of-fold residuals and the corrector.

The counter lives in a context variable: each trial opens its own (``count_fits``), so
parallel trials in threads never mix their counts. Outside a counting context
``record_fit`` does nothing. A user corrector that does not inherit from the library's
classes can call ``record_fit()`` in its ``fit`` to be counted.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

__all__ = ["FitCounter", "count_fits", "record_fit"]


@dataclass
class FitCounter:
    total: int = 0


_CURRENT: ContextVar[FitCounter | None] = ContextVar("autohybts_fit_counter", default=None)


@contextmanager
def count_fits() -> Iterator[FitCounter]:
    """Count the fits made inside the block (nested blocks count only their own)."""

    counter = FitCounter()
    token = _CURRENT.set(counter)
    try:
        yield counter
    finally:
        _CURRENT.reset(token)


def record_fit(n: int = 1) -> None:
    """Add ``n`` fits to the current counter, if any."""

    counter = _CURRENT.get()
    if counter is not None:
        counter.total += n
