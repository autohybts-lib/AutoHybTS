"""Variant B: rolling one-step forecasts with frozen parameters (§7.4, §13.3).

At each origin the model forecasts one step from the observations available so far;
only then is the actual value revealed and appended. Nothing is refitted: models
condition on the longer context with the parameters learned in ``fit`` (for hybrids,
the frozen base's one-step error on each revealed value joins the residual window).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from autohybts.adapters.base import BaseForecaster, FloatArray
from autohybts.domain.context import TemporalContext
from autohybts.exceptions import ConfigurationError

__all__ = ["rolling_one_step"]


def rolling_one_step(
    model: BaseForecaster,
    observed: npt.ArrayLike,
    future: npt.ArrayLike,
    *,
    origin_index: int | None = None,
    series_id: str = "__rolling__",
    frequency: str = "",
) -> FloatArray:
    """One-step forecasts of each value of ``future``, revealing it only after its forecast.

    ``observed`` is everything known at the first origin (the model's training data or an
    extension of it); ``origin_index`` is the absolute position of its last value.
    """

    history = np.asarray(observed, dtype=np.float64)
    upcoming = np.asarray(future, dtype=np.float64)
    if history.ndim != 1 or upcoming.ndim != 1 or upcoming.size == 0:
        raise ConfigurationError("rolling_one_step espera vetores 1-D e ao menos um valor futuro.")
    last = history.size - 1 if origin_index is None else origin_index
    start = last - history.size + 1
    path = np.concatenate([history, upcoming])
    forecasts = np.empty(upcoming.size, dtype=np.float64)
    for step in range(upcoming.size):
        end = history.size + step  # values revealed so far: path[:end]
        context = TemporalContext(
            series_id=series_id,
            frequency=frequency,
            origin_index=start + end - 1,
            origin_time=None,
            observed=path[:end],
            observed_start=start,
        )
        forecasts[step] = model.predict(1, context)[0]
    return forecasts
