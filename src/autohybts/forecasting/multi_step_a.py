"""Variant A engine: open multi-step forecast from an origin (§7.3, §13.2).

The recursion itself lives in the forecaster (for hybrids, in
``residuals.recursive.recursive_correction``); this engine turns a fitted model and a
context into a dated ``ForecastFrame`` and refuses anything but a single origin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autohybts.adapters.base import BaseForecaster
from autohybts.domain.context import TemporalContext
from autohybts.exceptions import ConfigurationError
from autohybts.forecasting.forecast_frame import ForecastFrame

__all__ = ["MultiStepEngineA"]


class MultiStepEngineA:
    def forecast(
        self,
        model: BaseForecaster,
        context: TemporalContext,
        horizon: int,
        *,
        candidate_id: str = "",
        fold_id: str = "",
    ) -> ForecastFrame:
        if context.origin_time is None:
            raise ConfigurationError("O motor A precisa de um contexto com data de origem.")
        prediction = model.predict(horizon, context)
        stamps = pd.date_range(context.origin_time, periods=horizon + 1, freq=context.frequency)[1:]
        frame = pd.DataFrame(
            {
                "series_id": context.series_id,
                "candidate_id": candidate_id,
                "fold_id": fold_id,
                "origin": context.origin_time,
                "timestamp": stamps,
                "horizon": np.arange(1, horizon + 1),
                "y_pred": prediction,
                "y_true": np.nan,
            }
        )
        return ForecastFrame(frame, context.frequency)
