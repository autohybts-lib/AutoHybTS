"""One-call estimator on top of ``AutoHybridExperiment``.

``AutoHybridForecaster(**params).fit(y)`` runs the experiment, selects the champion and
finalizes it; ``predict`` forecasts from the end of the permitted history.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from autohybts.adapters.base import Parametrized
from autohybts.data.panel import PanelFrame
from autohybts.data.schema import SeriesFrame
from autohybts.exceptions import ConfigurationError
from autohybts.experiment import AutoHybridExperiment
from autohybts.forecasting.finalized import FinalizedForecaster, PanelForecaster
from autohybts.results import CompareResult

__all__ = ["AutoHybridForecaster"]


class AutoHybridForecaster(Parametrized):
    """Experiment, selection and finalization in one call, for operational forecasting.

    Args:
        **params: any parameter of ``AutoHybridExperiment`` (``fh``, ``budget``,
            ``variant`` …); ``test_size`` defaults to 0 here, so the whole series is used
            and nothing is sealed. The fitted experiment, the comparison and the
            finalized model stay in ``experiment_``, ``compare_`` and ``model_``.
    """

    def __init__(self, **params: Any) -> None:
        params.setdefault("test_size", 0)
        self.params = params

    def get_params(self) -> dict[str, Any]:
        return {"params": dict(self.params)}

    def fit(
        self, y: pd.Series | SeriesFrame | PanelFrame | Mapping[str, pd.Series]
    ) -> AutoHybridForecaster:
        self.experiment_ = AutoHybridExperiment(**self.params).fit(y)
        self.compare_: CompareResult = self.experiment_.compare_models()
        self.model_: FinalizedForecaster | PanelForecaster = self.experiment_.finalize_model(
            self.compare_.best
        )
        return self

    def predict(self, fh: int | None = None) -> pd.DataFrame:
        self._fitted()
        return self.model_.predict(fh or self.experiment_.fh)

    def leaderboard(self) -> pd.DataFrame:
        self._fitted()
        return self.compare_.leaderboard

    @property
    def selection_history_(self) -> pd.DataFrame:
        self._fitted()
        return self.compare_.decisions

    def get_best_model(self) -> FinalizedForecaster | PanelForecaster:
        self._fitted()
        return self.model_

    def _fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise ConfigurationError("Chame fit(y) antes.")
