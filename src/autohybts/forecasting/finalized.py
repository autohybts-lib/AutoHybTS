"""The production object: a frozen candidate refitted on the permitted history (§10, R11)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from autohybts.adapters.base import BaseForecaster
from autohybts.data.schema import SeriesFrame
from autohybts.domain.context import TemporalContext
from autohybts.domain.identity import derive_seed
from autohybts.domain.specs import CandidateSpec
from autohybts.exceptions import ConfigurationError
from autohybts.forecasting.multi_step_a import MultiStepEngineA
from autohybts.registry.registry import ModelRegistry
from autohybts.residuals.generation import ResidualSettings
from autohybts.strategies.base import build_model

__all__ = [
    "FinalizedForecaster",
    "ForecastStream",
    "ForecastTicket",
    "PanelForecaster",
    "refit",
]


@dataclass(frozen=True, eq=False)
class FinalizedForecaster:
    """Structure and hyper-parameters frozen at selection; coefficients refitted on ``history``.

    Holds the history it was fitted on because forecasting needs it (lags, filter state,
    residual windows). Forecasts start right after ``training_cutoff``.
    """

    spec: CandidateSpec
    model: BaseForecaster
    history: SeriesFrame
    seed: int | None
    manifest: Mapping[str, Any] = field(default_factory=dict)
    # Observations used by fit; later ones arrived through update_state (no refit).
    n_trained: int | None = None

    @property
    def training_cutoff(self) -> pd.Timestamp:
        """Last observation the parameters were fitted on."""

        n = len(self.history) if self.n_trained is None else self.n_trained
        return self.history.timestamps[n - 1]

    @property
    def last_observation(self) -> pd.Timestamp:
        """Last observation known to the model (fit or later updates); forecasts start after it."""

        return self.history.end

    @property
    def variant(self) -> str:
        return self.spec.variant

    def update_state(
        self, observations: pd.Series | SeriesFrame, *, refit: bool = False
    ) -> FinalizedForecaster:
        """Same fitted model, longer history: new observations change the state, not the fit.

        Returns a new object; ``refit=True`` is refused because re-estimating parameters is
        a separate, explicit operation (``finalize_model`` on the longer series).
        """

        if refit:
            raise ConfigurationError(
                "update_state nunca reajusta; para reestimar, finalize o modelo de novo."
            )
        new = (
            observations
            if isinstance(observations, SeriesFrame)
            else SeriesFrame.from_series(
                observations, frequency=self.history.frequency, series_id=self.history.series_id
            )
        )
        expected = pd.date_range(self.last_observation, periods=2, freq=self.history.frequency)[1]
        if new.start != expected:
            raise ConfigurationError(
                f"Novas observações devem começar em {expected}; começam em {new.start}."
            )
        timestamps = pd.DatetimeIndex(self.history.timestamps.append(new.timestamps))
        values = np.concatenate([self.history.values, new.values])
        history = SeriesFrame(timestamps, values, self.history.frequency, self.history.series_id)
        n_trained = len(self.history) if self.n_trained is None else self.n_trained
        return FinalizedForecaster(
            self.spec, self.model, history, self.seed, self.manifest, n_trained
        )

    def stream(self) -> ForecastStream:
        """Rolling one-step forecasting with observations arriving one at a time (variant B)."""

        return ForecastStream(self)

    @property
    def model_id(self) -> str:
        """Identity of this fitted artifact: spec, data and seed."""

        return str(self.manifest.get("model_id", ""))

    def context(self) -> TemporalContext:
        return TemporalContext(
            series_id=self.history.series_id,
            frequency=self.history.frequency,
            origin_index=len(self.history) - 1,
            origin_time=self.history.end,
            observed=self.history.values,
        )

    def predict(self, fh: int) -> pd.DataFrame:
        """``fh`` steps after the last observation, indexed by date; no refit happens here."""

        frame = MultiStepEngineA().forecast(
            self.model, self.context(), fh, candidate_id=self.spec.candidate_id
        )
        return forecasts_table(frame.to_pandas())

    def __repr__(self) -> str:
        return (
            f"FinalizedForecaster({self.spec.label!r}, trained until {self.training_cutoff}, "
            f"{len(self.history)} observations)"
        )


@dataclass(frozen=True, slots=True)
class ForecastTicket:
    """A one-step forecast issued before the value it forecasts is observed."""

    origin: pd.Timestamp
    timestamp: pd.Timestamp
    prediction: float
    base_prediction: float
    correction: float


class ForecastStream:
    """``predict_next`` then ``observe``, in strict alternation; parameters stay frozen (T08)."""

    def __init__(self, model: FinalizedForecaster) -> None:
        self._model = model
        self._pending: ForecastTicket | None = None
        self._log: list[dict[str, Any]] = []

    @property
    def model(self) -> FinalizedForecaster:
        """The forecaster with every observation received so far."""

        return self._model

    def predict_next(self) -> ForecastTicket:
        if self._pending is not None:
            raise ConfigurationError("Observe o valor do passo anterior antes de prever o próximo.")
        model = self._model
        history = model.history.values
        components = getattr(model.model, "predict_components", None)
        prediction = float(model.model.predict(1, model.context())[0])
        if callable(components):
            base, correction = components(1, history)
            base_value, correction_value = float(base[0]), float(correction[0])
        else:
            base_value, correction_value = prediction, 0.0
        timestamp = pd.date_range(model.last_observation, periods=2, freq=model.history.frequency)[
            1
        ]
        self._pending = ForecastTicket(
            model.last_observation, timestamp, prediction, base_value, correction_value
        )
        return self._pending

    def observe(self, value: float) -> None:
        """Reveal the value forecast by the pending ticket; the model is not refitted."""

        if self._pending is None:
            raise ConfigurationError("Nenhuma previsão pendente: chame predict_next primeiro.")
        ticket = self._pending
        observation = pd.Series([float(value)], index=pd.DatetimeIndex([ticket.timestamp]))
        self._model = self._model.update_state(observation)
        self._log.append(
            {
                "origin": ticket.origin,
                "timestamp": ticket.timestamp,
                "y_pred": ticket.prediction,
                "y_base": ticket.base_prediction,
                "correction": ticket.correction,
                "y_true": float(value),
            }
        )
        self._pending = None

    def log(self) -> pd.DataFrame:
        """Every ticket with the value that was later observed."""

        return pd.DataFrame(self._log).set_index("timestamp") if self._log else pd.DataFrame()


def forecasts_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Public shape of predictions: indexed by timestamp, one column per field."""

    columns = ["series_id", "origin", "horizon", "y_pred", "y_true"]
    table = rows.set_index("timestamp")[columns]
    if bool(np.isnan(table["y_true"].to_numpy()).all()):
        table = table.drop(columns="y_true")
    return table


def refit(
    candidate: CandidateSpec,
    history: SeriesFrame,
    registry: ModelRegistry,
    *,
    session_seed: int,
    residual_settings: ResidualSettings,
) -> tuple[BaseForecaster, int | None]:
    """Fresh model for the frozen spec, fitted on ``history`` only (T13).

    No object fitted during validation is reused; residuals are regenerated with the same
    policy inside ``fit``.
    """

    if not isinstance(history, SeriesFrame):
        raise ConfigurationError("O refit exige um SeriesFrame com o histórico permitido.")
    model, seed = build_model(
        candidate,
        registry,
        unit_seed=derive_seed(session_seed, candidate.candidate_id, "refit"),
        residual_settings=residual_settings,
    )
    model.fit(history.values)
    return model, seed


@dataclass(frozen=True, eq=False)
class PanelForecaster:
    """One finalized model per series; each series keeps its own configuration and cutoff."""

    models: Mapping[str, FinalizedForecaster]
    spec: CandidateSpec
    manifest: Mapping[str, Any] = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        return str(self.manifest.get("model_id", ""))

    @property
    def variant(self) -> str:
        return self.spec.variant

    @property
    def series_ids(self) -> tuple[str, ...]:
        return tuple(self.models)

    @property
    def training_cutoff(self) -> dict[str, pd.Timestamp]:
        return {sid: model.training_cutoff for sid, model in self.models.items()}

    def __getitem__(self, series_id: str) -> FinalizedForecaster:
        try:
            return self.models[series_id]
        except KeyError:
            raise ConfigurationError(
                f"Série {series_id!r} não está no modelo; disponíveis: {list(self.models)}."
            ) from None

    def predict(self, fh: int) -> pd.DataFrame:
        """``fh`` steps after each series' last observation, stacked by series."""

        return pd.concat([model.predict(fh) for model in self.models.values()])

    def __repr__(self) -> str:
        return f"PanelForecaster({self.spec.label!r}, {len(self.models)} séries)"
