"""The production object: a frozen candidate refitted on the permitted history (§10, R11)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from autohybts.adapters.base import BaseForecaster
from autohybts.data.exog import ExogState, as_rows
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

    With exogenous features (ADR-0011) ``exog`` keeps the covariates up to the last
    observation; ``predict`` takes the values of ``future`` columns for the horizon
    (``X_future``) and ``update_state`` the covariates of the new dates (``X_new``).
    """

    spec: CandidateSpec
    model: BaseForecaster
    history: SeriesFrame
    seed: int | None
    manifest: Mapping[str, Any] = field(default_factory=dict)
    # Observations used by fit; later ones arrived through update_state (no refit).
    n_trained: int | None = None
    exog: ExogState | None = None

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
        self,
        observations: pd.Series | SeriesFrame,
        *,
        refit: bool = False,
        X_new: pd.DataFrame | None = None,
    ) -> FinalizedForecaster:
        """Same fitted model, longer history: new observations change the state, not the fit.

        Returns a new object; ``refit=True`` is refused because re-estimating parameters is
        a separate, explicit operation (``finalize_model`` on the longer series). A model
        with exogenous features takes the covariates of the new dates in ``X_new``.
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
        exog = self.exog
        if exog is None and X_new is not None:
            raise ConfigurationError("Este modelo não usa exógenas; X_new não se aplica.")
        if exog is not None:
            covariates = exog.extended(X_new)
            covariates = covariates.loc[covariates.index <= history.end]
            exog = ExogState(covariates, exog.kinds, exog.features, exog.start)
        return FinalizedForecaster(
            self.spec, self.model, history, self.seed, self.manifest, n_trained, exog
        )

    def stream(self) -> ForecastStream:
        """Rolling one-step forecasting with observations arriving one at a time (variant B)."""

        return ForecastStream(self)

    @property
    def model_id(self) -> str:
        """Identity of this fitted artifact: spec, data and seed."""

        return str(self.manifest.get("model_id", ""))

    def context(self, horizon: int = 0, X_future: pd.DataFrame | None = None) -> TemporalContext:
        """What the model sees at its last observation; with exogenous features, their rows
        up to ``horizon`` dates ahead (``future`` columns from ``X_future``)."""

        n = len(self.history)
        if self.exog is None:
            return TemporalContext(
                series_id=self.history.series_id,
                frequency=self.history.frequency,
                origin_index=n - 1,
                origin_time=self.history.end,
                observed=self.history.values,
            )
        limit = self.exog.max_ahead
        if limit is not None and horizon > limit:
            raise ConfigurationError(
                f"Horizonte {horizon} maior que a defasagem das exógenas past ({limit}): os "
                "valores necessários ainda não teriam sido observados."
            )
        timeline = pd.date_range(
            self.history.timestamps[0], periods=n + horizon, freq=self.history.frequency
        )
        design = self.exog.design(timeline, X_future)
        start = self.exog.start
        return TemporalContext(
            series_id=self.history.series_id,
            frequency=self.history.frequency,
            origin_index=n - 1,
            origin_time=self.history.end,
            observed=self.history.values[start:],
            observed_start=start,
            exog=design.rows(start, n + horizon),
            exog_ahead=horizon,
            exog_max_ahead=design.max_ahead,
        )

    def predict(self, fh: int, X_future: pd.DataFrame | None = None) -> pd.DataFrame:
        """``fh`` steps after the last observation, indexed by date; no refit happens here.

        A model with exogenous ``future`` columns needs their values for the ``fh`` dates in
        ``X_future``; ``past`` columns come from the history.
        """

        frame = MultiStepEngineA().forecast(
            self.model, self.context(fh, X_future), fh, candidate_id=self.spec.candidate_id
        )
        return forecasts_table(frame.to_pandas())

    def __repr__(self) -> str:
        return (
            f"FinalizedForecaster({self.spec.label!r}, trained until {self.training_cutoff}, "
            f"{len(self.history)} observations)"
        )


@dataclass(frozen=True, slots=True)
class ForecastTicket:
    """A one-step forecast issued before the value it forecasts is observed.

    Attributes:
        origin: last observation the forecast used.
        timestamp: date being forecast.
        prediction: the forecast (``base_prediction + correction``).
        base_prediction: the base model's part.
        correction: the residual corrector's part (0 without a corrector).
    """

    origin: pd.Timestamp
    timestamp: pd.Timestamp
    prediction: float
    base_prediction: float
    correction: float


class ForecastStream:
    """``predict_next`` then ``observe``, in strict alternation; parameters stay frozen."""

    def __init__(self, model: FinalizedForecaster) -> None:
        self._model = model
        self._pending: ForecastTicket | None = None
        self._log: list[dict[str, Any]] = []
        self._rows: pd.DataFrame | None = None

    @property
    def model(self) -> FinalizedForecaster:
        """The forecaster with every observation received so far."""

        return self._model

    def predict_next(self, exog: Any = None) -> ForecastTicket:
        """Forecast the next step; ``observe`` must reveal its value before the next call.

        ``exog`` holds the covariates of the date being forecast (a mapping or ``Series`` by
        column, or a one-row ``DataFrame``), needed for ``future`` columns.
        """

        if self._pending is not None:
            raise ConfigurationError("Observe o valor do passo anterior antes de prever o próximo.")
        model = self._model
        timestamp = pd.date_range(model.last_observation, periods=2, freq=model.history.frequency)[
            1
        ]
        rows = None if exog is None else as_rows(exog, timestamp)
        if model.exog is None and rows is not None:
            raise ConfigurationError("Este modelo não usa exógenas.")
        context = model.context(1, rows)
        components = getattr(model.model, "predict_components", None)
        prediction = float(model.model.predict(1, context)[0])
        if callable(components):
            if context.exog is None:
                base, correction = components(1, context.observed)
            else:
                base, correction = components(1, context.observed, context.exog)
            base_value, correction_value = float(base[0]), float(correction[0])
        else:
            base_value, correction_value = prediction, 0.0
        self._rows = rows
        self._pending = ForecastTicket(
            model.last_observation, timestamp, prediction, base_value, correction_value
        )
        return self._pending

    def observe(self, value: float, exog: Any = None) -> None:
        """Reveal the value forecast by the pending ticket; the model is not refitted.

        ``exog`` adds covariates observed with the value (``past`` columns of that date).
        """

        if self._pending is None:
            raise ConfigurationError("Nenhuma previsão pendente: chame predict_next primeiro.")
        ticket = self._pending
        observation = pd.Series([float(value)], index=pd.DatetimeIndex([ticket.timestamp]))
        rows = self._rows
        if exog is not None:
            extra = as_rows(exog, ticket.timestamp)
            rows = extra if rows is None else extra.combine_first(rows)
        if self._model.exog is None:
            self._model = self._model.update_state(observation)
        else:
            self._model = self._model.update_state(observation, X_new=rows)
        self._rows = None
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
    exog: ExogState | None = None,
) -> tuple[BaseForecaster, int | None]:
    """Fresh model for the frozen spec, fitted on ``history`` only (T13).

    No object fitted during validation is reused; residuals are regenerated with the same
    policy inside ``fit``. A spec with exogenous features is fitted on ``history`` from
    ``exog.start`` with the covariates of those dates.
    """

    if not isinstance(history, SeriesFrame):
        raise ConfigurationError("O refit exige um SeriesFrame com o histórico permitido.")
    model, seed = build_model(
        candidate,
        registry,
        unit_seed=derive_seed(session_seed, candidate.candidate_id, "refit"),
        residual_settings=residual_settings,
    )
    if candidate.exog:
        if exog is None:
            raise ConfigurationError("A candidata usa exógenas: informe o estado das exógenas.")
        rows = exog.design(history.timestamps).rows(exog.start, len(history))
        model.fit(history.values[exog.start :], rows)
    else:
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

    def predict(self, fh: int, X_future: Mapping[str, pd.DataFrame] | None = None) -> pd.DataFrame:
        """``fh`` steps after each series' last observation, stacked by series.

        ``X_future`` maps each series to the covariates of its horizon, for members with
        exogenous ``future`` columns.
        """

        rows = X_future or {}
        return pd.concat([model.predict(fh, rows.get(sid)) for sid, model in self.models.items()])

    def __repr__(self) -> str:
        return f"PanelForecaster({self.spec.label!r}, {len(self.models)} séries)"
