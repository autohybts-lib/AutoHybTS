"""Temporal evaluation of a candidate over the origins of a plan (architecture §4.1, §6).

For every origin: build a fresh model, fit it on that origin's training slice, forecast
``H`` steps from the origin's context and score against the targets. Nothing is reused
between origins. Candidate failures become auditable trials; configuration errors and
temporal leakage stop the run.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import autohybts
from autohybts.adapters.base import BaseForecaster, FloatArray
from autohybts.data.exog import ExogFrame, minimum_lag
from autohybts.data.panel import PanelPlan
from autohybts.data.splitter import (
    CrossValidationPlan,
    Origin,
    TemporalPlan,
    ensure_not_sealed,
)
from autohybts.domain.context import TemporalContext
from autohybts.domain.fit_count import count_fits
from autohybts.domain.identity import derive_seed
from autohybts.domain.results import (
    EvaluationResult,
    Layer,
    TrialResult,
    combine_results,
    unit_key,
)
from autohybts.domain.specs import CandidateSpec, trial_key
from autohybts.evaluation.metrics import compute_metrics, resolve_metrics
from autohybts.exceptions import (
    AutoHybTSError,
    CandidateFitError,
    ConfigurationError,
    InsufficientHistoryError,
    UnsupportedCapabilityError,
)
from autohybts.forecasting.rolling_one_step_b import rolling_one_step
from autohybts.registry.registry import ModelRegistry
from autohybts.residuals.generation import ResidualSettings
from autohybts.strategies.base import build_model
from autohybts.types import TrialStatus

__all__ = ["DEFAULT_METRICS", "TemporalEvaluator"]

DEFAULT_METRICS: Final = ("MASE", "MAE", "RMSE", "sMAPE")
_MAX_REASON: Final = 500


@dataclass(frozen=True, slots=True)
class TemporalEvaluator:
    """Scores candidates on the same origins with the same metrics.

    Args:
        registry: where model names are resolved.
        metrics: metric names; the first one is the default for rankings.
        seasonal_period: ``m`` of the MASE scale.
        session_seed: root of the per-(candidate, fold) seeds given to stochastic models.
        residual_settings: residual generation shared by every hybrid candidate.
        exog: covariates of the series (ADR-0011), used by candidates with ``exog``
            features; ``None`` when the experiment has none.
    """

    registry: ModelRegistry
    metrics: tuple[str, ...] = DEFAULT_METRICS
    seasonal_period: int = 1
    session_seed: int = 42
    residual_settings: ResidualSettings = field(default_factory=ResidualSettings)
    exog: ExogFrame | None = None

    def __post_init__(self) -> None:
        resolve_metrics(self.metrics)
        if (
            isinstance(self.seasonal_period, bool)
            or not isinstance(self.seasonal_period, int)
            or self.seasonal_period < 1
        ):
            raise ConfigurationError(
                f"seasonal_period deve ser inteiro >= 1; recebido {self.seasonal_period!r}."
            )

    def evaluate(
        self,
        candidate: CandidateSpec,
        plan: TemporalPlan | CrossValidationPlan | PanelPlan,
        layer: Layer | None = None,
    ) -> EvaluationResult:
        """Score ``candidate`` on every origin (of every series, for a panel plan)."""

        if isinstance(plan, PanelPlan):
            return self.evaluate_structure(dict.fromkeys(plan.series_ids, candidate), plan, layer)
        ensure_not_sealed(candidate, plan, layer)
        if not isinstance(plan, TemporalPlan | CrossValidationPlan):
            raise ConfigurationError("evaluate exige um plano temporal.")
        self._check_models(candidate)
        trials = tuple(self._trial(candidate, plan, origin) for origin in plan.origins(layer))
        return EvaluationResult(
            candidate=candidate,
            plan_fingerprint=plan.fingerprint,
            frequency=plan.history.frequency,
            seasonal_period=self.seasonal_period,
            metric_names=self.metrics,
            trials=trials,
        )

    def evaluate_structure(
        self,
        per_series: Mapping[str, CandidateSpec],
        plan: PanelPlan,
        layer: Layer | None = None,
        representative: CandidateSpec | None = None,
    ) -> EvaluationResult:
        """Each series with its own configuration (best configuration per series, ADR-0004).

        The trials of all series are combined; units are ``series:fold``.
        """

        ensure_not_sealed(plan, layer)
        if set(per_series) != set(plan.series_ids):
            raise ConfigurationError("Informe uma configuração para cada série do painel.")
        parts = [self.evaluate(per_series[sid], plan[sid], layer) for sid in plan.series_ids]
        chosen = representative or per_series[plan.series_ids[0]]
        return combine_results(parts, chosen, plan.fingerprint)

    def _check_models(self, candidate: CandidateSpec) -> None:
        # Unknown models are configuration errors, raised before any trial runs.
        for name in _base_names(candidate):
            self.registry.entry(name)
        for name in _corrector_names(candidate):
            self.registry.entry(name, "corrector")

    def _trial(
        self, candidate: CandidateSpec, plan: TemporalPlan | CrossValidationPlan, origin: Origin
    ) -> TrialResult:
        unsupported: UnsupportedCapabilityError | None = None
        try:
            model, seed = build_model(
                candidate,
                self.registry,
                unit_seed=derive_seed(
                    self.session_seed,
                    candidate.candidate_id,
                    unit_key(origin.series_id, origin.fold_id),
                ),
                residual_settings=self.residual_settings,
            )
        except UnsupportedCapabilityError as error:  # e.g. a strategy not built yet
            unsupported, seed = error, None
        key = trial_key(
            candidate,
            unit_id=origin.origin_id,
            seed=seed,
            data_fingerprint=plan.fingerprint,
            code_version=autohybts.__version__,
            protocol=self.protocol(),
        )

        def result(status: TrialStatus, **extra: Any) -> TrialResult:
            return TrialResult(
                candidate_id=candidate.candidate_id,
                series_id=origin.series_id,
                fold_id=origin.fold_id,
                layer=origin.layer,
                origin_index=origin.origin_index,
                origin_time=origin.origin_time,
                horizon=origin.horizon,
                trial_key=key,
                seed=seed,
                status=status,
                **extra,
            )

        if unsupported is not None:
            return result("SKIPPED", reason=_reason(unsupported), n_fits=0)
        train = plan.train_values(origin)
        started = time.perf_counter()
        with count_fits() as fits:
            outcome = self._fit_and_forecast(candidate, model, plan, origin, train)
        if isinstance(outcome, BaseException):
            status: TrialStatus = (
                "SKIPPED" if isinstance(outcome, UnsupportedCapabilityError) else "FAILED"
            )
            return result(status, reason=_reason(outcome), n_fits=fits.total)
        forecast, fitted, done = outcome

        actual = plan.target_values(origin)
        metrics = compute_metrics(
            resolve_metrics(self.metrics), actual, forecast, train, self.seasonal_period
        )
        return result(
            "SUCCEEDED",
            y_pred=tuple(float(v) for v in forecast),
            y_true=tuple(float(v) for v in actual),
            metrics=metrics,
            fit_seconds=fitted - started,
            predict_seconds=done - fitted,
            diagnostics=_diagnostics(model),
            n_fits=fits.total,
        )

    def _fit_and_forecast(
        self,
        candidate: CandidateSpec,
        model: BaseForecaster,
        plan: TemporalPlan | CrossValidationPlan,
        origin: Origin,
        train: FloatArray,
    ) -> tuple[FloatArray, float, float] | BaseException:
        """Fit at the origin and forecast; candidate failures are returned, not raised."""

        try:
            self._check_variant(candidate)
            self._check_capabilities(candidate, origin.horizon)
            if candidate.exog:
                return self._fit_and_forecast_exog(candidate, model, plan, origin)
            model.fit(train)
            fitted = time.perf_counter()
            if candidate.variant == "B":
                # H one-step forecasts; each target is revealed only after its forecast.
                forecast = rolling_one_step(
                    model,
                    train,
                    plan.target_values(origin),
                    origin_index=origin.origin_index,
                    series_id=origin.series_id,
                    frequency=plan.history.frequency,
                )
            else:
                forecast = model.predict(origin.horizon, plan.context(origin))
            done = time.perf_counter()
        except (UnsupportedCapabilityError, CandidateFitError, InsufficientHistoryError) as error:
            return error
        except AutoHybTSError:
            raise  # configuration errors and leakage are never swallowed
        except Exception as error:  # a third-party model outside ForecasterBase
            return error
        return forecast, fitted, done

    def _fit_and_forecast_exog(
        self,
        candidate: CandidateSpec,
        model: BaseForecaster,
        plan: TemporalPlan | CrossValidationPlan,
        origin: Origin,
    ) -> tuple[FloatArray, float, float]:
        """Fit and forecast with the exogenous features of ``candidate``.

        Leading rows where a lagged feature is not defined yet are left out of the model's
        training vector; metrics still use the origin's full training vector (same MASE
        scale as candidates without covariates).
        """

        if self.exog is None:
            raise ConfigurationError("A candidata usa exógenas, mas o experimento não tem X.")
        step = minimum_lag(candidate.variant, origin.horizon)
        design = self.exog.design(candidate.exog, plan.history.timestamps, step)
        first = design.first_complete(origin.train_start)
        end = origin.origin_index + 1
        if first >= end:
            raise InsufficientHistoryError("Nenhuma linha de treino com todas as exógenas.")
        train = plan.history.values[first:end]
        rows = design.rows(first, end + origin.horizon)
        model.fit(train, rows[: train.size])
        fitted = time.perf_counter()
        if candidate.variant == "B":
            forecast = rolling_one_step(
                model,
                train,
                plan.target_values(origin),
                origin_index=origin.origin_index,
                series_id=origin.series_id,
                frequency=plan.history.frequency,
                exog=rows,
                exog_max_ahead=design.max_ahead,
            )
        else:
            context = TemporalContext(
                series_id=origin.series_id,
                frequency=plan.history.frequency,
                origin_index=origin.origin_index,
                origin_time=origin.origin_time,
                observed=train,
                observed_start=first,
                exog=rows,
                exog_ahead=origin.horizon,
                exog_max_ahead=design.max_ahead,
            )
            forecast = model.predict(origin.horizon, context)
        return forecast, fitted, time.perf_counter()

    def _check_variant(self, candidate: CandidateSpec) -> None:
        if candidate.variant != "B":
            return
        for name in _base_names(candidate):
            if not self.registry.entry(name).capabilities.online_state_update:
                raise UnsupportedCapabilityError(
                    f"{name} não incorpora observações sem reajuste; incompatível com a variante B."
                )

    def _check_capabilities(self, candidate: CandidateSpec, horizon: int) -> None:
        """Models whose declared capabilities do not fit this protocol are SKIPPED, not fitted.

        The search is univariate, local and point-valued: a model trained globally on a
        panel, producing quantiles/samples, forecasting several targets or needing
        covariates needs its own contract (plano de ensembles, L02/L03); a model with a
        maximum horizon cannot forecast a longer one.
        """

        for kind, name in candidate.model_names():
            capabilities = self.registry.entry(name, kind).capabilities
            if capabilities.training_scope == "global":
                raise UnsupportedCapabilityError(
                    f"{name} treina globalmente num painel; a busca avalia modelos locais."
                )
            if capabilities.target_mode != "univariate":
                raise UnsupportedCapabilityError(
                    f"{name} é {capabilities.target_mode}; a busca é univariada, sem covariáveis."
                )
            if candidate.exog and kind == "base" and not capabilities.accepts_exog:
                raise UnsupportedCapabilityError(f"{name} não aceita exógenas.")
            if capabilities.output_type != "point":
                raise UnsupportedCapabilityError(
                    f"{name} produz {capabilities.output_type}; a busca compara previsões pontuais."
                )
            step = 1 if candidate.variant == "B" else horizon
            if capabilities.max_horizon is not None and step > capabilities.max_horizon:
                raise UnsupportedCapabilityError(
                    f"{name} prevê no máximo {capabilities.max_horizon} passos; pedido {step}."
                )

    def protocol(self) -> dict[str, Any]:
        """Settings shared by every trial; part of each trial key."""

        protocol: dict[str, Any] = {
            "metrics": list(self.metrics),
            "seasonal_period": self.seasonal_period,
            "residuals": self.residual_settings.describe(),
        }
        external = self.registry.external()
        if external:  # only when present, so keys of built-in models do not change
            protocol["external_models"] = external
        if self.exog is not None:  # likewise: experiments without covariates keep their keys
            protocol["exog"] = self.exog.describe()
        return protocol


def _reason(error: BaseException) -> str:
    """Type and message only: no traceback, so no local paths or data leak into logs."""

    text = f"{type(error).__name__}: {error}"
    return text if len(text) <= _MAX_REASON else text[: _MAX_REASON - 3] + "..."


def _diagnostics(model: object) -> dict[str, float]:
    method = getattr(model, "diagnostics", None)
    return {str(k): float(v) for k, v in method().items()} if callable(method) else {}


def _base_names(candidate: CandidateSpec) -> tuple[str, ...]:
    """Registered bases a candidate uses, components of compositions included."""

    return tuple(name for kind, name in candidate.model_names() if kind == "base")


def _corrector_names(candidate: CandidateSpec) -> tuple[str, ...]:
    """Registered correctors a candidate uses (extra stages are correctors)."""

    return tuple(name for kind, name in candidate.model_names() if kind == "corrector")
