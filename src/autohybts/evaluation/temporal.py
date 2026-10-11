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
from autohybts.data.panel import PanelPlan
from autohybts.data.splitter import (
    CrossValidationPlan,
    Origin,
    TemporalPlan,
    ensure_not_sealed,
)
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
    """

    registry: ModelRegistry
    metrics: tuple[str, ...] = DEFAULT_METRICS
    seasonal_period: int = 1
    session_seed: int = 42
    residual_settings: ResidualSettings = field(default_factory=ResidualSettings)

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
            return result("SKIPPED", reason=_reason(unsupported))
        train = plan.train_values(origin)
        started = time.perf_counter()
        try:
            self._check_variant(candidate)
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
        except UnsupportedCapabilityError as error:
            return result("SKIPPED", reason=_reason(error))
        except (CandidateFitError, InsufficientHistoryError) as error:
            return result("FAILED", reason=_reason(error))
        except AutoHybTSError:
            raise  # configuration errors and leakage are never swallowed
        except Exception as error:  # a third-party model outside ForecasterBase
            return result("FAILED", reason=_reason(error))

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
        )

    def _check_variant(self, candidate: CandidateSpec) -> None:
        if candidate.variant != "B":
            return
        for name in _base_names(candidate):
            if not self.registry.entry(name).capabilities.online_state_update:
                raise UnsupportedCapabilityError(
                    f"{name} não incorpora observações sem reajuste; incompatível com a variante B."
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
        return protocol


def _reason(error: BaseException) -> str:
    """Type and message only: no traceback, so no local paths or data leak into logs."""

    text = f"{type(error).__name__}: {error}"
    return text if len(text) <= _MAX_REASON else text[: _MAX_REASON - 3] + "..."


def _diagnostics(model: object) -> dict[str, float]:
    method = getattr(model, "diagnostics", None)
    return {str(k): float(v) for k, v in method().items()} if callable(method) else {}


def _base_names(candidate: CandidateSpec) -> tuple[str, ...]:
    """Registered bases a candidate uses (combination members are bases)."""

    if candidate.strategy == "fixed_combination":
        return tuple(spec.name for spec in (candidate.base, *candidate.members))
    return (candidate.base.name,)


def _corrector_names(candidate: CandidateSpec) -> tuple[str, ...]:
    """Registered correctors a candidate uses (extra stages are correctors)."""

    if candidate.corrector is None:
        return ()
    if candidate.strategy in ("residual_multistage", "dynamic_selection", "residual_combination"):
        return tuple(spec.name for spec in (candidate.corrector, *candidate.members))
    return (candidate.corrector.name,)
