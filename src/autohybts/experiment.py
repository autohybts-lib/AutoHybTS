"""Public experiment API in the PyCaret 4 style (ADR-0001).

``AutoHybridExperiment(**params).fit(y)`` validates the series, splits off the sealed
test and freezes the temporal plan. The verbs then return immutable results. There is no
global state: every call goes through this object, and everything that touches the
sealed test goes through ``evaluate_external`` only.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, overload

import pandas as pd

import autohybts
from autohybts.adapters.base import Parametrized
from autohybts.data.panel import PanelFrame, PanelPlan, plan_panel, split_panel_sealed_test
from autohybts.data.profiling import profile_series
from autohybts.data.schema import SeriesFrame
from autohybts.data.splitter import (
    CrossValidationPlan,
    SealedTest,
    TemporalCrossValidation,
    TemporalPlan,
    TemporalSplitter,
)
from autohybts.domain.identity import canonical_json, stable_hash
from autohybts.domain.results import EvaluationResult, Layer
from autohybts.domain.specs import CandidateSpec, ModelSpec
from autohybts.evaluation.metrics import compute_metrics, resolve_metrics
from autohybts.evaluation.temporal import DEFAULT_METRICS, TemporalEvaluator
from autohybts.exceptions import (
    ConfigurationError,
    CoverageError,
    InsufficientHistoryError,
    SealedTestAccessError,
)
from autohybts.forecasting.finalized import FinalizedForecaster, PanelForecaster, refit
from autohybts.forecasting.rolling_one_step_b import rolling_one_step
from autohybts.models.requirements import framework_versions
from autohybts.registry.capabilities import Capabilities
from autohybts.registry.registry import ModelRegistry, default_registry
from autohybts.reporting.comparison import ComparisonResult, diebold_mariano, holm
from autohybts.reporting.evidence import EvidenceBundle, write_evidence
from autohybts.reporting.leaderboard import leaderboard
from autohybts.reporting.plots import PlotArtifact, data_plot, model_plot
from autohybts.residuals.generation import ResidualSettings
from autohybts.results import (
    CompareResult,
    Event,
    ExternalComparison,
    ExternalResult,
    ModelResult,
    NestedCVResult,
    PredictResult,
)
from autohybts.search.budgets import BudgetPreset, SearchGuardConfig
from autohybts.search.runner import SearchEngine
from autohybts.search.spaces import DEFAULT_BASES, SearchSpace
from autohybts.search.tuning import Distribution, Tuner, TuningResult, default_space
from autohybts.selection.incremental import IncrementalSelector
from autohybts.selection.policy import DEFAULT_FAMILY_ORDER, SelectionPolicy
from autohybts.types import Profile, ProtocolVariant

__all__ = ["AutoHybridExperiment", "infer_seasonal_period"]

_INCLUDE: Final = (
    "standalone",
    "residual",
    "multistage",
    "decomposition",
    "combination",
    "dynamic",
    "ensemble",
)
_DEFAULT_INCLUDE: Final = ("standalone", "residual")
# Origins per validation layer of fold='auto': the acceptance rule's calibration
# (D5, ADR-0008) met its error targets with 7 units on one series.
AUTO_FOLD: Final = 7

# Seasonal period by frequency prefix, as commonly used for daily/weekly/monthly data.
_SEASONALITY: Final = {
    "h": 24,
    "H": 24,
    "D": 7,
    "B": 5,
    "W": 52,
    "MS": 12,
    "ME": 12,
    "M": 12,
    "QS": 4,
    "QE": 4,
    "Q": 4,
    "YS": 1,
    "YE": 1,
    "Y": 1,
    "A": 1,
}
# Experiment lifecycle (§9.5): which verbs are allowed in which state.
_STATES: Final = ("CREATED", "VALIDATED", "SELECTED")
LAYERS: Final[tuple[Layer, Layer]] = ("search", "select")


def infer_seasonal_period(frequency: str) -> int:
    """Conventional seasonal period for a pandas frequency alias (1 when unknown)."""

    prefix = frequency.split("-")[0].lstrip("0123456789")
    return _SEASONALITY.get(prefix, 1)


def _initial_champion(compare: CompareResult) -> CandidateSpec:
    """The standalone model the incremental chain started from (its first champion)."""

    selection = compare.selection
    initial = (
        selection.decisions[0].champion_before
        if selection.decisions
        else selection.champion.candidate_id
    )
    return next(r.candidate for r in selection.finalists if r.candidate_id == initial)


class AutoHybridExperiment(Parametrized):
    """Automated comparison of standalone and residual hybrid forecasters.

    Args:
        fh: forecast horizon (steps per origin, and default horizon of predictions).
        session_id: seed of the whole experiment.
        fold: origins in each validation layer (``V_search`` and ``V_select``).
            ``"auto"`` (default) uses 7 when the history allows and otherwise the largest
            number that fits, with a warning: the acceptance rule is calibrated for 7
            units on one series; with fewer it accepts spurious structures
            more often. An integer is used as given.
        fold_strategy: ``"expanding"`` or ``"rolling"`` (needs ``window_size``).
        window_size: training window length of each origin with ``fold_strategy="rolling"``.
        seasonal_period: ``None`` infers it from the frequency (daily: 7, monthly: 12 ...).
        test_size: points sealed at the end of the series; ``None`` means ``fh``, ``0``
            disables the sealed test (operational forecasting).
        profile: ``"quickstart"`` (variability criterion off), ``"standard"``,
            ``"research_strict"`` or ``"exploratory"``.
        variant: forecasting protocol. ``"A"`` (default): each origin forecasts ``fh``
            steps at once, and a corrector's window is fed with its own predicted
            residuals. ``"B"``: rolling one-step forecasts with frozen parameters; each
            observation is revealed only after its forecast and its residual then enters
            the corrector window. Bases that cannot take new observations without
            refitting are skipped in B; ``"dynamic"`` selection exists only in B.
        budget: search preset (``"smoke"``, ``"small"``, ``"medium"``, ``"large"``).
        max_trials: overrides the preset's total number of candidates.
        max_wall_time_s: optional search deadline in seconds (``None``: none, the default).
            Cooperative: no candidate starts after it, a running one finishes; candidates
            kept out are ``STOPPED`` in the ledger. With a deadline the searched set can
            change between machines; budgets by count are the reproducible ones.
        base_models, correctors, residual_windows, alphas, residual_sources: search space
            (``None`` keeps the default search space).
        base_params: grid of each base in stage S0, by model name: a list of parameter
            dicts, one candidate each (e.g. ``{"knn": [{"lags": 7}, {"lags": 14}]}``). A
            base without a grid uses its search defaults.
        corrector_params: alternative settings of each corrector, tried in stage S3
            (refinement) around the best residual structures; S2 uses each corrector's
            defaults.
        max_stages: most correction stages tried when ``compare_models`` includes
            ``"multistage"`` (2 or 3).
        decomposition_methods: decompositions tried when ``compare_models`` includes
            ``"decomposition"``; ``None`` means ``("ma", "stl")`` in variant A and
            ``("ma",)`` in variant B.
        combination_weights: weightings tried when ``compare_models`` includes
            ``"combination"``.
        ensemble_methods: aggregations tried when ``compare_models`` includes
            ``"ensemble"`` (``"equal"``, ``"median"``, ``"trimmed_mean"``,
            ``"inverse_mse"``, ``"nnls"``): ensembles of the best base of each family, of
            the two best residual structures and residual corrections of the equal-weight
            ensemble. Off the default ``include``: see the calibration gate.
        stage_params: ``"shared"`` (extra stages reuse window, alpha and source) or
            ``"per_stage"`` (the search also varies the extra stage's window and alpha).
        significance_alpha: optional extra acceptance predicate: a one-sided paired
            Wilcoxon test on ``V_select`` must give ``p < significance_alpha`` (off by
            default). It needs at least 5 paired units with different errors, all series
            together, for ``0.05``: with 3 origins on one series it can never pass. It
            controls each comparison, not the whole chain (see ``SelectionPolicy``).
        metrics: evaluation metrics; the first one ranks and selects.
        n_jobs: threads used to evaluate candidates in parallel.
        registry: custom model registry (built-ins by default).
    """

    def __init__(
        self,
        fh: int = 14,
        session_id: int = 42,
        fold: int | str = "auto",
        fold_strategy: str = "expanding",
        window_size: int | None = None,
        seasonal_period: int | None = None,
        test_size: int | None = None,
        profile: Profile = "standard",
        variant: ProtocolVariant = "A",
        budget: BudgetPreset = "small",
        max_trials: int | None = None,
        max_wall_time_s: float | None = None,
        base_models: Sequence[str] | None = None,
        correctors: Sequence[str] | None = None,
        residual_windows: Sequence[int] | None = None,
        alphas: Sequence[float] | None = None,
        residual_sources: Sequence[str] | None = None,
        base_params: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        corrector_params: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        max_stages: int = 2,
        decomposition_methods: Sequence[str] | None = None,
        combination_weights: Sequence[str] = ("equal", "inverse_mse"),
        ensemble_methods: Sequence[str] = ("equal", "median", "inverse_mse", "nnls"),
        stage_params: str = "shared",
        significance_alpha: float | None = None,
        metrics: Sequence[str] = DEFAULT_METRICS,
        n_jobs: int = 1,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.fh = fh
        self.session_id = session_id
        self.fold = fold
        self.fold_strategy = fold_strategy
        self.window_size = window_size
        self.seasonal_period = seasonal_period
        self.test_size = test_size
        self.profile = profile
        self.variant = variant
        self.budget = budget
        self.max_trials = max_trials
        self.max_wall_time_s = max_wall_time_s
        self.base_models = base_models
        self.correctors = correctors
        self.residual_windows = residual_windows
        self.alphas = alphas
        self.residual_sources = residual_sources
        self.base_params = base_params
        self.corrector_params = corrector_params
        self.max_stages = max_stages
        self.decomposition_methods = decomposition_methods
        self.combination_weights = combination_weights
        self.ensemble_methods = ensemble_methods
        self.stage_params = stage_params
        self.significance_alpha = significance_alpha
        self.metrics = metrics
        self.n_jobs = n_jobs
        self.registry = registry
        self.state_ = "CREATED"
        self.events_: list[Event] = []

    # ------------------------------------------------------------------ lifecycle
    def fit(
        self,
        y: pd.Series | SeriesFrame | PanelFrame | Mapping[str, pd.Series] | pd.DataFrame,
        *,
        series_col: str | None = None,
        date_col: str | None = None,
        target: str | None = None,
    ) -> AutoHybridExperiment:
        """Validate ``y``, seal the test segment and freeze the temporal plan.

        ``y`` is one series (``pd.Series`` or ``SeriesFrame``) or a panel: a ``PanelFrame``,
        a mapping ``series_id -> pd.Series`` or a long ``DataFrame`` with ``series_col``,
        ``date_col`` and ``target`` named explicitly. In a panel every series keeps its
        own sealed test, origins and MASE scale; candidates are compared on all of them.
        """

        started = time.perf_counter()
        if self.variant not in ("A", "B"):
            raise ConfigurationError(f"variant deve ser 'A' ou 'B'; recebido {self.variant!r}.")
        panel = _as_panel(y, series_col=series_col, date_col=date_col, target=target)
        test_size = self.fh if self.test_size is None else self.test_size
        dev_panel, sealed_map = split_panel_sealed_test(panel, test_size)
        frame = next(iter(panel))
        history = dev_panel[frame.series_id]
        sealed = sealed_map.get(frame.series_id)
        period = self.seasonal_period or infer_seasonal_period(panel.frequency)
        self.fold_ = self._resolve_fold(dev_panel)
        splitter = self._splitter_for(self.fold_)
        self.panel_: PanelFrame = dev_panel
        self._sealed_map: dict[str, SealedTest] = sealed_map
        self._splitter = splitter
        if dev_panel.is_single:
            self.history_: SeriesFrame = history
            self._sealed: SealedTest | None = sealed
            self.plan_: TemporalPlan = splitter.plan(history)
            self.work_plan_: TemporalPlan | PanelPlan = self.plan_
        else:
            self.work_plan_ = plan_panel(splitter, dev_panel)
        self.seasonal_period_ = period
        self.registry_: ModelRegistry = (self.registry or default_registry()).copy()
        self.evaluator_ = TemporalEvaluator(
            self.registry_,
            metrics=tuple(self.metrics),
            seasonal_period=period,
            session_seed=self.session_id,
            residual_settings=ResidualSettings(),
        )
        self.policy_ = (
            SelectionPolicy.quickstart(significance_alpha=self.significance_alpha)
            if self.profile == "quickstart"
            else SelectionPolicy(
                profile=self.profile,
                family_order=DEFAULT_FAMILY_ORDER,
                significance_alpha=self.significance_alpha,
            )
        )
        self._externally_evaluated: set[str] = set()
        # Evaluations already computed, per (candidate_id, layer), reused by plots and tables.
        self._evaluations: dict[tuple[str, str | None], EvaluationResult] = {}
        self.compare_: CompareResult | None = None
        self._external: dict[str, ExternalResult] = {}
        self._cv_evaluations: dict[tuple[str, str], EvaluationResult] = {}
        self.state_ = "VALIDATED"
        n_points = sum(len(frame) for frame in dev_panel)
        self._event(
            "experiment.fitted",
            f"{len(dev_panel)} série(s), {n_points} pontos de desenvolvimento, teste selado de "
            f"{test_size} por série, {len(self.splits)} origens",
            started,
        )
        return self

    @property
    def is_panel(self) -> bool:
        self._require("VALIDATED")
        return not self.panel_.is_single

    @property
    def splits(self) -> pd.DataFrame:
        """Origins of ``V_search`` and ``V_select`` (the sealed test is not listed)."""

        self._require("VALIDATED")
        return self.work_plan_.to_frame()

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self.events_)

    def models(self) -> pd.DataFrame:
        """Catalog of base models and correctors, with families and capabilities."""

        return (self.registry_ if hasattr(self, "registry_") else default_registry()).to_frame()

    # ------------------------------------------------------------------ verbs
    def register_model(
        self,
        name: str,
        factory: Callable[[Mapping[str, Any]], Any] | Any,
        *,
        kind: str = "base",
        family: str = "deep",
        requires: Mapping[str, str] | None = None,
        version: str = "1",
        capabilities: Capabilities | None = None,
        description: str = "",
        default_params: Mapping[str, Any] | None = None,
        tuning_space: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> AutoHybridExperiment:
        """Add a model built outside AutoHybTS (PyTorch, TensorFlow, any library).

        ``factory`` receives a parameter mapping and returns a fresh, unfitted model:
        ``torch_model(...)``/``keras_model(...)`` from ``autohybts.models``, or any
        callable returning a ``BaseForecaster`` (``kind="base"``) or a
        ``ResidualCorrector`` (``kind="corrector"``). A model *instance* is also accepted
        and cloned for every trial. Afterwards the name works everywhere: ``create_model``,
        ``base_models``/``correctors`` of the search, hybrids, ``tune_model``.

        ``requires`` (e.g. ``{"torch": ">=2.2"}``) is checked when the model is built; a
        missing or older library makes the trial ``SKIPPED``. ``version`` names the
        registration and enters every trial key: change it when the model code changes.
        Can be called before or after ``fit``; returns the experiment for chaining.
        """

        if kind not in ("base", "corrector"):
            raise ConfigurationError("kind deve ser 'base' ou 'corrector'.")
        if not callable(factory) or hasattr(factory, "get_params"):
            template = factory
            factory = lambda params: _clone_with(template, params)  # noqa: E731
        options: dict[str, Any] = {
            "description": description,
            "default_params": default_params,
            "replace": replace,
            "requires": requires,
            "version": str(version),
            "tuning_space": tuning_space,
            "capabilities": capabilities,
        }
        targets = [self.registry if self.registry is not None else default_registry()]
        self.registry = targets[0]
        if hasattr(self, "registry_"):
            targets.append(self.registry_)
        for registry in targets:
            if kind == "base":
                registry.register_base(name, factory, family=family, **options)
            else:
                registry.register_corrector(name, factory, **options)
        self._event("model.registered", f"{kind} {name!r} (versão {version})", None)
        return self

    def create_model(self, name: str, **params: Any) -> ModelResult:
        """Evaluate one standalone model on every origin of both layers."""

        self._require("VALIDATED")
        entry = self.registry_.entry(name)
        candidate = CandidateSpec.standalone(
            name, params, family=entry.family, variant=self.variant
        )
        return self._model_result(candidate, "model.created")

    def create_ensemble(
        self,
        models: Sequence[str | CandidateSpec | ModelResult],
        method: str = "equal",
        **options: Any,
    ) -> ModelResult:
        """Evaluate one ensemble of complete forecasters on every origin.

        ``models`` are names (default parameters of this experiment), specs or
        ``ModelResult`` — standalone models, hybrids or other ensembles. ``method`` is
        ``"equal"``, ``"median"``, ``"trimmed_mean"`` (``trim=``), ``"inverse_mse"`` or
        ``"nnls"`` (``shrinkage=`` towards equal weights); learned weights come from
        out-of-fold forecasts inside each origin's training vector.
        """

        self._require("VALIDATED")
        components = self._candidates(models)
        candidate = CandidateSpec.ensemble(components, method=method, **options)
        if candidate.variant != self.variant:
            raise ConfigurationError(
                f"Componentes da variante {candidate.variant}; o experimento é {self.variant}."
            )
        return self._model_result(candidate, "ensemble.created")

    def create_hybrid(
        self,
        base: str | CandidateSpec | ModelResult,
        corrector: str,
        window: int = 7,
        alpha: float = 1.0,
        residual_source: str = "historical_oof_1step",
        base_params: dict[str, Any] | None = None,
        corrector_params: dict[str, Any] | None = None,
        extra_stages: Sequence[str] = (),
        stage_params: Sequence[Mapping[str, Any]] | None = None,
        combine_correctors: Sequence[str] = (),
        combination_weights: str = "inverse_mse",
    ) -> ModelResult:
        """Evaluate one residual hybrid ``base + alpha * corrector`` on every origin.

        ``extra_stages`` names correctors of further stages: each one corrects the
        residuals of the model before it (``residual_multistage``). By default the extra
        stages share ``window``, ``alpha`` and ``residual_source``; ``stage_params`` gives
        one dict per extra stage to change them (e.g. ``[{"window_size": 28}]``).

        ``combine_correctors`` instead averages ``corrector`` and these correctors with
        fixed ``combination_weights`` (``"equal"`` or ``"inverse_mse"``) as one correction.

        ``base`` may also be a composed candidate (e.g. ``create_ensemble(...)``): the
        corrector then learns the out-of-fold residuals of that composition
        (``residual_composite``); stages and corrector combinations are for named bases.
        """

        self._require("VALIDATED")
        if not isinstance(base, str):
            spec = base.candidate if isinstance(base, ModelResult) else base
            if spec.strategy == "standalone":  # a single model: the usual residual hybrid
                return self.create_hybrid(
                    spec.base.name,
                    corrector,
                    window,
                    alpha,
                    residual_source,
                    spec.base.params,
                    corrector_params,
                    extra_stages,
                    stage_params,
                    combine_correctors,
                    combination_weights,
                )
            if extra_stages or combine_correctors or base_params:
                raise ConfigurationError(
                    "Com uma base composta, use só corrector, window, alpha e residual_source."
                )
            candidate = CandidateSpec.corrected(
                spec, ModelSpec.of(corrector, corrector_params), window, alpha, residual_source
            )
            return self._model_result(candidate, "hybrid.created")
        if extra_stages and combine_correctors:
            raise ConfigurationError("Use extra_stages ou combine_correctors, não os dois.")
        if stage_params is not None and not extra_stages:
            raise ConfigurationError("stage_params vale só com extra_stages.")
        options: dict[str, Any] = {}
        strategy = "residual_additive"
        members: tuple[str, ...] = ()
        if extra_stages:
            strategy, members = "residual_multistage", tuple(extra_stages)
            if stage_params is not None:
                options["stages"] = [dict(stage) for stage in stage_params]
        elif combine_correctors:
            strategy, members = "residual_combination", tuple(combine_correctors)
            options["weights"] = combination_weights
        candidate = CandidateSpec(
            "combination" if strategy == "residual_combination" else "residual",
            strategy,
            ModelSpec.of(base, base_params),
            ModelSpec.of(corrector, corrector_params),
            window,
            float(alpha),
            residual_source,
            variant=self.variant,
            members=tuple(ModelSpec.of(name) for name in members),
            options_json=canonical_json(options),
        )
        return self._model_result(candidate, "hybrid.created")

    def create_decomposition(
        self,
        base: str,
        corrector: str,
        method: str = "ma",
        window: int = 7,
        ma_window: int | None = None,
        base_params: dict[str, Any] | None = None,
        corrector_params: dict[str, Any] | None = None,
    ) -> ModelResult:
        """Evaluate ``base`` on the smooth part plus ``corrector`` on the volatile part.

        ``method="ma"`` is a causal moving average (window by kurtosis on each training
        vector when ``ma_window`` is ``None``); ``"stl"`` is variant A only.
        """

        self._require("VALIDATED")
        options: dict[str, Any] = {"method": method, "ma_window": ma_window}
        if method == "stl":
            options["period"] = self.seasonal_period_
        candidate = CandidateSpec(
            "decomposition",
            "decomposition",
            ModelSpec.of(
                base, base_params or self._space(with_residuals=False).default_params(base)
            ),
            ModelSpec.of(corrector, corrector_params),
            window,
            variant=self.variant,
            options_json=canonical_json({k: v for k, v in options.items() if v is not None}),
        )
        return self._model_result(candidate, "decomposition.created")

    def create_combination(
        self, models: Sequence[str], weights: str = "inverse_mse"
    ) -> ModelResult:
        """Evaluate a fixed combination of base models (weights from the training only)."""

        self._require("VALIDATED")
        if len(models) < 2:
            raise ConfigurationError("Informe ao menos dois modelos para combinar.")
        space = self._space(with_residuals=False)
        specs = [ModelSpec.of(name, space.default_params(name)) for name in models]
        for name in models:
            self.registry_.entry(name)
        candidate = CandidateSpec(
            "combination",
            "fixed_combination",
            specs[0],
            variant=self.variant,
            members=tuple(specs[1:]),
            options_json=canonical_json({"weights": weights}),
        )
        return self._model_result(candidate, "combination.created")

    @overload
    def tune_model(
        self,
        model: str | CandidateSpec | ModelResult,
        n_trials: int = ...,
        search_space: Mapping[str, Distribution | Sequence[Any]] | None = ...,
        sampler: str = ...,
        return_tuner: Literal[False] = ...,
    ) -> ModelResult: ...

    @overload
    def tune_model(
        self,
        model: str | CandidateSpec | ModelResult,
        n_trials: int = ...,
        search_space: Mapping[str, Distribution | Sequence[Any]] | None = ...,
        sampler: str = ...,
        *,
        return_tuner: Literal[True],
    ) -> tuple[ModelResult, TuningResult]: ...

    def tune_model(
        self,
        model: str | CandidateSpec | ModelResult,
        n_trials: int = 20,
        search_space: Mapping[str, Distribution | Sequence[Any]] | None = None,
        sampler: str = "auto",
        return_tuner: bool = False,
    ) -> ModelResult | tuple[ModelResult, TuningResult]:
        """Tune the parameters of the model's base on ``V_search`` only.

        ``search_space`` maps parameter names (dotted for nested ones, e.g.
        ``"estimator_params.C"``) to a ``Distribution``, a ``(low, high)`` pair (integers
        give an integer range) or a list of choices; ``None`` uses the built-in space of
        the base. ``sampler`` is ``"auto"`` (Optuna if installed, else seeded random
        search), ``"optuna"`` or ``"random"``. The tuned candidate is then evaluated on
        both layers like ``create_model``: its ``V_select`` score was never seen by the
        tuner. ``return_tuner=True`` also returns every trial.
        """

        self._require("VALIDATED")
        candidate = self._candidates([model])[0]
        space = (
            self._registered_space(candidate.base.name)
            if search_space is None
            else {name: _as_distribution(name, value) for name, value in search_space.items()}
        )
        started = time.perf_counter()
        tuner = Tuner(
            self.evaluator_,
            n_trials=n_trials,
            sampler=sampler,  # type: ignore[arg-type]
            session_seed=self.session_id,
        )
        tuning = tuner.tune(candidate, self.work_plan_, space)
        self._event(
            "model.tuned",
            f"{len(tuning.trials)} configurações ({tuning.sampler}); melhor: {tuning.best.label}",
            started,
        )
        result = self._model_result(tuning.best, "model.created")
        return (result, tuning) if return_tuner else result

    def compare_models(
        self, include: Sequence[str] = _DEFAULT_INCLUDE, sort: str | None = None
    ) -> CompareResult:
        """Budgeted search on ``V_search``, then incremental selection on ``V_select``.

        ``include`` picks the families: ``"standalone"`` (required: the initial champion
        is a single model), ``"residual"``, ``"multistage"`` (extra correction stages on
        the best residual structures; needs ``"residual"``), ``"decomposition"`` and
        ``"combination"``, ``"ensemble"`` (ensembles of bases, of residual structures and
        corrected ensembles; ``ensemble_methods``) and, in variant B, ``"dynamic"``
        (dynamic selection of the corrector, ``dref`` and ``local``).
        """

        self._require("VALIDATED")
        include = self._check_include(include)
        started = time.perf_counter()
        space = self._space(with_residuals="residual" in include, include=include)
        guard = self._guard()
        search = SearchEngine(self.evaluator_, space, guard, max_workers=self.n_jobs).run(
            self.work_plan_
        )
        self._event(
            "search.finished", f"{search.trial_count} tentativas ({search.stop_reason})", started
        )
        started = time.perf_counter()
        self.policy_ = replace(self.policy_, family_order=self._family_order(include))
        selection = IncrementalSelector(self.evaluator_, self.policy_).select(
            search, self.work_plan_
        )
        champion = selection.champion
        self._event("selection.finished", f"campeão: {champion.label}", started)
        self.state_ = "SELECTED"
        self.include_: tuple[str, ...] = include
        for evaluated in search.results:
            self._evaluations[(evaluated.candidate_id, "search")] = evaluated
        for evaluated in selection.finalists:
            self._evaluations[(evaluated.candidate_id, "select")] = evaluated
        sort = sort or self.evaluator_.metrics[0]
        self.compare_ = CompareResult(
            best=champion,
            leaderboard=leaderboard(search.results, sort, champion_id=champion.candidate_id),
            selection_leaderboard=leaderboard(
                selection.finalists, sort, champion_id=champion.candidate_id
            ),
            decisions=selection.decisions_frame(),
            audit=selection.audit,
            search=search,
            selection=selection,
            snapshot={
                **self._snapshot(space, guard, search.protocol, champion),
                "include": list(include),
            },
            comparison=self._finalists_comparison(selection.finalists, champion),
            events=self.events,
        )
        return self.compare_

    def finalize_model(
        self, candidate: CandidateSpec | ModelResult
    ) -> FinalizedForecaster | PanelForecaster:
        """Refit the frozen candidate on the history before the sealed test.

        In a panel each series is refitted with the configuration the selection chose for
        it (the champion's best configuration per series) and a ``PanelForecaster`` is
        returned.
        """

        self._require("VALIDATED")
        spec = candidate.candidate if isinstance(candidate, ModelResult) else candidate
        if not isinstance(spec, CandidateSpec):
            raise ConfigurationError("finalize_model espera uma CandidateSpec ou um ModelResult.")
        if not self.is_panel:
            return self._finalize_series(spec, self.history_, self._sealed)
        per_series = dict.fromkeys(self.panel_.series_ids, spec)
        if self.compare_ is not None and spec == self.compare_.best:
            per_series.update(self.compare_.selection.champion_per_series)
        models = {
            sid: self._finalize_series(per_series[sid], self.panel_[sid], self._sealed_map.get(sid))
            for sid in self.panel_.series_ids
        }
        manifest = {
            "model_id": stable_hash({sid: model.model_id for sid, model in models.items()}),
            "spec": {
                "candidate_id": spec.candidate_id,
                "label": spec.label,
                "describe": spec.describe(),
            },
            "series": {sid: dict(model.manifest) for sid, model in models.items()},
            "panel_fingerprint": self.panel_.fingerprint,
            "plan_fingerprint": self.work_plan_.fingerprint,
            "autohybts_version": autohybts.__version__,
        }
        return PanelForecaster(models, spec, manifest)

    def _finalize_series(
        self, spec: CandidateSpec, history: SeriesFrame, sealed: SealedTest | None
    ) -> FinalizedForecaster:
        started = time.perf_counter()
        model, seed = refit(
            spec,
            history,
            self.registry_,
            session_seed=self.session_id,
            residual_settings=self.evaluator_.residual_settings,
        )
        manifest = {
            "model_id": stable_hash(
                {"spec": spec.candidate_id, "data": history.fingerprint, "seed": seed}
            ),
            "spec": {
                "candidate_id": spec.candidate_id,
                "label": spec.label,
                "describe": spec.describe(),
            },
            "training_cutoff": str(history.end),
            "series_id": history.series_id,
            "frequency": history.frequency,
            "n_observations": len(history),
            "history_fingerprint": history.fingerprint,
            "plan_fingerprint": self.work_plan_.fingerprint,
            "seed": seed,
            "experiment": self._public_params(),
            "sealed_test": None
            if sealed is None
            else {"size": sealed.size, "fingerprint": sealed.fingerprint},
            "autohybts_version": autohybts.__version__,
        }
        external = self._external_requirements(spec)
        if external:
            manifest["frameworks"] = framework_versions(external)
        final = FinalizedForecaster(spec, model, history, seed, manifest, len(history))
        self._event("model.finalized", f"{spec.label} reajustado até {history.end}", started)
        return final

    def predict_model(
        self, model: FinalizedForecaster | PanelForecaster, fh: int | None = None
    ) -> PredictResult:
        """Forecast from the end of the model's training history; never refits."""

        if not isinstance(model, FinalizedForecaster | PanelForecaster):
            raise ConfigurationError(
                "predict_model espera um modelo finalizado (use finalize_model antes)."
            )
        steps = self.fh if fh is None else fh
        if model.variant == "B" and steps != 1:
            raise ConfigurationError(
                "Um modelo da variante B prevê um passo por vez; use fh=1 ou model.stream() "
                "para prever enquanto as observações chegam."
            )
        predictions = model.predict(steps)
        self._event("model.predicted", f"{len(predictions)} passos", None)
        return PredictResult(predictions, model.spec.label)

    def evaluate_external(
        self,
        model: FinalizedForecaster | PanelForecaster,
        y_test: pd.Series | SeriesFrame | None = None,
    ) -> ExternalResult:
        """Terminal evaluation, once per finalized model; the result never reaches the search.

        Without ``y_test`` the sealed test of this experiment is used. An explicit
        ``y_test`` (e.g. an official benchmark test) must start right after the model's
        training cutoff.
        """

        self._require("VALIDATED")
        if isinstance(model, PanelForecaster):
            return self._evaluate_external_panel(model)
        if not isinstance(model, FinalizedForecaster):
            raise ConfigurationError("evaluate_external espera um modelo finalizado.")
        if model.model_id in self._externally_evaluated:
            raise SealedTestAccessError(
                "Este modelo já foi avaliado no teste externo; repetir a avaliação transformaria "
                "o teste em validação."
            )
        if y_test is None:
            sealed = self._sealed_map.get(model.history.series_id)
            if sealed is None:
                raise ConfigurationError(
                    "Sem teste selado (test_size=0): passe y_test para a avaliação externa."
                )
            test = sealed._unseal()
        else:
            test = y_test if isinstance(y_test, SeriesFrame) else _as_frame(y_test)
        expected_start = pd.date_range(
            model.training_cutoff, periods=2, freq=model.history.frequency
        )[1]
        if test.start != expected_start or test.frequency != model.history.frequency:
            raise ConfigurationError(
                f"O teste deve começar em {expected_start} com frequência "
                f"{model.history.frequency!r}; começa em {test.start} ({test.frequency!r})."
            )
        self._externally_evaluated.add(model.model_id)
        started = time.perf_counter()
        if model.variant == "B":
            predictions = _rolling_table(model, test)
        else:
            predictions = model.predict(len(test))
        predictions["y_true"] = test.values
        values = compute_metrics(
            resolve_metrics(self.evaluator_.metrics),
            test.values,
            predictions["y_pred"].to_numpy(),
            model.history.values,
            self.seasonal_period_,
        )
        metrics = pd.DataFrame([{"model": model.spec.label, **values}]).set_index("model")
        self._event("model.evaluated_external", f"{len(test)} pontos", started)
        external = ExternalResult(metrics, predictions, model.spec.label)
        if y_test is None:
            self._external[model.model_id] = external
        return external

    # ------------------------------------------------------------------ evidence
    def data_profile(self) -> pd.DataFrame:
        """Data quality report of the development history (the sealed test is excluded).

        One column per series in a panel.
        """

        self._require("VALIDATED")
        if not self.is_panel:
            return profile_series(self.history_)
        return pd.concat(
            {sid: profile_series(frame)["value"] for sid, frame in self.panel_.series.items()},
            axis=1,
        )

    def plot_data(self, plot: str = "series", series_id: str | None = None) -> PlotArtifact:
        """``series``, ``acf``, ``seasonality`` or ``distribution`` of the development history."""

        self._require("VALIDATED")
        frame = self._series(series_id)
        sealed = self._sealed_map.get(frame.series_id)
        size = 0 if sealed is None else sealed.size
        return data_plot(plot, frame, self.seasonal_period_, size)

    def plot_model(
        self,
        candidate: CandidateSpec | ModelResult,
        plot: str = "cv_forecasts",
        series_id: str | None = None,
    ) -> PlotArtifact:
        """Validation diagnostics of a candidate; never uses the sealed test.

        In a panel the forecast plots show one series (``series_id``, default the first).
        """

        self._require("VALIDATED")
        spec = candidate.candidate if isinstance(candidate, ModelResult) else candidate
        frame = self._series(series_id)
        evaluation = self._evaluation(spec)
        if self.is_panel:
            evaluation = evaluation.for_series(frame.series_id)
        selection = None if self.compare_ is None else self.compare_.selection
        search_scores: dict[str, float] = {}
        if self.compare_ is not None:
            metric = self.evaluator_.metrics[0]
            search_scores = {r.candidate_id: r.mean(metric) for r in self.compare_.search.results}
        return model_plot(plot, spec, evaluation, frame, selection, search_scores)

    def get_metrics(self, candidate: CandidateSpec | ModelResult | None = None) -> pd.DataFrame:
        """Per-origin metrics of ``candidate``, or a per-layer summary of everything evaluated."""

        self._require("VALIDATED")
        if candidate is not None:
            spec = candidate.candidate if isinstance(candidate, ModelResult) else candidate
            return self._evaluation(spec).to_frame()
        metrics = list(self.evaluator_.metrics)
        columns = ["candidate_id", "model", "layer", *metrics, "n_units", "coverage"]
        rows = [
            {
                "candidate_id": candidate_id,
                "model": result.candidate.label,
                "layer": layer,
                **{m: result.mean(m) for m in metrics},
                "n_units": result.n_expected,
                "coverage": result.coverage(metrics[0]),
            }
            for (candidate_id, layer), result in sorted(
                self._evaluations.items(), key=lambda item: (item[0][0], str(item[0][1]))
            )
            if layer is not None
        ]
        return pd.DataFrame(rows, columns=columns)

    def export_evidence(self, path: str | Path, plots: Sequence[str] = ()) -> EvidenceBundle:
        """Tables, decisions, splits and manifest of the last ``compare_models``.

        Raw series values are not written. ``plots`` adds model plots of the champion
        (e.g. ``("selection_trace", "cv_forecasts")``).
        """

        self._require("SELECTED")
        compare = self.compare_
        if compare is None:
            raise ConfigurationError("Chame compare_models antes de export_evidence.")
        figures = [self.plot_model(compare.best, kind) for kind in plots]
        warnings = []
        if self.policy_.variability is None:
            warnings.append("critério de variabilidade desligado (perfil quickstart)")
        if self.policy_.exploratory:
            warnings.append("resultados exploratórios: não são avaliação imparcial")
        gap = compare.audit.selection_gap
        if gap > 0:
            warnings.append(f"o vencedor piorou da busca para a seleção (gap {gap:+.3f})")
        return write_evidence(
            path,
            experiment_id=compare.audit.protocol_hash,
            tables={
                "metrics": self.get_metrics(),
                "leaderboard": compare.leaderboard,
                "selection_leaderboard": compare.selection_leaderboard,
                "trials": compare.search.ledger_frame(),
            },
            documents={
                "selection_history": [asdict(d) for d in compare.selection.decisions],
                "splits": self.splits.reset_index().to_dict(orient="records"),
                "snapshot": compare.snapshot,
                "audit": {**asdict(compare.audit), "selection_gap": gap},
            },
            figures=figures,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ comparison
    def cross_validate(
        self,
        models: Sequence[str | CandidateSpec | ModelResult] | None = None,
        folds: int = 5,
        fold_strategy: str | None = None,
        step: int | None = None,
        window_size: int | None = None,
        horizon: int | None = None,
        reference: int | str = 0,
    ) -> ComparisonResult:
        """Rolling-origin temporal cross-validation of several models on the same folds.

        Runs on the development history only (never the sealed test) with its own folds,
        independent of ``V_search``/``V_select``. ``models`` accepts names (default
        parameters of this experiment), specs or ``ModelResult``; by default the
        finalists of ``compare_models`` (champion first) or every configured base model.
        Use the result for reporting; selection keeps its own protocol.
        """

        self._require("VALIDATED")
        splitter = TemporalCrossValidation(
            horizon=horizon or self.fh,
            n_folds=folds,
            step=step,
            fold_strategy=(fold_strategy or self.fold_strategy),  # type: ignore[arg-type]
            window_size=window_size if window_size is not None else self.window_size,
        )
        plan: CrossValidationPlan | PanelPlan = (
            plan_panel(splitter, self.panel_) if self.is_panel else splitter.split(self.history_)
        )
        started = time.perf_counter()
        candidates = self._candidates(models)
        results = []
        for candidate in candidates:
            key = (candidate.candidate_id, plan.fingerprint)
            if key not in self._cv_evaluations:
                self._cv_evaluations[key] = self.evaluator_.evaluate(candidate, plan)
            results.append(self._cv_evaluations[key])
        self._event("cv.finished", f"{len(candidates)} modelos x {folds} dobras", started)
        step_used = splitter.effective_step
        return ComparisonResult(
            tuple(results),
            self.evaluator_.metrics[0],
            self._series(None),
            f"cv {folds} dobras ({splitter.fold_strategy}, passo {step_used})",
            reference=self._reference_index(candidates, reference),
            max_lag=max(0, math.ceil(splitter.horizon / step_used) - 1),
            splits=plan.to_frame(),
        )

    def compare_external(
        self,
        models: Sequence[str | CandidateSpec | ModelResult] | None = None,
        reference: int | str = 0,
    ) -> ExternalComparison:
        """Terminal comparison on the sealed test, at most one evaluation per model.

        Each model is finalized (refit on the pre-test history) and evaluated once; a model
        already evaluated reuses its result. Report it; do not go back and change the
        configuration because of it, or the test becomes validation.
        """

        self._require("VALIDATED")
        if not self._sealed_map:
            raise ConfigurationError("Sem teste selado (test_size=0): nada para comparar.")
        candidates = self._candidates(models)
        externals = []
        for candidate in candidates:
            final = self.finalize_model(candidate)
            stored = self._external.get(final.model_id)
            externals.append(stored or self.evaluate_external(final))
        ref = self._reference_index(candidates, reference)
        metric = self.evaluator_.metrics[0]
        # In a panel each result has one row per series plus the mean; compare the means.
        table = pd.concat([e.metrics.tail(1) for e in externals])
        table[f"ganho_{metric}_vs_ref"] = 1 - table[metric] / table[metric].iloc[ref]
        steps = pd.DataFrame(
            {
                e.model_label: (e.predictions["y_true"] - e.predictions["y_pred"])
                .abs()
                .reset_index(drop=True)
                for e in externals
            }
        )
        rows = []
        reference_label = externals[ref].model_label
        for external in externals:
            if external.model_label == reference_label:
                continue
            stat, p_value = diebold_mariano(
                steps[reference_label].to_numpy(),
                steps[external.model_label].to_numpy(),
                max_lag=max(0, len(steps) - 1) if self.variant == "A" else 0,
            )
            rows.append({"model": external.model_label, "dm_statistic": stat, "dm_p": p_value})
        tests = pd.DataFrame(rows, columns=["model", "dm_statistic", "dm_p"])
        if not tests.empty:
            tests["dm_p_holm"] = holm(tests["dm_p"].tolist())
        return ExternalComparison(table, steps, tests, reference_label)

    def nested_cross_validate(
        self,
        outer_folds: int = 3,
        baselines: Sequence[str] = ("naive", "seasonal_naive"),
        budget: BudgetPreset | None = None,
        include: Sequence[str] | None = None,
    ) -> NestedCVResult:
        """Estimate how the whole automated procedure generalizes.

        For each outer origin (the last ``outer_folds`` blocks of ``fh`` points of the
        development history), a fresh experiment with these parameters searches and selects
        using only the history before that origin; its champion is refitted there and
        scored on the next block, side by side with fixed ``baselines`` and with
        ``best_standalone``: the best standalone model *of that inner selection* (its
        initial champion), chosen without the outer block.

        ``include`` are the families of the inner ``compare_models``; ``None`` repeats the
        ones of this experiment's last ``compare_models`` (the procedure being estimated),
        or its default when it has not run. ``result.cost`` has the work done per outer
        fold (candidates, units, seconds), so the procedure's cost can be reconstructed.

        In a panel every series is cut at its own outer origin, the inner experiment runs
        on the whole truncated panel and the folds table has one row per fold and series;
        the summary averages over all of them. The inner reference then uses the
        representative configuration of the standalone finalist in every series.
        """

        self._require("VALIDATED")
        families = self._check_include(
            include if include is not None else getattr(self, "include_", _DEFAULT_INCLUDE)
        )
        outer = TemporalCrossValidation(horizon=self.fh, n_folds=outer_folds, step=self.fh)
        outer_plan = plan_panel(outer, self.panel_)
        params = {**self._public_params(), "test_size": 0, "registry": self.registry}
        if budget is not None:
            params["budget"] = budget
        metrics = resolve_metrics(self.evaluator_.metrics)
        space = self._space(with_residuals=False)
        rows = []
        costs = []
        started = time.perf_counter()
        series_ids = self.panel_.series_ids
        fold_ids = [o.fold_id for o in outer_plan[series_ids[0]].origins()]
        for fold_id in fold_ids:
            # Each series is cut at its own origin of this outer fold.
            origins = {
                sid: next(o for o in outer_plan[sid].origins() if o.fold_id == fold_id)
                for sid in series_ids
            }
            inner = {
                sid: self.panel_[sid].head(origins[sid].origin_index + 1) for sid in series_ids
            }
            base_row: dict[str, Any] = {"fold_id": fold_id}
            cost: dict[str, Any] = {"fold_id": fold_id, "status": "SUCCEEDED"}
            fold_started = time.perf_counter()
            try:
                inner_input = inner[series_ids[0]] if not self.is_panel else PanelFrame(inner)
                inner_exp = AutoHybridExperiment(**params).fit(inner_input)
                inner_compare = inner_exp.compare_models(include=families)
                compared = time.perf_counter()
                final = inner_exp.finalize_model(inner_compare.best)
                reference = inner_exp.finalize_model(_initial_champion(inner_compare))
                finalized = time.perf_counter()
            except (InsufficientHistoryError, CoverageError) as error:
                for sid in series_ids:
                    rows.append(
                        {
                            **base_row,
                            "series_id": sid,
                            "origin": origins[sid].origin_time,
                            "status": "FAILED",
                            "reason": f"{type(error).__name__}: {error}",
                        }
                    )
                cost.update(status="FAILED", seconds_total=time.perf_counter() - fold_started)
                costs.append(cost)
                continue
            search = inner_compare.search
            finalists = inner_compare.selection.finalists
            cost.update(
                {
                    "search_candidates": search.trial_count,
                    "search_units": search.unit_count,
                    "search_fits": search.fit_count,
                    "cached_candidates": search.cached_count,
                    "max_trials_total": inner_compare.snapshot["guard"]["max_trials_total"],
                    "stop_reason": search.stop_reason,
                    "select_candidates": len(finalists),
                    "select_units": sum(len(r.trials) for r in finalists),
                    "select_fits": sum(t.n_fits or 0 for r in finalists for t in r.trials),
                    "seconds_search": search.elapsed_s,
                    # validation of the inner history, selection and bookkeeping
                    "seconds_selection": max(0.0, compared - fold_started - search.elapsed_s),
                    "seconds_refit": finalized - compared,
                }
            )
            for sid in series_ids:
                history = inner[sid]
                actual = outer_plan[sid].target_values(origins[sid])
                member = final[sid] if isinstance(final, PanelForecaster) else final
                forecast = member.predict(self.fh)["y_pred"].to_numpy()
                scores = compute_metrics(
                    metrics, actual, forecast, history.values, self.seasonal_period_
                )
                standalone = reference[sid] if isinstance(reference, PanelForecaster) else reference
                reference_scores = compute_metrics(
                    metrics,
                    actual,
                    standalone.predict(self.fh)["y_pred"].to_numpy(),
                    history.values,
                    self.seasonal_period_,
                )
                row = {
                    **base_row,
                    "series_id": sid,
                    "origin": origins[sid].origin_time,
                    "status": "SUCCEEDED",
                    "champion": member.spec.label,
                    "strategy": member.spec.strategy,
                    "best_standalone": standalone.spec.label,
                    "inner_selection_gap": inner_compare.audit.selection_gap,
                    **{f"autohybts_{m}": v for m, v in scores.items()},
                    **{f"best_standalone_{m}": v for m, v in reference_scores.items()},
                }
                for name in baselines:
                    spec = CandidateSpec.standalone(
                        name, space.default_params(name), family=self.registry_.entry(name).family
                    )
                    model, _ = refit(
                        spec,
                        history,
                        self.registry_,
                        session_seed=self.session_id,
                        residual_settings=self.evaluator_.residual_settings,
                    )
                    baseline = FinalizedForecaster(spec, model, history, None, {}, len(history))
                    predicted = baseline.predict(self.fh)["y_pred"].to_numpy()
                    baseline_scores = compute_metrics(
                        metrics, actual, predicted, history.values, self.seasonal_period_
                    )
                    row.update({f"{name}_{m}": v for m, v in baseline_scores.items()})
                rows.append(row)
            cost["seconds_total"] = time.perf_counter() - fold_started
            costs.append(cost)
        self._event("nested_cv.finished", f"{outer_folds} dobras externas", started)
        index = ["fold_id", "series_id"] if self.is_panel else ["fold_id"]
        folds_table = pd.DataFrame(rows)
        if not self.is_panel:
            folds_table = folds_table.drop(columns="series_id")
        folds_table = folds_table.set_index(index)
        metric = self.evaluator_.metrics[0]
        columns = [
            f"autohybts_{metric}",
            f"best_standalone_{metric}",
            *(f"{b}_{metric}" for b in baselines),
        ]
        present = [c for c in columns if c in folds_table.columns]
        summary = pd.DataFrame(
            {
                "mean": folds_table[present].mean(),
                "std": folds_table[present].std(ddof=1),
                "n_folds": folds_table[present].notna().sum(),
            }
        )
        if f"autohybts_{metric}" in folds_table.columns:
            auto = folds_table[f"autohybts_{metric}"]
            summary["autohybts_wins"] = [
                float((auto < folds_table[c]).mean()) if c != auto.name else float("nan")
                for c in present
            ]
        cost_table = pd.DataFrame(costs).set_index("fold_id")
        return NestedCVResult(folds_table, summary, metric, families, cost_table)

    def _finalists_comparison(
        self, finalists: Sequence[EvaluationResult], champion: CandidateSpec
    ) -> ComparisonResult:
        ordered = sorted(finalists, key=lambda r: r.candidate_id != champion.candidate_id)
        return ComparisonResult(
            tuple(ordered),
            self.evaluator_.metrics[0],
            self._series(None),
            "V_select (finalistas)",
            reference=0,
            max_lag=max(0, math.ceil(self.fh / self._splitter.effective_step) - 1),
        )

    def _candidates(
        self, models: Sequence[str | CandidateSpec | ModelResult] | None
    ) -> list[CandidateSpec]:
        space = self._space(with_residuals=False)
        if models is None:
            comparison = None if self.compare_ is None else self.compare_.comparison
            if comparison is not None:  # finalists of compare_models, champion first
                return [result.candidate for result in comparison.results]
            models = list(self.base_models or DEFAULT_BASES)
        candidates = []
        for model in models:
            if isinstance(model, ModelResult):
                candidates.append(model.candidate)
            elif isinstance(model, CandidateSpec):
                candidates.append(model)
            elif isinstance(model, str):
                entry = self.registry_.entry(model)
                candidates.append(
                    CandidateSpec.standalone(
                        model,
                        space.default_params(model),
                        family=entry.family,
                        variant=self.variant,
                    )
                )
            else:
                raise ConfigurationError(f"Modelo não reconhecido: {model!r}.")
        if not candidates:
            raise ConfigurationError("Informe ao menos um modelo.")
        return candidates

    @staticmethod
    def _reference_index(candidates: Sequence[CandidateSpec], reference: int | str) -> int:
        if isinstance(reference, int):
            if not 0 <= reference < len(candidates):
                raise ConfigurationError(f"Referência {reference} fora da lista de modelos.")
            return reference
        for index, candidate in enumerate(candidates):
            if reference in (candidate.label, candidate.base.name, candidate.candidate_id):
                return index
        raise ConfigurationError(f"Referência {reference!r} não está entre os modelos comparados.")

    # ------------------------------------------------------------------ helpers
    def _splitter_for(self, fold: int) -> TemporalSplitter:
        return TemporalSplitter(
            horizon=self.fh,
            n_search_folds=fold,
            n_select_folds=fold,
            fold_strategy=self.fold_strategy,  # type: ignore[arg-type]
            window_size=self.window_size,
        )

    def _resolve_fold(self, panel: PanelFrame) -> int:
        """``fold`` as given, or for ``"auto"`` the most origins up to ``AUTO_FOLD`` that fit."""

        if self.fold != "auto":
            if isinstance(self.fold, bool) or not isinstance(self.fold, int) or self.fold < 1:
                raise ConfigurationError(
                    f"fold deve ser inteiro >= 1 ou 'auto'; recebido {self.fold!r}."
                )
            return self.fold
        error: InsufficientHistoryError | None = None
        for fold in range(AUTO_FOLD, 0, -1):
            splitter = self._splitter_for(fold)
            try:
                for frame in panel:
                    splitter.plan(frame)
            except InsufficientHistoryError as exc:
                error = exc
                continue
            if fold < AUTO_FOLD:
                warnings.warn(
                    f"fold='auto': o histórico comporta {fold} origem(ns) por camada, menos que "
                    f"as {AUTO_FOLD} da calibração da regra de aceite; com menos "
                    "unidades a regra aceita estruturas espúrias com mais frequência.",
                    UserWarning,
                    stacklevel=3,
                )
            return fold
        assert error is not None
        raise error

    def _family_order(self, include: Sequence[str] = ()) -> tuple[str, ...]:
        """Default order with extra base families right before ``residual``.

        With ``"ensemble"`` included, the ensemble family challenges last but before
        ``dynamic`` (ADR-0007): only after every simpler structure had its chance. Without
        it the order, and the protocol hash, stay as before.
        """

        order = list(DEFAULT_FAMILY_ORDER)
        bases = set(self.base_models or DEFAULT_BASES)
        registry = self.registry_
        # A built-in family outside the default order (``intermittent``) joins only when the
        # search uses one of its models, so the default protocol does not change; families
        # of user-registered models join as soon as they are registered.
        extra = [
            family
            for family in registry.families("base")
            if family not in order
            and any(
                name in bases or registry.entry(name).external
                for name in registry.names("base", family)
            )
        ]
        position = order.index("residual")
        order = [*order[:position], *extra, *order[position:]]
        if "ensemble" in include:
            order.insert(order.index("dynamic"), "ensemble")
        return tuple(order)

    def _registered_space(self, name: str) -> dict[str, Distribution]:
        entry = self.registry_.entry(name)
        if entry.tuning_space is None:
            return default_space(name, self.seasonal_period_)
        return {key: _as_distribution(key, value) for key, value in entry.tuning_space.items()}

    def _external_requirements(self, spec: CandidateSpec) -> set[str]:
        packages: set[str] = set()
        for kind, model in spec.model_names():
            packages.update(self.registry_.entry(model, kind).requires)
        return packages

    def _series(self, series_id: str | None) -> SeriesFrame:
        """Development history of one series (the only one, or the first of a panel)."""

        if series_id is None:
            return next(iter(self.panel_))
        return self.panel_[series_id]

    def _evaluate_external_panel(self, model: PanelForecaster) -> ExternalResult:
        if not self._sealed_map:
            raise ConfigurationError("Sem teste selado (test_size=0) no painel.")
        members = {model.model_id, *(member.model_id for member in model.models.values())}
        if members & self._externally_evaluated:
            raise SealedTestAccessError(
                "Este modelo já foi avaliado no teste externo; repetir a avaliação transformaria "
                "o teste em validação."
            )
        self._externally_evaluated.update(members)
        rows, predictions = [], []
        metrics = resolve_metrics(self.evaluator_.metrics)
        for sid, member in model.models.items():
            test = self._sealed_map[sid]._unseal()
            if member.variant == "B":
                table = _rolling_table(member, test)
            else:
                table = member.predict(len(test))
            table["y_true"] = test.values
            values = compute_metrics(
                metrics,
                test.values,
                table["y_pred"].to_numpy(),
                member.history.values,
                self.seasonal_period_,
            )
            rows.append({"model": model.spec.label, "series_id": sid, **values})
            predictions.append(table)
        per_series = pd.DataFrame(rows)
        mean = per_series[list(self.evaluator_.metrics)].mean()
        summary = pd.concat(
            [
                per_series.set_index(["model", "series_id"]),
                pd.DataFrame(
                    [mean], index=pd.MultiIndex.from_tuples([(model.spec.label, "média")])
                ),
            ]
        )
        summary.index.names = ["model", "series_id"]
        external = ExternalResult(summary, pd.concat(predictions), model.spec.label)
        self._external[model.model_id] = external
        self._event("model.evaluated_external", f"{len(model.models)} séries", None)
        return external

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> AutoHybridExperiment:
        """Rebuild an experiment with the parameters recorded in a ``CompareResult`` snapshot."""

        if snapshot.get("schema") != "selection-snapshot/1":
            raise ConfigurationError("Snapshot com esquema desconhecido.")
        return cls(**snapshot["experiment"])

    def _model_result(self, candidate: CandidateSpec, kind: str) -> ModelResult:
        started = time.perf_counter()
        evaluation = self._evaluation(candidate)
        self._event(kind, candidate.label, started)
        return ModelResult(candidate, evaluation)

    def _evaluation(self, candidate: CandidateSpec) -> EvaluationResult:
        """Evaluation on every origin, reusing layers already evaluated by compare_models."""

        key = candidate.candidate_id
        if (key, None) in self._evaluations:
            return self._evaluations[(key, None)]
        parts = []
        for layer in LAYERS:
            part = self._evaluations.get((key, layer))
            if part is None:
                part = self.evaluator_.evaluate(candidate, self.work_plan_, layer)
                self._evaluations[(key, layer)] = part
            parts.append(part)
        combined = EvaluationResult(
            candidate,
            self.work_plan_.fingerprint,
            self.panel_.frequency,
            self.evaluator_.seasonal_period,
            self.evaluator_.metrics,
            parts[0].trials + parts[1].trials,
        )
        self._evaluations[(key, None)] = combined
        return combined

    def _space(self, *, with_residuals: bool, include: Sequence[str] = ()) -> SearchSpace:
        defaults = SearchSpace()
        methods: Sequence[str] = ()
        if "decomposition" in include:
            methods = self.decomposition_methods or (
                ("ma", "stl") if self.variant == "A" else ("ma",)
            )
        return SearchSpace(
            base_models=tuple(self.base_models or DEFAULT_BASES),
            correctors=tuple(self.correctors or defaults.correctors) if with_residuals else (),
            residual_windows=tuple(self.residual_windows or defaults.residual_windows),
            alphas=tuple(float(a) for a in (self.alphas or defaults.alphas)),
            residual_sources=tuple(self.residual_sources or defaults.residual_sources),
            base_params=_grids(self.base_params),
            corrector_params=_grids(self.corrector_params),
            seasonal_period=self.seasonal_period_,
            variant=self.variant,
            max_stages=self.max_stages if "multistage" in include else 1,
            stage_params=self.stage_params,
            decomposition_methods=tuple(methods),
            combination_weights=tuple(self.combination_weights) if "combination" in include else (),
            ensemble_methods=tuple(self.ensemble_methods) if "ensemble" in include else (),
            dynamic_methods=("dref", "local") if "dynamic" in include else (),
        )

    def _snapshot(
        self,
        space: SearchSpace,
        guard: SearchGuardConfig,
        search_protocol: dict[str, object],
        champion: CandidateSpec,
    ) -> dict[str, Any]:
        """What must stay fixed to reproduce the selection (§27)."""

        return {
            "schema": "selection-snapshot/1",
            "experiment": self._public_params(),
            "history_fingerprint": self.panel_.fingerprint
            if self.is_panel
            else self.history_.fingerprint,
            "series": list(self.panel_.series_ids),
            "plan_fingerprint": self.work_plan_.fingerprint,
            "splits": self._splitter.describe(),
            "space": space.describe(),
            "guard": guard.describe(),
            "policy": self.policy_.describe(),
            "evaluator": self.evaluator_.protocol(),
            "search_protocol_hash": search_protocol.get("protocol_hash"),
            "champion": {"candidate_id": champion.candidate_id, "label": champion.label},
            "autohybts_version": autohybts.__version__,
        }

    def _check_include(self, include: Sequence[str]) -> tuple[str, ...]:
        include = tuple(include)
        unknown = set(include) - set(_INCLUDE)
        if unknown or "standalone" not in include:
            raise ConfigurationError(
                f"include aceita {list(_INCLUDE)}, com 'standalone' obrigatório (o campeão "
                f"inicial é individual); recebido {list(include)}."
            )
        if "dynamic" in include and self.variant != "B":
            raise ConfigurationError(
                "'dynamic' só existe na variante B: use AutoHybridExperiment(variant='B')."
            )
        if "dynamic" in include and "residual" not in include:
            raise ConfigurationError("'dynamic' parte das estruturas residuais: inclua 'residual'.")
        if "multistage" in include and "residual" not in include:
            raise ConfigurationError("'multistage' refina estruturas residuais: inclua 'residual'.")
        return include

    def _guard(self) -> SearchGuardConfig:
        overrides: dict[str, Any] = {"max_wall_time_s": self.max_wall_time_s}
        if self.max_trials is not None:
            overrides["max_trials_total"] = self.max_trials
        return SearchGuardConfig.preset(self.budget, **overrides)

    def _public_params(self) -> dict[str, Any]:
        params = self.get_params()
        if params.pop("registry") is not None:
            params["registry"] = "custom (não reproduzível pelo snapshot)"
        for key in (
            "base_models",
            "correctors",
            "residual_windows",
            "alphas",
            "residual_sources",
            "metrics",
        ):
            if params.get(key) is not None:
                params[key] = list(params[key])
        for key in ("base_params", "corrector_params"):
            if params.get(key) is not None:
                params[key] = {k: [dict(p) for p in v] for k, v in _grids(params[key]).items()}
        return params

    def _require(self, state: str) -> None:
        if _STATES.index(self.state_) < _STATES.index(state):
            step = "fit(y)" if state == "VALIDATED" else "compare_models()"
            raise ConfigurationError(
                f"Operação exige o experimento em {state}; estado atual: {self.state_}. "
                f"Chame {step} primeiro."
            )

    def _event(self, kind: str, message: str, started: float | None) -> None:
        duration = None if started is None else (time.perf_counter() - started) * 1000
        self.events_.append(Event(kind, message, datetime.now(UTC), duration))

    def __repr__(self) -> str:
        return (
            f"AutoHybridExperiment(fh={self.fh}, session_id={self.session_id}, state={self.state_})"
        )


def _as_frame(y: object) -> SeriesFrame:
    if isinstance(y, pd.DataFrame):
        raise ConfigurationError(
            "Passe uma pd.Series indexada por data (ou um SeriesFrame); para DataFrames use "
            "SeriesFrame.from_pandas(df, target=..., date_col=...)."
        )
    if not isinstance(y, pd.Series):
        raise ConfigurationError(f"Esperada pd.Series ou SeriesFrame; recebido {type(y).__name__}.")
    return SeriesFrame.from_series(y)


def _rolling_table(model: FinalizedForecaster, test: SeriesFrame) -> pd.DataFrame:
    """Variant B on a test segment: one-step forecasts, each value revealed after its forecast."""

    forecasts = rolling_one_step(
        model.model,
        model.history.values,
        test.values,
        series_id=model.history.series_id,
        frequency=model.history.frequency,
    )
    origins = [model.last_observation, *test.timestamps[:-1]]
    return pd.DataFrame(
        {
            "series_id": model.history.series_id,
            "origin": origins,
            "horizon": 1,
            "y_pred": forecasts,
        },
        index=pd.DatetimeIndex(test.timestamps, name="timestamp"),
    )


def _as_panel(
    y: object, *, series_col: str | None, date_col: str | None, target: str | None
) -> PanelFrame:
    if isinstance(y, PanelFrame):
        return y
    if isinstance(y, SeriesFrame):
        return PanelFrame.single(y)
    if isinstance(y, pd.DataFrame):
        if series_col is None or date_col is None or target is None:
            raise ConfigurationError(
                "Para um DataFrame longo informe series_col, date_col e target; para uma série "
                "única passe uma pd.Series indexada por data."
            )
        return PanelFrame.from_long(y, series_col=series_col, date_col=date_col, target=target)
    if isinstance(y, Mapping):
        return PanelFrame.from_dict(y)
    return PanelFrame.single(_as_frame(y))


def _as_distribution(name: str, value: object) -> Distribution:
    if isinstance(value, Distribution):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        low, high = value
        if isinstance(low, int) and isinstance(high, int):
            return Distribution("int", low, high)
        return Distribution("float", float(low), float(high))
    if isinstance(value, list) and value:
        return Distribution("categorical", choices=tuple(value))
    raise ConfigurationError(
        f"Espaço de {name!r}: use Distribution, (mínimo, máximo) ou uma lista de opções."
    )


def _clone_with(template: Any, params: Mapping[str, Any]) -> Any:
    """Fresh copy of a registered model instance, with ``params`` overriding its own."""

    if not params:
        return template.clone()
    return type(template)(**{**template.get_params(), **dict(params)})


def _grids(
    grids: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Per-model parameter grids as tuples of plain dicts; a lone dict is a one-item grid."""

    if not grids:
        return {}
    if not isinstance(grids, Mapping):
        raise ConfigurationError("As grades devem mapear nome do modelo -> lista de parâmetros.")
    out: dict[str, tuple[dict[str, Any], ...]] = {}
    for name, grid in grids.items():
        items: list[object] = [grid] if isinstance(grid, Mapping) else list(grid)
        dicts = [dict(item) for item in items if isinstance(item, Mapping)]
        if not dicts or len(dicts) != len(items):
            raise ConfigurationError(f"A grade de {name!r} deve ser uma lista de dicionários.")
        out[str(name)] = tuple(dicts)
    return out
