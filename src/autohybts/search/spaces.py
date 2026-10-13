"""Search space and candidate generation per stage (§8.1, §8.2).

Generators return proposals in a deterministic priority order. A proposal that the
registry's capabilities rule out carries a ``skip_reason`` and is never evaluated.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from autohybts.data.stationarity import DIFFERENCING_TESTS
from autohybts.data.target_prep import check_preprocessing
from autohybts.domain.identity import canonical_json
from autohybts.domain.specs import (
    COMBINATION_WEIGHTS,
    DECOMPOSITION_METHODS,
    ENSEMBLE_METHODS,
    MAX_STAGES,
    CandidateSpec,
    ModelSpec,
)
from autohybts.exceptions import AutoHybTSError, ConfigurationError
from autohybts.registry.registry import ModelRegistry
from autohybts.types import RESIDUAL_SOURCES, ProtocolVariant

__all__ = ["DEFAULT_BASES", "Proposal", "SearchSpace"]

DEFAULT_BASES: Final = (
    "naive",
    "seasonal_naive",
    "linear",
    "knn",
    "svm",
    "tree",
    "random_forest",
    "arima",
    "mlp",
)
_LAG_MODELS: Final = frozenset(
    {
        "linear",
        "knn",
        "svm",
        "tree",
        "random_forest",
        "mlp",
        "hist_gradient_boosting",
        "adaboost_r2",
        "lightgbm_regressor",
        "ridge_regression",
        "lasso_regression",
        "elastic_net_regression",
        "extra_trees_regressor",
    }
)
# Dynamic selection grid of the legacy project (Silva et al., 2020); first value = default.
DYNAMIC_NEIGHBORS: Final = (10, 5, 20)
DYNAMIC_LAGS: Final = (7, 14)
LOCAL_ALPHAS: Final = (1.0, 0.1)
# Ensembles of the search start small (plano de ensembles §7.4: K = 2, 3 before more).
ENSEMBLE_SIZE: Final = 3
_WEIGHTED: Final = ("equal", "inverse_mse", "nnls")


@dataclass(frozen=True, slots=True)
class Proposal:
    candidate: CandidateSpec
    stage: str
    parent: str | None = None
    skip_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """What the search may try; the default search space by default.

    ``base_params`` and ``corrector_params`` map a model name to a grid (sequence of
    parameter dicts); a missing entry means the model's defaults. ``refine_sources`` are
    the residual sources tried in S3 around the best residual structures.

    Advanced families are off by default: ``max_stages > 1`` adds correction stages in
    S3; ``decomposition_methods``, ``combination_weights`` and ``ensemble_methods`` open
    stage S4.
    """

    base_models: tuple[str, ...] = DEFAULT_BASES
    base_params: Mapping[str, Sequence[Mapping[str, Any]]] = field(default_factory=dict)
    correctors: tuple[str, ...] = ("linear", "knn", "mlp")
    corrector_params: Mapping[str, Sequence[Mapping[str, Any]]] = field(default_factory=dict)
    residual_windows: tuple[int, ...] = (7, 28)
    alphas: tuple[float, ...] = (0.5, 1.0)
    residual_sources: tuple[str, ...] = ("historical_oof_1step",)
    refine_sources: tuple[str, ...] = ("historical_oof_multistep", "in_sample")
    seasonal_period: int = 1
    variant: ProtocolVariant = "A"
    max_stages: int = 1
    decomposition_methods: tuple[str, ...] = ()
    combination_weights: tuple[str, ...] = ()
    dynamic_methods: tuple[str, ...] = ()
    stage_params: str = "shared"
    ensemble_methods: tuple[str, ...] = ()
    differencing: str | None = None
    differencing_test: str = "kpss"
    transform: str | None = None
    outliers: str | None = None

    def __post_init__(self) -> None:
        if not self.base_models:
            raise ConfigurationError("A busca precisa de ao menos um modelo-base.")
        unknown = set(self.residual_sources + self.refine_sources) - set(RESIDUAL_SOURCES)
        if unknown:
            raise ConfigurationError(f"Fontes residuais desconhecidas: {sorted(unknown)}.")
        if any(
            isinstance(w, bool) or not isinstance(w, int) or w < 1 for w in self.residual_windows
        ):
            raise ConfigurationError("Janelas residuais devem ser inteiros >= 1.")
        if self.seasonal_period < 1:
            raise ConfigurationError("seasonal_period deve ser >= 1.")
        if (
            isinstance(self.max_stages, bool)
            or not isinstance(self.max_stages, int)
            or not 1 <= self.max_stages <= MAX_STAGES
        ):
            raise ConfigurationError(f"max_stages deve estar entre 1 e {MAX_STAGES}.")
        unknown_methods = set(self.decomposition_methods) - set(DECOMPOSITION_METHODS)
        if unknown_methods:
            raise ConfigurationError(f"Decomposições desconhecidas: {sorted(unknown_methods)}.")
        if "stl" in self.decomposition_methods and self.variant != "A":
            raise ConfigurationError("A STL só existe na variante A; use decomposition 'ma'.")
        if self.stage_params not in ("shared", "per_stage"):
            raise ConfigurationError("stage_params deve ser 'shared' ou 'per_stage'.")
        unknown_dynamic = set(self.dynamic_methods) - {"dref", "local"}
        if unknown_dynamic:
            raise ConfigurationError(f"Seleção dinâmica desconhecida: {sorted(unknown_dynamic)}.")
        if self.dynamic_methods and self.variant != "B":
            raise ConfigurationError(
                "A seleção dinâmica só existe na variante B: em A não há observação nova entre "
                "os passos para escolher o corretor."
            )
        unknown_ensembles = set(self.ensemble_methods) - set(ENSEMBLE_METHODS)
        if unknown_ensembles:
            raise ConfigurationError(
                f"Métodos de ensemble desconhecidos: {sorted(unknown_ensembles)}."
            )
        if self.differencing not in (None, "auto"):
            raise ConfigurationError(
                f"differencing da busca deve ser None ou 'auto'; recebido {self.differencing!r}."
            )
        if self.differencing_test not in DIFFERENCING_TESTS:
            raise ConfigurationError(
                f"differencing_test deve ser um de {list(DIFFERENCING_TESTS)}; "
                f"recebido {self.differencing_test!r}."
            )
        check_preprocessing(self.transform, self.outliers)
        unknown_weights = set(self.combination_weights) - set(COMBINATION_WEIGHTS)
        if unknown_weights:
            raise ConfigurationError(
                f"Pesos de combinação desconhecidos: {sorted(unknown_weights)}."
            )

    def describe(self) -> dict[str, Any]:
        described = self._describe_core()
        # Advanced settings appear only when used, so earlier protocol hashes still hold.
        if self.max_stages != 1:
            described["max_stages"] = self.max_stages
        if self.decomposition_methods:
            described["decomposition_methods"] = list(self.decomposition_methods)
        if self.combination_weights:
            described["combination_weights"] = list(self.combination_weights)
        if self.dynamic_methods:
            described["dynamic_methods"] = list(self.dynamic_methods)
        if self.stage_params != "shared":
            described["stage_params"] = self.stage_params
        if self.ensemble_methods:
            described["ensemble_methods"] = list(self.ensemble_methods)
        if self.differencing is not None:
            described["differencing"] = self.differencing
            described["differencing_test"] = self.differencing_test
        if self.transform is not None:
            described["transform"] = self.transform
        if self.outliers is not None:
            described["outliers"] = self.outliers
        return described

    def _describe_core(self) -> dict[str, Any]:
        return {
            "base_models": list(self.base_models),
            "base_params": {k: [dict(p) for p in v] for k, v in self.base_params.items()},
            "correctors": list(self.correctors),
            "corrector_params": {k: [dict(p) for p in v] for k, v in self.corrector_params.items()},
            "residual_windows": list(self.residual_windows),
            "alphas": list(self.alphas),
            "residual_sources": list(self.residual_sources),
            "refine_sources": list(self.refine_sources),
            "seasonal_period": self.seasonal_period,
            "variant": self.variant,
        }

    def base_proposals(self, registry: ModelRegistry) -> list[Proposal]:
        """S0: every configured base, in the configured order."""

        proposals = []
        for name in self.base_models:
            entry = registry.entry(name)
            for params in self._base_grid(name):
                candidate = CandidateSpec.standalone(
                    name, params, family=entry.family, variant=self.variant
                )
                reason = None
                if name == "seasonal_naive" and self.seasonal_period == 1:
                    reason = "sem sazonalidade declarada: equivale ao naive"
                elif name == "sarima" and self.seasonal_period == 1:
                    reason = "sem sazonalidade declarada: equivale ao arima"
                proposals.append(Proposal(candidate, "S0", skip_reason=reason))
        return proposals

    def residual_proposals(
        self, bases: Sequence[CandidateSpec], registry: ModelRegistry
    ) -> list[Proposal]:
        """S2: residual grid, breadth-first over the retained bases.

        Each (source, corrector, window, alpha) combination is proposed for every retained
        base (best first) before the next combination, so a budget cut keeps the bases
        diverse instead of exhausting the grid of the first one.
        """

        return [
            self._residual(
                base.base, corrector, {}, window, alpha, source, "S2", base.candidate_id, registry
            )
            for source in self.residual_sources
            for corrector in self.correctors
            for window in self._windows(corrector, registry)
            for alpha in self.alphas
            for base in bases
        ]

    def _windows(self, corrector: str, registry: ModelRegistry) -> tuple[int, ...]:
        """Windows to try with ``corrector``: one for a sequence corrector (it has none)."""

        if _is_sequence(corrector, registry):
            return self.residual_windows[:1]
        return self.residual_windows

    def _window_correctors(self, registry: ModelRegistry) -> tuple[str, ...]:
        """Correctors usable where a window regression is required."""

        return tuple(name for name in self.correctors if not _is_sequence(name, registry))

    def refinement_proposals(
        self, structures: Sequence[CandidateSpec], registry: ModelRegistry, max_refinements: int
    ) -> list[Proposal]:
        """S3: around each best residual structure, change one axis at a time.

        Axes: residual source, corrector hyper-parameters and, when ``max_stages > 1``,
        extra correction stages. At most ``max_refinements`` variants per axis and
        structure.
        """

        proposals = []
        for best in structures:
            corrector, window = best.corrector, best.window_size
            if corrector is None or window is None:
                raise ConfigurationError("Refinamento exige estruturas residuais.")
            alpha = float(best.alpha if best.alpha is not None else 1.0)
            source = str(best.residual_source)
            by_source = [
                self._residual(
                    best.base,
                    corrector.name,
                    corrector.params,
                    window,
                    alpha,
                    other,
                    "S3",
                    best.candidate_id,
                    registry,
                )
                for other in self.refine_sources
                if other != source
            ]
            by_params = [
                self._residual(
                    best.base,
                    corrector.name,
                    params,
                    window,
                    alpha,
                    source,
                    "S3",
                    best.candidate_id,
                    registry,
                )
                for params in self.corrector_params.get(corrector.name, ())
                if dict(params) != corrector.params
            ]
            by_stages = self._stage_proposals(best, registry)
            proposals.extend(
                by_source[:max_refinements]
                + by_params[:max_refinements]
                + by_stages[:max_refinements]
            )
        return proposals

    def _stage_proposals(self, best: CandidateSpec, registry: ModelRegistry) -> list[Proposal]:
        """Extra stages on ``best``: the same corrector again first, then the others."""

        if self.max_stages < 2 or best.strategy != "residual_additive" or best.corrector is None:
            return []
        same = best.corrector
        others = [ModelSpec.of(name) for name in self.correctors if name != same.name]
        chains: list[tuple[ModelSpec, ...]] = [(same,)]
        chains += [(other,) for other in others]
        if self.max_stages >= 3:
            chains.append((same, same))
        variants: list[dict[str, Any] | None] = [None]
        if self.stage_params == "per_stage":
            # One axis at a time on the extra stage: other windows, then other alphas.
            variants += [{"window_size": w} for w in self.residual_windows if w != best.window_size]
            variants += [{"alpha": float(a)} for a in self.alphas if float(a) != best.alpha]
        proposals = []
        for variant in variants:
            for chain in chains:
                for spec in chain:
                    registry.entry(spec.name, "corrector")
                options = {} if variant is None else {"stages": [dict(variant)] * len(chain)}
                candidate = CandidateSpec(
                    "residual",
                    "residual_multistage",
                    best.base,
                    same,
                    best.window_size,
                    best.alpha,
                    best.residual_source,
                    variant=self.variant,
                    members=chain,
                    options_json=canonical_json(options),
                )
                proposals.append(Proposal(candidate, "S3", best.candidate_id))
        return proposals

    def advanced_proposals(
        self, bases: Sequence[CandidateSpec], registry: ModelRegistry
    ) -> list[Proposal]:
        """S4: decompositions over the retained bases and fixed combinations of the best.

        Decompositions go breadth-first over the bases, like the residual grid. A
        combination joins the two best retained bases of different families. The two
        lists are interleaved, so a budget cut keeps both families represented.
        """

        proposals: list[Proposal] = []
        for method in self.decomposition_methods:
            reason = self._decomposition_skip(method)
            for corrector in self._window_correctors(registry):
                for window in self.residual_windows:
                    for base in bases:
                        options: dict[str, Any] = {"method": method}
                        if method == "stl":
                            options["period"] = max(2, self.seasonal_period)
                        candidate = CandidateSpec(
                            "decomposition",
                            "decomposition",
                            base.base,
                            ModelSpec.of(corrector),
                            window,
                            variant=self.variant,
                            options_json=canonical_json(options),
                        )
                        proposals.append(Proposal(candidate, "S4", base.candidate_id, reason))
        combinations: list[Proposal] = []
        pair: list[CandidateSpec] = []
        for base in bases:
            if all(base.family != chosen.family for chosen in pair):
                pair.append(base)
            if len(pair) == 2:
                break
        if len(pair) == 2:
            for weights in self.combination_weights:
                candidate = CandidateSpec(
                    "combination",
                    "fixed_combination",
                    pair[0].base,
                    variant=self.variant,
                    members=(pair[1].base,),
                    options_json=canonical_json({"weights": weights}),
                )
                combinations.append(Proposal(candidate, "S4", pair[0].candidate_id))
        return _interleave(proposals, combinations)

    def dynamic_proposals(
        self, structures: Sequence[CandidateSpec], registry: ModelRegistry
    ) -> list[Proposal]:
        """S4 (variant B): dynamic selection around the best residual structures.

        ``dref`` chooses, at each step, among the structure's corrector, the other
        configured correctors and no correction; ``local`` combines the base and the
        structure's corrector locally. Grid of the legacy project (Silva et al., 2020):
        neighbours ``k`` in ``DYNAMIC_NEIGHBORS``, lag size in ``DYNAMIC_LAGS`` and, for
        ``local``, the kernel ridge ``alpha`` in ``LOCAL_ALPHAS``. The first value of each
        axis is the default; defaults of every method come first, so a budget cut keeps
        them.
        """

        proposals: list[Proposal] = []
        rest: list[Proposal] = []
        for best in structures:
            if best.corrector is None or best.window_size is None:
                continue
            if _is_sequence(best.corrector.name, registry):
                continue  # dynamic selection needs window correctors
            others = tuple(
                ModelSpec.of(name)
                for name in self._window_correctors(registry)
                if name != best.corrector.name
            )
            for method in self.dynamic_methods:
                members = others if method == "dref" else ()
                for spec in members:
                    registry.entry(spec.name, "corrector")
                alphas = LOCAL_ALPHAS if method == "local" else (None,)
                grid = [
                    (k, lag, alpha)
                    for k in DYNAMIC_NEIGHBORS
                    for lag in DYNAMIC_LAGS
                    for alpha in alphas
                ]
                for index, (k, lag, alpha) in enumerate(grid):
                    options: dict[str, Any] = {"method": method, "neighbors": k, "lag_size": lag}
                    if alpha is not None:
                        options["local_alpha"] = alpha
                    candidate = CandidateSpec(
                        "dynamic",
                        "dynamic_selection",
                        best.base,
                        best.corrector,
                        best.window_size,
                        best.alpha,
                        best.residual_source,
                        variant=self.variant,
                        members=members,
                        options_json=canonical_json(options),
                    )
                    proposal = Proposal(candidate, "S4", best.candidate_id)
                    (proposals if index == 0 else rest).append(proposal)
        return proposals + rest

    def corrector_combination_proposals(
        self, structures: Sequence[CandidateSpec], registry: ModelRegistry
    ) -> list[Proposal]:
        """S4: around each best residual structure, its corrector combined with the others."""

        if len(self.correctors) < 2 or not self.combination_weights:
            return []
        proposals = []
        for best in structures:
            if best.strategy != "residual_additive" or best.corrector is None:
                continue
            if _is_sequence(best.corrector.name, registry):
                continue  # corrector combinations average window regressions
            others = tuple(
                ModelSpec.of(name)
                for name in self._window_correctors(registry)
                if name != best.corrector.name
            )
            if not others:
                continue
            for weights in self.combination_weights:
                candidate = CandidateSpec(
                    "combination",
                    "residual_combination",
                    best.base,
                    best.corrector,
                    best.window_size,
                    best.alpha,
                    best.residual_source,
                    variant=self.variant,
                    members=others,
                    options_json=canonical_json({"weights": weights}),
                )
                proposals.append(Proposal(candidate, "S4", best.candidate_id))
        return proposals

    def ensemble_proposals(
        self,
        bases: Sequence[CandidateSpec],
        structures: Sequence[CandidateSpec],
        registry: ModelRegistry,
    ) -> list[Proposal]:
        """S4: ensembles (family ``ensemble``), all from candidates already searched.

        * Ensembles of bases: the best retained base of each family, up to
          ``ENSEMBLE_SIZE`` (diversity of families before more members), with every
          configured method; ``median`` needs 3 members and ``trimmed_mean`` is left to
          explicit use (with 3 members it equals the median).
        * Ensemble of hybrids: the two best residual structures, weighted methods only.
        * Corrected ensemble: the equal-weight ensemble of bases corrected by each
          configured corrector (first window, alpha and source of the space).

        The equal-weight version of each kind comes first, so a budget cut keeps the
        simplest control of every kind.
        """

        if not self.ensemble_methods:
            return []
        pool: list[CandidateSpec] = []
        for base in bases:
            if all(base.family != chosen.family for chosen in pool):
                pool.append(base)
            if len(pool) == ENSEMBLE_SIZE:
                break
        kinds: list[list[Proposal]] = []
        if len(pool) >= 2:
            kinds.append(
                [
                    Proposal(
                        CandidateSpec.ensemble(pool, method=method), "S4", pool[0].candidate_id
                    )
                    for method in self.ensemble_methods
                    if method != "trimmed_mean" and (method != "median" or len(pool) >= 3)
                ]
            )
        if len(structures) >= 2:
            pair = list(structures[:2])
            kinds.append(
                [
                    Proposal(
                        CandidateSpec.ensemble(pair, method=method), "S4", pair[0].candidate_id
                    )
                    for method in self.ensemble_methods
                    if method in _WEIGHTED
                ]
            )
        if len(pool) >= 2 and self.correctors and self.residual_windows and self.alphas:
            base = CandidateSpec.ensemble(pool, method="equal")
            corrected = []
            for corrector in self.correctors:
                registry.entry(corrector, "corrector")
                candidate = CandidateSpec.corrected(
                    base,
                    ModelSpec.of(corrector),
                    self.residual_windows[0],
                    float(self.alphas[-1]),
                    self.residual_sources[0],
                )
                corrected.append(Proposal(candidate, "S4", base.candidate_id))
            kinds.append(corrected)
        ordered: list[Proposal] = []
        for index in range(max((len(k) for k in kinds), default=0)):
            ordered.extend(kind[index] for kind in kinds if index < len(kind))
        return ordered

    def _decomposition_skip(self, method: str) -> str | None:
        if method != "stl":
            return None
        if self.seasonal_period < 2:
            return "sem sazonalidade declarada: a STL exige período >= 2"
        if importlib.util.find_spec("statsmodels") is None:
            return 'a STL requer o extra `stats` (pip install "autohybts[stats]")'
        return None

    def default_params(self, name: str) -> dict[str, Any]:
        """Parameters a base gets when no grid is given (seasonality and lags of this space)."""

        if name == "seasonal_naive":
            return {"season_length": self.seasonal_period}
        if name in (
            "seasonal_window_average",
            "exponential_smoothing",
            "sarima",
        ) or name.startswith("sf_"):
            return {"season_length": self.seasonal_period}
        if name in _LAG_MODELS:
            params: dict[str, Any] = {"lags": max(7, self.seasonal_period)}
            if self.differencing is not None:
                params.update(
                    differencing=self.differencing,
                    season_length=self.seasonal_period,
                    differencing_test=self.differencing_test,
                )
            if self.transform is not None:
                params["transform"] = self.transform
            if self.outliers is not None:
                params["outliers"] = self.outliers
            return params
        return {}

    def _base_grid(self, name: str) -> list[dict[str, Any]]:
        grid = [dict(p) for p in self.base_params.get(name, ())] or [{}]
        defaults = self.default_params(name)
        return [{**defaults, **params} for params in grid]

    def _residual(
        self,
        base: ModelSpec,
        corrector: str,
        corrector_params: Mapping[str, Any],
        window: int,
        alpha: float,
        source: str,
        stage: str,
        parent: str,
        registry: ModelRegistry,
    ) -> Proposal:
        registry.entry(corrector, "corrector")
        candidate = CandidateSpec(
            "residual",
            "residual_additive",
            base,
            ModelSpec.of(corrector, corrector_params),
            window,
            float(alpha),
            source,
            variant=self.variant,
        )
        reason = None
        if source == "in_sample" and not registry.entry(base.name).capabilities.in_sample_fitted:
            reason = f"{base.name} não expõe valores ajustados (fonte in_sample)"
        return Proposal(candidate, stage, parent, reason)


def _interleave(first: list[Proposal], second: list[Proposal]) -> list[Proposal]:
    return round_robin([*first, *second])


def round_robin(proposals: Sequence[Proposal]) -> list[Proposal]:
    """One proposal per family in turn, keeping each family's order."""

    groups: dict[str, list[Proposal]] = {}
    for proposal in proposals:
        groups.setdefault(proposal.candidate.family, []).append(proposal)
    merged: list[Proposal] = []
    for index in range(max((len(g) for g in groups.values()), default=0)):
        merged.extend(group[index] for group in groups.values() if index < len(group))
    return merged


def _is_sequence(corrector: str, registry: ModelRegistry) -> bool:
    """Whether ``corrector`` forecasts the residual series (``residuals.sequence``)."""

    from autohybts.residuals.sequence import ResidualSequenceForecaster

    registry.entry(corrector, "corrector")  # unknown names fail here, as before
    try:
        made = registry.make_corrector(corrector)
    except AutoHybTSError:
        return False
    return isinstance(made, ResidualSequenceForecaster)
