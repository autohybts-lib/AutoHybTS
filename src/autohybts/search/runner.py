"""Staged, budgeted search over ``V_search`` (§8.2, §30.2).

S0 evaluates every base; S1 retains the best structures per family; S2 evaluates the
residual grid on the retained bases; S3 refines the best residual structures one axis
at a time (source, corrector parameters, extra stages); S4, when enabled, evaluates
decompositions and fixed combinations of the retained bases. Budgets are applied to the
proposal list *before* a stage runs, in a fixed priority order, so the number of
evaluated candidates never exceeds the limits and does not depend on how fast or in
which order parallel evaluations finish.

``max_wall_time_s`` is a cooperative deadline (§8.3: best effort, never a hard
guarantee without process isolation): it is checked before *each* candidate starts, in
the sequential and in the parallel path, and no candidate starts after it. A candidate
already running finishes all its origins (a partial candidate would have incomplete
coverage); admitted candidates that never started are ``STOPPED`` in the ledger. There is
no per-candidate timeout. Under a deadline the set of evaluated candidates depends on the
machine; budgets by count are the reproducible ones.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Final

import autohybts
from autohybts.data.panel import PanelPlan
from autohybts.data.splitter import TemporalPlan, ensure_not_sealed
from autohybts.domain.identity import stable_hash
from autohybts.domain.results import EvaluationResult, Layer
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.exceptions import ConfigurationError, CoverageError
from autohybts.progress import NullProgress, Progress
from autohybts.search.budgets import SearchGuardConfig
from autohybts.search.cache import MemoryTrialCache, evaluation_key
from autohybts.search.ledger import LedgerEntry, LedgerStatus, SearchResult
from autohybts.search.pruning import eligible, retain_bases, top_structures
from autohybts.search.spaces import Proposal, SearchSpace, round_robin

__all__ = ["SearchEngine"]

LAYER: Final[Layer] = "search"


@dataclass
class _State:
    started: float
    deadline: float | None = None
    ledger: list[LedgerEntry] = field(default_factory=list)
    results: dict[str, EvaluationResult] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)
    per_family: Counter[str] = field(default_factory=Counter)
    stopped: bool = False

    @property
    def evaluated(self) -> int:
        return sum(self.per_family.values())


class SearchEngine:
    def __init__(
        self,
        evaluator: TemporalEvaluator,
        space: SearchSpace,
        guard: SearchGuardConfig,
        *,
        cache: MemoryTrialCache | None = None,
        max_workers: int = 1,
        clock: Callable[[], float] = time.perf_counter,
        progress: Progress | None = None,
    ) -> None:
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
            raise ConfigurationError("max_workers deve ser inteiro >= 1.")
        self.evaluator = evaluator
        self.space = space
        self.guard = guard
        self.cache = cache
        self.progress: Progress = progress or NullProgress()
        self.max_workers = max_workers
        self.clock = clock

    @property
    def metric(self) -> str:
        return self.evaluator.metrics[0]

    def protocol(self, plan: TemporalPlan | PanelPlan) -> dict[str, object]:
        """Pre-registered settings of this search; its hash goes in the audit (§30.2)."""

        protocol: dict[str, object] = {
            "plan": plan.fingerprint,
            "layer": LAYER,
            "metric": self.metric,
            "space": self.space.describe(),
            "guard": self.guard.describe(),
            "evaluator": self.evaluator.protocol(),
            "session_seed": self.evaluator.session_seed,
            "code": autohybts.__version__,
        }
        protocol["protocol_hash"] = stable_hash(protocol)
        return protocol

    def run(self, plan: TemporalPlan | PanelPlan) -> SearchResult:
        ensure_not_sealed(plan)
        if not isinstance(plan, TemporalPlan | PanelPlan):
            raise ConfigurationError("A busca exige um TemporalPlan ou PanelPlan.")
        registry = self.evaluator.registry
        started = self.clock()
        limit = self.guard.max_wall_time_s
        state = _State(started=started, deadline=None if limit is None else started + limit)

        bases = self._stage(self.space.base_proposals(registry), plan, state)
        if not any(eligible(r, self.metric, self.guard.require_full_pairing) for r in bases):
            if state.stopped:
                raise CoverageError(
                    "O prazo max_wall_time_s acabou antes de algum modelo-base válido ser "
                    "avaliado; aumente o prazo ou use só o orçamento por quantidade."
                )
            raise CoverageError(
                "Nenhum modelo-base produziu previsões válidas em todas as origens de busca; "
                "não há candidata para selecionar."
            )
        retained = retain_bases(
            bases,
            self.metric,
            top_k=self.guard.top_k_per_family,
            full_pairing=self.guard.require_full_pairing,
        )

        retained_specs = [r.candidate for r in retained]
        advanced = self.space.advanced_proposals(retained_specs, registry)
        # S4 runs on what is left, with one slot kept per advanced family so each one is
        # represented; the slots come from the total budget, not from the residual family.
        advanced_reserve = len({p.candidate.family for p in advanced if p.skip_reason is None})
        if self.space.dynamic_methods and self.space.correctors:
            advanced_reserve += 1
        if self.space.ensemble_methods:
            advanced_reserve += 1
        # SX (covariates) runs after S3 with slots kept from the total budget: its bases and,
        # with correctors, as many residual structures as S2 keeps.
        exog = self.space.exog_proposals(registry)
        exog_admissible = [p for p in exog if p.skip_reason is None]
        exog_reserve = min(len(exog_admissible), self.guard.top_k_per_family * 2)
        if exog_reserve and self.space.correctors:
            exog_reserve += self.guard.top_k_residual_structures
        later_reserve = advanced_reserve + exog_reserve
        structures: list[EvaluationResult] = []
        if self.space.correctors and self.space.residual_windows and self.space.alphas:
            # Keep room for S3: refinement is the point of the staged search (§8.2).
            reserve = self.guard.top_k_residual_structures * self.guard.max_refinements
            residual = self._stage(
                self.space.residual_proposals(retained_specs, registry),
                plan,
                state,
                reserve=reserve + later_reserve,
                family_reserve=reserve,
            )
            structures = top_structures(
                residual,
                self.metric,
                k=self.guard.top_k_residual_structures,
                full_pairing=self.guard.require_full_pairing,
            )
            if structures and self.guard.max_refinements > 0:
                self._stage(
                    self.space.refinement_proposals(
                        [r.candidate for r in structures], registry, self.guard.max_refinements
                    ),
                    plan,
                    state,
                    reserve=later_reserve,
                    family_reserve=0,
                )
        if exog:
            self._exog_stage(exog, plan, state, advanced_reserve)
        best_structures = [r.candidate for r in structures]
        dynamic = self.space.dynamic_proposals(best_structures, registry)
        dynamic += self.space.corrector_combination_proposals(best_structures, registry)
        ensembles = self.space.ensemble_proposals(retained_specs, best_structures, registry)
        if advanced or dynamic or ensembles:
            self._stage(round_robin([*advanced, *dynamic, *ensembles]), plan, state)

        pruned = {e.reason for e in state.ledger if e.status == "PRUNED_BUDGET"}
        if state.stopped:
            reason = "max_wall_time"
        elif state.evaluated >= self.guard.max_trials_total:
            reason = "max_trials_total"
        elif pruned:
            reason = "budget_limited"
        else:
            reason = "completed"
        return SearchResult(
            metric=self.metric,
            plan_fingerprint=plan.fingerprint,
            ledger=tuple(state.ledger),
            results=tuple(state.results.values()),
            retained=tuple(r.candidate_id for r in retained),
            residual_structures=tuple(r.candidate_id for r in structures),
            stop_reason=reason,
            protocol=self.protocol(plan),
            elapsed_s=self.clock() - state.started,
        )

    def _exog_stage(
        self,
        proposals: Sequence[Proposal],
        plan: TemporalPlan | PanelPlan,
        state: _State,
        advanced_reserve: int,
    ) -> None:
        """SX: bases with covariates, then correctors on the best of them."""

        registry = self.evaluator.registry
        hybrid_reserve = self.guard.top_k_residual_structures if self.space.correctors else 0
        bases = self._stage(
            proposals,
            plan,
            state,
            reserve=advanced_reserve + hybrid_reserve,
            family_reserve=hybrid_reserve,
        )
        if not self.space.correctors:
            return
        best = top_structures(
            bases,
            self.metric,
            k=self.guard.top_k_residual_structures,
            full_pairing=self.guard.require_full_pairing,
        )
        if best:
            self._stage(
                self.space.exog_residual_proposals([r.candidate for r in best], registry),
                plan,
                state,
                reserve=advanced_reserve,
                family_reserve=0,
            )

    def _stage(
        self,
        proposals: Sequence[Proposal],
        plan: TemporalPlan | PanelPlan,
        state: _State,
        *,
        reserve: int = 0,
        family_reserve: int | None = None,
    ) -> list[EvaluationResult]:
        """Admit proposals in order under the budgets, keeping room for later stages.

        ``reserve`` is held back from the total budget, ``family_reserve`` (default:
        ``reserve``) from each family's budget.
        """

        family_reserve = reserve if family_reserve is None else family_reserve
        if self._past_deadline(state):
            state.stopped = True
        admitted: list[Proposal] = []
        pending: Counter[str] = Counter()
        for proposal in proposals:
            candidate = proposal.candidate
            status: LedgerStatus = "EVALUATED"
            reason: str | None = None
            if proposal.skip_reason is not None:
                status, reason = "SKIPPED_CAPABILITY", proposal.skip_reason
            elif candidate.candidate_id in state.seen:
                status, reason = "DUPLICATE", "candidate_id já avaliado"
            elif state.stopped:
                status, reason = "STOPPED", "max_wall_time_s atingido"
            elif len(admitted) >= self.guard.max_candidates_per_stage:
                status, reason = "PRUNED_BUDGET", "max_candidates_per_stage"
            elif state.evaluated + len(admitted) >= self.guard.max_trials_total - reserve:
                status, reason = "PRUNED_BUDGET", "max_trials_total"
            elif (
                state.per_family[candidate.family] + pending[candidate.family]
                >= self.guard.max_trials_per_family - family_reserve
            ):
                status, reason = "PRUNED_BUDGET", "max_trials_per_family"
            if status != "EVALUATED":
                state.ledger.append(_entry(proposal, status, reason))
                continue
            state.seen.add(candidate.candidate_id)
            pending[candidate.family] += 1
            admitted.append(proposal)

        outcomes = self._evaluate_all(admitted, plan, state)
        evaluated = []
        for proposal, outcome in zip(admitted, outcomes, strict=True):
            if outcome is None:  # admitted, but the deadline passed before it started
                state.stopped = True
                state.seen.discard(proposal.candidate.candidate_id)
                state.ledger.append(
                    _entry(proposal, "STOPPED", "max_wall_time_s atingido antes de iniciar")
                )
                continue
            result, cached, seconds = outcome
            state.per_family[proposal.candidate.family] += 1
            state.results[result.candidate_id] = result
            state.ledger.append(
                _entry(proposal, "EVALUATED", None, result, self.metric, seconds, cached)
            )
            evaluated.append(result)
        return evaluated

    def _evaluate_all(
        self, proposals: Sequence[Proposal], plan: TemporalPlan | PanelPlan, state: _State
    ) -> list[tuple[EvaluationResult, bool, float] | None]:
        """Evaluate in submission order; ``None`` for a candidate the deadline kept out."""

        def one(proposal: Proposal) -> tuple[EvaluationResult, bool, float] | None:
            if self._past_deadline(state):
                return None
            key = evaluation_key(
                proposal.candidate.candidate_id,
                plan.fingerprint,
                LAYER,
                self.evaluator.protocol(),
                self.evaluator.session_seed,
                autohybts.__version__,
            )
            if self.cache is not None and (hit := self.cache.get(key)) is not None:
                return hit, True, 0.0
            started = self.clock()
            result = self.evaluator.evaluate(proposal.candidate, plan, LAYER)
            if self.cache is not None:
                self.cache.put(key, result)
            return result, False, self.clock() - started

        if not proposals:
            return []
        stage = proposals[0].stage
        names = {
            "S0": "S0 bases",
            "S2": "S2 resíduos",
            "S3": "S3 refino",
            "SX": "SX exógenas",
            "S4": "S4 avançadas",
        }
        outcomes: list[tuple[EvaluationResult, bool, float] | None] = []
        with self.progress.bar(len(proposals), f"busca {names.get(stage, stage)}") as bar:
            if self.max_workers == 1:
                for proposal in proposals:
                    outcomes.append(one(proposal))
                    bar.update()
            else:
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    for outcome in pool.map(one, proposals):  # map keeps submission order
                        outcomes.append(outcome)
                        bar.update()
        return outcomes

    def _past_deadline(self, state: _State) -> bool:
        return state.deadline is not None and self.clock() >= state.deadline


def _entry(
    proposal: Proposal,
    status: LedgerStatus,
    reason: str | None,
    result: EvaluationResult | None = None,
    metric: str | None = None,
    seconds: float = 0.0,
    cached: bool = False,
) -> LedgerEntry:
    candidate = proposal.candidate
    return LedgerEntry(
        candidate_id=candidate.candidate_id,
        label=candidate.label,
        family=candidate.family,
        strategy=candidate.strategy,
        stage=proposal.stage,
        status=status,
        parent=proposal.parent,
        reason=reason,
        score=None if result is None or metric is None else result.mean(metric),
        coverage=None if result is None or metric is None else result.coverage(metric),
        n_failed_units=0 if result is None else sum(t.status != "SUCCEEDED" for t in result.trials),
        seconds=seconds,
        cached=cached,
    )
