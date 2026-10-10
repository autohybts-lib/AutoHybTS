"""Staged, budgeted search over ``V_search`` (§8.2, §30.2).

S0 evaluates every base; S1 retains the best structures per family; S2 evaluates the
residual grid on the retained bases; S3 refines the best residual structures one axis
at a time (source, corrector parameters, extra stages); S4, when enabled, evaluates
decompositions and fixed combinations of the retained bases. Budgets are applied to the
proposal list *before* a stage runs, in a fixed priority order, so the number of
evaluated candidates never exceeds the limits and does not depend on how fast or in
which order parallel evaluations finish.
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
from autohybts.search.budgets import SearchGuardConfig
from autohybts.search.cache import MemoryTrialCache, evaluation_key
from autohybts.search.ledger import LedgerEntry, LedgerStatus, SearchResult
from autohybts.search.pruning import eligible, retain_bases, top_structures
from autohybts.search.spaces import Proposal, SearchSpace

__all__ = ["SearchEngine"]

LAYER: Final[Layer] = "search"


@dataclass
class _State:
    started: float
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
    ) -> None:
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
            raise ConfigurationError("max_workers deve ser inteiro >= 1.")
        self.evaluator = evaluator
        self.space = space
        self.guard = guard
        self.cache = cache
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
        state = _State(started=self.clock())

        bases = self._stage(self.space.base_proposals(registry), plan, state)
        if not any(eligible(r, self.metric, self.guard.require_full_pairing) for r in bases):
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
        structures: list[EvaluationResult] = []
        if self.space.correctors and self.space.residual_windows and self.space.alphas:
            # Keep room for S3: refinement is the point of the staged search (§8.2).
            reserve = self.guard.top_k_residual_structures * self.guard.max_refinements
            residual = self._stage(
                self.space.residual_proposals(retained_specs, registry),
                plan,
                state,
                reserve=reserve + advanced_reserve,
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
                    reserve=advanced_reserve,
                    family_reserve=0,
                )
        if advanced:
            self._stage(advanced, plan, state)

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
        if self._out_of_time(state):
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
        for proposal, (result, cached, seconds) in zip(admitted, outcomes, strict=True):
            state.per_family[proposal.candidate.family] += 1
            state.results[result.candidate_id] = result
            state.ledger.append(
                _entry(proposal, "EVALUATED", None, result, self.metric, seconds, cached)
            )
            evaluated.append(result)
        return evaluated

    def _evaluate_all(
        self, proposals: Sequence[Proposal], plan: TemporalPlan | PanelPlan, state: _State
    ) -> list[tuple[EvaluationResult, bool, float]]:
        def one(proposal: Proposal) -> tuple[EvaluationResult, bool, float]:
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
            started = time.perf_counter()
            result = self.evaluator.evaluate(proposal.candidate, plan, LAYER)
            if self.cache is not None:
                self.cache.put(key, result)
            return result, False, time.perf_counter() - started

        if self.max_workers == 1:
            return [one(proposal) for proposal in proposals]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(one, proposals))  # map keeps submission order

    def _out_of_time(self, state: _State) -> bool:
        limit = self.guard.max_wall_time_s
        return limit is not None and self.clock() - state.started >= limit


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
