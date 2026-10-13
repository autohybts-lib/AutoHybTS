"""Incremental guarded selection on V_select (§9.4, §13.4, §30.1).

1. Finalists: for each family, the best structure found on ``V_search`` (ranking
   convention v1). They are re-evaluated on ``V_select``; the search never saw those
   targets.
2. The initial champion is the best standalone finalist with full ``V_select`` coverage.
3. Families are visited in ``family_order``; each family's finalist challenges the
   *current* champion and replaces it only if every enabled predicate passes. Rejections
   are recorded with their reasons; the champion is never replaced by default.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from autohybts.data.panel import PanelPlan
from autohybts.data.splitter import TemporalPlan, ensure_not_sealed
from autohybts.domain.identity import stable_hash
from autohybts.domain.results import EvaluationResult
from autohybts.domain.specs import CandidateSpec
from autohybts.evaluation.temporal import TemporalEvaluator
from autohybts.exceptions import ConfigurationError, CoverageError
from autohybts.search.ledger import SearchResult
from autohybts.selection.policy import SelectionPolicy, paired_comparison, paired_p_value
from autohybts.selection.rank import RANKING_CONVENTION, RankedStructure, RankingConvention

__all__ = [
    "IncrementalSelector",
    "SelectionAudit",
    "SelectionDecision",
    "SelectionResult",
    "decide",
    "run_chain",
]


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    """One challenger against the champion of that moment; complete even when rejected."""

    family: str
    champion_before: str
    challenger: str
    challenger_label: str
    accepted: bool
    reason_codes: tuple[str, ...]
    metric: str
    unit_ids: tuple[str, ...]
    n_expected_units: int
    challenger_coverage: float
    median_gain: float
    win_rate: float
    variability_status: str
    variability_value: float | None
    thresholds: dict[str, object] = field(default_factory=dict)
    # Short names (``arima-mlp``) of both sides, for tables and plots.
    champion_before_name: str = ""
    challenger_name: str = ""
    # One-sided paired Wilcoxon p-value; None when the significance predicate is off.
    p_value: float | None = None
    # Variability per series (NaN: no finite diagnostic, ignored by the aggregate);
    # empty when the criterion is disabled or not applicable.
    variability_by_series: Mapping[str, float] = field(default_factory=dict)
    # Paired errors and gains per compared unit, aligned with ``unit_ids``.
    champion_errors: tuple[float, ...] = ()
    challenger_errors: tuple[float, ...] = ()
    gains: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class SelectionAudit:
    """Evidence of how adaptive the selection was: trials, pruning, finalists, reasons.

    ``selection_gap()`` compares the winner's score on ``V_search`` with its score on
    ``V_select``: a large gap signals optimism from choosing where it was searched.
    """

    protocol_hash: str
    search_protocol_hash: str
    total_trials: int
    tried_by_family: dict[str, int]
    pruned_trials: int
    failed_trials: int
    finalist_ids: tuple[str, ...]
    winner_id: str
    comparison_origin_ids: tuple[str, ...]
    decision_reasons: tuple[str, ...]
    exploratory: bool
    search_best_score: float
    winner_search_score: float
    winner_select_score: float

    @property
    def selection_gap(self) -> float:
        """Select-layer minus search-layer score of the winner (positive = optimism)."""

        return self.winner_select_score - self.winner_search_score


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Outcome of the incremental selection on ``V_select``.

    ``champion`` is the final champion and ``champion_result`` its evaluation;
    ``finalists`` holds the best structure of each family, evaluated on ``V_select``;
    ``decisions`` records every challenge, accepted or not, with per-unit errors and gains
    (``decisions_frame()`` and ``decision_units_frame()`` give them as tables); ``audit``
    is the ``SelectionAudit``; in a panel, ``champion_per_series`` holds the
    configuration chosen for each series.
    """

    champion: CandidateSpec
    champion_result: EvaluationResult
    finalists: tuple[EvaluationResult, ...]
    decisions: tuple[SelectionDecision, ...]
    audit: SelectionAudit
    # Configuration of the champion structure in each series (best per series, ADR-0004).
    champion_per_series: Mapping[str, CandidateSpec] = field(default_factory=dict)

    def decisions_frame(self) -> pd.DataFrame:
        """One row per decision; models by short name, full label and ids at the end."""

        columns = [
            "family",
            "champion_before",
            "challenger",
            "accepted",
            "reason_codes",
            "median_gain",
            "win_rate",
            "variability_status",
            "variability_value",
            "p_value",
            "challenger_coverage",
            "challenger_label",
            "champion_before_id",
            "challenger_id",
        ]
        rows = [
            {
                "family": d.family,
                "champion_before": d.champion_before_name or d.champion_before,
                "challenger": d.challenger_name or d.challenger,
                "accepted": d.accepted,
                "reason_codes": d.reason_codes,
                "median_gain": d.median_gain,
                "win_rate": d.win_rate,
                "variability_status": d.variability_status,
                "variability_value": d.variability_value,
                "p_value": d.p_value,
                "challenger_coverage": d.challenger_coverage,
                "challenger_label": d.challenger_label,
                "champion_before_id": d.champion_before,
                "challenger_id": d.challenger,
            }
            for d in self.decisions
        ]
        return pd.DataFrame(rows, columns=columns)

    def decision_units_frame(self) -> pd.DataFrame:
        """One row per decision and compared unit: both errors, the gain and the win.

        Everything a decision used is here, so median gain and win rate can be recomputed
        (and resampled by unit or series) outside the library.
        """

        columns = [
            "family",
            "challenger_id",
            "accepted",
            "unit_id",
            "series_id",
            "champion_error",
            "challenger_error",
            "gain",
            "win",
        ]
        rows = [
            {
                "family": d.family,
                "challenger_id": d.challenger,
                "accepted": d.accepted,
                "unit_id": unit,
                "series_id": unit.rsplit(":", 1)[0] if ":" in unit else None,
                "champion_error": old,
                "challenger_error": new,
                "gain": gain,
                "win": new < old,
            }
            for d in self.decisions
            for unit, old, new, gain in zip(
                d.unit_ids, d.champion_errors, d.challenger_errors, d.gains, strict=True
            )
        ]
        return pd.DataFrame(rows, columns=columns)


class IncrementalSelector:
    def __init__(
        self,
        evaluator: TemporalEvaluator,
        policy: SelectionPolicy,
        ranking: RankingConvention = RANKING_CONVENTION,
    ) -> None:
        self.evaluator = evaluator
        self.policy = policy
        self.ranking = ranking

    @property
    def metric(self) -> str:
        return self.evaluator.metrics[0]

    def select(self, search: SearchResult, plan: TemporalPlan | PanelPlan) -> SelectionResult:
        ensure_not_sealed(search, plan)
        if search.plan_fingerprint != plan.fingerprint:
            raise ConfigurationError("A busca foi feita em outro plano temporal.")
        if search.metric != self.metric:
            raise ConfigurationError(
                f"Busca ordenada por {search.metric}, seleção por {self.metric}: incomparáveis."
            )
        structures = self._finalists(search)
        finalists_search = {family: s.representative for family, s in structures.items()}
        finalists = {
            family: self._evaluate_select(structure, plan)
            for family, structure in structures.items()
        }

        initial = self._initial_champion(structures, finalists)
        champion, decisions = run_chain(initial, finalists, self.policy, self.metric)

        audit = self._audit(search, finalists_search, finalists, champion, decisions)
        winner = structures[champion.candidate.family]
        return SelectionResult(
            champion=champion.candidate,
            champion_result=champion,
            finalists=tuple(finalists[f] for f in finalists),
            decisions=tuple(decisions),
            audit=audit,
            champion_per_series=dict(winner.per_series_candidates),
        )

    def _evaluate_select(
        self, structure: RankedStructure, plan: TemporalPlan | PanelPlan
    ) -> EvaluationResult:
        if isinstance(plan, PanelPlan):
            return self.evaluator.evaluate_structure(
                structure.per_series_candidates,
                plan,
                "select",
                representative=structure.representative.candidate,
            )
        return self.evaluator.evaluate(structure.representative.candidate, plan, "select")

    def _finalists(self, search: SearchResult) -> dict[str, RankedStructure]:
        full = self.policy.min_coverage >= 1.0
        best: dict[str, RankedStructure] = {}
        ranked = self.ranking.rank(list(search.results), self.metric, full_pairing=full)
        for structure in ranked:
            best.setdefault(structure.representative.candidate.family, structure)
        unknown = set(best) - set(self.policy.family_order)
        if unknown:
            raise ConfigurationError(
                f"Famílias {sorted(unknown)} fora de family_order {self.policy.family_order}."
            )
        return best

    def _initial_champion(
        self, structures: dict[str, RankedStructure], finalists: dict[str, EvaluationResult]
    ) -> EvaluationResult:
        standalone = sorted(
            (s for s in structures.values() if s.representative.candidate.strategy == "standalone"),
            key=RankedStructure.order,
        )
        for structure in standalone:
            result = finalists[structure.representative.candidate.family]
            if result.coverage(self.metric) >= self.policy.min_coverage:
                return result
        raise CoverageError(
            "Nenhuma base individual finalista tem cobertura suficiente em V_select."
        )

    def _audit(
        self,
        search: SearchResult,
        by_search: dict[str, EvaluationResult],
        finalists: dict[str, EvaluationResult],
        champion: EvaluationResult,
        decisions: Sequence[SelectionDecision],
    ) -> SelectionAudit:
        counts = search.counts()
        winner_family = champion.candidate.family
        protocol = {
            "search": search.protocol.get("protocol_hash"),
            "policy": self.policy.describe(),
            "ranking": self.ranking.describe(),
            "evaluator": self.evaluator.protocol(),
        }
        return SelectionAudit(
            protocol_hash=stable_hash(protocol),
            search_protocol_hash=str(search.protocol.get("protocol_hash", "")),
            total_trials=search.trial_count,
            tried_by_family=search.trials_by_family,
            pruned_trials=counts.get("PRUNED_BUDGET", 0),
            failed_trials=sum(
                any(t.status != "SUCCEEDED" for t in r.trials) for r in search.results
            ),
            finalist_ids=tuple(r.candidate_id for r in finalists.values()),
            winner_id=champion.candidate_id,
            comparison_origin_ids=champion.unit_ids,
            decision_reasons=tuple(f"{d.family}: {'+'.join(d.reason_codes)}" for d in decisions),
            exploratory=self.policy.exploratory,
            search_best_score=min(r.mean(self.metric) for r in by_search.values()),
            winner_search_score=by_search[winner_family].mean(self.metric),
            winner_select_score=champion.mean(self.metric),
        )


def decide(
    family: str,
    champion: EvaluationResult,
    challenger: EvaluationResult,
    policy: SelectionPolicy,
    metric: str,
) -> SelectionDecision:
    """Accept ``challenger`` only if every enabled predicate passes (§9.3); never by default."""

    reasons: list[str] = []
    expected = list(champion.unit_ids)
    if list(challenger.unit_ids) != expected:
        reasons.append("different_units")
    coverage = challenger.coverage(metric)
    if coverage < policy.min_coverage:
        reasons.append("coverage_incomplete")
    paired = paired_comparison(champion, challenger, metric, policy.epsilon)
    if not paired.unit_ids:
        reasons.append("no_comparable_units")
    else:
        if not paired.median_gain >= policy.median_gain_threshold:
            reasons.append("median_gain_below_threshold")
        if not paired.win_rate >= policy.win_rate_threshold:
            reasons.append("win_rate_below_threshold")
    p_value: float | None = None
    if policy.significance_alpha is not None and paired.unit_ids:
        p_value = paired_p_value(paired)
        if not p_value < policy.significance_alpha:
            reasons.append("not_significant")

    criterion = policy.variability
    value: float | None = None
    by_series: dict[str, float] = {}
    if criterion is None:
        status = "disabled"
    elif not criterion.applies_to(challenger):
        status = "not_applicable"
    else:
        by_series = criterion.per_series(challenger, paired.unit_ids)
        value = criterion.value(challenger, paired.unit_ids)
        if not math.isfinite(value):
            status = "undefined"
            reasons.append("variability_undefined")
        elif criterion.passes(value):
            status = "passed"
        else:
            status = "failed"
            reasons.append("variability_below_threshold")

    return SelectionDecision(
        family=family,
        champion_before=champion.candidate_id,
        challenger=challenger.candidate_id,
        challenger_label=challenger.candidate.label,
        accepted=not reasons,
        reason_codes=tuple(reasons) or ("accepted",),
        metric=metric,
        unit_ids=paired.unit_ids,
        n_expected_units=len(expected),
        challenger_coverage=coverage,
        median_gain=paired.median_gain,
        win_rate=paired.win_rate,
        variability_status=status,
        variability_value=value,
        thresholds=policy.describe(),
        champion_before_name=champion.candidate.name,
        challenger_name=challenger.candidate.name,
        p_value=p_value,
        variability_by_series=by_series,
        champion_errors=paired.champion_errors,
        challenger_errors=paired.challenger_errors,
        gains=paired.gains,
    )


def run_chain(
    initial: EvaluationResult,
    finalists: dict[str, EvaluationResult],
    policy: SelectionPolicy,
    metric: str,
) -> tuple[EvaluationResult, list[SelectionDecision]]:
    """Visit families in ``policy.family_order``; each finalist challenges the current champion."""

    champion = initial
    decisions: list[SelectionDecision] = []
    for family in policy.family_order:
        challenger = finalists.get(family)
        if challenger is None or challenger.candidate_id == champion.candidate_id:
            continue
        decision = decide(family, champion, challenger, policy, metric)
        decisions.append(decision)
        if decision.accepted:
            champion = challenger
    return champion, decisions
