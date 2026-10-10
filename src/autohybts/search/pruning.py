"""Stage-wise retention: best configuration per structure, then top-k per family (ADR-0004).

Ordering comes from ranking convention v1, so a panel is ranked by mean rank across
series and a single series by its score.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from autohybts.domain.results import EvaluationResult
from autohybts.selection.rank import RANKING_CONVENTION, eligible, structure_key

__all__ = ["best_per_structure", "eligible", "retain_bases", "structure_key", "top_structures"]


def best_per_structure(
    results: Iterable[EvaluationResult], metric: str, full_pairing: bool = True
) -> list[EvaluationResult]:
    """Best eligible configuration of each structure, ordered best first."""

    ranked = RANKING_CONVENTION.rank(list(results), metric, full_pairing=full_pairing)
    return [structure.representative for structure in ranked]


def retain_bases(
    results: Sequence[EvaluationResult],
    metric: str,
    *,
    top_k: int,
    always: Sequence[str] = ("arima",),
    full_pairing: bool = True,
) -> list[EvaluationResult]:
    """S1: top-k structures per family, plus ``always`` (if eligible) and the overall best.

    Deduplicated by ``candidate_id`` (T17) and ordered best first.
    """

    ranked = best_per_structure(results, metric, full_pairing)
    position = {result.candidate_id: index for index, result in enumerate(ranked)}
    kept: dict[str, EvaluationResult] = {}
    per_family: dict[str, int] = {}
    for result in ranked:
        family = result.candidate.family
        if per_family.get(family, 0) < top_k:
            per_family[family] = per_family.get(family, 0) + 1
            kept[result.candidate_id] = result
    for result in ranked:
        if result.candidate.base.name in always:
            kept.setdefault(result.candidate_id, result)
    if ranked:
        kept.setdefault(ranked[0].candidate_id, ranked[0])
    return sorted(kept.values(), key=lambda r: position[r.candidate_id])


def top_structures(
    results: Sequence[EvaluationResult], metric: str, *, k: int, full_pairing: bool = True
) -> list[EvaluationResult]:
    """Best ``k`` structures (each at its best configuration)."""

    return best_per_structure(results, metric, full_pairing)[:k]
