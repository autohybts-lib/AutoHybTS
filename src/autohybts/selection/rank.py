"""Ranking convention v1 (ADR-0004), for one series or a panel.

Per series, each structure is represented by its best configuration; a structure without
a valid result for every unit of some series is ineligible. Structures are ranked per
series by score (ties get the average rank) and the ranks are averaged across series.
Ordering then follows §9.4: mean rank, aggregated error, coverage (descending), cost and
finally ``candidate_id``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import pandas as pd

from autohybts.domain.results import EvaluationResult, combine_results
from autohybts.domain.specs import CandidateSpec

__all__ = [
    "RANKING_CONVENTION",
    "RankedStructure",
    "RankingConvention",
    "eligible",
    "structure_key",
]


def structure_key(candidate: CandidateSpec) -> tuple[str, ...]:
    """Same strategy and components; hyper-parameters, window, alpha and source may vary.

    Extra stages and combination members are components; the decomposition method and
    the combination weighting are part of the structure too.
    """

    corrector = candidate.corrector.name if candidate.corrector is not None else ""
    key = (candidate.strategy, candidate.base.name, corrector)
    extra = tuple(member.name for member in candidate.members)
    options = candidate.options
    for name in ("method", "weights"):
        if name in options:
            extra += (f"{name}={options[name]}",)
    if candidate.components:
        # Compositions: the structure is the set of component structures.
        parts = sorted("/".join(structure_key(c)) for c in candidate.components)
        extra += (f"components=[{';'.join(parts)}]",)
    return key + extra


def eligible(result: EvaluationResult, metric: str, full_pairing: bool) -> bool:
    if not math.isfinite(result.mean(metric)):
        return False
    return result.coverage(metric) == 1.0 if full_pairing else result.coverage(metric) > 0


def _best_configs(
    results: Iterable[EvaluationResult], metric: str, full_pairing: bool
) -> dict[tuple[str, ...], EvaluationResult]:
    """Best eligible configuration of each structure in one series."""

    best: dict[tuple[str, ...], EvaluationResult] = {}
    for result in results:
        if not eligible(result, metric, full_pairing):
            continue
        key = structure_key(result.candidate)
        current = best.get(key)
        if current is None or (result.mean(metric), result.candidate_id) < (
            current.mean(metric),
            current.candidate_id,
        ):
            best[key] = result
    return best


@dataclass(frozen=True, slots=True)
class RankedStructure:
    """A structure at its best configuration in every series.

    ``representative`` combines those per-series results (its candidate is the most
    frequent best configuration); ``per_series`` keeps each series' own result.
    """

    key: tuple[str, ...]
    mean_rank: float
    score: float
    coverage: float
    cost: float
    representative: EvaluationResult
    per_series: Mapping[str, EvaluationResult] = field(default_factory=dict)

    @property
    def candidate_id(self) -> str:
        return self.representative.candidate_id

    @property
    def per_series_candidates(self) -> dict[str, CandidateSpec]:
        return {sid: result.candidate for sid, result in self.per_series.items()}

    def order(self) -> tuple[float, float, float, float, str]:
        return (self.mean_rank, self.score, -self.coverage, self.cost, self.candidate_id)


@dataclass(frozen=True, slots=True)
class RankingConvention:
    name: str = "best_config_per_series_mean_rank"
    version: int = 1

    def rank(
        self,
        results: Sequence[EvaluationResult] | Mapping[str, Sequence[EvaluationResult]],
        metric: str,
        *,
        full_pairing: bool = True,
    ) -> list[RankedStructure]:
        """Structures ordered best first.

        ``results`` is either a mapping series id -> per-series results, or a list of
        results that may cover several series (they are split by series here).
        """

        by_series = results if isinstance(results, Mapping) else _split_by_series(results)
        per_series = {
            sid: _best_configs(group, metric, full_pairing) for sid, group in by_series.items()
        }
        if not per_series:
            return []
        common = set.intersection(*(set(best) for best in per_series.values()))
        if not common:
            return []
        # Ranked by position: keys are tuples of varying length, not a pandas index.
        keys = sorted(common)
        ranks = pd.DataFrame(
            {
                sid: pd.Series([best[key].mean(metric) for key in keys]).rank(method="average")
                for sid, best in per_series.items()
            }
        )
        mean_rank = dict(zip(keys, ranks.mean(axis=1).to_numpy(dtype=float), strict=True))
        ranked = []
        for key in common:
            members = {sid: best[key] for sid, best in per_series.items()}
            parts = list(members.values())
            counts = Counter(part.candidate_id for part in parts)
            chosen = max(parts, key=lambda part: counts[part.candidate_id])
            representative = (
                parts[0]
                if len(parts) == 1
                else combine_results(parts, chosen.candidate, parts[0].plan_fingerprint)
            )
            ranked.append(
                RankedStructure(
                    key=key,
                    mean_rank=float(mean_rank[key]),
                    score=_mean([part.mean(metric) for part in parts]),
                    coverage=_mean([part.coverage(metric) for part in parts]),
                    cost=sum(t.fit_seconds + t.predict_seconds for p in parts for t in p.trials),
                    representative=representative,
                    per_series=members,
                )
            )
        return sorted(ranked, key=RankedStructure.order)

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "version": self.version}


def _split_by_series(
    results: Sequence[EvaluationResult],
) -> dict[str, list[EvaluationResult]]:
    series = list(dict.fromkeys(sid for result in results for sid in result.series_ids))
    if len(series) <= 1:
        return {series[0] if series else "": list(results)}
    return {sid: [result.for_series(sid) for result in results] for sid in series}


def _mean(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else math.inf


RANKING_CONVENTION: Final = RankingConvention()
