"""Audit trail of the search: every proposal, evaluated or not (§30.2, T26)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal, TypeAlias

import pandas as pd

from autohybts.domain.results import EvaluationResult

__all__ = ["LedgerEntry", "LedgerStatus", "SearchResult"]

LedgerStatus: TypeAlias = Literal[
    "EVALUATED", "PRUNED_BUDGET", "SKIPPED_CAPABILITY", "DUPLICATE", "STOPPED"
]


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    candidate_id: str
    label: str
    family: str
    strategy: str
    stage: str
    status: LedgerStatus
    parent: str | None = None
    reason: str | None = None
    score: float | None = None
    coverage: float | None = None
    n_failed_units: int = 0
    seconds: float = 0.0
    cached: bool = False


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Everything the search tried and what it found, on ``V_search`` only."""

    metric: str
    plan_fingerprint: str
    ledger: tuple[LedgerEntry, ...]
    results: tuple[EvaluationResult, ...]
    retained: tuple[str, ...]
    residual_structures: tuple[str, ...]
    stop_reason: str
    protocol: dict[str, object]

    @property
    def trial_count(self) -> int:
        return sum(entry.status == "EVALUATED" for entry in self.ledger)

    @property
    def trials_by_family(self) -> dict[str, int]:
        return dict(Counter(entry.family for entry in self.ledger if entry.status == "EVALUATED"))

    def counts(self) -> dict[str, int]:
        return dict(Counter(entry.status for entry in self.ledger))

    def result(self, candidate_id: str) -> EvaluationResult:
        for result in self.results:
            if result.candidate_id == candidate_id:
                return result
        raise KeyError(candidate_id)

    def ledger_frame(self) -> pd.DataFrame:
        return pd.DataFrame([_row(entry) for entry in self.ledger])


def _row(entry: LedgerEntry) -> dict[str, object]:
    return {name: getattr(entry, name) for name in entry.__dataclass_fields__}
