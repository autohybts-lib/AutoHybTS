"""Cache of candidate evaluations (§8.4).

The key covers everything that can change an evaluation: candidate, plan fingerprint
(data and splits), layer, evaluation protocol (metrics, seasonal period, residual
settings), session seed and package version. Anything else changing means a new key,
so stale results are never reused.
"""

from __future__ import annotations

from autohybts.domain.identity import stable_hash
from autohybts.domain.results import EvaluationResult

__all__ = ["MemoryTrialCache", "evaluation_key"]


def evaluation_key(
    candidate_id: str,
    plan_fingerprint: str,
    layer: str | None,
    protocol: dict[str, object],
    session_seed: int,
    code_version: str,
) -> str:
    return stable_hash(
        {
            "schema": "evaluation/1",
            "candidate_id": candidate_id,
            "plan": plan_fingerprint,
            "layer": layer,
            "protocol": protocol,
            "seed": session_seed,
            "code": code_version,
        }
    )


class MemoryTrialCache:
    """In-process cache; a persistent store comes with experiment persistence (I8)."""

    def __init__(self) -> None:
        self._store: dict[str, EvaluationResult] = {}

    def get(self, key: str) -> EvaluationResult | None:
        return self._store.get(key)

    def put(self, key: str, result: EvaluationResult) -> None:
        self._store[key] = result

    def __len__(self) -> int:
        return len(self._store)
