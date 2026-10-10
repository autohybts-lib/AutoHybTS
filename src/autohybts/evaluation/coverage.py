"""Coverage: which expected evaluation units a candidate actually scored (§9.1)."""

from __future__ import annotations

from dataclasses import dataclass

from autohybts.domain.results import EvaluationResult

__all__ = ["CoverageReport", "coverage_report"]


@dataclass(frozen=True, slots=True)
class CoverageReport:
    metric: str | None
    expected: int
    valid: int
    missing: tuple[str, ...]

    @property
    def coverage(self) -> float:
        return self.valid / self.expected if self.expected else 0.0

    @property
    def complete(self) -> bool:
        return self.expected > 0 and self.valid == self.expected


def coverage_report(result: EvaluationResult, metric: str | None = None) -> CoverageReport:
    """Missing units are reported by ``unit_id``; they are never filled or scored as zero."""

    missing = tuple(trial.unit_id for trial in result.trials if not trial.is_valid(metric))
    return CoverageReport(metric, result.n_expected, result.n_expected - len(missing), missing)
