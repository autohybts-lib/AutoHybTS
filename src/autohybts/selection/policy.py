"""Acceptance policy: paired indicators, thresholds and profiles (§9.2, §9.3, §30.3).

Defaults follow the received protocol: median gain >= 1 %, win rate >= 60 % and, when
enabled, variability >= 0.05 as defined in ADR-0003. These thresholds are protocol rules,
not significance tests (§30.2).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, TypeAlias

import numpy as np

from autohybts.domain.results import EvaluationResult
from autohybts.exceptions import ConfigurationError, ProtocolConfigurationError
from autohybts.types import PROFILES, Profile

__all__ = [
    "DEFAULT_FAMILY_ORDER",
    "PairedComparison",
    "SelectionPolicy",
    "VariabilityCriterion",
    "paired_comparison",
]

DEFAULT_FAMILY_ORDER: Final = (
    "statistical",
    "linear",
    "kernel",
    "tree",
    "neural",
    "residual",
    "decomposition",
    "combination",
    "dynamic",
)
Direction: TypeAlias = Literal["gte", "lte"]


@dataclass(frozen=True, slots=True)
class VariabilityCriterion:
    """Variability predicate of R09 as a named, versioned plugin (ADR-0003)."""

    name: str = "correction_std_ratio"
    version: int = 1
    threshold: float = 0.05
    direction: Direction = "gte"
    aggregation: str = "median"

    def __post_init__(self) -> None:
        if self.direction not in ("gte", "lte"):
            raise ConfigurationError("direction deve ser 'gte' ou 'lte'.")
        if self.aggregation != "median":
            raise ConfigurationError("Só a agregação 'median' está definida (ADR-0003).")

    def applies_to(self, result: EvaluationResult) -> bool:
        """Only challengers with a corrector have a correction to measure."""

        return result.candidate.corrector is not None

    def value(self, result: EvaluationResult, unit_ids: Sequence[str]) -> float:
        """Median over the compared units of the per-unit diagnostic; NaN if none is finite.

        With several series the median is taken per series first, then across series;
        with one series this reduces to the median over its units.
        """

        wanted = set(unit_ids)
        values = [
            trial.diagnostics.get(self.name, math.nan)
            for trial in result.trials
            if trial.unit_id in wanted
        ]
        finite = [v for v in values if math.isfinite(v)]
        return float(np.median(finite)) if finite else math.nan

    def passes(self, value: float) -> bool:
        if not math.isfinite(value):
            return False
        return value >= self.threshold if self.direction == "gte" else value <= self.threshold

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "threshold": self.threshold,
            "direction": self.direction,
            "aggregation": self.aggregation,
        }


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """Profile and thresholds of the incremental guarded selection.

    Profiles:

    * ``quickstart``: variability off by default, declared in every decision.
    * ``standard``: protocol thresholds with variability (ADR-0003) on.
    * ``research_strict``: like ``standard``; refuses to run without a variability
      criterion or with ``min_coverage < 1``.
    * ``exploratory``: decisions are flagged as exploratory, never as unbiased.
    """

    profile: Profile = "standard"
    median_gain_threshold: float = 0.01
    win_rate_threshold: float = 0.60
    variability: VariabilityCriterion | None = field(default_factory=VariabilityCriterion)
    min_coverage: float = 1.0
    epsilon: float = 1e-12
    family_order: tuple[str, ...] = DEFAULT_FAMILY_ORDER

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise ConfigurationError(f"Perfil desconhecido {self.profile!r}; opções: {PROFILES}.")
        if not 0.0 < self.min_coverage <= 1.0:
            raise ConfigurationError("min_coverage deve estar em (0, 1].")
        if not 0.0 <= self.win_rate_threshold <= 1.0:
            raise ConfigurationError("win_rate_threshold deve estar em [0, 1].")
        if not self.epsilon > 0:
            raise ConfigurationError("epsilon deve ser positivo.")
        if len(set(self.family_order)) != len(self.family_order) or not self.family_order:
            raise ConfigurationError("family_order deve listar famílias sem repetição.")
        if self.profile == "research_strict":
            if self.variability is None:
                raise ProtocolConfigurationError(
                    "research_strict exige o critério de variabilidade definido (R09, ADR-0003)."
                )
            if self.min_coverage < 1.0:
                raise ProtocolConfigurationError("research_strict exige cobertura completa.")

    @classmethod
    def quickstart(cls, **overrides: object) -> SelectionPolicy:
        return cls(profile="quickstart", variability=None, **overrides)  # type: ignore[arg-type]

    @property
    def exploratory(self) -> bool:
        return self.profile == "exploratory"

    def describe(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "median_gain_threshold": self.median_gain_threshold,
            "win_rate_threshold": self.win_rate_threshold,
            "variability": None if self.variability is None else self.variability.describe(),
            "min_coverage": self.min_coverage,
            "epsilon": self.epsilon,
            "family_order": list(self.family_order),
        }


@dataclass(frozen=True, slots=True)
class PairedComparison:
    unit_ids: tuple[str, ...]
    champion_errors: tuple[float, ...]
    challenger_errors: tuple[float, ...]
    gains: tuple[float, ...]
    median_gain: float
    win_rate: float


def paired_comparison(
    champion: EvaluationResult, challenger: EvaluationResult, metric: str, epsilon: float
) -> PairedComparison:
    """Indicators of §9.2 on the units where both are valid.

    ``g_i = (E_c - E_n) / max(|E_c|, epsilon)``; ties are not wins.
    """

    champion_scores = champion.scores(metric)
    challenger_scores = challenger.scores(metric)
    units = [
        unit
        for unit in champion_scores.index
        if unit in challenger_scores.index
        and math.isfinite(champion_scores[unit])
        and math.isfinite(challenger_scores[unit])
    ]
    if not units:
        return PairedComparison((), (), (), (), math.nan, math.nan)
    current = np.array([champion_scores[u] for u in units], dtype=np.float64)
    new = np.array([challenger_scores[u] for u in units], dtype=np.float64)
    gains = (current - new) / np.maximum(np.abs(current), epsilon)
    return PairedComparison(
        unit_ids=tuple(units),
        champion_errors=tuple(current.tolist()),
        challenger_errors=tuple(new.tolist()),
        gains=tuple(gains.tolist()),
        median_gain=float(np.median(gains)),
        win_rate=float(np.mean(new < current)),
    )
