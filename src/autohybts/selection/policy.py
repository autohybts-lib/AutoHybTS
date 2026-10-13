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
    "paired_p_value",
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
    """Variability predicate of R09 as a named, versioned plugin (ADR-0003).

    Version 2 aggregates per series first (median over the series' compared units), then
    across series (median of the series values). Version 1 took a single median over all
    units, which differs in panels; single-series values are the same in both versions.
    """

    name: str = "correction_std_ratio"
    version: int = 2
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

    def per_series(self, result: EvaluationResult, unit_ids: Sequence[str]) -> dict[str, float]:
        """Median of the per-unit diagnostic over each series' compared units.

        Every series with a compared unit is listed; a series without any finite
        diagnostic is ``NaN`` (kept visible in the decision, ignored by ``value``).
        """

        wanted = set(unit_ids)
        grouped: dict[str, list[float]] = {}
        for trial in result.trials:
            if trial.unit_id in wanted:
                grouped.setdefault(trial.series_id, []).append(
                    trial.diagnostics.get(self.name, math.nan)
                )
        return {series: _finite_median(values) for series, values in sorted(grouped.items())}

    def value(self, result: EvaluationResult, unit_ids: Sequence[str]) -> float:
        """Median across series of the per-series medians; NaN if no series is finite.

        With one series this is the median over its compared units (ADR-0003).
        """

        return _finite_median(list(self.per_series(result, unit_ids).values()))

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


def _finite_median(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(np.median(finite)) if finite else math.nan


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """Profile and thresholds of the incremental guarded selection.

    Profiles:

    * ``quickstart``: variability off by default, declared in every decision.
    * ``standard``: protocol thresholds with variability (ADR-0003) on.
    * ``research_strict``: like ``standard``; refuses to run without a variability
      criterion or with ``min_coverage < 1``.
    * ``exploratory``: decisions are flagged as exploratory, never as unbiased.

    ``significance_alpha`` (off by default) adds a statistical predicate: the challenger
    must also beat the champion in a one-sided paired Wilcoxon signed-rank test on the same
    units, ``p < significance_alpha``. The protocol rule alone is not error control. The
    test bounds the false acceptance of *one* comparison at ``alpha``, under its
    assumptions (independent, symmetric paired differences); it does not control the
    whole chain, where several families challenge a champion that changes with the
    earlier decisions, nor the dependence between overlapping origins. Measure the
    chain's behaviour with ``autohybts.benchmarks.calibration``.

    Units with equal errors are dropped before the test, so ``n`` is the number of
    non-zero paired differences (all series of a panel together). With the exact null
    distribution the smallest one-sided p-value is ``1 / 2**n``: at least 5 such units
    for ``alpha = 0.05``; with ``fold=3`` on one series the predicate can never pass.
    The p-value comes from SciPy's default method, which depends on the SciPy version
    (exact for small ``n`` in recent versions, a normal approximation in some cases), so
    it can move slightly between environments.
    """

    profile: Profile = "standard"
    # 5% since the D5 calibration (ADR-0008); the architecture's §9.3 had 1%.
    median_gain_threshold: float = 0.05
    win_rate_threshold: float = 0.60
    variability: VariabilityCriterion | None = field(default_factory=VariabilityCriterion)
    min_coverage: float = 1.0
    epsilon: float = 1e-12
    family_order: tuple[str, ...] = DEFAULT_FAMILY_ORDER
    significance_alpha: float | None = None

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise ConfigurationError(f"Perfil desconhecido {self.profile!r}; opções: {PROFILES}.")
        if not 0.0 < self.min_coverage <= 1.0:
            raise ConfigurationError("min_coverage deve estar em (0, 1].")
        if not 0.0 <= self.win_rate_threshold <= 1.0:
            raise ConfigurationError("win_rate_threshold deve estar em [0, 1].")
        if not self.epsilon > 0:
            raise ConfigurationError("epsilon deve ser positivo.")
        if self.significance_alpha is not None and not 0.0 < self.significance_alpha < 1.0:
            raise ConfigurationError("significance_alpha deve estar em (0, 1) ou ser None.")
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
        described: dict[str, object] = {
            "profile": self.profile,
            "median_gain_threshold": self.median_gain_threshold,
            "win_rate_threshold": self.win_rate_threshold,
            "variability": None if self.variability is None else self.variability.describe(),
            "min_coverage": self.min_coverage,
            "epsilon": self.epsilon,
            "family_order": list(self.family_order),
        }
        # Only when used, so snapshots of earlier selections keep their content.
        if self.significance_alpha is not None:
            described["significance"] = {
                "test": "wilcoxon_one_sided",
                "alpha": self.significance_alpha,
            }
        return described


@dataclass(frozen=True, slots=True)
class PairedComparison:
    unit_ids: tuple[str, ...]
    champion_errors: tuple[float, ...]
    challenger_errors: tuple[float, ...]
    gains: tuple[float, ...]
    median_gain: float
    win_rate: float


def paired_p_value(paired: PairedComparison) -> float:
    """One-sided Wilcoxon signed-rank p-value that the challenger errs less (1 if no evidence)."""

    differences = np.asarray(paired.champion_errors) - np.asarray(paired.challenger_errors)
    differences = differences[differences != 0]
    if differences.size == 0:
        return 1.0
    from scipy import stats

    return float(stats.wilcoxon(differences, alternative="greater").pvalue)


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
