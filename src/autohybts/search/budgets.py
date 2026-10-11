"""Search budgets and anti-overfitting guards (§8.3, §30.4).

A *trial* here is one candidate evaluated on every origin of ``V_search``; the per-origin
results it contains are units, not separate trials. Budgets limit the number of
candidates looked at, which is what limits how much the search can adapt to the
validation noise (Cawley & Talbot, 2010).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Final, Literal, TypeAlias

from autohybts.exceptions import ConfigurationError

__all__ = ["PRESETS", "BudgetPreset", "SearchGuardConfig"]

BudgetPreset: TypeAlias = Literal["smoke", "small", "medium", "large"]
GuardMode: TypeAlias = Literal["research_strict", "standard", "exploratory"]


@dataclass(frozen=True, slots=True)
class SearchGuardConfig:
    mode: GuardMode = "standard"
    max_trials_total: int = 200
    max_trials_per_family: int = 40
    max_candidates_per_stage: int = 60
    max_refinements: int = 2
    max_wall_time_s: float | None = None
    top_k_per_family: int = 2
    top_k_residual_structures: int = 2
    require_full_pairing: bool = True
    prefer_simpler_on_tie: bool = True
    # Kept in the contract only to make the invariant explicit; True is refused.
    allow_test_feedback_to_search: bool = False

    def __post_init__(self) -> None:
        if self.allow_test_feedback_to_search:
            raise ConfigurationError(
                "allow_test_feedback_to_search deve permanecer False: o teste selado nunca "
                "alimenta a busca."
            )
        if self.mode not in ("research_strict", "standard", "exploratory"):
            raise ConfigurationError(f"Modo de busca desconhecido {self.mode!r}.")
        for name in (
            "max_trials_total",
            "max_trials_per_family",
            "max_candidates_per_stage",
            "top_k_per_family",
            "top_k_residual_structures",
        ):
            _positive(name, getattr(self, name), minimum=1)
        _positive("max_refinements", self.max_refinements, minimum=0)
        if self.max_wall_time_s is not None and not self.max_wall_time_s > 0:
            raise ConfigurationError("max_wall_time_s deve ser positivo ou None.")

    @classmethod
    def preset(cls, name: BudgetPreset, **overrides: Any) -> SearchGuardConfig:
        if name not in PRESETS:
            raise ConfigurationError(f"Preset desconhecido {name!r}; opções: {sorted(PRESETS)}.")
        return replace(cls(**PRESETS[name]["guard"]), **overrides)

    def describe(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _positive(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{name} deve ser inteiro >= {minimum}; recebido {value!r}.")


# Engineering starting points (§8.3), not experimental results; calibrate per study.
# Presets limit the search by *count* only, so the same data and seed give the same search
# on any machine. A wall-clock deadline is opt-in (``max_wall_time_s``, cooperative), and
# the number of origins per validation layer is the experiment's ``fold``, not the budget:
# it sets how many paired units the acceptance rule compares (§9.3).
PRESETS: Final[dict[str, dict[str, Any]]] = {
    "smoke": {
        "guard": {
            "max_trials_total": 10,
            "max_trials_per_family": 4,
            "max_candidates_per_stage": 6,
            "max_refinements": 1,
        },
    },
    "small": {
        "guard": {
            "max_trials_total": 50,
            "max_trials_per_family": 15,
            "max_candidates_per_stage": 25,
            "max_refinements": 2,
        },
    },
    "medium": {
        "guard": {
            "max_trials_total": 200,
            "max_trials_per_family": 40,
            "max_candidates_per_stage": 60,
            "max_refinements": 2,
        },
    },
    "large": {
        "guard": {
            "max_trials_total": 1000,
            "max_trials_per_family": 200,
            "max_candidates_per_stage": 300,
            "max_refinements": 3,
        },
    },
}
