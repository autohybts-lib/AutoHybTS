"""Shared literal types and the runtime tuples used to validate them."""

from __future__ import annotations

from typing import Final, Literal, TypeAlias, get_args

__all__ = [
    "PROFILES",
    "PROTOCOL_VARIANTS",
    "RESIDUAL_SOURCES",
    "TRIAL_STATUSES",
    "Profile",
    "ProtocolVariant",
    "ResidualSource",
    "TrialStatus",
]

# A: open multi-step forecast; B: rolling one-step with frozen parameters (§7.3-7.4).
ProtocolVariant: TypeAlias = Literal["A", "B"]

TrialStatus: TypeAlias = Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "SKIPPED", "TIMEOUT"]

# in_sample is allowed only as an explicitly labelled experimental source (§7.2).
ResidualSource: TypeAlias = Literal["in_sample", "historical_oof_1step", "historical_oof_multistep"]

Profile: TypeAlias = Literal["quickstart", "standard", "research_strict", "exploratory"]

PROTOCOL_VARIANTS: Final[tuple[ProtocolVariant, ...]] = get_args(ProtocolVariant)
TRIAL_STATUSES: Final[tuple[TrialStatus, ...]] = get_args(TrialStatus)
RESIDUAL_SOURCES: Final[tuple[ResidualSource, ...]] = get_args(ResidualSource)
PROFILES: Final[tuple[Profile, ...]] = get_args(Profile)
