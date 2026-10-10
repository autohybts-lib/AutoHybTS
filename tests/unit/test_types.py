from __future__ import annotations

from autohybts import types


def test_runtime_tuples_match_the_literals() -> None:
    assert types.PROTOCOL_VARIANTS == ("A", "B")
    assert types.TRIAL_STATUSES == (
        "PENDING",
        "RUNNING",
        "SUCCEEDED",
        "FAILED",
        "SKIPPED",
        "TIMEOUT",
    )
    assert types.RESIDUAL_SOURCES == (
        "in_sample",
        "historical_oof_1step",
        "historical_oof_multistep",
    )
    assert "research_strict" in types.PROFILES
