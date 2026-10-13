"""Acceptance rules on hand-built results: T09-T11, T24, ADR-0003 boundaries, legacy parity."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from autohybts.domain.results import EvaluationResult, TrialResult
from autohybts.domain.specs import CandidateSpec, ModelSpec
from autohybts.exceptions import ConfigurationError, ProtocolConfigurationError
from autohybts.selection.incremental import decide, run_chain
from autohybts.selection.policy import SelectionPolicy, VariabilityCriterion, paired_comparison
from autohybts.selection.rank import RankingConvention

BASE = CandidateSpec.standalone("naive", family="statistical")
LINEAR = CandidateSpec.standalone("linear", family="linear")
HYBRID = CandidateSpec(
    "residual",
    "residual_additive",
    ModelSpec.of("naive"),
    ModelSpec.of("linear"),
    7,
    1.0,
    "historical_oof_1step",
)


def result(
    candidate: CandidateSpec,
    errors: list[float | None],
    ratio: float | None = None,
    layer: str = "select",
) -> EvaluationResult:
    """``None`` in ``errors`` is a failed unit."""

    trials = []
    for i, error in enumerate(errors):
        diagnostics = {} if ratio is None else {"correction_std_ratio": ratio}
        trials.append(
            TrialResult(
                candidate_id=candidate.candidate_id,
                series_id="s",
                fold_id=f"{layer}-{i}",
                layer=layer,
                origin_index=100 + i,
                origin_time=pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                horizon=1,
                status="FAILED" if error is None else "SUCCEEDED",
                trial_key=f"k{i}",
                metrics={} if error is None else {"MASE": error},
                diagnostics=diagnostics,
                reason="boom" if error is None else None,
            )
        )
    return EvaluationResult(candidate, "plan", "D", 1, ("MASE",), tuple(trials))


CHAMPION = result(BASE, [1.0, 1.0, 1.0, 1.0, 1.0])
STANDARD = SelectionPolicy()


def test_paired_indicators() -> None:
    challenger = result(LINEAR, [0.5, 1.0, 1.2, 0.9, 0.0])
    paired = paired_comparison(CHAMPION, challenger, "MASE", 1e-12)
    assert paired.gains == pytest.approx((0.5, 0.0, -0.2, 0.1, 1.0))
    assert paired.median_gain == pytest.approx(0.1)
    assert paired.win_rate == pytest.approx(3 / 5)  # the tie at unit 1 is not a win


def test_epsilon_protects_zero_champion_errors() -> None:
    zero = result(BASE, [0.0, 0.0])
    paired = paired_comparison(zero, result(LINEAR, [1e-13, 0.0]), "MASE", 1e-12)
    assert paired.gains == pytest.approx((-0.1, 0.0))


def test_all_enabled_predicates_must_pass() -> None:
    # T09: each failing predicate alone is enough to reject.
    good = decide("linear", CHAMPION, result(LINEAR, [0.9] * 5), STANDARD, "MASE")
    assert good.accepted
    assert good.reason_codes == ("accepted",)
    assert good.variability_status == "not_applicable"

    low_gain = decide("linear", CHAMPION, result(LINEAR, [0.995] * 5), STANDARD, "MASE")
    assert low_gain.reason_codes == ("median_gain_below_threshold",)

    # With 5 units a median gain >= 1 % already implies >= 3 wins (60 %); use 4 units.
    four = result(BASE, [1.0] * 4)
    few_wins = decide("linear", four, result(LINEAR, [0.5, 0.5, 1.0, 1.0]), STANDARD, "MASE")
    assert few_wins.reason_codes == ("win_rate_below_threshold",)

    inert = decide("residual", CHAMPION, result(HYBRID, [0.8] * 5, ratio=0.01), STANDARD, "MASE")
    assert inert.reason_codes == ("variability_below_threshold",)


def test_rejection_keeps_champion_even_with_better_mean_error() -> None:
    # T10: one huge win makes the mean better, but median gain and win rate fail.
    challenger = result(LINEAR, [0.01, 1.05, 1.05, 1.05, 1.05])
    assert sum(t.metrics["MASE"] for t in challenger.trials) < 5.0
    champion, decisions = run_chain(CHAMPION, {"linear": challenger}, STANDARD, "MASE")
    assert champion is CHAMPION
    assert not decisions[0].accepted
    assert set(decisions[0].reason_codes) == {
        "median_gain_below_threshold",
        "win_rate_below_threshold",
    }


def test_incomplete_coverage_is_ineligible() -> None:
    # T11: better on the units it covered, still rejected.
    strict = SelectionPolicy(profile="research_strict")
    decision = decide(
        "linear", CHAMPION, result(LINEAR, [0.1, 0.1, 0.1, 0.1, None]), strict, "MASE"
    )
    assert not decision.accepted
    assert "coverage_incomplete" in decision.reason_codes
    assert decision.challenger_coverage == pytest.approx(0.8)


def test_tie_preserves_the_simpler_champion() -> None:
    # T24.
    champion, decisions = run_chain(
        CHAMPION, {"linear": result(LINEAR, [1.0] * 5)}, STANDARD, "MASE"
    )
    assert champion is CHAMPION
    assert decisions[0].median_gain == 0.0


@pytest.mark.parametrize(("ratio", "accepted"), [(0.049, False), (0.05, True), (0.051, True)])
def test_variability_boundaries(ratio: float, accepted: bool) -> None:
    # ADR-0003: median correction_std_ratio >= 0.05.
    decision = decide(
        "residual", CHAMPION, result(HYBRID, [0.8] * 5, ratio=ratio), STANDARD, "MASE"
    )
    assert decision.accepted is accepted
    assert decision.variability_value == pytest.approx(ratio)


def test_variability_uses_the_median_and_ignores_undefined_units() -> None:
    criterion = VariabilityCriterion()
    hybrid = result(HYBRID, [0.8] * 3)
    trials = tuple(
        TrialResult(
            **{
                **{f: getattr(t, f) for f in t.__dataclass_fields__},
                "diagnostics": {"correction_std_ratio": r},
            }
        )
        for t, r in zip(hybrid.trials, [0.01, math.nan, 0.2], strict=True)
    )
    mixed = EvaluationResult(HYBRID, "plan", "D", 1, ("MASE",), trials)
    assert criterion.value(mixed, [t.unit_id for t in trials]) == pytest.approx(0.105)


def panel_result(
    candidate: CandidateSpec, ratios: dict[str, list[float]], error: float = 0.8
) -> EvaluationResult:
    """One unit per ratio, grouped by series; every unit has the same error."""

    trials = [
        TrialResult(
            candidate_id=candidate.candidate_id,
            series_id=series,
            fold_id=f"select-{i}",
            layer="select",
            origin_index=100 + i,
            origin_time=pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            horizon=1,
            status="SUCCEEDED",
            trial_key=f"{series}-{i}",
            metrics={"MASE": error},
            diagnostics={"correction_std_ratio": ratio},
        )
        for series, values in ratios.items()
        for i, ratio in enumerate(values)
    ]
    return EvaluationResult(candidate, "plan", "D", 1, ("MASE",), tuple(trials))


def test_variability_takes_the_median_per_series_then_across_series() -> None:
    # Same number of units per series: the global median (0.3) is not the declared value.
    ratios = {"a": [0.1, 0.1, 100.0], "b": [0.2, 0.2, 100.0], "c": [0.3, 0.3, 100.0]}
    hybrid = panel_result(HYBRID, ratios)
    units = hybrid.unit_ids
    criterion = VariabilityCriterion()
    assert criterion.per_series(hybrid, units) == pytest.approx({"a": 0.1, "b": 0.2, "c": 0.3})
    assert criterion.value(hybrid, units) == pytest.approx(0.2)
    assert float(pd.Series([r for v in ratios.values() for r in v]).median()) == 0.3


def test_per_series_aggregation_can_flip_the_decision() -> None:
    # Per series 0.04 (refused); a single median over the 9 units would give 0.06.
    ratios = {"a": [0.04, 0.04, 1.0], "b": [0.04, 0.04, 1.0], "c": [0.06, 0.06, 1.0]}
    hybrid = panel_result(HYBRID, ratios)
    champion = panel_result(BASE, dict.fromkeys(ratios, [math.nan] * 3), error=1.0)
    decision = decide("residual", champion, hybrid, STANDARD, "MASE")
    assert decision.variability_value == pytest.approx(0.04)
    assert decision.reason_codes == ("variability_below_threshold",)
    assert dict(decision.variability_by_series) == pytest.approx({"a": 0.04, "b": 0.04, "c": 0.06})


def test_series_without_finite_variability_is_kept_visible_and_ignored() -> None:
    # Unequal unit counts, a NaN unit and a series with no finite diagnostic.
    ratios = {"a": [0.1, math.nan], "b": [0.2, 0.4, 0.6, 0.8], "c": [math.nan, math.nan]}
    hybrid = panel_result(HYBRID, ratios)
    criterion = VariabilityCriterion()
    per_series = criterion.per_series(hybrid, hybrid.unit_ids)
    assert per_series["a"] == pytest.approx(0.1)
    assert per_series["b"] == pytest.approx(0.5)
    assert math.isnan(per_series["c"])
    assert criterion.value(hybrid, hybrid.unit_ids) == pytest.approx(0.3)
    # Only the compared units count.
    only_b = [u for u in hybrid.unit_ids if u.startswith("b:")][:2]
    assert criterion.per_series(hybrid, only_b) == pytest.approx({"b": 0.3})


def test_variability_does_not_depend_on_the_order_of_series_or_units() -> None:
    ratios = {"a": [0.1, 0.5, 0.2], "b": [0.9, 0.3], "c": [0.05, 0.07, 0.06, 0.08]}
    reversed_ratios = {s: list(reversed(v)) for s, v in reversed(list(ratios.items()))}
    criterion = VariabilityCriterion()
    forward = panel_result(HYBRID, ratios)
    backward = panel_result(HYBRID, reversed_ratios)
    assert criterion.value(forward, forward.unit_ids) == criterion.value(
        backward, list(reversed(backward.unit_ids))
    )


@pytest.mark.parametrize(
    ("ratio", "gte_passes", "lte_passes"),
    [(0.049, False, True), (0.05, True, True), (0.051, True, False)],
)
def test_panel_variability_boundaries_in_both_directions(
    ratio: float, gte_passes: bool, lte_passes: bool
) -> None:
    # The middle series sets the value; the others sit on either side of it.
    hybrid = panel_result(HYBRID, {"a": [0.0, 0.01], "b": [ratio, ratio], "c": [1.0, math.nan]})
    for direction, expected in (("gte", gte_passes), ("lte", lte_passes)):
        criterion = VariabilityCriterion(direction=direction)  # type: ignore[arg-type]
        value = criterion.value(hybrid, hybrid.unit_ids)
        assert value == pytest.approx(ratio)
        assert criterion.passes(value) is expected


def test_variability_criterion_version_marks_the_per_series_aggregation() -> None:
    assert VariabilityCriterion().describe()["version"] == 2
    assert STANDARD.describe()["variability"]["version"] == 2  # type: ignore[index]


def test_undefined_variability_rejects() -> None:
    decision = decide(
        "residual", CHAMPION, result(HYBRID, [0.8] * 5, ratio=math.nan), STANDARD, "MASE"
    )
    assert decision.reason_codes == ("variability_undefined",)


def test_lte_direction() -> None:
    criterion = VariabilityCriterion(threshold=0.3, direction="lte")
    assert criterion.passes(0.3)
    assert not criterion.passes(0.31)


def test_quickstart_disables_variability_visibly() -> None:
    decision = decide(
        "residual",
        CHAMPION,
        result(HYBRID, [0.8] * 5, ratio=0.0),
        SelectionPolicy.quickstart(),
        "MASE",
    )
    assert decision.accepted
    assert decision.variability_status == "disabled"
    assert decision.thresholds["variability"] is None


def test_research_strict_requires_variability_and_full_coverage() -> None:
    with pytest.raises(ProtocolConfigurationError, match="variabilidade"):
        SelectionPolicy(profile="research_strict", variability=None)
    with pytest.raises(ProtocolConfigurationError, match="cobertura"):
        SelectionPolicy(profile="research_strict", min_coverage=0.8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"profile": "yolo"},
        {"min_coverage": 0.0},
        {"win_rate_threshold": 1.5},
        {"epsilon": 0.0},
        {"family_order": ("linear", "linear")},
    ],
)
def test_invalid_policies(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        SelectionPolicy(**kwargs)


def test_chain_compares_with_the_current_champion() -> None:
    linear = result(LINEAR, [0.8] * 5)
    hybrid = result(HYBRID, [0.75] * 5, ratio=0.2)  # 25 % better than the base, 6.25 % than linear
    champion, decisions = run_chain(
        CHAMPION, {"linear": linear, "residual": hybrid}, STANDARD, "MASE"
    )
    assert champion is hybrid
    assert [d.champion_before for d in decisions] == [BASE.candidate_id, LINEAR.candidate_id]
    assert decisions[1].median_gain == pytest.approx(0.0625)


def test_family_order_is_followed_and_matters() -> None:
    linear = result(LINEAR, [0.95] * 5)
    hybrid = result(HYBRID, [0.9] * 5, ratio=0.2)
    hybrid_first = SelectionPolicy(
        family_order=("residual", "linear", "statistical", "kernel", "tree", "neural")
    )
    _, default_decisions = run_chain(
        CHAMPION, {"linear": linear, "residual": hybrid}, STANDARD, "MASE"
    )
    _, other_decisions = run_chain(
        CHAMPION, {"linear": linear, "residual": hybrid}, hybrid_first, "MASE"
    )
    assert [d.family for d in default_decisions] == ["linear", "residual"]
    assert [d.family for d in other_decisions] == ["residual", "linear"]


def test_legacy_selection_chain_parity() -> None:
    # Behaviour of the legacy check_compose_selection: real gain accepted, marginal gain,
    # inert correction and failing structures refused.
    real = result(LINEAR, [0.9] * 5)
    marginal = result(HYBRID, [0.895] * 5, ratio=0.3)
    champion, decisions = run_chain(
        CHAMPION, {"linear": real, "residual": marginal}, STANDARD, "MASE"
    )
    assert champion is real
    assert decisions[1].reason_codes == ("median_gain_below_threshold",)

    inert = result(HYBRID, [0.5] * 5, ratio=0.02)
    _, decisions = run_chain(CHAMPION, {"residual": inert}, STANDARD, "MASE")
    assert decisions[0].reason_codes == ("variability_below_threshold",)

    failing = result(HYBRID, [0.5, 0.5, 0.5, 0.5, None], ratio=0.3)
    _, decisions = run_chain(CHAMPION, {"residual": failing}, STANDARD, "MASE")
    assert "coverage_incomplete" in decisions[0].reason_codes


def test_ranking_averages_tied_ranks_and_requires_every_series() -> None:
    convention = RankingConvention()
    a = result(BASE, [1.0, 1.0], layer="search")
    b = result(LINEAR, [1.0, 1.0], layer="search")
    c = result(CandidateSpec.standalone("knn", family="kernel"), [2.0, 2.0], layer="search")
    ranked = convention.rank({"s1": [a, b, c], "s2": [a, b]}, "MASE")
    assert [r.representative.candidate.base.name for r in ranked] == sorted(
        ["naive", "linear"],
        key=lambda n: (
            CandidateSpec.standalone(
                n, family="statistical" if n == "naive" else "linear"
            ).candidate_id
        ),
    )  # knn missing in s2: excluded; equal ranks broken by candidate_id
    assert {r.mean_rank for r in ranked} == {1.5}


def test_optional_significance_predicate() -> None:
    from autohybts.selection.policy import paired_p_value

    better = [0.9, 0.8, 0.85, 0.7, 0.9, 0.8, 0.75, 0.82]
    clear = result(HYBRID, better, ratio=0.3)
    base = result(BASE, [1.0] * 8)
    strict = SelectionPolicy(significance_alpha=0.05)
    accepted = decide("residual", base, clear, strict, "MASE")
    assert accepted.accepted
    assert accepted.p_value == pytest.approx(1 / 2**8)  # all 8 units better: exact p
    # Same gains on only 4 units: the rule passes but p >= 1/16 can never reach 0.05.
    few = decide(
        "residual",
        result(BASE, [1.0] * 4),
        result(HYBRID, better[:4], ratio=0.3),
        strict,
        "MASE",
    )
    assert few.reason_codes == ("not_significant",)
    assert decide("residual", base, clear, STANDARD, "MASE").p_value is None
    tied = paired_comparison(base, result(BASE, [1.0] * 8), "MASE", 1e-12)
    assert paired_p_value(tied) == 1.0
    assert strict.describe()["significance"] == {"test": "wilcoxon_one_sided", "alpha": 0.05}
    assert "significance" not in STANDARD.describe()
    with pytest.raises(ConfigurationError):
        SelectionPolicy(significance_alpha=1.5)


def test_significance_alpha_reaches_the_experiment_decisions() -> None:
    from autohybts import AutoHybridExperiment
    from autohybts.datasets import get_data

    exp = AutoHybridExperiment(
        fh=7,
        fold=5,
        budget="smoke",
        base_models=("naive", "seasonal_naive"),
        correctors=("linear",),
        residual_windows=(7,),
        alphas=(1.0,),
        significance_alpha=0.05,
    ).fit(get_data("synthetic_daily", n=260))
    table = exp.compare_models().decisions
    assert table["p_value"].notna().all()
    assert exp.policy_.significance_alpha == 0.05


def test_decisions_keep_the_paired_errors_and_gains_of_every_unit() -> None:
    from autohybts.selection.incremental import SelectionResult

    challenger = result(LINEAR, [0.5, 1.0, 1.2, 0.9, 0.0])
    decision = decide("linear", CHAMPION, challenger, STANDARD, "MASE")
    assert decision.champion_errors == (1.0,) * 5
    assert decision.challenger_errors == (0.5, 1.0, 1.2, 0.9, 0.0)
    assert decision.gains == pytest.approx((0.5, 0.0, -0.2, 0.1, 1.0))
    selection = SelectionResult(
        champion=challenger.candidate,
        champion_result=challenger,
        finalists=(CHAMPION, challenger),
        decisions=(decision,),
        audit=None,  # type: ignore[arg-type]
    )
    units = selection.decision_units_frame()
    assert list(units["unit_id"]) == list(decision.unit_ids)
    assert units["gain"].median() == pytest.approx(decision.median_gain)
    assert units["win"].mean() == pytest.approx(decision.win_rate)  # ties are not wins
    assert set(units["series_id"]) == {"s"}  # unit ids are "series:fold"
