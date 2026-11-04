from pathlib import Path

import pytest

from config.scoring import ObjectiveSpec
from pareto_explorer import ParetoCandidate, select_candidate


GROWTH = "cagr_fraction_strategy_equity_ew"
ULCER = "ulcer_index_fraction_strategy_equity"
SPECS = [ObjectiveSpec(metric=GROWTH, goal="max"), ObjectiveSpec(metric=ULCER, goal="min")]
SCALES = [f"{GROWTH}=2", f"{ULCER}=0.25"]
TARGETS = [f"{GROWTH}=9", f"{ULCER}=0.375"]


def candidate(name, growth, ulcer):
    return ParetoCandidate(
        path=Path(f"{name}.json"),
        entry={},
        objectives={GROWTH: growth, ULCER: ulcer},
        stats_flat={},
        aggregated_values={},
    )


def test_fixed_scales_apply_to_both_metric_directions():
    pool = [candidate("growth", 10, 0.75), candidate("risk", 8, 0.25)]
    selection = select_candidate(
        pool, SPECS, method="ideal", scale_pairs=[f"{GROWTH}=0.5", f"{ULCER}=0.25"]
    )

    assert selection.candidate.path.stem == "growth"
    assert selection.score == pytest.approx(-(2.0**0.5))
    assert selection.details["scaling"] == "fixed_raw"
    assert selection.details["selected_utilities"] == {GROWTH: 0.0, ULCER: -2.0}
    assert selection.details["distance_components"] == {GROWTH: 0.0, ULCER: 2.0}


def test_aspiration_uses_signed_tie_break_when_both_candidates_meet_targets():
    pool = [candidate("at_target", 9, 0.375), candidate("exceeds_target", 10, 0.25)]
    common = {"scale_pairs": SCALES, "target_pairs": TARGETS}
    aspiration = select_candidate(pool, SPECS, method="aspiration", **common)
    reference = select_candidate(pool, SPECS, method="reference", **common)

    assert reference.candidate.path.stem == "at_target"
    assert aspiration.candidate.path.stem == "exceeds_target"
    assert aspiration.details["shortfalls"] == {GROWTH: 0.0, ULCER: 0.0}
    assert aspiration.details["secondary_score_vector"] == [0.0, 0.5]
    assert aspiration.details["ranking_order"] == [1, 0]


def test_aspiration_prefers_smaller_worst_shortfall_over_better_signed_total():
    pool = [candidate("near_target", 8, 0.375), candidate("high_growth", 15, 0.625)]
    result = select_candidate(pool, SPECS, method="aspiration", scale_pairs=SCALES, target_pairs=TARGETS)

    assert result.candidate.path.stem == "near_target"
    assert result.details["shortfalls"] == {GROWTH: 0.5, ULCER: 0.0}
    assert result.details["worst_weighted_shortfall"] == 0.25
    assert result.details["secondary_score_vector"][1] > result.details["secondary_score_vector"][0]


def test_sensitivity_reports_weight_and_target_winner_changes():
    pool = [candidate("growth", 10, 0.5), candidate("risk", 8, 0.25)]
    result = select_candidate(
        pool,
        SPECS,
        method="aspiration",
        scale_pairs=SCALES,
        target_pairs=TARGETS,
        sensitivity_weight_fraction=0.05,
        sensitivity_target_scale=0.5,
    )
    sensitivity = result.details["sensitivity"]

    assert result.candidate.path.stem == "growth"
    assert sensitivity["check_count"] == 6
    assert sensitivity["changed_check_count"] == 3
    assert {
        "kind": "weight_transfer",
        "from_metric": GROWTH,
        "to_metric": ULCER,
        "winner": "risk.json",
    } in sensitivity["checks"]
    assert {
        "kind": "target_step",
        "metric": GROWTH,
        "target": 8.0,
        "winner": "risk.json",
    } in sensitivity["checks"]


def test_fixed_scale_rejects_infinite_metrics_while_legacy_ideal_accepts_them():
    pool = [candidate("finite", 7, 0.25), candidate("infinite", float("inf"), 0.25)]
    assert select_candidate(pool, SPECS, method="ideal").candidate.path.stem == "infinite"

    with pytest.raises(ValueError, match=f"finite '{GROWTH}' value in infinite.json"):
        select_candidate(pool, SPECS, method="ideal", scale_pairs=SCALES)


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"scale_pairs": [f"{GROWTH}=1"]}, f"Fixed scale for '{ULCER}'"),
        ({"scale_pairs": [f"{GROWTH}=1", f"{ULCER}=0"]}, "positive and finite"),
        ({"scale_pairs": [f"{GROWTH}=1", f"{ULCER}=inf"]}, "positive and finite"),
        ({"scale_pairs": [*SCALES, "typo=2"]}, "Unknown or inactive scale metric"),
        ({"sensitivity_weight_fraction": 1.0}, "fraction between 0 and 1"),
        ({"sensitivity_target_scale": 0.1}, "requires aspiration"),
    ],
)
def test_ideal_rejects_invalid_preference_options(options, error):
    with pytest.raises(ValueError, match=error):
        select_candidate([candidate("sample", 1, 0.5)], SPECS, method="ideal", **options)


@pytest.mark.parametrize(
    ("method", "options", "error"),
    [
        ("knee", {"weight_pairs": [f"{GROWTH}=2"]}, "does not support objective weights"),
        ("ideal", {"target_pairs": TARGETS}, "does not support objective targets"),
        ("ideal", {"objectives_arg": ", "}, "Choose at least one selection objective"),
        ("ideal", {"objectives_arg": f"{GROWTH},{GROWTH}"}, "Choose distinct selection objectives"),
        ("ideal", {"weight_pairs": [f"{GROWTH}=1e308", f"{ULCER}=1e308"]}, "sum of objective weights"),
    ],
)
def test_selection_rejects_inapplicable_or_ambiguous_settings(method, options, error):
    with pytest.raises(ValueError, match=error):
        select_candidate([candidate("sample", 1, 0.5)], SPECS, method=method, **options)


def test_fixed_ideal_rejects_overflowing_distances():
    pool = [candidate("growth", 10, 0.5), candidate("risk", 8, 0.25)]
    with pytest.raises(ValueError, match="selection distances are not finite"):
        select_candidate(
            pool, SPECS, method="ideal", scale_pairs=[f"{GROWTH}=1e-200", f"{ULCER}=0.25"]
        )


def test_aspiration_rejects_incomplete_targets_for_explicit_objectives():
    with pytest.raises(ValueError, match=f"requires a finite target for '{ULCER}'"):
        select_candidate(
            [candidate("sample", 1, 0.5)],
            SPECS,
            method="aspiration",
            objectives_arg=f"{GROWTH},{ULCER}",
            scale_pairs=SCALES,
            target_pairs=[f"{GROWTH}=8"],
        )


def test_fixed_reference_keeps_targets_beyond_the_observed_range():
    pool = [candidate("growth", 10, 0.5), candidate("risk", 8, 0.25)]
    result = select_candidate(
        pool, SPECS, method="reference", scale_pairs=SCALES, target_pairs=[f"{GROWTH}=14", f"{ULCER}=0.25"]
    )

    assert result.candidate.path.stem == "growth"
    assert result.details["selected_utilities"] == {GROWTH: -2.0, ULCER: -1.0}


def test_linear_outranking_has_the_same_order_as_utility():
    pool = [candidate("growth", 10, 0.5), candidate("risk", 8, 0.125), candidate("balanced", 9, 0.25)]
    options = {"scale_pairs": SCALES, "weight_pairs": [f"{GROWTH}=2", f"{ULCER}=1"]}
    utility = select_candidate(pool, SPECS, method="utility", **options)
    outranking = select_candidate(pool, SPECS, method="outranking", **options)

    assert outranking.details["ranking_order"] == utility.details["ranking_order"]
    assert sum(outranking.details["score_vector"]) == pytest.approx(0.0)
