"""Extended-real facts remain raw while selection distances stay finite."""

import json
import math
import operator
from pathlib import Path

import numpy as np
import pytest

from config.scoring import ObjectiveSpec
from metrics_schema import MetricAggregationError, encode_metric_values
from pareto_core import crowding_distances, extract_objectives
from pareto_explorer import (
    ParetoCandidate,
    _extract_suite_metrics,
    _resolve_limit_value,
    project_candidates_to_scenario,
    select_candidate,
)
from pareto_store import LimitSpec, ParetoStore, _evaluate_limits

OMEGA = "omega_ratio_strategy_equity"


def entry(value):
    return {
        "optimize": {"scoring": [OMEGA]},
        "metrics": {
            "definition_version": 1,
            "objectives": encode_metric_values({OMEGA: value}),
            "constraint_violation": 0.0,
        },
    }


def candidate(value, name="result.json"):
    return ParetoCandidate(Path(name), entry(value), {OMEGA: value}, {}, {})


def test_raw_objective_infinity_requires_current_status_encoding():
    assert extract_objectives(entry(math.inf))[0] == (math.inf,)
    old = entry(1.0)
    old["metrics"]["definition_version"] = 0
    with pytest.raises(MetricAggregationError, match="definition"):
        extract_objectives(old)
    malformed = entry(1.0)
    malformed["metrics"]["objectives"][OMEGA] = math.inf
    with pytest.raises(MetricAggregationError, match="malformed"):
        extract_objectives(malformed)


@pytest.mark.parametrize(
    "method", ["ideal", "reference", "utility", "lexicographic", "outranking"]
)
def test_selection_distances_are_finite_with_true_infinity(method):
    choices = [candidate(2.0, "finite.json"), candidate(math.inf, "infinite.json")]
    result = select_candidate(
        choices,
        [ObjectiveSpec(OMEGA, "max")],
        method=method,
        target_pairs=[OMEGA + "=inf"] if method == "reference" else None,
    )
    assert result.candidate.path.name == "infinite.json"
    assert math.isfinite(result.score)
    assert result.objective_values[OMEGA] == math.inf


def test_scenario_projection_and_limits_preserve_status():
    row = candidate(math.inf)
    row.entry["suite_metrics"] = {
        "definition_version": 1,
        "scenario_labels": ["one", "two"],
        "metrics": {
            OMEGA: {
                "stats": encode_metric_values({"mean": math.inf}),
                "scenarios": encode_metric_values({"one": math.inf, "two": math.inf}),
                **encode_metric_values({"aggregated": math.inf}),
            }
        },
    }
    projected = project_candidates_to_scenario(
        [row], [ObjectiveSpec(OMEGA, "max")], "one"
    )[0]
    assert projected.objectives[OMEGA] == math.inf
    assert (
        _resolve_limit_value(
            projected, {"metric": OMEGA, "penalize_if": "lower_than", "value": 1.0}
        )
        == math.inf
    )
    assert (
        _resolve_limit_value(
            row,
            {
                "metric": OMEGA,
                "scenario": "one",
                "penalize_if": "lower_than",
                "value": 1.0,
            },
        )
        == math.inf
    )
    stats, aggregate = _extract_suite_metrics(
        row.entry, reducer_cfg={"default": "mean"}
    )
    assert stats["metric_status"][OMEGA + "_mean"]["status"] == "positive_infinity"
    assert aggregate["metric_status"][OMEGA]["status"] == "positive_infinity"
    json.dumps(projected.stats_flat, allow_nan=False)
    assert _evaluate_limits(
        [LimitSpec(OMEGA, "auto", operator.gt, 1.0)], {}, {}, projected.objectives, {}
    )
    assert not _evaluate_limits(
        [LimitSpec(OMEGA, "auto", operator.lt, 1.0)], {}, {}, projected.objectives, {}
    )


def test_pareto_store_keeps_status_encoded_infinity_and_skips_invalid(tmp_path):
    store = ParetoStore(str(tmp_path))
    assert store.add_entry(entry(math.inf))
    assert not store.add_entry(entry(2.0))
    invalid = entry(3.0)
    invalid["metrics"]["fitness_status"] = "invalid"
    assert not store.add_entry(invalid)
    front = store.get_front()
    assert len(front) == 1
    assert front[0]["metrics"]["objectives"][OMEGA] is None
    json.dumps(front, allow_nan=False)
    assert not np.isnan(
        crowding_distances(np.array([[-math.inf], [1.0], [math.inf]]))
    ).any()


def test_pareto_cli_serializes_true_infinity_as_explicit_status(tmp_path, capsys):
    from pareto_explorer import build_parser, run_from_args

    (tmp_path / "infinite.json").write_text(
        json.dumps(entry(math.inf), allow_nan=False)
    )
    run_from_args(build_parser().parse_args([str(tmp_path), "--json"]))

    def reject_constant(value):
        raise AssertionError(f"nonstandard JSON number {value}")

    output = json.loads(capsys.readouterr().out, parse_constant=reject_constant)
    objectives = output["selected"]["objectives"]
    assert objectives[OMEGA] is None
    assert objectives["metric_status"][OMEGA]["status"] == "positive_infinity"
    assert (
        output["selected"]["details"]["ideal_point"]["metric_status"][OMEGA]["status"]
        == "positive_infinity"
    )


def test_compression_reports_raw_ranges_and_statuses():
    from pareto_compress import compress_candidates
    from pareto_explorer import _json_ready

    members, ranges, _ = compress_candidates(
        [candidate(2.0), candidate(math.inf, "infinite.json")],
        [ObjectiveSpec(OMEGA, "max")],
        count=2,
    )
    assert ranges[0]["min"] == 2.0
    assert ranges[0]["max"] == math.inf
    encoded = _json_ready({"objectives": ranges})
    assert (
        encoded["objectives"][0]["metric_status"]["max"]["status"]
        == "positive_infinity"
    )
    json.dumps(encoded, allow_nan=False)


@pytest.mark.parametrize(
    "option,value", [("weight", "inf"), ("weight", "nan"), ("target", "nan")]
)
def test_selection_rejects_undefined_control_values(option, value):
    args = (
        {"weight_pairs": [OMEGA + "=" + value]}
        if option == "weight"
        else {"target_pairs": [OMEGA + "=" + value]}
    )
    with pytest.raises(ValueError, match="finite|undefined"):
        select_candidate(
            [candidate(1.0), candidate(math.inf, "infinite.json")],
            [ObjectiveSpec(OMEGA, "max")],
            method="utility" if option == "weight" else "reference",
            **args,
        )


def test_numeric_result_metadata_cannot_become_an_objective():
    with pytest.raises(MetricAggregationError, match="objective mapping"):
        extract_objectives(
            {"metrics": {"definition_version": 1, "constraint_violation": 0.0}}
        )
    with pytest.raises(ValueError, match="unknown or retired"):
        extract_objectives(
            {
                "metrics": {
                    "definition_version": 1,
                    "objectives": {"observation_count": 12},
                }
            }
        )
