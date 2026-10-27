"""Exact core selectors, accepted outer reducers and monotonic objective transport."""

import math
import pytest
from config.metrics import (
    KNOWN_METRICS,
    metric_base_name,
    split_metric_stat_suffix,
    canonicalize_metric_name,
)
from config.scoring import (
    normalize_scoring_entries,
    ObjectiveSpec,
    to_engine_value,
    from_engine_value,
)
from config.limits import normalize_limit_entries
from limit_utils import expand_limit_checks, compute_limit_violation
from metrics_schema import MetricAggregationError


@pytest.mark.parametrize("name", sorted(KNOWN_METRICS))
def test_all_core_names_support_both_independent_penalty_directions(name):
    entries = normalize_limit_entries(
        [
            {"metric": name, "penalize_if": "less_than", "value": 0.1},
            {"metric": name, "penalize_if": "greater_than", "value": 0.9},
        ]
    )
    checks = expand_limit_checks(entries, {}, penalty_weight=1.0)
    assert len(checks) == 2
    assert sum(compute_limit_violation(check, 0.5) for check in checks) == 0
    assert sum(compute_limit_violation(check, 0.0) for check in checks) > 0
    assert sum(compute_limit_violation(check, 1.0) for check in checks) > 0


@pytest.mark.parametrize(
    "name",
    [
        "adg_fraction_strategy_equity",
        "sharpe_ratio_usd",
        "sortino_ratio_strategy_equity_ew",
        "hsl_panic_fill_loss_max_usd",
        "metric_a",
        "default",
        "cagr_fraction_strategy_equity_max",
    ],
)
def test_unknown_and_retired_object_form_selectors_fail_before_replay(name):
    with pytest.raises(ValueError, match="unknown or retired"):
        normalize_scoring_entries([{"metric": name, "goal": "min"}])


@pytest.mark.parametrize("name", sorted(KNOWN_METRICS))
def test_intrinsic_names_and_outer_statistics_remain_distinct(name):
    assert split_metric_stat_suffix(name) == (name, None)
    assert split_metric_stat_suffix(name + "_max") == (name, "max")
    assert metric_base_name(name + "_max") == name
    assert canonicalize_metric_name(name) == name


@pytest.mark.parametrize("goal", ["min", "max"])
def test_common_adapter_preserves_order_and_has_finite_infinite_endpoints(goal):
    spec = ObjectiveSpec("omega_ratio_strategy_equity", goal)
    raw = [-math.inf, -2.0, 0.0, 2.0, math.inf]
    values = [to_engine_value(spec, value) for value in raw]
    assert all(math.isfinite(x) for x in values)
    assert values == sorted(values, reverse=goal == "max")
    assert abs(values[0]) == abs(values[-1]) == 1
    assert from_engine_value(spec, values[0]) == raw[0]
    assert from_engine_value(spec, values[-1]) == raw[-1]


@pytest.mark.parametrize(
    "mode,bound,expected", [("less_than", 0.0, 0.0), ("greater_than", 1.0, math.inf)]
)
def test_infinite_limit_comparison_keeps_raw_order(mode, bound, expected):
    assert compute_limit_violation({"mode": mode, "bound": bound}, math.inf) == expected


def test_unavailable_limit_cannot_pass():
    with pytest.raises(MetricAggregationError):
        compute_limit_violation({"mode": "less_than", "bound": 0.0}, None)
