from limit_utils import expand_limit_checks, compute_limit_violation


def _single_violation(entry, value, weights=None, penalty_weight=1000.0):
    checks = expand_limit_checks([entry], weights or {}, penalty_weight=penalty_weight)
    assert len(checks) == 1
    return compute_limit_violation(checks[0], value)


def test_greater_than_limit_penalizes_excess():
    entry = {"metric": "drawdown_worst", "penalize_if": "greater_than", "value": 0.5}
    violation = _single_violation(entry, value=0.6, penalty_weight=10)
    assert violation == (0.6 - 0.5) * 10


def test_less_than_limit_penalizes_deficit_with_custom_stat():
    entry = {"metric": "adg", "penalize_if": "less_than", "value": 0.001, "stat": "min"}
    violation = _single_violation(entry, value=0.0002, penalty_weight=5)
    assert violation == (0.001 - 0.0002) * 5


def test_outside_range_penalty_hits_when_value_leaves_band():
    entry = {"metric": "loss_profit_ratio", "penalize_if": "outside_range", "range": [0.2, 0.5]}
    violation_low = _single_violation(entry, value=0.1, penalty_weight=7)
    violation_high = _single_violation(entry, value=0.9, penalty_weight=7)
    assert violation_low == (0.2 - 0.1) * 7
    assert violation_high == (0.9 - 0.5) * 7


def test_inside_range_penalty_hits_inside_band_only():
    entry = {"metric": "omega_ratio", "penalize_if": "inside_range", "range": [1.5, 2.0]}
    violation_inside = _single_violation(entry, value=1.7, penalty_weight=11)
    violation_outside = _single_violation(entry, value=2.5, penalty_weight=11)
    assert violation_inside == min(1.7 - 1.5, 2.0 - 1.7) * 11
    assert violation_outside == 0.0


def test_auto_penalize_if_respects_scoring_weight_sign():
    entry = {"metric": "adg", "penalize_if": "auto", "value": 0.0005}
    weights = {"adg": -1.0}
    violation = _single_violation(entry, value=0.0001, weights=weights, penalty_weight=3)
    assert violation == (0.0005 - 0.0001) * 3
