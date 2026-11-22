"""Verify the native shared account memory against an independent f32 fold."""

from copy import deepcopy
from fractions import Fraction
import json

import pytest

from v9_risk_reference import allowance, coefficients, f32, fma


@pytest.fixture(scope="module")
def native():
    import passivbot_rust

    if getattr(passivbot_rust, "__is_stub__", False):
        pytest.skip("requires the compiled Rust extension")
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    return passivbot_rust


def _payload():
    return {
        "start": 0,
        "now": 360_000,
        "known_origin": True,
        "memory_ema_span_minutes": 7.0,
        "fills": [
            {"identity": "entry-fee-long", "timestamp": 0, "net": -1.0},
            {"identity": "rebate-short", "timestamp": 0, "net": 1.25},
            {"identity": "panic-fill-short", "timestamp": 0, "net": -40.0},
            {"identity": "ordinary-loss-long", "timestamp": 60_000, "net": -2.0},
            {"identity": "profit-with-rebate", "timestamp": 120_000, "net": 10.125},
        ],
    }


def _evaluate(native, payload):
    return json.loads(native.realized_memory_v9(json.dumps(payload, allow_nan=False)))


def _reference_fold(payload):
    _, beta = coefficients(payload["memory_ema_span_minutes"])
    clock = payload["start"] // 60_000
    drawdown = 0.0
    for fill in payload["fills"]:
        minute = fill["timestamp"] // 60_000
        for _ in range(clock, minute):
            drawdown = fma(beta, drawdown, 0.0)
        drawdown = max(0.0, f32(Fraction(drawdown) - Fraction(f32(fill["net"]))))
        clock = minute
    for _ in range(clock, payload["now"] // 60_000):
        drawdown = fma(beta, drawdown, 0.0)
    return drawdown


def test_both_policies_consume_one_independently_replayed_drawdown(native):
    payload = _payload()
    outcome = _evaluate(native, payload)
    drawdown = outcome["drawdown"]
    assert drawdown == _reference_fold(payload)
    assert _evaluate(native, deepcopy(payload)) == outcome
    assert outcome["fills"] == len(payload["fills"])
    assert outcome["same_time_order_from_canonical_manager"] is True
    assert native.calc_loss_gate_headroom_v9(1000.0, True, 0.01, drawdown) == allowance(
        1000.0, 0.01, drawdown
    )
    assert native.calc_auto_unstuck_allowance_v9(1000.0, 0.04, 0.25, drawdown) == allowance(
        1000.0, 0.01, drawdown
    )
    assert native.calc_loss_gate_headroom_v9(1000.0, False, None, drawdown) is None
    assert native.calc_loss_gate_headroom_v9(1000.0, False, 0.01, drawdown) is None
    assert native.calc_loss_gate_headroom_v9(1000.0, True, 0.0, drawdown) == 0.0


def test_unknown_prefix_correction_reuses_the_same_initializer(native):
    payload = _payload()
    payload.update(start=60_000, known_origin=False)
    payload["fills"] = payload["fills"][3:]
    original = _evaluate(native, payload)
    payload["fills"][0]["net"] = -30.0
    corrected = _evaluate(native, payload)
    assert original["initialization"] == corrected["initialization"] == "unknown_prefix_zero"
    assert original["start"] == corrected["start"] == 60_000
    assert original["drawdown"] != corrected["drawdown"] == _reference_fold(payload)


@pytest.mark.parametrize("defect", ["duplicate", "future", "unordered", "invalid_span"])
def test_invalid_shared_facts_and_span_fail_before_replay(native, defect):
    payload = _payload()
    if defect == "duplicate":
        payload["fills"][1]["identity"] = payload["fills"][0]["identity"]
    elif defect == "future":
        payload["fills"][1]["timestamp"] = payload["now"] + 1
    elif defect == "unordered":
        payload["fills"].reverse()
    else:
        payload["memory_ema_span_minutes"] = 0.0
    with pytest.raises(ValueError):
        _evaluate(native, payload)
