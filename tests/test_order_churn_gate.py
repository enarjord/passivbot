from __future__ import annotations

import pytest

from live import reconciler
from live.order_churn_gate import (
    OrderChurnGateState,
    deterministic_one_to_one_matches,
    normalize_ideal_orders,
)


SYMBOL = "BTC/USDT:USDT"


def _order(
    *,
    price: float,
    qty: float = 1.0,
    pb_order_type: str = "entry_ema_anchor_long",
) -> dict:
    return {
        "symbol": SYMBOL,
        "position_side": "long",
        "side": "buy",
        "reduce_only": False,
        "type": "limit",
        "pb_order_type": pb_order_type,
        "qty": qty,
        "price": price,
    }


def _evaluate(
    state: OrderChurnGateState,
    orders: list[dict],
    *,
    now: float,
    stability_seconds: float = 120.0,
    max_sample_gap_seconds: float = 70.0,
):
    return state.evaluate_and_record(
        {SYMBOL: orders},
        now_monotonic=now,
        tolerance=0.0002,
        stability_seconds=stability_seconds,
        window_seconds=600.0,
        max_sample_gap_seconds=max_sample_gap_seconds,
    )


def test_no_history_and_single_move_fail_open():
    state = OrderChurnGateState()
    first = _order(price=100.0)
    assert _evaluate(state, [first], now=0.0)[id(first)].reason == "no_history"

    moved = _order(price=100.1)
    decision = _evaluate(state, [moved], now=60.0)[id(moved)]
    assert decision.churn_evidenced is False
    assert decision.reason == "history_short"


def test_sustained_monotonic_price_drift_is_evidence():
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=100.0)], now=0.0)
    _evaluate(state, [_order(price=100.1)], now=60.0)
    current = _order(price=100.2)

    decision = _evaluate(state, [current], now=120.0)[id(current)]

    assert decision.churn_evidenced is True
    assert decision.reason == "continuous_price_drift"


def test_sustained_monotonic_quantity_drift_is_evidence():
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=100.0, qty=1.0)], now=0.0)
    _evaluate(state, [_order(price=100.0, qty=1.001)], now=60.0)
    current = _order(price=100.0, qty=1.002)

    decision = _evaluate(state, [current], now=120.0)[id(current)]

    assert decision.churn_evidenced is True
    assert decision.reason == "continuous_qty_drift"


def test_oscillation_and_one_time_jump_do_not_prove_continuous_drift():
    oscillating = OrderChurnGateState()
    _evaluate(oscillating, [_order(price=100.0)], now=0.0)
    _evaluate(oscillating, [_order(price=100.1)], now=60.0)
    current = _order(price=100.0)
    decision = _evaluate(oscillating, [current], now=120.0)[id(current)]
    assert decision.churn_evidenced is False
    assert decision.reason == "no_continuous_drift"

    one_jump = OrderChurnGateState()
    _evaluate(one_jump, [_order(price=100.0)], now=0.0)
    _evaluate(one_jump, [_order(price=100.0)], now=60.0)
    current = _order(price=101.0)
    decision = _evaluate(one_jump, [current], now=120.0)[id(current)]
    assert decision.churn_evidenced is False
    assert decision.reason == "no_continuous_drift"


def test_recent_stability_clears_older_drift():
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=100.0)], now=0.0)
    _evaluate(state, [_order(price=100.1)], now=60.0)
    _evaluate(state, [_order(price=100.2)], now=120.0)
    _evaluate(state, [_order(price=100.2)], now=180.0)
    current = _order(price=100.2)

    decision = _evaluate(state, [current], now=240.0)[id(current)]

    assert decision.churn_evidenced is False
    assert decision.reason == "stable_tight_prefix"
    assert decision.tight_prefix_seconds >= 120.0


def test_time_gap_and_cohort_or_cardinality_change_fail_open():
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=100.0)], now=0.0)
    _evaluate(state, [_order(price=100.1)], now=60.0)
    current = _order(price=100.2)
    decision = _evaluate(state, [current], now=140.0)[id(current)]
    assert decision.churn_evidenced is False
    assert decision.reason == "no_history"

    changed_type = _order(
        price=100.3, pb_order_type="entry_grid_normal_long"
    )
    decision = _evaluate(state, [changed_type], now=200.0)[id(changed_type)]
    assert decision.churn_evidenced is False
    assert decision.reason == "no_history"

    state = OrderChurnGateState()
    _evaluate(state, [_order(price=100.0)], now=0.0)
    current = [_order(price=100.1), _order(price=101.0)]
    decisions = _evaluate(state, current, now=60.0)
    assert all(not decisions[id(order)].churn_evidenced for order in current)


def test_ladder_reordering_preserves_rank_based_decisions():
    state_a = OrderChurnGateState()
    state_b = OrderChurnGateState()
    for now, prices in (
        (0.0, [100.0, 101.0]),
        (60.0, [100.1, 101.1]),
    ):
        _evaluate(state_a, [_order(price=price) for price in prices], now=now)
        _evaluate(
            state_b,
            list(reversed([_order(price=price) for price in prices])),
            now=now,
        )
    current_a = [_order(price=100.2), _order(price=101.2)]
    current_b = list(reversed([_order(price=100.2), _order(price=101.2)]))
    decisions_a = _evaluate(state_a, current_a, now=120.0)
    decisions_b = _evaluate(state_b, current_b, now=120.0)

    assert {
        order["price"]: decisions_a[id(order)].reason for order in current_a
    } == {
        order["price"]: decisions_b[id(order)].reason for order in current_b
    }


def test_matching_is_deterministic_and_one_to_one():
    current = normalize_ideal_orders(
        [_order(price=100.01), _order(price=100.03)]
    )
    previous = normalize_ideal_orders(
        [_order(price=100.0), _order(price=100.02)]
    )

    expected = deterministic_one_to_one_matches(current, previous, 0.0002)

    assert len(expected) == len(set(expected.values()))
    for _ in range(5):
        assert deterministic_one_to_one_matches(
            current, previous, 0.0002
        ) == expected


def test_snapshot_and_attempt_windows_prune():
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=100.0)], now=0.0)
    state.record_action_attempts(2, now_monotonic=0.0)
    state.record_action_attempts(1, now_monotonic=5.0)
    assert (
        state.action_attempt_count(now_monotonic=9.0, window_seconds=10.0)
        == 3
    )
    assert (
        state.action_attempt_count(now_monotonic=11.0, window_seconds=10.0)
        == 1
    )

    current = _order(price=100.0)
    _evaluate(state, [current], now=601.0)
    snapshots = state.history_by_symbol[SYMBOL]
    assert len(snapshots) == 1
    assert snapshots[0].monotonic_seconds == 601.0


def test_prepare_normalization_failure_is_economy_only(monkeypatch, caplog):
    state = OrderChurnGateState()
    invalid = _order(price=100.0)
    invalid["qty"] = float("nan")

    class Bot:
        _order_churn_gate_state = state
        _order_churn_risk_active_pairs = ()
        active_symbols = [SYMBOL]
        open_orders = {}
        positions = {}

        @staticmethod
        def live_value(key):
            return {
                "order_replacement_churn_gate_activation_count": 10,
                "order_replacement_churn_gate_window_minutes": 10.0,
                "order_replacement_churn_gate_stability_minutes": 2.0,
                "order_match_tolerance_pct": 0.0002,
                "execution_delay_seconds": 2.0,
            }[key]

    monkeypatch.setattr(reconciler.time, "monotonic", lambda: 1.0)
    with caplog.at_level("ERROR"):
        result = reconciler.prepare_order_churn_evidence(
            Bot(), {SYMBOL: [invalid]}, generation=state.begin_generation()
        )

    assert result is None
    assert invalid["_churn_evidence"] is False
    assert invalid["_churn_reason"] == "normalization_unavailable"
    assert state.history_by_symbol == {}
    assert "reconciliation remains authoritative" in caplog.text


def test_active_rust_risk_pair_bypasses_observed_churn(monkeypatch):
    state = OrderChurnGateState()

    class Bot:
        _order_churn_gate_state = state
        _order_churn_risk_active_pairs = ((SYMBOL, "long"),)
        active_symbols = [SYMBOL]
        open_orders = {}
        positions = {}

        @staticmethod
        def live_value(key):
            return {
                "order_replacement_churn_gate_activation_count": 10,
                "order_replacement_churn_gate_window_minutes": 10.0,
                "order_replacement_churn_gate_stability_minutes": 2.0,
                "order_match_tolerance_pct": 0.0002,
                "execution_delay_seconds": 2.0,
            }[key]

    for now, price in ((0.0, 100.0), (60.0, 100.1), (120.0, 100.2)):
        monkeypatch.setattr(reconciler.time, "monotonic", lambda now=now: now)
        current = _order(price=price)
        reconciler.prepare_order_churn_evidence(
            Bot(), {SYMBOL: [current]}, generation=state.begin_generation()
        )

    assert current["_churn_evidence"] is False
    assert current["_churn_reason"] == "rust_risk_phase_active"


def test_console_projection_throttle():
    state = OrderChurnGateState()
    signature = (("allowance_exhausted", SYMBOL),)

    assert state.should_log_console_event(
        "create_deferred",
        signature,
        now_monotonic=100.0,
        repeat_seconds=300.0,
    ) == (True, 0)
    assert state.should_log_console_event(
        "create_deferred",
        signature,
        now_monotonic=101.0,
        repeat_seconds=300.0,
    ) == (False, 0)
    assert state.should_log_console_event(
        "create_deferred",
        signature,
        now_monotonic=400.0,
        repeat_seconds=300.0,
    ) == (True, 1)


def test_invalid_matching_tolerance_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        deterministic_one_to_one_matches([], [], -1.0)
