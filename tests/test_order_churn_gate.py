from __future__ import annotations

import pytest

from live import reconciler
from live.order_churn_gate import (
    OrderChurnGateState,
    deterministic_one_to_one_matches,
    normalize_ideal_orders,
)
from passivbot_exceptions import FatalBotException


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


def _short_order(*, price: float, qty: float = 1.0) -> dict:
    order = _order(
        price=price,
        qty=qty,
        pb_order_type="entry_ema_anchor_short",
    )
    order["position_side"] = "short"
    order["side"] = "sell"
    return order


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
    _evaluate(state, [_order(price=100.2)], now=120.0)
    current = _order(price=100.3)

    decision = _evaluate(state, [current], now=180.0)[id(current)]

    assert decision.churn_evidenced is True
    assert decision.reason == "continuous_price_drift"


def test_sustained_monotonic_quantity_drift_is_evidence():
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=100.0, qty=1.0)], now=0.0)
    _evaluate(state, [_order(price=100.0, qty=1.001)], now=60.0)
    _evaluate(state, [_order(price=100.0, qty=1.002)], now=120.0)
    current = _order(price=100.0, qty=1.003)

    decision = _evaluate(state, [current], now=180.0)[id(current)]

    assert decision.churn_evidenced is True
    assert decision.reason == "continuous_qty_drift"


def test_old_stable_history_does_not_count_toward_drift_duration():
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=100.0)], now=0.0)
    _evaluate(state, [_order(price=100.0)], now=60.0)
    _evaluate(state, [_order(price=100.0)], now=120.0)
    _evaluate(state, [_order(price=100.1)], now=180.0)
    current = _order(price=100.2)

    decision = _evaluate(state, [current], now=250.0)[id(current)]

    assert decision.churn_evidenced is False
    assert decision.reason == "drift_run_short"


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


def test_repeated_exclusive_long_short_switching_is_churn_evidence():
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=99.0)], now=0.0)
    _evaluate(state, [_short_order(price=101.0)], now=60.0)

    first_reappearance = _order(price=99.0)
    decision = _evaluate(state, [first_reappearance], now=120.0)[
        id(first_reappearance)
    ]
    assert decision.churn_evidenced is False

    _evaluate(state, [_short_order(price=101.0)], now=180.0)
    repeated = _order(price=99.0)
    decision = _evaluate(state, [repeated], now=240.0)[id(repeated)]

    assert decision.churn_evidenced is True
    assert decision.reason == "intermittent_cohort_reappearance"


def test_exclusive_switching_needs_stability_duration():
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=99.0)], now=0.0)
    _evaluate(state, [_short_order(price=101.0)], now=10.0)
    _evaluate(state, [_order(price=99.0)], now=20.0)
    _evaluate(state, [_short_order(price=101.0)], now=30.0)
    repeated = _order(price=99.0)

    decision = _evaluate(state, [repeated], now=40.0)[id(repeated)]

    assert decision.churn_evidenced is False
    assert decision.reason == "intermittent_run_short"

    _evaluate(state, [_order(price=99.0)], now=80.0)
    _evaluate(state, [_order(price=99.0)], now=120.0)
    still_stable = _order(price=99.0)
    decision = _evaluate(state, [still_stable], now=140.0)[id(still_stable)]

    assert decision.churn_evidenced is False
    assert decision.reason == "intermittent_run_short"


def test_sustained_drift_overrides_short_switching_interval():
    state = OrderChurnGateState()
    for now, orders in (
        (0.0, [_order(price=99.0)]),
        (10.0, [_short_order(price=101.0)]),
        (20.0, [_order(price=99.0)]),
        (30.0, [_short_order(price=101.0)]),
        (40.0, [_order(price=99.0)]),
        (80.0, [_order(price=99.1)]),
        (120.0, [_order(price=99.2)]),
        (160.0, [_order(price=99.3)]),
    ):
        _evaluate(state, orders, now=now)
    drifting = _order(price=99.4)

    decision = _evaluate(state, [drifting], now=200.0)[id(drifting)]

    assert decision.churn_evidenced is True
    assert decision.reason == "continuous_price_drift"


def test_continuous_stability_clears_exclusive_switching_evidence():
    state = OrderChurnGateState()
    for now, orders in (
        (0.0, [_order(price=99.0)]),
        (60.0, [_short_order(price=101.0)]),
        (120.0, [_order(price=99.0)]),
        (180.0, [_short_order(price=101.0)]),
        (240.0, [_order(price=99.0)]),
        (300.0, [_order(price=99.0)]),
    ):
        _evaluate(state, orders, now=now)
    stable = _order(price=99.0)

    decision = _evaluate(state, [stable], now=360.0)[id(stable)]

    assert decision.churn_evidenced is False
    assert decision.reason == "stable_tight_prefix"
    assert state.history_reset_during_evaluation is True
    assert state.reset_count == 1

    switched = _short_order(price=101.0)
    decision = _evaluate(state, [switched], now=420.0)[id(switched)]
    assert decision.churn_evidenced is False

    returned = _order(price=99.0)
    decision = _evaluate(state, [returned], now=480.0)[id(returned)]
    assert decision.churn_evidenced is False


@pytest.mark.parametrize(
    "snapshots",
    [
        # An empty ideal snapshot breaks provenance.
        [
            (0.0, [_order(price=99.0)]),
            (60.0, [_short_order(price=101.0)]),
            (120.0, []),
            (180.0, [_order(price=99.0)]),
            (240.0, [_short_order(price=101.0)]),
        ],
        # Coexisting cohorts are not mutually exclusive switching.
        [
            (0.0, [_order(price=99.0)]),
            (60.0, [_short_order(price=101.0)]),
            (120.0, [_order(price=99.0), _short_order(price=101.0)]),
            (180.0, [_short_order(price=101.0)]),
        ],
        # A changed ladder cardinality breaks cohort continuity.
        [
            (0.0, [_order(price=98.0), _order(price=99.0)]),
            (60.0, [_short_order(price=101.0)]),
            (120.0, [_order(price=98.0), _order(price=99.0)]),
            (180.0, [_short_order(price=101.0)]),
        ],
    ],
)
def test_uncertain_exclusive_switching_fails_open(snapshots):
    state = OrderChurnGateState()
    for now, orders in snapshots:
        _evaluate(state, orders, now=now)
    current = _order(price=99.0)

    decision = _evaluate(state, [current], now=240.0)[id(current)]

    assert decision.churn_evidenced is False


def test_alternate_cohort_cardinality_change_breaks_switching_proof():
    state = OrderChurnGateState()
    for now, orders in (
        (0.0, [_order(price=99.0)]),
        (60.0, [_short_order(price=101.0)]),
        (120.0, [_order(price=99.0)]),
        (180.0, [_short_order(price=101.0), _short_order(price=102.0)]),
    ):
        _evaluate(state, orders, now=now)
    current = _order(price=99.0)

    decision = _evaluate(state, [current], now=240.0)[id(current)]

    assert decision.churn_evidenced is False


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


@pytest.mark.asyncio
async def test_malformed_rust_ideal_is_fatal_before_reconciliation():
    invalid = _order(price=100.0)
    invalid["qty"] = float("nan")

    class Bot:
        exchange = "fake"

        async def calc_ideal_orders(self):
            return {SYMBOL: [invalid]}

        def _snapshot_actual_orders(self, *_args, **_kwargs):
            raise AssertionError("reconciliation must not inspect exchange orders")

    with pytest.raises(FatalBotException, match="malformed ideal orders"):
        await reconciler.calc_orders_to_cancel_and_create(Bot())


def test_departed_symbol_clears_history_instead_of_refreshing_empty_snapshots(
    monkeypatch,
):
    state = OrderChurnGateState()

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

    bot = Bot()
    monkeypatch.setattr(reconciler.time, "monotonic", lambda: 0.0)
    current = _order(price=100.0)
    reconciler.prepare_order_churn_evidence(
        bot, {SYMBOL: [current]}, generation=state.begin_generation()
    )
    assert state.symbols_with_history() == {SYMBOL}

    bot.active_symbols = []
    bot.positions = {
        SYMBOL: {
            "long": {"size": 0.0, "price": 0.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    monkeypatch.setattr(reconciler.time, "monotonic", lambda: 1.0)
    reconciler.prepare_order_churn_evidence(
        bot, {}, generation=state.begin_generation()
    )

    assert state.symbols_with_history() == set()
    assert state.reset_count == 1


def test_nonzero_position_keeps_churn_history_after_symbol_rotation(monkeypatch):
    state = OrderChurnGateState()

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

    bot = Bot()
    monkeypatch.setattr(reconciler.time, "monotonic", lambda: 0.0)
    reconciler.prepare_order_churn_evidence(
        bot, {SYMBOL: [_order(price=100.0)]}, generation=state.begin_generation()
    )

    bot.active_symbols = []
    bot.positions = {
        SYMBOL: {
            "long": {"size": 1.0, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    monkeypatch.setattr(reconciler.time, "monotonic", lambda: 1.0)
    reconciler.prepare_order_churn_evidence(
        bot, {}, generation=state.begin_generation()
    )

    assert state.symbols_with_history() == {SYMBOL}


def test_stable_switching_boundary_is_emitted_as_history_reset(monkeypatch):
    state = OrderChurnGateState()
    events = []

    class Bot:
        _order_churn_gate_state = state
        _order_churn_risk_active_pairs = ()
        active_symbols = [SYMBOL]
        open_orders = {}
        positions = {}
        _emit_order_churn_evidence_event = staticmethod(
            lambda **kwargs: events.append(kwargs)
        )

        @staticmethod
        def live_value(key):
            return {
                "order_replacement_churn_gate_activation_count": 10,
                "order_replacement_churn_gate_window_minutes": 10.0,
                "order_replacement_churn_gate_stability_minutes": 2.0,
                "order_match_tolerance_pct": 0.0002,
                "execution_delay_seconds": 2.0,
            }[key]

    for now, orders in (
        (0.0, [_order(price=99.0)]),
        (60.0, [_short_order(price=101.0)]),
        (120.0, [_order(price=99.0)]),
        (180.0, [_short_order(price=101.0)]),
        (240.0, [_order(price=99.0)]),
        (300.0, [_order(price=99.0)]),
        (360.0, [_order(price=99.0)]),
    ):
        monkeypatch.setattr(reconciler.time, "monotonic", lambda now=now: now)
        reconciler.prepare_order_churn_evidence(
            Bot(), {SYMBOL: orders}, generation=state.begin_generation()
        )

    assert events[-1]["reset"] is True
    assert events[-1]["reset_count"] == 1


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
