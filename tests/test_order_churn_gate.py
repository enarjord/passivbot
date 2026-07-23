from __future__ import annotations

import pytest

from live import reconciler
from live.order_churn_gate import (
    OrderChurnGateState,
    deterministic_one_to_one_matches,
    normalize_ideal_orders,
)


def _order(
    *,
    price: float,
    qty: float = 1.0,
    pb_order_type: str = "entry_ema_anchor_long",
) -> dict:
    return {
        "symbol": "BTC/USDT:USDT",
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
    generation: int | None = None,
    stability_seconds: float = 120.0,
):
    if generation is None:
        generation = state.begin_generation()
    return state.evaluate_and_record(
        {"BTC/USDT:USDT": orders},
        generation=generation,
        now_monotonic=now,
        tight_tolerance=0.0002,
        wider_tolerance=0.002,
        stability_seconds=stability_seconds,
        window_seconds=600.0,
        max_generation_gap_seconds=70.0,
    )


def test_wider_movement_is_churn_evidence_but_no_history_fails_open():
    state = OrderChurnGateState()
    first = _order(price=100.0)
    first_decision = _evaluate(state, [first], now=0.0)[id(first)]
    assert first_decision.churn_evidenced is False
    assert first_decision.reason == "no_history"

    moved = _order(price=100.1)
    moved_decision = _evaluate(state, [moved], now=5.0)[id(moved)]
    assert moved_decision.churn_evidenced is True
    assert moved_decision.reason == "wider_but_not_tight"


def test_newest_tight_prefix_clears_older_churn_evidence_after_stability_horizon():
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=100.0)], now=0.0)
    _evaluate(state, [_order(price=100.1)], now=5.0)
    _evaluate(state, [_order(price=100.1)], now=65.0)
    settled = _order(price=100.1)
    decision = _evaluate(state, [settled], now=125.0)[id(settled)]
    assert decision.churn_evidenced is False
    assert decision.reason == "stable_tight_prefix"
    assert decision.tight_prefix_count >= 2
    assert decision.tight_prefix_seconds >= 120.0


def test_generation_gap_breaks_provenance_and_fails_open():
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=100.0)], now=0.0)
    state.begin_generation()  # failed Rust planning attempt: no snapshot
    current = _order(price=100.1)
    decision = _evaluate(state, [current], now=10.0)[id(current)]
    assert decision.churn_evidenced is False
    assert decision.reason == "generation_gap"


def test_planning_time_gap_breaks_provenance_and_fails_open():
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=100.0)], now=0.0)
    current = _order(price=100.1)

    decision = _evaluate(state, [current], now=71.0)[id(current)]

    assert decision.churn_evidenced is False
    assert decision.reason == "time_gap"


def test_tight_snapshots_short_of_stability_horizon_do_not_prove_stability():
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=100.0)], now=0.0)
    _evaluate(state, [_order(price=100.0)], now=50.0)
    current = _order(price=100.0)

    decision = _evaluate(state, [current], now=100.0)[id(current)]

    assert decision.churn_evidenced is False
    assert decision.reason == "tight_history_short"
    assert decision.tight_prefix_count == 2
    assert decision.tight_prefix_seconds == 100.0


def test_slow_cumulative_drift_compares_current_to_each_snapshot():
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=100.0)], now=0.0)
    _evaluate(state, [_order(price=100.015)], now=5.0)
    current = _order(price=100.03)
    decision = _evaluate(state, [current], now=10.0)[id(current)]
    assert decision.churn_evidenced is True
    assert decision.reason == "wider_but_not_tight"


def test_pb_order_type_is_an_exact_historical_cohort_key():
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=100.0)], now=0.0)
    changed_type = _order(price=100.1, pb_order_type="entry_grid_normal_long")
    decision = _evaluate(state, [changed_type], now=5.0)[id(changed_type)]
    assert decision.churn_evidenced is False
    assert decision.reason == "uncertain_no_association"


def test_quantity_only_movement_is_churn_evidence():
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=100.0, qty=1.0)], now=0.0)
    moved = _order(price=100.0, qty=1.001)

    decision = _evaluate(state, [moved], now=5.0)[id(moved)]

    assert decision.churn_evidenced is True
    assert decision.reason == "wider_but_not_tight"


def test_move_beyond_wider_tolerance_is_uncertain_and_fails_open():
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=100.0)], now=0.0)
    moved = _order(price=101.0)

    decision = _evaluate(state, [moved], now=5.0)[id(moved)]

    assert decision.churn_evidenced is False
    assert decision.reason == "uncertain_no_association"


def test_semantic_cohort_keys_do_not_cross_associate():
    state = OrderChurnGateState()
    previous = _order(price=100.0)
    _evaluate(state, [previous], now=0.0)
    changed = _order(price=100.1)
    changed["reduce_only"] = True
    changed["side"] = "sell"
    changed["pb_order_type"] = "close_grid_long"

    decision = _evaluate(state, [changed], now=5.0)[id(changed)]

    assert decision.churn_evidenced is False
    assert decision.reason == "uncertain_no_association"


def test_one_to_one_matching_maximizes_cardinality_deterministically():
    current = normalize_ideal_orders(
        [_order(price=100.01), _order(price=100.03)]
    )
    previous = normalize_ideal_orders(
        [_order(price=100.0), _order(price=100.02)]
    )
    expected = deterministic_one_to_one_matches(current, previous, 0.0002)
    assert len(expected) == 2
    for _ in range(5):
        assert deterministic_one_to_one_matches(current, previous, 0.0002) == expected


def test_ladder_reordering_does_not_change_per_price_decisions():
    state_a = OrderChurnGateState()
    state_b = OrderChurnGateState()
    first = [_order(price=100.0), _order(price=101.0), _order(price=102.0)]
    _evaluate(state_a, list(first), now=0.0)
    _evaluate(state_b, list(reversed(first)), now=0.0)
    current_a = [_order(price=100.1), _order(price=101.0), _order(price=102.0)]
    current_b = list(reversed([dict(order) for order in current_a]))

    decisions_a = _evaluate(state_a, current_a, now=5.0)
    decisions_b = _evaluate(state_b, current_b, now=5.0)
    by_price_a = {
        order["price"]: decisions_a[id(order)].reason for order in current_a
    }
    by_price_b = {
        order["price"]: decisions_b[id(order)].reason for order in current_b
    }

    assert by_price_a == by_price_b


def test_complete_current_cohort_reserves_history_for_satisfied_peer():
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=100.0)], now=0.0)

    # The 100.0 ideal may already be satisfied by an actual resting order and
    # therefore never reach create admission. It must still reserve the only
    # historical observation before the unmatched 100.1 peer is classified.
    satisfied = _order(price=100.0)
    unmatched = _order(price=100.1)
    decisions = _evaluate(state, [satisfied, unmatched], now=5.0)

    assert decisions[id(satisfied)].reason == "tight_history_short"
    assert decisions[id(unmatched)].churn_evidenced is False
    assert decisions[id(unmatched)].reason == "uncertain_no_association"


def test_snapshot_and_attempt_windows_prune_one_item_at_a_time():
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=100.0)], now=0.0)
    _evaluate(state, [_order(price=100.0)], now=5.0)
    state.record_action_attempts(2, now_monotonic=0.0)
    state.record_action_attempts(1, now_monotonic=5.0)
    assert state.action_attempt_count(now_monotonic=9.0, window_seconds=10.0) == 3
    assert state.action_attempt_count(now_monotonic=11.0, window_seconds=10.0) == 1

    current = _order(price=100.0)
    _evaluate(state, [current], now=606.0)
    snapshots = state.history_by_symbol["BTC/USDT:USDT"]
    assert len(snapshots) == 1
    assert snapshots[0].monotonic_seconds == 606.0


def test_history_only_symbol_expires_without_refreshing_empty_snapshots():
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=100.0)], now=0.0)
    generation = state.begin_generation()

    state.evaluate_and_record(
        {},
        generation=generation,
        now_monotonic=601.0,
        tight_tolerance=0.0002,
        wider_tolerance=0.002,
        stability_seconds=120.0,
        window_seconds=600.0,
        max_generation_gap_seconds=70.0,
    )

    assert state.history_by_symbol == {}


def test_prepare_prunes_history_only_symbol_without_appending_empty_snapshot(monkeypatch):
    state = OrderChurnGateState()
    _evaluate(state, [_order(price=100.0)], now=0.0)

    class Bot:
        _order_churn_gate_state = state
        active_symbols = []
        open_orders = {}
        positions = {}

        @staticmethod
        def live_value(key):
            return {
                "order_replacement_churn_gate_activation_count": 10,
                "order_replacement_churn_gate_window_minutes": 10.0,
                "order_replacement_churn_gate_stability_minutes": 2.0,
                "order_match_tolerance_pct": 0.0002,
                "order_replacement_churn_gate_tracking_tolerance_pct": 0.002,
                "execution_delay_seconds": 2.0,
            }[key]

    monkeypatch.setattr(reconciler, "_order_churn_account_epoch", lambda _bot: ("e",))
    monkeypatch.setattr(
        reconciler,
        "_order_churn_symbol_compatibility_epochs",
        lambda _bot, symbols: {symbol: ("m",) for symbol in symbols},
    )
    monkeypatch.setattr(reconciler.time, "monotonic", lambda: 601.0)

    generation = state.begin_generation()
    reconciler.prepare_order_churn_evidence(Bot(), {}, generation=generation)

    assert state.history_by_symbol == {}


def test_live_generation_gap_allows_normal_scheduled_wait(monkeypatch):
    class Bot:
        @staticmethod
        def live_value(key):
            assert key == "execution_delay_seconds"
            return 2.0

    monkeypatch.setattr(
        reconciler,
        "_pb_const",
        lambda key: 30.0 if key == "EXECUTION_SCHEDULED_WAIT_SECONDS" else None,
    )

    max_gap = reconciler._order_churn_max_generation_gap_seconds(Bot())
    assert max_gap == 96.0

    state = OrderChurnGateState()
    first = _order(price=100.0)
    generation = state.begin_generation()
    state.evaluate_and_record(
        {first["symbol"]: [first]},
        generation=generation,
        now_monotonic=0.0,
        tight_tolerance=0.0002,
        wider_tolerance=0.002,
        stability_seconds=120.0,
        window_seconds=600.0,
        max_generation_gap_seconds=max_gap,
    )
    moved = _order(price=100.1)
    generation = state.begin_generation()
    decision = state.evaluate_and_record(
        {moved["symbol"]: [moved]},
        generation=generation,
        now_monotonic=40.0,
        tight_tolerance=0.0002,
        wider_tolerance=0.002,
        stability_seconds=120.0,
        window_seconds=600.0,
        max_generation_gap_seconds=max_gap,
    )[id(moved)]
    assert decision.churn_evidenced is True
    assert decision.reason == "wider_but_not_tight"


def test_first_prepare_emits_ram_reset_and_fails_open(monkeypatch):
    state = OrderChurnGateState()
    symbol = "BTC/USDT:USDT"
    ideal = _order(price=100.0)
    emitted = []

    class Bot:
        _order_churn_gate_state = state
        active_symbols = [symbol]
        open_orders = {}
        positions = {}

        @staticmethod
        def live_value(key):
            return {
                "order_replacement_churn_gate_activation_count": 10,
                "order_replacement_churn_gate_window_minutes": 10.0,
                "order_replacement_churn_gate_stability_minutes": 2.0,
                "order_match_tolerance_pct": 0.0002,
                "order_replacement_churn_gate_tracking_tolerance_pct": 0.002,
                "execution_delay_seconds": 2.0,
            }[key]

        @staticmethod
        def _emit_order_churn_evidence_event(**kwargs):
            emitted.append(kwargs)

    monkeypatch.setattr(reconciler, "_order_churn_account_epoch", lambda _bot: ("e",))
    monkeypatch.setattr(
        reconciler,
        "_order_churn_symbol_compatibility_epochs",
        lambda _bot, symbols: {item: ("m",) for item in symbols},
    )
    monkeypatch.setattr(reconciler.time, "monotonic", lambda: 1.0)

    generation = state.begin_generation()
    reconciler.prepare_order_churn_evidence(
        Bot(), {symbol: [ideal]}, generation=generation
    )

    assert ideal["_churn_evidence"] is False
    assert ideal["_churn_reason"] == "no_history"
    assert emitted[0]["reset"] is True
    assert emitted[0]["reset_count"] == 0


def test_active_rust_risk_pair_bypasses_observed_churn(monkeypatch):
    state = OrderChurnGateState()
    symbol = "BTC/USDT:USDT"

    class Bot:
        _order_churn_gate_state = state
        _order_churn_risk_active_pairs = ((symbol, "long"),)
        active_symbols = [symbol]
        open_orders = {}
        positions = {}

        @staticmethod
        def live_value(key):
            return {
                "order_replacement_churn_gate_activation_count": 10,
                "order_replacement_churn_gate_window_minutes": 10.0,
                "order_replacement_churn_gate_stability_minutes": 2.0,
                "order_match_tolerance_pct": 0.0002,
                "order_replacement_churn_gate_tracking_tolerance_pct": 0.002,
                "execution_delay_seconds": 2.0,
            }[key]

    monkeypatch.setattr(reconciler, "_order_churn_account_epoch", lambda _bot: ("e",))
    monkeypatch.setattr(
        reconciler,
        "_order_churn_symbol_compatibility_epochs",
        lambda _bot, symbols: {item: ("m",) for item in symbols},
    )
    monkeypatch.setattr(reconciler.time, "monotonic", lambda: 0.0)
    first = _order(price=100.0)
    reconciler.prepare_order_churn_evidence(
        Bot(), {symbol: [first]}, generation=state.begin_generation()
    )

    monkeypatch.setattr(reconciler.time, "monotonic", lambda: 5.0)
    moved = _order(price=100.1)
    reconciler.prepare_order_churn_evidence(
        Bot(), {symbol: [moved]}, generation=state.begin_generation()
    )

    assert moved["_churn_evidence"] is False
    assert moved["_churn_reason"] == "rust_risk_phase_active"


def test_downstream_normalization_outage_is_symbol_scoped(monkeypatch, caplog):
    state = OrderChurnGateState()
    btc = "BTC/USDT:USDT"
    eth = "ETH/USDT:USDT"
    valid = _order(price=100.0)
    invalid = {**_order(price=10.0), "symbol": eth}
    hostile_error = type("ApiKeySecretError", (ValueError,), {})
    secret = "normalization-secret https://example.invalid/normalization"

    class Bot:
        _order_churn_gate_state = state
        active_symbols = [btc, eth]
        open_orders = {}
        positions = {}

        @staticmethod
        def live_value(key):
            return {
                "order_replacement_churn_gate_activation_count": 10,
                "order_replacement_churn_gate_window_minutes": 10.0,
                "order_replacement_churn_gate_stability_minutes": 2.0,
                "order_match_tolerance_pct": 0.0002,
                "order_replacement_churn_gate_tracking_tolerance_pct": 0.002,
                "execution_delay_seconds": 2.0,
            }[key]

    monkeypatch.setattr(reconciler, "_order_churn_account_epoch", lambda _bot: ("e",))
    monkeypatch.setattr(
        reconciler,
        "_order_churn_symbol_compatibility_epochs",
        lambda _bot, symbols: {symbol: ("m",) for symbol in symbols},
    )
    monkeypatch.setattr(reconciler.time, "monotonic", lambda: 1.0)
    original_normalize = reconciler.normalize_ideal_orders

    def normalize_or_fail(orders):
        if orders[0]["symbol"] == eth:
            raise hostile_error(secret)
        return original_normalize(orders)

    monkeypatch.setattr(reconciler, "normalize_ideal_orders", normalize_or_fail)

    generation = state.begin_generation()
    with caplog.at_level("ERROR"):
        unavailable = reconciler.prepare_order_churn_evidence(
            Bot(), {btc: [valid], eth: [invalid]}, generation=generation
        )

    assert unavailable == {eth}
    assert btc in state.history_by_symbol
    assert eth not in state.history_by_symbol
    assert valid["_churn_reason"] == "no_history"
    assert invalid["_churn_reason"] == "normalization_unavailable"
    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert "error_type=ValueError" in rendered
    assert hostile_error.__name__ not in rendered
    assert secret not in rendered
    assert "example.invalid" not in rendered


def test_churn_evidence_emitter_failure_redacts_hostile_metadata(caplog):
    hostile_error = type("ApiKeySecretError", (RuntimeError,), {})
    secret = "emitter-secret https://example.invalid/emitter"
    state = OrderChurnGateState()

    class Bot:
        def _emit_order_churn_evidence_event(self, **_kwargs):
            raise hostile_error(secret)

    with caplog.at_level("DEBUG"):
        reconciler._emit_order_churn_evidence_summary(
            Bot(),
            state=state,
            generation=1,
            reset=False,
            decisions=[],
            symbols=["BTC/USDT:USDT"],
        )

    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert "error_type=RuntimeError" in rendered
    assert hostile_error.__name__ not in rendered
    assert secret not in rendered
    assert "example.invalid" not in rendered
    assert state.history_by_symbol == {}
    assert state.reset_count == 0


@pytest.mark.asyncio
async def test_unavailable_churn_normalization_leaves_only_that_symbol_untouched():
    btc = "BTC/USDT:USDT"
    eth = "ETH/USDT:USDT"
    btc_ideal = _order(price=100.0)
    eth_ideal = {**_order(price=10.0), "symbol": eth}

    class Bot:
        exchange = "fake"
        active_symbols = [btc, eth]
        open_orders = {btc: [], eth: []}
        positions = {
            btc: {"long": {"size": 0.0}, "short": {"size": 0.0}},
            eth: {"long": {"size": 0.0}, "short": {"size": 0.0}},
        }
        actual_orders = {btc: [], eth: []}

        @classmethod
        def _snapshot_actual_orders(cls, symbols, psides_by_symbol=None):
            del psides_by_symbol
            return {symbol: list(cls.actual_orders.get(symbol, [])) for symbol in symbols}

        @staticmethod
        def _reconcile_symbol_orders(symbol, actual, ideal, keys, **kwargs):
            return reconciler.reconcile_symbol_orders(
                Bot(), symbol, actual, ideal, keys, **kwargs
            )

        @staticmethod
        def _annotate_order_deltas(to_cancel, to_create):
            return to_cancel, to_create

        @staticmethod
        def _apply_order_match_tolerance(to_cancel, to_create):
            return to_cancel, to_create, 0

        @staticmethod
        async def _sort_orders_by_market_diff(orders, _label):
            return orders

        @staticmethod
        def _order_plan_summary_is_interesting(**_kwargs):
            return False

    _to_cancel, to_create = await reconciler.calc_orders_to_cancel_and_create_from_ideal(
        Bot(),
        {btc: [btc_ideal], eth: [eth_ideal]},
        apply_creation_guardrails=False,
        apply_mode_filters=False,
        collect_fresh_entry_eligibility=False,
        order_churn_unavailable_symbols={eth},
    )

    assert to_create == [btc_ideal]

    Bot.actual_orders = {btc: [], eth: [{"symbol": eth}]}
    _to_cancel, to_create = await reconciler.calc_orders_to_cancel_and_create_from_ideal(
        Bot(),
        {btc: [btc_ideal], eth: [eth_ideal]},
        apply_creation_guardrails=False,
        apply_mode_filters=False,
        collect_fresh_entry_eligibility=False,
        order_churn_unavailable_symbols={eth},
    )
    assert to_create == []

    Bot.actual_orders = {btc: [], eth: []}
    Bot.positions = {eth: {"long": {"size": 1.0}, "short": {"size": 0.0}}}
    _to_cancel, to_create = await reconciler.calc_orders_to_cancel_and_create_from_ideal(
        Bot(),
        {btc: [btc_ideal], eth: [eth_ideal]},
        apply_creation_guardrails=False,
        apply_mode_filters=False,
        collect_fresh_entry_eligibility=False,
        order_churn_unavailable_symbols={eth},
    )
    assert to_create == []

    Bot.positions = {eth: {"long": {"size": 0.0}}}
    _to_cancel, to_create = await reconciler.calc_orders_to_cancel_and_create_from_ideal(
        Bot(),
        {btc: [btc_ideal], eth: [eth_ideal]},
        apply_creation_guardrails=False,
        apply_mode_filters=False,
        collect_fresh_entry_eligibility=False,
        order_churn_unavailable_symbols={eth},
    )
    assert to_create == []


@pytest.mark.asyncio
async def test_unavailable_stateful_symbol_preserves_only_market_panic_create():
    btc = "BTC/USDT:USDT"
    eth = "ETH/USDT:USDT"
    ordinary = _order(price=100.0)
    market_panic = {
        **_order(price=100.0, pb_order_type="close_panic_long"),
        "side": "sell",
        "reduce_only": True,
        "type": "market",
    }
    unavailable = {**_order(price=10.0), "symbol": eth}

    class Bot:
        exchange = "fake"
        active_symbols = [btc, eth]
        open_orders = {btc: [], eth: [{"symbol": eth}]}
        positions = {
            btc: {"long": {"size": 0.0}, "short": {"size": 0.0}},
            eth: {"long": {"size": 0.0}, "short": {"size": 0.0}},
        }

        @staticmethod
        def _snapshot_actual_orders(symbols, psides_by_symbol=None):
            del psides_by_symbol
            return {
                symbol: ([{"symbol": eth}] if symbol == eth else [])
                for symbol in symbols
            }

        @staticmethod
        def _reconcile_symbol_orders(symbol, actual, ideal, keys, **kwargs):
            return reconciler.reconcile_symbol_orders(
                Bot(), symbol, actual, ideal, keys, **kwargs
            )

        @staticmethod
        def _annotate_order_deltas(to_cancel, to_create):
            return to_cancel, to_create

        @staticmethod
        def _apply_order_match_tolerance(to_cancel, to_create):
            return to_cancel, to_create, 0

        @staticmethod
        async def _sort_orders_by_market_diff(orders, _label):
            return orders

        @staticmethod
        def _order_plan_summary_is_interesting(**_kwargs):
            return False

    _to_cancel, to_create = await reconciler.calc_orders_to_cancel_and_create_from_ideal(
        Bot(),
        {btc: [ordinary, market_panic], eth: [unavailable]},
        apply_creation_guardrails=False,
        apply_mode_filters=False,
        collect_fresh_entry_eligibility=False,
        order_churn_unavailable_symbols={eth},
    )

    assert to_create == [market_panic]

    Bot.positions = {eth: {"long": {"size": 1.0}}}
    _to_cancel, to_create = await reconciler.calc_orders_to_cancel_and_create_from_ideal(
        Bot(),
        {btc: [ordinary, market_panic], eth: [unavailable]},
        apply_creation_guardrails=False,
        apply_mode_filters=False,
        collect_fresh_entry_eligibility=False,
        order_churn_unavailable_symbols={eth},
    )
    assert to_create == []


def test_epoch_reset_clears_history_but_not_attempts():
    state = OrderChurnGateState()
    assert state.reset_history_for_epoch(("epoch", 1)) is False
    _evaluate(state, [_order(price=100.0)], now=0.0)
    state.record_action_attempts(1, now_monotonic=0.0)
    assert state.reset_history_for_epoch(("epoch", 2)) is True
    assert state.history_by_symbol == {}
    assert state.action_attempt_count(now_monotonic=1.0, window_seconds=600.0) == 1


def test_symbol_epoch_reset_is_scoped_and_preserves_attempts():
    state = OrderChurnGateState()
    state.reset_history_for_symbol_epochs(
        {"BTC/USDT:USDT": (0.1, 0.001), "ETH/USDT:USDT": (0.01, 0.001)}
    )
    generation = state.begin_generation()
    state.evaluate_and_record(
        {
            "BTC/USDT:USDT": [_order(price=100.0)],
            "ETH/USDT:USDT": [
                {**_order(price=10.0), "symbol": "ETH/USDT:USDT"}
            ],
        },
        generation=generation,
        now_monotonic=0.0,
        tight_tolerance=0.0002,
        wider_tolerance=0.002,
        stability_seconds=120.0,
        window_seconds=600.0,
        max_generation_gap_seconds=70.0,
    )
    state.record_action_attempts(1, now_monotonic=0.0)

    changed = state.reset_history_for_symbol_epochs(
        {"BTC/USDT:USDT": (0.01, 0.001), "ETH/USDT:USDT": (0.01, 0.001)}
    )

    assert changed == {"BTC/USDT:USDT"}
    assert "BTC/USDT:USDT" not in state.history_by_symbol
    assert "ETH/USDT:USDT" in state.history_by_symbol
    assert state.action_attempt_count(now_monotonic=1.0, window_seconds=600.0) == 1


def test_console_projection_throttle_preserves_first_transition_and_periodic_summary():
    state = OrderChurnGateState()
    signature = (("allowance_exhausted", "BTC/USDT:USDT"),)

    assert state.should_log_console_event(
        "create_deferred", signature, now_monotonic=100.0, repeat_seconds=300.0
    ) == (True, 0)
    assert state.should_log_console_event(
        "create_deferred", signature, now_monotonic=101.0, repeat_seconds=300.0
    ) == (False, 0)
    assert state.should_log_console_event(
        "create_deferred", signature, now_monotonic=399.9, repeat_seconds=300.0
    ) == (False, 0)
    assert state.should_log_console_event(
        "create_deferred", signature, now_monotonic=400.0, repeat_seconds=300.0
    ) == (True, 2)


def test_console_projection_throttle_logs_material_signature_change_immediately():
    state = OrderChurnGateState()

    assert state.should_log_console_event(
        "history_reset", "account", now_monotonic=1.0
    ) == (True, 0)
    assert state.should_log_console_event(
        "history_reset", "account", now_monotonic=2.0
    ) == (False, 0)
    assert state.should_log_console_event(
        "history_reset", ("market", "BTC"), now_monotonic=3.0
    ) == (True, 0)
