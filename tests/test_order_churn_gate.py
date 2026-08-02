from __future__ import annotations

import json
import sys
from copy import deepcopy

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


def _raw_rust_order(**overrides) -> dict:
    order = {
        "symbol_idx": 0,
        "pside": "long",
        "qty": 1.0,
        "price": 100.0,
        "order_type": "entry_initial_normal_long",
        "execution_type": "limit",
        "execution_priority": "ordinary",
    }
    order.update(overrides)
    return order


def _raw_rust_input(
    *,
    long_mode=None,
    short_mode="manual",
    long_pos_size=1.0,
    short_pos_size=0.0,
    tradable=True,
    long_wallet_exposure_limit=1.0,
    short_wallet_exposure_limit=1.0,
    long_n_positions=1,
    short_n_positions=1,
    long_total_wallet_exposure_limit=1.0,
    short_total_wallet_exposure_limit=1.0,
    bid=100.0,
    ask=100.0,
    qty_step=0.001,
    price_step=0.01,
    min_qty=0.001,
    min_cost=1.0,
    c_mult=1.0,
    **global_overrides,
) -> dict:
    global_input = {
        "hedge_mode": True,
        "strategy_kind": "trailing_martingale",
        "auto_unstuck_allowed": True,
        "market_orders_allowed": False,
        "market_order_near_touch_threshold": 0.001,
        "panic_close_market": False,
        "max_realized_loss_pct": 0.1,
        "global_bot_params": {
            "long": {
                "n_positions": long_n_positions,
                "total_wallet_exposure_limit": long_total_wallet_exposure_limit,
                "risk_twel_enforcer_enabled": True,
                "risk_twel_enforcer_threshold": 1.0,
                "hsl_enabled": False,
                "hsl_panic_close_order_type": "market",
            },
            "short": {
                "n_positions": short_n_positions,
                "total_wallet_exposure_limit": short_total_wallet_exposure_limit,
                "risk_twel_enforcer_enabled": True,
                "risk_twel_enforcer_threshold": 1.0,
                "hsl_enabled": False,
                "hsl_panic_close_order_type": "market",
            },
        },
    }
    global_bot_params_override = global_overrides.pop("global_bot_params", None)
    global_input.update(global_overrides)
    if global_bot_params_override is not None:
        for pside in ("long", "short"):
            global_input["global_bot_params"][pside].update(
                global_bot_params_override.get(pside, {})
            )
    return {
        "global": global_input,
        "symbols": [
            {
                "symbol_idx": 0,
                "order_book": {"bid": bid, "ask": ask},
                "exchange": {
                    "qty_step": qty_step,
                    "price_step": price_step,
                    "min_qty": min_qty,
                    "min_cost": min_cost,
                    "c_mult": c_mult,
                },
                "tradable": tradable,
                "long": {
                    "mode": long_mode,
                    "position": {"size": long_pos_size, "price": 100.0},
                    "bot_params": {
                        "wallet_exposure_limit": long_wallet_exposure_limit,
                        "risk_wel_enforcer_enabled": True,
                        "risk_wel_enforcer_threshold": 1.0,
                        "unstuck_enabled": True,
                        "unstuck_close_pct": 0.1,
                        "unstuck_loss_allowance_pct": 0.1,
                        "unstuck_threshold": 0.1,
                    },
                },
                "short": {
                    "mode": short_mode,
                    "position": {"size": short_pos_size, "price": 100.0},
                    "bot_params": {
                        "wallet_exposure_limit": short_wallet_exposure_limit,
                        "risk_wel_enforcer_enabled": True,
                        "risk_wel_enforcer_threshold": 1.0,
                        "unstuck_enabled": True,
                        "unstuck_close_pct": 0.1,
                        "unstuck_loss_allowance_pct": 0.1,
                        "unstuck_threshold": 0.1,
                    },
                },
            }
        ],
    }


def _raw_rust_output(orders=None, *, symbol_states=None) -> dict:
    if orders is None:
        orders = []
    if symbol_states is None:
        symbol_states = [
            {
                "symbol_idx": 0,
                "long": {
                    "input_mode": None,
                    "effective_mode": "normal",
                    "active": True,
                    "allow_initial": True,
                },
                "short": {
                    "input_mode": "manual",
                    "effective_mode": "manual",
                    "active": False,
                    "allow_initial": False,
                },
            }
        ]
    return {
        "orders": orders,
        "diagnostics": {
            "warnings": [],
            "symbol_states": symbol_states,
            "loss_gate_blocks": [],
            "min_effective_cost_blocks": [],
            "forager_selections": [],
        },
    }


def _raw_rust_output_for_long_mode(orders, input_mode: str) -> dict:
    out = _raw_rust_output(orders)
    out["diagnostics"]["symbol_states"][0]["long"].update(
        input_mode=input_mode,
        effective_mode=input_mode,
    )
    return out


def test_passivbot_rust_stub_emits_required_diagnostic_collections():
    import passivbot_rust as pbr

    if not getattr(pbr, "__is_stub__", False):
        pytest.skip("real Rust extension is loaded")
    out = json.loads(pbr.compute_ideal_orders_json(json.dumps({"symbols": []})))

    for field in (
        "loss_gate_blocks",
        "min_effective_cost_blocks",
        "forager_selections",
    ):
        assert out["diagnostics"][field] == []

    with pytest.raises(ValueError, match="unknown order type"):
        pbr.order_type_snake_to_id("entry_bogus_long")

    stub_input = _raw_rust_input(long_pos_size=0.0, long_wallet_exposure_limit=0.0)
    stub_out = json.loads(pbr.compute_ideal_orders_json(json.dumps(stub_input)))
    assert stub_out["diagnostics"]["symbol_states"][0]["long"]["active"] is False


def _raw_loss_gate_block(**overrides) -> dict:
    block = {
        "symbol_idx": 0,
        "pside": "long",
        "order_type": "close_auto_reduce_wel_long",
        "qty": -1.0,
        "price": 100.0,
        "projected_pnl": -110.0,
        "balance_before": 1_000.0,
        "projected_balance_after": 890.0,
        "balance_peak": 1_000.0,
        "balance_floor": 900.0,
        "max_realized_loss_pct": 0.1,
    }
    block.update(overrides)
    return block


def _raw_min_effective_cost_block(**overrides) -> dict:
    block = {
        "symbol_idx": 0,
        "pside": "long",
        "balance": 1_000.0,
        "effective_limit": 10.0,
        "entry_initial_qty_pct": 0.01,
        "projected_initial_cost": 5.0,
        "effective_min_cost": 10.0,
    }
    block.update(overrides)
    return block


def _raw_forager_selection(**overrides) -> dict:
    selection = {
        "pside": "long",
        "slots_to_fill": 1,
        "score_hysteresis_pct": 0.1,
        "selected_symbol_indices": [0],
        "incumbent_symbol_indices": [0],
        "top_scores": [
            {
                "symbol_idx": 0,
                "rank": 0,
                "score": 1.0,
                "volume_component": 1.0,
                "ema_readiness_component": 1.0,
                "volatility_component": 1.0,
                "selected": True,
                "incumbent": True,
            }
        ],
        "hysteresis_events": [
            {
                "incumbent_symbol_idx": 0,
                "incumbent_score": 1.0,
                "challenger_symbol_idx": 0,
                "challenger_score": 1.0,
                "score_gap": 0.0,
                "kept_incumbent": True,
            }
        ],
    }
    selection.update(overrides)
    return selection


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


def test_raw_rust_order_batch_with_unknown_symbol_is_fatal_as_a_whole():
    orders = [
        _raw_rust_order(),
        _raw_rust_order(symbol_idx=999),
    ]

    with pytest.raises(FatalBotException, match="unknown symbol_idx 999"):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output(orders), {0: SYMBOL}, _raw_rust_input()
        )


def test_raw_rust_output_requires_orders_field():
    with pytest.raises(FatalBotException, match="missing required orders field"):
        reconciler.validate_rust_orchestrator_output(
            {}, {0: SYMBOL}, _raw_rust_input()
        )


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"pside": "both"}, "invalid pside"),
        ({"pside": []}, "invalid pside"),
        ({"qty": 0.0}, "invalid qty"),
        ({"qty": float("nan")}, "invalid qty"),
        ({"qty": 10**400}, "invalid qty"),
        ({"price": 0.0}, "invalid price"),
        ({"price": 10**400}, "invalid price"),
        ({"order_type": ""}, "invalid order_type"),
        ({"order_type": "not_an_order_long"}, "invalid order_type"),
        ({"qty": -1.0}, "qty sign disagrees"),
        ({"execution_type": "stop"}, "invalid execution_type"),
        ({"execution_type": {}}, "invalid execution_type"),
        ({"execution_priority": "optional"}, "invalid execution_priority"),
        ({"execution_priority": []}, "invalid execution_priority"),
    ],
)
def test_raw_rust_output_rejects_every_malformed_order_field(overrides, error):
    with pytest.raises(FatalBotException, match=error):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output([_raw_rust_order(**overrides)]),
            {0: SYMBOL},
            _raw_rust_input(),
        )


def test_raw_rust_output_rejects_unknown_order_type_when_lookup_is_permissive(
    monkeypatch,
):
    import passivbot_rust as pbr

    monkeypatch.setattr(pbr, "order_type_snake_to_id", lambda _name: 0)
    monkeypatch.setattr(
        pbr, "order_type_id_to_snake", lambda _type_id: "entry_initial_normal_long"
    )

    with pytest.raises(FatalBotException, match="invalid order_type"):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output(
                [_raw_rust_order(order_type="entry_bogus_long")]
            ),
            {0: SYMBOL},
            _raw_rust_input(),
        )


@pytest.mark.parametrize(
    "orders",
    [
        [_raw_rust_order(qty=-1.01, order_type="close_grid_long")],
        [
            _raw_rust_order(qty=-0.6, order_type="close_grid_long"),
            _raw_rust_order(
                qty=-0.5,
                price=101.0,
                order_type="close_trailing_long",
            ),
        ],
    ],
)
def test_raw_rust_output_rejects_close_wave_exceeding_submitted_position(orders):
    with pytest.raises(FatalBotException, match="exceeds submitted position"):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output(orders),
            {0: SYMBOL},
            _raw_rust_input(long_pos_size=1.0),
        )


def test_raw_rust_output_accepts_close_wave_equal_to_submitted_position():
    orders = [
        _raw_rust_order(qty=-0.6, order_type="close_grid_long"),
        _raw_rust_order(
            qty=-0.4,
            price=101.0,
            order_type="close_trailing_long",
        ),
    ]

    assert reconciler.validate_rust_orchestrator_output(
        _raw_rust_output(orders),
        {0: SYMBOL},
        _raw_rust_input(long_pos_size=1.0),
    ) == orders


@pytest.mark.parametrize(
    ("input_mode", "position_size", "order"),
    [
        ("manual", 1.0, _raw_rust_order()),
        ("panic", 1.0, _raw_rust_order()),
        (
            "panic",
            1.0,
            _raw_rust_order(qty=-1.0, order_type="close_grid_long"),
        ),
        (
            "normal",
            1.0,
            _raw_rust_order(
                qty=-1.0,
                order_type="close_panic_long",
                execution_priority="risk_critical",
            ),
        ),
        ("tp_only", 1.0, _raw_rust_order()),
        ("graceful_stop", 0.0, _raw_rust_order()),
    ],
    ids=[
        "manual-entry",
        "panic-entry",
        "panic-ordinary-close",
        "normal-panic-close",
        "tp-only-entry",
        "flat-graceful-stop-entry",
    ],
)
def test_raw_rust_output_rejects_order_family_forbidden_by_submitted_mode(
    input_mode, position_size, order
):
    out = _raw_rust_output([order])
    out["diagnostics"]["symbol_states"][0]["long"]["input_mode"] = input_mode

    with pytest.raises(FatalBotException, match="inconsistent with its submitted mode"):
        reconciler.validate_rust_orchestrator_output(
            out,
            {0: SYMBOL},
            _raw_rust_input(
                long_mode=input_mode,
                long_pos_size=position_size,
            ),
        )


def test_raw_rust_output_keeps_graceful_stop_dca_with_open_position():
    order = _raw_rust_order()
    out = _raw_rust_output([order])
    out["diagnostics"]["symbol_states"][0]["long"]["input_mode"] = (
        "graceful_stop"
    )

    assert reconciler.validate_rust_orchestrator_output(
        out,
        {0: SYMBOL},
        _raw_rust_input(long_mode="graceful_stop", long_pos_size=1.0),
    ) == [order]


@pytest.mark.parametrize(
    "order_type",
    [
        "close_panic_long",
        "close_auto_reduce_twel_long",
        "close_auto_reduce_wel_long",
        "close_unstuck_long",
    ],
)
def test_raw_rust_output_rejects_ordinary_priority_for_protective_orders(
    order_type,
):
    order = _raw_rust_order(qty=-1.0, order_type=order_type)
    input_mode = "panic" if order_type == "close_panic_long" else None
    out = (
        _raw_rust_output_for_long_mode([order], input_mode)
        if input_mode is not None
        else _raw_rust_output([order])
    )

    with pytest.raises(FatalBotException, match="inconsistent with its order_type"):
        reconciler.validate_rust_orchestrator_output(
            out,
            {0: SYMBOL},
            _raw_rust_input(long_mode=input_mode),
        )


def test_raw_rust_output_accepts_risk_critical_priority_for_protective_order():
    order = _raw_rust_order(
        qty=-1.0,
        order_type="close_unstuck_long",
        execution_priority="risk_critical",
    )

    assert reconciler.validate_rust_orchestrator_output(
        _raw_rust_output([order]),
        {0: SYMBOL},
        _raw_rust_input(),
    ) == [order]


def test_raw_rust_output_rejects_unstuck_when_submitted_gate_is_false():
    order = _raw_rust_order(
        qty=-1.0,
        order_type="close_unstuck_long",
        execution_priority="risk_critical",
    )

    with pytest.raises(FatalBotException, match="submitted auto-unstuck gate"):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output([order]),
            {0: SYMBOL},
            _raw_rust_input(auto_unstuck_allowed=False),
        )


@pytest.mark.parametrize(
    ("order_type", "scope", "field", "value"),
    [
        ("close_unstuck_long", "symbol", "unstuck_enabled", False),
        ("close_unstuck_long", "symbol", "unstuck_loss_allowance_pct", 0.0),
        ("close_unstuck_long", "symbol", "unstuck_close_pct", 0.0),
        ("close_unstuck_long", "symbol", "unstuck_threshold", 0.0),
        (
            "close_auto_reduce_wel_long",
            "symbol",
            "risk_wel_enforcer_enabled",
            False,
        ),
        (
            "close_auto_reduce_wel_long",
            "symbol",
            "risk_wel_enforcer_threshold",
            0.0,
        ),
        (
            "close_auto_reduce_twel_long",
            "global",
            "risk_twel_enforcer_enabled",
            False,
        ),
        (
            "close_auto_reduce_twel_long",
            "global",
            "risk_twel_enforcer_threshold",
            0.0,
        ),
    ],
)
def test_raw_rust_output_rejects_disabled_protective_reducer_family(
    order_type, scope, field, value
):
    orchestrator_input = _raw_rust_input()
    params = (
        orchestrator_input["symbols"][0]["long"]["bot_params"]
        if scope == "symbol"
        else orchestrator_input["global"]["global_bot_params"]["long"]
    )
    params[field] = value
    order = _raw_rust_order(
        qty=-1.0,
        order_type=order_type,
        execution_priority="risk_critical",
    )

    with pytest.raises(FatalBotException, match="submitted reducer enablement"):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output([order]),
            {0: SYMBOL},
            orchestrator_input,
        )


@pytest.mark.parametrize(
    ("order", "input_mode"),
    [
        (
            _raw_rust_order(execution_priority="risk_critical"),
            None,
        ),
        (
            _raw_rust_order(
                qty=-1.0,
                order_type="close_grid_long",
                execution_priority="risk_critical",
            ),
            None,
        ),
        (
            _raw_rust_order(qty=-1.0, order_type="close_grid_long"),
            "graceful_stop",
        ),
    ],
    ids=[
        "risk-critical-entry",
        "risk-critical-normal-close",
        "ordinary-graceful-stop-close",
    ],
)
def test_raw_rust_output_rejects_priority_inconsistent_with_full_rust_rule(
    order, input_mode
):
    out = _raw_rust_output([order])
    out["diagnostics"]["symbol_states"][0]["long"]["input_mode"] = input_mode

    with pytest.raises(FatalBotException, match="inconsistent with"):
        reconciler.validate_rust_orchestrator_output(
            out, {0: SYMBOL}, _raw_rust_input(long_mode=input_mode)
        )


def test_raw_rust_output_accepts_risk_critical_graceful_stop_close():
    order = _raw_rust_order(
        qty=-1.0,
        order_type="close_grid_long",
        execution_priority="risk_critical",
    )
    out = _raw_rust_output([order])
    out["diagnostics"]["symbol_states"][0]["long"]["input_mode"] = "graceful_stop"

    assert reconciler.validate_rust_orchestrator_output(
        out, {0: SYMBOL}, _raw_rust_input(long_mode="graceful_stop")
    ) == [order]


def test_raw_rust_output_rejects_input_mode_echo_changed_from_submitted_mode():
    order = _raw_rust_order(
        qty=-1.0,
        order_type="close_grid_long",
        execution_priority="risk_critical",
    )
    out = _raw_rust_output([order])
    out["diagnostics"]["symbol_states"][0]["long"]["input_mode"] = "graceful_stop"

    with pytest.raises(
        FatalBotException, match="inconsistent with its submitted input"
    ):
        reconciler.validate_rust_orchestrator_output(
            out, {0: SYMBOL}, _raw_rust_input(long_mode="normal")
        )


def test_raw_rust_output_rejects_market_entry_when_input_forbids_it():
    order = _raw_rust_order(execution_type="market")

    with pytest.raises(
        FatalBotException, match="inconsistent with its submitted input"
    ):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output([order]),
            {0: SYMBOL},
            _raw_rust_input(market_orders_allowed=False),
        )


def test_raw_rust_output_accepts_market_entry_when_policy_selects_it():
    order = _raw_rust_order(execution_type="market")

    assert reconciler.validate_rust_orchestrator_output(
        _raw_rust_output([order]),
        {0: SYMBOL},
        _raw_rust_input(market_orders_allowed=True),
    ) == [order]


@pytest.mark.parametrize(
    ("qty", "price", "execution_type"),
    [
        (1.0, 90.0, "limit"),
        (-1.0, 110.0, "limit"),
        (1.0, 99.95, "market"),
        (-1.0, 100.05, "market"),
    ],
    ids=["far-buy-limit", "far-sell-limit", "near-buy-market", "near-sell-market"],
)
def test_raw_rust_output_accepts_non_panic_execution_type_matching_policy(
    qty, price, execution_type
):
    order = _raw_rust_order(
        qty=qty,
        price=price,
        order_type=(
            "entry_initial_normal_long" if qty > 0.0 else "close_grid_long"
        ),
        execution_type=execution_type,
    )

    assert reconciler.validate_rust_orchestrator_output(
        _raw_rust_output([order]),
        {0: SYMBOL},
        _raw_rust_input(market_orders_allowed=True),
    ) == [order]


@pytest.mark.parametrize(
    ("qty", "price", "execution_type"),
    [
        (1.0, 90.0, "market"),
        (-1.0, 110.0, "market"),
        (1.0, 99.95, "limit"),
        (-1.0, 100.05, "limit"),
        (1.0, 101.0, "limit"),
        (-1.0, 99.0, "limit"),
    ],
    ids=[
        "far-buy-market",
        "far-sell-market",
        "near-buy-limit",
        "near-sell-limit",
        "crossing-buy-limit",
        "crossing-sell-limit",
    ],
)
def test_raw_rust_output_rejects_non_panic_execution_type_mismatch(
    qty, price, execution_type
):
    order = _raw_rust_order(
        qty=qty,
        price=price,
        order_type=(
            "entry_initial_normal_long" if qty > 0.0 else "close_grid_long"
        ),
        execution_type=execution_type,
    )

    with pytest.raises(
        FatalBotException, match="inconsistent with its submitted input"
    ):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output([order]),
            {0: SYMBOL},
            _raw_rust_input(market_orders_allowed=True),
        )


@pytest.mark.parametrize(
    "global_overrides",
    [
        {
            "global_bot_params": {
                "long": {
                    "hsl_enabled": True,
                    "hsl_panic_close_order_type": "market",
                },
                "short": {
                    "hsl_enabled": False,
                    "hsl_panic_close_order_type": "market",
                },
            }
        },
        {"panic_close_market": True},
    ],
    ids=["side-local-hsl-market", "global-panic-market"],
)
def test_raw_rust_output_allows_configured_market_panic_close_when_markets_disabled(
    global_overrides,
):
    order = _raw_rust_order(
        qty=-1.0,
        order_type="close_panic_long",
        execution_type="market",
        execution_priority="risk_critical",
    )

    assert reconciler.validate_rust_orchestrator_output(
        _raw_rust_output_for_long_mode([order], "panic"),
        {0: SYMBOL},
        _raw_rust_input(
            long_mode="panic",
            market_orders_allowed=False,
            **global_overrides,
        ),
    ) == [order]


def test_raw_rust_output_rejects_partial_panic_close():
    order = _raw_rust_order(
        qty=-0.5,
        order_type="close_panic_long",
        execution_priority="risk_critical",
    )

    with pytest.raises(FatalBotException, match="panic quantity"):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output_for_long_mode([order], "panic"),
            {0: SYMBOL},
            _raw_rust_input(long_mode="panic", long_pos_size=1.0),
        )


@pytest.mark.parametrize(
    "global_overrides",
    [
        {
            "global_bot_params": {
                "long": {
                    "hsl_enabled": True,
                    "hsl_panic_close_order_type": "market",
                },
                "short": {
                    "hsl_enabled": False,
                    "hsl_panic_close_order_type": "market",
                },
            }
        },
        {"panic_close_market": True},
    ],
    ids=["side-local-hsl-market", "global-panic-market"],
)
def test_raw_rust_output_rejects_limit_panic_close_when_market_is_configured(
    global_overrides,
):
    order = _raw_rust_order(
        qty=-1.0,
        order_type="close_panic_long",
        execution_type="limit",
        execution_priority="risk_critical",
    )

    with pytest.raises(
        FatalBotException, match="inconsistent with its submitted input"
    ):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output_for_long_mode([order], "panic"),
            {0: SYMBOL},
            _raw_rust_input(
                long_mode="panic",
                market_orders_allowed=False,
                **global_overrides,
            ),
        )


def test_raw_rust_output_rejects_market_panic_close_when_limit_is_configured():
    order = _raw_rust_order(
        qty=-1.0,
        order_type="close_panic_long",
        execution_type="market",
        execution_priority="risk_critical",
    )
    global_bot_params = {
        "long": {
            "hsl_enabled": True,
            "hsl_panic_close_order_type": "limit",
        },
        "short": {
            "hsl_enabled": False,
            "hsl_panic_close_order_type": "market",
        },
    }

    with pytest.raises(
        FatalBotException, match="inconsistent with its submitted input"
    ):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output_for_long_mode([order], "panic"),
            {0: SYMBOL},
            _raw_rust_input(
                long_mode="panic",
                market_orders_allowed=True,
                global_bot_params=global_bot_params,
            ),
        )


@pytest.mark.parametrize(
    "global_overrides",
    [
        {},
        {
            "global_bot_params": {
                "long": {
                    "hsl_enabled": True,
                    "hsl_panic_close_order_type": "limit",
                },
                "short": {
                    "hsl_enabled": False,
                    "hsl_panic_close_order_type": "market",
                },
            }
        },
    ],
    ids=["hsl-disabled", "hsl-limit"],
)
def test_raw_rust_output_rejects_unconfigured_market_panic_close(global_overrides):
    order = _raw_rust_order(
        qty=-1.0,
        order_type="close_panic_long",
        execution_type="market",
        execution_priority="risk_critical",
    )

    with pytest.raises(
        FatalBotException, match="inconsistent with its submitted input"
    ):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output_for_long_mode([order], "panic"),
            {0: SYMBOL},
            _raw_rust_input(
                long_mode="panic",
                market_orders_allowed=False,
                **global_overrides,
            ),
        )


def test_raw_rust_output_rejects_colliding_conversion_identities():
    orders = [_raw_rust_order(), _raw_rust_order()]

    with pytest.raises(FatalBotException, match="collide under conversion identity"):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output(orders),
            {0: SYMBOL},
            _raw_rust_input(),
        )


def test_raw_rust_output_keeps_distinct_structured_conversion_identities():
    orders = [
        _raw_rust_order(qty=1.0, price=23.0),
        _raw_rust_order(qty=1.02, price=3.0),
    ]

    reconciler.validate_rust_orchestrator_output(
        _raw_rust_output(orders),
        {0: SYMBOL},
        _raw_rust_input(),
    )


def test_raw_rust_output_requires_complete_symbol_state_coverage():
    with pytest.raises(FatalBotException, match="do not cover"):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output(symbol_states=[]),
            {0: SYMBOL},
            _raw_rust_input(),
        )


def test_raw_rust_output_requires_explicit_input_mode():
    out = _raw_rust_output()
    del out["diagnostics"]["symbol_states"][0]["long"]["input_mode"]

    with pytest.raises(FatalBotException, match="invalid long input_mode"):
        reconciler.validate_rust_orchestrator_output(
            out, {0: SYMBOL}, _raw_rust_input()
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("input_mode", []), ("effective_mode", {})],
)
def test_raw_rust_output_rejects_unhashable_symbol_state_modes(field, value):
    out = _raw_rust_output()
    out["diagnostics"]["symbol_states"][0]["long"][field] = value

    with pytest.raises(FatalBotException, match=rf"invalid long {field}"):
        reconciler.validate_rust_orchestrator_output(
            out, {0: SYMBOL}, _raw_rust_input()
        )


@pytest.mark.parametrize(
    "input_overrides",
    [
        {"tradable": False},
        {"long_wallet_exposure_limit": 0.0},
    ],
    ids=["non-tradable", "zero-wallet-exposure-limit"],
)
def test_raw_rust_output_rejects_active_state_for_submitted_ineligible_side(
    input_overrides,
):
    with pytest.raises(FatalBotException, match="inconsistent with submitted eligibility"):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output(),
            {0: SYMBOL},
            _raw_rust_input(**input_overrides),
        )


@pytest.mark.parametrize(
    "input_overrides",
    [
        {"long_n_positions": 0},
        {"long_total_wallet_exposure_limit": 0.0},
    ],
    ids=["zero-global-position-cap", "zero-global-wallet-exposure-limit"],
)
def test_raw_rust_output_rejects_active_state_for_flat_globally_disabled_side(
    input_overrides,
):
    out = _raw_rust_output()
    out["diagnostics"]["symbol_states"][0]["long"]["effective_mode"] = "manual"
    with pytest.raises(FatalBotException, match="inconsistent with submitted eligibility"):
        reconciler.validate_rust_orchestrator_output(
            out,
            {0: SYMBOL},
            _raw_rust_input(long_pos_size=0.0, **input_overrides),
        )


def test_raw_rust_output_accepts_inactive_state_for_submitted_ineligible_side():
    out = _raw_rust_output()
    out["diagnostics"]["symbol_states"][0]["long"]["active"] = False
    out["diagnostics"]["symbol_states"][0]["long"]["allow_initial"] = False

    assert reconciler.validate_rust_orchestrator_output(
        out,
        {0: SYMBOL},
        _raw_rust_input(tradable=False),
    ) == []


@pytest.mark.parametrize(
    "input_overrides",
    [
        {"tradable": False},
        {"long_wallet_exposure_limit": 0.0},
    ],
    ids=["non-tradable", "zero-wallet-exposure-limit"],
)
def test_raw_rust_output_rejects_flat_entry_for_submitted_ineligible_side(
    input_overrides,
):
    out = _raw_rust_output([_raw_rust_order()])
    out["diagnostics"]["symbol_states"][0]["long"]["active"] = False
    out["diagnostics"]["symbol_states"][0]["long"]["allow_initial"] = False

    with pytest.raises(FatalBotException, match="submitted mode or eligibility"):
        reconciler.validate_rust_orchestrator_output(
            out,
            {0: SYMBOL},
            _raw_rust_input(long_pos_size=0.0, **input_overrides),
        )


@pytest.mark.parametrize(
    "input_overrides",
    [
        {"long_n_positions": 0},
        {"long_total_wallet_exposure_limit": 0.0},
    ],
    ids=["zero-global-position-cap", "zero-global-wallet-exposure-limit"],
)
def test_raw_rust_output_rejects_flat_entry_for_globally_disabled_side(
    input_overrides,
):
    out = _raw_rust_output([_raw_rust_order()])
    out["diagnostics"]["symbol_states"][0]["long"].update(
        effective_mode="manual",
        active=False,
        allow_initial=False,
    )

    with pytest.raises(FatalBotException, match="globally disabled long"):
        reconciler.validate_rust_orchestrator_output(
            out,
            {0: SYMBOL},
            _raw_rust_input(long_pos_size=0.0, **input_overrides),
        )


@pytest.mark.parametrize(
    "order",
    [
        _raw_rust_order(order_type="entry_grid_normal_long"),
        _raw_rust_order(qty=-1.0, order_type="close_grid_long"),
    ],
    ids=["entry", "close"],
)
def test_raw_rust_output_rejects_orders_for_globally_disabled_held_side(order):
    out = _raw_rust_output([order])
    out["diagnostics"]["symbol_states"][0]["long"]["effective_mode"] = "manual"
    with pytest.raises(FatalBotException, match="globally disabled long"):
        reconciler.validate_rust_orchestrator_output(
            out,
            {0: SYMBOL},
            _raw_rust_input(
                long_pos_size=1.0,
                long_total_wallet_exposure_limit=0.0,
            ),
        )


def test_raw_rust_output_rejects_flat_entry_contradicted_by_symbol_state():
    out = _raw_rust_output([_raw_rust_order()])
    out["diagnostics"]["symbol_states"][0]["long"].update(
        active=False,
        allow_initial=False,
    )

    with pytest.raises(FatalBotException, match="contradicts submitted symbol state"):
        reconciler.validate_rust_orchestrator_output(
            out,
            {0: SYMBOL},
            _raw_rust_input(long_pos_size=0.0),
        )


@pytest.mark.parametrize(
    "order_type",
    [
        "entry_grid_normal_long",
        "entry_initial_partial_long",
        "entry_trailing_cropped_long",
    ],
)
def test_raw_rust_output_rejects_impossible_flat_entry_batch(order_type):
    with pytest.raises(FatalBotException, match="flat submitted side"):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output([_raw_rust_order(order_type=order_type)]),
            {0: SYMBOL},
            _raw_rust_input(long_pos_size=0.0),
        )


@pytest.mark.parametrize("strategy_kind", ["trailing_martingale", "trailing_grid_v7"])
def test_raw_rust_output_accepts_recursive_flat_grid_entry_batch(strategy_kind):
    orders = [
        _raw_rust_order(order_type="entry_initial_normal_long"),
        _raw_rust_order(
            qty=1.1,
            price=99.0,
            order_type="entry_grid_normal_long",
        ),
    ]

    assert reconciler.validate_rust_orchestrator_output(
        _raw_rust_output(orders),
        {0: SYMBOL},
        _raw_rust_input(long_pos_size=0.0, strategy_kind=strategy_kind),
    ) == orders


def test_raw_rust_output_accepts_flat_ema_anchor_entry_batch():
    order = _raw_rust_order(order_type="entry_ema_anchor_long")

    assert reconciler.validate_rust_orchestrator_output(
        _raw_rust_output([order]),
        {0: SYMBOL},
        _raw_rust_input(long_pos_size=0.0, strategy_kind="ema_anchor"),
    ) == [order]


def _two_flat_long_symbol_inputs(*, forced_normal: bool = False):
    orchestrator_input = _raw_rust_input(
        long_pos_size=0.0,
        long_mode="normal" if forced_normal else None,
    )
    second_symbol = deepcopy(orchestrator_input["symbols"][0])
    second_symbol["symbol_idx"] = 1
    orchestrator_input["symbols"].append(second_symbol)
    symbol_states = []
    for symbol_idx in (0, 1):
        symbol_states.append(
            {
                "symbol_idx": symbol_idx,
                "long": {
                    "input_mode": "normal" if forced_normal else None,
                    "effective_mode": "normal",
                    "active": True,
                    "allow_initial": True,
                },
                "short": {
                    "input_mode": "manual",
                    "effective_mode": "manual",
                    "active": False,
                    "allow_initial": False,
                },
            }
        )
    return orchestrator_input, _raw_rust_output(symbol_states=symbol_states)


def test_raw_rust_output_rejects_flat_active_set_above_submitted_position_cap():
    orchestrator_input, out = _two_flat_long_symbol_inputs()

    with pytest.raises(FatalBotException, match="exceeds submitted position cap"):
        reconciler.validate_rust_orchestrator_output(
            out,
            {0: SYMBOL, 1: "ETH/USDT:USDT"},
            orchestrator_input,
        )


def test_raw_rust_output_allows_forced_normal_position_cap_expansion():
    orchestrator_input, out = _two_flat_long_symbol_inputs(forced_normal=True)

    assert reconciler.validate_rust_orchestrator_output(
        out,
        {0: SYMBOL, 1: "ETH/USDT:USDT"},
        orchestrator_input,
    ) == []


def test_raw_rust_output_rejects_one_way_initial_against_opposite_position():
    out = _raw_rust_output(
        [
            _raw_rust_order(
                pside="short",
                qty=-1.0,
                order_type="entry_initial_normal_short",
            )
        ]
    )
    out["diagnostics"]["symbol_states"][0]["short"].update(
        input_mode=None,
        effective_mode="normal",
        active=True,
        allow_initial=True,
    )

    with pytest.raises(FatalBotException, match="one-way position-side exclusion"):
        reconciler.validate_rust_orchestrator_output(
            out,
            {0: SYMBOL},
            _raw_rust_input(
                long_pos_size=1.0,
                short_pos_size=0.0,
                short_mode=None,
                hedge_mode=False,
            ),
        )


def test_raw_rust_output_rejects_two_one_way_initial_sides():
    out = _raw_rust_output()
    out["diagnostics"]["symbol_states"][0]["long"].update(
        active=True,
        allow_initial=True,
    )
    out["diagnostics"]["symbol_states"][0]["short"].update(
        input_mode=None,
        effective_mode="normal",
        active=True,
        allow_initial=True,
    )

    with pytest.raises(FatalBotException, match="one-way position-side exclusion"):
        reconciler.validate_rust_orchestrator_output(
            out,
            {0: SYMBOL},
            _raw_rust_input(
                long_pos_size=0.0,
                short_pos_size=0.0,
                short_mode=None,
                hedge_mode=False,
            ),
        )


@pytest.mark.parametrize(
    ("strategy_kind", "order_type"),
    [
        ("trailing_martingale", "entry_ema_anchor_long"),
        ("trailing_grid_v7", "close_ema_anchor_long"),
        ("ema_anchor", "entry_initial_normal_long"),
        ("ema_anchor", "close_grid_long"),
    ],
)
def test_raw_rust_output_rejects_order_family_from_another_strategy(
    strategy_kind,
    order_type,
):
    is_close = order_type.startswith("close_")
    order = _raw_rust_order(
        qty=-1.0 if is_close else 1.0,
        order_type=order_type,
    )

    with pytest.raises(FatalBotException, match="inconsistent with submitted strategy"):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output([order]),
            {0: SYMBOL},
            _raw_rust_input(strategy_kind=strategy_kind),
        )


def test_raw_rust_output_rejects_competing_protective_reducers():
    orders = [
        _raw_rust_order(
            qty=-0.4,
            order_type="close_unstuck_long",
            execution_priority="risk_critical",
        ),
        _raw_rust_order(
            qty=-0.5,
            price=101.0,
            order_type="close_auto_reduce_wel_long",
            execution_priority="risk_critical",
        ),
    ]

    with pytest.raises(FatalBotException, match="competing protective reducers"):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output(orders),
            {0: SYMBOL},
            _raw_rust_input(long_pos_size=1.0),
        )


@pytest.mark.parametrize(
    ("order_overrides", "input_overrides", "error"),
    [
        ({"qty": 0.05}, {"qty_step": 0.01, "min_qty": 0.1}, "entry minimum"),
        ({"qty": 0.1}, {"min_cost": 20.0}, "entry minimum"),
        ({"qty": 1.05}, {"qty_step": 0.1}, "qty_step"),
    ],
    ids=["min-qty", "min-cost", "qty-step"],
)
def test_raw_rust_output_rejects_entry_quantity_outside_exchange_constraints(
    order_overrides,
    input_overrides,
    error,
):
    with pytest.raises(FatalBotException, match=error):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output([_raw_rust_order(**order_overrides)]),
            {0: SYMBOL},
            _raw_rust_input(**input_overrides),
        )


def test_raw_rust_output_accepts_step_aligned_entry_at_effective_minimum():
    assert reconciler.validate_rust_orchestrator_output(
        _raw_rust_output([_raw_rust_order(qty=1.1)]),
        {0: SYMBOL},
        _raw_rust_input(qty_step=0.1, min_qty=0.1, min_cost=105.0),
    ) == [
        _raw_rust_order(qty=1.1),
    ]


def test_raw_rust_output_rejects_off_step_tiny_entry_quantity():
    with pytest.raises(FatalBotException, match="qty_step"):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output([_raw_rust_order(qty=1.5e-12)]),
            {0: SYMBOL},
            _raw_rust_input(qty_step=1e-12, min_qty=1e-12, min_cost=0.0),
        )


def test_raw_rust_output_accepts_aligned_tiny_entry_quantity():
    order = _raw_rust_order(qty=1e-12)
    assert reconciler.validate_rust_orchestrator_output(
        _raw_rust_output([order]),
        {0: SYMBOL},
        _raw_rust_input(qty_step=1e-12, min_qty=1e-12, min_cost=0.0),
    ) == [order]


def test_raw_rust_output_rejects_quantity_below_tiny_minimum_cost():
    with pytest.raises(FatalBotException, match="entry minimum"):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output([_raw_rust_order(qty=1e-12, price=1.0)]),
            {0: SYMBOL},
            _raw_rust_input(
                bid=1.0,
                ask=1.0,
                qty_step=1e-12,
                min_qty=0.0,
                min_cost=1e-9,
            ),
        )


def test_raw_rust_output_accepts_quantity_at_tiny_minimum_cost():
    order = _raw_rust_order(qty=1e-9, price=1.0)
    assert reconciler.validate_rust_orchestrator_output(
        _raw_rust_output([order]),
        {0: SYMBOL},
        _raw_rust_input(
            bid=1.0,
            ask=1.0,
            qty_step=1e-12,
            min_qty=0.0,
            min_cost=1e-9,
        ),
    ) == [order]


def test_rust_effective_min_qty_rounds_positive_sub_step_minimum_up():
    assert reconciler._rust_effective_min_qty(
        1e9,
        (1.0, 0.01, 0.0, 1.0, 1.0),
    ) == 1.0


def test_rust_effective_min_qty_ceils_genuinely_above_step_minimum():
    assert reconciler._rust_effective_min_qty(
        4999.999975,
        (0.001, 0.01, 0.0, 5.0, 1.0),
    ) == 0.002


def test_rust_effective_min_qty_preserves_tiny_aligned_exchange_minimum():
    assert reconciler._rust_effective_min_qty(
        100.0,
        (1e-12, 0.01, 1e-12, 0.0, 1.0),
    ) == 1e-12


def test_rust_effective_min_qty_preserves_aligned_multiplication_rounding_down():
    assert reconciler._rust_effective_min_qty(
        100.0,
        (0.03, 0.01, 0.33, 0.0, 1.0),
    ) == 0.33


def test_raw_rust_output_rejects_limit_price_off_submitted_price_step():
    with pytest.raises(FatalBotException, match="price_step"):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output([_raw_rust_order(price=100.005)]),
            {0: SYMBOL},
            _raw_rust_input(price_step=0.01),
        )


def test_raw_rust_output_accepts_limit_price_aligned_to_submitted_price_step():
    order = _raw_rust_order(price=100.01)
    assert reconciler.validate_rust_orchestrator_output(
        _raw_rust_output([order]),
        {0: SYMBOL},
        _raw_rust_input(price_step=0.01, bid=100.01, ask=100.01),
    ) == [order]


def test_raw_rust_output_rejects_off_step_tiny_limit_price():
    with pytest.raises(FatalBotException, match="price_step"):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output([_raw_rust_order(price=1.5e-12)]),
            {0: SYMBOL},
            _raw_rust_input(
                bid=1.5e-12,
                ask=1.5e-12,
                price_step=1e-12,
                min_cost=0.0,
            ),
        )


def test_raw_rust_output_accepts_aligned_tiny_limit_price():
    order = _raw_rust_order(price=1e-12)
    assert reconciler.validate_rust_orchestrator_output(
        _raw_rust_output([order]),
        {0: SYMBOL},
        _raw_rust_input(
            bid=1e-12,
            ask=1e-12,
            price_step=1e-12,
            min_cost=0.0,
        ),
    ) == [order]


@pytest.mark.parametrize(
    ("order_overrides", "input_overrides", "error"),
    [
        (
            {"qty": -0.5, "order_type": "close_grid_long"},
            {"long_pos_size": 2.0, "qty_step": 0.1, "min_qty": 1.0},
            "close minimum",
        ),
        (
            {"qty": -1.05, "order_type": "close_grid_long"},
            {"long_pos_size": 2.0, "qty_step": 0.1},
            "qty_step",
        ),
    ],
    ids=["below-effective-minimum", "off-quantity-step"],
)
def test_raw_rust_output_rejects_close_quantity_outside_exchange_constraints(
    order_overrides,
    input_overrides,
    error,
):
    with pytest.raises(FatalBotException, match=error):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output([_raw_rust_order(**order_overrides)]),
            {0: SYMBOL},
            _raw_rust_input(**input_overrides),
        )


def test_raw_rust_output_accepts_exact_remaining_below_minimum_dust_close():
    order = _raw_rust_order(
        qty=-0.55,
        order_type="close_grid_long",
    )
    assert reconciler.validate_rust_orchestrator_output(
        _raw_rust_output([order]),
        {0: SYMBOL},
        _raw_rust_input(long_pos_size=0.55, qty_step=0.1, min_qty=1.0),
    ) == [order]


@pytest.mark.parametrize("position_size", [1e-13, 1e-12])
def test_raw_rust_output_rejects_close_for_position_rust_trims_as_dust(position_size):
    order = _raw_rust_order(
        qty=-position_size,
        order_type="close_grid_long",
    )

    with pytest.raises(FatalBotException, match="Rust's dust threshold"):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output([order]),
            {0: SYMBOL},
            _raw_rust_input(long_pos_size=position_size),
        )


def test_raw_rust_output_accepts_exact_remaining_off_step_full_close():
    order = _raw_rust_order(
        qty=-1.005,
        order_type="close_grid_long",
    )
    assert reconciler.validate_rust_orchestrator_output(
        _raw_rust_output([order]),
        {0: SYMBOL},
        _raw_rust_input(long_pos_size=1.005, qty_step=0.01, min_qty=0.01),
    ) == [order]


def test_raw_rust_output_accepts_aligned_partial_close_at_exchange_minimum():
    order = _raw_rust_order(
        qty=-0.07,
        order_type="close_grid_long",
    )
    assert reconciler.validate_rust_orchestrator_output(
        _raw_rust_output([order]),
        {0: SYMBOL},
        _raw_rust_input(long_pos_size=0.2, qty_step=0.01, min_qty=0.07),
    ) == [order]


def test_raw_rust_output_rejects_off_step_tiny_close_quantity():
    order = _raw_rust_order(
        qty=-1.5e-12,
        order_type="close_grid_long",
    )
    with pytest.raises(FatalBotException, match="qty_step"):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output([order]),
            {0: SYMBOL},
            _raw_rust_input(
                long_pos_size=1e-11,
                qty_step=1e-12,
                min_qty=1e-12,
                min_cost=0.0,
            ),
        )


def test_raw_rust_output_keeps_held_entry_for_submitted_nontradable_side():
    order = _raw_rust_order(order_type="entry_grid_normal_long")
    out = _raw_rust_output([order])
    out["diagnostics"]["symbol_states"][0]["long"]["active"] = False
    out["diagnostics"]["symbol_states"][0]["long"]["allow_initial"] = False

    assert reconciler.validate_rust_orchestrator_output(
        out,
        {0: SYMBOL},
        _raw_rust_input(tradable=False, long_pos_size=1.0),
    ) == [order]


def test_raw_rust_output_keeps_active_held_side_when_globally_disabled():
    out = _raw_rust_output()
    out["diagnostics"]["symbol_states"][0]["long"]["effective_mode"] = "manual"
    assert reconciler.validate_rust_orchestrator_output(
        out,
        {0: SYMBOL},
        _raw_rust_input(
            long_pos_size=1.0,
            long_n_positions=0,
            long_total_wallet_exposure_limit=0.0,
        ),
    ) == []


def test_raw_rust_output_rejects_incorrect_recognized_effective_mode():
    out = _raw_rust_output_for_long_mode([], "panic")
    out["diagnostics"]["symbol_states"][0]["long"]["effective_mode"] = "normal"

    with pytest.raises(FatalBotException, match="effective_mode inconsistent"):
        reconciler.validate_rust_orchestrator_output(
            out,
            {0: SYMBOL},
            _raw_rust_input(long_mode="panic", long_pos_size=1.0),
        )


def test_raw_rust_output_accepts_held_graceful_stop_effective_normal_mode():
    out = _raw_rust_output_for_long_mode([], "graceful_stop")
    out["diagnostics"]["symbol_states"][0]["long"]["effective_mode"] = "normal"

    assert reconciler.validate_rust_orchestrator_output(
        out,
        {0: SYMBOL},
        _raw_rust_input(long_mode="graceful_stop", long_pos_size=1.0),
    ) == []


def test_raw_rust_output_accepts_tiny_nonzero_graceful_stop_position_order():
    order = _raw_rust_order(order_type="entry_grid_normal_long")
    out = _raw_rust_output_for_long_mode([order], "graceful_stop")
    out["diagnostics"]["symbol_states"][0]["long"]["effective_mode"] = "normal"

    assert reconciler.validate_rust_orchestrator_output(
        out,
        {0: SYMBOL},
        _raw_rust_input(long_mode="graceful_stop", long_pos_size=1e-13),
    ) == [order]


def test_raw_rust_output_rejects_inactive_state_for_eligible_managed_position():
    out = _raw_rust_output()
    out["diagnostics"]["symbol_states"][0]["long"]["active"] = False
    out["diagnostics"]["symbol_states"][0]["long"]["allow_initial"] = False

    with pytest.raises(FatalBotException, match="inconsistent with submitted managed position"):
        reconciler.validate_rust_orchestrator_output(
            out,
            {0: SYMBOL},
            _raw_rust_input(long_pos_size=1.0),
        )


def test_raw_rust_output_accepts_inactive_state_for_manual_managed_position():
    out = _raw_rust_output_for_long_mode([], "manual")
    out["diagnostics"]["symbol_states"][0]["long"]["active"] = False
    out["diagnostics"]["symbol_states"][0]["long"]["allow_initial"] = False

    assert reconciler.validate_rust_orchestrator_output(
        out,
        {0: SYMBOL},
        _raw_rust_input(long_mode="manual", long_pos_size=1.0),
    ) == []


@pytest.mark.parametrize(
    "field",
    ["loss_gate_blocks", "min_effective_cost_blocks", "forager_selections"],
)
def test_raw_rust_output_requires_consumed_diagnostic_collection(field):
    out = _raw_rust_output()
    del out["diagnostics"][field]

    with pytest.raises(FatalBotException, match=rf"missing required {field}"):
        reconciler.validate_rust_orchestrator_output(
            out, {0: SYMBOL}, _raw_rust_input()
        )


@pytest.mark.parametrize(
    ("loss_gate_blocks", "error"),
    [
        ({}, "must be a list"),
        (["invalid"], "must be a mapping"),
        ([{"symbol_idx": 999}], "invalid symbol_idx"),
        ([{"symbol_idx": 0, "pside": "both"}], "invalid pside"),
        ([_raw_loss_gate_block(projected_pnl=10**400)], "invalid projected_pnl"),
    ],
)
def test_raw_rust_output_rejects_malformed_loss_gate_diagnostics(
    loss_gate_blocks, error
):
    out = _raw_rust_output()
    out["diagnostics"]["loss_gate_blocks"] = loss_gate_blocks

    with pytest.raises(FatalBotException, match=error):
        reconciler.validate_rust_orchestrator_output(
            out, {0: SYMBOL}, _raw_rust_input()
        )


def test_raw_rust_output_rejects_incomplete_loss_gate_block():
    out = _raw_rust_output()
    out["diagnostics"]["loss_gate_blocks"] = [
        {
            "symbol_idx": 0,
            "pside": "long",
            "order_type": "close_auto_reduce_wel_long",
        }
    ]

    with pytest.raises(FatalBotException, match="invalid qty"):
        reconciler.validate_rust_orchestrator_output(
            out, {0: SYMBOL}, _raw_rust_input()
        )


def test_raw_rust_output_accepts_consistent_loss_gate_block():
    out = _raw_rust_output()
    out["diagnostics"]["loss_gate_blocks"] = [_raw_loss_gate_block()]

    assert reconciler.validate_rust_orchestrator_output(
        out, {0: SYMBOL}, _raw_rust_input()
    ) == []


def test_raw_rust_output_rejects_loss_gate_block_for_flat_side():
    out = _raw_rust_output()
    out["diagnostics"]["loss_gate_blocks"] = [_raw_loss_gate_block()]

    with pytest.raises(FatalBotException, match="requires a submitted position"):
        reconciler.validate_rust_orchestrator_output(
            out, {0: SYMBOL}, _raw_rust_input(long_pos_size=0.0)
        )


@pytest.mark.parametrize("long_mode", ["manual", "panic"])
def test_raw_rust_output_rejects_loss_gate_block_for_non_gateable_mode(long_mode):
    out = _raw_rust_output_for_long_mode([], long_mode)
    if long_mode == "manual":
        out["diagnostics"]["symbol_states"][0]["long"]["active"] = False
        out["diagnostics"]["symbol_states"][0]["long"]["allow_initial"] = False
    out["diagnostics"]["loss_gate_blocks"] = [_raw_loss_gate_block()]

    with pytest.raises(FatalBotException, match="submitted mode or eligibility"):
        reconciler.validate_rust_orchestrator_output(
            out,
            {0: SYMBOL},
            _raw_rust_input(long_mode=long_mode),
        )


def test_raw_rust_output_rejects_loss_gate_block_for_globally_disabled_side():
    out = _raw_rust_output()
    out["diagnostics"]["symbol_states"][0]["long"]["effective_mode"] = "manual"
    out["diagnostics"]["loss_gate_blocks"] = [_raw_loss_gate_block()]

    with pytest.raises(FatalBotException, match="globally disabled long"):
        reconciler.validate_rust_orchestrator_output(
            out,
            {0: SYMBOL},
            _raw_rust_input(long_total_wallet_exposure_limit=0.0),
        )


def test_raw_rust_output_rejects_loss_gate_block_from_another_strategy():
    out = _raw_rust_output()
    out["diagnostics"]["loss_gate_blocks"] = [
        _raw_loss_gate_block(order_type="close_ema_anchor_long")
    ]

    with pytest.raises(FatalBotException, match="inconsistent with submitted strategy"):
        reconciler.validate_rust_orchestrator_output(
            out,
            {0: SYMBOL},
            _raw_rust_input(strategy_kind="trailing_martingale"),
        )


@pytest.mark.parametrize(
    ("submitted_max_realized_loss_pct", "block_max_realized_loss_pct", "error"),
    [
        (1.0, 0.1, "submitted realized-loss gate is disabled"),
        (0.2, 0.1, "inconsistent with submitted policy"),
    ],
)
def test_raw_rust_output_rejects_loss_gate_block_inconsistent_with_submitted_policy(
    submitted_max_realized_loss_pct, block_max_realized_loss_pct, error
):
    out = _raw_rust_output()
    out["diagnostics"]["loss_gate_blocks"] = [
        _raw_loss_gate_block(max_realized_loss_pct=block_max_realized_loss_pct)
    ]

    with pytest.raises(FatalBotException, match=error):
        reconciler.validate_rust_orchestrator_output(
            out,
            {0: SYMBOL},
            _raw_rust_input(
                max_realized_loss_pct=submitted_max_realized_loss_pct
            ),
        )


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"order_type": "entry_initial_normal_long"}, "must be a close order"),
        (
            {"order_type": "close_panic_long"},
            "panic order_type bypasses the realized-loss gate",
        ),
        ({"qty": 1.0}, "qty sign disagrees"),
        ({"projected_pnl": 0.0}, "negative projected_pnl"),
        ({"projected_balance_after": 891.0}, "inconsistent projected balance"),
        ({"balance_floor": 899.0}, "inconsistent balance floor"),
        (
            {"projected_pnl": -100.0, "projected_balance_after": 900.0},
            "does not cross balance floor",
        ),
    ],
)
def test_raw_rust_output_rejects_impossible_loss_gate_block(overrides, error):
    out = _raw_rust_output()
    out["diagnostics"]["loss_gate_blocks"] = [
        _raw_loss_gate_block(**overrides)
    ]

    with pytest.raises(FatalBotException, match=error):
        reconciler.validate_rust_orchestrator_output(
            out, {0: SYMBOL}, _raw_rust_input()
        )


@pytest.mark.parametrize(
    ("diagnostics", "error"),
    [
        ({"min_effective_cost_blocks": {}}, "must be a list"),
        (
            {
                "min_effective_cost_blocks": [
                    _raw_min_effective_cost_block(symbol_idx="0")
                ]
            },
            "invalid symbol_idx",
        ),
        (
            {
                "min_effective_cost_blocks": [
                    _raw_min_effective_cost_block(balance=None)
                ]
            },
            "invalid balance",
        ),
        (
            {
                "min_effective_cost_blocks": [
                    _raw_min_effective_cost_block(balance=10**400)
                ]
            },
            "invalid balance",
        ),
        ({"forager_selections": {}}, "must be a list"),
        (
            {"forager_selections": [_raw_forager_selection(slots_to_fill="1")]},
            "invalid slots_to_fill",
        ),
        (
            {
                "forager_selections": [
                    _raw_forager_selection(selected_symbol_indices=["0"])
                ]
            },
            "invalid symbol_idx",
        ),
        (
            {"forager_selections": [_raw_forager_selection(top_scores={})]},
            "top_scores must be a list",
        ),
        (
            {
                "forager_selections": [
                    _raw_forager_selection(
                        top_scores=[
                            {
                                **_raw_forager_selection()["top_scores"][0],
                                "score": "1.0",
                            }
                        ]
                    )
                ]
            },
            "invalid score",
        ),
        (
            {
                "forager_selections": [
                    _raw_forager_selection(
                        top_scores=[
                            {
                                **_raw_forager_selection()["top_scores"][0],
                                "score": 10**400,
                            }
                        ]
                    )
                ]
            },
            "invalid score",
        ),
        (
            {"forager_selections": [_raw_forager_selection(hysteresis_events={})]},
            "hysteresis_events must be a list",
        ),
        (
            {
                "forager_selections": [
                    _raw_forager_selection(
                        hysteresis_events=[
                            {
                                **_raw_forager_selection()["hysteresis_events"][0],
                                "kept_incumbent": 1,
                            }
                        ]
                    )
                ]
            },
            "invalid kept_incumbent",
        ),
        (
            {
                "forager_selections": [
                    _raw_forager_selection(
                        hysteresis_events=[
                            {
                                **_raw_forager_selection()["hysteresis_events"][0],
                                "score_gap": 10**400,
                            }
                        ]
                    )
                ]
            },
            "invalid score_gap",
        ),
    ],
)
def test_raw_rust_output_rejects_malformed_consumed_diagnostics(diagnostics, error):
    out = _raw_rust_output()
    out["diagnostics"].update(diagnostics)

    with pytest.raises(FatalBotException, match=error):
        reconciler.validate_rust_orchestrator_output(
            out, {0: SYMBOL}, _raw_rust_input(long_pos_size=0.0)
        )


def test_raw_rust_output_accepts_complete_consumed_diagnostics():
    out = _raw_rust_output()
    out["diagnostics"].update(
        {
            "min_effective_cost_blocks": [_raw_min_effective_cost_block()],
            "forager_selections": [_raw_forager_selection()],
        }
    )

    assert reconciler.validate_rust_orchestrator_output(
        out, {0: SYMBOL}, _raw_rust_input(long_pos_size=0.0)
    ) == []


def test_raw_rust_output_rejects_forager_selection_disagreeing_with_active_state():
    orchestrator_input, out = _two_flat_long_symbol_inputs()
    out["diagnostics"]["symbol_states"][1]["long"].update(
        active=False,
        allow_initial=False,
    )
    out["diagnostics"]["forager_selections"] = [
        _raw_forager_selection(selected_symbol_indices=[1])
    ]

    with pytest.raises(FatalBotException, match="disagree with submitted flat active"):
        reconciler.validate_rust_orchestrator_output(
            out,
            {0: SYMBOL, 1: "ETH/USDT:USDT"},
            orchestrator_input,
        )


def test_raw_rust_output_accepts_forager_selection_losing_one_way_tie_break():
    order = _raw_rust_order(order_type="entry_initial_normal_long")
    out = _raw_rust_output([order])
    out["diagnostics"]["symbol_states"][0]["short"].update(
        input_mode=None,
        effective_mode="normal",
        active=True,
        allow_initial=False,
    )
    out["diagnostics"]["forager_selections"] = [
        _raw_forager_selection(pside="long"),
        _raw_forager_selection(pside="short"),
    ]

    assert reconciler.validate_rust_orchestrator_output(
        out,
        {0: SYMBOL},
        _raw_rust_input(
            hedge_mode=False,
            long_pos_size=0.0,
            short_pos_size=0.0,
            short_mode=None,
            min_cost=0.0,
        ),
    ) == [order]


@pytest.mark.parametrize(
    ("warnings", "error"),
    [
        (None, "warnings must be a list"),
        ({}, "warnings must be a list"),
        (["bad"], "exactly one warning variant"),
        ([{}], "exactly one warning variant"),
        ([{"unknown": {}}], "invalid warning variant"),
        (
            [{"disabled_pside_has_position": {"symbol_idx": 0}}],
            "invalid warning fields",
        ),
        (
            [
                {
                    "non_tradable_has_position": {
                        "symbol_idx": 0,
                        "pside": [],
                    }
                }
            ],
            "invalid pside",
        ),
        (
            [
                {
                    "twel_repair_blocked_by_loss_gate": {
                        "pside": "long",
                        "current_twe": 0.6,
                        "twel_repair_target": 0.5,
                        "policy": "unknown",
                        "candidate_count": 1,
                        "blocked_order_count": 1,
                        "projected_twe_after_allowed_reductions": 0.6,
                    }
                }
            ],
            "invalid policy",
        ),
    ],
)
def test_raw_rust_output_rejects_malformed_warnings(warnings, error):
    out = _raw_rust_output()
    out["diagnostics"]["warnings"] = warnings

    with pytest.raises(FatalBotException, match=error):
        reconciler.validate_rust_orchestrator_output(
            out, {0: SYMBOL}, _raw_rust_input()
        )


@pytest.mark.parametrize("policy", [[], {}], ids=["array", "object"])
def test_raw_rust_output_rejects_unhashable_warning_policy_as_fatal(policy):
    out = _raw_rust_output()
    out["diagnostics"]["warnings"] = [
        {
            "twel_repair_blocked_by_loss_gate": {
                "pside": "long",
                "current_twe": 0.6,
                "twel_repair_target": 0.5,
                "policy": policy,
                "candidate_count": 1,
                "blocked_order_count": 1,
                "projected_twe_after_allowed_reductions": 0.6,
            }
        }
    ]

    with pytest.raises(FatalBotException, match="invalid policy"):
        reconciler.parse_and_validate_rust_orchestrator_output(
            json.dumps(out), {0: SYMBOL}, _raw_rust_input()
        )


def test_raw_rust_output_requires_warnings_collection():
    out = _raw_rust_output()
    del out["diagnostics"]["warnings"]

    with pytest.raises(FatalBotException, match="missing required warnings"):
        reconciler.validate_rust_orchestrator_output(
            out, {0: SYMBOL}, _raw_rust_input()
        )


def test_raw_rust_output_accepts_each_warning_variant():
    out = _raw_rust_output()
    out["diagnostics"]["warnings"] = [
        {
            "disabled_pside_has_position": {
                "symbol_idx": 0,
                "pside": "long",
            }
        },
        {
            "non_tradable_has_position": {
                "symbol_idx": 0,
                "pside": "short",
            }
        },
        {
            "twel_repair_blocked_by_loss_gate": {
                "pside": "long",
                "current_twe": 0.6,
                "twel_repair_target": 0.5,
                "policy": "reduce_portfolio",
                "candidate_count": 1,
                "blocked_order_count": 1,
                "projected_twe_after_allowed_reductions": 0.6,
            }
        },
    ]

    assert reconciler.validate_rust_orchestrator_output(
        out, {0: SYMBOL}, _raw_rust_input()
    ) == []


def test_raw_rust_output_malformed_json_is_fatal():
    with pytest.raises(FatalBotException, match="malformed JSON"):
        reconciler.parse_and_validate_rust_orchestrator_output(
            "{", {0: SYMBOL}, _raw_rust_input()
        )


def test_raw_rust_output_duplicate_json_key_is_fatal():
    out_json = '{"orders":[{"malformed":true}],"orders":[]}'

    with pytest.raises(FatalBotException, match="malformed JSON"):
        reconciler.parse_and_validate_rust_orchestrator_output(
            out_json, {0: SYMBOL}, _raw_rust_input()
        )


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_raw_rust_output_nonstandard_json_numeric_constant_is_fatal(constant):
    out_json = f'{{"diagnostics":{{"warnings":[{constant}]}}}}'

    with pytest.raises(FatalBotException, match="malformed JSON"):
        reconciler.parse_and_validate_rust_orchestrator_output(
            out_json, {0: SYMBOL}, _raw_rust_input()
        )


def test_raw_rust_output_exponent_overflow_json_number_is_fatal():
    out_json = '{"diagnostics":{"warnings":[1e400]}}'

    with pytest.raises(FatalBotException, match="malformed JSON"):
        reconciler.parse_and_validate_rust_orchestrator_output(
            out_json, {0: SYMBOL}, _raw_rust_input()
        )


def test_raw_rust_output_integer_digit_limit_failure_is_fatal():
    max_digits = sys.get_int_max_str_digits()
    if max_digits == 0:
        pytest.skip("Python integer string conversion limit is disabled")
    out_json = '{"value":' + ("1" * (max_digits + 1)) + "}"

    with pytest.raises(FatalBotException, match="malformed JSON"):
        reconciler.parse_and_validate_rust_orchestrator_output(
            out_json, {0: SYMBOL}, _raw_rust_input()
        )


def test_raw_rust_output_decoder_recursion_failure_is_fatal(monkeypatch):
    def raise_recursion_error(*_args, **_kwargs):
        raise RecursionError("decoder nesting limit exceeded")

    monkeypatch.setattr(reconciler.json, "loads", raise_recursion_error)

    with pytest.raises(FatalBotException, match="malformed JSON"):
        reconciler.parse_and_validate_rust_orchestrator_output(
            "{}", {0: SYMBOL}, _raw_rust_input()
        )


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
