from __future__ import annotations

import json
import sys

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


def _raw_rust_input(*, long_mode=None, short_mode="manual", **global_overrides) -> dict:
    global_input = {
        "market_orders_allowed": False,
        "panic_close_market": False,
        "global_bot_params": {
            "long": {
                "hsl_enabled": False,
                "hsl_panic_close_order_type": "market",
            },
            "short": {
                "hsl_enabled": False,
                "hsl_panic_close_order_type": "market",
            },
        },
    }
    global_input.update(global_overrides)
    return {
        "global": global_input,
        "symbols": [
            {
                "symbol_idx": 0,
                "long": {"mode": long_mode},
                "short": {"mode": short_mode},
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
            "symbol_states": symbol_states,
            "loss_gate_blocks": [],
        },
    }


def test_passivbot_rust_stub_emits_required_loss_gate_collection():
    import passivbot_rust as pbr

    if not getattr(pbr, "__is_stub__", False):
        pytest.skip("real Rust extension is loaded")
    out = json.loads(pbr.compute_ideal_orders_json(json.dumps({"symbols": []})))

    assert out["diagnostics"]["loss_gate_blocks"] == []


def _raw_loss_gate_block(**overrides) -> dict:
    block = {
        "symbol_idx": 0,
        "pside": "long",
        "order_type": "close_auto_reduce_wel_long",
        "qty": -1.0,
        "price": 100.0,
        "projected_pnl": -1.0,
        "balance_before": 1_000.0,
        "projected_balance_after": 999.0,
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
        ({"qty": 0.0}, "invalid qty"),
        ({"qty": float("nan")}, "invalid qty"),
        ({"qty": 10**400}, "invalid qty"),
        ({"price": 0.0}, "invalid price"),
        ({"price": 10**400}, "invalid price"),
        ({"order_type": ""}, "invalid order_type"),
        ({"order_type": "not_an_order_long"}, "invalid order_type"),
        ({"qty": -1.0}, "qty sign disagrees"),
        ({"execution_type": "stop"}, "invalid execution_type"),
        ({"execution_priority": "optional"}, "invalid execution_priority"),
    ],
)
def test_raw_rust_output_rejects_every_malformed_order_field(overrides, error):
    with pytest.raises(FatalBotException, match=error):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output([_raw_rust_order(**overrides)]),
            {0: SYMBOL},
            _raw_rust_input(),
        )


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

    with pytest.raises(FatalBotException, match="inconsistent with its order_type"):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output([order]),
            {0: SYMBOL},
            _raw_rust_input(),
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


def test_raw_rust_output_accepts_market_entry_when_input_allows_it():
    order = _raw_rust_order(execution_type="market")

    assert reconciler.validate_rust_orchestrator_output(
        _raw_rust_output([order]),
        {0: SYMBOL},
        _raw_rust_input(market_orders_allowed=True),
    ) == [order]


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
        _raw_rust_output([order]),
        {0: SYMBOL},
        _raw_rust_input(market_orders_allowed=False, **global_overrides),
    ) == [order]


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
            _raw_rust_output([order]),
            {0: SYMBOL},
            _raw_rust_input(market_orders_allowed=False, **global_overrides),
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
            _raw_rust_output([order]),
            {0: SYMBOL},
            _raw_rust_input(
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
            _raw_rust_output([order]),
            {0: SYMBOL},
            _raw_rust_input(market_orders_allowed=False, **global_overrides),
        )


@pytest.mark.parametrize(
    "orders",
    [
        [_raw_rust_order(), _raw_rust_order()],
        [_raw_rust_order(), _raw_rust_order(execution_type="market")],
        [_raw_rust_order(execution_type="market"), _raw_rust_order()],
    ],
    ids=[
        "exact-duplicate",
        "conflicting-execution-type",
        "conflicting-execution-type-reversed",
    ],
)
def test_raw_rust_output_rejects_colliding_conversion_identities(orders):
    with pytest.raises(FatalBotException, match="collide under conversion identity"):
        reconciler.validate_rust_orchestrator_output(
            _raw_rust_output(orders),
            {0: SYMBOL},
            _raw_rust_input(market_orders_allowed=True),
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


def test_raw_rust_output_requires_loss_gate_blocks_collection():
    out = _raw_rust_output()
    del out["diagnostics"]["loss_gate_blocks"]

    with pytest.raises(FatalBotException, match="missing required loss_gate_blocks"):
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
            out, {0: SYMBOL}, _raw_rust_input()
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
