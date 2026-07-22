from __future__ import annotations

import math

import pytest

from live.executor import _apply_order_churn_final_admission
from live.order_churn_gate import OrderChurnGateState


def _order(
    name: str,
    *,
    price: float,
    side: str = "buy",
    execution_type: str = "limit",
    priority: str = "ordinary",
    churn: bool = True,
) -> dict:
    return {
        "name": name,
        "symbol": "BTC/USDT:USDT",
        "position_side": "long",
        "side": side,
        "reduce_only": False,
        "type": execution_type,
        "pb_order_type": "entry_ema_anchor_long",
        "execution_priority": priority,
        "qty": 1.0,
        "price": price,
        "_churn_evidence": churn,
    }


class _Bot:
    def __init__(self):
        self._order_churn_gate_state = OrderChurnGateState()
        self.values = {
            "order_replacement_churn_gate_activation_count": 10,
            "order_replacement_churn_gate_window_minutes": 10.0,
            "order_replacement_churn_gate_market_dist_pct": 0.005,
            "max_n_creations_per_batch": 20,
        }
        self.prices = {"BTC/USDT:USDT": 100.0}
        self.action_headroom: float | int | None = math.inf

    def live_value(self, key: str):
        return self.values[key]

    def _is_market_execution_order(self, order: dict) -> bool:
        return order.get("type") == "market"

    async def _fetch_fresh_order_churn_market_prices(
        self, symbols: set[str], *, max_age_ms: int = 10_000
    ):
        return {symbol: self.prices.get(symbol) for symbol in symbols}

    async def _order_churn_far_create_headroom(self):
        return self.action_headroom


@pytest.mark.asyncio
async def test_allowance_blocks_only_far_churn_evidenced_ordinary_orders():
    bot = _Bot()
    bot._order_churn_gate_state.record_create_attempts(10, now_monotonic=0.0)
    # Keep the attempts in whatever monotonic window the implementation observes.
    import time

    bot._order_churn_gate_state.create_attempt_timestamps.clear()
    bot._order_churn_gate_state.record_create_attempts(
        10, now_monotonic=time.monotonic()
    )
    far = _order("far", price=99.0)
    near = _order("near", price=99.8)
    market = _order("market", price=90.0, execution_type="market")
    critical = _order("critical", price=90.0, priority="risk_critical")
    stable = _order("stable", price=90.0, churn=False)

    admitted = await _apply_order_churn_final_admission(
        bot, [far, near, market, stable, critical]
    )

    assert [order["name"] for order in admitted] == [
        "critical",
        "near",
        "market",
        "stable",
    ]
    assert far["_churn_gate_reason"] == "allowance_exhausted"


@pytest.mark.asyncio
async def test_unsupported_generic_connector_bypasses_churn_admission_policy():
    bot = _Bot()
    bot._order_churn_gate_enabled_for_connector = False

    def fail_live_value(_key):
        raise AssertionError("generic fallback must not read churn configuration")

    bot.live_value = fail_live_value
    orders = [_order("far", price=90.0), _order("near", price=99.8)]

    admitted = await _apply_order_churn_final_admission(bot, orders)

    assert admitted == orders


@pytest.mark.asyncio
async def test_exempt_orders_count_against_later_far_candidate_in_same_batch():
    bot = _Bot()
    bot.values["order_replacement_churn_gate_activation_count"] = 1
    near = _order("near", price=99.8)
    far = _order("far", price=99.0)

    admitted = await _apply_order_churn_final_admission(bot, [near, far])

    assert [order["name"] for order in admitted] == ["near"]
    assert far["_churn_gate_reason"] == "allowance_exhausted"


@pytest.mark.asyncio
async def test_missing_final_market_data_defers_only_churn_evidenced_candidate():
    bot = _Bot()
    bot.prices = {}
    churn = _order("churn", price=99.0)
    stable = _order("stable", price=99.0, churn=False)

    admitted = await _apply_order_churn_final_admission(bot, [churn, stable])

    assert [order["name"] for order in admitted] == ["stable"]
    assert churn["_churn_gate_reason"] == "market_price_unavailable"


@pytest.mark.asyncio
async def test_hyperliquid_style_headroom_only_blocks_far_churn_candidate():
    bot = _Bot()
    bot.action_headroom = 0
    far = _order("far", price=99.0)
    near = _order("near", price=99.8)
    critical = _order("critical", price=90.0, priority="risk_critical")

    admitted = await _apply_order_churn_final_admission(bot, [far, near, critical])

    assert [order["name"] for order in admitted] == ["critical", "near"]
    assert far["_churn_gate_reason"] == "action_headroom_exhausted"


@pytest.mark.asyncio
async def test_far_candidate_cannot_consume_headroom_needed_by_later_exempt_order():
    bot = _Bot()
    bot.action_headroom = 1
    far = _order("far", price=99.0)
    near = _order("near", price=99.8)

    admitted = await _apply_order_churn_final_admission(bot, [far, near])

    assert [order["name"] for order in admitted] == ["near"]
    assert far["_churn_gate_reason"] == "action_headroom_exhausted"
    assert near["_churn_gate_reason"] == "market_distance_exempt"


@pytest.mark.asyncio
async def test_far_candidate_reserves_required_config_action_with_create():
    bot = _Bot()
    bot.action_headroom = 2
    far = _order("far", price=99.0)

    admitted = await _apply_order_churn_final_admission(
        bot,
        [far],
        config_action_costs_by_symbol={far["symbol"]: 1},
    )

    assert admitted == []
    assert far["_churn_gate_reason"] == "action_headroom_exhausted"

    bot.action_headroom = 3
    admitted = await _apply_order_churn_final_admission(
        bot,
        [far],
        config_action_costs_by_symbol={far["symbol"]: 1},
    )
    assert admitted == [far]


@pytest.mark.asyncio
async def test_disabled_gate_preserves_priority_and_batch_cap_without_distance_fetch():
    bot = _Bot()
    bot.values["order_replacement_churn_gate_activation_count"] = 0
    bot.values["max_n_creations_per_batch"] = 2
    ordinary = _order("ordinary", price=90.0)
    critical = _order("critical", price=90.0, priority="risk_critical")
    another = _order("another", price=90.0)

    admitted = await _apply_order_churn_final_admission(
        bot, [ordinary, another, critical]
    )

    assert [order["name"] for order in admitted] == ["critical", "ordinary"]


@pytest.mark.asyncio
async def test_diagnostic_emitter_failure_cannot_change_admission():
    bot = _Bot()

    def fail_event(**_kwargs):
        raise RuntimeError("diagnostic sink failed")

    bot._emit_order_churn_admission_event = fail_event
    stable = _order("stable", price=90.0, churn=False)

    admitted = await _apply_order_churn_final_admission(bot, [stable])

    assert admitted == [stable]
