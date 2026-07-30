from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from outcome.adapters import hyperliquid
from outcome.hyperliquid_live import (
    HyperliquidOutcomeAccountSnapshot,
    HyperliquidOutcomeFeeRates,
)
from outcome.live_planning import build_ema_anchor_outcome_live_plan
from outcome.models import (
    OutcomeCollateralBalance,
    OutcomeOpenOrder,
    OutcomeOrderSide,
    OutcomeSignalCandle1s,
    OutcomeSide,
    OutcomeTokenBalance,
)
from outcome.order_ownership import managed_outcome_client_order_id


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "outcome"


def market():
    return replace(
        hyperliquid.normalize_market(
            json.loads((FIXTURES / "hyperliquid_price_binary.json").read_text())
        ),
        qty_step=1.0,
        min_order_qty=1.0,
        min_order_notional=10.0,
    )


def account(
    received_time_ms: int,
    *,
    yes_qty: float = 0.0,
    no_qty: float = 0.0,
    yes_held_qty: float = 0.0,
    no_held_qty: float = 0.0,
    held_collateral: float = 0.0,
    available_after_maintenance: float = 100.0,
    open_orders: tuple[OutcomeOpenOrder, ...] = (),
) -> HyperliquidOutcomeAccountSnapshot:
    return HyperliquidOutcomeAccountSnapshot(
        received_time_ms=received_time_ms,
        collateral=OutcomeCollateralBalance(
            asset="USDC",
            total=100.0,
            held=held_collateral,
            available_after_maintenance=available_after_maintenance,
        ),
        fee_rates=HyperliquidOutcomeFeeRates(
            user_add_rate=0.0001,
            user_cross_rate=0.0003,
            user_spot_add_rate=0.0004,
            user_spot_cross_rate=0.0007,
        ),
        token_balances=(
            OutcomeTokenBalance(
                market_id="913",
                asset_id="+9130",
                outcome=OutcomeSide.YES,
                total_qty=yes_qty,
                held_qty=yes_held_qty,
                entry_notional=yes_qty * 0.4,
            ),
            OutcomeTokenBalance(
                market_id="913",
                asset_id="+9131",
                outcome=OutcomeSide.NO,
                total_qty=no_qty,
                held_qty=no_held_qty,
                entry_notional=no_qty * 0.4,
            ),
        ),
        open_orders=open_orders,
        recent_fills=(),
        unknown_outcome_balance_coins=(),
        unknown_outcome_order_coins=(),
        unknown_outcome_fill_coins=(),
    )


def params() -> dict:
    return {
        "ema_span_fast_seconds": 2.0,
        "ema_span_slow_seconds": 4.0,
        "ema_warmup_seconds": 3,
        "quote_offset": 0.01,
        "inventory_skew": 0.0,
        "clip_qty": 25.0,
        "max_total_inventory_qty": 100.0,
        "max_abs_residual_qty": 50.0,
        "min_locked_pair_edge": 0.005,
        "estimated_fee_per_share": 0.0,
        "risk_reduction_only_ms_before_close": 30_000,
        "entry_cutoff_ms_before_close": 5_000,
        "execution_mode": "accumulate_pairs",
    }


def candles() -> list[OutcomeSignalCandle1s]:
    return [
        OutcomeSignalCandle1s(
            timestamp_ms=1784872800000 + index * 1_000,
            open=close,
            high=close,
            low=close,
            close=close,
            volume=1.0 if index == 0 else 0.0,
            trade_count=1 if index == 0 else 0,
            carried_forward=index != 0,
        )
        for index, close in enumerate((0.50, 0.51, 0.52))
    ]


def test_live_plan_reconstructs_rust_state_from_dense_fill_candles():
    signal_candles = candles()
    now_ms = signal_candles[-1].timestamp_ms + 1_000
    plan = build_ema_anchor_outcome_live_plan(
        market(),
        params(),
        signal_candles,
        account(now_ms),
        now_ms=now_ms,
    )

    assert plan.strategy_kind == "ema_anchor_outcome"
    assert plan.observation_count == 3
    assert plan.configured_estimated_fee_per_share == 0.0
    assert plan.effective_estimated_fee_per_share == pytest.approx(0.0004)
    assert (
        plan.estimated_fee_source
        == "hyperliquid_user_fees_conservative_maker_floor"
    )
    assert len(plan.intents) == 2


def test_live_plan_retains_higher_configured_fee_estimate():
    signal_candles = candles()
    now_ms = signal_candles[-1].timestamp_ms + 1_000
    configured = params()
    configured["estimated_fee_per_share"] = 0.001
    plan = build_ema_anchor_outcome_live_plan(
        market(),
        configured,
        signal_candles,
        account(now_ms),
        now_ms=now_ms,
    )

    assert plan.configured_estimated_fee_per_share == pytest.approx(0.001)
    assert plan.effective_estimated_fee_per_share == pytest.approx(0.001)
    assert plan.estimated_fee_source == "configured"


def test_live_plan_restores_only_managed_buy_reserve_for_replacement_sizing():
    signal_candles = candles()
    now_ms = signal_candles[-1].timestamp_ms + 1_000
    managed_order = OutcomeOpenOrder(
        market_id="913",
        order_id="managed",
        asset_id="+9130",
        outcome=OutcomeSide.YES,
        side=OutcomeOrderSide.BUY,
        native_price=0.49,
        qty=50.0,
        original_qty=50.0,
        timestamp_ms=now_ms - 1_000,
        client_order_id=managed_outcome_client_order_id(
            "913",
            slot="canonical_bid",
            observation_end_ms=signal_candles[-1].timestamp_ms,
        ),
    )
    unmanaged_order = OutcomeOpenOrder(
        market_id="913",
        order_id="unmanaged",
        asset_id="+9131",
        outcome=OutcomeSide.NO,
        side=OutcomeOrderSide.BUY,
        native_price=0.49,
        qty=100.0,
        original_qty=100.0,
        timestamp_ms=now_ms - 1_000,
        client_order_id="0x" + "00" * 16,
    )
    plan = build_ema_anchor_outcome_live_plan(
        market(),
        params(),
        signal_candles,
        account(
            now_ms,
            held_collateral=99.0,
            available_after_maintenance=1.0,
            open_orders=(managed_order, unmanaged_order),
        ),
        now_ms=now_ms,
    )

    assert len(plan.intents) == 2
    assert all(intent.qty == pytest.approx(25.0) for intent in plan.intents)


def test_live_plan_rejects_stale_account_and_signal_inputs():
    signal_candles = candles()
    now_ms = signal_candles[-1].timestamp_ms + 10_000
    with pytest.raises(ValueError, match="snapshot is stale"):
        build_ema_anchor_outcome_live_plan(
            market(),
            params(),
            signal_candles,
            account(signal_candles[-1].timestamp_ms),
            now_ms=now_ms,
        )


def test_live_plan_sells_excess_yes_during_risk_reduction_window():
    outcome_market = market()
    signal_candles = candles()
    risk_timestamp_ms = outcome_market.lifecycle.scheduled_event_time_ms - 10_000
    shifted = [
        OutcomeSignalCandle1s(
            timestamp_ms=risk_timestamp_ms - (len(signal_candles) - index) * 1_000,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            trade_count=candle.trade_count,
            carried_forward=candle.carried_forward,
        )
        for index, candle in enumerate(signal_candles)
    ]
    now_ms = shifted[-1].timestamp_ms + 1_000
    plan = build_ema_anchor_outcome_live_plan(
        outcome_market,
        params(),
        shifted,
        account(now_ms, yes_qty=30.0, no_qty=5.0),
        now_ms=now_ms,
    )

    assert len(plan.intents) == 1
    intent = plan.intents[0]
    assert intent.slot == "canonical_ask"
    assert intent.outcome is OutcomeSide.YES
    assert intent.side is OutcomeOrderSide.SELL
    assert intent.qty == 25.0
    assert intent.close_all is True


def test_live_plan_marks_below_minimum_residual_exit_as_close_all():
    outcome_market = replace(
        market(),
        min_order_qty=5.0,
        min_order_notional=10.0,
    )
    signal_candles = candles()
    risk_timestamp_ms = outcome_market.lifecycle.scheduled_event_time_ms - 10_000
    shifted = [
        replace(
            candle,
            timestamp_ms=risk_timestamp_ms
            - (len(signal_candles) - index) * 1_000,
        )
        for index, candle in enumerate(signal_candles)
    ]
    now_ms = shifted[-1].timestamp_ms + 1_000

    plan = build_ema_anchor_outcome_live_plan(
        outcome_market,
        params(),
        shifted,
        account(now_ms, yes_qty=10.0, no_qty=9.0),
        now_ms=now_ms,
    )

    assert len(plan.intents) == 1
    intent = plan.intents[0]
    assert intent.side is OutcomeOrderSide.SELL
    assert intent.qty == 1.0
    assert intent.close_all is True


def test_live_plan_does_not_sell_inventory_reserved_by_unmanaged_orders():
    signal_candles = candles()
    now_ms = signal_candles[-1].timestamp_ms + 1_000
    configured = params()
    configured["execution_mode"] = "yes_only"
    unmanaged_sell = OutcomeOpenOrder(
        market_id="913",
        order_id="unmanaged-sell",
        asset_id="+9130",
        outcome=OutcomeSide.YES,
        side=OutcomeOrderSide.SELL,
        native_price=0.60,
        qty=30.0,
        original_qty=30.0,
        timestamp_ms=now_ms - 1_000,
        client_order_id="0x" + "00" * 16,
    )

    plan = build_ema_anchor_outcome_live_plan(
        market(),
        configured,
        signal_candles,
        account(
            now_ms,
            yes_qty=30.0,
            yes_held_qty=30.0,
            open_orders=(unmanaged_sell,),
        ),
        now_ms=now_ms,
    )

    assert all(intent.side is not OutcomeOrderSide.SELL for intent in plan.intents)


def test_live_plan_can_reclaim_inventory_reserved_by_managed_sell_orders():
    signal_candles = candles()
    now_ms = signal_candles[-1].timestamp_ms + 1_000
    configured = params()
    configured["execution_mode"] = "yes_only"
    managed_sell = OutcomeOpenOrder(
        market_id="913",
        order_id="managed-sell",
        asset_id="+9130",
        outcome=OutcomeSide.YES,
        side=OutcomeOrderSide.SELL,
        native_price=0.60,
        qty=30.0,
        original_qty=30.0,
        timestamp_ms=now_ms - 1_000,
        client_order_id=managed_outcome_client_order_id(
            "913",
            slot="canonical_ask",
            observation_end_ms=signal_candles[-1].timestamp_ms,
        ),
    )

    plan = build_ema_anchor_outcome_live_plan(
        market(),
        configured,
        signal_candles,
        account(
            now_ms,
            yes_qty=30.0,
            yes_held_qty=30.0,
            open_orders=(managed_sell,),
        ),
        now_ms=now_ms,
    )

    sells = [intent for intent in plan.intents if intent.side is OutcomeOrderSide.SELL]
    assert len(sells) == 1
    assert sells[0].qty == pytest.approx(25.0)
