from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3

import pytest

import outcome.live_bot as live_bot
from outcome.adapters import hyperliquid
from outcome.archive import OutcomeTradeArchive
from outcome.hyperliquid_live import (
    HyperliquidOutcomeAccountSnapshot,
    HyperliquidOutcomeFeeRates,
    HyperliquidOutcomeLifecycleSnapshot,
    HyperliquidOutcomeLifecycleState,
    HyperliquidOutcomeMutationResult,
)
from outcome.live_bot import (
    OutcomePlanningUnavailableReason,
    run_hip4_outcome_collected_cycle,
    run_hip4_outcome_cycle,
    run_hip4_outcome_unavailable_cycle,
)
from outcome.live_reconciliation import managed_outcome_client_order_id
from outcome.models import (
    OutcomeOpenOrder,
    OutcomeCollateralBalance,
    NormalizedOutcomeTrade,
    OutcomeOrderSide,
    OutcomeSettlementEvidence,
    OutcomeSignalCandle1s,
    OutcomeSide,
    OutcomeTokenBalance,
    OutcomeVenue,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "outcome"


def market():
    return hyperliquid.normalize_market(
        json.loads((FIXTURES / "hyperliquid_price_binary.json").read_text())
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


def candles():
    start = market().lifecycle.trading_open_time_ms
    assert start is not None
    return [
        OutcomeSignalCandle1s(
            timestamp_ms=start + index * 1_000,
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


def snapshot(received_time_ms, *, open_orders=()):
    return HyperliquidOutcomeAccountSnapshot(
        received_time_ms=received_time_ms,
        collateral=OutcomeCollateralBalance(asset="USDC", total=100.0, held=0.0),
        fee_rates=HyperliquidOutcomeFeeRates(
            user_add_rate=0.0001,
            user_cross_rate=0.0003,
            user_spot_add_rate=0.0004,
            user_spot_cross_rate=0.0007,
        ),
        token_balances=(
            OutcomeTokenBalance("913", "+9130", OutcomeSide.YES, 0.0, 0.0, 0.0),
            OutcomeTokenBalance("913", "+9131", OutcomeSide.NO, 0.0, 0.0, 0.0),
        ),
        open_orders=tuple(open_orders),
        recent_fills=(),
        unknown_outcome_balance_coins=(),
        unknown_outcome_order_coins=(),
        unknown_outcome_fill_coins=(),
    )


def lifecycle_snapshot(
    observed_at_ms,
    *,
    state=HyperliquidOutcomeLifecycleState.ACTIVE,
):
    settlement = None
    if state is HyperliquidOutcomeLifecycleState.SETTLED:
        settlement = OutcomeSettlementEvidence(
            venue=OutcomeVenue.HYPERLIQUID,
            market_id="913",
            yes_fraction=1.0,
            payout_unit=1.0,
            settlement_time_ms=market().lifecycle.scheduled_event_time_ms + 1_000,
            capital_release_time_ms=market().lifecycle.scheduled_event_time_ms + 1_000,
            received_time_ms=observed_at_ms,
            source_event_id="0xsettlement",
            evidence_source="hyperliquid_user_fills",
            observed_yes_qty=25.0,
            observed_no_qty=0.0,
            collateral_payout=25.0,
            fee=0.0,
            fee_asset="USDC",
        )
    return HyperliquidOutcomeLifecycleSnapshot(
        market_id="913",
        state=state,
        observed_at_ms=observed_at_ms,
        settlement=settlement,
    )


class ReadOnlyClient:
    def __init__(self, account, *, lifecycle=None):
        self.account = account
        self.lifecycle = lifecycle
        self.market_checks = 0
        self.snapshot_fetches = 0

    async def fetch_market_lifecycle(self, market, *, account=None, now_ms=None):
        self.market_checks += 1
        return self.lifecycle or lifecycle_snapshot(
            self.account.received_time_ms if now_ms is None else now_ms
        )

    async def fetch_account_snapshot(self, markets):
        self.snapshot_fetches += 1
        return self.account


@pytest.mark.asyncio
async def test_default_cycle_is_read_only_and_returns_explicit_reconciliation():
    signal_candles = candles()
    now_ms = signal_candles[-1].timestamp_ms + 1_000
    client = ReadOnlyClient(snapshot(now_ms))

    cycle = await run_hip4_outcome_cycle(
        client,
        market(),
        params(),
        signal_candles,
        now_ms=now_ms,
    )

    assert cycle.is_dry_run is True
    assert cycle.planning_available is True
    assert cycle.planning_unavailable_reason is None
    assert cycle.market_id == "913"
    assert len(cycle.plan.intents) == 2
    assert len(cycle.reconciliation.creates) == 2
    assert cycle.reconciliation.cancels == ()
    assert client.market_checks == 1
    assert client.snapshot_fetches == 1


@pytest.mark.asyncio
async def test_default_cycle_timestamps_decision_after_required_reads(monkeypatch):
    signal_candles = candles()
    completed_at_ms = signal_candles[-1].timestamp_ms + 1_000
    clock = {"now_ms": completed_at_ms - 2_000}

    class AdvancingClient(ReadOnlyClient):
        async def fetch_account_snapshot(self, markets):
            clock["now_ms"] = completed_at_ms
            self.account = snapshot(completed_at_ms)
            return await super().fetch_account_snapshot(markets)

        async def fetch_market_lifecycle(self, market, *, account=None, now_ms=None):
            assert now_ms is None
            return lifecycle_snapshot(clock["now_ms"])

    monkeypatch.setattr(
        live_bot.time,
        "time",
        lambda: clock["now_ms"] / 1_000,
    )
    cycle = await run_hip4_outcome_cycle(
        AdvancingClient(snapshot(clock["now_ms"])),
        market(),
        params(),
        signal_candles,
    )

    assert cycle.planned_at_ms == completed_at_ms
    assert cycle.account.received_time_ms == completed_at_ms
    assert cycle.planning_available is True


@pytest.mark.asyncio
async def test_unavailable_signal_cycle_is_cancel_only_and_observable():
    now_ms = candles()[-1].timestamp_ms + 1_000
    client = ReadOnlyClient(snapshot(now_ms))

    cycle = await run_hip4_outcome_unavailable_cycle(
        client,
        market(),
        reason=OutcomePlanningUnavailableReason.NO_PUBLIC_FILL,
        now_ms=now_ms,
    )

    assert cycle.is_dry_run is True
    assert cycle.planning_available is False
    assert cycle.plan is None
    assert cycle.planning_unavailable_reason is OutcomePlanningUnavailableReason.NO_PUBLIC_FILL
    assert cycle.reconciliation.creates == ()
    assert cycle.reconciliation.cancels == ()
    assert client.market_checks == 1
    assert client.snapshot_fetches == 1


@pytest.mark.asyncio
async def test_stale_signal_routes_normal_cycle_to_cancel_only_safety():
    signal_candles = candles()
    now_ms = signal_candles[-1].timestamp_ms + 10_000
    client = ReadOnlyClient(snapshot(now_ms))

    cycle = await run_hip4_outcome_cycle(
        client,
        market(),
        params(),
        signal_candles,
        now_ms=now_ms,
    )

    assert cycle.plan is None
    assert (
        cycle.planning_unavailable_reason
        is OutcomePlanningUnavailableReason.STALE_VERIFIED_SIGNAL
    )
    assert cycle.reconciliation.creates == ()
    assert client.market_checks == 1
    assert client.snapshot_fetches == 1


@pytest.mark.asyncio
async def test_collected_cycle_routes_public_silence_to_cancel_only_safety():
    async def silent_stream():
        while True:
            await asyncio.sleep(60)
            if False:  # pragma: no cover - keeps this an async generator
                yield None

    now_ms = candles()[-1].timestamp_ms + 1_000
    client = ReadOnlyClient(snapshot(now_ms))
    collected = await run_hip4_outcome_collected_cycle(
        client,
        market(),
        params(),
        min_observations=3,
        max_wait_seconds=0.01,
        trade_stream=silent_stream(),
        now_ms=now_ms,
    )

    assert collected.signal_window is None
    assert collected.cycle.planning_available is False
    assert (
        collected.cycle.planning_unavailable_reason
        is OutcomePlanningUnavailableReason.NO_PUBLIC_FILL
    )
    assert collected.cycle.reconciliation.creates == ()


@pytest.mark.asyncio
async def test_collected_cycle_uses_verified_fill_seed_for_live_rust_plan():
    outcome_market = market()
    start_ms = outcome_market.lifecycle.trading_open_time_ms
    assert start_ms is not None
    received_time_ms = start_ms + 950

    async def seeded_stream():
        yield NormalizedOutcomeTrade(
            venue=outcome_market.venue,
            market_id=outcome_market.market_id,
            asset_id=outcome_market.yes_asset.asset_id,
            outcome=OutcomeSide.YES,
            native_side=OutcomeOrderSide.BUY,
            native_price=0.5,
            canonical_yes_price=0.5,
            qty=1.0,
            exchange_time_ms=start_ms + 900,
            received_time_ms=received_time_ms,
            source_event_id="seed",
            collector_sequence=1,
        )

    now_ms = start_ms + 4_100
    client = ReadOnlyClient(snapshot(now_ms))
    collected = await run_hip4_outcome_collected_cycle(
        client,
        outcome_market,
        params(),
        min_observations=3,
        delivery_lag_ms=0,
        wall_clock_ms=lambda: now_ms,
        trade_stream=seeded_stream(),
        now_ms=now_ms,
    )

    assert collected.signal_window is not None
    assert len(collected.signal_window.candles) == 3
    assert collected.cycle.planning_available is True
    assert collected.cycle.plan is not None
    assert len(collected.cycle.plan.intents) == 2


@pytest.mark.asyncio
async def test_collected_cycle_routes_partial_verified_window_to_cancel_only_safety():
    outcome_market = market()
    start_ms = outcome_market.lifecycle.trading_open_time_ms
    assert start_ms is not None

    async def partial_stream():
        yield NormalizedOutcomeTrade(
            venue=outcome_market.venue,
            market_id=outcome_market.market_id,
            asset_id=outcome_market.yes_asset.asset_id,
            outcome=OutcomeSide.YES,
            native_side=OutcomeOrderSide.BUY,
            native_price=0.5,
            canonical_yes_price=0.5,
            qty=1.0,
            exchange_time_ms=start_ms + 900,
            received_time_ms=start_ms + 950,
            source_event_id="partial-seed",
            collector_sequence=1,
        )
        await asyncio.sleep(60)

    now_ms = start_ms + 1_500
    client = ReadOnlyClient(snapshot(now_ms))
    collected = await run_hip4_outcome_collected_cycle(
        client,
        outcome_market,
        params(),
        min_observations=3,
        max_wait_seconds=0.01,
        delivery_lag_ms=0,
        wall_clock_ms=lambda: now_ms,
        trade_stream=partial_stream(),
        now_ms=now_ms,
    )

    assert collected.signal_window is None
    assert collected.cycle.plan is None
    assert (
        collected.cycle.planning_unavailable_reason
        is OutcomePlanningUnavailableReason.INCOMPLETE_VERIFIED_SIGNAL
    )
    assert collected.cycle.reconciliation.creates == ()


@pytest.mark.asyncio
async def test_expired_market_routes_normal_cycle_to_cancel_only_safety():
    signal_candles = candles()
    now_ms = market().lifecycle.scheduled_event_time_ms + 1_000
    client = ReadOnlyClient(
        snapshot(now_ms),
        lifecycle=lifecycle_snapshot(
            now_ms,
            state=HyperliquidOutcomeLifecycleState.EXPIRED_AWAITING_SETTLEMENT,
        ),
    )

    cycle = await run_hip4_outcome_cycle(
        client,
        market(),
        params(),
        signal_candles,
        now_ms=now_ms,
    )

    assert cycle.plan is None
    assert cycle.lifecycle.state is HyperliquidOutcomeLifecycleState.EXPIRED_AWAITING_SETTLEMENT
    assert (
        cycle.planning_unavailable_reason
        is OutcomePlanningUnavailableReason.MARKET_EXPIRED_AWAITING_SETTLEMENT
    )
    assert cycle.reconciliation.creates == ()


@pytest.mark.asyncio
async def test_settled_market_routes_normal_cycle_to_cancel_only_with_evidence():
    signal_candles = candles()
    now_ms = market().lifecycle.scheduled_event_time_ms + 2_000
    client = ReadOnlyClient(
        snapshot(now_ms),
        lifecycle=lifecycle_snapshot(
            now_ms,
            state=HyperliquidOutcomeLifecycleState.SETTLED,
        ),
    )

    cycle = await run_hip4_outcome_cycle(
        client,
        market(),
        params(),
        signal_candles,
        now_ms=now_ms,
    )

    assert cycle.plan is None
    assert cycle.lifecycle.state is HyperliquidOutcomeLifecycleState.SETTLED
    assert cycle.lifecycle.settlement is not None
    assert cycle.lifecycle.settlement.yes_fraction == 1.0
    assert (
        cycle.planning_unavailable_reason
        is OutcomePlanningUnavailableReason.MARKET_SETTLED
    )
    assert cycle.reconciliation.creates == ()


@pytest.mark.asyncio
async def test_settled_cycle_persists_authoritative_evidence(tmp_path):
    signal_candles = candles()
    now_ms = market().lifecycle.scheduled_event_time_ms + 2_000
    archive = OutcomeTradeArchive(tmp_path / "live-settlement.sqlite")
    client = ReadOnlyClient(
        snapshot(now_ms),
        lifecycle=lifecycle_snapshot(
            now_ms,
            state=HyperliquidOutcomeLifecycleState.SETTLED,
        ),
    )

    cycle = await run_hip4_outcome_cycle(
        client,
        market(),
        params(),
        signal_candles,
        now_ms=now_ms,
        archive=archive,
        collector_session="live-settlement",
    )

    assert cycle.lifecycle.settlement is not None
    assert archive.load_settlements(OutcomeVenue.HYPERLIQUID, "913") == [
        cycle.lifecycle.settlement
    ]


class SafetyMutationClient:
    def __init__(self, snapshots, *, lifecycle=None):
        self.snapshots = list(snapshots)
        self.lifecycle = lifecycle
        self.cancelled = []
        self.created = []

    async def fetch_market_lifecycle(self, market, *, account=None, now_ms=None):
        observed_at_ms = (
            account.received_time_ms
            if now_ms is None and account is not None
            else now_ms
        )
        return self.lifecycle or lifecycle_snapshot(observed_at_ms)

    async def fetch_account_snapshot(self, markets):
        return self.snapshots.pop(0)

    async def cancel_order(
        self,
        market,
        *,
        outcome,
        order_id,
        expected_client_order_id,
    ):
        self.cancelled.append(
            (market.market_id, outcome, order_id, expected_client_order_id)
        )
        return HyperliquidOutcomeMutationResult(
            kind="cancelled",
            order_id=None,
            filled_qty=0.0,
            average_price=None,
            raw_response={},
        )

    async def submit_limit_order(self, *args, **kwargs):
        self.created.append((args, kwargs))
        raise AssertionError("cancel-only safety cycle must never create an order")


@pytest.mark.asyncio
async def test_executed_unavailable_signal_cycle_cancels_managed_quote_only():
    now_ms = candles()[-1].timestamp_ms + 1_000
    cloid = managed_outcome_client_order_id(
        "913",
        slot="canonical_bid",
        observation_end_ms=candles()[-1].timestamp_ms,
    )
    managed_order = OutcomeOpenOrder(
        market_id="913",
        order_id="7",
        asset_id="+9130",
        outcome=OutcomeSide.YES,
        side=OutcomeOrderSide.BUY,
        native_price=0.49,
        qty=25.0,
        original_qty=25.0,
        timestamp_ms=candles()[-1].timestamp_ms,
        client_order_id=cloid,
    )
    initial = snapshot(now_ms, open_orders=(managed_order,))
    after_cancel = snapshot(now_ms)
    final = snapshot(now_ms)
    client = SafetyMutationClient((initial, after_cancel, final))

    cycle = await run_hip4_outcome_unavailable_cycle(
        client,
        market(),
        reason=OutcomePlanningUnavailableReason.NO_PUBLIC_FILL,
        execute=True,
        now_ms=now_ms,
    )

    assert cycle.is_dry_run is False
    assert client.cancelled == [("913", OutcomeSide.YES, 7, cloid)]
    assert client.created == []
    assert cycle.mutation_result is not None
    assert len(cycle.mutation_result.cancelled) == 1
    assert cycle.mutation_result.created == ()


@pytest.mark.asyncio
async def test_stream_disconnect_routes_to_executed_cancel_only_cycle():
    outcome_market = market()
    start_ms = outcome_market.lifecycle.trading_open_time_ms
    assert start_ms is not None

    async def disconnected_stream():
        yield NormalizedOutcomeTrade(
            venue=outcome_market.venue,
            market_id=outcome_market.market_id,
            asset_id=outcome_market.yes_asset.asset_id,
            outcome=OutcomeSide.YES,
            native_side=OutcomeOrderSide.BUY,
            native_price=0.5,
            canonical_yes_price=0.5,
            qty=1.0,
            exchange_time_ms=start_ms + 900,
            received_time_ms=start_ms + 950,
            source_event_id="disconnect-seed",
            collector_sequence=1,
        )

    now_ms = start_ms + 1_500
    cloid = managed_outcome_client_order_id(
        "913",
        slot="canonical_bid",
        observation_end_ms=candles()[-1].timestamp_ms,
    )
    managed_order = OutcomeOpenOrder(
        market_id="913",
        order_id="7",
        asset_id="+9130",
        outcome=OutcomeSide.YES,
        side=OutcomeOrderSide.BUY,
        native_price=0.49,
        qty=25.0,
        original_qty=25.0,
        timestamp_ms=start_ms,
        client_order_id=cloid,
    )
    client = SafetyMutationClient(
        (
            snapshot(now_ms, open_orders=(managed_order,)),
            snapshot(now_ms),
            snapshot(now_ms),
        )
    )

    collected = await run_hip4_outcome_collected_cycle(
        client,
        outcome_market,
        params(),
        min_observations=3,
        max_wait_seconds=1.0,
        delivery_lag_ms=0,
        wall_clock_ms=lambda: now_ms,
        trade_stream=disconnected_stream(),
        execute=True,
        now_ms=now_ms,
    )

    assert collected.signal_window is None
    assert (
        collected.cycle.planning_unavailable_reason
        is OutcomePlanningUnavailableReason.SIGNAL_COLLECTION_FAILED
    )
    assert client.cancelled == [("913", OutcomeSide.YES, 7, cloid)]
    assert client.created == []


@pytest.mark.asyncio
async def test_malformed_public_signal_routes_to_executed_cancel_only_cycle():
    outcome_market = market()
    start_ms = outcome_market.lifecycle.trading_open_time_ms
    assert start_ms is not None

    async def malformed_stream():
        if False:  # pragma: no cover - keeps this an async generator
            yield None
        raise json.JSONDecodeError("malformed websocket payload", "{", 0)

    now_ms = start_ms + 1_500
    cloid = managed_outcome_client_order_id(
        "913",
        slot="canonical_bid",
        observation_end_ms=candles()[-1].timestamp_ms,
    )
    managed_order = OutcomeOpenOrder(
        market_id="913",
        order_id="7",
        asset_id="+9130",
        outcome=OutcomeSide.YES,
        side=OutcomeOrderSide.BUY,
        native_price=0.49,
        qty=25.0,
        original_qty=25.0,
        timestamp_ms=start_ms,
        client_order_id=cloid,
    )
    client = SafetyMutationClient(
        (
            snapshot(now_ms, open_orders=(managed_order,)),
            snapshot(now_ms),
            snapshot(now_ms),
        )
    )

    collected = await run_hip4_outcome_collected_cycle(
        client,
        outcome_market,
        params(),
        min_observations=3,
        max_wait_seconds=1.0,
        delivery_lag_ms=0,
        wall_clock_ms=lambda: now_ms,
        trade_stream=malformed_stream(),
        execute=True,
        now_ms=now_ms,
    )

    assert collected.signal_window is None
    assert (
        collected.cycle.planning_unavailable_reason
        is OutcomePlanningUnavailableReason.SIGNAL_COLLECTION_FAILED
    )
    assert client.cancelled == [("913", OutcomeSide.YES, 7, cloid)]
    assert client.created == []


@pytest.mark.asyncio
async def test_signal_collection_configuration_errors_are_not_downgraded():
    with pytest.raises(ValueError, match="min_observations must be positive"):
        await run_hip4_outcome_collected_cycle(
            ReadOnlyClient(snapshot(candles()[-1].timestamp_ms + 1_000)),
            market(),
            params(),
            min_observations=0,
        )


@pytest.mark.asyncio
async def test_archive_failure_routes_to_cancel_only_cycle():
    class FailingArchive:
        def append_market_metadata(self, *args, **kwargs):
            raise sqlite3.OperationalError("archive unavailable")

    now_ms = candles()[-1].timestamp_ms + 1_000
    client = ReadOnlyClient(snapshot(now_ms))
    collected = await run_hip4_outcome_collected_cycle(
        client,
        market(),
        params(),
        min_observations=3,
        archive=FailingArchive(),
        collector_session="failing-archive",
        now_ms=now_ms,
    )

    assert collected.signal_window is None
    assert (
        collected.cycle.planning_unavailable_reason
        is OutcomePlanningUnavailableReason.SIGNAL_COLLECTION_FAILED
    )
    assert collected.cycle.reconciliation.creates == ()
