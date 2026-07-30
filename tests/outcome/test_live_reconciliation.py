from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from outcome.adapters import hyperliquid
from outcome.hyperliquid_live import (
    HyperliquidOutcomeAccountSnapshot,
    HyperliquidOutcomeFeeRates,
    HyperliquidOutcomeMutationResult,
    OutcomeCreateDeadlineExpired,
)
from outcome.live_planning import OutcomeLiveOrderIntent, OutcomeLivePlan
from outcome.live_reconciliation import (
    OutcomeOrderCancel,
    OutcomeOrderMutationSkippedReason,
    execute_hip4_order_reconciliation,
    is_managed_outcome_client_order_id,
    managed_outcome_client_order_id,
    reconcile_outcome_orders,
    reconcile_outcome_orders_to_empty,
)
from outcome.models import (
    OutcomeCollateralBalance,
    OutcomeOpenOrder,
    OutcomeOrderSide,
    OutcomeSide,
    OutcomeTokenBalance,
)


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


def plan() -> OutcomeLivePlan:
    return OutcomeLivePlan(
        strategy_kind="ema_anchor_outcome",
        market_id="913",
        observation_start_ms=1_000,
        observation_end_ms=3_000,
        observation_count=3,
        ema_fast=0.5,
        ema_slow=0.5,
        inventory_shift=0.0,
        configured_estimated_fee_per_share=0.0,
        effective_estimated_fee_per_share=0.0004,
        estimated_fee_source="hyperliquid_user_fees_conservative_maker_floor",
        intents=(
            OutcomeLiveOrderIntent(
                slot="canonical_bid",
                outcome=OutcomeSide.YES,
                side=OutcomeOrderSide.BUY,
                native_price=0.49,
                canonical_yes_price=0.49,
                qty=25.0,
            ),
            OutcomeLiveOrderIntent(
                slot="canonical_ask",
                outcome=OutcomeSide.NO,
                side=OutcomeOrderSide.BUY,
                native_price=0.49,
                canonical_yes_price=0.51,
                qty=25.0,
            ),
        ),
    )


def snapshot(open_orders=()) -> HyperliquidOutcomeAccountSnapshot:
    return HyperliquidOutcomeAccountSnapshot(
        received_time_ms=4_000,
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


def order(
    order_id: str,
    *,
    outcome: OutcomeSide,
    price: float,
    cloid: str | None,
    qty: float = 25.0,
) -> OutcomeOpenOrder:
    return OutcomeOpenOrder(
        market_id="913",
        order_id=order_id,
        asset_id="+9130" if outcome is OutcomeSide.YES else "+9131",
        outcome=outcome,
        side=OutcomeOrderSide.BUY,
        native_price=price,
        qty=qty,
        original_qty=25.0,
        timestamp_ms=3_000,
        client_order_id=cloid,
    )


def test_reconciliation_keeps_exact_managed_order_and_never_cancels_unmanaged():
    desired = plan()
    bid_cloid = managed_outcome_client_order_id(
        "913",
        slot="canonical_bid",
        observation_end_ms=2_000,
    )
    stale_cloid = managed_outcome_client_order_id(
        "913",
        slot="canonical_ask",
        observation_end_ms=2_000,
    )
    account = snapshot(
        (
            order("1", outcome=OutcomeSide.YES, price=0.49, cloid=bid_cloid),
            order("2", outcome=OutcomeSide.NO, price=0.48, cloid=stale_cloid),
            order("3", outcome=OutcomeSide.YES, price=0.47, cloid=None),
        )
    )

    reconciliation = reconcile_outcome_orders(market(), desired, account)

    assert reconciliation.kept_order_ids == ("1",)
    assert [item.order_id for item in reconciliation.cancels] == ["2"]
    assert len(reconciliation.creates) == 1
    assert reconciliation.creates[0].intent.slot == "canonical_ask"
    assert reconciliation.unmanaged_order_ids == ("3",)
    assert is_managed_outcome_client_order_id(bid_cloid, "913") is True
    assert is_managed_outcome_client_order_id(bid_cloid, "914") is False


def test_unavailable_signal_cancels_only_managed_orders_without_creates():
    managed_cloid = managed_outcome_client_order_id(
        "913",
        slot="canonical_bid",
        observation_end_ms=2_000,
    )
    account = snapshot(
        (
            order("1", outcome=OutcomeSide.YES, price=0.49, cloid=managed_cloid),
            order("2", outcome=OutcomeSide.YES, price=0.48, cloid=None),
        )
    )

    reconciliation = reconcile_outcome_orders_to_empty(
        market(),
        account,
        decision_time_ms=4_000,
    )

    assert reconciliation.creates == ()
    assert [item.order_id for item in reconciliation.cancels] == ["1"]
    assert reconciliation.kept_order_ids == ()
    assert reconciliation.unmanaged_order_ids == ("2",)


def test_unavailable_signal_reconciliation_requires_second_aligned_decision_time():
    with pytest.raises(ValueError, match="second-aligned"):
        reconcile_outcome_orders_to_empty(
            market(),
            snapshot(),
            decision_time_ms=4_001,
        )


class FakeClient:
    def __init__(self, *snapshots):
        self.snapshots = list(snapshots)
        self.cancelled = []
        self.created = []

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

    async def submit_limit_order(
        self,
        market,
        *,
        outcome,
        side,
        native_price,
        qty,
        client_order_id,
        post_only,
        create_deadline_ms=None,
        wall_clock_ms=None,
    ):
        self.created.append(
            (market.market_id, outcome, side, native_price, qty, client_order_id, post_only)
        )
        return HyperliquidOutcomeMutationResult(
            kind="resting",
            order_id="9",
            filled_qty=0.0,
            average_price=None,
            raw_response={},
        )

    async def fetch_account_snapshot(self, markets):
        return self.snapshots.pop(0)


@pytest.mark.asyncio
async def test_executor_verifies_cancel_before_create_and_final_resting_state():
    desired = replace(plan(), intents=(plan().intents[0],))
    stale_cloid = managed_outcome_client_order_id(
        "913",
        slot="canonical_bid",
        observation_end_ms=2_000,
    )
    current_cloid = managed_outcome_client_order_id(
        "913",
        slot="canonical_bid",
        observation_end_ms=3_000,
    )
    initial = snapshot((order("2", outcome=OutcomeSide.YES, price=0.48, cloid=stale_cloid),))
    after_cancels = snapshot(())
    final = snapshot((order("9", outcome=OutcomeSide.YES, price=0.49, cloid=current_cloid),))
    reconciliation = reconcile_outcome_orders(market(), desired, initial)
    client = FakeClient(after_cancels, final)

    result = await execute_hip4_order_reconciliation(client, market(), reconciliation)

    assert client.cancelled == [("913", OutcomeSide.YES, 2, stale_cloid)]
    assert len(client.created) == 1
    assert result.final_snapshot.open_orders[0].client_order_id == current_cloid


@pytest.mark.asyncio
async def test_executor_cancels_to_empty_if_signal_expires_after_cancellation():
    desired = replace(plan(), intents=(plan().intents[0],))
    stale_cloid = managed_outcome_client_order_id(
        "913",
        slot="canonical_bid",
        observation_end_ms=2_000,
    )
    initial = snapshot(
        (order("2", outcome=OutcomeSide.YES, price=0.48, cloid=stale_cloid),)
    )
    reconciliation = reconcile_outcome_orders(market(), desired, initial)
    client = FakeClient(snapshot(), snapshot(), snapshot())
    clock_values = iter((3_000, 8_001))

    result = await execute_hip4_order_reconciliation(
        client,
        market(),
        reconciliation,
        create_deadline_ms=8_000,
        wall_clock_ms=lambda: next(clock_values),
    )

    assert result.create_skipped_reason is (
        OutcomeOrderMutationSkippedReason.STALE_VERIFIED_SIGNAL
    )
    assert result.create_skipped_at_ms == 8_001
    assert result.final_snapshot.open_orders == ()
    assert client.cancelled == [("913", OutcomeSide.YES, 2, stale_cloid)]
    assert client.created == []


@pytest.mark.asyncio
async def test_executor_cancels_to_empty_if_create_preflight_crosses_deadline():
    desired = replace(plan(), intents=(plan().intents[0],))
    reconciliation = reconcile_outcome_orders(market(), desired, snapshot())

    class DeadlineClient(FakeClient):
        async def submit_limit_order(self, *args, **kwargs):
            raise OutcomeCreateDeadlineExpired(
                observed_at_ms=8_001,
                deadline_ms=8_000,
            )

    client = DeadlineClient(snapshot(), snapshot())
    result = await execute_hip4_order_reconciliation(
        client,
        market(),
        reconciliation,
        create_deadline_ms=8_000,
        wall_clock_ms=lambda: 4_000,
    )

    assert result.create_skipped_reason is (
        OutcomeOrderMutationSkippedReason.STALE_VERIFIED_SIGNAL
    )
    assert result.create_skipped_at_ms == 8_001
    assert result.final_snapshot.open_orders == ()
    assert result.created == ()


@pytest.mark.asyncio
async def test_executor_cancels_all_attempted_orders_after_partial_create_failure():
    reconciliation = reconcile_outcome_orders(market(), plan(), snapshot())
    first, second = reconciliation.creates
    cleanup_snapshot = snapshot(
        (
            order(
                "9",
                outcome=first.intent.outcome,
                price=first.intent.native_price,
                cloid=first.client_order_id,
            ),
            order(
                "10",
                outcome=second.intent.outcome,
                price=second.intent.native_price,
                cloid=second.client_order_id,
            ),
        )
    )
    client = FakeClient(cleanup_snapshot, snapshot())
    original_submit = client.submit_limit_order

    async def fail_second_submit(*args, **kwargs):
        if len(client.created) == 1:
            client.created.append(("ambiguous-second", kwargs["client_order_id"]))
            raise TimeoutError("ambiguous create timeout")
        return await original_submit(*args, **kwargs)

    client.submit_limit_order = fail_second_submit

    with pytest.raises(TimeoutError, match="ambiguous create timeout"):
        await execute_hip4_order_reconciliation(client, market(), reconciliation)

    assert [item[2] for item in client.cancelled] == [9, 10]
    assert [item[3] for item in client.cancelled] == [
        first.client_order_id,
        second.client_order_id,
    ]


@pytest.mark.asyncio
async def test_partial_create_cleanup_continues_after_individual_cancel_failure():
    reconciliation = reconcile_outcome_orders(market(), plan(), snapshot())
    first, second = reconciliation.creates
    cleanup_snapshot = snapshot(
        (
            order(
                "9",
                outcome=first.intent.outcome,
                price=first.intent.native_price,
                cloid=first.client_order_id,
            ),
            order(
                "10",
                outcome=second.intent.outcome,
                price=second.intent.native_price,
                cloid=second.client_order_id,
            ),
        )
    )
    client = FakeClient(cleanup_snapshot, snapshot())
    original_submit = client.submit_limit_order
    original_cancel = client.cancel_order

    async def fail_second_submit(*args, **kwargs):
        if len(client.created) == 1:
            client.created.append(("ambiguous-second", kwargs["client_order_id"]))
            raise TimeoutError("ambiguous create timeout")
        return await original_submit(*args, **kwargs)

    async def fail_first_cleanup_cancel(*args, **kwargs):
        if kwargs["order_id"] == 9:
            client.cancelled.append(
                (
                    args[0].market_id,
                    kwargs["outcome"],
                    kwargs["order_id"],
                    kwargs["expected_client_order_id"],
                )
            )
            raise RuntimeError("order filled before cleanup cancellation")
        return await original_cancel(*args, **kwargs)

    client.submit_limit_order = fail_second_submit
    client.cancel_order = fail_first_cleanup_cancel

    with pytest.raises(TimeoutError, match="ambiguous create timeout"):
        await execute_hip4_order_reconciliation(client, market(), reconciliation)

    assert [item[2] for item in client.cancelled] == [9, 10]


@pytest.mark.asyncio
async def test_executor_rejects_forged_unnamespaced_cancel_before_any_mutation():
    desired = replace(plan(), intents=())
    unmanaged = order(
        "2",
        outcome=OutcomeSide.YES,
        price=0.48,
        cloid="0x00000000000000000000000000000001",
    )
    reconciliation = reconcile_outcome_orders(market(), desired, snapshot((unmanaged,)))
    forged = replace(
        reconciliation,
        cancels=(
            OutcomeOrderCancel(
                order_id="2",
                outcome=OutcomeSide.YES,
                client_order_id=unmanaged.client_order_id,
            ),
        ),
    )
    client = FakeClient(snapshot(), snapshot())

    with pytest.raises(ValueError, match="unnamespaced"):
        await execute_hip4_order_reconciliation(client, market(), forged)

    assert client.cancelled == []
    assert client.created == []


@pytest.mark.asyncio
async def test_executor_rejects_forged_create_cloid_before_any_mutation():
    reconciliation = reconcile_outcome_orders(market(), plan(), snapshot())
    forged = replace(
        reconciliation,
        creates=(
            replace(
                reconciliation.creates[0],
                client_order_id="0x00000000000000000000000000000001",
            ),
        ),
    )
    client = FakeClient(snapshot(), snapshot())

    with pytest.raises(ValueError, match="invalid managed client-order ID"):
        await execute_hip4_order_reconciliation(client, market(), forged)

    assert client.cancelled == []
    assert client.created == []


@pytest.mark.asyncio
async def test_executor_rejects_create_for_slot_occupied_by_kept_order():
    desired = plan()
    bid_cloid = managed_outcome_client_order_id(
        "913",
        slot="canonical_bid",
        observation_end_ms=2_000,
    )
    account = snapshot(
        (order("1", outcome=OutcomeSide.YES, price=0.49, cloid=bid_cloid),)
    )
    reconciliation = reconcile_outcome_orders(market(), desired, account)
    assert reconciliation.kept[0].intent.slot == "canonical_bid"
    assert reconciliation.creates[0].intent.slot == "canonical_ask"
    forged_create = replace(
        reconciliation.creates[0],
        intent=reconciliation.kept[0].intent,
        client_order_id=managed_outcome_client_order_id(
            "913",
            slot="canonical_bid",
            observation_end_ms=reconciliation.observation_end_ms,
        ),
    )
    forged = replace(reconciliation, creates=(forged_create,))
    client = FakeClient()

    with pytest.raises(ValueError, match="slot occupied by a kept order"):
        await execute_hip4_order_reconciliation(client, market(), forged)

    assert client.cancelled == []
    assert client.created == []


@pytest.mark.asyncio
async def test_executor_rejects_created_order_with_wrong_authoritative_terms():
    reconciliation = reconcile_outcome_orders(
        market(),
        replace(plan(), intents=(plan().intents[0],)),
        snapshot(),
    )
    cloid = reconciliation.creates[0].client_order_id
    wrong_final = snapshot(
        (order("9", outcome=OutcomeSide.YES, price=0.48, cloid=cloid),)
    )
    client = FakeClient(wrong_final, snapshot())

    with pytest.raises(RuntimeError, match="exact intent"):
        await execute_hip4_order_reconciliation(client, market(), reconciliation)

    assert [item[2] for item in client.cancelled] == [9]


@pytest.mark.asyncio
async def test_executor_cleans_up_attempted_orders_after_final_verification_failure():
    reconciliation = reconcile_outcome_orders(market(), plan(), snapshot())
    first, second = reconciliation.creates
    incomplete_final = snapshot(
        (
            order(
                "9",
                outcome=first.intent.outcome,
                price=first.intent.native_price,
                cloid=first.client_order_id,
            ),
        )
    )
    client = FakeClient(incomplete_final, snapshot())

    with pytest.raises(RuntimeError, match="exact intent"):
        await execute_hip4_order_reconciliation(client, market(), reconciliation)

    assert [item[2] for item in client.cancelled] == [9]
    assert [item[3] for item in client.cancelled] == [first.client_order_id]
    assert len(client.created) == 2


@pytest.mark.asyncio
async def test_executor_cancels_surviving_kept_quote_after_partial_fill_race():
    desired = replace(plan(), intents=(plan().intents[0],))
    cloid = managed_outcome_client_order_id(
        "913",
        slot="canonical_bid",
        observation_end_ms=2_000,
    )
    kept = order("1", outcome=OutcomeSide.YES, price=0.49, cloid=cloid)
    reconciliation = reconcile_outcome_orders(
        market(),
        desired,
        snapshot((kept,)),
    )
    partially_filled = order(
        "1",
        outcome=OutcomeSide.YES,
        price=0.49,
        cloid=cloid,
        qty=10.0,
    )
    client = FakeClient(snapshot((partially_filled,)), snapshot())

    with pytest.raises(RuntimeError, match="kept managed order"):
        await execute_hip4_order_reconciliation(client, market(), reconciliation)

    assert [item[2] for item in client.cancelled] == [1]


@pytest.mark.asyncio
async def test_executor_recovers_all_targets_after_partial_cancellation_failure():
    stale_bid = managed_outcome_client_order_id(
        "913",
        slot="canonical_bid",
        observation_end_ms=2_000,
    )
    stale_ask = managed_outcome_client_order_id(
        "913",
        slot="canonical_ask",
        observation_end_ms=2_000,
    )
    first_order = order(
        "1",
        outcome=OutcomeSide.YES,
        price=0.48,
        cloid=stale_bid,
    )
    second_order = order(
        "2",
        outcome=OutcomeSide.NO,
        price=0.48,
        cloid=stale_ask,
    )
    reconciliation = reconcile_outcome_orders_to_empty(
        market(),
        snapshot((first_order, second_order)),
        decision_time_ms=4_000,
    )
    client = FakeClient(snapshot((second_order,)), snapshot())
    original_cancel = client.cancel_order
    failed_once = False

    async def fail_second_cancel(*args, **kwargs):
        nonlocal failed_once
        result = await original_cancel(*args, **kwargs)
        if kwargs["order_id"] == 2 and not failed_once:
            failed_once = True
            raise TimeoutError("ambiguous second cancellation")
        return result

    client.cancel_order = fail_second_cancel

    with pytest.raises(TimeoutError, match="ambiguous second cancellation"):
        await execute_hip4_order_reconciliation(client, market(), reconciliation)

    assert [item[2] for item in client.cancelled] == [1, 2, 2]
    assert [item[3] for item in client.cancelled] == [
        stale_bid,
        stale_ask,
        stale_ask,
    ]
