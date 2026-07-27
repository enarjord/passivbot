from __future__ import annotations

from dataclasses import dataclass
import math

from outcome.hyperliquid_live import (
    HyperliquidOutcomeAccountSnapshot,
    HyperliquidOutcomeLiveClient,
    HyperliquidOutcomeMutationResult,
)
from outcome.live_planning import OutcomeLivePlan, OutcomeLiveOrderIntent
from outcome.models import NormalizedOutcomeMarket, OutcomeOpenOrder, OutcomeSide
from outcome.order_ownership import (
    is_managed_outcome_client_order_id,
    managed_outcome_client_order_id,
    managed_outcome_client_order_slot,
)


@dataclass(frozen=True)
class OutcomeOrderCreate:
    intent: OutcomeLiveOrderIntent
    client_order_id: str


@dataclass(frozen=True)
class OutcomeOrderCancel:
    order_id: str
    outcome: OutcomeSide
    client_order_id: str


@dataclass(frozen=True)
class OutcomeOrderReconciliation:
    market_id: str
    observation_end_ms: int
    creates: tuple[OutcomeOrderCreate, ...]
    cancels: tuple[OutcomeOrderCancel, ...]
    kept_order_ids: tuple[str, ...]
    unmanaged_order_ids: tuple[str, ...]


@dataclass(frozen=True)
class OutcomeOrderReconciliationResult:
    cancelled: tuple[HyperliquidOutcomeMutationResult, ...]
    created: tuple[HyperliquidOutcomeMutationResult, ...]
    final_snapshot: HyperliquidOutcomeAccountSnapshot


async def _cancel_attempted_creates(
    client: HyperliquidOutcomeLiveClient,
    market: NormalizedOutcomeMarket,
    attempted: tuple[OutcomeOrderCreate, ...],
    authoritative: HyperliquidOutcomeAccountSnapshot | None = None,
) -> None:
    """Restore a verified create-free state after a rejected or ambiguous submission."""

    attempted_cloids = {creation.client_order_id for creation in attempted}
    if authoritative is None:
        authoritative = await client.fetch_account_snapshot((market,))
    cleanup_orders = tuple(
        order
        for order in authoritative.open_orders
        if order.market_id == market.market_id
        and order.client_order_id in attempted_cloids
    )
    for order in cleanup_orders:
        await client.cancel_order(
            market,
            outcome=order.outcome,
            order_id=int(order.order_id),
            expected_client_order_id=order.client_order_id,
        )
    verified = await client.fetch_account_snapshot((market,))
    remaining = sorted(
        order.order_id
        for order in verified.open_orders
        if order.market_id == market.market_id
        and order.client_order_id in attempted_cloids
    )
    if remaining:
        raise RuntimeError(
            "HIP-4 partial-create cleanup is not authoritative: "
            f"{remaining}"
        )


async def _cancel_reconciliation_targets(
    client: HyperliquidOutcomeLiveClient,
    market: NormalizedOutcomeMarket,
    cancellations: tuple[OutcomeOrderCancel, ...],
) -> None:
    """Drive every targeted managed order to a verified absent state after cancel failure."""

    targets = {cancellation.order_id: cancellation for cancellation in cancellations}
    authoritative = await client.fetch_account_snapshot((market,))
    cleanup_errors: list[Exception] = []
    for order in authoritative.open_orders:
        cancellation = targets.get(order.order_id)
        if cancellation is None or order.market_id != market.market_id:
            continue
        if (
            order.outcome is not cancellation.outcome
            or order.client_order_id != cancellation.client_order_id
        ):
            cleanup_errors.append(
                RuntimeError(
                    f"HIP-4 cancellation target {order.order_id} changed authoritative identity"
                )
            )
            continue
        try:
            await client.cancel_order(
                market,
                outcome=cancellation.outcome,
                order_id=int(cancellation.order_id),
                expected_client_order_id=cancellation.client_order_id,
            )
        except Exception as exc:
            cleanup_errors.append(exc)

    verified = await client.fetch_account_snapshot((market,))
    remaining = sorted(
        order.order_id
        for order in verified.open_orders
        if order.market_id == market.market_id and order.order_id in targets
    )
    if remaining:
        error = RuntimeError(
            f"HIP-4 cancellation recovery is not authoritative: {remaining}"
        )
        if cleanup_errors:
            raise error from cleanup_errors[0]
        raise error


def _order_matches_intent(
    order: OutcomeOpenOrder,
    intent: OutcomeLiveOrderIntent,
    market_id: str,
) -> bool:
    return (
        managed_outcome_client_order_slot(order.client_order_id, market_id) == intent.slot
        and order.outcome is intent.outcome
        and order.side is intent.side
        and math.isclose(order.native_price, intent.native_price, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(order.qty, intent.qty, rel_tol=0.0, abs_tol=1e-12)
    )


def reconcile_outcome_orders(
    market: NormalizedOutcomeMarket,
    plan: OutcomeLivePlan,
    account: HyperliquidOutcomeAccountSnapshot,
) -> OutcomeOrderReconciliation:
    """Diff one market's desired intents without touching unnamespaced user orders."""

    if plan.market_id != market.market_id:
        raise ValueError("outcome live plan belongs to a different market")
    current_market_orders = tuple(
        order for order in account.open_orders if order.market_id == market.market_id
    )
    managed = [
        order
        for order in current_market_orders
        if is_managed_outcome_client_order_id(order.client_order_id, market.market_id)
    ]
    unmanaged = [
        order
        for order in current_market_orders
        if not is_managed_outcome_client_order_id(order.client_order_id, market.market_id)
    ]

    creates = []
    cancels = []
    kept = []
    retained_order_ids: set[str] = set()
    for intent in plan.intents:
        cloid = managed_outcome_client_order_id(
            market.market_id,
            slot=intent.slot,
            observation_end_ms=plan.observation_end_ms,
        )
        matches = [
            order
            for order in managed
            if order.order_id not in retained_order_ids
            and _order_matches_intent(order, intent, market.market_id)
        ]
        if matches:
            keep = sorted(matches, key=lambda order: order.order_id)[0]
            kept.append(keep.order_id)
            retained_order_ids.add(keep.order_id)
        else:
            creates.append(OutcomeOrderCreate(intent=intent, client_order_id=cloid))

    for order in managed:
        if order.order_id in retained_order_ids:
            continue
        if order.client_order_id is None:  # guarded by managed selection
            raise AssertionError("managed outcome order unexpectedly omitted client_order_id")
        cancels.append(
            OutcomeOrderCancel(
                order_id=order.order_id,
                outcome=order.outcome,
                client_order_id=order.client_order_id,
            )
        )
    return OutcomeOrderReconciliation(
        market_id=market.market_id,
        observation_end_ms=plan.observation_end_ms,
        creates=tuple(creates),
        cancels=tuple(cancels),
        kept_order_ids=tuple(sorted(kept)),
        unmanaged_order_ids=tuple(sorted(order.order_id for order in unmanaged)),
    )


def reconcile_outcome_orders_to_empty(
    market: NormalizedOutcomeMarket,
    account: HyperliquidOutcomeAccountSnapshot,
    *,
    decision_time_ms: int,
) -> OutcomeOrderReconciliation:
    """Cancel this market's managed quotes while preserving every unmanaged order."""

    if decision_time_ms < 0 or decision_time_ms % 1_000 != 0:
        raise ValueError("outcome safety decision time must be second-aligned")
    current_market_orders = tuple(
        order for order in account.open_orders if order.market_id == market.market_id
    )
    managed = [
        order
        for order in current_market_orders
        if is_managed_outcome_client_order_id(order.client_order_id, market.market_id)
    ]
    unmanaged = [
        order
        for order in current_market_orders
        if not is_managed_outcome_client_order_id(order.client_order_id, market.market_id)
    ]
    cancels = []
    for order in managed:
        if order.client_order_id is None:  # guarded by managed selection
            raise AssertionError("managed outcome order unexpectedly omitted client_order_id")
        cancels.append(
            OutcomeOrderCancel(
                order_id=order.order_id,
                outcome=order.outcome,
                client_order_id=order.client_order_id,
            )
        )
    return OutcomeOrderReconciliation(
        market_id=market.market_id,
        observation_end_ms=decision_time_ms,
        creates=(),
        cancels=tuple(sorted(cancels, key=lambda item: item.order_id)),
        kept_order_ids=(),
        unmanaged_order_ids=tuple(sorted(order.order_id for order in unmanaged)),
    )


async def execute_hip4_order_reconciliation(
    client: HyperliquidOutcomeLiveClient,
    market: NormalizedOutcomeMarket,
    reconciliation: OutcomeOrderReconciliation,
) -> OutcomeOrderReconciliationResult:
    """Cancel stale managed orders, verify cancellation, then create desired ALO orders."""

    if reconciliation.market_id != market.market_id:
        raise ValueError("outcome reconciliation belongs to a different market")
    _validate_managed_reconciliation(market, reconciliation)
    cancelled = []
    try:
        for cancellation in reconciliation.cancels:
            cancelled.append(
                await client.cancel_order(
                    market,
                    outcome=cancellation.outcome,
                    order_id=int(cancellation.order_id),
                    expected_client_order_id=cancellation.client_order_id,
                )
            )
        if reconciliation.cancels:
            after_cancels = await client.fetch_account_snapshot((market,))
            still_open = {
                order.order_id
                for order in after_cancels.open_orders
                if order.order_id in {item.order_id for item in reconciliation.cancels}
            }
            if still_open:
                raise RuntimeError(
                    f"HIP-4 managed order cancellation not yet authoritative: {sorted(still_open)}"
                )
    except Exception:
        try:
            await _cancel_reconciliation_targets(
                client,
                market,
                reconciliation.cancels,
            )
        except Exception as cleanup_error:
            raise RuntimeError(
                "HIP-4 order cancellation failed and recovery could not establish "
                "an authoritative safe state"
            ) from cleanup_error
        raise

    created = []
    attempted_creates = []
    try:
        for creation in reconciliation.creates:
            attempted_creates.append(creation)
            intent = creation.intent
            created.append(
                await client.submit_limit_order(
                    market,
                    outcome=intent.outcome,
                    side=intent.side,
                    native_price=intent.native_price,
                    qty=intent.qty,
                    client_order_id=creation.client_order_id,
                    post_only=True,
                )
            )
    except Exception:
        if not attempted_creates:
            raise
        try:
            await _cancel_attempted_creates(
                client,
                market,
                tuple(attempted_creates),
            )
        except Exception as cleanup_error:
            raise RuntimeError(
                "HIP-4 order creation failed and partial-create cleanup could "
                "not establish an authoritative safe state"
            ) from cleanup_error
        raise

    final_snapshot = None
    try:
        final_snapshot = await client.fetch_account_snapshot((market,))
        expected_cloids = {creation.client_order_id for creation in reconciliation.creates}
        final_market_orders = tuple(
            order
            for order in final_snapshot.open_orders
            if order.market_id == market.market_id
        )
        for creation in reconciliation.creates:
            matches = [
                order
                for order in final_market_orders
                if order.client_order_id == creation.client_order_id
            ]
            if len(matches) != 1 or not _order_matches_intent(
                matches[0],
                creation.intent,
                market.market_id,
            ):
                raise RuntimeError(
                    "HIP-4 created managed order is not authoritative with its exact "
                    f"intent: {creation.client_order_id}"
                )
        final_order_ids = {order.order_id for order in final_market_orders}
        missing_kept = set(reconciliation.kept_order_ids) - final_order_ids
        if missing_kept:
            raise RuntimeError(
                f"HIP-4 kept managed orders are no longer authoritative: {sorted(missing_kept)}"
            )
        cancelled_still_open = {
            cancellation.order_id
            for cancellation in reconciliation.cancels
            if cancellation.order_id in final_order_ids
        }
        if cancelled_still_open:
            raise RuntimeError(
                "HIP-4 cancelled managed orders reappeared in final state: "
                f"{sorted(cancelled_still_open)}"
            )
        expected_managed_ids = set(reconciliation.kept_order_ids)
        unexpected_managed = {
            order.order_id
            for order in final_market_orders
            if (
                is_managed_outcome_client_order_id(
                    order.client_order_id,
                    market.market_id,
                )
                and order.order_id not in expected_managed_ids
                and order.client_order_id not in expected_cloids
            )
        }
        if unexpected_managed:
            raise RuntimeError(
                "HIP-4 final state contains unexpected managed orders: "
                f"{sorted(unexpected_managed)}"
            )
    except Exception:
        if not attempted_creates:
            raise
        try:
            await _cancel_attempted_creates(
                client,
                market,
                tuple(attempted_creates),
                final_snapshot,
            )
        except Exception as cleanup_error:
            raise RuntimeError(
                "HIP-4 final verification failed and attempted-create cleanup could "
                "not establish an authoritative safe state"
            ) from cleanup_error
        raise
    return OutcomeOrderReconciliationResult(
        cancelled=tuple(cancelled),
        created=tuple(created),
        final_snapshot=final_snapshot,
    )


def _validate_managed_reconciliation(
    market: NormalizedOutcomeMarket,
    reconciliation: OutcomeOrderReconciliation,
) -> None:
    """Reject forged or internally inconsistent mutation instructions before any write."""

    cancel_ids = [cancellation.order_id for cancellation in reconciliation.cancels]
    if len(cancel_ids) != len(set(cancel_ids)):
        raise ValueError("outcome reconciliation contains duplicate cancellation order IDs")
    kept_ids = list(reconciliation.kept_order_ids)
    if len(kept_ids) != len(set(kept_ids)):
        raise ValueError("outcome reconciliation contains duplicate kept order IDs")
    if set(cancel_ids) & set(kept_ids):
        raise ValueError("outcome reconciliation cannot both keep and cancel an order")
    for cancellation in reconciliation.cancels:
        if not is_managed_outcome_client_order_id(
            cancellation.client_order_id,
            market.market_id,
        ):
            raise ValueError(
                "outcome reconciliation cannot cancel an unnamespaced order"
            )
        try:
            parsed_order_id = int(cancellation.order_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("outcome cancellation order ID must be an integer") from exc
        if parsed_order_id < 0 or str(parsed_order_id) != cancellation.order_id:
            raise ValueError("outcome cancellation order ID is not canonical")

    create_slots = [creation.intent.slot for creation in reconciliation.creates]
    if len(create_slots) != len(set(create_slots)):
        raise ValueError("outcome reconciliation contains duplicate create slots")
    for creation in reconciliation.creates:
        expected_cloid = managed_outcome_client_order_id(
            market.market_id,
            slot=creation.intent.slot,
            observation_end_ms=reconciliation.observation_end_ms,
        )
        if creation.client_order_id != expected_cloid:
            raise ValueError(
                "outcome reconciliation create has an invalid managed client-order ID"
            )
