from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import sqlite3
import time
from typing import Any, AsyncIterator, Callable, Mapping, Sequence

import aiohttp

from outcome.archive import OutcomeTradeArchive
from outcome.hyperliquid_live import (
    HyperliquidOutcomeAccountSnapshot,
    HyperliquidOutcomeLifecycleSnapshot,
    HyperliquidOutcomeLifecycleState,
    HyperliquidOutcomeLiveClient,
)
from outcome.live_data import (
    ContinuousVerifiedOutcomeSignalCollector,
    OutcomeIncompleteVerifiedSignal,
    OutcomeInvalidPublicSignal,
    OutcomeNoPublicFill,
    VerifiedOutcomeSignalWindow,
    collect_verified_hyperliquid_signal_window,
)
from outcome.live_planning import (
    DEFAULT_OUTCOME_MAX_SIGNAL_AGE_MS,
    OutcomeLivePlan,
    OutcomeSignalPlanningUnavailable,
    build_ema_anchor_outcome_live_plan,
)
from outcome.live_reconciliation import (
    OutcomeOrderMutationSkippedReason,
    OutcomeOrderReconciliation,
    OutcomeOrderReconciliationResult,
    execute_hip4_order_reconciliation,
    reconcile_outcome_orders,
    reconcile_outcome_orders_to_empty,
)
from outcome.models import (
    NormalizedOutcomeMarket,
    OutcomeSignalCandle1s,
)
from outcome.public_streams import OutcomeTradeStreamItem


class OutcomePlanningUnavailableReason(str, Enum):
    NO_PUBLIC_FILL = "no_public_fill"
    INCOMPLETE_VERIFIED_SIGNAL = "incomplete_verified_signal"
    STALE_VERIFIED_SIGNAL = "stale_verified_signal"
    STALE_ACCOUNT_SNAPSHOT = "stale_account_snapshot"
    SIGNAL_COLLECTION_FAILED = "signal_collection_failed"
    MARKET_CONSTRAINTS_UNAVAILABLE = "market_constraints_unavailable"
    MARKET_EXPIRED_AWAITING_SETTLEMENT = "market_expired_awaiting_settlement"
    MARKET_SETTLED = "market_settled"


@dataclass(frozen=True)
class HyperliquidOutcomeCycle:
    market_id: str
    planned_at_ms: int
    account: HyperliquidOutcomeAccountSnapshot
    lifecycle: HyperliquidOutcomeLifecycleSnapshot
    plan: OutcomeLivePlan | None
    reconciliation: OutcomeOrderReconciliation
    mutation_result: OutcomeOrderReconciliationResult | None
    planning_unavailable_reason: OutcomePlanningUnavailableReason | None = None

    @property
    def is_dry_run(self) -> bool:
        return self.mutation_result is None

    @property
    def planning_available(self) -> bool:
        return self.plan is not None


@dataclass(frozen=True)
class HyperliquidOutcomeCollectedCycle:
    cycle: HyperliquidOutcomeCycle
    signal_window: VerifiedOutcomeSignalWindow | None


async def run_hip4_outcome_cycle(
    client: HyperliquidOutcomeLiveClient,
    market: NormalizedOutcomeMarket,
    strategy_params: Mapping[str, Any],
    signal_candles: Sequence[OutcomeSignalCandle1s],
    *,
    execute: bool = False,
    now_ms: int | None = None,
    wall_clock_ms: Callable[[], int] | None = None,
    archive: OutcomeTradeArchive | None = None,
    collector_session: str | None = None,
) -> HyperliquidOutcomeCycle:
    """Plan and optionally reconcile one HIP-4 market from authoritative current inputs."""

    _validate_archive_session(archive, collector_session)
    account = await client.fetch_account_snapshot((market,))
    lifecycle = await client.fetch_market_lifecycle(
        market,
        account=account,
        now_ms=now_ms,
    )
    clock = wall_clock_ms or (lambda: int(time.time() * 1_000))
    planned_at_ms = int(clock()) if now_ms is None else int(now_ms)
    lifecycle_reason = _planning_reason_for_lifecycle(lifecycle)
    if lifecycle_reason is not None:
        cycle = await _finish_unavailable_cycle(
            client,
            market,
            account,
            lifecycle=lifecycle,
            reason=lifecycle_reason,
            execute=execute,
            planned_at_ms=planned_at_ms,
        )
        _persist_lifecycle_settlement(
            lifecycle,
            archive=archive,
            collector_session=collector_session,
        )
        return cycle
    _persist_lifecycle_settlement(
        lifecycle,
        archive=archive,
        collector_session=collector_session,
    )
    try:
        plan = build_ema_anchor_outcome_live_plan(
            market,
            strategy_params,
            signal_candles,
            account,
            now_ms=planned_at_ms,
        )
    except OutcomeSignalPlanningUnavailable as exc:
        return await _finish_unavailable_cycle(
            client,
            market,
            account,
            lifecycle=lifecycle,
            reason=OutcomePlanningUnavailableReason(exc.reason),
            execute=execute,
            planned_at_ms=planned_at_ms,
        )
    reconciliation = reconcile_outcome_orders(market, plan, account)
    mutation_result = None
    if execute:
        mutation_result = await execute_hip4_order_reconciliation(
            client,
            market,
            reconciliation,
            create_deadline_ms=(
                plan.observation_end_ms + DEFAULT_OUTCOME_MAX_SIGNAL_AGE_MS
            ),
            wall_clock_ms=(
                clock
                if wall_clock_ms is not None or now_ms is None
                else lambda: int(now_ms)
            ),
        )
        if (
            mutation_result.create_skipped_reason
            is OutcomeOrderMutationSkippedReason.STALE_VERIFIED_SIGNAL
        ):
            if mutation_result.create_skipped_at_ms is None:
                raise AssertionError("stale outcome execution omitted its decision time")
            decision_time_ms = mutation_result.create_skipped_at_ms // 1_000 * 1_000
            return HyperliquidOutcomeCycle(
                market_id=market.market_id,
                planned_at_ms=planned_at_ms,
                account=account,
                lifecycle=lifecycle,
                plan=None,
                reconciliation=reconcile_outcome_orders_to_empty(
                    market,
                    mutation_result.final_snapshot,
                    decision_time_ms=decision_time_ms,
                ),
                mutation_result=mutation_result,
                planning_unavailable_reason=(
                    OutcomePlanningUnavailableReason.STALE_VERIFIED_SIGNAL
                ),
            )
    return HyperliquidOutcomeCycle(
        market_id=market.market_id,
        planned_at_ms=planned_at_ms,
        account=account,
        lifecycle=lifecycle,
        plan=plan,
        reconciliation=reconciliation,
        mutation_result=mutation_result,
        planning_unavailable_reason=None,
    )


async def run_hip4_outcome_unavailable_cycle(
    client: HyperliquidOutcomeLiveClient,
    market: NormalizedOutcomeMarket,
    *,
    reason: OutcomePlanningUnavailableReason,
    execute: bool = False,
    now_ms: int | None = None,
    archive: OutcomeTradeArchive | None = None,
    collector_session: str | None = None,
) -> HyperliquidOutcomeCycle:
    """Produce a cancel-only decision when a verified strategy signal is unavailable."""

    if not isinstance(reason, OutcomePlanningUnavailableReason):
        raise TypeError("outcome planning unavailability requires a stable reason")
    _validate_archive_session(archive, collector_session)
    account = await client.fetch_account_snapshot((market,))
    lifecycle = await client.fetch_market_lifecycle(
        market,
        account=account,
        now_ms=now_ms,
    )
    planned_at_ms = int(time.time() * 1_000) if now_ms is None else int(now_ms)
    lifecycle_reason = _planning_reason_for_lifecycle(lifecycle)
    cycle = await _finish_unavailable_cycle(
        client,
        market,
        account,
        lifecycle=lifecycle,
        reason=lifecycle_reason or reason,
        execute=execute,
        planned_at_ms=planned_at_ms,
    )
    _persist_lifecycle_settlement(
        lifecycle,
        archive=archive,
        collector_session=collector_session,
    )
    return cycle


def _planning_reason_for_lifecycle(
    lifecycle: HyperliquidOutcomeLifecycleSnapshot,
) -> OutcomePlanningUnavailableReason | None:
    if lifecycle.state is HyperliquidOutcomeLifecycleState.ACTIVE:
        return None
    if lifecycle.state is HyperliquidOutcomeLifecycleState.SETTLED:
        return OutcomePlanningUnavailableReason.MARKET_SETTLED
    if (
        lifecycle.state
        is HyperliquidOutcomeLifecycleState.EXPIRED_AWAITING_SETTLEMENT
    ):
        return OutcomePlanningUnavailableReason.MARKET_EXPIRED_AWAITING_SETTLEMENT
    raise ValueError(f"unsupported HIP-4 lifecycle state {lifecycle.state!r}")


def _validate_archive_session(
    archive: OutcomeTradeArchive | None,
    collector_session: str | None,
) -> None:
    if archive is not None and not collector_session:
        raise ValueError("archived outcome cycles require collector_session")


def _persist_lifecycle_settlement(
    lifecycle: HyperliquidOutcomeLifecycleSnapshot,
    *,
    archive: OutcomeTradeArchive | None,
    collector_session: str | None,
) -> None:
    if archive is not None and lifecycle.settlement is not None:
        archive.append_settlement(
            lifecycle.settlement,
            collector_session=collector_session,
        )


async def _finish_unavailable_cycle(
    client: HyperliquidOutcomeLiveClient,
    market: NormalizedOutcomeMarket,
    account: HyperliquidOutcomeAccountSnapshot,
    *,
    lifecycle: HyperliquidOutcomeLifecycleSnapshot,
    reason: OutcomePlanningUnavailableReason,
    execute: bool,
    planned_at_ms: int,
) -> HyperliquidOutcomeCycle:
    decision_time_ms = planned_at_ms // 1_000 * 1_000
    reconciliation = reconcile_outcome_orders_to_empty(
        market,
        account,
        decision_time_ms=decision_time_ms,
    )
    mutation_result = (
        await execute_hip4_order_reconciliation(client, market, reconciliation)
        if execute
        else None
    )
    return HyperliquidOutcomeCycle(
        market_id=market.market_id,
        planned_at_ms=planned_at_ms,
        account=account,
        lifecycle=lifecycle,
        plan=None,
        reconciliation=reconciliation,
        mutation_result=mutation_result,
        planning_unavailable_reason=reason,
    )


async def run_hip4_outcome_collected_cycle(
    client: HyperliquidOutcomeLiveClient,
    market: NormalizedOutcomeMarket,
    strategy_params: Mapping[str, Any],
    *,
    min_observations: int,
    max_wait_seconds: float = 120.0,
    delivery_lag_ms: int = 1_000,
    max_live_trade_lag_ms: int = 2_000,
    archive: OutcomeTradeArchive | None = None,
    collector_session: str | None = None,
    execute: bool = False,
    now_ms: int | None = None,
    wall_clock_ms: Callable[[], int] | None = None,
    trade_stream: AsyncIterator[OutcomeTradeStreamItem] | None = None,
    continuous_collector: ContinuousVerifiedOutcomeSignalCollector | None = None,
) -> HyperliquidOutcomeCollectedCycle:
    """Consume a verified HIP-4 signal and reconcile, or cancel managed quotes on silence.

    Executing live loops should retain one ``continuous_collector`` across cycles so verified
    zero-volume seconds advance while account reconciliation runs.  A one-shot executable call
    clears managed quotes before its bounded bootstrap collection.
    """

    effective_archive = archive
    effective_collector_session = collector_session
    if continuous_collector is not None:
        if continuous_collector.market != market:
            raise ValueError("continuous outcome collector belongs to a different market")
        if trade_stream is not None:
            raise ValueError(
                "continuous collector and one-shot trade stream are mutually exclusive"
            )
        if archive is not None and archive is not continuous_collector.archive:
            raise ValueError("continuous collector owns a different outcome archive")
        if (
            collector_session is not None
            and collector_session != continuous_collector.collector_session
        ):
            raise ValueError("continuous collector owns a different archive session")
        effective_archive = continuous_collector.archive
        effective_collector_session = continuous_collector.collector_session

    if execute and (
        continuous_collector is None or not continuous_collector.has_emitted_window
    ):
        bootstrap = await run_hip4_outcome_unavailable_cycle(
            client,
            market,
            reason=OutcomePlanningUnavailableReason.INCOMPLETE_VERIFIED_SIGNAL,
            execute=True,
            now_ms=now_ms,
            archive=effective_archive,
            collector_session=effective_collector_session,
        )
        if bootstrap.lifecycle.state is not HyperliquidOutcomeLifecycleState.ACTIVE:
            return HyperliquidOutcomeCollectedCycle(cycle=bootstrap, signal_window=None)

    try:
        if continuous_collector is None:
            window = await collect_verified_hyperliquid_signal_window(
                market,
                min_observations=min_observations,
                max_wait_seconds=max_wait_seconds,
                delivery_lag_ms=delivery_lag_ms,
                max_live_trade_lag_ms=max_live_trade_lag_ms,
                wall_clock_ms=wall_clock_ms,
                trade_stream=trade_stream,
                archive=archive,
                collector_session=collector_session,
            )
        else:
            window = await continuous_collector.next_window(
                min_observations=min_observations,
                max_wait_seconds=max_wait_seconds,
                max_signal_age_ms=DEFAULT_OUTCOME_MAX_SIGNAL_AGE_MS,
            )
    except OutcomeNoPublicFill:
        cycle = await run_hip4_outcome_unavailable_cycle(
            client,
            market,
            reason=OutcomePlanningUnavailableReason.NO_PUBLIC_FILL,
            execute=execute,
            now_ms=now_ms,
            archive=effective_archive,
            collector_session=effective_collector_session,
        )
        return HyperliquidOutcomeCollectedCycle(cycle=cycle, signal_window=None)
    except OutcomeIncompleteVerifiedSignal:
        cycle = await run_hip4_outcome_unavailable_cycle(
            client,
            market,
            reason=OutcomePlanningUnavailableReason.INCOMPLETE_VERIFIED_SIGNAL,
            execute=execute,
            now_ms=now_ms,
            archive=effective_archive,
            collector_session=effective_collector_session,
        )
        return HyperliquidOutcomeCollectedCycle(cycle=cycle, signal_window=None)
    except (
        OutcomeInvalidPublicSignal,
        ConnectionError,
        OSError,
        aiohttp.ClientError,
        sqlite3.Error,
    ):
        cycle = await run_hip4_outcome_unavailable_cycle(
            client,
            market,
            reason=OutcomePlanningUnavailableReason.SIGNAL_COLLECTION_FAILED,
            execute=execute,
            now_ms=now_ms,
            archive=effective_archive,
            collector_session=effective_collector_session,
        )
        return HyperliquidOutcomeCollectedCycle(cycle=cycle, signal_window=None)

    cycle = await run_hip4_outcome_cycle(
        client,
        market,
        strategy_params,
        window.candles,
        execute=execute,
        now_ms=now_ms,
        wall_clock_ms=wall_clock_ms,
        archive=effective_archive,
        collector_session=effective_collector_session,
    )
    return HyperliquidOutcomeCollectedCycle(cycle=cycle, signal_window=window)
