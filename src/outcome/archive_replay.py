from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from outcome.archive import OutcomeTradeArchive
from outcome.backtest_input import build_trade_derived_ema_anchor_input
from outcome.candles import VerifiedCoverage
from outcome.models import (
    MarketLifecycle,
    NormalizedOutcomeMarket,
    OutcomeSettlementEvidence,
    OutcomeVenue,
)
from outcome.rust_runner import normalized_market_to_rust_spec


@dataclass(frozen=True)
class ArchivedOutcomeReplay:
    market: NormalizedOutcomeMarket
    settlement: OutcomeSettlementEvidence
    coverage: VerifiedCoverage
    actual_fill_records: int
    payload: Mapping[str, Any]


def _proves_interval(
    intervals: list[VerifiedCoverage],
    start_ms: int,
    end_ms: int,
) -> bool:
    cursor = start_ms
    for interval in intervals:
        if interval.end_ms <= cursor:
            continue
        if interval.start_ms > cursor:
            return False
        cursor = max(cursor, interval.end_ms)
        if cursor >= end_ms:
            return True
    return False


def _authoritative_settlement(
    settlements: list[OutcomeSettlementEvidence],
    *,
    market_id: str,
) -> OutcomeSettlementEvidence:
    if not settlements:
        raise ValueError(f"outcome archive has no settlement evidence for {market_id}")
    yes_fractions = {settlement.yes_fraction for settlement in settlements}
    payout_units = {settlement.payout_unit for settlement in settlements}
    if len(yes_fractions) != 1 or len(payout_units) != 1:
        raise ValueError(f"outcome archive has conflicting settlement evidence for {market_id}")
    return max(
        settlements,
        key=lambda settlement: (
            settlement.capital_release_time_ms is not None,
            settlement.capital_release_time_ms or settlement.settlement_time_ms,
            settlement.received_time_ms,
            settlement.source_event_id,
        ),
    )


def consolidated_archived_market(
    market_versions: list[NormalizedOutcomeMarket],
) -> NormalizedOutcomeMarket:
    """Keep initial trading terms while merging later lifecycle observations.

    Polymarket metadata observed while a contract is live has no actual close time. A later closed
    observation supplies that field, so replay must not blindly select the first archived version.
    """

    if not market_versions:
        raise ValueError("cannot consolidate an empty outcome market history")
    first = market_versions[0]
    initial_constraints = (
        first.qty_step,
        first.min_order_qty,
        first.min_order_notional,
    )
    for version in market_versions[1:]:
        constraints = (
            version.qty_step,
            version.min_order_qty,
            version.min_order_notional,
        )
        if constraints != initial_constraints:
            raise ValueError(
                "full-contract outcome replay does not yet support changing "
                "quantity or minimum-order constraints"
            )

    def first_present(name: str):
        return next(
            (
                getattr(market.lifecycle, name)
                for market in market_versions
                if getattr(market.lifecycle, name) is not None
            ),
            None,
        )

    def last_present(name: str):
        return next(
            (
                getattr(market.lifecycle, name)
                for market in reversed(market_versions)
                if getattr(market.lifecycle, name) is not None
            ),
            None,
        )

    lifecycle = MarketLifecycle(
        discovery_time_ms=first_present("discovery_time_ms"),
        trading_open_time_ms=first_present("trading_open_time_ms"),
        order_acceptance_time_ms=first_present("order_acceptance_time_ms"),
        trading_close_time_ms=last_present("trading_close_time_ms"),
        scheduled_event_time_ms=first_present("scheduled_event_time_ms"),
        resolution_time_ms=last_present("resolution_time_ms"),
        settlement_time_ms=last_present("settlement_time_ms"),
        accepting_orders=last_present("accepting_orders"),
        resolved=last_present("resolved"),
        yes_payout_fraction=last_present("yes_payout_fraction"),
    )
    return replace(first, lifecycle=lifecycle)


def load_archived_opening_market(
    archive: OutcomeTradeArchive,
    *,
    venue: OutcomeVenue,
    market_id: str,
) -> tuple[NormalizedOutcomeMarket, int]:
    """Load the latest retained market state observed no later than trading open."""

    market_versions = archive.load_market_metadata(venue, market_id)
    if not market_versions:
        raise ValueError(f"outcome archive has no market metadata for {market_id}")
    start_ms = next(
        (
            version.lifecycle.trading_open_time_ms
            for version in market_versions
            if version.lifecycle.trading_open_time_ms is not None
        ),
        None,
    )
    if start_ms is None:
        raise ValueError(f"outcome market {market_id} has no complete trading lifecycle")
    opening_market = archive.load_market_metadata_at(
        venue,
        market_id,
        observed_at_or_before_ms=start_ms,
    )
    if opening_market is None:
        raise ValueError(
            f"outcome archive has no market metadata observed by trading open "
            f"for {market_id}"
        )
    return opening_market, start_ms


def build_archived_ema_anchor_replay(
    archive: OutcomeTradeArchive,
    *,
    venue: OutcomeVenue,
    market_id: str,
    fee_schedule: Mapping[str, Any],
    requested_collateral: float,
    strategy_params: Mapping[str, Any],
    qty_step: float | None = None,
) -> ArchivedOutcomeReplay:
    """Build one authoritative full-contract replay from retained metadata and fills."""

    opening_market, start_ms = load_archived_opening_market(
        archive,
        venue=venue,
        market_id=market_id,
    )
    market = consolidated_archived_market(
        [
            opening_market,
            *archive.load_market_metadata_observed_after(
                venue,
                market_id,
                observed_after_ms=start_ms,
            ),
        ]
    )
    end_ms = market.lifecycle.trading_close_time_ms
    if end_ms is None or end_ms <= start_ms:
        raise ValueError(f"outcome market {market_id} has no complete trading lifecycle")
    settlement = _authoritative_settlement(
        archive.load_settlements(venue, market_id),
        market_id=market_id,
    )
    capital_release_time_ms = settlement.capital_release_time_ms
    if capital_release_time_ms is None:
        raise ValueError(
            f"outcome archive has resolution but no authoritative capital release "
            f"evidence for {market_id}"
        )
    if capital_release_time_ms < end_ms:
        raise ValueError(f"outcome capital release for {market_id} predates trading close")

    trades = []
    for asset in (market.yes_asset, market.no_asset):
        coverage = archive.load_verified_coverage(
            venue,
            market_id,
            asset.asset_id,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        if not _proves_interval(coverage, start_ms, end_ms):
            raise ValueError(
                f"outcome archive does not prove full-contract coverage for "
                f"{market_id} {asset.side.value}"
            )
        trades.extend(
            archive.load_trades(
                venue,
                market_id,
                asset.asset_id,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
    price_grid_changes = archive.load_price_grid_changes(
        venue,
        market_id,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    if venue is OutcomeVenue.POLYMARKET:
        grid_coverage = archive.load_verified_price_grid_coverage(
            venue,
            market_id,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        if not _proves_interval(grid_coverage, start_ms, end_ms):
            raise ValueError(
                f"outcome archive does not prove full-contract price-grid coverage "
                f"for {market_id}"
            )
    full_coverage = VerifiedCoverage(start_ms, end_ms)
    payload = build_trade_derived_ema_anchor_input(
        market_spec=normalized_market_to_rust_spec(market, qty_step=qty_step),
        trades=trades,
        verified_coverage=(full_coverage,),
        fee_schedule=fee_schedule,
        starting_collateral=requested_collateral,
        strategy_params=strategy_params,
        settlement_time_ms=settlement.settlement_time_ms,
        yes_fraction=settlement.yes_fraction,
        price_grid_changes=price_grid_changes,
    )
    return ArchivedOutcomeReplay(
        market=market,
        settlement=settlement,
        coverage=full_coverage,
        actual_fill_records=len(trades),
        payload=payload,
    )
