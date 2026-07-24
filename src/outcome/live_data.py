from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import AsyncIterator, Callable, Sequence

from outcome.archive import OutcomeTradeArchive
from outcome.candles import (
    VerifiedCoverage,
    trades_to_canonical_signal_1s_candles,
)
from outcome.models import (
    NormalizedOutcomeMarket,
    NormalizedOutcomeTrade,
    OutcomeSignalCandle1s,
    OutcomeVenue,
)
from outcome.public_streams import (
    stream_hyperliquid_public_trades,
    stream_polymarket_public_trades,
)


class OutcomeNoPublicFill(TimeoutError):
    """No actual fill arrived before the bounded collection deadline."""


class OutcomeIncompleteVerifiedSignal(ValueError):
    """Actual fills arrived, but they did not prove the required dense signal window."""


@dataclass(frozen=True)
class VerifiedOutcomeSignalWindow:
    coverage: VerifiedCoverage
    trades: tuple[NormalizedOutcomeTrade, ...]
    candles: tuple[OutcomeSignalCandle1s, ...]

    def __post_init__(self) -> None:
        if not self.trades or not self.candles:
            raise ValueError("verified outcome signal window must contain trades and candles")
        if self.candles[0].timestamp_ms < self.coverage.start_ms:
            raise ValueError("verified outcome signal window starts before coverage")
        if self.candles[-1].timestamp_ms + 1_000 > self.coverage.end_ms:
            raise ValueError("verified outcome signal window ends after coverage")

    @property
    def covered_trades(self) -> tuple[NormalizedOutcomeTrade, ...]:
        """Return only fills eligible for execution inside the verified interval.

        `trades` may include a boundary fill observed just before coverage starts. It can seed
        the prior signal close, but it must not be eligible for simulated execution.
        """

        return tuple(
            trade
            for trade in self.trades
            if self.coverage.start_ms
            <= trade.exchange_time_ms
            < self.coverage.end_ms
        )


def build_verified_outcome_signal_window(
    trades: Sequence[NormalizedOutcomeTrade],
    coverage: VerifiedCoverage,
    *,
    min_observations: int,
) -> VerifiedOutcomeSignalWindow:
    if min_observations <= 0:
        raise ValueError("min_observations must be positive")
    candles = [
        candle
        for candle in trades_to_canonical_signal_1s_candles(
            trades,
            verified_coverage=(coverage,),
        )
        if coverage.start_ms <= candle.timestamp_ms
        and candle.timestamp_ms + 1_000 <= coverage.end_ms
    ]
    if len(candles) < min_observations:
        raise OutcomeIncompleteVerifiedSignal(
            f"verified outcome window has {len(candles)} observations; "
            f"requires {min_observations}"
        )
    if any(
        candles[index].timestamp_ms != candles[index - 1].timestamp_ms + 1_000
        for index in range(1, len(candles))
    ):
        raise OutcomeIncompleteVerifiedSignal(
            "verified outcome signal window is not contiguous"
        )
    return VerifiedOutcomeSignalWindow(
        coverage=coverage,
        trades=tuple(trades),
        candles=tuple(candles),
    )


async def collect_verified_outcome_signal_window(
    market: NormalizedOutcomeMarket,
    *,
    min_observations: int,
    max_wait_seconds: float = 120.0,
    delivery_lag_ms: int = 1_000,
    max_live_trade_lag_ms: int = 2_000,
    wall_clock_ms: Callable[[], int] | None = None,
    trade_stream: AsyncIterator[NormalizedOutcomeTrade] | None = None,
    archive: OutcomeTradeArchive | None = None,
    collector_session: str | None = None,
) -> VerifiedOutcomeSignalWindow:
    """Collect actual public fills, then materialize only fully observed one-second bars."""

    if market.venue not in {OutcomeVenue.HYPERLIQUID, OutcomeVenue.POLYMARKET}:
        raise ValueError("unsupported venue for outcome signal collection")
    if min_observations <= 0:
        raise ValueError("min_observations must be positive")
    if max_wait_seconds <= 0.0:
        raise ValueError("max_wait_seconds must be positive")
    if delivery_lag_ms < 0:
        raise ValueError("delivery_lag_ms must be non-negative")
    if max_live_trade_lag_ms < 0:
        raise ValueError("max_live_trade_lag_ms must be non-negative")
    if archive is not None and not collector_session:
        raise ValueError("archived outcome collection requires collector_session")
    clock = wall_clock_ms or (lambda: int(time.time() * 1_000))
    if archive is not None:
        archive.append_market_metadata(
            market,
            observed_at_ms=clock(),
            observation_source="live_fill_collection_start",
            collector_session=str(collector_session),
        )
    owns_stream = trade_stream is None
    stream_factory = (
        (lambda: stream_hyperliquid_public_trades((market,)))
        if market.venue is OutcomeVenue.HYPERLIQUID
        else (lambda: stream_polymarket_public_trades((market,)))
    )
    stream = trade_stream or stream_factory()
    loop = asyncio.get_running_loop()
    overall_deadline = loop.time() + max_wait_seconds
    collection_end_ms: int | None = None
    first_received_ms: int | None = None
    trades: list[NormalizedOutcomeTrade] = []
    try:
        while True:
            now_monotonic = loop.time()
            if now_monotonic >= overall_deadline:
                if first_received_ms is None:
                    raise OutcomeNoPublicFill(
                        "no outcome public fill arrived before the collection deadline"
                    )
                break
            if collection_end_ms is not None and clock() >= collection_end_ms:
                break
            wait_seconds = overall_deadline - now_monotonic
            if collection_end_ms is not None:
                wait_seconds = min(
                    wait_seconds,
                    max(0.001, (collection_end_ms - clock()) / 1_000),
                )
            try:
                trade = await asyncio.wait_for(anext(stream), timeout=wait_seconds)
            except asyncio.TimeoutError:
                if first_received_ms is None:
                    raise OutcomeNoPublicFill(
                        "no outcome public fill arrived before the collection deadline"
                    ) from None
                break
            except StopAsyncIteration as exc:
                if not owns_stream:
                    raise ConnectionError(
                        "outcome public trade stream ended during collection"
                    ) from exc
                trades.clear()
                first_received_ms = None
                collection_end_ms = None
                stream = stream_factory()
                continue
            if trade.market_id != market.market_id:
                raise ValueError("outcome public trade stream returned a different market")
            if trade.collector_sequence is None:
                raise ValueError("outcome live trade omitted collector chronology")
            if archive is not None:
                archive.append_trade(trade, collector_session=collector_session)
            delivery_delay_ms = trade.received_time_ms - trade.exchange_time_ms
            if not -1_000 <= delivery_delay_ms <= max_live_trade_lag_ms:
                continue
            trades.append(trade)
            if first_received_ms is None:
                first_received_ms = trade.received_time_ms
                coverage_start_ms = ((first_received_ms + 999) // 1_000) * 1_000
                collection_end_ms = (
                    coverage_start_ms
                    + min_observations * 1_000
                    + delivery_lag_ms
                )
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()

    if first_received_ms is None:
        raise OutcomeNoPublicFill("outcome signal collection completed without a public fill")
    coverage_start_ms = ((first_received_ms + 999) // 1_000) * 1_000
    coverage_end_ms = ((clock() - delivery_lag_ms) // 1_000) * 1_000
    if coverage_end_ms <= coverage_start_ms:
        raise OutcomeIncompleteVerifiedSignal(
            "outcome collection did not complete one verified signal second"
        )
    coverage = VerifiedCoverage(coverage_start_ms, coverage_end_ms)
    window = build_verified_outcome_signal_window(
        trades,
        coverage,
        min_observations=min_observations,
    )
    if archive is not None:
        for asset in (market.yes_asset, market.no_asset):
            archive.record_verified_coverage(
                market.venue,
                market.market_id,
                asset.asset_id,
                coverage,
                collector_session=str(collector_session),
            )
    return window


async def collect_verified_hyperliquid_signal_window(
    market: NormalizedOutcomeMarket,
    **kwargs,
) -> VerifiedOutcomeSignalWindow:
    if market.venue is not OutcomeVenue.HYPERLIQUID:
        raise ValueError("HIP-4 signal collection requires a Hyperliquid market")
    return await collect_verified_outcome_signal_window(market, **kwargs)


async def collect_verified_polymarket_signal_window(
    market: NormalizedOutcomeMarket,
    **kwargs,
) -> VerifiedOutcomeSignalWindow:
    if market.venue is not OutcomeVenue.POLYMARKET:
        raise ValueError("Polymarket signal collection requires a Polymarket market")
    return await collect_verified_outcome_signal_window(market, **kwargs)
