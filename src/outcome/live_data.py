from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import time
from typing import AsyncIterator, Callable, Sequence

from outcome.archive import OutcomeTradeArchive, OutcomeVerifiedCoverageOverlap
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
    OutcomeTradeStreamItem,
    stream_hyperliquid_public_trades,
    stream_polymarket_public_trades,
)


class OutcomeNoPublicFill(TimeoutError):
    """No actual fill arrived before the bounded collection deadline."""


class OutcomeIncompleteVerifiedSignal(ValueError):
    """Actual fills arrived, but they did not prove the required dense signal window."""


class OutcomeInvalidPublicSignal(ValueError):
    """Public fill data or retained evidence failed validation during collection."""


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


class ContinuousVerifiedOutcomeSignalCollector:
    """Own one continuous public-fill stream and expose verified dense 1s snapshots.

    The collector consumes the websocket outside reconciliation cycles.  Once an actual fill
    establishes the canonical close, verified no-trade seconds advance the signal with flat,
    zero-volume candles.  Stream failure stops that progression instead of fabricating coverage.
    """

    def __init__(
        self,
        market: NormalizedOutcomeMarket,
        *,
        delivery_lag_ms: int = 1_000,
        max_live_trade_lag_ms: int = 2_000,
        wall_clock_ms: Callable[[], int] | None = None,
        trade_stream: AsyncIterator[OutcomeTradeStreamItem] | None = None,
        archive: OutcomeTradeArchive | None = None,
        collector_session: str | None = None,
    ) -> None:
        if market.venue not in {OutcomeVenue.HYPERLIQUID, OutcomeVenue.POLYMARKET}:
            raise ValueError("unsupported venue for continuous outcome signal collection")
        if delivery_lag_ms < 0:
            raise ValueError("delivery_lag_ms must be non-negative")
        if max_live_trade_lag_ms < 0:
            raise ValueError("max_live_trade_lag_ms must be non-negative")
        if archive is not None and not collector_session:
            raise ValueError("archived outcome collection requires collector_session")
        if archive is not None and market.venue is OutcomeVenue.POLYMARKET:
            raise ValueError(
                "continuous Polymarket archival requires an atomic identity-less segment writer"
            )
        self.market = market
        self.delivery_lag_ms = delivery_lag_ms
        self.max_live_trade_lag_ms = max_live_trade_lag_ms
        self.verification_lag_ms = max(delivery_lag_ms, max_live_trade_lag_ms)
        self._clock = wall_clock_ms or (lambda: int(time.time() * 1_000))
        self._supplied_stream = trade_stream
        self._stream: AsyncIterator[OutcomeTradeStreamItem] | None = None
        self._archive = archive
        self._collector_session = collector_session
        self._condition = asyncio.Condition()
        self._task: asyncio.Task[None] | None = None
        self._trades: list[NormalizedOutcomeTrade] = []
        self._rejected_trade_times_ms: list[int] = []
        self._coverage_start_ms: int | None = None
        self._last_returned_end_ms: int | None = None
        self._last_archived_end_ms: int | None = None
        self._failure: Exception | None = None

    @property
    def has_emitted_window(self) -> bool:
        return self._last_returned_end_ms is not None

    @property
    def archive(self) -> OutcomeTradeArchive | None:
        return self._archive

    @property
    def collector_session(self) -> str | None:
        return self._collector_session

    async def __aenter__(self) -> "ContinuousVerifiedOutcomeSignalCollector":
        await self.start()
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("continuous outcome collector is already started")
        if self._archive is not None:
            try:
                self._archive.append_market_metadata(
                    self.market,
                    observed_at_ms=self._clock(),
                    observation_source="continuous_live_fill_collection_start",
                    collector_session=str(self._collector_session),
                )
            except ValueError as exc:
                raise OutcomeInvalidPublicSignal(
                    "outcome market metadata conflicted with retained live evidence"
                ) from exc
        self._stream = self._supplied_stream or (
            stream_hyperliquid_public_trades((self.market,))
            if self.market.venue is OutcomeVenue.HYPERLIQUID
            else stream_polymarket_public_trades((self.market,))
        )
        self._task = asyncio.create_task(self._consume(), name="outcome-signal-collector")

    async def aclose(self) -> None:
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        stream = self._stream
        self._stream = None
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()

    async def _consume(self) -> None:
        assert self._stream is not None
        try:
            async for stream_item in self._stream:
                batch = (
                    (stream_item,)
                    if isinstance(stream_item, NormalizedOutcomeTrade)
                    else stream_item
                )
                self._accept_batch(batch)
                async with self._condition:
                    self._condition.notify_all()
            raise ConnectionError("continuous outcome public trade stream ended")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failure = exc
            async with self._condition:
                self._condition.notify_all()

    def _accept_batch(self, batch: tuple[NormalizedOutcomeTrade, ...]) -> None:
        if (
            not isinstance(batch, tuple)
            or not batch
            or not all(isinstance(trade, NormalizedOutcomeTrade) for trade in batch)
        ):
            raise OutcomeInvalidPublicSignal(
                "outcome public trade stream returned a malformed trade batch"
            )
        if len({trade.received_time_ms for trade in batch}) != 1:
            raise OutcomeInvalidPublicSignal(
                "outcome public trade batch does not share one receive timestamp"
            )
        for trade in batch:
            if trade.market_id != self.market.market_id or trade.venue is not self.market.venue:
                raise OutcomeInvalidPublicSignal(
                    "outcome public trade stream returned a different market"
                )
            if trade.collector_sequence is None:
                raise OutcomeInvalidPublicSignal(
                    "outcome live trade omitted collector chronology"
                )
            if (
                self._last_returned_end_ms is not None
                and trade.exchange_time_ms < self._last_returned_end_ms
            ):
                raise OutcomeInvalidPublicSignal(
                    "outcome fill arrived after its signal second was already emitted"
                )
            delivery_delay_ms = trade.received_time_ms - trade.exchange_time_ms
            if not -1_000 <= delivery_delay_ms <= self.max_live_trade_lag_ms:
                self._rejected_trade_times_ms.append(trade.exchange_time_ms)
                continue
            if self._archive is not None:
                try:
                    self._archive.append_trade(
                        trade,
                        collector_session=self._collector_session,
                    )
                except ValueError as exc:
                    raise OutcomeInvalidPublicSignal(
                        "outcome public trade conflicted with retained live evidence"
                    ) from exc
            self._trades.append(trade)
            if self._coverage_start_ms is None:
                self._coverage_start_ms = (
                    (trade.received_time_ms + 999) // 1_000
                ) * 1_000

    def _verified_end_ms(self) -> int | None:
        if self._coverage_start_ms is None:
            return None
        verified_through_ms = self._clock() - self.verification_lag_ms
        verified_end_ms = max(
            self._coverage_start_ms,
            (verified_through_ms // 1_000) * 1_000,
        )
        trading_close_ms = self.market.lifecycle.trading_close_time_ms
        return (
            verified_end_ms
            if trading_close_ms is None
            else min(verified_end_ms, trading_close_ms)
        )

    async def next_window(
        self,
        *,
        min_observations: int,
        max_wait_seconds: float = 120.0,
        max_signal_age_ms: int | None = None,
    ) -> VerifiedOutcomeSignalWindow:
        """Return the newest verified snapshot after at least one new completed second."""

        if min_observations <= 0:
            raise ValueError("min_observations must be positive")
        if max_wait_seconds <= 0.0:
            raise ValueError("max_wait_seconds must be positive")
        if max_signal_age_ms is not None and max_signal_age_ms < 0:
            raise ValueError("max_signal_age_ms must be non-negative")
        if self._task is None:
            raise RuntimeError("continuous outcome collector is not started")
        loop = asyncio.get_running_loop()
        effective_wait_seconds = max_wait_seconds
        if self._last_returned_end_ms is not None and max_signal_age_ms is not None:
            freshness_remaining_seconds = (
                self._last_returned_end_ms + max_signal_age_ms - self._clock()
            ) / 1_000
            effective_wait_seconds = min(
                effective_wait_seconds,
                max(0.0, freshness_remaining_seconds),
            )
        deadline = loop.time() + effective_wait_seconds
        while True:
            if self._failure is not None:
                raise self._failure
            verified_end_ms = self._verified_end_ms()
            if self._coverage_start_ms is not None and verified_end_ms is not None:
                required_end_ms = (
                    self._coverage_start_ms + min_observations * 1_000
                    if self._last_returned_end_ms is None
                    else self._last_returned_end_ms + 1_000
                )
                if verified_end_ms >= required_end_ms:
                    coverage = VerifiedCoverage(self._coverage_start_ms, verified_end_ms)
                    if any(
                        coverage.start_ms <= timestamp_ms < coverage.end_ms
                        for timestamp_ms in self._rejected_trade_times_ms
                    ):
                        raise OutcomeIncompleteVerifiedSignal(
                            "outcome collection observed an in-window fill outside the "
                            "allowed delivery lag"
                        )
                    try:
                        window = build_verified_outcome_signal_window(
                            self._trades,
                            coverage,
                            min_observations=min_observations,
                        )
                    except OutcomeIncompleteVerifiedSignal:
                        raise
                    except ValueError as exc:
                        raise OutcomeInvalidPublicSignal(
                            "outcome public fills could not produce a valid signal window"
                        ) from exc
                    self._archive_new_coverage(verified_end_ms)
                    self._last_returned_end_ms = verified_end_ms
                    return window
            remaining = deadline - loop.time()
            if remaining <= 0.0:
                if self._coverage_start_ms is None:
                    raise OutcomeNoPublicFill(
                        "no outcome public fill arrived before the collection deadline"
                    )
                raise OutcomeIncompleteVerifiedSignal(
                    "continuous outcome signal did not advance before the deadline"
                )
            poll_seconds = min(remaining, 0.25)
            async with self._condition:
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=poll_seconds)
                except asyncio.TimeoutError:
                    pass

    def _archive_new_coverage(self, verified_end_ms: int) -> None:
        if self._archive is None:
            return
        start_ms = self._last_archived_end_ms or self._coverage_start_ms
        assert start_ms is not None
        if verified_end_ms <= start_ms:
            return
        coverage = VerifiedCoverage(start_ms, verified_end_ms)
        for asset in (self.market.yes_asset, self.market.no_asset):
            self._archive.record_verified_coverage(
                self.market.venue,
                self.market.market_id,
                asset.asset_id,
                coverage,
                collector_session=str(self._collector_session),
            )
        self._last_archived_end_ms = verified_end_ms


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
    trade_stream: AsyncIterator[OutcomeTradeStreamItem] | None = None,
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
    verification_lag_ms = max(delivery_lag_ms, max_live_trade_lag_ms)
    clock = wall_clock_ms or (lambda: int(time.time() * 1_000))
    if archive is not None:
        try:
            archive.append_market_metadata(
                market,
                observed_at_ms=clock(),
                observation_source="live_fill_collection_start",
                collector_session=str(collector_session),
            )
        except ValueError as exc:
            raise OutcomeInvalidPublicSignal(
                "outcome market metadata conflicted with retained live evidence"
            ) from exc
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
    deferred_archive_trades: list[NormalizedOutcomeTrade] = []
    rejected_trade_times_ms: list[int] = []
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
                stream_item = await asyncio.wait_for(
                    anext(stream),
                    timeout=wait_seconds,
                )
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
                deferred_archive_trades.clear()
                rejected_trade_times_ms.clear()
                first_received_ms = None
                collection_end_ms = None
                stream = stream_factory()
                continue
            except ValueError as exc:
                raise OutcomeInvalidPublicSignal(
                    "outcome public trade stream returned malformed data"
                ) from exc
            batch = (
                (stream_item,)
                if isinstance(stream_item, NormalizedOutcomeTrade)
                else stream_item
            )
            if (
                not isinstance(batch, tuple)
                or not batch
                or not all(isinstance(trade, NormalizedOutcomeTrade) for trade in batch)
            ):
                raise OutcomeInvalidPublicSignal(
                    "outcome public trade stream returned a malformed trade batch"
                )
            if len({trade.received_time_ms for trade in batch}) != 1:
                raise OutcomeInvalidPublicSignal(
                    "outcome public trade batch does not share one receive timestamp"
                )
            # One stream item is one already-decoded websocket message. Process every
            # trade before checking the deadline again so certified coverage cannot
            # omit a fill that was already received.
            for trade in batch:
                if trade.market_id != market.market_id:
                    raise OutcomeInvalidPublicSignal(
                        "outcome public trade stream returned a different market"
                    )
                if trade.collector_sequence is None:
                    raise OutcomeInvalidPublicSignal(
                        "outcome live trade omitted collector chronology"
                    )
                delivery_delay_ms = trade.received_time_ms - trade.exchange_time_ms
                if not -1_000 <= delivery_delay_ms <= max_live_trade_lag_ms:
                    rejected_trade_times_ms.append(trade.exchange_time_ms)
                    continue
                if archive is not None:
                    if market.venue is OutcomeVenue.POLYMARKET:
                        deferred_archive_trades.append(trade)
                    else:
                        try:
                            archive.append_trade(
                                trade,
                                collector_session=collector_session,
                            )
                        except ValueError as exc:
                            raise OutcomeInvalidPublicSignal(
                                "outcome public trade conflicted with retained live evidence"
                            ) from exc
                trades.append(trade)
                if first_received_ms is None:
                    first_received_ms = trade.received_time_ms
                    coverage_start_ms = ((first_received_ms + 999) // 1_000) * 1_000
                    collection_end_ms = (
                        coverage_start_ms
                        + min_observations * 1_000
                        + verification_lag_ms
                    )
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()

    if first_received_ms is None:
        raise OutcomeNoPublicFill("outcome signal collection completed without a public fill")
    assert collection_end_ms is not None
    coverage_start_ms = ((first_received_ms + 999) // 1_000) * 1_000
    verified_through_ms = min(clock(), collection_end_ms) - verification_lag_ms
    coverage_end_ms = (verified_through_ms // 1_000) * 1_000
    if coverage_end_ms <= coverage_start_ms:
        raise OutcomeIncompleteVerifiedSignal(
            "outcome collection did not complete one verified signal second"
        )
    coverage = VerifiedCoverage(coverage_start_ms, coverage_end_ms)
    if any(
        coverage.start_ms <= exchange_time_ms < coverage.end_ms
        for exchange_time_ms in rejected_trade_times_ms
    ):
        raise OutcomeIncompleteVerifiedSignal(
            "outcome collection observed an in-window fill outside the allowed delivery lag"
        )

    def materialize_window() -> VerifiedOutcomeSignalWindow:
        try:
            return build_verified_outcome_signal_window(
                trades,
                coverage,
                min_observations=min_observations,
            )
        except OutcomeIncompleteVerifiedSignal:
            raise
        except ValueError as exc:
            raise OutcomeInvalidPublicSignal(
                "outcome public fills could not produce a valid signal window"
            ) from exc

    if archive is not None and market.venue is OutcomeVenue.POLYMARKET:
        try:
            with archive.write_transaction():
                has_identityless_trades = any(
                    trade.source_event_id is None and trade.sequence_id is None
                    for trade in deferred_archive_trades
                )
                if has_identityless_trades:
                    for asset in (market.yes_asset, market.no_asset):
                        archive.require_no_verified_coverage_overlap(
                            market.venue,
                            market.market_id,
                            asset.asset_id,
                            coverage,
                        )
                archivable_trades = [
                    trade
                    for trade in deferred_archive_trades
                    if (
                        trade.source_event_id is not None
                        or trade.sequence_id is not None
                        or coverage.start_ms
                        <= trade.exchange_time_ms
                        < coverage.end_ms
                    )
                ]
                for trade in archivable_trades:
                    archive.append_trade(
                        trade,
                        collector_session=collector_session,
                    )
                window = materialize_window()
                for asset in (market.yes_asset, market.no_asset):
                    archive.record_verified_coverage(
                        market.venue,
                        market.market_id,
                        asset.asset_id,
                        coverage,
                        collector_session=str(collector_session),
                    )
        except OutcomeIncompleteVerifiedSignal:
            raise
        except OutcomeInvalidPublicSignal:
            raise
        except OutcomeVerifiedCoverageOverlap as exc:
            raise OutcomeInvalidPublicSignal(
                "Polymarket identity-less fills cannot certify overlapping coverage"
            ) from exc
        except ValueError as exc:
            raise OutcomeInvalidPublicSignal(
                "outcome public trade conflicted with retained live evidence"
            ) from exc
    else:
        window = materialize_window()
        if archive is not None:
            try:
                for asset in (market.yes_asset, market.no_asset):
                    archive.record_verified_coverage(
                        market.venue,
                        market.market_id,
                        asset.asset_id,
                        coverage,
                        collector_session=str(collector_session),
                    )
            except ValueError as exc:
                raise OutcomeInvalidPublicSignal(
                    "outcome verified coverage conflicted with retained live evidence"
                ) from exc
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
