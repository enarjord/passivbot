from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import AsyncIterator

from outcome.archive import OutcomeTradeArchive
from outcome.live_data import (
    VerifiedOutcomeSignalWindow,
    collect_verified_outcome_signal_window,
)
from outcome.models import (
    NormalizedOutcomeMarket,
    NormalizedOutcomeTrade,
    OutcomeBookSnapshot,
    OutcomePriceGridChange,
    OutcomeVenue,
)
from outcome.public_streams import (
    stream_hyperliquid_public_books,
    stream_polymarket_public_books,
    stream_polymarket_public_price_grid_changes,
)


@dataclass(frozen=True)
class OutcomePublicCaptureResult:
    market_id: str
    collector_session: str
    signal_window: VerifiedOutcomeSignalWindow
    books_inserted: int
    books_ignored_as_duplicates: int
    price_grid_changes_inserted: int
    price_grid_changes_ignored_as_duplicates: int


async def _archive_books(
    stream: AsyncIterator[OutcomeBookSnapshot],
    *,
    market: NormalizedOutcomeMarket,
    archive: OutcomeTradeArchive,
    collector_session: str,
    counts: dict[str, int],
) -> None:
    try:
        async for book in stream:
            if book.venue is not market.venue or book.market_id != market.market_id:
                raise ValueError("outcome book stream returned a different market")
            if archive.append_book(book, collector_session=collector_session):
                counts["books_inserted"] += 1
            else:
                counts["books_ignored"] += 1
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()
    raise ConnectionError("outcome public book stream ended during capture")


async def _archive_price_grid_changes(
    stream: AsyncIterator[OutcomePriceGridChange],
    *,
    market: NormalizedOutcomeMarket,
    archive: OutcomeTradeArchive,
    collector_session: str,
    counts: dict[str, int],
) -> None:
    try:
        async for change in stream:
            if change.venue is not market.venue or change.market_id != market.market_id:
                raise ValueError("outcome price-grid stream returned a different market")
            if archive.append_price_grid_change(
                change,
                collector_session=collector_session,
            ):
                counts["grid_inserted"] += 1
            else:
                counts["grid_ignored"] += 1
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()
    raise ConnectionError("outcome public price-grid stream ended during capture")


async def capture_outcome_public_session(
    market: NormalizedOutcomeMarket,
    *,
    archive: OutcomeTradeArchive,
    collector_session: str,
    min_observations: int,
    max_wait_seconds: float = 120.0,
    delivery_lag_ms: int = 1_000,
    max_live_trade_lag_ms: int = 2_000,
    wall_clock_ms=None,
    trade_stream: AsyncIterator[NormalizedOutcomeTrade] | None = None,
    book_stream: AsyncIterator[OutcomeBookSnapshot] | None = None,
    price_grid_stream: AsyncIterator[OutcomePriceGridChange] | None = None,
    capture_books: bool = True,
    capture_price_grid_changes: bool = True,
) -> OutcomePublicCaptureResult:
    """Archive one bounded public session and return a verified trade-derived signal window.

    Books and price-grid changes are retained as independent audit/replay inputs. Only the trade
    stream can create verified candle coverage. The current Polymarket grid stream has no
    authoritative subscription-readiness boundary, so this bounded capture intentionally does not
    certify price-grid coverage. Any required auxiliary stream failure aborts the session instead
    of silently claiming a complete capture.
    """

    if not collector_session.strip():
        raise ValueError("collector_session must not be empty")
    if market.venue not in {OutcomeVenue.HYPERLIQUID, OutcomeVenue.POLYMARKET}:
        raise ValueError("unsupported venue for public outcome capture")
    if price_grid_stream is not None and market.venue is not OutcomeVenue.POLYMARKET:
        raise ValueError("only Polymarket exposes public outcome price-grid changes")
    if not capture_books and book_stream is not None:
        raise ValueError("book_stream was supplied while book capture is disabled")
    if not capture_price_grid_changes and price_grid_stream is not None:
        raise ValueError(
            "price_grid_stream was supplied while price-grid capture is disabled"
        )
    observed_at_ms = (
        int(time.time() * 1_000)
        if wall_clock_ms is None
        else int(wall_clock_ms())
    )
    archive.append_market_metadata(
        market,
        observed_at_ms=observed_at_ms,
        observation_source="public_capture_start",
        collector_session=collector_session,
    )

    counts = {
        "books_inserted": 0,
        "books_ignored": 0,
        "grid_inserted": 0,
        "grid_ignored": 0,
    }
    signal_task = asyncio.create_task(
        collect_verified_outcome_signal_window(
            market,
            min_observations=min_observations,
            max_wait_seconds=max_wait_seconds,
            delivery_lag_ms=delivery_lag_ms,
            max_live_trade_lag_ms=max_live_trade_lag_ms,
            wall_clock_ms=wall_clock_ms,
            trade_stream=trade_stream,
            archive=archive,
            collector_session=collector_session,
        ),
        name=f"outcome-trades-{market.market_id}",
    )
    auxiliary_tasks: list[asyncio.Task[None]] = []
    if capture_books:
        if book_stream is None:
            book_stream = (
                stream_hyperliquid_public_books((market,))
                if market.venue is OutcomeVenue.HYPERLIQUID
                else stream_polymarket_public_books((market,))
            )
        auxiliary_tasks.append(
            asyncio.create_task(
                _archive_books(
                    book_stream,
                    market=market,
                    archive=archive,
                    collector_session=collector_session,
                    counts=counts,
                ),
                name=f"outcome-books-{market.market_id}",
            )
        )
    if capture_price_grid_changes and market.venue is OutcomeVenue.POLYMARKET:
        if price_grid_stream is None:
            price_grid_stream = stream_polymarket_public_price_grid_changes((market,))
        auxiliary_tasks.append(
            asyncio.create_task(
                _archive_price_grid_changes(
                    price_grid_stream,
                    market=market,
                    archive=archive,
                    collector_session=collector_session,
                    counts=counts,
                ),
                name=f"outcome-grid-{market.market_id}",
            )
        )

    tasks: set[asyncio.Task] = {signal_task, *auxiliary_tasks}
    try:
        while True:
            done, _pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            auxiliary_failure = next(
                (
                    task.exception()
                    for task in done
                    if task is not signal_task
                    and not task.cancelled()
                    and task.exception() is not None
                ),
                None,
            )
            if auxiliary_failure is not None:
                raise auxiliary_failure
            if signal_task in done:
                window = signal_task.result()
                break
            raise ConnectionError("required outcome auxiliary stream stopped unexpectedly")
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    return OutcomePublicCaptureResult(
        market_id=market.market_id,
        collector_session=collector_session,
        signal_window=window,
        books_inserted=counts["books_inserted"],
        books_ignored_as_duplicates=counts["books_ignored"],
        price_grid_changes_inserted=counts["grid_inserted"],
        price_grid_changes_ignored_as_duplicates=counts["grid_ignored"],
    )
