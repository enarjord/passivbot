from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from outcome.adapters import polymarket
from outcome.archive import OutcomeTradeArchive
from outcome.capture import capture_outcome_public_session
from outcome.candles import VerifiedCoverage
from outcome.models import (
    OutcomeBookLevel,
    OutcomeBookSnapshot,
    OutcomePriceGridChange,
    OutcomePriceGridMetadata,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "outcome"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.mark.asyncio
async def test_capture_archives_books_and_grid_without_using_them_as_candles(tmp_path):
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))
    trade = polymarket.normalize_market_ws_trade(
        {
            "event_type": "last_trade_price",
            "market": market.market_id,
            "asset_id": market.yes_asset.asset_id,
            "price": "0.40",
            "size": "2",
            "side": "BUY",
            "timestamp": "1900",
        },
        market,
        received_time_ms=1_950,
        collector_sequence=1,
    )
    book = OutcomeBookSnapshot(
        venue=market.venue,
        market_id=market.market_id,
        asset_id=market.yes_asset.asset_id,
        outcome=market.yes_asset.side,
        timestamp_ms=1_900,
        received_time_ms=1_950,
        bids=(OutcomeBookLevel(native_price=0.89, qty=10.0, order_count=None),),
        asks=(OutcomeBookLevel(native_price=0.90, qty=10.0, order_count=None),),
        raw_payload={"event_type": "book"},
    )
    grid_change = OutcomePriceGridChange(
        venue=market.venue,
        market_id=market.market_id,
        timestamp_ms=2_500,
        received_time_ms=2_550,
        old_grid=OutcomePriceGridMetadata(kind="fixed_step", fixed_step=0.01),
        new_grid=OutcomePriceGridMetadata(kind="fixed_step", fixed_step=0.001),
        raw_payload={"event_type": "tick_size_change"},
    )

    async def trades():
        await asyncio.sleep(0.01)
        yield trade

    async def books():
        yield book
        await asyncio.Event().wait()

    async def grids():
        yield grid_change
        await asyncio.Event().wait()

    archive = OutcomeTradeArchive(tmp_path / "outcomes.sqlite")
    result = await capture_outcome_public_session(
        market,
        archive=archive,
        collector_session="capture-1",
        min_observations=3,
        max_wait_seconds=1.0,
        delivery_lag_ms=0,
        wall_clock_ms=lambda: 5_100,
        trade_stream=trades(),
        book_stream=books(),
        price_grid_stream=grids(),
    )

    assert result.books_inserted == 1
    assert result.price_grid_changes_inserted == 1
    assert result.signal_window.coverage == VerifiedCoverage(2_000, 5_000)
    assert [candle.close for candle in result.signal_window.candles] == [0.4, 0.4, 0.4]
    assert all(candle.volume == 0.0 for candle in result.signal_window.candles)
    assert archive.load_market_metadata(market.venue, market.market_id) == [market]
    assert archive.load_books(
        market.venue,
        market.market_id,
        market.yes_asset.asset_id,
        start_ms=0,
        end_ms=10_000,
    ) == [book]
    assert archive.load_price_grid_changes(
        market.venue,
        market.market_id,
        start_ms=0,
        end_ms=10_000,
    ) == [grid_change]


@pytest.mark.asyncio
async def test_required_book_stream_failure_cancels_trade_capture_without_coverage(tmp_path):
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))
    waiting = asyncio.Event()

    async def trades():
        await waiting.wait()
        if False:  # pragma: no cover - keeps this an async generator
            yield

    async def failed_books():
        raise RuntimeError("book feed failed")
        if False:  # pragma: no cover - keeps this an async generator
            yield

    archive = OutcomeTradeArchive(tmp_path / "outcomes.sqlite")
    with pytest.raises(RuntimeError, match="book feed failed"):
        await capture_outcome_public_session(
            market,
            archive=archive,
            collector_session="capture-failure",
            min_observations=3,
            max_wait_seconds=1.0,
            trade_stream=trades(),
            book_stream=failed_books(),
            capture_price_grid_changes=False,
        )

    for asset in (market.yes_asset, market.no_asset):
        assert archive.load_verified_coverage(
            market.venue,
            market.market_id,
            asset.asset_id,
            start_ms=0,
            end_ms=10_000,
        ) == []


@pytest.mark.asyncio
async def test_capture_rejects_disabled_supplied_auxiliary_stream(tmp_path):
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))

    async def books():
        if False:  # pragma: no cover
            yield

    archive = OutcomeTradeArchive(tmp_path / "outcomes.sqlite")
    with pytest.raises(ValueError, match="book capture is disabled"):
        await capture_outcome_public_session(
            market,
            archive=archive,
            collector_session="capture-invalid",
            min_observations=3,
            trade_stream=None,
            book_stream=books(),
            capture_books=False,
        )
