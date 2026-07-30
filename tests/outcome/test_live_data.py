from __future__ import annotations

import pytest

from outcome.archive import OutcomeTradeArchive
from outcome.candles import VerifiedCoverage
from outcome.live_data import (
    OutcomeIncompleteVerifiedSignal,
    OutcomeInvalidPublicSignal,
    build_verified_outcome_signal_window,
    collect_verified_polymarket_signal_window,
)
from outcome.models import (
    MarketLifecycle,
    NormalizedOutcomeAsset,
    NormalizedOutcomeMarket,
    NormalizedOutcomeTrade,
    OutcomeCapabilities,
    OutcomeFeeMetadata,
    OutcomeOrderSide,
    OutcomePriceGridMetadata,
    OutcomeSide,
    OutcomeVenue,
)


def trade(timestamp_ms: int, received_time_ms: int, price: float):
    return NormalizedOutcomeTrade(
        venue=OutcomeVenue.HYPERLIQUID,
        market_id="913",
        asset_id="+9130",
        outcome=OutcomeSide.YES,
        native_side=OutcomeOrderSide.BUY,
        native_price=price,
        canonical_yes_price=price,
        qty=1.0,
        exchange_time_ms=timestamp_ms,
        received_time_ms=received_time_ms,
        source_event_id=f"trade-{timestamp_ms}",
        collector_sequence=timestamp_ms,
    )


def polymarket_market() -> NormalizedOutcomeMarket:
    return NormalizedOutcomeMarket(
        venue=OutcomeVenue.POLYMARKET,
        market_id="condition",
        title="BTC above target?",
        description="",
        quote_asset="USDC",
        yes_asset=NormalizedOutcomeAsset(
            side=OutcomeSide.YES,
            label="Yes",
            asset_id="yes-token",
            market_data_symbol="yes-token",
            order_asset_id="yes-token",
        ),
        no_asset=NormalizedOutcomeAsset(
            side=OutcomeSide.NO,
            label="No",
            asset_id="no-token",
            market_data_symbol="no-token",
            order_asset_id="no-token",
        ),
        payout_unit=1.0,
        price_grid=OutcomePriceGridMetadata(kind="fixed_step", fixed_step=0.001),
        qty_step=None,
        min_order_qty=1.0,
        min_order_notional=None,
        lifecycle=MarketLifecycle(accepting_orders=True),
        capabilities=OutcomeCapabilities(
            complementary_books_merged=False,
            sell_requires_inventory=True,
            supports_split=True,
            supports_merge=True,
            supports_redeem=True,
            supports_post_only=True,
            supports_gtd=True,
        ),
        fee_metadata=OutcomeFeeMetadata(formula="venue_reported_zero"),
    )


def test_verified_live_window_uses_trade_seed_and_carries_only_covered_seconds():
    window = build_verified_outcome_signal_window(
        [trade(1_900, 1_950, 0.4)],
        VerifiedCoverage(2_000, 5_000),
        min_observations=3,
    )

    assert [candle.timestamp_ms for candle in window.candles] == [2_000, 3_000, 4_000]
    assert all(candle.open == candle.high == candle.low == candle.close == 0.4 for candle in window.candles)
    assert all(candle.volume == 0.0 for candle in window.candles)
    assert window.trades == (trade(1_900, 1_950, 0.4),)
    assert window.covered_trades == ()


def test_verified_live_window_rejects_insufficient_completed_seconds():
    with pytest.raises(ValueError, match="requires 3"):
        build_verified_outcome_signal_window(
            [trade(1_900, 1_950, 0.4)],
            VerifiedCoverage(2_000, 4_000),
            min_observations=3,
        )


@pytest.mark.asyncio
async def test_polymarket_collector_uses_same_actual_fill_and_zero_second_contract(
    tmp_path,
):
    market = polymarket_market()
    archive = OutcomeTradeArchive(tmp_path / "outcomes.sqlite")

    async def stream():
        yield NormalizedOutcomeTrade(
            venue=OutcomeVenue.POLYMARKET,
            market_id=market.market_id,
            asset_id=market.yes_asset.asset_id,
            outcome=OutcomeSide.YES,
            native_side=OutcomeOrderSide.BUY,
            native_price=0.4,
            canonical_yes_price=0.4,
            qty=2.0,
            exchange_time_ms=1_900,
            received_time_ms=1_950,
            source_event_id="fill",
            collector_sequence=1,
        )

    window = await collect_verified_polymarket_signal_window(
        market,
        min_observations=3,
        delivery_lag_ms=0,
        max_live_trade_lag_ms=100,
        wall_clock_ms=lambda: 5_100,
        trade_stream=stream(),
        archive=archive,
        collector_session="live-data-test",
    )

    assert window.coverage == VerifiedCoverage(2_000, 5_000)
    assert [candle.timestamp_ms for candle in window.candles] == [2_000, 3_000, 4_000]
    assert all(candle.volume == 0.0 for candle in window.candles)
    assert len(window.trades) == 1
    assert window.covered_trades == ()
    assert archive.load_market_metadata(market.venue, market.market_id) == [market]


@pytest.mark.asyncio
async def test_late_in_window_fill_prevents_verified_coverage(tmp_path):
    market = polymarket_market()
    archive = OutcomeTradeArchive(tmp_path / "late-fill.sqlite")

    async def stream():
        for sequence, (exchange_time_ms, received_time_ms) in enumerate(
            ((1_900, 1_950), (2_500, 5_001)),
            start=1,
        ):
            yield NormalizedOutcomeTrade(
                venue=market.venue,
                market_id=market.market_id,
                asset_id=market.yes_asset.asset_id,
                outcome=OutcomeSide.YES,
                native_side=OutcomeOrderSide.BUY,
                native_price=0.4,
                canonical_yes_price=0.4,
                qty=1.0,
                exchange_time_ms=exchange_time_ms,
                received_time_ms=received_time_ms,
                source_event_id=f"late-{sequence}",
                collector_sequence=sequence,
            )

    clock_values = iter((1_900, 2_500, 2_500, 5_000, 5_000))
    with pytest.raises(
        OutcomeIncompleteVerifiedSignal,
        match="outside the allowed delivery lag",
    ):
        await collect_verified_polymarket_signal_window(
            market,
            min_observations=1,
            delivery_lag_ms=0,
            max_live_trade_lag_ms=2_000,
            wall_clock_ms=lambda: next(clock_values),
            trade_stream=stream(),
            archive=archive,
            collector_session="late-fill-test",
        )

    assert archive.load_verified_coverage(
        market.venue,
        market.market_id,
        market.yes_asset.asset_id,
        start_ms=0,
        end_ms=10_000,
    ) == []


@pytest.mark.asyncio
async def test_overlag_fill_outside_new_window_is_rejected_before_archive(tmp_path):
    market = polymarket_market()
    archive = OutcomeTradeArchive(tmp_path / "overlag-fill.sqlite")
    retained_coverage = VerifiedCoverage(1_000, 2_000)
    for asset in (market.yes_asset, market.no_asset):
        archive.record_verified_coverage(
            market.venue,
            market.market_id,
            asset.asset_id,
            retained_coverage,
            collector_session="retained",
        )

    async def stream():
        for sequence, (exchange_time_ms, received_time_ms, price) in enumerate(
            ((4_900, 4_950, 0.4), (1_500, 5_000, 0.6)),
            start=1,
        ):
            yield NormalizedOutcomeTrade(
                venue=market.venue,
                market_id=market.market_id,
                asset_id=market.yes_asset.asset_id,
                outcome=OutcomeSide.YES,
                native_side=OutcomeOrderSide.BUY,
                native_price=price,
                canonical_yes_price=price,
                qty=1.0,
                exchange_time_ms=exchange_time_ms,
                received_time_ms=received_time_ms,
                source_event_id=None,
                collector_sequence=sequence,
            )

    clock_values = iter((4_900, 5_000, 5_000, 6_200, 6_200))
    window = await collect_verified_polymarket_signal_window(
        market,
        min_observations=1,
        delivery_lag_ms=0,
        max_live_trade_lag_ms=100,
        wall_clock_ms=lambda: next(clock_values),
        trade_stream=stream(),
        archive=archive,
        collector_session="new-window",
    )

    retained = archive.load_trades(
        market.venue,
        market.market_id,
        market.yes_asset.asset_id,
        start_ms=0,
        end_ms=10_000,
    )
    assert window.coverage == VerifiedCoverage(5_000, 6_000)
    assert [item.exchange_time_ms for item in retained] == [4_900]


@pytest.mark.asyncio
async def test_collector_waits_through_maximum_accepted_trade_lag():
    market = polymarket_market()

    async def stream():
        yield NormalizedOutcomeTrade(
            venue=market.venue,
            market_id=market.market_id,
            asset_id=market.yes_asset.asset_id,
            outcome=OutcomeSide.YES,
            native_side=OutcomeOrderSide.BUY,
            native_price=0.4,
            canonical_yes_price=0.4,
            qty=1.0,
            exchange_time_ms=1_900,
            received_time_ms=1_950,
            source_event_id="seed",
            collector_sequence=1,
        )
        yield NormalizedOutcomeTrade(
            venue=market.venue,
            market_id=market.market_id,
            asset_id=market.yes_asset.asset_id,
            outcome=OutcomeSide.YES,
            native_side=OutcomeOrderSide.BUY,
            native_price=0.6,
            canonical_yes_price=0.6,
            qty=1.0,
            exchange_time_ms=2_900,
            received_time_ms=4_500,
            source_event_id="accepted-late-fill",
            collector_sequence=2,
        )

    clock_values = iter((2_500, 2_500, 5_000, 5_000))
    window = await collect_verified_polymarket_signal_window(
        market,
        min_observations=1,
        delivery_lag_ms=0,
        max_live_trade_lag_ms=2_000,
        wall_clock_ms=lambda: next(clock_values),
        trade_stream=stream(),
    )

    assert window.coverage == VerifiedCoverage(2_000, 3_000)
    assert window.candles[0].close == pytest.approx(0.6)
    assert window.candles[0].volume == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_polymarket_identityless_collector_rejects_overlapping_coverage(
    tmp_path,
):
    market = polymarket_market()
    archive = OutcomeTradeArchive(tmp_path / "overlap.sqlite")

    def stream():
        async def _stream():
            yield NormalizedOutcomeTrade(
                venue=market.venue,
                market_id=market.market_id,
                asset_id=market.yes_asset.asset_id,
                outcome=OutcomeSide.YES,
                native_side=OutcomeOrderSide.BUY,
                native_price=0.4,
                canonical_yes_price=0.4,
                qty=1.0,
                exchange_time_ms=1_900,
                received_time_ms=1_950,
                source_event_id=None,
                collector_sequence=1,
                raw_payload={"event_type": "last_trade_price"},
            )

        return _stream()

    await collect_verified_polymarket_signal_window(
        market,
        min_observations=3,
        delivery_lag_ms=0,
        max_live_trade_lag_ms=100,
        wall_clock_ms=lambda: 5_100,
        trade_stream=stream(),
        archive=archive,
        collector_session="first",
    )
    with pytest.raises(
        OutcomeInvalidPublicSignal,
        match="cannot certify overlapping coverage",
    ):
        await collect_verified_polymarket_signal_window(
            market,
            min_observations=3,
            delivery_lag_ms=0,
            max_live_trade_lag_ms=100,
            wall_clock_ms=lambda: 5_100,
            trade_stream=stream(),
            archive=archive,
            collector_session="second",
        )

    assert len(
        archive.load_trades(
            market.venue,
            market.market_id,
            market.yes_asset.asset_id,
            start_ms=0,
            end_ms=10_000,
        )
    ) == 1


@pytest.mark.asyncio
async def test_owned_polymarket_reconnect_discards_abandoned_session_fills(
    tmp_path,
    monkeypatch,
):
    market = polymarket_market()
    archive = OutcomeTradeArchive(tmp_path / "reconnect.sqlite")
    sessions = iter(
        (
            (
                NormalizedOutcomeTrade(
                    venue=market.venue,
                    market_id=market.market_id,
                    asset_id=market.yes_asset.asset_id,
                    outcome=OutcomeSide.YES,
                    native_side=OutcomeOrderSide.BUY,
                    native_price=0.4,
                    canonical_yes_price=0.4,
                    qty=1.0,
                    exchange_time_ms=1_900,
                    received_time_ms=1_950,
                    source_event_id=None,
                    collector_sequence=1,
                ),
            ),
            (
                NormalizedOutcomeTrade(
                    venue=market.venue,
                    market_id=market.market_id,
                    asset_id=market.yes_asset.asset_id,
                    outcome=OutcomeSide.YES,
                    native_side=OutcomeOrderSide.BUY,
                    native_price=0.6,
                    canonical_yes_price=0.6,
                    qty=2.0,
                    exchange_time_ms=3_900,
                    received_time_ms=3_950,
                    source_event_id=None,
                    collector_sequence=1,
                ),
            ),
        )
    )

    def stream_factory(markets):
        assert markets == (market,)
        session_trades = next(sessions)

        async def stream():
            for item in session_trades:
                yield item

        return stream()

    clock_values = iter((1_000, 3_000, 3_000, 7_100, 7_100))
    monkeypatch.setattr(
        "outcome.live_data.stream_polymarket_public_trades",
        stream_factory,
    )

    window = await collect_verified_polymarket_signal_window(
        market,
        min_observations=3,
        delivery_lag_ms=0,
        max_live_trade_lag_ms=100,
        wall_clock_ms=lambda: next(clock_values),
        archive=archive,
        collector_session="reconnect",
    )

    retained = archive.load_trades(
        market.venue,
        market.market_id,
        market.yes_asset.asset_id,
        start_ms=0,
        end_ms=10_000,
    )
    assert [item.native_price for item in retained] == [0.6]
    assert window.coverage == VerifiedCoverage(4_000, 7_000)
    assert window.candles[0].close == pytest.approx(0.6)
