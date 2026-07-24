from __future__ import annotations

import pytest

from outcome.archive import OutcomeTradeArchive
from outcome.candles import VerifiedCoverage
from outcome.live_data import (
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
