from __future__ import annotations

from dataclasses import replace

import pytest

from outcome.candles import (
    AmbiguousTradeOrderingError,
    VerifiedCoverage,
    trades_to_1s_candles,
    trades_to_canonical_signal_1s_candles,
)
from outcome.models import (
    NormalizedOutcomeTrade,
    OutcomeOrderSide,
    OutcomeSide,
    OutcomeVenue,
)


def trade(
    timestamp_ms: int,
    price: float,
    qty: float,
    event_id: str | None,
    *,
    outcome: OutcomeSide = OutcomeSide.YES,
    received_time_ms: int | None = None,
    sequence_id: str | None = None,
) -> NormalizedOutcomeTrade:
    native_price = price if outcome is OutcomeSide.YES else 1.0 - price
    return NormalizedOutcomeTrade(
        venue=OutcomeVenue.HYPERLIQUID,
        market_id="913",
        asset_id="+9130" if outcome is OutcomeSide.YES else "+9131",
        outcome=outcome,
        native_side=OutcomeOrderSide.BUY,
        native_price=native_price,
        canonical_yes_price=price,
        qty=qty,
        exchange_time_ms=timestamp_ms,
        received_time_ms=timestamp_ms + 10 if received_time_ms is None else received_time_ms,
        source_event_id=event_id,
        sequence_id=sequence_id,
    )


def test_trade_derived_ohlcv_and_explicit_identity_deduplication():
    first = trade(1_100, 0.40, 2.0, "one")
    candles = trades_to_1s_candles(
        [
            trade(1_900, 0.45, 1.0, "three"),
            first,
            replace(first, received_time_ms=9_999),
            trade(1_500, 0.35, 3.0, "two"),
        ]
    )

    assert len(candles) == 1
    candle = candles[0]
    assert candle.open == pytest.approx(0.40)
    assert candle.high == pytest.approx(0.45)
    assert candle.low == pytest.approx(0.35)
    assert candle.close == pytest.approx(0.45)
    assert candle.volume == pytest.approx(6.0)
    assert candle.trade_count == 3
    assert candle.carried_forward is False


@pytest.mark.parametrize(
    "builder",
    [trades_to_1s_candles, trades_to_canonical_signal_1s_candles],
)
@pytest.mark.parametrize("identity_kind", ["source", "sequence"])
def test_conflicting_duplicate_trade_identity_is_rejected(builder, identity_kind):
    original = trade(
        1_100,
        0.4,
        1.0,
        "same" if identity_kind == "source" else None,
        sequence_id="same" if identity_kind == "sequence" else None,
    )
    conflicting = replace(
        original,
        native_price=0.45,
        canonical_yes_price=0.45,
    )

    with pytest.raises(ValueError, match="conflicting outcome trade evidence"):
        builder([original, conflicting])


def test_trade_rejects_inconsistent_native_to_canonical_price_mapping():
    with pytest.raises(ValueError, match="canonical_yes_price must equal"):
        NormalizedOutcomeTrade(
            venue=OutcomeVenue.HYPERLIQUID,
            market_id="913",
            asset_id="+9131",
            outcome=OutcomeSide.NO,
            native_side=OutcomeOrderSide.BUY,
            native_price=0.2,
            canonical_yes_price=0.2,
            qty=1.0,
            exchange_time_ms=1_000,
            received_time_ms=1_010,
        )


def test_verified_no_trade_seconds_carry_close_but_unknown_gap_breaks_continuity():
    candles = trades_to_1s_candles(
        [trade(1_100, 0.40, 1.0, "one"), trade(5_100, 0.55, 1.0, "two")],
        verified_coverage=[
            VerifiedCoverage(1_000, 3_000),
            VerifiedCoverage(4_000, 7_000),
        ],
    )

    assert [candle.timestamp_ms for candle in candles] == [1_000, 2_000, 5_000, 6_000]
    assert candles[1].carried_forward is True
    assert candles[1].open == candles[1].high == candles[1].low == candles[1].close == 0.40
    assert candles[1].volume == 0.0
    assert candles[3].carried_forward is True
    assert candles[3].close == 0.55


def test_native_yes_and_no_books_cannot_be_collapsed_for_execution_candles():
    with pytest.raises(ValueError, match="native outcome book"):
        trades_to_1s_candles(
            [
                trade(1_100, 0.4, 1.0, "yes", outcome=OutcomeSide.YES),
                trade(1_200, 0.4, 1.0, "no", outcome=OutcomeSide.NO),
            ]
        )


def test_no_trade_input_does_not_invent_a_starting_price():
    assert trades_to_1s_candles([], verified_coverage=[VerifiedCoverage(0, 10_000)]) == []


def test_signal_candle_combines_only_actual_fills_from_yes_and_no_books():
    candles = trades_to_canonical_signal_1s_candles(
        [
            trade(1_100, 0.4, 1.0, "yes", outcome=OutcomeSide.YES),
            trade(1_200, 0.6, 2.0, "no", outcome=OutcomeSide.NO),
        ]
    )
    assert len(candles) == 1
    assert candles[0].open == 0.4
    assert candles[0].close == 0.6
    assert candles[0].volume == 3.0
    assert candles[0].trade_count == 2


def test_signal_candle_counts_mirrored_merged_book_trade_once():
    yes = replace(
        trade(1_100, 0.4, 1.0, "yes", outcome=OutcomeSide.YES),
        economic_event_id="transaction-leg",
    )
    no = replace(
        trade(1_100, 0.4, 1.0, "no", outcome=OutcomeSide.NO),
        economic_event_id="transaction-leg",
    )
    candles = trades_to_canonical_signal_1s_candles([yes, no])
    assert candles[0].volume == 1.0
    assert candles[0].trade_count == 1


def test_signal_candle_pairs_repeated_equal_merged_book_trades_by_occurrence():
    trades = [
        replace(
            trade(
                1_100,
                0.4,
                1.0,
                f"{outcome.value}-{occurrence}",
                outcome=outcome,
                sequence_id=str(sequence),
            ),
            economic_event_id="same-transaction-price-and-quantity",
        )
        for sequence, (occurrence, outcome) in enumerate(
            (
                (0, OutcomeSide.YES),
                (1, OutcomeSide.YES),
                (0, OutcomeSide.NO),
                (1, OutcomeSide.NO),
            ),
            start=1,
        )
    ]

    candles = trades_to_canonical_signal_1s_candles(trades)

    assert candles[0].volume == 2.0
    assert candles[0].trade_count == 2


@pytest.mark.parametrize(
    "builder",
    [trades_to_1s_candles, trades_to_canonical_signal_1s_candles],
)
def test_same_exchange_timestamp_without_ordering_evidence_is_rejected(builder):
    with pytest.raises(AmbiguousTradeOrderingError, match="without a unique source sequence"):
        builder(
            [
                trade(1_000, 0.40, 1.0, "one", received_time_ms=2_000),
                trade(1_000, 0.45, 1.0, "two", received_time_ms=2_000),
            ]
        )


def test_unique_receive_times_establish_open_and_close_order():
    candles = trades_to_1s_candles(
        [
            trade(1_000, 0.45, 1.0, "later", received_time_ms=2_002),
            trade(1_000, 0.40, 1.0, "earlier", received_time_ms=2_001),
        ]
    )

    assert candles[0].open == pytest.approx(0.40)
    assert candles[0].close == pytest.approx(0.45)


def test_numeric_source_sequence_wins_over_receive_order():
    candles = trades_to_1s_candles(
        [
            trade(
                1_000,
                0.45,
                1.0,
                None,
                received_time_ms=2_001,
                sequence_id="10",
            ),
            trade(
                1_000,
                0.40,
                1.0,
                None,
                received_time_ms=2_002,
                sequence_id="2",
            ),
        ]
    )

    assert candles[0].open == pytest.approx(0.40)
    assert candles[0].close == pytest.approx(0.45)


def test_collector_sequence_orders_tied_websocket_fills():
    candles = trades_to_1s_candles(
        [
            replace(
                trade(1_000, 0.45, 1.0, "later", received_time_ms=2_000),
                collector_sequence=11,
            ),
            replace(
                trade(1_000, 0.40, 1.0, "earlier", received_time_ms=2_000),
                collector_sequence=10,
            ),
        ]
    )

    assert candles[0].open == pytest.approx(0.40)
    assert candles[0].close == pytest.approx(0.45)
