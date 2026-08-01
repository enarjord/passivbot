from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from outcome.adapters import hyperliquid, polymarket
from outcome.archive import OutcomeTradeArchive
from outcome.candles import VerifiedCoverage, trades_to_canonical_signal_1s_candles
from outcome.historical import (
    archive_historical_batch,
    parse_hyperliquid_node_fills_by_block,
    parse_polymarket_order_filled_logs,
)
from outcome.models import OutcomeOrderSide, OutcomeSide, OutcomeVenue


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "outcome"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_hyperliquid_block_archive_deduplicates_participant_fills_and_proves_zero_seconds(
    tmp_path,
):
    market = hyperliquid.normalize_market(fixture("hyperliquid_price_binary.json"))
    fill = {
        "coin": market.yes_asset.market_data_symbol,
        "px": "0.335",
        "sz": "2",
        "side": "B",
        "time": 1_500,
        "hash": "0xtrade",
        "tid": 7,
    }
    opposite = dict(fill, side="A")
    lines = [
        json.dumps(
            {
                "local_time": "1970-01-01T00:00:01.600Z",
                "block_time": "1970-01-01T00:00:01.000Z",
                "block_number": 100,
                "events": [["0xbuyer", fill], ["0xseller", opposite]],
            }
        ),
        json.dumps(
            {
                "local_time": "1970-01-01T00:00:02.600Z",
                "block_time": "1970-01-01T00:00:03.000Z",
                "block_number": 101,
                "events": [],
            }
        ),
    ]

    batch = parse_hyperliquid_node_fills_by_block(
        lines,
        market,
        source_cursor="node_fills_by_block/hourly/19700101/0.lz4",
        received_time_ms=9_000,
    )

    assert batch.venue is OutcomeVenue.HYPERLIQUID
    assert len(batch.trades) == 1
    assert batch.trades[0].sequence_id == "00000000000000000100:00000000:00000000"
    coverage = batch.coverage_by_asset[market.yes_asset.asset_id]
    assert coverage == (VerifiedCoverage(2_000, 3_000),)
    candles = trades_to_canonical_signal_1s_candles(
        batch.trades,
        verified_coverage=coverage,
    )
    assert [(c.timestamp_ms, c.volume, c.close) for c in candles] == [
        (1_000, 2.0, 0.335),
        (2_000, 0.0, 0.335),
    ]

    archive = OutcomeTradeArchive(tmp_path / "outcomes.sqlite")
    assert archive_historical_batch(archive, batch, collector_session="historical-1") == (1, 0)
    assert archive_historical_batch(archive, batch, collector_session="historical-1") == (0, 1)
    assert archive.load_verified_coverage(
        OutcomeVenue.HYPERLIQUID,
        market.market_id,
        market.yes_asset.asset_id,
        start_ms=0,
        end_ms=10_000,
    ) == [VerifiedCoverage(2_000, 3_000)]


def test_historical_batch_rolls_back_all_rows_on_late_trade_conflict(tmp_path):
    market = hyperliquid.normalize_market(fixture("hyperliquid_price_binary.json"))
    fill = {
        "coin": market.yes_asset.market_data_symbol,
        "px": "0.335",
        "sz": "2",
        "side": "B",
        "time": 1_500,
        "hash": "0xtrade",
        "tid": 7,
    }
    lines = [
        json.dumps(
            {
                "block_time": "1970-01-01T00:00:01.000Z",
                "block_number": 100,
                "events": [["0xbuyer", fill]],
            }
        ),
        json.dumps(
            {
                "block_time": "1970-01-01T00:00:03.000Z",
                "block_number": 101,
                "events": [],
            }
        ),
    ]
    batch = parse_hyperliquid_node_fills_by_block(
        lines,
        market,
        source_cursor="atomic-batch",
        received_time_ms=9_000,
    )
    original = batch.trades[0]
    new_trade = replace(
        original,
        exchange_time_ms=1_600,
        received_time_ms=9_001,
        source_event_id="0xnew",
        economic_event_id="0xnew",
        sequence_id="00000000000000000100:00000000:00000001",
    )
    conflicting = replace(original, qty=3.0)
    failed_batch = replace(batch, trades=(new_trade, conflicting))
    archive = OutcomeTradeArchive(tmp_path / "atomic-historical.sqlite")
    assert archive.append_trade(original) is True

    with pytest.raises(ValueError, match="conflicting outcome trade evidence"):
        archive_historical_batch(
            archive,
            failed_batch,
            collector_session="atomic-batch",
        )

    assert archive.load_market_metadata(market.venue, market.market_id) == []
    assert archive.load_trades(
        market.venue,
        market.market_id,
        market.yes_asset.asset_id,
        start_ms=0,
        end_ms=10_000,
    ) == [original]
    assert archive.load_verified_coverage(
        market.venue,
        market.market_id,
        market.yes_asset.asset_id,
        start_ms=0,
        end_ms=10_000,
    ) == []


def test_hyperliquid_coverage_excludes_partial_boundary_seconds():
    market = hyperliquid.normalize_market(fixture("hyperliquid_price_binary.json"))
    lines = [
        json.dumps(
            {
                "block_time": "1970-01-01T00:00:01.500Z",
                "block_number": 100,
                "events": [],
            }
        ),
        json.dumps(
            {
                "block_time": "1970-01-01T00:00:02.500Z",
                "block_number": 101,
                "events": [],
            }
        ),
    ]

    batch = parse_hyperliquid_node_fills_by_block(
        lines,
        market,
        source_cursor="partial-boundaries",
    )

    assert batch.coverage_by_asset == {}


def test_hyperliquid_mirrored_assets_share_historical_economic_event_id():
    market = hyperliquid.normalize_market(fixture("hyperliquid_price_binary.json"))
    yes_fill = {
        "coin": market.yes_asset.market_data_symbol,
        "px": "0.335",
        "sz": "2",
        "side": "A",
        "time": 1_500,
        "hash": "0xmirror",
        "tid": 7,
    }
    no_fill = {
        **yes_fill,
        "coin": market.no_asset.market_data_symbol,
        "px": "0.665",
        "side": "B",
        "tid": 8,
    }
    lines = [
        json.dumps(
            {
                "block_time": "1970-01-01T00:00:01.000Z",
                "block_number": 100,
                "events": [["0xparticipant", yes_fill], ["0xparticipant", no_fill]],
            }
        ),
        json.dumps(
            {
                "block_time": "1970-01-01T00:00:02.000Z",
                "block_number": 101,
                "events": [],
            }
        ),
    ]

    batch = parse_hyperliquid_node_fills_by_block(
        lines,
        market,
        source_cursor="mirrored-assets",
    )

    assert len(batch.trades) == 2
    assert batch.trades[0].source_event_id != batch.trades[1].source_event_id
    assert batch.trades[0].economic_event_id == batch.trades[1].economic_event_id


def test_hyperliquid_block_gap_fails_before_claiming_coverage():
    market = hyperliquid.normalize_market(fixture("hyperliquid_price_binary.json"))
    lines = [
        json.dumps(
            {
                "block_time": "1970-01-01T00:00:01.500Z",
                "block_number": 100,
                "events": [],
            }
        ),
        json.dumps(
            {
                "block_time": "1970-01-01T00:00:02.500Z",
                "block_number": 102,
                "events": [],
            }
        ),
    ]

    with pytest.raises(ValueError, match="block gap"):
        parse_hyperliquid_node_fills_by_block(
            lines,
            market,
            source_cursor="incomplete",
        )


def test_hyperliquid_fill_times_must_follow_block_event_order():
    market = hyperliquid.normalize_market(fixture("hyperliquid_price_binary.json"))

    def fill(*, time_ms: int, tid: int) -> dict:
        return {
            "coin": market.yes_asset.market_data_symbol,
            "px": "0.335",
            "sz": "2",
            "side": "B",
            "time": time_ms,
            "hash": f"0xtrade{tid}",
            "tid": tid,
        }

    lines = [
        json.dumps(
            {
                "block_time": "1970-01-01T00:00:01.000Z",
                "block_number": 100,
                "events": [["0xfirst", fill(time_ms=2_500, tid=7)]],
            }
        ),
        json.dumps(
            {
                "block_time": "1970-01-01T00:00:02.000Z",
                "block_number": 101,
                "events": [["0xsecond", fill(time_ms=1_500, tid=8)]],
            }
        ),
    ]

    with pytest.raises(ValueError, match="fill times contradict block/event order"):
        parse_hyperliquid_node_fills_by_block(
            lines,
            market,
            source_cursor="contradictory-fill-order",
        )


def test_hyperliquid_settlement_is_archived_separately_and_never_becomes_a_candle(
    tmp_path,
):
    market = hyperliquid.normalize_market(fixture("hyperliquid_price_binary.json"))

    def settlement(
        *,
        coin: str,
        price: str,
        qty: str,
        tid: int,
        oid: int,
    ) -> dict:
        return {
            "coin": coin,
            "crossed": True,
            "dir": "Settlement",
            "fee": "0.0",
            "feeToken": "USDC",
            "hash": "0xsettlement",
            "oid": oid,
            "px": price,
            "side": "A",
            "startPosition": qty,
            "sz": qty,
            "tid": tid,
            "time": 2_500,
        }

    lines = [
        json.dumps(
            {
                "block_time": "1970-01-01T00:00:02.500Z",
                "block_number": 100,
                "events": [
                    [
                        "0xyesholder",
                        settlement(
                            coin=market.yes_asset.market_data_symbol,
                            price="1.0",
                            qty="2.0",
                            tid=1,
                            oid=11,
                        ),
                    ],
                    [
                        "0xnoholder",
                        settlement(
                            coin=market.no_asset.market_data_symbol,
                            price="0.0",
                            qty="3.0",
                            tid=2,
                            oid=12,
                        ),
                    ],
                ],
            }
        )
    ]

    batch = parse_hyperliquid_node_fills_by_block(
        lines,
        market,
        source_cursor="node_fills_by_block/hourly/19700101/0.lz4",
        received_time_ms=9_000,
    )

    assert batch.trades == ()
    assert len(batch.settlements) == 1
    evidence = batch.settlements[0]
    assert evidence.yes_fraction == 1.0
    assert evidence.observed_yes_qty == 2.0
    assert evidence.observed_no_qty == 3.0
    assert evidence.collateral_payout == 2.0
    assert evidence.fee == 0.0
    assert evidence.fee_asset == "USDC"

    archive = OutcomeTradeArchive(tmp_path / "settlements.sqlite")
    assert archive_historical_batch(
        archive,
        batch,
        collector_session="historical-settlement",
    ) == (0, 0)
    assert archive.load_trades(
        market.venue,
        market.market_id,
        market.yes_asset.asset_id,
        start_ms=0,
        end_ms=10_000,
    ) == []
    assert archive.load_settlements(market.venue, market.market_id) == [evidence]


def test_polymarket_v1_and_v2_logs_share_one_ordered_trade_contract():
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))
    contract = "0xexchange"
    v1_buy_yes = {
        "address": contract,
        "blockNumber": 10,
        "transactionIndex": 2,
        "logIndex": 3,
        "blockTimeMs": 10_500,
        "transactionHash": "0xv1",
        "args": {
            "orderHash": "0xorder1",
            "maker": "0xmaker1",
            "taker": "0xtaker1",
            "makerAssetId": "0",
            "takerAssetId": market.yes_asset.asset_id,
            "makerAmountFilled": 335_000,
            "takerAmountFilled": 1_000_000,
            "fee": 0,
        },
    }
    v2_sell_no = {
        "address": contract,
        "blockNumber": 10,
        "transactionIndex": 2,
        "logIndex": 4,
        "blockTimeMs": 10_500,
        "transactionHash": "0xv2",
        "args": {
            "orderHash": "0xorder2",
            "maker": "0xmaker2",
            "taker": "0xtaker2",
            "side": 1,
            "tokenId": market.no_asset.asset_id,
            "makerAmountFilled": 2_000_000,
            "takerAmountFilled": 1_330_000,
            "fee": 0,
            "builder": "0x0",
            "metadata": "0x0",
        },
    }
    taker_aggregate = {
        **v2_sell_no,
        "logIndex": 5,
        "transactionHash": "0xaggregate",
        "args": {**v2_sell_no["args"], "taker": contract},
    }

    batch = parse_polymarket_order_filled_logs(
        [taker_aggregate, v2_sell_no, v1_buy_yes],
        market,
        source_cursor="polygon:10:10",
        coverage=VerifiedCoverage(10_000, 11_000),
        received_time_ms=20_000,
    )

    assert len(batch.trades) == 2
    assert [trade.native_side for trade in batch.trades] == [
        OutcomeOrderSide.BUY,
        OutcomeOrderSide.SELL,
    ]
    assert [trade.outcome for trade in batch.trades] == [OutcomeSide.YES, OutcomeSide.NO]
    assert [trade.native_price for trade in batch.trades] == pytest.approx([0.335, 0.665])
    assert [trade.canonical_yes_price for trade in batch.trades] == pytest.approx([0.335, 0.335])
    assert [trade.sequence_id for trade in batch.trades] == [
        "00000000000000000010:00000002:00000003",
        "00000000000000000010:00000002:00000004",
    ]


def test_polymarket_duplicate_log_position_is_rejected():
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))
    event = {
        "address": "0xexchange",
        "blockNumber": 10,
        "transactionIndex": 2,
        "logIndex": 3,
        "blockTimeMs": 10_500,
        "transactionHash": "0xv2",
        "args": {
            "taker": "0xtaker",
            "side": 0,
            "tokenId": market.yes_asset.asset_id,
            "makerAmountFilled": 335_000,
            "takerAmountFilled": 1_000_000,
        },
    }

    with pytest.raises(ValueError, match="duplicate Polymarket log position"):
        parse_polymarket_order_filled_logs(
            [event, event],
            market,
            source_cursor="polygon:10:10",
            coverage=VerifiedCoverage(10_000, 11_000),
        )


def test_polymarket_log_outside_proven_time_range_is_rejected():
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))
    event = {
        "address": "0xexchange",
        "blockNumber": 10,
        "transactionIndex": 2,
        "logIndex": 3,
        "blockTimeMs": 9_999,
        "transactionHash": "0xv2",
        "args": {
            "taker": "0xtaker",
            "side": 0,
            "tokenId": market.yes_asset.asset_id,
            "makerAmountFilled": 335_000,
            "takerAmountFilled": 1_000_000,
        },
    }

    with pytest.raises(ValueError, match="outside proven coverage"):
        parse_polymarket_order_filled_logs(
            [event],
            market,
            source_cursor="polygon:10:10",
            coverage=VerifiedCoverage(10_000, 11_000),
        )
