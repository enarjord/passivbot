from __future__ import annotations

import json
from pathlib import Path

import pytest

from outcome.adapters import polymarket
from outcome.backtest_input import build_trade_derived_ema_anchor_input
from outcome.candles import VerifiedCoverage


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "outcome"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_trade_derived_input_uses_dense_signal_but_only_positive_volume_execution():
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))
    yes = polymarket.normalize_market_ws_trade(
        {
            "event_type": "last_trade_price",
            "market": market.market_id,
            "asset_id": market.yes_asset.asset_id,
            "price": "0.40",
            "size": "2",
            "side": "BUY",
            "timestamp": "1100",
        },
        market,
        received_time_ms=1_200,
        collector_sequence=1,
    )
    no = polymarket.normalize_market_ws_trade(
        {
            "event_type": "last_trade_price",
            "market": market.market_id,
            "asset_id": market.no_asset.asset_id,
            "price": "0.55",
            "size": "3",
            "side": "SELL",
            "timestamp": "3200",
        },
        market,
        received_time_ms=3_300,
        collector_sequence=2,
    )
    payload = build_trade_derived_ema_anchor_input(
        market_spec={"market_id": market.market_id},
        trades=[yes, no],
        verified_coverage=(VerifiedCoverage(1_000, 5_000),),
        fee_schedule={"maker_rate": 0.0, "taker_rate": 0.0, "formula": "notional"},
        starting_collateral=100.0,
        strategy_params={"execution_mode": "accumulate_pairs"},
        settlement_time_ms=5_000,
        yes_fraction=1.0,
    )

    assert [candle["timestamp_ms"] for candle in payload["signal_candles"]] == [
        1_000,
        2_000,
        3_000,
        4_000,
    ]
    assert [candle["volume"] for candle in payload["signal_candles"]] == [
        2.0,
        0.0,
        3.0,
        0.0,
    ]
    assert payload["signal_candles"][2]["close"] == pytest.approx(0.45)
    assert [
        (candle["timestamp_ms"], candle["outcome"], candle["volume"])
        for candle in payload["execution_candles"]
    ] == [(1_000, "yes", 2.0), (3_000, "no", 3.0)]


def test_trade_derived_input_rejects_uncovered_fill():
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))
    trade = polymarket.normalize_market_ws_trade(
        {
            "event_type": "last_trade_price",
            "market": market.market_id,
            "asset_id": market.yes_asset.asset_id,
            "price": "0.40",
            "size": "2",
            "side": "BUY",
            "timestamp": "900",
        },
        market,
        received_time_ms=1_000,
    )

    with pytest.raises(ValueError, match="outside verified coverage"):
        build_trade_derived_ema_anchor_input(
            market_spec={"market_id": market.market_id},
            trades=[trade],
            verified_coverage=(VerifiedCoverage(1_000, 2_000),),
            fee_schedule={},
            starting_collateral=100.0,
            strategy_params={},
            settlement_time_ms=2_000,
            yes_fraction=0.0,
        )
