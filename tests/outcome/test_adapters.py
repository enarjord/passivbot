from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from outcome.adapters import hyperliquid, polymarket
from outcome.models import MarketLifecycle, OutcomeOrderSide, OutcomeSide, OutcomeVenue
from outcome.rust_runner import normalized_market_to_rust_spec


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "outcome"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_hyperliquid_price_binary_market_maps_official_asset_encodings():
    market = hyperliquid.normalize_market(load_fixture("hyperliquid_price_binary.json"))

    assert market.venue is OutcomeVenue.HYPERLIQUID
    assert market.market_id == "913"
    assert market.yes_asset.asset_id == "+9130"
    assert market.yes_asset.market_data_symbol == "#9130"
    assert market.yes_asset.order_asset_id == "100009130"
    assert market.no_asset.asset_id == "+9131"
    assert market.no_asset.market_data_symbol == "#9131"
    assert market.no_asset.order_asset_id == "100009131"
    assert market.lifecycle.trading_open_time_ms == 1784872800000
    assert market.lifecycle.trading_close_time_ms == 1784959200000
    assert market.lifecycle.scheduled_event_time_ms == 1784959200000
    assert market.capabilities.complementary_books_merged is True
    assert market.price_grid.kind == "significant_figures"
    assert market.price_grid.max_significant_figures == 5
    assert market.qty_step == 1.0
    assert market.min_order_qty == 1.0
    assert market.min_order_notional == 10.0


@pytest.mark.parametrize(
    "description",
    [
        "",
        "other",
        "class:priceBucket|underlying:BTC|expiry:20260725-0600|targetPrice:1|period:1d",
        "class:priceBinary|underlying:BTC|expiry:bad|targetPrice:1|period:1d",
        "class:priceBinary|underlying:BTC|expiry:20260725-0600|targetPrice:1|period:1d|extra:x",
    ],
)
def test_hyperliquid_rejects_non_price_binary_or_malformed_descriptions(description):
    payload = load_fixture("hyperliquid_price_binary.json")
    payload["description"] = description
    with pytest.raises(ValueError):
        hyperliquid.normalize_market(payload)


def test_hyperliquid_no_trade_maps_to_canonical_yes_price_and_exposure():
    market = hyperliquid.normalize_market(load_fixture("hyperliquid_price_binary.json"))
    trade = hyperliquid.normalize_trade(
        {
            "coin": "#9131",
            "side": "B",
            "px": "0.665",
            "sz": "4.2",
            "time": 1784950000123,
            "tid": 123456,
            "hash": "0xabc",
            "users": ["0xbuyer", "0xseller"],
        },
        market,
        received_time_ms=1784950000200,
    )

    assert trade.outcome is OutcomeSide.NO
    assert trade.native_side is OutcomeOrderSide.BUY
    assert trade.canonical_yes_price == pytest.approx(0.335)
    assert trade.canonical_exposure_delta == pytest.approx(-4.2)
    assert trade.deduplication_key is not None
    assert trade.economic_deduplication_key is not None

    mirrored_yes = hyperliquid.normalize_trade(
        {
            "coin": "#9130",
            "side": "A",
            "px": "0.335",
            "sz": "4.2",
            "time": 1784950000123,
            "tid": 654321,
            "hash": "0xabc",
            "users": ["0xseller", "0xbuyer"],
        },
        market,
        received_time_ms=1784950000200,
    )
    assert mirrored_yes.source_event_id != trade.source_event_id
    assert mirrored_yes.economic_event_id == trade.economic_event_id


def test_polymarket_binary_market_preserves_tokens_lifecycle_and_fee_curve():
    market = polymarket.normalize_market(load_fixture("polymarket_binary.json"))

    assert market.venue is OutcomeVenue.POLYMARKET
    assert market.yes_asset.label == "Yes"
    assert market.no_asset.label == "No"
    assert market.price_grid.fixed_step == 0.01
    assert market.min_order_qty == 5.0
    assert market.lifecycle.accepting_orders is True
    assert market.lifecycle.trading_close_time_ms is None
    assert market.lifecycle.resolution_time_ms is None
    assert market.fee_metadata.formula == "polymarket_probability_curve"
    assert market.fee_metadata.parameters["taker_only"] is True
    assert market.native_metadata["yes_outcome_index"] == 0
    assert market.native_metadata["no_outcome_index"] == 1


def test_polymarket_rust_spec_requires_explicit_qty_step_and_uses_tick_bounds():
    market = polymarket.normalize_market(load_fixture("polymarket_binary.json"))
    market = replace(
        market,
        lifecycle=MarketLifecycle(
            trading_open_time_ms=1_000,
            trading_close_time_ms=5_000,
            scheduled_event_time_ms=5_000,
        ),
    )

    with pytest.raises(ValueError, match="quantity constraints"):
        normalized_market_to_rust_spec(market)

    spec = normalized_market_to_rust_spec(market, qty_step=0.01)
    assert spec["qty_step"] == 0.01
    assert spec["min_price"] == pytest.approx(0.01)
    assert spec["max_price"] == pytest.approx(0.99)


def test_hyperliquid_rust_spec_bounds_follow_significant_figure_grid():
    market = hyperliquid.normalize_market(load_fixture("hyperliquid_price_binary.json"))

    spec = normalized_market_to_rust_spec(market)

    assert spec["min_price"] == pytest.approx(1e-8)
    assert spec["max_price"] == pytest.approx(0.99999)


def test_polymarket_closed_time_is_actual_trading_close_not_scheduled_event():
    payload = load_fixture("polymarket_binary.json")
    payload["closed"] = True
    payload["acceptingOrders"] = False
    payload["closedTime"] = "2026-08-01T14:03:02Z"
    market = polymarket.normalize_market(payload)

    assert market.lifecycle.trading_close_time_ms == 1785592982000
    assert market.lifecycle.scheduled_event_time_ms == 1785499200000


def test_polymarket_arbitrary_binary_labels_use_explicit_first_outcome_reference():
    payload = load_fixture("polymarket_binary.json")
    payload["outcomes"] = '["Heroic", "TheMongolz"]'
    market = polymarket.normalize_market(payload)

    assert market.yes_asset.label == "Heroic"
    assert market.no_asset.label == "TheMongolz"
    assert market.native_metadata["canonical_reference_is_semantic_yes"] is False
    assert market.capabilities.supports_post_only is True


def test_polymarket_condition_resolution_maps_original_outcome_order_to_canonical_yes():
    payload = load_fixture("polymarket_binary.json")
    payload["outcomes"] = '["No", "Yes"]'
    market = polymarket.normalize_market(payload)
    event = {
        "blockTimeMs": 20_000,
        "transactionHash": "0xresolution",
        "logIndex": 7,
        "args": {
            "conditionId": market.market_id,
            "oracle": "0xoracle",
            "questionId": payload["questionID"],
            "outcomeSlotCount": 2,
            "payoutNumerators": [0, 1],
        },
    }

    settlement = polymarket.normalize_condition_resolution(
        event,
        market,
        received_time_ms=21_000,
    )

    assert settlement.yes_fraction == 1.0
    assert settlement.payout_unit == 1.0
    assert settlement.settlement_time_ms == 20_000
    assert settlement.capital_release_time_ms is None
    assert settlement.source_event_id == "0xresolution:7"
    assert settlement.evidence_source == "polymarket_ctf_condition_resolution"


def test_polymarket_rejects_negative_risk_and_non_binary_markets():
    payload = load_fixture("polymarket_binary.json")
    payload["negRisk"] = True
    with pytest.raises(ValueError, match="negative-risk"):
        polymarket.normalize_market(payload)

    payload = load_fixture("polymarket_binary.json")
    payload["outcomes"] = '["A", "B", "C"]'
    with pytest.raises(ValueError, match="exactly two"):
        polymarket.normalize_market(payload)


def test_polymarket_requires_explicit_collateral_when_gamma_omits_it():
    payload = load_fixture("polymarket_binary.json")
    payload.pop("denominationToken")
    with pytest.raises(ValueError, match="collateral identity"):
        polymarket.normalize_market(payload)

    market = polymarket.normalize_market(payload, quote_asset="pUSD")
    assert market.quote_asset == "pUSD"


def test_polymarket_public_no_trade_maps_price_but_does_not_invent_unique_id():
    market = polymarket.normalize_market(load_fixture("polymarket_binary.json"))
    trade = polymarket.normalize_public_trade(
        {
            "proxyWallet": "0xabc",
            "side": "SELL",
            "asset": market.no_asset.asset_id,
            "conditionId": market.market_id,
            "size": 3.0,
            "price": 0.67,
            "timestamp": 1784950000,
            "outcome": "No",
            "transactionHash": "0xtransaction",
        },
        market,
        received_time_ms=1784950000200,
    )

    assert trade.canonical_yes_price == pytest.approx(0.33)
    assert trade.canonical_exposure_delta == pytest.approx(3.0)
    assert trade.deduplication_key is None


def test_polymarket_market_websocket_trade_uses_milliseconds_and_preserves_missing_identity():
    market = polymarket.normalize_market(load_fixture("polymarket_binary.json"))
    trade = polymarket.normalize_market_ws_trade(
        {
            "event_type": "last_trade_price",
            "market": market.market_id,
            "asset_id": market.yes_asset.asset_id,
            "price": "0.456",
            "size": "219.217767",
            "side": "BUY",
            "timestamp": "1750428146322",
            "transaction_hash": "0xtransaction",
        },
        market,
        received_time_ms=1750428146400,
    )

    assert trade.exchange_time_ms == 1750428146322
    assert trade.canonical_yes_price == pytest.approx(0.456)
    assert trade.deduplication_key is None
