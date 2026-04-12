from copy import deepcopy

import pytest

from ccxt_contracts import build_contract_bot, derive_market_contracts, summarize_market_snapshot
from exchanges.ccxt_bot import CCXTBot
from utils import _build_coin_symbol_maps, filter_markets


def test_derive_market_contracts_normalizes_generic_market_metadata():
    markets = {
        "BTC/USDT:USDT": {
            "id": "BTCUSDT",
            "limits": {"cost": {"min": None}, "amount": {"min": None}},
            "precision": {"amount": 0.001, "price": 0.1},
        },
        "ETH/USDT:USDT": {
            "id": "ETHUSDT",
            "limits": {"cost": {"min": 5.0}, "amount": {"min": 0.0}},
            "precision": {"amount": 0.01, "price": 0.01},
            "contractSize": 2.0,
        },
    }

    contracts = derive_market_contracts("testexchange", "USDT", markets)

    assert contracts["BTC/USDT:USDT"]["min_cost"] == 0.1
    assert contracts["BTC/USDT:USDT"]["min_qty"] == 0.001
    assert contracts["BTC/USDT:USDT"]["contract_size"] == 1
    assert contracts["ETH/USDT:USDT"]["min_cost"] == 5.0
    assert contracts["ETH/USDT:USDT"]["min_qty"] == 0.01
    assert contracts["ETH/USDT:USDT"]["contract_size"] == 2.0


def test_derive_market_contracts_applies_bitget_min_cost_floor():
    markets = {
        "XRP/USDT:USDT": {
            "id": "XRPUSDT",
            "limits": {"cost": {"min": 0.1}, "amount": {"min": 1.0}},
            "precision": {"amount": 1.0, "price": 0.0001},
            "contractSize": 1.0,
        }
    }

    contracts = derive_market_contracts("bitget", "USDT", markets)

    assert contracts["XRP/USDT:USDT"]["min_cost"] == pytest.approx(5.1)


def test_derive_market_contracts_applies_hyperliquid_specific_defaults():
    markets = {
        "HYPE/USDC:USDC": {
            "id": "HYPE",
            "limits": {"cost": {"min": None}, "amount": {"min": None}},
            "precision": {"amount": 1.0, "price": 0.01},
            "contractSize": 1.0,
            "info": {"maxLeverage": "25"},
        },
        "XYZ-TSLA/USDC:USDC": {
            "id": "XYZTSLA",
            "limits": {"cost": {"min": 10.0}, "amount": {"min": 1.0}},
            "precision": {"amount": 1.0, "price": 0.01},
            "contractSize": 1.0,
            "info": {"maxLeverage": "20", "onlyIsolated": True, "baseName": "XYZ-TSLA"},
        },
    }

    contracts = derive_market_contracts("hyperliquid", "USDC", markets)

    assert contracts["HYPE/USDC:USDC"]["min_cost"] == pytest.approx(10.1)
    assert contracts["HYPE/USDC:USDC"]["max_leverage"] == 25
    assert contracts["XYZ-TSLA/USDC:USDC"]["requires_isolated_margin"] is True
    assert contracts["XYZ-TSLA/USDC:USDC"]["margin_capability"] == "isolated_only"
    assert contracts["XYZ-TSLA/USDC:USDC"]["max_leverage"] == 10


def test_filter_markets_catches_ccxt_market_flag_changes():
    markets = {
        "BTC/USDT:USDT": {"active": True, "swap": True, "linear": True},
        "ETH/USDT:USDT": {"active": False, "swap": True, "linear": True},
        "SOL/USDT:USDT": {"active": True, "swap": False, "linear": True},
        "DOGE/USDT:USDT": {"active": True, "swap": True, "linear": False},
        "BTC/USDC:USDC": {"active": True, "swap": True, "linear": True},
    }

    eligible, ineligible, reasons = filter_markets(markets, "binance", quote="USDT")

    assert sorted(eligible) == ["BTC/USDT:USDT"]
    assert reasons["ETH/USDT:USDT"] == "not active"
    assert reasons["SOL/USDT:USDT"] == "not swap"
    assert reasons["DOGE/USDT:USDT"] == "not linear"
    assert reasons["BTC/USDC:USDC"] == "wrong quote"
    assert set(ineligible) == {
        "ETH/USDT:USDT",
        "SOL/USDT:USDT",
        "DOGE/USDT:USDT",
        "BTC/USDC:USDC",
    }


def test_filter_markets_handles_hyperliquid_zero_open_interest():
    markets = {
        "ABC/USDC:USDC": {
            "active": True,
            "swap": True,
            "linear": True,
            "info": {"openInterest": "0"},
        },
        "DEF/USDC:USDC": {
            "active": True,
            "swap": True,
            "linear": True,
            "info": {"openInterest": "123"},
        },
    }

    eligible, _, reasons = filter_markets(markets, "hyperliquid", quote="USDC")

    assert sorted(eligible) == ["DEF/USDC:USDC"]
    assert reasons["ABC/USDC:USDC"] == "ineligible on hyperliquid"


def test_coin_symbol_maps_cover_ccxt_base_variants_and_stock_perps():
    markets = {
        "1000SHIB/USDT:USDT": {
            "swap": True,
            "linear": True,
            "base": "1000SHIB",
            "id": "1000SHIBUSDT",
        },
        "XYZ-TSLA/USDC:USDC": {
            "swap": True,
            "linear": True,
            "baseName": "XYZ-TSLA",
            "id": "XYZTSLA",
        },
        "xyz:AAPL/USDC:USDC": {
            "swap": True,
            "linear": True,
            "baseName": "xyz:AAPL",
            "id": "XYZAAPL",
        },
    }

    coin_to_symbol_map, symbol_to_coin_map = _build_coin_symbol_maps(markets, "USDC")

    assert "XYZ-TSLA/USDC:USDC" in coin_to_symbol_map["TSLA"]
    assert "XYZ-TSLA/USDC:USDC" in coin_to_symbol_map["xyz:TSLA"]
    assert "xyz:AAPL/USDC:USDC" in coin_to_symbol_map["AAPL"]
    assert symbol_to_coin_map["XYZTSLA"] == "XYZ-TSLA"


def test_coin_symbol_maps_cover_namespaced_hip3_aliases():
    markets = {
        "PARA-BTCD/USDC:USDC": {
            "swap": True,
            "linear": True,
            "base": "PARA-BTCD",
            "baseName": "para:BTCD",
            "id": "180002",
            "info": {"hip3": True},
        },
        "ABCD-USA500/USDC:USDC": {
            "swap": True,
            "linear": True,
            "base": "ABCD-USA500",
            "baseName": "abcd:USA500",
            "id": "160000",
            "info": {"hip3": True},
        },
    }

    coin_to_symbol_map, symbol_to_coin_map = _build_coin_symbol_maps(markets, "USDC")

    assert "PARA-BTCD/USDC:USDC" in coin_to_symbol_map["BTCD"]
    assert "ABCD-USA500/USDC:USDC" in coin_to_symbol_map["USA500"]
    assert "PARA-BTCD/USDC:USDC" in coin_to_symbol_map["para:BTCD"]
    assert "ABCD-USA500/USDC:USDC" in coin_to_symbol_map["ABCD-USA500"]
    assert symbol_to_coin_map["BTCD"] == "para:BTCD"
    assert symbol_to_coin_map["USA500"] == "abcd:USA500"
    assert symbol_to_coin_map["180002"] == "para:BTCD"
    assert symbol_to_coin_map["160000"] == "abcd:USA500"


def test_extract_live_margin_mode_accepts_ccxt_field_renames():
    bot = build_contract_bot("testexchange")

    assert bot._extract_live_margin_mode({"marginMode": "cross"}) == "cross"
    assert bot._extract_live_margin_mode({"tradeMode": "isolated"}) == "isolated"
    assert bot._extract_live_margin_mode({"tdMode": "cross"}) == "cross"
    assert bot._extract_live_margin_mode({"mgnMode": "isolated"}) == "isolated"
    assert bot._extract_live_margin_mode({"isolated": True}) == "isolated"
    assert bot._extract_live_margin_mode({"info": {"marginType": "crossed"}}) == "cross"


def test_margin_policy_uses_margin_modes_and_live_state_consistently():
    bot = build_contract_bot("testexchange")
    bot.markets_dict = {
        "ISO/USDT:USDT": {
            "marginModes": {"cross": False, "isolated": True},
            "info": {},
        },
        "BOTH/USDT:USDT": {
            "marginModes": {"cross": True, "isolated": True},
            "info": {},
        },
    }

    isolated_policy = bot._resolve_margin_policy_for_symbol("ISO/USDT:USDT")
    assert isolated_policy["mode"] == "isolated"
    assert isolated_policy["blocked"] is True

    bot.open_orders = {"ISO/USDT:USDT": [{"id": "1"}]}
    bot._live_margin_modes = {"ISO/USDT:USDT": "isolated"}
    live_policy = bot._resolve_margin_policy_for_symbol("ISO/USDT:USDT")
    assert live_policy["mode"] == "isolated"
    assert live_policy["blocked"] is False
    assert live_policy["live_margin_mode"] == "isolated"

    both_policy = bot._resolve_margin_policy_for_symbol("BOTH/USDT:USDT")
    assert both_policy["mode"] == "cross"
    assert both_policy["capability"] == "both"


def test_market_snapshot_summary_contains_contracts_filtering_and_maps():
    markets = {
        "BTC/USDT:USDT": {
            "id": "BTCUSDT",
            "active": True,
            "swap": True,
            "linear": True,
            "limits": {"cost": {"min": 5.0}, "amount": {"min": 0.001}},
            "precision": {"amount": 0.001, "price": 0.1},
            "base": "BTC",
        }
    }

    summary = summarize_market_snapshot("binance", "USDT", deepcopy(markets))

    assert summary["eligible_symbols"] == ["BTC/USDT:USDT"]
    assert summary["contracts"]["BTC/USDT:USDT"]["id"] == "BTCUSDT"
    assert summary["coin_to_symbol_map"]["BTC"] == ["BTC/USDT:USDT"]
    assert summary["symbol_to_coin_map"]["BTCUSDT"] == "BTC"
