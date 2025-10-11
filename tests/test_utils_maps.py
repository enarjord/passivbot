import os
import json
import time
import pytest

import utils


def make_dummy_exchange_class(markets):
    class DummyCCXT:
        def __init__(self, config=None):
            self.options = {}

        async def load_markets(self, reload):
            return markets

        async def close(self):
            return None

    return DummyCCXT


@pytest.mark.asyncio
async def test_load_markets_fetch_and_cache_creates_maps(tmp_path, monkeypatch):
    # Work inside an isolated temp directory
    monkeypatch.chdir(tmp_path)

    # Prepare dummy markets for a USDT-quoted futures exchange
    markets = {
        "BTC/USDT:USDT": {"swap": True, "baseName": "BTC", "base": "BTC"},
        "1000SHIB/USDT:USDT": {"swap": True, "baseName": "1000SHIB", "base": "1000SHIB"},
        "FOO/USDT:USDT": {"swap": True, "base": "FOO"},  # no baseName
        "ETH/USDC:USDC": {"swap": True, "base": "ETH"},  # different quote -> ignored for USDT
        "SPOT/USDT": {"swap": False, "base": "SPOT"},  # not swap
    }

    # Stub ccxt to return the dummy markets for binanceusdm
    monkeypatch.setattr(utils.ccxt, "exchanges", ["binanceusdm"], raising=False)
    monkeypatch.setattr(utils.ccxt, "binanceusdm", make_dummy_exchange_class(markets), raising=False)

    # Call with "binance" to exercise normalize_exchange_name -> "binanceusdm"
    result = await utils.load_markets("binance")
    assert result == markets

    # Cached markets file exists
    markets_path = os.path.join("caches", "binanceusdm", "markets.json")
    assert os.path.exists(markets_path)

    # Maps should be created
    c2s_path = os.path.join("caches", "binanceusdm", "coin_to_symbol_map.json")
    s2c_path = os.path.join("caches", "symbol_to_coin_map.json")
    assert os.path.exists(c2s_path)
    assert os.path.exists(s2c_path)

    c2s = json.load(open(c2s_path))
    s2c = json.load(open(s2c_path))

    # BTC should map uniquely
    assert set(c2s["BTC"]) == {"BTC/USDT:USDT"}
    # SHIB should be derived from "1000SHIB"
    assert set(c2s["SHIB"]) == {"1000SHIB/USDT:USDT"}
    # FOO comes from base without baseName
    assert set(c2s["FOO"]) == {"FOO/USDT:USDT"}

    # symbol_to_coin should resolve "1000SHIB" to "SHIB"
    assert s2c["1000SHIB/USDT:USDT"] == "SHIB"

    # Runtime helpers use caches
    assert utils.coin_to_symbol("BTC", "binanceusdm") == "BTC/USDT:USDT"
    assert utils.coin_to_symbol("SHIB", "binanceusdm") == "1000SHIB/USDT:USDT"

    # Heuristic for hyperliquid-style "kSHIB"
    assert utils.symbol_to_coin("kSHIB/USDT:USDT") == "SHIB"


@pytest.mark.asyncio
async def test_load_markets_uses_fresh_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ex = "binanceusdm"

    # Prepare a fresh cached markets file
    markets = {
        "BTC/USDT:USDT": {"swap": True, "baseName": "BTC", "base": "BTC"},
        "FOO/USDT:USDT": {"swap": True, "base": "FOO"},
    }
    markets_path = os.path.join("caches", ex, "markets.json")
    os.makedirs(os.path.dirname(markets_path), exist_ok=True)
    json.dump(markets, open(markets_path, "w"))

    # Ensure cache is considered fresh by controlling utc_ms
    fresh_now = utils.get_file_mod_ms(markets_path) + 100.0
    monkeypatch.setattr(utils, "utc_ms", lambda: fresh_now, raising=False)

    # Should read from cache (no ccxt stub needed) and also populate maps
    result = await utils.load_markets(ex)
    assert result == markets

    c2s_path = os.path.join("caches", ex, "coin_to_symbol_map.json")
    s2c_path = os.path.join("caches", "symbol_to_coin_map.json")
    assert os.path.exists(c2s_path)
    assert os.path.exists(s2c_path)

    assert utils.coin_to_symbol("BTC", ex) == "BTC/USDT:USDT"
    assert utils.symbol_to_coin("FOO/USDT:USDT") == "FOO"


def test_coin_to_symbol_in_memory_reload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ex = "binanceusdm"
    path = os.path.join("caches", ex, "coin_to_symbol_map.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Initial mapping
    json.dump({"BTC": ["BTC/USDT:USDT"]}, open(path, "w"))
    # Prime cache
    assert utils.coin_to_symbol("BTC", ex) == "BTC/USDT:USDT"

    # Modify on disk and bump mtime
    time.sleep(0.01)  # ensure mtime can change across filesystems
    json.dump({"BTC": ["BTCX/USDT:USDT"]}, open(path, "w"))
    os.utime(path, None)

    # Should reload and reflect new value
    assert utils.coin_to_symbol("BTC", ex) == "BTCX/USDT:USDT"


def test_symbol_to_coin_in_memory_reload_and_heuristics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s2c_path = os.path.join("caches", "symbol_to_coin_map.json")
    os.makedirs(os.path.dirname(s2c_path), exist_ok=True)

    # No mapping -> use heuristics
    if os.path.exists(s2c_path):
        os.remove(s2c_path)
    assert utils.symbol_to_coin("kSHIB/USDT:USDT") == "SHIB"

    # Add mapping and ensure it takes precedence over heuristics
    json.dump({"kSHIB/USDT:USDT": "SHIBBIE"}, open(s2c_path, "w"))
    os.utime(s2c_path, None)
    assert utils.symbol_to_coin("kSHIB/USDT:USDT") == "SHIBBIE"

    # Update mapping -> should reload
    time.sleep(0.01)
    json.dump({"kSHIB/USDT:USDT": "SHIB"}, open(s2c_path, "w"))
    os.utime(s2c_path, None)
    assert utils.symbol_to_coin("kSHIB/USDT:USDT") == "SHIB"
