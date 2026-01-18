import os
import json
import logging
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

    # Call with "binance" - cache uses non-normalized name "binance" (not "binanceusdm")
    result = await utils.load_markets("binance")
    assert result == markets

    # Cached markets file exists (uses non-normalized exchange name for cache path)
    markets_path = os.path.join("caches", "binance", "markets.json")
    assert os.path.exists(markets_path)

    # Maps should be created
    c2s_path = os.path.join("caches", "binance", "coin_to_symbol_map.json")
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

    # Runtime helpers use caches (use non-normalized exchange name)
    assert utils.coin_to_symbol("BTC", "binance") == "BTC/USDT:USDT"
    assert utils.coin_to_symbol("SHIB", "binance") == "1000SHIB/USDT:USDT"

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


def test_coin_to_symbol_fallback_and_logging(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    ex = "hyperliquid"
    path = os.path.join("caches", ex, "coin_to_symbol_map.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({}, open(path, "w"))
    # sanitize a noisy input
    caplog.set_level(logging.INFO)
    sym = utils.coin_to_symbol("BTCUSDC", ex)
    assert sym == "BTC/USDC:USDC"
    assert any("BTCUSDC" in rec.message for rec in caplog.records)


def test_coin_to_symbol_multiple_candidates(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    ex = "binanceusdm"
    path = os.path.join("caches", ex, "coin_to_symbol_map.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"BTC": ["A", "B"]}, open(path, "w"))
    caplog.set_level(logging.INFO)
    assert utils.coin_to_symbol("BTC", ex) == "A"
    assert any("Multiple candidates" in rec.message for rec in caplog.records)


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


def test_symbol_to_coin_warns_only_once(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    utils._SYMBOL_TO_COIN_WARNINGS.clear()
    caplog.set_level(logging.WARNING)
    assert utils.symbol_to_coin("FOO/USDT:USDT") == "FOO"
    assert utils.symbol_to_coin("FOO/USDT:USDT") == "FOO"
    warnings = [rec for rec in caplog.records if "heuristics to guess coin" in rec.message]
    assert len(warnings) == 1


def test_get_quote_with_explicit_override():
    """get_quote() returns explicit quote when provided, ignoring exchange defaults."""
    # Without override - uses hardcoded defaults
    assert utils.get_quote("binance") == "USDT"
    assert utils.get_quote("hyperliquid") == "USDC"

    # With explicit override - returns the override
    assert utils.get_quote("binance", quote="USDC") == "USDC"
    assert utils.get_quote("hyperliquid", quote="USDT") == "USDT"
    assert utils.get_quote("paradex", quote="USDC") == "USDC"


def test_filter_markets_with_explicit_quote():
    """filter_markets() uses explicit quote when provided."""
    markets = {
        "BTC/USDT:USDT": {"active": True, "swap": True, "linear": True},
        "BTC/USDC:USDC": {"active": True, "swap": True, "linear": True},
        "ETH/USDT:USDT": {"active": True, "swap": True, "linear": True},
    }

    # Default for binance is USDT
    eligible, ineligible, reasons = utils.filter_markets(markets, "binance")
    assert "BTC/USDT:USDT" in eligible
    assert "BTC/USDC:USDC" in ineligible
    assert reasons["BTC/USDC:USDC"] == "wrong quote"

    # Override to USDC
    eligible, ineligible, reasons = utils.filter_markets(markets, "binance", quote="USDC")
    assert "BTC/USDC:USDC" in eligible
    assert "BTC/USDT:USDT" in ineligible
    assert reasons["BTC/USDT:USDT"] == "wrong quote"


def test_coin_to_symbol_with_explicit_quote(tmp_path, monkeypatch):
    """coin_to_symbol() uses explicit quote for fallback symbol construction."""
    monkeypatch.chdir(tmp_path)
    ex = "paradex"

    # No cache exists, so fallback is used
    # Default would be USDT, but with explicit USDC override
    sym = utils.coin_to_symbol("BTC", ex, quote="USDC")
    assert sym == "BTC/USDC:USDC"

    # Verify default still works for legacy exchanges
    sym2 = utils.coin_to_symbol("ETH", "binance")
    assert sym2 == "ETH/USDT:USDT"


def test_concurrent_write_symbol_maps(tmp_path, monkeypatch):
    """
    Multiple threads writing to symbol_to_coin_map shouldn't corrupt it.
    This tests the race condition fix for parallel bot startup.
    """
    import concurrent.futures
    import threading

    monkeypatch.chdir(tmp_path)
    os.makedirs("caches", exist_ok=True)

    # Reset the stale cleanup flag so each call triggers cleanup check
    utils._SYMBOL_MAP_STALE_CLEANUP_DONE = False

    errors = []
    results = []
    lock = threading.Lock()

    def write_maps(thread_id):
        try:
            # Each thread creates different market data
            markets = {
                f"COIN{thread_id}/USDT:USDT": {
                    "swap": True,
                    "base": f"COIN{thread_id}",
                    "baseName": f"COIN{thread_id}",
                }
            }
            result = utils.create_coin_symbol_map_cache("binanceusdm", markets, verbose=False)
            with lock:
                results.append((thread_id, result))
        except Exception as e:
            with lock:
                errors.append((thread_id, e))

    # Launch 10 concurrent writers to simulate parallel bot startup
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(write_maps, i) for i in range(10)]
        concurrent.futures.wait(futures)

    # No exceptions should have occurred
    assert not errors, f"Concurrent writes failed: {errors}"

    # Verify file is valid JSON (not corrupted)
    s2c_path = os.path.join("caches", "symbol_to_coin_map.json")
    assert os.path.exists(s2c_path), "symbol_to_coin_map.json should exist"
    with open(s2c_path) as f:
        data = json.load(f)  # Should not raise JSONDecodeError
    assert isinstance(data, dict), "symbol_to_coin_map should be a dict"

    # Verify coin_to_symbol_map is also valid
    c2s_path = os.path.join("caches", "binanceusdm", "coin_to_symbol_map.json")
    assert os.path.exists(c2s_path), "coin_to_symbol_map.json should exist"
    with open(c2s_path) as f:
        c2s_data = json.load(f)  # Should not raise JSONDecodeError
    assert isinstance(c2s_data, dict), "coin_to_symbol_map should be a dict"


def test_stale_lock_cleanup(tmp_path, monkeypatch):
    """Stale lock files should be cleaned up on first access."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("caches/binanceusdm", exist_ok=True)

    # Reset cleanup flag
    utils._SYMBOL_MAP_STALE_CLEANUP_DONE = False

    # Create a stale lock file (older than threshold)
    stale_lock = os.path.join("caches", "symbol_to_coin_map.json.lock")
    with open(stale_lock, "w") as f:
        f.write("")
    # Set mtime to 5 minutes ago (older than 180s threshold)
    old_time = time.time() - 300
    os.utime(stale_lock, (old_time, old_time))

    # Create a fresh lock file (should NOT be removed)
    fresh_lock = os.path.join("caches", "binanceusdm", "coin_to_symbol_map.json.lock")
    with open(fresh_lock, "w") as f:
        f.write("")

    # Trigger cleanup via any symbol map operation
    utils._cleanup_stale_symbol_map_locks()

    # Stale lock should be removed
    assert not os.path.exists(stale_lock), "Stale lock should be removed"
    # Fresh lock should still exist
    assert os.path.exists(fresh_lock), "Fresh lock should not be removed"
