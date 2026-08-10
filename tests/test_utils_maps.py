import os
import json
import logging
import time
import pytest

import utils


@pytest.fixture(autouse=True)
def clear_utils_caches():
    """Clear global in-memory caches before each test to ensure isolation."""
    utils._COIN_TO_SYMBOL_CACHE.clear()
    utils._SYMBOL_TO_COIN_CACHE.clear()
    utils._SYMBOL_TO_COIN_CACHE["map"] = None
    utils._SYMBOL_TO_COIN_CACHE["mtime_ns"] = None
    utils._SYMBOL_TO_COIN_CACHE["size"] = None
    utils._SYMBOL_TO_COIN_WARNINGS.clear()
    utils._COIN_TO_SYMBOL_FALLBACKS.clear()
    utils._SYMBOL_MAP_STALE_CLEANUP_DONE = False
    yield
    # Cleanup after test as well
    utils._COIN_TO_SYMBOL_CACHE.clear()
    utils._SYMBOL_TO_COIN_CACHE.clear()
    utils._SYMBOL_TO_COIN_CACHE["map"] = None
    utils._SYMBOL_TO_COIN_CACHE["mtime_ns"] = None
    utils._SYMBOL_TO_COIN_CACHE["size"] = None


def make_dummy_exchange_class(markets):
    class DummyCCXT:
        def __init__(self, config=None):
            self.options = {}

        async def load_markets(self, reload):
            return markets

        async def close(self):
            return None

    return DummyCCXT


def test_load_ccxt_instance_defaults_okx_to_swap_only_markets(monkeypatch):
    class DummyOKX:
        def __init__(self, config=None):
            self.config = config or {}
            self.options = {}

    monkeypatch.setattr(utils.ccxt, "exchanges", ["okx"], raising=False)
    monkeypatch.setattr(utils.ccxt, "okx", DummyOKX, raising=False)
    monkeypatch.setattr(utils, "resolve_custom_endpoint_override", lambda _exchange: None)

    cc = utils.load_ccxt_instance("okx")

    assert cc.options["defaultType"] == "swap"
    assert cc.options["fetchMarkets"] == {"types": ["swap"]}


def test_defx_exchange_qualified_identifier_is_recognized():
    assert utils.split_exchange_qualified_market_identifier("defx::ABCUSDC") == (
        "defx",
        "ABCUSDC",
    )


@pytest.mark.parametrize(
    "identifier", ["BTC-USDT-SWAP", "HOTUSDTM", "EDGEUSDTM", "XUSDT", "XUSDTM"]
)
def test_native_market_id_is_exact(identifier):
    assert utils.looks_like_exact_market_identifier(identifier)


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
    # Exact bases and CCXT symbols remain lossless aliases.
    assert c2s["1000SHIB"] == ["1000SHIB/USDT:USDT"]
    assert c2s["1000SHIB/USDT:USDT"] == ["1000SHIB/USDT:USDT"]
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
async def test_load_markets_uses_native_bitunix_client_on_cold_cache(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    markets = {
        "BTC/USDT:USDT": {
            "swap": True,
            "baseName": "BTC",
            "base": "BTC",
        }
    }
    observed = {}

    class DummyBitunixClient:
        def __init__(self, config):
            observed["config"] = config

        async def load_markets(self, reload):
            observed["reload"] = reload
            return markets

        async def close(self):
            observed["closed"] = True

    import exchanges.bitunix as bitunix

    def unexpected_ccxt(*_args, **_kwargs):
        raise AssertionError("Bitunix cold start must not use CCXT")

    monkeypatch.setattr(bitunix, "BitunixClient", DummyBitunixClient)
    monkeypatch.setattr(
        utils,
        "resolve_custom_endpoint_override",
        lambda _exchange: None,
    )
    monkeypatch.setattr(
        utils,
        "load_ccxt_instance",
        unexpected_ccxt,
    )

    result = await utils.load_markets("bitunix")

    assert result == markets
    assert observed == {
        "config": {
            "enableRateLimit": True,
            "timeout": 60_000,
            "wsEnabled": False,
        },
        "reload": True,
        "closed": True,
    }
    assert os.path.exists("caches/bitunix/markets.json")


@pytest.mark.asyncio
async def test_load_markets_uses_fresh_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ex = "binance"

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
    ex = "binance"
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
    caplog.set_level(logging.INFO)
    sym = utils.coin_to_symbol("BTC", ex)
    assert sym == "BTC/USDC:USDC"
    assert any("BTC" in rec.message for rec in caplog.records)


@pytest.mark.parametrize("identifier", ["bitget::1000ABCUSDT", "1000ABCUSDT"])
def test_exact_identifier_without_market_map_fails_closed(identifier, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(utils.UnknownMarketIdentifier, match="is unavailable on bitget"):
        utils.coin_to_symbol(identifier, "bitget")


def test_unknown_namespaced_hip3_identifier_fails_closed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert utils.looks_like_exact_market_identifier("xyz:UNKNOWN")
    with pytest.raises(utils.UnknownMarketIdentifier, match="is unavailable on hyperliquid"):
        utils.coin_to_symbol("xyz:UNKNOWN", "hyperliquid")


def test_coin_to_symbol_rejects_multiple_candidates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ex = "binance"
    path = os.path.join("caches", ex, "coin_to_symbol_map.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"BTC": ["A", "B"]}, open(path, "w"))
    with pytest.raises(utils.AmbiguousMarketIdentifier, match="matches \['A', 'B'\]"):
        utils.coin_to_symbol("BTC", ex)


@pytest.mark.parametrize(
    ("symbol", "kwargs", "expected"),
    [
        ("SHIB/USDT:USDT", {"exchange": "bitget"}, ("SHIB", 1)),
        ("1000SHIB/USDT:USDT", {"exchange": "binance"}, ("SHIB", 1000)),
        ("SHIB1000/USDT:USDT", {"exchange": "bybit"}, ("SHIB", 1000)),
        ("SHIB100000/USDT:USDT", {"exchange": "bybit"}, ("SHIB", 100000)),
        (
            "KSHIB/USDC:USDC",
            {"exchange": "hyperliquid", "market": {"baseName": "kSHIB"}},
            ("SHIB", 1000),
        ),
        (
            "KAVA/USDC:USDC",
            {"exchange": "hyperliquid", "market": {"baseName": "KAVA"}},
            ("KAVA", 1),
        ),
        ("LUNA2/USDT:USDT", {"exchange": "binance"}, ("LUNA2", 1)),
        ("1INCH/USDT:USDT", {"exchange": "binance"}, ("1INCH", 1)),
        ("NDX100/USDT:USDT", {"exchange": "bitget"}, ("NDX100", 1)),
        ("NAS100/USDT:USDT", {"exchange": "gate"}, ("NAS100", 1)),
        (
            "XYZ-XYZ100/USDC:USDC",
            {"exchange": "hyperliquid", "market": {"info": {"hip3": True}}},
            ("XYZ-XYZ100", 1),
        ),
    ],
)
def test_market_denomination_identity(symbol, kwargs, expected):
    assert utils.market_denomination_identity(symbol, **kwargs) == expected


def test_collision_aware_maps_preserve_exact_market_identifiers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ex = "testexchange"
    markets = {
        "ABC/USDT:USDT": {
            "swap": True,
            "linear": True,
            "base": "ABC",
            "baseName": "ABC",
            "id": "ABCUSDT",
        },
        "1000ABC/USDT:USDT": {
            "swap": True,
            "linear": True,
            "base": "1000ABC",
            "baseName": "1000ABC",
            "id": "1000ABCUSDT",
        },
    }

    c2s, s2c = utils._build_coin_symbol_maps(markets, "USDT", exchange="bitget")
    reverse_c2s, reverse_s2c = utils._build_coin_symbol_maps(
        dict(reversed(list(markets.items()))), "USDT", exchange="bitget"
    )
    assert c2s == reverse_c2s
    assert s2c == reverse_s2c
    assert c2s["ABC"] == ["1000ABC/USDT:USDT", "ABC/USDT:USDT"]
    assert c2s["1000ABC"] == ["1000ABC/USDT:USDT"]
    assert c2s["ABCUSDT"] == ["ABC/USDT:USDT"]
    assert c2s["1000ABCUSDT"] == ["1000ABC/USDT:USDT"]
    assert c2s["ABC/USDT:USDT"] == ["ABC/USDT:USDT"]
    assert c2s["1000ABC/USDT:USDT"] == ["1000ABC/USDT:USDT"]

    path = os.path.join("caches", ex, "coin_to_symbol_map.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(c2s, open(path, "w"))

    assert utils.coin_to_symbol("1000ABC", ex) == "1000ABC/USDT:USDT"
    assert utils.coin_to_symbol("ABCUSDT", ex) == "ABC/USDT:USDT"
    assert utils.coin_to_symbol("1000ABCUSDT", ex) == "1000ABC/USDT:USDT"
    assert utils.coin_to_symbol("ABC/USDT:USDT", ex) == "ABC/USDT:USDT"
    assert utils.coin_to_symbol("1000ABC/USDT:USDT", ex) == "1000ABC/USDT:USDT"
    assert utils.coin_to_symbol("ABC", ex) == "ABC/USDT:USDT"


def test_plain_alias_outranks_colliding_native_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    markets = {
        "ABC/USDT:USDT": {
            "swap": True,
            "linear": True,
            "base": "ABC",
            "id": "ABCUSDT",
        },
        "1000ABC/USDT:USDT": {
            "swap": True,
            "linear": True,
            "base": "1000ABC",
            "id": "ABC",
        },
    }

    c2s, _ = utils._build_coin_symbol_maps(markets, "USDT", exchange="bitget")
    assert c2s["ABC"] == ["1000ABC/USDT:USDT", "ABC/USDT:USDT"]
    assert c2s["bitget::ABC"] == ["1000ABC/USDT:USDT"]

    path = tmp_path / "caches" / "bitget" / "coin_to_symbol_map.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(c2s))

    assert utils.coin_to_symbol("ABC", "bitget") == "ABC/USDT:USDT"
    assert utils.coin_to_symbol("bitget::ABC", "bitget") == "1000ABC/USDT:USDT"


def test_plain_coin_prefers_smallest_available_denomination(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ex = "testexchange"
    path = os.path.join("caches", ex, "coin_to_symbol_map.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(
        {
            "SHIB": [
                "SHIB100000/USDT:USDT",
                "SHIB1000/USDT:USDT",
            ]
        },
        open(path, "w"),
    )

    assert utils.coin_to_symbol("SHIB", ex) == "SHIB1000/USDT:USDT"


def test_plain_coin_rejects_duplicate_markets_at_same_denomination(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ex = "testexchange"
    path = os.path.join("caches", ex, "coin_to_symbol_map.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(
        {
            "SHIB": [
                "1000SHIB/USDT:USDT",
                "SHIB1000/USDT:USDT",
            ]
        },
        open(path, "w"),
    )

    with pytest.raises(utils.AmbiguousMarketIdentifier, match="ambiguous market identifier"):
        utils.coin_to_symbol("SHIB", ex)


def test_single_multiplier_market_keeps_legacy_convenience_alias():
    markets = {
        "1000SHIB/USDT:USDT": {
            "id": "1000SHIBUSDT",
            "swap": True,
            "linear": True,
            "base": "1000SHIB",
            "baseName": "SHIB",
        }
    }

    coin_to_symbol_map, symbol_to_coin_map = utils._build_coin_symbol_maps(markets, "USDT")

    assert coin_to_symbol_map["SHIB"] == ["1000SHIB/USDT:USDT"]
    assert coin_to_symbol_map["1000SHIB"] == ["1000SHIB/USDT:USDT"]
    assert symbol_to_coin_map["1000SHIBUSDT"] == "SHIB"


def test_numeric_ticker_suffix_is_not_fabricated_as_bitget_denomination():
    markets = {
        "NDX/USDT:USDT": {
            "id": "NDXUSDT",
            "swap": True,
            "linear": True,
            "base": "NDX",
        },
        "NDX100/USDT:USDT": {
            "id": "NDX100USDT",
            "swap": True,
            "linear": True,
            "base": "NDX100",
        },
    }

    coin_to_symbol_map, symbol_to_coin_map = utils._build_coin_symbol_maps(
        markets, "USDT", exchange="bitget"
    )

    assert coin_to_symbol_map["NDX"] == ["NDX/USDT:USDT"]
    assert coin_to_symbol_map["NDX100"] == ["NDX100/USDT:USDT"]
    assert symbol_to_coin_map["NDX100/USDT:USDT"] == "NDX100"


def test_hyperliquid_k_prefix_requires_lowercase_k_basename_metadata():
    markets = {
        "KSHIB/USDC:USDC": {
            "id": "38",
            "swap": True,
            "linear": True,
            "base": "KSHIB",
            "baseName": "kSHIB",
        },
        "KAVA/USDC:USDC": {
            "id": "KAVA",
            "swap": True,
            "linear": True,
            "base": "KAVA",
            "baseName": "KAVA",
        },
    }

    coin_to_symbol_map, symbol_to_coin_map = utils._build_coin_symbol_maps(
        markets, "USDC", exchange="hyperliquid"
    )

    assert coin_to_symbol_map["SHIB"] == ["KSHIB/USDC:USDC"]
    assert coin_to_symbol_map["KAVA"] == ["KAVA/USDC:USDC"]
    assert "AVA" not in coin_to_symbol_map
    assert symbol_to_coin_map["KAVA/USDC:USDC"] == "KAVA"


def test_inactive_multiplier_market_does_not_pollute_convenience_alias():
    markets = {
        "ABC/USDT:USDT": {
            "id": "ABCUSDT",
            "swap": True,
            "linear": True,
            "active": True,
            "base": "ABC",
        },
        "1000ABC/USDT:USDT": {
            "id": "1000ABCUSDT",
            "swap": True,
            "linear": True,
            "active": False,
            "base": "1000ABC",
        },
    }

    coin_to_symbol_map, _ = utils._build_coin_symbol_maps(markets, "USDT")

    assert coin_to_symbol_map["ABC"] == ["ABC/USDT:USDT"]
    assert coin_to_symbol_map["1000ABCUSDT"] == ["1000ABC/USDT:USDT"]
    assert coin_to_symbol_map["1000ABC/USDT:USDT"] == ["1000ABC/USDT:USDT"]


def test_map_builder_omits_ambiguous_canonical_label_deterministically():
    markets = {
        "ABC/USDC:USDC": {
            "swap": True,
            "linear": True,
            "base": "ABC",
            "id": "ABC-USDC",
        },
        "VENUE-ABC/USDC:USDC": {
            "swap": True,
            "linear": True,
            "baseName": "venue:ABC",
            "id": "12345",
            "info": {"hip3": True},
        },
    }

    c2s, s2c = utils._build_coin_symbol_maps(markets, "USDC")
    reverse_c2s, reverse_s2c = utils._build_coin_symbol_maps(
        dict(reversed(list(markets.items()))), "USDC"
    )

    assert c2s == reverse_c2s
    assert s2c == reverse_s2c
    assert c2s["ABC"] == ["ABC/USDC:USDC", "VENUE-ABC/USDC:USDC"]
    assert "ABC" not in s2c
    assert s2c["ABC-USDC"] == "ABC"
    assert s2c["12345"] == "venue:ABC"


def test_ambiguity_tombstones_survive_other_exchange_refreshes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    colliding_markets = {
        "ABC/USDC:USDC": {
            "swap": True,
            "linear": True,
            "base": "ABC",
            "id": "ABC-USDC",
        },
        "VENUE-ABC/USDC:USDC": {
            "swap": True,
            "linear": True,
            "baseName": "venue:ABC",
            "id": "12345",
            "info": {"hip3": True},
        },
    }
    unambiguous_markets = {
        "ABC/USDT:USDT": {
            "swap": True,
            "linear": True,
            "base": "ABC",
            "id": "ABCUSDT",
        }
    }

    assert utils.create_coin_symbol_map_cache(
        "hyperliquid", colliding_markets, quote="USDC", verbose=False
    )
    assert utils.create_coin_symbol_map_cache("bybit", unambiguous_markets, verbose=False)

    symbol_map = json.loads((tmp_path / "caches" / "symbol_to_coin_map.json").read_text())
    ambiguities = json.loads(
        (tmp_path / "caches" / "symbol_to_coin_ambiguities.json").read_text()
    )
    assert "ABC" not in symbol_map
    assert ambiguities["hyperliquid"] == ["ABC"]
    assert utils.coin_to_symbol("ABC", "bybit") == "ABC/USDT:USDT"
    with pytest.raises(utils.AmbiguousMarketIdentifier):
        utils.coin_to_symbol("ABC", "hyperliquid")


def test_ambiguity_tombstone_is_released_after_collision_disappears(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    colliding_markets = {
        "ABC/USDT:USDT": {
            "swap": True,
            "linear": True,
            "base": "ABC",
            "id": "ABCUSDT",
        },
        "1000ABC/USDT:USDT": {
            "swap": True,
            "linear": True,
            "base": "1000ABC",
            "id": "1000ABCUSDT",
        },
    }
    remaining_market = {"ABC/USDT:USDT": colliding_markets["ABC/USDT:USDT"]}

    assert utils.create_coin_symbol_map_cache("bitget", colliding_markets, verbose=False)
    assert "ABC" not in utils._load_symbol_to_coin_map()

    assert utils.create_coin_symbol_map_cache("bitget", remaining_market, verbose=False)
    assert utils.symbol_to_coin("ABC", verbose=False) == "ABC"
    assert json.loads(
        (tmp_path / "caches" / "symbol_to_coin_ambiguities.json").read_text()
    )["bitget"] == []


def test_exact_native_id_resolution_is_exchange_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bitget_markets = {
        "ABC/USDT:USDT": {
            "swap": True,
            "linear": True,
            "base": "ABC",
            "id": "12345",
        }
    }
    bybit_markets = {
        "OTHER/USDT:USDT": {
            "swap": True,
            "linear": True,
            "base": "OTHER",
            "id": "12345",
        }
    }

    assert utils.create_coin_symbol_map_cache("bitget", bitget_markets, verbose=False)
    assert utils.create_coin_symbol_map_cache("bybit", bybit_markets, verbose=False)

    assert utils.coin_to_symbol("12345", "bitget") == "ABC/USDT:USDT"
    assert utils.coin_to_symbol("12345", "bybit") == "OTHER/USDT:USDT"


def test_exchange_qualified_native_id_is_scoped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = os.path.join("caches", "bitget", "coin_to_symbol_map.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"ABCUSDT": ["ABC/USDT:USDT"]}, open(path, "w"))

    assert utils.coin_to_symbol("bitget::ABCUSDT", "bitget") == "ABC/USDT:USDT"
    with pytest.raises(utils.MarketIdentifierExchangeMismatch, match="targets bitget, not bybit"):
        utils.coin_to_symbol("bitget::ABCUSDT", "bybit")


@pytest.mark.parametrize("identifier", ["bitget::1000ABCUSDT", "1000ABCUSDT"])
def test_exact_identifier_miss_does_not_fall_back_to_normalized_market(
    identifier, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    path = os.path.join("caches", "bitget", "coin_to_symbol_map.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"ABC": ["ABC/USDT:USDT"]}, open(path, "w"))

    with pytest.raises(utils.UnknownMarketIdentifier, match="is unavailable on bitget"):
        utils.coin_to_symbol(identifier, "bitget")


def test_symbol_to_coin_preserves_exchange_qualified_identifier(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert utils.symbol_to_coin("bitget::ABCUSDT", verbose=False) == "bitget::ABCUSDT"


def test_namespaced_non_exchange_alias_is_not_treated_as_qualification(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    path = os.path.join("caches", "hyperliquid", "coin_to_symbol_map.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"xyz:TSLA": ["XYZ-TSLA/USDC:USDC"]}, open(path, "w"))

    assert utils.coin_to_symbol("xyz:TSLA", "hyperliquid") == "XYZ-TSLA/USDC:USDC"


def test_namespaced_full_symbol_alias_resolves_losslessly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    markets = {
        "XYZ-TSLA/USDC:USDC": {
            "id": "12345",
            "swap": True,
            "linear": True,
            "active": True,
            "base": "XYZ-TSLA",
            "baseName": "xyz:TSLA",
            "info": {"hip3": True},
        }
    }
    assert utils.create_coin_symbol_map_cache(
        "hyperliquid", markets, quote="USDC", verbose=False
    )

    assert (
        utils.coin_to_symbol("xyz:TSLA/USDC:USDC", "hyperliquid")
        == "XYZ-TSLA/USDC:USDC"
    )


def test_hip3_namespace_matching_exchange_name_is_not_qualification(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = os.path.join("caches", "hyperliquid", "coin_to_symbol_map.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"bitget:FOO": ["BITGET-FOO/USDC:USDC"]}, open(path, "w"))

    assert (
        utils.coin_to_symbol("bitget:FOO", "hyperliquid")
        == "BITGET-FOO/USDC:USDC"
    )


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


def test_filter_markets_demotes_boot_inventory_chatter_to_debug(caplog):
    markets = {
        "BTC/USDT:USDT": {"active": True, "swap": True, "linear": True},
        "SPOT/USDT": {"active": True, "swap": False, "linear": True},
        "DOGE/USDT:USDT": {"active": True, "swap": True, "linear": False},
        "BTC/USDC:USDC": {"active": True, "swap": True, "linear": True},
    }

    with caplog.at_level(logging.DEBUG):
        utils.filter_markets(markets, "binance", quote="USDT", verbose=True)

    debug_messages = {
        (record.levelname, record.message)
        for record in caplog.records
        if record.message.startswith(("not swap", "not linear", "wrong quote"))
    }
    assert ("DEBUG", "not swap: SPOT/USDT") in debug_messages
    assert ("DEBUG", "not linear: DOGE/USDT:USDT") in debug_messages
    assert ("DEBUG", "wrong quote: BTC/USDC:USDC") in debug_messages


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
