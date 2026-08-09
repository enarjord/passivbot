import json
import os
import types

import pytest

import procedures


class _DummyCC:
    def __init__(self, exchange_name: str, first_ts: int):
        self.exchange_name = exchange_name
        self.first_ts = first_ts
        self.fetch_calls = []

    async def fetch_ohlcv(self, symbol, since=None, timeframe=None, limit=None, **kwargs):
        self.fetch_calls.append(
            {
                "symbol": symbol,
                "since": since,
                "timeframe": timeframe,
                "limit": limit,
                **kwargs,
            }
        )
        if self.exchange_name == "binanceusdm" and symbol == "HYPE/USDT:USDT":
            return [[self.first_ts, 1.0, 1.0, 1.0, 1.0, 1.0]]
        return []

    async def close(self):
        return None


def _mark_first_ohlcv_cache_current(cache_dir):
    (cache_dir / "first_ohlcv_timestamps_unified.version").write_text(
        str(procedures.FIRST_OHLCV_TIMESTAMPS_CACHE_VERSION), encoding="utf-8"
    )


def _write_first_ohlcv_provenance(cache_dir, data):
    (cache_dir / "first_ohlcv_timestamps_unified_exchange_specific_symbols.json").write_text(
        json.dumps(data), encoding="utf-8"
    )


@pytest.mark.asyncio
async def test_get_first_timestamps_preserves_exact_market_identifiers(monkeypatch, tmp_path):
    cache_dir = tmp_path / "caches"
    cache_dir.mkdir()
    _mark_first_ohlcv_cache_current(cache_dir)
    exact_identifiers = {
        "1000ABC": 1710115200000,
        "ABC/USDT:USDT": 1710115260000,
        "bitget::ABCUSDT": 1710115320000,
    }
    (cache_dir / "first_ohlcv_timestamps_unified.json").write_text(
        json.dumps(exact_identifiers), encoding="utf-8"
    )
    (cache_dir / "first_ohlcv_timestamps_unified_exchange_specific.json").write_text(
        json.dumps({key: {"bitget": value} for key, value in exact_identifiers.items()}),
        encoding="utf-8",
    )
    _write_first_ohlcv_provenance(
        cache_dir,
        {key: {"bitget": key} for key in exact_identifiers},
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(procedures, "make_get_filepath", lambda p: str(tmp_path / p))
    monkeypatch.setattr(
        procedures,
        "load_ccxt_instance",
        lambda *_args, **_kwargs: pytest.fail("valid exact cache keys must not fetch markets"),
    )
    monkeypatch.setattr(procedures, "coin_to_symbol", lambda coin, _exchange: coin)

    result = await procedures.get_first_timestamps_unified(list(exact_identifiers))

    assert result == exact_identifiers


@pytest.mark.asyncio
async def test_get_first_timestamps_returns_only_requested_cache_keys(monkeypatch, tmp_path):
    cache_dir = tmp_path / "caches"
    cache_dir.mkdir()
    _mark_first_ohlcv_cache_current(cache_dir)
    (cache_dir / "first_ohlcv_timestamps_unified.json").write_text(
        json.dumps(
            {
                "BTC": 1710115200000,
                "bybit::ABCUSDT": 1710115260000,
            }
        ),
        encoding="utf-8",
    )
    (cache_dir / "first_ohlcv_timestamps_unified_exchange_specific.json").write_text(
        json.dumps({"BTC": {"binanceusdm": 1710115200000}}), encoding="utf-8"
    )
    _write_first_ohlcv_provenance(
        cache_dir, {"BTC": {"binanceusdm": "BTC/USDT:USDT"}}
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(procedures, "make_get_filepath", lambda p: str(tmp_path / p))
    monkeypatch.setattr(
        procedures,
        "load_ccxt_instance",
        lambda *_args, **_kwargs: pytest.fail("valid requested cache key must not fetch markets"),
    )
    monkeypatch.setattr(
        procedures, "coin_to_symbol", lambda _coin, _exchange: "BTC/USDT:USDT"
    )

    result = await procedures.get_first_timestamps_unified(["BTC"])

    assert result == {"BTC": 1710115200000}


@pytest.mark.asyncio
async def test_exchange_timestamp_cache_rebind_refetches_current_market(monkeypatch, tmp_path):
    cache_dir = tmp_path / "caches"
    cache_dir.mkdir()
    _mark_first_ohlcv_cache_current(cache_dir)
    old_ts = 1609459200000
    new_ts = 1710115200000
    (cache_dir / "first_ohlcv_timestamps_unified.json").write_text(
        json.dumps({"ABC": old_ts}), encoding="utf-8"
    )
    (cache_dir / "first_ohlcv_timestamps_unified_exchange_specific.json").write_text(
        json.dumps({"ABC": {"bybit": old_ts}}), encoding="utf-8"
    )
    _write_first_ohlcv_provenance(
        cache_dir, {"ABC": {"bybit": "ABC/USDT:USDT"}}
    )

    class ReboundClient(_DummyCC):
        async def fetch_ohlcv(self, symbol, since=None, timeframe=None, limit=None, **kwargs):
            self.fetch_calls.append({"symbol": symbol, "since": since, "timeframe": timeframe})
            return [[new_ts, 1.0, 1.0, 1.0, 1.0, 1.0]]

    client = ReboundClient("bybit", new_ts)

    async def fake_load_markets(_exchange):
        return {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(procedures, "make_get_filepath", lambda p: str(tmp_path / p))
    monkeypatch.setattr(procedures, "coin_to_symbol", lambda *_args: "1000ABC/USDT:USDT")
    monkeypatch.setattr(procedures, "load_ccxt_instance", lambda _exchange: client)
    monkeypatch.setattr(procedures, "load_markets", fake_load_markets)

    result = await procedures.get_first_timestamps_unified(["ABC"], exchange="bybit")

    assert result == {"ABC": new_ts}
    assert client.fetch_calls
    provenance = json.loads(
        (cache_dir / "first_ohlcv_timestamps_unified_exchange_specific_symbols.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["ABC"]["bybit"] == "1000ABC/USDT:USDT"


@pytest.mark.asyncio
async def test_get_first_timestamps_unified_refreshes_zero_cached_entries(monkeypatch, tmp_path):
    cache_dir = tmp_path / "caches"
    cache_dir.mkdir()
    _mark_first_ohlcv_cache_current(cache_dir)
    unified_cache = cache_dir / "first_ohlcv_timestamps_unified.json"
    exchange_cache = cache_dir / "first_ohlcv_timestamps_unified_exchange_specific.json"
    unified_cache.write_text(json.dumps({"HYPE": 0.0}), encoding="utf-8")
    exchange_cache.write_text(
        json.dumps(
            {
                "HYPE": {
                    "binanceusdm": 0.0,
                    "bitget": 0.0,
                    "bybit": 0.0,
                    "gateio": 0.0,
                    "hyperliquid": 0.0,
                    "okx": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(procedures, "make_get_filepath", lambda p: str(tmp_path / p))
    monkeypatch.setattr(procedures, "coin_to_symbol", lambda coin, ex: "HYPE/USDT:USDT")
    monkeypatch.setattr(
        procedures,
        "load_ccxt_instance",
        lambda ex_name: _DummyCC(ex_name, 1710115200000),
    )

    async def _fake_load_markets(_exchange_name):
        return {}

    monkeypatch.setattr(procedures, "load_markets", _fake_load_markets)

    result = await procedures.get_first_timestamps_unified(["HYPE"])

    assert result["HYPE"] == 1710115200000
    assert json.loads(unified_cache.read_text(encoding="utf-8"))["HYPE"] == 1710115200000
    assert (
        json.loads(exchange_cache.read_text(encoding="utf-8"))["HYPE"]["binanceusdm"]
        == 1710115200000
    )


@pytest.mark.asyncio
async def test_get_first_timestamps_ignores_unversioned_legacy_cache(monkeypatch, tmp_path):
    cache_dir = tmp_path / "caches"
    cache_dir.mkdir()
    unified_cache = cache_dir / "first_ohlcv_timestamps_unified.json"
    exchange_cache = cache_dir / "first_ohlcv_timestamps_unified_exchange_specific.json"
    unified_cache.write_text(json.dumps({"HYPE": 1609459200000}), encoding="utf-8")
    exchange_cache.write_text(
        json.dumps({"HYPE": {"binanceusdm": 1609459200000}}), encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(procedures, "make_get_filepath", lambda p: str(tmp_path / p))
    monkeypatch.setattr(procedures, "coin_to_symbol", lambda coin, ex: "HYPE/USDT:USDT")
    monkeypatch.setattr(
        procedures,
        "load_ccxt_instance",
        lambda ex_name: _DummyCC(ex_name, 1710115200000),
    )

    async def _fake_load_markets(_exchange_name):
        return {}

    monkeypatch.setattr(procedures, "load_markets", _fake_load_markets)

    result = await procedures.get_first_timestamps_unified(["HYPE"])

    assert result["HYPE"] == 1710115200000
    assert (
        cache_dir / "first_ohlcv_timestamps_unified.version"
    ).read_text(encoding="utf-8") == str(
        procedures.FIRST_OHLCV_TIMESTAMPS_CACHE_VERSION
    )


@pytest.mark.asyncio
async def test_get_first_timestamps_scoped_identifier_skips_other_exchanges(monkeypatch, tmp_path):
    cache_dir = tmp_path / "caches"
    cache_dir.mkdir()
    _mark_first_ohlcv_cache_current(cache_dir)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(procedures, "make_get_filepath", lambda p: str(tmp_path / p))
    monkeypatch.setattr(
        procedures,
        "load_ccxt_instance",
        lambda ex_name: _DummyCC(ex_name, 1710115200000),
    )

    async def _fake_load_markets(_exchange_name):
        return {}

    monkeypatch.setattr(procedures, "load_markets", _fake_load_markets)
    resolved_on = []

    def _coin_to_symbol(coin, exchange):
        resolved_on.append((coin, exchange))
        return "ABC/USDT:USDT"

    monkeypatch.setattr(procedures, "coin_to_symbol", _coin_to_symbol)

    result = await procedures.get_first_timestamps_unified(["bybit::ABCUSDT"])

    assert resolved_on == [("bybit::ABCUSDT", "bybit")]
    assert result["bybit::ABCUSDT"] == 0.0


@pytest.mark.asyncio
async def test_exchange_specific_first_timestamps_skips_other_exchanges(monkeypatch, tmp_path):
    cache_dir = tmp_path / "caches"
    cache_dir.mkdir()
    _mark_first_ohlcv_cache_current(cache_dir)
    unified_cache = cache_dir / "first_ohlcv_timestamps_unified.json"
    unified_cache.write_text(json.dumps({"ABC": 1710115200000}), encoding="utf-8")
    exchange_cache = cache_dir / "first_ohlcv_timestamps_unified_exchange_specific.json"
    exchange_cache.write_text(
        json.dumps({"ABC": {"binanceusdm": 1710115200000, "bybit": 0.0}}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(procedures, "make_get_filepath", lambda p: str(tmp_path / p))
    loaded_exchanges = []

    def _load_ccxt(exchange_name):
        loaded_exchanges.append(exchange_name)
        return _DummyCC(exchange_name, 1710115200000)

    monkeypatch.setattr(procedures, "load_ccxt_instance", _load_ccxt)

    async def _load_markets(_exchange):
        return {}

    monkeypatch.setattr(procedures, "load_markets", _load_markets)

    def _coin_to_symbol(coin, exchange):
        assert exchange == "bybit"
        return "ABC/USDT:USDT"

    monkeypatch.setattr(procedures, "coin_to_symbol", _coin_to_symbol)

    result = await procedures.get_first_timestamps_unified(["ABC"], exchange="bybit")

    assert loaded_exchanges == ["bybit"]
    assert result == {"ABC": 0.0}
    assert json.loads(unified_cache.read_text(encoding="utf-8"))["ABC"] == 1710115200000
    assert json.loads(exchange_cache.read_text(encoding="utf-8"))["ABC"] == {
        "binanceusdm": 1710115200000,
        "bybit": 0.0,
    }


@pytest.mark.asyncio
async def test_exchange_specific_first_timestamps_supports_non_default_exchange(
    monkeypatch, tmp_path
):
    cache_dir = tmp_path / "caches"
    cache_dir.mkdir()
    _mark_first_ohlcv_cache_current(cache_dir)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(procedures, "make_get_filepath", lambda p: str(tmp_path / p))
    loaded_exchanges = []
    loaded_clients = []

    def _load_ccxt(exchange_name):
        loaded_exchanges.append(exchange_name)
        client = _DummyCC(exchange_name, 1710115200000)
        loaded_clients.append(client)
        return client

    monkeypatch.setattr(procedures, "load_ccxt_instance", _load_ccxt)

    async def _load_markets(_exchange):
        return {}

    monkeypatch.setattr(procedures, "load_markets", _load_markets)

    def _coin_to_symbol(_coin, exchange):
        assert exchange == "kucoin"
        return "ABC/USDT:USDT"

    monkeypatch.setattr(procedures, "coin_to_symbol", _coin_to_symbol)

    result = await procedures.get_first_timestamps_unified(
        ["ABC"], exchange="kucoinfutures"
    )

    assert loaded_exchanges == ["kucoinfutures"]
    assert result == {"ABC": 0.0}
    assert loaded_clients[0].fetch_calls == [
        {
            "symbol": "ABC/USDT:USDT",
            "since": 1,
            "timeframe": "1d",
            "limit": 1,
        }
    ]
    exchange_cache = cache_dir / "first_ohlcv_timestamps_unified_exchange_specific.json"
    assert json.loads(exchange_cache.read_text(encoding="utf-8"))["ABC"] == {
        "kucoin": 0.0
    }


@pytest.mark.asyncio
async def test_qualified_dynamic_exchange_is_discovered_without_exchange_argument(
    monkeypatch, tmp_path
):
    cache_dir = tmp_path / "caches"
    cache_dir.mkdir()
    _mark_first_ohlcv_cache_current(cache_dir)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(procedures, "make_get_filepath", lambda p: str(tmp_path / p))
    loaded_clients = {}

    def _load_ccxt(exchange_name):
        client = _DummyCC(exchange_name, 1710115200000)
        loaded_clients[exchange_name] = client
        return client

    monkeypatch.setattr(procedures, "load_ccxt_instance", _load_ccxt)

    async def _load_markets(_exchange):
        return {}

    monkeypatch.setattr(procedures, "load_markets", _load_markets)
    resolved_on = []

    def _coin_to_symbol(coin, exchange):
        resolved_on.append((coin, exchange))
        return "ABC/USDT:USDT"

    monkeypatch.setattr(procedures, "coin_to_symbol", _coin_to_symbol)

    result = await procedures.get_first_timestamps_unified(["kucoin::ABCUSDT"])

    assert resolved_on == [("kucoin::ABCUSDT", "kucoin")]
    assert result == {"kucoin::ABCUSDT": 0.0}
    assert loaded_clients["kucoinfutures"].fetch_calls == [
        {
            "symbol": "ABC/USDT:USDT",
            "since": 1,
            "timeframe": "1d",
            "limit": 1,
        }
    ]
    exchange_cache = cache_dir / "first_ohlcv_timestamps_unified_exchange_specific.json"
    assert json.loads(exchange_cache.read_text(encoding="utf-8"))["kucoin::ABCUSDT"] == {
        "kucoin": 0.0
    }


@pytest.mark.asyncio
async def test_configured_dynamic_exchange_scopes_unqualified_discovery(monkeypatch, tmp_path):
    cache_dir = tmp_path / "caches"
    cache_dir.mkdir()
    _mark_first_ohlcv_cache_current(cache_dir)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(procedures, "make_get_filepath", lambda p: str(tmp_path / p))
    loaded_clients = {}

    def _load_ccxt(exchange_name):
        client = _DummyCC(exchange_name, 1710115200000)
        loaded_clients[exchange_name] = client
        return client

    monkeypatch.setattr(procedures, "load_ccxt_instance", _load_ccxt)

    async def _load_markets(_exchange):
        return {}

    monkeypatch.setattr(procedures, "load_markets", _load_markets)

    resolved_on = []

    def _coin_to_symbol(coin, exchange):
        resolved_on.append((coin, exchange))
        return "EDGE/USDT:USDT"

    monkeypatch.setattr(procedures, "coin_to_symbol", _coin_to_symbol)

    result = await procedures.get_first_timestamps_unified(
        ["EDGE"], exchanges=["kucoin"]
    )

    assert result == {"EDGE": 0.0}
    assert set(loaded_clients) == {"kucoinfutures"}
    assert resolved_on == [("EDGE", "kucoin")]
    assert loaded_clients["kucoinfutures"].fetch_calls == [
        {
            "symbol": "EDGE/USDT:USDT",
            "since": 1,
            "timeframe": "1d",
            "limit": 1,
        }
    ]


@pytest.mark.asyncio
async def test_exchange_specific_first_timestamps_uses_native_bitunix_client(
    monkeypatch, tmp_path
):
    import custom_endpoint_overrides
    import exchanges.bitunix as bitunix

    cache_dir = tmp_path / "caches"
    cache_dir.mkdir()
    _mark_first_ohlcv_cache_current(cache_dir)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(procedures, "make_get_filepath", lambda p: str(tmp_path / p))
    dummy = _DummyCC("bitunix", 1710115200000)
    observed_configs = []

    def _bitunix_client(config):
        observed_configs.append(config)
        return dummy

    monkeypatch.setattr(bitunix, "BitunixClient", _bitunix_client)
    monkeypatch.setattr(
        bitunix,
        "apply_bitunix_endpoint_override",
        lambda config, _override: config,
    )
    monkeypatch.setattr(
        custom_endpoint_overrides,
        "resolve_custom_endpoint_override",
        lambda _exchange: None,
    )
    monkeypatch.setattr(
        procedures,
        "load_ccxt_instance",
        lambda *_args, **_kwargs: pytest.fail("Bitunix must not use a CCXT client"),
    )

    async def _load_markets(exchange):
        assert exchange == "bitunix"
        return {}

    monkeypatch.setattr(procedures, "load_markets", _load_markets)

    def _coin_to_symbol(_coin, exchange):
        assert exchange == "bitunix"
        return "ABC/USDT:USDT"

    monkeypatch.setattr(procedures, "coin_to_symbol", _coin_to_symbol)

    result = await procedures.get_first_timestamps_unified(["ABC"], exchange="bitunix")

    assert result == {"ABC": 0.0}
    assert observed_configs == [
        {"enableRateLimit": True, "timeout": 60_000, "wsEnabled": False}
    ]
