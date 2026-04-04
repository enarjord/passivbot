import json
import os
import types

import pytest

import procedures


class _DummyCC:
    def __init__(self, exchange_name: str, first_ts: int):
        self.exchange_name = exchange_name
        self.first_ts = first_ts

    async def fetch_ohlcv(self, symbol, since=None, timeframe=None):
        if self.exchange_name == "binanceusdm" and symbol == "HYPE/USDT:USDT":
            return [[self.first_ts, 1.0, 1.0, 1.0, 1.0, 1.0]]
        return []

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_get_first_timestamps_unified_refreshes_zero_cached_entries(monkeypatch, tmp_path):
    cache_dir = tmp_path / "caches"
    cache_dir.mkdir()
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
