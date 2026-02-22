import os
import json
import types
import asyncio
import numpy as np
import pandas as pd
import pytest

import ccxt.async_support as ccxt
from ohlcv_utils import (
    deduplicate_rows,
    ensure_millis_df,
    fill_gaps_in_ohlcvs,
    attempt_gap_fix_ohlcvs,
    dump_ohlcv_data,
    load_ohlcv_data,
)
from utils import to_ccxt_exchange_id, load_markets, utc_ms


def test_to_ccxt_exchange_id(monkeypatch):
    # Make ccxt.exchanges predictable
    monkeypatch.setattr(
        ccxt,
        "exchanges",
        ["binance", "binanceusdm", "kucoin", "kucoinfutures", "bybit"],
        raising=False,
    )

    assert to_ccxt_exchange_id("binance") == "binanceusdm"
    assert to_ccxt_exchange_id("kucoin") == "kucoinfutures"
    # Unchanged when no matching futures suffix exists
    assert to_ccxt_exchange_id("bybit") == "bybit"
    # Already ccxt ID stays as-is
    assert to_ccxt_exchange_id("binanceusdm") == "binanceusdm"


def test_deduplicate_rows():
    arr = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [5.0, 6.0]])
    out = deduplicate_rows(arr)
    assert out.shape == (3, 2)
    assert (out[0] == np.array([1.0, 2.0])).all()
    assert (out[1] == np.array([3.0, 4.0])).all()
    assert (out[2] == np.array([5.0, 6.0])).all()


def test_ensure_millis_seconds_to_ms():
    df = pd.DataFrame(
        {
            "timestamp": [1_600_000_000],
            "open": [1],
            "high": [1],
            "low": [1],
            "close": [1],
            "volume": [1],
        }
    )
    out = ensure_millis_df(df.copy())
    assert out["timestamp"].iloc[0] == 1_600_000_000_000


def test_ensure_millis_micros_to_ms():
    df = pd.DataFrame(
        {
            "timestamp": [1_600_000_000_000_000],
            "open": [1],
            "high": [1],
            "low": [1],
            "close": [1],
            "volume": [1],
        }
    )
    out = ensure_millis_df(df.copy())
    assert out["timestamp"].iloc[0] == 1_600_000_000_000


def test_ensure_millis_ms_unchanged():
    df = pd.DataFrame(
        {
            "timestamp": [1_600_000_000_000],
            "open": [1],
            "high": [1],
            "low": [1],
            "close": [1],
            "volume": [1],
        }
    )
    out = ensure_millis_df(df.copy())
    assert out["timestamp"].iloc[0] == 1_600_000_000_000


def _make_df_with_gap():
    # timestamps: 0, 60000, 180000 (missing 120000)
    ts = [0, 60_000, 180_000]
    base = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [1.0, 2.0, 4.0],
            "high": [1.5, 2.5, 4.5],
            "low": [0.5, 1.5, 3.5],
            "close": [1.1, 2.1, 4.1],
            "volume": [10.0, 20.0, 40.0],
        }
    )
    return base


def test_fill_gaps_in_ohlcvs():
    df = _make_df_with_gap()
    filled = fill_gaps_in_ohlcvs(df)
    # Should now have 4 rows: 0, 60000, 120000, 180000
    assert len(filled) == 4
    assert filled["timestamp"].tolist() == [0, 60_000, 120_000, 180_000]
    # The inserted row's O/H/L should be filled with previous close
    assert filled.loc[2, "open"] == filled.loc[1, "close"]


def test_attempt_gap_fix_ohlcvs_small_gap():
    df = _make_df_with_gap()
    fixed = attempt_gap_fix_ohlcvs(df, symbol="TEST", verbose=False)
    assert len(fixed) == 4
    assert fixed["timestamp"].tolist() == [0, 60_000, 120_000, 180_000]


def test_attempt_gap_fix_ohlcvs_raises_on_huge_gap():
    # create a huge gap of 13 hours
    gap_ms = 13 * 60 * 60 * 1000
    df = pd.DataFrame(
        {
            "timestamp": [0, gap_ms],
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [0.0, 0.0],
        }
    )
    with pytest.raises(Exception):
        attempt_gap_fix_ohlcvs(df, symbol="HUGE", verbose=False)


def test_dump_and_load_ohlcvs_roundtrip(tmp_path):
    fp = tmp_path / "ohlcv.npy"
    df = pd.DataFrame(
        {
            "timestamp": [0, 60_000, 120_000, 120_000],  # duplicate last row
            "open": [1.0, 2.0, 3.0, 3.0],
            "high": [1.0, 2.0, 3.0, 3.0],
            "low": [1.0, 2.0, 3.0, 3.0],
            "close": [1.0, 2.0, 3.0, 3.0],
            "volume": [10.0, 20.0, 30.0, 30.0],
        }
    )
    dump_ohlcv_data(df, str(fp))
    loaded = load_ohlcv_data(str(fp))
    # Deduplicated so should be 3 rows
    assert loaded.shape[0] == 3
    assert loaded["timestamp"].tolist() == [0, 60_000, 120_000]


@pytest.mark.asyncio
async def test_load_markets_fetch_and_cache(tmp_path, monkeypatch):
    import utils

    # Make ccxt.exchanges predictable
    monkeypatch.setattr(ccxt, "exchanges", ["binance", "binanceusdm"], raising=False)

    # Dummy exchange class to inject
    class DummyCCXT:
        def __init__(self, config=None):
            self.options = {}

        async def load_markets(self, reload=False, params=None):
            return {"BTC/USDT:USDT": {"id": "BTCUSDT", "swap": True}}

        async def close(self):
            return

    # Point ccxt.binanceusdm to our dummy
    monkeypatch.setattr(ccxt, "binanceusdm", DummyCCXT, raising=False)

    # Control time so cache-aging checks are predictable
    monkeypatch.setattr(utils, "utc_ms", lambda: 1_000_000_000_000)
    # First call: force cache miss by reporting old mod time
    monkeypatch.setattr(utils, "get_file_mod_ms", lambda p: 0)

    # Run inside temporary directory so "caches/..." is isolated
    monkeypatch.chdir(tmp_path)

    markets = await load_markets("binance")
    assert "BTC/USDT:USDT" in markets

    # Ensure cache file exists (uses non-normalized exchange name for cache path)
    cache_fp = tmp_path / "caches" / "binance" / "markets.json"
    assert cache_fp.exists()

    # Second call: make cache fresh to test cache path
    monkeypatch.setattr(
        utils, "get_file_mod_ms", lambda p: utils.utc_ms() - 1000
    )
    markets2 = await load_markets("binance")
    assert markets2 == markets


@pytest.mark.asyncio
async def test_compute_exchange_volume_ratios_simple(monkeypatch):
    from hlcv_preparation import compute_exchange_volume_ratios

    # Build stub OMs for two exchanges
    class StubOM:
        def __init__(self, df):
            self._df = df

        def has_coin(self, coin):
            return True

        def update_date_range(self, start, end):
            pass

        async def get_ohlcvs(self, coin):
            return self._df.copy()

    # Day 0 in ms is 0.. so use two bars
    df_a = pd.DataFrame(
        {
            "timestamp": [0, 60_000],
            "open": [1, 1],
            "high": [1, 1],
            "low": [1, 1],
            "close": [1, 1],
            "volume": [10.0, 20.0],  # sum = 30
        }
    )
    df_b = pd.DataFrame(
        {
            "timestamp": [0, 60_000],
            "open": [1, 1],
            "high": [1, 1],
            "low": [1, 1],
            "close": [1, 1],
            "volume": [30.0, 30.0],  # sum = 60
        }
    )

    om_dict = {"exA": StubOM(df_a), "exB": StubOM(df_b)}
    ratios = await compute_exchange_volume_ratios(
        exchanges=["exA", "exB"],
        coins=["BTC"],
        start_date="1970-01-01",
        end_date="1970-01-02",
        om_dict=om_dict,
    )
    # Pair order follows (ex0, ex1) as constructed; exA < exB
    assert ("exA", "exB") in ratios
    assert ratios[("exA", "exB")] == pytest.approx(0.5)
