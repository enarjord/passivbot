import importlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

from hlcv_preparation import prepare_hlcvs, try_prepare_hlcvs_v2_local
from ohlcv_catalog import OhlcvCatalog
from ohlcv_legacy_import import resolve_legacy_symbol_dir
from ohlcv_store import month_start_ts


LEGACY_DTYPE = np.dtype(
    [
        ("ts", "int64"),
        ("o", "float32"),
        ("h", "float32"),
        ("l", "float32"),
        ("c", "float32"),
        ("bv", "float32"),
    ]
)


def _write_day(root, exchange, symbol, day, rows):
    symbol_dir = resolve_legacy_symbol_dir(root, exchange, "1m", symbol)
    symbol_dir.mkdir(parents=True, exist_ok=True)
    np.save(symbol_dir / f"{day}.npy", np.array(rows, dtype=LEGACY_DTYPE))


@pytest.mark.asyncio
async def test_try_prepare_hlcvs_v2_local_uses_local_cache(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    legacy_root = tmp_path / "caches" / "ohlcv"
    (tmp_path / "caches" / "binance").mkdir(parents=True, exist_ok=True)

    start = month_start_ts(2026, 4)
    ts = np.array([start, start + 60_000, start + 2 * 60_000], dtype=np.int64)
    _write_day(
        legacy_root,
        "binance",
        "ETH/USDT:USDT",
        "2026-04-01",
        [
            (int(ts[0]), 0.0, 101.0, 99.0, 100.0, 10.0),
            (int(ts[1]), 0.0, 102.0, 100.0, 101.0, 11.0),
            (int(ts[2]), 0.0, 103.0, 101.0, 102.0, 12.0),
        ],
    )
    _write_day(
        legacy_root,
        "binance",
        "BTC/USDT:USDT",
        "2026-04-01",
        [
            (int(ts[0]), 0.0, 50001.0, 49999.0, 50000.0, 100.0),
            (int(ts[1]), 0.0, 50011.0, 50009.0, 50010.0, 101.0),
            (int(ts[2]), 0.0, 50021.0, 50019.0, 50020.0, 102.0),
        ],
    )

    with open(tmp_path / "caches" / "binance" / "first_timestamps.json", "w", encoding="utf-8") as f:
        json.dump({"ETH": int(ts[0]), "BTC": int(ts[0])}, f)

    async def fake_load_markets(exchange, verbose=False, **kwargs):
        return {
            "ETH/USDT:USDT": {
                "base": "ETH",
                "maker": 0.0002,
                "taker": 0.00055,
                "contractSize": 1.0,
                "limits": {"cost": {"min": 0.01}, "amount": {"min": 0.001}},
                "precision": {"price": 0.1, "amount": 0.001},
            },
            "BTC/USDT:USDT": {
                "base": "BTC",
                "maker": 0.0002,
                "taker": 0.00055,
                "contractSize": 1.0,
                "limits": {"cost": {"min": 0.01}, "amount": {"min": 0.001}},
                "precision": {"price": 0.1, "amount": 0.001},
            },
        }

    async def fake_first_timestamps_unified(coins, exchange=None):
        return {coin: int(ts[0]) for coin in coins}

    monkeypatch.setattr("hlcv_preparation.load_markets", fake_load_markets)
    monkeypatch.setattr("hlcv_preparation.get_first_timestamps_unified", fake_first_timestamps_unified)

    config = {
        "backtest": {
            "start_date": "2026-04-01",
            "end_date": "2026-04-01",
            "gap_tolerance_ohlcvs_minutes": 120.0,
            "cm_debug_level": 0,
            "cm_progress_log_interval_seconds": 0.0,
        },
        "live": {
            "approved_coins": {"long": ["ETH"], "short": []},
            "minimum_coin_age_days": 0.0,
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0.0,
        },
        "bot": {"long": {}, "short": {}},
    }

    prepared = await try_prepare_hlcvs_v2_local(config, "binance")
    assert prepared is not None
    mss, timestamps, hlcvs, btc_prices = prepared
    np.testing.assert_array_equal(timestamps, np.array([ts[0]], dtype=np.int64))
    assert hlcvs.shape == (1, 1, 4)
    np.testing.assert_allclose(hlcvs[:, 0, 2], np.array([100.0]))
    np.testing.assert_allclose(btc_prices, np.array([50000.0]))
    assert mss["ETH"]["first_valid_index"] == 0
    assert mss["ETH"]["last_valid_index"] == 0
    assert mss["__meta__"]["btc_source_exchange"] == "binance"


@pytest.mark.asyncio
async def test_prepare_hlcvs_mss_prefers_local_v2_before_full_prepare(monkeypatch, tmp_path):
    import rust_utils

    prepared = (
        {
            "ETH": {"first_valid_index": 0, "last_valid_index": 0},
            "__meta__": {"btc_source_exchange": "binance"},
        },
        np.array([month_start_ts(2026, 4)], dtype=np.int64),
        np.array([[[101.0, 99.0, 100.0, 10.0]]], dtype=np.float64),
        np.array([50_000.0], dtype=np.float64),
    )

    config = {
        "backtest": {
            "base_dir": str(tmp_path / "results"),
            "start_date": "2026-04-01",
            "end_date": "2026-04-01",
            "gap_tolerance_ohlcvs_minutes": 120.0,
        },
        "live": {
            "approved_coins": {"long": ["ETH"], "short": []},
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0.0,
        },
        "bot": {"long": {}, "short": {}},
    }

    monkeypatch.setattr(rust_utils, "check_and_maybe_compile", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        rust_utils,
        "verify_loaded_runtime_extension",
        lambda *args, **kwargs: {"skipped": "test"},
    )
    sys.modules.pop("backtest", None)
    backtest = importlib.import_module("backtest")

    monkeypatch.setattr(backtest, "load_coins_hlcvs_from_cache", lambda *args, **kwargs: None)

    async def fake_try_prepare(*args, **kwargs):
        return prepared

    async def fail_prepare(*args, **kwargs):
        raise AssertionError("full prepare_hlcvs should not be called when local v2 succeeds")

    monkeypatch.setattr(backtest, "try_prepare_hlcvs_v2_local", fake_try_prepare)
    monkeypatch.setattr(backtest, "prepare_hlcvs", fail_prepare)
    monkeypatch.setattr(backtest, "save_coins_hlcvs_to_cache", lambda *args, **kwargs: None)

    coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices, timestamps = await backtest.prepare_hlcvs_mss(
        config, "binance"
    )

    assert coins == ["ETH"]
    assert cache_dir is None
    assert results_path.endswith("/binance/")
    np.testing.assert_allclose(hlcvs, prepared[2])
    np.testing.assert_allclose(btc_usd_prices, prepared[3])
    np.testing.assert_array_equal(timestamps, prepared[1])
    assert mss["ETH"]["first_valid_index"] == 0
    assert mss["ETH"]["last_valid_index"] == 0


@pytest.mark.asyncio
async def test_prepare_hlcvs_prefers_local_v2_before_legacy_prepare(monkeypatch):
    prepared = (
        {
            "ETH": {"first_valid_index": 0, "last_valid_index": 0},
            "__meta__": {"btc_source_exchange": "binance"},
        },
        np.array([month_start_ts(2026, 4)], dtype=np.int64),
        np.array([[[101.0, 99.0, 100.0, 10.0]]], dtype=np.float64),
        np.array([50_000.0], dtype=np.float64),
    )

    config = {
        "backtest": {
            "start_date": "2026-04-01",
            "end_date": "2026-04-01",
            "gap_tolerance_ohlcvs_minutes": 120.0,
        },
        "live": {
            "approved_coins": {"long": ["ETH"], "short": []},
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0.0,
        },
        "bot": {"long": {}, "short": {}},
    }

    async def fake_try_prepare(*args, **kwargs):
        return prepared

    async def fail_prepare_internal(*args, **kwargs):
        raise AssertionError("legacy prepare_hlcvs_internal should not run when local v2 succeeds")

    monkeypatch.setattr("hlcv_preparation.try_prepare_hlcvs_v2_local", fake_try_prepare)
    monkeypatch.setattr("hlcv_preparation.prepare_hlcvs_internal", fail_prepare_internal)

    mss, timestamps, hlcvs, btc_usd_prices = await prepare_hlcvs(config, "binance")

    np.testing.assert_allclose(hlcvs, prepared[2])
    np.testing.assert_allclose(btc_usd_prices, prepared[3])
    np.testing.assert_array_equal(timestamps, prepared[1])
    assert mss["ETH"]["first_valid_index"] == 0
    assert mss["ETH"]["last_valid_index"] == 0


@pytest.mark.asyncio
async def test_try_prepare_hlcvs_v2_local_fetches_missing_remote_range_into_store(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    start = month_start_ts(2026, 4)
    ts = np.array([start], dtype=np.int64)

    async def fake_load_markets(exchange, verbose=False, **kwargs):
        return {
            "ETH/USDT:USDT": {
                "base": "ETH",
                "maker": 0.0002,
                "taker": 0.00055,
                "contractSize": 1.0,
                "limits": {"cost": {"min": 0.01}, "amount": {"min": 0.001}},
                "precision": {"price": 0.1, "amount": 0.001},
            },
            "BTC/USDT:USDT": {
                "base": "BTC",
                "maker": 0.0002,
                "taker": 0.00055,
                "contractSize": 1.0,
                "limits": {"cost": {"min": 0.01}, "amount": {"min": 0.001}},
                "precision": {"price": 0.1, "amount": 0.001},
            },
        }

    async def fake_first_timestamps_unified(coins, exchange=None):
        return {coin: int(ts[0]) for coin in coins}

    async def fake_get_ohlcvs(self, coin, start_date=None, end_date=None):
        if coin == "ETH":
            closes = np.array([100.0], dtype=np.float64)
        elif coin == "BTC":
            closes = np.array([50_000.0], dtype=np.float64)
        else:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
        return pd.DataFrame(
            {
                "timestamp": ts,
                "open": closes,
                "high": closes + 1.0,
                "low": closes - 1.0,
                "close": closes,
                "volume": np.array([10.0], dtype=np.float64),
            }
        )

    monkeypatch.setattr("hlcv_preparation.load_markets", fake_load_markets)
    monkeypatch.setattr("hlcv_preparation.get_first_timestamps_unified", fake_first_timestamps_unified)
    monkeypatch.setattr("hlcv_preparation.HLCVManager.get_ohlcvs", fake_get_ohlcvs)

    config = {
        "backtest": {
            "start_date": "2026-04-01",
            "end_date": "2026-04-01",
            "gap_tolerance_ohlcvs_minutes": 120.0,
            "cm_debug_level": 0,
            "cm_progress_log_interval_seconds": 0.0,
        },
        "live": {
            "approved_coins": {"long": ["ETH"], "short": []},
            "minimum_coin_age_days": 0.0,
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0.0,
        },
        "bot": {"long": {}, "short": {}},
    }

    prepared = await try_prepare_hlcvs_v2_local(config, "binance")
    assert prepared is not None
    mss, timestamps, hlcvs, btc_prices = prepared

    np.testing.assert_array_equal(timestamps, ts)
    assert hlcvs.shape == (1, 1, 4)
    np.testing.assert_allclose(hlcvs[:, 0, 2], np.array([100.0]))
    np.testing.assert_allclose(btc_prices, np.array([50_000.0]))
    assert mss["ETH"]["first_valid_index"] == 0
    assert mss["ETH"]["last_valid_index"] == 0

    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    attempts = catalog.list_fetch_attempts(
        "binance", "1m", "ETH/USDT:USDT", int(ts[0]), int(ts[0])
    )
    assert len(attempts) == 1
    assert attempts[0].outcome == "ok"


@pytest.mark.asyncio
async def test_try_prepare_hlcvs_v2_local_persists_persistent_cm_gap_after_empty_fetch(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    start = month_start_ts(2026, 4)
    ts = np.array([start], dtype=np.int64)

    async def fake_load_markets(exchange, verbose=False, **kwargs):
        return {
            "ETH/USDT:USDT": {
                "base": "ETH",
                "maker": 0.0002,
                "taker": 0.00055,
                "contractSize": 1.0,
                "limits": {"cost": {"min": 0.01}, "amount": {"min": 0.001}},
                "precision": {"price": 0.1, "amount": 0.001},
            }
        }

    async def fake_first_timestamps_unified(coins, exchange=None):
        return {coin: int(ts[0]) for coin in coins}

    class FakeCM:
        def get_gap_summary(self, symbol):
            return {
                "gaps": [
                    {
                        "start_ts": int(ts[0]),
                        "end_ts": int(ts[0]),
                        "retry_count": 3,
                        "reason": "no_archive",
                        "persistent": True,
                    }
                ]
            }

    async def fake_get_ohlcvs(self, coin, start_date=None, end_date=None):
        self.cm = FakeCM()
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    monkeypatch.setattr("hlcv_preparation.load_markets", fake_load_markets)
    monkeypatch.setattr("hlcv_preparation.get_first_timestamps_unified", fake_first_timestamps_unified)
    monkeypatch.setattr("hlcv_preparation.HLCVManager.get_ohlcvs", fake_get_ohlcvs)

    config = {
        "backtest": {
            "start_date": "2026-04-01",
            "end_date": "2026-04-01",
            "gap_tolerance_ohlcvs_minutes": 120.0,
            "cm_debug_level": 0,
            "cm_progress_log_interval_seconds": 0.0,
        },
        "live": {
            "approved_coins": {"long": ["ETH"], "short": []},
            "minimum_coin_age_days": 0.0,
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0.0,
        },
        "bot": {"long": {}, "short": {}},
    }

    prepared = await try_prepare_hlcvs_v2_local(config, "binance")
    assert prepared is None

    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    gaps = catalog.get_persistent_gaps("binance", "1m", "ETH/USDT:USDT", int(ts[0]), int(ts[0]))
    assert len(gaps) == 1
    assert gaps[0].reason == "no_archive"
    attempts = catalog.list_fetch_attempts(
        "binance", "1m", "ETH/USDT:USDT", int(ts[0]), int(ts[0])
    )
    assert len(attempts) == 1
    assert attempts[0].outcome == "empty"

    async def fail_get_ohlcvs(self, coin, start_date=None, end_date=None):
        raise AssertionError("persistent v2 gap should block a second remote fetch attempt")

    monkeypatch.setattr("hlcv_preparation.HLCVManager.get_ohlcvs", fail_get_ohlcvs)
    second = await try_prepare_hlcvs_v2_local(config, "binance")
    assert second is None
