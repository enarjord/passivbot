"""Round-trip test for the persistent hlcvs cache: save_coins_hlcvs_to_cache
followed by load_coins_hlcvs_from_cache must return the identical arrays.

Guards the verified-arrays reuse in load_coins_hlcvs_from_cache (the loader
consumes the arrays decompressed by manifest verification instead of paying a
second decompression).
"""

import numpy as np
import pytest

import backtest as bt


def _config(base_dir: str, compress: bool) -> dict:
    return {
        "backtest": {
            "base_dir": base_dir,
            "compress_cache": compress,
            "end_date": "2025-01-02",
            "exchanges": ["binance"],
            "gap_tolerance_ohlcvs_minutes": 120,
            "ohlcv_source_dir": None,
            "start_date": "2025-01-01",
        },
        "bot": {
            "long": {"n_positions": 1, "total_wallet_exposure_limit": 1.0},
            "short": {"n_positions": 1, "total_wallet_exposure_limit": 1.0},
        },
        "live": {
            "approved_coins": {"long": ["BTC"], "short": ["BTC"]},
            "minimum_coin_age_days": 0,
        },
    }


@pytest.mark.parametrize("compress", [True, False])
def test_cache_save_load_roundtrip(tmp_path, monkeypatch, compress):
    monkeypatch.setattr(bt, "HLCVS_CACHE_ROOT", tmp_path / "hlcvs_data")
    config = _config(str(tmp_path / "backtests"), compress)

    coins = ["BTC"]
    hlcvs = np.arange(12, dtype=np.float64).reshape(3, 1, 4)
    timestamps = np.array(
        [1735689600000, 1735689660000, 1735689720000], dtype=np.int64
    )
    btc_usd_prices = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    mss = {
        "BTC": {"exchange": "binance", "first_valid_index": 0, "last_valid_index": 2},
        "__meta__": {"btc_source_exchange": "binanceusdm"},
    }

    saved_dir = bt.save_coins_hlcvs_to_cache(
        config,
        coins,
        hlcvs,
        "binance",
        mss,
        btc_usd_prices,
        timestamps=timestamps,
        warmup_minutes=7,
    )
    assert saved_dir is not None

    result = bt.load_coins_hlcvs_from_cache(config, "binance", warmup_minutes=7)
    assert result is not None
    cache_dir, loaded_coins, loaded_hlcvs, loaded_mss, _results_path, loaded_btc, loaded_ts = result

    assert cache_dir == saved_dir
    assert loaded_coins == coins
    assert loaded_mss["BTC"]["exchange"] == "binance"
    np.testing.assert_array_equal(loaded_hlcvs, hlcvs)
    np.testing.assert_array_equal(loaded_ts, timestamps)
    np.testing.assert_array_equal(loaded_btc, btc_usd_prices)

    # Insufficient cached warmup must still invalidate the cache.
    assert bt.load_coins_hlcvs_from_cache(config, "binance", warmup_minutes=8) is None
