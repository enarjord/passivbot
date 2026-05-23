import json

import numpy as np
import pytest

from hlcvs_manifest import HlcvsManifestError, build_hlcvs_manifest, write_hlcvs_manifest
from hlcvs_override import load_hlcvs_data_override


def _base_config(cache_dir, *, mode="dataset"):
    return {
        "backtest": {
            "base_dir": "backtests",
            "compress_cache": False,
            "end_date": "2025-01-03",
            "exchanges": ["binance"],
            "gap_tolerance_ohlcvs_minutes": 120,
            "hlcvs_cache_permissive": False,
            "hlcvs_data_dir": str(cache_dir),
            "hlcvs_data_override_mode": mode,
            "start_date": "2025-01-01",
        },
        "bot": {
            "long": {"n_positions": 1, "total_wallet_exposure_limit": 1.0},
            "short": {"n_positions": 1, "total_wallet_exposure_limit": 1.0},
        },
        "live": {
            "approved_coins": {"long": ["BTC"], "short": []},
            "minimum_coin_age_days": 0,
        },
    }


def _write_dataset(cache_dir):
    cache_dir.mkdir(parents=True)
    coins = ["BTC", "ETH"]
    timestamps = np.array([1735689600000, 1735776000000, 1735862400000], dtype=np.int64)
    hlcvs = np.arange(24, dtype=np.float64).reshape(3, 2, 4)
    btc_usd_prices = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    mss = {
        "BTC": {"exchange": "binance"},
        "ETH": {"exchange": "binance"},
        "__meta__": {"btc_source_exchange": "binanceusdm"},
    }
    np.save(cache_dir / "hlcvs.npy", hlcvs)
    np.save(cache_dir / "timestamps.npy", timestamps)
    np.save(cache_dir / "btc_usd_prices.npy", btc_usd_prices)
    (cache_dir / "coins.json").write_text(json.dumps(coins))
    (cache_dir / "market_specific_settings.json").write_text(json.dumps(mss))
    manifest_config = _base_config(cache_dir)
    manifest_config["live"]["approved_coins"] = {"long": ["BTC", "ETH"], "short": ["ETH"]}
    manifest = build_hlcvs_manifest(
        config=manifest_config,
        exchange="binance",
        cache_hash="abc123",
        coins=coins,
        hlcvs=hlcvs,
        mss=mss,
        btc_usd_prices=btc_usd_prices,
        timestamps=timestamps,
        warmup_minutes=0,
        compressed=False,
    )
    write_hlcvs_manifest(cache_dir, manifest)
    return timestamps, hlcvs


def test_hlcvs_dataset_override_dataset_mode_updates_side_membership_and_effective_config(tmp_path):
    cache_dir = tmp_path / "hlcvs_data" / "custom__abc123"
    timestamps, hlcvs = _write_dataset(cache_dir)
    config = _base_config(cache_dir, mode="dataset")

    cache_dir_out, coins, out_hlcvs, mss, _results_path, btc_usd_prices, out_timestamps = (
        load_hlcvs_data_override(config, "binance")
    )

    assert cache_dir_out == cache_dir.resolve()
    assert coins == ["BTC", "ETH"]
    assert config["live"]["approved_coins"] == {"long": ["BTC", "ETH"], "short": ["ETH"]}
    assert config["backtest"]["coins"]["binance"] == ["BTC", "ETH"]
    np.testing.assert_array_equal(out_timestamps, timestamps)
    np.testing.assert_allclose(out_hlcvs, hlcvs)
    np.testing.assert_allclose(btc_usd_prices, np.array([100.0, 101.0, 102.0]))
    assert mss["__meta__"]["dataset_override"] is True


def test_hlcvs_dataset_override_intersection_mode_preserves_input_side_membership(tmp_path):
    cache_dir = tmp_path / "hlcvs_data" / "custom__abc123"
    _write_dataset(cache_dir)
    config = _base_config(cache_dir, mode="intersection")

    _cache_dir, coins, _hlcvs, _mss, _results_path, _btc, _timestamps = (
        load_hlcvs_data_override(config, "binance")
    )

    assert coins == ["BTC"]
    assert config["live"]["approved_coins"] == {"long": ["BTC"], "short": []}


def test_hlcvs_dataset_override_rejects_manifest_hash_mismatch(tmp_path):
    cache_dir = tmp_path / "hlcvs_data" / "custom__abc123"
    _write_dataset(cache_dir)
    config = _base_config(cache_dir, mode="dataset")
    np.save(cache_dir / "btc_usd_prices.npy", np.array([999.0, 998.0, 997.0]))

    with pytest.raises(HlcvsManifestError, match="hash mismatch"):
        load_hlcvs_data_override(config, "binance")
