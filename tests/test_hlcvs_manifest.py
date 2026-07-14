import json

import numpy as np
import pytest

from hlcvs_manifest import (
    HlcvsManifestError,
    build_hlcvs_manifest,
    hash_logical_array,
    verify_hlcvs_manifest,
    write_hlcvs_manifest,
)


def _minimal_config():
    return {
        "backtest": {
            "end_date": "2025-01-02",
            "exchanges": ["binance", "bybit"],
            "gap_tolerance_ohlcvs_minutes": 120,
            "ohlcv_source_dir": None,
            "start_date": "2025-01-01",
        },
        "bot": {
            "long": {"n_positions": 1, "total_wallet_exposure_limit": 1.0},
            "short": {"n_positions": 1, "total_wallet_exposure_limit": 1.0},
        },
        "live": {
            "approved_coins": {"long": ["BTC", "ETH"], "short": ["BTC"]},
        },
    }


def _write_cache_files(cache_dir, *, hlcvs, timestamps, btc_usd_prices, coins, mss):
    cache_dir.mkdir(parents=True)
    np.save(cache_dir / "hlcvs.npy", hlcvs)
    np.save(cache_dir / "timestamps.npy", timestamps)
    np.save(cache_dir / "btc_usd_prices.npy", btc_usd_prices)
    (cache_dir / "coins.json").write_text(json.dumps(coins))
    (cache_dir / "market_specific_settings.json").write_text(json.dumps(mss))


def test_build_hlcvs_manifest_hashes_logical_arrays_and_side_membership(tmp_path):
    cache_dir = tmp_path / "hlcvs"
    coins = ["BTC", "ETH"]
    hlcvs = np.arange(24, dtype=np.float64).reshape(3, 2, 4)
    timestamps = np.array([1735689600000, 1735689660000, 1735689720000], dtype=np.int64)
    btc_usd_prices = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    mss = {
        "BTC": {"exchange": "binance", "first_valid_index": 0, "last_valid_index": 2},
        "ETH": {"exchange": "bybit", "first_valid_index": 0, "last_valid_index": 2},
        "__meta__": {"btc_source_exchange": "binanceusdm"},
    }
    _write_cache_files(
        cache_dir,
        hlcvs=hlcvs,
        timestamps=timestamps,
        btc_usd_prices=btc_usd_prices,
        coins=coins,
        mss=mss,
    )

    manifest = build_hlcvs_manifest(
        config=_minimal_config(),
        exchange="combined",
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

    assert manifest["files"]["hlcvs"]["sha256"] == hash_logical_array(hlcvs)
    assert manifest["btc_benchmark"]["sha256"] == hash_logical_array(btc_usd_prices)
    assert manifest["effective"]["side_membership"] == {
        "long": ["BTC", "ETH"],
        "short": ["BTC"],
    }
    assert verify_hlcvs_manifest(cache_dir)["config_hash"] == "abc123"

    changed_btc = btc_usd_prices.copy()
    changed_btc[0] += 1.0
    changed_manifest = build_hlcvs_manifest(
        config=_minimal_config(),
        exchange="combined",
        cache_hash="abc123",
        coins=coins,
        hlcvs=hlcvs,
        mss=mss,
        btc_usd_prices=changed_btc,
        timestamps=timestamps,
        warmup_minutes=0,
        compressed=False,
    )
    assert changed_manifest["btc_benchmark"]["sha256"] != manifest["btc_benchmark"]["sha256"]


def test_build_hlcvs_manifest_requires_timestamps():
    coins = ["BTC"]
    hlcvs = np.ones((2, 1, 4), dtype=np.float64)
    btc_usd_prices = np.array([100.0, 101.0], dtype=np.float64)
    mss = {"BTC": {"exchange": "binance"}, "__meta__": {"btc_source_exchange": "binanceusdm"}}

    with pytest.raises(HlcvsManifestError, match="require timestamps"):
        build_hlcvs_manifest(
            config=_minimal_config(),
            exchange="binance",
            cache_hash="abc123",
            coins=coins,
            hlcvs=hlcvs,
            mss=mss,
            btc_usd_prices=btc_usd_prices,
            timestamps=None,
            warmup_minutes=0,
            compressed=False,
        )


def test_verify_hlcvs_manifest_rejects_modified_file(tmp_path):
    cache_dir = tmp_path / "hlcvs"
    coins = ["BTC"]
    hlcvs = np.ones((2, 1, 4), dtype=np.float64)
    timestamps = np.array([1735689600000, 1735689660000], dtype=np.int64)
    btc_usd_prices = np.array([100.0, 101.0], dtype=np.float64)
    mss = {"BTC": {"exchange": "binance"}, "__meta__": {"btc_source_exchange": "binanceusdm"}}
    _write_cache_files(
        cache_dir,
        hlcvs=hlcvs,
        timestamps=timestamps,
        btc_usd_prices=btc_usd_prices,
        coins=coins,
        mss=mss,
    )
    manifest = build_hlcvs_manifest(
        config=_minimal_config(),
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
    tampered = hlcvs.copy()
    tampered[0, 0, 0] = 999.0
    np.save(cache_dir / "hlcvs.npy", tampered)

    with pytest.raises(HlcvsManifestError, match="hash mismatch"):
        verify_hlcvs_manifest(cache_dir)


def test_verify_hlcvs_manifest_rejects_missing_required_file_entry(tmp_path):
    cache_dir = tmp_path / "hlcvs"
    coins = ["BTC"]
    hlcvs = np.ones((2, 1, 4), dtype=np.float64)
    timestamps = np.array([1735689600000, 1735689660000], dtype=np.int64)
    btc_usd_prices = np.array([100.0, 101.0], dtype=np.float64)
    mss = {"BTC": {"exchange": "binance"}, "__meta__": {"btc_source_exchange": "binanceusdm"}}
    _write_cache_files(
        cache_dir,
        hlcvs=hlcvs,
        timestamps=timestamps,
        btc_usd_prices=btc_usd_prices,
        coins=coins,
        mss=mss,
    )
    manifest = build_hlcvs_manifest(
        config=_minimal_config(),
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
    del manifest["files"]["hlcvs"]
    write_hlcvs_manifest(cache_dir, manifest)

    with pytest.raises(HlcvsManifestError, match="missing required file entries"):
        verify_hlcvs_manifest(cache_dir)


def test_hash_logical_array_matches_tobytes_reference():
    """Digest format is a compatibility contract with existing on-disk manifests:
    it must equal the original tobytes(order="C")-based formula exactly."""
    import hashlib

    def reference_hash(array):
        arr = np.ascontiguousarray(np.asarray(array))
        hasher = hashlib.sha256()
        hasher.update(str(arr.dtype).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(arr.tobytes(order="C"))
        return hasher.hexdigest()

    arrays = [
        np.arange(24, dtype=np.float64).reshape(2, 3, 4),
        np.arange(6, dtype=np.float32).reshape(3, 2),
        np.array([1735689600000, 1735689660000], dtype=np.int64),
        np.arange(24, dtype=np.float64).reshape(2, 3, 4)[:, :2, :],  # non-contiguous
        np.asfortranarray(np.arange(6, dtype=np.float64).reshape(2, 3)),  # F-order
    ]
    for arr in arrays:
        assert hash_logical_array(arr) == reference_hash(arr)


def test_verify_hlcvs_manifest_returns_verified_arrays(tmp_path):
    cache_dir = tmp_path / "hlcvs"
    coins = ["BTC"]
    hlcvs = np.arange(8, dtype=np.float64).reshape(2, 1, 4)
    timestamps = np.array([1735689600000, 1735689660000], dtype=np.int64)
    btc_usd_prices = np.array([100.0, 101.0], dtype=np.float64)
    mss = {"BTC": {"exchange": "binance"}, "__meta__": {"btc_source_exchange": "binanceusdm"}}
    _write_cache_files(
        cache_dir,
        hlcvs=hlcvs,
        timestamps=timestamps,
        btc_usd_prices=btc_usd_prices,
        coins=coins,
        mss=mss,
    )
    manifest = build_hlcvs_manifest(
        config=_minimal_config(),
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

    out_arrays = {}
    verify_hlcvs_manifest(cache_dir, out_arrays=out_arrays)

    assert set(out_arrays) == {"hlcvs", "btc_usd_prices", "timestamps"}
    np.testing.assert_array_equal(out_arrays["hlcvs"], hlcvs)
    np.testing.assert_array_equal(out_arrays["timestamps"], timestamps)
    np.testing.assert_array_equal(out_arrays["btc_usd_prices"], btc_usd_prices)
