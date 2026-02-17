"""
Tests for backtest HLCV cache warmup decoupling.

The backtest HLCV cache key no longer includes warmup_minutes. Instead,
warmup sufficiency is checked at load time: if the cached data was
produced with warmup >= the currently needed warmup, the cache is reused.
This allows configs that differ only in EMA spans (and thus warmup) to
share the same cache slot, with a ratchet-up behavior on re-fetch.

Tests cover:
- Cache hash independence from warmup_minutes
- Warmup sufficiency gate in load_coins_hlcvs_from_cache
- cache_meta.json persistence in save_coins_hlcvs_to_cache
- Ratchet-up: smaller warmup reuses cache built with larger warmup
- Legacy caches (no cache_meta.json) treated as warmup=0
"""

import copy
import gzip
import json
import os

import numpy as np

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backtest import (
    ensure_valid_index_metadata,
    get_cache_hash,
    load_coins_hlcvs_from_cache,
    save_coins_hlcvs_to_cache,
)


# ============================================================================
# Fixtures
# ============================================================================


def _base_config(**overrides):
    """Minimal config sufficient for cache hash computation."""
    cfg = {
        "backtest": {
            "base_dir": "backtests",
            "compress_cache": False,
            "end_date": "2025-06-01",
            "start_date": "2024-01-01",
            "exchanges": ["binance"],
            "gap_tolerance_ohlcvs_minutes": 120,
        },
        "bot": {
            "long": {
                "ema_span_0": 1000.0,
                "ema_span_1": 1500.0,
                "filter_volume_ema_span": 2000.0,
                "filter_volatility_ema_span": 100.0,
                "entry_volatility_ema_span_hours": 1.0,
            },
            "short": {
                "ema_span_0": 100.0,
                "ema_span_1": 100.0,
                "filter_volume_ema_span": 360.0,
                "filter_volatility_ema_span": 10.0,
                "entry_volatility_ema_span_hours": 1.0,
            },
        },
        "live": {
            "approved_coins": {"long": ["BTC/USDT:USDT"], "short": []},
            "ignored_coins": {"long": [], "short": []},
            "minimum_coin_age_days": 30,
            "warmup_ratio": 0.3,
            "max_warmup_minutes": 0,
        },
        "optimize": {"bounds": {}},
    }
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


def _write_fake_cache(cache_dir, *, compress=False, warmup_minutes=None):
    """Write minimal valid cache files so load_coins_hlcvs_from_cache succeeds."""
    os.makedirs(cache_dir, exist_ok=True)

    coins = ["BTC"]
    mss = {"BTC": {"first_valid_index": 0, "last_valid_index": 9}}

    hlcvs = np.zeros((10, 1, 4), dtype=np.float64)
    btc_usd = np.ones(10, dtype=np.float64) * 50000.0
    timestamps = np.arange(10, dtype=np.int64) * 60_000

    json.dump(coins, open(os.path.join(cache_dir, "coins.json"), "w"))
    json.dump(mss, open(os.path.join(cache_dir, "market_specific_settings.json"), "w"))

    if compress:
        with gzip.open(os.path.join(cache_dir, "hlcvs.npy.gz"), "wb", compresslevel=1) as f:
            np.save(f, hlcvs)
        with gzip.open(os.path.join(cache_dir, "btc_usd_prices.npy.gz"), "wb", compresslevel=1) as f:
            np.save(f, btc_usd)
        with gzip.open(os.path.join(cache_dir, "timestamps.npy.gz"), "wb", compresslevel=1) as f:
            np.save(f, timestamps)
    else:
        np.save(os.path.join(cache_dir, "hlcvs.npy"), hlcvs)
        np.save(os.path.join(cache_dir, "btc_usd_prices.npy"), btc_usd)
        np.save(os.path.join(cache_dir, "timestamps.npy"), timestamps)

    if warmup_minutes is not None:
        json.dump(
            {"warmup_minutes": int(warmup_minutes)},
            open(os.path.join(cache_dir, "cache_meta.json"), "w"),
        )


# ============================================================================
# Test Class: Cache Hash Independence from Warmup
# ============================================================================


class TestCacheHashIndependence:
    """Verify that warmup_minutes no longer affects the cache hash."""

    def test_different_ema_spans_same_hash(self):
        """Two configs differing only in EMA spans produce the same cache hash."""
        cfg_a = _base_config()
        cfg_b = copy.deepcopy(cfg_a)
        cfg_b["bot"]["long"]["entry_volatility_ema_span_hours"] = 500.0

        hash_a = get_cache_hash(cfg_a, "binance")
        hash_b = get_cache_hash(cfg_b, "binance")

        assert hash_a == hash_b

    def test_different_warmup_ratio_same_hash(self):
        """Changing warmup_ratio does not change the hash."""
        cfg_a = _base_config()
        cfg_b = copy.deepcopy(cfg_a)
        cfg_b["live"]["warmup_ratio"] = 10.0

        hash_a = get_cache_hash(cfg_a, "binance")
        hash_b = get_cache_hash(cfg_b, "binance")

        assert hash_a == hash_b

    def test_different_coins_different_hash(self):
        """Changing approved_coins still produces a different hash."""
        cfg_a = _base_config()
        cfg_b = copy.deepcopy(cfg_a)
        cfg_b["live"]["approved_coins"]["long"] = ["ETH/USDT:USDT"]

        hash_a = get_cache_hash(cfg_a, "binance")
        hash_b = get_cache_hash(cfg_b, "binance")

        assert hash_a != hash_b

    def test_different_dates_different_hash(self):
        """Changing start_date still produces a different hash."""
        cfg_a = _base_config()
        cfg_b = copy.deepcopy(cfg_a)
        cfg_b["backtest"]["start_date"] = "2023-06-01"

        hash_a = get_cache_hash(cfg_a, "binance")
        hash_b = get_cache_hash(cfg_b, "binance")

        assert hash_a != hash_b


# ============================================================================
# Test Class: Warmup Sufficiency Gate
# ============================================================================


class TestWarmupSufficiencyGate:
    """Verify that load_coins_hlcvs_from_cache checks warmup sufficiency."""

    def test_sufficient_warmup_returns_cache(self, tmp_path, monkeypatch):
        """Cache with warmup >= needed returns data."""
        cfg = _base_config()
        cache_hash = get_cache_hash(cfg, "binance")
        cache_dir = tmp_path / "caches" / "hlcvs_data" / cache_hash[:16]
        _write_fake_cache(str(cache_dir), warmup_minutes=5000)

        monkeypatch.chdir(tmp_path)
        result = load_coins_hlcvs_from_cache(cfg, "binance", warmup_minutes=3000)

        assert result is not None

    def test_insufficient_warmup_returns_none(self, tmp_path, monkeypatch):
        """Cache with warmup < needed returns None."""
        cfg = _base_config()
        cache_hash = get_cache_hash(cfg, "binance")
        cache_dir = tmp_path / "caches" / "hlcvs_data" / cache_hash[:16]
        _write_fake_cache(str(cache_dir), warmup_minutes=1000)

        monkeypatch.chdir(tmp_path)
        result = load_coins_hlcvs_from_cache(cfg, "binance", warmup_minutes=5000)

        assert result is None

    def test_exact_warmup_match_returns_cache(self, tmp_path, monkeypatch):
        """Cache with warmup == needed returns data (boundary condition)."""
        cfg = _base_config()
        cache_hash = get_cache_hash(cfg, "binance")
        cache_dir = tmp_path / "caches" / "hlcvs_data" / cache_hash[:16]
        _write_fake_cache(str(cache_dir), warmup_minutes=5000)

        monkeypatch.chdir(tmp_path)
        result = load_coins_hlcvs_from_cache(cfg, "binance", warmup_minutes=5000)

        assert result is not None

    def test_zero_needed_warmup_always_hits(self, tmp_path, monkeypatch):
        """When needed warmup is 0, any cache is sufficient."""
        cfg = _base_config()
        cache_hash = get_cache_hash(cfg, "binance")
        cache_dir = tmp_path / "caches" / "hlcvs_data" / cache_hash[:16]
        _write_fake_cache(str(cache_dir), warmup_minutes=0)

        monkeypatch.chdir(tmp_path)
        result = load_coins_hlcvs_from_cache(cfg, "binance", warmup_minutes=0)

        assert result is not None

    def test_default_warmup_param_is_zero(self, tmp_path, monkeypatch):
        """Calling without warmup_minutes defaults to 0 (always hits)."""
        cfg = _base_config()
        cache_hash = get_cache_hash(cfg, "binance")
        cache_dir = tmp_path / "caches" / "hlcvs_data" / cache_hash[:16]
        _write_fake_cache(str(cache_dir), warmup_minutes=0)

        monkeypatch.chdir(tmp_path)
        result = load_coins_hlcvs_from_cache(cfg, "binance")

        assert result is not None


# ============================================================================
# Test Class: Legacy Cache Compatibility
# ============================================================================


class TestLegacyCacheCompatibility:
    """Verify behavior with pre-existing caches that lack cache_meta.json."""

    def test_missing_cache_meta_treated_as_zero_warmup(self, tmp_path, monkeypatch):
        """Legacy cache without cache_meta.json has cached_warmup=0."""
        cfg = _base_config()
        cache_hash = get_cache_hash(cfg, "binance")
        cache_dir = tmp_path / "caches" / "hlcvs_data" / cache_hash[:16]
        _write_fake_cache(str(cache_dir), warmup_minutes=None)

        monkeypatch.chdir(tmp_path)

        # Needed warmup > 0 → miss (legacy cache has no warmup metadata)
        result = load_coins_hlcvs_from_cache(cfg, "binance", warmup_minutes=100)
        assert result is None

    def test_missing_cache_meta_hits_when_zero_needed(self, tmp_path, monkeypatch):
        """Legacy cache still works when needed warmup is 0."""
        cfg = _base_config()
        cache_hash = get_cache_hash(cfg, "binance")
        cache_dir = tmp_path / "caches" / "hlcvs_data" / cache_hash[:16]
        _write_fake_cache(str(cache_dir), warmup_minutes=None)

        monkeypatch.chdir(tmp_path)

        result = load_coins_hlcvs_from_cache(cfg, "binance", warmup_minutes=0)
        assert result is not None

    def test_corrupt_cache_meta_treated_as_zero(self, tmp_path, monkeypatch):
        """Corrupt cache_meta.json falls back to cached_warmup=0."""
        cfg = _base_config()
        cache_hash = get_cache_hash(cfg, "binance")
        cache_dir = tmp_path / "caches" / "hlcvs_data" / cache_hash[:16]
        _write_fake_cache(str(cache_dir), warmup_minutes=5000)

        # Corrupt the meta file
        meta_path = os.path.join(str(cache_dir), "cache_meta.json")
        with open(meta_path, "w") as f:
            f.write("not valid json{{{")

        monkeypatch.chdir(tmp_path)
        result = load_coins_hlcvs_from_cache(cfg, "binance", warmup_minutes=100)
        assert result is None


# ============================================================================
# Test Class: Save Persists Warmup Metadata
# ============================================================================


class TestSavePersistsWarmupMetadata:
    """Verify that save_coins_hlcvs_to_cache writes cache_meta.json."""

    def test_save_writes_cache_meta(self, tmp_path, monkeypatch):
        """save_coins_hlcvs_to_cache creates cache_meta.json with warmup_minutes."""
        monkeypatch.chdir(tmp_path)
        cfg = _base_config()

        coins = ["BTC"]
        hlcvs = np.zeros((10, 1, 4), dtype=np.float64)
        mss = {"BTC": {}}
        btc_usd = np.ones(10, dtype=np.float64)
        timestamps = np.arange(10, dtype=np.int64) * 60_000

        cache_dir = save_coins_hlcvs_to_cache(
            cfg, coins, hlcvs, "binance", mss, btc_usd, timestamps,
            warmup_minutes=12345,
        )

        meta_path = os.path.join(str(cache_dir), "cache_meta.json")
        assert os.path.exists(meta_path)

        meta = json.load(open(meta_path))
        assert meta["warmup_minutes"] == 12345

    def test_save_overwrites_existing_cache(self, tmp_path, monkeypatch):
        """Saving to an existing cache dir overwrites files (ratchet-up)."""
        monkeypatch.chdir(tmp_path)
        cfg = _base_config()

        coins = ["BTC"]
        hlcvs = np.zeros((10, 1, 4), dtype=np.float64)
        mss = {"BTC": {}}
        btc_usd = np.ones(10, dtype=np.float64)
        timestamps = np.arange(10, dtype=np.int64) * 60_000

        # First save with warmup=1000
        cache_dir = save_coins_hlcvs_to_cache(
            cfg, coins, hlcvs, "binance", mss, btc_usd, timestamps,
            warmup_minutes=1000,
        )
        meta = json.load(open(os.path.join(str(cache_dir), "cache_meta.json")))
        assert meta["warmup_minutes"] == 1000

        # Second save with warmup=5000 (ratchet up)
        hlcvs_bigger = np.zeros((20, 1, 4), dtype=np.float64)
        btc_bigger = np.ones(20, dtype=np.float64)
        ts_bigger = np.arange(20, dtype=np.int64) * 60_000
        cache_dir2 = save_coins_hlcvs_to_cache(
            cfg, coins, hlcvs_bigger, "binance", mss, btc_bigger, ts_bigger,
            warmup_minutes=5000,
        )

        assert str(cache_dir) == str(cache_dir2)
        meta = json.load(open(os.path.join(str(cache_dir2), "cache_meta.json")))
        assert meta["warmup_minutes"] == 5000

    def test_save_does_not_downgrade_existing_warmup(self, tmp_path, monkeypatch):
        """Saving with smaller warmup must not downgrade cache metadata."""
        monkeypatch.chdir(tmp_path)
        cfg = _base_config()

        coins = ["BTC"]
        hlcvs = np.zeros((10, 1, 4), dtype=np.float64)
        mss = {"BTC": {}}
        btc_usd = np.ones(10, dtype=np.float64)
        timestamps = np.arange(10, dtype=np.int64) * 60_000

        cache_dir = save_coins_hlcvs_to_cache(
            cfg, coins, hlcvs, "binance", mss, btc_usd, timestamps,
            warmup_minutes=5000,
        )
        save_coins_hlcvs_to_cache(
            cfg, coins, hlcvs, "binance", mss, btc_usd, timestamps,
            warmup_minutes=1000,
        )

        meta = json.load(open(os.path.join(str(cache_dir), "cache_meta.json")))
        assert meta["warmup_minutes"] == 5000

    def test_save_with_compression(self, tmp_path, monkeypatch):
        """cache_meta.json is written even with compressed cache."""
        monkeypatch.chdir(tmp_path)
        cfg = _base_config()
        cfg["backtest"]["compress_cache"] = True

        coins = ["BTC"]
        hlcvs = np.zeros((10, 1, 4), dtype=np.float64)
        mss = {"BTC": {}}
        btc_usd = np.ones(10, dtype=np.float64)
        timestamps = np.arange(10, dtype=np.int64) * 60_000

        cache_dir = save_coins_hlcvs_to_cache(
            cfg, coins, hlcvs, "binance", mss, btc_usd, timestamps,
            warmup_minutes=9999,
        )

        meta_path = os.path.join(str(cache_dir), "cache_meta.json")
        assert os.path.exists(meta_path)
        meta = json.load(open(meta_path))
        assert meta["warmup_minutes"] == 9999


# ============================================================================
# Test Class: Ratchet-Up End-to-End
# ============================================================================


class TestRatchetUpEndToEnd:
    """Simulate the full ratchet-up scenario with save then load."""

    def test_ratchet_scenario(self, tmp_path, monkeypatch):
        """
        Scenario:
        1. Save cache with warmup=3000.
        2. Load with warmup=3000 → hit.
        3. Load with warmup=5000 → miss (insufficient).
        4. Save cache with warmup=5000 (overwrite).
        5. Load with warmup=3000 → hit (5000 >= 3000).
        6. Load with warmup=5000 → hit (5000 >= 5000).
        """
        monkeypatch.chdir(tmp_path)
        cfg = _base_config()

        coins = ["BTC"]
        hlcvs = np.zeros((10, 1, 4), dtype=np.float64)
        mss = {"BTC": {}}
        btc_usd = np.ones(10, dtype=np.float64)
        timestamps = np.arange(10, dtype=np.int64) * 60_000

        # Step 1: Save with warmup=3000
        save_coins_hlcvs_to_cache(
            cfg, coins, hlcvs, "binance", mss, btc_usd, timestamps,
            warmup_minutes=3000,
        )

        # Step 2: Load with warmup=3000 → hit
        result = load_coins_hlcvs_from_cache(cfg, "binance", warmup_minutes=3000)
        assert result is not None

        # Step 3: Load with warmup=5000 → miss
        result = load_coins_hlcvs_from_cache(cfg, "binance", warmup_minutes=5000)
        assert result is None

        # Step 4: Save with warmup=5000 (overwrite same slot)
        hlcvs_bigger = np.zeros((20, 1, 4), dtype=np.float64)
        btc_bigger = np.ones(20, dtype=np.float64)
        ts_bigger = np.arange(20, dtype=np.int64) * 60_000
        save_coins_hlcvs_to_cache(
            cfg, coins, hlcvs_bigger, "binance", mss, btc_bigger, ts_bigger,
            warmup_minutes=5000,
        )

        # Step 5: Load with warmup=3000 → hit (ratcheted up)
        result = load_coins_hlcvs_from_cache(cfg, "binance", warmup_minutes=3000)
        assert result is not None

        # Step 6: Load with warmup=5000 → hit
        result = load_coins_hlcvs_from_cache(cfg, "binance", warmup_minutes=5000)
        assert result is not None

    def test_cache_hit_uses_current_warmup_after_metadata_normalization(self, tmp_path, monkeypatch):
        """Cache hit must apply current run warmup, not stale cached mss warmup."""
        monkeypatch.chdir(tmp_path)
        cfg = _base_config()

        coins = ["BTC"]
        hlcvs = np.zeros((10000, 1, 4), dtype=np.float64)
        mss = {"BTC": {"first_valid_index": 0, "last_valid_index": 9999, "warmup_minutes": 5000}}
        btc_usd = np.ones(10000, dtype=np.float64)
        timestamps = np.arange(10000, dtype=np.int64) * 60_000

        save_coins_hlcvs_to_cache(
            cfg, coins, hlcvs, "binance", mss, btc_usd, timestamps,
            warmup_minutes=5000,
        )
        result = load_coins_hlcvs_from_cache(cfg, "binance", warmup_minutes=3000)
        assert result is not None

        mss_loaded = result[3]
        ensure_valid_index_metadata(mss_loaded, hlcvs, coins, {"__default__": 3000, "BTC": 3000})
        assert mss_loaded["BTC"]["warmup_minutes"] == 3000
        assert mss_loaded["BTC"]["trade_start_index"] == 3000
