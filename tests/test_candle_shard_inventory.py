import os

import numpy as np
import pytest

from candlestick_manager import CANDLE_DTYPE, ONE_MIN_MS, CandlestickManager


def test_shard_pruning_batches_parent_scan_and_observes_later_deletion(tmp_path, monkeypatch):
    manager = CandlestickManager(exchange=None, exchange_name="test", cache_dir=str(tmp_path))
    paths = [tmp_path / f"{i}.npy" for i in range(200)]
    for path in paths:
        path.touch()
    index = {"shards": {
        str(i): {"path": str(path), "min_ts": i, "max_ts": i}
        for i, path in enumerate(paths)
    }}
    scans = []
    scan = os.scandir

    def counted_scan(parent):
        scans.append(parent)
        return scan(parent)

    monkeypatch.setattr(os, "scandir", counted_scan)
    monkeypatch.setattr(os.path, "exists", lambda path: (_ for _ in ()).throw(
        AssertionError("regular shards should not require per-file stat")))
    assert manager._prune_missing_shards_from_index(index) == 0
    assert scans == [str(tmp_path)]
    paths[-1].unlink()
    assert manager._prune_missing_shards_from_index(index) == 1
    assert "199" not in index["shards"]
    assert index["meta"]["last_final_ts"] == 198
    assert len(scans) == 2


def test_shard_pruning_follows_symlinks_and_handles_multiple_parents(tmp_path):
    manager = CandlestickManager(exchange=None, exchange_name="test", cache_dir=str(tmp_path))
    shard = tmp_path / "existing.npy"
    shard.touch()
    good = tmp_path / "good.npy"
    good.symlink_to(shard)
    broken = tmp_path / "broken.npy"
    broken.symlink_to(tmp_path / "absent.npy")
    index = {"shards": {
        "good": {"path": str(good), "max_ts": 1},
        "bad": {"path": str(broken), "max_ts": 2},
        "absent_parent": {"path": str(tmp_path / "absent" / "data.npy"), "max_ts": 3},
    }}
    assert manager._prune_missing_shards_from_index(index) == 2
    assert set(index["shards"]) == {"good"}


def test_shard_pruning_uses_existing_path_fallback_when_listing_fails(tmp_path, monkeypatch):
    manager = CandlestickManager(exchange=None, exchange_name="test", cache_dir=str(tmp_path))
    shard = tmp_path / "existing.npy"
    shard.touch()
    index = {"shards": {"present": {"path": str(shard)},
                        "missing": {"path": str(tmp_path / "missing.npy")}}}

    def denied(_):
        raise PermissionError("cannot enumerate")

    monkeypatch.setattr(os, "scandir", denied)
    assert manager._prune_missing_shards_from_index(index) == 1
    assert set(index["shards"]) == {"present"}


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root can search mode-0400 directories")
def test_readable_but_unsearchable_directory_does_not_prove_shard_exists(tmp_path):
    manager = CandlestickManager(exchange=None, exchange_name="test", cache_dir=str(tmp_path))
    parent = tmp_path / "readable"
    parent.mkdir()
    shard = parent / "data.npy"
    shard.touch()
    index = {"shards": {"day": {"path": str(shard)}}}
    parent.chmod(0o400)
    try:
        if os.path.exists(shard):
            pytest.skip("filesystem does not enforce search permissions")
        assert manager._prune_missing_shards_from_index(index) == 1
        assert not index["shards"]
    finally:
        parent.chmod(0o700)

def test_candle_health_reuses_index_within_read_and_prunes_between_reads(tmp_path, monkeypatch):
    manager = CandlestickManager(exchange=None, exchange_name="test", cache_dir=str(tmp_path))
    symbol = "TEST/USDT:USDT"
    candle = np.array([(ONE_MIN_MS, 1, 1, 1, 1, 1)], dtype=CANDLE_DTYPE)
    manager._persist_batch(symbol, candle)
    scans = 0
    original = manager._prune_missing_shards_from_index

    def prune(index):
        nonlocal scans
        scans += 1
        return original(index)

    monkeypatch.setattr(manager, "_prune_missing_shards_from_index", prune)
    first = manager.get_completed_candle_health(symbol, {"1m": 1}, now_ms=2 * 24 * 60 * ONE_MIN_MS)
    assert first["timeframes"]["1m"]["last_cached_ts"] == ONE_MIN_MS
    assert scans == 1

    indexed = manager._index[f"{symbol}::1m"]
    [shard] = indexed["shards"].values()
    os.unlink(shard["path"])
    manager._cache.pop(symbol, None)
    invalidations = []
    monkeypatch.setattr(manager, "_invalidate_ema_cache",
                        lambda *args, **kwargs: invalidations.append((args, kwargs)))
    second = manager.get_completed_candle_health(symbol, {"1m": 1}, now_ms=2 * 24 * 60 * ONE_MIN_MS)
    assert second["timeframes"]["1m"]["last_cached_ts"] is None
    assert scans == 2
    assert invalidations == [((symbol,), {"timeframe": "1m"})]
