from concurrent.futures import ThreadPoolExecutor
from threading import Event
from pathlib import Path
import os
import subprocess
import sys

import numpy as np

import materialized_cache as cache
from ohlcv_catalog import OhlcvCatalog
from ohlcv_store import OhlcvStore, month_start_ts


def test_concurrent_first_month_writers_preserve_both_rows(tmp_path, monkeypatch):
    root = tmp_path / "ohlcv"
    stores = [OhlcvStore(root, OhlcvCatalog(root / "catalog.sqlite")) for _ in range(2)]
    first_initializing, release_first, second_started, second_done = [Event() for _ in range(4)]
    original = np.lib.format.open_memmap
    paused = False

    def pause_initialization(filename, *args, **kwargs):
        nonlocal paused
        if kwargs.get("mode") == "w+" and not paused:
            paused = True
            first_initializing.set()
            assert release_first.wait(5)
        return original(filename, *args, **kwargs)

    monkeypatch.setattr(np.lib.format, "open_memmap", pause_initialization)
    base = month_start_ts(2026, 4)
    values = np.array([[101, 99, 100, 10], [102, 100, 101, 11]], dtype=np.float32)

    def write(index):
        if index:
            second_started.set()
        stores[index].write_rows("binance", "1m", "BTC/USDT", np.array([base + index * 60000]),
                                 values[index:index + 1])
        if index:
            second_done.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(write, 0)
        try:
            assert first_initializing.wait(5)
            second = pool.submit(write, 1)
            assert second_started.wait(5)
            assert not second_done.wait(0.2)
        finally:
            release_first.set()
        first.result(timeout=5)
        second.result(timeout=5)
    fresh = OhlcvStore(root, OhlcvCatalog(root / "catalog.sqlite"))
    result = fresh.read_range("binance", "1m", "BTC/USDT", base, base + 60000)
    np.testing.assert_array_equal(result.valid, [True, True])
    np.testing.assert_array_equal(result.values, values)


def test_prune_waits_for_allocation_before_owner_metadata_exists(tmp_path, monkeypatch):
    writing_owner, release_owner, prune_started, prune_done = [Event() for _ in range(4)]
    original = cache._write_lock
    paused = False

    def pause_owner(path):
        nonlocal paused
        if not paused:
            paused = True
            writing_owner.set()
            assert release_owner.wait(5)
        original(path)

    monkeypatch.setattr(cache, "_write_lock", pause_owner)

    def prune():
        prune_started.set()
        cache.prune_materialized_cache(tmp_path)
        prune_done.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        allocation = pool.submit(cache.prepare_materialized_run, tmp_path, "run")
        try:
            assert writing_owner.wait(5)
            pruning = pool.submit(prune)
            assert prune_started.wait(5)
            assert not prune_done.wait(0.2)
        finally:
            release_owner.set()
        run = allocation.result(timeout=5)
        pruning.result(timeout=5)
    assert run.is_dir()
    assert cache.materialized_lock_path(run).is_file()


def test_operation_lock_survives_owner_exit_without_replacing_inode(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(Path(cache.__file__).parent))
    subprocess.run([sys.executable, "-c",
                    "import os,sys; from materialized_cache import materialized_operation_lock; "
                    "lock=materialized_operation_lock(sys.argv[1]); lock.__enter__(); os._exit(0)",
                    str(tmp_path)], check=True, env=env, timeout=10)
    path = cache.materialized_operation_lock_path(tmp_path)
    inode = path.stat().st_ino
    with cache.materialized_operation_lock(tmp_path):
        assert path.stat().st_ino == inode
    assert path.stat().st_ino == inode
