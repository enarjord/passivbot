from concurrent.futures import ThreadPoolExecutor
from threading import Event
from pathlib import Path
import os
import json
import multiprocessing
import shutil
import time
import subprocess
import sys

import numpy as np
import pytest

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


def _contending_process(root, start, iterations):
    # Separate processes must exclude each other even before run-owner metadata
    # exists. O_EXCL makes overlapping critical sections an immediate failure.
    start.wait(10)
    root = Path(root)
    marker = root / "critical-section"
    for _ in range(iterations):
        with cache.materialized_operation_lock(root):
            fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                with (root / "owners.log").open("a") as output:
                    output.write(f"{os.getpid()}\n")
                time.sleep(0.005)
            finally:
                os.close(fd)
                marker.unlink()


def test_spawned_operation_owners_exclude_each_other(tmp_path):
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    workers = [context.Process(target=_contending_process, args=(str(tmp_path), start, 8)) for _ in range(4)]
    for worker in workers:
        worker.start()
    start.set()
    try:
        for worker in workers:
            worker.join(15)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(5)
    owners = (tmp_path / "owners.log").read_text().splitlines()
    assert len(owners) == 32
    assert len(set(owners)) == 4
    assert cache.materialized_operation_lock_path(tmp_path).is_file()


@pytest.mark.parametrize("metadata", [None, "invalid-json", {}, {"pid": "invalid"}, {"pid": 0}])
def test_unknown_legacy_owner_is_preserved_with_actionable_recovery(tmp_path, metadata):
    legacy = tmp_path / cache.OP_LOCK_DIRNAME
    legacy.mkdir()
    if metadata is not None:
        text = metadata if isinstance(metadata, str) else json.dumps(metadata)
        (legacy / cache.OP_LOCK_FILENAME).write_text(text)
    with pytest.raises(RuntimeError, match="Stop all workers using the previous lock protocol"):
        cache.prepare_materialized_run(tmp_path, "new-run")
    assert legacy.is_dir()
    assert not (tmp_path / "new-run").exists()
    # Explicit operator cleanup is safe only after all old workers have stopped.
    shutil.rmtree(legacy)
    assert cache.prepare_materialized_run(tmp_path, "new-run").is_dir()


@pytest.mark.parametrize("host,pid", [("local", "live"), ("foreign-host", "dead")])
def test_active_or_foreign_legacy_owner_is_never_reclaimed(tmp_path, host, pid, monkeypatch):
    legacy = tmp_path / cache.OP_LOCK_DIRNAME
    legacy.mkdir()
    owner = {"hostname": cache._hostname() if host == "local" else host,
             "pid": os.getpid() if pid == "live" else 12345}
    (legacy / cache.OP_LOCK_FILENAME).write_text(json.dumps(owner))
    if pid == "dead":
        monkeypatch.setattr(cache, "_process_exists", lambda _pid: False)
    with pytest.raises(RuntimeError, match="ownership is unknown or active"):
        cache.prune_materialized_cache(tmp_path)
    assert json.loads((legacy / cache.OP_LOCK_FILENAME).read_text()) == owner


def test_confirmed_dead_local_legacy_owner_does_not_block(tmp_path, monkeypatch):
    legacy = tmp_path / cache.OP_LOCK_DIRNAME
    legacy.mkdir()
    (legacy / cache.OP_LOCK_FILENAME).write_text(json.dumps({"hostname": cache._hostname(), "pid": 12345}))
    monkeypatch.setattr(cache, "_process_exists", lambda _pid: False)
    assert cache.prepare_materialized_run(tmp_path, "new-run").is_dir()


@pytest.mark.parametrize("pid", [999999999.5, 999999999.0, True, False, "999999999"])
def test_non_integer_legacy_pid_cannot_prove_dead_owner(tmp_path, monkeypatch, pid):
    legacy = tmp_path / cache.OP_LOCK_DIRNAME
    legacy.mkdir()
    owner = {"hostname": cache._hostname(), "pid": pid}
    (legacy / cache.OP_LOCK_FILENAME).write_text(json.dumps(owner))
    def unexpected_probe(_pid):
        pytest.fail("Malformed owner PID must not be probed as a process")
    monkeypatch.setattr(cache, "_process_exists", unexpected_probe)
    with pytest.raises(RuntimeError, match="ownership is unknown or active"):
        cache.prepare_materialized_run(tmp_path, "new-run")
    assert json.loads((legacy / cache.OP_LOCK_FILENAME).read_text()) == owner
    assert not (tmp_path / "new-run").exists()


def test_unrepresentable_legacy_pid_has_actionable_recovery(tmp_path):
    legacy = tmp_path / cache.OP_LOCK_DIRNAME
    legacy.mkdir()
    owner = {"hostname": cache._hostname(), "pid": 10**100}
    (legacy / cache.OP_LOCK_FILENAME).write_text(json.dumps(owner))
    with pytest.raises(RuntimeError, match="Stop all workers using the previous lock protocol"):
        cache.prepare_materialized_run(tmp_path, "new-run")
    assert json.loads((legacy / cache.OP_LOCK_FILENAME).read_text()) == owner
    assert not (tmp_path / "new-run").exists()


def test_confirmed_dead_legacy_migration_survives_pid_reuse(tmp_path, monkeypatch):
    legacy = tmp_path / cache.OP_LOCK_DIRNAME
    legacy.mkdir()
    owner = {"hostname": cache._hostname(), "pid": 12345}
    (legacy / cache.OP_LOCK_FILENAME).write_text(json.dumps(owner))
    monkeypatch.setattr(cache, "_process_exists", lambda _pid: False)
    with cache.materialized_operation_lock(tmp_path):
        assert not legacy.exists()
    monkeypatch.setattr(cache, "_process_exists", lambda _pid: True)
    with cache.materialized_operation_lock(tmp_path):
        assert not legacy.exists()
    assert cache.materialized_operation_lock_path(tmp_path).is_file()
