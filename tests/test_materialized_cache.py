import json
import socket
import time

import numpy as np
import pytest

import materialized_cache
from backtest_dataset_materializer import materialize_frames
from materialized_cache import (
    LOCK_FILENAME,
    create_materialized_lock,
    materialized_lock_path,
    prepare_materialized_run,
    prune_materialized_cache,
    release_materialized_root,
    release_materialized_payload,
)


def test_prune_materialized_cache_removes_unlocked_payloads_only(tmp_path):
    cache_root = tmp_path / "caches"
    materialized = cache_root / "ohlcvs" / "materialized"
    data_root = cache_root / "ohlcvs" / "data"
    hlcvs_data = cache_root / "hlcvs_data"
    stale_payload = materialized / "combined_old"
    stale_payload.mkdir(parents=True)
    (stale_payload / "hlcvs.dat").write_bytes(b"scratch")
    data_root.mkdir(parents=True)
    (data_root / "keep.npy").write_bytes(b"raw")
    hlcvs_data.mkdir(parents=True)
    (hlcvs_data / "keep.npy").write_bytes(b"prepared")

    prune_materialized_cache(materialized)

    assert not stale_payload.exists()
    assert (data_root / "keep.npy").exists()
    assert (hlcvs_data / "keep.npy").exists()


def test_prune_materialized_cache_preserves_live_lock(tmp_path):
    materialized = tmp_path / "materialized"
    active_payload = materialized / "binance_active"
    active_payload.mkdir(parents=True)
    create_materialized_lock(active_payload)

    prune_materialized_cache(materialized)

    assert active_payload.exists()
    assert materialized_lock_path(active_payload).exists()


def test_prune_materialized_cache_recovers_stale_local_lock(tmp_path):
    materialized = tmp_path / "materialized"
    stale_payload = materialized / "binance_stale"
    stale_payload.mkdir(parents=True)
    lock_path = stale_payload / LOCK_FILENAME
    with open(lock_path, "w") as f:
        json.dump(
            {
                "pid": 999_999_999,
                "hostname": socket.gethostname(),
                "created_at_ms": int(time.time() * 1000),
                "updated_at_ms": int(time.time() * 1000),
            },
            f,
        )

    prune_materialized_cache(materialized)

    assert not stale_payload.exists()


def test_prune_materialized_cache_preserves_and_logs_foreign_lock(tmp_path, caplog):
    materialized = tmp_path / "materialized"
    foreign_payload = materialized / "foreign_active"
    foreign_payload.mkdir(parents=True)
    lock_path = foreign_payload / LOCK_FILENAME
    with open(lock_path, "w") as f:
        json.dump(
            {
                "pid": 999_999_999,
                "hostname": "different-host",
                "created_at_ms": int(time.time() * 1000) - 25 * 60 * 60 * 1000,
                "updated_at_ms": int(time.time() * 1000) - 25 * 60 * 60 * 1000,
            },
            f,
        )

    prune_materialized_cache(materialized)

    assert foreign_payload.exists()
    assert materialized_lock_path(foreign_payload).exists()
    assert "preserving foreign materialized scratch lock" in caplog.text
    assert "different-host" in caplog.text


def test_release_materialized_payload_deletes_own_locked_run(tmp_path):
    materialized = tmp_path / "materialized"
    run_root = materialized / "combined_current"
    run_root.mkdir(parents=True)
    create_materialized_lock(run_root)
    path = run_root / "hlcvs.dat"
    arr = np.memmap(path, mode="w+", dtype=np.float64, shape=(2, 1, 4))
    arr[:] = 1.0
    arr.flush()

    assert release_materialized_payload(arr)
    del arr

    assert not run_root.exists()


def test_release_materialized_root_keeps_lock_when_delete_fails(tmp_path, monkeypatch):
    materialized = tmp_path / "materialized"
    run_root = materialized / "delete_fails"
    run_root.mkdir(parents=True)
    create_materialized_lock(run_root)

    def fail_rmtree(_path):
        raise OSError("simulated delete failure")

    monkeypatch.setattr(materialized_cache.shutil, "rmtree", fail_rmtree)

    assert not release_materialized_root(run_root)
    assert run_root.exists()
    assert materialized_lock_path(run_root).exists()


def test_materialize_frames_failure_removes_partial_locked_run(tmp_path):
    materialized = tmp_path / "materialized"
    start = 1_800_000_000_000
    timestamps = np.array([start, start + 60_000], dtype=np.int64)

    with pytest.raises(ValueError):
        materialize_frames(
            output_root=materialized,
            exchange="combined",
            coins=["ETH"],
            timestamps=timestamps,
            aligned_values_by_coin={"ETH": np.ones((1, 4), dtype=np.float64)},
            btc_usd_prices=np.array([30_000.0, 30_100.0]),
            mss={"ETH": {}},
            run_id="bad_shape",
        )

    assert not (materialized / "bad_shape").exists()


def test_prepare_materialized_run_prunes_before_exposing_locked_run(tmp_path):
    materialized = tmp_path / "materialized"
    old_payload = materialized / "old_unlocked"
    old_payload.mkdir(parents=True)
    (old_payload / "hlcvs.dat").write_bytes(b"old")

    run_root = prepare_materialized_run(materialized, "new_run")

    assert not old_payload.exists()
    assert run_root.exists()
    assert materialized_lock_path(run_root).exists()
