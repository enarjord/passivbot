from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


LOCK_FILENAME = ".materialized.lock.json"
OP_LOCK_DIRNAME = ".materialized.op.lock"
OP_LOCK_FILENAME = "lock.json"
OP_LOCK_TIMEOUT_SECONDS = 30.0
OP_LOCK_POLL_SECONDS = 0.05


def _utc_ms() -> int:
    return int(time.time() * 1000)


def _hostname() -> str:
    return socket.gethostname()


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_lock(lock_path: Path) -> dict[str, Any] | None:
    try:
        with open(lock_path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_lock(lock_path: Path) -> None:
    now = _utc_ms()
    payload = {
        "pid": os.getpid(),
        "hostname": _hostname(),
        "created_at_ms": now,
        "updated_at_ms": now,
    }
    tmp_path = lock_path.with_suffix(lock_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, lock_path)


def _lock_is_active(lock_path: Path) -> bool:
    lock = _read_lock(lock_path)
    if lock is None:
        return False
    try:
        pid = int(lock.get("pid"))
    except (TypeError, ValueError):
        return False
    host = str(lock.get("hostname") or "")
    if host == _hostname():
        return _process_exists(pid)
    return True


def _warn_foreign_lock(run_root: Path, lock_path: Path) -> None:
    lock = _read_lock(lock_path)
    if lock is None:
        return
    host = str(lock.get("hostname") or "")
    if not host or host == _hostname():
        return
    try:
        updated_at = int(lock.get("updated_at_ms") or lock.get("created_at_ms") or 0)
    except (TypeError, ValueError):
        updated_at = 0
    age_hours = ((_utc_ms() - updated_at) / (60 * 60 * 1000)) if updated_at > 0 else None
    try:
        pid = int(lock.get("pid"))
    except (TypeError, ValueError):
        pid = None
    if age_hours is None:
        logging.warning(
            "[hlcvs] preserving foreign materialized scratch lock path=%s host=%s pid=%s",
            run_root,
            host,
            pid,
        )
    else:
        logging.warning(
            "[hlcvs] preserving foreign materialized scratch lock path=%s host=%s pid=%s age_hours=%.1f",
            run_root,
            host,
            pid,
            age_hours,
        )


def materialized_lock_path(run_root: str | Path) -> Path:
    return Path(run_root) / LOCK_FILENAME


def materialized_operation_lock_path(output_root: str | Path) -> Path:
    return Path(output_root) / OP_LOCK_DIRNAME


def _operation_lock_is_active(lock_dir: Path) -> bool:
    return _lock_is_active(lock_dir / OP_LOCK_FILENAME)


@contextmanager
def materialized_operation_lock(output_root: str | Path):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    lock_dir = materialized_operation_lock_path(root)
    deadline = time.monotonic() + OP_LOCK_TIMEOUT_SECONDS
    acquired = False
    while True:
        try:
            lock_dir.mkdir()
            _write_lock(lock_dir / OP_LOCK_FILENAME)
            acquired = True
            break
        except FileExistsError:
            if not _operation_lock_is_active(lock_dir):
                shutil.rmtree(lock_dir, ignore_errors=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for materialized cache lock {lock_dir}")
            time.sleep(OP_LOCK_POLL_SECONDS)
    try:
        yield
    finally:
        if acquired:
            shutil.rmtree(lock_dir, ignore_errors=True)


def create_materialized_lock(run_root: str | Path) -> Path:
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = materialized_lock_path(root)
    _write_lock(lock_path)
    return lock_path


def _prune_materialized_cache(output_root: str | Path) -> None:
    root = Path(output_root)
    if not root.exists():
        return
    for run_root in root.iterdir():
        if not run_root.is_dir():
            continue
        if run_root.name == OP_LOCK_DIRNAME:
            continue
        lock_path = materialized_lock_path(run_root)
        if lock_path.exists() and _lock_is_active(lock_path):
            _warn_foreign_lock(run_root, lock_path)
            continue
        try:
            shutil.rmtree(run_root)
            logging.info("[hlcvs] pruned materialized scratch payload %s", run_root)
        except FileNotFoundError:
            continue
        except OSError as exc:
            logging.warning(
                "[hlcvs] failed to prune materialized scratch payload %s: %s",
                run_root,
                exc,
            )


def prune_materialized_cache(output_root: str | Path) -> None:
    with materialized_operation_lock(output_root):
        _prune_materialized_cache(output_root)


def prepare_materialized_run(output_root: str | Path, run_id: str) -> Path:
    root = Path(output_root)
    with materialized_operation_lock(root):
        _prune_materialized_cache(root)
        run_root = root / run_id
        if run_root.exists():
            lock_path = materialized_lock_path(run_root)
            if lock_path.exists() and _lock_is_active(lock_path):
                raise FileExistsError(f"materialized run is already active: {run_root}")
            shutil.rmtree(run_root)
        run_root.mkdir(parents=True, exist_ok=False)
        create_materialized_lock(run_root)
        return run_root


def materialized_root_from_array(array: Any) -> Path | None:
    if not isinstance(array, np.memmap):
        return None
    filename = getattr(array, "filename", None)
    if not filename:
        return None
    path = Path(filename)
    if path.name not in {"hlcvs.dat", "timestamps.dat", "btc_usd_prices.dat"}:
        return None
    return path.parent


def release_materialized_root(run_root: str | Path, *, delete: bool = True) -> bool:
    root = Path(run_root)
    lock_path = materialized_lock_path(root)
    if lock_path.exists():
        lock = _read_lock(lock_path)
        lock_pid = None
        if lock is not None:
            try:
                lock_pid = int(lock.get("pid"))
            except (TypeError, ValueError):
                lock_pid = None
        if lock_pid not in {None, os.getpid()} and _lock_is_active(lock_path):
            return False
    if delete:
        try:
            shutil.rmtree(root)
            logging.info("[hlcvs] deleted materialized scratch payload %s", root)
            return True
        except FileNotFoundError:
            return True
        except OSError as exc:
            try:
                create_materialized_lock(root)
            except OSError:
                pass
            logging.warning(
                "[hlcvs] failed to delete materialized scratch payload %s: %s",
                root,
                exc,
            )
            return False
    if lock_path.exists():
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logging.warning("[hlcvs] failed to remove materialized lock %s: %s", lock_path, exc)
            return False
    return True


def release_materialized_payload(array: Any, *, delete: bool = True) -> bool:
    root = materialized_root_from_array(array)
    if root is None:
        return False
    return release_materialized_root(root, delete=delete)
