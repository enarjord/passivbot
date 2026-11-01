"""Task-local data policy for simulation preparation; never enabled by live callers."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
import logging
from pathlib import Path
import time


class OfflineDataError(ValueError):
    """A required simulation input is unavailable without network access."""


_policy = ContextVar("simulation_data_policy", default=None)
_acquisition_mode = ContextVar("simulation_acquisition_mode", default="auto")


def normalize_acquisition_mode(value):
    if value not in ("auto", "rest"):
        raise ValueError("backtest.acquisition_mode must be one of ['auto', 'rest']")
    return value


def acquisition_mode():
    """Public-data transport policy, independent of verified candle identity."""
    return _acquisition_mode.get()


def is_offline():
    return _policy.get() is not None


def require_online(operation):
    if is_offline():
        raise OfflineDataError(
            f"Offline data unavailable: {operation}. Refresh on a connected host and copy "
            "the candle cache and metadata, or run with backtest.offline=false."
        )


def snapshot_file(path):
    """Fingerprint small metadata snapshots, including their age, once per preparation."""
    state = _policy.get()
    path = Path(path)
    key = str(path)
    if state is not None and key not in state["metadata"]:
        raw = path.read_bytes()
        stat = path.stat()
        state["metadata"][key] = {
            "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw),
            "mtime_ns": stat.st_mtime_ns,
        }
        logging.info("[offline] metadata=%s age_hours=%.1f sha256=%s", key,
                     max(0, time.time() - stat.st_mtime) / 3600,
                     state["metadata"][key]["sha256"])


def record_range(exchange, symbol, start_ts, end_ts, rng):
    state = _policy.get()
    if state is None:
        return
    # Hash only the range being consumed, never an entire historical cache.
    digest = hashlib.sha256()
    for array in (rng.timestamps, rng.values, rng.valid):
        digest.update(array.tobytes())
    state["ranges"].append({
        "exchange": exchange, "symbol": symbol,
        "requested_start_ms": int(start_ts), "requested_end_ms": int(end_ts),
        "actual_start_ms": int(rng.timestamps[0]), "actual_end_ms": int(rng.timestamps[-1]),
        "valid_rows": int(rng.valid.sum()), "sha256": digest.hexdigest(),
    })


def data_manifest():
    state = _policy.get()
    return json.loads(json.dumps(state)) if state is not None else None


def has_current_offline_proof(snapshot):
    """Old presence-only snapshots cannot attest the current boundary contract."""
    return (isinstance(snapshot, dict)
            and type(snapshot.get("version")) is int and snapshot["version"] == 2
            and type(snapshot.get("boundary_proof_version")) is int
            and snapshot["boundary_proof_version"] == 2)


@contextmanager
def simulation_data_policy(config):
    enabled = config.get("backtest", {}).get("offline", False)
    if not isinstance(enabled, bool):
        raise ValueError("backtest.offline must be a boolean")
    mode = normalize_acquisition_mode(config.get("backtest", {}).get("acquisition_mode", "auto"))
    acquisition_token = _acquisition_mode.set(mode)
    token = None
    try:
        if enabled and not is_offline():
            token = _policy.set({"version": 2, "boundary_proof_version": 2, "metadata": {}, "ranges": []})
        yield
    finally:
        if token is not None:
            _policy.reset(token)
        _acquisition_mode.reset(acquisition_token)


def simulation_data_scope(func):
    """Scope policy to an async simulation API, including tasks it creates."""
    @wraps(func)
    async def wrapped(config, *args, **kwargs):
        with simulation_data_policy(config):
            return await func(config, *args, **kwargs)
    return wrapped
