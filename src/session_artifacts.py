"""Readable run directories and explicit, versioned artifact discovery."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import secrets
import tempfile
from uuid import uuid4

import numpy as np

LAYOUT_VERSION = 2
SESSION_MANIFEST = "session.json"


def utc_timestamp(timestamp_ms=None) -> str:
    now = (
        datetime.now(timezone.utc)
        if timestamp_ms is None
        else datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc)
    )
    return now.strftime("%Y-%m-%dT%H_%M_%SZ")


def utc_datetime(timestamp_ms=None) -> str:
    now = (
        datetime.now(timezone.utc)
        if timestamp_ms is None
        else datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc)
    )
    return now.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def resolve_optimizer_seed(options, resume_directory=None):
    if options.get("seed") is not None:
        return
    if resume_directory is None:
        options["seed"] = secrets.randbits(32)
    else:
        manifest = Path(resume_directory) / SESSION_MANIFEST
        if manifest.is_file():
            options["seed"] = json.loads(manifest.read_text(encoding="utf-8"))["seed"]


def safe_component(value: str, *, max_length=80) -> str:
    original = str(value)
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", original).strip("_-") or "unnamed"
    if label != original or len(label) > max_length:
        suffix = hashlib.sha256(original.encode()).hexdigest()[:8]
        label = f"{label[:max_length - 9]}-{suffix}"
    return label


def coins_label(coins) -> str:
    coins = sorted(set(coins))
    labels = [safe_component(coin) for coin in coins]
    joined = "_".join(labels)
    return (
        joined
        if coins and len(coins) <= 6 and len(joined) <= 80
        else f"{len(coins)}_coins"
    )


def canonical_json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else _unsupported(x),
    )


def _unsupported(value):
    raise TypeError(f"Unsupported setup input: {type(value).__name__}")


def setup_hash(payload) -> str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def snapshot_starting_configs(configs):
    """Freeze selected seeds on disk so hashing and execution consume the same bytes."""
    stream = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
    digest, count = hashlib.sha256(), 0
    try:
        for config in configs:
            identity = {
                key: config[key]
                for key in (
                    "bot",
                    "live",
                    "optimize",
                    "_optimizer_anchor",
                    "optimizer_anchor",
                )
                if key in config
            }
            digest.update(canonical_json(identity).encode() + b"\n")
            stream.write(canonical_json(config) + "\n")
            count += 1
        stream.flush()
        return stream, {"count": count, "sha256": digest.hexdigest()}
    except BaseException:
        stream.close()
        raise


def iter_starting_snapshot(stream):
    stream.seek(0)
    for line in stream:
        yield json.loads(line)


def effective_setup_config(config, *, optimize=False) -> dict:
    """Project execution inputs; never hash helper state or machine-local paths."""
    from config_utils import clean_config
    from optimization.evaluation_contract import BACKTEST_LIVE_KEYS
    from optimization.fine_tune_anchors import get_anchor_plan

    effective = clean_config(config)
    backtest = deepcopy(effective["backtest"])
    for key in (
        "base_dir",
        "cache_dir",
        "coins",
        "hlcvs_data_dir",
        "ohlcv_source_dir",
        "offline",
        "visible_metrics",
        "balance_sample_divider",
        "cm_debug_level",
        "cm_progress_log_interval_seconds",
        "compress_cache",
    ):
        backtest.pop(key, None)
    result = {
        "bot": effective["bot"],
        "live": {
            key: effective["live"][key]
            for key in sorted(BACKTEST_LIVE_KEYS)
            if key in effective["live"]
        },
        "backtest": backtest,
        "coin_overrides": effective.get("coin_overrides", {}),
    }
    if optimize:
        options = deepcopy(effective["optimize"])
        for key in ("compress_results_file", "write_all_results", "pareto_max_size"):
            options.pop(key, None)
        result["optimize"] = options
        plan = deepcopy(get_anchor_plan(config))
        if plan is not None:
            for anchor in plan.get("anchors", []):
                for key in ("source", "label", "path"):
                    anchor.pop(key, None)
            result["anchors"] = plan
    return result


def date_span(configs) -> tuple[str, dict]:
    from utils import date_to_ts, format_end_date

    windows = [
        {
            "start_date": cfg["backtest"]["start_date"],
            "end_date": format_end_date(cfg["backtest"]["end_date"]),
        }
        for cfg in configs
    ]
    start = min(date_to_ts(window["start_date"]) for window in windows)
    end = max(date_to_ts(window["end_date"]) for window in windows)
    days = int(round((end - start) / 86_400_000))
    return f"{days}days", {"windows": windows, "envelope_days": days}


def create_session_dir(root, *, coins, source, span, setup, scenarios=0, metadata=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = utc_timestamp()
    fingerprint = setup_hash(setup)
    prefix = f"{timestamp}_{coins_label(coins)}_{safe_component(source)}_{span}_"
    if scenarios:
        prefix += f"suite-{scenarios}sc_"
    for _ in range(10):
        run_id = uuid4().hex
        directory = root / f"{prefix}setup-{fingerprint[:12]}_run-{run_id[:8]}"
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        manifest = {
            "layout_version": LAYOUT_VERSION,
            "started_at": utc_datetime(),
            "setup_sha256": fingerprint,
            "run_id": run_id,
            "setup": setup,
            **(metadata or {}),
        }
        write_json(directory / SESSION_MANIFEST, manifest)
        return directory, manifest
    raise FileExistsError("Unable to allocate a unique session directory")


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def artifact_paths(directory, relative_to):
    directory, relative_to = Path(directory), Path(relative_to)
    return {
        path.relative_to(directory).as_posix(): path.relative_to(relative_to).as_posix()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
