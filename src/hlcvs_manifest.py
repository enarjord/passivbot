from __future__ import annotations

import gzip
import hashlib
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from backtest_universe import effective_backtest_approved_coins_by_side
from config.access import get_optional_config_value, require_config_value
from utils import date_to_ts, format_end_date, utc_ms


HLCVS_MANIFEST_SCHEMA_VERSION = 1
HLCVS_MATERIALIZATION_SCHEMA_VERSION = 1
REQUIRED_HLCVS_MANIFEST_FILES = (
    "hlcvs",
    "btc_usd_prices",
    "timestamps",
    "coins",
    "market_specific_settings",
)


class HlcvsManifestError(RuntimeError):
    pass


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_logical_array(array: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    hasher = hashlib.sha256()
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(arr.tobytes(order="C"))
    return hasher.hexdigest()


def hash_json_value(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def load_numpy_artifact(path: Path) -> np.ndarray:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return np.load(f)
    return np.load(path)


def _array_file_entry(path: str, array: Any) -> dict[str, Any]:
    arr = np.asarray(array)
    return {
        "path": path,
        "sha256": hash_logical_array(arr),
        "shape": [int(x) for x in arr.shape],
        "dtype": str(arr.dtype),
    }


def _json_file_entry(path: str, value: Any) -> dict[str, Any]:
    return {"path": path, "sha256": hash_json_value(value)}


def _ts_from_config_date(value: Any) -> int:
    return int(date_to_ts(str(value)))


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _ccxt_version() -> str | None:
    try:
        import ccxt  # type: ignore
    except Exception:
        return None
    return str(getattr(ccxt, "__version__", "")) or None


def _side_membership(config: dict, coins: list[str]) -> dict[str, list[str]]:
    coin_set = set(coins)
    approved = effective_backtest_approved_coins_by_side(config)
    return {
        pside: sorted([coin for coin in side_coins if coin in coin_set])
        for pside, side_coins in approved.items()
    }


def build_hlcvs_manifest(
    *,
    config: dict,
    exchange: str,
    cache_hash: str,
    coins: list[str],
    hlcvs: Any,
    mss: dict,
    btc_usd_prices: Any,
    timestamps: Any | None,
    warmup_minutes: int,
    compressed: bool,
) -> dict[str, Any]:
    if timestamps is None:
        raise HlcvsManifestError("HLCV manifests require timestamps artifact data")
    ts_arr = np.asarray(timestamps)
    effective_start_ts = int(ts_arr[0]) if ts_arr.size else None
    effective_end_ts = int(ts_arr[-1]) if ts_arr.size else None

    requested_end_date = format_end_date(require_config_value(config, "backtest.end_date"))
    files = {
        "hlcvs": _array_file_entry("hlcvs.npy.gz" if compressed else "hlcvs.npy", hlcvs),
        "btc_usd_prices": _array_file_entry(
            "btc_usd_prices.npy.gz" if compressed else "btc_usd_prices.npy",
            btc_usd_prices,
        ),
        "coins": _json_file_entry("coins.json", coins),
        "market_specific_settings": _json_file_entry("market_specific_settings.json", mss),
    }
    files["timestamps"] = _array_file_entry(
        "timestamps.npy.gz" if compressed else "timestamps.npy",
        timestamps,
    )
    candidate_report = None
    if isinstance(mss, dict):
        candidate_report = (mss.get("__meta__", {}) or {}).get("candidate_report")
    if candidate_report is not None:
        files["candidate_report"] = _json_file_entry("candidate_report.json", candidate_report)

    sources = {}
    for coin in coins:
        meta = mss.get(coin, {}) if isinstance(mss, dict) else {}
        sources[coin] = {
            "ohlcv_exchange": meta.get("ohlcv_source") or meta.get("exchange"),
            "market_settings_exchange": meta.get("exchange"),
            "symbol": meta.get("symbol"),
            "first_valid_index": meta.get("first_valid_index"),
            "last_valid_index": meta.get("last_valid_index"),
        }

    meta = mss.get("__meta__", {}) if isinstance(mss, dict) else {}
    manifest = {
        "schema_version": HLCVS_MANIFEST_SCHEMA_VERSION,
        "materialization_schema_version": HLCVS_MATERIALIZATION_SCHEMA_VERSION,
        "dataset_kind": "hlcvs_data",
        "config_hash": str(cache_hash),
        "built_at": utc_ms(),
        "requested": {
            "exchange": exchange,
            "exchanges": list(require_config_value(config, "backtest.exchanges")),
            "start_ts": _ts_from_config_date(require_config_value(config, "backtest.start_date")),
            "end_ts": _ts_from_config_date(requested_end_date),
            "warmup_minutes": int(warmup_minutes),
            "gap_tolerance_ohlcvs_minutes": require_config_value(
                config, "backtest.gap_tolerance_ohlcvs_minutes"
            ),
            "ohlcv_source_dir": get_optional_config_value(config, "backtest.ohlcv_source_dir"),
        },
        "effective": {
            "coins": list(coins),
            "side_membership": _side_membership(config, coins),
            "start_ts": effective_start_ts,
            "end_ts": effective_end_ts,
        },
        "files": files,
        "sources": sources,
        "btc_benchmark": {
            "exchange": meta.get("btc_source_exchange"),
            "symbol": "BTC/USDT:USDT",
            "sha256": files["btc_usd_prices"]["sha256"],
        },
        "environment": {
            "passivbot_git_commit": _git_commit(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "ccxt": _ccxt_version(),
        },
    }
    return manifest


def write_hlcvs_manifest(cache_dir: Path, manifest: dict[str, Any]) -> Path:
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def load_hlcvs_manifest(cache_dir: Path) -> dict[str, Any] | None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open() as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise HlcvsManifestError(f"HLCV manifest must be a JSON object: {manifest_path}")
    return manifest


def manifest_has_required_schema(manifest: dict[str, Any] | None) -> bool:
    if not isinstance(manifest, dict):
        return False
    return (
        int(manifest.get("schema_version", 0) or 0) == HLCVS_MANIFEST_SCHEMA_VERSION
        and int(manifest.get("materialization_schema_version", 0) or 0)
        == HLCVS_MATERIALIZATION_SCHEMA_VERSION
    )


def _verify_array_artifact(cache_dir: Path, name: str, entry: dict[str, Any]) -> None:
    rel_path = entry.get("path")
    if not rel_path:
        raise HlcvsManifestError(f"HLCV manifest file entry {name!r} is missing path")
    path = cache_dir / str(rel_path)
    if not path.exists():
        raise HlcvsManifestError(f"HLCV manifest file is missing: {path}")
    actual = load_numpy_artifact(path)
    actual_hash = hash_logical_array(actual)
    if actual_hash != entry.get("sha256"):
        raise HlcvsManifestError(
            f"HLCV manifest hash mismatch for {name}: expected {entry.get('sha256')} got {actual_hash}"
        )
    expected_shape = entry.get("shape")
    if expected_shape is not None and [int(x) for x in actual.shape] != list(expected_shape):
        raise HlcvsManifestError(
            f"HLCV manifest shape mismatch for {name}: expected {expected_shape} got {list(actual.shape)}"
        )
    expected_dtype = entry.get("dtype")
    if expected_dtype is not None and str(actual.dtype) != str(expected_dtype):
        raise HlcvsManifestError(
            f"HLCV manifest dtype mismatch for {name}: expected {expected_dtype} got {actual.dtype}"
        )


def _verify_json_artifact(cache_dir: Path, name: str, entry: dict[str, Any]) -> None:
    rel_path = entry.get("path")
    if not rel_path:
        raise HlcvsManifestError(f"HLCV manifest file entry {name!r} is missing path")
    path = cache_dir / str(rel_path)
    if not path.exists():
        raise HlcvsManifestError(f"HLCV manifest file is missing: {path}")
    with path.open() as f:
        value = json.load(f)
    actual_hash = hash_json_value(value)
    if actual_hash != entry.get("sha256"):
        raise HlcvsManifestError(
            f"HLCV manifest hash mismatch for {name}: expected {entry.get('sha256')} got {actual_hash}"
        )


def verify_hlcvs_manifest(cache_dir: Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = load_hlcvs_manifest(cache_dir) if manifest is None else manifest
    if manifest is None:
        raise HlcvsManifestError(f"HLCV manifest missing from {cache_dir}")
    if not manifest_has_required_schema(manifest):
        raise HlcvsManifestError(f"HLCV manifest schema is missing or unsupported in {cache_dir}")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise HlcvsManifestError(f"HLCV manifest files section is missing in {cache_dir}")
    missing = [name for name in REQUIRED_HLCVS_MANIFEST_FILES if name not in files]
    if missing:
        raise HlcvsManifestError(
            f"HLCV manifest is missing required file entries in {cache_dir}: {missing}"
        )
    for name in ("hlcvs", "btc_usd_prices", "timestamps"):
        _verify_array_artifact(cache_dir, name, files[name])
    for name in ("coins", "market_specific_settings", "candidate_report"):
        if name in files:
            _verify_json_artifact(cache_dir, name, files[name])
    logging.info("[hlcvs] verified manifest for %s", cache_dir)
    return manifest
