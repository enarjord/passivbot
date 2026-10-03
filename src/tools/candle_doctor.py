"""
Candlestick Doctor: audit and repair OHLCV shard caches.

Scans ``caches/ohlcv/{exchange}/{timeframe}/{symbol}/`` directories for
corrupted files, stale index entries, legacy formats, and data anomalies.

Examples
--------
Report-only scan of all caches::

    python -m src.tools.candle_doctor --progress

Filtered scan with JSON output::

    python -m src.tools.candle_doctor --exchange binance --json

Apply automatic fixes::

    python -m src.tools.candle_doctor --fix --progress
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add src/ to path so we can import candlestick_manager
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from candlestick_manager import CANDLE_DTYPE, ONE_MIN_MS  # noqa: E402

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

SHARD_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class Issue:
    exchange: str
    timeframe: str
    symbol: str
    shard: str  # date key or "index.json"
    check: str  # machine-readable check name
    severity: str  # "error" | "warning"
    message: str
    fixable: bool
    fixed: bool = False


@dataclass
class DoctorSummary:
    exchanges_scanned: int
    symbols_scanned: int
    shards_scanned: int
    issues_found: int
    issues_fixed: int
    by_check: Dict[str, int] = field(default_factory=dict)
    by_severity: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Atomic write helpers (mirrors candlestick_manager patterns)
# ---------------------------------------------------------------------------


def _atomic_save_npy(path: str, arr: np.ndarray) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_save_json(path: str, data: dict) -> None:
    payload = json.dumps(data, sort_keys=True).encode("utf-8")
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _compute_crc(arr: np.ndarray) -> int:
    """CRC32 matching candlestick_manager: sorted by ts, then tobytes."""
    sorted_arr = np.sort(arr, order="ts")
    return int(zlib.crc32(sorted_arr.tobytes()) & 0xFFFFFFFF)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_corrupted(path: str) -> Optional[tuple]:
    """Try to load a .npy file. Returns (arr, None) on success or (None, msg) on failure."""
    try:
        with open(path, "rb") as f:
            arr = np.load(f, allow_pickle=False)
    except Exception as exc:
        return None, f"Failed to load: {exc}"
    if arr.ndim == 0:
        return None, "Scalar array (ndim=0)"
    return arr, None


def check_wrong_format(arr: np.ndarray) -> bool:
    """Return True if array is legacy 2D float format, not CANDLE_DTYPE."""
    if arr.dtype == CANDLE_DTYPE:
        return False
    if arr.ndim == 2 and arr.shape[1] >= 6:
        return True
    return False


def convert_legacy_to_dtype(arr: np.ndarray) -> np.ndarray:
    """Convert a legacy 2D float array to CANDLE_DTYPE structured array."""
    raw = np.asarray(arr[:, :6], dtype=np.float64)
    out = np.empty((raw.shape[0],), dtype=CANDLE_DTYPE)
    out["ts"] = raw[:, 0].astype(np.int64)
    out["o"] = raw[:, 1].astype(np.float32)
    out["h"] = raw[:, 2].astype(np.float32)
    out["l"] = raw[:, 3].astype(np.float32)
    out["c"] = raw[:, 4].astype(np.float32)
    out["bv"] = raw[:, 5].astype(np.float32)
    return out


def check_timestamp_alignment(arr: np.ndarray) -> np.ndarray:
    """Return mask of rows where ts is not aligned to minute boundaries."""
    ts = arr["ts"]
    return (ts % ONE_MIN_MS) != 0


def check_duplicate_timestamps(arr: np.ndarray) -> bool:
    """Return True if any timestamp is duplicated."""
    ts = arr["ts"]
    return len(np.unique(ts)) < len(ts)


def check_shard_date_mismatch(arr: np.ndarray, date_key: str) -> bool:
    """Return True if any timestamps fall outside the UTC day indicated by filename."""
    try:
        day_start = int(
            datetime.strptime(date_key, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000
        )
    except ValueError:
        return True
    day_end = day_start + 24 * 60 * 60 * 1000  # exclusive
    ts = arr["ts"]
    return bool(np.any(ts < day_start) or np.any(ts >= day_end))


def check_ohlc_sanity(arr: np.ndarray) -> List[tuple]:
    """Return list of (severity, message) for OHLC anomalies."""
    issues = []
    for col in ("o", "h", "l", "c", "bv"):
        if np.any(np.isnan(arr[col])):
            issues.append(("error", f"NaN values in {col}"))
        if np.any(np.isinf(arr[col])):
            issues.append(("error", f"Inf values in {col}"))
    h, l, c = arr["h"], arr["l"], arr["c"]
    if np.any(h < l):
        issues.append(("warning", "high < low detected"))
    # close outside [low, high] range (allow float32 epsilon)
    eps = np.float32(1e-7)
    if np.any((c < l - eps) | (c > h + eps)):
        issues.append(("warning", "close outside [low, high] range"))
    if np.any(arr["o"] == 0) or np.any(arr["c"] == 0):
        issues.append(("warning", "Zero price detected"))
    return issues


def check_gap_inside_shard(arr: np.ndarray) -> Optional[str]:
    """Check for non-continuous minute spacing within a shard."""
    if arr.size < 2:
        return None
    ts = np.sort(arr["ts"])
    diffs = np.diff(ts)
    non_one_min = diffs[diffs != ONE_MIN_MS]
    if len(non_one_min) > 0:
        gap_count = len(non_one_min)
        max_gap_min = int(np.max(non_one_min)) // ONE_MIN_MS
        return f"{gap_count} gap(s), max {max_gap_min} min"
    return None


def check_crc_mismatch(arr: np.ndarray, stored_crc: int) -> Optional[str]:
    """Return message if CRC doesn't match, None if ok."""
    computed = _compute_crc(arr)
    if computed != stored_crc:
        return f"CRC stored={stored_crc} computed={computed}"
    return None


# ---------------------------------------------------------------------------
# Fix operations
# ---------------------------------------------------------------------------


def fix_corrupted(path: str, index: dict, date_key: str) -> None:
    """Delete corrupted .npy and remove from index."""
    try:
        os.unlink(path)
    except OSError:
        pass
    shards = index.get("shards", {})
    shards.pop(date_key, None)


def fix_wrong_format(path: str, arr: np.ndarray) -> np.ndarray:
    """Convert legacy array and save atomically. Returns the converted array."""
    converted = convert_legacy_to_dtype(arr)
    converted = np.sort(converted, order="ts")
    _atomic_save_npy(path, converted)
    return converted


def fix_crc(index: dict, date_key: str, arr: np.ndarray) -> None:
    """Recompute CRC and update index."""
    new_crc = _compute_crc(arr)
    shards = index.get("shards", {})
    if date_key in shards:
        shards[date_key]["crc32"] = new_crc


def fix_index_orphan_entries(index: dict, symbol_dir: str) -> int:
    """Remove index entries whose .npy files don't exist. Returns count removed."""
    shards = index.get("shards", {})
    to_remove = []
    for dk, meta in list(shards.items()):
        if not isinstance(meta, dict):
            to_remove.append(dk)
            continue
        path = meta.get("path")
        if not path:
            path = os.path.join(symbol_dir, f"{dk}.npy")
        if not os.path.exists(path):
            to_remove.append(dk)
    for dk in to_remove:
        shards.pop(dk, None)
    return len(to_remove)


def fix_index_missing_entries(
    index: dict, symbol_dir: str, npy_files: Dict[str, str], loaded_arrays: Dict[str, np.ndarray]
) -> int:
    """Add index entries for .npy files not in index. Returns count added."""
    shards = index.setdefault("shards", {})
    added = 0
    for dk, path in npy_files.items():
        if dk not in shards:
            arr = loaded_arrays.get(dk)
            if arr is not None and arr.size > 0:
                shards[dk] = {
                    "path": path,
                    "min_ts": int(arr["ts"].min()),
                    "max_ts": int(arr["ts"].max()),
                    "count": int(arr.size),
                    "crc32": _compute_crc(arr),
                }
                added += 1
    return added


def fix_timestamp_alignment(path: str, arr: np.ndarray) -> np.ndarray:
    """Floor timestamps to minute boundary, dedup, resave."""
    arr["ts"] = (arr["ts"] // ONE_MIN_MS) * ONE_MIN_MS
    # dedup keeping last occurrence
    _, idx = np.unique(arr["ts"][::-1], return_index=True)
    arr = arr[::-1][idx][::-1]
    arr = np.sort(arr, order="ts")
    _atomic_save_npy(path, arr)
    return arr


def fix_duplicate_timestamps(path: str, arr: np.ndarray) -> np.ndarray:
    """Dedup keeping last occurrence per timestamp, resave."""
    _, idx = np.unique(arr["ts"][::-1], return_index=True)
    arr = arr[::-1][idx][::-1]
    arr = np.sort(arr, order="ts")
    _atomic_save_npy(path, arr)
    return arr


def fix_ohlc_delete(path: str, index: dict, date_key: str) -> None:
    """Delete shard with NaN/Inf (will be re-fetched)."""
    try:
        os.unlink(path)
    except OSError:
        pass
    shards = index.get("shards", {})
    shards.pop(date_key, None)


# ---------------------------------------------------------------------------
# Symbol scanner
# ---------------------------------------------------------------------------


def scan_symbol(
    symbol_dir: str,
    exchange: str,
    timeframe: str,
    symbol: str,
    do_fix: bool,
) -> tuple:
    """Scan one symbol directory. Returns (issues, shards_scanned)."""
    issues: List[Issue] = []
    shards_scanned = 0

    # Discover .npy files
    npy_files: Dict[str, str] = {}  # date_key -> path
    try:
        for entry in os.scandir(symbol_dir):
            if entry.is_file() and entry.name.endswith(".npy"):
                stem = entry.name[:-4]
                if SHARD_DATE_RE.match(stem):
                    npy_files[stem] = entry.path
    except OSError:
        return issues, 0

    # Load index.json
    index_path = os.path.join(symbol_dir, "index.json")
    index: dict = {}
    if os.path.exists(index_path):
        try:
            with open(index_path, "r") as f:
                index = json.load(f)
        except (json.JSONDecodeError, OSError):
            index = {}

    index_modified = False

    # --- Check index consistency ---
    shards_meta = index.get("shards", {})

    # Orphan index entries (reference missing files)
    orphan_keys = []
    for dk, meta in list(shards_meta.items()):
        if not isinstance(meta, dict):
            orphan_keys.append(dk)
            continue
        path = meta.get("path")
        if not path:
            path = os.path.join(symbol_dir, f"{dk}.npy")
        if not os.path.exists(path):
            orphan_keys.append(dk)

    if orphan_keys:
        issue = Issue(
            exchange=exchange,
            timeframe=timeframe,
            symbol=symbol,
            shard="index.json",
            check="index_inconsistency",
            severity="error",
            message=f"{len(orphan_keys)} orphan index entries (missing files)",
            fixable=True,
        )
        if do_fix:
            removed = fix_index_orphan_entries(index, symbol_dir)
            if removed:
                issue.fixed = True
                index_modified = True
        issues.append(issue)

    # Unindexed files (files exist without index entry)
    unindexed = [dk for dk in npy_files if dk not in shards_meta]

    # We'll populate this as we load arrays for unindexed entry fix
    loaded_arrays: Dict[str, np.ndarray] = {}

    # Collect volume data for suspicious_volume check
    all_bv: List[np.ndarray] = []
    all_close: List[np.ndarray] = []

    # --- Per-shard checks ---
    for date_key, npy_path in sorted(npy_files.items()):
        shards_scanned += 1

        # Try load
        result = check_corrupted(npy_path)
        raw_arr, err_msg = result

        if err_msg is not None:
            issue = Issue(
                exchange=exchange,
                timeframe=timeframe,
                symbol=symbol,
                shard=date_key,
                check="corrupted_file",
                severity="error",
                message=err_msg,
                fixable=True,
            )
            if do_fix:
                fix_corrupted(npy_path, index, date_key)
                issue.fixed = True
                index_modified = True
            issues.append(issue)
            continue

        arr = raw_arr

        # Check wrong format (legacy 2D float64)
        if check_wrong_format(arr):
            issue = Issue(
                exchange=exchange,
                timeframe=timeframe,
                symbol=symbol,
                shard=date_key,
                check="wrong_format",
                severity="error",
                message=f"Legacy 2D float64 array shape={arr.shape}",
                fixable=True,
            )
            if do_fix:
                arr = fix_wrong_format(npy_path, arr)
                issue.fixed = True
                index_modified = True
            issues.append(issue)

        # Ensure we have CANDLE_DTYPE from here on
        if arr.dtype != CANDLE_DTYPE:
            # If not fixable wrong_format and not CANDLE_DTYPE, skip further checks
            if arr.ndim == 2 and arr.shape[1] >= 6:
                arr = convert_legacy_to_dtype(arr)
            else:
                continue

        if arr.size == 0:
            continue

        loaded_arrays[date_key] = arr

        # Check timestamp alignment
        misaligned = check_timestamp_alignment(arr)
        if np.any(misaligned):
            count = int(np.sum(misaligned))
            issue = Issue(
                exchange=exchange,
                timeframe=timeframe,
                symbol=symbol,
                shard=date_key,
                check="timestamp_misalignment",
                severity="error",
                message=f"{count} timestamps not aligned to minute boundary",
                fixable=True,
            )
            if do_fix:
                arr = fix_timestamp_alignment(npy_path, arr)
                loaded_arrays[date_key] = arr
                issue.fixed = True
                index_modified = True
            issues.append(issue)

        # Check duplicate timestamps
        if check_duplicate_timestamps(arr):
            n_unique = len(np.unique(arr["ts"]))
            issue = Issue(
                exchange=exchange,
                timeframe=timeframe,
                symbol=symbol,
                shard=date_key,
                check="duplicate_timestamps",
                severity="error",
                message=f"{arr.size - n_unique} duplicate timestamps",
                fixable=True,
            )
            if do_fix:
                arr = fix_duplicate_timestamps(npy_path, arr)
                loaded_arrays[date_key] = arr
                issue.fixed = True
                index_modified = True
            issues.append(issue)

        # Check shard date mismatch
        if check_shard_date_mismatch(arr, date_key):
            ts_min = int(arr["ts"].min())
            ts_max = int(arr["ts"].max())
            issue = Issue(
                exchange=exchange,
                timeframe=timeframe,
                symbol=symbol,
                shard=date_key,
                check="shard_date_mismatch",
                severity="error",
                message=f"Timestamps outside {date_key} UTC day (ts range {ts_min}-{ts_max})",
                fixable=False,
            )
            issues.append(issue)

        # Check OHLC sanity
        ohlc_issues = check_ohlc_sanity(arr)
        has_nan_inf = any("NaN" in msg or "Inf" in msg for _, msg in ohlc_issues)
        for severity, msg in ohlc_issues:
            fixable = "NaN" in msg or "Inf" in msg
            issue = Issue(
                exchange=exchange,
                timeframe=timeframe,
                symbol=symbol,
                shard=date_key,
                check="ohlc_sanity",
                severity=severity,
                message=msg,
                fixable=fixable,
            )
            if do_fix and fixable and has_nan_inf:
                # Only delete once for all NaN/Inf issues in same shard
                pass
            issues.append(issue)

        if do_fix and has_nan_inf:
            fix_ohlc_delete(npy_path, index, date_key)
            index_modified = True
            # Mark all NaN/Inf issues as fixed
            for iss in issues:
                if (
                    iss.shard == date_key
                    and iss.check == "ohlc_sanity"
                    and iss.fixable
                    and not iss.fixed
                ):
                    iss.fixed = True
            # Don't run further checks on deleted shard
            loaded_arrays.pop(date_key, None)
            continue

        # Check CRC mismatch (if index entry exists)
        shard_meta = shards_meta.get(date_key)
        if shard_meta and isinstance(shard_meta, dict) and "crc32" in shard_meta:
            crc_msg = check_crc_mismatch(arr, shard_meta["crc32"])
            if crc_msg is not None:
                issue = Issue(
                    exchange=exchange,
                    timeframe=timeframe,
                    symbol=symbol,
                    shard=date_key,
                    check="crc_mismatch",
                    severity="error",
                    message=crc_msg,
                    fixable=True,
                )
                if do_fix:
                    fix_crc(index, date_key, arr)
                    issue.fixed = True
                    index_modified = True
                issues.append(issue)

        # Check gaps inside shard
        gap_msg = check_gap_inside_shard(arr)
        if gap_msg is not None:
            issues.append(
                Issue(
                    exchange=exchange,
                    timeframe=timeframe,
                    symbol=symbol,
                    shard=date_key,
                    check="gap_inside_shard",
                    severity="warning",
                    message=gap_msg,
                    fixable=False,
                )
            )

        # Accumulate volume data
        all_bv.append(arr["bv"])
        all_close.append(arr["c"])

    # --- Fix unindexed files ---
    if unindexed:
        issue = Issue(
            exchange=exchange,
            timeframe=timeframe,
            symbol=symbol,
            shard="index.json",
            check="index_inconsistency",
            severity="error",
            message=f"{len(unindexed)} files without index entries",
            fixable=True,
        )
        if do_fix:
            added = fix_index_missing_entries(index, symbol_dir, npy_files, loaded_arrays)
            if added:
                issue.fixed = True
                index_modified = True
        issues.append(issue)

    # --- Check suspicious volume (across all shards) ---
    if all_bv and all_close:
        bv_cat = np.concatenate(all_bv)
        close_cat = np.concatenate(all_close)
        # Filter out zeros
        mask = close_cat > 0
        if np.any(mask):
            med_bv = float(np.median(bv_cat[mask]))
            med_close = float(np.median(close_cat[mask]))
            if med_close > 0 and med_bv / med_close > 100:
                issues.append(
                    Issue(
                        exchange=exchange,
                        timeframe=timeframe,
                        symbol=symbol,
                        shard="*",
                        check="suspicious_volume",
                        severity="warning",
                        message=f"median(bv)/median(close) = {med_bv/med_close:.1f} (>100, likely quote-vol)",
                        fixable=False,
                    )
                )

    # --- Save modified index ---
    if do_fix and index_modified:
        _atomic_save_json(index_path, index)

    return issues, shards_scanned


# ---------------------------------------------------------------------------
# Top-level scanner
# ---------------------------------------------------------------------------


def scan_all(
    cache_dir: str,
    do_fix: bool,
    exchange_filter: Optional[str],
    symbol_filter: Optional[str],
    show_progress: bool,
) -> tuple:
    """Scan all caches. Returns (all_issues, summary)."""
    ohlcv_root = os.path.join(cache_dir, "ohlcv")
    if not os.path.isdir(ohlcv_root):
        print(f"No OHLCV cache directory found at {ohlcv_root}")
        return [], DoctorSummary(0, 0, 0, 0, 0)

    # Collect work items: (exchange, timeframe, symbol, symbol_dir)
    work: List[tuple] = []
    exchanges = set()

    for exchange_entry in sorted(os.scandir(ohlcv_root), key=lambda e: e.name):
        if not exchange_entry.is_dir():
            continue
        ex_name = exchange_entry.name
        if exchange_filter and ex_name != exchange_filter:
            continue
        exchanges.add(ex_name)

        for tf_entry in sorted(os.scandir(exchange_entry.path), key=lambda e: e.name):
            if not tf_entry.is_dir():
                continue
            tf_name = tf_entry.name

            for sym_entry in sorted(os.scandir(tf_entry.path), key=lambda e: e.name):
                if not sym_entry.is_dir():
                    continue
                sym_name = sym_entry.name
                if symbol_filter and sym_name != symbol_filter:
                    continue
                work.append((ex_name, tf_name, sym_name, sym_entry.path))

    iterator = work
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(work, desc="Scanning symbols", unit="sym")
        except ImportError:
            pass

    all_issues: List[Issue] = []
    total_shards = 0
    symbols_scanned = 0

    for ex_name, tf_name, sym_name, sym_dir in iterator:
        symbol_issues, n_shards = scan_symbol(sym_dir, ex_name, tf_name, sym_name, do_fix)
        all_issues.extend(symbol_issues)
        total_shards += n_shards
        symbols_scanned += 1

    # Build summary
    by_check: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}
    fixed_count = 0

    for iss in all_issues:
        by_check[iss.check] = by_check.get(iss.check, 0) + 1
        by_severity[iss.severity] = by_severity.get(iss.severity, 0) + 1
        if iss.fixed:
            fixed_count += 1

    summary = DoctorSummary(
        exchanges_scanned=len(exchanges),
        symbols_scanned=symbols_scanned,
        shards_scanned=total_shards,
        issues_found=len(all_issues),
        issues_fixed=fixed_count,
        by_check=by_check,
        by_severity=by_severity,
    )

    return all_issues, summary


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_human(all_issues: List[Issue], summary: DoctorSummary) -> None:
    # Count fixed per check for summary line
    fixed_per_check: Dict[str, int] = {}
    for iss in all_issues:
        if iss.fixed:
            fixed_per_check[iss.check] = fixed_per_check.get(iss.check, 0) + 1

    for iss in all_issues:
        tag = "FIXED" if iss.fixed else "FOUND"
        loc = f"{iss.exchange}/{iss.timeframe}/{iss.symbol}/{iss.shard}"
        print(f"[{tag}] {loc} :: {iss.check} :: {iss.message}")

    print()
    print("=== Doctor Summary ===")
    print(
        f"Exchanges: {summary.exchanges_scanned}  "
        f"Symbols: {summary.symbols_scanned}  "
        f"Shards: {summary.shards_scanned}"
    )
    print(f"Issues found: {summary.issues_found}  Fixed: {summary.issues_fixed}")
    for check, count in sorted(summary.by_check.items()):
        fixed = fixed_per_check.get(check, 0)
        suffix = f" ({fixed} fixed)" if fixed else ""
        print(f"  {check}: {count}{suffix}")


def print_json(all_issues: List[Issue], summary: DoctorSummary) -> None:
    output = {
        "summary": asdict(summary),
        "issues": [asdict(iss) for iss in all_issues],
    }
    print(json.dumps(output, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Candlestick Doctor: audit and repair OHLCV shard caches.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to repair fixable issues (default: report only).",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default=None,
        help="Only scan this exchange.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Only scan this symbol (directory name, e.g. BTC_USDT:USDT).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm progress bar.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Machine-readable JSON output.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="caches",
        help="Root cache directory (default: caches).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    all_issues, summary = scan_all(
        cache_dir=args.cache_dir,
        do_fix=args.fix,
        exchange_filter=args.exchange,
        symbol_filter=args.symbol,
        show_progress=args.progress,
    )

    if args.json_output:
        print_json(all_issues, summary)
    else:
        print_human(all_issues, summary)


if __name__ == "__main__":
    main()
