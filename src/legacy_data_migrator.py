"""
Legacy data migration utilities.

This module provides functions to:
1. Standardize cache directory names (e.g., binanceusdm -> binance)
2. Migrate legacy data from historical_data/ to caches/ohlcv/
3. Merge duplicate symbol directories caused by inconsistent path sanitization

Migration is non-destructive: legacy data is copied (not moved) and the
historical_data/ directory is left untouched for the user to delete manually.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import sys

VOLUME_AUDIT_REFERENCE_EXCHANGES = ("binance", "bybit")

# Windows compatibility check (same logic as candlestick_manager.py)
# See: https://github.com/enarjord/passivbot/issues/547
windows_compatibility = (
    sys.platform.startswith("win") or os.environ.get("WINDOWS_COMPATIBILITY") == "1"
)


def _sanitize_symbol(symbol: str) -> str:
    """
    Convert symbol to filesystem-safe path component.

    Same logic as candlestick_manager._sanitize_symbol() to ensure consistency.
    On non-Windows: LINK/USDT:USDT -> LINK_USDT:USDT (keeps colon)
    On Windows: LINK/USDT:USDT -> LINK_USDT_USDT (replaces colon)
    """
    sanitized = symbol.replace("/", "_")
    if windows_compatibility:
        sanitized = sanitized.replace(":", "_")
    return sanitized

# Mapping from ccxt exchange IDs to standard (short) names.
# Only include entries where the ccxt ID differs from the standard name.
# IMPORTANT: Never add identity mappings (e.g., "gateio": "gateio") as this
# causes the merge logic to delete the directory after merging into itself!
CCXT_ID_TO_STANDARD = {
    "binanceusdm": "binance",
    "kucoinfutures": "kucoin",
    "krakenfutures": "kraken",
    # gateio, bybit, okx, hyperliquid use the same name in ccxt and standard
}

# Reverse mapping
STANDARD_TO_CCXT_ID = {v: k for k, v in CCXT_ID_TO_STANDARD.items()}

# Legacy directory name patterns to search for in historical_data/
LEGACY_DIR_PATTERNS = [
    "ohlcvs_binanceusdm",
    "ohlcvs_binance",
    "ohlcvs_futures",  # Old Binance futures path
    "ohlcvs_bybit",
    "ohlcvs_kucoinfutures",
    "ohlcvs_kucoin",
    "ohlcvs_okx",
    "ohlcvs_gateio",
    "ohlcvs_hyperliquid",
]


def standardize_cache_directories(cache_base: str = "caches/ohlcv", dry_run: bool = False) -> int:
    """
    Rename cache directories from ccxt IDs to standard names.

    For example:
    - caches/ohlcv/binanceusdm/ -> caches/ohlcv/binance/
    - caches/ohlcv/kucoinfutures/ -> caches/ohlcv/kucoin/

    Also removes any symlinks that were created as workarounds.

    Args:
        cache_base: Base directory for OHLCV cache (default: "caches/ohlcv")
        dry_run: If True, only log what would be done without making changes

    Returns:
        Number of directories renamed/cleaned up
    """
    base_path = Path(cache_base)
    if not base_path.exists():
        return 0

    changes = 0

    # First, remove any symlinks
    for item in base_path.iterdir():
        if item.is_symlink():
            target = os.readlink(str(item))
            if dry_run:
                logging.info("[dry-run] Would remove symlink %s -> %s", item, target)
            else:
                logging.info("Removing cache symlink %s -> %s", item, target)
                item.unlink()
            changes += 1

    # Then, rename directories from ccxt IDs to standard names
    for ccxt_id, standard_name in CCXT_ID_TO_STANDARD.items():
        # Safety: skip identity mappings to avoid merging a directory into itself
        # and then deleting it (catastrophic data loss)
        if ccxt_id == standard_name:
            continue

        ccxt_path = base_path / ccxt_id
        standard_path = base_path / standard_name

        if not ccxt_path.exists() or ccxt_path.is_symlink():
            continue

        if standard_path.exists() and not standard_path.is_symlink():
            # Both exist - need to merge
            if dry_run:
                logging.info(
                    "[dry-run] Would merge %s into %s",
                    ccxt_path, standard_path
                )
            else:
                logging.info(
                    "Merging cache directory %s into %s",
                    ccxt_path, standard_path
                )
                _merge_cache_directories(ccxt_path, standard_path)
                shutil.rmtree(ccxt_path)
            changes += 1
        else:
            # Simple rename
            if dry_run:
                logging.info(
                    "[dry-run] Would rename %s to %s",
                    ccxt_path, standard_path
                )
            else:
                logging.info(
                    "Renaming cache directory %s to %s",
                    ccxt_path, standard_path
                )
                ccxt_path.rename(standard_path)
            changes += 1

    return changes


def _merge_cache_directories(source: Path, dest: Path) -> None:
    """
    Merge source cache directory into dest, preserving newer files.

    For conflicts (same file in both), keeps the file with the newer mtime.
    """
    for item in source.rglob("*"):
        if not item.is_file():
            continue

        rel_path = item.relative_to(source)
        dest_file = dest / rel_path

        if dest_file.exists():
            # Keep newer file
            if item.stat().st_mtime > dest_file.stat().st_mtime:
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest_file)
        else:
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest_file)


def merge_duplicate_symbol_directories(
    cache_base: str = "caches/ohlcv",
    dry_run: bool = False,
) -> int:
    """
    Merge duplicate symbol directories caused by inconsistent path sanitization.

    Problem: Legacy migrator used `symbol.replace("/", "_").replace(":", "_")`
    but CandlestickManager uses `_sanitize_symbol()` which only replaces ":" on Windows.

    This results in duplicate directories like:
    - LINK_USDT_USDT (wrong - from legacy migrator)
    - LINK_USDT:USDT (correct - from CandlestickManager)

    This function finds and merges these duplicates.

    Returns:
        Number of directories merged/removed
    """
    base_path = Path(cache_base)
    if not base_path.exists():
        return 0

    merged_count = 0

    # Iterate over exchange directories
    for exchange_dir in base_path.iterdir():
        if not exchange_dir.is_dir():
            continue

        # Iterate over timeframe directories (e.g., 1m, 5m)
        for tf_dir in exchange_dir.iterdir():
            if not tf_dir.is_dir():
                continue

            # Find all symbol directories
            symbol_dirs = [d for d in tf_dir.iterdir() if d.is_dir()]

            # Group by canonical symbol (what _sanitize_symbol would produce)
            # We need to detect directories that differ only by ":" vs "_"
            groups: Dict[str, List[Path]] = {}

            for sym_dir in symbol_dirs:
                dir_name = sym_dir.name

                # Try to extract the original symbol and compute canonical path
                # Pattern: COIN_QUOTE_QUOTE or COIN_QUOTE:QUOTE
                # e.g., LINK_USDT_USDT or LINK_USDT:USDT

                # The canonical form depends on windows_compatibility
                # On non-Windows: LINK_USDT:USDT
                # On Windows: LINK_USDT_USDT

                # Normalize to a key that groups both variants together
                # Replace all : with _ for grouping purposes
                group_key = dir_name.replace(":", "_")

                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(sym_dir)

            # Process groups with duplicates
            for group_key, dirs in groups.items():
                if len(dirs) <= 1:
                    continue

                # Determine the canonical directory name
                # Reconstruct the symbol from the underscore-only form
                # e.g., LINK_USDT_USDT -> LINK/USDT:USDT -> _sanitize_symbol -> canonical

                # Find the "correct" directory (the one matching _sanitize_symbol output)
                correct_dir = None
                wrong_dirs = []

                for d in dirs:
                    # Check if this matches what _sanitize_symbol would produce
                    # The correct one will have ":" if not windows_compatibility
                    if windows_compatibility:
                        # On Windows, underscore-only is correct
                        if ":" not in d.name:
                            correct_dir = d
                        else:
                            wrong_dirs.append(d)
                    else:
                        # On non-Windows, colon version is correct
                        if ":" in d.name:
                            correct_dir = d
                        else:
                            wrong_dirs.append(d)

                if correct_dir is None:
                    # All directories use the wrong format - pick one as the target
                    # and convert its name to the correct format
                    source_dir = dirs[0]
                    wrong_dirs = dirs[1:]

                    # Reconstruct correct name: replace the last _ before USDT/USDC with :
                    # e.g., LINK_USDT_USDT -> LINK_USDT:USDT
                    correct_name = _convert_to_canonical_symbol_path(source_dir.name)
                    correct_dir = source_dir.parent / correct_name

                    if dry_run:
                        logging.info(
                            "[dry-run] Would rename %s to %s",
                            source_dir, correct_dir
                        )
                    else:
                        logging.info(
                            "[boot] Renaming symbol directory %s to %s",
                            source_dir.name, correct_name
                        )
                        source_dir.rename(correct_dir)
                    merged_count += 1

                # Merge wrong directories into correct one
                for wrong_dir in wrong_dirs:
                    if not wrong_dir.exists():
                        continue

                    if dry_run:
                        logging.info(
                            "[dry-run] Would merge %s into %s and delete",
                            wrong_dir, correct_dir
                        )
                    else:
                        logging.info(
                            "[boot] Merging duplicate symbol directory %s into %s",
                            wrong_dir.name, correct_dir.name
                        )
                        _merge_cache_directories(wrong_dir, correct_dir)
                        shutil.rmtree(wrong_dir)
                    merged_count += 1

    return merged_count


def _load_day_array(path: Path) -> Optional[np.ndarray]:
    try:
        return np.load(path)
    except Exception:
        return None


def _summarize_day_volume(arr: np.ndarray) -> Tuple[float, float]:
    """
    Return (base_volume_sum, quote_to_base_sum) for a day array.
    """
    try:
        vol = arr["bv"].astype(np.float64, copy=False)
        close = arr["c"].astype(np.float64, copy=False)
    except Exception:
        return 0.0, 0.0
    base_sum = float(np.nansum(vol))
    with np.errstate(divide="ignore", invalid="ignore"):
        base_from_quote = np.where(close > 0, vol / close, np.nan)
    quote_to_base_sum = float(np.nansum(base_from_quote[np.isfinite(base_from_quote)]))
    return base_sum, quote_to_base_sum


def _choose_volume_mode(
    base_sum: float,
    quote_to_base_sum: float,
    ref_sum: Optional[float],
    *,
    ratio_threshold: float = 3.0,
) -> Optional[str]:
    """
    Decide whether a day's volume appears to be base or quote.

    Returns:
        "base", "quote", or None if uncertain.
    """
    if ref_sum is None or ref_sum <= 0.0:
        return None
    if base_sum <= 0.0 and quote_to_base_sum <= 0.0:
        return None
    # Compare log-distance to reference to reduce scale bias.
    def log_dist(a: float, b: float) -> float:
        if a <= 0 or b <= 0:
            return float("inf")
        return abs(np.log10(a / b))

    base_dist = log_dist(base_sum, ref_sum)
    quote_dist = log_dist(quote_to_base_sum, ref_sum)

    if base_dist < quote_dist:
        # Only accept if reasonably close to reference.
        if base_sum / ref_sum < ratio_threshold and ref_sum / base_sum < ratio_threshold:
            return "base"
        return None
    if quote_dist < base_dist:
        if quote_to_base_sum / ref_sum < ratio_threshold and ref_sum / quote_to_base_sum < ratio_threshold:
            return "quote"
        return None
    return None


def audit_and_fix_gateio_volume(
    cache_base: str = "caches/ohlcv",
    *,
    dry_run: bool = True,
    reference_exchanges: Tuple[str, ...] = VOLUME_AUDIT_REFERENCE_EXCHANGES,
    max_files: Optional[int] = None,
    log_interval: int = 200,
) -> Dict[str, int]:
    """
    Audit GateIO cache for quote/base volume mix and optionally fix to base volume.

    Uses reference exchanges (binance/bybit by default) to infer whether a day
    file is base or quote volume. Only fixes when inference is confident.
    """
    stats = {
        "scanned": 0,
        "converted": 0,
        "already_base": 0,
        "skipped_no_reference": 0,
        "skipped_uncertain": 0,
        "errors": 0,
    }
    gateio_root = Path(cache_base) / "gateio" / "1m"
    if not gateio_root.exists():
        return stats

    files_processed = 0
    for sym_dir in sorted(gateio_root.iterdir()):
        if not sym_dir.is_dir():
            continue
        # Try to find a reference exchange that has the same symbol directory.
        ref_dirs = []
        for ex in reference_exchanges:
            ref_path = Path(cache_base) / ex / "1m" / sym_dir.name
            if ref_path.exists():
                ref_dirs.append(ref_path)
        for day_path in sorted(sym_dir.glob("*.npy")):
            if max_files is not None and files_processed >= max_files:
                return stats
            files_processed += 1
            stats["scanned"] += 1
            arr = _load_day_array(day_path)
            if arr is None:
                stats["errors"] += 1
                continue
            base_sum, quote_to_base_sum = _summarize_day_volume(arr)
            ref_sum = None
            for ref_dir in ref_dirs:
                ref_day = ref_dir / day_path.name
                if ref_day.exists():
                    ref_arr = _load_day_array(ref_day)
                    if ref_arr is None:
                        continue
                    ref_sum, _ = _summarize_day_volume(ref_arr)
                    if ref_sum > 0:
                        break
            mode = _choose_volume_mode(base_sum, quote_to_base_sum, ref_sum)
            if mode is None:
                if ref_sum is None:
                    stats["skipped_no_reference"] += 1
                else:
                    stats["skipped_uncertain"] += 1
                continue
            if mode == "base":
                stats["already_base"] += 1
                continue

            # mode == "quote": convert to base
            if dry_run:
                stats["converted"] += 1
            else:
                try:
                    close = arr["c"].astype(np.float64, copy=False)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        arr["bv"] = np.where(close > 0, arr["bv"] / close, arr["bv"])
                    tmp_path = day_path.with_suffix(".npy.tmp")
                    np.save(tmp_path, arr)
                    tmp_path.replace(day_path)
                    stats["converted"] += 1
                except Exception:
                    stats["errors"] += 1
            if log_interval and stats["scanned"] % log_interval == 0:
                logging.info(
                    "gateio volume audit progress | scanned=%d converted=%d base=%d no_ref=%d uncertain=%d errors=%d",
                    stats["scanned"],
                    stats["converted"],
                    stats["already_base"],
                    stats["skipped_no_reference"],
                    stats["skipped_uncertain"],
                    stats["errors"],
                )

    return stats


def normalize_ccxt_volume_to_base(
    exchange_id: str, close: float, volume: float
) -> float:
    """
    Normalize ccxt OHLCV volume to base volume.

    Some exchanges (notably gateio swap) report quote-volume in the ccxt OHLCV
    "volume" field. For those exchanges, divide by close to get base.
    """
    exid = str(exchange_id).lower()
    if exid == "gateio" and close > 0:
        return float(volume) / float(close)
    return float(volume)

def _convert_to_canonical_symbol_path(dir_name: str) -> str:
    """
    Convert a symbol directory name to canonical format.

    On non-Windows: LINK_USDT_USDT -> LINK_USDT:USDT
    On Windows: keeps as-is (underscore only)

    Handles patterns like:
    - COIN_QUOTE_QUOTE (e.g., LINK_USDT_USDT)
    - 1000COIN_QUOTE_QUOTE (e.g., 1000PEPE_USDT_USDT)
    """
    if windows_compatibility:
        return dir_name

    # Pattern: everything up to the last occurrence of _USDT or _USDC,
    # then replace the underscore before the final quote with :
    # e.g., LINK_USDT_USDT -> LINK_USDT:USDT
    #       BTC_USDC_USDC -> BTC_USDC:USDC

    for quote in ["USDT", "USDC"]:
        suffix = f"_{quote}_{quote}"
        if dir_name.endswith(suffix):
            base = dir_name[:-len(suffix)]
            return f"{base}_{quote}:{quote}"

    # Fallback: no change if pattern doesn't match
    return dir_name


def get_legacy_exchange_name(dir_name: str) -> Optional[str]:
    """
    Extract exchange name from legacy directory name.

    Examples:
    - "ohlcvs_binanceusdm" -> "binance"
    - "ohlcvs_bybit" -> "bybit"
    - "ohlcvs_futures" -> "binance" (old Binance futures path)
    """
    if not dir_name.startswith("ohlcvs_"):
        return None

    suffix = dir_name[7:]  # Remove "ohlcvs_" prefix

    # Special case for old Binance futures path
    if suffix == "futures":
        return "binance"

    # Check if it's a ccxt ID that needs standardization
    if suffix in CCXT_ID_TO_STANDARD:
        return CCXT_ID_TO_STANDARD[suffix]

    return suffix


def scan_legacy_data(
    historical_data_path: str = "historical_data",
) -> Dict[str, Dict[str, List[str]]]:
    """
    Scan historical_data/ for legacy OHLCV shards.

    Returns:
        Dict mapping exchange -> coin -> list of date strings (YYYY-MM-DD)
    """
    result: Dict[str, Dict[str, List[str]]] = {}
    base = Path(historical_data_path)

    if not base.exists():
        return result

    for legacy_dir in base.iterdir():
        if not legacy_dir.is_dir():
            continue

        exchange = get_legacy_exchange_name(legacy_dir.name)
        if exchange is None:
            continue

        if exchange not in result:
            result[exchange] = {}

        # Scan for coin directories
        for coin_dir in legacy_dir.iterdir():
            if not coin_dir.is_dir():
                continue

            coin = coin_dir.name
            dates = []

            # Find all .npy shard files
            for shard_file in coin_dir.glob("*.npy"):
                # Extract date from filename (YYYY-MM-DD.npy)
                date_str = shard_file.stem
                if len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-":
                    dates.append(date_str)

            if dates:
                result[exchange][coin] = sorted(dates)

    return result


def migrate_legacy_data_for_exchange(
    exchange: str,
    cache_base: str = "caches/ohlcv",
    historical_data_path: str = "historical_data",
    dry_run: bool = False,
    quote: str = "USDT",
) -> Tuple[int, int]:
    """
    Migrate legacy data for a specific exchange.

    Args:
        exchange: Standard exchange name (e.g., "binance")
        cache_base: Base directory for OHLCV cache
        historical_data_path: Path to legacy historical_data directory
        dry_run: If True, only log what would be done
        quote: Quote currency for building symbol paths

    Returns:
        Tuple of (files_migrated, files_skipped)
    """
    # Import CANDLE_DTYPE here to avoid circular imports
    from candlestick_manager import CANDLE_DTYPE
    import time

    migrated = 0
    skipped = 0

    legacy_data = scan_legacy_data(historical_data_path)

    if exchange not in legacy_data:
        return migrated, skipped

    # Count total shards for progress reporting
    total_shards = sum(len(dates) for dates in legacy_data[exchange].values())
    processed = 0
    last_log_time = time.monotonic()
    log_interval_seconds = 10.0  # Log progress every 10 seconds

    for coin, dates in legacy_data[exchange].items():
        symbol = f"{coin}/{quote}:{quote}"
        safe_symbol = _sanitize_symbol(symbol)

        for date_str in dates:
            processed += 1

            # Log progress periodically
            now = time.monotonic()
            if now - last_log_time >= log_interval_seconds:
                pct = int(100 * processed / total_shards) if total_shards > 0 else 0
                logging.info(
                    "[boot] Migration progress for %s: %d%% (%d/%d shards, %d migrated, %d skipped)",
                    exchange, pct, processed, total_shards, migrated, skipped
                )
                last_log_time = now
            # Build target path
            target_path = Path(cache_base) / exchange / "1m" / safe_symbol / f"{date_str}.npy"

            if target_path.exists():
                skipped += 1
                continue

            # Find source file(s)
            source_paths = _find_legacy_source_paths(
                exchange, coin, date_str, historical_data_path
            )

            if not source_paths:
                continue

            # Use first valid source
            source_data = None
            for source_path in source_paths:
                if not os.path.exists(source_path):
                    continue
                try:
                    source_data = _load_and_convert_legacy_shard(source_path, CANDLE_DTYPE)
                    if source_data is not None and len(source_data) > 0:
                        break
                except Exception as e:
                    logging.debug("Failed to load %s: %s", source_path, e)
                    continue

            if source_data is None or len(source_data) == 0:
                continue

            if dry_run:
                logging.info(
                    "[dry-run] Would migrate %s/%s/%s (%d candles)",
                    exchange, coin, date_str, len(source_data)
                )
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                # Atomic write: write to temp, then rename
                # Use .tmp.npy suffix so numpy doesn't add another .npy extension
                tmp_path = target_path.with_suffix(".tmp.npy")
                np.save(tmp_path, source_data)
                tmp_path.rename(target_path)
                logging.debug(
                    "Migrated %s/%s/%s (%d candles)",
                    exchange, coin, date_str, len(source_data)
                )

            migrated += 1

    return migrated, skipped


def _find_legacy_source_paths(
    exchange: str, coin: str, date_str: str, historical_data_path: str
) -> List[str]:
    """
    Find potential source paths for a legacy shard.

    Returns list of candidate paths, ordered by preference.
    """
    paths = []
    base = Path(historical_data_path)

    # Try various legacy path patterns
    patterns = [
        f"ohlcvs_{exchange}",
        f"ohlcvs_{STANDARD_TO_CCXT_ID.get(exchange, exchange)}",
    ]

    # Special case for Binance
    if exchange == "binance":
        patterns.extend(["ohlcvs_binanceusdm", "ohlcvs_futures"])

    for pattern in patterns:
        candidate = base / pattern / coin / f"{date_str}.npy"
        if candidate.exists():
            paths.append(str(candidate))

    return paths


def _load_and_convert_legacy_shard(path: str, candle_dtype) -> Optional[np.ndarray]:
    """
    Load a legacy .npy shard and convert to CANDLE_DTYPE.

    Legacy format: unstructured array with columns [ts, o, h, l, c, volume]
    New format: structured array with CANDLE_DTYPE
    """
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception:
        return None

    if arr is None or arr.size == 0:
        return None

    # Check if already in correct dtype
    if arr.dtype == candle_dtype:
        return arr

    # Convert from legacy format
    if arr.ndim == 2 and arr.shape[1] >= 6:
        result = np.empty(arr.shape[0], dtype=candle_dtype)
        result["ts"] = arr[:, 0].astype(np.int64)
        result["o"] = arr[:, 1].astype(np.float32)
        result["h"] = arr[:, 2].astype(np.float32)
        result["l"] = arr[:, 3].astype(np.float32)
        result["c"] = arr[:, 4].astype(np.float32)
        result["bv"] = arr[:, 5].astype(np.float32)
        return result

    return None


# Track whether migration message has been logged this session
_MIGRATION_LOGGED: Set[str] = set()


def migrate_legacy_data_on_init(
    exchange: str,
    cache_base: str = "caches/ohlcv",
    historical_data_path: str = "historical_data",
    quote: str = "USDT",
    *,
    audit_gateio_volume: bool = True,
) -> int:
    """
    Check for and migrate legacy data on CandlestickManager initialization.

    This is called once per exchange per session. It:
    1. Logs a message if legacy data exists
    2. Copies missing data to the new cache location
    3. Leaves historical_data/ untouched

    Args:
        exchange: Standard exchange name
        cache_base: Base directory for OHLCV cache
        historical_data_path: Path to legacy historical_data directory
        quote: Quote currency

    Returns:
        Number of files migrated
    """
    global _MIGRATION_LOGGED

    if exchange in _MIGRATION_LOGGED:
        return 0

    legacy_data = scan_legacy_data(historical_data_path)

    if exchange not in legacy_data:
        return 0

    total_coins = len(legacy_data[exchange])
    total_shards = sum(len(dates) for dates in legacy_data[exchange].values())

    if total_shards == 0:
        return 0

    # Log once per session
    _MIGRATION_LOGGED.add(exchange)
    logging.info(
        "[boot] Legacy data found in %s/ohlcvs_%s/ (%d coins, %d shards). "
        "Migrating missing files to %s/",
        historical_data_path,
        exchange,
        total_coins,
        total_shards,
        cache_base,
    )

    migrated, skipped = migrate_legacy_data_for_exchange(
        exchange=exchange,
        cache_base=cache_base,
        historical_data_path=historical_data_path,
        dry_run=False,
        quote=quote,
    )

    if migrated > 0:
        logging.info(
            "[boot] Migrated %d legacy shards for %s (%d already existed). "
            "You may safely delete %s/ohlcvs_%s/ to save disk space.",
            migrated,
            exchange,
            skipped,
            historical_data_path,
            exchange,
        )
    elif skipped > 0:
        logging.info(
            "[boot] Legacy data for %s already migrated (%d shards). "
            "You may safely delete %s/ohlcvs_%s/ to save disk space.",
            exchange,
            skipped,
            historical_data_path,
            exchange,
        )

    if audit_gateio_volume and exchange == "gateio":
        stats = audit_and_fix_gateio_volume(cache_base=cache_base, dry_run=True)
        if stats["scanned"] > 0:
            logging.info(
                "[boot] gateio volume audit (dry-run) | scanned=%d converted=%d base=%d no_ref=%d uncertain=%d errors=%d",
                stats["scanned"],
                stats["converted"],
                stats["already_base"],
                stats["skipped_no_reference"],
                stats["skipped_uncertain"],
                stats["errors"],
            )
    return migrated
