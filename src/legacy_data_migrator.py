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
# Include cache_base/historical_data_path so isolated caches can migrate independently.
_MIGRATION_LOGGED: Set[Tuple[str, str, str]] = set()


def migrate_legacy_data_all_on_init(
    cache_base: str = "caches/ohlcv",
    historical_data_path: str = "historical_data",
    quote: str = "USDT",
    *,
    audit_gateio_volume: bool = True,
) -> int:
    """
    Migrate legacy data for all exchanges once per process.

    This is intended to be called once globally (e.g., on first CandlestickManager init)
    and will migrate all exchanges discovered under historical_data/.

    Args:
        cache_base: Base directory for OHLCV cache
        historical_data_path: Path to legacy historical_data directory
        quote: Quote currency for building symbol paths
        audit_gateio_volume: If True, skip gateio migration due to volume differences

    Returns:
        Total number of files migrated across all exchanges
    """
    legacy_data = scan_legacy_data(historical_data_path)
    if not legacy_data:
        return 0

    total_exchanges = len(legacy_data)
    total_coins = sum(len(coins) for coins in legacy_data.values())
    total_shards = sum(len(dates) for coins in legacy_data.values() for dates in coins.values())

    logging.info(
        "[boot] Legacy data found in %s/ (%d exchanges, %d coins, %d shards). "
        "Migrating missing files to %s/",
        historical_data_path,
        total_exchanges,
        total_coins,
        total_shards,
        cache_base,
    )

    migrated_total = 0

    for exchange in sorted(legacy_data.keys()):
        key = (exchange, cache_base, historical_data_path)
        if key in _MIGRATION_LOGGED:
            continue
        _MIGRATION_LOGGED.add(key)

        if audit_gateio_volume and exchange == "gateio":
            logging.info(
                "[boot] skipping gateio legacy migration audit; gateio cache should be refreshed from remote data"
            )
            continue

        migrated, skipped = migrate_legacy_data_for_exchange(
            exchange=exchange,
            cache_base=cache_base,
            historical_data_path=historical_data_path,
            dry_run=False,
            quote=quote,
        )
        migrated_total += migrated

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

    return migrated_total


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

    key = (exchange, cache_base, historical_data_path)
    if key in _MIGRATION_LOGGED:
        return 0

    legacy_data = scan_legacy_data(historical_data_path)

    if exchange not in legacy_data:
        return 0

    total_coins = len(legacy_data[exchange])
    total_shards = sum(len(dates) for dates in legacy_data[exchange].values())

    if total_shards == 0:
        return 0

    # Log once per session
    _MIGRATION_LOGGED.add(key)
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
        logging.info(
            "[boot] skipping gateio legacy migration audit; gateio cache should be refreshed from remote data"
        )
    return migrated
