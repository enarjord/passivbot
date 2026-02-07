#!/usr/bin/env python3
"""
Migrate historical_data/ to caches/ohlcv/ structure.

This optional utility converts legacy downloader data to the CandlestickManager format.

Usage:
    python src/tools/migrate_historical_data.py --exchange binanceusdm --dry-run
    python src/tools/migrate_historical_data.py --exchange binanceusdm --execute
    python src/tools/migrate_historical_data.py --exchange binanceusdm --execute --delete-legacy
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from candlestick_manager import CANDLE_DTYPE, ONE_MIN_MS
from utils import get_quote, to_ccxt_exchange_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def coin_to_symbol(coin: str, exchange: str) -> str:
    """Convert legacy coin name to full symbol format."""
    quote = get_quote(exchange)
    return f"{coin}/{quote}:{quote}"


def symbol_to_safe_path(symbol: str) -> str:
    """Convert symbol to safe filesystem path component."""
    return symbol.replace("/", "_").replace(":", "_")


def load_legacy_shard(path: str) -> Optional[np.ndarray]:
    """Load a legacy .npy shard and convert to CANDLE_DTYPE."""
    try:
        arr = np.load(path, allow_pickle=False)
        if isinstance(arr, np.ndarray) and arr.dtype == CANDLE_DTYPE:
            return arr
        # Legacy format: 2D float array [timestamp, open, high, low, close, volume]
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 6:
            raw = np.asarray(arr[:, :6], dtype=np.float64)
            out = np.empty((raw.shape[0],), dtype=CANDLE_DTYPE)
            out["ts"] = raw[:, 0].astype(np.int64)
            out["o"] = raw[:, 1].astype(np.float32)
            out["h"] = raw[:, 2].astype(np.float32)
            out["l"] = raw[:, 3].astype(np.float32)
            out["c"] = raw[:, 4].astype(np.float32)
            out["bv"] = raw[:, 5].astype(np.float32)
            return out
        logging.warning(f"Unknown format in {path}: shape={arr.shape}, dtype={arr.dtype}")
        return None
    except Exception as e:
        logging.warning(f"Failed to load {path}: {e}")
        return None


def scan_legacy_data(exchange: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Scan historical_data/ for legacy shards.

    Returns:
        Dict of coin -> [(date_key, path), ...]
    """
    legacy_dir = Path(f"historical_data/ohlcvs_{exchange}")
    if not legacy_dir.exists():
        return {}

    result: Dict[str, List[Tuple[str, str]]] = {}
    for coin_dir in legacy_dir.iterdir():
        if not coin_dir.is_dir():
            continue
        coin = coin_dir.name
        shards = []
        for npy_file in coin_dir.glob("*.npy"):
            name = npy_file.stem
            # Accept YYYY-MM-DD format
            if len(name) == 10 and name[4] == "-" and name[7] == "-":
                shards.append((name, str(npy_file)))
        if shards:
            result[coin] = sorted(shards)
    return result


def compute_crc32(arr: np.ndarray) -> int:
    """Compute CRC32 of numpy array bytes."""
    return zlib.crc32(arr.tobytes()) & 0xFFFFFFFF


def migrate_coin(
    exchange: str,
    coin: str,
    shards: List[Tuple[str, str]],
    cache_dir: str = "caches",
    dry_run: bool = True,
) -> Tuple[int, int, int]:
    """
    Migrate a single coin's data from legacy to new format.

    Returns:
        (migrated_count, skipped_count, error_count)
    """
    symbol = coin_to_symbol(coin, exchange)
    safe_symbol = symbol_to_safe_path(symbol)
    target_dir = Path(cache_dir) / "ohlcv" / exchange / "1m" / safe_symbol
    index_path = target_dir / "index.json"

    # Load existing index if present
    existing_shards = {}
    if index_path.exists():
        try:
            with open(index_path, "r") as f:
                idx = json.load(f)
            existing_shards = idx.get("shards", {})
        except Exception:
            pass

    migrated = 0
    skipped = 0
    errors = 0
    new_shards = {}

    for date_key, legacy_path in shards:
        target_path = target_dir / f"{date_key}.npy"

        # Skip if already exists
        if date_key in existing_shards or target_path.exists():
            skipped += 1
            continue

        # Load and convert
        arr = load_legacy_shard(legacy_path)
        if arr is None or arr.size == 0:
            errors += 1
            continue

        # Sort by timestamp
        arr = np.sort(arr, order="ts")

        if dry_run:
            logging.info(
                f"  [DRY RUN] Would migrate {legacy_path} -> {target_path} ({arr.size} rows)"
            )
            migrated += 1
            continue

        # Create directory and save
        target_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(target_path), arr)

        # Add to index
        ts_arr = arr["ts"]
        new_shards[date_key] = {
            "path": str(target_path),
            "min_ts": int(ts_arr.min()),
            "max_ts": int(ts_arr.max()),
            "count": int(arr.size),
            "crc32": compute_crc32(arr),
        }
        migrated += 1
        logging.info(f"  Migrated {date_key}: {arr.size} rows")

    # Update index if we migrated any shards
    if new_shards and not dry_run:
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    idx = json.load(f)
            except Exception:
                idx = {"shards": {}, "meta": {}}
        else:
            idx = {"shards": {}, "meta": {}}

        idx["shards"].update(new_shards)
        idx["meta"]["last_refresh_ms"] = int(time.time() * 1000)

        # Atomic write
        tmp_path = str(index_path) + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(idx, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(index_path))

    return migrated, skipped, errors


def delete_legacy_data(exchange: str, coins: Optional[List[str]] = None) -> int:
    """
    Delete legacy data for specified coins (or all if None).

    Returns:
        Number of directories deleted
    """
    legacy_dir = Path(f"historical_data/ohlcvs_{exchange}")
    if not legacy_dir.exists():
        return 0

    deleted = 0
    if coins is None:
        # Delete entire exchange directory
        shutil.rmtree(str(legacy_dir))
        logging.info(f"Deleted {legacy_dir}")
        deleted = 1
    else:
        for coin in coins:
            coin_dir = legacy_dir / coin
            if coin_dir.exists():
                shutil.rmtree(str(coin_dir))
                logging.info(f"Deleted {coin_dir}")
                deleted += 1

    return deleted


def main():
    parser = argparse.ArgumentParser(
        description="Migrate historical_data/ to caches/ohlcv/ structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Preview what would be migrated
    python src/tools/migrate_historical_data.py --exchange binanceusdm --dry-run

    # Migrate specific coins
    python src/tools/migrate_historical_data.py --exchange binanceusdm --coins BTC,ETH --execute

    # Migrate all and delete legacy
    python src/tools/migrate_historical_data.py --exchange binanceusdm --execute --delete-legacy
        """,
    )
    parser.add_argument(
        "--exchange",
        required=True,
        help="Exchange to migrate (e.g., binanceusdm, bybit, bitget)",
    )
    parser.add_argument(
        "--coins",
        type=str,
        default=None,
        help="Comma-separated list of coins to migrate (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without making changes",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the migration",
    )
    parser.add_argument(
        "--delete-legacy",
        action="store_true",
        help="Delete legacy data after successful migration (requires --execute)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="caches",
        help="Target cache directory (default: caches)",
    )

    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        parser.error("Must specify either --dry-run or --execute")

    if args.delete_legacy and not args.execute:
        parser.error("--delete-legacy requires --execute")

    exchange = to_ccxt_exchange_id(args.exchange)
    logging.info(f"Scanning legacy data for {exchange}...")

    legacy_data = scan_legacy_data(exchange)
    if not legacy_data:
        logging.info("No legacy data found to migrate.")
        return

    # Filter by coins if specified
    if args.coins:
        requested_coins = [c.strip().upper() for c in args.coins.split(",")]
        legacy_data = {k: v for k, v in legacy_data.items() if k.upper() in requested_coins}

    if not legacy_data:
        logging.info("No matching coins found.")
        return

    total_shards = sum(len(v) for v in legacy_data.values())
    logging.info(f"Found {len(legacy_data)} coins with {total_shards} shards to migrate")

    # Calculate disk space
    total_size = 0
    for coin, shards in legacy_data.items():
        for _, path in shards:
            try:
                total_size += os.path.getsize(path)
            except Exception:
                pass
    logging.info(f"Total legacy data size: {total_size / (1024 * 1024):.2f} MB")

    # Migrate each coin
    total_migrated = 0
    total_skipped = 0
    total_errors = 0
    successful_coins = []

    for coin in sorted(legacy_data.keys()):
        shards = legacy_data[coin]
        logging.info(f"Migrating {coin} ({len(shards)} shards)...")

        migrated, skipped, errors = migrate_coin(
            exchange,
            coin,
            shards,
            cache_dir=args.cache_dir,
            dry_run=args.dry_run,
        )

        total_migrated += migrated
        total_skipped += skipped
        total_errors += errors

        if migrated > 0 or skipped > 0:
            successful_coins.append(coin)

    # Summary
    logging.info("=" * 60)
    logging.info("Migration Summary:")
    logging.info(f"  Coins processed: {len(legacy_data)}")
    logging.info(f"  Shards migrated: {total_migrated}")
    logging.info(f"  Shards skipped (already exist): {total_skipped}")
    logging.info(f"  Shards with errors: {total_errors}")

    if args.dry_run:
        logging.info("  Mode: DRY RUN (no changes made)")
        logging.info("  Run with --execute to perform migration")
    else:
        logging.info("  Mode: EXECUTED")

        # Delete legacy if requested
        if args.delete_legacy and total_errors == 0:
            logging.info("Deleting legacy data...")
            deleted = delete_legacy_data(exchange, successful_coins if args.coins else None)
            logging.info(f"  Deleted {deleted} legacy director(y/ies)")
        elif args.delete_legacy and total_errors > 0:
            logging.warning("  Skipping legacy deletion due to errors")


if __name__ == "__main__":
    main()
