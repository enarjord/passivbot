#!/usr/bin/env python3
"""
Audit and optionally fix mixed base/quote volume in OHLCV caches.

Currently supports GateIO cache audit/fix by comparing daily volumes to
reference exchanges (binance/bybit) where available.
"""

from __future__ import annotations

import argparse
import logging

from legacy_data_migrator import audit_and_fix_gateio_volume
from logging_setup import configure_logging, resolve_log_level


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit/fix OHLCV volume in caches/ohlcv")
    parser.add_argument(
        "--cache-base",
        type=str,
        default="caches/ohlcv",
        help="Base directory for OHLCV cache (default: caches/ohlcv)",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="gateio",
        help="Exchange to audit (default: gateio)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply fixes (default: dry-run)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of day files scanned (for testing)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Logging verbosity (warning, info, debug, trace or 0-3).",
    )

    args = parser.parse_args()
    configure_logging(debug=resolve_log_level(args.log_level, None, fallback=1))

    if args.exchange.lower() != "gateio":
        logging.error("Only gateio is supported by this tool for now.")
        return 1

    stats = audit_and_fix_gateio_volume(
        cache_base=args.cache_base,
        dry_run=not args.fix,
        max_files=args.max_files,
    )
    logging.info(
        "gateio volume audit | dry_run=%s scanned=%d converted=%d base=%d no_ref=%d uncertain=%d errors=%d",
        not args.fix,
        stats["scanned"],
        stats["converted"],
        stats["already_base"],
        stats["skipped_no_reference"],
        stats["skipped_uncertain"],
        stats["errors"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
