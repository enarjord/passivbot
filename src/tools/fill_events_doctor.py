"""Audit and optionally repair cached fill-event anomalies."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

# Ensure we can import modules from src/
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from config_utils import format_config, load_config
from fill_events_manager import (
    FillEventsManager,
    _build_fetcher_for_bot,
    _extract_symbol_pool,
    _instantiate_bot,
    _parse_log_level,
    _parse_time_arg,
)
from logging_setup import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit/repair fill-events cache")
    parser.add_argument("--config", default="configs/template.json", help="Path to config JSON")
    parser.add_argument("--user", help="Override live.user from config")
    parser.add_argument(
        "--cache-root",
        default="caches/fill_events",
        help="Root fill-events cache directory (default: caches/fill_events)",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Auto-repair detected anomalies by refetching from exchange",
    )
    parser.add_argument(
        "--repair-start",
        help="Optional repair start timestamp (ms/seconds/ISO). Defaults to earliest anomaly - 24h",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Optional symbol override for fetcher construction",
    )
    parser.add_argument(
        "--logging-level",
        default="info",
        help="Logging level name or integer (warning=0, info=1, debug=2, trace=3)",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config, verbose=False)
    config = format_config(config, verbose=False)
    if args.user:
        config.setdefault("live", {})["user"] = args.user

    bot = _instantiate_bot(config)
    symbols = _extract_symbol_pool(config, args.symbols)
    fetcher = _build_fetcher_for_bot(bot, symbols)
    cache_path = Path(args.cache_root) / bot.exchange / bot.user
    manager = FillEventsManager(
        exchange=bot.exchange,
        user=bot.user,
        fetcher=fetcher,
        cache_path=cache_path,
    )

    repair_start_ms: Optional[int] = _parse_time_arg(args.repair_start) if args.repair_start else None
    report = await manager.run_doctor(
        auto_repair=bool(args.repair),
        repair_start_ms=repair_start_ms,
    )
    print(json.dumps(report, indent=2, sort_keys=True))

    if report.get("anomaly_events") and not args.repair:
        return 2
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(_parse_log_level(args.logging_level))
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
