from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logging_setup import configure_logging
from monitor_tui import MonitorTuiClient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal terminal dashboard for the Passivbot monitor relay."
    )
    parser.add_argument(
        "--relay-url",
        type=str,
        default="http://127.0.0.1:8765",
        help="Base URL for the monitor relay.",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default=None,
        help="Exchange name when selecting one bot from a multi-bot relay.",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="User/account name when selecting one bot from a multi-bot relay.",
    )
    parser.add_argument(
        "--focus-symbol",
        type=str,
        default=None,
        help="Optional symbol to prioritize in the TUI panels.",
    )
    parser.add_argument(
        "--snapshot-refresh-seconds",
        type=float,
        default=2.0,
        help="How often to refresh /snapshot for current-state panels.",
    )
    parser.add_argument(
        "--render-interval-ms",
        type=int,
        default=250,
        help="How often to redraw the terminal.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        help="Logging level for relay connectivity diagnostics.",
    )
    return parser.parse_args()


async def _run_async(args: argparse.Namespace) -> None:
    client = MonitorTuiClient(
        relay_url=args.relay_url,
        exchange=args.exchange,
        user=args.user,
        focus_symbol=args.focus_symbol,
        snapshot_refresh_seconds=args.snapshot_refresh_seconds,
        render_interval_ms=args.render_interval_ms,
    )
    await client.run()


def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level.upper())
    logging.info(
        "[monitor-tui] connecting relay_url=%s exchange=%s user=%s focus_symbol=%s",
        args.relay_url,
        args.exchange,
        args.user,
        args.focus_symbol,
    )
    try:
        asyncio.run(_run_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
