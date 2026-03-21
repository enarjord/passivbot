from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logging_setup import configure_logging
from monitor_dev import run_monitor_dev


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the monitor relay if needed and open the minimal TUI with bot-log tailing."
    )
    parser.add_argument("--relay-url", type=str, default="http://127.0.0.1:8765")
    parser.add_argument("--exchange", type=str, default=None)
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--focus-symbol", type=str, default=None)
    parser.add_argument("--monitor-root", type=str, default="monitor")
    parser.add_argument("--logs-dir", type=str, default="logs")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--snapshot-refresh-seconds", type=float, default=2.0)
    parser.add_argument("--render-interval-ms", type=int, default=250)
    parser.add_argument("--log-poll-interval-ms", type=int, default=500)
    parser.add_argument("--log-bootstrap-lines", type=int, default=12)
    parser.add_argument("--relay-poll-interval-ms", type=int, default=250)
    parser.add_argument("--relay-queue-size", type=int, default=1000)
    parser.add_argument("--relay-log-file", type=str, default="tmp/monitor_dev/relay.log")
    parser.add_argument("--log-level", type=str, default="WARNING")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level.upper())
    try:
        asyncio.run(
            run_monitor_dev(
                relay_url=args.relay_url,
                exchange=args.exchange,
                user=args.user,
                focus_symbol=args.focus_symbol,
                monitor_root=args.monitor_root,
                logs_dir=args.logs_dir,
                explicit_log_file=args.log_file,
                snapshot_refresh_seconds=args.snapshot_refresh_seconds,
                render_interval_ms=args.render_interval_ms,
                log_poll_interval_ms=args.log_poll_interval_ms,
                log_bootstrap_lines=args.log_bootstrap_lines,
                relay_poll_interval_ms=args.relay_poll_interval_ms,
                relay_queue_size=args.relay_queue_size,
                relay_log_file=args.relay_log_file,
                repo_root=str(Path(__file__).resolve().parents[2]),
            )
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
