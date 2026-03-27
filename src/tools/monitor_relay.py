from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from aiohttp import web

SRC_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logging_setup import configure_logging
from monitor_relay import create_monitor_relay_app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve read-only Passivbot monitor snapshots and live streams."
    )
    parser.add_argument(
        "--monitor-root",
        type=str,
        default="monitor",
        help="Base monitor root containing {exchange}/{user}/ manifests and snapshots.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind host for the relay server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Bind port for the relay server.",
    )
    parser.add_argument(
        "--poll-interval-ms",
        type=int,
        default=250,
        help="Polling interval for current event/history files.",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=1000,
        help="Per-subscriber outbound queue size before a resync is required.",
    )
    parser.add_argument(
        "--ws-replay-limit",
        type=int,
        default=50,
        help="How many recent lines per current event/history file to replay on websocket connect.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level for the relay process.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level.upper())
    app = create_monitor_relay_app(
        monitor_root=args.monitor_root,
        poll_interval_ms=args.poll_interval_ms,
        subscriber_queue_size=args.queue_size,
        ws_replay_limit=args.ws_replay_limit,
    )
    logging.info(
        "[monitor-relay] serving monitor_root=%s host=%s port=%s poll_interval_ms=%s ws_replay_limit=%s",
        args.monitor_root,
        args.host,
        args.port,
        args.poll_interval_ms,
        args.ws_replay_limit,
    )
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
