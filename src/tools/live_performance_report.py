from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live.performance_report import (  # noqa: E402
    build_live_performance_report,
    summarize_live_performance_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize local live bot performance timing from structured monitor "
            "events. This is read-only and does not contact exchanges."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="monitor",
        help="Monitor root, bot root, events directory, or NDJSON segment.",
    )
    parser.add_argument(
        "--since-ms",
        type=int,
        help="Only include live events with monitor ts at or after this epoch-ms value.",
    )
    parser.add_argument(
        "--until-ms",
        type=int,
        help="Only include live events with monitor ts at or before this epoch-ms value.",
    )
    parser.add_argument(
        "--recent-minutes",
        type=float,
        help="Only include live events from the last N minutes.",
    )
    parser.add_argument(
        "--include-rotated",
        action="store_true",
        help=(
            "Also scan rotated/compressed history segments. By default directory "
            "scans read current.ndjson files only."
        ),
    )
    parser.add_argument(
        "--event-tail-lines",
        type=int,
        default=0,
        help=(
            "Opt-in bound for monitor event segments: inspect only the last N rows "
            "from each event file. Plain NDJSON segments seek from file end; "
            "compressed segments may still scan sequentially. Default 0 keeps "
            "full monitor validation."
        ),
    )
    parser.add_argument(
        "--group-limit",
        type=int,
        default=80,
        help="Maximum performance timing groups to include.",
    )
    parser.add_argument(
        "--bot",
        action="append",
        default=[],
        help="Only include one bot key, formatted as exchange/user. May be repeated.",
    )
    parser.add_argument(
        "--exchange",
        action="append",
        default=[],
        help="Only include one exchange name. May be repeated.",
    )
    parser.add_argument(
        "--user",
        action="append",
        default=[],
        help="Only include one user/account name. May be repeated.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Emit a bounded operator summary instead of the full report.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact single-line JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.since_ms is not None and args.recent_minutes is not None:
        parser.error("--since-ms and --recent-minutes are mutually exclusive")
    since_ms = args.since_ms
    if args.recent_minutes is not None:
        if args.recent_minutes <= 0:
            parser.error("--recent-minutes must be greater than 0")
        since_ms = int(time.time() * 1000) - int(args.recent_minutes * 60_000)
    if since_ms is not None and args.until_ms is not None and since_ms > args.until_ms:
        parser.error("--since-ms/--recent-minutes must be <= --until-ms")
    if args.group_limit < 0:
        parser.error("--group-limit must be >= 0")
    if args.event_tail_lines < 0:
        parser.error("--event-tail-lines must be >= 0")
    report = build_live_performance_report(
        args.path,
        since_ms=since_ms,
        until_ms=args.until_ms,
        include_rotated=bool(args.include_rotated),
        event_tail_lines=int(args.event_tail_lines),
        group_limit=int(args.group_limit),
        bot_filters=args.bot,
        exchange_filters=args.exchange,
        user_filters=args.user,
    )
    if args.summary:
        report = summarize_live_performance_report(report, group_limit=int(args.group_limit))
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
