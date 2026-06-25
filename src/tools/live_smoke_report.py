from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live.smoke_report import build_live_smoke_report, default_logs_root_for_monitor  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize local live monitor events and text logs for smoke-test "
            "evidence."
        )
    )
    parser.add_argument(
        "monitor_root",
        nargs="?",
        default="monitor",
        help="Monitor root, bot root, events directory, or NDJSON segment.",
    )
    parser.add_argument(
        "--logs-root",
        default=None,
        help=(
            "Text log directory to scan. Defaults to a sibling logs directory "
            "found from monitor_root; use an empty string to skip logs."
        ),
    )
    parser.add_argument(
        "--include-rotated",
        action="store_true",
        help="Also scan rotated/compressed monitor event segments.",
    )
    parser.add_argument(
        "--max-problem-events",
        type=int,
        default=50,
        help="Maximum recent problem events to include.",
    )
    parser.add_argument(
        "--max-log-files",
        type=int,
        default=8,
        help="Maximum recent log files to inspect.",
    )
    parser.add_argument(
        "--log-tail-lines",
        type=int,
        default=300,
        help="Tail this many lines from each inspected log file.",
    )
    parser.add_argument(
        "--max-log-matches",
        type=int,
        default=50,
        help="Maximum matching log lines to include.",
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
    if args.logs_root is None:
        logs_root = default_logs_root_for_monitor(args.monitor_root)
    else:
        logs_root = args.logs_root if str(args.logs_root).strip() else None
    report = build_live_smoke_report(
        args.monitor_root,
        logs_root=logs_root,
        include_rotated=bool(args.include_rotated),
        max_problem_events=int(args.max_problem_events),
        max_log_files=int(args.max_log_files),
        log_tail_lines=int(args.log_tail_lines),
        max_log_matches=int(args.max_log_matches),
    )
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
