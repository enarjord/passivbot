from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live.smoke_report import (  # noqa: E402
    DEFAULT_LOG_WINDOW_UNPARSED_POLICY,
    LOG_WINDOW_UNPARSED_POLICIES,
    build_live_smoke_report,
    default_logs_root_for_monitor,
    summarize_live_smoke_report,
    summarize_live_smoke_report_brief,
)


def _since_ms_from_recent_minutes(value: float | None) -> int | None:
    if value is None:
        return None
    minutes = float(value)
    if minutes <= 0:
        raise ValueError("--recent-minutes must be positive")
    return int(time.time() * 1000) - int(minutes * 60_000)


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
        "--processes",
        action="store_true",
        help="Include read-only passivbot live process liveness details.",
    )
    parser.add_argument(
        "--supervisor-config",
        help=(
            "Optional tmuxp-style config listing expected passivbot live panes. "
            "Implies --processes and reports missing expected bots as hard failures."
        ),
    )
    parser.add_argument(
        "--process-match",
        default="passivbot live",
        help="Substring used before canonicalizing passivbot live process rows.",
    )
    parser.add_argument(
        "--include-rotated",
        action="store_true",
        help="Also scan rotated/compressed monitor event segments.",
    )
    parser.add_argument(
        "--since-ms",
        type=int,
        help=(
            "Only include structured monitor events and timestamped text log "
            "lines at or after this epoch-ms timestamp."
        ),
    )
    parser.add_argument(
        "--until-ms",
        type=int,
        help=(
            "Only include structured monitor events and timestamped text log "
            "lines at or before this epoch-ms timestamp."
        ),
    )
    parser.add_argument(
        "--recent-minutes",
        type=float,
        help=(
            "Only include structured monitor events and timestamped text log "
            "lines from the last N minutes. Equivalent to --since-ms based on "
            "local wall-clock time."
        ),
    )
    parser.add_argument(
        "--log-window-unparsed-policy",
        choices=sorted(LOG_WINDOW_UNPARSED_POLICIES),
        default=DEFAULT_LOG_WINDOW_UNPARSED_POLICY,
        help=(
            "When a log time window is active, keep unparseable text log lines "
            "visible by default, or drop unparseable lines without in-window "
            "timestamp context."
        ),
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
        "--event-tail-lines",
        type=int,
        default=0,
        help=(
            "Opt-in bound for monitor event segments: inspect only the last N "
            "rows from each event file. Plain NDJSON segments seek from file "
            "end; compressed segments may still scan sequentially. Default 0 "
            "keeps full monitor validation."
        ),
    )
    parser.add_argument(
        "--max-event-files-per-bot",
        type=int,
        default=0,
        help=(
            "With --include-rotated, scan at most N event segments per bot/events "
            "directory, preferring current.ndjson and then newest rotated files. "
            "Default 0 scans all discovered segments."
        ),
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
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Emit concise smoke-test summary JSON instead of the full report.",
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help=(
            "Emit top-level smoke-test counters without event groups or log "
            "matches. Implies --summary."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.logs_root is None:
        logs_root = default_logs_root_for_monitor(args.monitor_root)
    else:
        logs_root = args.logs_root if str(args.logs_root).strip() else None
    since_ms = args.since_ms
    try:
        recent_since_ms = _since_ms_from_recent_minutes(args.recent_minutes)
    except ValueError as exc:
        parser.error(str(exc))
    if since_ms is not None and recent_since_ms is not None:
        parser.error("--since-ms and --recent-minutes are mutually exclusive")
    if recent_since_ms is not None:
        since_ms = recent_since_ms
    until_ms = args.until_ms
    if since_ms is not None and until_ms is not None and since_ms > until_ms:
        parser.error("--since-ms must be <= --until-ms")
    if int(args.event_tail_lines) < 0:
        parser.error("--event-tail-lines must be non-negative")
    if int(args.max_event_files_per_bot) < 0:
        parser.error("--max-event-files-per-bot must be non-negative")
    report = build_live_smoke_report(
        args.monitor_root,
        logs_root=logs_root,
        include_processes=bool(args.processes),
        supervisor_config=args.supervisor_config,
        process_command_match=str(args.process_match),
        include_rotated=bool(args.include_rotated),
        since_ms=since_ms,
        until_ms=until_ms,
        max_problem_events=int(args.max_problem_events),
        max_log_files=int(args.max_log_files),
        event_tail_lines=int(args.event_tail_lines),
        max_event_files_per_bot=int(args.max_event_files_per_bot),
        log_tail_lines=int(args.log_tail_lines),
        max_log_matches=int(args.max_log_matches),
        log_window_unparsed_policy=str(args.log_window_unparsed_policy),
    )
    if args.brief:
        output = summarize_live_smoke_report_brief(report)
    elif args.summary:
        output = summarize_live_smoke_report(report)
    else:
        output = report
    print(json.dumps(output, indent=None if args.compact else 2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
