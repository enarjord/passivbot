from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live.restart_smoke_plan import (  # noqa: E402
    DEFAULT_INCIDENT_BUNDLE_OUTPUT,
    DEFAULT_SMOKE_EVENT_TAIL_LINES,
    DEFAULT_SMOKE_LOG_TAIL_LINES,
    DEFAULT_SMOKE_MAX_LOG_FILES,
    DEFAULT_SMOKE_MAX_LOG_MATCHES,
    DEFAULT_SMOKE_MAX_EVENT_FILES_PER_BOT,
    DEFAULT_LOGS_ROOT,
    DEFAULT_MONITOR_ROOT,
    DEFAULT_SHUTDOWN_TIMEOUT_S,
    DEFAULT_SMOKE_WINDOW_MINUTES,
    DEFAULT_STARTUP_WAIT_S,
    build_live_restart_smoke_plan,
    summarize_live_restart_smoke_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool live-restart-smoke-plan",
        description=(
            "Build a read-only dry-run plan for a live restart smoke routine. "
            "This tool does not execute the plan."
        ),
    )
    parser.add_argument("supervisor_config", help="tmuxp-style supervisor config path.")
    parser.add_argument(
        "--repo-path",
        help="Repository path to include in planned git checks.",
    )
    parser.add_argument(
        "--monitor-root",
        default=DEFAULT_MONITOR_ROOT,
        help="Monitor root for the planned smoke-report command.",
    )
    parser.add_argument(
        "--logs-root",
        default=DEFAULT_LOGS_ROOT,
        help="Logs root for the planned smoke-report command. Use an empty string to omit.",
    )
    parser.add_argument(
        "--shutdown-timeout-s",
        type=int,
        default=DEFAULT_SHUTDOWN_TIMEOUT_S,
        help="Per-bot graceful shutdown timeout for the plan.",
    )
    parser.add_argument(
        "--startup-wait-s",
        type=int,
        default=DEFAULT_STARTUP_WAIT_S,
        help="Post-start wait duration before smoke-report collection.",
    )
    parser.add_argument(
        "--smoke-window-minutes",
        type=int,
        default=DEFAULT_SMOKE_WINDOW_MINUTES,
        help="Recent time window for the planned smoke-report command.",
    )
    parser.add_argument(
        "--event-tail-lines",
        type=int,
        default=DEFAULT_SMOKE_EVENT_TAIL_LINES,
        help=(
            "Rows per monitor event file for the planned smoke-report command. "
            "Use 0 for full event-file validation."
        ),
    )
    parser.add_argument(
        "--max-event-files-per-bot",
        type=int,
        default=DEFAULT_SMOKE_MAX_EVENT_FILES_PER_BOT,
        help=(
            "Recent event files per bot for the planned smoke-report command. "
            "Use 0 to disable the per-bot file cap."
        ),
    )
    parser.add_argument(
        "--max-log-files",
        type=int,
        default=DEFAULT_SMOKE_MAX_LOG_FILES,
        help=(
            "Recent text log files for the planned smoke-report command. "
            "Use 0 to omit the explicit smoke-report override."
        ),
    )
    parser.add_argument(
        "--log-tail-lines",
        type=int,
        default=DEFAULT_SMOKE_LOG_TAIL_LINES,
        help=(
            "Rows per text log file for the planned smoke-report command. "
            "Use 0 to omit the explicit smoke-report override."
        ),
    )
    parser.add_argument(
        "--max-log-matches",
        type=int,
        default=DEFAULT_SMOKE_MAX_LOG_MATCHES,
        help=(
            "Maximum matched text log lines for the planned smoke-report command. "
            "Use 0 to omit the explicit smoke-report override."
        ),
    )
    parser.add_argument(
        "--incident-bundle-output",
        default=DEFAULT_INCIDENT_BUNDLE_OUTPUT,
        help=(
            "Output path for the planned live-incident-bundle evidence command. "
            "The plan never creates the bundle."
        ),
    )
    smoke_projection = parser.add_mutually_exclusive_group()
    smoke_projection.add_argument(
        "--brief-smoke-report",
        action="store_true",
        help="Plan a brief live-smoke-report command; this is the default.",
    )
    smoke_projection.add_argument(
        "--summary-smoke-report",
        action="store_true",
        help="Plan a summary live-smoke-report command instead of --brief.",
    )
    smoke_projection.add_argument(
        "--full-smoke-report",
        action="store_true",
        help="Plan a full live-smoke-report command instead of --brief.",
    )
    parser.add_argument(
        "--pretty-smoke-report",
        action="store_true",
        help="Plan a pretty-printed live-smoke-report command instead of --compact.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Rejected: execution is not implemented for this plan-only tool.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact single-line JSON.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Emit a concise plan summary instead of the full per-bot plan.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.execute:
        parser.error(
            "--execute is not implemented; live-restart-smoke-plan is plan-only"
        )
    try:
        report = build_live_restart_smoke_plan(
            args.supervisor_config,
            repo_path=args.repo_path,
            monitor_root=args.monitor_root,
            logs_root=args.logs_root if str(args.logs_root).strip() else None,
            shutdown_timeout_s=int(args.shutdown_timeout_s),
            startup_wait_s=int(args.startup_wait_s),
            smoke_window_minutes=int(args.smoke_window_minutes),
            smoke_event_tail_lines=int(args.event_tail_lines),
            smoke_max_event_files_per_bot=int(args.max_event_files_per_bot),
            smoke_max_log_files=int(args.max_log_files),
            smoke_log_tail_lines=int(args.log_tail_lines),
            smoke_max_log_matches=int(args.max_log_matches),
            incident_bundle_output=args.incident_bundle_output,
            compact_smoke_report=not bool(args.pretty_smoke_report),
            brief_smoke_report=not (
                bool(args.full_smoke_report) or bool(args.summary_smoke_report)
            ),
            summary_smoke_report=bool(args.summary_smoke_report),
            execute=False,
        )
    except (NotImplementedError, ValueError) as exc:
        parser.error(str(exc))
    if args.summary:
        report = summarize_live_restart_smoke_plan(report)
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
