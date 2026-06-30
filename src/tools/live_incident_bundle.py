from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live.incident_bundle import build_live_incident_bundle  # noqa: E402
from live.smoke_report import (  # noqa: E402
    DEFAULT_LOG_WINDOW_UNPARSED_POLICY,
    LOG_WINDOW_UNPARSED_POLICIES,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a local incident evidence bundle from monitor event NDJSON, "
            "selected text log excerpts, monitor snapshots, config hashes, and "
            "runtime metadata."
        )
    )
    parser.add_argument(
        "monitor_root",
        nargs="?",
        default="monitor",
        help="Monitor root, bot root, events directory, or NDJSON segment.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output .tar.gz path. Defaults to passivbot_incident_bundle_<utc>.tar.gz.",
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
        "--config",
        action="append",
        default=[],
        help="Config path to hash into the bundle manifest. May be repeated.",
    )
    parser.add_argument(
        "--processes",
        action="store_true",
        help="Include read-only passivbot live process liveness details in smoke_report.json.",
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
    parser.add_argument("--cycle-id", help="Include compact records for one cycle_id.")
    parser.add_argument(
        "--event-type",
        "--kind",
        dest="event_types",
        action="append",
        help="Filter compact records by event type. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--order-wave-id",
        action="append",
        help="Filter compact records by order_wave_id. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--remote-call-id",
        action="append",
        help="Filter compact records by remote_call_id. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--symbol",
        action="append",
        help="Filter compact records by symbol. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--pside",
        action="append",
        help="Filter compact records by position side. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--reason-code",
        action="append",
        help="Filter compact records by reason_code. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--status",
        action="append",
        help="Filter compact records by status. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--since-ms",
        type=int,
        help="Include live events at or after this monitor ts.",
    )
    parser.add_argument(
        "--until-ms",
        type=int,
        help="Include live events at or before this monitor ts.",
    )
    parser.add_argument(
        "--recent-minutes",
        type=float,
        help=(
            "Include live events and timestamped text log lines from the last "
            "N minutes. Equivalent to --since-ms based on local wall-clock time."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum compact event records or timeline rows to include per report.",
    )
    parser.add_argument(
        "--include-data",
        action="store_true",
        help="Include each matched event's bounded data payload in compact event reports.",
    )
    parser.add_argument(
        "--no-trace-report",
        action="store_true",
        help=(
            "Do not include live-event-query trace summary/order-trace outputs "
            "in the bundled event report."
        ),
    )
    parser.add_argument(
        "--include-rotated",
        action="store_true",
        help="Also scan rotated/compressed monitor event segments.",
    )
    parser.add_argument(
        "--event-tail-lines",
        type=int,
        default=0,
        help=(
            "Only scan the last N lines from each monitor event segment for "
            "event reports and smoke checks. Plain NDJSON segments seek from "
            "file end; compressed segments may still be scanned sequentially. "
            "Default 0 scans full segments."
        ),
    )
    parser.add_argument(
        "--no-event-segments",
        action="store_true",
        help="Do not copy bounded event segment files into the bundle.",
    )
    parser.add_argument(
        "--max-event-segment-bytes",
        type=int,
        default=10_000_000,
        help="Maximum total event segment bytes copied into the bundle.",
    )
    parser.add_argument(
        "--max-snapshot-files",
        type=int,
        default=20,
        help="Maximum monitor snapshot JSON files copied into the bundle.",
    )
    parser.add_argument(
        "--max-snapshot-file-bytes",
        type=int,
        default=1_000_000,
        help="Maximum size for one monitor snapshot JSON file copied into the bundle.",
    )
    parser.add_argument(
        "--max-log-files",
        type=int,
        default=8,
        help="Maximum recent text log files to inspect.",
    )
    parser.add_argument(
        "--log-tail-lines",
        type=int,
        default=500,
        help="Tail this many lines from each inspected text log file.",
    )
    parser.add_argument(
        "--max-log-matches",
        type=int,
        default=100,
        help="Maximum matching log lines to include in the smoke report.",
    )
    parser.add_argument(
        "--log-window-unparsed-policy",
        choices=sorted(LOG_WINDOW_UNPARSED_POLICIES),
        default=DEFAULT_LOG_WINDOW_UNPARSED_POLICY,
        help=(
            "When the embedded smoke-report log window is active, keep "
            "unparseable text log lines visible by default, or drop unparseable "
            "lines without in-window timestamp context."
        ),
    )
    parser.add_argument("--compact", action="store_true", help="Emit compact single-line JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.since_ms is not None and args.recent_minutes is not None:
        parser.error("--since-ms and --recent-minutes are mutually exclusive")
    if int(args.event_tail_lines) < 0:
        parser.error("--event-tail-lines must be >= 0")
    since_ms = args.since_ms
    if args.recent_minutes is not None:
        if args.recent_minutes <= 0:
            parser.error("--recent-minutes must be greater than 0")
        since_ms = int(time.time() * 1000) - int(args.recent_minutes * 60_000)
    if since_ms is not None and args.until_ms is not None and since_ms > args.until_ms:
        parser.error("--since-ms/--recent-minutes must be <= --until-ms")
    report = build_live_incident_bundle(
        args.monitor_root,
        output_path=args.output,
        logs_root=args.logs_root,
        config_paths=list(args.config or []),
        include_processes=bool(args.processes),
        supervisor_config=args.supervisor_config,
        process_command_match=str(args.process_match),
        cycle_id=args.cycle_id,
        event_type=args.event_types,
        order_wave_id=args.order_wave_id,
        remote_call_id=args.remote_call_id,
        symbol=args.symbol,
        pside=args.pside,
        reason_code=args.reason_code,
        status=args.status,
        since_ms=since_ms,
        until_ms=args.until_ms,
        include_data=bool(args.include_data),
        include_trace_report=not bool(args.no_trace_report),
        include_rotated=bool(args.include_rotated),
        event_tail_lines=int(args.event_tail_lines),
        include_event_segments=not bool(args.no_event_segments),
        max_events=int(args.limit),
        max_log_files=int(args.max_log_files),
        log_tail_lines=int(args.log_tail_lines),
        max_log_matches=int(args.max_log_matches),
        log_window_unparsed_policy=str(args.log_window_unparsed_policy),
        max_snapshot_files=int(args.max_snapshot_files),
        max_snapshot_file_bytes=int(args.max_snapshot_file_bytes),
        max_event_segment_bytes=int(args.max_event_segment_bytes),
    )
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
