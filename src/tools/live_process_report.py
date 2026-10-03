from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live.smoke_report import (  # noqa: E402
    DEFAULT_PROCESS_MATCH,
    MAX_PROCESS_SAMPLES,
    MAX_PROCESS_SAMPLE_INTERVAL_S,
    build_live_process_report,
    summarize_live_process_report,
)


SAFETY_CONTRACT = {
    "local_only": True,
    "reads": [
        "process_table",
        "optional_supervisor_config",
        "optional_referenced_bot_configs",
    ],
    "monitor_events": False,
    "text_logs": False,
    "network": False,
    "exchange_access": False,
    "credential_store_access": False,
    "process_control": False,
    "writes_files": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool live-process-report",
        description=(
            "Sample the local passivbot live process table without reading monitor "
            "events or text logs, contacting exchanges, or controlling processes."
        ),
    )
    parser.add_argument(
        "--supervisor-config",
        help=(
            "Optional tmuxp-style config listing expected passivbot live panes. "
            "Missing, duplicate, and extra command matches are hard failures."
        ),
    )
    parser.add_argument(
        "--process-match",
        default=DEFAULT_PROCESS_MATCH,
        help="Substring used before canonicalizing passivbot live process rows.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help=(
            "Number of bounded process-table samples. "
            f"Default 1; maximum {MAX_PROCESS_SAMPLES}."
        ),
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=5.0,
        help=(
            "Seconds between samples when --samples is greater than 1. "
            f"Maximum {MAX_PROCESS_SAMPLE_INTERVAL_S:g}."
        ),
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help=(
            "Emit aggregate process/config/sampling counters without command, "
            "account, path, PID, or per-process rows."
        ),
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
    try:
        processes = build_live_process_report(
            supervisor_config=args.supervisor_config,
            process_command_match=str(args.process_match),
            process_samples=int(args.samples),
            process_sample_interval_s=float(args.interval_s),
            config_base_dir=Path.cwd(),
        )
    except ValueError as exc:
        parser.error(str(exc))
    output_processes = (
        summarize_live_process_report(processes) if args.brief else processes
    )
    report = {
        "schema_version": 1,
        "tool": "live-process-report",
        "ok": bool(processes.get("ok")),
        "hard_failures": int(processes.get("hard_failures") or 0),
        "safety": SAFETY_CONTRACT,
        "processes": output_processes,
    }
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
