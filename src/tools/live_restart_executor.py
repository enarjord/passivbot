from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live.restart_executor import execute_live_restart  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool live-restart-executor",
        description=(
            "Gracefully restart exact verified local tmux targets. This tool "
            "requires a clean expected repository/runtime and does not pull or "
            "build code or use automatic force signals."
        ),
    )
    parser.add_argument("supervisor_config", help="Tmuxp-style supervisor config.")
    parser.add_argument("--session-name", required=True, help="Exact tmux session.")
    parser.add_argument(
        "--expected-supervisor-fingerprint",
        required=True,
        help="Exact lowercase SHA-256 fingerprint from the target preflight.",
    )
    parser.add_argument(
        "--expected-repository-head",
        required=True,
        help="Exact 40-character lowercase Git commit expected at execution.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Confirm exact-pane graceful stop and relaunch execution.",
    )
    parser.add_argument("--shutdown-timeout-s", type=float, default=90.0)
    parser.add_argument("--startup-timeout-s", type=float, default=180.0)
    parser.add_argument("--poll-interval-s", type=float, default=2.0)
    parser.add_argument("--preflight-samples", type=int, default=3)
    parser.add_argument("--preflight-interval-s", type=float, default=5.0)
    parser.add_argument("--verification-samples", type=int, default=3)
    parser.add_argument("--verification-interval-s", type=float, default=5.0)
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("--execute is required; use live-restart-target-report first")
    try:
        report = execute_live_restart(
            args.supervisor_config,
            session_name=args.session_name,
            expected_supervisor_fingerprint=args.expected_supervisor_fingerprint,
            expected_repository_head=args.expected_repository_head,
            config_base_dir=Path.cwd(),
            preflight_samples=args.preflight_samples,
            preflight_interval_s=args.preflight_interval_s,
            shutdown_timeout_s=args.shutdown_timeout_s,
            startup_timeout_s=args.startup_timeout_s,
            poll_interval_s=args.poll_interval_s,
            verification_samples=args.verification_samples,
            verification_interval_s=args.verification_interval_s,
            execute=True,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
