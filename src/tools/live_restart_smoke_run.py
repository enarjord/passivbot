from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live.restart_smoke_orchestrator import (  # noqa: E402  # isort: skip
    DEFAULT_SMOKE_WAIT_S,
    execute_live_restart_smoke,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool live-restart-smoke-run",
        description=(
            "Restart exact verified local tmux targets and collect one "
            "bounded post-restart smoke window without pull/build or force "
            "escalation."
        ),
    )
    parser.add_argument("supervisor_config")
    parser.add_argument("monitor_root")
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--logs-root")
    parser.add_argument("--expected-repository-head", required=True)
    parser.add_argument("--expected-rust-source-fingerprint", required=True)
    parser.add_argument("--expected-supervisor-fingerprint", required=True)
    parser.add_argument("--expected-targets", required=True, type=int)
    parser.add_argument(
        "--smoke-wait-s", type=float, default=DEFAULT_SMOKE_WAIT_S
    )
    parser.add_argument("--shutdown-timeout-s", type=float, default=90.0)
    parser.add_argument("--startup-timeout-s", type=float, default=180.0)
    parser.add_argument("--poll-interval-s", type=float, default=2.0)
    parser.add_argument("--preflight-samples", type=int, default=3)
    parser.add_argument("--preflight-interval-s", type=float, default=5.0)
    parser.add_argument("--verification-samples", type=int, default=3)
    parser.add_argument("--verification-interval-s", type=float, default=5.0)
    parser.add_argument("--smoke-target-samples", type=int, default=3)
    parser.add_argument("--smoke-target-interval-s", type=float, default=1.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("--execute is required")
    try:
        report = execute_live_restart_smoke(
            args.supervisor_config,
            session_name=args.session_name,
            monitor_root=args.monitor_root,
            logs_root=args.logs_root,
            expected_repository_head=args.expected_repository_head,
            expected_rust_source_fingerprint=(
                args.expected_rust_source_fingerprint
            ),
            expected_supervisor_fingerprint=(
                args.expected_supervisor_fingerprint
            ),
            expected_targets=args.expected_targets,
            smoke_wait_s=args.smoke_wait_s,
            shutdown_timeout_s=args.shutdown_timeout_s,
            startup_timeout_s=args.startup_timeout_s,
            poll_interval_s=args.poll_interval_s,
            preflight_samples=args.preflight_samples,
            preflight_interval_s=args.preflight_interval_s,
            verification_samples=args.verification_samples,
            verification_interval_s=args.verification_interval_s,
            smoke_target_samples=args.smoke_target_samples,
            smoke_target_interval_s=args.smoke_target_interval_s,
            execute=True,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            report,
            indent=None if args.compact else 2,
            sort_keys=True,
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
