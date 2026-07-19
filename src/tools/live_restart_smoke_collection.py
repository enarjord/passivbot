from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live.restart_smoke_collection import (  # noqa: E402
    DEFAULT_TARGET_SAMPLE_INTERVAL_S,
    DEFAULT_TARGET_SAMPLES,
    build_live_restart_smoke_collection,
)
from live.restart_smoke_evidence import (  # noqa: E402
    validate_live_restart_smoke_epoch_window,
    validate_live_restart_smoke_expectations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool live-restart-smoke-collect",
        description=(
            "Collect bounded local restart-target and smoke evidence in memory, "
            "then emit a sanitized verdict without controlling processes."
        ),
    )
    parser.add_argument("supervisor_config", help="Tmuxp-style supervisor config.")
    parser.add_argument("monitor_root", help="Monitor root to scan.")
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--logs-root")
    parser.add_argument("--expected-repository-head", required=True)
    parser.add_argument("--expected-supervisor-fingerprint", required=True)
    parser.add_argument("--expected-targets", required=True, type=int)
    parser.add_argument("--since-ms", required=True, type=int)
    parser.add_argument("--until-ms", required=True, type=int)
    parser.add_argument(
        "--target-samples",
        type=int,
        default=DEFAULT_TARGET_SAMPLES,
    )
    parser.add_argument(
        "--target-sample-interval-s",
        type=float,
        default=DEFAULT_TARGET_SAMPLE_INTERVAL_S,
    )
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_live_restart_smoke_expectations(
            expected_repository_head=args.expected_repository_head,
            expected_supervisor_fingerprint=args.expected_supervisor_fingerprint,
            expected_targets=args.expected_targets,
        )
        validate_live_restart_smoke_epoch_window(
            since_ms=args.since_ms,
            until_ms=args.until_ms,
        )
        report = build_live_restart_smoke_collection(
            args.supervisor_config,
            session_name=args.session_name,
            monitor_root=args.monitor_root,
            logs_root=args.logs_root,
            expected_repository_head=args.expected_repository_head,
            expected_supervisor_fingerprint=args.expected_supervisor_fingerprint,
            expected_targets=args.expected_targets,
            since_ms=args.since_ms,
            until_ms=args.until_ms,
            target_samples=args.target_samples,
            target_sample_interval_s=args.target_sample_interval_s,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
