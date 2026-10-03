from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live.repository_prepare import (  # noqa: E402  # isort: skip
    DEFAULT_BUILD_TIMEOUT_S,
    prepare_live_repository,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool live-repository-prepare",
        description=(
            "Fast-forward clean local canonical master to one exact "
            "origin/master commit and prove its Rust runtime is "
            "restart-ready."
        ),
    )
    parser.add_argument("--expected-current-head", required=True)
    parser.add_argument("--expected-target-head", required=True)
    parser.add_argument("--expected-rust-source-fingerprint", required=True)
    parser.add_argument(
        "--build-timeout-s", type=float, default=DEFAULT_BUILD_TIMEOUT_S
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Confirm canonical fetch, fast-forward, and Rust preparation.",
    )
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("--execute is required")
    try:
        report = prepare_live_repository(
            expected_current_head=args.expected_current_head,
            expected_target_head=args.expected_target_head,
            expected_rust_source_fingerprint=(
                args.expected_rust_source_fingerprint
            ),
            build_timeout_s=args.build_timeout_s,
            execute=True,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            report, indent=None if args.compact else 2, sort_keys=True
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
