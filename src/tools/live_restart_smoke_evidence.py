from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live.restart_smoke_evidence import (  # noqa: E402
    MAX_RESTART_TARGETS,
    build_live_restart_smoke_evidence,
)


MAX_INPUT_JSON_BYTES = 16 * 1024 * 1024


def _file_state(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _load_json_object(path_text: str, *, label: str) -> dict[str, Any]:
    try:
        path = Path(path_text)
        before = path.stat()
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            raw = handle.read(MAX_INPUT_JSON_BYTES + 1)
        after = path.stat()
        if (
            _file_state(before) != _file_state(opened)
            or _file_state(opened) != _file_state(after)
            or opened.st_size != len(raw)
            or len(raw) > MAX_INPUT_JSON_BYTES
        ):
            raise ValueError
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError(
            f"{label} must be a readable JSON object within the input limit"
        ) from None
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool live-restart-smoke-evidence",
        description=(
            "Evaluate existing local restart-target and smoke-report JSON evidence "
            "without running commands or controlling processes."
        ),
    )
    parser.add_argument("target_report_json")
    parser.add_argument("smoke_report_json")
    parser.add_argument("--expected-repository-head", required=True)
    parser.add_argument("--expected-supervisor-fingerprint", required=True)
    parser.add_argument("--expected-targets", required=True, type=int)
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        target_report = _load_json_object(args.target_report_json, label="target report")
        smoke_report = _load_json_object(args.smoke_report_json, label="smoke report")
        report = build_live_restart_smoke_evidence(
            target_report,
            smoke_report,
            expected_repository_head=str(args.expected_repository_head),
            expected_supervisor_fingerprint=str(args.expected_supervisor_fingerprint),
            expected_targets=args.expected_targets,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
