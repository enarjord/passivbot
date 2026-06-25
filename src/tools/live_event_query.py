from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live.event_query import build_event_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate monitor event NDJSON and reconstruct a live cycle by cycle_id."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="monitor",
        help="Monitor root, bot root, events directory, or NDJSON segment.",
    )
    parser.add_argument("--cycle-id", help="Return compact records for one cycle_id.")
    parser.add_argument(
        "--event-type",
        "--kind",
        dest="event_types",
        action="append",
        help=(
            "Return compact records matching one event type. May be repeated or "
            "comma-separated; --kind is accepted as an alias for monitor terminology."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum cycle records to include in output.",
    )
    parser.add_argument(
        "--include-data",
        action="store_true",
        help="Include each matched event's bounded data payload.",
    )
    parser.add_argument(
        "--include-rotated",
        action="store_true",
        help=(
            "Also scan rotated/compressed history segments. By default directory "
            "scans read current.ndjson files only."
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
    report = build_event_report(
        args.path,
        cycle_id=args.cycle_id,
        event_type=args.event_types,
        limit=args.limit,
        include_data=bool(args.include_data),
        include_rotated=bool(args.include_rotated),
    )
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
