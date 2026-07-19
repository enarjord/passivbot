from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live.runtime_attribution import (  # noqa: E402
    AttributionScanLimitError,
    DEFAULT_MAX_BYTES_PER_FILE,
    DEFAULT_MAX_FILES,
    DEFAULT_MAX_FILLS,
    DEFAULT_MAX_TOTAL_BYTES,
    build_runtime_attribution_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool runtime-attribution",
        description=(
            "Build a read-only local report linking fills to recorded first-ingestion "
            "identity and candidate runtime windows."
        ),
    )
    parser.add_argument("--fill-root", action="append", default=None)
    parser.add_argument("--runtime-root", action="append", default=None)
    parser.add_argument("--monitor-root", action="append", default=None)
    parser.add_argument("--log-root", action="append", default=None)
    parser.add_argument("--exchange", action="append", default=[])
    parser.add_argument("--user", action="append", default=[])
    parser.add_argument("--symbol", action="append", default=[])
    parser.add_argument("--since-ms", type=int)
    parser.add_argument("--until-ms", type=int)
    parser.add_argument(
        "--trailing-only",
        action="store_true",
        help="Return only fills whose Passivbot order type is trailing.",
    )
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument("--max-fills", type=int, default=DEFAULT_MAX_FILLS)
    parser.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    parser.add_argument(
        "--max-bytes-per-file",
        type=int,
        default=DEFAULT_MAX_BYTES_PER_FILE,
    )
    parser.add_argument(
        "--fail-on-unattributed",
        action="store_true",
        help="Exit 1 when any selected fill lacks recorded first-ingestion provenance.",
    )
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_runtime_attribution_report(
            fill_roots=args.fill_root or ("caches/fill_events",),
            runtime_roots=args.runtime_root or ("caches/runtime",),
            monitor_roots=args.monitor_root or ("monitor",),
            log_roots=args.log_root or ("logs",),
            exchanges=args.exchange,
            users=args.user,
            symbols=args.symbol,
            since_ms=args.since_ms,
            until_ms=args.until_ms,
            trailing_only=args.trailing_only,
            max_files=args.max_files,
            max_bytes_per_file=args.max_bytes_per_file,
            max_total_bytes=args.max_total_bytes,
            max_fills=args.max_fills,
        )
    except (AttributionScanLimitError, ValueError) as exc:
        print(f"passivbot tool runtime-attribution: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    if args.fail_on_unattributed:
        unattributed = report["summary"]["first_ingestion_status_counts"].get(
            "unattributed", 0
        )
        if unattributed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
