from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live.log_secret_inventory import (  # noqa: E402
    DEFAULT_MAX_BYTES_PER_FILE,
    DEFAULT_MAX_FILES,
    build_log_secret_inventory,
    summarize_log_secret_inventory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool log-secret-inventory",
        description="Read-only, value-free inventory of secret-like material in local logs.",
    )
    parser.add_argument("logs_root", nargs="?", default="logs", help="Local log directory to scan.")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument("--max-bytes-per-file", type=int, default=DEFAULT_MAX_BYTES_PER_FILE)
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Omit per-file paths and hashes; emit aggregate scan evidence only.",
    )
    parser.add_argument("--compact", action="store_true", help="Emit compact single-line JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_log_secret_inventory(
            args.logs_root,
            max_files=args.max_files,
            max_bytes_per_file=args.max_bytes_per_file,
        )
    except ValueError as exc:
        print(f"passivbot tool log-secret-inventory: {exc}", file=sys.stderr)
        return 2
    if args.summary:
        report = summarize_log_secret_inventory(report)
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
