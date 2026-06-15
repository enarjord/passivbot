from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config.migrations.trailing_grid_v7 import (
    migrate_v7_trailing_grid_file,
    migration_report_has_unresolved,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool migrate-config-v7",
        description="Convert a legacy v7 trailing-grid config to canonical v8 trailing_grid_v7 shape.",
    )
    parser.add_argument("input_config", type=Path, help="Path to the legacy v7 config")
    parser.add_argument("output_config", type=Path, help="Path for the migrated v8 config")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write the transform report as JSON",
    )
    parser.add_argument(
        "--allow-manual-review-output",
        action="store_true",
        help=(
            "Write best-effort migrated output even when the report contains manual-review "
            "or dropped unsupported fields. By default unresolved migrations return nonzero "
            "and do not write the output config."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _migrated, report = migrate_v7_trailing_grid_file(
        args.input_config,
        args.output_config,
        allow_manual_review_output=args.allow_manual_review_output,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if migration_report_has_unresolved(report) and not args.allow_manual_review_output:
        print(
            "migration requires manual review; output config was not written. "
            "Use --allow-manual-review-output only to intentionally write best-effort output.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
