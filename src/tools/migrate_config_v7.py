from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config.migrations.trailing_grid_v7 import (
    migrate_v7_trailing_grid_file,
    migration_report_has_invalid_output,
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
        help=(
            "Path to write the transform report as JSON. Defaults to a "
            "*.migration-report.json sibling of the requested output config."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete JSON report to stdout instead of the concise summary.",
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


def default_report_path(output_config: Path) -> Path:
    if output_config.suffix:
        candidate = output_config.with_suffix(".migration-report.json")
    else:
        candidate = output_config.with_name(output_config.name + ".migration-report.json")
    if candidate == output_config:
        return output_config.with_name(output_config.name + ".report.json")
    return candidate


def format_summary(report: dict) -> str:
    validation = report.get("canonical_validation")
    validation_status = validation.get("status") if isinstance(validation, dict) else "unknown"
    output_state = "written" if report.get("output_written") else "not written"
    lines = [
        f"Migration status: {report.get('status', 'unknown')}",
        f"Canonical validation: {validation_status}",
        f"Output config: {output_state} ({report.get('output_config_path')})",
        f"Manual-review fields: {len(report.get('manual_review_fields', []))}",
        f"Unsupported fields dropped: {len(report.get('dropped_unsupported_fields', []))}",
        f"Behavior-change warnings: {len(report.get('behavior_change_warnings', []))}",
        f"V8 defaults inserted: {len(report.get('inserted_v8_defaults', []))}",
        f"Report: {report.get('report_path')}",
    ]
    return "\n".join(lines)


def print_action_items(report: dict) -> None:
    for path in report.get("manual_review_fields", []):
        print(f"manual review: {path}", file=sys.stderr)
    for path in report.get("dropped_unsupported_fields", []):
        print(f"dropped unsupported field: {path}", file=sys.stderr)
    for warning in report.get("behavior_change_warnings", []):
        print(f"behavior warning: {warning}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input_config.resolve() == args.output_config.resolve():
        print("input config and output config paths must differ", file=sys.stderr)
        return 2
    report_path = args.report or default_report_path(args.output_config)
    if report_path.resolve() == args.output_config.resolve():
        print("migration report path must differ from output config path", file=sys.stderr)
        return 2
    _migrated, report = migrate_v7_trailing_grid_file(
        args.input_config,
        args.output_config,
        allow_manual_review_output=args.allow_manual_review_output,
    )
    report["report_path"] = str(report_path)
    payload = json.dumps(report, indent=2, sort_keys=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(payload + "\n", encoding="utf-8")
    if args.json:
        print(payload)
    else:
        print(format_summary(report))
        print_action_items(report)
    if migration_report_has_invalid_output(report):
        validation = report.get("canonical_validation", {})
        print(
            "migration produced an invalid canonical config; output config was not written: "
            f"{validation.get('error_type', 'validation error')}: "
            f"{validation.get('message', 'unknown error')}",
            file=sys.stderr,
        )
        return 2
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
