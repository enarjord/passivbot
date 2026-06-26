from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable


NUMPY_CACHE_SUFFIXES = {".npy"}
DEFAULT_ROOTS = ("caches",)


def _issue(
    severity: str,
    code: str,
    path: Path,
    message: str,
    *,
    family: str | None = None,
) -> dict[str, Any]:
    issue = {
        "severity": severity,
        "code": code,
        "path": str(path),
        "message": message,
    }
    if family:
        issue["family"] = family
    return issue


def _empty_file_issue(path: Path) -> dict[str, Any] | None:
    try:
        if path.stat().st_size == 0:
            return _issue("warning", "empty_file", path, "cache file is empty")
    except OSError as exc:
        return _issue("error", "stat_failed", path, f"could not stat file: {exc}")
    return None


def _inspect_json_file(path: Path) -> list[dict[str, Any]]:
    empty = _empty_file_issue(path)
    if empty is not None:
        return [empty]
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        return [_issue("error", "json_decode_failed", path, f"invalid UTF-8: {exc}")]
    except json.JSONDecodeError as exc:
        return [
            _issue(
                "error",
                "json_decode_failed",
                path,
                f"invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}",
            )
        ]
    except OSError as exc:
        return [_issue("error", "read_failed", path, f"could not read file: {exc}")]
    return []


def _inspect_ndjson_file(path: Path) -> list[dict[str, Any]]:
    empty = _empty_file_issue(path)
    if empty is not None:
        return [empty]
    issues: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    json.loads(raw)
                except json.JSONDecodeError as exc:
                    issues.append(
                        _issue(
                            "error",
                            "ndjson_decode_failed",
                            path,
                            f"invalid NDJSON at line {line_no} column {exc.colno}: {exc.msg}",
                        )
                    )
                    break
    except UnicodeDecodeError as exc:
        return [_issue("error", "ndjson_decode_failed", path, f"invalid UTF-8: {exc}")]
    except OSError as exc:
        return [_issue("error", "read_failed", path, f"could not read file: {exc}")]
    return issues


def _inspect_npy_file(path: Path) -> list[dict[str, Any]]:
    empty = _empty_file_issue(path)
    if empty is not None:
        return [empty]
    try:
        import numpy as np
    except ImportError as exc:
        return [
            _issue(
                "warning",
                "npy_check_unavailable",
                path,
                f"numpy unavailable; skipped NPY validation: {exc}",
            )
        ]
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        _ = arr.shape
        _ = str(arr.dtype)
    except (OSError, ValueError, EOFError, AttributeError, TypeError) as exc:
        return [_issue("error", "npy_load_failed", path, f"could not load NPY: {exc}")]
    return []


def _inspect_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _inspect_json_file(path)
    if suffix == ".ndjson":
        return _inspect_ndjson_file(path)
    if suffix in NUMPY_CACHE_SUFFIXES:
        return _inspect_npy_file(path)
    return []


def _cache_family(root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(root)
        parts = [part.lower() for part in rel.parts]
    except ValueError:
        parts = [part.lower() for part in path.parts]
    stem = path.stem.lower()
    suffix = path.suffix.lower()
    joined = "/".join(parts)
    if suffix == ".lock" or path.name.lower().endswith(".lock"):
        return "locks"
    if any(part.startswith("ohlcvs") for part in parts) or "ohlcv" in joined:
        return "candles"
    if "hlcvs_data" in parts or "hlcvs" in joined:
        return "historical_hlcvs"
    if any(token in joined for token in ("fills", "fill_events", "pnls", "pnl")):
        return "fills"
    if any(token in joined for token in ("hsl", "equity_hard_stop", "risk")):
        return "risk"
    if "markets" in stem or "market" in stem:
        return "markets"
    if any(token in joined for token in ("monitor", "events", "snapshots")):
        return "monitor"
    if any(token in joined for token in ("user", "api_key", "api-keys", "api_keys")):
        return "user_config"
    return "unknown"


def _family_summary_entry() -> dict[str, Any]:
    return {
        "file_count": 0,
        "total_bytes": 0,
        "issue_count": 0,
        "by_extension": Counter(),
    }


def _finalize_family_summary(by_family: dict[str, dict[str, Any]]) -> dict[str, Any]:
    finalized: dict[str, Any] = {}
    for family, summary in sorted(by_family.items()):
        finalized[family] = {
            "file_count": int(summary["file_count"]),
            "total_bytes": int(summary["total_bytes"]),
            "issue_count": int(summary["issue_count"]),
            "by_extension": dict(sorted(summary["by_extension"].items())),
        }
    return finalized


def _scan_root(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {
        "path": str(root),
        "exists": root.exists(),
        "is_dir": root.is_dir(),
        "file_count": 0,
        "dir_count": 0,
        "total_bytes": 0,
        "by_extension": {},
        "by_family": {},
    }
    issues: list[dict[str, Any]] = []
    if not root.exists():
        issues.append(
            _issue(
                "warning",
                "root_missing",
                root,
                "cache root does not exist",
                family="root",
            )
        )
        return summary, issues
    if not root.is_dir():
        issues.append(
            _issue(
                "error",
                "root_not_directory",
                root,
                "cache root is not a directory",
                family="root",
            )
        )
        return summary, issues
    try:
        entries = list(root.rglob("*"))
    except OSError as exc:
        issues.append(
            _issue(
                "error",
                "root_scan_failed",
                root,
                f"could not scan root: {exc}",
                family="root",
            )
        )
        return summary, issues

    by_extension: Counter[str] = Counter()
    by_family: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if entry.is_dir():
            summary["dir_count"] += 1
            continue
        if not entry.is_file():
            continue
        summary["file_count"] += 1
        suffix = entry.suffix.lower() or "<none>"
        by_extension[suffix] += 1
        family = _cache_family(root, entry)
        family_summary = by_family.setdefault(family, _family_summary_entry())
        family_summary["file_count"] += 1
        family_summary["by_extension"][suffix] += 1
        try:
            file_size = entry.stat().st_size
            summary["total_bytes"] += file_size
            family_summary["total_bytes"] += file_size
        except OSError as exc:
            family_summary["issue_count"] += 1
            issues.append(
                _issue(
                    "error",
                    "stat_failed",
                    entry,
                    f"could not stat file: {exc}",
                    family=family,
                )
            )
            continue
        file_issues = _inspect_file(entry)
        for issue in file_issues:
            issue.setdefault("family", family)
        family_summary["issue_count"] += len(file_issues)
        issues.extend(file_issues)
    summary["by_extension"] = dict(sorted(by_extension.items()))
    summary["by_family"] = _finalize_family_summary(by_family)
    return summary, issues


def build_cache_integrity_report(roots: Iterable[str | Path]) -> dict[str, Any]:
    root_paths = [Path(root).expanduser() for root in roots]
    root_reports: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for root in root_paths:
        root_report, root_issues = _scan_root(root)
        root_reports.append(root_report)
        issues.extend(root_issues)
    by_severity: Counter[str] = Counter()
    by_family: dict[str, dict[str, Any]] = {}
    for item in issues:
        by_severity[item["severity"]] += 1
    for root in root_reports:
        for family, family_report in root["by_family"].items():
            summary = by_family.setdefault(family, _family_summary_entry())
            summary["file_count"] += int(family_report["file_count"])
            summary["total_bytes"] += int(family_report["total_bytes"])
            summary["issue_count"] += int(family_report["issue_count"])
            summary["by_extension"].update(family_report["by_extension"])
    summary = {
        "root_count": len(root_reports),
        "file_count": sum(int(root["file_count"]) for root in root_reports),
        "dir_count": sum(int(root["dir_count"]) for root in root_reports),
        "total_bytes": sum(int(root["total_bytes"]) for root in root_reports),
        "issue_count": len(issues),
        "by_severity": dict(sorted(by_severity.items())),
        "by_family": _finalize_family_summary(by_family),
    }
    return {
        "ok": "error" not in by_severity,
        "summary": summary,
        "roots": root_reports,
        "issues": issues,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool cache-integrity-doctor",
        description=(
            "Read-only local cache integrity smoke report for JSON, NDJSON, and NPY files."
        ),
    )
    parser.add_argument(
        "roots",
        nargs="*",
        default=list(DEFAULT_ROOTS),
        help="Cache root directories to scan. Defaults to caches.",
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
    report = build_cache_integrity_report(args.roots)
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
