from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable


NUMPY_CACHE_SUFFIXES = {".npy"}
DEFAULT_ROOTS = ("caches",)
MAX_COVERAGE_ARTIFACT_SAMPLES = 8
MAX_COVERAGE_GAP_SAMPLES = 8


def _timeframe_interval_ms(timeframe: str) -> int | None:
    normalized = str(timeframe).strip().lower()
    if normalized == "1m":
        return 60_000
    if normalized == "1h":
        return 60 * 60_000
    return None


def _format_ms(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc).isoformat()


def _month_start_ms(year: int, month: int) -> int:
    return int(datetime(int(year), int(month), 1, tzinfo=timezone.utc).timestamp() * 1000)


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


def _coverage_summary_entry() -> dict[str, Any]:
    return {
        "artifact_count": 0,
        "covered_artifact_count": 0,
        "row_count": 0,
        "valid_row_count": 0,
        "gap_count": 0,
        "max_gap_rows": 0,
        "max_gap_ms": 0,
        "first_valid_ms": None,
        "last_valid_ms": None,
        "artifact_samples": [],
        "gap_samples": [],
    }


def _family_summary_entry() -> dict[str, Any]:
    return {
        "file_count": 0,
        "total_bytes": 0,
        "issue_count": 0,
        "by_extension": Counter(),
        "coverage": _coverage_summary_entry(),
    }


def _parse_ohlcv_valid_artifact(root: Path, path: Path) -> dict[str, Any] | None:
    name = path.name.lower()
    if not name.endswith(".valid.npy"):
        return None
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    parts = [part.lower() for part in rel_parts]
    if "data" in parts:
        data_idx = parts.index("data")
        offset = data_idx + 1
    else:
        offset = len(parts) - 5
    if offset < 0 or len(parts) < offset + 5:
        return None
    exchange = rel_parts[offset]
    timeframe = rel_parts[offset + 1]
    symbol = rel_parts[offset + 2]
    year_raw = rel_parts[offset + 3]
    month_raw = path.name.split(".", maxsplit=1)[0]
    interval_ms = _timeframe_interval_ms(timeframe)
    if interval_ms is None:
        return None
    try:
        year = int(year_raw)
        month = int(month_raw)
    except ValueError:
        return None
    if not 1 <= month <= 12:
        return None
    return {
        "exchange": exchange,
        "timeframe": timeframe,
        "symbol": symbol,
        "year": year,
        "month": month,
        "interval_ms": interval_ms,
        "month_start_ms": _month_start_ms(year, month),
    }


def _inspect_coverage_artifact(root: Path, path: Path, family: str) -> dict[str, Any] | None:
    if family != "candles":
        return None
    metadata = _parse_ohlcv_valid_artifact(root, path)
    if metadata is None:
        return None
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        valid = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False), dtype=bool)
    except (OSError, ValueError, EOFError, AttributeError, TypeError):
        return None
    if valid.ndim != 1:
        return None
    valid_indices = np.flatnonzero(valid)
    row_count = int(valid.shape[0])
    valid_row_count = int(valid_indices.size)
    interval_ms = int(metadata["interval_ms"])
    first_valid_idx: int | None = None
    last_valid_idx: int | None = None
    first_valid_ms: int | None = None
    last_valid_ms: int | None = None
    gaps: list[dict[str, Any]] = []
    if valid_row_count:
        first_valid_idx = int(valid_indices[0])
        last_valid_idx = int(valid_indices[-1])
        first_valid_ms = int(metadata["month_start_ms"]) + first_valid_idx * interval_ms
        last_valid_ms = int(metadata["month_start_ms"]) + last_valid_idx * interval_ms
        diffs = np.diff(valid_indices)
        gap_positions = np.flatnonzero(diffs > 1)
        for gap_pos in gap_positions:
            prev_idx = int(valid_indices[int(gap_pos)])
            next_idx = int(valid_indices[int(gap_pos) + 1])
            gap_start_idx = prev_idx + 1
            gap_end_idx = next_idx - 1
            gap_rows = gap_end_idx - gap_start_idx + 1
            gap_start_ms = int(metadata["month_start_ms"]) + gap_start_idx * interval_ms
            gap_end_ms = int(metadata["month_start_ms"]) + gap_end_idx * interval_ms
            gaps.append(
                {
                    "path": str(path),
                    "exchange": metadata["exchange"],
                    "timeframe": metadata["timeframe"],
                    "symbol": metadata["symbol"],
                    "start_ms": gap_start_ms,
                    "end_ms": gap_end_ms,
                    "start_date": _format_ms(gap_start_ms),
                    "end_date": _format_ms(gap_end_ms),
                    "rows": int(gap_rows),
                    "duration_ms": int(gap_rows * interval_ms),
                }
            )
    max_gap_rows = max((int(gap["rows"]) for gap in gaps), default=0)
    return {
        "path": str(path),
        "exchange": metadata["exchange"],
        "timeframe": metadata["timeframe"],
        "symbol": metadata["symbol"],
        "year": metadata["year"],
        "month": metadata["month"],
        "interval_ms": interval_ms,
        "row_count": row_count,
        "valid_row_count": valid_row_count,
        "first_valid_ms": first_valid_ms,
        "last_valid_ms": last_valid_ms,
        "first_valid_date": _format_ms(first_valid_ms),
        "last_valid_date": _format_ms(last_valid_ms),
        "gap_count": len(gaps),
        "max_gap_rows": max_gap_rows,
        "max_gap_ms": int(max_gap_rows * interval_ms),
        "gaps": gaps,
    }


def _add_limited_sample(samples: list[dict[str, Any]], item: dict[str, Any], limit: int) -> None:
    if len(samples) < limit:
        samples.append(item)


def _add_gap_samples(samples: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> None:
    samples.extend(gaps)
    samples.sort(key=lambda item: (int(item["rows"]), str(item["path"])), reverse=True)
    del samples[MAX_COVERAGE_GAP_SAMPLES:]


def _merge_coverage(summary: dict[str, Any], artifact: dict[str, Any]) -> None:
    coverage = summary["coverage"]
    coverage["artifact_count"] += 1
    coverage["row_count"] += int(artifact["row_count"])
    coverage["valid_row_count"] += int(artifact["valid_row_count"])
    coverage["gap_count"] += int(artifact["gap_count"])
    coverage["max_gap_rows"] = max(int(coverage["max_gap_rows"]), int(artifact["max_gap_rows"]))
    coverage["max_gap_ms"] = max(int(coverage["max_gap_ms"]), int(artifact["max_gap_ms"]))
    first_valid_ms = artifact["first_valid_ms"]
    last_valid_ms = artifact["last_valid_ms"]
    if first_valid_ms is not None:
        coverage["covered_artifact_count"] += 1
        if coverage["first_valid_ms"] is None or int(first_valid_ms) < int(
            coverage["first_valid_ms"]
        ):
            coverage["first_valid_ms"] = int(first_valid_ms)
        if coverage["last_valid_ms"] is None or int(last_valid_ms) > int(
            coverage["last_valid_ms"]
        ):
            coverage["last_valid_ms"] = int(last_valid_ms)
    sample = {key: value for key, value in artifact.items() if key != "gaps"}
    _add_limited_sample(coverage["artifact_samples"], sample, MAX_COVERAGE_ARTIFACT_SAMPLES)
    _add_gap_samples(coverage["gap_samples"], artifact["gaps"])


def _merge_finalized_coverage(summary: dict[str, Any], coverage_report: dict[str, Any]) -> None:
    coverage = summary["coverage"]
    coverage["artifact_count"] += int(coverage_report["artifact_count"])
    coverage["covered_artifact_count"] += int(coverage_report["covered_artifact_count"])
    coverage["row_count"] += int(coverage_report["row_count"])
    coverage["valid_row_count"] += int(coverage_report["valid_row_count"])
    coverage["gap_count"] += int(coverage_report["gap_count"])
    coverage["max_gap_rows"] = max(
        int(coverage["max_gap_rows"]),
        int(coverage_report["max_gap_rows"]),
    )
    coverage["max_gap_ms"] = max(
        int(coverage["max_gap_ms"]),
        int(coverage_report["max_gap_ms"]),
    )
    first_valid_ms = coverage_report["first_valid_ms"]
    last_valid_ms = coverage_report["last_valid_ms"]
    if first_valid_ms is not None:
        if coverage["first_valid_ms"] is None or int(first_valid_ms) < int(
            coverage["first_valid_ms"]
        ):
            coverage["first_valid_ms"] = int(first_valid_ms)
        if coverage["last_valid_ms"] is None or int(last_valid_ms) > int(
            coverage["last_valid_ms"]
        ):
            coverage["last_valid_ms"] = int(last_valid_ms)
    for artifact in coverage_report["artifact_samples"]:
        _add_limited_sample(
            coverage["artifact_samples"],
            artifact,
            MAX_COVERAGE_ARTIFACT_SAMPLES,
        )
    _add_gap_samples(coverage["gap_samples"], coverage_report["gap_samples"])


def _finalize_coverage(coverage: dict[str, Any], issue_count: int) -> dict[str, Any]:
    artifact_count = int(coverage["artifact_count"])
    valid_row_count = int(coverage["valid_row_count"])
    gap_count = int(coverage["gap_count"])
    if artifact_count == 0:
        evidence = "no_coverage_metadata"
    elif issue_count:
        evidence = "issues_present"
    elif valid_row_count == 0:
        evidence = "no_valid_rows"
    elif gap_count:
        evidence = "coverage_with_gaps"
    else:
        evidence = "coverage_observed"
    return {
        "warm_cache_evidence": evidence,
        "artifact_count": artifact_count,
        "covered_artifact_count": int(coverage["covered_artifact_count"]),
        "row_count": int(coverage["row_count"]),
        "valid_row_count": valid_row_count,
        "gap_count": gap_count,
        "max_gap_rows": int(coverage["max_gap_rows"]),
        "max_gap_ms": int(coverage["max_gap_ms"]),
        "first_valid_ms": coverage["first_valid_ms"],
        "last_valid_ms": coverage["last_valid_ms"],
        "first_valid_date": _format_ms(coverage["first_valid_ms"]),
        "last_valid_date": _format_ms(coverage["last_valid_ms"]),
        "artifact_samples": list(coverage["artifact_samples"]),
        "gap_samples": list(coverage["gap_samples"]),
    }


def _finalize_family_summary(by_family: dict[str, dict[str, Any]]) -> dict[str, Any]:
    finalized: dict[str, Any] = {}
    for family, summary in sorted(by_family.items()):
        issue_count = int(summary["issue_count"])
        family_report = {
            "file_count": int(summary["file_count"]),
            "total_bytes": int(summary["total_bytes"]),
            "issue_count": issue_count,
            "by_extension": dict(sorted(summary["by_extension"].items())),
        }
        coverage = _finalize_coverage(summary["coverage"], issue_count)
        if family == "candles" or int(coverage["artifact_count"]):
            family_report["coverage"] = coverage
        finalized[family] = family_report
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
        coverage_artifact = _inspect_coverage_artifact(root, entry, family)
        if coverage_artifact is not None:
            _merge_coverage(family_summary, coverage_artifact)
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
            if "coverage" in family_report:
                _merge_finalized_coverage(summary, family_report["coverage"])
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
