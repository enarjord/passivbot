from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from ohlcv_store import month_start_ts, rows_in_month, timeframe_to_interval_ms


NUMPY_CACHE_SUFFIXES = {".npy"}
DEFAULT_ROOTS = ("caches",)
MAX_COVERAGE_ARTIFACT_SAMPLES = 8
MAX_COVERAGE_GAP_SAMPLES = 8
MAX_METADATA_ARTIFACT_SAMPLES = 8
CURRENT_FILL_PNL_CONTRACT = "gross_pnl_quote_fee_best_effort_v2"
TIMESTAMP_FIELD_SUFFIXES = ("_ms", "_ts", "_at")
TIMESTAMP_FIELD_NAMES = {
    "timestamp",
    "time",
    "event_time",
    "event_time_ms",
    "last_refresh_ms",
    "oldest_event_ts",
    "newest_event_ts",
    "covered_start_ms",
    "start_ts",
    "end_ts",
    "last_red_ts",
    "cooldown_until_ms",
    "pending_red_since_ms",
}


def _timeframe_interval_ms(timeframe: str) -> int | None:
    try:
        return int(timeframe_to_interval_ms(str(timeframe)))
    except ValueError:
        return None
    return None


def _format_ms(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc).isoformat()


def _month_start_ms(year: int, month: int) -> int:
    return int(month_start_ts(int(year), int(month)))


def _int_ms(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return None
    return candidate if candidate > 0 else None


def _timestamp_field_name(key: str) -> bool:
    normalized = str(key).lower()
    return normalized in TIMESTAMP_FIELD_NAMES or normalized.endswith(TIMESTAMP_FIELD_SUFFIXES)


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


def _read_json_payload(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return None


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
        "length_mismatch_count": 0,
        "row_count": 0,
        "expected_row_count": 0,
        "valid_row_count": 0,
        "gap_count": 0,
        "max_gap_rows": 0,
        "max_gap_ms": 0,
        "first_valid_ms": None,
        "last_valid_ms": None,
        "artifact_samples": [],
        "gap_samples": [],
    }


def _metadata_summary_entry() -> dict[str, Any]:
    return {
        "artifact_count": 0,
        "metadata_file_count": 0,
        "record_count": 0,
        "current_pnl_contract_count": 0,
        "legacy_pnl_contract_count": 0,
        "missing_pnl_contract_count": 0,
        "history_scope_counts": Counter(),
        "known_gap_count": 0,
        "first_event_ms": None,
        "last_event_ms": None,
        "covered_start_ms": None,
        "newest_event_ms": None,
        "last_refresh_ms": None,
        "timestamp_field_count": 0,
        "artifact_samples": [],
    }


def _family_summary_entry() -> dict[str, Any]:
    return {
        "file_count": 0,
        "total_bytes": 0,
        "issue_count": 0,
        "by_extension": Counter(),
        "coverage": _coverage_summary_entry(),
        "metadata": _metadata_summary_entry(),
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
    expected_row_count = int(
        rows_in_month(
            int(metadata["year"]),
            int(metadata["month"]),
            metadata["timeframe"],
        )
    )
    valid_row_count = int(valid_indices.size)
    interval_ms = int(metadata["interval_ms"])
    length_mismatch = row_count != expected_row_count
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
    if row_count < expected_row_count:
        gap_start_idx = row_count
        gap_end_idx = expected_row_count - 1
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
                "boundary": "trailing_shortfall",
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
        "expected_row_count": expected_row_count,
        "length_mismatch": length_mismatch,
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
    coverage["expected_row_count"] += int(artifact["expected_row_count"])
    coverage["length_mismatch_count"] += int(bool(artifact["length_mismatch"]))
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
    coverage["length_mismatch_count"] += int(coverage_report["length_mismatch_count"])
    coverage["row_count"] += int(coverage_report["row_count"])
    coverage["expected_row_count"] += int(coverage_report["expected_row_count"])
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


def _update_min_max_ms(
    metadata: dict[str, Any],
    *,
    first_key: str,
    last_key: str,
    value: Any,
) -> None:
    ts_ms = _int_ms(value)
    if ts_ms is None:
        return
    if metadata[first_key] is None or ts_ms < int(metadata[first_key]):
        metadata[first_key] = ts_ms
    if metadata[last_key] is None or ts_ms > int(metadata[last_key]):
        metadata[last_key] = ts_ms


def _iter_timestamp_fields(payload: Any, *, prefix: str = ""):
    if isinstance(payload, dict):
        for key, value in payload.items():
            path_key = f"{prefix}.{key}" if prefix else str(key)
            if _timestamp_field_name(str(key)):
                ts_ms = _int_ms(value)
                if ts_ms is not None:
                    yield path_key, ts_ms
            if isinstance(value, (dict, list)):
                yield from _iter_timestamp_fields(value, prefix=path_key)
    elif isinstance(payload, list):
        for index, item in enumerate(payload[:MAX_METADATA_ARTIFACT_SAMPLES]):
            if isinstance(item, (dict, list)):
                path_key = f"{prefix}[{index}]" if prefix else f"[{index}]"
                yield from _iter_timestamp_fields(item, prefix=path_key)


def _fill_contract_bucket(value: Any) -> str:
    contract = str(value or "")
    if contract == CURRENT_FILL_PNL_CONTRACT:
        return "current"
    if not contract:
        return "missing"
    return "legacy"


def _add_fill_record_metadata(out: dict[str, Any], record: dict[str, Any]) -> bool:
    if "timestamp" not in record and "pnl_contract" not in record:
        return False
    out["record_count"] += 1
    bucket = _fill_contract_bucket(record.get("pnl_contract"))
    out[f"{bucket}_pnl_contract_count"] += 1
    _update_min_max_ms(
        out,
        first_key="first_event_ms",
        last_key="last_event_ms",
        value=record.get("timestamp"),
    )
    return True


def _finish_record_sample(
    out: dict[str, Any],
    sample: dict[str, Any],
) -> dict[str, Any] | None:
    if int(out["record_count"]) == 0:
        return None
    sample["record_count"] = out["record_count"]
    sample["first_event_ms"] = out["first_event_ms"]
    sample["last_event_ms"] = out["last_event_ms"]
    _add_limited_sample(out["artifact_samples"], sample, MAX_METADATA_ARTIFACT_SAMPLES)
    return out


def _summarize_fill_payload(path: Path, payload: Any) -> dict[str, Any] | None:
    out = _metadata_summary_entry()
    out["artifact_count"] = 1
    sample: dict[str, Any] = {"path": str(path)}
    records: list[Any] = []
    if path.name == "metadata.json" and isinstance(payload, dict):
        out["metadata_file_count"] = 1
        bucket = _fill_contract_bucket(payload.get("pnl_contract"))
        out[f"{bucket}_pnl_contract_count"] += 1
        scope = str(payload.get("history_scope") or "unknown")
        out["history_scope_counts"][scope] += 1
        known_gaps = payload.get("known_gaps")
        if isinstance(known_gaps, list):
            out["known_gap_count"] = len(known_gaps)
        for key, target in (
            ("covered_start_ms", "covered_start_ms"),
            ("newest_event_ts", "newest_event_ms"),
            ("last_refresh_ms", "last_refresh_ms"),
        ):
            ts_ms = _int_ms(payload.get(key))
            if ts_ms is not None:
                out[target] = ts_ms
        for key in ("oldest_event_ts", "newest_event_ts"):
            _update_min_max_ms(
                out,
                first_key="first_event_ms",
                last_key="last_event_ms",
                value=payload.get(key),
            )
        sample.update(
            {
                "kind": "fill_metadata",
                "pnl_contract": payload.get("pnl_contract"),
                "history_scope": payload.get("history_scope"),
                "known_gap_count": out["known_gap_count"],
                "covered_start_ms": out["covered_start_ms"],
                "newest_event_ms": out["newest_event_ms"],
            }
        )
    elif isinstance(payload, list):
        records = [
            item
            for item in payload
            if isinstance(item, dict) and ("timestamp" in item or "pnl_contract" in item)
        ]
        if not records:
            return None
        sample["kind"] = "fill_records"
    elif isinstance(payload, dict) and ("timestamp" in payload or "pnl_contract" in payload):
        records = [payload]
        sample["kind"] = "fill_record"
    else:
        return None

    for record in records:
        if not isinstance(record, dict):
            continue
        _add_fill_record_metadata(out, record)
    if records:
        sample["record_count"] = out["record_count"]
        sample["first_event_ms"] = out["first_event_ms"]
        sample["last_event_ms"] = out["last_event_ms"]
    _add_limited_sample(out["artifact_samples"], sample, MAX_METADATA_ARTIFACT_SAMPLES)
    return out


def _summarize_fill_ndjson_payload(path: Path) -> dict[str, Any] | None:
    out = _metadata_summary_entry()
    out["artifact_count"] = 1
    sample: dict[str, Any] = {"path": str(path), "kind": "fill_records"}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                record = json.loads(raw)
                if isinstance(record, dict):
                    _add_fill_record_metadata(out, record)
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return None
    return _finish_record_sample(out, sample)


def _add_risk_timestamp_metadata(
    out: dict[str, Any],
    payload: Any,
    timestamp_fields: list[dict[str, Any]],
) -> None:
    for key, ts_ms in _iter_timestamp_fields(payload):
        out["timestamp_field_count"] += 1
        _update_min_max_ms(
            out,
            first_key="first_event_ms",
            last_key="last_event_ms",
            value=ts_ms,
        )
        _add_limited_sample(
            timestamp_fields,
            {"field": key, "timestamp_ms": ts_ms, "date": _format_ms(ts_ms)},
            MAX_METADATA_ARTIFACT_SAMPLES,
        )


def _summarize_risk_payload(path: Path, payload: Any) -> dict[str, Any] | None:
    out = _metadata_summary_entry()
    out["artifact_count"] = 1
    sample: dict[str, Any] = {"path": str(path)}
    if isinstance(payload, dict):
        sample["kind"] = "risk_state"
        sample["top_level_keys"] = sorted(str(key) for key in payload.keys())[
            :MAX_METADATA_ARTIFACT_SAMPLES
        ]
        if any(token in str(path).lower() for token in ("hsl", "equity_hard_stop")):
            sample["hsl_related"] = True
    elif isinstance(payload, list):
        sample["kind"] = "risk_records"
        out["record_count"] = sum(1 for item in payload if isinstance(item, dict))
    else:
        return None

    timestamp_fields: list[dict[str, Any]] = []
    _add_risk_timestamp_metadata(out, payload, timestamp_fields)
    if timestamp_fields:
        sample["timestamp_fields"] = timestamp_fields
        sample["first_timestamp_ms"] = out["first_event_ms"]
        sample["last_timestamp_ms"] = out["last_event_ms"]
    _add_limited_sample(out["artifact_samples"], sample, MAX_METADATA_ARTIFACT_SAMPLES)
    return out


def _summarize_risk_ndjson_payload(path: Path) -> dict[str, Any] | None:
    out = _metadata_summary_entry()
    out["artifact_count"] = 1
    sample: dict[str, Any] = {"path": str(path), "kind": "risk_records"}
    if any(token in str(path).lower() for token in ("hsl", "equity_hard_stop")):
        sample["hsl_related"] = True
    timestamp_fields: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                record = json.loads(raw)
                if not isinstance(record, dict):
                    continue
                out["record_count"] += 1
                _add_risk_timestamp_metadata(out, record, timestamp_fields)
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return None
    if int(out["record_count"]) == 0:
        return None
    if timestamp_fields:
        sample["timestamp_fields"] = timestamp_fields
        sample["first_timestamp_ms"] = out["first_event_ms"]
        sample["last_timestamp_ms"] = out["last_event_ms"]
    _add_limited_sample(out["artifact_samples"], sample, MAX_METADATA_ARTIFACT_SAMPLES)
    return out


def _inspect_metadata_artifact(path: Path, family: str) -> dict[str, Any] | None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = _read_json_payload(path)
        if payload is None:
            return None
        if family == "fills":
            return _summarize_fill_payload(path, payload)
        if family == "risk":
            return _summarize_risk_payload(path, payload)
        return None
    if suffix == ".ndjson":
        if family == "fills":
            return _summarize_fill_ndjson_payload(path)
        if family == "risk":
            return _summarize_risk_ndjson_payload(path)
        return None
    else:
        return None


def _merge_metadata(summary: dict[str, Any], artifact: dict[str, Any]) -> None:
    metadata = summary["metadata"]
    for key in (
        "artifact_count",
        "metadata_file_count",
        "record_count",
        "current_pnl_contract_count",
        "legacy_pnl_contract_count",
        "missing_pnl_contract_count",
        "known_gap_count",
        "timestamp_field_count",
    ):
        metadata[key] += int(artifact[key])
    metadata["history_scope_counts"].update(artifact["history_scope_counts"])
    for key in ("first_event_ms", "covered_start_ms"):
        value = artifact[key]
        if value is not None and (metadata[key] is None or int(value) < int(metadata[key])):
            metadata[key] = int(value)
    for key in ("last_event_ms", "newest_event_ms", "last_refresh_ms"):
        value = artifact[key]
        if value is not None and (metadata[key] is None or int(value) > int(metadata[key])):
            metadata[key] = int(value)
    for sample in artifact["artifact_samples"]:
        _add_limited_sample(
            metadata["artifact_samples"],
            sample,
            MAX_METADATA_ARTIFACT_SAMPLES,
        )


def _merge_finalized_metadata(summary: dict[str, Any], metadata_report: dict[str, Any]) -> None:
    metadata = summary["metadata"]
    for key in (
        "artifact_count",
        "metadata_file_count",
        "record_count",
        "current_pnl_contract_count",
        "legacy_pnl_contract_count",
        "missing_pnl_contract_count",
        "known_gap_count",
        "timestamp_field_count",
    ):
        metadata[key] += int(metadata_report[key])
    metadata["history_scope_counts"].update(metadata_report["history_scope_counts"])
    for key in ("first_event_ms", "covered_start_ms"):
        value = metadata_report[key]
        if value is not None and (metadata[key] is None or int(value) < int(metadata[key])):
            metadata[key] = int(value)
    for key in ("last_event_ms", "newest_event_ms", "last_refresh_ms"):
        value = metadata_report[key]
        if value is not None and (metadata[key] is None or int(value) > int(metadata[key])):
            metadata[key] = int(value)
    for sample in metadata_report["artifact_samples"]:
        _add_limited_sample(
            metadata["artifact_samples"],
            sample,
            MAX_METADATA_ARTIFACT_SAMPLES,
        )


def _finalize_metadata(
    metadata: dict[str, Any],
    *,
    family: str,
    issue_count: int,
) -> dict[str, Any]:
    artifact_count = int(metadata["artifact_count"])
    legacy_count = int(metadata["legacy_pnl_contract_count"])
    missing_count = int(metadata["missing_pnl_contract_count"])
    current_count = int(metadata["current_pnl_contract_count"])
    record_count = int(metadata["record_count"])
    if artifact_count == 0:
        compatibility = "no_local_metadata"
    elif issue_count:
        compatibility = "issues_present"
    elif family == "fills" and (legacy_count or missing_count):
        compatibility = "legacy_or_missing_pnl_contract"
    elif family == "fills" and current_count:
        compatibility = "current_pnl_contract"
    elif family == "fills":
        compatibility = "no_pnl_contract_evidence"
    elif int(metadata["timestamp_field_count"]):
        compatibility = "local_state_with_timestamps"
    else:
        compatibility = "local_state_observed"
    return {
        "compatibility": compatibility,
        "artifact_count": artifact_count,
        "metadata_file_count": int(metadata["metadata_file_count"]),
        "record_count": record_count,
        "current_pnl_contract_count": current_count,
        "legacy_pnl_contract_count": legacy_count,
        "missing_pnl_contract_count": missing_count,
        "history_scope_counts": dict(sorted(metadata["history_scope_counts"].items())),
        "known_gap_count": int(metadata["known_gap_count"]),
        "first_event_ms": metadata["first_event_ms"],
        "last_event_ms": metadata["last_event_ms"],
        "first_event_date": _format_ms(metadata["first_event_ms"]),
        "last_event_date": _format_ms(metadata["last_event_ms"]),
        "covered_start_ms": metadata["covered_start_ms"],
        "covered_start_date": _format_ms(metadata["covered_start_ms"]),
        "newest_event_ms": metadata["newest_event_ms"],
        "newest_event_date": _format_ms(metadata["newest_event_ms"]),
        "last_refresh_ms": metadata["last_refresh_ms"],
        "last_refresh_date": _format_ms(metadata["last_refresh_ms"]),
        "timestamp_field_count": int(metadata["timestamp_field_count"]),
        "artifact_samples": list(metadata["artifact_samples"]),
    }


def _finalize_coverage(coverage: dict[str, Any], issue_count: int) -> dict[str, Any]:
    artifact_count = int(coverage["artifact_count"])
    valid_row_count = int(coverage["valid_row_count"])
    gap_count = int(coverage["gap_count"])
    length_mismatch_count = int(coverage["length_mismatch_count"])
    if artifact_count == 0:
        evidence = "no_coverage_metadata"
    elif length_mismatch_count:
        evidence = "coverage_length_mismatch"
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
        "length_mismatch_count": length_mismatch_count,
        "row_count": int(coverage["row_count"]),
        "expected_row_count": int(coverage["expected_row_count"]),
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
        metadata = _finalize_metadata(
            summary["metadata"],
            family=family,
            issue_count=issue_count,
        )
        if family in {"fills", "risk"} or int(metadata["artifact_count"]):
            family_report["metadata"] = metadata
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
            if coverage_artifact["length_mismatch"]:
                family_summary["issue_count"] += 1
                issues.append(
                    _issue(
                        "warning",
                        "coverage_length_mismatch",
                        entry,
                        (
                            "candle valid mask length does not match canonical "
                            f"month rows: actual={coverage_artifact['row_count']} "
                            f"expected={coverage_artifact['expected_row_count']}"
                        ),
                        family=family,
                    )
                )
        metadata_artifact = _inspect_metadata_artifact(entry, family)
        if metadata_artifact is not None:
            _merge_metadata(family_summary, metadata_artifact)
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
            if "metadata" in family_report:
                _merge_finalized_metadata(summary, family_report["metadata"])
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
