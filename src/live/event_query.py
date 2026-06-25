from __future__ import annotations

import gzip
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from live.event_bus import LIVE_EVENT_ID_KEYS, LIVE_EVENT_MONITOR_PAYLOAD_KEY

EVENT_ID_KEYS = LIVE_EVENT_ID_KEYS


@dataclass(frozen=True)
class EventIssue:
    path: str
    line: int | None
    severity: str
    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_event_segment(path: Path) -> bool:
    return path.name.endswith(".ndjson") or path.name.endswith(".ndjson.gz")


def _event_path_sort_key(path: Path) -> tuple[str, int, str]:
    return (str(path.parent), 1 if path.name == "current.ndjson" else 0, path.name)


def discover_event_files(root: str | Path, *, include_rotated: bool = False) -> list[Path]:
    """Find monitor event NDJSON segments below a monitor root, bot root, or events dir."""
    path = Path(root).expanduser()
    if path.is_file():
        return [path] if _is_event_segment(path) else []
    if not path.exists():
        raise FileNotFoundError(str(path))
    if not path.is_dir():
        return []
    return sorted(
        (
            candidate
            for candidate in path.rglob("*.ndjson*")
            if candidate.is_file()
            and candidate.parent.name == "events"
            and _is_event_segment(candidate)
            and (include_rotated or candidate.name == "current.ndjson")
        ),
        key=_event_path_sort_key,
    )


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _live_event_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    payload = row.get("payload")
    if not isinstance(payload, dict):
        return None
    live_event = payload.get(LIVE_EVENT_MONITOR_PAYLOAD_KEY)
    return live_event if isinstance(live_event, dict) else None


def _event_ids(live_event: dict[str, Any]) -> dict[str, Any]:
    ids = live_event.get("ids")
    return dict(ids) if isinstance(ids, dict) else {}


def _normalize_filter_values(values: str | Iterable[str] | None) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = list(values)
    out: set[str] = set()
    for value in raw_values:
        for part in str(value).split(","):
            cleaned = part.strip()
            if cleaned:
                out.add(cleaned)
    return out


def _filter_matches(value: Any, filter_values: set[str]) -> bool:
    return not filter_values or (value is not None and str(value) in filter_values)


def _filter_report(
    *,
    cycle_id: str | None,
    event_types: set[str],
    order_wave_ids: set[str],
    remote_call_ids: set[str],
    symbols: set[str],
    psides: set[str],
    reason_codes: set[str],
    statuses: set[str],
) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if cycle_id is not None:
        filters["cycle_id"] = str(cycle_id)
    if event_types:
        filters["event_types"] = sorted(event_types)
    if order_wave_ids:
        filters["order_wave_ids"] = sorted(order_wave_ids)
    if remote_call_ids:
        filters["remote_call_ids"] = sorted(remote_call_ids)
    if symbols:
        filters["symbols"] = sorted(symbols)
    if psides:
        filters["psides"] = sorted(psides)
    if reason_codes:
        filters["reason_codes"] = sorted(reason_codes)
    if statuses:
        filters["statuses"] = sorted(statuses)
    return filters


def _compact_record(
    *,
    path: Path,
    line_no: int,
    row: dict[str, Any],
    live_event: dict[str, Any],
    include_data: bool,
) -> dict[str, Any]:
    ids = _event_ids(live_event)
    compact = {
        "path": str(path),
        "line": int(line_no),
        "ts": row.get("ts"),
        "seq": row.get("seq"),
        "event_type": live_event.get("event_type") or row.get("kind"),
        "level": live_event.get("level"),
        "status": live_event.get("status"),
        "reason_code": live_event.get("reason_code"),
        "component": live_event.get("component"),
        "exchange": live_event.get("exchange") or row.get("exchange"),
        "user": live_event.get("user") or row.get("user"),
        "symbol": live_event.get("symbol") or row.get("symbol"),
        "pside": live_event.get("pside") or row.get("pside"),
        "side": live_event.get("side"),
        "ids": {
            key: ids.get(key) for key in EVENT_ID_KEYS if ids.get(key) is not None
        },
    }
    if include_data:
        compact["data"] = (
            live_event.get("data") if isinstance(live_event.get("data"), dict) else {}
        )
    return {key: value for key, value in compact.items() if value is not None}


def _timeline_line(record: dict[str, Any]) -> str:
    parts = [str(record.get("ts", "-"))]
    if record.get("seq") is not None:
        parts.append(f"seq={record['seq']}")
    parts.append(str(record.get("event_type") or "unknown"))
    for key in ("status", "reason_code", "symbol", "pside", "side"):
        value = record.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    ids = record.get("ids")
    if isinstance(ids, dict) and ids:
        id_parts = [
            f"{key}={value}"
            for key, value in ids.items()
            if value is not None
            and key
            in {
                "cycle_id",
                "action_id",
                "order_wave_id",
                "remote_call_id",
                "remote_call_group_id",
            }
        ]
        if id_parts:
            parts.append("ids=" + ",".join(id_parts))
    return " ".join(parts)


def build_event_report(
    root: str | Path,
    *,
    cycle_id: str | None = None,
    event_type: str | Iterable[str] | None = None,
    order_wave_id: str | Iterable[str] | None = None,
    remote_call_id: str | Iterable[str] | None = None,
    symbol: str | Iterable[str] | None = None,
    pside: str | Iterable[str] | None = None,
    reason_code: str | Iterable[str] | None = None,
    status: str | Iterable[str] | None = None,
    limit: int = 200,
    include_data: bool = False,
    include_rotated: bool = False,
    timeline: bool = False,
) -> dict[str, Any]:
    issues: list[EventIssue] = []
    try:
        files = discover_event_files(root, include_rotated=include_rotated)
    except FileNotFoundError as exc:
        files = []
        issues.append(
            EventIssue(str(root), None, "error", "path_not_found", str(exc))
        )
    if not files and not issues:
        issues.append(
            EventIssue(
                str(root),
                None,
                "error",
                "no_event_files",
                "no event NDJSON files found",
            )
        )

    event_type_counts: Counter[str] = Counter()
    cycle_counts: Counter[str] = Counter()
    records_total = 0
    live_events = 0
    legacy_events = 0
    missing_cycle_id = 0
    cycle_events: list[dict[str, Any]] = []
    cycle_match_count = 0
    query_events: list[dict[str, Any]] = []
    query_match_count = 0
    max_events = max(0, int(limit))
    event_type_filter = _normalize_filter_values(event_type)
    order_wave_filter = _normalize_filter_values(order_wave_id)
    remote_call_filter = _normalize_filter_values(remote_call_id)
    symbol_filter = _normalize_filter_values(symbol)
    pside_filter = _normalize_filter_values(pside)
    reason_code_filter = _normalize_filter_values(reason_code)
    status_filter = _normalize_filter_values(status)
    has_non_cycle_filter = any(
        (
            event_type_filter,
            order_wave_filter,
            remote_call_filter,
            symbol_filter,
            pside_filter,
            reason_code_filter,
            status_filter,
        )
    )
    has_query_filter = has_non_cycle_filter or bool(timeline and cycle_id is None)

    for path in files:
        try:
            with _open_text(path) as stream:
                for line_no, raw_line in enumerate(stream, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    records_total += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        issues.append(
                            EventIssue(
                                str(path),
                                line_no,
                                "error",
                                "invalid_json",
                                str(exc),
                            )
                        )
                        continue
                    if not isinstance(row, dict):
                        issues.append(
                            EventIssue(
                                str(path),
                                line_no,
                                "error",
                                "invalid_record",
                                "event row is not a JSON object",
                            )
                        )
                        continue

                    live_event = _live_event_payload(row)
                    if live_event is None:
                        legacy_events += 1
                        continue
                    live_events += 1

                    record_event_type = live_event.get("event_type") or row.get("kind")
                    if not record_event_type:
                        issues.append(
                            EventIssue(
                                str(path),
                                line_no,
                                "error",
                                "missing_event_type",
                                "live event is missing event_type",
                            )
                        )
                    else:
                        record_event_type = str(record_event_type)
                        event_type_counts[record_event_type] += 1
                        if row.get("kind") and str(row.get("kind")) != record_event_type:
                            issues.append(
                                EventIssue(
                                    str(path),
                                    line_no,
                                    "warning",
                                    "kind_event_type_mismatch",
                                    f"kind={row.get('kind')} event_type={record_event_type}",
                                )
                            )

                    ids = _event_ids(live_event)
                    if live_event.get("ids") is not None and not isinstance(
                        live_event.get("ids"), dict
                    ):
                        issues.append(
                            EventIssue(
                                str(path),
                                line_no,
                                "error",
                                "invalid_ids",
                                "live event ids field is not an object",
                            )
                        )
                    record_cycle_id = ids.get("cycle_id")
                    if record_cycle_id is None:
                        missing_cycle_id += 1
                    else:
                        record_cycle_id = str(record_cycle_id)
                        cycle_counts[record_cycle_id] += 1

                    record_symbol = live_event.get("symbol") or row.get("symbol")
                    record_pside = live_event.get("pside") or row.get("pside")
                    record_status = live_event.get("status")
                    record_reason_code = live_event.get("reason_code")
                    event_type_matches = _filter_matches(
                        record_event_type, event_type_filter
                    )
                    cycle_matches = cycle_id is None or record_cycle_id == str(cycle_id)
                    order_wave_matches = _filter_matches(
                        ids.get("order_wave_id"), order_wave_filter
                    )
                    remote_call_matches = _filter_matches(
                        ids.get("remote_call_id"), remote_call_filter
                    )
                    symbol_matches = _filter_matches(record_symbol, symbol_filter)
                    pside_matches = _filter_matches(record_pside, pside_filter)
                    reason_code_matches = _filter_matches(
                        record_reason_code, reason_code_filter
                    )
                    status_matches = _filter_matches(record_status, status_filter)
                    query_matches = (
                        event_type_matches
                        and cycle_matches
                        and order_wave_matches
                        and remote_call_matches
                        and symbol_matches
                        and pside_matches
                        and reason_code_matches
                        and status_matches
                    )

                    if has_query_filter and query_matches:
                        query_match_count += 1
                        if len(query_events) < max_events:
                            query_events.append(
                                _compact_record(
                                    path=path,
                                    line_no=line_no,
                                    row=row,
                                    live_event=live_event,
                                    include_data=include_data,
                                )
                            )
                    if cycle_id is not None and query_matches:
                        cycle_match_count += 1
                        if len(cycle_events) < max_events:
                            cycle_events.append(
                                _compact_record(
                                    path=path,
                                    line_no=line_no,
                                    row=row,
                                    live_event=live_event,
                                    include_data=include_data,
                                )
                            )
        except OSError as exc:
            issues.append(
                EventIssue(str(path), None, "error", "read_failed", str(exc))
            )

    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    report: dict[str, Any] = {
        "ok": error_count == 0,
        "root": str(Path(root).expanduser()),
        "include_rotated": bool(include_rotated),
        "files": [str(path) for path in files],
        "files_scanned": len(files),
        "records_total": records_total,
        "live_events": live_events,
        "legacy_events": legacy_events,
        "missing_cycle_id": missing_cycle_id,
        "issues": [issue.to_dict() for issue in issues],
        "error_count": error_count,
        "warning_count": warning_count,
        "event_types": dict(sorted(event_type_counts.items())),
        "cycle_ids_sample": [
            {"cycle_id": key, "events": value}
            for key, value in cycle_counts.most_common(20)
        ],
    }
    if has_query_filter:
        query_filters = _filter_report(
            cycle_id=cycle_id,
            event_types=event_type_filter,
            order_wave_ids=order_wave_filter,
            remote_call_ids=remote_call_filter,
            symbols=symbol_filter,
            psides=pside_filter,
            reason_codes=reason_code_filter,
            statuses=status_filter,
        )
        report["query"] = {
            "filters": query_filters,
            "matched_events": query_match_count,
            "events_truncated": query_match_count > len(query_events),
            "events": query_events,
        }
        if timeline:
            report["query"]["timeline"] = [
                _timeline_line(event) for event in query_events
            ]
    if cycle_id is not None:
        cycle_filters = _filter_report(
            cycle_id=cycle_id,
            event_types=event_type_filter,
            order_wave_ids=order_wave_filter,
            remote_call_ids=remote_call_filter,
            symbols=symbol_filter,
            psides=pside_filter,
            reason_codes=reason_code_filter,
            statuses=status_filter,
        )
        report["cycle"] = {
            "cycle_id": str(cycle_id),
            "filters": cycle_filters,
            "event_types": sorted(event_type_filter) if event_type_filter else None,
            "matched_events": cycle_match_count,
            "events_truncated": cycle_match_count > len(cycle_events),
            "events": cycle_events,
        }
        if timeline:
            report["cycle"]["timeline"] = [
                _timeline_line(event) for event in cycle_events
            ]
        if not event_type_filter:
            report["cycle"].pop("event_types", None)
    return report
