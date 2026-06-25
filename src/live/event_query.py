from __future__ import annotations

import gzip
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


EVENT_ID_KEYS = (
    "cycle_id",
    "snapshot_id",
    "plan_id",
    "action_id",
    "order_wave_id",
    "remote_call_id",
    "remote_call_group_id",
)


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


def discover_event_files(root: str | Path) -> list[Path]:
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
    live_event = payload.get("_live_event")
    return live_event if isinstance(live_event, dict) else None


def _event_ids(live_event: dict[str, Any]) -> dict[str, Any]:
    ids = live_event.get("ids")
    return dict(ids) if isinstance(ids, dict) else {}


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
        "ids": {key: ids.get(key) for key in EVENT_ID_KEYS if ids.get(key) is not None},
    }
    if include_data:
        compact["data"] = live_event.get("data") if isinstance(live_event.get("data"), dict) else {}
    return {key: value for key, value in compact.items() if value is not None}


def build_event_report(
    root: str | Path,
    *,
    cycle_id: str | None = None,
    limit: int = 200,
    include_data: bool = False,
) -> dict[str, Any]:
    issues: list[EventIssue] = []
    try:
        files = discover_event_files(root)
    except FileNotFoundError as exc:
        files = []
        issues.append(
            EventIssue(str(root), None, "error", "path_not_found", str(exc))
        )
    if not files and not issues:
        issues.append(
            EventIssue(str(root), None, "error", "no_event_files", "no event NDJSON files found")
        )

    event_type_counts: Counter[str] = Counter()
    cycle_counts: Counter[str] = Counter()
    records_total = 0
    live_events = 0
    legacy_events = 0
    missing_cycle_id = 0
    cycle_events: list[dict[str, Any]] = []
    cycle_match_count = 0
    max_events = max(0, int(limit))

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

                    event_type = live_event.get("event_type") or row.get("kind")
                    if not event_type:
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
                        event_type = str(event_type)
                        event_type_counts[event_type] += 1
                        if row.get("kind") and str(row.get("kind")) != event_type:
                            issues.append(
                                EventIssue(
                                    str(path),
                                    line_no,
                                    "warning",
                                    "kind_event_type_mismatch",
                                    f"kind={row.get('kind')} event_type={event_type}",
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
                    if cycle_id is not None and record_cycle_id == str(cycle_id):
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
    if cycle_id is not None:
        report["cycle"] = {
            "cycle_id": str(cycle_id),
            "matched_events": cycle_match_count,
            "events_truncated": cycle_match_count > len(cycle_events),
            "events": cycle_events,
        }
    return report

