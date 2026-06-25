from __future__ import annotations

import gzip
import json
import re
import stat
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from live.event_bus import LIVE_EVENT_MONITOR_PAYLOAD_KEY, EventTypes
from live.event_query import build_event_report, discover_event_files


HARD_LOG_PATTERN = re.compile(
    r"\bTraceback\b|(?:^|\s|\[)CRITICAL(?:\s|\]|:|$)|\blevel=critical\b|\bfatal\b|\buncaught\b",
    re.IGNORECASE,
)
ATTENTION_LOG_PATTERN = re.compile(
    r"\bTraceback\b|(?:^|\s|\[)(?:CRITICAL|ERROR)(?:\s|\]|:|$)|\blevel=(?:critical|error)\b|\bfatal\b|\buncaught\b",
    re.IGNORECASE,
)
SENSITIVE_LOG_HEADER_PATTERN = re.compile(
    r"(?i)\b(authorization|proxy-authorization|x-mbx-apikey|cookie|set-cookie)"
    r"(\s*[:=]\s*)(?:bearer|basic)?\s*[^,\s;]+"
)
SENSITIVE_LOG_VALUE_PATTERN = re.compile(
    r"(?i)\b(api[-_]?key|apikey|secret|token|signature|password|passphrase|private[-_]?key)"
    r"([\"']?\s*(?:[:=]|\s)\s*)[\"']?[^,\s;&\"'}]+"
)
AUTH_SCHEME_PATTERN = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]+")
STARTUP_TIMING_BASELINE_WINDOW = 20


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _live_event_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    payload = row.get("payload")
    if not isinstance(payload, dict):
        return None
    live_event = payload.get(LIVE_EVENT_MONITOR_PAYLOAD_KEY)
    return live_event if isinstance(live_event, dict) else None


def _event_ids(live_event: dict[str, Any]) -> dict[str, Any]:
    ids = live_event.get("ids")
    return dict(ids) if isinstance(ids, dict) else {}


def _bot_key(live_event: dict[str, Any], row: dict[str, Any]) -> str:
    exchange = live_event.get("exchange") or row.get("exchange") or "unknown_exchange"
    user = live_event.get("user") or row.get("user") or "unknown_user"
    return f"{exchange}/{user}"


def _compact_problem_event(
    *,
    path: Path,
    line_no: int,
    row: dict[str, Any],
    live_event: dict[str, Any],
) -> dict[str, Any]:
    ids = _event_ids(live_event)
    return {
        key: value
        for key, value in {
            "path": str(path),
            "line": int(line_no),
            "ts": row.get("ts"),
            "seq": row.get("seq"),
            "event_type": live_event.get("event_type") or row.get("kind"),
            "level": live_event.get("level"),
            "status": live_event.get("status"),
            "reason_code": live_event.get("reason_code"),
            "exchange": live_event.get("exchange") or row.get("exchange"),
            "user": live_event.get("user") or row.get("user"),
            "symbol": live_event.get("symbol") or row.get("symbol"),
            "pside": live_event.get("pside") or row.get("pside"),
            "ids": {
                key: ids.get(key)
                for key in (
                    "cycle_id",
                    "order_wave_id",
                    "remote_call_id",
                    "remote_call_group_id",
                )
                if ids.get(key) is not None
            },
        }.items()
        if value not in (None, {}, [])
    }


def _redact_log_text(value: str, *, max_len: int = 500) -> str:
    text = str(value)
    text = SENSITIVE_LOG_HEADER_PATTERN.sub(r"\1\2[redacted]", text)
    text = SENSITIVE_LOG_VALUE_PATTERN.sub(r"\1\2[redacted]", text)
    text = AUTH_SCHEME_PATTERN.sub(r"\1 [redacted]", text)
    if len(text) > max_len:
        text = f"{text[:max_len]}...<truncated>"
    return text


def _non_negative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, parsed)


def _nearest_rank(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    rank = max(1, int(len(ordered) * float(percentile) / 100.0 + 0.999999))
    return ordered[min(len(ordered) - 1, rank - 1)]


def _median(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return int(round((ordered[midpoint - 1] + ordered[midpoint]) / 2.0))


def _ms_summary(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {
            "median_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    return {
        "median_ms": _median(values),
        "p95_ms": _nearest_rank(values, 95.0),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _usage_pct(value: int | None, budget: int | None) -> int | None:
    if value is None or budget is None or budget <= 0:
        return None
    return int(round(float(value) * 100.0 / float(budget)))


def _startup_timing_record(
    *,
    row: dict[str, Any],
    live_event: dict[str, Any],
    path: Path,
    line_no: int,
) -> dict[str, Any] | None:
    if live_event.get("event_type") != EventTypes.BOT_STARTUP_TIMING:
        return None
    data = live_event.get("data")
    if not isinstance(data, dict):
        return None
    phase = str(data.get("phase") or "").strip()
    if not phase:
        return None
    elapsed_ms = _non_negative_int(data.get("elapsed_ms"))
    since_previous_ms = _non_negative_int(data.get("since_previous_ms"))
    if elapsed_ms is None and since_previous_ms is None:
        return None
    return {
        "phase": phase,
        "elapsed_ms": elapsed_ms,
        "since_previous_ms": since_previous_ms,
        "details": data.get("details"),
        "ts": row.get("ts"),
        "seq": row.get("seq"),
        "path": str(path),
        "line": int(line_no),
    }


def _sort_startup_record_key(record: dict[str, Any]) -> tuple[int, int, str, int]:
    ts = _non_negative_int(record.get("ts"))
    seq = _non_negative_int(record.get("seq"))
    return (
        -1 if ts is None else ts,
        -1 if seq is None else seq,
        str(record.get("path") or ""),
        int(record.get("line") or 0),
    )


def _summarize_startup_timings(
    records_by_bot: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for bot_key, records in sorted(records_by_bot.items()):
        phases: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in sorted(records, key=_sort_startup_record_key):
            phases[str(record["phase"])].append(record)
        phase_summaries: dict[str, dict[str, Any]] = {}
        for phase, phase_records in sorted(phases.items()):
            window = phase_records[-STARTUP_TIMING_BASELINE_WINDOW:]
            latest = window[-1]
            elapsed_values = [
                int(record["elapsed_ms"])
                for record in window
                if record.get("elapsed_ms") is not None
            ]
            phase_values = [
                int(record["since_previous_ms"])
                for record in window
                if record.get("since_previous_ms") is not None
            ]
            elapsed_summary = _ms_summary(elapsed_values)
            phase_summary = _ms_summary(phase_values)
            latest_elapsed = latest.get("elapsed_ms")
            latest_phase = latest.get("since_previous_ms")
            phase_summaries[phase] = {
                "samples": len(window),
                "latest_ts": latest.get("ts"),
                "latest_elapsed_ms": latest_elapsed,
                "latest_since_previous_ms": latest_phase,
                "elapsed_baseline": elapsed_summary,
                "phase_baseline": phase_summary,
                "latest_elapsed_vs_p95_pct": _usage_pct(
                    latest_elapsed, elapsed_summary["p95_ms"]
                ),
                "latest_phase_vs_p95_pct": _usage_pct(
                    latest_phase, phase_summary["p95_ms"]
                ),
            }
            details = latest.get("details")
            if details not in (None, ""):
                phase_summaries[phase]["latest_details"] = _redact_log_text(
                    str(details)
                )
        summaries.append(
            {
                "bot": bot_key,
                "baseline_window": STARTUP_TIMING_BASELINE_WINDOW,
                "phases": phase_summaries,
            }
        )
    return summaries


def _is_problem_event(live_event: dict[str, Any]) -> bool:
    level = str(live_event.get("level") or "").lower()
    status = str(live_event.get("status") or "").lower()
    event_type = str(live_event.get("event_type") or "")
    return (
        level in {"error", "critical"}
        or status in {"failed", "degraded"}
        or event_type == "sink.degraded"
    )


def _is_hard_problem_event(live_event: dict[str, Any]) -> bool:
    level = str(live_event.get("level") or "").lower()
    event_type = str(live_event.get("event_type") or "")
    return level in {"error", "critical"} or event_type == "sink.degraded"


def _scan_events(
    root: str | Path,
    *,
    include_rotated: bool,
    max_problem_events: int,
) -> dict[str, Any]:
    files = discover_event_files(root, include_rotated=include_rotated)
    bots: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "events": 0,
            "invalid_ts": 0,
            "last_ts": None,
            "event_types": Counter(),
            "levels": Counter(),
            "statuses": Counter(),
            "problem_events": 0,
            "hard_problem_events": 0,
        }
    )
    problem_events: deque[dict[str, Any]] = deque(maxlen=max(0, int(max_problem_events)))
    invalid_rows = 0
    startup_timing_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for path in files:
        try:
            with _open_text(path) as stream:
                for line_no, raw_line in enumerate(stream, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_rows += 1
                        continue
                    if not isinstance(row, dict):
                        invalid_rows += 1
                        continue
                    live_event = _live_event_payload(row)
                    if live_event is None:
                        continue
                    bot_key = _bot_key(live_event, row)
                    bot = bots[bot_key]
                    bot["events"] += 1
                    ts = row.get("ts")
                    if ts is not None:
                        try:
                            bot["last_ts"] = max(int(bot["last_ts"] or 0), int(ts))
                        except (TypeError, ValueError):
                            bot["invalid_ts"] += 1
                    event_type = live_event.get("event_type") or row.get("kind")
                    if event_type:
                        bot["event_types"][str(event_type)] += 1
                    level = live_event.get("level")
                    if level:
                        bot["levels"][str(level).lower()] += 1
                    status = live_event.get("status")
                    if status:
                        bot["statuses"][str(status).lower()] += 1
                    if _is_problem_event(live_event):
                        bot["problem_events"] += 1
                        hard = _is_hard_problem_event(live_event)
                        if hard:
                            bot["hard_problem_events"] += 1
                        problem_event = _compact_problem_event(
                            path=path,
                            line_no=line_no,
                            row=row,
                            live_event=live_event,
                        )
                        problem_event["hard"] = hard
                        problem_events.append(
                            problem_event
                        )
                    startup_timing = _startup_timing_record(
                        row=row,
                        live_event=live_event,
                        path=path,
                        line_no=line_no,
                    )
                    if startup_timing is not None:
                        startup_timing_records[bot_key].append(startup_timing)
        except OSError:
            invalid_rows += 1

    return {
        "invalid_rows": invalid_rows,
        "bots": [
            {
                "bot": key,
                "events": int(value["events"]),
                "invalid_ts": int(value["invalid_ts"]),
                "last_ts": value["last_ts"],
                "problem_events": int(value["problem_events"]),
                "hard_problem_events": int(value["hard_problem_events"]),
                "event_types": dict(value["event_types"].most_common(10)),
                "levels": dict(sorted(value["levels"].items())),
                "statuses": dict(sorted(value["statuses"].items())),
            }
            for key, value in sorted(bots.items())
        ],
        "problem_events": list(problem_events),
        "problem_event_count": sum(int(value["problem_events"]) for value in bots.values()),
        "hard_problem_event_count": sum(
            int(value["hard_problem_events"]) for value in bots.values()
        ),
        "startup_timings": _summarize_startup_timings(startup_timing_records),
    }


def _recent_log_files(root: str | Path, *, max_files: int) -> list[Path]:
    path = Path(root).expanduser()
    if not path.exists() or not path.is_dir():
        return []
    files: list[tuple[float, Path]] = []
    seen_inodes: set[tuple[int, int]] = set()
    for candidate in path.glob("*.log*"):
        try:
            file_stat = candidate.stat()
        except OSError:
            continue
        if not stat.S_ISREG(file_stat.st_mode):
            continue
        inode_key = (int(file_stat.st_dev), int(file_stat.st_ino))
        if inode_key in seen_inodes:
            continue
        seen_inodes.add(inode_key)
        files.append((float(file_stat.st_mtime), candidate))
    return [
        candidate
        for _, candidate in sorted(files, key=lambda item: item[0], reverse=True)[
            : max(0, int(max_files))
        ]
    ]


def _tail_lines(path: Path, *, max_lines: int) -> list[tuple[int, str]]:
    try:
        with _open_text(path) as stream:
            rows = deque(enumerate(stream, start=1), maxlen=max(0, int(max_lines)))
    except OSError:
        return []
    return [(line_no, line.rstrip("\n")) for line_no, line in rows]


def default_logs_root_for_monitor(monitor_root: str | Path) -> Path | None:
    path = Path(monitor_root).expanduser()
    start = path.parent if path.is_file() else path
    for ancestor in (start, *start.parents):
        candidate = ancestor / "logs"
        if candidate.is_dir():
            return candidate
    return None


def _scan_logs(
    root: str | Path | None,
    *,
    max_files: int,
    tail_lines: int,
    max_matches: int,
) -> dict[str, Any]:
    if root is None:
        return {
            "root": None,
            "files_scanned": 0,
            "hard_matches": 0,
            "attention_matches": 0,
            "matches": [],
        }
    files = _recent_log_files(root, max_files=max_files)
    matches: list[dict[str, Any]] = []
    hard_matches = 0
    attention_matches = 0
    for path in files:
        for line_no, line in _tail_lines(path, max_lines=tail_lines):
            if not ATTENTION_LOG_PATTERN.search(line):
                continue
            attention_matches += 1
            hard = bool(HARD_LOG_PATTERN.search(line))
            if hard:
                hard_matches += 1
            if len(matches) < max(0, int(max_matches)):
                matches.append(
                    {
                        "path": str(path),
                        "line": int(line_no),
                        "hard": hard,
                        "text": _redact_log_text(line, max_len=500),
                    }
                )
    return {
        "root": str(Path(root).expanduser()),
        "files_scanned": len(files),
        "hard_matches": hard_matches,
        "attention_matches": attention_matches,
        "matches": matches,
    }


def build_live_smoke_report(
    monitor_root: str | Path = "monitor",
    *,
    logs_root: str | Path | None = "logs",
    include_rotated: bool = False,
    max_problem_events: int = 50,
    max_log_files: int = 8,
    log_tail_lines: int = 300,
    max_log_matches: int = 50,
) -> dict[str, Any]:
    event_report = build_event_report(
        monitor_root,
        include_rotated=include_rotated,
        limit=max_problem_events,
    )
    event_scan: dict[str, Any]
    if event_report.get("files_scanned"):
        event_scan = _scan_events(
            monitor_root,
            include_rotated=include_rotated,
            max_problem_events=max_problem_events,
        )
    else:
        event_scan = {
            "invalid_rows": 0,
            "bots": [],
            "problem_events": [],
            "problem_event_count": 0,
            "hard_problem_event_count": 0,
            "startup_timings": [],
        }
    log_scan = _scan_logs(
        logs_root,
        max_files=max_log_files,
        tail_lines=log_tail_lines,
        max_matches=max_log_matches,
    )
    hard_failures = (
        int(event_report["error_count"])
        + int(event_scan["invalid_rows"])
        + int(event_scan["hard_problem_event_count"])
        + int(log_scan["hard_matches"])
    )
    attention_count = int(event_scan["problem_event_count"]) + int(
        log_scan["attention_matches"]
    )
    return {
        "ok": hard_failures == 0,
        "attention": attention_count > 0,
        "hard_failures": hard_failures,
        "attention_count": attention_count,
        "monitor": {
            "root": event_report.get("root"),
            "include_rotated": event_report.get("include_rotated"),
            "files_scanned": event_report.get("files_scanned"),
            "records_total": event_report.get("records_total"),
            "live_events": event_report.get("live_events"),
            "legacy_events": event_report.get("legacy_events"),
            "error_count": event_report.get("error_count"),
            "warning_count": event_report.get("warning_count"),
            "event_types": event_report["event_types"],
            "cycle_ids_sample": event_report["cycle_ids_sample"],
            "issues": event_report["issues"],
        },
        "bots": event_scan["bots"],
        "startup_timings": event_scan["startup_timings"],
        "problem_events": event_scan["problem_events"],
        "hard_problem_event_count": event_scan["hard_problem_event_count"],
        "logs": log_scan,
    }
