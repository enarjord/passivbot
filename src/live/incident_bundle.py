from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from live.event_bus import LIVE_EVENT_ID_KEYS, LIVE_EVENT_MONITOR_PAYLOAD_KEY
from live.event_file_rows import event_file_rows
from live.event_query import (
    build_event_report,
    discover_event_files,
    discover_event_files_with_metadata,
)
from live.smoke_report import (
    DEFAULT_LOG_WINDOW_UNPARSED_POLICY,
    _redact_log_text,
    build_live_smoke_report,
    default_logs_root_for_monitor,
)


BUNDLE_VERSION = 1
MONITOR_SNAPSHOT_FILE_NAMES = frozenset({"state.latest.json", "manifest.json"})
SNAPSHOT_SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "passphrase",
    "password",
    "private_key",
    "secret",
    "signature",
    "token",
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: Any) -> str:
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _live_event_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    payload = row.get("payload")
    if not isinstance(payload, dict):
        return None
    live_event = payload.get(LIVE_EVENT_MONITOR_PAYLOAD_KEY)
    return live_event if isinstance(live_event, dict) else None


def _event_ids(live_event: dict[str, Any]) -> dict[str, Any]:
    ids = live_event.get("ids")
    return dict(ids) if isinstance(ids, dict) else {}


def _compact_event(
    *,
    path: Path,
    line_no: int,
    row: dict[str, Any],
    live_event: dict[str, Any],
    include_data: bool,
) -> dict[str, Any]:
    ids = _event_ids(live_event)
    record = {
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
        "ids": {key: ids.get(key) for key in LIVE_EVENT_ID_KEYS if ids.get(key) is not None},
    }
    if include_data:
        data = live_event.get("data")
        record["data"] = dict(data) if isinstance(data, dict) else {}
    return {key: value for key, value in record.items() if value not in (None, {}, [])}


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


def _build_time_window_report(
    monitor_root: str | Path,
    *,
    since_ms: int | None,
    until_ms: int | None,
    include_rotated: bool,
    include_data: bool,
    event_tail_lines: int = 0,
    limit: int,
) -> dict[str, Any]:
    if since_ms is None and until_ms is None:
        return {
            "enabled": False,
            "filters": {},
            "matched_events": 0,
            "events_truncated": False,
            "events": [],
            "timeline": [],
            "issues": [],
        }
    filters = {
        key: value
        for key, value in {"since_ms": since_ms, "until_ms": until_ms}.items()
        if value is not None
    }
    issues: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    matched_events = 0
    max_events = max(0, int(limit))
    max_event_tail_lines = max(0, int(event_tail_lines))
    event_tail_limited_files = 0
    event_tail_skipped_lines = 0
    event_tail_skipped_lines_exact = True
    event_tail_skipped_bytes = 0
    event_tail_line_numbers_exact = True
    event_tail_methods: Counter[str] = Counter()
    try:
        files = discover_event_files(monitor_root, include_rotated=include_rotated)
    except FileNotFoundError as exc:
        files = []
        issues.append(
            {
                "path": str(monitor_root),
                "line": None,
                "severity": "error",
                "code": "path_not_found",
                "message": str(exc),
            }
        )

    for path in files:
        try:
            with event_file_rows(path, max_tail_lines=max_event_tail_lines) as (
                rows,
                row_window,
            ):
                if max_event_tail_lines and row_window.limited:
                    event_tail_limited_files += 1
                    if row_window.skipped_lines is None:
                        event_tail_skipped_lines_exact = False
                    else:
                        event_tail_skipped_lines += int(row_window.skipped_lines)
                    event_tail_skipped_bytes += int(row_window.skipped_bytes)
                    event_tail_line_numbers_exact = (
                        event_tail_line_numbers_exact
                        and bool(row_window.line_numbers_exact)
                    )
                    event_tail_methods[str(row_window.method)] += 1
                for line_no, raw_line in rows:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        issues.append(
                            {
                                "path": str(path),
                                "line": line_no,
                                "severity": "warning",
                                "code": "invalid_json",
                                "message": str(exc),
                            }
                        )
                        continue
                    if not isinstance(row, dict):
                        continue
                    ts_ms = _safe_int(row.get("ts"))
                    if ts_ms is None:
                        continue
                    if since_ms is not None and ts_ms < int(since_ms):
                        continue
                    if until_ms is not None and ts_ms > int(until_ms):
                        continue
                    live_event = _live_event_payload(row)
                    if live_event is None:
                        continue
                    matched_events += 1
                    if len(events) < max_events:
                        events.append(
                            _compact_event(
                                path=path,
                                line_no=line_no,
                                row=row,
                                live_event=live_event,
                                include_data=include_data,
                            )
                        )
        except OSError as exc:
            issues.append(
                {
                    "path": str(path),
                    "line": None,
                    "severity": "error",
                    "code": "read_failed",
                    "message": str(exc),
                }
            )
    report = {
        "enabled": True,
        "filters": filters,
        "matched_events": matched_events,
        "events_truncated": matched_events > len(events),
        "events": events,
        "timeline": [_timeline_line(event) for event in events],
        "issues": issues,
    }
    if max_event_tail_lines:
        report["event_tail_lines"] = max_event_tail_lines
        report["event_tail_limited_files"] = event_tail_limited_files
        report["event_tail_skipped_lines"] = event_tail_skipped_lines
        report["event_tail_skipped_lines_exact"] = event_tail_skipped_lines_exact
        report["event_tail_skipped_bytes"] = event_tail_skipped_bytes
        report["event_tail_line_numbers_exact"] = event_tail_line_numbers_exact
        report["event_tail_methods"] = dict(sorted(event_tail_methods.items()))
    return report


def _git_metadata(cwd: str | Path | None = None) -> dict[str, Any]:
    root = Path(cwd or Path.cwd()).expanduser()

    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    return {
        "cwd": str(root),
        "head": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "status_short": run_git(["status", "--short"]),
        "remote_url": _redact_url_userinfo(run_git(["remote", "get-url", "origin"])),
    }


def _redact_url_userinfo(value: str | None) -> str | None:
    if not value:
        return value
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if not parts.scheme or not parts.netloc:
        return value
    if parts.username is None and parts.password is None:
        return value
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit(
        (parts.scheme, f"[redacted]@{host}", parts.path, parts.query, parts.fragment)
    )


def _runtime_metadata() -> dict[str, Any]:
    loadavg: tuple[float, ...] | None
    try:
        loadavg = tuple(float(value) for value in os.getloadavg())
    except (AttributeError, OSError):
        loadavg = None
    return {
        "created_at_ms": int(time.time() * 1000),
        "created_at_iso": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "loadavg": loadavg,
    }


def _config_hashes(paths: list[str | Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        item: dict[str, Any] = {"path": str(path)}
        try:
            stat = path.stat()
        except OSError as exc:
            item.update({"ok": False, "error": str(exc)})
            out.append(item)
            continue
        if not path.is_file():
            item.update({"ok": False, "error": "not a regular file"})
            out.append(item)
            continue
        item.update(
            {
                "ok": True,
                "size_bytes": int(stat.st_size),
                "mtime_ms": int(stat.st_mtime * 1000),
                "sha256": _sha256_file(path),
            }
        )
        out.append(item)
    return out


def _is_sensitive_snapshot_key(key: Any) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return any(fragment in normalized for fragment in SNAPSHOT_SENSITIVE_KEY_FRAGMENTS)


def _redact_snapshot_value(value: Any, *, sensitive_key: bool = False) -> Any:
    if sensitive_key and value is not None:
        return "[redacted]"
    if isinstance(value, str):
        redacted = _redact_log_text(value, max_len=1_000_000)
        return _redact_url_userinfo(redacted) or redacted
    if isinstance(value, list):
        return [_redact_snapshot_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _redact_snapshot_value(
                item,
                sensitive_key=_is_sensitive_snapshot_key(key),
            )
            for key, item in value.items()
        }
    return value


def _redacted_snapshot_bytes(path: Path) -> bytes:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return (_redact_log_text(text, max_len=10_000_000) + "\n").encode("utf-8")
    redacted = _redact_snapshot_value(parsed)
    return (json.dumps(redacted, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _snapshot_files(
    monitor_root: str | Path,
    *,
    max_files: int,
    max_file_bytes: int,
) -> list[Path]:
    root = Path(monitor_root).expanduser()
    if root.is_file():
        root = root.parent
    if not root.exists() or not root.is_dir():
        return []
    candidates: list[tuple[float, Path]] = []
    for path in root.rglob("*.json"):
        if any(part == "events" for part in path.parts):
            continue
        if path.name not in MONITOR_SNAPSHOT_FILE_NAMES:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if not path.is_file() or stat.st_size > max_file_bytes:
            continue
        candidates.append((float(stat.st_mtime), path))
    return [
        path
        for _, path in sorted(candidates, key=lambda item: item[0], reverse=True)[
            : max(0, int(max_files))
        ]
    ]


def _copy_snapshot_files(
    *,
    monitor_root: str | Path,
    bundle_root: Path,
    max_files: int,
    max_file_bytes: int,
) -> list[dict[str, Any]]:
    root = Path(monitor_root).expanduser()
    if root.is_file():
        root = root.parent
    copied: list[dict[str, Any]] = []
    for path in _snapshot_files(
        root,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
    ):
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = Path(hashlib.sha256(str(path).encode()).hexdigest()[:12]) / path.name
        dest = bundle_root / "monitor_snapshots" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = _redacted_snapshot_bytes(path)
            dest.write_bytes(data)
            copied.append(
                {
                    "path": str(path),
                    "bundle_path": str(dest.relative_to(bundle_root)),
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "redacted": True,
                }
            )
        except OSError as exc:
            copied.append({"path": str(path), "ok": False, "error": str(exc)})
    return copied


def _matched_segment_paths(*reports: dict[str, Any]) -> set[str]:
    paths: set[str] = set()

    def collect_events(events: Any) -> None:
        if not isinstance(events, list):
            return
        for event in events:
            if isinstance(event, dict) and event.get("path"):
                paths.add(str(event["path"]))

    for report in reports:
        query = report.get("query")
        if isinstance(query, dict):
            collect_events(query.get("events"))
        cycle = report.get("cycle")
        if isinstance(cycle, dict):
            collect_events(cycle.get("events"))
        collect_events(report.get("events"))
    return paths


def _event_report_result_summary(event_report: dict[str, Any]) -> dict[str, Any]:
    cycle = event_report.get("cycle")
    query = event_report.get("query")
    cycle_section = cycle if isinstance(cycle, dict) else None
    query_section = query if isinstance(query, dict) else None
    trace_section = cycle_section or query_section or {}

    trace_summary = trace_section.get("trace_summary")
    order_trace = trace_section.get("order_trace")
    cycle_trace = trace_section.get("cycle_trace")
    return {
        "files_scanned": event_report.get("files_scanned"),
        "file_discovery": event_report.get("file_discovery") or {},
        "live_events": event_report.get("live_events"),
        "error_count": event_report.get("error_count"),
        "warning_count": event_report.get("warning_count"),
        "event_window": event_report.get("event_window"),
        "cycle_matched_events": (
            cycle_section.get("matched_events") if cycle_section is not None else None
        ),
        "query_matched_events": (
            query_section.get("matched_events") if query_section is not None else None
        ),
        "trace_summary_matched_events": (
            trace_summary.get("matched_events")
            if isinstance(trace_summary, dict)
            else None
        ),
        "order_trace_matched_events": (
            order_trace.get("matched_order_events")
            if isinstance(order_trace, dict)
            else None
        ),
        "cycle_trace_matched_events": (
            cycle_trace.get("matched_cycle_events")
            if isinstance(cycle_trace, dict)
            else None
        ),
    }


def _problem_report_result_summary(problem_report: dict[str, Any]) -> dict[str, Any]:
    query = problem_report.get("query")
    query_section = query if isinstance(query, dict) else {}
    trace_summary = query_section.get("trace_summary")
    return {
        "enabled": bool(problem_report),
        "files_scanned": problem_report.get("files_scanned"),
        "live_events": problem_report.get("live_events"),
        "error_count": problem_report.get("error_count"),
        "warning_count": problem_report.get("warning_count"),
        "matched_events": query_section.get("matched_events"),
        "events_truncated": query_section.get("events_truncated"),
        "trace_summary_matched_events": (
            trace_summary.get("matched_events")
            if isinstance(trace_summary, dict)
            else None
        ),
    }


def _copy_event_segments(
    *,
    monitor_root: str | Path,
    bundle_root: Path,
    event_report: dict[str, Any],
    window_report: dict[str, Any],
    problem_report: dict[str, Any],
    include_rotated: bool,
    include_segments: bool,
    max_total_bytes: int,
) -> dict[str, Any]:
    file_discovery: dict[str, Any] = {
        "candidate_files": 0,
        "event_segments": 0,
        "rotated_skipped": 0,
        "scope_pruned": 0,
        "bot_path_pruning_applied": False,
        "opaque_bot_id_full_scan": False,
    }
    try:
        discovery = discover_event_files_with_metadata(
            monitor_root, include_rotated=include_rotated
        )
        discovered = discovery.files
        file_discovery = discovery.to_dict()
    except FileNotFoundError:
        discovered = []
    matched_paths = _matched_segment_paths(event_report, window_report, problem_report)
    if matched_paths:
        paths = [Path(path) for path in sorted(matched_paths)]
    else:
        paths = discovered
    total_bytes = 0
    files: list[dict[str, Any]] = []
    for path in paths:
        item: dict[str, Any] = {"path": str(path)}
        try:
            stat = path.stat()
        except OSError as exc:
            item.update({"included": False, "reason": str(exc)})
            files.append(item)
            continue
        size = int(stat.st_size)
        item["size_bytes"] = size
        if not include_segments:
            item.update({"included": False, "reason": "disabled"})
            files.append(item)
            continue
        if total_bytes + size > max(0, int(max_total_bytes)):
            item.update({"included": False, "reason": "max_total_bytes"})
            files.append(item)
            continue
        digest = hashlib.sha256(str(path).encode()).hexdigest()[:12]
        dest = bundle_root / "event_segments" / f"{digest}_{path.name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = path.read_bytes()
            item["sha256"] = hashlib.sha256(data).hexdigest()
            dest.write_bytes(data)
        except OSError as exc:
            item.update({"included": False, "reason": str(exc)})
            files.append(item)
            continue
        total_bytes += size
        item.update(
            {
                "included": True,
                "bundle_path": str(dest.relative_to(bundle_root)),
            }
        )
        files.append(item)
    return {
        "include_segments": bool(include_segments),
        "include_rotated": bool(include_rotated),
        "max_total_bytes": int(max_total_bytes),
        "file_discovery": file_discovery,
        "total_included_bytes": total_bytes,
        "files": files,
    }


def _tar_directory(source_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as tar:
        for path in sorted(source_dir.rglob("*")):
            tar.add(path, arcname=str(path.relative_to(source_dir)), recursive=False)


def build_live_incident_bundle(
    monitor_root: str | Path = "monitor",
    *,
    output_path: str | Path | None = None,
    logs_root: str | Path | None = None,
    config_paths: list[str | Path] | None = None,
    include_processes: bool = False,
    supervisor_config: str | Path | None = None,
    process_command_match: str = "passivbot live",
    cycle_id: str | None = None,
    event_type: str | list[str] | None = None,
    level: str | list[str] | None = None,
    exchange: str | list[str] | None = None,
    user: str | list[str] | None = None,
    bot_id: str | list[str] | None = None,
    order_wave_id: str | list[str] | None = None,
    remote_call_id: str | list[str] | None = None,
    remote_call_group_id: str | list[str] | None = None,
    symbol: str | list[str] | None = None,
    pside: str | list[str] | None = None,
    side: str | list[str] | None = None,
    reason_code: str | list[str] | None = None,
    status: str | list[str] | None = None,
    source: str | list[str] | None = None,
    component: str | list[str] | None = None,
    tag: str | list[str] | None = None,
    data_eq: str | list[str] | None = None,
    since_ms: int | None = None,
    until_ms: int | None = None,
    include_data: bool = False,
    include_trace_report: bool = True,
    include_problem_report: bool = True,
    include_rotated: bool = False,
    include_event_segments: bool = True,
    max_events: int = 500,
    max_problem_events: int = 50,
    max_log_files: int = 8,
    log_tail_lines: int = 500,
    max_log_matches: int = 100,
    log_window_unparsed_policy: str = DEFAULT_LOG_WINDOW_UNPARSED_POLICY,
    event_tail_lines: int = 0,
    max_snapshot_files: int = 20,
    max_snapshot_file_bytes: int = 1_000_000,
    max_event_segment_bytes: int = 10_000_000,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    monitor_path = Path(monitor_root).expanduser()
    if logs_root is None:
        logs_path = default_logs_root_for_monitor(monitor_path)
    else:
        logs_path = Path(logs_root).expanduser() if str(logs_root).strip() else None
    out_path = (
        Path(output_path).expanduser()
        if output_path is not None
        else Path(f"passivbot_incident_bundle_{_utc_stamp()}.tar.gz")
    )

    event_report = build_event_report(
        monitor_path,
        cycle_id=cycle_id,
        event_type=event_type,
        level=level,
        exchange=exchange,
        user=user,
        bot_id=bot_id,
        order_wave_id=order_wave_id,
        remote_call_id=remote_call_id,
        remote_call_group_id=remote_call_group_id,
        symbol=symbol,
        pside=pside,
        side=side,
        reason_code=reason_code,
        status=status,
        source=source,
        component=component,
        tag=tag,
        data_eq=data_eq,
        event_tail_lines=event_tail_lines,
        limit=max_events,
        include_data=include_data,
        include_rotated=include_rotated,
        timeline=True,
        trace_summary=include_trace_report,
        order_trace=include_trace_report,
        cycle_trace=include_trace_report and cycle_id is not None,
    )
    problem_report = (
        build_event_report(
            monitor_path,
            cycle_id=cycle_id,
            event_type=event_type,
            level=level,
            exchange=exchange,
            user=user,
            bot_id=bot_id,
            order_wave_id=order_wave_id,
            remote_call_id=remote_call_id,
            remote_call_group_id=remote_call_group_id,
            symbol=symbol,
            pside=pside,
            side=side,
            reason_code=reason_code,
            status=status,
            source=source,
            component=component,
            tag=tag,
            data_eq=data_eq,
            problem_events=True,
            since_ms=since_ms,
            until_ms=until_ms,
            event_tail_lines=event_tail_lines,
            limit=max_problem_events,
            include_data=include_data,
            include_rotated=include_rotated,
            timeline=True,
            trace_summary=True,
        )
        if include_problem_report
        else {}
    )
    window_report = _build_time_window_report(
        monitor_path,
        since_ms=since_ms,
        until_ms=until_ms,
        include_rotated=include_rotated,
        include_data=include_data,
        event_tail_lines=event_tail_lines,
        limit=max_events,
    )
    smoke_report = build_live_smoke_report(
        monitor_path,
        logs_root=logs_path,
        include_processes=include_processes,
        supervisor_config=supervisor_config,
        process_command_match=process_command_match,
        include_rotated=include_rotated,
        since_ms=since_ms,
        until_ms=until_ms,
        event_tail_lines=event_tail_lines,
        max_problem_events=max_problem_events,
        max_log_files=max_log_files,
        log_tail_lines=log_tail_lines,
        max_log_matches=max_log_matches,
        log_window_unparsed_policy=log_window_unparsed_policy,
    )

    with tempfile.TemporaryDirectory(prefix="passivbot_incident_bundle_") as tmp_name:
        bundle_root = Path(tmp_name)
        config_hashes = _config_hashes(list(config_paths or []))
        snapshots = _copy_snapshot_files(
            monitor_root=monitor_path,
            bundle_root=bundle_root,
            max_files=max_snapshot_files,
            max_file_bytes=max_snapshot_file_bytes,
        )
        segment_manifest = _copy_event_segments(
            monitor_root=monitor_path,
            bundle_root=bundle_root,
            event_report=event_report,
            window_report=window_report,
            problem_report=problem_report,
            include_rotated=include_rotated,
            include_segments=include_event_segments,
            max_total_bytes=max_event_segment_bytes,
        )
        metadata = {
            "bundle_version": BUNDLE_VERSION,
            "monitor_root": str(monitor_path),
            "logs_root": str(logs_path) if logs_path is not None else None,
            "output_path": str(out_path),
            "filters": {
                key: value
                for key, value in {
                    "cycle_id": cycle_id,
                    "event_type": event_type,
                    "level": level,
                    "exchange": exchange,
                    "user": user,
                    "bot_id": bot_id,
                    "order_wave_id": order_wave_id,
                    "remote_call_id": remote_call_id,
                    "remote_call_group_id": remote_call_group_id,
                    "symbol": symbol,
                    "pside": pside,
                    "side": side,
                    "reason_code": reason_code,
                    "status": status,
                    "source": source,
                    "component": component,
                    "tag": tag,
                    "data_eq": data_eq,
                    "since_ms": since_ms,
                    "until_ms": until_ms,
                    "event_tail_lines": event_tail_lines if event_tail_lines else None,
                    "include_rotated": include_rotated,
                    "include_data": include_data,
                    "include_trace_report": include_trace_report,
                    "include_problem_report": include_problem_report,
                    "include_processes": include_processes,
                    "supervisor_config": str(supervisor_config)
                    if supervisor_config is not None
                    else None,
                }.items()
                if value not in (None, [], "")
            },
            "runtime": _runtime_metadata(),
            "git": _git_metadata(cwd=cwd),
            "config_hashes": config_hashes,
            "monitor_snapshots": snapshots,
            "event_segments": segment_manifest,
        }

        _write_json(bundle_root / "manifest.json", metadata)
        _write_json(bundle_root / "event_report.json", event_report)
        if include_problem_report:
            _write_json(bundle_root / "problem_event_report.json", problem_report)
        _write_json(bundle_root / "time_window_report.json", window_report)
        _write_json(bundle_root / "smoke_report.json", smoke_report)
        timeline_lines: list[str] = []
        for section in ("cycle", "query"):
            value = event_report.get(section)
            if isinstance(value, dict):
                timeline_lines.extend(str(line) for line in value.get("timeline", []))
        problem_query = problem_report.get("query")
        if isinstance(problem_query, dict):
            problem_timeline = problem_query.get("timeline")
            if isinstance(problem_timeline, list):
                timeline_lines.extend(str(line) for line in problem_timeline)
        timeline_lines.extend(str(line) for line in window_report.get("timeline", []))
        (bundle_root / "timeline.txt").write_text(
            "\n".join(timeline_lines) + ("\n" if timeline_lines else ""),
            encoding="utf-8",
        )
        _write_json(bundle_root / "config_hashes.json", config_hashes)
        _write_json(bundle_root / "event_segments_manifest.json", segment_manifest)
        _tar_directory(bundle_root, out_path)

    hard_failures = int(smoke_report.get("hard_failures", 0)) + int(
        event_report.get("error_count", 0)
    )
    hard_failures += sum(
        1
        for issue in window_report.get("issues", [])
        if isinstance(issue, dict) and issue.get("severity") == "error"
    )
    return {
        "ok": hard_failures == 0,
        "bundle_path": str(out_path),
        "bundle_version": BUNDLE_VERSION,
        "hard_failures": hard_failures,
        "event_report": _event_report_result_summary(event_report),
        "problem_event_report": _problem_report_result_summary(problem_report),
        "time_window": {
            "enabled": window_report.get("enabled"),
            "matched_events": window_report.get("matched_events"),
            "events_truncated": window_report.get("events_truncated"),
            "event_tail_lines": window_report.get("event_tail_lines"),
            "event_tail_limited_files": window_report.get("event_tail_limited_files"),
            "event_tail_skipped_lines": window_report.get("event_tail_skipped_lines"),
            "event_tail_skipped_lines_exact": window_report.get(
                "event_tail_skipped_lines_exact"
            ),
            "event_tail_skipped_bytes": window_report.get("event_tail_skipped_bytes"),
            "event_tail_line_numbers_exact": window_report.get(
                "event_tail_line_numbers_exact"
            ),
            "event_tail_methods": window_report.get("event_tail_methods"),
        },
        "smoke_report": {
            "ok": smoke_report.get("ok"),
            "attention": smoke_report.get("attention"),
            "hard_failures": smoke_report.get("hard_failures"),
            "attention_count": smoke_report.get("attention_count"),
            "event_window": smoke_report.get("event_window"),
            "processes": {
                "enabled": smoke_report.get("processes", {}).get("enabled"),
                "ok": smoke_report.get("processes", {}).get("ok"),
                "expected_total": smoke_report.get("processes", {}).get(
                    "expected_total"
                ),
                "running_live_total": smoke_report.get("processes", {}).get(
                    "running_live_total"
                ),
                "missing_expected": len(
                    smoke_report.get("processes", {}).get("missing_expected", [])
                ),
            },
        },
        "config_hashes": len(config_hashes),
        "monitor_snapshots": len(metadata["monitor_snapshots"]),
        "event_segments": {
            "files": len(segment_manifest["files"]),
            "included": sum(1 for item in segment_manifest["files"] if item.get("included")),
            "file_discovery": segment_manifest.get("file_discovery") or {},
            "total_included_bytes": segment_manifest["total_included_bytes"],
        },
    }
