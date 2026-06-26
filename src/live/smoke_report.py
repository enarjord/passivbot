from __future__ import annotations

import gzip
import json
import re
import stat
import subprocess
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

from live.event_bus import LIVE_EVENT_MONITOR_PAYLOAD_KEY, EventTypes
from live.event_query import build_event_report, discover_event_files


PYTHON_TRACEBACK_HEADER_PATTERN = r"\bTraceback\s+\(most recent call last\):"
HARD_LOG_PATTERN = re.compile(
    rf"{PYTHON_TRACEBACK_HEADER_PATTERN}|"
    r"(?:^|\s|\[)CRITICAL(?:\s|\]|:|$)|"
    r"\blevel=critical\b|\bfatal\b|\buncaught\b",
    re.IGNORECASE,
)
ATTENTION_LOG_PATTERN = re.compile(
    rf"{PYTHON_TRACEBACK_HEADER_PATTERN}|"
    r"(?:^|\s|\[)(?:CRITICAL|ERROR)(?:\s|\]|:|$)|"
    r"\blevel=(?:critical|error)\b|\bfatal\b|\buncaught\b",
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
LOG_LINE_TS_PATTERN = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\b"
)
STARTUP_TIMING_BASELINE_WINDOW = 20
DEFAULT_PROCESS_MATCH = "passivbot live"
LOG_WINDOW_UNPARSED_POLICIES = {"keep", "drop"}
DEFAULT_LOG_WINDOW_UNPARSED_POLICY = "keep"
REMOTE_CALL_FAILURE_GROUP_LIMIT = 20
RISK_EVENT_GROUP_LIMIT = 20
PROBLEM_EVENT_GROUP_LIMIT = 20
RISK_EVENT_TYPES = {
    EventTypes.RISK_MODE_CHANGED,
    EventTypes.HSL_TRANSITION,
    EventTypes.HSL_STATUS,
    EventTypes.HSL_RED_TRIGGERED,
    EventTypes.HSL_COOLDOWN_STARTED,
    EventTypes.HSL_COOLDOWN_ENDED,
}
PROBLEM_EVENT_DATA_KEYS: dict[str, tuple[str, ...]] = {
    EventTypes.CYCLE_DEGRADED: ("details", "authoritative_epoch"),
    EventTypes.EMA_UNAVAILABLE: (
        "optional_drop_count",
        "optional_drop_groups",
        "candidate_unavailable",
        "candidate_unavailable_groups",
        "unavailable",
        "unavailable_reasons",
    ),
}
PROBLEM_EVENT_DATA_MAX_DEPTH = 4
PROBLEM_EVENT_DATA_MAX_ITEMS = 8
PROBLEM_EVENT_DATA_MAX_TEXT = 240
PROCESS_REPORT_FIELDS = {
    "pid",
    "age_s",
    "stat",
    "cpu_pct",
    "mem_pct",
    "command",
    "command_key",
}


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


def _compact_problem_event_data_value(value: Any, *, depth: int = 0) -> Any:
    if value in (None, "", [], {}, ()):
        return None
    if isinstance(value, bool) or isinstance(value, int) or isinstance(value, float):
        return value
    if isinstance(value, str):
        text = _redact_log_text(value)
        if len(text) > PROBLEM_EVENT_DATA_MAX_TEXT:
            text = text[:PROBLEM_EVENT_DATA_MAX_TEXT] + "..."
        return text
    if depth >= PROBLEM_EVENT_DATA_MAX_DEPTH:
        return _compact_problem_event_data_value(str(value), depth=depth)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value, key=str)[:PROBLEM_EVENT_DATA_MAX_ITEMS]:
            compact = _compact_problem_event_data_value(
                value.get(key),
                depth=depth + 1,
            )
            if compact not in (None, "", [], {}):
                out[str(key)] = compact
        truncated = max(0, len(value) - PROBLEM_EVENT_DATA_MAX_ITEMS)
        if truncated:
            out["truncated"] = truncated
        return out or None
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        out = [
            compact
            for item in values[:PROBLEM_EVENT_DATA_MAX_ITEMS]
            if (compact := _compact_problem_event_data_value(item, depth=depth + 1))
            not in (None, "", [], {})
        ]
        truncated = max(0, len(values) - PROBLEM_EVENT_DATA_MAX_ITEMS)
        if truncated:
            out.append({"truncated": truncated})
        return out or None
    return _compact_problem_event_data_value(str(value), depth=depth)


def _compact_problem_event_data(live_event: dict[str, Any]) -> dict[str, Any]:
    event_type = str(live_event.get("event_type") or "")
    keys = PROBLEM_EVENT_DATA_KEYS.get(event_type)
    if not keys:
        return {}
    data = live_event.get("data")
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for key in keys:
        if key not in data:
            continue
        compact = _compact_problem_event_data_value(data.get(key))
        if compact not in (None, "", [], {}):
            out[key] = compact
    return out


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
            "latest_data": _compact_problem_event_data(live_event),
        }.items()
        if value not in (None, {}, [])
    }


def _sort_event_position_key(
    *,
    ts: Any,
    seq: Any,
    path: Path | str,
    line_no: int,
) -> tuple[int, int, str, int]:
    ts_value = _non_negative_int(ts)
    seq_value = _non_negative_int(seq)
    return (
        -1 if ts_value is None else ts_value,
        -1 if seq_value is None else seq_value,
        str(path),
        int(line_no),
    )


def _remote_call_failure_group(
    *,
    bot_key: str,
    row: dict[str, Any],
    live_event: dict[str, Any],
    path: Path,
    line_no: int,
) -> dict[str, Any]:
    data = live_event.get("data")
    payload = data if isinstance(data, dict) else {}
    ids = _event_ids(live_event)
    latest_error = payload.get("error_repr") or payload.get("error")
    return {
        "bot": bot_key,
        "reason_code": live_event.get("reason_code"),
        "surface": payload.get("surface"),
        "error_type": payload.get("error_type"),
        "component": live_event.get("component"),
        "count": 1,
        "latest_ts": row.get("ts"),
        "latest_seq": row.get("seq"),
        "latest_path": str(path),
        "latest_line": int(line_no),
        "latest_elapsed_ms": _non_negative_int(payload.get("elapsed_ms")),
        "latest_error": _redact_log_text(str(latest_error))
        if latest_error not in (None, "")
        else None,
        "latest_ids": {
            key: ids.get(key)
            for key in ("cycle_id", "remote_call_id", "remote_call_group_id")
            if ids.get(key) is not None
        },
    }


def _merge_remote_call_failure_group(
    groups: dict[tuple[Any, ...], dict[str, Any]],
    group: dict[str, Any],
) -> None:
    key = (
        group.get("bot"),
        group.get("reason_code"),
        group.get("surface"),
        group.get("error_type"),
        group.get("component"),
    )
    existing = groups.get(key)
    if existing is None:
        groups[key] = group
        return
    existing["count"] = int(existing.get("count", 0)) + 1
    current_key = _sort_event_position_key(
        ts=group.get("latest_ts"),
        seq=group.get("latest_seq"),
        path=group.get("latest_path") or "",
        line_no=int(group.get("latest_line") or 0),
    )
    existing_key = _sort_event_position_key(
        ts=existing.get("latest_ts"),
        seq=existing.get("latest_seq"),
        path=existing.get("latest_path") or "",
        line_no=int(existing.get("latest_line") or 0),
    )
    if current_key > existing_key:
        for field in (
            "latest_ts",
            "latest_seq",
            "latest_path",
            "latest_line",
            "latest_elapsed_ms",
            "latest_error",
            "latest_ids",
        ):
            existing[field] = group.get(field)


def _summarize_remote_call_failures(
    groups: dict[tuple[Any, ...], dict[str, Any]]
) -> dict[str, Any]:
    ordered = sorted(
        groups.values(),
        key=lambda item: (
            -int(item.get("count", 0)),
            -int(_non_negative_int(item.get("latest_ts")) or 0),
            str(item.get("bot") or ""),
            str(item.get("reason_code") or ""),
            str(item.get("surface") or ""),
        ),
    )
    compact_groups = [
        {
            key: value
            for key, value in group.items()
            if key not in {"latest_path", "latest_line", "latest_seq"}
            and value not in (None, {}, [])
        }
        for group in ordered[:REMOTE_CALL_FAILURE_GROUP_LIMIT]
    ]
    return {
        "total": sum(int(group.get("count", 0)) for group in groups.values()),
        "groups_truncated": len(ordered) > REMOTE_CALL_FAILURE_GROUP_LIMIT,
        "groups": compact_groups,
    }


def _problem_event_group(
    *,
    bot_key: str,
    row: dict[str, Any],
    live_event: dict[str, Any],
    path: Path,
    line_no: int,
    hard: bool,
) -> dict[str, Any]:
    ids = _event_ids(live_event)
    return {
        "bot": bot_key,
        "event_type": live_event.get("event_type") or row.get("kind"),
        "reason_code": live_event.get("reason_code"),
        "status": live_event.get("status"),
        "level": live_event.get("level"),
        "hard": bool(hard),
        "symbol": live_event.get("symbol") or row.get("symbol"),
        "pside": live_event.get("pside") or row.get("pside"),
        "component": live_event.get("component"),
        "count": 1,
        "latest_ts": row.get("ts"),
        "latest_seq": row.get("seq"),
        "latest_path": str(path),
        "latest_line": int(line_no),
        "latest_data": _compact_problem_event_data(live_event),
        "latest_ids": {
            key: ids.get(key)
            for key in (
                "cycle_id",
                "order_wave_id",
                "remote_call_id",
                "remote_call_group_id",
            )
            if ids.get(key) is not None
        },
    }


def _merge_problem_event_group(
    groups: dict[tuple[Any, ...], dict[str, Any]],
    group: dict[str, Any],
) -> None:
    key = (
        group.get("bot"),
        group.get("event_type"),
        group.get("reason_code"),
        group.get("status"),
        group.get("hard"),
        group.get("symbol"),
        group.get("pside"),
    )
    existing = groups.get(key)
    if existing is None:
        groups[key] = group
        return
    existing["count"] = int(existing.get("count", 0)) + 1
    current_key = _sort_event_position_key(
        ts=group.get("latest_ts"),
        seq=group.get("latest_seq"),
        path=group.get("latest_path") or "",
        line_no=int(group.get("latest_line") or 0),
    )
    existing_key = _sort_event_position_key(
        ts=existing.get("latest_ts"),
        seq=existing.get("latest_seq"),
        path=existing.get("latest_path") or "",
        line_no=int(existing.get("latest_line") or 0),
    )
    if current_key > existing_key:
        for field in (
            "level",
            "component",
            "latest_ts",
            "latest_seq",
            "latest_path",
            "latest_line",
            "latest_data",
            "latest_ids",
        ):
            existing[field] = group.get(field)


def _summarize_problem_event_groups(
    groups: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(
        groups.values(),
        key=lambda item: (
            not bool(item.get("hard")),
            -int(item.get("count", 0)),
            -int(_non_negative_int(item.get("latest_ts")) or 0),
            str(item.get("bot") or ""),
            str(item.get("event_type") or ""),
            str(item.get("reason_code") or ""),
        ),
    )
    compact_groups = [
        {
            key: value
            for key, value in group.items()
            if key not in {"latest_path", "latest_line", "latest_seq"}
            and value not in (None, {}, [])
        }
        for group in ordered[:PROBLEM_EVENT_GROUP_LIMIT]
    ]
    return {
        "total": sum(int(group.get("count", 0)) for group in groups.values()),
        "groups_truncated": len(ordered) > PROBLEM_EVENT_GROUP_LIMIT,
        "groups": compact_groups,
    }


def _risk_event_group(
    *,
    bot_key: str,
    row: dict[str, Any],
    live_event: dict[str, Any],
    path: Path,
    line_no: int,
) -> dict[str, Any]:
    data = live_event.get("data")
    payload = data if isinstance(data, dict) else {}
    ids = _event_ids(live_event)
    latest_data = {
        key: payload.get(key)
        for key in (
            "signal_mode",
            "tier",
            "previous_tier",
            "action",
            "mode",
            "previous_mode",
            "drawdown_score",
            "drawdown_raw",
            "drawdown_ema",
            "dist_to_red",
            "red_threshold",
            "cooldown_until_ms",
            "cooldown_remaining",
            "cooldown_remaining_seconds",
            "last_red_ts",
            "pending_red_since_ms",
        )
        if payload.get(key) is not None
    }
    return {
        "bot": bot_key,
        "event_type": live_event.get("event_type") or row.get("kind"),
        "reason_code": live_event.get("reason_code"),
        "status": live_event.get("status"),
        "level": live_event.get("level"),
        "symbol": live_event.get("symbol") or row.get("symbol"),
        "pside": live_event.get("pside") or row.get("pside"),
        "component": live_event.get("component"),
        "count": 1,
        "latest_ts": row.get("ts"),
        "latest_seq": row.get("seq"),
        "latest_path": str(path),
        "latest_line": int(line_no),
        "latest_data": latest_data,
        "latest_ids": {
            key: ids.get(key)
            for key in ("cycle_id", "snapshot_id", "plan_id", "action_id")
            if ids.get(key) is not None
        },
    }


def _merge_risk_event_group(
    groups: dict[tuple[Any, ...], dict[str, Any]],
    group: dict[str, Any],
) -> None:
    key = (
        group.get("bot"),
        group.get("event_type"),
        group.get("symbol"),
        group.get("pside"),
        group.get("reason_code"),
    )
    existing = groups.get(key)
    if existing is None:
        groups[key] = group
        return
    existing["count"] = int(existing.get("count", 0)) + 1
    current_key = _sort_event_position_key(
        ts=group.get("latest_ts"),
        seq=group.get("latest_seq"),
        path=group.get("latest_path") or "",
        line_no=int(group.get("latest_line") or 0),
    )
    existing_key = _sort_event_position_key(
        ts=existing.get("latest_ts"),
        seq=existing.get("latest_seq"),
        path=existing.get("latest_path") or "",
        line_no=int(existing.get("latest_line") or 0),
    )
    if current_key > existing_key:
        for field in (
            "status",
            "level",
            "component",
            "latest_ts",
            "latest_seq",
            "latest_path",
            "latest_line",
            "latest_data",
            "latest_ids",
        ):
            existing[field] = group.get(field)


def _summarize_risk_events(
    groups: dict[tuple[Any, ...], dict[str, Any]],
    event_type_counts: Counter[str],
) -> dict[str, Any]:
    ordered = sorted(
        groups.values(),
        key=lambda item: (
            -int(_non_negative_int(item.get("latest_ts")) or 0),
            str(item.get("bot") or ""),
            str(item.get("event_type") or ""),
            str(item.get("symbol") or ""),
            str(item.get("pside") or ""),
        ),
    )
    compact_groups = [
        {
            key: value
            for key, value in group.items()
            if key not in {"latest_path", "latest_line", "latest_seq"}
            and value not in (None, {}, [])
        }
        for group in ordered[:RISK_EVENT_GROUP_LIMIT]
    ]
    return {
        "total": sum(int(group.get("count", 0)) for group in groups.values()),
        "groups_truncated": len(ordered) > RISK_EVENT_GROUP_LIMIT,
        "event_types": dict(event_type_counts.most_common()),
        "groups": compact_groups,
    }


def _redact_log_text(value: str, *, max_len: int = 500) -> str:
    text = str(value)
    text = SENSITIVE_LOG_HEADER_PATTERN.sub(r"\1\2[redacted]", text)
    text = SENSITIVE_LOG_VALUE_PATTERN.sub(r"\1\2[redacted]", text)
    text = AUTH_SCHEME_PATTERN.sub(r"\1 [redacted]", text)
    if len(text) > max_len:
        text = f"{text[:max_len]}...<truncated>"
    return text


def _shell_tokens(value: str) -> list[str]:
    try:
        import shlex

        return shlex.split(str(value))
    except (ImportError, ValueError):
        return str(value).split()


def _canonical_live_command(value: str) -> str:
    tokens = _shell_tokens(value)
    for index, token in enumerate(tokens[:-1]):
        if Path(token).name == "passivbot" and tokens[index + 1] == "live":
            return " ".join([Path(tokens[index]).name, *tokens[index + 1 :]])
    return ""


def _parse_tmuxp_live_commands(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None or str(config_path).strip() == "":
        return {
            "path": None,
            "exists": False,
            "error": None,
            "expected": [],
        }
    path = Path(config_path).expanduser()
    expected: list[dict[str, Any]] = []
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "error": "config_not_found",
            "expected": expected,
        }
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return {
            "path": str(path),
            "exists": True,
            "error": f"config_read_failed:{exc.__class__.__name__}",
            "expected": expected,
        }

    current_window: str | None = None
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        window_line = stripped[2:].strip() if stripped.startswith("- ") else stripped
        if window_line.startswith("window_name:"):
            current_window = window_line.split(":", 1)[1].strip().strip("\"'")
            continue
        command = stripped
        if command.startswith("- "):
            command = command[2:].strip()
        command = command.strip().strip("\"'")
        command_key = _canonical_live_command(command)
        if not command_key:
            continue
        expected.append(
            {
                "name": current_window,
                "command": _redact_log_text(command, max_len=400),
                "command_key": _redact_log_text(command_key, max_len=400),
                "_match_key": command_key,
            }
        )
    return {
        "path": str(path),
        "exists": True,
        "error": None,
        "expected": expected,
    }


def _ps_process_rows() -> tuple[list[str], str | None]:
    commands = [
        ["ps", "-eo", "pid=,ppid=,etimes=,stat=,pcpu=,pmem=,command="],
        ["ps", "-axo", "pid=,ppid=,stat=,pcpu=,pmem=,command="],
    ]
    last_error: str | None = None
    for command in commands:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            last_error = f"{command[0]}_failed:{exc.__class__.__name__}"
            continue
        if result.returncode != 0:
            last_error = (result.stderr or result.stdout or "ps_failed").strip()
            continue
        return result.stdout.splitlines(), None
    return [], last_error or "ps_failed"


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _process_record_from_ps_line(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped:
        return None
    parts = stripped.split(None, 6)
    if len(parts) == 7 and _int_or_none(parts[2]) is not None:
        pid, ppid, etimes, stat_value, pcpu, pmem, command = parts
        age_s = _int_or_none(etimes)
    else:
        parts = stripped.split(None, 5)
        if len(parts) != 6:
            return None
        pid, ppid, stat_value, pcpu, pmem, command = parts
        age_s = None
    command_key = _canonical_live_command(command)
    if not command_key:
        return None
    return {
        "pid": _int_or_none(pid),
        "ppid": _int_or_none(ppid),
        "age_s": age_s,
        "stat": stat_value,
        "cpu_pct": _float_or_none(pcpu),
        "mem_pct": _float_or_none(pmem),
        "command": _redact_log_text(command, max_len=500),
        "command_key": _redact_log_text(command_key, max_len=500),
        "_match_key": command_key,
    }


def _running_live_processes(*, command_match: str) -> dict[str, Any]:
    rows, error = _ps_process_rows()
    processes: list[dict[str, Any]] = []
    for row in rows:
        record = _process_record_from_ps_line(row)
        if record is not None:
            match_key = str(record.get("_match_key") or "")
            command_key = str(record.get("command_key") or "")
            if (
                command_match
                and command_match not in row
                and command_match not in match_key
                and command_match not in command_key
            ):
                continue
            processes.append(record)
    processes.sort(
        key=lambda item: (str(item.get("command_key") or ""), int(item.get("pid") or 0))
    )
    return {
        "scan_error": error,
        "running": processes,
    }


def _build_process_report(
    *,
    include_processes: bool,
    supervisor_config: str | Path | None,
    process_command_match: str,
) -> dict[str, Any]:
    enabled = bool(include_processes or supervisor_config)
    if not enabled:
        return {
            "enabled": False,
            "ok": True,
            "hard_failures": 0,
            "expected_total": 0,
            "running_live_total": 0,
            "missing_expected": [],
            "unexpected_running": [],
        }

    config = _parse_tmuxp_live_commands(supervisor_config)
    running_scan = _running_live_processes(command_match=process_command_match)
    running = running_scan["running"]
    expected = config["expected"]
    missing: list[dict[str, Any]] = []
    matched_process_indexes: set[int] = set()
    for item in expected:
        match_key = item.get("_match_key")
        matches = [
            (index, process)
            for index, process in enumerate(running)
            if process.get("_match_key") == match_key
        ]
        if matches:
            matched_process_indexes.add(matches[0][0])
            continue
        missing.append(
            {
                "name": item.get("name"),
                "command": item.get("command"),
                "command_key": item.get("command_key"),
            }
        )
    unexpected = [
        {
            key: value
            for key, value in process.items()
            if key in PROCESS_REPORT_FIELDS
        }
        for index, process in enumerate(running)
        if expected and index not in matched_process_indexes
    ]
    hard_failures = len(missing)
    if config.get("error"):
        hard_failures += 1
    elif supervisor_config and not expected:
        hard_failures += 1
        config["error"] = "no_expected_live_commands"
    if running_scan.get("scan_error"):
        hard_failures += 1
    return {
        "enabled": True,
        "ok": hard_failures == 0,
        "hard_failures": hard_failures,
        "scan_error": running_scan.get("scan_error"),
        "config": {
            "path": config.get("path"),
            "exists": config.get("exists"),
            "error": config.get("error"),
        },
        "expected_total": len(expected),
        "running_live_total": len(running),
        "matched_expected": max(0, len(expected) - len(missing)),
        "missing_expected": missing,
        "unexpected_running": unexpected,
        "running": [
            {
                key: value
                for key, value in process.items()
                if key in PROCESS_REPORT_FIELDS
            }
            for process in running
        ],
    }


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


def _event_window_report(
    *,
    since_ms: int | None,
    until_ms: int | None,
    events_considered: int,
    events_skipped_before: int,
    events_skipped_after: int,
    invalid_window_ts: int,
) -> dict[str, Any]:
    return {
        "enabled": since_ms is not None or until_ms is not None,
        "since_ms": since_ms,
        "until_ms": until_ms,
        "events_considered": int(events_considered),
        "events_skipped_before": int(events_skipped_before),
        "events_skipped_after": int(events_skipped_after),
        "invalid_window_ts": int(invalid_window_ts),
    }


def _scan_events(
    root: str | Path,
    *,
    include_rotated: bool,
    max_problem_events: int,
    since_ms: int | None = None,
    until_ms: int | None = None,
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
    events_considered = 0
    events_skipped_before = 0
    events_skipped_after = 0
    invalid_window_ts = 0
    startup_timing_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    remote_call_failure_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    risk_event_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    risk_event_type_counts: Counter[str] = Counter()
    problem_event_groups: dict[tuple[Any, ...], dict[str, Any]] = {}

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
                    row_ts = _non_negative_int(row.get("ts"))
                    if since_ms is not None or until_ms is not None:
                        if row_ts is None:
                            invalid_window_ts += 1
                            continue
                        if since_ms is not None and row_ts < since_ms:
                            events_skipped_before += 1
                            continue
                        if until_ms is not None and row_ts > until_ms:
                            events_skipped_after += 1
                            continue
                    events_considered += 1
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
                    if event_type == EventTypes.REMOTE_CALL_FAILED:
                        _merge_remote_call_failure_group(
                            remote_call_failure_groups,
                            _remote_call_failure_group(
                                bot_key=bot_key,
                                row=row,
                                live_event=live_event,
                                path=path,
                                line_no=line_no,
                            ),
                        )
                    if event_type in RISK_EVENT_TYPES:
                        risk_event_type_counts[str(event_type)] += 1
                        _merge_risk_event_group(
                            risk_event_groups,
                            _risk_event_group(
                                bot_key=bot_key,
                                row=row,
                                live_event=live_event,
                                path=path,
                                line_no=line_no,
                            ),
                        )
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
                        _merge_problem_event_group(
                            problem_event_groups,
                            _problem_event_group(
                                bot_key=bot_key,
                                row=row,
                                live_event=live_event,
                                path=path,
                                line_no=line_no,
                                hard=hard,
                            ),
                        )
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
        "problem_event_groups": _summarize_problem_event_groups(problem_event_groups),
        "problem_event_count": sum(int(value["problem_events"]) for value in bots.values()),
        "hard_problem_event_count": sum(
            int(value["hard_problem_events"]) for value in bots.values()
        ),
        "startup_timings": _summarize_startup_timings(startup_timing_records),
        "remote_call_failures": _summarize_remote_call_failures(
            remote_call_failure_groups
        ),
        "risk_events": _summarize_risk_events(
            risk_event_groups,
            risk_event_type_counts,
        ),
        "event_window": _event_window_report(
            since_ms=since_ms,
            until_ms=until_ms,
            events_considered=events_considered,
            events_skipped_before=events_skipped_before,
            events_skipped_after=events_skipped_after,
            invalid_window_ts=invalid_window_ts,
        ),
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


def _parse_log_line_ts_ms(line: str) -> int | None:
    match = LOG_LINE_TS_PATTERN.match(str(line))
    if match is None:
        return None
    try:
        parsed = datetime.fromisoformat(match.group("ts").replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(parsed.timestamp() * 1000)


def _normalize_log_window_unparsed_policy(value: str | None) -> str:
    policy = str(value or DEFAULT_LOG_WINDOW_UNPARSED_POLICY).strip().lower()
    if policy not in LOG_WINDOW_UNPARSED_POLICIES:
        raise ValueError(
            "log_window_unparsed_policy must be one of "
            f"{sorted(LOG_WINDOW_UNPARSED_POLICIES)}"
        )
    return policy


def _log_window_report(
    *,
    since_ms: int | None,
    until_ms: int | None,
    lines_considered: int,
    lines_skipped_before: int,
    lines_skipped_after: int,
    unparsed_ts: int,
    unparsed_policy: str,
    lines_skipped_unparsed: int,
) -> dict[str, Any]:
    return {
        "enabled": since_ms is not None or until_ms is not None,
        "since_ms": since_ms,
        "until_ms": until_ms,
        "lines_considered": int(lines_considered),
        "lines_skipped_before": int(lines_skipped_before),
        "lines_skipped_after": int(lines_skipped_after),
        "unparsed_ts": int(unparsed_ts),
        "unparsed_policy": _normalize_log_window_unparsed_policy(unparsed_policy),
        "lines_skipped_unparsed": int(lines_skipped_unparsed),
    }


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
    since_ms: int | None,
    until_ms: int | None,
    log_window_unparsed_policy: str = DEFAULT_LOG_WINDOW_UNPARSED_POLICY,
) -> dict[str, Any]:
    window_enabled = since_ms is not None or until_ms is not None
    unparsed_policy = _normalize_log_window_unparsed_policy(log_window_unparsed_policy)
    window_report = _log_window_report(
        since_ms=since_ms,
        until_ms=until_ms,
        lines_considered=0,
        lines_skipped_before=0,
        lines_skipped_after=0,
        unparsed_ts=0,
        unparsed_policy=unparsed_policy,
        lines_skipped_unparsed=0,
    )
    if root is None:
        return {
            "root": None,
            "files_scanned": 0,
            "hard_matches": 0,
            "attention_matches": 0,
            "window": window_report,
            "matches": [],
        }
    files = _recent_log_files(root, max_files=max_files)
    matches: list[dict[str, Any]] = []
    hard_matches = 0
    attention_matches = 0
    lines_considered = 0
    lines_skipped_before = 0
    lines_skipped_after = 0
    unparsed_ts = 0
    lines_skipped_unparsed = 0
    for path in files:
        log_context_ts_ms: int | None = None
        for line_no, line in _tail_lines(path, max_lines=tail_lines):
            line_ts = _parse_log_line_ts_ms(line)
            if line_ts is not None:
                log_context_ts_ms = line_ts
            attention = bool(ATTENTION_LOG_PATTERN.search(line))
            if window_enabled:
                if line_ts is None:
                    unparsed_ts += 1
                    if log_context_ts_ms is not None:
                        if since_ms is not None and log_context_ts_ms < since_ms:
                            lines_skipped_before += 1
                            continue
                        if until_ms is not None and log_context_ts_ms > until_ms:
                            lines_skipped_after += 1
                            continue
                    if unparsed_policy == "drop" and (
                        log_context_ts_ms is None or not attention
                    ):
                        lines_skipped_unparsed += 1
                        continue
                else:
                    if since_ms is not None and line_ts < since_ms:
                        lines_skipped_before += 1
                        continue
                    if until_ms is not None and line_ts > until_ms:
                        lines_skipped_after += 1
                        continue
            lines_considered += 1
            if not attention:
                continue
            attention_matches += 1
            hard = bool(HARD_LOG_PATTERN.search(line))
            if hard:
                hard_matches += 1
            if len(matches) < max(0, int(max_matches)):
                match = {
                    "path": str(path),
                    "line": int(line_no),
                    "hard": hard,
                    "text": _redact_log_text(line, max_len=500),
                }
                if line_ts is not None:
                    match["ts"] = line_ts
                matches.append(match)
    return {
        "root": str(Path(root).expanduser()),
        "files_scanned": len(files),
        "hard_matches": hard_matches,
        "attention_matches": attention_matches,
        "window": _log_window_report(
            since_ms=since_ms,
            until_ms=until_ms,
            lines_considered=lines_considered,
            lines_skipped_before=lines_skipped_before,
            lines_skipped_after=lines_skipped_after,
            unparsed_ts=unparsed_ts,
            unparsed_policy=unparsed_policy,
            lines_skipped_unparsed=lines_skipped_unparsed,
        ),
        "matches": matches,
    }


def build_live_smoke_report(
    monitor_root: str | Path = "monitor",
    *,
    logs_root: str | Path | None = "logs",
    include_processes: bool = False,
    supervisor_config: str | Path | None = None,
    process_command_match: str = DEFAULT_PROCESS_MATCH,
    include_rotated: bool = False,
    since_ms: int | None = None,
    until_ms: int | None = None,
    max_problem_events: int = 50,
    max_log_files: int = 8,
    log_tail_lines: int = 300,
    max_log_matches: int = 50,
    log_window_unparsed_policy: str = DEFAULT_LOG_WINDOW_UNPARSED_POLICY,
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
            since_ms=since_ms,
            until_ms=until_ms,
        )
    else:
        event_scan = {
            "invalid_rows": 0,
            "bots": [],
            "problem_events": [],
            "problem_event_groups": {
                "total": 0,
                "groups_truncated": False,
                "groups": [],
            },
            "problem_event_count": 0,
            "hard_problem_event_count": 0,
            "startup_timings": [],
            "remote_call_failures": {
                "total": 0,
                "groups_truncated": False,
                "groups": [],
            },
            "risk_events": {
                "total": 0,
                "groups_truncated": False,
                "event_types": {},
                "groups": [],
            },
            "event_window": _event_window_report(
                since_ms=since_ms,
                until_ms=until_ms,
                events_considered=0,
                events_skipped_before=0,
                events_skipped_after=0,
                invalid_window_ts=0,
            ),
        }
    log_scan = _scan_logs(
        logs_root,
        max_files=max_log_files,
        tail_lines=log_tail_lines,
        max_matches=max_log_matches,
        since_ms=since_ms,
        until_ms=until_ms,
        log_window_unparsed_policy=log_window_unparsed_policy,
    )
    process_report = _build_process_report(
        include_processes=include_processes,
        supervisor_config=supervisor_config,
        process_command_match=process_command_match,
    )
    hard_failures = (
        int(event_report["error_count"])
        + int(event_scan["invalid_rows"])
        + int(event_scan["hard_problem_event_count"])
        + int(log_scan["hard_matches"])
        + int(process_report["hard_failures"])
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
        "remote_call_failures": event_scan["remote_call_failures"],
        "risk_events": event_scan["risk_events"],
        "event_window": event_scan["event_window"],
        "problem_events": event_scan["problem_events"],
        "problem_event_groups": event_scan["problem_event_groups"],
        "problem_event_count": event_scan["problem_event_count"],
        "hard_problem_event_count": event_scan["hard_problem_event_count"],
        "logs": log_scan,
        "processes": process_report,
    }
