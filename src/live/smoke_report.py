from __future__ import annotations

import gzip
import json
import math
import re
import stat
import subprocess
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from live.event_bus import LIVE_EVENT_MONITOR_PAYLOAD_KEY, EventTypes, utc_ms
from live.event_file_rows import event_file_rows
from live.event_query import (
    _limit_recent_event_files_per_bot,
    discover_event_files_with_metadata,
)
from live.problem_events import is_hard_problem_event, is_problem_event


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
RISK_LOG_PATTERN = re.compile(
    r"(?:\[(?:risk|hsl|wel|twel|unstuck)\]|\bHSL\[|\b(?:risk|hsl|wel|twel|unstuck)\.)",
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
REMOTE_CALL_HEALTH_GROUP_LIMIT = 20
REMOTE_CALL_TIMING_GROUP_LIMIT = 20
REMOTE_CALL_HEALTH_VALUE_LIMIT = 8
FILL_REFRESH_HEALTH_GROUP_LIMIT = 20
FILL_REFRESH_HEALTH_VALUE_LIMIT = 8
EXECUTION_HEALTH_GROUP_LIMIT = 20
EXECUTION_HEALTH_VALUE_LIMIT = 8
SMOKE_REPORT_SUMMARY_GROUP_LIMIT = 8
SMOKE_REPORT_BRIEF_REMOTE_CALL_SLOWEST_LIMIT = 3
_SMOKE_REPORT_SECTION_BASE_KEYS = (
    "ok",
    "attention",
    "hard_failures",
    "attention_count",
    "hard_failure_sources",
    "attention_sources",
    "repository",
    "monitor",
    "event_window",
    "problem_event_count",
    "hard_problem_event_count",
)
SMOKE_REPORT_BRIEF_PROBLEM_GROUP_LIMIT = 5
ACCOUNT_CRITICAL_REMOTE_CALL_KIND = "authoritative_state_fetch"
ACCOUNT_CRITICAL_REMOTE_CALL_SURFACES = frozenset(
    {
        "balance",
        "positions",
        "open_orders",
        # Hyperliquid state refresh splits account-critical position/balance
        # surfaces more finely than the standard staged refresh path.
        "positions_balance",
        "core_positions",
        "hip3_positions",
    }
)
RISK_EVENT_GROUP_LIMIT = 20
SHUTDOWN_EVENT_GROUP_LIMIT = 20
PROBLEM_EVENT_GROUP_LIMIT = 20
EMA_READINESS_GROUP_LIMIT = 20
EMA_READINESS_REASON_SYMBOL_SAMPLE_LIMIT = 8
STAGED_READINESS_GROUP_LIMIT = 20
EVENT_PIPELINE_HEALTH_GROUP_LIMIT = 20
HSL_REPLAY_HEALTH_GROUP_LIMIT = 20
HSL_REPLAY_STALE_ACTIVE_EVENT_AGE_MS = 5 * 60 * 1000
HSL_REPLAY_LONG_RUNNING_ACTIVE_MS = 10 * 60 * 1000
EXCHANGE_CONFIG_REFRESH_HEALTH_GROUP_LIMIT = 20
RISK_EVENT_TYPES = {
    EventTypes.RISK_MODE_CHANGED,
    EventTypes.HSL_TRANSITION,
    EventTypes.HSL_STATUS,
    EventTypes.HSL_RAW_RED_PENDING,
    EventTypes.HSL_RED_TRIGGERED,
    EventTypes.HSL_RED_FINALIZED_WITHOUT_ORDER,
    EventTypes.HSL_COOLDOWN_STARTED,
    EventTypes.HSL_COOLDOWN_ENDED,
    EventTypes.UNSTUCK_STATUS,
    EventTypes.UNSTUCK_SELECTION,
    EventTypes.RISK_ENTRY_COOLDOWN_DELTA_ANCHORED,
}
SHUTDOWN_EVENT_TYPES = {
    EventTypes.BOT_STOPPING,
    EventTypes.BOT_SHUTDOWN_STAGE,
    EventTypes.BOT_STOPPED,
}
HSL_REPLAY_EVENT_TYPES = {
    EventTypes.HSL_REPLAY_STARTED,
    EventTypes.HSL_REPLAY_PROGRESS,
    EventTypes.HSL_REPLAY_COMPLETED,
    EventTypes.HSL_REPLAY_FAILED,
}
REMOTE_CALL_TIMING_EVENT_TYPES = {
    EventTypes.REMOTE_CALL_SUCCEEDED,
    EventTypes.REMOTE_CALL_FAILED,
    EventTypes.REMOTE_CALL_THROTTLED,
}
EXECUTION_HEALTH_EVENT_TYPES = {
    EventTypes.ORDER_WAVE_STARTED,
    EventTypes.ORDER_WAVE_COMPLETED,
    EventTypes.EXECUTION_CREATE_SENT,
    EventTypes.EXECUTION_CREATE_SUCCEEDED,
    EventTypes.EXECUTION_CREATE_FAILED,
    EventTypes.EXECUTION_CREATE_REJECTED,
    EventTypes.EXECUTION_CREATE_DEFERRED,
    EventTypes.EXECUTION_CREATE_SKIPPED,
    EventTypes.EXECUTION_CANCEL_SENT,
    EventTypes.EXECUTION_CANCEL_SUCCEEDED,
    EventTypes.EXECUTION_CANCEL_FAILED,
    EventTypes.EXECUTION_CANCEL_AMBIGUOUS_TERMINAL,
    EventTypes.EXECUTION_AMBIGUOUS,
    EventTypes.EXECUTION_CONFIRMATION_REQUESTED,
    EventTypes.EXECUTION_CONFIRMATION_SATISFIED,
    EventTypes.EXECUTION_CONFIRMATION_TIMEOUT,
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
    EventTypes.STATE_REFRESH_PROGRESS: (
        "plan",
        "pending",
        "elapsed_ms",
        "completed_timings_ms",
        "threshold_s",
        "repeated",
    ),
}
PROBLEM_EVENT_DATA_MAX_DEPTH = 4
PROBLEM_EVENT_DATA_MAX_ITEMS = 8
PROBLEM_EVENT_DATA_MAX_TEXT = 240
PROCESS_REPORT_FIELDS = {
    "pid",
    "ppid",
    "age_s",
    "stat",
    "cpu_pct",
    "mem_pct",
    "rss_kb",
    "account",
    "config_path",
    "config_key",
    "command",
    "command_key",
}
EXPECTED_PROCESS_FIELDS = {
    "name",
    "account",
    "config_path",
    "config_key",
    "command",
    "command_key",
}
LIVE_COMMAND_VALUE_FLAGS = {
    "-u": "account",
    "--user": "account",
    "-bo": "balance_override",
    "--balance-override": "balance_override",
    "--balance_override": "balance_override",
    "--live.balance_override": "balance_override",
    "--live.balance-override": "balance_override",
}
LIVE_COMMAND_EQ_PREFIXES = {
    "--user=": "account",
    "--balance-override=": "balance_override",
    "--balance_override=": "balance_override",
    "--live.balance_override=": "balance_override",
    "--live.balance-override=": "balance_override",
}
ACCOUNT_LEVEL_HSL_SIGNAL_MODES = {"unified", "pside", "per_side"}
HSL_BALANCE_OVERRIDE_UNSAFE_CODE = "hsl_balance_override_account_level_replay_unsafe"


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    try:
        return candidate.resolve()
    except OSError:
        return candidate.absolute()


def _user_safe_display_path(path: str | Path) -> str:
    resolved = _resolve_path(path)

    def _tilde_path(tail: tuple[str, ...]) -> str:
        return "~" if not tail else "~/" + "/".join(tail)

    try:
        home = _resolve_path(Path.home())
        relative = resolved.relative_to(home)
        return _tilde_path(relative.parts)
    except (OSError, RuntimeError, ValueError):
        pass

    display_candidate = Path(path).expanduser()
    try:
        display_path = display_candidate.absolute()
    except OSError:
        display_path = resolved
    parts = display_path.parts
    if len(parts) >= 2 and parts[:2] == ("/", "root"):
        return _tilde_path(tuple(parts[2:]))
    if len(parts) >= 3 and parts[0] == "/" and parts[1] in {"home", "Users"}:
        return _tilde_path(tuple(parts[3:]))
    return str(resolved)


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
    existing["count"] = _count_value(existing.get("count")) + 1
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


def _remote_call_status(live_event: dict[str, Any], row: dict[str, Any]) -> str | None:
    event_type = live_event.get("event_type") or row.get("kind")
    if event_type == EventTypes.REMOTE_CALL_SUCCEEDED:
        return "succeeded"
    if event_type == EventTypes.REMOTE_CALL_FAILED:
        return "failed"
    if event_type == EventTypes.REMOTE_CALL_THROTTLED:
        return "throttled"
    status = live_event.get("status")
    if status not in (None, ""):
        return str(status).lower()
    return None


def _remote_call_raw_status(live_event: dict[str, Any]) -> str | None:
    status = live_event.get("status")
    if status not in (None, ""):
        return str(status).lower()
    return None


def _remote_call_health_group(
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
    elapsed_ms = _non_negative_int(payload.get("elapsed_ms"))
    status = _remote_call_status(live_event, row)
    raw_status = _remote_call_raw_status(live_event)
    reason_code = live_event.get("reason_code")
    error_type = payload.get("error_type")
    kind = payload.get("kind")
    surface = payload.get("surface")
    symbol = live_event.get("symbol")
    failed = status == "failed"
    return {
        "bot": bot_key,
        "component": live_event.get("component"),
        "kind": kind,
        "surface": surface,
        "count": 1,
        "elapsed_values": [elapsed_ms] if elapsed_ms is not None else [],
        "statuses": Counter([status]) if status else Counter(),
        "raw_statuses": Counter([raw_status])
        if raw_status and raw_status != status
        else Counter(),
        "reason_codes": Counter([str(reason_code)]) if reason_code not in (None, "") else Counter(),
        "error_types": Counter([str(error_type)]) if error_type not in (None, "") else Counter(),
        "symbols": Counter([str(symbol)]) if symbol not in (None, "") else Counter(),
        "failed_reason_codes": Counter([str(reason_code)])
        if failed and reason_code not in (None, "")
        else Counter(),
        "failed_error_types": Counter([str(error_type)])
        if failed and error_type not in (None, "")
        else Counter(),
        "failed_kinds": Counter([str(kind)])
        if failed and kind not in (None, "")
        else Counter(),
        "failed_surfaces": Counter([str(surface)])
        if failed and surface not in (None, "")
        else Counter(),
        "latest_ts": row.get("ts"),
        "latest_seq": row.get("seq"),
        "latest_path": str(path),
        "latest_line": int(line_no),
        "latest_event_type": live_event.get("event_type") or row.get("kind"),
        "latest_status": status,
        "latest_raw_status": raw_status if raw_status and raw_status != status else None,
        "latest_elapsed_ms": elapsed_ms,
        "latest_symbol": symbol,
        "latest_error_type": error_type,
        "latest_ids": {
            key: ids.get(key)
            for key in ("cycle_id", "remote_call_id", "remote_call_group_id")
            if ids.get(key) is not None
        },
    }


def _merge_remote_call_health_group(
    groups: dict[tuple[Any, ...], dict[str, Any]],
    group: dict[str, Any],
) -> None:
    key = (
        group.get("bot"),
        group.get("component"),
        group.get("kind"),
        group.get("surface"),
    )
    existing = groups.get(key)
    if existing is None:
        groups[key] = group
        return
    existing["count"] = _count_value(existing.get("count")) + 1
    existing.setdefault("elapsed_values", []).extend(group.get("elapsed_values") or [])
    for field in (
        "statuses",
        "raw_statuses",
        "reason_codes",
        "error_types",
        "symbols",
        "failed_reason_codes",
        "failed_error_types",
        "failed_kinds",
        "failed_surfaces",
    ):
        counter = existing.setdefault(field, Counter())
        counter.update(group.get(field) or Counter())
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
            "latest_event_type",
            "latest_status",
            "latest_raw_status",
            "latest_elapsed_ms",
            "latest_symbol",
            "latest_error_type",
            "latest_ids",
        ):
            existing[field] = group.get(field)


def _top_counter_values(counter: Counter[str], *, limit: int) -> dict[str, int]:
    if not counter:
        return {}
    ordered = sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return {str(key): int(value) for key, value in ordered[:limit]}


def _symbol_sample(counter: Counter[str], *, limit: int) -> dict[str, Any]:
    if not counter:
        return {}
    ordered = sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0])))
    sample = [str(key) for key, _value in ordered[:limit]]
    return {
        "count": len(counter),
        "sample": sample,
        "truncated": max(0, len(counter) - limit),
    }


def _remote_call_health_sort_key(
    group: dict[str, Any],
) -> tuple[int, int, int, int, int, str, str]:
    statuses = group.get("statuses") if isinstance(group.get("statuses"), Counter) else Counter()
    elapsed = _ms_summary(
        [
            int(value)
            for value in (group.get("elapsed_values") or [])
            if _non_negative_int(value) is not None
        ]
    )
    return (
        -int(statuses.get("failed", 0)),
        -int(statuses.get("throttled", 0)),
        -int(elapsed.get("p95_ms") or 0),
        -int(elapsed.get("max_ms") or 0),
        -int(group.get("count", 0)),
        str(group.get("bot") or ""),
        str(group.get("kind") or group.get("surface") or ""),
    )


def _summarize_remote_call_health(
    groups: dict[tuple[Any, ...], dict[str, Any]]
) -> dict[str, Any]:
    ordered = sorted(groups.values(), key=_remote_call_health_sort_key)
    status_totals: Counter[str] = Counter()
    failed_reason_codes: Counter[str] = Counter()
    failed_error_types: Counter[str] = Counter()
    failed_kinds: Counter[str] = Counter()
    failed_surfaces: Counter[str] = Counter()
    for group in groups.values():
        statuses = group.get("statuses") if isinstance(group.get("statuses"), Counter) else Counter()
        status_totals.update(statuses)
        for source, target in (
            ("failed_reason_codes", failed_reason_codes),
            ("failed_error_types", failed_error_types),
            ("failed_kinds", failed_kinds),
            ("failed_surfaces", failed_surfaces),
        ):
            values = group.get(source)
            if isinstance(values, Counter):
                target.update(values)
    total = sum(int(group.get("count", 0)) for group in groups.values())
    total_succeeded_count = int(status_totals.get("succeeded", 0))
    total_failed_count = int(status_totals.get("failed", 0))
    total_throttled_count = int(status_totals.get("throttled", 0))
    compact_groups = []
    for group in ordered[:REMOTE_CALL_HEALTH_GROUP_LIMIT]:
        statuses = group.get("statuses") if isinstance(group.get("statuses"), Counter) else Counter()
        reason_codes = (
            group.get("reason_codes")
            if isinstance(group.get("reason_codes"), Counter)
            else Counter()
        )
        raw_statuses = (
            group.get("raw_statuses")
            if isinstance(group.get("raw_statuses"), Counter)
            else Counter()
        )
        error_types = (
            group.get("error_types") if isinstance(group.get("error_types"), Counter) else Counter()
        )
        symbols = group.get("symbols") if isinstance(group.get("symbols"), Counter) else Counter()
        count = int(group.get("count", 0))
        group_failed_count = int(statuses.get("failed", 0))
        group_throttled_count = int(statuses.get("throttled", 0))
        compact = {
            "bot": group.get("bot"),
            "component": group.get("component"),
            "kind": group.get("kind"),
            "surface": group.get("surface"),
            "count": count,
            "succeeded": int(statuses.get("succeeded", 0)),
            "failed": group_failed_count,
            "throttled": group_throttled_count,
            "failure_pct": _usage_pct(group_failed_count, count),
            "throttled_pct": _usage_pct(group_throttled_count, count),
            "statuses": _top_counter_values(
                statuses,
                limit=REMOTE_CALL_HEALTH_VALUE_LIMIT,
            ),
            "raw_statuses": _top_counter_values(
                raw_statuses,
                limit=REMOTE_CALL_HEALTH_VALUE_LIMIT,
            ),
            "reason_codes": _top_counter_values(
                reason_codes,
                limit=REMOTE_CALL_HEALTH_VALUE_LIMIT,
            ),
            "error_types": _top_counter_values(
                error_types,
                limit=REMOTE_CALL_HEALTH_VALUE_LIMIT,
            ),
            "symbols": _symbol_sample(
                symbols,
                limit=REMOTE_CALL_HEALTH_VALUE_LIMIT,
            ),
            "elapsed_ms": _ms_summary(
                [
                    int(value)
                    for value in (group.get("elapsed_values") or [])
                    if _non_negative_int(value) is not None
                ]
            ),
            "latest_ts": group.get("latest_ts"),
            "latest_event_type": group.get("latest_event_type"),
            "latest_status": group.get("latest_status"),
            "latest_raw_status": group.get("latest_raw_status"),
            "latest_elapsed_ms": group.get("latest_elapsed_ms"),
            "latest_symbol": group.get("latest_symbol"),
            "latest_error_type": group.get("latest_error_type"),
            "latest_ids": group.get("latest_ids"),
        }
        compact_groups.append(
            {
                key: value
                for key, value in compact.items()
                if key not in {"latest_path", "latest_line", "latest_seq"}
                and value not in (None, {}, [])
            }
        )
    out = {
        "total": total,
        "succeeded": total_succeeded_count,
        "failed": total_failed_count,
        "throttled": total_throttled_count,
        "failure_pct": _usage_pct(total_failed_count, total),
        "throttled_pct": _usage_pct(total_throttled_count, total),
        "groups_truncated": len(ordered) > REMOTE_CALL_HEALTH_GROUP_LIMIT,
        "groups": compact_groups,
    }
    for key, values in (
        ("failed_reason_codes", failed_reason_codes),
        ("failed_error_types", failed_error_types),
        ("failed_kinds", failed_kinds),
        ("failed_surfaces", failed_surfaces),
    ):
        compact_values = _top_counter_values(
            values,
            limit=REMOTE_CALL_HEALTH_VALUE_LIMIT,
        )
        if compact_values:
            out[key] = compact_values
    return out


def _execution_health_outcome(event_type: str, status: str | None) -> str:
    if event_type == EventTypes.ORDER_WAVE_STARTED:
        return "wave_started"
    if event_type == EventTypes.ORDER_WAVE_COMPLETED:
        return "wave_completed"
    if event_type == EventTypes.EXECUTION_CREATE_SENT:
        return "create_sent"
    if event_type == EventTypes.EXECUTION_CREATE_SUCCEEDED:
        return "create_succeeded"
    if event_type == EventTypes.EXECUTION_CREATE_FAILED:
        return "create_failed"
    if event_type == EventTypes.EXECUTION_CREATE_REJECTED:
        return "create_rejected"
    if event_type == EventTypes.EXECUTION_CREATE_DEFERRED:
        return "create_deferred"
    if event_type == EventTypes.EXECUTION_CREATE_SKIPPED:
        return "create_skipped"
    if event_type == EventTypes.EXECUTION_CANCEL_SENT:
        return "cancel_sent"
    if event_type == EventTypes.EXECUTION_CANCEL_SUCCEEDED:
        return "cancel_succeeded"
    if event_type == EventTypes.EXECUTION_CANCEL_FAILED:
        return "cancel_failed"
    if event_type == EventTypes.EXECUTION_CANCEL_AMBIGUOUS_TERMINAL:
        return "cancel_ambiguous_terminal"
    if event_type == EventTypes.EXECUTION_AMBIGUOUS:
        return "ambiguous"
    if event_type == EventTypes.EXECUTION_CONFIRMATION_REQUESTED:
        return "confirmation_requested"
    if event_type == EventTypes.EXECUTION_CONFIRMATION_SATISFIED:
        return "confirmation_satisfied"
    if event_type == EventTypes.EXECUTION_CONFIRMATION_TIMEOUT:
        return "confirmation_timeout"
    if status not in (None, ""):
        return str(status).lower()
    return "unknown"


def _execution_health_elapsed_values(payload: dict[str, Any]) -> list[int]:
    values: list[int] = []
    for key in ("elapsed_ms", "confirm_ms", "cancel_ms", "create_ms"):
        value = _non_negative_int(payload.get(key))
        if value is not None:
            values.append(int(value))
    return values


def _safe_string_sample(values: Any, *, limit: int) -> dict[str, Any]:
    if not isinstance(values, (list, tuple, set)):
        return {}
    safe_values = sorted(
        {
            _redact_log_text(str(value), max_len=80)
            for value in values
            if value not in (None, "")
        }
    )
    return {
        "count": len(safe_values),
        "sample": safe_values[:limit],
        "truncated": max(0, len(safe_values) - limit),
    }


def _execution_health_latest_data(payload: dict[str, Any]) -> dict[str, Any]:
    safe_keys = (
        "index",
        "order_type",
        "pb_order_type",
        "context",
        "reason",
        "reduce_only",
        "elapsed_ms",
        "confirm_ms",
        "timeout_ms",
        "planned_cancel",
        "planned_create",
        "cancel_posted",
        "create_posted",
        "skipped_cancel",
        "deferred_create",
        "skipped_create",
        "order_count",
        "symbols_count",
        "symbols_truncated",
        "pending_surfaces",
        "fresh_surfaces",
        "changed_surfaces",
        "error_type",
        "result_status",
    )
    latest: dict[str, Any] = {}
    for key in safe_keys:
        value = payload.get(key)
        if value in (None, "", {}, []):
            continue
        latest[key] = _compact_problem_event_data_value(value)
    symbol_sample = _safe_string_sample(
        payload.get("symbols"),
        limit=EXECUTION_HEALTH_VALUE_LIMIT,
    )
    if symbol_sample:
        latest["symbols"] = symbol_sample
    return latest


def _execution_health_group(
    *,
    bot_key: str,
    row: dict[str, Any],
    live_event: dict[str, Any],
    path: Path,
    line_no: int,
) -> dict[str, Any]:
    event_type = str(live_event.get("event_type") or row.get("kind") or "")
    data = live_event.get("data")
    payload = data if isinstance(data, dict) else {}
    status = (
        str(live_event.get("status")).lower()
        if live_event.get("status") not in (None, "")
        else None
    )
    reason_code = live_event.get("reason_code")
    error_type = payload.get("error_type")
    symbol = live_event.get("symbol")
    pside = live_event.get("pside")
    side = live_event.get("side")
    outcome = _execution_health_outcome(event_type, status)
    ids = _event_ids(live_event)
    return {
        "bot": bot_key,
        "event_type": event_type,
        "component": live_event.get("component"),
        "status": status,
        "outcome": outcome,
        "reason_code": reason_code,
        "symbol": symbol,
        "pside": pside,
        "side": side,
        "count": 1,
        "event_types": Counter([event_type]) if event_type else Counter(),
        "statuses": Counter([status]) if status else Counter(),
        "outcomes": Counter([outcome]) if outcome else Counter(),
        "reason_codes": Counter([str(reason_code)]) if reason_code not in (None, "") else Counter(),
        "error_types": Counter([str(error_type)]) if error_type not in (None, "") else Counter(),
        "symbols": Counter([str(symbol)]) if symbol not in (None, "") else Counter(),
        "psides": Counter([str(pside)]) if pside not in (None, "") else Counter(),
        "sides": Counter([str(side)]) if side not in (None, "") else Counter(),
        "elapsed_values": _execution_health_elapsed_values(payload),
        "latest_ts": row.get("ts"),
        "latest_seq": row.get("seq"),
        "latest_path": str(path),
        "latest_line": int(line_no),
        "latest_level": live_event.get("level"),
        "latest_error_type": error_type,
        "latest_data": _execution_health_latest_data(payload),
        "latest_ids": {
            key: ids.get(key)
            for key in ("cycle_id", "order_wave_id", "action_id")
            if ids.get(key) is not None
        },
    }


def _merge_execution_health_group(
    groups: dict[tuple[Any, ...], dict[str, Any]],
    group: dict[str, Any],
) -> None:
    key = (
        group.get("bot"),
        group.get("event_type"),
        group.get("component"),
        group.get("status"),
        group.get("reason_code"),
        group.get("symbol"),
        group.get("pside"),
        group.get("side"),
    )
    existing = groups.get(key)
    if existing is None:
        groups[key] = group
        return
    existing["count"] = _count_value(existing.get("count")) + 1
    existing.setdefault("elapsed_values", []).extend(group.get("elapsed_values") or [])
    for field in (
        "event_types",
        "statuses",
        "outcomes",
        "reason_codes",
        "error_types",
        "symbols",
        "psides",
        "sides",
    ):
        counter = existing.setdefault(field, Counter())
        counter.update(group.get(field) or Counter())
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
            "latest_level",
            "latest_error_type",
            "latest_data",
            "latest_ids",
        ):
            existing[field] = group.get(field)


def _execution_health_sort_key(
    group: dict[str, Any],
) -> tuple[int, int, int, int, int, int, str, str]:
    outcomes = group.get("outcomes") if isinstance(group.get("outcomes"), Counter) else Counter()
    elapsed = _ms_summary(
        [
            int(value)
            for value in (group.get("elapsed_values") or [])
            if _non_negative_int(value) is not None
        ]
    )
    return (
        -_count_value(outcomes.get("ambiguous")),
        -_count_value(outcomes.get("cancel_ambiguous_terminal")),
        -_count_value(outcomes.get("confirmation_timeout")),
        -(
            _count_value(outcomes.get("create_failed"))
            + _count_value(outcomes.get("cancel_failed"))
        ),
        -_count_value(outcomes.get("create_rejected")),
        -int(elapsed.get("p95_ms") or 0),
        str(group.get("bot") or ""),
        str(group.get("event_type") or ""),
    )


def _summarize_execution_health(
    groups: dict[tuple[Any, ...], dict[str, Any]]
) -> dict[str, Any]:
    ordered = sorted(groups.values(), key=_execution_health_sort_key)
    event_types: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    bots: Counter[str] = Counter()
    for group in groups.values():
        event_types.update(group.get("event_types") or Counter())
        statuses.update(group.get("statuses") or Counter())
        outcomes.update(group.get("outcomes") or Counter())
        if group.get("bot") not in (None, ""):
            bots[str(group.get("bot"))] += int(group.get("count") or 0)
    total = sum(_count_value(group.get("count")) for group in groups.values())
    failed = _count_value(outcomes.get("create_failed")) + _count_value(
        outcomes.get("cancel_failed")
    )
    rejected = _count_value(outcomes.get("create_rejected"))
    ambiguous = int(
        _count_value(outcomes.get("ambiguous"))
        + _count_value(outcomes.get("cancel_ambiguous_terminal"))
    )
    confirmation_timeout = _count_value(outcomes.get("confirmation_timeout"))
    compact_groups = []
    for group in ordered[:EXECUTION_HEALTH_GROUP_LIMIT]:
        statuses_counter = (
            group.get("statuses") if isinstance(group.get("statuses"), Counter) else Counter()
        )
        outcomes_counter = (
            group.get("outcomes") if isinstance(group.get("outcomes"), Counter) else Counter()
        )
        reason_codes = (
            group.get("reason_codes")
            if isinstance(group.get("reason_codes"), Counter)
            else Counter()
        )
        error_types = (
            group.get("error_types") if isinstance(group.get("error_types"), Counter) else Counter()
        )
        symbols = group.get("symbols") if isinstance(group.get("symbols"), Counter) else Counter()
        psides = group.get("psides") if isinstance(group.get("psides"), Counter) else Counter()
        sides = group.get("sides") if isinstance(group.get("sides"), Counter) else Counter()
        compact = {
            "bot": group.get("bot"),
            "event_type": group.get("event_type"),
            "component": group.get("component"),
            "status": group.get("status"),
            "outcome": group.get("outcome"),
            "reason_code": group.get("reason_code"),
            "symbol": group.get("symbol"),
            "pside": group.get("pside"),
            "side": group.get("side"),
            "count": _count_value(group.get("count")),
            "statuses": _top_counter_values(
                statuses_counter,
                limit=EXECUTION_HEALTH_VALUE_LIMIT,
            ),
            "outcomes": _top_counter_values(
                outcomes_counter,
                limit=EXECUTION_HEALTH_VALUE_LIMIT,
            ),
            "reason_codes": _top_counter_values(
                reason_codes,
                limit=EXECUTION_HEALTH_VALUE_LIMIT,
            ),
            "error_types": _top_counter_values(
                error_types,
                limit=EXECUTION_HEALTH_VALUE_LIMIT,
            ),
            "symbols": _symbol_sample(
                symbols,
                limit=EXECUTION_HEALTH_VALUE_LIMIT,
            ),
            "psides": _symbol_sample(
                psides,
                limit=EXECUTION_HEALTH_VALUE_LIMIT,
            ),
            "sides": _symbol_sample(
                sides,
                limit=EXECUTION_HEALTH_VALUE_LIMIT,
            ),
            "elapsed_ms": _ms_summary(
                [
                    int(value)
                    for value in (group.get("elapsed_values") or [])
                    if _non_negative_int(value) is not None
                ]
            ),
            "latest_ts": group.get("latest_ts"),
            "latest_level": group.get("latest_level"),
            "latest_error_type": group.get("latest_error_type"),
            "latest_data": group.get("latest_data"),
            "latest_ids": group.get("latest_ids"),
        }
        compact_groups.append(
            {
                key: value
                for key, value in compact.items()
                if key not in {"latest_path", "latest_line", "latest_seq"}
                and value not in (None, {}, [])
            }
        )
    return {
        "total": total,
        "bots": len(bots),
        "failed": failed,
        "rejected": rejected,
        "ambiguous": ambiguous,
        "confirmation_timeout": confirmation_timeout,
        "event_types": _top_counter_values(
            event_types,
            limit=EXECUTION_HEALTH_VALUE_LIMIT,
        ),
        "statuses": _top_counter_values(
            statuses,
            limit=EXECUTION_HEALTH_VALUE_LIMIT,
        ),
        "outcomes": _top_counter_values(
            outcomes,
            limit=EXECUTION_HEALTH_VALUE_LIMIT,
        ),
        "groups_truncated": len(ordered) > EXECUTION_HEALTH_GROUP_LIMIT,
        "groups": compact_groups,
    }


def _fill_refresh_health_group(
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
    elapsed_ms = _non_negative_int(payload.get("elapsed_ms"))
    status = live_event.get("status")
    status_value = str(status).lower() if status not in (None, "") else None
    reason_code = live_event.get("reason_code")
    error_type = payload.get("error_type")
    return {
        "bot": bot_key,
        "source": payload.get("source"),
        "refresh_mode": payload.get("refresh_mode"),
        "history_scope": payload.get("history_scope"),
        "count": 1,
        "elapsed_values": [elapsed_ms] if elapsed_ms is not None else [],
        "statuses": Counter([status_value]) if status_value else Counter(),
        "reason_codes": Counter([str(reason_code)])
        if reason_code not in (None, "")
        else Counter(),
        "error_types": Counter([str(error_type)])
        if error_type not in (None, "")
        else Counter(),
        "latest_ts": row.get("ts"),
        "latest_seq": row.get("seq"),
        "latest_path": str(path),
        "latest_line": int(line_no),
        "latest_status": status_value,
        "latest_reason_code": reason_code,
        "latest_elapsed_ms": elapsed_ms,
        "latest_history_scope": payload.get("history_scope"),
        "latest_event_count_after": _non_negative_int(payload.get("event_count_after")),
        "latest_new_count": _non_negative_int(payload.get("new_count")),
        "latest_enriched_count": _non_negative_int(payload.get("enriched_count")),
        "latest_pending_pnl_count": _non_negative_int(payload.get("pending_pnl_count")),
        "latest_coverage_ready_after": payload.get("coverage_ready_after"),
        "latest_coverage_reason_after": payload.get("coverage_reason_after"),
        "latest_retry_count": _non_negative_int(payload.get("retry_count")),
        "latest_next_retry_in_ms": _non_negative_int(payload.get("next_retry_in_ms")),
        "latest_error_type": error_type,
        "latest_degraded_events_after": _non_negative_int(
            payload.get("degraded_events_after")
        ),
        "latest_ids": {
            key: ids.get(key)
            for key in ("cycle_id",)
            if ids.get(key) is not None
        },
    }


def _merge_fill_refresh_health_group(
    groups: dict[tuple[Any, ...], dict[str, Any]],
    group: dict[str, Any],
) -> None:
    key = (
        group.get("bot"),
        group.get("source"),
        group.get("refresh_mode"),
    )
    existing = groups.get(key)
    if existing is None:
        groups[key] = group
        return
    existing["count"] = int(existing.get("count") or 0) + 1
    existing.setdefault("elapsed_values", []).extend(group.get("elapsed_values") or [])
    for field in ("statuses", "reason_codes", "error_types"):
        counter = existing.setdefault(field, Counter())
        counter.update(group.get(field) or Counter())
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
            "history_scope",
            "latest_ts",
            "latest_seq",
            "latest_path",
            "latest_line",
            "latest_status",
            "latest_reason_code",
            "latest_elapsed_ms",
            "latest_history_scope",
            "latest_event_count_after",
            "latest_new_count",
            "latest_enriched_count",
            "latest_pending_pnl_count",
            "latest_coverage_ready_after",
            "latest_coverage_reason_after",
            "latest_retry_count",
            "latest_next_retry_in_ms",
            "latest_error_type",
            "latest_degraded_events_after",
            "latest_ids",
        ):
            existing[field] = group.get(field)


def _fill_refresh_health_sort_key(
    group: dict[str, Any],
) -> tuple[int, int, int, int, int, str, str]:
    statuses = (
        group.get("statuses") if isinstance(group.get("statuses"), Counter) else Counter()
    )
    elapsed = _ms_summary(
        [
            int(value)
            for value in (group.get("elapsed_values") or [])
            if _non_negative_int(value) is not None
        ]
    )
    latest_failed = 1 if group.get("latest_status") == "failed" else 0
    return (
        -latest_failed,
        -int(statuses.get("failed") or 0),
        -int(elapsed.get("p95_ms") or 0),
        -int(elapsed.get("max_ms") or 0),
        -int(group.get("count") or 0),
        str(group.get("bot") or ""),
        str(group.get("source") or ""),
    )


def _summarize_fill_refresh_health(
    groups: dict[tuple[Any, ...], dict[str, Any]]
) -> dict[str, Any]:
    ordered = sorted(groups.values(), key=_fill_refresh_health_sort_key)
    status_totals: Counter[str] = Counter()
    bots: set[str] = set()
    failed_bots: set[str] = set()
    latest_failed_bots: set[str] = set()
    recovered_groups = 0
    for group in groups.values():
        bot = group.get("bot")
        if bot not in (None, ""):
            bots.add(str(bot))
        statuses = (
            group.get("statuses")
            if isinstance(group.get("statuses"), Counter)
            else Counter()
        )
        status_totals.update(statuses)
        if int(statuses.get("failed") or 0):
            if bot not in (None, ""):
                failed_bots.add(str(bot))
            if group.get("latest_status") != "failed":
                recovered_groups += 1
        if group.get("latest_status") == "failed" and bot not in (None, ""):
            latest_failed_bots.add(str(bot))
    total = sum(int(group.get("count") or 0) for group in groups.values())
    total_failed_count = int(status_totals.get("failed") or 0)
    compact_groups = []
    for group in ordered[:FILL_REFRESH_HEALTH_GROUP_LIMIT]:
        statuses = (
            group.get("statuses")
            if isinstance(group.get("statuses"), Counter)
            else Counter()
        )
        reason_codes = (
            group.get("reason_codes")
            if isinstance(group.get("reason_codes"), Counter)
            else Counter()
        )
        error_types = (
            group.get("error_types")
            if isinstance(group.get("error_types"), Counter)
            else Counter()
        )
        count = int(group.get("count") or 0)
        failed_count = int(statuses.get("failed") or 0)
        compact = {
            "bot": group.get("bot"),
            "source": group.get("source"),
            "refresh_mode": group.get("refresh_mode"),
            "history_scope": group.get("history_scope"),
            "count": count,
            "failed": failed_count,
            "failure_pct": _usage_pct(failed_count, count),
            "statuses": _top_counter_values(
                statuses,
                limit=FILL_REFRESH_HEALTH_VALUE_LIMIT,
            ),
            "reason_codes": _top_counter_values(
                reason_codes,
                limit=FILL_REFRESH_HEALTH_VALUE_LIMIT,
            ),
            "error_types": _top_counter_values(
                error_types,
                limit=FILL_REFRESH_HEALTH_VALUE_LIMIT,
            ),
            "elapsed_ms": _ms_summary(
                [
                    int(value)
                    for value in (group.get("elapsed_values") or [])
                    if _non_negative_int(value) is not None
                ]
            ),
            "latest_ts": group.get("latest_ts"),
            "latest_status": group.get("latest_status"),
            "latest_reason_code": group.get("latest_reason_code"),
            "latest_elapsed_ms": group.get("latest_elapsed_ms"),
            "latest_history_scope": group.get("latest_history_scope"),
            "latest_event_count_after": group.get("latest_event_count_after"),
            "latest_new_count": group.get("latest_new_count"),
            "latest_enriched_count": group.get("latest_enriched_count"),
            "latest_pending_pnl_count": group.get("latest_pending_pnl_count"),
            "latest_coverage_ready_after": group.get("latest_coverage_ready_after"),
            "latest_coverage_reason_after": group.get("latest_coverage_reason_after"),
            "latest_retry_count": group.get("latest_retry_count"),
            "latest_next_retry_in_ms": group.get("latest_next_retry_in_ms"),
            "latest_error_type": group.get("latest_error_type"),
            "latest_degraded_events_after": group.get(
                "latest_degraded_events_after"
            ),
            "recovered": bool(failed_count and group.get("latest_status") != "failed"),
            "latest_ids": group.get("latest_ids"),
        }
        compact_groups.append(
            {
                key: value
                for key, value in compact.items()
                if key not in {"latest_path", "latest_line", "latest_seq"}
                and value not in (None, {}, [])
            }
        )
    return {
        "total": total,
        "bots": len(bots),
        "failed": total_failed_count,
        "failure_pct": _usage_pct(total_failed_count, total),
        "failed_bots": len(failed_bots),
        "latest_failed_bots": len(latest_failed_bots),
        "recovered_groups": recovered_groups,
        "statuses": _top_counter_values(
            status_totals,
            limit=FILL_REFRESH_HEALTH_VALUE_LIMIT,
        ),
        "groups_truncated": len(ordered) > FILL_REFRESH_HEALTH_GROUP_LIMIT,
        "groups": compact_groups,
    }


def _account_critical_remote_call_health_groups(
    groups: dict[tuple[Any, ...], dict[str, Any]]
) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {
        key: group
        for key, group in groups.items()
        if group.get("kind") == ACCOUNT_CRITICAL_REMOTE_CALL_KIND
        and group.get("surface") in ACCOUNT_CRITICAL_REMOTE_CALL_SURFACES
    }


def _remote_call_timing_group(
    *,
    bot_key: str,
    row: dict[str, Any],
    live_event: dict[str, Any],
    path: Path,
    line_no: int,
) -> dict[str, Any] | None:
    data = live_event.get("data")
    payload = data if isinstance(data, dict) else {}
    elapsed_ms = _non_negative_int(payload.get("elapsed_ms"))
    if elapsed_ms is None:
        return None
    ids = _event_ids(live_event)
    return {
        "bot": bot_key,
        "event_type": live_event.get("event_type") or row.get("kind"),
        "reason_code": live_event.get("reason_code"),
        "surface": payload.get("surface"),
        "kind": payload.get("kind"),
        "error_type": payload.get("error_type"),
        "component": live_event.get("component"),
        "status": live_event.get("status"),
        "symbol": live_event.get("symbol"),
        "count": 1,
        "elapsed_values": [elapsed_ms],
        "latest_ts": row.get("ts"),
        "latest_seq": row.get("seq"),
        "latest_path": str(path),
        "latest_line": int(line_no),
        "latest_elapsed_ms": elapsed_ms,
        "latest_ids": {
            key: ids.get(key)
            for key in ("cycle_id", "remote_call_id", "remote_call_group_id")
            if ids.get(key) is not None
        },
    }


def _merge_remote_call_timing_group(
    groups: dict[tuple[Any, ...], dict[str, Any]],
    group: dict[str, Any],
) -> None:
    key = (
        group.get("bot"),
        group.get("event_type"),
        group.get("reason_code"),
        group.get("surface"),
        group.get("kind"),
        group.get("error_type"),
        group.get("component"),
        group.get("status"),
        group.get("symbol"),
    )
    existing = groups.get(key)
    if existing is None:
        groups[key] = group
        return
    existing["count"] = int(existing.get("count", 0)) + 1
    existing.setdefault("elapsed_values", []).extend(group.get("elapsed_values") or [])
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
            "latest_ids",
        ):
            existing[field] = group.get(field)


def _remote_call_timing_sort_key(
    group: dict[str, Any],
) -> tuple[int, int, int, int, str, str]:
    summary = _ms_summary(
        [
            int(value)
            for value in (group.get("elapsed_values") or [])
            if _non_negative_int(value) is not None
        ]
    )
    return (
        -int(summary.get("p95_ms") or 0),
        -int(summary.get("max_ms") or 0),
        -int(group.get("count", 0)),
        -int(_non_negative_int(group.get("latest_ts")) or 0),
        str(group.get("bot") or ""),
        str(group.get("reason_code") or ""),
    )


def _summarize_remote_call_timings(
    groups: dict[tuple[Any, ...], dict[str, Any]]
) -> dict[str, Any]:
    ordered = sorted(groups.values(), key=_remote_call_timing_sort_key)
    compact_groups = []
    for group in ordered[:REMOTE_CALL_TIMING_GROUP_LIMIT]:
        compact = {
            key: value
            for key, value in group.items()
            if key
            not in {
                "elapsed_values",
                "latest_path",
                "latest_line",
                "latest_seq",
            }
            and value not in (None, {}, [])
        }
        compact["elapsed_ms"] = _ms_summary(
            [
                int(value)
                for value in (group.get("elapsed_values") or [])
                if _non_negative_int(value) is not None
            ]
        )
        compact_groups.append(compact)
    return {
        "total": sum(int(group.get("count", 0)) for group in groups.values()),
        "groups_truncated": len(ordered) > REMOTE_CALL_TIMING_GROUP_LIMIT,
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
    recovered: bool = False,
    recovery: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ids = _event_ids(live_event)
    group = {
        "bot": bot_key,
        "event_type": live_event.get("event_type") or row.get("kind"),
        "reason_code": live_event.get("reason_code"),
        "status": live_event.get("status"),
        "level": live_event.get("level"),
        "hard": bool(hard),
        "recovered": bool(recovered),
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
    if recovery:
        group["latest_recovery"] = recovery
    return group


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
        group.get("recovered"),
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
            "latest_recovery",
        ):
            existing[field] = group.get(field)


def _summarize_problem_event_groups(
    groups: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    event_types: Counter[str] = Counter()
    for group in groups.values():
        event_type = group.get("event_type")
        if isinstance(event_type, str) and event_type:
            event_types[event_type] += int(_non_negative_int(group.get("count")) or 0)
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
            and not (key == "recovered" and value is False)
            and value not in (None, {}, [])
        }
        for group in ordered[:PROBLEM_EVENT_GROUP_LIMIT]
    ]
    return {
        "total": sum(int(group.get("count", 0)) for group in groups.values()),
        "groups_truncated": len(ordered) > PROBLEM_EVENT_GROUP_LIMIT,
        "event_types": dict(event_types.most_common()),
        "groups": compact_groups,
    }


def _count_from_summary(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    return int(_non_negative_int(value.get("count")) or 0)


def _reason_symbol_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, list):
        return {}
    counts: Counter[str] = Counter()
    for item in value:
        if not isinstance(item, dict):
            continue
        reason = str(item.get("reason") or "unknown")
        counts[reason] += _count_from_summary(item.get("symbols")) or 1
    return dict(counts.most_common())


def _summary_symbol_sample(value: Any, *, limit: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    count = int(_non_negative_int(value.get("count")) or 0)
    raw_sample = value.get("sample")
    sample: list[str] = []
    if isinstance(raw_sample, list):
        for symbol in raw_sample:
            if symbol is None:
                continue
            safe_symbol = _redact_log_text(str(symbol), max_len=80)
            if safe_symbol and safe_symbol not in sample:
                sample.append(safe_symbol)
            if len(sample) >= limit:
                break
    if count <= 0:
        count = len(sample)
    if count <= 0 and not sample:
        return {}
    return {
        "count": count,
        "sample": sample,
        "truncated": max(0, count - len(sample)),
    }


def _reason_symbol_summaries(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        return {}
    summaries: dict[str, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        reason = str(item.get("reason") or "unknown")
        symbol_summary = _summary_symbol_sample(
            item.get("symbols"),
            limit=EMA_READINESS_REASON_SYMBOL_SAMPLE_LIMIT,
        )
        if symbol_summary:
            summaries[reason] = symbol_summary
    return summaries


def _sum_reason_symbol_summaries(values: Iterable[Any]) -> dict[str, dict[str, Any]]:
    totals: dict[str, int] = defaultdict(int)
    samples: dict[str, Counter[str]] = defaultdict(Counter)
    for value in values:
        if not isinstance(value, dict):
            continue
        for reason, summary in value.items():
            if not isinstance(summary, dict):
                continue
            key = str(reason)
            totals[key] += int(_non_negative_int(summary.get("count")) or 0)
            raw_sample = summary.get("sample")
            if isinstance(raw_sample, list):
                for symbol in raw_sample:
                    if symbol is not None:
                        safe_symbol = _redact_log_text(str(symbol), max_len=80)
                        if safe_symbol:
                            samples[key][safe_symbol] += 1
    merged: dict[str, dict[str, Any]] = {}
    for reason, count in sorted(totals.items(), key=lambda item: (-item[1], item[0])):
        sample_summary = _symbol_sample(
            samples.get(reason, Counter()),
            limit=EMA_READINESS_REASON_SYMBOL_SAMPLE_LIMIT,
        )
        sample = sample_summary.get("sample") if isinstance(sample_summary, dict) else []
        merged[reason] = {
            "count": int(count),
            "sample": sample or [],
            "truncated": max(0, int(count) - len(sample or [])),
        }
    return merged


def _candidate_error_type_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, list):
        return {}
    counts: Counter[str] = Counter()
    for item in value:
        if not isinstance(item, dict):
            continue
        error_types = item.get("error_types")
        if isinstance(error_types, list):
            for error_type in error_types:
                if error_type:
                    counts[str(error_type)] += 1
    return dict(counts.most_common())


def _sum_counter_maps(values: Iterable[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values:
        if not isinstance(value, dict):
            continue
        for key, count in value.items():
            if key is None:
                continue
            counts[str(key)] += int(_non_negative_int(count) or 0)
    return dict(counts.most_common())


def _ema_readiness_group(
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
    return {
        "bot": bot_key,
        "reason_code": live_event.get("reason_code"),
        "status": live_event.get("status"),
        "level": live_event.get("level"),
        "component": live_event.get("component"),
        "count": 1,
        "latest_ts": row.get("ts"),
        "latest_seq": row.get("seq"),
        "latest_path": str(path),
        "latest_line": int(line_no),
        "latest_candidate_unavailable_count": _count_from_summary(
            payload.get("candidate_unavailable")
        ),
        "latest_unavailable_count": _count_from_summary(payload.get("unavailable")),
        "latest_optional_drop_count": int(
            _non_negative_int(payload.get("optional_drop_count")) or 0
        ),
        "candidate_reason_counts": _reason_symbol_counts(
            payload.get("candidate_unavailable_groups")
        ),
        "unavailable_reason_counts": _reason_symbol_counts(
            payload.get("unavailable_reasons")
        ),
        "candidate_reason_symbols": _reason_symbol_summaries(
            payload.get("candidate_unavailable_groups")
        ),
        "unavailable_reason_symbols": _reason_symbol_summaries(
            payload.get("unavailable_reasons")
        ),
        "candidate_error_type_counts": _candidate_error_type_counts(
            payload.get("candidate_unavailable_groups")
        ),
        "latest_ids": {
            key: ids.get(key)
            for key in ("cycle_id", "snapshot_id", "plan_id", "action_id")
            if ids.get(key) is not None
        },
        "latest_data": _compact_problem_event_data(live_event),
    }


def _merge_ema_readiness_group(
    groups: dict[tuple[Any, ...], dict[str, Any]],
    group: dict[str, Any],
) -> None:
    key = (
        group.get("bot"),
        group.get("reason_code"),
        group.get("status"),
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
            "latest_candidate_unavailable_count",
            "latest_unavailable_count",
            "latest_optional_drop_count",
            "candidate_reason_counts",
            "unavailable_reason_counts",
            "candidate_reason_symbols",
            "unavailable_reason_symbols",
            "candidate_error_type_counts",
            "latest_ids",
            "latest_data",
        ):
            existing[field] = group.get(field)


def _summarize_ema_readiness_health(
    groups: dict[tuple[Any, ...], dict[str, Any]],
    event_type_counts: Counter[str],
) -> dict[str, Any]:
    ordered = sorted(
        groups.values(),
        key=lambda item: (
            -int(_non_negative_int(item.get("latest_ts")) or 0),
            -int(item.get("latest_candidate_unavailable_count") or 0),
            -int(item.get("latest_unavailable_count") or 0),
            -int(item.get("count") or 0),
            str(item.get("bot") or ""),
        ),
    )
    compact_groups = [
        {
            key: value
            for key, value in group.items()
            if key not in {"latest_path", "latest_line", "latest_seq"}
            and value not in (None, {}, [])
        }
        for group in ordered[:EMA_READINESS_GROUP_LIMIT]
    ]
    return {
        "total": sum(int(group.get("count", 0)) for group in groups.values()),
        "groups_truncated": len(ordered) > EMA_READINESS_GROUP_LIMIT,
        "event_types": dict(event_type_counts.most_common()),
        "bots": len({group.get("bot") for group in groups.values()}),
        "latest_candidate_unavailable_total": sum(
            int(group.get("latest_candidate_unavailable_count") or 0)
            for group in groups.values()
        ),
        "latest_unavailable_total": sum(
            int(group.get("latest_unavailable_count") or 0)
            for group in groups.values()
        ),
        "latest_optional_drop_total": sum(
            int(group.get("latest_optional_drop_count") or 0)
            for group in groups.values()
        ),
        "latest_candidate_reason_counts": _sum_counter_maps(
            group.get("candidate_reason_counts") for group in groups.values()
        ),
        "latest_unavailable_reason_counts": _sum_counter_maps(
            group.get("unavailable_reason_counts") for group in groups.values()
        ),
        "latest_candidate_reason_symbols": _sum_reason_symbol_summaries(
            group.get("candidate_reason_symbols") for group in groups.values()
        ),
        "latest_unavailable_reason_symbols": _sum_reason_symbol_summaries(
            group.get("unavailable_reason_symbols") for group in groups.values()
        ),
        "latest_candidate_error_type_counts": _sum_counter_maps(
            group.get("candidate_error_type_counts") for group in groups.values()
        ),
        "groups": compact_groups,
    }


def _exchange_config_refresh_group(
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
    latest_data: dict[str, Any] = {}
    for key in ("context", "operation", "error_type"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            latest_data[key] = _redact_log_text(value, max_len=120)
    elapsed_ms = _non_negative_int(payload.get("elapsed_ms"))
    if elapsed_ms is not None:
        latest_data["elapsed_ms"] = elapsed_ms
    started_ms = _non_negative_int(payload.get("started_ms"))
    if started_ms is not None:
        latest_data["started_ms"] = started_ms
    return {
        "bot": bot_key,
        "event_type": live_event.get("event_type") or row.get("kind"),
        "reason_code": live_event.get("reason_code"),
        "status": live_event.get("status"),
        "level": live_event.get("level"),
        "component": live_event.get("component"),
        "count": 1,
        "latest_ts": row.get("ts"),
        "latest_seq": row.get("seq"),
        "latest_path": str(path),
        "latest_line": int(line_no),
        "latest_data": latest_data,
        "latest_ids": {
            key: ids.get(key)
            for key in ("cycle_id", "remote_call_id")
            if ids.get(key) is not None
        },
    }


def _merge_exchange_config_refresh_group(
    groups: dict[tuple[Any, ...], dict[str, Any]],
    group: dict[str, Any],
) -> None:
    key = (
        group.get("bot"),
        group.get("status"),
        group.get("reason_code"),
        (group.get("latest_data") or {}).get("operation"),
        (group.get("latest_data") or {}).get("error_type"),
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


def _summarize_exchange_config_refresh_health(
    groups: dict[tuple[Any, ...], dict[str, Any]],
    event_type_counts: Counter[str],
) -> dict[str, Any]:
    ordered = sorted(
        groups.values(),
        key=lambda item: (
            -int(_non_negative_int(item.get("latest_ts")) or 0),
            str(item.get("bot") or ""),
            str(item.get("status") or ""),
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
        for group in ordered[:EXCHANGE_CONFIG_REFRESH_HEALTH_GROUP_LIMIT]
    ]
    status_counts: Counter[str] = Counter()
    failed_bots: set[str] = set()
    for group in groups.values():
        count = int(group.get("count") or 0)
        status = str(group.get("status") or "unknown")
        status_counts[status] += count
        if status == "failed" and count > 0 and group.get("bot") not in (None, ""):
            failed_bots.add(str(group.get("bot")))
    total = sum(int(group.get("count", 0)) for group in groups.values())
    failed = int(status_counts.get("failed", 0))
    return {
        "total": total,
        "succeeded": int(status_counts.get("succeeded", 0)),
        "failed": failed,
        "failure_pct": round((failed / total) * 100.0, 3) if total else 0,
        "groups_truncated": len(ordered) > EXCHANGE_CONFIG_REFRESH_HEALTH_GROUP_LIMIT,
        "event_types": dict(event_type_counts.most_common()),
        "statuses": dict(status_counts.most_common()),
        "bots": len({group.get("bot") for group in groups.values()}),
        "failed_bots": len(failed_bots),
        "groups": compact_groups,
    }


def _staged_readiness_event(live_event: dict[str, Any]) -> bool:
    if (live_event.get("event_type") or "") != EventTypes.CYCLE_DEGRADED:
        return False
    reason_code = str(live_event.get("reason_code") or "")
    return reason_code.startswith("staged_execution")


def _string_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, list):
        return {}
    counts: Counter[str] = Counter()
    for item in value:
        if item is not None:
            counts[str(item)] += 1
    return dict(counts.most_common())


def _invalid_surface_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts: Counter[str] = Counter()
    for key, rows in value.items():
        if isinstance(rows, list):
            counts[str(key)] += max(1, len(rows))
        else:
            counts[str(key)] += 1
    return dict(counts.most_common())


def _completed_candle_mismatch_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    rows = value.get("completed_candles")
    if not isinstance(rows, list):
        return {}
    counts: Counter[str] = Counter()
    for row in rows:
        if isinstance(row, dict):
            counts[str(row.get("mismatch_type") or "unknown")] += 1
    return dict(counts.most_common())


def _staged_readiness_group(
    *,
    bot_key: str,
    row: dict[str, Any],
    live_event: dict[str, Any],
    path: Path,
    line_no: int,
) -> dict[str, Any]:
    data = live_event.get("data")
    payload = data if isinstance(data, dict) else {}
    details = payload.get("details")
    detail_payload = details if isinstance(details, dict) else {}
    ids = _event_ids(live_event)
    invalid = detail_payload.get("invalid")
    missing_surfaces = _string_counts(detail_payload.get("missing"))
    invalid_surfaces = _invalid_surface_counts(invalid)
    return {
        "bot": bot_key,
        "reason_code": live_event.get("reason_code"),
        "status": live_event.get("status"),
        "level": live_event.get("level"),
        "component": live_event.get("component"),
        "count": 1,
        "latest_ts": row.get("ts"),
        "latest_seq": row.get("seq"),
        "latest_path": str(path),
        "latest_line": int(line_no),
        "latest_context": detail_payload.get("context"),
        "latest_defer_reason": detail_payload.get("defer_reason"),
        "latest_missing_surfaces": missing_surfaces,
        "latest_invalid_surfaces": invalid_surfaces,
        "latest_completed_candle_mismatch_counts": _completed_candle_mismatch_counts(
            invalid
        ),
        "latest_missing_surface_count": sum(missing_surfaces.values()),
        "latest_invalid_surface_count": sum(invalid_surfaces.values()),
        "latest_ids": {
            key: ids.get(key)
            for key in ("cycle_id", "snapshot_id", "plan_id", "action_id")
            if ids.get(key) is not None
        },
        "latest_data": _compact_problem_event_data(live_event),
    }


def _merge_staged_readiness_group(
    groups: dict[tuple[Any, ...], dict[str, Any]],
    group: dict[str, Any],
) -> None:
    key = (
        group.get("bot"),
        group.get("reason_code"),
        group.get("status"),
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
            "latest_context",
            "latest_defer_reason",
            "latest_missing_surfaces",
            "latest_invalid_surfaces",
            "latest_completed_candle_mismatch_counts",
            "latest_missing_surface_count",
            "latest_invalid_surface_count",
            "latest_ids",
            "latest_data",
        ):
            existing[field] = group.get(field)


def _summarize_staged_readiness_health(
    groups: dict[tuple[Any, ...], dict[str, Any]],
    event_type_counts: Counter[str],
) -> dict[str, Any]:
    ordered = sorted(
        groups.values(),
        key=lambda item: (
            -int(_non_negative_int(item.get("latest_ts")) or 0),
            -int(item.get("latest_invalid_surface_count") or 0),
            -int(item.get("latest_missing_surface_count") or 0),
            -int(item.get("count") or 0),
            str(item.get("bot") or ""),
        ),
    )
    compact_groups = [
        {
            key: value
            for key, value in group.items()
            if key not in {"latest_path", "latest_line", "latest_seq"}
            and value not in (None, {}, [])
        }
        for group in ordered[:STAGED_READINESS_GROUP_LIMIT]
    ]
    return {
        "total": sum(int(group.get("count", 0)) for group in groups.values()),
        "groups_truncated": len(ordered) > STAGED_READINESS_GROUP_LIMIT,
        "event_types": dict(event_type_counts.most_common()),
        "bots": len({group.get("bot") for group in groups.values()}),
        "latest_missing_surface_total": sum(
            int(group.get("latest_missing_surface_count") or 0)
            for group in groups.values()
        ),
        "latest_invalid_surface_total": sum(
            int(group.get("latest_invalid_surface_count") or 0)
            for group in groups.values()
        ),
        "latest_missing_surfaces": _sum_counter_maps(
            group.get("latest_missing_surfaces") for group in groups.values()
        ),
        "latest_invalid_surfaces": _sum_counter_maps(
            group.get("latest_invalid_surfaces") for group in groups.values()
        ),
        "groups": compact_groups,
    }


def _safe_counter_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, raw in value.items():
        parsed = _non_negative_int(raw)
        if parsed is not None:
            out[str(key)] = int(parsed)
    return out


def _event_pipeline_health_group(
    *,
    bot_key: str,
    row: dict[str, Any],
    live_event: dict[str, Any],
    path: Path,
    line_no: int,
) -> dict[str, Any] | None:
    data = live_event.get("data")
    payload = data if isinstance(data, dict) else {}
    observed_keys = {
        "event_queue_depth",
        "event_queue_maxsize",
        "event_queue_unfinished_tasks",
        "event_dropped_total",
        "event_drop_counts",
        "event_sink_error_total",
        "event_sink_error_counts",
        "event_degraded_count",
        "event_pipeline_stopping",
        "event_pipeline_worker_alive",
    }
    if not any(key in payload for key in observed_keys):
        return None
    ids = _event_ids(live_event)
    return {
        "bot": bot_key,
        "count": 1,
        "latest_ts": row.get("ts"),
        "latest_seq": row.get("seq"),
        "latest_path": str(path),
        "latest_line": int(line_no),
        "latest_queue_depth": _non_negative_int(payload.get("event_queue_depth")),
        "latest_queue_maxsize": _non_negative_int(payload.get("event_queue_maxsize")),
        "latest_queue_unfinished_tasks": _non_negative_int(
            payload.get("event_queue_unfinished_tasks")
        ),
        "latest_dropped_total": _non_negative_int(payload.get("event_dropped_total")),
        "latest_drop_counts": _safe_counter_mapping(payload.get("event_drop_counts")),
        "latest_sink_error_total": _non_negative_int(
            payload.get("event_sink_error_total")
        ),
        "latest_sink_error_counts": _safe_counter_mapping(
            payload.get("event_sink_error_counts")
        ),
        "latest_degraded_count": _non_negative_int(payload.get("event_degraded_count")),
        "latest_pipeline_stopping": (
            bool(payload.get("event_pipeline_stopping"))
            if "event_pipeline_stopping" in payload
            else None
        ),
        "latest_worker_alive": (
            bool(payload.get("event_pipeline_worker_alive"))
            if "event_pipeline_worker_alive" in payload
            else None
        ),
        "latest_ids": {
            key: ids.get(key)
            for key in ("cycle_id", "snapshot_id", "plan_id", "action_id")
            if ids.get(key) is not None
        },
    }


def _merge_event_pipeline_health_group(
    groups: dict[tuple[Any, ...], dict[str, Any]],
    group: dict[str, Any],
) -> None:
    key = (group.get("bot"),)
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
            "latest_queue_depth",
            "latest_queue_maxsize",
            "latest_queue_unfinished_tasks",
            "latest_dropped_total",
            "latest_drop_counts",
            "latest_sink_error_total",
            "latest_sink_error_counts",
            "latest_degraded_count",
            "latest_pipeline_stopping",
            "latest_worker_alive",
            "latest_ids",
        ):
            existing[field] = group.get(field)


def _summarize_event_pipeline_health(
    groups: dict[tuple[Any, ...], dict[str, Any]],
    event_type_counts: Counter[str],
) -> dict[str, Any]:
    ordered = sorted(
        groups.values(),
        key=lambda item: (
            -int(_non_negative_int(item.get("latest_dropped_total")) or 0),
            -int(_non_negative_int(item.get("latest_sink_error_total")) or 0),
            -int(_non_negative_int(item.get("latest_degraded_count")) or 0),
            -int(_non_negative_int(item.get("latest_queue_depth")) or 0),
            -int(_non_negative_int(item.get("latest_ts")) or 0),
            str(item.get("bot") or ""),
        ),
    )
    compact_groups = [
        {
            key: value
            for key, value in group.items()
            if key not in {"latest_path", "latest_line", "latest_seq"}
            and value not in (None, {}, [])
        }
        for group in ordered[:EVENT_PIPELINE_HEALTH_GROUP_LIMIT]
    ]
    return {
        "total": sum(int(group.get("count", 0)) for group in groups.values()),
        "groups_truncated": len(ordered) > EVENT_PIPELINE_HEALTH_GROUP_LIMIT,
        "event_types": dict(event_type_counts.most_common()),
        "bots": len({group.get("bot") for group in groups.values()}),
        "latest_queue_depth_total": sum(
            int(group.get("latest_queue_depth") or 0) for group in groups.values()
        ),
        "latest_queue_unfinished_total": sum(
            int(group.get("latest_queue_unfinished_tasks") or 0)
            for group in groups.values()
        ),
        "latest_dropped_total": sum(
            int(group.get("latest_dropped_total") or 0) for group in groups.values()
        ),
        "latest_sink_error_total": sum(
            int(group.get("latest_sink_error_total") or 0)
            for group in groups.values()
        ),
        "latest_degraded_total": sum(
            int(group.get("latest_degraded_count") or 0) for group in groups.values()
        ),
        "latest_worker_not_alive_count": sum(
            1 for group in groups.values() if group.get("latest_worker_alive") is False
        ),
        "latest_stopping_count": sum(
            1 for group in groups.values() if group.get("latest_pipeline_stopping") is True
        ),
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
            "ema_gap_to_red",
            "elapsed_minutes",
            "red_threshold",
            "cooldown_until_ms",
            "cooldown_remaining",
            "cooldown_remaining_seconds",
            "last_red_ts",
            "pending_red_since_ms",
            "stop_event_timestamp_ms",
            "stop_event_anchor_source",
            "stop_event_anchor_timestamp_ms",
            "stop_event_anchor_fallback_used",
            "no_exchange_close_needed",
            "exchange_close_order_submitted",
            "panic_order_submitted_count",
            "symbol_position_open",
            "position_count",
            "entry_orders",
            "nonpanic_close_orders",
            "flat_confirmations",
            "changed",
            "price_diff_pct",
            "status_counts",
            "over_budget_sides",
        )
        if payload.get(key) is not None
    }
    event_type = live_event.get("event_type") or row.get("kind")
    hsl_anchor_sources: Counter[str] = Counter()
    hsl_anchor_fallback_used = 0
    hsl_raw_red_pending_data: dict[str, Any] = {}
    if event_type == EventTypes.HSL_RED_FINALIZED_WITHOUT_ORDER:
        for key in (
            "drawdown_score",
            "drawdown_raw",
            "drawdown_ema",
            "dist_to_red",
            "red_threshold",
        ):
            latest_data.pop(key, None)
        anchor_source = latest_data.get("stop_event_anchor_source")
        if anchor_source not in (None, ""):
            hsl_anchor_sources[str(anchor_source)] += 1
        if latest_data.get("stop_event_anchor_fallback_used") is True:
            hsl_anchor_fallback_used = 1
    if event_type == EventTypes.HSL_RAW_RED_PENDING:
        hsl_raw_red_pending_data = {
            key: latest_data.get(key)
            for key in (
                "signal_mode",
                "tier",
                "drawdown_score",
                "ema_gap_to_red",
                "elapsed_minutes",
                "red_threshold",
            )
            if latest_data.get(key) is not None
        }
        for key in (
            "drawdown_score",
            "drawdown_raw",
            "drawdown_ema",
            "dist_to_red",
            "ema_gap_to_red",
            "red_threshold",
        ):
            latest_data.pop(key, None)
    return {
        "bot": bot_key,
        "event_type": event_type,
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
        "_hsl_anchor_sources": hsl_anchor_sources,
        "_hsl_anchor_fallback_used": hsl_anchor_fallback_used,
        "_hsl_raw_red_pending_data": hsl_raw_red_pending_data,
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
    existing_sources = existing.setdefault("_hsl_anchor_sources", Counter())
    if isinstance(existing_sources, Counter):
        existing_sources.update(group.get("_hsl_anchor_sources") or Counter())
    existing["_hsl_anchor_fallback_used"] = int(
        existing.get("_hsl_anchor_fallback_used") or 0
    ) + int(group.get("_hsl_anchor_fallback_used") or 0)
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
            "_hsl_raw_red_pending_data",
            "latest_ids",
        ):
            existing[field] = group.get(field)


def _summarize_hsl_flat_finalization_anchors(
    groups: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    total = 0
    fallback_used = 0
    bots: set[str] = set()
    symbols: Counter[str] = Counter()
    for group in groups.values():
        if group.get("event_type") != EventTypes.HSL_RED_FINALIZED_WITHOUT_ORDER:
            continue
        count = int(group.get("count") or 0)
        total += count
        bot = group.get("bot")
        if bot not in (None, ""):
            bots.add(str(bot))
        symbol = group.get("symbol")
        if symbol not in (None, ""):
            symbols[str(symbol)] += count
        sources = group.get("_hsl_anchor_sources")
        if isinstance(sources, Counter):
            source_counts.update(sources)
        latest_data = group.get("latest_data")
        if (
            count > 0
            and isinstance(latest_data, dict)
            and not sources
            and latest_data.get("stop_event_anchor_source") not in (None, "")
        ):
            source_counts[str(latest_data.get("stop_event_anchor_source"))] += count
        fallback_used += int(group.get("_hsl_anchor_fallback_used") or 0)
    if total <= 0:
        return {
            "total": 0,
            "source_counts": {},
            "fallback_used": 0,
            "fallback_used_pct": 0,
            "bots": 0,
            "symbols": {"count": 0, "sample": [], "truncated": 0},
        }
    return {
        "total": total,
        "source_counts": dict(source_counts.most_common()),
        "fallback_used": fallback_used,
        "fallback_used_pct": round((fallback_used / total) * 100.0, 3),
        "bots": len(bots),
        "symbols": _symbol_sample(symbols, limit=8),
    }


def _summarize_hsl_status(
    groups: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    total = 0
    bots: set[str] = set()
    symbols: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    signal_mode_counts: Counter[str] = Counter()
    closest: list[dict[str, Any]] = []
    cooldown_active: list[dict[str, Any]] = []
    for group in groups.values():
        if group.get("event_type") != EventTypes.HSL_STATUS:
            continue
        count = int(group.get("count") or 0)
        if count <= 0:
            continue
        total += count
        bot = group.get("bot")
        if bot not in (None, ""):
            bots.add(str(bot))
        symbol = group.get("symbol")
        if symbol not in (None, ""):
            symbols[str(symbol)] += count
        latest_data = group.get("latest_data")
        data = latest_data if isinstance(latest_data, dict) else {}
        tier = data.get("tier")
        if tier not in (None, ""):
            tier_counts[str(tier)] += count
        signal_mode = data.get("signal_mode")
        if signal_mode not in (None, ""):
            signal_mode_counts[str(signal_mode)] += count
        cooldown_until_ms = _non_negative_int(data.get("cooldown_until_ms"))
        cooldown_remaining_seconds = _numeric_value(
            data.get("cooldown_remaining_seconds")
        )
        if cooldown_until_ms is not None or cooldown_remaining_seconds is not None:
            cooldown_active.append(
                {
                    key: value
                    for key, value in {
                        "bot": bot,
                        "symbol": symbol,
                        "pside": group.get("pside"),
                        "tier": str(tier) if tier not in (None, "") else None,
                        "reason_code": group.get("reason_code"),
                        "cooldown_remaining_seconds": cooldown_remaining_seconds,
                        "cooldown_until_ms": cooldown_until_ms,
                        "latest_ts": _non_negative_int(group.get("latest_ts")),
                    }.items()
                    if value not in (None, "", {})
                }
            )
        dist_to_red = _numeric_value(data.get("dist_to_red"))
        if dist_to_red is None:
            continue
        red_threshold = _numeric_value(data.get("red_threshold"))
        drawdown_score = _numeric_value(data.get("drawdown_score"))
        red_proximity_pct = None
        if red_threshold is not None and red_threshold > 0:
            if drawdown_score is not None:
                red_proximity_pct = round((drawdown_score / red_threshold) * 100.0, 3)
            else:
                red_proximity_pct = round(
                    max(0.0, 1.0 - (dist_to_red / red_threshold)) * 100.0,
                    3,
                )
        sample = {
            key: value
            for key, value in {
                "bot": bot,
                "symbol": symbol,
                "pside": group.get("pside"),
                "tier": str(tier) if tier not in (None, "") else None,
                "signal_mode": str(signal_mode)
                if signal_mode not in (None, "")
                else None,
                "dist_to_red": dist_to_red,
                "red_threshold": red_threshold,
                "red_proximity_pct": red_proximity_pct,
                "latest_ts": _non_negative_int(group.get("latest_ts")),
            }.items()
            if value not in (None, "", {})
        }
        closest.append(sample)
    if total <= 0:
        return {
            "total": 0,
            "bots": 0,
            "symbols": {"count": 0, "sample": [], "truncated": 0},
            "tier_counts": {},
            "signal_mode_counts": {},
            "closest_to_red": [],
        }
    closest_sorted = sorted(
        closest,
        key=lambda item: (
            float(item.get("dist_to_red")),
            str(item.get("bot") or ""),
            str(item.get("symbol") or ""),
            str(item.get("pside") or ""),
        ),
    )
    cooldown_sorted = sorted(
        cooldown_active,
        key=lambda item: (
            -int(item.get("latest_ts") or 0),
            str(item.get("bot") or ""),
            str(item.get("symbol") or ""),
            str(item.get("pside") or ""),
        ),
    )
    out = {
        "total": total,
        "bots": len(bots),
        "symbols": _symbol_sample(symbols, limit=8),
        "tier_counts": dict(tier_counts.most_common()),
        "signal_mode_counts": dict(signal_mode_counts.most_common()),
        "closest_to_red": closest_sorted[:5],
        "closest_to_red_truncated": max(0, len(closest_sorted) - 5),
    }
    if cooldown_sorted:
        out["cooldown_active"] = cooldown_sorted[:5]
        out["cooldown_active_truncated"] = max(0, len(cooldown_sorted) - 5)
    return out


def _summarize_hsl_raw_red_pending(
    groups: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    total = 0
    bots: set[str] = set()
    symbols: Counter[str] = Counter()
    signal_mode_counts: Counter[str] = Counter()
    pending: list[dict[str, Any]] = []
    for group in groups.values():
        if group.get("event_type") != EventTypes.HSL_RAW_RED_PENDING:
            continue
        count = int(group.get("count") or 0)
        if count <= 0:
            continue
        total += count
        bot = group.get("bot")
        if bot not in (None, ""):
            bots.add(str(bot))
        symbol = group.get("symbol")
        if symbol not in (None, ""):
            symbols[str(symbol)] += count
        latest_data = group.get("latest_data")
        raw_pending_data = group.get("_hsl_raw_red_pending_data")
        data = (
            raw_pending_data
            if isinstance(raw_pending_data, dict) and raw_pending_data
            else latest_data
            if isinstance(latest_data, dict)
            else {}
        )
        signal_mode = data.get("signal_mode")
        if signal_mode not in (None, ""):
            signal_mode_counts[str(signal_mode)] += count
        tier = data.get("tier")
        red_threshold = _numeric_value(data.get("red_threshold"))
        drawdown_score = _numeric_value(data.get("drawdown_score"))
        red_proximity_pct = None
        if red_threshold is not None and red_threshold > 0 and drawdown_score is not None:
            red_proximity_pct = round((drawdown_score / red_threshold) * 100.0, 3)
        ema_gap_to_red = _numeric_value(data.get("ema_gap_to_red"))
        ema_gap_to_red_pct = None
        if red_threshold is not None and red_threshold > 0 and ema_gap_to_red is not None:
            ema_gap_to_red_pct = round((ema_gap_to_red / red_threshold) * 100.0, 3)
        pending.append(
            {
                key: value
                for key, value in {
                    "bot": bot,
                    "symbol": symbol,
                    "pside": group.get("pside"),
                    "tier": str(tier) if tier not in (None, "") else None,
                    "signal_mode": str(signal_mode)
                    if signal_mode not in (None, "")
                    else None,
                    "red_proximity_pct": red_proximity_pct,
                    "ema_gap_to_red_pct": ema_gap_to_red_pct,
                    "elapsed_minutes": _non_negative_int(data.get("elapsed_minutes")),
                    "latest_ts": _non_negative_int(group.get("latest_ts")),
                }.items()
                if value not in (None, "", {})
            }
        )
    if total <= 0:
        return {
            "total": 0,
            "bots": 0,
            "symbols": {"count": 0, "sample": [], "truncated": 0},
            "signal_mode_counts": {},
            "pending": [],
            "pending_truncated": 0,
        }
    pending_sorted = sorted(
        pending,
        key=lambda item: (
            -float(item.get("red_proximity_pct") or 0.0),
            -int(item.get("latest_ts") or 0),
            str(item.get("bot") or ""),
            str(item.get("symbol") or ""),
            str(item.get("pside") or ""),
        ),
    )
    return {
        "total": total,
        "bots": len(bots),
        "symbols": _symbol_sample(symbols, limit=8),
        "signal_mode_counts": dict(signal_mode_counts.most_common()),
        "pending": pending_sorted[:5],
        "pending_truncated": max(0, len(pending_sorted) - 5),
    }


def _shareable_hsl_status(hsl_status: Any) -> dict[str, Any]:
    if not isinstance(hsl_status, dict):
        return {}
    out = {
        key: hsl_status.get(key)
        for key in (
            "total",
            "bots",
            "symbols",
            "tier_counts",
            "signal_mode_counts",
            "closest_to_red_truncated",
            "cooldown_active_truncated",
        )
        if hsl_status.get(key) is not None
    }
    closest = hsl_status.get("closest_to_red")
    if isinstance(closest, list):
        out["closest_to_red"] = [
            {
                key: item.get(key)
                for key in (
                    "bot",
                    "symbol",
                    "pside",
                    "tier",
                    "signal_mode",
                    "red_proximity_pct",
                    "latest_ts",
                )
                if isinstance(item, dict) and item.get(key) not in (None, "", {})
            }
            for item in closest[:5]
            if isinstance(item, dict)
        ]
    cooldown_active = hsl_status.get("cooldown_active")
    if isinstance(cooldown_active, list):
        compact_cooldown_active = [
            {
                key: item.get(key)
                for key in (
                    "bot",
                    "symbol",
                    "pside",
                    "tier",
                    "reason_code",
                    "cooldown_remaining_seconds",
                    "cooldown_until_ms",
                    "latest_ts",
                )
                if isinstance(item, dict) and item.get(key) not in (None, "", {})
            }
            for item in cooldown_active[:5]
            if isinstance(item, dict)
        ]
        if compact_cooldown_active:
            out["cooldown_active"] = compact_cooldown_active
    return out


def _shareable_hsl_raw_red_pending(raw_red_pending: Any) -> dict[str, Any]:
    if not isinstance(raw_red_pending, dict):
        return {}
    out = {
        key: raw_red_pending.get(key)
        for key in (
            "total",
            "bots",
            "symbols",
            "signal_mode_counts",
            "pending_truncated",
        )
        if raw_red_pending.get(key) is not None
    }
    pending = raw_red_pending.get("pending")
    if isinstance(pending, list):
        compact_pending = [
            {
                key: item.get(key)
                for key in (
                    "bot",
                    "symbol",
                    "pside",
                    "tier",
                    "signal_mode",
                    "red_proximity_pct",
                    "ema_gap_to_red_pct",
                    "elapsed_minutes",
                    "latest_ts",
                )
                if isinstance(item, dict) and item.get(key) not in (None, "", {})
            }
            for item in pending[:5]
            if isinstance(item, dict)
        ]
        if compact_pending:
            out["pending"] = compact_pending
    return out


SHAREABLE_RISK_LATEST_DATA_KEYS = frozenset(
    {
        "signal_mode",
        "tier",
        "previous_tier",
        "action",
        "mode",
        "previous_mode",
        "cooldown_until_ms",
        "cooldown_remaining",
        "cooldown_remaining_seconds",
        "last_red_ts",
        "pending_red_since_ms",
        "elapsed_minutes",
        "stop_event_timestamp_ms",
        "stop_event_anchor_source",
        "stop_event_anchor_timestamp_ms",
        "stop_event_anchor_fallback_used",
        "no_exchange_close_needed",
        "exchange_close_order_submitted",
        "panic_order_submitted_count",
        "symbol_position_open",
        "position_count",
        "entry_orders",
        "nonpanic_close_orders",
        "flat_confirmations",
        "changed",
        "status_counts",
        "over_budget_sides",
    }
)


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
            if key
            not in {
                "latest_path",
                "latest_line",
                "latest_seq",
                "_hsl_anchor_sources",
                "_hsl_anchor_fallback_used",
                "_hsl_raw_red_pending_data",
            }
            and value not in (None, {}, [])
        }
        for group in ordered[:RISK_EVENT_GROUP_LIMIT]
    ]
    out = {
        "total": sum(int(group.get("count", 0)) for group in groups.values()),
        "groups_truncated": len(ordered) > RISK_EVENT_GROUP_LIMIT,
        "event_types": dict(event_type_counts.most_common()),
        "hsl_flat_finalization_anchors": _summarize_hsl_flat_finalization_anchors(
            groups
        ),
        "hsl_status": _summarize_hsl_status(groups),
        "groups": compact_groups,
    }
    hsl_raw_red_pending = _summarize_hsl_raw_red_pending(groups)
    if int(hsl_raw_red_pending.get("total") or 0) > 0:
        out["hsl_raw_red_pending"] = hsl_raw_red_pending
    return out


def _compact_shutdown_event_data(live_event: dict[str, Any]) -> dict[str, Any]:
    data = live_event.get("data")
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for key in (
        "reason",
        "stage",
        "elapsed_s",
        "task_count",
        "timeout_s",
        "cancel_timeout_s",
        "threshold_s",
        "closed",
        "inline",
        "error",
    ):
        if key not in data or data.get(key) is None:
            continue
        value = data.get(key)
        if isinstance(value, str):
            out[key] = _redact_log_text(value, max_len=240)
        elif isinstance(value, (bool, int, float)):
            out[key] = value
        else:
            compact = _compact_problem_event_data_value(value)
            if compact not in (None, "", [], {}):
                out[key] = compact
    return out


def _shutdown_event_group(
    *,
    bot_key: str,
    row: dict[str, Any],
    live_event: dict[str, Any],
    path: Path,
    line_no: int,
) -> dict[str, Any]:
    ids = _event_ids(live_event)
    return {
        "bot": bot_key,
        "event_type": live_event.get("event_type") or row.get("kind"),
        "reason_code": live_event.get("reason_code"),
        "status": live_event.get("status"),
        "level": live_event.get("level"),
        "component": live_event.get("component"),
        "count": 1,
        "latest_ts": row.get("ts"),
        "latest_seq": row.get("seq"),
        "latest_path": str(path),
        "latest_line": int(line_no),
        "latest_data": _compact_shutdown_event_data(live_event),
        "latest_ids": {
            key: ids.get(key)
            for key in ("cycle_id", "snapshot_id", "plan_id", "action_id")
            if ids.get(key) is not None
        },
    }


def _merge_shutdown_event_group(
    groups: dict[tuple[Any, ...], dict[str, Any]],
    group: dict[str, Any],
) -> None:
    key = (
        group.get("bot"),
        group.get("event_type"),
        group.get("reason_code"),
        group.get("status"),
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
            "latest_message",
            "latest_data",
            "latest_ids",
        ):
            existing[field] = group.get(field)


def _summarize_shutdown_events(
    groups: dict[tuple[Any, ...], dict[str, Any]],
    event_type_counts: Counter[str],
) -> dict[str, Any]:
    ordered = sorted(
        groups.values(),
        key=lambda item: (
            -int(_non_negative_int(item.get("latest_ts")) or 0),
            str(item.get("bot") or ""),
            str(item.get("event_type") or ""),
            str(item.get("reason_code") or ""),
            str(item.get("status") or ""),
        ),
    )
    compact_groups = [
        {
            key: value
            for key, value in group.items()
            if key not in {"latest_path", "latest_line", "latest_seq"}
            and value not in (None, {}, [])
        }
        for group in ordered[:SHUTDOWN_EVENT_GROUP_LIMIT]
    ]
    return {
        "total": sum(int(group.get("count", 0)) for group in groups.values()),
        "groups_truncated": len(ordered) > SHUTDOWN_EVENT_GROUP_LIMIT,
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


def _numeric_value(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if value != value else round(value, 6)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    if parsed.is_integer():
        return int(parsed)
    return round(parsed, 6)


def _compact_hsl_replay_data(live_event: dict[str, Any]) -> dict[str, Any]:
    data = live_event.get("data")
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("error_type", "signal_mode", "stage", "timeframe"):
        value = data.get(key)
        if isinstance(value, str) and value:
            out[key] = _redact_log_text(value, max_len=80)
    for key in ("is_held_pair", "is_cooldown_pair"):
        value = data.get(key)
        if isinstance(value, bool):
            out[key] = value
    for key in (
        "lookback_days",
        "symbols",
        "pairs",
        "held_pairs",
        "cooldown_pairs",
        "required_pairs",
        "timeline_rows",
        "fill_events",
        "panic_events",
        "skipped_unsupported_symbols",
        "events",
        "current_position_pairs",
        "price_replay_symbols",
        "priced_symbols",
        "empty_price_symbols",
        "approximate_price_symbols",
        "skipped_price_symbols",
        "missing_price_symbols",
        "history_minutes",
        "replay_concurrency",
        "start_ts",
        "end_ts",
        "record_start_ts",
        "pair_idx",
        "applied_rows",
        "total_applied_rows",
        "rows",
        "skipped_pairs",
        "rows_per_second",
        "elapsed_s",
        "history_build_elapsed_s",
        "price_history_fetch_elapsed_s",
        "timeline_replay_elapsed_s",
        "full_elapsed_s",
        "startup_blocking_elapsed_s",
    ):
        value = _numeric_value(data.get(key))
        if value is not None:
            out[key] = value
    return out


def _hsl_observed_rows(data: dict[str, Any]) -> int | None:
    for key in ("total_applied_rows", "rows", "applied_rows"):
        value = _non_negative_int(data.get(key))
        if value is not None:
            return value
    return None


def _hsl_replay_remaining_rows(
    *,
    estimated_work: int | None,
    observed_rows: int | None,
) -> int | None:
    if estimated_work is None or observed_rows is None:
        return None
    return max(0, int(estimated_work) - int(observed_rows))


def _hsl_replay_eta_ms(
    *,
    remaining_rows: int | None,
    rows_per_second: Any,
) -> int | None:
    if remaining_rows is None:
        return None
    try:
        rate = float(rows_per_second)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rate) or rate <= 0.0:
        return None
    return int(round(1000.0 * float(remaining_rows) / rate))


def _hsl_replay_derived(data: dict[str, Any]) -> dict[str, Any]:
    timeline_rows = _non_negative_int(data.get("timeline_rows"))
    pairs = _non_negative_int(data.get("pairs"))
    required_pairs = _non_negative_int(data.get("required_pairs"))
    held_pairs = _non_negative_int(data.get("held_pairs"))
    cooldown_pairs = _non_negative_int(data.get("cooldown_pairs"))
    observed_rows = _hsl_observed_rows(data)
    out: dict[str, Any] = {}
    dense_work: int | None = None
    required_work: int | None = None
    if timeline_rows is not None and pairs is not None:
        dense_work = int(timeline_rows) * int(pairs)
        out["estimated_dense_pair_row_work"] = dense_work
        if observed_rows is not None and dense_work > 0:
            out["observed_work_pct"] = round(
                min(100.0, max(0.0, 100.0 * float(observed_rows) / float(dense_work))),
                3,
            )
    if timeline_rows is not None and required_pairs is not None:
        required_work = int(timeline_rows) * int(required_pairs)
        out["estimated_required_pair_row_work"] = required_work
        if observed_rows is not None and required_work > 0:
            out["observed_required_work_pct"] = round(
                min(100.0, max(0.0, 100.0 * float(observed_rows) / float(required_work))),
                3,
            )
    if timeline_rows is not None and held_pairs is not None:
        out["estimated_held_pair_row_work"] = int(timeline_rows) * int(held_pairs)
    if timeline_rows is not None and cooldown_pairs is not None:
        out["estimated_cooldown_pair_row_work"] = int(timeline_rows) * int(cooldown_pairs)
    if observed_rows is not None:
        out["observed_applied_rows"] = int(observed_rows)
    dense_remaining_rows = _hsl_replay_remaining_rows(
        estimated_work=dense_work,
        observed_rows=observed_rows,
    )
    if dense_remaining_rows is not None:
        out["estimated_dense_remaining_rows"] = dense_remaining_rows
        dense_remaining_ms = _hsl_replay_eta_ms(
            remaining_rows=dense_remaining_rows,
            rows_per_second=data.get("rows_per_second"),
        )
        if dense_remaining_ms is not None:
            out["estimated_dense_remaining_ms"] = dense_remaining_ms
    required_remaining_rows = _hsl_replay_remaining_rows(
        estimated_work=required_work,
        observed_rows=observed_rows,
    )
    if required_remaining_rows is not None:
        out["estimated_required_remaining_rows"] = required_remaining_rows
        required_remaining_ms = _hsl_replay_eta_ms(
            remaining_rows=required_remaining_rows,
            rows_per_second=data.get("rows_per_second"),
        )
        if required_remaining_ms is not None:
            out["estimated_required_remaining_ms"] = required_remaining_ms
    primary_remaining_rows = (
        required_remaining_rows
        if required_remaining_rows is not None
        else dense_remaining_rows
    )
    if primary_remaining_rows is not None:
        out["estimated_remaining_rows"] = primary_remaining_rows
        primary_remaining_ms = _hsl_replay_eta_ms(
            remaining_rows=primary_remaining_rows,
            rows_per_second=data.get("rows_per_second"),
        )
        if primary_remaining_ms is not None:
            out["estimated_remaining_ms"] = primary_remaining_ms
    for source_key, target_key in (
        ("elapsed_s", "latest_elapsed_ms"),
        ("history_build_elapsed_s", "history_build_elapsed_ms"),
        ("price_history_fetch_elapsed_s", "price_history_fetch_elapsed_ms"),
        ("timeline_replay_elapsed_s", "timeline_replay_elapsed_ms"),
        ("full_elapsed_s", "full_elapsed_ms"),
        ("startup_blocking_elapsed_s", "startup_blocking_elapsed_ms"),
    ):
        value = data.get(source_key)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed == parsed and parsed >= 0.0:
            out[target_key] = int(round(parsed * 1000.0))
            if source_key == "history_build_elapsed_s" and "latest_elapsed_ms" not in out:
                out["latest_elapsed_ms"] = out[target_key]
    return out


def _hsl_replay_record(
    *,
    row: dict[str, Any],
    live_event: dict[str, Any],
    path: Path,
    line_no: int,
) -> dict[str, Any]:
    data = _compact_hsl_replay_data(live_event)
    ids = _event_ids(live_event)
    return {
        key: value
        for key, value in {
            "event_type": live_event.get("event_type") or row.get("kind"),
            "reason_code": live_event.get("reason_code"),
            "status": live_event.get("status"),
            "level": live_event.get("level"),
            "component": live_event.get("component"),
            "ts": row.get("ts"),
            "seq": row.get("seq"),
            "path": str(path),
            "line": int(line_no),
            "symbol": live_event.get("symbol") or row.get("symbol"),
            "pside": live_event.get("pside") or row.get("pside"),
            "data": data,
            "derived": _hsl_replay_derived(data),
            "ids": {
                key: ids.get(key)
                for key in ("cycle_id", "snapshot_id", "plan_id", "action_id")
                if ids.get(key) is not None
            },
        }.items()
        if value not in (None, {}, [])
    }


def _merge_hsl_replay_group(
    groups: dict[str, dict[str, Any]],
    *,
    bot_key: str,
    row: dict[str, Any],
    live_event: dict[str, Any],
    path: Path,
    line_no: int,
) -> None:
    event_type = str(live_event.get("event_type") or row.get("kind") or "")
    record = _hsl_replay_record(
        row=row,
        live_event=live_event,
        path=path,
        line_no=line_no,
    )
    group = groups.get(bot_key)
    if group is None:
        group = {
            "bot": bot_key,
            "count": 0,
            "event_types": Counter(),
            "latest": None,
            "started": None,
            "loaded": None,
            "progress": None,
            "completed": None,
        }
        groups[bot_key] = group
    group["count"] = int(group.get("count", 0)) + 1
    group["event_types"][event_type] += 1

    def is_newer(candidate: dict[str, Any], existing: dict[str, Any] | None) -> bool:
        if existing is None:
            return True
        return _sort_event_position_key(
            ts=candidate.get("ts"),
            seq=candidate.get("seq"),
            path=candidate.get("path") or "",
            line_no=int(candidate.get("line") or 0),
        ) > _sort_event_position_key(
            ts=existing.get("ts"),
            seq=existing.get("seq"),
            path=existing.get("path") or "",
            line_no=int(existing.get("line") or 0),
        )

    if is_newer(record, group.get("latest")):
        group["latest"] = record
    data = record.get("data") if isinstance(record.get("data"), dict) else {}
    if event_type == EventTypes.HSL_REPLAY_STARTED and is_newer(
        record, group.get("started")
    ):
        group["started"] = record
    elif (
        event_type == EventTypes.HSL_REPLAY_PROGRESS
        and data.get("stage") == "loaded"
        and is_newer(record, group.get("loaded"))
    ):
        group["loaded"] = record
    elif event_type == EventTypes.HSL_REPLAY_PROGRESS and is_newer(
        record, group.get("progress")
    ):
        group["progress"] = record
    elif event_type == EventTypes.HSL_REPLAY_COMPLETED and is_newer(
        record, group.get("completed")
    ):
        group["completed"] = record
    elif event_type == EventTypes.HSL_REPLAY_FAILED and is_newer(
        record, group.get("failed")
    ):
        group["failed"] = record


def _public_hsl_replay_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    return {
        key: value
        for key, value in record.items()
        if key not in {"path", "line", "seq"} and value not in (None, {}, [])
    }


def _hsl_replay_group_active(group: dict[str, Any]) -> bool:
    latest = group.get("latest")
    if not isinstance(latest, dict):
        return False
    return latest.get("event_type") not in {
        EventTypes.HSL_REPLAY_COMPLETED,
        EventTypes.HSL_REPLAY_FAILED,
    }


def _hsl_replay_latest_event_age_ms(
    record: dict[str, Any],
    *,
    report_ts_ms: int,
) -> int | None:
    ts = _non_negative_int(record.get("ts"))
    if ts is None:
        return None
    return int(max(0, int(report_ts_ms) - int(ts)))


def _hsl_replay_record_elapsed_ms(record: Any) -> int | None:
    if not isinstance(record, dict):
        return None
    derived = record.get("derived") if isinstance(record.get("derived"), dict) else {}
    elapsed_candidates = [
        _non_negative_int(derived.get(key))
        for key in (
            "latest_elapsed_ms",
            "startup_blocking_elapsed_ms",
            "full_elapsed_ms",
            "history_build_elapsed_ms",
            "price_history_fetch_elapsed_ms",
            "timeline_replay_elapsed_ms",
        )
    ]
    elapsed_values = [int(value) for value in elapsed_candidates if value is not None]
    return max(elapsed_values) if elapsed_values else None


def _with_hsl_replay_active_age(
    group: dict[str, Any],
    *,
    report_ts_ms: int,
) -> dict[str, Any]:
    latest = group.get("latest")
    if not _hsl_replay_group_active(group) or not isinstance(latest, dict):
        return group
    age_ms = _hsl_replay_latest_event_age_ms(latest, report_ts_ms=report_ts_ms)
    if age_ms is None:
        return group
    out = dict(group)
    latest_out = dict(latest)
    derived = latest_out.get("derived")
    derived_out = dict(derived) if isinstance(derived, dict) else {}
    derived_out["latest_event_age_ms"] = int(age_ms)
    latest_out["derived"] = derived_out
    out["latest"] = latest_out
    out["active_latest_event_age_ms"] = int(age_ms)
    elapsed_ms = _hsl_replay_record_elapsed_ms(latest_out)
    stale = int(age_ms) >= HSL_REPLAY_STALE_ACTIVE_EVENT_AGE_MS
    long_running = (
        elapsed_ms is not None
        and int(elapsed_ms) >= HSL_REPLAY_LONG_RUNNING_ACTIVE_MS
    )
    if stale:
        out["active_stale"] = True
        out["active_stale_threshold_ms"] = HSL_REPLAY_STALE_ACTIVE_EVENT_AGE_MS
    if long_running:
        out["active_long_running"] = True
        out["active_long_running_threshold_ms"] = HSL_REPLAY_LONG_RUNNING_ACTIVE_MS
    return out


def _summarize_hsl_replay_health(
    groups: dict[str, dict[str, Any]],
    event_type_counts: Counter[str],
    *,
    report_ts_ms: int | None = None,
) -> dict[str, Any]:
    if report_ts_ms is None:
        report_ts_ms = utc_ms()
    ordered = sorted(
        groups.values(),
        key=lambda item: (
            0 if _hsl_replay_group_active(item) else 1,
            -int(
                (((item.get("latest") or {}).get("derived") or {}).get(
                    "startup_blocking_elapsed_ms"
                ))
                or 0
            ),
            -int(((item.get("latest") or {}).get("ts")) or 0),
            str(item.get("bot") or ""),
        ),
    )
    compact_groups: list[dict[str, Any]] = []
    active_bots = 0
    stale_active_bots = 0
    long_running_active_bots = 0
    completed_bots = 0
    failed_bots = 0
    failed_attention_bots = 0
    for group in ordered:
        active = _hsl_replay_group_active(group)
        latest = _public_hsl_replay_record(group.get("latest"))
        completed = _public_hsl_replay_record(group.get("completed"))
        failed = _public_hsl_replay_record(group.get("failed"))
        if active:
            active_bots += 1
        elif latest.get("event_type") == EventTypes.HSL_REPLAY_FAILED:
            failed_bots += 1
            if latest.get("reason_code") != "shutdown_cancelled":
                failed_attention_bots += 1
        elif latest.get("event_type") == EventTypes.HSL_REPLAY_COMPLETED:
            if completed.get("status") == "succeeded":
                completed_bots += 1
            elif completed.get("status") == "failed":
                failed_bots += 1
        elif failed:
            failed_bots += 1
            if failed.get("reason_code") != "shutdown_cancelled":
                failed_attention_bots += 1
        public_group = {
            "bot": group.get("bot"),
            "active": active,
            "count": int(group.get("count") or 0),
            "event_types": dict(group.get("event_types").most_common())
            if isinstance(group.get("event_types"), Counter)
            else {},
            "latest": latest,
            "started": _public_hsl_replay_record(group.get("started")),
            "loaded": _public_hsl_replay_record(group.get("loaded")),
            "progress": _public_hsl_replay_record(group.get("progress")),
            "completed": completed,
            "failed": failed,
        }
        public_group = _with_hsl_replay_active_age(
            public_group,
            report_ts_ms=int(report_ts_ms),
        )
        if active and bool(public_group.get("active_stale")):
            stale_active_bots += 1
        if active and bool(public_group.get("active_long_running")):
            long_running_active_bots += 1
        compact_groups.append(
            {
                key: value
                for key, value in public_group.items()
                if value not in (None, {}, [])
            }
        )
    return {
        "total": sum(int(group.get("count", 0)) for group in groups.values()),
        "groups_truncated": len(ordered) > HSL_REPLAY_HEALTH_GROUP_LIMIT,
        "event_types": dict(event_type_counts.most_common()),
        "bots": len(groups),
        "active_bots": int(active_bots),
        "stale_active_bots": int(stale_active_bots),
        "long_running_active_bots": int(long_running_active_bots),
        "completed_bots": int(completed_bots),
        "failed_bots": int(failed_bots),
        "failed_attention_bots": int(failed_attention_bots),
        "groups": compact_groups[:HSL_REPLAY_HEALTH_GROUP_LIMIT],
    }


def _shell_tokens(value: str) -> list[str]:
    try:
        import shlex

        return shlex.split(str(value))
    except (ImportError, ValueError):
        return str(value).split()


def _redact_live_command_for_report(value: str, *, max_len: int = 500) -> str:
    tokens = _shell_tokens(value)
    redacted: list[str] = []
    index = 0
    balance_flags = {
        flag
        for flag, target in LIVE_COMMAND_VALUE_FLAGS.items()
        if target == "balance_override"
    }
    balance_eq_prefixes = {
        prefix
        for prefix, target in LIVE_COMMAND_EQ_PREFIXES.items()
        if target == "balance_override"
    }
    while index < len(tokens):
        token = tokens[index]
        if token in balance_flags:
            redacted.append(token)
            if index + 1 < len(tokens):
                redacted.append("[redacted]")
                index += 2
            else:
                index += 1
            continue
        matched_eq = False
        for prefix in balance_eq_prefixes:
            if token.startswith(prefix):
                redacted.append(f"{prefix}[redacted]")
                matched_eq = True
                break
        if not matched_eq:
            redacted.append(token)
        index += 1
    return _redact_log_text(" ".join(redacted), max_len=max_len)


def _canonical_live_command(value: str) -> str:
    tokens = _shell_tokens(value)
    for index, token in enumerate(tokens[:-1]):
        if Path(token).name == "passivbot" and tokens[index + 1] == "live":
            return " ".join([Path(tokens[index]).name, *tokens[index + 1 :]])
    return ""


def _live_command_context(command_key: str) -> dict[str, str]:
    tokens = _shell_tokens(command_key)
    if len(tokens) < 2 or tokens[0] != "passivbot" or tokens[1] != "live":
        return {}
    account: str | None = None
    config_path: str | None = None
    balance_override: str | None = None
    args = tokens[2:]
    index = 0
    while index < len(args):
        token = args[index]
        if token in LIVE_COMMAND_VALUE_FLAGS:
            if index + 1 < len(args):
                value = args[index + 1]
                target = LIVE_COMMAND_VALUE_FLAGS[token]
                if target == "account":
                    account = value
                elif target == "balance_override":
                    balance_override = value
            index += 2
            continue
        matched_eq = False
        for prefix, target in LIVE_COMMAND_EQ_PREFIXES.items():
            if token.startswith(prefix):
                value = token.split("=", 1)[1]
                if target == "account":
                    account = value
                elif target == "balance_override":
                    balance_override = value
                matched_eq = True
                break
        if matched_eq:
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        if config_path is None:
            config_path = token
        index += 1

    out: dict[str, str] = {}
    if account:
        out["account"] = _redact_log_text(account, max_len=160)
    if balance_override is not None and str(balance_override).strip() != "":
        out["balance_override"] = _redact_log_text(balance_override, max_len=80)
    if config_path:
        out["config_path"] = _redact_log_text(config_path, max_len=260)
    if account and config_path:
        out["config_key"] = _redact_log_text(f"{account}:{config_path}", max_len=320)
    elif account:
        out["config_key"] = _redact_log_text(account, max_len=160)
    elif config_path:
        out["config_key"] = _redact_log_text(config_path, max_len=260)
    return out


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
        context = _live_command_context(command_key)
        expected.append(
            {
                "name": current_window,
                "command": _redact_live_command_for_report(command, max_len=400),
                "command_key": _redact_live_command_for_report(command_key, max_len=400),
                "_match_key": command_key,
                **context,
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
        ["ps", "-eo", "pid=,ppid=,etimes=,stat=,pcpu=,pmem=,rss=,command="],
        ["ps", "-axo", "pid=,ppid=,stat=,pcpu=,pmem=,rss=,command="],
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


def _find_repository_root(
    monitor_root: str | Path,
    *,
    repo_root: str | Path | None,
) -> Path:
    if repo_root is not None:
        return _resolve_path(repo_root)

    starts: list[Path] = []
    monitor_path = _resolve_path(monitor_root)
    starts.append(monitor_path.parent if monitor_path.name == "monitor" else monitor_path)
    starts.append(_resolve_path(Path.cwd()))

    seen: set[Path] = set()
    for start in starts:
        for ancestor in (start, *start.parents):
            if ancestor in seen:
                continue
            seen.add(ancestor)
            if (ancestor / ".git").exists():
                return ancestor
    return starts[0]


def _git_output(
    repo_root: Path,
    args: list[str],
    *,
    timeout: float = 2.0,
) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"git_failed:{exc.__class__.__name__}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or f"git_exit_{result.returncode}").strip()
        return None, _redact_log_text(detail, max_len=300)
    return result.stdout.strip(), None


def _build_repository_report(
    monitor_root: str | Path,
    *,
    repo_root: str | Path | None,
) -> dict[str, Any]:
    root = _find_repository_root(monitor_root, repo_root=repo_root)
    report: dict[str, Any] = {
        "root": _user_safe_display_path(root),
        "is_git_repo": False,
        "branch": None,
        "head": None,
        "head_full": None,
        "dirty": None,
        "tracked_changes": None,
        "error": None,
    }
    inside_work_tree, error = _git_output(root, ["rev-parse", "--is-inside-work-tree"])
    if error is not None or inside_work_tree != "true":
        report["error"] = error
        return report

    report["is_git_repo"] = True
    errors: list[str] = []
    head_full, error = _git_output(root, ["rev-parse", "HEAD"])
    if error is None:
        report["head_full"] = head_full
    else:
        errors.append(f"head:{error}")
    head, error = _git_output(root, ["rev-parse", "--short", "HEAD"])
    if error is None:
        report["head"] = head
    else:
        errors.append(f"head_short:{error}")
    branch, error = _git_output(root, ["rev-parse", "--abbrev-ref", "HEAD"])
    if error is None:
        report["branch"] = branch
    else:
        errors.append(f"branch:{error}")
    status, error = _git_output(root, ["status", "--porcelain", "--untracked-files=no"])
    if error is None:
        changes = [line for line in (status or "").splitlines() if line.strip()]
        report["dirty"] = bool(changes)
        report["tracked_changes"] = len(changes)
    else:
        errors.append(f"status:{error}")
    report["error"] = ";".join(errors) or None
    return report


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
    parts = stripped.split(None, 7)
    if (
        len(parts) == 8
        and _int_or_none(parts[2]) is not None
        and _int_or_none(parts[6]) is not None
    ):
        pid, ppid, etimes, stat_value, pcpu, pmem, rss, command = parts
        age_s = _int_or_none(etimes)
    else:
        parts = stripped.split(None, 6)
        if len(parts) == 7 and _int_or_none(parts[2]) is not None:
            pid, ppid, etimes, stat_value, pcpu, pmem, command = parts
            age_s = _int_or_none(etimes)
            rss = None
        elif len(parts) == 7 and _int_or_none(parts[5]) is not None:
            pid, ppid, stat_value, pcpu, pmem, rss, command = parts
            age_s = None
        else:
            parts = stripped.split(None, 5)
            if len(parts) != 6:
                return None
            pid, ppid, stat_value, pcpu, pmem, command = parts
            age_s = None
            rss = None
    command_key = _canonical_live_command(command)
    if not command_key:
        return None
    record = {
        "pid": _int_or_none(pid),
        "ppid": _int_or_none(ppid),
        "age_s": age_s,
        "stat": stat_value,
        "cpu_pct": _float_or_none(pcpu),
        "mem_pct": _float_or_none(pmem),
        "rss_kb": _int_or_none(rss) if rss is not None else None,
        "command": _redact_live_command_for_report(command, max_len=500),
        "command_key": _redact_live_command_for_report(command_key, max_len=500),
        "_match_key": command_key,
    }
    record.update(_live_command_context(command_key))
    return record


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


def _public_process_record(process: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in process.items()
        if key in PROCESS_REPORT_FIELDS and value is not None
    }


def _public_expected_record(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in item.items()
        if key in EXPECTED_PROCESS_FIELDS and value is not None
    }


def _load_smoke_config_file(
    path: str | Path,
    *,
    base_dir: Path | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = (base_dir or Path.cwd()) / config_path
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"config_read_failed:{exc.__class__.__name__}"
    try:
        import hjson

        parsed = hjson.loads(raw)
    except Exception as exc:
        return None, f"config_parse_failed:{exc.__class__.__name__}"
    if not isinstance(parsed, dict):
        return None, "config_root_not_object"
    return parsed, None


def _truthy_config_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value) and math.isfinite(float(value))
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return False


def _smoke_hsl_enabled_psides(config: dict[str, Any]) -> list[str]:
    enabled: list[str] = []
    bot = config.get("bot")
    if not isinstance(bot, dict):
        return enabled
    for pside in ("long", "short"):
        side_config = bot.get(pside)
        if not isinstance(side_config, dict):
            continue
        hsl_config = side_config.get("hsl")
        nested_enabled = (
            hsl_config.get("enabled") if isinstance(hsl_config, dict) else None
        )
        flat_enabled = side_config.get("hsl_enabled")
        if _truthy_config_flag(nested_enabled) or _truthy_config_flag(flat_enabled):
            enabled.append(pside)
    return enabled


def _normalize_smoke_hsl_signal_mode(value: Any) -> str:
    text = str(value if value not in (None, "") else "unified").strip().lower()
    aliases = {
        "per_side": "pside",
        "side": "pside",
        "symbol": "coin",
        "coins": "coin",
    }
    return aliases.get(text, text)


def _balance_override_active(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def _smoke_config_check_records(
    records: list[dict[str, Any]],
    *,
    config_base_dir: Path | None = None,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    checked_keys: set[str] = set()
    skipped = 0
    for record in records:
        config_path = record.get("config_path")
        command_key = record.get("command_key")
        if not config_path:
            skipped += 1
            continue
        dedupe_key = str(command_key or config_path)
        if dedupe_key in checked_keys:
            continue
        checked_keys.add(dedupe_key)
        config, error = _load_smoke_config_file(
            str(config_path),
            base_dir=config_base_dir,
        )
        if error is not None or config is None:
            issues.append(
                {
                    "severity": "warning",
                    "code": str(error or "config_unavailable"),
                    "account": record.get("account"),
                    "config_path": config_path,
                    "command_key": _redact_live_command_for_report(str(command_key))
                    if command_key is not None
                    else None,
                }
            )
            continue
        live = config.get("live") if isinstance(config.get("live"), dict) else {}
        signal_mode = _normalize_smoke_hsl_signal_mode(live.get("hsl_signal_mode"))
        config_balance_override = live.get("balance_override")
        launch_balance_override = record.get("balance_override")
        balance_override_source = None
        if _balance_override_active(launch_balance_override):
            balance_override_source = "argument"
        elif _balance_override_active(config_balance_override):
            balance_override_source = "live.balance_override"
        enabled_psides = _smoke_hsl_enabled_psides(config)
        if (
            enabled_psides
            and balance_override_source is not None
            and signal_mode in ACCOUNT_LEVEL_HSL_SIGNAL_MODES
        ):
            issues.append(
                {
                    "severity": "error",
                    "code": HSL_BALANCE_OVERRIDE_UNSAFE_CODE,
                    "message": (
                        "account-level HSL replay is unsafe with an active "
                        "balance override"
                    ),
                    "account": record.get("account"),
                    "config_path": config_path,
                    "command_key": _redact_live_command_for_report(str(command_key))
                    if command_key is not None
                    else None,
                    "hsl_signal_mode": signal_mode,
                    "enabled_psides": enabled_psides,
                    "balance_override_active": True,
                    "balance_override_source": balance_override_source,
                }
            )
    hard_failures = sum(1 for issue in issues if issue.get("severity") == "error")
    return {
        "enabled": True,
        "ok": hard_failures == 0,
        "checked": len(checked_keys),
        "skipped": skipped,
        "hard_failures": hard_failures,
        "issues": issues,
    }


def _build_live_config_checks(
    *,
    expected: list[dict[str, Any]],
    running: list[dict[str, Any]],
    config_base_dir: Path | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    records.extend(expected or [])
    records.extend(running or [])
    if not records:
        return {
            "enabled": False,
            "ok": True,
            "checked": 0,
            "skipped": 0,
            "hard_failures": 0,
            "issues": [],
        }
    return _smoke_config_check_records(records, config_base_dir=config_base_dir)


def parse_tmuxp_live_commands(config_path: str | Path | None) -> dict[str, Any]:
    """Return sanitized passivbot live commands from a tmuxp-style config."""
    config = _parse_tmuxp_live_commands(config_path)
    return {
        "path": _user_safe_display_path(config["path"])
        if config.get("path") is not None
        else None,
        "exists": bool(config.get("exists")),
        "error": config.get("error"),
        "expected": [
            _public_expected_record(item) for item in config.get("expected") or []
        ],
    }


def _build_process_report(
    *,
    include_processes: bool,
    supervisor_config: str | Path | None,
    process_command_match: str,
    config_base_dir: Path | None = None,
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
            "duplicate_configured_command_matches": [],
            "extra_passivbot_live_processes": [],
            "config_checks": {
                "enabled": False,
                "ok": True,
                "checked": 0,
                "skipped": 0,
                "hard_failures": 0,
                "issues": [],
            },
        }

    config = _parse_tmuxp_live_commands(supervisor_config)
    running_scan = _running_live_processes(command_match=process_command_match)
    running = running_scan["running"]
    expected = config["expected"]
    missing: list[dict[str, Any]] = []
    duplicate_matches: list[dict[str, Any]] = []
    expected_status: list[dict[str, Any]] = []
    matched_process_indexes: set[int] = set()
    for item in expected:
        match_key = item.get("_match_key")
        matches = [
            (index, process)
            for index, process in enumerate(running)
            if process.get("_match_key") == match_key
        ]
        if matches:
            matched_process_indexes.update(index for index, _process in matches)
            matched_processes = [
                _public_process_record(process) for _index, process in matches
            ]
            expected_row = _public_expected_record(item)
            expected_row["match_count"] = len(matches)
            expected_row["matched_processes"] = matched_processes
            expected_status.append(expected_row)
            if len(matches) > 1:
                duplicate_row = _public_expected_record(item)
                duplicate_row["match_count"] = len(matches)
                duplicate_row["matched_processes"] = matched_processes
                duplicate_matches.append(duplicate_row)
            continue
        missing_row = _public_expected_record(item)
        missing_row["match_count"] = 0
        missing.append(missing_row)
        expected_row = dict(missing_row)
        expected_row["matched_processes"] = []
        expected_status.append(expected_row)
    unexpected = [
        _public_process_record(process)
        for index, process in enumerate(running)
        if expected and index not in matched_process_indexes
    ]
    hard_failures = len(missing)
    if supervisor_config:
        hard_failures += len(duplicate_matches) + len(unexpected)
    if config.get("error"):
        hard_failures += 1
    elif supervisor_config and not expected:
        hard_failures += 1
        config["error"] = "no_expected_live_commands"
    if running_scan.get("scan_error"):
        hard_failures += 1
    config_checks = _build_live_config_checks(
        expected=expected,
        running=running,
        config_base_dir=config_base_dir,
    )
    hard_failures += int(config_checks.get("hard_failures") or 0)
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
        "classification_source": "local_process_table_command_match",
        "tmux_pane_ownership": "not_available_from_process_table",
        "expected": expected_status,
        "missing_expected": missing,
        "duplicate_configured_command_matches": duplicate_matches,
        "extra_passivbot_live_processes": unexpected,
        "unexpected_running": unexpected,
        "running": [_public_process_record(process) for process in running],
        "config_checks": config_checks,
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


def _startup_budget_projection(
    *,
    latest_ms: int | None,
    baseline_values: list[int],
) -> dict[str, int | str | None]:
    if latest_ms is None:
        return {
            "status": "unavailable",
            "latest_ms": None,
            "budget_ms": None,
            "baseline_samples": len(baseline_values),
            "usage_pct": None,
            "over_budget_by_ms": None,
            "source": "prior_p95_ms",
        }
    if not baseline_values:
        return {
            "status": "no_baseline",
            "latest_ms": latest_ms,
            "budget_ms": None,
            "baseline_samples": 0,
            "usage_pct": None,
            "over_budget_by_ms": None,
            "source": "prior_p95_ms",
        }
    budget_ms = _nearest_rank(baseline_values, 95.0)
    over_budget_by_ms = max(0, latest_ms - budget_ms)
    return {
        "status": "over_budget" if over_budget_by_ms > 0 else "within_budget",
        "latest_ms": latest_ms,
        "budget_ms": budget_ms,
        "baseline_samples": len(baseline_values),
        "usage_pct": _usage_pct(latest_ms, budget_ms),
        "over_budget_by_ms": over_budget_by_ms,
        "source": "prior_p95_ms",
    }


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
            elapsed_budget_values = (
                elapsed_values[:-1] if latest_elapsed is not None else elapsed_values
            )
            phase_budget_values = (
                phase_values[:-1] if latest_phase is not None else phase_values
            )
            phase_summaries[phase] = {
                "samples": len(window),
                "latest_ts": latest.get("ts"),
                "latest_elapsed_ms": latest_elapsed,
                "latest_since_previous_ms": latest_phase,
                "elapsed_baseline": elapsed_summary,
                "phase_baseline": phase_summary,
                "elapsed_budget": _startup_budget_projection(
                    latest_ms=latest_elapsed,
                    baseline_values=elapsed_budget_values,
                ),
                "phase_budget": _startup_budget_projection(
                    latest_ms=latest_phase,
                    baseline_values=phase_budget_values,
                ),
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


def _event_window_report(
    *,
    since_ms: int | None,
    until_ms: int | None,
    events_considered: int,
    events_skipped_before: int,
    events_skipped_after: int,
    invalid_window_ts: int,
    event_tail_lines: int = 0,
    event_tail_limited_files: int = 0,
    event_tail_skipped_lines: int = 0,
    event_tail_skipped_lines_exact: bool = True,
    event_tail_skipped_bytes: int = 0,
    event_tail_line_numbers_exact: bool = True,
    event_tail_methods: dict[str, int] | None = None,
    max_event_files_per_bot: int = 0,
    event_file_limit_groups: int = 0,
    event_files_before_limit: int = 0,
    event_files_skipped_by_limit: int = 0,
) -> dict[str, Any]:
    report = {
        "enabled": since_ms is not None or until_ms is not None,
        "since_ms": since_ms,
        "until_ms": until_ms,
        "events_considered": int(events_considered),
        "events_skipped_before": int(events_skipped_before),
        "events_skipped_after": int(events_skipped_after),
        "invalid_window_ts": int(invalid_window_ts),
    }
    if int(event_tail_lines) > 0:
        report["event_tail_lines"] = int(event_tail_lines)
        report["event_tail_limited_files"] = int(event_tail_limited_files)
        report["event_tail_skipped_lines"] = int(event_tail_skipped_lines)
        report["event_tail_skipped_lines_exact"] = bool(event_tail_skipped_lines_exact)
        report["event_tail_skipped_bytes"] = int(event_tail_skipped_bytes)
        report["event_tail_line_numbers_exact"] = bool(event_tail_line_numbers_exact)
        report["event_tail_methods"] = dict(sorted((event_tail_methods or {}).items()))
    if int(max_event_files_per_bot) > 0:
        report["max_event_files_per_bot"] = int(max_event_files_per_bot)
        report["event_file_limit_scope"] = "per_bot"
        report["event_file_limit_groups"] = int(event_file_limit_groups)
        report["event_files_before_limit"] = int(event_files_before_limit)
        report["event_files_skipped_by_limit"] = int(event_files_skipped_by_limit)
        report["event_file_limit_order"] = "current_then_recent_mtime"
    return report


RECOVERABLE_TIME_SYNC_REASON_CODES = frozenset(
    {
        "InvalidNonce",
        "exchange_timestamp_nonce_error",
    }
)


def _is_successful_time_sync_event(live_event: dict[str, Any]) -> bool:
    if str(live_event.get("event_type") or "") != EventTypes.EXCHANGE_TIME_SYNC:
        return False
    if str(live_event.get("status") or "").lower() != "succeeded":
        return False
    data = live_event.get("data")
    if isinstance(data, dict) and data.get("recovered") is False:
        return False
    return True


def _is_recoverable_time_sync_problem(live_event: dict[str, Any]) -> bool:
    if str(live_event.get("event_type") or "") != EventTypes.CYCLE_DEGRADED:
        return False
    reason_code = str(live_event.get("reason_code") or "")
    data = live_event.get("data")
    error_type = str(data.get("error_type") or "") if isinstance(data, dict) else ""
    if reason_code in RECOVERABLE_TIME_SYNC_REASON_CODES:
        return True
    if error_type in RECOVERABLE_TIME_SYNC_REASON_CODES:
        return True
    text = " ".join(str(item).lower() for item in (reason_code, error_type))
    return "nonce" in text or "timestamp" in text


def _recovery_summary_for_event(
    problem_record: dict[str, Any],
    recoveries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    live_event = problem_record["live_event"]
    if not bool(problem_record.get("base_hard")):
        return None
    if not _is_recoverable_time_sync_problem(live_event):
        return None
    bot_key = str(problem_record.get("bot_key") or "")
    problem_ts = _non_negative_int(problem_record["row"].get("ts"))
    if problem_ts is None:
        return None
    ids = _event_ids(live_event)
    cycle_id = ids.get("cycle_id")
    data = live_event.get("data")
    problem_error_type = (
        str(data.get("error_type") or "") if isinstance(data, dict) else ""
    )
    candidates: list[dict[str, Any]] = []
    for recovery in recoveries:
        if str(recovery.get("bot_key") or "") != bot_key:
            continue
        recovery_ts = _non_negative_int(recovery.get("ts"))
        if recovery_ts is None or recovery_ts < problem_ts:
            continue
        recovery_cycle_id = recovery.get("cycle_id")
        if cycle_id is not None:
            if recovery_cycle_id == cycle_id:
                pass
            elif recovery_cycle_id is None:
                if recovery_ts - problem_ts > 5 * 60 * 1000:
                    continue
                recovery_error_type = str(recovery.get("error_type") or "")
                if (
                    problem_error_type
                    and recovery_error_type
                    and recovery_error_type != problem_error_type
                ):
                    continue
            else:
                continue
        elif recovery_ts - problem_ts > 5 * 60 * 1000:
            continue
        candidates.append(recovery)
    if not candidates:
        return None
    recovery = min(candidates, key=lambda item: int(item.get("ts") or 0))
    return {
        "event_type": EventTypes.EXCHANGE_TIME_SYNC,
        "reason_code": recovery.get("reason_code"),
        "status": recovery.get("status"),
        "ts": recovery.get("ts"),
        "cycle_id": recovery.get("cycle_id"),
    }


def _problem_event_recovery_report(
    problem_records: list[dict[str, Any]],
    recoveries: list[dict[str, Any]],
) -> dict[str, Any]:
    recovered = 0
    recovered_hard = 0
    by_event_type: Counter[str] = Counter()
    for record in problem_records:
        recovery = _recovery_summary_for_event(record, recoveries)
        record["recovery"] = recovery
        record["recovered"] = recovery is not None
        if recovery is not None:
            recovered += 1
            if bool(record.get("base_hard")):
                recovered_hard += 1
            event_type = str(record["live_event"].get("event_type") or "")
            if event_type:
                by_event_type[event_type] += 1
    return {
        "total": int(recovered),
        "hard": int(recovered_hard),
        "event_types": dict(by_event_type.most_common()),
    }


def _monitor_issue(
    path: str | Path,
    line: int | None,
    severity: str,
    code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "line": line,
        "severity": str(severity),
        "code": str(code),
        "message": str(message),
    }


def _scan_events(
    root: str | Path,
    *,
    include_rotated: bool,
    max_problem_events: int,
    since_ms: int | None = None,
    until_ms: int | None = None,
    event_tail_lines: int = 0,
    max_event_files_per_bot: int = 0,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
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
            root, include_rotated=include_rotated
        )
        files = discovery.files
        file_discovery = discovery.to_dict()
    except FileNotFoundError as exc:
        files = []
        issues.append(
            _monitor_issue(str(root), None, "error", "path_not_found", str(exc))
        )
    if not files and not issues:
        issues.append(
            _monitor_issue(
                str(root),
                None,
                "error",
                "no_event_files",
                "no event NDJSON files found",
            )
        )
    report_ts_ms = utc_ms()
    records_total = 0
    live_events = 0
    legacy_events = 0
    missing_cycle_id = 0
    monitor_event_type_counts: Counter[str] = Counter()
    cycle_counts: Counter[str] = Counter()
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
    problem_records: list[dict[str, Any]] = []
    time_sync_recoveries: list[dict[str, Any]] = []
    invalid_rows = 0
    events_considered = 0
    events_skipped_before = 0
    events_skipped_after = 0
    invalid_window_ts = 0
    max_event_tail_lines = max(0, int(event_tail_lines))
    max_event_file_count_per_bot = max(0, int(max_event_files_per_bot))
    event_tail_limited_files = 0
    event_tail_skipped_lines = 0
    event_tail_skipped_lines_exact = True
    event_tail_skipped_bytes = 0
    event_tail_line_numbers_exact = True
    event_tail_methods: Counter[str] = Counter()
    event_files_before_limit = 0
    event_files_skipped_by_limit = 0
    event_file_limit_groups = 0
    startup_timing_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    remote_call_failure_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    remote_call_health_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    remote_call_timing_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    execution_health_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    fill_refresh_health_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    ema_readiness_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    ema_readiness_event_type_counts: Counter[str] = Counter()
    exchange_config_refresh_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    exchange_config_refresh_event_type_counts: Counter[str] = Counter()
    staged_readiness_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    staged_readiness_event_type_counts: Counter[str] = Counter()
    event_pipeline_health_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    event_pipeline_health_event_type_counts: Counter[str] = Counter()
    hsl_replay_health_groups: dict[str, dict[str, Any]] = {}
    hsl_replay_health_event_type_counts: Counter[str] = Counter()
    risk_event_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    risk_event_type_counts: Counter[str] = Counter()
    shutdown_event_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    shutdown_event_type_counts: Counter[str] = Counter()
    problem_event_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    if max_event_file_count_per_bot and files:
        event_files_before_limit = len(files)
        files, event_files_skipped_by_limit, event_file_limit_groups = (
            _limit_recent_event_files_per_bot(files, max_event_file_count_per_bot)
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
                    records_total += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        issues.append(
                            _monitor_issue(
                                path,
                                line_no,
                                "error",
                                "invalid_json",
                                str(exc),
                            )
                        )
                        invalid_rows += 1
                        continue
                    if not isinstance(row, dict):
                        issues.append(
                            _monitor_issue(
                                path,
                                line_no,
                                "error",
                                "invalid_record",
                                "event row is not a JSON object",
                            )
                        )
                        invalid_rows += 1
                        continue
                    live_event = _live_event_payload(row)
                    if live_event is None:
                        legacy_events += 1
                        continue
                    live_events += 1
                    event_type = live_event.get("event_type") or row.get("kind")
                    if event_type:
                        event_type = str(event_type)
                        monitor_event_type_counts[event_type] += 1
                        if row.get("kind") and str(row.get("kind")) != event_type:
                            issues.append(
                                _monitor_issue(
                                    path,
                                    line_no,
                                    "warning",
                                    "kind_event_type_mismatch",
                                    f"kind={row.get('kind')} event_type={event_type}",
                                )
                            )
                    else:
                        issues.append(
                            _monitor_issue(
                                path,
                                line_no,
                                "error",
                                "missing_event_type",
                                "live event is missing event_type",
                            )
                        )
                    if live_event.get("ids") is not None and not isinstance(
                        live_event.get("ids"), dict
                    ):
                        issues.append(
                            _monitor_issue(
                                path,
                                line_no,
                                "error",
                                "invalid_ids",
                                "live event ids field is not an object",
                            )
                        )
                    ids = _event_ids(live_event)
                    cycle_id = ids.get("cycle_id")
                    if cycle_id is None:
                        missing_cycle_id += 1
                    else:
                        cycle_counts[str(cycle_id)] += 1
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
                    if event_type:
                        bot["event_types"][str(event_type)] += 1
                    if _is_successful_time_sync_event(live_event):
                        time_sync_recoveries.append(
                            {
                                "bot_key": bot_key,
                                "ts": row.get("ts"),
                                "cycle_id": _event_ids(live_event).get("cycle_id"),
                                "reason_code": live_event.get("reason_code"),
                                "status": live_event.get("status"),
                                "error_type": (
                                    str(live_event.get("data", {}).get("error_type") or "")
                                    if isinstance(live_event.get("data"), dict)
                                    else ""
                                ),
                            }
                        )
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
                    if event_type in REMOTE_CALL_TIMING_EVENT_TYPES:
                        _merge_remote_call_health_group(
                            remote_call_health_groups,
                            _remote_call_health_group(
                                bot_key=bot_key,
                                row=row,
                                live_event=live_event,
                                path=path,
                                line_no=line_no,
                            ),
                        )
                        remote_call_timing_group = _remote_call_timing_group(
                            bot_key=bot_key,
                            row=row,
                            live_event=live_event,
                            path=path,
                            line_no=line_no,
                        )
                        if remote_call_timing_group is not None:
                            _merge_remote_call_timing_group(
                                remote_call_timing_groups,
                                remote_call_timing_group,
                            )
                    if event_type in EXECUTION_HEALTH_EVENT_TYPES:
                        _merge_execution_health_group(
                            execution_health_groups,
                            _execution_health_group(
                                bot_key=bot_key,
                                row=row,
                                live_event=live_event,
                                path=path,
                                line_no=line_no,
                            ),
                        )
                    if event_type == EventTypes.FILLS_REFRESH_SUMMARY:
                        _merge_fill_refresh_health_group(
                            fill_refresh_health_groups,
                            _fill_refresh_health_group(
                                bot_key=bot_key,
                                row=row,
                                live_event=live_event,
                                path=path,
                                line_no=line_no,
                            ),
                        )
                    if event_type == EventTypes.EMA_UNAVAILABLE:
                        ema_readiness_event_type_counts[str(event_type)] += 1
                        _merge_ema_readiness_group(
                            ema_readiness_groups,
                            _ema_readiness_group(
                                bot_key=bot_key,
                                row=row,
                                live_event=live_event,
                                path=path,
                                line_no=line_no,
                            ),
                        )
                    if event_type == EventTypes.EXCHANGE_CONFIG_REFRESH:
                        exchange_config_refresh_event_type_counts[str(event_type)] += 1
                        _merge_exchange_config_refresh_group(
                            exchange_config_refresh_groups,
                            _exchange_config_refresh_group(
                                bot_key=bot_key,
                                row=row,
                                live_event=live_event,
                                path=path,
                                line_no=line_no,
                            ),
                        )
                    if _staged_readiness_event(live_event):
                        staged_readiness_event_type_counts[str(event_type)] += 1
                        _merge_staged_readiness_group(
                            staged_readiness_groups,
                            _staged_readiness_group(
                                bot_key=bot_key,
                                row=row,
                                live_event=live_event,
                                path=path,
                                line_no=line_no,
                            ),
                        )
                    if event_type == EventTypes.HEALTH_SUMMARY:
                        event_pipeline_health_group = _event_pipeline_health_group(
                            bot_key=bot_key,
                            row=row,
                            live_event=live_event,
                            path=path,
                            line_no=line_no,
                        )
                        if event_pipeline_health_group is not None:
                            event_pipeline_health_event_type_counts[str(event_type)] += 1
                            _merge_event_pipeline_health_group(
                                event_pipeline_health_groups,
                                event_pipeline_health_group,
                            )
                    if event_type in HSL_REPLAY_EVENT_TYPES:
                        hsl_replay_health_event_type_counts[str(event_type)] += 1
                        _merge_hsl_replay_group(
                            hsl_replay_health_groups,
                            bot_key=bot_key,
                            row=row,
                            live_event=live_event,
                            path=path,
                            line_no=line_no,
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
                    if event_type in SHUTDOWN_EVENT_TYPES:
                        shutdown_event_type_counts[str(event_type)] += 1
                        _merge_shutdown_event_group(
                            shutdown_event_groups,
                            _shutdown_event_group(
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
                    if is_problem_event(live_event):
                        bot["problem_events"] += 1
                        problem_records.append(
                            {
                                "bot_key": bot_key,
                                "row": row,
                                "live_event": live_event,
                                "path": path,
                                "line_no": line_no,
                                "base_hard": is_hard_problem_event(live_event),
                            }
                        )
                    startup_timing = _startup_timing_record(
                        row=row,
                        live_event=live_event,
                        path=path,
                        line_no=line_no,
                    )
                    if startup_timing is not None:
                        startup_timing_records[bot_key].append(startup_timing)
        except OSError as exc:
            issues.append(
                _monitor_issue(path, None, "error", "read_failed", str(exc))
            )
            invalid_rows += 1

    recovered_problem_events = _problem_event_recovery_report(
        problem_records, time_sync_recoveries
    )
    for record in problem_records:
        hard = bool(record.get("base_hard")) and not bool(record.get("recovered"))
        bot_key = str(record.get("bot_key") or "unknown")
        if hard:
            bots[bot_key]["hard_problem_events"] += 1
        _merge_problem_event_group(
            problem_event_groups,
            _problem_event_group(
                bot_key=bot_key,
                row=record["row"],
                live_event=record["live_event"],
                path=record["path"],
                line_no=int(record["line_no"]),
                hard=hard,
                recovered=bool(record.get("recovered")),
                recovery=record.get("recovery"),
            ),
        )
        problem_event = _compact_problem_event(
            path=record["path"],
            line_no=int(record["line_no"]),
            row=record["row"],
            live_event=record["live_event"],
        )
        problem_event["hard"] = hard
        if bool(record.get("recovered")):
            problem_event["recovered"] = True
            problem_event["recovery"] = record.get("recovery")
        problem_events.append(problem_event)

    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")
    return {
        "monitor": {
            "root": str(Path(root).expanduser()),
            "include_rotated": bool(include_rotated),
            "files_scanned": len(files),
            "records_total": records_total,
            "live_events": live_events,
            "legacy_events": legacy_events,
            "missing_cycle_id": missing_cycle_id,
            "issues": issues,
            "error_count": error_count,
            "warning_count": warning_count,
            "file_discovery": file_discovery,
            "event_types": dict(sorted(monitor_event_type_counts.items())),
            "cycle_ids_sample": [
                {"cycle_id": key, "events": value}
                for key, value in cycle_counts.most_common(20)
            ],
        },
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
        "recovered_problem_events": recovered_problem_events,
        "problem_event_count": sum(int(value["problem_events"]) for value in bots.values()),
        "hard_problem_event_count": sum(
            int(value["hard_problem_events"]) for value in bots.values()
        ),
        "startup_timings": _summarize_startup_timings(startup_timing_records),
        "remote_call_failures": _summarize_remote_call_failures(
            remote_call_failure_groups
        ),
        "remote_call_health": _summarize_remote_call_health(remote_call_health_groups),
        "account_critical_remote_call_health": _summarize_remote_call_health(
            _account_critical_remote_call_health_groups(remote_call_health_groups)
        ),
        "remote_call_timings": _summarize_remote_call_timings(remote_call_timing_groups),
        "execution_health": _summarize_execution_health(execution_health_groups),
        "fill_refresh_health": _summarize_fill_refresh_health(
            fill_refresh_health_groups
        ),
        "ema_readiness_health": _summarize_ema_readiness_health(
            ema_readiness_groups,
            ema_readiness_event_type_counts,
        ),
        "exchange_config_refresh_health": _summarize_exchange_config_refresh_health(
            exchange_config_refresh_groups,
            exchange_config_refresh_event_type_counts,
        ),
        "staged_readiness_health": _summarize_staged_readiness_health(
            staged_readiness_groups,
            staged_readiness_event_type_counts,
        ),
        "event_pipeline_health": _summarize_event_pipeline_health(
            event_pipeline_health_groups,
            event_pipeline_health_event_type_counts,
        ),
        "hsl_replay_health": _summarize_hsl_replay_health(
            hsl_replay_health_groups,
            hsl_replay_health_event_type_counts,
            report_ts_ms=report_ts_ms,
        ),
        "risk_events": _summarize_risk_events(
            risk_event_groups,
            risk_event_type_counts,
        ),
        "shutdown_events": _summarize_shutdown_events(
            shutdown_event_groups,
            shutdown_event_type_counts,
        ),
        "event_window": _event_window_report(
            since_ms=since_ms,
            until_ms=until_ms,
            events_considered=events_considered,
            events_skipped_before=events_skipped_before,
            events_skipped_after=events_skipped_after,
            invalid_window_ts=invalid_window_ts,
            event_tail_lines=max_event_tail_lines,
            event_tail_limited_files=event_tail_limited_files,
            event_tail_skipped_lines=event_tail_skipped_lines,
            event_tail_skipped_lines_exact=event_tail_skipped_lines_exact,
            event_tail_skipped_bytes=event_tail_skipped_bytes,
            event_tail_line_numbers_exact=event_tail_line_numbers_exact,
            event_tail_methods=dict(event_tail_methods),
            max_event_files_per_bot=max_event_file_count_per_bot,
            event_file_limit_groups=event_file_limit_groups,
            event_files_before_limit=event_files_before_limit or len(files),
            event_files_skipped_by_limit=event_files_skipped_by_limit,
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
    dropped_unparsed_attention_matches: int,
    dropped_unparsed_hard_matches: int,
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
        "dropped_unparsed_attention_matches": int(
            dropped_unparsed_attention_matches
        ),
        "dropped_unparsed_hard_matches": int(dropped_unparsed_hard_matches),
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
        dropped_unparsed_attention_matches=0,
        dropped_unparsed_hard_matches=0,
    )
    if root is None:
        return {
            "root": None,
            "max_files": max(0, int(max_files)),
            "tail_lines": max(0, int(tail_lines)),
            "max_matches": max(0, int(max_matches)),
            "files_scanned": 0,
            "hard_matches": 0,
            "attention_matches": 0,
            "risk_attention_matches": 0,
            "risk_hard_matches": 0,
            "non_risk_attention_matches": 0,
            "non_risk_hard_matches": 0,
            "window": window_report,
            "matches": [],
            "dropped_unparsed_attention_matches": 0,
            "dropped_unparsed_hard_matches": 0,
        }
    files = _recent_log_files(root, max_files=max_files)
    matches: list[dict[str, Any]] = []
    hard_matches = 0
    attention_matches = 0
    risk_attention_matches = 0
    risk_hard_matches = 0
    non_risk_attention_matches = 0
    non_risk_hard_matches = 0
    lines_considered = 0
    lines_skipped_before = 0
    lines_skipped_after = 0
    unparsed_ts = 0
    lines_skipped_unparsed = 0
    dropped_unparsed_attention_matches = 0
    dropped_unparsed_hard_matches = 0
    for path in files:
        log_context_ts_ms: int | None = None
        log_context_line_no: int | None = None
        log_context_text: str | None = None
        for line_no, line in _tail_lines(path, max_lines=tail_lines):
            line_ts = _parse_log_line_ts_ms(line)
            if line_ts is not None:
                log_context_ts_ms = line_ts
                log_context_line_no = int(line_no)
                log_context_text = line
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
                        if log_context_ts_ms is None and attention:
                            dropped_unparsed_attention_matches += 1
                            if HARD_LOG_PATTERN.search(line):
                                dropped_unparsed_hard_matches += 1
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
            risk_match = bool(RISK_LOG_PATTERN.search(line))
            if risk_match:
                risk_attention_matches += 1
            else:
                non_risk_attention_matches += 1
            if hard:
                hard_matches += 1
                if risk_match:
                    risk_hard_matches += 1
                else:
                    non_risk_hard_matches += 1
            if len(matches) < max(0, int(max_matches)):
                match = {
                    "path": str(path),
                    "line": int(line_no),
                    "hard": hard,
                    "category": "risk" if risk_match else "general",
                    "text": _redact_log_text(line, max_len=500),
                }
                if line_ts is not None:
                    match["ts"] = line_ts
                elif log_context_ts_ms is not None:
                    match["context_ts"] = int(log_context_ts_ms)
                    if log_context_line_no is not None:
                        match["context_line"] = int(log_context_line_no)
                    if log_context_text:
                        match["context_text"] = _redact_log_text(
                            log_context_text, max_len=500
                        )
                matches.append(match)
    return {
        "root": str(Path(root).expanduser()),
        "max_files": max(0, int(max_files)),
        "tail_lines": max(0, int(tail_lines)),
        "max_matches": max(0, int(max_matches)),
        "files_scanned": len(files),
        "hard_matches": hard_matches,
        "attention_matches": attention_matches,
        "risk_attention_matches": risk_attention_matches,
        "risk_hard_matches": risk_hard_matches,
        "non_risk_attention_matches": non_risk_attention_matches,
        "non_risk_hard_matches": non_risk_hard_matches,
        "window": _log_window_report(
            since_ms=since_ms,
            until_ms=until_ms,
            lines_considered=lines_considered,
            lines_skipped_before=lines_skipped_before,
            lines_skipped_after=lines_skipped_after,
            unparsed_ts=unparsed_ts,
            unparsed_policy=unparsed_policy,
            lines_skipped_unparsed=lines_skipped_unparsed,
            dropped_unparsed_attention_matches=dropped_unparsed_attention_matches,
            dropped_unparsed_hard_matches=dropped_unparsed_hard_matches,
        ),
        "matches": matches,
        "dropped_unparsed_attention_matches": dropped_unparsed_attention_matches,
        "dropped_unparsed_hard_matches": dropped_unparsed_hard_matches,
    }


def build_live_smoke_report(
    monitor_root: str | Path = "monitor",
    *,
    logs_root: str | Path | None = "logs",
    repo_root: str | Path | None = None,
    include_processes: bool = False,
    supervisor_config: str | Path | None = None,
    process_command_match: str = DEFAULT_PROCESS_MATCH,
    include_rotated: bool = False,
    since_ms: int | None = None,
    until_ms: int | None = None,
    max_problem_events: int = 50,
    max_log_files: int = 8,
    event_tail_lines: int = 0,
    max_event_files_per_bot: int = 0,
    log_tail_lines: int = 300,
    max_log_matches: int = 50,
    log_window_unparsed_policy: str = DEFAULT_LOG_WINDOW_UNPARSED_POLICY,
) -> dict[str, Any]:
    repository_root = _find_repository_root(monitor_root, repo_root=repo_root)
    event_scan = _scan_events(
        monitor_root,
        include_rotated=include_rotated,
        max_problem_events=max_problem_events,
        since_ms=since_ms,
        until_ms=until_ms,
        event_tail_lines=event_tail_lines,
        max_event_files_per_bot=max_event_files_per_bot,
    )
    event_report = event_scan["monitor"]
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
        config_base_dir=repository_root,
    )
    repository_report = _build_repository_report(monitor_root, repo_root=repository_root)
    hard_failures = (
        int(event_report["error_count"])
        + int(event_scan["invalid_rows"])
        + int(event_scan["hard_problem_event_count"])
        + int(log_scan["hard_matches"])
        + int(process_report["hard_failures"])
    )
    hsl_replay_active_bots = int(
        event_scan["hsl_replay_health"].get("active_bots") or 0
    )
    hsl_replay_failed_attention_bots = int(
        event_scan["hsl_replay_health"].get("failed_attention_bots") or 0
    )
    attention_count = (
        int(event_scan["problem_event_count"])
        + int(log_scan["attention_matches"])
        + int(log_scan.get("dropped_unparsed_attention_matches", 0))
        + hsl_replay_active_bots
        + hsl_replay_failed_attention_bots
    )
    hard_failure_sources = {
        "monitor_errors": int(event_report["error_count"]),
        "invalid_event_rows": int(event_scan["invalid_rows"]),
        "hard_problem_events": int(event_scan["hard_problem_event_count"]),
        "log_hard_matches": int(log_scan["hard_matches"]),
        "process_hard_failures": int(process_report["hard_failures"]),
        "total": int(hard_failures),
    }
    attention_sources = {
        "problem_events": int(event_scan["problem_event_count"]),
        "log_attention_matches": int(log_scan["attention_matches"]),
        "dropped_unparsed_attention_matches": int(
            log_scan.get("dropped_unparsed_attention_matches", 0)
        ),
        "total": int(attention_count),
    }
    if hsl_replay_active_bots:
        attention_sources["hsl_replay_active_bots"] = hsl_replay_active_bots
    if hsl_replay_failed_attention_bots:
        attention_sources["hsl_replay_failed_bots"] = hsl_replay_failed_attention_bots
    return {
        "ok": hard_failures == 0,
        "attention": attention_count > 0,
        "hard_failures": hard_failures,
        "attention_count": attention_count,
        "hard_failure_sources": hard_failure_sources,
        "attention_sources": attention_sources,
        "monitor": {
            "root": event_report.get("root"),
            "include_rotated": event_report.get("include_rotated"),
            "files_scanned": event_report.get("files_scanned"),
            "records_total": event_report.get("records_total"),
            "live_events": event_report.get("live_events"),
            "legacy_events": event_report.get("legacy_events"),
            "error_count": event_report.get("error_count"),
            "warning_count": event_report.get("warning_count"),
            "file_discovery": event_report.get("file_discovery"),
            "event_types": event_report["event_types"],
            "cycle_ids_sample": event_report["cycle_ids_sample"],
            "issues": event_report["issues"],
        },
        "bots": event_scan["bots"],
        "startup_timings": event_scan["startup_timings"],
        "remote_call_failures": event_scan["remote_call_failures"],
        "remote_call_health": event_scan["remote_call_health"],
        "account_critical_remote_call_health": event_scan[
            "account_critical_remote_call_health"
        ],
        "remote_call_timings": event_scan["remote_call_timings"],
        "execution_health": event_scan["execution_health"],
        "fill_refresh_health": event_scan["fill_refresh_health"],
        "ema_readiness_health": event_scan["ema_readiness_health"],
        "exchange_config_refresh_health": event_scan[
            "exchange_config_refresh_health"
        ],
        "staged_readiness_health": event_scan["staged_readiness_health"],
        "event_pipeline_health": event_scan["event_pipeline_health"],
        "hsl_replay_health": event_scan["hsl_replay_health"],
        "risk_events": event_scan["risk_events"],
        "shutdown_events": event_scan["shutdown_events"],
        "event_window": event_scan["event_window"],
        "problem_events": event_scan["problem_events"],
        "problem_event_groups": event_scan["problem_event_groups"],
        "recovered_problem_events": event_scan["recovered_problem_events"],
        "problem_event_count": event_scan["problem_event_count"],
        "hard_problem_event_count": event_scan["hard_problem_event_count"],
        "logs": log_scan,
        "processes": process_report,
        "repository": repository_report,
    }


def _summary_limited_groups(
    summary: dict[str, Any],
    *,
    limit: int,
    shareable_latest_data_keys: frozenset[str] | None = None,
) -> dict[str, Any]:
    groups = summary.get("groups")
    if not isinstance(groups, list):
        groups = []
    limit = max(0, int(limit))
    limited_groups = groups[:limit]
    if shareable_latest_data_keys is not None:
        limited_groups = [
            _shareable_summary_group(group, latest_data_keys=shareable_latest_data_keys)
            for group in limited_groups
        ]
    return {
        key: value
        for key, value in {
            "total": summary.get("total"),
            "succeeded": summary.get("succeeded"),
            "failed": summary.get("failed"),
            "throttled": summary.get("throttled"),
            "rejected": summary.get("rejected"),
            "ambiguous": summary.get("ambiguous"),
            "confirmation_timeout": summary.get("confirmation_timeout"),
            "failure_pct": summary.get("failure_pct"),
            "throttled_pct": summary.get("throttled_pct"),
            "event_types": summary.get("event_types"),
            "statuses": summary.get("statuses"),
            "outcomes": summary.get("outcomes"),
            "failed_reason_codes": summary.get("failed_reason_codes") or None,
            "failed_error_types": summary.get("failed_error_types") or None,
            "failed_kinds": summary.get("failed_kinds") or None,
            "failed_surfaces": summary.get("failed_surfaces") or None,
            "bots": summary.get("bots"),
            "active_bots": summary.get("active_bots"),
            "stale_active_bots": summary.get("stale_active_bots"),
            "long_running_active_bots": summary.get("long_running_active_bots"),
            "completed_bots": summary.get("completed_bots"),
            "failed_bots": summary.get("failed_bots"),
            "latest_failed_bots": summary.get("latest_failed_bots"),
            "failed_attention_bots": summary.get("failed_attention_bots"),
            "recovered_groups": summary.get("recovered_groups"),
            "latest_candidate_unavailable_total": summary.get(
                "latest_candidate_unavailable_total"
            ),
            "latest_unavailable_total": summary.get("latest_unavailable_total"),
            "latest_optional_drop_total": summary.get("latest_optional_drop_total"),
            "latest_candidate_reason_counts": summary.get(
                "latest_candidate_reason_counts"
            )
            or None,
            "latest_unavailable_reason_counts": summary.get(
                "latest_unavailable_reason_counts"
            )
            or None,
            "latest_candidate_reason_symbols": summary.get(
                "latest_candidate_reason_symbols"
            )
            or None,
            "latest_unavailable_reason_symbols": summary.get(
                "latest_unavailable_reason_symbols"
            )
            or None,
            "latest_candidate_error_type_counts": summary.get(
                "latest_candidate_error_type_counts"
            )
            or None,
            "latest_missing_surface_total": summary.get("latest_missing_surface_total"),
            "latest_invalid_surface_total": summary.get("latest_invalid_surface_total"),
            "latest_missing_surfaces": summary.get("latest_missing_surfaces") or None,
            "latest_invalid_surfaces": summary.get("latest_invalid_surfaces") or None,
            "latest_queue_depth_total": summary.get("latest_queue_depth_total"),
            "latest_queue_unfinished_total": summary.get(
                "latest_queue_unfinished_total"
            ),
            "latest_dropped_total": summary.get("latest_dropped_total"),
            "latest_sink_error_total": summary.get("latest_sink_error_total"),
            "latest_degraded_total": summary.get("latest_degraded_total"),
            "latest_worker_not_alive_count": summary.get(
                "latest_worker_not_alive_count"
            ),
            "latest_stopping_count": summary.get("latest_stopping_count"),
            "hsl_flat_finalization_anchors": summary.get(
                "hsl_flat_finalization_anchors"
            ),
            "hsl_status": _shareable_hsl_status(summary.get("hsl_status")),
            "hsl_raw_red_pending": _shareable_hsl_raw_red_pending(
                summary.get("hsl_raw_red_pending")
            ),
            "groups_truncated": bool(summary.get("groups_truncated"))
            or len(groups) > limit,
            "groups": limited_groups,
        }.items()
        if value is not None
    }


def _shareable_summary_group(
    group: Any,
    *,
    latest_data_keys: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(group, dict):
        return {}
    out = dict(group)
    latest_data = out.get("latest_data")
    if isinstance(latest_data, dict):
        safe_latest_data = {
            key: value
            for key, value in latest_data.items()
            if key in latest_data_keys and value not in (None, {}, [])
        }
        if safe_latest_data:
            out["latest_data"] = safe_latest_data
        else:
            out.pop("latest_data", None)
    return {key: value for key, value in out.items() if value not in (None, {}, [])}


def _summary_startup_timings(
    startup_timings: Any,
    *,
    limit: int,
) -> dict[str, Any]:
    rows = startup_timings if isinstance(startup_timings, list) else []
    limit = max(0, int(limit))
    return {
        "bots": len(rows),
        "groups_truncated": len(rows) > limit,
        "groups": rows[:limit],
    }


def _brief_startup_timings(startup_timings: Any) -> dict[str, Any]:
    rows = startup_timings if isinstance(startup_timings, list) else []
    phase_count = 0
    over_budget_phases = 0
    max_latest_elapsed_ms: int | None = None
    max_latest_phase_ms: int | None = None
    max_startup_elapsed_ms: int | None = None
    startup_phase_bots = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        phases = row.get("phases")
        if not isinstance(phases, dict):
            continue
        for phase, phase_data in phases.items():
            if not isinstance(phase_data, dict):
                continue
            phase_count += 1
            latest_elapsed = _non_negative_int(phase_data.get("latest_elapsed_ms"))
            latest_phase = _non_negative_int(phase_data.get("latest_since_previous_ms"))
            if latest_elapsed is not None:
                max_latest_elapsed_ms = max(
                    latest_elapsed,
                    max_latest_elapsed_ms if max_latest_elapsed_ms is not None else 0,
                )
                if str(phase) == "startup":
                    startup_phase_bots += 1
                    max_startup_elapsed_ms = max(
                        latest_elapsed,
                        max_startup_elapsed_ms
                        if max_startup_elapsed_ms is not None
                        else 0,
                    )
            if latest_phase is not None:
                max_latest_phase_ms = max(
                    latest_phase,
                    max_latest_phase_ms if max_latest_phase_ms is not None else 0,
                )
            elapsed_budget = phase_data.get("elapsed_budget")
            phase_budget = phase_data.get("phase_budget")
            if (
                isinstance(elapsed_budget, dict)
                and elapsed_budget.get("status") == "over_budget"
            ) or (
                isinstance(phase_budget, dict)
                and phase_budget.get("status") == "over_budget"
            ):
                over_budget_phases += 1
    return {
        key: value
        for key, value in {
            "bots": len(rows),
            "phases": phase_count,
            "over_budget_phases": over_budget_phases,
            "startup_phase_bots": startup_phase_bots,
            "max_latest_elapsed_ms": max_latest_elapsed_ms,
            "max_latest_phase_ms": max_latest_phase_ms,
            "max_startup_elapsed_ms": max_startup_elapsed_ms,
        }.items()
        if value is not None
    }


def summarize_live_smoke_report(
    report: dict[str, Any],
    *,
    max_groups: int = SMOKE_REPORT_SUMMARY_GROUP_LIMIT,
) -> dict[str, Any]:
    """Project a full smoke report into concise operator/debugging evidence."""

    max_groups = max(0, int(max_groups))
    logs = report.get("logs") if isinstance(report.get("logs"), dict) else {}
    matches = logs.get("matches") if isinstance(logs.get("matches"), list) else []
    processes = (
        report.get("processes") if isinstance(report.get("processes"), dict) else {}
    )
    process_config_checks = (
        processes.get("config_checks")
        if isinstance(processes.get("config_checks"), dict)
        else {}
    )
    repository = (
        report.get("repository") if isinstance(report.get("repository"), dict) else {}
    )
    monitor = report.get("monitor") if isinstance(report.get("monitor"), dict) else {}
    problem_groups = (
        report.get("problem_event_groups")
        if isinstance(report.get("problem_event_groups"), dict)
        else {}
    )
    problem_group_rows = (
        problem_groups.get("groups")
        if isinstance(problem_groups.get("groups"), list)
        else []
    )
    risk_events = (
        report.get("risk_events") if isinstance(report.get("risk_events"), dict) else {}
    )
    ema_readiness_health = (
        report.get("ema_readiness_health")
        if isinstance(report.get("ema_readiness_health"), dict)
        else {}
    )
    fill_refresh_health = (
        report.get("fill_refresh_health")
        if isinstance(report.get("fill_refresh_health"), dict)
        else {}
    )
    execution_health = (
        report.get("execution_health")
        if isinstance(report.get("execution_health"), dict)
        else {}
    )
    exchange_config_refresh_health = (
        report.get("exchange_config_refresh_health")
        if isinstance(report.get("exchange_config_refresh_health"), dict)
        else {}
    )
    staged_readiness_health = (
        report.get("staged_readiness_health")
        if isinstance(report.get("staged_readiness_health"), dict)
        else {}
    )
    event_pipeline_health = (
        report.get("event_pipeline_health")
        if isinstance(report.get("event_pipeline_health"), dict)
        else {}
    )
    hsl_replay_health = (
        report.get("hsl_replay_health")
        if isinstance(report.get("hsl_replay_health"), dict)
        else {}
    )
    shutdown_events = (
        report.get("shutdown_events")
        if isinstance(report.get("shutdown_events"), dict)
        else {}
    )

    return {
        "ok": bool(report.get("ok", False)),
        "attention": bool(report.get("attention", False)),
        "hard_failures": int(report.get("hard_failures") or 0),
        "attention_count": int(report.get("attention_count") or 0),
        "hard_failure_sources": dict(report.get("hard_failure_sources") or {}),
        "attention_sources": dict(report.get("attention_sources") or {}),
        "recovered_problem_events": dict(report.get("recovered_problem_events") or {}),
        "repository": {
            key: repository.get(key)
            for key in (
                "branch",
                "head",
                "dirty",
                "tracked_changes",
                "error",
            )
            if key in repository
        },
        "monitor": {
            key: monitor.get(key)
            for key in (
                "root",
                "files_scanned",
                "records_total",
                "live_events",
                "legacy_events",
                "error_count",
                "warning_count",
                "file_discovery",
            )
            if key in monitor
        },
        "event_window": report.get("event_window"),
        "processes": {
            key: processes.get(key)
            for key in (
                "enabled",
                "ok",
                "hard_failures",
                "expected_total",
                "matched_expected",
                "running_live_total",
                "classification_source",
                "tmux_pane_ownership",
                "scan_error",
            )
            if key in processes
        }
        | {
            "missing_expected_count": len(processes.get("missing_expected") or []),
            "duplicate_configured_command_matches_count": len(
                processes.get("duplicate_configured_command_matches") or []
            ),
            "extra_passivbot_live_processes_count": len(
                processes.get("extra_passivbot_live_processes") or []
            ),
            "unexpected_running_count": len(processes.get("unexpected_running") or []),
            "missing_expected": (processes.get("missing_expected") or [])[:max_groups],
            "duplicate_configured_command_matches": (
                processes.get("duplicate_configured_command_matches") or []
            )[:max_groups],
            "extra_passivbot_live_processes": (
                processes.get("extra_passivbot_live_processes") or []
            )[:max_groups],
            "unexpected_running": (processes.get("unexpected_running") or [])[:max_groups],
            "config_checks": {
                key: process_config_checks.get(key)
                for key in (
                    "enabled",
                    "ok",
                    "checked",
                    "skipped",
                    "hard_failures",
                )
                if key in process_config_checks
            }
            | {
                "issues": (process_config_checks.get("issues") or [])[:max_groups],
                "issues_truncated": len(process_config_checks.get("issues") or [])
                > max_groups,
            },
        },
        "logs": {
            "root": logs.get("root"),
            "max_files": int(logs.get("max_files") or 0),
            "tail_lines": int(logs.get("tail_lines") or 0),
            "max_matches": int(logs.get("max_matches") or 0),
            "files_scanned": int(logs.get("files_scanned") or 0),
            "hard_matches": int(logs.get("hard_matches") or 0),
            "attention_matches": int(logs.get("attention_matches") or 0),
            "risk_attention_matches": int(logs.get("risk_attention_matches") or 0),
            "risk_hard_matches": int(logs.get("risk_hard_matches") or 0),
            "non_risk_attention_matches": int(
                logs.get("non_risk_attention_matches") or 0
            ),
            "non_risk_hard_matches": int(logs.get("non_risk_hard_matches") or 0),
            "dropped_unparsed_attention_matches": int(
                logs.get("dropped_unparsed_attention_matches") or 0
            ),
            "dropped_unparsed_hard_matches": int(
                logs.get("dropped_unparsed_hard_matches") or 0
            ),
            "matches_truncated": len(matches) > max_groups,
            "matches": matches[:max_groups],
            "window": logs.get("window"),
        },
        "problem_events": {
            "total": int(report.get("problem_event_count") or 0),
            "hard": int(report.get("hard_problem_event_count") or 0),
            "groups_truncated": bool(problem_groups.get("groups_truncated"))
            or len(problem_group_rows) > max_groups,
            "groups": problem_group_rows[:max_groups],
        },
        "remote_calls": _summary_limited_groups(
            report.get("remote_call_health") or {},
            limit=max_groups,
        ),
        "account_critical_remote_calls": _summary_limited_groups(
            report.get("account_critical_remote_call_health") or {},
            limit=max_groups,
        ),
        "startup_timings": _summary_startup_timings(
            report.get("startup_timings"),
            limit=max_groups,
        ),
        "execution_health": _summary_limited_groups(
            execution_health,
            limit=max_groups,
        ),
        "fill_refresh_health": _summary_limited_groups(
            fill_refresh_health,
            limit=max_groups,
        ),
        "ema_readiness_health": _summary_limited_groups(
            ema_readiness_health,
            limit=max_groups,
        ),
        "exchange_config_refresh_health": _summary_limited_groups(
            exchange_config_refresh_health,
            limit=max_groups,
        ),
        "staged_readiness_health": _summary_limited_groups(
            staged_readiness_health,
            limit=max_groups,
        ),
        "event_pipeline_health": _summary_limited_groups(
            event_pipeline_health,
            limit=max_groups,
        ),
        "hsl_replay_health": _summary_limited_groups(
            hsl_replay_health,
            limit=max_groups,
        ),
        "risk_events": _summary_limited_groups(
            risk_events,
            limit=max_groups,
            shareable_latest_data_keys=SHAREABLE_RISK_LATEST_DATA_KEYS,
        ),
        "shutdown_events": _summary_limited_groups(
            shutdown_events,
            limit=max_groups,
        ),
    }


def _count_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _brief_remote_call_slowest(summary: dict[str, Any]) -> list[dict[str, Any]]:
    groups = summary.get("groups")
    if not isinstance(groups, list):
        return []
    rows: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        elapsed = group.get("elapsed_ms")
        elapsed_map = elapsed if isinstance(elapsed, dict) else {}
        max_ms = _non_negative_int(elapsed_map.get("max_ms"))
        p95_ms = _non_negative_int(elapsed_map.get("p95_ms"))
        latest_elapsed_ms = _non_negative_int(group.get("latest_elapsed_ms"))
        sort_ms = max(
            int(max_ms or 0),
            int(p95_ms or 0),
            int(latest_elapsed_ms or 0),
        )
        if sort_ms <= 0:
            continue
        latest_symbol = group.get("latest_symbol")
        row = {
            "bot": _redact_log_text(str(group.get("bot")), max_len=120)
            if group.get("bot") not in (None, "")
            else None,
            "kind": _redact_log_text(str(group.get("kind")), max_len=80)
            if group.get("kind") not in (None, "")
            else None,
            "surface": _redact_log_text(str(group.get("surface")), max_len=80)
            if group.get("surface") not in (None, "")
            else None,
            "count": _count_value(group.get("count")),
            "failed": _count_value(group.get("failed")),
            "throttled": _count_value(group.get("throttled")),
            "max_ms": int(max_ms or 0),
            "p95_ms": int(p95_ms or 0),
            "latest_elapsed_ms": int(latest_elapsed_ms or 0),
            "latest_symbol": _redact_log_text(str(latest_symbol), max_len=120)
            if latest_symbol not in (None, "")
            else None,
        }
        rows.append(
            {
                key: value
                for key, value in row.items()
                if value not in (None, "", {}, [])
                and not (key in {"failed", "throttled"} and value == 0)
            }
        )
    rows.sort(
        key=lambda item: (
            -int(item.get("max_ms") or 0),
            -int(item.get("p95_ms") or 0),
            -int(item.get("latest_elapsed_ms") or 0),
            str(item.get("bot") or ""),
            str(item.get("kind") or item.get("surface") or ""),
        )
    )
    return rows[:SMOKE_REPORT_BRIEF_REMOTE_CALL_SLOWEST_LIMIT]


def _brief_remote_call_health(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        summary = {}
    out = {
        key: value
        for key, value in {
            "total": _count_value(summary.get("total")),
            "succeeded": _count_value(summary.get("succeeded")),
            "failed": _count_value(summary.get("failed")),
            "throttled": _count_value(summary.get("throttled")),
            "failure_pct": summary.get("failure_pct"),
            "throttled_pct": summary.get("throttled_pct"),
        }.items()
        if value is not None
    }
    for key in (
        "failed_reason_codes",
        "failed_error_types",
        "failed_kinds",
        "failed_surfaces",
    ):
        value = summary.get(key)
        if isinstance(value, dict) and value:
            out[key] = value
    slowest = _brief_remote_call_slowest(summary)
    if slowest:
        out["slowest"] = slowest
    return out


def _brief_fill_refresh_health(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        summary = {}
    return {
        key: value
        for key, value in {
            "total": _count_value(summary.get("total")),
            "bots": _count_value(summary.get("bots")),
            "failed": _count_value(summary.get("failed")),
            "failure_pct": summary.get("failure_pct"),
            "failed_bots": _count_value(summary.get("failed_bots")),
            "latest_failed_bots": _count_value(summary.get("latest_failed_bots")),
            "recovered_groups": _count_value(summary.get("recovered_groups")),
            "statuses": summary.get("statuses") or {},
        }.items()
        if value is not None
    }


def _brief_execution_health(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        summary = {}
    return {
        key: value
        for key, value in {
            "total": _count_value(summary.get("total")),
            "bots": _count_value(summary.get("bots")),
            "failed": _count_value(summary.get("failed")),
            "rejected": _count_value(summary.get("rejected")),
            "ambiguous": _count_value(summary.get("ambiguous")),
            "confirmation_timeout": _count_value(
                summary.get("confirmation_timeout")
            ),
            "event_types": summary.get("event_types") or {},
            "statuses": summary.get("statuses") or {},
            "outcomes": summary.get("outcomes") or {},
        }.items()
        if value is not None
    }


def _brief_problem_event_groups(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        summary = {}
    groups = summary.get("groups")
    if not isinstance(groups, list):
        groups = []
    limit = max(0, int(SMOKE_REPORT_BRIEF_PROBLEM_GROUP_LIMIT))
    safe_group_keys = (
        "bot",
        "event_type",
        "reason_code",
        "status",
        "level",
        "hard",
        "symbol",
        "pside",
        "component",
        "count",
        "latest_ts",
    )
    compact_groups: list[dict[str, Any]] = []
    for group in groups[:limit]:
        if not isinstance(group, dict):
            continue
        compact = {
            key: group.get(key)
            for key in safe_group_keys
            if group.get(key) not in (None, "", {}, [])
        }
        if compact:
            compact_groups.append(compact)
    event_types = summary.get("event_types")
    if not isinstance(event_types, dict):
        event_types = {}
    ordered_event_types = sorted(
        event_types.items(),
        key=lambda item: (-_count_value(item[1]), str(item[0])),
    )
    return {
        "groups_truncated": bool(summary.get("groups_truncated")) or len(groups) > limit,
        "event_types_truncated": len(ordered_event_types) > limit,
        "event_types": {
            str(key): _count_value(value)
            for key, value in ordered_event_types[:limit]
            if str(key)
        },
        "groups": compact_groups,
    }


def _brief_log_window(logs: dict[str, Any]) -> dict[str, Any]:
    window = logs.get("window") if isinstance(logs.get("window"), dict) else {}
    return {
        key: window.get(key)
        for key in (
            "enabled",
            "since_ms",
            "until_ms",
            "lines_considered",
            "lines_skipped_before",
            "lines_skipped_after",
            "unparsed_ts",
            "unparsed_policy",
            "lines_skipped_unparsed",
            "dropped_unparsed_attention_matches",
            "dropped_unparsed_hard_matches",
        )
        if key in window
    }


def _brief_hsl_replay_health(hsl_replay_health: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "total": _count_value(hsl_replay_health.get("total")),
        "bots": _count_value(hsl_replay_health.get("bots")),
        "active_bots": _count_value(hsl_replay_health.get("active_bots")),
        "stale_active_bots": _count_value(
            hsl_replay_health.get("stale_active_bots")
        ),
        "long_running_active_bots": _count_value(
            hsl_replay_health.get("long_running_active_bots")
        ),
        "completed_bots": _count_value(hsl_replay_health.get("completed_bots")),
        "failed_bots": _count_value(hsl_replay_health.get("failed_bots")),
        "failed_attention_bots": _count_value(
            hsl_replay_health.get("failed_attention_bots")
        ),
        "event_types": hsl_replay_health.get("event_types") or {},
    }
    groups = hsl_replay_health.get("groups")
    if not isinstance(groups, list):
        return out

    max_active_latest_elapsed_ms: int | None = None
    max_active_latest_event_age_ms: int | None = None
    max_active_estimated_remaining_rows: int | None = None
    max_active_estimated_remaining_ms: int | None = None
    max_completed_elapsed_ms: int | None = None
    active_stage_counts: Counter[str] = Counter()

    for group in groups:
        if not isinstance(group, dict):
            continue
        completed_elapsed_ms = _hsl_replay_record_elapsed_ms(group.get("completed"))
        if completed_elapsed_ms is not None:
            if max_completed_elapsed_ms is None:
                max_completed_elapsed_ms = int(completed_elapsed_ms)
            else:
                max_completed_elapsed_ms = max(
                    max_completed_elapsed_ms,
                    int(completed_elapsed_ms),
                )
        if not bool(group.get("active")):
            continue
        event_age_ms = _non_negative_int(group.get("active_latest_event_age_ms"))
        if event_age_ms is not None:
            if max_active_latest_event_age_ms is None:
                max_active_latest_event_age_ms = int(event_age_ms)
            else:
                max_active_latest_event_age_ms = max(
                    max_active_latest_event_age_ms,
                    int(event_age_ms),
                )
        latest = group.get("latest") if isinstance(group.get("latest"), dict) else {}
        elapsed_ms = _hsl_replay_record_elapsed_ms(latest)
        if elapsed_ms is not None:
            if max_active_latest_elapsed_ms is None:
                max_active_latest_elapsed_ms = int(elapsed_ms)
            else:
                max_active_latest_elapsed_ms = max(
                    max_active_latest_elapsed_ms,
                    int(elapsed_ms),
                )
        derived = latest.get("derived") if isinstance(latest.get("derived"), dict) else {}
        estimated_remaining_rows = _non_negative_int(
            derived.get("estimated_remaining_rows")
        )
        if estimated_remaining_rows is not None:
            if max_active_estimated_remaining_rows is None:
                max_active_estimated_remaining_rows = int(estimated_remaining_rows)
            else:
                max_active_estimated_remaining_rows = max(
                    max_active_estimated_remaining_rows,
                    int(estimated_remaining_rows),
                )
        estimated_remaining_ms = _non_negative_int(
            derived.get("estimated_remaining_ms")
        )
        if estimated_remaining_ms is not None:
            if max_active_estimated_remaining_ms is None:
                max_active_estimated_remaining_ms = int(estimated_remaining_ms)
            else:
                max_active_estimated_remaining_ms = max(
                    max_active_estimated_remaining_ms,
                    int(estimated_remaining_ms),
                )
        data = latest.get("data") if isinstance(latest.get("data"), dict) else {}
        stage = data.get("stage")
        if isinstance(stage, str) and stage:
            active_stage_counts[_redact_log_text(stage, max_len=80)] += 1

    if max_active_latest_elapsed_ms is not None:
        out["max_active_latest_elapsed_ms"] = int(max_active_latest_elapsed_ms)
    if max_active_latest_event_age_ms is not None:
        out["max_active_latest_event_age_ms"] = int(max_active_latest_event_age_ms)
    if max_active_estimated_remaining_rows is not None:
        out["max_active_estimated_remaining_rows"] = int(
            max_active_estimated_remaining_rows
        )
    if max_active_estimated_remaining_ms is not None:
        out["max_active_estimated_remaining_ms"] = int(
            max_active_estimated_remaining_ms
        )
    if max_completed_elapsed_ms is not None:
        out["max_completed_elapsed_ms"] = int(max_completed_elapsed_ms)
    if active_stage_counts:
        out["active_stage_counts"] = dict(active_stage_counts.most_common())
    return out


def summarize_live_smoke_report_brief(report: dict[str, Any]) -> dict[str, Any]:
    """Project a full smoke report into top-level smoke-loop counters."""

    logs = report.get("logs") if isinstance(report.get("logs"), dict) else {}
    processes = (
        report.get("processes") if isinstance(report.get("processes"), dict) else {}
    )
    process_config_checks = (
        processes.get("config_checks")
        if isinstance(processes.get("config_checks"), dict)
        else {}
    )
    repository = (
        report.get("repository") if isinstance(report.get("repository"), dict) else {}
    )
    monitor = report.get("monitor") if isinstance(report.get("monitor"), dict) else {}
    risk_events = (
        report.get("risk_events") if isinstance(report.get("risk_events"), dict) else {}
    )
    ema_readiness_health = (
        report.get("ema_readiness_health")
        if isinstance(report.get("ema_readiness_health"), dict)
        else {}
    )
    fill_refresh_health = (
        report.get("fill_refresh_health")
        if isinstance(report.get("fill_refresh_health"), dict)
        else {}
    )
    execution_health = (
        report.get("execution_health")
        if isinstance(report.get("execution_health"), dict)
        else {}
    )
    exchange_config_refresh_health = (
        report.get("exchange_config_refresh_health")
        if isinstance(report.get("exchange_config_refresh_health"), dict)
        else {}
    )
    staged_readiness_health = (
        report.get("staged_readiness_health")
        if isinstance(report.get("staged_readiness_health"), dict)
        else {}
    )
    event_pipeline_health = (
        report.get("event_pipeline_health")
        if isinstance(report.get("event_pipeline_health"), dict)
        else {}
    )
    hsl_replay_health = (
        report.get("hsl_replay_health")
        if isinstance(report.get("hsl_replay_health"), dict)
        else {}
    )
    shutdown_events = (
        report.get("shutdown_events")
        if isinstance(report.get("shutdown_events"), dict)
        else {}
    )
    event_window = (
        report.get("event_window") if isinstance(report.get("event_window"), dict) else {}
    )
    ema_readiness_brief = {
        "total": _count_value(ema_readiness_health.get("total")),
        "bots": _count_value(ema_readiness_health.get("bots")),
        "latest_candidate_unavailable_total": _count_value(
            ema_readiness_health.get("latest_candidate_unavailable_total")
        ),
        "latest_unavailable_total": _count_value(
            ema_readiness_health.get("latest_unavailable_total")
        ),
        "latest_optional_drop_total": _count_value(
            ema_readiness_health.get("latest_optional_drop_total")
        ),
        "event_types": ema_readiness_health.get("event_types") or {},
    }
    for key in (
        "latest_candidate_reason_counts",
        "latest_unavailable_reason_counts",
        "latest_candidate_reason_symbols",
        "latest_unavailable_reason_symbols",
        "latest_candidate_error_type_counts",
    ):
        value = ema_readiness_health.get(key)
        if isinstance(value, dict) and value:
            ema_readiness_brief[key] = value
    staged_readiness_brief = {
        "total": _count_value(staged_readiness_health.get("total")),
        "bots": _count_value(staged_readiness_health.get("bots")),
        "latest_missing_surface_total": _count_value(
            staged_readiness_health.get("latest_missing_surface_total")
        ),
        "latest_invalid_surface_total": _count_value(
            staged_readiness_health.get("latest_invalid_surface_total")
        ),
        "event_types": staged_readiness_health.get("event_types") or {},
    }
    for key in ("latest_missing_surfaces", "latest_invalid_surfaces"):
        value = staged_readiness_health.get(key)
        if isinstance(value, dict) and value:
            staged_readiness_brief[key] = value
    risk_events_brief = {
        "total": _count_value(risk_events.get("total")),
        "event_types": risk_events.get("event_types") or {},
        "hsl_flat_finalization_anchors": risk_events.get(
            "hsl_flat_finalization_anchors"
        )
        or {},
        "hsl_status": _shareable_hsl_status(risk_events.get("hsl_status")),
    }
    raw_red_pending = _shareable_hsl_raw_red_pending(
        risk_events.get("hsl_raw_red_pending")
    )
    if raw_red_pending:
        risk_events_brief["hsl_raw_red_pending"] = raw_red_pending
    return {
        "ok": bool(report.get("ok")),
        "attention": bool(report.get("attention")),
        "hard_failures": _count_value(report.get("hard_failures")),
        "attention_count": _count_value(report.get("attention_count")),
        "hard_failure_sources": {
            key: _count_value(value)
            for key, value in (report.get("hard_failure_sources") or {}).items()
        },
        "attention_sources": {
            key: _count_value(value)
            for key, value in (report.get("attention_sources") or {}).items()
        },
        "recovered_problem_events": {
            "total": _count_value((report.get("recovered_problem_events") or {}).get("total")),
            "hard": _count_value((report.get("recovered_problem_events") or {}).get("hard")),
            "event_types": (
                (report.get("recovered_problem_events") or {}).get("event_types")
                or {}
            ),
        },
        "repository": {
            key: repository.get(key)
            for key in ("branch", "head", "dirty", "tracked_changes")
            if key in repository
        },
        "monitor": {
            key: monitor.get(key)
            for key in (
                "files_scanned",
                "records_total",
                "live_events",
                "legacy_events",
                "error_count",
                "warning_count",
                "file_discovery",
            )
            if key in monitor
        },
        "event_window": {
            key: event_window.get(key)
            for key in (
                "enabled",
                "since_ms",
                "until_ms",
                "events_considered",
                "events_skipped_before",
                "events_skipped_after",
                "invalid_window_ts",
                "event_tail_lines",
                "event_tail_limited_files",
                "event_tail_skipped_lines",
                "event_tail_skipped_lines_exact",
                "event_tail_skipped_bytes",
                "event_tail_line_numbers_exact",
                "event_tail_methods",
                "max_event_files_per_bot",
                "event_file_limit_scope",
                "event_file_limit_groups",
                "event_files_before_limit",
                "event_files_skipped_by_limit",
                "event_file_limit_order",
            )
            if key in event_window
        },
        "processes": {
            key: processes.get(key)
            for key in (
                "enabled",
                "ok",
                "hard_failures",
                "expected_total",
                "matched_expected",
                "running_live_total",
                "classification_source",
                "tmux_pane_ownership",
                "scan_error",
            )
            if key in processes
        }
        | {
            "missing_expected_count": len(processes.get("missing_expected") or []),
            "duplicate_configured_command_matches_count": len(
                processes.get("duplicate_configured_command_matches") or []
            ),
            "extra_passivbot_live_processes_count": len(
                processes.get("extra_passivbot_live_processes") or []
            ),
            "unexpected_running_count": len(processes.get("unexpected_running") or []),
            "config_checks": {
                key: process_config_checks.get(key)
                for key in (
                    "enabled",
                    "ok",
                    "checked",
                    "skipped",
                    "hard_failures",
                )
                if key in process_config_checks
            }
            | {
                "issues_count": len(process_config_checks.get("issues") or []),
            },
        },
        "logs": {
            "max_files": _count_value(logs.get("max_files")),
            "tail_lines": _count_value(logs.get("tail_lines")),
            "max_matches": _count_value(logs.get("max_matches")),
            "files_scanned": _count_value(logs.get("files_scanned")),
            "hard_matches": _count_value(logs.get("hard_matches")),
            "attention_matches": _count_value(logs.get("attention_matches")),
            "risk_attention_matches": _count_value(
                logs.get("risk_attention_matches")
            ),
            "risk_hard_matches": _count_value(logs.get("risk_hard_matches")),
            "non_risk_attention_matches": _count_value(
                logs.get("non_risk_attention_matches")
            ),
            "non_risk_hard_matches": _count_value(
                logs.get("non_risk_hard_matches")
            ),
            "dropped_unparsed_attention_matches": _count_value(
                logs.get("dropped_unparsed_attention_matches")
            ),
            "dropped_unparsed_hard_matches": _count_value(
                logs.get("dropped_unparsed_hard_matches")
            ),
            "window": _brief_log_window(logs),
        },
        "problem_events": {
            "total": _count_value(report.get("problem_event_count")),
            "hard": _count_value(report.get("hard_problem_event_count")),
        }
        | _brief_problem_event_groups(report.get("problem_event_groups")),
        "remote_calls": _brief_remote_call_health(report.get("remote_call_health")),
        "account_critical_remote_calls": _brief_remote_call_health(
            report.get("account_critical_remote_call_health")
        ),
        "startup_timings": _brief_startup_timings(report.get("startup_timings")),
        "execution": _brief_execution_health(execution_health),
        "fill_refresh": _brief_fill_refresh_health(fill_refresh_health),
        "ema_readiness": ema_readiness_brief,
        "exchange_config_refresh": {
            "total": _count_value(exchange_config_refresh_health.get("total")),
            "bots": _count_value(exchange_config_refresh_health.get("bots")),
            "succeeded": _count_value(
                exchange_config_refresh_health.get("succeeded")
            ),
            "failed": _count_value(exchange_config_refresh_health.get("failed")),
            "failure_pct": exchange_config_refresh_health.get("failure_pct"),
            "failed_bots": _count_value(
                exchange_config_refresh_health.get("failed_bots")
            ),
            "event_types": exchange_config_refresh_health.get("event_types") or {},
        },
        "staged_readiness": staged_readiness_brief,
        "event_pipeline": {
            "total": _count_value(event_pipeline_health.get("total")),
            "bots": _count_value(event_pipeline_health.get("bots")),
            "latest_queue_depth_total": _count_value(
                event_pipeline_health.get("latest_queue_depth_total")
            ),
            "latest_queue_unfinished_total": _count_value(
                event_pipeline_health.get("latest_queue_unfinished_total")
            ),
            "latest_dropped_total": _count_value(
                event_pipeline_health.get("latest_dropped_total")
            ),
            "latest_sink_error_total": _count_value(
                event_pipeline_health.get("latest_sink_error_total")
            ),
            "latest_degraded_total": _count_value(
                event_pipeline_health.get("latest_degraded_total")
            ),
            "latest_worker_not_alive_count": _count_value(
                event_pipeline_health.get("latest_worker_not_alive_count")
            ),
            "latest_stopping_count": _count_value(
                event_pipeline_health.get("latest_stopping_count")
            ),
            "event_types": event_pipeline_health.get("event_types") or {},
        },
        "hsl_replay": _brief_hsl_replay_health(hsl_replay_health),
        "risk_events": risk_events_brief,
        "shutdown_events": {
            "total": _count_value(shutdown_events.get("total")),
            "event_types": shutdown_events.get("event_types") or {},
        },
    }


def available_live_smoke_report_sections(report: dict[str, Any]) -> list[str]:
    return sorted(key for key in report if key not in _SMOKE_REPORT_SECTION_BASE_KEYS)


def project_live_smoke_report_sections(
    report: dict[str, Any],
    sections: list[str] | tuple[str, ...] | set[str],
) -> dict[str, Any]:
    requested = []
    for section in sections:
        normalized = str(section).strip()
        if normalized and normalized not in requested:
            requested.append(normalized)
    if not requested or "all" in requested:
        return report

    available = available_live_smoke_report_sections(report)
    available_set = set(available)
    unknown = [section for section in requested if section not in available_set]
    if unknown:
        raise ValueError(
            "unknown --section value(s): "
            + ", ".join(unknown)
            + "; available sections: "
            + ", ".join(available)
            + "; use all for the full report"
        )

    projected = {key: report[key] for key in _SMOKE_REPORT_SECTION_BASE_KEYS if key in report}
    for section in requested:
        projected[section] = report[section]
    return projected
