from __future__ import annotations

import json
import time
from collections.abc import Iterable as IterableABC
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from live.event_bus import EventTypes, LIVE_EVENT_ID_KEYS, LIVE_EVENT_MONITOR_PAYLOAD_KEY
from live.event_file_rows import event_file_rows
from live.problem_events import is_hard_problem_event, is_problem_event

EVENT_ID_KEYS = LIVE_EVENT_ID_KEYS
ORDER_TRACE_EVENT_TYPES = {
    EventTypes.ORDER_WAVE_STARTED,
    EventTypes.ORDER_WAVE_COMPLETED,
    EventTypes.EXECUTION_CREATE_SENT,
    EventTypes.EXECUTION_CREATE_SUCCEEDED,
    EventTypes.EXECUTION_CREATE_FAILED,
    EventTypes.EXECUTION_CREATE_REJECTED,
    EventTypes.EXECUTION_CANCEL_SENT,
    EventTypes.EXECUTION_CANCEL_SUCCEEDED,
    EventTypes.EXECUTION_CANCEL_FAILED,
    EventTypes.EXECUTION_CANCEL_AMBIGUOUS_TERMINAL,
    EventTypes.EXECUTION_AMBIGUOUS,
    EventTypes.EXECUTION_CONFIRMATION_REQUESTED,
    EventTypes.EXECUTION_CONFIRMATION_SATISFIED,
    EventTypes.EXECUTION_CONFIRMATION_TIMEOUT,
}
ORDER_TRACE_EVENT_DETAIL_KEYS = (
    "index",
    "pb_order_type",
    "order_type",
    "context",
    "reason",
    "reduce_only",
    "price",
    "qty",
    "elapsed_ms",
    "confirm_ms",
    "timeout_ms",
    "result_status",
    "current_epoch",
    "target_epoch",
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


@dataclass(frozen=True)
class EventFileDiscovery:
    files: list[Path]
    candidate_files: int = 0
    event_segments: int = 0
    rotated_skipped: int = 0
    scope_pruned: int = 0
    bot_path_pruning_applied: bool = False
    opaque_bot_id_full_scan: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_files": int(self.candidate_files),
            "event_segments": int(self.event_segments),
            "rotated_skipped": int(self.rotated_skipped),
            "scope_pruned": int(self.scope_pruned),
            "bot_path_pruning_applied": bool(self.bot_path_pruning_applied),
            "opaque_bot_id_full_scan": bool(self.opaque_bot_id_full_scan),
        }


def _is_event_segment(path: Path) -> bool:
    return path.name.endswith(".ndjson") or path.name.endswith(".ndjson.gz")


def _event_path_sort_key(path: Path) -> tuple[str, int, str]:
    return (str(path.parent), 1 if path.name == "current.ndjson" else 0, path.name)


def discover_event_files(
    root: str | Path,
    *,
    include_rotated: bool = False,
    exchange: str | Iterable[str] | None = None,
    user: str | Iterable[str] | None = None,
    bot_id: str | Iterable[str] | None = None,
) -> list[Path]:
    """Find monitor event NDJSON segments below a monitor root, bot root, or events dir."""
    return _discover_event_files(
        root,
        include_rotated=include_rotated,
        exchange=exchange,
        user=user,
        bot_id=bot_id,
    ).files


def discover_event_files_with_metadata(
    root: str | Path,
    *,
    include_rotated: bool = False,
    exchange: str | Iterable[str] | None = None,
    user: str | Iterable[str] | None = None,
    bot_id: str | Iterable[str] | None = None,
) -> EventFileDiscovery:
    """Find monitor event NDJSON segments and return bounded discovery metadata."""
    return _discover_event_files(
        root,
        include_rotated=include_rotated,
        exchange=exchange,
        user=user,
        bot_id=bot_id,
    )


def _discover_event_files(
    root: str | Path,
    *,
    include_rotated: bool = False,
    exchange: str | Iterable[str] | None = None,
    user: str | Iterable[str] | None = None,
    bot_id: str | Iterable[str] | None = None,
) -> EventFileDiscovery:
    """Find monitor event NDJSON segments and report bounded discovery metadata."""
    path = Path(root).expanduser()
    exchange_filter = _normalize_filter_values(exchange)
    user_filter = _normalize_filter_values(user)
    bot_filter = _normalize_filter_values(bot_id)
    bot_path_filter = _path_like_bot_id_filter(bot_filter)
    opaque_bot_full_scan = bool(bot_filter and not bot_path_filter)
    if path.is_file():
        files = [path] if _is_event_segment(path) else []
        return EventFileDiscovery(
            files=files,
            candidate_files=1,
            event_segments=len(files),
            bot_path_pruning_applied=False,
            opaque_bot_id_full_scan=opaque_bot_full_scan,
        )
    if not path.exists():
        raise FileNotFoundError(str(path))
    if not path.is_dir():
        return EventFileDiscovery(
            files=[],
            bot_path_pruning_applied=bool(bot_path_filter),
            opaque_bot_id_full_scan=opaque_bot_full_scan,
        )
    files: list[Path] = []
    candidate_files = 0
    event_segments = 0
    rotated_skipped = 0
    scope_pruned = 0
    for candidate in path.rglob("*.ndjson*"):
        if not candidate.is_file():
            continue
        candidate_files += 1
        if not (
            candidate.parent.name == "events"
            and _is_event_segment(candidate)
        ):
            continue
        event_segments += 1
        if not _monitor_path_scope_matches_under_root(
            path,
            candidate,
            exchange_filter=exchange_filter,
            user_filter=user_filter,
            bot_path_filter=bot_path_filter,
        ):
            scope_pruned += 1
            continue
        if not include_rotated and candidate.name != "current.ndjson":
            rotated_skipped += 1
            continue
        files.append(candidate)
    return EventFileDiscovery(
        files=sorted(files, key=_event_path_sort_key),
        candidate_files=candidate_files,
        event_segments=event_segments,
        rotated_skipped=rotated_skipped,
        scope_pruned=scope_pruned,
        bot_path_pruning_applied=bool(bot_path_filter),
        opaque_bot_id_full_scan=opaque_bot_full_scan,
    )


def _file_mtime_ms(path: Path) -> int | None:
    try:
        return int(path.stat().st_mtime * 1000)
    except OSError:
        return None


def _recent_event_file_sort_key(path: Path) -> tuple[int, int, str, str]:
    mtime_ms = _file_mtime_ms(path)
    return (
        0 if path.name == "current.ndjson" else 1,
        -(mtime_ms if mtime_ms is not None else -1),
        str(path.parent),
        path.name,
    )


def _limit_recent_event_files_per_bot(
    files: list[Path],
    max_event_files_per_bot: int,
) -> tuple[list[Path], int, int]:
    if max_event_files_per_bot <= 0:
        return files, 0, 0
    grouped: dict[Path, list[Path]] = {}
    for path in files:
        grouped.setdefault(path.parent, []).append(path)
    selected: list[Path] = []
    skipped = 0
    for _, group_files in sorted(grouped.items(), key=lambda item: str(item[0])):
        ordered = sorted(group_files, key=_recent_event_file_sort_key)
        selected.extend(ordered[:max_event_files_per_bot])
        skipped += max(0, len(ordered) - max_event_files_per_bot)
    return sorted(selected, key=_recent_event_file_sort_key), skipped, len(grouped)


def _live_event_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    payload = row.get("payload")
    if not isinstance(payload, dict):
        return None
    live_event = payload.get(LIVE_EVENT_MONITOR_PAYLOAD_KEY)
    return live_event if isinstance(live_event, dict) else None


def _event_ids(live_event: dict[str, Any]) -> dict[str, Any]:
    ids = live_event.get("ids")
    out = dict(ids) if isinstance(ids, dict) else {}
    if out.get("snapshot_id") is None:
        data = live_event.get("data")
        if isinstance(data, dict) and data.get("snapshot_id") is not None:
            out["snapshot_id"] = data["snapshot_id"]
    return out


def _monitor_path_exchange_user(path: Path) -> tuple[str, str] | None:
    """Return monitor path-derived exchange/user for .../<exchange>/<user>/events/*.ndjson."""
    if path.parent.name != "events":
        return None
    try:
        user = path.parent.parent.name
        exchange = path.parent.parent.parent.name
    except IndexError:
        return None
    if not exchange or not user:
        return None
    return exchange, user


def _monitor_path_bot_id(path: Path) -> str | None:
    """Return monitor path-derived bot id for .../<exchange>/<user>/events/*.ndjson."""
    scope = _monitor_path_exchange_user(path)
    if scope is None:
        return None
    exchange, user = scope
    return f"{exchange}/{user}"


def _monitor_path_scope_matches_under_root(
    root: Path,
    path: Path,
    *,
    exchange_filter: set[str],
    user_filter: set[str],
    bot_path_filter: set[str],
) -> bool:
    """Conservatively prune only obvious <monitor>/<exchange>/<user>/events files."""
    if not exchange_filter and not user_filter and not bot_path_filter:
        return True
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        return True
    if len(relative_parts) != 4 or relative_parts[2] != "events":
        return True
    exchange, user = relative_parts[0], relative_parts[1]
    if bot_path_filter and f"{exchange}/{user}" not in bot_path_filter:
        return False
    return _filter_matches(exchange, exchange_filter) and _filter_matches(
        user,
        user_filter,
    )


def _path_like_bot_id_filter(bot_filter: set[str]) -> set[str]:
    """Return path-derived bot ids only when pruning cannot drop opaque bot ids."""
    if not bot_filter:
        return set()
    return set(bot_filter) if all("/" in value for value in bot_filter) else set()


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


def _normalize_data_eq_filters(
    values: str | Iterable[str] | None,
) -> dict[str, set[str]]:
    if values is None:
        return {}
    if isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = list(values)
    out: dict[str, set[str]] = defaultdict(set)
    for raw_value in raw_values:
        text = str(raw_value)
        if "=" not in text:
            raise ValueError("data_eq filters must use key=value")
        key, expected = text.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("data_eq filters must include a non-empty key")
        out[key].add(expected.strip())
    return {key: set(values) for key, values in sorted(out.items())}


def _data_value_matches(value: Any, expected_values: set[str]) -> bool:
    candidates = {str(value)}
    if isinstance(value, bool):
        candidates.add(str(value).lower())
    elif value is None:
        candidates.add("null")
    return bool(candidates.intersection(expected_values))


def _data_filters_match(
    live_event: dict[str, Any],
    data_eq_filters: dict[str, set[str]],
) -> bool:
    if not data_eq_filters:
        return True
    data = live_event.get("data")
    if not isinstance(data, dict):
        return False
    for key, expected_values in data_eq_filters.items():
        if key not in data:
            return False
        if not _data_value_matches(data.get(key), expected_values):
            return False
    return True


def _filter_report(
    *,
    cycle_id: str | None,
    event_types: set[str],
    levels: set[str],
    exchanges: set[str],
    users: set[str],
    bot_ids: set[str],
    snapshot_ids: set[str],
    plan_ids: set[str],
    action_ids: set[str],
    order_wave_ids: set[str],
    remote_call_ids: set[str],
    remote_call_group_ids: set[str],
    symbols: set[str],
    psides: set[str],
    sides: set[str],
    reason_codes: set[str],
    statuses: set[str],
    sources: set[str],
    components: set[str],
    tags: set[str],
    debug_profiles: set[str],
    data_eq_filters: dict[str, set[str]],
    problem_events: bool,
    hard_problem_events: bool,
    since_ms: int | None = None,
    until_ms: int | None = None,
) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if cycle_id is not None:
        filters["cycle_id"] = str(cycle_id)
    if since_ms is not None:
        filters["since_ms"] = int(since_ms)
    if until_ms is not None:
        filters["until_ms"] = int(until_ms)
    if event_types:
        filters["event_types"] = sorted(event_types)
    if levels:
        filters["levels"] = sorted(levels)
    if exchanges:
        filters["exchanges"] = sorted(exchanges)
    if users:
        filters["users"] = sorted(users)
    if bot_ids:
        filters["bot_ids"] = sorted(bot_ids)
    if snapshot_ids:
        filters["snapshot_ids"] = sorted(snapshot_ids)
    if plan_ids:
        filters["plan_ids"] = sorted(plan_ids)
    if action_ids:
        filters["action_ids"] = sorted(action_ids)
    if order_wave_ids:
        filters["order_wave_ids"] = sorted(order_wave_ids)
    if remote_call_ids:
        filters["remote_call_ids"] = sorted(remote_call_ids)
    if remote_call_group_ids:
        filters["remote_call_group_ids"] = sorted(remote_call_group_ids)
    if symbols:
        filters["symbols"] = sorted(symbols)
    if psides:
        filters["psides"] = sorted(psides)
    if sides:
        filters["sides"] = sorted(sides)
    if reason_codes:
        filters["reason_codes"] = sorted(reason_codes)
    if statuses:
        filters["statuses"] = sorted(statuses)
    if sources:
        filters["sources"] = sorted(sources)
    if components:
        filters["components"] = sorted(components)
    if tags:
        filters["tags"] = sorted(tags)
    if debug_profiles:
        filters["debug_profiles"] = sorted(debug_profiles)
    if data_eq_filters:
        filters["data_eq"] = {
            key: sorted(values) for key, values in sorted(data_eq_filters.items())
        }
    if problem_events:
        filters["problem_events"] = True
    if hard_problem_events:
        filters["hard_problem_events"] = True
    return filters


def build_event_query_filters(
    *,
    cycle_id: str | None = None,
    event_type: str | Iterable[str] | None = None,
    level: str | Iterable[str] | None = None,
    exchange: str | Iterable[str] | None = None,
    user: str | Iterable[str] | None = None,
    bot_id: str | Iterable[str] | None = None,
    snapshot_id: str | Iterable[str] | None = None,
    plan_id: str | Iterable[str] | None = None,
    action_id: str | Iterable[str] | None = None,
    order_wave_id: str | Iterable[str] | None = None,
    remote_call_id: str | Iterable[str] | None = None,
    remote_call_group_id: str | Iterable[str] | None = None,
    symbol: str | Iterable[str] | None = None,
    pside: str | Iterable[str] | None = None,
    side: str | Iterable[str] | None = None,
    reason_code: str | Iterable[str] | None = None,
    status: str | Iterable[str] | None = None,
    source: str | Iterable[str] | None = None,
    component: str | Iterable[str] | None = None,
    tag: str | Iterable[str] | None = None,
    debug_profile: str | Iterable[str] | None = None,
    data_eq: str | Iterable[str] | None = None,
    problem_events: bool = False,
    hard_problem_events: bool = False,
    since_ms: int | None = None,
    until_ms: int | None = None,
) -> dict[str, Any]:
    since_filter = int(since_ms) if since_ms is not None else None
    until_filter = int(until_ms) if until_ms is not None else None
    if (
        since_filter is not None
        and until_filter is not None
        and since_filter > until_filter
    ):
        raise ValueError("since_ms must be <= until_ms")
    debug_profiles = _normalize_filter_values(debug_profile)
    return {
        "cycle_id": str(cycle_id) if cycle_id is not None else None,
        "event_types": _normalize_filter_values(event_type),
        "levels": _normalize_filter_values(level),
        "exchanges": _normalize_filter_values(exchange),
        "users": _normalize_filter_values(user),
        "bot_ids": _normalize_filter_values(bot_id),
        "snapshot_ids": _normalize_filter_values(snapshot_id),
        "plan_ids": _normalize_filter_values(plan_id),
        "action_ids": _normalize_filter_values(action_id),
        "order_wave_ids": _normalize_filter_values(order_wave_id),
        "remote_call_ids": _normalize_filter_values(remote_call_id),
        "remote_call_group_ids": _normalize_filter_values(remote_call_group_id),
        "symbols": _normalize_filter_values(symbol),
        "psides": _normalize_filter_values(pside),
        "sides": _normalize_filter_values(side),
        "reason_codes": _normalize_filter_values(reason_code),
        "statuses": _normalize_filter_values(status),
        "sources": _normalize_filter_values(source),
        "components": _normalize_filter_values(component),
        "tags": _normalize_filter_values(tag),
        "debug_profiles": debug_profiles,
        "data_eq_filters": _normalize_data_eq_filters(data_eq),
        "problem_events": bool(problem_events),
        "hard_problem_events": bool(hard_problem_events),
        "since_ms": since_filter,
        "until_ms": until_filter,
    }


def event_query_filter_report(filters: dict[str, Any]) -> dict[str, Any]:
    return _filter_report(
        cycle_id=filters.get("cycle_id"),
        event_types=filters["event_types"],
        levels=filters["levels"],
        exchanges=filters["exchanges"],
        users=filters["users"],
        bot_ids=filters["bot_ids"],
        snapshot_ids=filters["snapshot_ids"],
        plan_ids=filters["plan_ids"],
        action_ids=filters["action_ids"],
        order_wave_ids=filters["order_wave_ids"],
        remote_call_ids=filters["remote_call_ids"],
        remote_call_group_ids=filters["remote_call_group_ids"],
        symbols=filters["symbols"],
        psides=filters["psides"],
        sides=filters["sides"],
        reason_codes=filters["reason_codes"],
        statuses=filters["statuses"],
        sources=filters["sources"],
        components=filters["components"],
        tags=filters["tags"],
        debug_profiles=filters["debug_profiles"],
        data_eq_filters=filters["data_eq_filters"],
        problem_events=bool(filters["problem_events"]),
        hard_problem_events=bool(filters["hard_problem_events"]),
        since_ms=filters.get("since_ms"),
        until_ms=filters.get("until_ms"),
    )


def event_matches_query_filters(
    *,
    path: Path,
    row: dict[str, Any],
    live_event: dict[str, Any],
    filters: dict[str, Any],
) -> bool:
    ids = _event_ids(live_event)
    record_event_type = live_event.get("event_type") or row.get("kind")
    record_level = live_event.get("level")
    record_symbol = live_event.get("symbol") or row.get("symbol")
    record_pside = live_event.get("pside") or row.get("pside")
    record_side = live_event.get("side")
    record_status = live_event.get("status")
    record_reason_code = live_event.get("reason_code")
    record_source = live_event.get("source")
    record_component = live_event.get("component")
    record_tags = _event_tags(row, live_event)
    record_data = (
        live_event.get("data") if isinstance(live_event.get("data"), dict) else {}
    )
    path_scope = _monitor_path_exchange_user(path)
    record_exchange = live_event.get("exchange") or row.get("exchange")
    record_user = live_event.get("user") or row.get("user")
    if path_scope is not None:
        if record_exchange is None:
            record_exchange = path_scope[0]
        if record_user is None:
            record_user = path_scope[1]
    cycle_id = filters.get("cycle_id")
    if cycle_id is not None and str(ids.get("cycle_id")) != str(cycle_id):
        return False
    if not _filter_matches(record_event_type, filters["event_types"]):
        return False
    if not _filter_matches(record_level, filters["levels"]):
        return False
    if not _filter_matches(record_exchange, filters["exchanges"]):
        return False
    if not _filter_matches(record_user, filters["users"]):
        return False
    if not _filter_matches(
        ids.get("bot_id") or _monitor_path_bot_id(path), filters["bot_ids"]
    ):
        return False
    if not _filter_matches(ids.get("snapshot_id"), filters["snapshot_ids"]):
        return False
    if not _filter_matches(ids.get("plan_id"), filters["plan_ids"]):
        return False
    if not _filter_matches(ids.get("action_id"), filters["action_ids"]):
        return False
    if not _filter_matches(ids.get("order_wave_id"), filters["order_wave_ids"]):
        return False
    if not _filter_matches(ids.get("remote_call_id"), filters["remote_call_ids"]):
        return False
    if not _filter_matches(
        ids.get("remote_call_group_id"), filters["remote_call_group_ids"]
    ):
        return False
    if not _filter_matches(record_symbol, filters["symbols"]):
        return False
    if not _filter_matches(record_pside, filters["psides"]):
        return False
    if not _filter_matches(record_side, filters["sides"]):
        return False
    if not _filter_matches(record_reason_code, filters["reason_codes"]):
        return False
    if not _filter_matches(record_status, filters["statuses"]):
        return False
    if not _filter_matches(record_source, filters["sources"]):
        return False
    if not _filter_matches(record_component, filters["components"]):
        return False
    if filters["tags"] and not set(record_tags).intersection(filters["tags"]):
        return False
    if not _filter_matches(
        record_data.get("debug_profile"), filters["debug_profiles"]
    ):
        return False
    if not _data_filters_match(live_event, filters["data_eq_filters"]):
        return False
    if filters["hard_problem_events"]:
        return is_hard_problem_event(live_event)
    if filters["problem_events"]:
        return is_problem_event(live_event)
    return True


def _compact_record(
    *,
    path: Path,
    line_no: int,
    row: dict[str, Any],
    live_event: dict[str, Any],
    include_data: bool,
    exchange: Any = None,
    user: Any = None,
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
        "source": live_event.get("source"),
        "component": live_event.get("component"),
        "exchange": exchange or live_event.get("exchange") or row.get("exchange"),
        "user": user or live_event.get("user") or row.get("user"),
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
            for key in EVENT_ID_KEYS
            if (value := ids.get(key)) is not None
        ]
        if id_parts:
            parts.append("ids=" + ",".join(id_parts))
    return " ".join(parts)


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _sorted_counter_items(
    counter: Counter[str], *, limit: int = 20
) -> list[dict[str, Any]]:
    return [
        {"id": str(key), "events": int(value)}
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[
            : max(0, int(limit))
        ]
    ]


def _sorted_values(values: set[str], *, limit: int = 20) -> list[str]:
    return sorted(values)[: max(0, int(limit))]


def _record_ts(row: dict[str, Any]) -> int | None:
    try:
        return int(row.get("ts"))
    except (TypeError, ValueError):
        return None


def _event_tags(row: dict[str, Any], live_event: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for source in (row.get("tags"), live_event.get("tags")):
        if isinstance(source, str):
            tags.append(source)
        elif isinstance(source, IterableABC):
            tags.extend(str(item) for item in source if item is not None)
    return sorted(set(tags))


def _short_ref(value: Any, *, max_len: int = 32) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    if len(text) <= max_len:
        return text
    keep = max(4, (max_len - 3) // 2)
    return f"{text[:keep]}...{text[-keep:]}"


def _first_data_value(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _action_from_event(event_type: str | None, action_id: Any) -> str | None:
    if action_id is not None:
        parts = str(action_id).split(":")
        if len(parts) >= 2 and parts[-2]:
            return parts[-2]
    text = str(event_type or "")
    if ".create_" in text or text.endswith(".create_sent"):
        return "create"
    if ".cancel_" in text or text.endswith(".cancel_sent"):
        return "cancel"
    return None


def _action_index(action_id: Any) -> int | None:
    if action_id is None:
        return None
    try:
        return int(str(action_id).split(":")[-1])
    except (TypeError, ValueError):
        return None


def _is_order_trace_event(
    live_event: dict[str, Any],
    event_type: str | None,
    ids: dict[str, Any],
) -> bool:
    return bool(
        event_type in ORDER_TRACE_EVENT_TYPES
        or ids.get("order_wave_id") is not None
        or ids.get("action_id") is not None
        or str(event_type or "").startswith("execution.")
    )


def _order_trace_event(
    *,
    path: Path,
    line_no: int,
    row: dict[str, Any],
    live_event: dict[str, Any],
    event_type: str | None,
) -> dict[str, Any]:
    ids = _event_ids(live_event)
    data = live_event.get("data") if isinstance(live_event.get("data"), dict) else {}
    order_id = _first_data_value(data, "result_order_id_short", "order_id_short")
    client_order_id = _first_data_value(
        data,
        "result_client_order_id_short",
        "client_order_id_short",
    )
    if order_id is None:
        order_id = live_event.get("order_id")
    if client_order_id is None:
        client_order_id = live_event.get("client_order_id")
    item: dict[str, Any] = {
        "path": str(path),
        "line": int(line_no),
        "ts": row.get("ts"),
        "seq": row.get("seq"),
        "event_type": event_type or row.get("kind"),
        "level": live_event.get("level"),
        "status": live_event.get("status"),
        "reason_code": live_event.get("reason_code"),
        "symbol": live_event.get("symbol") or row.get("symbol"),
        "pside": live_event.get("pside") or row.get("pside"),
        "side": live_event.get("side"),
        "order_id_short": _short_ref(order_id),
        "client_order_id_short": _short_ref(client_order_id),
        "ids": {
            key: ids.get(key)
            for key in EVENT_ID_KEYS
            if ids.get(key) is not None
        },
    }
    for key in ORDER_TRACE_EVENT_DETAIL_KEYS:
        value = data.get(key)
        if value is not None:
            item[key] = value
    return {key: value for key, value in item.items() if value not in (None, {}, [])}


class _OrderTraceBuilder:
    def __init__(self, *, event_limit: int) -> None:
        self.event_limit = max(0, int(event_limit))
        self.matched_order_events = 0
        self.unscoped_events: list[dict[str, Any]] = []
        self.unscoped_event_count = 0
        self.unscoped_events_truncated = False
        self.waves: dict[str, dict[str, Any]] = {}

    def add(
        self,
        *,
        path: Path,
        line_no: int,
        row: dict[str, Any],
        live_event: dict[str, Any],
        event_type: str | None,
        ids: dict[str, Any],
    ) -> None:
        if not _is_order_trace_event(live_event, event_type, ids):
            return
        self.matched_order_events += 1
        item = _order_trace_event(
            path=path,
            line_no=line_no,
            row=row,
            live_event=live_event,
            event_type=event_type,
        )
        order_wave_id = ids.get("order_wave_id")
        if order_wave_id is None:
            self.unscoped_event_count += 1
            if len(self.unscoped_events) < self.event_limit:
                self.unscoped_events.append(item)
            else:
                self.unscoped_events_truncated = True
            return
        wave_id = str(order_wave_id)
        wave = self.waves.setdefault(
            wave_id,
            {
                "order_wave_id": wave_id,
                "event_count": 0,
                "events_truncated": False,
                "timeline": [],
                "event_types": Counter(),
                "statuses": Counter(),
                "reason_codes": Counter(),
                "symbols": set(),
                "psides": set(),
                "sides": set(),
                "actions": {},
                "confirmations": [],
                "confirmation_count": 0,
                "confirmations_truncated": False,
                "first_ts": None,
                "last_ts": None,
            },
        )
        self._add_event_summary(wave, item)
        action_id = ids.get("action_id")
        if action_id is None:
            if str(event_type or "").startswith("execution.confirmation_"):
                wave["confirmation_count"] += 1
                if len(wave["confirmations"]) < self.event_limit:
                    wave["confirmations"].append(item)
                else:
                    wave["confirmations_truncated"] = True
            return
        action_key = str(action_id)
        action = wave["actions"].setdefault(
            action_key,
            {
                "action_id": action_key,
                "action": _action_from_event(event_type, action_id),
                "index": _action_index(action_id),
                "event_count": 0,
                "events_truncated": False,
                "events": [],
                "event_types": Counter(),
                "statuses": Counter(),
                "reason_codes": Counter(),
                "symbols": set(),
                "psides": set(),
                "sides": set(),
                "order_ids_short": set(),
                "client_order_ids_short": set(),
                "first_ts": None,
                "last_ts": None,
                "latest_event_type": None,
                "latest_status": None,
                "latest_reason_code": None,
            },
        )
        self._add_event_summary(action, item, keep_timeline_key="events")

    def _add_event_summary(
        self,
        summary: dict[str, Any],
        item: dict[str, Any],
        *,
        keep_timeline_key: str = "timeline",
    ) -> None:
        summary["event_count"] += 1
        ts = _record_ts(item)
        if ts is not None:
            if summary.get("first_ts") is None:
                summary["first_ts"] = ts
            summary["last_ts"] = ts
        event_type = item.get("event_type")
        if event_type is not None:
            summary["event_types"][str(event_type)] += 1
            summary["latest_event_type"] = str(event_type)
        status = item.get("status")
        if status is not None:
            summary["statuses"][str(status)] += 1
            summary["latest_status"] = str(status)
        reason_code = item.get("reason_code")
        if reason_code is not None:
            summary["reason_codes"][str(reason_code)] += 1
            summary["latest_reason_code"] = str(reason_code)
        for key, target in (
            ("symbol", "symbols"),
            ("pside", "psides"),
            ("side", "sides"),
            ("order_id_short", "order_ids_short"),
            ("client_order_id_short", "client_order_ids_short"),
        ):
            value = item.get(key)
            if value is not None and target in summary:
                summary[target].add(str(value))
        kept = summary[keep_timeline_key]
        if len(kept) < self.event_limit:
            kept.append(item)
        else:
            summary["events_truncated"] = True

    def to_dict(self) -> dict[str, Any]:
        waves = []
        for wave_id, summary in sorted(self.waves.items()):
            actions = []
            for action_id, action in sorted(summary["actions"].items()):
                actions.append(self._finalize_action(action))
            waves.append(
                {
                    "order_wave_id": wave_id,
                    "event_count": int(summary["event_count"]),
                    "events_truncated": bool(summary["events_truncated"]),
                    "first_ts": summary.get("first_ts"),
                    "last_ts": summary.get("last_ts"),
                    "event_types": _sorted_counter(summary["event_types"]),
                    "statuses": _sorted_counter(summary["statuses"]),
                    "reason_codes": _sorted_counter(summary["reason_codes"]),
                    "symbols": _sorted_values(summary["symbols"]),
                    "psides": _sorted_values(summary["psides"]),
                    "sides": _sorted_values(summary["sides"]),
                    "action_count": len(summary["actions"]),
                    "actions": actions,
                    "confirmation_count": int(summary["confirmation_count"]),
                    "confirmations_truncated": bool(
                        summary["confirmations_truncated"]
                    ),
                    "confirmations": summary["confirmations"],
                    "timeline": summary["timeline"],
                }
            )
        out: dict[str, Any] = {
            "matched_order_events": int(self.matched_order_events),
            "order_wave_count": len(self.waves),
            "order_waves": waves,
            "unscoped_event_count": int(self.unscoped_event_count),
        }
        if self.unscoped_events or self.unscoped_events_truncated:
            out["unscoped_events"] = self.unscoped_events
            out["unscoped_events_truncated"] = bool(self.unscoped_events_truncated)
        return out

    def _finalize_action(self, action: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "action_id": action["action_id"],
                "action": action.get("action"),
                "index": action.get("index"),
                "event_count": int(action["event_count"]),
                "events_truncated": bool(action["events_truncated"]),
                "first_ts": action.get("first_ts"),
                "last_ts": action.get("last_ts"),
                "latest_event_type": action.get("latest_event_type"),
                "latest_status": action.get("latest_status"),
                "latest_reason_code": action.get("latest_reason_code"),
                "event_types": _sorted_counter(action["event_types"]),
                "statuses": _sorted_counter(action["statuses"]),
                "reason_codes": _sorted_counter(action["reason_codes"]),
                "symbols": _sorted_values(action["symbols"]),
                "psides": _sorted_values(action["psides"]),
                "sides": _sorted_values(action["sides"]),
                "order_ids_short": _sorted_values(action["order_ids_short"]),
                "client_order_ids_short": _sorted_values(
                    action["client_order_ids_short"]
                ),
                "events": action["events"],
            }.items()
            if value not in (None, {}, [])
        }


class _CycleTraceBuilder:
    def __init__(self, *, event_limit: int, include_data: bool) -> None:
        self.event_limit = max(0, int(event_limit))
        self.include_data = bool(include_data)
        self.matched_cycle_events = 0
        self.missing_cycle_id_event_count = 0
        self.missing_cycle_id_events: list[dict[str, Any]] = []
        self.missing_cycle_id_events_truncated = False
        self.cycles: dict[str, dict[str, Any]] = {}

    def add(
        self,
        *,
        path: Path,
        line_no: int,
        row: dict[str, Any],
        live_event: dict[str, Any],
        event_type: str | None,
        ids: dict[str, Any],
        exchange: Any = None,
        user: Any = None,
    ) -> None:
        item = _compact_record(
            path=path,
            line_no=line_no,
            row=row,
            live_event=live_event,
            include_data=self.include_data,
            exchange=exchange,
            user=user,
        )
        cycle_id = ids.get("cycle_id")
        if cycle_id is None:
            self.missing_cycle_id_event_count += 1
            if len(self.missing_cycle_id_events) < self.event_limit:
                self.missing_cycle_id_events.append(item)
            else:
                self.missing_cycle_id_events_truncated = True
            return
        self.matched_cycle_events += 1
        cycle_key = str(cycle_id)
        cycle = self.cycles.setdefault(
            cycle_key,
            {
                "cycle_id": cycle_key,
                "event_count": 0,
                "events_truncated": False,
                "timeline": [],
                "trace_summary": _TraceSummaryBuilder(),
                "order_trace": _OrderTraceBuilder(event_limit=self.event_limit),
            },
        )
        cycle["event_count"] += 1
        if len(cycle["timeline"]) < self.event_limit:
            cycle["timeline"].append(item)
        else:
            cycle["events_truncated"] = True
        cycle["trace_summary"].add(
            row=row,
            live_event=live_event,
            event_type=event_type,
            exchange=exchange,
            user=user,
        )
        cycle["order_trace"].add(
            path=path,
            line_no=line_no,
            row=row,
            live_event=live_event,
            event_type=event_type,
            ids=ids,
        )

    def to_dict(self) -> dict[str, Any]:
        cycles = []
        for cycle_id, summary in sorted(self.cycles.items()):
            trace_summary = summary["trace_summary"].to_dict()
            cycles.append(
                {
                    "cycle_id": cycle_id,
                    "event_count": int(summary["event_count"]),
                    "events_truncated": bool(summary["events_truncated"]),
                    "first_ts": trace_summary.get("first_ts"),
                    "last_ts": trace_summary.get("last_ts"),
                    "trace_summary": trace_summary,
                    "order_trace": summary["order_trace"].to_dict(),
                    "timeline": summary["timeline"],
                }
            )
        out: dict[str, Any] = {
            "matched_cycle_events": int(self.matched_cycle_events),
            "cycle_count": len(self.cycles),
            "cycles": cycles,
            "missing_cycle_id_event_count": int(self.missing_cycle_id_event_count),
        }
        if self.missing_cycle_id_events or self.missing_cycle_id_events_truncated:
            out["missing_cycle_id_events"] = self.missing_cycle_id_events
            out["missing_cycle_id_events_truncated"] = bool(
                self.missing_cycle_id_events_truncated
            )
        return out


class _TraceSummaryBuilder:
    def __init__(self) -> None:
        self.events = 0
        self.first_ts: Any = None
        self.last_ts: Any = None
        self.event_types: Counter[str] = Counter()
        self.levels: Counter[str] = Counter()
        self.sources: Counter[str] = Counter()
        self.components: Counter[str] = Counter()
        self.tags: Counter[str] = Counter()
        self.exchanges: Counter[str] = Counter()
        self.users: Counter[str] = Counter()
        self.statuses: Counter[str] = Counter()
        self.reason_codes: Counter[str] = Counter()
        self.symbols: set[str] = set()
        self.psides: set[str] = set()
        self.sides: set[str] = set()
        self.ids: dict[str, Counter[str]] = {
            key: Counter() for key in EVENT_ID_KEYS
        }
        self.order_waves: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "events": 0,
                "event_types": Counter(),
                "statuses": Counter(),
                "reason_codes": Counter(),
                "symbols": set(),
                "psides": set(),
                "action_ids": set(),
            }
        )

    def add(
        self,
        *,
        row: dict[str, Any],
        live_event: dict[str, Any],
        event_type: str | None,
        exchange: Any = None,
        user: Any = None,
    ) -> None:
        self.events += 1
        ts = row.get("ts")
        if self.first_ts is None or (ts is not None and ts < self.first_ts):
            self.first_ts = ts
        if self.last_ts is None or (ts is not None and ts > self.last_ts):
            self.last_ts = ts
        if event_type:
            self.event_types[str(event_type)] += 1
        for key, counter in (
            ("level", self.levels),
            ("source", self.sources),
            ("component", self.components),
            ("status", self.statuses),
            ("reason_code", self.reason_codes),
        ):
            value = live_event.get(key)
            if value is not None:
                counter[str(value)] += 1
        for value, counter in (
            (
                exchange or live_event.get("exchange") or row.get("exchange"),
                self.exchanges,
            ),
            (user or live_event.get("user") or row.get("user"), self.users),
        ):
            if value is not None:
                counter[str(value)] += 1
        for tag in _event_tags(row, live_event):
            self.tags[tag] += 1
        symbol = live_event.get("symbol") or row.get("symbol")
        if symbol is not None:
            self.symbols.add(str(symbol))
        pside = live_event.get("pside") or row.get("pside")
        if pside is not None:
            self.psides.add(str(pside))
        side = live_event.get("side")
        if side is not None:
            self.sides.add(str(side))

        ids = _event_ids(live_event)
        for key in EVENT_ID_KEYS:
            value = ids.get(key)
            if value is not None:
                self.ids[key][str(value)] += 1

        order_wave_id = ids.get("order_wave_id")
        if order_wave_id is None:
            return
        wave = self.order_waves[str(order_wave_id)]
        wave["events"] += 1
        if event_type:
            wave["event_types"][str(event_type)] += 1
        status = live_event.get("status")
        if status is not None:
            wave["statuses"][str(status)] += 1
        reason_code = live_event.get("reason_code")
        if reason_code is not None:
            wave["reason_codes"][str(reason_code)] += 1
        if symbol is not None:
            wave["symbols"].add(str(symbol))
        if pside is not None:
            wave["psides"].add(str(pside))
        action_id = ids.get("action_id")
        if action_id is not None:
            wave["action_ids"].add(str(action_id))

    def to_dict(self) -> dict[str, Any]:
        ids = {
            key: _sorted_counter_items(counter)
            for key, counter in self.ids.items()
            if counter
        }
        order_waves = {
            wave_id: {
                "events": int(summary["events"]),
                "event_types": _sorted_counter(summary["event_types"]),
                "statuses": _sorted_counter(summary["statuses"]),
                "reason_codes": _sorted_counter(summary["reason_codes"]),
                "symbols": _sorted_values(summary["symbols"]),
                "psides": _sorted_values(summary["psides"]),
                "action_ids": _sorted_values(summary["action_ids"]),
            }
            for wave_id, summary in sorted(self.order_waves.items())
        }
        out: dict[str, Any] = {
            "matched_events": int(self.events),
            "event_types": _sorted_counter(self.event_types),
            "levels": _sorted_counter(self.levels),
            "sources": _sorted_counter(self.sources),
            "components": _sorted_counter(self.components),
            "tags": _sorted_counter(self.tags),
            "exchanges": _sorted_counter(self.exchanges),
            "users": _sorted_counter(self.users),
            "statuses": _sorted_counter(self.statuses),
            "reason_codes": _sorted_counter(self.reason_codes),
            "symbols": _sorted_values(self.symbols),
            "psides": _sorted_values(self.psides),
            "sides": _sorted_values(self.sides),
        }
        if self.first_ts is not None:
            out["first_ts"] = self.first_ts
        if self.last_ts is not None:
            out["last_ts"] = self.last_ts
        if ids:
            out["ids"] = ids
        if order_waves:
            out["order_waves"] = order_waves
        return out


def build_event_report(
    root: str | Path,
    *,
    cycle_id: str | None = None,
    event_type: str | Iterable[str] | None = None,
    level: str | Iterable[str] | None = None,
    exchange: str | Iterable[str] | None = None,
    user: str | Iterable[str] | None = None,
    bot_id: str | Iterable[str] | None = None,
    snapshot_id: str | Iterable[str] | None = None,
    plan_id: str | Iterable[str] | None = None,
    action_id: str | Iterable[str] | None = None,
    order_wave_id: str | Iterable[str] | None = None,
    remote_call_id: str | Iterable[str] | None = None,
    remote_call_group_id: str | Iterable[str] | None = None,
    symbol: str | Iterable[str] | None = None,
    pside: str | Iterable[str] | None = None,
    side: str | Iterable[str] | None = None,
    reason_code: str | Iterable[str] | None = None,
    status: str | Iterable[str] | None = None,
    source: str | Iterable[str] | None = None,
    component: str | Iterable[str] | None = None,
    tag: str | Iterable[str] | None = None,
    data_eq: str | Iterable[str] | None = None,
    debug_profile: str | Iterable[str] | None = None,
    problem_events: bool = False,
    hard_problem_events: bool = False,
    since_ms: int | None = None,
    until_ms: int | None = None,
    event_tail_lines: int = 0,
    max_event_files_per_bot: int = 0,
    limit: int = 200,
    include_data: bool = False,
    include_rotated: bool = False,
    timeline: bool = False,
    trace_summary: bool = False,
    order_trace: bool = False,
    cycle_trace: bool = False,
) -> dict[str, Any]:
    query_filters = build_event_query_filters(
        cycle_id=cycle_id,
        event_type=event_type,
        level=level,
        exchange=exchange,
        user=user,
        bot_id=bot_id,
        snapshot_id=snapshot_id,
        plan_id=plan_id,
        action_id=action_id,
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
        debug_profile=debug_profile,
        problem_events=problem_events,
        hard_problem_events=hard_problem_events,
        since_ms=since_ms,
        until_ms=until_ms,
    )
    since_filter = query_filters["since_ms"]
    until_filter = query_filters["until_ms"]
    window_enabled = since_filter is not None or until_filter is not None
    window_considered = 0
    window_skipped_before = 0
    window_skipped_after = 0
    window_invalid_ts = 0
    files_skipped_before_window = 0
    max_event_tail_lines = max(0, int(event_tail_lines))
    max_event_file_count_per_bot = max(0, int(max_event_files_per_bot))
    event_files_before_limit = 0
    event_files_skipped_by_limit = 0
    event_file_limit_groups = 0
    event_tail_limited_files = 0
    event_tail_skipped_lines = 0
    event_tail_skipped_lines_exact = True
    event_tail_skipped_bytes = 0
    event_tail_line_numbers_exact = True
    event_tail_methods: Counter[str] = Counter()
    scan_physical_bytes_read = 0
    scan_physical_bytes_known = True
    scan_decoded_bytes_read = 0
    scan_decoded_bytes_known = True
    scan_files_read = 0
    scan_read_methods: Counter[str] = Counter()
    issues: list[EventIssue] = []
    discovery = EventFileDiscovery(files=[])
    try:
        discovery = _discover_event_files(
            root,
            include_rotated=include_rotated,
            exchange=exchange,
            user=user,
            bot_id=bot_id,
        )
        files = discovery.files
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
    if since_filter is not None and files:
        window_files: list[Path] = []
        for event_file in files:
            file_mtime_ms = _file_mtime_ms(event_file)
            if file_mtime_ms is not None and file_mtime_ms < since_filter:
                files_skipped_before_window += 1
                continue
            window_files.append(event_file)
        files = window_files
    if max_event_file_count_per_bot and files:
        event_files_before_limit = len(files)
        files, event_files_skipped_by_limit, event_file_limit_groups = (
            _limit_recent_event_files_per_bot(files, max_event_file_count_per_bot)
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
    query_trace = _TraceSummaryBuilder()
    cycle_trace_summary = _TraceSummaryBuilder()
    max_events = max(0, int(limit))
    query_order_trace = _OrderTraceBuilder(event_limit=max_events)
    cycle_order_trace = _OrderTraceBuilder(event_limit=max_events)
    query_cycle_trace = _CycleTraceBuilder(
        event_limit=max_events,
        include_data=include_data,
    )
    cycle_cycle_trace = _CycleTraceBuilder(
        event_limit=max_events,
        include_data=include_data,
    )
    event_type_filter = query_filters["event_types"]
    level_filter = query_filters["levels"]
    exchange_filter = query_filters["exchanges"]
    user_filter = query_filters["users"]
    bot_filter = query_filters["bot_ids"]
    snapshot_filter = query_filters["snapshot_ids"]
    plan_filter = query_filters["plan_ids"]
    action_filter = query_filters["action_ids"]
    order_wave_filter = query_filters["order_wave_ids"]
    remote_call_filter = query_filters["remote_call_ids"]
    remote_call_group_filter = query_filters["remote_call_group_ids"]
    symbol_filter = query_filters["symbols"]
    pside_filter = query_filters["psides"]
    side_filter = query_filters["sides"]
    reason_code_filter = query_filters["reason_codes"]
    status_filter = query_filters["statuses"]
    source_filter = query_filters["sources"]
    component_filter = query_filters["components"]
    tag_filter = query_filters["tags"]
    debug_profile_filter = query_filters["debug_profiles"]
    data_eq_filters = query_filters["data_eq_filters"]
    problem_event_filter = bool(query_filters["problem_events"])
    hard_problem_event_filter = bool(query_filters["hard_problem_events"])
    has_non_cycle_filter = any(
        (
            event_type_filter,
            level_filter,
            exchange_filter,
            user_filter,
            bot_filter,
            snapshot_filter,
            plan_filter,
            action_filter,
            order_wave_filter,
            remote_call_filter,
            remote_call_group_filter,
            symbol_filter,
            pside_filter,
            side_filter,
            reason_code_filter,
            status_filter,
            source_filter,
            component_filter,
            tag_filter,
            debug_profile_filter,
            data_eq_filters,
            problem_event_filter,
            hard_problem_event_filter,
        )
    )
    trace_without_cycle = (
        (timeline or trace_summary or order_trace or cycle_trace) and cycle_id is None
    )
    has_query_filter = has_non_cycle_filter or window_enabled or trace_without_cycle

    scan_started_at = time.perf_counter()
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
                    if window_enabled:
                        record_ts = _record_ts(row)
                        if record_ts is None:
                            window_invalid_ts += 1
                            continue
                        if since_filter is not None and record_ts < since_filter:
                            window_skipped_before += 1
                            continue
                        if until_filter is not None and record_ts > until_filter:
                            window_skipped_after += 1
                            continue
                        window_considered += 1

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

                    path_scope = _monitor_path_exchange_user(path)
                    record_exchange = live_event.get("exchange") or row.get("exchange")
                    record_user = live_event.get("user") or row.get("user")
                    if path_scope is not None:
                        if record_exchange is None:
                            record_exchange = path_scope[0]
                        if record_user is None:
                            record_user = path_scope[1]
                    query_matches = event_matches_query_filters(
                        path=path,
                        row=row,
                        live_event=live_event,
                        filters=query_filters,
                    )

                    if has_query_filter and query_matches:
                        query_match_count += 1
                        if trace_summary:
                            query_trace.add(
                                row=row,
                                live_event=live_event,
                                event_type=record_event_type,
                                exchange=record_exchange,
                                user=record_user,
                            )
                        if order_trace:
                            query_order_trace.add(
                                path=path,
                                line_no=line_no,
                                row=row,
                                live_event=live_event,
                                event_type=record_event_type,
                                ids=ids,
                            )
                        if cycle_trace:
                            query_cycle_trace.add(
                                path=path,
                                line_no=line_no,
                                row=row,
                                live_event=live_event,
                                event_type=record_event_type,
                                ids=ids,
                                exchange=record_exchange,
                                user=record_user,
                            )
                        if len(query_events) < max_events:
                            query_events.append(
                                _compact_record(
                                    path=path,
                                    line_no=line_no,
                                    row=row,
                                    live_event=live_event,
                                    include_data=include_data,
                                    exchange=record_exchange,
                                    user=record_user,
                                )
                            )
                    if cycle_id is not None and query_matches:
                        cycle_match_count += 1
                        if trace_summary:
                            cycle_trace_summary.add(
                                row=row,
                                live_event=live_event,
                                event_type=record_event_type,
                                exchange=record_exchange,
                                user=record_user,
                            )
                        if order_trace:
                            cycle_order_trace.add(
                                path=path,
                                line_no=line_no,
                                row=row,
                                live_event=live_event,
                                event_type=record_event_type,
                                ids=ids,
                            )
                        if cycle_trace:
                            cycle_cycle_trace.add(
                                path=path,
                                line_no=line_no,
                                row=row,
                                live_event=live_event,
                                event_type=record_event_type,
                                ids=ids,
                                exchange=record_exchange,
                                user=record_user,
                            )
                        if len(cycle_events) < max_events:
                            cycle_events.append(
                                _compact_record(
                                    path=path,
                                    line_no=line_no,
                                    row=row,
                                    live_event=live_event,
                                    include_data=include_data,
                                    exchange=record_exchange,
                                    user=record_user,
                                )
                            )
            scan_files_read += 1
            if row_window.physical_bytes_read is None:
                scan_physical_bytes_known = False
            else:
                scan_physical_bytes_read += int(row_window.physical_bytes_read)
            if row_window.decoded_bytes_read is None:
                scan_decoded_bytes_known = False
            else:
                scan_decoded_bytes_read += int(row_window.decoded_bytes_read)
            scan_read_methods[str(row_window.method)] += 1
        except OSError as exc:
            scan_physical_bytes_known = False
            scan_decoded_bytes_known = False
            issues.append(
                EventIssue(str(path), None, "error", "read_failed", str(exc))
            )

    scan_elapsed_ms = round((time.perf_counter() - scan_started_at) * 1000, 3)

    if has_query_filter and not include_rotated and discovery.rotated_skipped > 0:
        issues.append(
            EventIssue(
                str(root),
                None,
                "warning",
                "current_only_rotated_segments_skipped",
                (
                    "query scanned current.ndjson only while rotated event segments "
                    "were present; rerun with --include-rotated for a complete "
                    "history query"
                ),
            )
        )

    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    report: dict[str, Any] = {
        "ok": error_count == 0,
        "root": str(Path(root).expanduser()),
        "include_rotated": bool(include_rotated),
        "files": [str(path) for path in files],
        "files_scanned": len(files),
        "scan_cost": {
            "elapsed_ms": scan_elapsed_ms,
            "physical_bytes_read": (
                scan_physical_bytes_read if scan_physical_bytes_known else None
            ),
            "physical_bytes_known": scan_physical_bytes_known,
            "decoded_bytes_read": (
                scan_decoded_bytes_read if scan_decoded_bytes_known else None
            ),
            "decoded_bytes_known": scan_decoded_bytes_known,
            "files_read": scan_files_read,
            "records_read": records_total,
            "read_methods": dict(sorted(scan_read_methods.items())),
        },
        "file_discovery": discovery.to_dict(),
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
            levels=level_filter,
            exchanges=exchange_filter,
            users=user_filter,
            bot_ids=bot_filter,
            snapshot_ids=snapshot_filter,
            plan_ids=plan_filter,
            action_ids=action_filter,
            order_wave_ids=order_wave_filter,
            remote_call_ids=remote_call_filter,
            remote_call_group_ids=remote_call_group_filter,
            symbols=symbol_filter,
            psides=pside_filter,
            sides=side_filter,
            reason_codes=reason_code_filter,
            statuses=status_filter,
            sources=source_filter,
            components=component_filter,
            tags=tag_filter,
            data_eq_filters=data_eq_filters,
            debug_profiles=debug_profile_filter,
            problem_events=problem_event_filter,
            hard_problem_events=hard_problem_event_filter,
            since_ms=since_filter,
            until_ms=until_filter,
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
        if trace_summary:
            report["query"]["trace_summary"] = query_trace.to_dict()
        if order_trace:
            report["query"]["order_trace"] = query_order_trace.to_dict()
        if cycle_trace:
            report["query"]["cycle_trace"] = query_cycle_trace.to_dict()
    if cycle_id is not None:
        cycle_filters = _filter_report(
            cycle_id=cycle_id,
            event_types=event_type_filter,
            levels=level_filter,
            exchanges=exchange_filter,
            users=user_filter,
            bot_ids=bot_filter,
            snapshot_ids=snapshot_filter,
            plan_ids=plan_filter,
            action_ids=action_filter,
            order_wave_ids=order_wave_filter,
            remote_call_ids=remote_call_filter,
            remote_call_group_ids=remote_call_group_filter,
            symbols=symbol_filter,
            psides=pside_filter,
            sides=side_filter,
            reason_codes=reason_code_filter,
            statuses=status_filter,
            sources=source_filter,
            components=component_filter,
            tags=tag_filter,
            data_eq_filters=data_eq_filters,
            debug_profiles=debug_profile_filter,
            problem_events=problem_event_filter,
            hard_problem_events=hard_problem_event_filter,
            since_ms=since_filter,
            until_ms=until_filter,
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
        if trace_summary:
            report["cycle"]["trace_summary"] = cycle_trace_summary.to_dict()
        if order_trace:
            report["cycle"]["order_trace"] = cycle_order_trace.to_dict()
        if cycle_trace:
            report["cycle"]["cycle_trace"] = cycle_cycle_trace.to_dict()
        if not event_type_filter:
            report["cycle"].pop("event_types", None)
    if window_enabled or max_event_tail_lines or max_event_file_count_per_bot:
        report["event_window"] = {
            "enabled": window_enabled,
            "since_ms": since_filter,
            "until_ms": until_filter,
            "events_considered": window_considered if window_enabled else live_events,
            "events_skipped_before": window_skipped_before,
            "events_skipped_after": window_skipped_after,
            "invalid_window_ts": window_invalid_ts,
            "files_skipped_before_window": files_skipped_before_window,
        }
        if max_event_file_count_per_bot:
            report["event_window"]["max_event_files_per_bot"] = (
                max_event_file_count_per_bot
            )
            report["event_window"]["event_file_limit_scope"] = "per_bot"
            report["event_window"]["event_file_limit_groups"] = event_file_limit_groups
            report["event_window"]["event_files_before_limit"] = event_files_before_limit
            report["event_window"]["event_files_skipped_by_limit"] = (
                event_files_skipped_by_limit
            )
            report["event_window"]["event_file_limit_order"] = (
                "current_then_recent_mtime"
            )
        if max_event_tail_lines:
            report["event_window"]["event_tail_lines"] = max_event_tail_lines
            report["event_window"]["event_tail_limited_files"] = event_tail_limited_files
            report["event_window"]["event_tail_skipped_lines"] = event_tail_skipped_lines
            report["event_window"]["event_tail_skipped_lines_exact"] = (
                event_tail_skipped_lines_exact
            )
            report["event_window"]["event_tail_skipped_bytes"] = event_tail_skipped_bytes
            report["event_window"]["event_tail_line_numbers_exact"] = (
                event_tail_line_numbers_exact
            )
            report["event_window"]["event_tail_methods"] = dict(
                sorted(event_tail_methods.items())
            )
    return report
