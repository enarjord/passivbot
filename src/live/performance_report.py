from __future__ import annotations

import gzip
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from live.event_bus import LIVE_EVENT_MONITOR_PAYLOAD_KEY
from live.event_query import discover_event_files
from live.smoke_report import _user_safe_display_path


GROUP_LIMIT = 80
SUMMARY_GROUP_LIMIT = 12


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _record_ts(row: dict[str, Any]) -> int | None:
    try:
        return int(row.get("ts"))
    except (TypeError, ValueError):
        return None


def _live_event_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    payload = row.get("payload")
    if not isinstance(payload, dict):
        return None
    live_event = payload.get(LIVE_EVENT_MONITOR_PAYLOAD_KEY)
    return live_event if isinstance(live_event, dict) else None


def _bot_key(row: dict[str, Any], live_event: dict[str, Any]) -> str:
    exchange = live_event.get("exchange") or row.get("exchange") or "-"
    user = live_event.get("user") or row.get("user") or "-"
    return f"{exchange}/{user}"


def _string_filter(values: list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    if not values:
        return set()
    return {str(value) for value in values if str(value)}


def _matches_filters(
    row: dict[str, Any],
    live_event: dict[str, Any],
    *,
    bot_filters: set[str],
    exchange_filters: set[str],
    user_filters: set[str],
) -> bool:
    if bot_filters and _bot_key(row, live_event) not in bot_filters:
        return False
    exchange = str(live_event.get("exchange") or row.get("exchange") or "")
    if exchange_filters and exchange not in exchange_filters:
        return False
    user = str(live_event.get("user") or row.get("user") or "")
    if user_filters and user not in user_filters:
        return False
    return True


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _non_negative_ms(value: Any) -> int | None:
    number = _finite_float(value)
    if number is None or number < 0:
        return None
    return int(round(number))


def _elapsed_s_to_ms(value: Any) -> int | None:
    number = _finite_float(value)
    if number is None or number < 0:
        return None
    return int(round(number * 1000.0))


def _percentile(sorted_values: list[int], pct: float) -> int | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return int(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return int(sorted_values[lower])
    weight = rank - lower
    return int(round(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight))


class _MetricGroup:
    def __init__(
        self,
        *,
        bot: str,
        operation: str,
        component: str | None,
        event_type: str,
        trading_impact: str,
        timing_kind: str,
    ) -> None:
        self.bot = bot
        self.operation = operation
        self.component = component
        self.event_type = event_type
        self.trading_impact = trading_impact
        self.timing_kind = timing_kind
        self.values: list[int] = []
        self.statuses: Counter[str] = Counter()
        self.reason_codes: Counter[str] = Counter()
        self.symbols: Counter[str] = Counter()
        self.latest_ts: int | None = None

    def add(
        self,
        value_ms: int,
        *,
        row: dict[str, Any],
        live_event: dict[str, Any],
    ) -> None:
        self.values.append(int(value_ms))
        status = live_event.get("status")
        if status is not None:
            self.statuses[str(status)] += 1
        reason_code = live_event.get("reason_code")
        if reason_code is not None:
            self.reason_codes[str(reason_code)] += 1
        symbol = live_event.get("symbol") or row.get("symbol")
        if symbol is not None:
            self.symbols[str(symbol)] += 1
        ts = _record_ts(row)
        if ts is not None and (self.latest_ts is None or ts > self.latest_ts):
            self.latest_ts = ts

    def to_dict(self) -> dict[str, Any]:
        values = sorted(self.values)
        mean_value = statistics.fmean(values) if values else None
        return {
            key: value
            for key, value in {
                "bot": self.bot,
                "operation": self.operation,
                "component": self.component,
                "event_type": self.event_type,
                "trading_impact": self.trading_impact,
                "timing_kind": self.timing_kind,
                "count": len(values),
                "min_ms": int(values[0]) if values else None,
                "max_ms": int(values[-1]) if values else None,
                "mean_ms": int(round(mean_value)) if mean_value is not None else None,
                "median_ms": int(round(statistics.median(values))) if values else None,
                "p95_ms": _percentile(values, 0.95),
                "latest_ts": self.latest_ts,
                "statuses": dict(sorted(self.statuses.items())),
                "reason_codes": dict(sorted(self.reason_codes.items())),
                "symbols_sample": [
                    symbol
                    for symbol, _count in sorted(
                        self.symbols.items(), key=lambda item: (-item[1], item[0])
                    )[:10]
                ],
            }.items()
            if value not in (None, {}, [])
        }


class _PerformanceAccumulator:
    def __init__(self) -> None:
        self.groups: dict[tuple[str, str, str | None, str], _MetricGroup] = {}
        self.trading_impact_counts: Counter[str] = Counter()

    def add(
        self,
        *,
        row: dict[str, Any],
        live_event: dict[str, Any],
        operation: str,
        value_ms: int | None,
        trading_impact: str,
        timing_kind: str = "duration",
    ) -> None:
        if value_ms is None:
            return
        bot = _bot_key(row, live_event)
        event_type = str(live_event.get("event_type") or row.get("kind") or "unknown")
        component = live_event.get("component")
        component_key = str(component) if component is not None else None
        key = (bot, operation, component_key, timing_kind)
        group = self.groups.get(key)
        if group is None:
            group = _MetricGroup(
                bot=bot,
                operation=operation,
                component=component_key,
                event_type=event_type,
                trading_impact=trading_impact,
                timing_kind=timing_kind,
            )
            self.groups[key] = group
        group.add(value_ms, row=row, live_event=live_event)
        self.trading_impact_counts[trading_impact] += 1

    def to_dict(self, *, group_limit: int = GROUP_LIMIT) -> dict[str, Any]:
        groups = sorted(
            (group.to_dict() for group in self.groups.values()),
            key=lambda item: (
                -int(item.get("p95_ms", 0) or 0),
                -int(item.get("max_ms", 0) or 0),
                -int(item.get("count", 0) or 0),
                str(item.get("bot") or ""),
                str(item.get("operation") or ""),
            ),
        )
        return {
            "total_groups": len(groups),
            "groups_truncated": len(groups) > int(group_limit),
            "trading_impact_counts": dict(sorted(self.trading_impact_counts.items())),
            "groups": groups[: max(0, int(group_limit))],
        }


def _event_cycle_id(live_event: dict[str, Any]) -> str | None:
    ids = live_event.get("ids")
    if not isinstance(ids, dict):
        return None
    cycle_id = ids.get("cycle_id")
    if cycle_id is None:
        return None
    return str(cycle_id)


def _minute_boundary_ms(timestamp_ms: int) -> int:
    return int(timestamp_ms) - (int(timestamp_ms) % 60_000)


def _decision_milestone(event_type: str) -> str | None:
    return {
        "cycle.started": "cycle_started",
        "rust_orchestrator.called": "rust_called",
        "rust_orchestrator.returned": "rust_returned",
        "action.planned": "action_planned",
        "order_wave.started": "order_wave_started",
        "execution.create_sent": "first_write_sent",
        "execution.cancel_sent": "first_write_sent",
        "execution.confirmation_requested": "confirmation_requested",
        "execution.confirmation_satisfied": "confirmation_satisfied",
        "cycle.completed": "cycle_completed",
    }.get(event_type)


def _decision_trading_impact(milestone: str) -> str:
    if milestone in {"first_write_sent", "confirmation_requested", "confirmation_satisfied"}:
        return "blocks_exchange_actions"
    if milestone == "cycle_completed":
        return "blocks_next_cycle"
    return "blocks_cycle_decision"


class _DecisionBoundaryAccumulator:
    def __init__(self) -> None:
        self.groups: dict[tuple[str, str], _MetricGroup] = {}
        self.cycle_boundaries: dict[tuple[str, str], int] = {}
        self.cycles_seen: set[tuple[str, str]] = set()
        self.cycles_with_write: set[tuple[str, str]] = set()
        self.events_without_cycle_id = 0

    def add(self, *, row: dict[str, Any], live_event: dict[str, Any]) -> None:
        event_type = str(live_event.get("event_type") or row.get("kind") or "")
        milestone = _decision_milestone(event_type)
        if milestone is None:
            return
        timestamp_ms = _record_ts(row)
        if timestamp_ms is None:
            return
        cycle_id = _event_cycle_id(live_event)
        if not cycle_id:
            self.events_without_cycle_id += 1
            return
        bot = _bot_key(row, live_event)
        cycle_key = (bot, cycle_id)
        self.cycles_seen.add(cycle_key)
        if milestone == "cycle_started" or cycle_key not in self.cycle_boundaries:
            self.cycle_boundaries[cycle_key] = _minute_boundary_ms(timestamp_ms)
        if milestone == "first_write_sent":
            self.cycles_with_write.add(cycle_key)
        lag_ms = max(0, int(timestamp_ms) - int(self.cycle_boundaries[cycle_key]))
        group_key = (bot, milestone)
        group = self.groups.get(group_key)
        if group is None:
            group = _MetricGroup(
                bot=bot,
                operation=f"decision_boundary.{milestone}",
                component="decision_boundary",
                event_type=event_type,
                trading_impact=_decision_trading_impact(milestone),
                timing_kind="lag_from_minute_boundary",
            )
            self.groups[group_key] = group
        group.add(lag_ms, row=row, live_event=live_event)

    def to_dict(self, *, group_limit: int = GROUP_LIMIT) -> dict[str, Any]:
        groups = sorted(
            (group.to_dict() for group in self.groups.values()),
            key=lambda item: (
                -int(item.get("p95_ms", 0) or 0),
                -int(item.get("max_ms", 0) or 0),
                -int(item.get("count", 0) or 0),
                str(item.get("bot") or ""),
                str(item.get("operation") or ""),
            ),
        )
        return {
            "minute_ms": 60_000,
            "cycles": len(self.cycles_seen),
            "cycles_with_write": len(self.cycles_with_write),
            "events_without_cycle_id": int(self.events_without_cycle_id),
            "total_groups": len(groups),
            "groups_truncated": len(groups) > int(group_limit),
            "groups": groups[: max(0, int(group_limit))],
        }


def _timings_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out = {}
    for key, raw in value.items():
        duration = _non_negative_ms(raw)
        if duration is not None:
            out[str(key)] = duration
    return out


def _summary_timing_maps(value: Any) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, dict[str, int]] = {}
    for surface, summary in value.items():
        if not isinstance(summary, dict):
            continue
        surface_summary = {}
        for field in ("min", "mean", "max"):
            duration = _non_negative_ms(summary.get(field))
            if duration is not None:
                surface_summary[field] = duration
        if surface_summary:
            out[str(surface)] = surface_summary
    return out


def _remote_call_operation(data: dict[str, Any], live_event: dict[str, Any]) -> str:
    component = str(live_event.get("component") or "")
    prefix = "remote_call"
    if "authoritative" in component:
        prefix = "remote_call.authoritative"
    elif "candle" in component:
        prefix = "remote_call.candle"
    surface = data.get("surface")
    kind = data.get("kind")
    reason_code = live_event.get("reason_code")
    if surface is not None:
        return f"{prefix}.{surface}"
    if kind is not None:
        return f"{prefix}.{kind}"
    if reason_code is not None:
        return f"{prefix}.{reason_code}"
    return prefix


def _trading_impact_for_event(event_type: str, operation: str) -> str:
    if event_type == "cycle.completed":
        if operation == "cycle.phase.monitor_flush":
            return "diagnostics_only"
        if operation.startswith("cycle.phase."):
            return "blocks_cycle_decision"
        return "blocks_next_cycle"
    if event_type == "state.refresh_timing":
        return "blocks_exchange_actions"
    if event_type.startswith("remote_call."):
        if "authoritative" in operation:
            return "blocks_exchange_actions"
        if "ccxt_fetch_ohlcv" in operation or "candle" in operation:
            return "blocks_indicator_readiness"
        return "exchange_io"
    if event_type.startswith("hsl.replay."):
        return "blocks_or_delays_hsl_readiness"
    if event_type == "bot.startup_timing":
        return "blocks_startup_readiness"
    return "observability"


def _add_event_timings(
    accumulator: _PerformanceAccumulator,
    *,
    row: dict[str, Any],
    live_event: dict[str, Any],
) -> None:
    event_type = str(live_event.get("event_type") or row.get("kind") or "")
    data = live_event.get("data") if isinstance(live_event.get("data"), dict) else {}

    if event_type == "cycle.completed":
        elapsed_ms = _non_negative_ms(data.get("elapsed_ms"))
        accumulator.add(
            row=row,
            live_event=live_event,
            operation="cycle.elapsed",
            value_ms=elapsed_ms,
            trading_impact=_trading_impact_for_event(event_type, "cycle.elapsed"),
        )
        for phase, duration_ms in _timings_map(data.get("timings_ms")).items():
            operation = f"cycle.phase.{phase}"
            accumulator.add(
                row=row,
                live_event=live_event,
                operation=operation,
                value_ms=duration_ms,
                trading_impact=_trading_impact_for_event(event_type, operation),
            )
        return

    if event_type == "state.refresh_timing":
        for field in ("wall_ms", "surface_max_ms", "surface_sum_ms", "residual_ms"):
            operation = f"state_refresh.{field.removesuffix('_ms')}"
            accumulator.add(
                row=row,
                live_event=live_event,
                operation=operation,
                value_ms=_non_negative_ms(data.get(field)),
                trading_impact=_trading_impact_for_event(event_type, operation),
            )
        for surface, duration_ms in _timings_map(data.get("timings_ms")).items():
            operation = f"state_refresh.surface.{surface}"
            accumulator.add(
                row=row,
                live_event=live_event,
                operation=operation,
                value_ms=duration_ms,
                trading_impact=_trading_impact_for_event(event_type, operation),
            )
        for surface, summary in _summary_timing_maps(data.get("surfaces_ms")).items():
            for field, duration_ms in summary.items():
                operation = f"state_refresh.surface.{surface}.{field}"
                accumulator.add(
                    row=row,
                    live_event=live_event,
                    operation=operation,
                    value_ms=duration_ms,
                    trading_impact=_trading_impact_for_event(event_type, operation),
                    timing_kind="summary_stat",
                )
        return

    if event_type.startswith("remote_call."):
        operation = _remote_call_operation(data, live_event)
        accumulator.add(
            row=row,
            live_event=live_event,
            operation=operation,
            value_ms=_non_negative_ms(data.get("elapsed_ms")),
            trading_impact=_trading_impact_for_event(event_type, operation),
        )
        return

    if event_type.startswith("hsl.replay."):
        stage = data.get("stage")
        operation = "hsl_replay.elapsed"
        if stage is not None:
            operation = f"hsl_replay.{stage}.elapsed"
        accumulator.add(
            row=row,
            live_event=live_event,
            operation=operation,
            value_ms=_elapsed_s_to_ms(data.get("elapsed_s")),
            trading_impact=_trading_impact_for_event(event_type, operation),
            timing_kind="cumulative",
        )
        return

    if event_type == "bot.startup_timing":
        stage = data.get("stage") or data.get("phase") or "startup"
        operation = f"startup.{stage}"
        value_ms = _non_negative_ms(data.get("elapsed_ms"))
        if value_ms is None:
            value_ms = _elapsed_s_to_ms(data.get("elapsed_s"))
        accumulator.add(
            row=row,
            live_event=live_event,
            operation=operation,
            value_ms=value_ms,
            trading_impact=_trading_impact_for_event(event_type, operation),
        )


def build_live_performance_report(
    root: str | Path = "monitor",
    *,
    since_ms: int | None = None,
    until_ms: int | None = None,
    include_rotated: bool = False,
    group_limit: int = GROUP_LIMIT,
    bot_filters: list[str] | tuple[str, ...] | set[str] | None = None,
    exchange_filters: list[str] | tuple[str, ...] | set[str] | None = None,
    user_filters: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, Any]:
    since_filter = int(since_ms) if since_ms is not None else None
    until_filter = int(until_ms) if until_ms is not None else None
    if (
        since_filter is not None
        and until_filter is not None
        and since_filter > until_filter
    ):
        raise ValueError("since_ms must be <= until_ms")
    window_enabled = since_filter is not None or until_filter is not None
    event_window = {
        "enabled": bool(window_enabled),
        "since_ms": since_filter,
        "until_ms": until_filter,
        "events_considered": 0,
        "events_skipped_before": 0,
        "events_skipped_after": 0,
        "invalid_window_ts": 0,
    }
    bot_filter_set = _string_filter(bot_filters)
    exchange_filter_set = _string_filter(exchange_filters)
    user_filter_set = _string_filter(user_filters)
    filters = {
        "enabled": bool(bot_filter_set or exchange_filter_set or user_filter_set),
        "bots": sorted(bot_filter_set),
        "exchanges": sorted(exchange_filter_set),
        "users": sorted(user_filter_set),
        "events_skipped": 0,
    }
    issues: list[dict[str, Any]] = []
    try:
        files = discover_event_files(root, include_rotated=include_rotated)
    except FileNotFoundError as exc:
        files = []
        issues.append(
            {
                "path": _user_safe_display_path(root),
                "line": None,
                "severity": "error",
                "code": "path_not_found",
                "message": getattr(exc, "strerror", None) or "path not found",
            }
        )
    if not files and not issues:
        issues.append(
            {
                "path": _user_safe_display_path(root),
                "line": None,
                "severity": "error",
                "code": "no_event_files",
                "message": "no event NDJSON files found",
            }
        )

    accumulator = _PerformanceAccumulator()
    decision_boundary = _DecisionBoundaryAccumulator()
    records_total = 0
    live_events = 0
    legacy_events = 0
    event_types: Counter[str] = Counter()
    bots: Counter[str] = Counter()
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
                            {
                                "path": _user_safe_display_path(path),
                                "line": int(line_no),
                                "severity": "error",
                                "code": "invalid_json",
                                "message": str(exc),
                            }
                        )
                        continue
                    if not isinstance(row, dict):
                        issues.append(
                            {
                                "path": _user_safe_display_path(path),
                                "line": int(line_no),
                                "severity": "error",
                                "code": "invalid_record",
                                "message": "event row is not a JSON object",
                            }
                        )
                        continue
                    live_event = _live_event_payload(row)
                    if live_event is None:
                        legacy_events += 1
                        continue
                    if not _matches_filters(
                        row,
                        live_event,
                        bot_filters=bot_filter_set,
                        exchange_filters=exchange_filter_set,
                        user_filters=user_filter_set,
                    ):
                        filters["events_skipped"] += 1
                        continue
                    live_events += 1
                    if window_enabled:
                        record_ts = _record_ts(row)
                        if record_ts is None:
                            event_window["invalid_window_ts"] += 1
                            continue
                        if since_filter is not None and record_ts < since_filter:
                            event_window["events_skipped_before"] += 1
                            continue
                        if until_filter is not None and record_ts > until_filter:
                            event_window["events_skipped_after"] += 1
                            continue
                        event_window["events_considered"] += 1
                    event_type = str(live_event.get("event_type") or row.get("kind") or "unknown")
                    event_types[event_type] += 1
                    bots[_bot_key(row, live_event)] += 1
                    _add_event_timings(
                        accumulator,
                        row=row,
                        live_event=live_event,
                    )
                    decision_boundary.add(row=row, live_event=live_event)
        except OSError as exc:
            issues.append(
                {
                    "path": _user_safe_display_path(path),
                    "line": None,
                    "severity": "error",
                    "code": "read_failed",
                    "message": getattr(exc, "strerror", None) or type(exc).__name__,
                }
            )

    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")
    report = {
        "ok": error_count == 0,
        "root": _user_safe_display_path(root),
        "include_rotated": bool(include_rotated),
        "files": [_user_safe_display_path(path) for path in files],
        "files_scanned": len(files),
        "records_total": int(records_total),
        "live_events": int(live_events),
        "legacy_events": int(legacy_events),
        "issues": issues,
        "error_count": int(error_count),
        "warning_count": int(warning_count),
        "event_types": dict(sorted(event_types.items())),
        "bots": [
            {"bot": bot, "events": int(count)}
            for bot, count in sorted(bots.items(), key=lambda item: (-item[1], item[0]))
        ],
        "performance": accumulator.to_dict(group_limit=group_limit),
        "decision_boundary_lag": decision_boundary.to_dict(group_limit=group_limit),
    }
    if window_enabled:
        report["event_window"] = event_window
    if filters["enabled"]:
        report["filters"] = filters
    return report


def summarize_live_performance_report(
    report: dict[str, Any],
    *,
    group_limit: int = SUMMARY_GROUP_LIMIT,
) -> dict[str, Any]:
    performance = report.get("performance") if isinstance(report.get("performance"), dict) else {}
    groups = performance.get("groups") if isinstance(performance.get("groups"), list) else []
    summary = {
        "ok": bool(report.get("ok")),
        "root": report.get("root"),
        "include_rotated": bool(report.get("include_rotated")),
        "files_scanned": int(report.get("files_scanned") or 0),
        "records_total": int(report.get("records_total") or 0),
        "live_events": int(report.get("live_events") or 0),
        "legacy_events": int(report.get("legacy_events") or 0),
        "error_count": int(report.get("error_count") or 0),
        "warning_count": int(report.get("warning_count") or 0),
        "bots": report.get("bots") or [],
        "performance": {
            "total_groups": int(performance.get("total_groups") or 0),
            "groups_truncated": bool(performance.get("groups_truncated")),
            "trading_impact_counts": performance.get("trading_impact_counts") or {},
            "groups": groups[: max(0, int(group_limit))],
        },
    }
    if isinstance(report.get("decision_boundary_lag"), dict):
        decision_lag = report["decision_boundary_lag"]
        decision_groups = (
            decision_lag.get("groups") if isinstance(decision_lag.get("groups"), list) else []
        )
        summary["decision_boundary_lag"] = {
            "minute_ms": int(decision_lag.get("minute_ms") or 60_000),
            "cycles": int(decision_lag.get("cycles") or 0),
            "cycles_with_write": int(decision_lag.get("cycles_with_write") or 0),
            "events_without_cycle_id": int(decision_lag.get("events_without_cycle_id") or 0),
            "total_groups": int(decision_lag.get("total_groups") or 0),
            "groups_truncated": bool(decision_lag.get("groups_truncated")),
            "groups": decision_groups[: max(0, int(group_limit))],
        }
    if report.get("event_window") is not None:
        summary["event_window"] = report.get("event_window")
    if report.get("filters") is not None:
        summary["filters"] = report.get("filters")
    if report.get("issues"):
        summary["issues"] = report.get("issues")
    return summary
