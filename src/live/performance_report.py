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

_NON_BLOCKING_IMPACTS = {"diagnostics_only", "observability"}
_BLOCKING_SCOPE_BY_IMPACT = {
    "blocks_or_delays_hsl_readiness": "delays_protective_readiness",
    "blocks_exchange_actions": "delays_exchange_actions",
    "blocks_cycle_decision": "delays_cycle_decision",
    "blocks_indicator_readiness": "delays_indicator_readiness",
    "blocks_startup_readiness": "delays_startup_readiness",
    "blocks_next_cycle": "delays_next_cycle",
    "exchange_io": "exchange_io",
}
_RESOURCE_PRESSURE_FIELDS = (
    "rss_bytes",
    "memory_percent",
    "open_fds",
    "loadavg_1m",
    "loadavg_5m",
    "loadavg_15m",
    "cpu_count",
    "uptime_ms",
    "last_loop_duration_ms",
    "errors_last_hour",
    "ws_reconnects",
    "rate_limits",
    "event_queue_depth",
    "event_queue_maxsize",
    "event_queue_unfinished_tasks",
    "event_dropped_total",
    "event_sink_error_total",
    "event_degraded_count",
)
_RESOURCE_PRESSURE_COUNTER_FIELDS = (
    "event_drop_counts",
    "event_sink_error_counts",
)
_RESOURCE_PRESSURE_BOOL_FIELDS = (
    "event_pipeline_stopping",
    "event_pipeline_worker_alive",
)


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


def _non_negative_number(value: Any) -> float | int | None:
    number = _finite_float(value)
    if number is None or number < 0:
        return None
    if float(number).is_integer():
        return int(number)
    return float(number)


def _safe_counter_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, raw in value.items():
        number = _finite_float(raw)
        if number is not None and number >= 0:
            out[str(key)] = int(number)
    return out


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

    def groups_list(self) -> list[dict[str, Any]]:
        return sorted(
            (group.to_dict() for group in self.groups.values()),
            key=lambda item: (
                -int(item.get("p95_ms", 0) or 0),
                -int(item.get("max_ms", 0) or 0),
                -int(item.get("count", 0) or 0),
                str(item.get("bot") or ""),
                str(item.get("operation") or ""),
            ),
        )

    def to_dict(self, *, group_limit: int = GROUP_LIMIT) -> dict[str, Any]:
        groups = self.groups_list()
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


def _cycle_number(cycle_id: str | None) -> int | None:
    if not cycle_id:
        return None
    text = str(cycle_id)
    if text.startswith("cy_"):
        text = text[3:]
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


class _CycleScopeTracker:
    def __init__(self) -> None:
        self.generation_by_bot: dict[str, int] = {}
        self.last_cycle_number_by_bot: dict[str, int] = {}

    def observe(self, *, row: dict[str, Any], live_event: dict[str, Any]) -> int:
        bot = _bot_key(row, live_event)
        generation = int(self.generation_by_bot.get(bot, 0))
        event_type = str(live_event.get("event_type") or row.get("kind") or "")
        if event_type == "bot.started":
            generation += 1
            self.generation_by_bot[bot] = generation
            self.last_cycle_number_by_bot.pop(bot, None)
            return generation
        if event_type == "cycle.started":
            cycle_number = _cycle_number(_event_cycle_id(live_event))
            if cycle_number is not None:
                previous = self.last_cycle_number_by_bot.get(bot)
                if previous is not None and cycle_number <= previous:
                    generation += 1
                    self.generation_by_bot[bot] = generation
                self.last_cycle_number_by_bot[bot] = cycle_number
        return generation


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
        self.cycle_boundaries: dict[tuple[str, int, str], int] = {}
        self.cycles_seen: set[tuple[str, int, str]] = set()
        self.cycles_with_write: set[tuple[str, int, str]] = set()
        self.events_without_cycle_id = 0

    def add(
        self,
        *,
        row: dict[str, Any],
        live_event: dict[str, Any],
        cycle_scope: int,
    ) -> None:
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
        cycle_key = (bot, int(cycle_scope), cycle_id)
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

    def groups_list(self) -> list[dict[str, Any]]:
        return sorted(
            (group.to_dict() for group in self.groups.values()),
            key=lambda item: (
                -int(item.get("p95_ms", 0) or 0),
                -int(item.get("max_ms", 0) or 0),
                -int(item.get("count", 0) or 0),
                str(item.get("bot") or ""),
                str(item.get("operation") or ""),
            ),
        )

    def to_dict(self, *, group_limit: int = GROUP_LIMIT) -> dict[str, Any]:
        groups = self.groups_list()
        return {
            "minute_ms": 60_000,
            "cycles": len(self.cycles_seen),
            "cycles_with_write": len(self.cycles_with_write),
            "events_without_cycle_id": int(self.events_without_cycle_id),
            "total_groups": len(groups),
            "groups_truncated": len(groups) > int(group_limit),
            "groups": groups[: max(0, int(group_limit))],
        }


def _cycle_id_from_snapshot_data(data: dict[str, Any]) -> str | None:
    cycle_id = data.get("cycle_id")
    if cycle_id is None:
        return None
    text = str(cycle_id)
    if text.startswith("cy_"):
        return text
    return f"cy_{text}"


class _InputStalenessAccumulator:
    def __init__(self) -> None:
        self.groups: dict[tuple[str, str, str], _MetricGroup] = {}
        self.packet_received_ts: dict[tuple[str, int, str, str], int] = {}
        self.snapshot_ts_by_cycle: dict[tuple[str, int, str], int] = {}
        self.ema_ts_by_cycle: dict[tuple[str, int, str], int] = {}
        self.snapshots_seen = 0
        self.rust_calls_seen = 0
        self.packet_refs_missing = 0
        self.rust_calls_missing_snapshot = 0
        self.rust_calls_missing_ema = 0

    def _add_group(
        self,
        *,
        row: dict[str, Any],
        live_event: dict[str, Any],
        operation: str,
        value_ms: int,
        timing_kind: str,
        trading_impact: str,
    ) -> None:
        bot = _bot_key(row, live_event)
        group_key = (bot, operation, timing_kind)
        group = self.groups.get(group_key)
        if group is None:
            group = _MetricGroup(
                bot=bot,
                operation=operation,
                component="input_staleness",
                event_type=str(live_event.get("event_type") or row.get("kind") or "unknown"),
                trading_impact=trading_impact,
                timing_kind=timing_kind,
            )
            self.groups[group_key] = group
        group.add(value_ms, row=row, live_event=live_event)

    def add(
        self,
        *,
        row: dict[str, Any],
        live_event: dict[str, Any],
        cycle_scope: int,
    ) -> None:
        event_type = str(live_event.get("event_type") or row.get("kind") or "")
        data = live_event.get("data") if isinstance(live_event.get("data"), dict) else {}
        timestamp_ms = _record_ts(row)
        if timestamp_ms is None:
            return
        bot = _bot_key(row, live_event)
        if event_type == "data_packet.updated":
            kind = data.get("kind")
            revision = data.get("revision")
            received_ts = _non_negative_ms(data.get("response_received_ts_ms"))
            if kind is None or revision is None or received_ts is None:
                return
            self.packet_received_ts[(bot, int(cycle_scope), str(kind), str(revision))] = int(
                received_ts
            )
            return
        if event_type == "snapshot.built":
            self.snapshots_seen += 1
            cycle_id = _cycle_id_from_snapshot_data(data)
            if cycle_id:
                self.snapshot_ts_by_cycle[(bot, int(cycle_scope), cycle_id)] = int(timestamp_ms)
            packets = data.get("data_packets")
            if not isinstance(packets, list):
                return
            for packet in packets:
                if not isinstance(packet, dict):
                    continue
                kind = packet.get("kind")
                revision = packet.get("revision")
                if kind is None or revision is None:
                    continue
                received_ts = self.packet_received_ts.get(
                    (bot, int(cycle_scope), str(kind), str(revision))
                )
                if received_ts is None:
                    self.packet_refs_missing += 1
                    continue
                self._add_group(
                    row=row,
                    live_event=live_event,
                    operation=f"input_staleness.data_packet.{kind}",
                    value_ms=max(0, int(timestamp_ms) - int(received_ts)),
                    timing_kind="age_at_snapshot",
                    trading_impact="blocks_exchange_actions",
                )
            return
        if event_type == "ema.bundle.completed":
            cycle_id = _event_cycle_id(live_event)
            if cycle_id:
                self.ema_ts_by_cycle[(bot, int(cycle_scope), cycle_id)] = int(timestamp_ms)
            return
        if event_type == "rust_orchestrator.called":
            self.rust_calls_seen += 1
            cycle_id = _event_cycle_id(live_event)
            if not cycle_id:
                return
            snapshot_ts = self.snapshot_ts_by_cycle.get((bot, int(cycle_scope), cycle_id))
            if snapshot_ts is None:
                self.rust_calls_missing_snapshot += 1
            else:
                self._add_group(
                    row=row,
                    live_event=live_event,
                    operation="input_staleness.snapshot_to_rust",
                    value_ms=max(0, int(timestamp_ms) - int(snapshot_ts)),
                    timing_kind="age_at_rust_call",
                    trading_impact="blocks_cycle_decision",
                )
            ema_ts = self.ema_ts_by_cycle.get((bot, int(cycle_scope), cycle_id))
            if ema_ts is None:
                self.rust_calls_missing_ema += 1
            else:
                self._add_group(
                    row=row,
                    live_event=live_event,
                    operation="input_staleness.ema_bundle_to_rust",
                    value_ms=max(0, int(timestamp_ms) - int(ema_ts)),
                    timing_kind="age_at_rust_call",
                    trading_impact="blocks_indicator_readiness",
                )

    def groups_list(self) -> list[dict[str, Any]]:
        return sorted(
            (group.to_dict() for group in self.groups.values()),
            key=lambda item: (
                -int(item.get("p95_ms", 0) or 0),
                -int(item.get("max_ms", 0) or 0),
                -int(item.get("count", 0) or 0),
                str(item.get("bot") or ""),
                str(item.get("operation") or ""),
            ),
        )

    def to_dict(self, *, group_limit: int = GROUP_LIMIT) -> dict[str, Any]:
        groups = self.groups_list()
        return {
            "snapshots_seen": int(self.snapshots_seen),
            "rust_calls_seen": int(self.rust_calls_seen),
            "packet_refs_missing": int(self.packet_refs_missing),
            "rust_calls_missing_snapshot": int(self.rust_calls_missing_snapshot),
            "rust_calls_missing_ema": int(self.rust_calls_missing_ema),
            "total_groups": len(groups),
            "groups_truncated": len(groups) > int(group_limit),
            "groups": groups[: max(0, int(group_limit))],
        }


def _startup_elapsed_ms(data: dict[str, Any]) -> int | None:
    value_ms = _non_negative_ms(data.get("elapsed_ms"))
    if value_ms is not None:
        return value_ms
    return _elapsed_s_to_ms(data.get("elapsed_s"))


class _StartupReadinessAccumulator:
    def __init__(self) -> None:
        self.bots: dict[str, dict[str, Any]] = {}

    def _bot_state(self, *, row: dict[str, Any], live_event: dict[str, Any]) -> dict[str, Any]:
        bot = _bot_key(row, live_event)
        state = self.bots.get(bot)
        if state is None:
            state = {
                "bot": bot,
                "startup_phases_ms": {},
                "latest_ts": None,
            }
            self.bots[bot] = state
        ts = _record_ts(row)
        if ts is not None and (
            state.get("latest_ts") is None or int(ts) > int(state["latest_ts"])
        ):
            state["latest_ts"] = int(ts)
        return state

    def add(self, *, row: dict[str, Any], live_event: dict[str, Any]) -> None:
        event_type = str(live_event.get("event_type") or row.get("kind") or "")
        data = live_event.get("data") if isinstance(live_event.get("data"), dict) else {}
        ts = _record_ts(row)
        if event_type == "bot.started":
            state = self._bot_state(row=row, live_event=live_event)
            bot = str(state["bot"])
            state.clear()
            state.update(
                {
                    "bot": bot,
                    "startup_phases_ms": {},
                    "latest_ts": int(ts) if ts is not None else None,
                }
            )
            if ts is not None:
                state["bot_started_ts"] = int(ts)
            state["lifecycle_status"] = "started"
            return
        if event_type == "bot.ready":
            state = self._bot_state(row=row, live_event=live_event)
            if ts is not None:
                state["bot_ready_ts"] = int(ts)
            state["lifecycle_status"] = "ready"
            return
        if event_type == "bot.startup_timing":
            state = self._bot_state(row=row, live_event=live_event)
            stage = data.get("stage") or data.get("phase") or "startup"
            elapsed_ms = _startup_elapsed_ms(data)
            if elapsed_ms is not None:
                state["startup_phases_ms"][str(stage)] = int(elapsed_ms)
            return
        if event_type.startswith("hsl.replay."):
            state = self._bot_state(row=row, live_event=live_event)
            hsl_state = state.get("hsl_replay")
            if not isinstance(hsl_state, dict):
                hsl_state = {}
                state["hsl_replay"] = hsl_state
            if ts is not None:
                hsl_state["latest_ts"] = int(ts)
            hsl_state["event_type"] = event_type
            hsl_state["status"] = live_event.get("status")
            hsl_state["reason_code"] = live_event.get("reason_code")
            for key in (
                "signal_mode",
                "stage",
                "pairs",
                "held_pairs",
                "cooldown_pairs",
                "required_pairs",
                "timeline_rows",
                "applied_rows",
                "total_applied_rows",
                "skipped_pairs",
                "rows_per_second",
                "elapsed_s",
                "full_elapsed_s",
                "startup_blocking_elapsed_s",
            ):
                if key in data:
                    hsl_state[key] = data[key]

    def to_dict(self, *, group_limit: int = GROUP_LIMIT) -> dict[str, Any]:
        bot_items = []
        ready_count = 0
        hsl_active_count = 0
        limit = max(0, int(group_limit))
        for bot, state in sorted(self.bots.items()):
            phases = dict(sorted(state.get("startup_phases_ms", {}).items()))
            item = {
                "bot": bot,
                "latest_ts": state.get("latest_ts"),
                "lifecycle_status": state.get("lifecycle_status"),
                "startup_phases_ms": dict(list(phases.items())[:limit]),
            }
            if len(phases) > limit:
                item["startup_phases_truncated"] = True
            if state.get("bot_started_ts") is not None:
                item["bot_started_ts"] = int(state["bot_started_ts"])
            if state.get("bot_ready_ts") is not None:
                item["bot_ready_ts"] = int(state["bot_ready_ts"])
                ready_count += 1
            if isinstance(state.get("hsl_replay"), dict):
                hsl_state = {
                    key: value
                    for key, value in state["hsl_replay"].items()
                    if value is not None
                }
                item["hsl_replay"] = hsl_state
                if hsl_state.get("status") not in ("succeeded", "failed"):
                    hsl_active_count += 1
            bot_items.append({key: value for key, value in item.items() if value not in (None, {})})
        return {
            "bot_count": len(bot_items),
            "ready_count": int(ready_count),
            "hsl_replay_active_count": int(hsl_active_count),
            "bots_truncated": len(bot_items) > limit,
            "bots": bot_items[:limit],
        }


class _ResourcePressureAccumulator:
    def __init__(self) -> None:
        self.bots: dict[str, dict[str, Any]] = {}
        self.event_types: Counter[str] = Counter()

    def add(self, *, row: dict[str, Any], live_event: dict[str, Any]) -> None:
        event_type = str(live_event.get("event_type") or row.get("kind") or "")
        if event_type != "health.summary":
            return
        data = live_event.get("data") if isinstance(live_event.get("data"), dict) else {}
        observed_values = {}
        for key in _RESOURCE_PRESSURE_FIELDS:
            value = _non_negative_number(data.get(key))
            if value is not None:
                observed_values[key] = value
        observed_counters = {}
        for key in _RESOURCE_PRESSURE_COUNTER_FIELDS:
            counters = _safe_counter_mapping(data.get(key))
            if counters:
                observed_counters[key] = counters
        observed_bools = {
            key: bool(data.get(key))
            for key in _RESOURCE_PRESSURE_BOOL_FIELDS
            if key in data
        }
        if not observed_values and not observed_counters and not observed_bools:
            return
        bot = _bot_key(row, live_event)
        state = self.bots.get(bot)
        if state is None:
            state = {
                "bot": bot,
                "count": 0,
                "latest_ts": None,
                "values": {key: [] for key in _RESOURCE_PRESSURE_FIELDS},
                "latest": {},
                "latest_counters": {},
            }
            self.bots[bot] = state
        state["count"] = int(state["count"]) + 1
        ts = _record_ts(row)
        latest_changed = ts is None or state.get("latest_ts") is None
        if ts is not None and state.get("latest_ts") is not None:
            latest_changed = int(ts) >= int(state["latest_ts"])
        if ts is not None and latest_changed:
            state["latest_ts"] = int(ts)
        for key, value in observed_values.items():
            state["values"][key].append(value)
            if latest_changed:
                state["latest"][key] = value
        if latest_changed:
            state["latest"].update(observed_bools)
            state["latest_counters"] = observed_counters
        self.event_types[event_type] += 1

    @staticmethod
    def _field_stats(values: list[float | int], latest: Any) -> dict[str, Any]:
        numeric = [float(value) for value in values]
        if not numeric:
            return {}
        out: dict[str, Any] = {
            "latest": latest,
            "min": min(numeric),
            "max": max(numeric),
            "mean": statistics.fmean(numeric),
        }
        return {
            key: int(value) if isinstance(value, float) and value.is_integer() else value
            for key, value in out.items()
        }

    def to_dict(self, *, group_limit: int = GROUP_LIMIT) -> dict[str, Any]:
        groups = []
        for bot, state in self.bots.items():
            latest = state.get("latest") if isinstance(state.get("latest"), dict) else {}
            fields = {}
            for key in _RESOURCE_PRESSURE_FIELDS:
                values = state["values"].get(key) if isinstance(state.get("values"), dict) else []
                stats = self._field_stats(values or [], latest.get(key))
                if stats:
                    fields[key] = stats
            latest_counters = (
                state.get("latest_counters")
                if isinstance(state.get("latest_counters"), dict)
                else {}
            )
            group = {
                "bot": bot,
                "count": int(state.get("count") or 0),
                "latest_ts": state.get("latest_ts"),
                "fields": fields,
                "latest_event_pipeline_stopping": latest.get("event_pipeline_stopping"),
                "latest_event_pipeline_worker_alive": latest.get(
                    "event_pipeline_worker_alive"
                ),
                "latest_event_drop_counts": latest_counters.get("event_drop_counts"),
                "latest_event_sink_error_counts": latest_counters.get(
                    "event_sink_error_counts"
                ),
            }
            groups.append(
                {key: value for key, value in group.items() if value not in (None, {}, [])}
            )
        groups = sorted(
            groups,
            key=lambda item: (
                -int(item.get("fields", {}).get("event_dropped_total", {}).get("latest", 0) or 0),
                -int(
                    item.get("fields", {}).get("event_sink_error_total", {}).get("latest", 0)
                    or 0
                ),
                -int(item.get("fields", {}).get("event_degraded_count", {}).get("latest", 0) or 0),
                -int(item.get("fields", {}).get("event_queue_depth", {}).get("latest", 0) or 0),
                -int(item.get("fields", {}).get("rss_bytes", {}).get("max", 0) or 0),
                str(item.get("bot") or ""),
            ),
        )
        limit = max(0, int(group_limit))
        return {
            "total": sum(int(group.get("count") or 0) for group in groups),
            "bots": len(groups),
            "event_types": dict(self.event_types.most_common()),
            "groups_truncated": len(groups) > limit,
            "groups": groups[:limit],
        }


def _slowest_blockers(
    *,
    performance_groups: list[dict[str, Any]],
    decision_boundary_groups: list[dict[str, Any]],
    input_staleness_groups: list[dict[str, Any]],
    group_limit: int = GROUP_LIMIT,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for source_section, groups in (
        ("performance", performance_groups),
        ("decision_boundary_lag", decision_boundary_groups),
        ("input_staleness", input_staleness_groups),
    ):
        for group in groups:
            impact = str(group.get("trading_impact") or "")
            if not impact or impact in _NON_BLOCKING_IMPACTS:
                continue
            item = {
                key: group[key]
                for key in (
                    "bot",
                    "operation",
                    "component",
                    "event_type",
                    "trading_impact",
                    "timing_kind",
                    "count",
                    "min_ms",
                    "max_ms",
                    "mean_ms",
                    "median_ms",
                    "p95_ms",
                    "latest_ts",
                    "statuses",
                    "reason_codes",
                    "symbols_sample",
                )
                if key in group
            }
            item["source_section"] = source_section
            item["blocking_scope"] = _BLOCKING_SCOPE_BY_IMPACT.get(impact, impact)
            items.append(item)
    items = sorted(
        items,
        key=lambda item: (
            -int(item.get("p95_ms", 0) or 0),
            -int(item.get("max_ms", 0) or 0),
            -int(item.get("count", 0) or 0),
            str(item.get("bot") or ""),
            str(item.get("source_section") or ""),
            str(item.get("operation") or ""),
        ),
    )
    limit = max(0, int(group_limit))
    return {
        "total_groups": len(items),
        "groups_truncated": len(items) > limit,
        "groups": items[:limit],
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
    input_staleness = _InputStalenessAccumulator()
    startup_readiness = _StartupReadinessAccumulator()
    resource_pressure = _ResourcePressureAccumulator()
    cycle_scope_tracker = _CycleScopeTracker()
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
                    cycle_scope = cycle_scope_tracker.observe(row=row, live_event=live_event)
                    event_types[event_type] += 1
                    bots[_bot_key(row, live_event)] += 1
                    _add_event_timings(
                        accumulator,
                        row=row,
                        live_event=live_event,
                    )
                    decision_boundary.add(
                        row=row,
                        live_event=live_event,
                        cycle_scope=cycle_scope,
                    )
                    input_staleness.add(
                        row=row,
                        live_event=live_event,
                        cycle_scope=cycle_scope,
                    )
                    startup_readiness.add(row=row, live_event=live_event)
                    resource_pressure.add(row=row, live_event=live_event)
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
    performance_groups = accumulator.groups_list()
    decision_boundary_groups = decision_boundary.groups_list()
    input_staleness_groups = input_staleness.groups_list()
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
        "input_staleness": input_staleness.to_dict(group_limit=group_limit),
        "startup_readiness": startup_readiness.to_dict(group_limit=group_limit),
        "resource_pressure": resource_pressure.to_dict(group_limit=group_limit),
        "slowest_blockers": _slowest_blockers(
            performance_groups=performance_groups,
            decision_boundary_groups=decision_boundary_groups,
            input_staleness_groups=input_staleness_groups,
            group_limit=group_limit,
        ),
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
    if isinstance(report.get("input_staleness"), dict):
        input_staleness = report["input_staleness"]
        staleness_groups = (
            input_staleness.get("groups")
            if isinstance(input_staleness.get("groups"), list)
            else []
        )
        summary["input_staleness"] = {
            "snapshots_seen": int(input_staleness.get("snapshots_seen") or 0),
            "rust_calls_seen": int(input_staleness.get("rust_calls_seen") or 0),
            "packet_refs_missing": int(input_staleness.get("packet_refs_missing") or 0),
            "rust_calls_missing_snapshot": int(
                input_staleness.get("rust_calls_missing_snapshot") or 0
            ),
            "rust_calls_missing_ema": int(input_staleness.get("rust_calls_missing_ema") or 0),
            "total_groups": int(input_staleness.get("total_groups") or 0),
            "groups_truncated": bool(input_staleness.get("groups_truncated")),
            "groups": staleness_groups[: max(0, int(group_limit))],
        }
    if isinstance(report.get("startup_readiness"), dict):
        startup_readiness = dict(report["startup_readiness"])
        startup_bots = (
            startup_readiness.get("bots")
            if isinstance(startup_readiness.get("bots"), list)
            else []
        )
        startup_readiness["bots"] = startup_bots[: max(0, int(group_limit))]
        if len(startup_bots) > max(0, int(group_limit)):
            startup_readiness["bots_truncated"] = True
        summary["startup_readiness"] = startup_readiness
    if isinstance(report.get("resource_pressure"), dict):
        resource_pressure = dict(report["resource_pressure"])
        pressure_groups = (
            resource_pressure.get("groups")
            if isinstance(resource_pressure.get("groups"), list)
            else []
        )
        resource_pressure["groups"] = pressure_groups[: max(0, int(group_limit))]
        if len(pressure_groups) > max(0, int(group_limit)):
            resource_pressure["groups_truncated"] = True
        summary["resource_pressure"] = resource_pressure
    if isinstance(report.get("slowest_blockers"), dict):
        slowest_blockers = report["slowest_blockers"]
        blocker_groups = (
            slowest_blockers.get("groups")
            if isinstance(slowest_blockers.get("groups"), list)
            else []
        )
        summary["slowest_blockers"] = {
            "total_groups": int(slowest_blockers.get("total_groups") or 0),
            "groups_truncated": bool(slowest_blockers.get("groups_truncated")),
            "groups": blocker_groups[: max(0, int(group_limit))],
        }
    if report.get("event_window") is not None:
        summary["event_window"] = report.get("event_window")
    if report.get("filters") is not None:
        summary["filters"] = report.get("filters")
    if report.get("issues"):
        summary["issues"] = report.get("issues")
    return summary
