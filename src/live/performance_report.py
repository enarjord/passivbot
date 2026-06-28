from __future__ import annotations

import gzip
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from live.event_bus import LIVE_EVENT_DEBUG_PROFILES, LIVE_EVENT_MONITOR_PAYLOAD_KEY
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
_SHUTDOWN_EVENT_TYPES = {
    "bot.stopping",
    "bot.shutdown.stage",
    "bot.stopped",
}
_EXECUTION_CREATE_TERMINALS = {
    "execution.create_succeeded",
    "execution.create_failed",
    "execution.create_rejected",
}
_EXECUTION_CANCEL_TERMINALS = {
    "execution.cancel_succeeded",
    "execution.cancel_failed",
    "execution.cancel_ambiguous_terminal",
}
_EXECUTION_CONFIRMATION_TERMINALS = {
    "execution.confirmation_satisfied",
    "execution.confirmation_timeout",
}
_HSL_REPLAY_STRING_FIELDS = (
    "signal_mode",
    "stage",
)
_HSL_REPLAY_BOOL_FIELDS = (
    "is_held_pair",
    "is_cooldown_pair",
)
_HSL_REPLAY_NUMERIC_FIELDS = (
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
    "pair_idx",
    "applied_rows",
    "total_applied_rows",
    "rows",
    "skipped_pairs",
    "rows_per_second",
    "elapsed_s",
    "full_elapsed_s",
    "startup_blocking_elapsed_s",
)
_CACHE_EVENT_TYPES = {
    "cache.load.completed",
    "cache.flush.completed",
    "cache.warmup_decision",
}
_FORAGER_EMA_READINESS_EVENT_TYPES = {
    "forager.selection",
    "forager.feature_unavailable",
    "ema.unavailable",
    "ema.fallback_used",
}
_ACCOUNT_STATE_CHANGE_EVENT_TYPES = {
    "fill.ingested",
    "position.changed",
    "balance.changed",
}
_RISK_ACTIVITY_EVENT_TYPES = {
    "risk.mode_changed",
    "hsl.transition",
    "hsl.status",
    "hsl.red_triggered",
    "hsl.cooldown_started",
    "hsl.cooldown_ended",
    "unstuck.status",
    "unstuck.selection",
}
_CACHE_STRING_FIELDS = (
    "context",
    "timeframe",
    "stage",
)
_CACHE_BOOL_FIELDS = (
    "cold_path_required",
)
_CACHE_NUMERIC_FIELDS = (
    "start_ts",
    "end_ts",
    "loaded_rows",
    "loaded_start_ts",
    "loaded_end_ts",
    "days",
    "primary_days",
    "legacy_days",
    "merged_days",
    "elapsed_ms",
    "suppressed_count",
    "persisted_rows",
    "persisted_start_ts",
    "persisted_end_ts",
    "suppressed_rows",
    "symbol_count",
    "reused_count",
    "cold_count",
    "concurrency",
    "ttl_ms",
    "window_min_candles",
    "window_max_candles",
)
_CACHE_COUNTER_FIELDS = (
    "source_days",
    "reason_counts",
)
_FORAGER_SELECTION_NUMERIC_FIELDS = (
    "candidate_count",
    "eligible_count",
    "selected_count",
    "incumbent_count",
    "max_n_positions",
    "slots_to_fill",
    "max_age_ms",
    "fetch_budget",
    "feature_unavailable_count",
    "volatility_dropped_count",
    "hysteresis_event_count",
)
_FORAGER_FEATURE_UNAVAILABLE_NUMERIC_FIELDS = (
    "candidate_count",
    "volume_count",
    "log_range_count",
    "max_age_ms",
    "fetch_budget",
)
_EMA_UNAVAILABLE_NUMERIC_FIELDS = (
    "optional_drop_count",
)
_EMA_FALLBACK_NUMERIC_FIELDS = (
    "close_recovered_count",
    "close_fallback_count",
    "forager_cached_fallback_count",
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


def _safe_string_list(value: Any, *, limit: int = 12) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item)
        if not text:
            continue
        out.append(text)
        if len(out) >= int(limit):
            break
    return out


def _elapsed_s_to_ms(value: Any) -> int | None:
    number = _finite_float(value)
    if number is None or number < 0:
        return None
    return int(round(number * 1000.0))


def _rounded_float(value: float, ndigits: int = 3) -> float | int:
    rounded = round(float(value), int(ndigits))
    if rounded.is_integer():
        return int(rounded)
    return rounded


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


def _event_id_value(live_event: dict[str, Any], key: str) -> str | None:
    ids = live_event.get("ids")
    if not isinstance(ids, dict):
        return None
    value = ids.get(key)
    if value is None:
        return None
    return str(value)


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


class _InputStalenessAccumulator:
    def __init__(self) -> None:
        self.groups: dict[tuple[str, str, str], _MetricGroup] = {}
        self.packet_received_ts: dict[tuple[str, int, str, str], int] = {}
        self.snapshot_ts_by_cycle: dict[tuple[str, int, str], int] = {}
        self.latest_snapshot_ts_by_scope: dict[tuple[str, int], int] = {}
        self.ema_ts_by_cycle: dict[tuple[str, int, str], int] = {}
        self.snapshots_seen = 0
        self.snapshot_surface_age_rows = 0
        self.snapshot_market_summaries_seen = 0
        self.snapshot_market_stale_count = 0
        self.rust_calls_seen = 0
        self.packet_refs_missing = 0
        self.snapshot_to_rust_exact_matches = 0
        self.snapshot_to_rust_latest_snapshot_matches = 0
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
            scope_key = (bot, int(cycle_scope))
            self.latest_snapshot_ts_by_scope[scope_key] = int(timestamp_ms)
            cycle_id = _event_cycle_id(live_event)
            if cycle_id:
                self.snapshot_ts_by_cycle[(bot, int(cycle_scope), cycle_id)] = int(timestamp_ms)
            surface_ages = data.get("surface_ages")
            if isinstance(surface_ages, list):
                for surface in surface_ages:
                    if not isinstance(surface, dict):
                        continue
                    name = surface.get("name")
                    age_ms = _non_negative_ms(surface.get("age_ms"))
                    if name is None or age_ms is None:
                        continue
                    self.snapshot_surface_age_rows += 1
                    surface_name = str(name)
                    impact = (
                        "blocks_indicator_readiness"
                        if surface_name == "completed_candles"
                        else "blocks_exchange_actions"
                    )
                    self._add_group(
                        row=row,
                        live_event=live_event,
                        operation=f"input_staleness.surface.{surface_name}",
                        value_ms=age_ms,
                        timing_kind="age_at_snapshot",
                        trading_impact=impact,
                    )
            market_summary = data.get("market_snapshot_summary")
            if isinstance(market_summary, dict):
                max_age_ms = _non_negative_ms(market_summary.get("max_age_ms"))
                mean_age_ms = _non_negative_ms(market_summary.get("mean_age_ms"))
                configured_max_age_ms = _non_negative_ms(
                    market_summary.get("configured_max_age_ms")
                )
                if max_age_ms is not None:
                    self.snapshot_market_summaries_seen += 1
                    self._add_group(
                        row=row,
                        live_event=live_event,
                        operation="input_staleness.market_snapshot.max",
                        value_ms=max_age_ms,
                        timing_kind="age_at_snapshot",
                        trading_impact="blocks_exchange_actions",
                    )
                    if (
                        configured_max_age_ms is not None
                        and max_age_ms > configured_max_age_ms
                    ):
                        self.snapshot_market_stale_count += 1
                        self._add_group(
                            row=row,
                            live_event=live_event,
                            operation="input_staleness.market_snapshot.configured_excess",
                            value_ms=max_age_ms - configured_max_age_ms,
                            timing_kind="configured_age_excess",
                            trading_impact="blocks_exchange_actions",
                        )
                if mean_age_ms is not None:
                    self._add_group(
                        row=row,
                        live_event=live_event,
                        operation="input_staleness.market_snapshot.mean",
                        value_ms=mean_age_ms,
                        timing_kind="age_at_snapshot",
                        trading_impact="blocks_exchange_actions",
                    )
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
            if snapshot_ts is not None:
                self.snapshot_to_rust_exact_matches += 1
            else:
                latest_snapshot_ts = self.latest_snapshot_ts_by_scope.get((bot, int(cycle_scope)))
                if latest_snapshot_ts is not None and latest_snapshot_ts <= int(timestamp_ms):
                    snapshot_ts = int(latest_snapshot_ts)
                    self.snapshot_to_rust_latest_snapshot_matches += 1
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
            "snapshot_surface_age_rows": int(self.snapshot_surface_age_rows),
            "snapshot_market_summaries_seen": int(self.snapshot_market_summaries_seen),
            "snapshot_market_stale_count": int(self.snapshot_market_stale_count),
            "rust_calls_seen": int(self.rust_calls_seen),
            "packet_refs_missing": int(self.packet_refs_missing),
            "snapshot_to_rust_exact_matches": int(self.snapshot_to_rust_exact_matches),
            "snapshot_to_rust_latest_snapshot_matches": int(
                self.snapshot_to_rust_latest_snapshot_matches
            ),
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


_LIVE_EVENT_DEBUG_PROFILE_SET = set(LIVE_EVENT_DEBUG_PROFILES)


def _known_debug_profiles(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_values: list[Any] = [
            part for part in value.replace(";", ",").replace(" ", ",").split(",") if part
        ]
    elif isinstance(value, (list, tuple, set, frozenset)):
        raw_values = list(value)
    else:
        return []
    profiles = {
        str(item).strip()
        for item in raw_values
        if str(item).strip() in _LIVE_EVENT_DEBUG_PROFILE_SET
    }
    return sorted(profiles)


class _StartupReadinessAccumulator:
    def __init__(self) -> None:
        self.bots: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _update_debug_profiles(state: dict[str, Any], data: dict[str, Any]) -> None:
        profiles = _known_debug_profiles(data.get("live_event_debug_profiles"))
        if not profiles:
            return
        existing = state.get("debug_profiles")
        if not isinstance(existing, set):
            existing = set()
            state["debug_profiles"] = existing
        existing.update(profiles)

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
            self._update_debug_profiles(state, data)
            return
        if event_type == "bot.ready":
            state = self._bot_state(row=row, live_event=live_event)
            if ts is not None:
                state["bot_ready_ts"] = int(ts)
            state["lifecycle_status"] = "ready"
            self._update_debug_profiles(state, data)
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
        debug_profile_counts: Counter[str] = Counter()
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
            debug_profiles = sorted(state.get("debug_profiles") or [])
            if debug_profiles:
                item["debug_profiles"] = debug_profiles
                debug_profile_counts.update(debug_profiles)
            bot_items.append({key: value for key, value in item.items() if value not in (None, {})})
        return {
            "bot_count": len(bot_items),
            "ready_count": int(ready_count),
            "hsl_replay_active_count": int(hsl_active_count),
            "debug_profile_counts": dict(sorted(debug_profile_counts.items())),
            "bots_truncated": len(bot_items) > limit,
            "bots": bot_items[:limit],
        }


def _bounded_hsl_replay_data(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for key in _HSL_REPLAY_STRING_FIELDS:
        value = data.get(key)
        if value is not None:
            out[key] = str(value)
    for key in _HSL_REPLAY_BOOL_FIELDS:
        if key in data:
            out[key] = bool(data.get(key))
    for key in _HSL_REPLAY_NUMERIC_FIELDS:
        value = _non_negative_number(data.get(key))
        if value is not None:
            out[key] = value
    return out


def _bounded_cache_event_data(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for key in _CACHE_STRING_FIELDS:
        value = data.get(key)
        if value is not None:
            out[key] = str(value)
    for key in _CACHE_BOOL_FIELDS:
        if key in data:
            out[key] = bool(data.get(key))
    for key in _CACHE_NUMERIC_FIELDS:
        value = _non_negative_number(data.get(key))
        if value is not None:
            out[key] = value
    for key in _CACHE_COUNTER_FIELDS:
        mapping = _safe_counter_mapping(data.get(key))
        if mapping:
            out[key] = mapping
    return out


def _bounded_forager_selection_data(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for key in _FORAGER_SELECTION_NUMERIC_FIELDS:
        value = _non_negative_number(data.get(key))
        if value is not None:
            out[key] = value
    for key in ("slots_open",):
        if key in data:
            out[key] = bool(data.get(key))
    for key in ("source",):
        value = data.get(key)
        if value is not None:
            out[key] = str(value)
    for key in ("selected_symbols", "incumbent_symbols"):
        values = _safe_string_list(data.get(key), limit=12)
        if values:
            out[key] = values
    return out


def _bounded_forager_feature_unavailable_data(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for key in _FORAGER_FEATURE_UNAVAILABLE_NUMERIC_FIELDS:
        value = _non_negative_number(data.get(key))
        if value is not None:
            out[key] = value
    unavailable = _safe_string_list(data.get("unavailable"), limit=12)
    if unavailable:
        out["unavailable"] = unavailable
    return out


def _bounded_ema_unavailable_data(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for key in _EMA_UNAVAILABLE_NUMERIC_FIELDS:
        value = _non_negative_number(data.get(key))
        if value is not None:
            out[key] = value
    for key in ("candidate_unavailable", "unavailable"):
        values = _safe_string_list(data.get(key), limit=12)
        if values:
            out[key] = values
    candidate_groups = []
    raw_candidate_groups = data.get("candidate_unavailable_groups")
    if isinstance(raw_candidate_groups, list):
        for item in raw_candidate_groups[:8]:
            if not isinstance(item, dict):
                continue
            group = {
                "reason": str(item.get("reason")) if item.get("reason") is not None else None,
                "symbols": _safe_string_list(item.get("symbols"), limit=12),
                "error_types": _safe_string_list(item.get("error_types"), limit=4),
            }
            candidate_groups.append(
                {key: value for key, value in group.items() if value not in (None, [], {})}
            )
    if candidate_groups:
        out["candidate_unavailable_groups"] = candidate_groups
    unavailable_reasons = []
    raw_unavailable_reasons = data.get("unavailable_reasons")
    if isinstance(raw_unavailable_reasons, list):
        for item in raw_unavailable_reasons[:8]:
            if not isinstance(item, dict):
                continue
            group = {
                "reason": str(item.get("reason")) if item.get("reason") is not None else None,
                "symbols": _safe_string_list(item.get("symbols"), limit=12),
            }
            unavailable_reasons.append(
                {key: value for key, value in group.items() if value not in (None, [], {})}
            )
    if unavailable_reasons:
        out["unavailable_reasons"] = unavailable_reasons
    optional_groups = []
    raw_optional_groups = data.get("optional_drop_groups")
    if isinstance(raw_optional_groups, list):
        for item in raw_optional_groups[:8]:
            if not isinstance(item, dict):
                continue
            group = {
                "ema_type": (
                    str(item.get("ema_type")) if item.get("ema_type") is not None else None
                ),
                "reason": str(item.get("reason")) if item.get("reason") is not None else None,
                "symbols": _safe_string_list(item.get("symbols"), limit=12),
            }
            optional_groups.append(
                {key: value for key, value in group.items() if value not in (None, [], {})}
            )
    if optional_groups:
        out["optional_drop_groups"] = optional_groups
    return out


def _bounded_ema_fallback_data(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for key in _EMA_FALLBACK_NUMERIC_FIELDS:
        value = _non_negative_number(data.get(key))
        if value is not None:
            out[key] = value
    for key in (
        "close_recovered_symbols",
        "close_fallback_symbols",
        "forager_cached_fallback_symbols",
    ):
        values = _safe_string_list(data.get(key), limit=12)
        if values:
            out[key] = values
    return out


def _bounded_forager_ema_readiness_data(event_type: str, data: Any) -> dict[str, Any]:
    if event_type == "forager.selection":
        return _bounded_forager_selection_data(data)
    if event_type == "forager.feature_unavailable":
        return _bounded_forager_feature_unavailable_data(data)
    if event_type == "ema.unavailable":
        return _bounded_ema_unavailable_data(data)
    if event_type == "ema.fallback_used":
        return _bounded_ema_fallback_data(data)
    return {}


def _number_summary(values: list[int]) -> dict[str, Any]:
    sorted_values = sorted(int(value) for value in values)
    if not sorted_values:
        return {}
    mean_value = statistics.fmean(sorted_values)
    return {
        "count": len(sorted_values),
        "min": int(sorted_values[0]),
        "max": int(sorted_values[-1]),
        "mean": int(round(mean_value)),
        "median": int(round(statistics.median(sorted_values))),
        "p95": _percentile(sorted_values, 0.95),
    }


def _hsl_replay_observed_rows(data: dict[str, Any]) -> int | None:
    for key in ("total_applied_rows", "applied_rows", "rows"):
        value = _non_negative_number(data.get(key))
        if value is not None:
            return int(value)
    return None


def _derive_hsl_replay_profile(data: dict[str, Any]) -> dict[str, Any]:
    timeline_rows = _non_negative_number(data.get("timeline_rows"))
    pairs = _non_negative_number(data.get("pairs"))
    required_pairs = _non_negative_number(data.get("required_pairs"))
    held_pairs = _non_negative_number(data.get("held_pairs"))
    cooldown_pairs = _non_negative_number(data.get("cooldown_pairs"))
    observed_rows = _hsl_replay_observed_rows(data)
    out: dict[str, Any] = {}
    if timeline_rows is not None and pairs is not None:
        dense_work = int(timeline_rows) * int(pairs)
        out["estimated_dense_pair_row_work"] = dense_work
        if observed_rows is not None and dense_work > 0:
            out["observed_work_pct"] = _rounded_float(
                min(100.0, max(0.0, 100.0 * float(observed_rows) / float(dense_work)))
            )
    if timeline_rows is not None and required_pairs is not None:
        out["estimated_required_pair_row_work"] = int(timeline_rows) * int(required_pairs)
    if timeline_rows is not None and held_pairs is not None:
        out["estimated_held_pair_row_work"] = int(timeline_rows) * int(held_pairs)
    if timeline_rows is not None and cooldown_pairs is not None:
        out["estimated_cooldown_pair_row_work"] = int(timeline_rows) * int(cooldown_pairs)
    if observed_rows is not None:
        out["observed_applied_rows"] = int(observed_rows)
    for source_key, target_key in (
        ("elapsed_s", "latest_elapsed_ms"),
        ("full_elapsed_s", "full_elapsed_ms"),
        ("startup_blocking_elapsed_s", "startup_blocking_elapsed_ms"),
    ):
        value_ms = _elapsed_s_to_ms(data.get(source_key))
        if value_ms is not None:
            out[target_key] = value_ms
    if data.get("startup_blocking_elapsed_s") is not None:
        out["startup_blocking"] = True
    return out


class _HslReplayProfileAccumulator:
    def __init__(self) -> None:
        self.bots: dict[str, dict[str, Any]] = {}
        self.event_types: Counter[str] = Counter()
        self.total_events = 0

    def add(self, *, row: dict[str, Any], live_event: dict[str, Any]) -> None:
        event_type = str(live_event.get("event_type") or row.get("kind") or "")
        if not event_type.startswith("hsl.replay."):
            return
        data = _bounded_hsl_replay_data(live_event.get("data"))
        if not data:
            return
        bot = _bot_key(row, live_event)
        state = self.bots.get(bot)
        if state is None:
            state = {
                "bot": bot,
                "event_types": Counter(),
                "total_events": 0,
                "latest_ts": None,
            }
            self.bots[bot] = state
        self.total_events += 1
        self.event_types[event_type] += 1
        state["total_events"] = int(state["total_events"]) + 1
        state["event_types"][event_type] += 1
        ts = _record_ts(row)
        latest_changed = ts is None or state.get("latest_ts") is None
        if ts is not None and state.get("latest_ts") is not None:
            latest_changed = int(ts) >= int(state["latest_ts"])
        record = {
            key: value
            for key, value in {
                "event_type": event_type,
                "status": live_event.get("status"),
                "reason_code": live_event.get("reason_code"),
                "ts": int(ts) if ts is not None else None,
                "symbol": live_event.get("symbol") or row.get("symbol"),
                "pside": live_event.get("pside") or row.get("pside"),
                "data": data,
                "derived": _derive_hsl_replay_profile(data),
            }.items()
            if value not in (None, {}, [])
        }
        stage = str(data.get("stage") or "")
        if event_type == "hsl.replay.progress" and stage == "loaded":
            state["loaded"] = record
        elif event_type == "hsl.replay.completed":
            state["completed"] = record
        elif event_type == "hsl.replay.progress":
            state["progress"] = record
        elif event_type == "hsl.replay.started":
            state["started"] = record
        if latest_changed:
            if ts is not None:
                state["latest_ts"] = int(ts)
            state["latest"] = record

    def to_dict(self, *, group_limit: int = GROUP_LIMIT) -> dict[str, Any]:
        groups = []
        for bot, state in self.bots.items():
            group = {
                "bot": bot,
                "total_events": int(state.get("total_events") or 0),
                "latest_ts": state.get("latest_ts"),
                "event_types": dict(state["event_types"].most_common())
                if isinstance(state.get("event_types"), Counter)
                else {},
                "latest": state.get("latest"),
                "started": state.get("started"),
                "loaded": state.get("loaded"),
                "progress": state.get("progress"),
                "completed": state.get("completed"),
            }
            groups.append({key: value for key, value in group.items() if value not in (None, {}, [])})
        groups = sorted(
            groups,
            key=lambda item: (
                -int(
                    (
                        item.get("latest", {})
                        .get("derived", {})
                        .get("startup_blocking_elapsed_ms", 0)
                    )
                    or 0
                ),
                -int(
                    (
                        item.get("latest", {})
                        .get("derived", {})
                        .get("latest_elapsed_ms", 0)
                    )
                    or 0
                ),
                -int(item.get("latest_ts", 0) or 0),
                str(item.get("bot") or ""),
            ),
        )
        limit = max(0, int(group_limit))
        return {
            "total_events": int(self.total_events),
            "bot_count": len(groups),
            "event_types": dict(self.event_types.most_common()),
            "groups_truncated": len(groups) > limit,
            "groups": groups[:limit],
        }


class _CacheWarmupAccumulator:
    def __init__(self) -> None:
        self.bots: dict[str, dict[str, Any]] = {}
        self.event_types: Counter[str] = Counter()
        self.total_events = 0

    def add(self, *, row: dict[str, Any], live_event: dict[str, Any]) -> None:
        event_type = str(live_event.get("event_type") or row.get("kind") or "")
        if event_type not in _CACHE_EVENT_TYPES:
            return
        data = _bounded_cache_event_data(live_event.get("data"))
        if not data:
            return
        bot = _bot_key(row, live_event)
        state = self.bots.get(bot)
        if state is None:
            state = {
                "bot": bot,
                "event_types": Counter(),
                "total_events": 0,
                "latest_ts": None,
                "symbols": Counter(),
                "timeframes": Counter(),
                "warmup_contexts": Counter(),
                "warmup_reason_counts": Counter(),
                "load_source_days": Counter(),
                "warmup_elapsed_ms": [],
                "load_elapsed_ms": [],
                "symbol_count": 0,
                "reused_count": 0,
                "cold_count": 0,
                "cold_path_decisions": 0,
                "loaded_rows": 0,
                "persisted_rows": 0,
                "suppressed_load_events": 0,
                "suppressed_flush_events": 0,
                "suppressed_flush_rows": 0,
            }
            self.bots[bot] = state

        self.total_events += 1
        self.event_types[event_type] += 1
        state["total_events"] = int(state["total_events"]) + 1
        state["event_types"][event_type] += 1
        ts = _record_ts(row)
        symbol = live_event.get("symbol") or row.get("symbol")
        if symbol is not None:
            state["symbols"][str(symbol)] += 1
        timeframe = data.get("timeframe")
        if timeframe is not None:
            state["timeframes"][str(timeframe)] += 1

        record = {
            key: value
            for key, value in {
                "event_type": event_type,
                "status": live_event.get("status"),
                "reason_code": live_event.get("reason_code"),
                "ts": int(ts) if ts is not None else None,
                "symbol": str(symbol) if symbol is not None else None,
                "data": data,
            }.items()
            if value not in (None, {}, [])
        }

        if event_type == "cache.warmup_decision":
            context = data.get("context")
            if context is not None:
                state["warmup_contexts"][str(context)] += 1
            for key in ("symbol_count", "reused_count", "cold_count"):
                value = _non_negative_number(data.get(key))
                if value is not None:
                    state[key] = int(state.get(key) or 0) + int(value)
            if bool(data.get("cold_path_required")):
                state["cold_path_decisions"] = int(state["cold_path_decisions"]) + 1
            elapsed_ms = _non_negative_ms(data.get("elapsed_ms"))
            if elapsed_ms is not None:
                state["warmup_elapsed_ms"].append(elapsed_ms)
            for reason, count in _safe_counter_mapping(data.get("reason_counts")).items():
                state["warmup_reason_counts"][reason] += int(count)
            state["latest_warmup_decision"] = record
        elif event_type == "cache.load.completed":
            loaded_rows = _non_negative_number(data.get("loaded_rows"))
            if loaded_rows is not None:
                state["loaded_rows"] = int(state["loaded_rows"]) + int(loaded_rows)
            suppressed = _non_negative_number(data.get("suppressed_count"))
            if suppressed is not None:
                state["suppressed_load_events"] = int(state["suppressed_load_events"]) + int(
                    suppressed
                )
            elapsed_ms = _non_negative_ms(data.get("elapsed_ms"))
            if elapsed_ms is not None:
                state["load_elapsed_ms"].append(elapsed_ms)
            for source, count in _safe_counter_mapping(data.get("source_days")).items():
                state["load_source_days"][source] += int(count)
            state["latest_load_completed"] = record
        elif event_type == "cache.flush.completed":
            persisted_rows = _non_negative_number(data.get("persisted_rows"))
            if persisted_rows is not None:
                state["persisted_rows"] = int(state["persisted_rows"]) + int(persisted_rows)
            suppressed = _non_negative_number(data.get("suppressed_count"))
            if suppressed is not None:
                state["suppressed_flush_events"] = int(state["suppressed_flush_events"]) + int(
                    suppressed
                )
            suppressed_rows = _non_negative_number(data.get("suppressed_rows"))
            if suppressed_rows is not None:
                state["suppressed_flush_rows"] = int(state["suppressed_flush_rows"]) + int(
                    suppressed_rows
                )
            state["latest_flush_completed"] = record

        latest_changed = ts is None or state.get("latest_ts") is None
        if ts is not None and state.get("latest_ts") is not None:
            latest_changed = int(ts) >= int(state["latest_ts"])
        if latest_changed:
            if ts is not None:
                state["latest_ts"] = int(ts)
            state["latest"] = record

    def to_dict(self, *, group_limit: int = GROUP_LIMIT) -> dict[str, Any]:
        groups = []
        for bot, state in self.bots.items():
            event_counts = (
                state["event_types"]
                if isinstance(state.get("event_types"), Counter)
                else Counter()
            )
            warmup = {}
            if event_counts.get("cache.warmup_decision", 0) > 0:
                warmup = {
                    "contexts": dict(state["warmup_contexts"].most_common()),
                    "reason_counts": dict(state["warmup_reason_counts"].most_common()),
                    "symbol_count": int(state.get("symbol_count") or 0),
                    "reused_count": int(state.get("reused_count") or 0),
                    "cold_count": int(state.get("cold_count") or 0),
                    "cold_path_decisions": int(state.get("cold_path_decisions") or 0),
                    "elapsed_ms": _number_summary(state.get("warmup_elapsed_ms") or []),
                    "latest": state.get("latest_warmup_decision"),
                }
            load = {}
            if event_counts.get("cache.load.completed", 0) > 0:
                load = {
                    "loaded_rows": int(state.get("loaded_rows") or 0),
                    "source_days": dict(state["load_source_days"].most_common()),
                    "suppressed_events": int(state.get("suppressed_load_events") or 0),
                    "elapsed_ms": _number_summary(state.get("load_elapsed_ms") or []),
                    "latest": state.get("latest_load_completed"),
                }
            flush = {}
            if event_counts.get("cache.flush.completed", 0) > 0:
                flush = {
                    "persisted_rows": int(state.get("persisted_rows") or 0),
                    "suppressed_events": int(state.get("suppressed_flush_events") or 0),
                    "suppressed_rows": int(state.get("suppressed_flush_rows") or 0),
                    "latest": state.get("latest_flush_completed"),
                }
            group = {
                "bot": bot,
                "total_events": int(state.get("total_events") or 0),
                "latest_ts": state.get("latest_ts"),
                "event_types": dict(event_counts.most_common()),
                "timeframes": dict(state["timeframes"].most_common()),
                "symbols": {
                    "count": len(state["symbols"]),
                    "sample": [
                        symbol
                        for symbol, _count in state["symbols"].most_common(10)
                    ],
                }
                if state.get("symbols")
                else {},
                "latest": state.get("latest"),
                "warmup": {
                    key: value
                    for key, value in warmup.items()
                    if value not in (None, {}, [])
                },
                "load": {
                    key: value
                    for key, value in load.items()
                    if value not in (None, {}, [])
                },
                "flush": {
                    key: value
                    for key, value in flush.items()
                    if value not in (None, {}, [])
                },
            }
            groups.append(
                {
                    key: value
                    for key, value in group.items()
                    if value not in (None, {}, [])
                }
            )
        groups = sorted(
            groups,
            key=lambda item: (
                -int((item.get("warmup", {}) or {}).get("cold_path_decisions", 0) or 0),
                -int((item.get("warmup", {}) or {}).get("cold_count", 0) or 0),
                -int(
                    ((item.get("load", {}) or {}).get("elapsed_ms", {}) or {}).get(
                        "p95", 0
                    )
                    or 0
                ),
                -int(item.get("latest_ts", 0) or 0),
                str(item.get("bot") or ""),
            ),
        )
        limit = max(0, int(group_limit))
        return {
            "total_events": int(self.total_events),
            "bot_count": len(groups),
            "event_types": dict(self.event_types.most_common()),
            "groups_truncated": len(groups) > limit,
            "groups": groups[:limit],
        }


class _ForagerEmaReadinessAccumulator:
    def __init__(self) -> None:
        self.bots: dict[str, dict[str, Any]] = {}
        self.event_types: Counter[str] = Counter()
        self.total_events = 0

    def _bot_state(self, bot: str) -> dict[str, Any]:
        state = self.bots.get(bot)
        if state is None:
            state = {
                "bot": bot,
                "event_types": Counter(),
                "statuses": Counter(),
                "reason_codes": Counter(),
                "psides": Counter(),
                "total_events": 0,
                "latest_ts": None,
                "selected_symbols": Counter(),
                "incumbent_symbols": Counter(),
                "feature_unavailable_symbols": Counter(),
                "ema_candidate_symbols": Counter(),
                "ema_unavailable_symbols": Counter(),
                "fallback_symbols": Counter(),
                "selection_events": 0,
                "selection_candidate_counts": [],
                "selection_eligible_counts": [],
                "selection_selected_counts": [],
                "selection_feature_unavailable_counts": [],
                "selection_volatility_dropped_counts": [],
                "selection_max_age_ms": [],
                "selection_fetch_budget": [],
                "feature_unavailable_events": 0,
                "feature_unavailable_candidate_counts": [],
                "feature_unavailable_volume_counts": [],
                "feature_unavailable_log_range_counts": [],
                "feature_unavailable_max_age_ms": [],
                "feature_unavailable_fetch_budget": [],
                "ema_unavailable_events": 0,
                "ema_optional_drop_count": 0,
                "ema_candidate_unavailable_count": 0,
                "ema_unavailable_count": 0,
                "ema_candidate_reasons": Counter(),
                "ema_unavailable_reasons": Counter(),
                "ema_optional_drop_reasons": Counter(),
                "ema_error_types": Counter(),
                "ema_fallback_events": 0,
                "close_recovered_count": 0,
                "close_fallback_count": 0,
                "forager_cached_fallback_count": 0,
            }
            self.bots[bot] = state
        return state

    @staticmethod
    def _add_number(container: list[int], value: Any) -> None:
        number = _non_negative_number(value)
        if number is not None:
            container.append(int(number))

    @staticmethod
    def _add_counter(counter: Counter[str], values: Any) -> None:
        for value in _safe_string_list(values, limit=12):
            counter[value] += 1

    @staticmethod
    def _symbols_summary(counter: Counter[str], *, limit: int = 10) -> dict[str, Any]:
        if not counter:
            return {}
        return {
            "count": len(counter),
            "sample": [symbol for symbol, _count in counter.most_common(limit)],
        }

    @staticmethod
    def _latest_changed(state: dict[str, Any], ts: int | None) -> bool:
        if ts is None or state.get("latest_ts") is None:
            return True
        return int(ts) >= int(state["latest_ts"])

    def add(self, *, row: dict[str, Any], live_event: dict[str, Any]) -> None:
        event_type = str(live_event.get("event_type") or row.get("kind") or "")
        if event_type not in _FORAGER_EMA_READINESS_EVENT_TYPES:
            return
        data = _bounded_forager_ema_readiness_data(event_type, live_event.get("data"))
        if not data:
            return
        bot = _bot_key(row, live_event)
        state = self._bot_state(bot)
        self.total_events += 1
        self.event_types[event_type] += 1
        state["total_events"] = int(state["total_events"]) + 1
        state["event_types"][event_type] += 1
        status = live_event.get("status")
        if status is not None:
            state["statuses"][str(status)] += 1
        reason_code = live_event.get("reason_code")
        if reason_code is not None:
            state["reason_codes"][str(reason_code)] += 1
        pside = live_event.get("pside") or row.get("pside")
        if pside is not None:
            state["psides"][str(pside)] += 1
        ts = _record_ts(row)
        record = {
            key: value
            for key, value in {
                "event_type": event_type,
                "status": status,
                "reason_code": reason_code,
                "ts": int(ts) if ts is not None else None,
                "pside": str(pside) if pside is not None else None,
                "data": data,
            }.items()
            if value not in (None, {}, [])
        }

        if event_type == "forager.selection":
            state["selection_events"] = int(state["selection_events"]) + 1
            for field, bucket in (
                ("candidate_count", "selection_candidate_counts"),
                ("eligible_count", "selection_eligible_counts"),
                ("selected_count", "selection_selected_counts"),
                ("feature_unavailable_count", "selection_feature_unavailable_counts"),
                ("volatility_dropped_count", "selection_volatility_dropped_counts"),
                ("max_age_ms", "selection_max_age_ms"),
                ("fetch_budget", "selection_fetch_budget"),
            ):
                self._add_number(state[bucket], data.get(field))
            self._add_counter(state["selected_symbols"], data.get("selected_symbols"))
            self._add_counter(state["incumbent_symbols"], data.get("incumbent_symbols"))
            state["latest_selection"] = record
        elif event_type == "forager.feature_unavailable":
            state["feature_unavailable_events"] = int(state["feature_unavailable_events"]) + 1
            for field, bucket in (
                ("candidate_count", "feature_unavailable_candidate_counts"),
                ("volume_count", "feature_unavailable_volume_counts"),
                ("log_range_count", "feature_unavailable_log_range_counts"),
                ("max_age_ms", "feature_unavailable_max_age_ms"),
                ("fetch_budget", "feature_unavailable_fetch_budget"),
            ):
                self._add_number(state[bucket], data.get(field))
            self._add_counter(state["feature_unavailable_symbols"], data.get("unavailable"))
            state["latest_feature_unavailable"] = record
        elif event_type == "ema.unavailable":
            state["ema_unavailable_events"] = int(state["ema_unavailable_events"]) + 1
            state["ema_optional_drop_count"] = int(state["ema_optional_drop_count"]) + int(
                _non_negative_number(data.get("optional_drop_count")) or 0
            )
            candidate_symbols = data.get("candidate_unavailable")
            unavailable_symbols = data.get("unavailable")
            state["ema_candidate_unavailable_count"] = int(
                state["ema_candidate_unavailable_count"]
            ) + len(_safe_string_list(candidate_symbols, limit=12))
            state["ema_unavailable_count"] = int(state["ema_unavailable_count"]) + len(
                _safe_string_list(unavailable_symbols, limit=12)
            )
            self._add_counter(state["ema_candidate_symbols"], candidate_symbols)
            self._add_counter(state["ema_unavailable_symbols"], unavailable_symbols)
            for item in data.get("candidate_unavailable_groups") or []:
                if not isinstance(item, dict):
                    continue
                reason = item.get("reason")
                if reason is not None:
                    state["ema_candidate_reasons"][str(reason)] += 1
                for error_type in _safe_string_list(item.get("error_types"), limit=4):
                    state["ema_error_types"][error_type] += 1
            for item in data.get("unavailable_reasons") or []:
                if isinstance(item, dict) and item.get("reason") is not None:
                    state["ema_unavailable_reasons"][str(item["reason"])] += 1
            for item in data.get("optional_drop_groups") or []:
                if isinstance(item, dict) and item.get("reason") is not None:
                    state["ema_optional_drop_reasons"][str(item["reason"])] += 1
            state["latest_ema_unavailable"] = record
        elif event_type == "ema.fallback_used":
            state["ema_fallback_events"] = int(state["ema_fallback_events"]) + 1
            for field in _EMA_FALLBACK_NUMERIC_FIELDS:
                state[field] = int(state[field]) + int(_non_negative_number(data.get(field)) or 0)
            for key in (
                "close_recovered_symbols",
                "close_fallback_symbols",
                "forager_cached_fallback_symbols",
            ):
                self._add_counter(state["fallback_symbols"], data.get(key))
            state["latest_ema_fallback"] = record

        if self._latest_changed(state, ts):
            if ts is not None:
                state["latest_ts"] = int(ts)
            state["latest"] = record

    def to_dict(self, *, group_limit: int = GROUP_LIMIT) -> dict[str, Any]:
        groups = []
        for bot, state in self.bots.items():
            selection = {}
            if int(state.get("selection_events") or 0):
                selection = {
                    "events": int(state.get("selection_events") or 0),
                    "candidate_count": _number_summary(state["selection_candidate_counts"]),
                    "eligible_count": _number_summary(state["selection_eligible_counts"]),
                    "selected_count": _number_summary(state["selection_selected_counts"]),
                    "feature_unavailable_count": _number_summary(
                        state["selection_feature_unavailable_counts"]
                    ),
                    "volatility_dropped_count": _number_summary(
                        state["selection_volatility_dropped_counts"]
                    ),
                    "max_age_ms": _number_summary(state["selection_max_age_ms"]),
                    "fetch_budget": _number_summary(state["selection_fetch_budget"]),
                    "selected_symbols": self._symbols_summary(state["selected_symbols"]),
                    "incumbent_symbols": self._symbols_summary(state["incumbent_symbols"]),
                    "latest": state.get("latest_selection"),
                }
            feature_unavailable = {}
            if int(state.get("feature_unavailable_events") or 0):
                feature_unavailable = {
                    "events": int(state.get("feature_unavailable_events") or 0),
                    "candidate_count": _number_summary(
                        state["feature_unavailable_candidate_counts"]
                    ),
                    "volume_count": _number_summary(state["feature_unavailable_volume_counts"]),
                    "log_range_count": _number_summary(
                        state["feature_unavailable_log_range_counts"]
                    ),
                    "max_age_ms": _number_summary(state["feature_unavailable_max_age_ms"]),
                    "fetch_budget": _number_summary(state["feature_unavailable_fetch_budget"]),
                    "unavailable_symbols": self._symbols_summary(
                        state["feature_unavailable_symbols"]
                    ),
                    "latest": state.get("latest_feature_unavailable"),
                }
            ema_unavailable = {}
            if int(state.get("ema_unavailable_events") or 0):
                ema_unavailable = {
                    "events": int(state.get("ema_unavailable_events") or 0),
                    "optional_drop_count": int(state.get("ema_optional_drop_count") or 0),
                    "candidate_symbol_sample_count": int(
                        state.get("ema_candidate_unavailable_count") or 0
                    ),
                    "unavailable_symbol_sample_count": int(
                        state.get("ema_unavailable_count") or 0
                    ),
                    "candidate_reasons": dict(state["ema_candidate_reasons"].most_common()),
                    "unavailable_reasons": dict(state["ema_unavailable_reasons"].most_common()),
                    "optional_drop_reasons": dict(
                        state["ema_optional_drop_reasons"].most_common()
                    ),
                    "error_types": dict(state["ema_error_types"].most_common()),
                    "candidate_symbols": self._symbols_summary(state["ema_candidate_symbols"]),
                    "unavailable_symbols": self._symbols_summary(
                        state["ema_unavailable_symbols"]
                    ),
                    "latest": state.get("latest_ema_unavailable"),
                }
            ema_fallback = {}
            if int(state.get("ema_fallback_events") or 0):
                ema_fallback = {
                    "events": int(state.get("ema_fallback_events") or 0),
                    "close_recovered_count": int(state.get("close_recovered_count") or 0),
                    "close_fallback_count": int(state.get("close_fallback_count") or 0),
                    "forager_cached_fallback_count": int(
                        state.get("forager_cached_fallback_count") or 0
                    ),
                    "symbols": self._symbols_summary(state["fallback_symbols"]),
                    "latest": state.get("latest_ema_fallback"),
                }
            group = {
                "bot": bot,
                "total_events": int(state.get("total_events") or 0),
                "latest_ts": state.get("latest_ts"),
                "event_types": dict(state["event_types"].most_common()),
                "statuses": dict(state["statuses"].most_common()),
                "reason_codes": dict(state["reason_codes"].most_common()),
                "psides": dict(state["psides"].most_common()),
                "latest": state.get("latest"),
                "forager_selection": {
                    key: value
                    for key, value in selection.items()
                    if value not in (None, {}, [])
                },
                "forager_feature_unavailable": {
                    key: value
                    for key, value in feature_unavailable.items()
                    if value not in (None, {}, [])
                },
                "ema_unavailable": {
                    key: value
                    for key, value in ema_unavailable.items()
                    if value not in (None, {}, [])
                },
                "ema_fallback": {
                    key: value
                    for key, value in ema_fallback.items()
                    if value not in (None, {}, [])
                },
            }
            groups.append(
                {
                    key: value
                    for key, value in group.items()
                    if value not in (None, {}, [])
                }
            )
        groups = sorted(
            groups,
            key=lambda item: (
                -int(
                    (
                        item.get("ema_unavailable", {}) or {}
                    ).get("candidate_symbol_sample_count", 0)
                    or 0
                ),
                -int(
                    (
                        item.get("forager_selection", {}) or {}
                    ).get("feature_unavailable_count", {})
                    .get("max", 0)
                    or 0
                ),
                -int(item.get("latest_ts", 0) or 0),
                str(item.get("bot") or ""),
            ),
        )
        limit = max(0, int(group_limit))
        return {
            "total_events": int(self.total_events),
            "bot_count": len(groups),
            "event_types": dict(self.event_types.most_common()),
            "groups_truncated": len(groups) > limit,
            "groups": groups[:limit],
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
        numeric = sorted(float(value) for value in values)
        if not numeric:
            return {}
        p95_index = (len(numeric) - 1) * 0.95
        p95_lower = int(math.floor(p95_index))
        p95_upper = int(math.ceil(p95_index))
        if p95_lower == p95_upper:
            p95 = numeric[p95_lower]
        else:
            p95_weight = p95_index - p95_lower
            p95 = numeric[p95_lower] * (1.0 - p95_weight) + numeric[p95_upper] * p95_weight

        integral_series = all(value.is_integer() for value in numeric)

        def clean(value: Any) -> float | int:
            number = float(value)
            if integral_series:
                return int(round(number))
            rounded = round(number, 3)
            if rounded.is_integer():
                return int(rounded)
            return rounded

        out: dict[str, Any] = {
            "latest": clean(latest),
            "count": len(numeric),
            "min": clean(numeric[0]),
            "max": clean(numeric[-1]),
            "mean": clean(statistics.fmean(numeric)),
            "median": clean(statistics.median(numeric)),
            "p95": clean(p95),
        }
        return out

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


class _ShutdownLatencyAccumulator:
    def __init__(self) -> None:
        self.accumulator = _PerformanceAccumulator()
        self.event_types: Counter[str] = Counter()
        self.stage_counts: Counter[str] = Counter()
        self.shutdowns_started = 0
        self.shutdowns_completed = 0

    def add(self, *, row: dict[str, Any], live_event: dict[str, Any]) -> None:
        event_type = str(live_event.get("event_type") or row.get("kind") or "")
        if event_type not in _SHUTDOWN_EVENT_TYPES:
            return
        self.event_types[event_type] += 1
        data = live_event.get("data") if isinstance(live_event.get("data"), dict) else {}
        if event_type == "bot.stopping":
            self.shutdowns_started += 1
            return
        if event_type == "bot.shutdown.stage":
            stage = data.get("stage") or live_event.get("reason_code") or "unknown"
            stage = str(stage)
            self.stage_counts[stage] += 1
            self.accumulator.add(
                row=row,
                live_event=live_event,
                operation=f"shutdown.stage.{stage}",
                value_ms=_elapsed_s_to_ms(data.get("elapsed_s")),
                trading_impact="observability",
                timing_kind="cumulative",
            )
            return
        if event_type == "bot.stopped":
            self.shutdowns_completed += 1
            self.accumulator.add(
                row=row,
                live_event=live_event,
                operation="shutdown.total",
                value_ms=_elapsed_s_to_ms(data.get("elapsed_s")),
                trading_impact="observability",
                timing_kind="duration",
            )

    def to_dict(self, *, group_limit: int = GROUP_LIMIT) -> dict[str, Any]:
        timing = self.accumulator.to_dict(group_limit=group_limit)
        return {
            "total_events": sum(self.event_types.values()),
            "shutdowns_started": int(self.shutdowns_started),
            "shutdowns_completed": int(self.shutdowns_completed),
            "event_types": dict(self.event_types.most_common()),
            "stage_counts": dict(self.stage_counts.most_common()),
            "total_groups": int(timing.get("total_groups") or 0),
            "groups_truncated": bool(timing.get("groups_truncated")),
            "groups": timing.get("groups") or [],
        }


class _ExecutionTimingAccumulator:
    def __init__(self) -> None:
        self.accumulator = _PerformanceAccumulator()
        self.action_starts: dict[tuple[str, int, str], int] = {}
        self.confirmation_starts: dict[tuple[str, int, str], int] = {}
        self.wave_starts: dict[tuple[str, int, str], int] = {}
        self.event_types: Counter[str] = Counter()
        self.missing_id_counts: Counter[str] = Counter()
        self.unpaired_terminal_counts: Counter[str] = Counter()
        self.starts_seen: Counter[str] = Counter()
        self.terminals_seen: Counter[str] = Counter()
        self.timing_observations: Counter[str] = Counter()
        self.terminal_outcome_counts: Counter[str] = Counter()

    def _record_terminal_outcome(self, event_type: str) -> None:
        if event_type in _EXECUTION_CREATE_TERMINALS:
            outcome = event_type.removeprefix("execution.create_")
            self.terminal_outcome_counts[f"create.{outcome}"] += 1
            return
        if event_type in _EXECUTION_CANCEL_TERMINALS:
            outcome = event_type.removeprefix("execution.cancel_")
            self.terminal_outcome_counts[f"cancel.{outcome}"] += 1
            return
        if event_type in _EXECUTION_CONFIRMATION_TERMINALS:
            outcome = event_type.removeprefix("execution.confirmation_")
            self.terminal_outcome_counts[f"confirmation.{outcome}"] += 1

    def _add_timing(
        self,
        *,
        row: dict[str, Any],
        live_event: dict[str, Any],
        operation: str,
        value_ms: int | None,
        timing_kind: str = "duration",
    ) -> None:
        self.accumulator.add(
            row=row,
            live_event=live_event,
            operation=operation,
            value_ms=value_ms,
            trading_impact="exchange_io",
            timing_kind=timing_kind,
        )
        if value_ms is not None:
            self.timing_observations[operation] += 1

    def _pair_elapsed(
        self,
        *,
        starts: dict[tuple[str, int, str], int],
        key: tuple[str, int, str],
        row: dict[str, Any],
        live_event: dict[str, Any],
        operation: str,
    ) -> None:
        start_ts = starts.pop(key, None)
        end_ts = _record_ts(row)
        if start_ts is None or end_ts is None:
            self.unpaired_terminal_counts[operation] += 1
            return
        self._add_timing(
            row=row,
            live_event=live_event,
            operation=operation,
            value_ms=max(0, int(end_ts) - int(start_ts)),
        )

    def add(
        self,
        *,
        row: dict[str, Any],
        live_event: dict[str, Any],
        cycle_scope: int,
    ) -> None:
        event_type = str(live_event.get("event_type") or row.get("kind") or "")
        if not (
            event_type.startswith("execution.")
            or event_type.startswith("order_wave.")
        ):
            return
        self.event_types[event_type] += 1
        bot = _bot_key(row, live_event)
        timestamp_ms = _record_ts(row)
        data = live_event.get("data") if isinstance(live_event.get("data"), dict) else {}

        if event_type == "order_wave.started":
            order_wave_id = _event_id_value(live_event, "order_wave_id")
            self.starts_seen["order_wave.total"] += 1
            if order_wave_id is None or timestamp_ms is None:
                self.missing_id_counts["order_wave.total"] += 1
                return
            self.wave_starts[(bot, int(cycle_scope), order_wave_id)] = int(timestamp_ms)
            return

        if event_type == "order_wave.completed":
            self.terminals_seen["order_wave.total"] += 1
            order_wave_id = _event_id_value(live_event, "order_wave_id")
            emitted_elapsed = _non_negative_ms(data.get("elapsed_ms"))
            if emitted_elapsed is not None:
                if order_wave_id is not None:
                    self.wave_starts.pop((bot, int(cycle_scope), order_wave_id), None)
                self._add_timing(
                    row=row,
                    live_event=live_event,
                    operation="order_wave.total",
                    value_ms=emitted_elapsed,
                )
                return
            if order_wave_id is None:
                self.missing_id_counts["order_wave.total"] += 1
                return
            self._pair_elapsed(
                starts=self.wave_starts,
                key=(bot, int(cycle_scope), order_wave_id),
                row=row,
                live_event=live_event,
                operation="order_wave.total",
            )
            return

        if event_type in {"execution.create_sent", "execution.cancel_sent"}:
            operation = (
                "execution.create_response"
                if event_type == "execution.create_sent"
                else "execution.cancel_response"
            )
            action_id = _event_id_value(live_event, "action_id")
            self.starts_seen[operation] += 1
            if action_id is None or timestamp_ms is None:
                self.missing_id_counts[operation] += 1
                return
            self.action_starts[(bot, int(cycle_scope), action_id)] = int(timestamp_ms)
            return

        if event_type in _EXECUTION_CREATE_TERMINALS | _EXECUTION_CANCEL_TERMINALS:
            self._record_terminal_outcome(event_type)
            operation = (
                "execution.create_response"
                if event_type in _EXECUTION_CREATE_TERMINALS
                else "execution.cancel_response"
            )
            self.terminals_seen[operation] += 1
            action_id = _event_id_value(live_event, "action_id")
            if action_id is None:
                self.missing_id_counts[operation] += 1
                return
            self._pair_elapsed(
                starts=self.action_starts,
                key=(bot, int(cycle_scope), action_id),
                row=row,
                live_event=live_event,
                operation=operation,
            )
            return

        if event_type == "execution.confirmation_requested":
            order_wave_id = _event_id_value(live_event, "order_wave_id")
            self.starts_seen["execution.confirmation"] += 1
            if order_wave_id is None or timestamp_ms is None:
                self.missing_id_counts["execution.confirmation"] += 1
                return
            self.confirmation_starts[(bot, int(cycle_scope), order_wave_id)] = int(timestamp_ms)
            return

        if event_type in _EXECUTION_CONFIRMATION_TERMINALS:
            self._record_terminal_outcome(event_type)
            operation = "execution.confirmation"
            self.terminals_seen[operation] += 1
            order_wave_id = _event_id_value(live_event, "order_wave_id")
            emitted_elapsed = _non_negative_ms(data.get("confirm_ms"))
            if emitted_elapsed is None:
                emitted_elapsed = _non_negative_ms(data.get("elapsed_ms"))
            if emitted_elapsed is not None:
                if order_wave_id is not None:
                    self.confirmation_starts.pop(
                        (bot, int(cycle_scope), order_wave_id),
                        None,
                    )
                self._add_timing(
                    row=row,
                    live_event=live_event,
                    operation=operation,
                    value_ms=emitted_elapsed,
                )
                return
            if order_wave_id is None:
                self.missing_id_counts[operation] += 1
                return
            self._pair_elapsed(
                starts=self.confirmation_starts,
                key=(bot, int(cycle_scope), order_wave_id),
                row=row,
                live_event=live_event,
                operation=operation,
            )

    def to_dict(self, *, group_limit: int = GROUP_LIMIT) -> dict[str, Any]:
        timing = self.accumulator.to_dict(group_limit=group_limit)
        pending_starts = Counter()
        for _key in self.action_starts:
            pending_starts["execution.write_response"] += 1
        for _key in self.confirmation_starts:
            pending_starts["execution.confirmation"] += 1
        for _key in self.wave_starts:
            pending_starts["order_wave.total"] += 1
        return {
            "total_events": sum(self.event_types.values()),
            "event_types": dict(self.event_types.most_common()),
            "starts_seen": dict(sorted(self.starts_seen.items())),
            "terminals_seen": dict(sorted(self.terminals_seen.items())),
            "timing_observations": dict(sorted(self.timing_observations.items())),
            "terminal_outcome_counts": dict(sorted(self.terminal_outcome_counts.items())),
            "missing_id_counts": dict(sorted(self.missing_id_counts.items())),
            "unpaired_terminal_counts": dict(sorted(self.unpaired_terminal_counts.items())),
            "pending_start_counts": dict(sorted(pending_starts.items())),
            "total_groups": int(timing.get("total_groups") or 0),
            "groups_truncated": bool(timing.get("groups_truncated")),
            "groups": timing.get("groups") or [],
        }


class _AccountStateChangeGroup:
    def __init__(self, *, bot: str, event_type: str) -> None:
        self.bot = bot
        self.event_type = event_type
        self.count = 0
        self.statuses: Counter[str] = Counter()
        self.reason_codes: Counter[str] = Counter()
        self.symbols: Counter[str] = Counter()
        self.psides: Counter[str] = Counter()
        self.sides: Counter[str] = Counter()
        self.components: Counter[str] = Counter()
        self.latest_ts: int | None = None

    def add(self, *, row: dict[str, Any], live_event: dict[str, Any]) -> None:
        self.count += 1
        status = live_event.get("status")
        if status is not None:
            self.statuses[str(status)] += 1
        reason_code = live_event.get("reason_code")
        if reason_code is not None:
            self.reason_codes[str(reason_code)] += 1
        symbol = live_event.get("symbol") or row.get("symbol")
        if symbol is not None:
            self.symbols[str(symbol)] += 1
        pside = live_event.get("pside") or row.get("pside")
        if pside is not None:
            self.psides[str(pside)] += 1
        side = live_event.get("side") or row.get("side")
        if side is not None:
            self.sides[str(side)] += 1
        component = live_event.get("component")
        if component is not None:
            self.components[str(component)] += 1
        ts = _record_ts(row)
        if ts is not None and (self.latest_ts is None or ts > self.latest_ts):
            self.latest_ts = ts

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "bot": self.bot,
                "event_type": self.event_type,
                "count": int(self.count),
                "latest_ts": self.latest_ts,
                "statuses": dict(sorted(self.statuses.items())),
                "reason_codes": dict(sorted(self.reason_codes.items())),
                "psides": dict(sorted(self.psides.items())),
                "sides": dict(sorted(self.sides.items())),
                "components": dict(sorted(self.components.items())),
                "symbols_sample": [
                    symbol
                    for symbol, _count in sorted(
                        self.symbols.items(), key=lambda item: (-item[1], item[0])
                    )[:10]
                ],
                "symbols_count": len(self.symbols),
            }.items()
            if value not in (None, {}, [])
        }


class _AccountStateChangeAccumulator:
    def __init__(self) -> None:
        self.groups: dict[tuple[str, str], _AccountStateChangeGroup] = {}
        self.event_types: Counter[str] = Counter()
        self.statuses: Counter[str] = Counter()
        self.reason_codes: Counter[str] = Counter()
        self.bot_counts: Counter[str] = Counter()

    def add(self, *, row: dict[str, Any], live_event: dict[str, Any]) -> None:
        event_type = str(live_event.get("event_type") or row.get("kind") or "")
        if event_type not in _ACCOUNT_STATE_CHANGE_EVENT_TYPES:
            return
        bot = _bot_key(row, live_event)
        self.event_types[event_type] += 1
        self.bot_counts[bot] += 1
        status = live_event.get("status")
        if status is not None:
            self.statuses[str(status)] += 1
        reason_code = live_event.get("reason_code")
        if reason_code is not None:
            self.reason_codes[str(reason_code)] += 1
        key = (bot, event_type)
        group = self.groups.get(key)
        if group is None:
            group = _AccountStateChangeGroup(bot=bot, event_type=event_type)
            self.groups[key] = group
        group.add(row=row, live_event=live_event)

    def groups_list(self) -> list[dict[str, Any]]:
        return sorted(
            (group.to_dict() for group in self.groups.values()),
            key=lambda item: (
                -int(item.get("count", 0) or 0),
                -int(item.get("latest_ts", 0) or 0),
                str(item.get("bot") or ""),
                str(item.get("event_type") or ""),
            ),
        )

    def to_dict(self, *, group_limit: int = GROUP_LIMIT) -> dict[str, Any]:
        groups = self.groups_list()
        return {
            "total_events": sum(self.event_types.values()),
            "event_types": dict(sorted(self.event_types.items())),
            "statuses": dict(sorted(self.statuses.items())),
            "reason_codes": dict(sorted(self.reason_codes.items())),
            "bot_count": len(self.bot_counts),
            "bots": [
                {"bot": bot, "events": int(count)}
                for bot, count in sorted(
                    self.bot_counts.items(), key=lambda item: (-item[1], item[0])
                )[: max(0, int(group_limit))]
            ],
            "bots_truncated": len(self.bot_counts) > int(group_limit),
            "total_groups": len(groups),
            "groups_truncated": len(groups) > int(group_limit),
            "groups": groups[: max(0, int(group_limit))],
        }


class _RiskActivityGroup:
    def __init__(self, *, bot: str, event_type: str) -> None:
        self.bot = bot
        self.event_type = event_type
        self.count = 0
        self.statuses: Counter[str] = Counter()
        self.reason_codes: Counter[str] = Counter()
        self.symbols: Counter[str] = Counter()
        self.psides: Counter[str] = Counter()
        self.components: Counter[str] = Counter()
        self.latest_ts: int | None = None

    def add(self, *, row: dict[str, Any], live_event: dict[str, Any]) -> None:
        self.count += 1
        status = live_event.get("status")
        if status is not None:
            self.statuses[str(status)] += 1
        reason_code = live_event.get("reason_code")
        if reason_code is not None:
            self.reason_codes[str(reason_code)] += 1
        symbol = live_event.get("symbol") or row.get("symbol")
        if symbol is not None:
            self.symbols[str(symbol)] += 1
        pside = live_event.get("pside") or row.get("pside")
        if pside is not None:
            self.psides[str(pside)] += 1
        component = live_event.get("component")
        if component is not None:
            self.components[str(component)] += 1
        ts = _record_ts(row)
        if ts is not None and (self.latest_ts is None or ts > self.latest_ts):
            self.latest_ts = ts

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "bot": self.bot,
                "event_type": self.event_type,
                "count": int(self.count),
                "latest_ts": self.latest_ts,
                "statuses": dict(sorted(self.statuses.items())),
                "reason_codes": dict(sorted(self.reason_codes.items())),
                "psides": dict(sorted(self.psides.items())),
                "components": dict(sorted(self.components.items())),
                "symbols_sample": [
                    symbol
                    for symbol, _count in sorted(
                        self.symbols.items(), key=lambda item: (-item[1], item[0])
                    )[:10]
                ],
                "symbols_count": len(self.symbols),
            }.items()
            if value not in (None, {}, [])
        }


class _RiskActivityAccumulator:
    def __init__(self) -> None:
        self.groups: dict[tuple[str, str], _RiskActivityGroup] = {}
        self.event_types: Counter[str] = Counter()
        self.statuses: Counter[str] = Counter()
        self.reason_codes: Counter[str] = Counter()
        self.bot_counts: Counter[str] = Counter()

    def add(self, *, row: dict[str, Any], live_event: dict[str, Any]) -> None:
        event_type = str(live_event.get("event_type") or row.get("kind") or "")
        if event_type not in _RISK_ACTIVITY_EVENT_TYPES:
            return
        bot = _bot_key(row, live_event)
        self.event_types[event_type] += 1
        self.bot_counts[bot] += 1
        status = live_event.get("status")
        if status is not None:
            self.statuses[str(status)] += 1
        reason_code = live_event.get("reason_code")
        if reason_code is not None:
            self.reason_codes[str(reason_code)] += 1
        key = (bot, event_type)
        group = self.groups.get(key)
        if group is None:
            group = _RiskActivityGroup(bot=bot, event_type=event_type)
            self.groups[key] = group
        group.add(row=row, live_event=live_event)

    def groups_list(self) -> list[dict[str, Any]]:
        return sorted(
            (group.to_dict() for group in self.groups.values()),
            key=lambda item: (
                -int(item.get("count", 0) or 0),
                -int(item.get("latest_ts", 0) or 0),
                str(item.get("bot") or ""),
                str(item.get("event_type") or ""),
            ),
        )

    def to_dict(self, *, group_limit: int = GROUP_LIMIT) -> dict[str, Any]:
        groups = self.groups_list()
        return {
            "total_events": sum(self.event_types.values()),
            "event_types": dict(sorted(self.event_types.items())),
            "statuses": dict(sorted(self.statuses.items())),
            "reason_codes": dict(sorted(self.reason_codes.items())),
            "bot_count": len(self.bot_counts),
            "bots": [
                {"bot": bot, "events": int(count)}
                for bot, count in sorted(
                    self.bot_counts.items(), key=lambda item: (-item[1], item[0])
                )[: max(0, int(group_limit))]
            ],
            "bots_truncated": len(self.bot_counts) > int(group_limit),
            "total_groups": len(groups),
            "groups_truncated": len(groups) > int(group_limit),
            "groups": groups[: max(0, int(group_limit))],
        }


def _slowest_blockers(
    *,
    performance_groups: list[dict[str, Any]],
    decision_boundary_groups: list[dict[str, Any]],
    input_staleness_groups: list[dict[str, Any]],
    execution_timing_groups: list[dict[str, Any]],
    group_limit: int = GROUP_LIMIT,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for source_section, groups in (
        ("performance", performance_groups),
        ("decision_boundary_lag", decision_boundary_groups),
        ("input_staleness", input_staleness_groups),
        ("execution_timing", execution_timing_groups),
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


def _operation_category(operation: Any) -> str:
    text = str(operation or "")
    for prefix, category in (
        ("startup.", "startup"),
        ("state_refresh.", "state_refresh"),
        ("remote_call.", "remote_call"),
        ("hsl_replay.", "hsl_replay"),
        ("cache_", "cache"),
        ("forager_", "forager"),
        ("decision_boundary.", "decision_boundary"),
        ("input_staleness.", "input_staleness"),
        ("execution.", "execution"),
        ("order_wave.", "execution"),
        ("shutdown.", "shutdown"),
        ("cycle.", "cycle"),
    ):
        if text.startswith(prefix):
            return category
    return "other"


def _operation_duration_table(
    *,
    performance_groups: list[dict[str, Any]],
    decision_boundary_groups: list[dict[str, Any]],
    input_staleness_groups: list[dict[str, Any]],
    execution_timing_groups: list[dict[str, Any]],
    shutdown_latency_groups: list[dict[str, Any]],
    group_limit: int = GROUP_LIMIT,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    trading_impact_counts: Counter[str] = Counter()
    blocking_scope_counts: Counter[str] = Counter()
    operation_category_counts: Counter[str] = Counter()
    timing_kind_counts: Counter[str] = Counter()
    for source_section, groups in (
        ("performance", performance_groups),
        ("decision_boundary_lag", decision_boundary_groups),
        ("input_staleness", input_staleness_groups),
        ("execution_timing", execution_timing_groups),
        ("shutdown_latency", shutdown_latency_groups),
    ):
        for group in groups:
            operation = group.get("operation")
            category = _operation_category(operation)
            impact = str(group.get("trading_impact") or "unknown")
            blocking_scope = _BLOCKING_SCOPE_BY_IMPACT.get(impact, impact)
            timing_kind = str(group.get("timing_kind") or "duration")
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
            item["operation_category"] = category
            item["blocking_scope"] = blocking_scope
            items.append(item)
            trading_impact_counts[impact] += 1
            blocking_scope_counts[blocking_scope] += 1
            operation_category_counts[category] += 1
            timing_kind_counts[timing_kind] += 1
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
        "trading_impact_counts": dict(sorted(trading_impact_counts.items())),
        "blocking_scope_counts": dict(sorted(blocking_scope_counts.items())),
        "operation_category_counts": dict(sorted(operation_category_counts.items())),
        "timing_kind_counts": dict(sorted(timing_kind_counts.items())),
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
    if event_type == "cache.warmup_decision":
        return "blocks_indicator_readiness"
    if event_type == "cache.load.completed":
        return "blocks_indicator_readiness"
    if event_type == "cache.flush.completed":
        return "diagnostics_only"
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

    if event_type in _CACHE_EVENT_TYPES:
        operation = event_type.replace(".", "_")
        accumulator.add(
            row=row,
            live_event=live_event,
            operation=operation,
            value_ms=_non_negative_ms(data.get("elapsed_ms")),
            trading_impact=_trading_impact_for_event(event_type, operation),
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
    hsl_replay_profile = _HslReplayProfileAccumulator()
    cache_warmup = _CacheWarmupAccumulator()
    forager_ema_readiness = _ForagerEmaReadinessAccumulator()
    resource_pressure = _ResourcePressureAccumulator()
    shutdown_latency = _ShutdownLatencyAccumulator()
    execution_timing = _ExecutionTimingAccumulator()
    account_state_changes = _AccountStateChangeAccumulator()
    risk_activity = _RiskActivityAccumulator()
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
                    hsl_replay_profile.add(row=row, live_event=live_event)
                    cache_warmup.add(row=row, live_event=live_event)
                    forager_ema_readiness.add(row=row, live_event=live_event)
                    resource_pressure.add(row=row, live_event=live_event)
                    shutdown_latency.add(row=row, live_event=live_event)
                    execution_timing.add(
                        row=row,
                        live_event=live_event,
                        cycle_scope=cycle_scope,
                    )
                    account_state_changes.add(row=row, live_event=live_event)
                    risk_activity.add(row=row, live_event=live_event)
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
    execution_timing_groups = execution_timing.accumulator.groups_list()
    shutdown_latency_groups = shutdown_latency.accumulator.groups_list()
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
        "hsl_replay_profile": hsl_replay_profile.to_dict(group_limit=group_limit),
        "cache_warmup": cache_warmup.to_dict(group_limit=group_limit),
        "forager_ema_readiness": forager_ema_readiness.to_dict(group_limit=group_limit),
        "resource_pressure": resource_pressure.to_dict(group_limit=group_limit),
        "shutdown_latency": shutdown_latency.to_dict(group_limit=group_limit),
        "execution_timing": execution_timing.to_dict(group_limit=group_limit),
        "account_state_changes": account_state_changes.to_dict(group_limit=group_limit),
        "risk_activity": risk_activity.to_dict(group_limit=group_limit),
        "operation_durations": _operation_duration_table(
            performance_groups=performance_groups,
            decision_boundary_groups=decision_boundary_groups,
            input_staleness_groups=input_staleness_groups,
            execution_timing_groups=execution_timing_groups,
            shutdown_latency_groups=shutdown_latency_groups,
            group_limit=group_limit,
        ),
        "slowest_blockers": _slowest_blockers(
            performance_groups=performance_groups,
            decision_boundary_groups=decision_boundary_groups,
            input_staleness_groups=input_staleness_groups,
            execution_timing_groups=execution_timing_groups,
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
            "snapshot_surface_age_rows": int(
                input_staleness.get("snapshot_surface_age_rows") or 0
            ),
            "snapshot_market_summaries_seen": int(
                input_staleness.get("snapshot_market_summaries_seen") or 0
            ),
            "snapshot_market_stale_count": int(
                input_staleness.get("snapshot_market_stale_count") or 0
            ),
            "rust_calls_seen": int(input_staleness.get("rust_calls_seen") or 0),
            "packet_refs_missing": int(input_staleness.get("packet_refs_missing") or 0),
            "snapshot_to_rust_exact_matches": int(
                input_staleness.get("snapshot_to_rust_exact_matches") or 0
            ),
            "snapshot_to_rust_latest_snapshot_matches": int(
                input_staleness.get("snapshot_to_rust_latest_snapshot_matches") or 0
            ),
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
    if isinstance(report.get("hsl_replay_profile"), dict):
        hsl_replay_profile = dict(report["hsl_replay_profile"])
        hsl_groups = (
            hsl_replay_profile.get("groups")
            if isinstance(hsl_replay_profile.get("groups"), list)
            else []
        )
        hsl_replay_profile["groups"] = hsl_groups[: max(0, int(group_limit))]
        if len(hsl_groups) > max(0, int(group_limit)):
            hsl_replay_profile["groups_truncated"] = True
        summary["hsl_replay_profile"] = hsl_replay_profile
    if isinstance(report.get("cache_warmup"), dict):
        cache_warmup = dict(report["cache_warmup"])
        cache_groups = (
            cache_warmup.get("groups")
            if isinstance(cache_warmup.get("groups"), list)
            else []
        )
        cache_warmup["groups"] = cache_groups[: max(0, int(group_limit))]
        if len(cache_groups) > max(0, int(group_limit)):
            cache_warmup["groups_truncated"] = True
        summary["cache_warmup"] = cache_warmup
    if isinstance(report.get("forager_ema_readiness"), dict):
        forager_ema_readiness = dict(report["forager_ema_readiness"])
        readiness_groups = (
            forager_ema_readiness.get("groups")
            if isinstance(forager_ema_readiness.get("groups"), list)
            else []
        )
        forager_ema_readiness["groups"] = readiness_groups[: max(0, int(group_limit))]
        if len(readiness_groups) > max(0, int(group_limit)):
            forager_ema_readiness["groups_truncated"] = True
        summary["forager_ema_readiness"] = forager_ema_readiness
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
    if isinstance(report.get("shutdown_latency"), dict):
        shutdown_latency = dict(report["shutdown_latency"])
        shutdown_groups = (
            shutdown_latency.get("groups")
            if isinstance(shutdown_latency.get("groups"), list)
            else []
        )
        shutdown_latency["groups"] = shutdown_groups[: max(0, int(group_limit))]
        if len(shutdown_groups) > max(0, int(group_limit)):
            shutdown_latency["groups_truncated"] = True
        summary["shutdown_latency"] = shutdown_latency
    if isinstance(report.get("execution_timing"), dict):
        execution_timing = dict(report["execution_timing"])
        execution_groups = (
            execution_timing.get("groups")
            if isinstance(execution_timing.get("groups"), list)
            else []
        )
        execution_timing["groups"] = execution_groups[: max(0, int(group_limit))]
        if len(execution_groups) > max(0, int(group_limit)):
            execution_timing["groups_truncated"] = True
        summary["execution_timing"] = execution_timing
    if isinstance(report.get("account_state_changes"), dict):
        account_state_changes = dict(report["account_state_changes"])
        state_groups = (
            account_state_changes.get("groups")
            if isinstance(account_state_changes.get("groups"), list)
            else []
        )
        state_bots = (
            account_state_changes.get("bots")
            if isinstance(account_state_changes.get("bots"), list)
            else []
        )
        account_state_changes["groups"] = state_groups[: max(0, int(group_limit))]
        account_state_changes["bots"] = state_bots[: max(0, int(group_limit))]
        if len(state_groups) > max(0, int(group_limit)):
            account_state_changes["groups_truncated"] = True
        if len(state_bots) > max(0, int(group_limit)):
            account_state_changes["bots_truncated"] = True
        summary["account_state_changes"] = account_state_changes
    if isinstance(report.get("risk_activity"), dict):
        risk_activity = dict(report["risk_activity"])
        risk_groups = (
            risk_activity.get("groups")
            if isinstance(risk_activity.get("groups"), list)
            else []
        )
        risk_bots = (
            risk_activity.get("bots")
            if isinstance(risk_activity.get("bots"), list)
            else []
        )
        risk_activity["groups"] = risk_groups[: max(0, int(group_limit))]
        risk_activity["bots"] = risk_bots[: max(0, int(group_limit))]
        if len(risk_groups) > max(0, int(group_limit)):
            risk_activity["groups_truncated"] = True
        if len(risk_bots) > max(0, int(group_limit)):
            risk_activity["bots_truncated"] = True
        summary["risk_activity"] = risk_activity
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
    if isinstance(report.get("operation_durations"), dict):
        operation_durations = report["operation_durations"]
        operation_groups = (
            operation_durations.get("groups")
            if isinstance(operation_durations.get("groups"), list)
            else []
        )
        summary["operation_durations"] = {
            "total_groups": int(operation_durations.get("total_groups") or 0),
            "groups_truncated": bool(operation_durations.get("groups_truncated")),
            "trading_impact_counts": operation_durations.get("trading_impact_counts") or {},
            "blocking_scope_counts": operation_durations.get("blocking_scope_counts") or {},
            "operation_category_counts": operation_durations.get(
                "operation_category_counts"
            )
            or {},
            "timing_kind_counts": operation_durations.get("timing_kind_counts") or {},
            "groups": operation_groups[: max(0, int(group_limit))],
        }
    if report.get("event_window") is not None:
        summary["event_window"] = report.get("event_window")
    if report.get("filters") is not None:
        summary["filters"] = report.get("filters")
    if report.get("issues"):
        summary["issues"] = report.get("issues")
    return summary
