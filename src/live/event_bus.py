from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import queue
import re
import threading
import time
from types import MappingProxyType
import uuid
from typing import Any, Iterable, Mapping, Protocol


SCHEMA_VERSION = 1
REDACTED = "[redacted]"
LIVE_EVENT_MONITOR_PAYLOAD_KEY = "_live_event"
LIVE_EVENT_ID_KEYS = (
    "bot_id",
    "cycle_id",
    "snapshot_id",
    "plan_id",
    "action_id",
    "order_wave_id",
    "remote_call_id",
    "remote_call_group_id",
)
LIVE_EVENT_DEBUG_PROFILE_ENV = "PASSIVBOT_LIVE_EVENT_DEBUG_PROFILES"
LIVE_EVENT_CONSOLE_ENV = "PASSIVBOT_LIVE_EVENT_CONSOLE"
LIVE_EVENT_DEBUG_PROFILES = (
    "cache",
    "candles",
    "ema",
    "execution",
    "fills",
    "forager",
    "hsl",
    "remote_calls",
    "rust",
    "startup",
    "state",
)
_LIVE_EVENT_DEBUG_PROFILE_SET = frozenset(LIVE_EVENT_DEBUG_PROFILES)
_LIVE_EVENT_DEBUG_PROFILE_ALIASES = {
    "warmup": "cache",
    "candle": "candles",
    "caches": "cache",
    "emas": "ema",
    "ema-readiness": "ema",
    "ema_readiness": "ema",
    "remote": "remote_calls",
    "remote_call": "remote_calls",
    "remote-call": "remote_calls",
    "remote-calls": "remote_calls",
}
_MONITOR_PUBLISHER_PHASE_TIMING_KEYS = (
    "lock_wait_ns",
    "rotation_ns",
    "persist_ns",
    "maintenance_ns",
    "manifest_checkpoint_count",
    "manifest_checkpoint_ns_total",
    "manifest_checkpoint_ns_max",
    "retention_run_count",
    "retention_ns_total",
    "retention_ns_max",
    "retention_thread_cpu_ns_total",
    "retention_thread_cpu_ns_max",
    "retention_non_cpu_ns_total",
    "retention_non_cpu_ns_max",
    "retention_inventory_ns_total",
    "retention_inventory_ns_max",
    "retention_age_filter_ns_total",
    "retention_age_filter_ns_max",
    "retention_cap_prune_ns_total",
    "retention_cap_prune_ns_max",
    "retention_age_unlink_ns_total",
    "retention_age_unlink_ns_max",
    "retention_cap_unlink_ns_total",
    "retention_cap_unlink_ns_max",
    "retention_inventory_entries_visited",
    "retention_inventory_candidates",
    "retention_age_deleted",
    "retention_cap_deleted",
)
_MONITOR_PHASE_TIMING_KEYS = ("prepare_ns", *_MONITOR_PUBLISHER_PHASE_TIMING_KEYS)
_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_CONTROL_CHARACTER_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _empty_monitor_phase_timing() -> dict[str, int]:
    return {key: 0 for key in _MONITOR_PHASE_TIMING_KEYS}

_STARTUP_PHASE_READINESS_CONTRACTS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "account": ("account_critical", "protective_blocker"),
        "hsl": ("held_position_protective", "protective_blocker"),
        "startup": ("execution_loop", "protective_blocker"),
        "market": ("first_market_state", "cycle_delay"),
        "full-warmup": ("background_candles_complete", "entry_blocker"),
    }
)
STARTUP_TIMING_PHASES = frozenset(
    {
        "account",
        "active-candle",
        "full-warmup",
        "hsl",
        "market",
        "startup",
    }
)


def startup_phase_readiness_contract(phase: object) -> dict[str, str] | None:
    """Return bounded readiness metadata for a known startup phase."""
    if not isinstance(phase, str):
        return None
    values = _STARTUP_PHASE_READINESS_CONTRACTS.get(phase)
    if values is None:
        return None
    return {
        "readiness_scope": values[0],
        "trading_impact": values[1],
    }


def startup_timing_phase(data: object) -> str | None:
    """Return canonical startup phase, with stage accepted only as legacy fallback."""
    if not isinstance(data, Mapping):
        return None
    phase = str(data.get("phase") or "").strip()
    stage = str(data.get("stage") or "").strip()
    if phase and stage and phase != stage:
        return None
    value = phase or stage
    if not value:
        return None
    return value if value in STARTUP_TIMING_PHASES else "other"


class EventTypes:
    BOT_STARTED = "bot.started"
    BOT_READY = "bot.ready"
    BOT_STARTUP_TIMING = "bot.startup_timing"
    BOT_STOPPING = "bot.stopping"
    BOT_SHUTDOWN_STAGE = "bot.shutdown.stage"
    BOT_STOPPED = "bot.stopped"
    HEALTH_SUMMARY = "health.summary"
    RESOURCE_MEMORY_SNAPSHOT = "resource.memory_snapshot"
    MARKET_SNAPSHOT_DIAGNOSTIC_SKIPPED = "market.snapshot_diagnostic_skipped"
    CYCLE_STARTED = "cycle.started"
    CYCLE_COMPLETED = "cycle.completed"
    CYCLE_DEGRADED = "cycle.degraded"
    DATA_PACKET_UPDATED = "data_packet.updated"
    SNAPSHOT_BUILT = "snapshot.built"
    PLANNING_UNAVAILABLE = "planning.unavailable"
    PLANNING_DEFER_SUMMARY = "planning.defer_summary"
    PLANNING_SYMBOL_STATE = "planning.symbol_state"
    FORAGER_SELECTION = "forager.selection"
    FORAGER_FEATURE_UNAVAILABLE = "forager.feature_unavailable"
    FORAGER_ELIGIBILITY_CHANGED = "forager.eligibility_changed"
    CONFIG_MARKET_COMPATIBILITY = "config.market_compatibility"
    EMA_BUNDLE_STARTED = "ema.bundle.started"
    EMA_BUNDLE_COMPLETED = "ema.bundle.completed"
    EMA_FALLBACK_USED = "ema.fallback_used"
    EMA_UNAVAILABLE = "ema.unavailable"
    CANDLE_COVERAGE_CHECKED = "candle.coverage_checked"
    CANDLE_TAIL_PROJECTED = "candle.tail_projected"
    CACHE_LOAD_COMPLETED = "cache.load.completed"
    CACHE_FLUSH_COMPLETED = "cache.flush.completed"
    CACHE_WARMUP_DECISION = "cache.warmup_decision"
    EXCHANGE_CONFIG_REFRESH = "exchange.config_refresh"
    EXCHANGE_TIME_SYNC = "exchange.time_sync"
    WEBSOCKET_RECONNECT = "websocket.reconnect"
    REMOTE_CALL_STARTED = "remote_call.started"
    REMOTE_CALL_SUCCEEDED = "remote_call.succeeded"
    REMOTE_CALL_FAILED = "remote_call.failed"
    REMOTE_CALL_THROTTLED = "remote_call.throttled"
    STATE_REFRESH_PROGRESS = "state.refresh_progress"
    STATE_REFRESH_TIMING = "state.refresh_timing"
    RUST_ORCHESTRATOR_CALLED = "rust_orchestrator.called"
    RUST_ORCHESTRATOR_RETURNED = "rust_orchestrator.returned"
    ACTION_PLANNED = "action.planned"
    ORDER_WAVE_STARTED = "order_wave.started"
    ORDER_WAVE_COMPLETED = "order_wave.completed"
    EXECUTION_CREATE_SENT = "execution.create_sent"
    EXECUTION_CREATE_CONNECTOR_CALL_STARTED = (
        "execution.create_connector_call_started"
    )
    EXECUTION_CREATE_SUCCEEDED = "execution.create_succeeded"
    EXECUTION_CREATE_FAILED = "execution.create_failed"
    EXECUTION_CREATE_REJECTED = "execution.create_rejected"
    EXECUTION_CREATE_DEFERRED = "execution.create_deferred"
    EXECUTION_CREATE_SKIPPED = "execution.create_skipped"
    ENTRY_INITIAL_ELIGIBILITY = "entry.initial_eligibility"
    ENTRY_INITIAL_DISTANCE_GATE_BLOCKED = "entry.initial_distance_gate_blocked"
    ENTRY_INITIAL_DISTANCE_GATE_CLEARED = "entry.initial_distance_gate_cleared"
    ENTRY_MIN_EFFECTIVE_COST_BLOCKED = "entry.min_effective_cost_blocked"
    EXECUTION_CANCEL_SENT = "execution.cancel_sent"
    EXECUTION_CANCEL_CONNECTOR_CALL_STARTED = (
        "execution.cancel_connector_call_started"
    )
    EXECUTION_CANCEL_SUCCEEDED = "execution.cancel_succeeded"
    EXECUTION_CANCEL_FAILED = "execution.cancel_failed"
    EXECUTION_CANCEL_AMBIGUOUS_TERMINAL = "execution.cancel_ambiguous_terminal"
    EXECUTION_AMBIGUOUS = "execution.ambiguous"
    EXECUTION_CONFIRMATION_REQUESTED = "execution.confirmation_requested"
    EXECUTION_CONFIRMATION_SATISFIED = "execution.confirmation_satisfied"
    EXECUTION_CONFIRMATION_TIMEOUT = "execution.confirmation_timeout"
    FILLS_REFRESH_SUMMARY = "fills.refresh_summary"
    FILL_INGESTED = "fill.ingested"
    FILLS_INGESTED_SUMMARY = "fills.ingested_summary"
    POSITION_CHANGED = "position.changed"
    BALANCE_CHANGED = "balance.changed"
    RISK_MODE_CHANGED = "risk.mode_changed"
    HSL_TRANSITION = "hsl.transition"
    HSL_STATUS = "hsl.status"
    HSL_RAW_RED_PENDING = "hsl.raw_red_pending"
    HSL_REPLAY_STARTED = "hsl.replay.started"
    HSL_REPLAY_PROGRESS = "hsl.replay.progress"
    HSL_REPLAY_COMPLETED = "hsl.replay.completed"
    HSL_REPLAY_FAILED = "hsl.replay.failed"
    HSL_REPLAY_CACHE = "hsl.replay.cache"
    HSL_RED_TRIGGERED = "hsl.red_triggered"
    HSL_RED_FINALIZED_WITHOUT_ORDER = "hsl.red_finalized_without_order"
    HSL_COOLDOWN_STARTED = "hsl.cooldown_started"
    HSL_COOLDOWN_ENDED = "hsl.cooldown_ended"
    TRAILING_STATUS = "trailing.status"
    UNSTUCK_STATUS = "unstuck.status"
    UNSTUCK_SELECTION = "unstuck.selection"
    RISK_ENTRY_COOLDOWN_DELTA_ANCHORED = "risk.entry_cooldown_delta_anchored"
    REALIZED_LOSS_GATE_BLOCKED = "risk.realized_loss_gate_blocked"
    SINK_DEGRADED = "sink.degraded"


class EventTags:
    ACCOUNT = "account"
    ACTION = "action"
    AUTHORITATIVE = "authoritative"
    AVAILABILITY = "availability"
    BALANCE = "balance"
    BUNDLE = "bundle"
    CACHE = "cache"
    CANDLE = "candle"
    CONFIRMATION = "confirmation"
    COVERAGE = "coverage"
    CYCLE = "cycle"
    DEFER = "defer"
    DEGRADED = "degraded"
    EMA = "ema"
    ENTRY = "entry"
    EXECUTION = "execution"
    EXCHANGE = "exchange"
    FALLBACK = "fallback"
    FILL = "fill"
    FILLS = "fills"
    FLUSH = "flush"
    FORAGER = "forager"
    GATE = "gate"
    HEALTH = "health"
    LOAD = "load"
    LOGGING = "logging"
    MARKET = "market"
    MEMORY = "memory"
    MODE = "mode"
    ORDER = "order"
    PLANNING = "planning"
    POSITION = "position"
    REFRESH = "refresh"
    REMOTE_CALL = "remote_call"
    RESOURCE = "resource"
    RISK = "risk"
    RUST = "rust"
    SELECTION = "selection"
    SINK = "sink"
    SNAPSHOT = "snapshot"
    STATE = "state"
    SUMMARY = "summary"
    TAIL = "tail"
    TIMEOUT = "timeout"
    TIME_SYNC = "time_sync"
    TRAILING = "trailing"
    UNAVAILABLE = "unavailable"
    UNSTUCK = "unstuck"
    WARMUP = "warmup"
    WAVE = "wave"
    WEBSOCKET = "websocket"


class ReasonCodes:
    AUTHORITATIVE_CONFIRMATION = "authoritative_confirmation"
    AUTHORITATIVE_CONFIRMATION_TIMEOUT = "authoritative_confirmation_timeout"
    BALANCE_CHANGED = "balance_changed"
    CANDLE_DISK_FLUSH_COMPLETED = "candle_disk_flush_completed"
    CANDLE_DISK_LOAD_COMPLETED = "candle_disk_load_completed"
    CONNECTOR_CALL_STARTED = "connector_call_started"
    EMA_FALLBACK_USED = "ema_fallback_used"
    EXCHANGE_ACKNOWLEDGED = "exchange_acknowledged"
    EXCHANGE_CONFIG_REFRESH = "exchange_config_refresh"
    EXCHANGE_CONFIG_REFRESH_FAILED = "exchange_config_refresh_failed"
    EXCHANGE_EXCEPTION = "exchange_exception"
    EXCHANGE_TIME_SYNC = "exchange_time_sync"
    EXCHANGE_TIME_SYNC_UNAVAILABLE = "exchange_time_sync_unavailable"
    WEBSOCKET_RECONNECT = "websocket_reconnect"
    EXECUTION_LOOP_ERROR_BURST = "execution_loop_error_burst"
    FRESH_ENTRY_ELIGIBILITY = "fresh_entry_eligibility"
    FILL_CACHE_DOCTOR_REPORT = "fill_cache_doctor_report"
    FILL_CACHE_QUARANTINED = "fill_cache_quarantined"
    FILL_CACHE_READY = "fill_cache_ready"
    FILL_CACHE_REBUILD_STARTED = "fill_cache_rebuild_started"
    INITIAL_ENTRY_DISTANCE_GATE = "initial_entry_distance_gate"
    LENGTH_MISMATCH = "length_mismatch"
    LIMIT_ORDER_CREATE_MARKET_DISTANCE = "limit_order_create_market_distance"
    LOW_BALANCE = "low_balance"
    MARKET_SNAPSHOT_DIAGNOSTIC_SKIPPED = "market_snapshot_diagnostic_skipped"
    MIN_EFFECTIVE_COST_BLOCKED = "min_effective_cost_blocked"
    MEMORY_SNAPSHOT = "memory_snapshot"
    NEW_FILL = "new_fill"
    NEW_FILL_BATCH = "new_fill_batch"
    OPEN_TAIL_PROJECTION = "open_tail_projection"
    OPTIONAL_EMA_DROPPED = "optional_ema_dropped"
    PENDING_EXCHANGE_CONFIG = "pending_exchange_config"
    PERIODIC_HEALTH_SUMMARY = "periodic_health_summary"
    PRE_CREATE_MARKET_SNAPSHOT_UNAVAILABLE = (
        "pre_create_market_snapshot_unavailable"
    )
    PRE_CREATE_PLANNING_SNAPSHOT_INVALID = "pre_create_planning_snapshot_invalid"
    QUEUE_FULL = "queue_full"
    RANKING_FEATURES_UNAVAILABLE = "ranking_features_unavailable"
    FORAGER_ELIGIBILITY_MEMBERSHIP_CHANGED = (
        "forager_eligibility_membership_changed"
    )
    CONFIG_MARKET_UNSUPPORTED = "config_market_unsupported"
    CONFIG_ISOLATED_ONLY_MARKET_BLOCKED = "config_isolated_only_market_blocked"
    CONFIG_STOCK_PERP_WRONG_EXCHANGE = "config_stock_perp_wrong_exchange"
    CONFIG_STOCK_PERP_UNAVAILABLE_MARKET = "config_stock_perp_unavailable_market"
    CONFIG_HIP3_ACCOUNT_MODE_UNSUPPORTED = "config_hip3_account_mode_unsupported"
    RECENT_EXECUTION = "recent_execution"
    REMOTE_FETCH = "remote_fetch"
    RISK_ENTRY_COOLDOWN_POSITION_DELTA = "entry_cooldown_position_delta"
    REQUIRED_CANDLE_DISK_COVERAGE = "required_candle_disk_coverage"
    REQUIRED_EMA_UNAVAILABLE = "required_ema_unavailable"
    HSL_BALANCE_OVERRIDE_ACCOUNT_LEVEL_REPLAY_UNSAFE = (
        "hsl_balance_override_account_level_replay_unsafe"
    )
    HSL_HISTORY_EMPTY = "hsl_history_empty"
    HSL_HISTORY_INPUTS_LOADED = "hsl_history_inputs_loaded"
    HSL_HELD_PROTECTIVE_READY = "hsl_held_protective_ready"
    HSL_PRICE_HISTORY_FETCH_COMPLETED = "hsl_price_history_fetch_completed"
    HSL_PRICE_HISTORY_FETCH_STARTED = "hsl_price_history_fetch_started"
    HSL_PRICE_HISTORY_SYMBOL_FETCH_COMPLETED = (
        "hsl_price_history_symbol_fetch_completed"
    )
    HSL_PRICE_HISTORY_SYMBOL_FETCH_STARTED = "hsl_price_history_symbol_fetch_started"
    HSL_REPLAY_CACHE_HIT = "hsl_replay_cache_hit"
    HSL_REPLAY_CACHE_MISS = "hsl_replay_cache_miss"
    HSL_REPLAY_CACHE_REJECTED = "hsl_replay_cache_rejected"
    HSL_REPLAY_CACHE_WRITTEN = "hsl_replay_cache_written"
    HSL_REPLAY_CACHE_WRITE_FAILED = "hsl_replay_cache_write_failed"
    HSL_REPLAY_PENDING = "hsl_replay_pending"
    HSL_RAW_RED_PENDING_EMA_CONFIRMATION = "hsl_raw_red_pending_ema_confirmation"
    HSL_RED_FINALIZED_WITHOUT_EXCHANGE_ORDER = (
        "hsl_red_finalized_without_exchange_order"
    )
    HSL_TIMELINE_REPLAY_COMPLETED = "hsl_timeline_replay_completed"
    HSL_TIMELINE_REPLAY_STARTED = "hsl_timeline_replay_started"
    RUST_OUTPUT_ACTIONS = "rust_output_actions"
    SINK_PIPELINE_CLOSING = "pipeline_closing"
    SNAPSHOT_SYMBOL_STATE = "snapshot_symbol_state"
    STARTUP_PHASE_READY = "startup_phase_ready"
    STAGED_REFRESH_PROGRESS = "staged_refresh_progress"
    STAGED_REFRESH_TIMING = "staged_refresh_timing"
    STATE_CHANGE_DETECTED = "state_change_detected"
    SUBMITTED_TO_EXCHANGE = "submitted_to_exchange"
    REALIZED_LOSS_GATE_BLOCKED = "realized_loss_gate_blocked"
    TRAILING_STATUS = "trailing_status"
    UNSTUCK_SELECTION = "unstuck_selection"
    UNSTUCK_STATUS = "unstuck_status"
    WARMUP_CACHE_DECISION = "warmup_cache_decision"


def authoritative_reason_code(surface: object) -> str:
    return f"authoritative_{surface}"


def sink_failed_reason_code(name: object) -> str:
    return f"{name}_sink_failed"


def _split_debug_profile_string(value: str) -> list[str]:
    return [part for part in re.split(r"[\s,;]+", value.strip()) if part]


def normalize_live_event_debug_profiles(value: Any) -> tuple[str, ...]:
    """Normalize live-event debug profile config/env values.

    The special profile ``all`` expands to all currently known profiles. Empty,
    false-like, or ``none`` values disable profile enrichment.
    """
    if value is None or value is False:
        return ()
    if value is True:
        raw_values: list[Any] = ["all"]
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"0", "false", "no", "none", "off"}:
            return ()
        raw_values = _split_debug_profile_string(stripped)
    elif isinstance(value, (list, tuple, set, frozenset)):
        raw_values = list(value)
    else:
        raw_values = [value]

    normalized: list[str] = []
    unknown: list[str] = []
    for raw in raw_values:
        item = str(raw).strip().lower().replace("-", "_")
        if not item or item in {"0", "false", "no", "none", "off"}:
            continue
        if item == "all":
            normalized.extend(LIVE_EVENT_DEBUG_PROFILES)
            continue
        item = _LIVE_EVENT_DEBUG_PROFILE_ALIASES.get(item, item)
        if item not in _LIVE_EVENT_DEBUG_PROFILE_SET:
            unknown.append(str(raw))
            continue
        normalized.append(item)
    if unknown:
        allowed = ", ".join((*LIVE_EVENT_DEBUG_PROFILES, "all"))
        raise ValueError(
            f"unknown live event debug profile(s): {', '.join(unknown)}; "
            f"allowed values: {allowed}"
        )
    return tuple(sorted(set(normalized)))


def live_event_debug_profile_enabled(holder: Any, profile: str) -> bool:
    item = str(profile).strip().lower().replace("-", "_")
    item = _LIVE_EVENT_DEBUG_PROFILE_ALIASES.get(item, item)
    profiles = getattr(holder, "live_event_debug_profiles", None)
    if profiles is None:
        pipeline = getattr(holder, "_live_event_pipeline", None)
        profiles = getattr(pipeline, "debug_profiles", ())
    return item in set(profiles or ())


def normalize_live_event_console_enabled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"", "0", "false", "no", "none", "off"}:
            return False
        if cleaned in {"1", "true", "yes", "on", "console"}:
            return True
    return bool(value)


def resolve_live_event_console_enabled(
    *,
    config_value: Any = None,
    env_value: Any = None,
    default: bool = True,
) -> bool:
    if env_value is not None:
        return normalize_live_event_console_enabled(env_value)
    if config_value is not None:
        return normalize_live_event_console_enabled(config_value)
    return bool(default)


PHASE1_EVENT_TYPES = {
    EventTypes.BOT_STARTED,
    EventTypes.BOT_READY,
    EventTypes.BOT_STARTUP_TIMING,
    EventTypes.BOT_STOPPING,
    EventTypes.BOT_SHUTDOWN_STAGE,
    EventTypes.BOT_STOPPED,
    EventTypes.HEALTH_SUMMARY,
    EventTypes.RESOURCE_MEMORY_SNAPSHOT,
    EventTypes.CYCLE_STARTED,
    EventTypes.CYCLE_COMPLETED,
    EventTypes.CYCLE_DEGRADED,
    EventTypes.DATA_PACKET_UPDATED,
    EventTypes.SNAPSHOT_BUILT,
    EventTypes.PLANNING_UNAVAILABLE,
    EventTypes.PLANNING_DEFER_SUMMARY,
    EventTypes.PLANNING_SYMBOL_STATE,
    EventTypes.FORAGER_SELECTION,
    EventTypes.FORAGER_FEATURE_UNAVAILABLE,
    EventTypes.FORAGER_ELIGIBILITY_CHANGED,
    EventTypes.CONFIG_MARKET_COMPATIBILITY,
    EventTypes.EMA_BUNDLE_STARTED,
    EventTypes.EMA_BUNDLE_COMPLETED,
    EventTypes.EMA_FALLBACK_USED,
    EventTypes.EMA_UNAVAILABLE,
    EventTypes.CANDLE_COVERAGE_CHECKED,
    EventTypes.CANDLE_TAIL_PROJECTED,
    EventTypes.CACHE_LOAD_COMPLETED,
    EventTypes.CACHE_FLUSH_COMPLETED,
    EventTypes.CACHE_WARMUP_DECISION,
    EventTypes.EXCHANGE_CONFIG_REFRESH,
    EventTypes.EXCHANGE_TIME_SYNC,
    EventTypes.WEBSOCKET_RECONNECT,
    EventTypes.REMOTE_CALL_STARTED,
    EventTypes.REMOTE_CALL_SUCCEEDED,
    EventTypes.REMOTE_CALL_FAILED,
    EventTypes.REMOTE_CALL_THROTTLED,
    EventTypes.RUST_ORCHESTRATOR_CALLED,
    EventTypes.RUST_ORCHESTRATOR_RETURNED,
    EventTypes.ACTION_PLANNED,
    EventTypes.ORDER_WAVE_STARTED,
    EventTypes.ORDER_WAVE_COMPLETED,
    EventTypes.EXECUTION_CREATE_SENT,
    EventTypes.EXECUTION_CREATE_CONNECTOR_CALL_STARTED,
    EventTypes.EXECUTION_CREATE_SUCCEEDED,
    EventTypes.EXECUTION_CREATE_FAILED,
    EventTypes.EXECUTION_CREATE_REJECTED,
    EventTypes.EXECUTION_CREATE_DEFERRED,
    EventTypes.EXECUTION_CREATE_SKIPPED,
    EventTypes.ENTRY_INITIAL_ELIGIBILITY,
    EventTypes.ENTRY_INITIAL_DISTANCE_GATE_BLOCKED,
    EventTypes.ENTRY_INITIAL_DISTANCE_GATE_CLEARED,
    EventTypes.ENTRY_MIN_EFFECTIVE_COST_BLOCKED,
    EventTypes.EXECUTION_CANCEL_SENT,
    EventTypes.EXECUTION_CANCEL_CONNECTOR_CALL_STARTED,
    EventTypes.EXECUTION_CANCEL_SUCCEEDED,
    EventTypes.EXECUTION_CANCEL_FAILED,
    EventTypes.EXECUTION_CANCEL_AMBIGUOUS_TERMINAL,
    EventTypes.EXECUTION_AMBIGUOUS,
    EventTypes.EXECUTION_CONFIRMATION_REQUESTED,
    EventTypes.EXECUTION_CONFIRMATION_SATISFIED,
    EventTypes.EXECUTION_CONFIRMATION_TIMEOUT,
    EventTypes.FILLS_REFRESH_SUMMARY,
    EventTypes.FILL_INGESTED,
    EventTypes.FILLS_INGESTED_SUMMARY,
    EventTypes.POSITION_CHANGED,
    EventTypes.BALANCE_CHANGED,
    EventTypes.RISK_MODE_CHANGED,
    EventTypes.HSL_TRANSITION,
    EventTypes.HSL_STATUS,
    EventTypes.HSL_REPLAY_STARTED,
    EventTypes.HSL_REPLAY_PROGRESS,
    EventTypes.HSL_REPLAY_COMPLETED,
    EventTypes.HSL_REPLAY_FAILED,
    EventTypes.HSL_REPLAY_CACHE,
    EventTypes.HSL_RED_TRIGGERED,
    EventTypes.HSL_RED_FINALIZED_WITHOUT_ORDER,
    EventTypes.HSL_COOLDOWN_STARTED,
    EventTypes.HSL_COOLDOWN_ENDED,
    EventTypes.TRAILING_STATUS,
    EventTypes.UNSTUCK_STATUS,
    EventTypes.UNSTUCK_SELECTION,
    EventTypes.RISK_ENTRY_COOLDOWN_DELTA_ANCHORED,
    EventTypes.REALIZED_LOSS_GATE_BLOCKED,
    EventTypes.SINK_DEGRADED,
}


LEGACY_EVENT_TYPE_ALIASES = {
    "planning_unavailable": EventTypes.PLANNING_UNAVAILABLE,
}


VALID_LEVELS = {"trace", "debug", "info", "warning", "error", "critical"}
VALID_STATUSES = {
    "started",
    "succeeded",
    "failed",
    "deferred",
    "skipped",
    "recovered",
    "degraded",
}


_SENSITIVE_KEY_FRAGMENTS = {
    "apikey",
    "api_key",
    "api-key",
    "authorization",
    "auth",
    "cookie",
    "passphrase",
    "password",
    "privatekey",
    "private_key",
    "private-key",
    "secret",
    "signature",
    "token",
    "walletaddress",
    "wallet_address",
    "wallet-address",
    "x-mbx-apikey",
}


def utc_ms() -> int:
    return int(time.time() * 1000)


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


def normalize_event_type(event_type: object) -> str:
    raw = str(event_type)
    return LEGACY_EVENT_TYPE_ALIASES.get(raw, raw)


def payload_hash(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def payload_hash_raw(payload: bytes | str) -> str:
    raw = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    return hashlib.sha256(raw).hexdigest()


def _is_sensitive_key(key: object) -> bool:
    cleaned = "".join(ch for ch in str(key).lower() if ch.isalnum() or ch in "_-")
    compact = cleaned.replace("-", "").replace("_", "")
    return any(
        fragment in cleaned or fragment.replace("-", "").replace("_", "") in compact
        for fragment in _SENSITIVE_KEY_FRAGMENTS
    )


def redact_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): REDACTED if _is_sensitive_key(key) else redact_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    if isinstance(value, tuple):
        return [redact_payload(item) for item in value]
    return value


@dataclass(frozen=True)
class LiveEvent:
    event_type: str
    level: str = "info"
    source: str = "live"
    component: str | None = None
    tags: tuple[str, ...] = ()
    exchange: str | None = None
    user: str | None = None
    bot_id: str | None = None
    symbol: str | None = None
    pside: str | None = None
    side: str | None = None
    order_id: str | None = None
    client_order_id: str | None = None
    cycle_id: str | None = None
    snapshot_id: str | None = None
    plan_id: str | None = None
    action_id: str | None = None
    order_wave_id: str | None = None
    remote_call_id: str | None = None
    remote_call_group_id: str | None = None
    status: str | None = None
    reason_code: str | None = None
    message: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict)
    raw_ref: str | None = None
    raw_hash: str | None = None
    schema_version: int = SCHEMA_VERSION
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts_ms: int = field(default_factory=utc_ms)
    monotonic_ms: int = field(default_factory=monotonic_ms)

    def __post_init__(self) -> None:
        level = str(self.level).lower()
        if level not in VALID_LEVELS:
            raise ValueError(f"invalid live event level: {self.level}")
        object.__setattr__(self, "level", level)
        if self.status is not None:
            status = str(self.status).lower()
            if status not in VALID_STATUSES:
                raise ValueError(f"invalid live event status: {self.status}")
            object.__setattr__(self, "status", status)
        object.__setattr__(self, "event_type", normalize_event_type(self.event_type))
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))
        object.__setattr__(self, "data", dict(redact_payload(dict(self.data or {}))))

    def with_context(self, context: "LiveEventContext") -> "LiveEvent":
        return context.apply(self)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tags"] = list(self.tags)
        data["data"] = dict(self.data)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_monitor_event(self) -> tuple[str, tuple[str, ...], dict[str, Any]]:
        payload = dict(self.data)
        payload[LIVE_EVENT_MONITOR_PAYLOAD_KEY] = {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "level": self.level,
            "source": self.source,
            "component": self.component,
            "exchange": self.exchange,
            "user": self.user,
            "bot_id": self.bot_id,
            "symbol": self.symbol,
            "pside": self.pside,
            "side": self.side,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "message": self.message,
            "data": dict(self.data),
            "raw_ref": self.raw_ref,
            "raw_hash": self.raw_hash,
            "ids": {key: getattr(self, key) for key in LIVE_EVENT_ID_KEYS},
        }
        return self.event_type, self.tags, payload


@dataclass(frozen=True)
class LiveEventContext:
    exchange: str | None = None
    user: str | None = None
    bot_id: str | None = None
    cycle_id: str | None = None
    snapshot_id: str | None = None
    plan_id: str | None = None
    action_id: str | None = None
    order_wave_id: str | None = None
    remote_call_id: str | None = None
    remote_call_group_id: str | None = None

    def with_ids(self, **kwargs: str | None) -> "LiveEventContext":
        valid = {key: value for key, value in kwargs.items() if hasattr(self, key)}
        unknown = set(kwargs) - set(valid)
        if unknown:
            raise KeyError(f"unknown live event context fields: {sorted(unknown)}")
        return replace(self, **valid)

    def apply(self, event: LiveEvent) -> LiveEvent:
        updates: dict[str, str | None] = {}
        for key, value in asdict(self).items():
            if value is not None and getattr(event, key) is None:
                updates[key] = value
        return replace(event, **updates) if updates else event


@dataclass(frozen=True)
class EventRoute:
    structured: bool = True
    monitor: bool = True
    console: bool = False
    text: bool = False
    throttle_interval_ms: int = 0
    raw_payload_policy: str = "summary_hash_only"


DEFAULT_ROUTE = EventRoute(structured=True, monitor=True, console=False, text=False)
DEFAULT_ROUTES: dict[str, EventRoute] = {
    EventTypes.BOT_STARTED: EventRoute(console=True, text=True),
    EventTypes.BOT_READY: EventRoute(),
    EventTypes.BOT_STARTUP_TIMING: EventRoute(console=True, text=True),
    EventTypes.BOT_STOPPING: EventRoute(console=True, text=True),
    EventTypes.BOT_SHUTDOWN_STAGE: EventRoute(console=True, text=True),
    EventTypes.BOT_STOPPED: EventRoute(console=True, text=True),
    EventTypes.HEALTH_SUMMARY: EventRoute(console=True, text=True),
    EventTypes.RESOURCE_MEMORY_SNAPSHOT: EventRoute(console=True, text=True),
    EventTypes.MARKET_SNAPSHOT_DIAGNOSTIC_SKIPPED: EventRoute(
        console=False, text=False
    ),
    EventTypes.CYCLE_STARTED: EventRoute(
        console=True, text=True, throttle_interval_ms=60_000
    ),
    EventTypes.CYCLE_COMPLETED: EventRoute(
        console=True, text=True, throttle_interval_ms=60_000
    ),
    EventTypes.CYCLE_DEGRADED: EventRoute(console=True, text=True),
    EventTypes.DATA_PACKET_UPDATED: EventRoute(console=False),
    EventTypes.SNAPSHOT_BUILT: EventRoute(console=False),
    EventTypes.PLANNING_UNAVAILABLE: EventRoute(
        console=True, text=True, throttle_interval_ms=60_000
    ),
    EventTypes.PLANNING_DEFER_SUMMARY: EventRoute(console=False, text=False),
    EventTypes.PLANNING_SYMBOL_STATE: EventRoute(console=False, text=False),
    EventTypes.FORAGER_SELECTION: EventRoute(
        console=True, text=True, throttle_interval_ms=5 * 60 * 1000
    ),
    EventTypes.FORAGER_FEATURE_UNAVAILABLE: EventRoute(console=False, text=False),
    EventTypes.FORAGER_ELIGIBILITY_CHANGED: EventRoute(console=True, text=True),
    EventTypes.CONFIG_MARKET_COMPATIBILITY: EventRoute(console=False, text=False),
    EventTypes.EMA_BUNDLE_STARTED: EventRoute(console=False, text=False),
    EventTypes.EMA_BUNDLE_COMPLETED: EventRoute(console=False, text=False),
    EventTypes.EMA_FALLBACK_USED: EventRoute(
        console=True, text=True, throttle_interval_ms=15 * 60 * 1000
    ),
    EventTypes.EMA_UNAVAILABLE: EventRoute(
        console=True, text=True, throttle_interval_ms=15 * 60 * 1000
    ),
    EventTypes.CANDLE_COVERAGE_CHECKED: EventRoute(console=False, text=False),
    EventTypes.CANDLE_TAIL_PROJECTED: EventRoute(console=False, text=False),
    EventTypes.CACHE_LOAD_COMPLETED: EventRoute(console=False, text=False),
    EventTypes.CACHE_FLUSH_COMPLETED: EventRoute(console=False, text=False),
    EventTypes.CACHE_WARMUP_DECISION: EventRoute(console=False, text=False),
    EventTypes.EXCHANGE_CONFIG_REFRESH: EventRoute(console=False, text=False),
    EventTypes.EXCHANGE_TIME_SYNC: EventRoute(console=False, text=False),
    EventTypes.WEBSOCKET_RECONNECT: EventRoute(console=False, text=False),
    EventTypes.REMOTE_CALL_STARTED: EventRoute(console=False),
    EventTypes.REMOTE_CALL_SUCCEEDED: EventRoute(console=False),
    EventTypes.REMOTE_CALL_FAILED: EventRoute(console=False),
    EventTypes.REMOTE_CALL_THROTTLED: EventRoute(console=False),
    EventTypes.STATE_REFRESH_TIMING: EventRoute(console=True, text=True),
    EventTypes.RUST_ORCHESTRATOR_CALLED: EventRoute(console=False),
    EventTypes.RUST_ORCHESTRATOR_RETURNED: EventRoute(
        console=True, text=True, throttle_interval_ms=60_000
    ),
    EventTypes.ACTION_PLANNED: EventRoute(console=False, text=False),
    EventTypes.ORDER_WAVE_STARTED: EventRoute(console=False),
    EventTypes.ORDER_WAVE_COMPLETED: EventRoute(console=True, text=True),
    EventTypes.EXECUTION_CREATE_SENT: EventRoute(console=False),
    EventTypes.EXECUTION_CREATE_CONNECTOR_CALL_STARTED: EventRoute(
        console=False, text=False
    ),
    EventTypes.EXECUTION_CREATE_SUCCEEDED: EventRoute(console=True, text=True),
    EventTypes.EXECUTION_CREATE_FAILED: EventRoute(console=True, text=True),
    EventTypes.EXECUTION_CREATE_REJECTED: EventRoute(console=True, text=True),
    EventTypes.EXECUTION_CREATE_DEFERRED: EventRoute(console=False, text=False),
    EventTypes.EXECUTION_CREATE_SKIPPED: EventRoute(console=True, text=True),
    EventTypes.ENTRY_INITIAL_ELIGIBILITY: EventRoute(console=False, text=False),
    EventTypes.ENTRY_INITIAL_DISTANCE_GATE_BLOCKED: EventRoute(
        console=True, text=True
    ),
    EventTypes.ENTRY_INITIAL_DISTANCE_GATE_CLEARED: EventRoute(
        console=True, text=True
    ),
    EventTypes.ENTRY_MIN_EFFECTIVE_COST_BLOCKED: EventRoute(console=True, text=True),
    EventTypes.EXECUTION_CANCEL_SENT: EventRoute(console=False),
    EventTypes.EXECUTION_CANCEL_CONNECTOR_CALL_STARTED: EventRoute(
        console=False, text=False
    ),
    EventTypes.EXECUTION_CANCEL_SUCCEEDED: EventRoute(console=True, text=True),
    EventTypes.EXECUTION_CANCEL_FAILED: EventRoute(console=True, text=True),
    EventTypes.EXECUTION_CANCEL_AMBIGUOUS_TERMINAL: EventRoute(console=True, text=True),
    EventTypes.EXECUTION_AMBIGUOUS: EventRoute(console=True, text=True),
    EventTypes.EXECUTION_CONFIRMATION_REQUESTED: EventRoute(console=False),
    EventTypes.EXECUTION_CONFIRMATION_SATISFIED: EventRoute(console=True, text=True),
    EventTypes.EXECUTION_CONFIRMATION_TIMEOUT: EventRoute(console=True, text=True),
    EventTypes.FILLS_REFRESH_SUMMARY: EventRoute(console=False, text=False),
    EventTypes.FILL_INGESTED: EventRoute(console=True, text=True),
    EventTypes.FILLS_INGESTED_SUMMARY: EventRoute(console=True, text=True),
    EventTypes.POSITION_CHANGED: EventRoute(console=True, text=True),
    EventTypes.BALANCE_CHANGED: EventRoute(console=True, text=True),
    EventTypes.RISK_MODE_CHANGED: EventRoute(console=True, text=True),
    EventTypes.HSL_TRANSITION: EventRoute(console=True, text=True),
    EventTypes.HSL_STATUS: EventRoute(console=True, text=True),
    EventTypes.HSL_RAW_RED_PENDING: EventRoute(console=False, text=False),
    EventTypes.HSL_REPLAY_STARTED: EventRoute(console=False, text=False),
    EventTypes.HSL_REPLAY_PROGRESS: EventRoute(console=False, text=False),
    EventTypes.HSL_REPLAY_COMPLETED: EventRoute(console=False, text=False),
    EventTypes.HSL_REPLAY_FAILED: EventRoute(console=False, text=False),
    EventTypes.HSL_REPLAY_CACHE: EventRoute(console=False, text=False),
    EventTypes.HSL_RED_TRIGGERED: EventRoute(console=False, text=False),
    EventTypes.HSL_RED_FINALIZED_WITHOUT_ORDER: EventRoute(console=False, text=False),
    EventTypes.HSL_COOLDOWN_STARTED: EventRoute(console=False, text=False),
    EventTypes.HSL_COOLDOWN_ENDED: EventRoute(console=False, text=False),
    EventTypes.TRAILING_STATUS: EventRoute(console=True, text=True),
    EventTypes.UNSTUCK_STATUS: EventRoute(console=True, text=True),
    EventTypes.UNSTUCK_SELECTION: EventRoute(console=True, text=True),
    EventTypes.RISK_ENTRY_COOLDOWN_DELTA_ANCHORED: EventRoute(
        console=False, text=False
    ),
    EventTypes.REALIZED_LOSS_GATE_BLOCKED: EventRoute(console=True, text=True),
    EventTypes.SINK_DEGRADED: EventRoute(console=True, text=True),
}


class LiveEventSink(Protocol):
    def write(self, event: LiveEvent) -> Any:
        ...


class _MonitorEventPrepareError(Exception):
    def __init__(self, error: Exception, timing: dict[str, int]):
        super().__init__(str(error))
        self.error = error
        self.timing = timing


class MonitorEventSink:
    def __init__(self, publisher: Any, *, publisher_phase_timing: bool = False):
        self.publisher = publisher
        self.publisher_phase_timing = bool(publisher_phase_timing)

    def write(self, event: LiveEvent) -> Any:
        try:
            result, _timing = self._write_with_timing(event)
        except _MonitorEventPrepareError as exc:
            raise exc.error from exc
        if result is None:
            raise RuntimeError(f"monitor publisher returned None for {event.event_type}")
        return result

    def _write_with_timing(self, event: LiveEvent) -> tuple[Any, dict[str, int]]:
        timing = _empty_monitor_phase_timing()
        prepare_started_ns = time.monotonic_ns()
        try:
            kind, tags, payload = event.to_monitor_event()
        except Exception as exc:
            raise _MonitorEventPrepareError(exc, timing) from exc
        finally:
            timing["prepare_ns"] = max(0, time.monotonic_ns() - prepare_started_ns)

        if self.publisher_phase_timing:
            record_event_timed = getattr(self.publisher, "_record_event_timed", None)
            if not callable(record_event_timed):
                raise TypeError(
                    "monitor publisher phase timing requires _record_event_timed()"
                )
            result, publisher_timing = record_event_timed(
                kind,
                tags,
                payload,
                ts=event.ts_ms,
                symbol=event.symbol,
                pside=event.pside,
            )
            for key in _MONITOR_PUBLISHER_PHASE_TIMING_KEYS:
                timing[key] = max(0, int(publisher_timing.get(key, 0)))
            return result, timing

        result = self.publisher.record_event(
            kind,
            tags,
            payload,
            ts=event.ts_ms,
            symbol=event.symbol,
            pside=event.pside,
        )
        return result, timing


class ListEventSink:
    def __init__(self) -> None:
        self.events: list[LiveEvent] = []

    def write(self, event: LiveEvent) -> LiveEvent:
        self.events.append(event)
        return event


_CONSOLE_EVENT_TAGS = {
    EventTypes.BOT_STARTED: "bot",
    EventTypes.BOT_READY: "bot",
    EventTypes.BOT_STARTUP_TIMING: "boot",
    EventTypes.BOT_STOPPING: "bot",
    EventTypes.BOT_SHUTDOWN_STAGE: "shutdown",
    EventTypes.BOT_STOPPED: "bot",
    EventTypes.HEALTH_SUMMARY: "health",
    EventTypes.CYCLE_STARTED: "cycle",
    EventTypes.CYCLE_COMPLETED: "cycle",
    EventTypes.CYCLE_DEGRADED: "cycle",
    EventTypes.PLANNING_UNAVAILABLE: "gate",
    EventTypes.FORAGER_SELECTION: "forager",
    EventTypes.RUST_ORCHESTRATOR_RETURNED: "rust",
    EventTypes.ORDER_WAVE_COMPLETED: "execute",
    EventTypes.EXECUTION_CREATE_SUCCEEDED: "order",
    EventTypes.EXECUTION_CREATE_FAILED: "order",
    EventTypes.EXECUTION_CREATE_REJECTED: "order",
    EventTypes.EXECUTION_CREATE_DEFERRED: "gate",
    EventTypes.EXECUTION_CREATE_SKIPPED: "gate",
    EventTypes.ENTRY_INITIAL_DISTANCE_GATE_BLOCKED: "entry",
    EventTypes.ENTRY_INITIAL_DISTANCE_GATE_CLEARED: "entry",
    EventTypes.ENTRY_MIN_EFFECTIVE_COST_BLOCKED: "entry",
    EventTypes.EXECUTION_CANCEL_SUCCEEDED: "order",
    EventTypes.EXECUTION_CANCEL_FAILED: "order",
    EventTypes.EXECUTION_CANCEL_AMBIGUOUS_TERMINAL: "order",
    EventTypes.EXECUTION_AMBIGUOUS: "order",
    EventTypes.EXECUTION_CONFIRMATION_SATISFIED: "execute",
    EventTypes.EXECUTION_CONFIRMATION_TIMEOUT: "execute",
    EventTypes.FILL_INGESTED: "fill",
    EventTypes.FILLS_INGESTED_SUMMARY: "fill",
    EventTypes.POSITION_CHANGED: "pos",
    EventTypes.BALANCE_CHANGED: "balance",
    EventTypes.RESOURCE_MEMORY_SNAPSHOT: "memory",
    EventTypes.EMA_FALLBACK_USED: "ema",
    EventTypes.EMA_UNAVAILABLE: "ema",
    EventTypes.RISK_MODE_CHANGED: "risk",
    EventTypes.HSL_TRANSITION: "risk",
    EventTypes.HSL_STATUS: "risk",
    EventTypes.TRAILING_STATUS: "trailing",
    EventTypes.UNSTUCK_STATUS: "unstuck",
    EventTypes.UNSTUCK_SELECTION: "unstuck",
    EventTypes.REALIZED_LOSS_GATE_BLOCKED: "risk",
    EventTypes.SINK_DEGRADED: "logging",
}


def _console_tag(event: LiveEvent) -> str:
    return _CONSOLE_EVENT_TAGS.get(event.event_type, event.event_type)


def _data_int(data: Mapping[str, Any], key: str) -> int | None:
    try:
        value = data.get(key)
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _data_str(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None or value == "":
        return None
    return str(value)


def _data_float(data: Mapping[str, Any], key: str) -> str | None:
    try:
        value = data.get(key)
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return f"{number:.10g}"


def _data_number(data: Mapping[str, Any], key: str) -> float | None:
    try:
        value = data.get(key)
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _compact_csv(values: Any, *, limit: int = 4) -> str | None:
    if not isinstance(values, (list, tuple, set)):
        return None
    items = [str(item) for item in values if item is not None and str(item) != ""]
    if not items:
        return None
    shown = items[:limit]
    suffix = f",+{len(items) - len(shown)}" if len(items) > len(shown) else ""
    return ",".join(shown) + suffix


def _count_pair(data: Mapping[str, Any], done_key: str, planned_key: str) -> str | None:
    done = _data_int(data, done_key)
    planned = _data_int(data, planned_key)
    if done is None and planned is None:
        return None
    if planned is None:
        return str(done or 0)
    return f"{done or 0}/{planned}"


def _console_order_wave_summary(event: LiveEvent) -> list[str]:
    data = event.data if isinstance(event.data, Mapping) else {}
    parts: list[str] = []
    if event.order_wave_id:
        parts.append(f"wave={event.order_wave_id}")
    cancel = _count_pair(data, "cancel_posted", "planned_cancel")
    create = _count_pair(data, "create_posted", "planned_create")
    if cancel is not None:
        parts.append(f"cancel={cancel}")
    if create is not None:
        parts.append(f"create={create}")
    for key in ("deferred_create", "skipped_create", "skipped_cancel"):
        value = _data_int(data, key)
        if value:
            parts.append(f"{key}={value}")
    elapsed = _data_int(data, "elapsed_ms")
    if elapsed is not None:
        parts.append(f"elapsed={elapsed}ms")
    symbols = _compact_csv(data.get("symbols"), limit=4)
    if symbols:
        parts.append(f"symbols={symbols}")
    return parts


def _console_order_summary(event: LiveEvent) -> list[str]:
    data = event.data if isinstance(event.data, Mapping) else {}
    parts: list[str] = []
    if event.event_type == EventTypes.EXECUTION_CANCEL_AMBIGUOUS_TERMINAL:
        parts.append("confirmation=full_account_required")
    if event.order_wave_id:
        parts.append(f"wave={event.order_wave_id}")
    if event.side:
        parts.append(f"side={event.side}")
    order_type = _data_str(data, "order_type")
    if order_type:
        parts.append(f"type={order_type}")
    qty = _data_float(data, "qty")
    price = _data_float(data, "price")
    if qty:
        parts.append(f"qty={qty}")
    if price:
        parts.append(f"price={price}")
    if data.get("reduce_only") is True:
        parts.append("reduce_only=true")
    result_status = _data_str(data, "result_status")
    if result_status:
        parts.append(f"exchange_status={result_status}")
    order_id = _data_str(data, "result_order_id_short") or _data_str(data, "order_id_short")
    if order_id:
        parts.append(f"order_id={order_id}")
    client_id = _data_str(data, "result_client_order_id_short") or _data_str(
        data, "client_order_id_short"
    )
    if client_id:
        parts.append(f"client_id={client_id}")
    error_type = _data_str(data, "error_type")
    if error_type:
        parts.append(f"error_type={error_type}")
    return parts


def _console_create_filter_summary(event: LiveEvent) -> list[str]:
    data = event.data if isinstance(event.data, Mapping) else {}
    parts: list[str] = []
    if event.order_wave_id:
        parts.append(f"wave={event.order_wave_id}")
    count = _data_int(data, "order_count")
    if count is not None:
        parts.append(f"orders={count}")
    symbols = _compact_csv(data.get("symbols"), limit=4)
    if symbols:
        parts.append(f"symbols={symbols}")
    raw_balance = _data_number(data, "raw_balance")
    quote = _data_str(data, "quote")
    if raw_balance is not None:
        suffix = f" {quote}" if quote else ""
        parts.append(f"balance={raw_balance:g}{suffix}")
    balance_threshold = _data_number(data, "balance_threshold")
    if balance_threshold is not None:
        suffix = f" {quote}" if quote else ""
        parts.append(f"threshold={balance_threshold:g}{suffix}")
    allowed_cancel = _data_int(data, "allowed_cancel")
    if allowed_cancel is not None:
        parts.append(f"allow_cancel={allowed_cancel}")
    allowed_protective_create = _data_int(data, "allowed_protective_create")
    if allowed_protective_create is not None:
        parts.append(f"allow_protective_create={allowed_protective_create}")
    return parts


def _console_entry_gate_summary(event: LiveEvent) -> list[str]:
    data = event.data if isinstance(event.data, Mapping) else {}
    parts: list[str] = []
    action = _data_str(data, "action")
    if action:
        parts.append(f"action={action}")
    order_type = _data_str(data, "order_type")
    if order_type:
        parts.append(f"type={order_type}")
    qty = _data_float(data, "qty")
    price = _data_float(data, "price")
    market = _data_float(data, "market_price")
    if qty:
        parts.append(f"qty={qty}")
    if price:
        parts.append(f"price={price}")
    if market:
        parts.append(f"market={market}")
    for label, key in (
        ("dist", "distance_pct"),
        ("threshold", "threshold_pct"),
        ("tolerance", "tolerance_pct"),
    ):
        value = _data_number(data, key)
        if value is not None:
            parts.append(f"{label}={value:.4f}%")
    return parts


def _console_min_effective_cost_summary(event: LiveEvent) -> list[str]:
    data = event.data if isinstance(event.data, Mapping) else {}
    parts: list[str] = []
    action = _data_str(data, "action")
    if action:
        parts.append(f"action={action}")
    wanted = _data_float(data, "projected_initial_cost")
    required = _data_float(data, "effective_min_cost")
    if wanted and required:
        parts.append(f"notional={wanted}/{required}")
    else:
        if wanted:
            parts.append(f"wanted={wanted}")
        if required:
            parts.append(f"required={required}")
    effective_limit = _data_number(data, "effective_limit")
    if effective_limit is not None:
        parts.append(f"effective_limit={effective_limit * 100.0:.4f}%")
    entry_qty_pct = _data_number(data, "entry_initial_qty_pct")
    if entry_qty_pct is not None:
        parts.append(f"initial_qty={entry_qty_pct * 100.0:.4f}%")
    return parts


def _console_realized_loss_gate_summary(event: LiveEvent) -> list[str]:
    data = event.data if isinstance(event.data, Mapping) else {}
    parts: list[str] = []
    order_type = _data_str(data, "order_type")
    if order_type:
        parts.append(f"type={order_type}")
    qty = _data_float(data, "qty")
    price = _data_float(data, "price")
    if qty:
        parts.append(f"qty={qty}")
    if price:
        parts.append(f"price={price}")
    projected_pnl = _data_float(data, "projected_pnl")
    if projected_pnl:
        parts.append(f"projected_pnl={projected_pnl}")
    projected_balance = _data_float(data, "projected_balance_after")
    if projected_balance:
        parts.append(f"projected_balance={projected_balance}")
    balance_floor = _data_float(data, "balance_floor")
    if balance_floor:
        parts.append(f"floor={balance_floor}")
    max_loss = _data_number(data, "max_realized_loss_pct")
    if max_loss is not None:
        parts.append(f"max_loss={max_loss * 100.0:.4f}%")
    return parts


def _console_confirmation_summary(event: LiveEvent) -> list[str]:
    data = event.data if isinstance(event.data, Mapping) else {}
    parts: list[str] = []
    if event.order_wave_id:
        parts.append(f"wave={event.order_wave_id}")
    surfaces = _compact_csv(data.get("fresh_surfaces"), limit=4)
    if surfaces:
        parts.append(f"fresh={surfaces}")
    pending = _compact_csv(data.get("pending_surfaces"), limit=4)
    if pending:
        parts.append(f"pending={pending}")
    elapsed = _data_int(data, "elapsed_ms")
    timeout = _data_int(data, "timeout_ms")
    if elapsed is not None:
        parts.append(f"elapsed={elapsed}ms")
    if timeout is not None:
        parts.append(f"timeout={timeout}ms")
    return parts


def _console_rust_summary(event: LiveEvent) -> list[str]:
    data = event.data if isinstance(event.data, Mapping) else {}
    parts: list[str] = []
    elapsed = _data_int(data, "elapsed_ms")
    order_count = _data_int(data, "order_count")
    if order_count is not None:
        parts.append(f"orders={order_count}")
    if elapsed is not None:
        parts.append(f"elapsed={elapsed}ms")
    error_type = _data_str(data, "error_type")
    if error_type:
        parts.append(f"error_type={error_type}")
    return parts


def _format_fill_console_timestamp(value: object) -> str | None:
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return None
    if timestamp <= 0:
        return None
    try:
        return datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except (OverflowError, OSError, ValueError):
        return None


def _format_console_fill_ingested(event: LiveEvent) -> str:
    data = event.data if isinstance(event.data, Mapping) else {}
    parts = ["[fill]"]
    timestamp = _format_fill_console_timestamp(data.get("timestamp"))
    if timestamp:
        parts.append(timestamp)
    parts.append(_format_position_console_coin(event.symbol))
    parts.append(_format_console_label(event.pside))
    order_type = _format_console_label(_data_str(data, "pb_order_type"))
    parts.append(order_type)

    qty = _data_number(data, "qty")
    signed_qty = qty
    if qty is not None and str(event.side or "").lower() == "sell":
        signed_qty = -abs(qty)
    elif qty is not None and str(event.side or "").lower() == "buy":
        signed_qty = abs(qty)
    qty_rendered = _format_console_number(signed_qty)
    if signed_qty is not None and str(event.side or "").lower() == "buy":
        qty_rendered = f"+{qty_rendered}"
    parts.append(qty_rendered)
    parts.extend(("@", _format_console_number(_data_number(data, "price"))))

    is_close = "close" in str(_data_str(data, "pb_order_type") or "").lower()
    pnl = _data_number(data, "pnl")
    pnl_status = str(_data_str(data, "pnl_status") or "complete").lower()
    if is_close or (pnl is not None and pnl != 0.0):
        if pnl_status == "pending":
            parts.append(", pnl=pending")
        elif pnl is not None:
            pnl_rendered = _format_console_number(pnl)
            pnl_sign = "+" if pnl >= 0.0 else ""
            parts.append(f", pnl={pnl_sign}{pnl_rendered} USDT")
    fee = _data_number(data, "fee")
    if fee is not None and fee != 0.0:
        parts.append(f", fee={_format_console_number(fee)} USDT")
    if order_type == "unknown":
        client_id = _data_str(data, "client_order_id_short")
        if client_id:
            parts.append(f"(coid={_format_console_label(client_id)})")
    fill_id_hash = _data_str(data, "fill_id_hash")
    if fill_id_hash:
        parts.append(f"id={fill_id_hash[:12]}")
    return " ".join(parts).replace(" ,", ",")


def _format_console_fills_ingested_summary(event: LiveEvent) -> str:
    data = event.data if isinstance(event.data, Mapping) else {}
    count = _data_int(data, "count")
    total_pnl = _data_number(data, "known_net_realized_pnl")
    known_pnl_count = _data_int(data, "known_pnl_count")
    pending_pnl_count = _data_int(data, "pending_pnl_count")
    count_label = "-" if count is None else str(count)
    if known_pnl_count == 0:
        message = f"[fill] {count_label} fills, pnl=-"
    else:
        pnl_label = "-" if total_pnl is None else _format_console_number(total_pnl)
        pnl_sign = "+" if total_pnl is not None and total_pnl >= 0.0 else ""
        message = f"[fill] {count_label} fills, pnl={pnl_sign}{pnl_label} USDT"
    if known_pnl_count is not None:
        message += f", pnl_known={known_pnl_count}"
    if pending_pnl_count:
        message += f", pnl_pending={pending_pnl_count}"
    return message


def _format_console_number(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    if value == 0.0:
        return "0"
    return f"{value:.10g}"


def _format_balance_console_delta(value: float | None) -> str:
    rendered = _format_console_number(value)
    if value is not None and math.isfinite(value) and value > 0.0:
        return f"+{rendered}"
    return rendered


def _format_balance_console_transition(
    previous: float | None, current: float | None, delta: float | None
) -> str:
    if (
        previous is None
        or current is None
        or not math.isfinite(previous)
        or not math.isfinite(current)
    ):
        return "unavailable"
    return (
        f"{_format_console_number(previous)} -> {_format_console_number(current)} "
        f"({_format_balance_console_delta(delta)})"
    )


def _format_position_console_percentage(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    percentage = value * 100.0
    if not math.isfinite(percentage):
        return "-"
    if percentage == 0.0:
        percentage = 0.0
    return f"{percentage:.4f}%"


def _format_console_label(value: object) -> str:
    text = _ANSI_ESCAPE_RE.sub("", str(value or ""))
    text = _CONTROL_CHARACTER_RE.sub(" ", text)
    return " ".join(text.split()) or "-"


def _format_position_console_coin(symbol: object) -> str:
    coin = str(symbol or "").partition("/")[0]
    if "1000" in coin:
        start = coin.find("1000")
        end = start + 1
        while end < len(coin) and coin[end] == "0":
            end += 1
        coin = coin[:start] + coin[end:]
    if coin.startswith("k") and coin[1:].isupper():
        coin = coin[1:]
    return _format_console_label(coin)


def _format_console_position_changed(event: LiveEvent) -> str:
    data = event.data if isinstance(event.data, Mapping) else {}
    action = _format_console_label(_data_str(data, "action"))
    coin = _format_position_console_coin(event.symbol)
    pside = _format_console_label(event.pside)
    old_leg = (
        f"{_format_console_number(_data_number(data, 'old_size'))} @ "
        f"{_format_console_number(_data_number(data, 'old_price'))}"
    )
    new_leg = (
        f"{_format_console_number(_data_number(data, 'new_size'))} @ "
        f"{_format_console_number(_data_number(data, 'new_price'))}"
    )
    metrics = " ".join(
        (
            "WE="
            f"{_format_position_console_percentage(_data_number(data, 'wallet_exposure'))}",
            "WEL="
            f"{_format_position_console_percentage(_data_number(data, 'wel_ratio'))}",
            "eWEL="
            f"{_format_position_console_percentage(_data_number(data, 'wele_ratio'))}",
            "TWEL="
            f"{_format_position_console_percentage(_data_number(data, 'twel_ratio'))}",
            "uPnL="
            f"{_format_console_number(_data_number(data, 'upnl'))}",
        )
    )
    return (
        f"[pos] {action:>7} {coin:<10} {pside:<5} {old_leg:<22} "
        f"-> {new_leg:<22} | {metrics}"
    )


def _format_console_balance_changed(event: LiveEvent) -> str:
    data = event.data if isinstance(event.data, Mapping) else {}
    raw_transition = _format_balance_console_transition(
        _data_number(data, "previous_balance_raw"),
        _data_number(data, "balance_raw"),
        _data_number(data, "balance_raw_delta"),
    )
    snapped_transition = _format_balance_console_transition(
        _data_number(data, "previous_balance_snapped"),
        _data_number(data, "balance_snapped"),
        _data_number(data, "balance_snapped_delta"),
    )
    equity = _format_console_number(_data_number(data, "equity"))
    source = _format_console_label(_data_str(data, "source"))
    return (
        f"[balance] {'raw':<5}{raw_transition} | "
        f"{'snap':<5}{snapped_transition} | "
        f"equity={equity} source={source}"
    )


def format_memory_snapshot_console(data: Mapping[str, Any]) -> str:
    """Project a bounded memory snapshot without retaining diagnostic samples."""
    rss_bytes = _data_int(data, "rss_bytes")
    delta_pct = _data_number(data, "rss_delta_pct")
    cache = data.get("cache") if isinstance(data.get("cache"), Mapping) else {}
    timeframe_cache = (
        data.get("timeframe_cache")
        if isinstance(data.get("timeframe_cache"), Mapping)
        else {}
    )
    tasks = data.get("tasks") if isinstance(data.get("tasks"), Mapping) else {}

    parts = ["[memory]"]
    if rss_bytes is not None:
        parts.append(f"rss={rss_bytes / 1024.0 / 1024.0:.1f}MiB")
    if delta_pct is not None and math.isfinite(delta_pct):
        parts.append(f"delta={delta_pct:+.1f}%")
    cache_bytes = _data_int(cache, "bytes")
    cache_symbols = _data_int(cache, "symbols")
    if cache_bytes is not None:
        cache_part = f"cache={cache_bytes / 1024.0 / 1024.0:.1f}MiB"
        if cache_symbols is not None:
            cache_part += f"/{cache_symbols}sym"
        parts.append(cache_part)
    tf_bytes = _data_int(timeframe_cache, "bytes")
    tf_ranges = _data_int(timeframe_cache, "ranges")
    if tf_bytes is not None:
        tf_part = f"tf={tf_bytes / 1024.0 / 1024.0:.1f}MiB"
        if tf_ranges is not None:
            tf_part += f"/{tf_ranges}rng"
        parts.append(tf_part)
    total_tasks = _data_int(tasks, "total")
    pending_tasks = _data_int(tasks, "pending")
    if total_tasks is not None:
        task_part = f"tasks={total_tasks}"
        if pending_tasks is not None:
            task_part += f"/{pending_tasks}"
        parts.append(task_part)
    return " ".join(parts)[:240]


def _console_risk_mode_changed_summary(event: LiveEvent) -> list[str]:
    data = event.data if isinstance(event.data, Mapping) else {}
    parts: list[str] = []
    source = _data_str(data, "source")
    if source:
        parts.append(f"source={source}")
    action = _data_str(data, "action")
    if action:
        parts.append(f"action={action}")
    previous = _data_str(data, "previous_mode")
    mode = _data_str(data, "mode")
    if previous and mode:
        parts.append(f"mode={previous}->{mode}")
    elif mode:
        parts.append(f"mode={mode}")
    symbols = data.get("symbols")
    if isinstance(symbols, Mapping):
        count = _data_int(symbols, "count")
        sample = _compact_csv(symbols.get("sample"), limit=4)
        if count is not None:
            bit = f"symbols={count}"
            if sample:
                bit += f":{sample}"
            parts.append(bit)
    prev_counts = data.get("previous_mode_counts")
    if isinstance(prev_counts, Mapping) and prev_counts:
        parts.append(
            "previous_counts="
            + ",".join(f"{key}:{prev_counts[key]}" for key in sorted(prev_counts))
        )
    mode_counts = data.get("mode_counts")
    if isinstance(mode_counts, Mapping) and mode_counts:
        parts.append(
            "mode_counts="
            + ",".join(f"{key}:{mode_counts[key]}" for key in sorted(mode_counts))
        )
    return parts


def _console_hsl_transition_summary(event: LiveEvent) -> list[str]:
    data = event.data if isinstance(event.data, Mapping) else {}
    parts: list[str] = []
    signal_mode = _data_str(data, "signal_mode")
    if signal_mode:
        parts.append(f"mode={signal_mode}")
    previous = _data_str(data, "previous_tier")
    tier = _data_str(data, "tier")
    if previous and tier:
        parts.append(f"tier={previous}->{tier}")
    elif tier:
        parts.append(f"tier={tier}")
    for label, key in (
        ("dist_to_red", "dist_to_red"),
        ("drawdown_score", "drawdown_score"),
        ("red_threshold", "red_threshold"),
    ):
        value = _format_console_ratio(_data_float(data, key))
        if value is not None:
            parts.append(f"{label}={value}")
    timestamp_ms = _data_int(data, "timestamp_ms")
    if timestamp_ms is not None:
        parts.append(f"ts={timestamp_ms}")
    return parts


def _console_forager_selection_summary(event: LiveEvent) -> list[str]:
    data = event.data if isinstance(event.data, Mapping) else {}
    parts: list[str] = []
    selected = _data_int(data, "selected_count")
    candidate = _data_int(data, "candidate_count")
    eligible = _data_int(data, "eligible_count")
    if selected is not None and candidate is not None:
        if eligible is not None and eligible != candidate:
            parts.append(f"selected={selected}/{eligible}/{candidate}")
        else:
            parts.append(f"selected={selected}/{candidate}")
    elif selected is not None:
        parts.append(f"selected={selected}")
    slots_to_fill = _data_int(data, "slots_to_fill")
    max_positions = _data_int(data, "max_n_positions")
    if slots_to_fill is not None:
        if max_positions is not None:
            parts.append(f"slots={slots_to_fill}/{max_positions}")
        else:
            parts.append(f"slots={slots_to_fill}")
    unavailable = _data_int(data, "feature_unavailable_count")
    if unavailable:
        parts.append(f"unavailable={unavailable}")
    dropped = _data_int(data, "volatility_dropped_count")
    if dropped:
        parts.append(f"volatility_dropped={dropped}")
    selected_symbols = _compact_csv(data.get("selected_symbols"), limit=4)
    if selected_symbols:
        parts.append(f"symbols={selected_symbols}")
    incumbent_symbols = _compact_csv(data.get("incumbent_symbols"), limit=4)
    if incumbent_symbols:
        parts.append(f"incumbents={incumbent_symbols}")
    max_age_ms = _data_int(data, "max_age_ms")
    if max_age_ms is not None:
        parts.append(f"max_age={max_age_ms / 1000.0:.0f}s")
    fetch_budget = _data_int(data, "fetch_budget")
    if fetch_budget is not None:
        parts.append(f"fetch_budget={fetch_budget}")
    hysteresis_events = _data_int(data, "hysteresis_event_count")
    if hysteresis_events:
        parts.append(f"hysteresis_events={hysteresis_events}")
    source = _data_str(data, "source")
    if source:
        parts.append(f"source={source}")
    return parts


def _console_health_summary(event: LiveEvent) -> list[str]:
    data = event.data if isinstance(event.data, Mapping) else {}
    parts: list[str] = []
    if event.reason_code == ReasonCodes.EXECUTION_LOOP_ERROR_BURST:
        count = _data_int(data, "count")
        window = _data_int(data, "window_s")
        if count is not None:
            if window is not None:
                parts.append(f"errors={count}/{window}s")
            else:
                parts.append(f"errors={count}")
        endpoints = data.get("top_endpoints")
        if isinstance(endpoints, list):
            shown = []
            for item in endpoints[:3]:
                if not isinstance(item, Mapping):
                    continue
                endpoint = _data_str(item, "endpoint")
                endpoint_count = _data_int(item, "count")
                if endpoint and endpoint_count is not None:
                    shown.append(f"{endpoint}:{endpoint_count}")
            if shown:
                parts.append("top=" + ",".join(shown))
        latest_type = _data_str(data, "latest_error_type")
        if latest_type:
            parts.append(f"latest={latest_type}")
        latest_status = _data_str(data, "latest_status")
        if latest_status:
            parts.append(f"status={latest_status}")
        latest_code = _data_str(data, "latest_code")
        if latest_code:
            parts.append(f"code={latest_code}")
        return parts

    uptime_ms = _data_int(data, "uptime_ms")
    if uptime_ms is not None:
        parts.append(f"uptime={uptime_ms // 1000}s")
    loop_ms = _data_int(data, "last_loop_duration_ms")
    if loop_ms is not None:
        parts.append(f"loop={loop_ms / 1000.0:.1f}s")
    summary_lag_ms = _data_int(data, "health_summary_lag_ms")
    if summary_lag_ms:
        parts.append(f"health_lag={summary_lag_ms / 1000.0:.1f}s")
    long_count = _data_int(data, "positions_long")
    short_count = _data_int(data, "positions_short")
    if long_count is not None or short_count is not None:
        parts.append(f"positions={long_count or 0}L/{short_count or 0}S")
    balance = _data_float(data, "balance_raw")
    if balance:
        parts.append(f"balance={balance}")
    equity = _data_float(data, "equity")
    if equity:
        parts.append(f"equity={equity}")
    placed = _data_int(data, "orders_placed")
    cancelled = _data_int(data, "orders_cancelled")
    if placed is not None or cancelled is not None:
        parts.append(f"orders=+{placed or 0}/-{cancelled or 0}")
    fills = _data_int(data, "fills")
    if fills is not None:
        fill_part = f"fills={fills}"
        pnl = _data_float(data, "pnl")
        if pnl and fills:
            fill_part += f":pnl={pnl}"
        parts.append(fill_part)
    errors = _data_int(data, "errors_last_hour")
    if errors is not None:
        parts.append(f"errors={errors}/h")
    ws = _data_int(data, "ws_reconnects")
    if ws:
        parts.append(f"ws={ws}")
    rate_limits = _data_int(data, "rate_limits")
    if rate_limits:
        parts.append(f"rate_limits={rate_limits}")
    rss_bytes = _data_int(data, "rss_bytes")
    if rss_bytes is not None:
        parts.append(f"rss={rss_bytes / 1024.0 / 1024.0:.1f}MiB")
    queue_depth = _data_int(data, "event_queue_depth")
    if queue_depth:
        queue_max = _data_int(data, "event_queue_maxsize")
        if queue_max:
            parts.append(f"event_q={queue_depth}/{queue_max}")
        else:
            parts.append(f"event_q={queue_depth}")
    dropped = _data_int(data, "event_dropped_total")
    if dropped:
        parts.append(f"event_dropped={dropped}")
    sink_errors = _data_int(data, "event_sink_error_total")
    if sink_errors:
        parts.append(f"sink_errors={sink_errors}")
    if data.get("event_pipeline_worker_alive") is False:
        parts.append("event_worker=dead")
    return parts


def _format_console_duration_ms(duration_ms: int) -> str:
    total_seconds = max(0, duration_ms // 1000)
    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d{hours}h{minutes}m"
    if hours:
        return f"{hours}h{minutes}m{seconds}s"
    if minutes:
        return f"{minutes}m{seconds}s"
    return f"{seconds}s"


def format_periodic_health_summary(data: Mapping[str, Any]) -> str:
    """Render the bounded operator projection for a periodic health summary."""
    uptime_ms = _data_int(data, "uptime_ms")
    loop_ms = _data_int(data, "last_loop_duration_ms")
    long_count = _data_int(data, "positions_long")
    short_count = _data_int(data, "positions_short")
    parts = [
        f"up={_format_console_duration_ms(uptime_ms or 0)}",
        f"loop={loop_ms / 1000.0:.1f}s" if loop_ms and loop_ms > 0 else "loop=n/a",
        f"pos={long_count or 0}L/{short_count or 0}S",
    ]

    balance = _data_number(data, "balance_raw")
    quote = _data_str(data, "quote")
    if balance is not None:
        suffix = f" {quote}" if quote else ""
        balance_part = f"bal={balance:.2f}{suffix}"
        snapped = _data_number(data, "balance_snapped")
        if snapped is not None and abs(balance - snapped) > 1e-9:
            balance_part += f" (snap {snapped:.2f})"
        parts.append(balance_part)

    placed = _data_int(data, "orders_placed")
    cancelled = _data_int(data, "orders_cancelled")
    parts.append(f"ord=+{placed or 0}/-{cancelled or 0}")

    fills = _data_int(data, "fills")
    fills = fills if fills is not None else 0
    fills_part = f"fills={fills}"
    if fills > 0:
        pnl = _data_number(data, "pnl")
        if pnl is not None:
            pnl_suffix = f" {quote}" if quote else ""
            fills_part += f" (pnl={pnl:+.2f}{pnl_suffix})"
    parts.append(fills_part)

    errors = _data_int(data, "errors_last_hour")
    error_budget_max = _data_int(data, "error_budget_max")
    parts.append(f"err={errors or 0}/{error_budget_max or 10}")

    ws = _data_int(data, "ws_reconnects")
    if ws:
        parts.append(f"ws={ws}")
    rate_limits = _data_int(data, "rate_limits")
    if rate_limits:
        parts.append(f"rate_lim={rate_limits}")
    rss_bytes = _data_int(data, "rss_bytes")
    if rss_bytes is not None:
        parts.append(f"rss={rss_bytes / 1024.0 / 1024.0:.1f}MiB")
    summary_lag_ms = _data_int(data, "health_summary_lag_ms")
    if summary_lag_ms:
        parts.append(f"lag={summary_lag_ms / 1000.0:.1f}s")

    slow_phases = data.get("slow_phases")
    if isinstance(slow_phases, list):
        shown = []
        for item in slow_phases[:3]:
            if not isinstance(item, Mapping):
                continue
            phase = _data_str(item, "phase")
            duration_ms = _data_int(item, "duration_ms")
            if phase and duration_ms and duration_ms > 0:
                shown.append(f"{phase}:{duration_ms / 1000.0:.1f}s")
        if shown:
            parts.append("slow=" + ",".join(shown))

    queue_depth = _data_int(data, "event_queue_depth")
    if queue_depth:
        queue_max = _data_int(data, "event_queue_maxsize")
        parts.append(
            f"event_q={queue_depth}/{queue_max}"
            if queue_max
            else f"event_q={queue_depth}"
        )
    dropped = _data_int(data, "event_dropped_total")
    if dropped:
        parts.append(f"event_drop={dropped}")
    sink_errors = _data_int(data, "event_sink_error_total")
    if sink_errors:
        parts.append(f"sink_err={sink_errors}")
    if data.get("event_pipeline_worker_alive") is False:
        parts.append("event_worker=dead")
    return "[health] " + " ".join(parts)


def _format_console_ratio(value: Any) -> str | None:
    if value is None:
        return None
    return f"{float(value):.6f}"


def _console_hsl_status_summary(event: LiveEvent) -> list[str]:
    data = event.data if isinstance(event.data, Mapping) else {}
    parts: list[str] = []
    signal_mode = _data_str(data, "signal_mode")
    if signal_mode:
        parts.append(f"mode={signal_mode}")
    tier = _data_str(data, "tier")
    if tier:
        parts.append(f"tier={tier}")
    for label, key in (
        ("dist_to_red", "dist_to_red"),
        ("drawdown_score", "drawdown_score"),
        ("red_threshold", "red_threshold"),
    ):
        value = _format_console_ratio(_data_float(data, key))
        if value is not None:
            parts.append(f"{label}={value}")
    cooldown = _data_str(data, "cooldown_remaining")
    if not cooldown:
        seconds = _data_number(data, "cooldown_remaining_seconds")
        if seconds is not None:
            cooldown = f"{seconds:.0f}s"
    if cooldown:
        parts.append(f"cooldown={cooldown}")
    last_red_ts = _data_int(data, "last_red_ts")
    if last_red_ts is not None:
        parts.append(f"last_red_ts={last_red_ts}")
    pending_red_since_ms = _data_int(data, "pending_red_since_ms")
    if pending_red_since_ms is not None:
        parts.append(f"pending_red_since_ms={pending_red_since_ms}")
    return parts


def _console_startup_timing_summary(event: LiveEvent) -> list[str]:
    data = event.data if isinstance(event.data, Mapping) else {}
    parts: list[str] = []
    phase = _data_str(data, "phase")
    if phase:
        parts.append(f"phase={phase}-ready")
    elapsed_ms = _data_number(data, "elapsed_ms")
    if elapsed_ms is not None:
        parts.append(f"elapsed={elapsed_ms / 1000.0:.2f}s")
    since_previous_ms = _data_number(data, "since_previous_ms")
    if since_previous_ms is not None:
        parts.append(f"since_previous={since_previous_ms / 1000.0:.2f}s")
    details = _data_str(data, "details")
    if details:
        parts.append(f"details={details}")
    return parts


def _console_trailing_status_summary(event: LiveEvent) -> list[str]:
    data = event.data if isinstance(event.data, Mapping) else {}
    parts: list[str] = []
    kind = _data_str(data, "kind")
    if kind:
        parts.append(f"kind={kind}")
    if data.get("diagnostics_supported") is False:
        strategy_kind = _data_str(data, "strategy_kind")
        if strategy_kind:
            parts.append(f"strategy={strategy_kind}")
        reason = _data_str(data, "unsupported_reason")
        if reason:
            parts.append(f"unsupported={reason}")
    return parts


def _compact_trailing_console_label(value: object, *, limit: int) -> str:
    label = _format_console_label(value)
    if len(label) <= limit:
        return label
    return label[: limit - 3] + "..."


def _format_console_trailing_status(event: LiveEvent) -> str | None:
    data = event.data if isinstance(event.data, Mapping) else {}
    if data.get("diagnostics_supported") is False:
        return None

    parts = ["[trailing]"]
    if event.status:
        parts.append(event.status)
    if event.cycle_id:
        parts.append(f"cycle={_compact_trailing_console_label(event.cycle_id, limit=36)}")

    kind = _data_str(data, "kind")
    trailing_status = _data_str(data, "trailing_status")
    if kind and trailing_status:
        parts.append(
            f"{_compact_trailing_console_label(kind, limit=12)}/"
            f"{_compact_trailing_console_label(trailing_status, limit=24)}"
        )
    elif kind:
        parts.append(f"kind={_compact_trailing_console_label(kind, limit=12)}")
    elif trailing_status:
        parts.append(f"status={_compact_trailing_console_label(trailing_status, limit=24)}")

    selected_mode = _data_str(data, "selected_mode")
    if selected_mode:
        parts.append(f"mode={_compact_trailing_console_label(selected_mode, limit=20)}")

    gate_states: list[str] = []
    threshold_met = data.get("threshold_met")
    if threshold_met is not None:
        gate_states.append(f"t:{'y' if bool(threshold_met) else 'n'}")
    retracement_met = data.get("retracement_met")
    if retracement_met is not None:
        gate_states.append(f"r:{'y' if bool(retracement_met) else 'n'}")
    if gate_states:
        parts.append(f"gates={'/'.join(gate_states)}")

    threshold_pct = _data_number(data, "threshold_pct")
    threshold_price = _data_number(data, "threshold_price")
    if threshold_pct is not None and threshold_price:
        parts.append(f"threshold={threshold_pct * 100.0:.4f}%@{threshold_price:g}")
    elif threshold_pct is not None:
        parts.append(f"threshold={threshold_pct * 100.0:.4f}%")
    elif threshold_price:
        parts.append(f"threshold={threshold_price:g}")

    retracement_pct = _data_number(data, "retracement_pct")
    retracement_price = _data_number(data, "retracement_price")
    if retracement_pct is not None and retracement_price:
        parts.append(f"retracement={retracement_pct * 100.0:.4f}%@{retracement_price:g}")
    elif retracement_pct is not None:
        parts.append(f"retracement={retracement_pct * 100.0:.4f}%")
    elif retracement_price:
        parts.append(f"retracement={retracement_price:g}")

    current_price = _data_number(data, "current_price")
    if current_price:
        parts.append(f"cur={current_price:g}")
    if event.symbol:
        parts.append(f"symbol={_compact_trailing_console_label(event.symbol, limit=48)}")
    if event.pside:
        parts.append(f"pside={_compact_trailing_console_label(event.pside, limit=8)}")
    return " ".join(parts)


def _console_unstuck_status_summary(event: LiveEvent) -> list[str]:
    data = event.data if isinstance(event.data, Mapping) else {}
    parts: list[str] = []
    sides = data.get("sides")
    if isinstance(sides, Mapping):
        side_parts = []
        for pside in ("long", "short"):
            info = sides.get(pside)
            if not isinstance(info, Mapping):
                continue
            status = str(info.get("status") or "unknown")
            bit = f"{pside}:{status}"
            allowance = _data_number(info, "allowance")
            if allowance is not None:
                bit += f" allowance={allowance:g}"
            if info.get("over_budget") is True:
                bit += " over_budget"
            next_symbol = _data_str(info, "next_symbol")
            if next_symbol:
                bit += f" candidate={next_symbol}"
            target_price = _data_number(info, "next_target_price")
            if target_price:
                bit += f" target={target_price:g}"
            target_dist = _data_number(info, "next_target_distance_ratio")
            if target_dist is not None:
                bit += f" target_dist={target_dist * 100.0:.4f}%"
            trigger_dist = _data_number(info, "next_unstuck_trigger_distance_ratio")
            if trigger_dist is not None:
                bit += f" ema_gate_dist={trigger_dist * 100.0:.4f}%"
            side_parts.append(bit)
        if side_parts:
            parts.append(" ".join(side_parts))
    over_budget = _compact_csv(data.get("over_budget_sides"), limit=2)
    if over_budget:
        parts.append(f"over_budget={over_budget}")
    if data.get("changed") is True:
        parts.append("changed=true")
    return parts


def _console_unstuck_selection_summary(event: LiveEvent) -> list[str]:
    data = event.data if isinstance(event.data, Mapping) else {}
    parts: list[str] = []
    entry_price = _data_number(data, "entry_price")
    if entry_price:
        parts.append(f"entry={entry_price:g}")
    current_price = _data_number(data, "current_price")
    if current_price:
        parts.append(f"current={current_price:g}")
    diff_pct = _data_number(data, "price_diff_pct")
    if diff_pct is not None:
        parts.append(f"pos_pnl_dist={diff_pct:.4f}%")
    allowance = _data_number(data, "allowance")
    if allowance is not None:
        parts.append(f"allowance={allowance:g}")
    if data.get("changed") is True:
        parts.append("changed=true")
    return parts


def _console_data_summary(event: LiveEvent) -> list[str]:
    if event.event_type == EventTypes.BOT_STARTUP_TIMING:
        return _console_startup_timing_summary(event)
    if event.event_type == EventTypes.ORDER_WAVE_COMPLETED:
        return _console_order_wave_summary(event)
    if event.event_type in {
        EventTypes.EXECUTION_CREATE_DEFERRED,
        EventTypes.EXECUTION_CREATE_SKIPPED,
    }:
        return _console_create_filter_summary(event)
    if event.event_type in {
        EventTypes.ENTRY_INITIAL_DISTANCE_GATE_BLOCKED,
        EventTypes.ENTRY_INITIAL_DISTANCE_GATE_CLEARED,
    }:
        return _console_entry_gate_summary(event)
    if event.event_type == EventTypes.ENTRY_MIN_EFFECTIVE_COST_BLOCKED:
        return _console_min_effective_cost_summary(event)
    if event.event_type in {
        EventTypes.EXECUTION_CREATE_SUCCEEDED,
        EventTypes.EXECUTION_CREATE_FAILED,
        EventTypes.EXECUTION_CREATE_REJECTED,
        EventTypes.EXECUTION_CANCEL_SUCCEEDED,
        EventTypes.EXECUTION_CANCEL_FAILED,
        EventTypes.EXECUTION_CANCEL_AMBIGUOUS_TERMINAL,
        EventTypes.EXECUTION_AMBIGUOUS,
    }:
        return _console_order_summary(event)
    if event.event_type in {
        EventTypes.EXECUTION_CONFIRMATION_SATISFIED,
        EventTypes.EXECUTION_CONFIRMATION_TIMEOUT,
    }:
        return _console_confirmation_summary(event)
    if event.event_type == EventTypes.RUST_ORCHESTRATOR_RETURNED:
        return _console_rust_summary(event)
    if event.event_type == EventTypes.FORAGER_SELECTION:
        return _console_forager_selection_summary(event)
    if event.event_type == EventTypes.HEALTH_SUMMARY:
        return _console_health_summary(event)
    if event.event_type == EventTypes.RISK_MODE_CHANGED:
        return _console_risk_mode_changed_summary(event)
    if event.event_type == EventTypes.HSL_TRANSITION:
        return _console_hsl_transition_summary(event)
    if event.event_type == EventTypes.HSL_STATUS:
        return _console_hsl_status_summary(event)
    if event.event_type == EventTypes.TRAILING_STATUS:
        return _console_trailing_status_summary(event)
    if event.event_type == EventTypes.UNSTUCK_STATUS:
        return _console_unstuck_status_summary(event)
    if event.event_type == EventTypes.UNSTUCK_SELECTION:
        return _console_unstuck_selection_summary(event)
    if event.event_type == EventTypes.REALIZED_LOSS_GATE_BLOCKED:
        return _console_realized_loss_gate_summary(event)
    return []


def _hsl_status_operator_visible(event: LiveEvent) -> bool:
    data = event.data if isinstance(event.data, Mapping) else {}
    if event.symbol is None:
        return True
    if data.get("has_open_position") is True:
        return True
    if str(event.reason_code or "") == "cooldown_active":
        return True
    return str(data.get("tier") or "").lower() == "red"


def _operator_sink_event_visible(event: LiveEvent) -> bool:
    data = event.data if isinstance(event.data, Mapping) else {}
    if data.get("operator_visible") is False:
        return False
    if event.event_type == EventTypes.STATE_REFRESH_TIMING:
        return data.get("summary") is True or _state_refresh_timing_wall_ms(data) >= 10_000
    if event.event_type == EventTypes.FILL_INGESTED:
        return True
    if event.event_type == EventTypes.FORAGER_SELECTION:
        return str(data.get("source") or "") != "rust_orchestrator"
    if event.event_type == EventTypes.HSL_STATUS:
        return _hsl_status_operator_visible(event)
    if event.event_type == EventTypes.EMA_FALLBACK_USED:
        return (_data_int(data, "close_fallback_count") or 0) > 0
    if event.event_type == EventTypes.EMA_UNAVAILABLE:
        candidate_unavailable = data.get("candidate_unavailable")
        return isinstance(candidate_unavailable, Mapping) and (
            (_data_int(candidate_unavailable, "count") or 0) > 0
        )
    return True


def _console_sink_event_visible(event: LiveEvent) -> bool:
    if not _operator_sink_event_visible(event):
        return False
    if event.event_type == EventTypes.BALANCE_CHANGED:
        data = event.data if isinstance(event.data, Mapping) else {}
        snapped_delta = _data_number(data, "balance_snapped_delta")
        # Missing or malformed materiality metadata must remain operator-visible.
        return snapped_delta is None or snapped_delta != 0.0
    return True


_STATE_REFRESH_CONSOLE_LABELS = {
    "balance": "bal",
    "fills": "fills",
    "open_orders": "orders",
    "positions": "pos",
    "positions_balance": "pos+bal",
}
_STATE_REFRESH_CONSOLE_LABEL_MAX = 7
_STATE_REFRESH_CONSOLE_MAX_MS = (1 << 63) - 1


def _state_refresh_timing_wall_ms(data: Mapping[str, Any]) -> int:
    try:
        value = int(data.get("wall_ms", 0))
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, min(value, _STATE_REFRESH_CONSOLE_MAX_MS))


def _format_state_refresh_console_label(value: object) -> str:
    label = _STATE_REFRESH_CONSOLE_LABELS.get(str(value), None)
    if label is not None:
        return label
    label = _format_console_label(value)
    return label[:_STATE_REFRESH_CONSOLE_LABEL_MAX]


def _format_state_refresh_console_ms(value: object) -> str:
    try:
        milliseconds = int(value)
    except (TypeError, ValueError, OverflowError):
        return "-"
    milliseconds = max(0, min(milliseconds, _STATE_REFRESH_CONSOLE_MAX_MS))
    if milliseconds < 1_000_000:
        return str(milliseconds)
    return f"{milliseconds:.4g}"


def _format_state_refresh_console_stats(value: object) -> str:
    if not isinstance(value, Mapping):
        return "-"
    return "/".join(
        _format_state_refresh_console_ms(value.get(key)) for key in ("min", "mean", "max")
    )


def _format_state_refresh_console_labels(value: object) -> str:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return "-"
    labels = [_format_state_refresh_console_label(item) for item in value]
    labels = [label for label in labels if label and label != "-"]
    if len(labels) <= 4:
        return ",".join(labels) or "-"
    return ",".join(labels[:3]) + ",+more"


def _format_state_refresh_console_timings(value: object) -> str:
    if not isinstance(value, Mapping):
        return "-"
    timings = [
        (_format_state_refresh_console_label(surface), _format_state_refresh_console_ms(elapsed))
        for surface, elapsed in value.items()
    ]
    timings.sort(key=lambda item: item[0])
    if len(timings) <= 4:
        shown = timings
        suffix = ""
    else:
        shown = timings[:3]
        suffix = ",+more"
    return ",".join(f"{surface}:{elapsed}" for surface, elapsed in shown) + suffix or "-"


def _format_state_refresh_console_slowest_surface(value: object) -> str:
    if not isinstance(value, Mapping) or not value:
        return "-"
    candidates: list[tuple[int, str, Mapping[str, Any]]] = []
    for surface, stats in value.items():
        if not isinstance(stats, Mapping):
            continue
        try:
            maximum = int(stats.get("max", 0))
        except (TypeError, ValueError, OverflowError):
            maximum = 0
        candidates.append((max(0, maximum), _format_state_refresh_console_label(surface), stats))
    if not candidates:
        return "-"
    _maximum, surface, stats = max(candidates, key=lambda item: (item[0], item[1]))
    return f"{surface}:{_format_state_refresh_console_stats(stats)}"


def format_state_refresh_timing_console(data: Mapping[str, Any]) -> str:
    """Render bounded operator projections for staged-refresh timing events."""
    plan = _format_state_refresh_console_labels(data.get("plan"))
    if data.get("summary") is True:
        try:
            count = max(0, int(data.get("count", 0)))
        except (TypeError, ValueError, OverflowError):
            count = 0
        count_label = str(min(count, 1_000_000))
        if count > 1_000_000:
            count_label += "+"
        return " ".join(
            (
                "[state] refresh summary",
                f"plan={plan}",
                f"n={count_label}",
                f"wall_ms={_format_state_refresh_console_stats(data.get('wall_ms'))}",
                f"slow={_format_state_refresh_console_slowest_surface(data.get('surfaces_ms'))}",
                f"resid_ms={_format_state_refresh_console_stats(data.get('residual_ms'))}",
            )
        )

    parts = [
        "[state] refresh",
        f"plan={plan}",
        f"wall={_format_state_refresh_console_ms(data.get('wall_ms'))}ms",
        f"surfaces_ms={_format_state_refresh_console_timings(data.get('timings_ms'))}",
    ]
    if data.get("parallel") is True:
        parts.append("parallel")
    residual_ms = _state_refresh_timing_wall_ms({"wall_ms": data.get("residual_ms")})
    if residual_ms >= 500:
        parts.append(f"residual={_format_state_refresh_console_ms(residual_ms)}ms")
    return " ".join(parts)


_EMA_FALLBACK_CONSOLE_TOKEN_LIMIT = 16
_EMA_FALLBACK_CONSOLE_REASON_LIMIT = 18
_EMA_FALLBACK_CONSOLE_SAMPLE_LIMIT = 2
_EMA_FALLBACK_CONSOLE_EXAMPLE_LIMIT = 2
_EMA_FALLBACK_CONSOLE_RECORD_LIMIT = 188
_EMA_FALLBACK_CONSOLE_COUNT_LIMIT = 999_999_999


def _bounded_ema_console_text(value: object, *, limit: int) -> str:
    """Return a compact token without exposing arbitrary payload text."""
    cleaned = re.sub(r"\s+", "-", str(value or "").strip())
    bounded = "".join(
        char for index, char in enumerate(cleaned) if index < max(0, int(limit))
    )
    return bounded or "-"


def _bounded_ema_console_count(value: object) -> str:
    try:
        count = max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return "-"
    if count > _EMA_FALLBACK_CONSOLE_COUNT_LIMIT:
        return f"{_EMA_FALLBACK_CONSOLE_COUNT_LIMIT}+"
    return str(count)


def _bounded_ema_console_span(value: object) -> str:
    try:
        span = float(value)
    except (TypeError, ValueError, OverflowError):
        return "-"
    if not math.isfinite(span):
        return "-"
    return f"{span:.6g}"


def _bounded_ema_console_reason(value: object, *, limit: int) -> str:
    reason = str(value or "").strip()
    if reason.startswith("non-finite close EMA value"):
        return "non_finite"
    if re.fullmatch(r"[A-Za-z0-9_.-]+", reason):
        return _bounded_ema_console_text(reason, limit=limit)
    if ":" in reason:
        error_type = reason.split(":", 1)[0].strip()
        if re.fullmatch(r"[A-Za-z0-9_.-]+", error_type):
            return _bounded_ema_console_text(error_type, limit=limit)
    return "error"


def _ema_fallback_examples(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    examples = data.get("examples")
    if not isinstance(examples, Mapping):
        return []
    close_fallback = examples.get("close_fallback")
    if not isinstance(close_fallback, (list, tuple)):
        return []
    return [item for item in close_fallback if isinstance(item, Mapping)]


def _format_ema_fallback_console_example(
    example: Mapping[str, Any],
    *,
    symbol_limit: int = _EMA_FALLBACK_CONSOLE_TOKEN_LIMIT,
    span_limit: int = _EMA_FALLBACK_CONSOLE_SAMPLE_LIMIT,
    reason_limit: int = _EMA_FALLBACK_CONSOLE_REASON_LIMIT,
) -> str:
    spans = example.get("spans")
    if isinstance(spans, (list, tuple)):
        span_text = ",".join(
            _bounded_ema_console_span(span)
            for index, span in enumerate(spans)
            if index < span_limit
        )
    else:
        span_text = "-"
    if not span_text:
        span_text = "-"
    return "".join(
        (
            _bounded_ema_console_text(example.get("symbol"), limit=symbol_limit),
            f"[{span_text}]",
            f" age={_bounded_ema_console_count(example.get('max_age_ms'))}ms",
            f" n={_bounded_ema_console_count(example.get('max_fallbacks'))}",
            f" why={_bounded_ema_console_reason(example.get('reason'), limit=reason_limit)}",
        )
    )


def format_ema_fallback_console(data: Mapping[str, Any]) -> str:
    """Render the close-EMA fallback warning without generic event decoration."""
    examples = _ema_fallback_examples(data)
    max_age_ms = max(
        (_data_int(example, "max_age_ms") or 0 for example in examples), default=0
    )
    max_fallbacks = max(
        (_data_int(example, "max_fallbacks") or 0 for example in examples), default=0
    )
    symbols = data.get("close_fallback_symbols")
    if isinstance(symbols, Mapping):
        sample_values = symbols.get("sample")
        symbol_count = _bounded_ema_console_count(symbols.get("count"))
    else:
        sample_values = ()
        symbol_count = "-"
    if isinstance(sample_values, (list, tuple, set, frozenset)):
        sample = ",".join(
            _bounded_ema_console_text(
                value, limit=_EMA_FALLBACK_CONSOLE_TOKEN_LIMIT
            )
            for index, value in enumerate(sample_values)
            if index < _EMA_FALLBACK_CONSOLE_SAMPLE_LIMIT
        )
    else:
        sample = "-"
    parts = [
        "[ema] close fallback",
        f"n={_bounded_ema_console_count(data.get('close_fallback_count'))}",
        f"sym={symbol_count}({sample or '-'})",
        f"age={_bounded_ema_console_count(max_age_ms)}ms",
        f"streak={_bounded_ema_console_count(max_fallbacks)}",
    ]
    emitted_examples = 0
    for example in examples:
        if emitted_examples >= _EMA_FALLBACK_CONSOLE_EXAMPLE_LIMIT:
            break
        candidate = f"ex={_format_ema_fallback_console_example(example)}"
        if len(" ".join((*parts, candidate))) > _EMA_FALLBACK_CONSOLE_RECORD_LIMIT:
            candidate = "ex=" + _format_ema_fallback_console_example(
                example, symbol_limit=8, span_limit=1, reason_limit=8
            )
        if len(" ".join((*parts, candidate))) > _EMA_FALLBACK_CONSOLE_RECORD_LIMIT:
            break
        parts.append(candidate)
        emitted_examples += 1
    if not emitted_examples:
        parts.append("ex=-")
    return " ".join(parts)


_EMA_UNAVAILABLE_CONSOLE_GROUP_LIMIT = 36
_EMA_UNAVAILABLE_CONSOLE_ERROR_LIMIT = 24
_EMA_UNAVAILABLE_CONSOLE_SAMPLE_LIMIT = 2
_EMA_UNAVAILABLE_CONSOLE_RECORD_LIMIT = 188
_EMA_UNAVAILABLE_COMPACT_GROUP_LIMIT = 19
_EMA_UNAVAILABLE_COMPACT_ERROR_LIMIT = 13
_EMA_UNAVAILABLE_COMPACT_SYMBOL_LIMIT = 14
_EMA_UNAVAILABLE_EMA_TYPE_RE = re.compile(
    r"\b(?P<ema_type>m1_close|m1_volume|m1_log_range|h1_log_range)\s+EMA\b",
    re.IGNORECASE,
)


def _ema_unavailable_console_group(data: Mapping[str, Any]) -> Mapping[str, Any]:
    groups = data.get("candidate_unavailable_groups")
    if not isinstance(groups, (list, tuple)):
        return {}
    return next((group for group in groups if isinstance(group, Mapping)), {})


def _ema_unavailable_console_symbol_preview(
    group: Mapping[str, Any], *, symbol_limit: int = _EMA_FALLBACK_CONSOLE_TOKEN_LIMIT
) -> str:
    symbols = group.get("symbols")
    if not isinstance(symbols, Mapping):
        return "-"
    count = _bounded_ema_console_count(symbols.get("count"))
    sample_values = symbols.get("sample")
    if not isinstance(sample_values, (list, tuple, set, frozenset)):
        return f"{count}(-)"
    sample = [
        _bounded_ema_console_text(
            _format_position_console_coin(value), limit=symbol_limit
        )
        for index, value in enumerate(sample_values)
        if index < _EMA_UNAVAILABLE_CONSOLE_SAMPLE_LIMIT
    ]
    try:
        omitted = max(0, int(symbols.get("count", 0)) - len(sample))
    except (TypeError, ValueError, OverflowError):
        omitted = 0
    if omitted:
        sample.append(f"+{_bounded_ema_console_count(omitted)}")
    return f"{count}({','.join(sample) or '-'})"


def _ema_unavailable_console_ema_type(group: Mapping[str, Any]) -> str | None:
    example_error = group.get("example_error")
    match = _EMA_UNAVAILABLE_EMA_TYPE_RE.search(str(example_error or ""))
    if match is None:
        return None
    return _bounded_ema_console_text(
        match.group("ema_type"), limit=_EMA_UNAVAILABLE_CONSOLE_ERROR_LIMIT
    )


def format_ema_unavailable_console(data: Mapping[str, Any]) -> str:
    """Project required EMA unavailability without reducing structured diagnostics."""
    candidate_unavailable = data.get("candidate_unavailable")
    count = (
        _bounded_ema_console_count(candidate_unavailable.get("count"))
        if isinstance(candidate_unavailable, Mapping)
        else "-"
    )
    group = _ema_unavailable_console_group(data)
    reason = group.get("reason")
    parts = [
        "[ema] unavailable",
        f"n={count}",
        "group="
        + _bounded_ema_console_text(
            reason, limit=_EMA_UNAVAILABLE_CONSOLE_GROUP_LIMIT
        ),
        "action=mark_nontradable_until_fresh",
        f"sym={_ema_unavailable_console_symbol_preview(group)}",
    ]
    error_types = group.get("error_types")
    if isinstance(error_types, (list, tuple)) and error_types:
        parts.append(
            "err="
            + _bounded_ema_console_text(
                error_types[0], limit=_EMA_UNAVAILABLE_CONSOLE_ERROR_LIMIT
            )
        )
    ema_type = _ema_unavailable_console_ema_type(group)
    if ema_type:
        parts.append(f"ema={ema_type}")
    message = " ".join(parts)
    if len(message) <= _EMA_UNAVAILABLE_CONSOLE_RECORD_LIMIT:
        return message

    parts[2] = "group=" + _bounded_ema_console_text(
        reason, limit=_EMA_UNAVAILABLE_COMPACT_GROUP_LIMIT
    )
    parts[4] = "sym=" + _ema_unavailable_console_symbol_preview(
        group, symbol_limit=_EMA_UNAVAILABLE_COMPACT_SYMBOL_LIMIT
    )
    for index, part in enumerate(parts):
        if part.startswith("err="):
            parts[index] = "err=" + _bounded_ema_console_text(
                error_types[0], limit=_EMA_UNAVAILABLE_COMPACT_ERROR_LIMIT
            )
            break
    return " ".join(parts)


_INITIAL_ENTRY_DISTANCE_GATE_CONSOLE_RECORD_LIMIT = 188
_INITIAL_ENTRY_DISTANCE_GATE_CONSOLE_CYCLE_LIMIT = 24
_INITIAL_ENTRY_DISTANCE_GATE_CONSOLE_SYMBOL_LIMIT = 24
_INITIAL_ENTRY_DISTANCE_GATE_CONSOLE_PSIDE_LIMIT = 8
_INITIAL_ENTRY_DISTANCE_GATE_CONSOLE_TYPE_LIMIT = 32


def _bounded_initial_entry_distance_gate_console_text(
    value: object, *, limit: int
) -> str:
    """Return a sanitized, bounded display token for the gate projection."""
    return _format_console_label(value).replace(" ", "-")[:limit] or "-"


def _format_initial_entry_distance_gate_console_pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.4f}"


def _format_initial_entry_distance_gate_console(event: LiveEvent) -> str:
    """Render a compact blocked or cleared initial-entry distance-gate transition."""
    data = event.data if isinstance(event.data, Mapping) else {}
    transition = (
        "blocked"
        if event.event_type == EventTypes.ENTRY_INITIAL_DISTANCE_GATE_BLOCKED
        else "cleared"
    )
    parts = ["[entry]", transition]
    cycle_candidate = (
        (
            "cy="
            + _bounded_initial_entry_distance_gate_console_text(
                event.cycle_id,
                limit=_INITIAL_ENTRY_DISTANCE_GATE_CONSOLE_CYCLE_LIMIT,
            ),
        )
        if event.cycle_id
        else ()
    )
    tolerance_candidate = (
        "tol="
        + _format_initial_entry_distance_gate_console_pct(
            _data_number(data, "tolerance_pct")
        )
        + "%",
    ) if "tolerance_pct" in data else ()
    aggregate_candidates = (
        "active=" + str(min(max(_data_int(data, "active_count") or 0, 0), 999)),
        "sup=" + str(min(max(_data_int(data, "suppressed_count") or 0, 0), 999)),
    ) if (
        event.event_type == EventTypes.ENTRY_INITIAL_DISTANCE_GATE_BLOCKED
        and ("active_count" in data or "suppressed_count" in data)
    ) else ()
    candidates = (
        *cycle_candidate,
        "symbol="
        + _bounded_initial_entry_distance_gate_console_text(
            event.symbol, limit=_INITIAL_ENTRY_DISTANCE_GATE_CONSOLE_SYMBOL_LIMIT
        ),
        "pside="
        + _bounded_initial_entry_distance_gate_console_text(
            event.pside, limit=_INITIAL_ENTRY_DISTANCE_GATE_CONSOLE_PSIDE_LIMIT
        ),
        "type="
        + _bounded_initial_entry_distance_gate_console_text(
            _data_str(data, "order_type"),
            limit=_INITIAL_ENTRY_DISTANCE_GATE_CONSOLE_TYPE_LIMIT,
        ),
        f"q={_format_console_number(_data_number(data, 'qty'))}",
        f"px={_format_console_number(_data_number(data, 'price'))}",
        f"mkt={_format_console_number(_data_number(data, 'market_price'))}",
        "d="
        + _format_initial_entry_distance_gate_console_pct(
            _data_number(data, "distance_pct")
        )
        + "%",
        "max="
        + _format_initial_entry_distance_gate_console_pct(
            _data_number(data, "threshold_pct")
        )
        + "%",
        *tolerance_candidate,
        *aggregate_candidates,
    )
    for candidate in candidates:
        if (
            len(" ".join((*parts, candidate)))
            > _INITIAL_ENTRY_DISTANCE_GATE_CONSOLE_RECORD_LIMIT
        ):
            break
        parts.append(candidate)
    return " ".join(parts)


_FORAGER_ELIGIBILITY_CONSOLE_RECORD_LIMIT = 188
_FORAGER_ELIGIBILITY_CONSOLE_TOKEN_LIMIT = 16
_FORAGER_ELIGIBILITY_CONSOLE_SAMPLE_LIMIT = 3
_FORAGER_ELIGIBILITY_CONSOLE_COUNT_LIMIT = 999_999_999


def _bounded_forager_eligibility_console_token(value: object, *, limit: int) -> str:
    """Return a one-field, sanitized token for a coin-list console projection."""
    cleaned = _ANSI_ESCAPE_RE.sub("", str(value or ""))
    token = "".join(
        char
        if char.isascii()
        and char.isprintable()
        and not char.isspace()
        and char not in {",", "=", "|", "[", "]"}
        else "_"
        for char in cleaned
    )
    token = token[:limit]
    return token or "-"


def _forager_eligibility_console_count(value: object) -> int:
    try:
        return min(_FORAGER_ELIGIBILITY_CONSOLE_COUNT_LIMIT, max(0, int(value)))
    except (TypeError, ValueError, OverflowError):
        return 0


def _format_forager_eligibility_console_count(value: int) -> str:
    if value >= _FORAGER_ELIGIBILITY_CONSOLE_COUNT_LIMIT:
        return f"{_FORAGER_ELIGIBILITY_CONSOLE_COUNT_LIMIT}+"
    return str(value)


def format_forager_eligibility_console(data: Mapping[str, Any]) -> str:
    """Project eligibility membership changes without exposing unbounded symbol lists."""
    counts = {"long": 0, "short": 0}
    samples: list[str] = []
    changes = data.get("changes") if isinstance(data, Mapping) else None
    if isinstance(changes, (list, tuple)):
        for change in changes:
            if not isinstance(change, Mapping):
                continue
            pside = str(change.get("pside") or "")
            if pside not in counts:
                continue
            counts[pside] = min(
                _FORAGER_ELIGIBILITY_CONSOLE_COUNT_LIMIT,
                counts[pside] + _forager_eligibility_console_count(change.get("count")),
            )
            symbols = change.get("symbols")
            if isinstance(symbols, (list, tuple)):
                samples.extend(
                    _bounded_forager_eligibility_console_token(
                        _format_position_console_coin(symbol),
                        limit=_FORAGER_ELIGIBILITY_CONSOLE_TOKEN_LIMIT,
                    )
                    for symbol in symbols[:12]
                )
    samples = sorted(samples)[:_FORAGER_ELIGIBILITY_CONSOLE_SAMPLE_LIMIT]
    total = min(
        _FORAGER_ELIGIBILITY_CONSOLE_COUNT_LIMIT, counts["long"] + counts["short"]
    )
    omitted = max(0, total - len(samples))
    message = " ".join(
        (
            "[forager]",
            "list="
            + _bounded_forager_eligibility_console_token(
                data.get("list_kind") if isinstance(data, Mapping) else None,
                limit=_FORAGER_ELIGIBILITY_CONSOLE_TOKEN_LIMIT,
            ),
            "op="
            + _bounded_forager_eligibility_console_token(
                data.get("operation") if isinstance(data, Mapping) else None,
                limit=_FORAGER_ELIGIBILITY_CONSOLE_TOKEN_LIMIT,
            ),
            "counts=long:"
            + _format_forager_eligibility_console_count(counts["long"])
            + ",short:"
            + _format_forager_eligibility_console_count(counts["short"]),
            "samples=" + (",".join(samples) or "-"),
            "omitted=" + _format_forager_eligibility_console_count(omitted),
        )
    )
    return message[:_FORAGER_ELIGIBILITY_CONSOLE_RECORD_LIMIT]


def format_console_event(event: LiveEvent) -> str:
    if (
        event.event_type == EventTypes.HEALTH_SUMMARY
        and event.reason_code == ReasonCodes.PERIODIC_HEALTH_SUMMARY
    ):
        return format_periodic_health_summary(event.data)
    if event.event_type == EventTypes.FILL_INGESTED:
        return _format_console_fill_ingested(event)
    if event.event_type == EventTypes.FILLS_INGESTED_SUMMARY:
        return _format_console_fills_ingested_summary(event)
    if event.event_type == EventTypes.POSITION_CHANGED:
        return _format_console_position_changed(event)
    if event.event_type == EventTypes.BALANCE_CHANGED:
        return _format_console_balance_changed(event)
    if event.event_type == EventTypes.RESOURCE_MEMORY_SNAPSHOT:
        return format_memory_snapshot_console(event.data)
    if event.event_type == EventTypes.EMA_FALLBACK_USED:
        return format_ema_fallback_console(event.data)
    if event.event_type == EventTypes.EMA_UNAVAILABLE:
        return format_ema_unavailable_console(event.data)
    if event.event_type == EventTypes.FORAGER_ELIGIBILITY_CHANGED:
        return format_forager_eligibility_console(event.data)
    if event.event_type in {
        EventTypes.ENTRY_INITIAL_DISTANCE_GATE_BLOCKED,
        EventTypes.ENTRY_INITIAL_DISTANCE_GATE_CLEARED,
    }:
        return _format_initial_entry_distance_gate_console(event)
    if event.event_type == EventTypes.STATE_REFRESH_TIMING:
        data = event.data if isinstance(event.data, Mapping) else {}
        return format_state_refresh_timing_console(data)
    if event.event_type == EventTypes.TRAILING_STATUS:
        trailing_status = _format_console_trailing_status(event)
        if trailing_status is not None:
            return trailing_status
    base = f"[{_console_tag(event)}]"
    if event.status:
        base += f" {event.status}"
    if event.cycle_id:
        base += f" cycle={event.cycle_id}"
    for part in _console_data_summary(event):
        base += f" {part}"
    if event.symbol:
        base += f" symbol={event.symbol}"
    if event.pside:
        base += f" pside={event.pside}"
    if event.reason_code:
        base += f" reason={event.reason_code}"
    if event.message:
        base += f" {event.message}"
    return base


class ConsoleSummarySink:
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    def write(self, event: LiveEvent) -> str:
        message = format_console_event(event)
        self.logger.log(_logging_level(event.level), message)
        return message


def _logging_level(level: str) -> int:
    if level == "critical":
        return logging.CRITICAL
    if level == "error":
        return logging.ERROR
    if level == "warning":
        return logging.WARNING
    if level == "debug":
        return logging.DEBUG
    trace_level = getattr(logging, "TRACE", 5)
    if level == "trace":
        return int(trace_level)
    return logging.INFO


@dataclass(frozen=True)
class _QueuedLiveEvent:
    event: LiveEvent
    enqueued_ns: int


@dataclass(frozen=True)
class _EventPipelineTimingWindow:
    started_ns: int
    ended_ns: int
    processed_count: int
    queue_wait_ns_total: int
    queue_wait_ns_max: int
    worker_service_ns_total: int
    worker_service_ns_max: int
    structured_sink_write_count: int
    structured_sink_service_ns_total: int
    structured_sink_service_ns_max: int
    monitor_sink_write_count: int
    monitor_sink_service_ns_total: int
    monitor_sink_service_ns_max: int
    monitor_prepare_ns_total: int
    monitor_prepare_ns_max: int
    monitor_publisher_lock_wait_ns_total: int
    monitor_publisher_lock_wait_ns_max: int
    monitor_publisher_rotation_ns_total: int
    monitor_publisher_rotation_ns_max: int
    monitor_publisher_persist_ns_total: int
    monitor_publisher_persist_ns_max: int
    monitor_publisher_maintenance_ns_total: int
    monitor_publisher_maintenance_ns_max: int
    monitor_publisher_manifest_checkpoint_count: int
    monitor_publisher_manifest_checkpoint_ns_total: int
    monitor_publisher_manifest_checkpoint_ns_max: int
    monitor_publisher_retention_run_count: int
    monitor_publisher_retention_ns_total: int
    monitor_publisher_retention_ns_max: int
    monitor_publisher_retention_thread_cpu_ns_total: int
    monitor_publisher_retention_thread_cpu_ns_max: int
    monitor_publisher_retention_non_cpu_ns_total: int
    monitor_publisher_retention_non_cpu_ns_max: int
    monitor_publisher_retention_inventory_ns_total: int
    monitor_publisher_retention_inventory_ns_max: int
    monitor_publisher_retention_age_filter_ns_total: int
    monitor_publisher_retention_age_filter_ns_max: int
    monitor_publisher_retention_cap_prune_ns_total: int
    monitor_publisher_retention_cap_prune_ns_max: int
    monitor_publisher_retention_age_unlink_ns_total: int
    monitor_publisher_retention_age_unlink_ns_max: int
    monitor_publisher_retention_cap_unlink_ns_total: int
    monitor_publisher_retention_cap_unlink_ns_max: int
    monitor_publisher_retention_inventory_entries_visited: int
    monitor_publisher_retention_inventory_candidates: int
    monitor_publisher_retention_age_deleted: int
    monitor_publisher_retention_cap_deleted: int


@dataclass
class _EventPipelineSinkWriteTiming:
    structured_sink_write_count: int = 0
    structured_sink_service_ns_total: int = 0
    structured_sink_service_ns_max: int = 0
    monitor_sink_write_count: int = 0
    monitor_sink_service_ns_total: int = 0
    monitor_sink_service_ns_max: int = 0
    monitor_prepare_ns_total: int = 0
    monitor_prepare_ns_max: int = 0
    monitor_publisher_lock_wait_ns_total: int = 0
    monitor_publisher_lock_wait_ns_max: int = 0
    monitor_publisher_rotation_ns_total: int = 0
    monitor_publisher_rotation_ns_max: int = 0
    monitor_publisher_persist_ns_total: int = 0
    monitor_publisher_persist_ns_max: int = 0
    monitor_publisher_maintenance_ns_total: int = 0
    monitor_publisher_maintenance_ns_max: int = 0
    monitor_publisher_manifest_checkpoint_count: int = 0
    monitor_publisher_manifest_checkpoint_ns_total: int = 0
    monitor_publisher_manifest_checkpoint_ns_max: int = 0
    monitor_publisher_retention_run_count: int = 0
    monitor_publisher_retention_ns_total: int = 0
    monitor_publisher_retention_ns_max: int = 0
    monitor_publisher_retention_thread_cpu_ns_total: int = 0
    monitor_publisher_retention_thread_cpu_ns_max: int = 0
    monitor_publisher_retention_non_cpu_ns_total: int = 0
    monitor_publisher_retention_non_cpu_ns_max: int = 0
    monitor_publisher_retention_inventory_ns_total: int = 0
    monitor_publisher_retention_inventory_ns_max: int = 0
    monitor_publisher_retention_age_filter_ns_total: int = 0
    monitor_publisher_retention_age_filter_ns_max: int = 0
    monitor_publisher_retention_cap_prune_ns_total: int = 0
    monitor_publisher_retention_cap_prune_ns_max: int = 0
    monitor_publisher_retention_age_unlink_ns_total: int = 0
    monitor_publisher_retention_age_unlink_ns_max: int = 0
    monitor_publisher_retention_cap_unlink_ns_total: int = 0
    monitor_publisher_retention_cap_unlink_ns_max: int = 0
    monitor_publisher_retention_inventory_entries_visited: int = 0
    monitor_publisher_retention_inventory_candidates: int = 0
    monitor_publisher_retention_age_deleted: int = 0
    monitor_publisher_retention_cap_deleted: int = 0

    def record(self, sink_name: str, service_ns: int) -> None:
        service_ns = max(0, int(service_ns))
        if sink_name == "structured":
            self.structured_sink_write_count += 1
            self.structured_sink_service_ns_total += service_ns
            self.structured_sink_service_ns_max = max(
                self.structured_sink_service_ns_max, service_ns
            )
        elif sink_name == "monitor":
            self.monitor_sink_write_count += 1
            self.monitor_sink_service_ns_total += service_ns
            self.monitor_sink_service_ns_max = max(
                self.monitor_sink_service_ns_max, service_ns
            )

    def record_monitor_phase_timing(self, timing: Mapping[str, int]) -> None:
        for source_key, field_prefix in (
            ("prepare_ns", "monitor_prepare_ns"),
            ("lock_wait_ns", "monitor_publisher_lock_wait_ns"),
            ("rotation_ns", "monitor_publisher_rotation_ns"),
            ("persist_ns", "monitor_publisher_persist_ns"),
            ("maintenance_ns", "monitor_publisher_maintenance_ns"),
        ):
            value_ns = max(0, int(timing.get(source_key, 0)))
            total_field = f"{field_prefix}_total"
            max_field = f"{field_prefix}_max"
            setattr(self, total_field, int(getattr(self, total_field)) + value_ns)
            setattr(self, max_field, max(int(getattr(self, max_field)), value_ns))
        for source_key, field_name in (
            ("manifest_checkpoint_count", "monitor_publisher_manifest_checkpoint_count"),
            ("retention_run_count", "monitor_publisher_retention_run_count"),
            (
                "retention_inventory_entries_visited",
                "monitor_publisher_retention_inventory_entries_visited",
            ),
            (
                "retention_inventory_candidates",
                "monitor_publisher_retention_inventory_candidates",
            ),
            ("retention_age_deleted", "monitor_publisher_retention_age_deleted"),
            ("retention_cap_deleted", "monitor_publisher_retention_cap_deleted"),
        ):
            setattr(
                self,
                field_name,
                int(getattr(self, field_name))
                + max(0, int(timing.get(source_key, 0))),
            )
        for source_key, field_prefix in (
            ("manifest_checkpoint_ns", "monitor_publisher_manifest_checkpoint_ns"),
            ("retention_ns", "monitor_publisher_retention_ns"),
            (
                "retention_thread_cpu_ns",
                "monitor_publisher_retention_thread_cpu_ns",
            ),
            (
                "retention_non_cpu_ns",
                "monitor_publisher_retention_non_cpu_ns",
            ),
            ("retention_inventory_ns", "monitor_publisher_retention_inventory_ns"),
            ("retention_age_filter_ns", "monitor_publisher_retention_age_filter_ns"),
            ("retention_cap_prune_ns", "monitor_publisher_retention_cap_prune_ns"),
            ("retention_age_unlink_ns", "monitor_publisher_retention_age_unlink_ns"),
            ("retention_cap_unlink_ns", "monitor_publisher_retention_cap_unlink_ns"),
        ):
            total_field = f"{field_prefix}_total"
            max_field = f"{field_prefix}_max"
            value_total_ns = max(0, int(timing.get(f"{source_key}_total", 0)))
            value_max_ns = max(0, int(timing.get(f"{source_key}_max", 0)))
            setattr(self, total_field, int(getattr(self, total_field)) + value_total_ns)
            setattr(self, max_field, max(int(getattr(self, max_field)), value_max_ns))


class LiveEventPipeline:
    def __init__(
        self,
        *,
        context: LiveEventContext | None = None,
        routes: Mapping[str, EventRoute] | None = None,
        structured_sinks: Iterable[LiveEventSink] = (),
        monitor_sinks: Iterable[LiveEventSink] = (),
        console_sink: LiveEventSink | None = None,
        text_sink: LiveEventSink | None = None,
        queue_maxsize: int = 10_000,
        debug_profiles: Iterable[str] = (),
        start: bool = True,
    ):
        self.context = context or LiveEventContext()
        self.routes = dict(DEFAULT_ROUTES)
        if routes:
            self.routes.update(routes)
        self.structured_sinks = tuple(structured_sinks)
        self.monitor_sinks = tuple(monitor_sinks)
        self.console_sink = console_sink
        self.text_sink = text_sink
        self.debug_profiles = normalize_live_event_debug_profiles(debug_profiles)
        self.drop_counters: Counter[str] = Counter()
        self.sink_error_counters: Counter[str] = Counter()
        self.degraded_events: deque[LiveEvent] = deque(maxlen=1_000)
        self._throttle_last_emit_ms: dict[tuple[str, str], int] = {}
        self._queue: queue.Queue[_QueuedLiveEvent | None] = queue.Queue(
            maxsize=queue_maxsize
        )
        self._stop = threading.Event()
        self._enqueue_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._timing_window_started_ns = time.monotonic_ns()
        self._timing_processed_count = 0
        self._timing_queue_wait_ns_total = 0
        self._timing_queue_wait_ns_max = 0
        self._timing_worker_service_ns_total = 0
        self._timing_worker_service_ns_max = 0
        self._timing_structured_sink_write_count = 0
        self._timing_structured_sink_service_ns_total = 0
        self._timing_structured_sink_service_ns_max = 0
        self._timing_monitor_sink_write_count = 0
        self._timing_monitor_sink_service_ns_total = 0
        self._timing_monitor_sink_service_ns_max = 0
        self._timing_monitor_prepare_ns_total = 0
        self._timing_monitor_prepare_ns_max = 0
        self._timing_monitor_publisher_lock_wait_ns_total = 0
        self._timing_monitor_publisher_lock_wait_ns_max = 0
        self._timing_monitor_publisher_rotation_ns_total = 0
        self._timing_monitor_publisher_rotation_ns_max = 0
        self._timing_monitor_publisher_persist_ns_total = 0
        self._timing_monitor_publisher_persist_ns_max = 0
        self._timing_monitor_publisher_maintenance_ns_total = 0
        self._timing_monitor_publisher_maintenance_ns_max = 0
        self._timing_monitor_publisher_manifest_checkpoint_count = 0
        self._timing_monitor_publisher_manifest_checkpoint_ns_total = 0
        self._timing_monitor_publisher_manifest_checkpoint_ns_max = 0
        self._timing_monitor_publisher_retention_run_count = 0
        self._timing_monitor_publisher_retention_ns_total = 0
        self._timing_monitor_publisher_retention_ns_max = 0
        self._timing_monitor_publisher_retention_thread_cpu_ns_total = 0
        self._timing_monitor_publisher_retention_thread_cpu_ns_max = 0
        self._timing_monitor_publisher_retention_non_cpu_ns_total = 0
        self._timing_monitor_publisher_retention_non_cpu_ns_max = 0
        self._timing_monitor_publisher_retention_inventory_ns_total = 0
        self._timing_monitor_publisher_retention_inventory_ns_max = 0
        self._timing_monitor_publisher_retention_age_filter_ns_total = 0
        self._timing_monitor_publisher_retention_age_filter_ns_max = 0
        self._timing_monitor_publisher_retention_cap_prune_ns_total = 0
        self._timing_monitor_publisher_retention_cap_prune_ns_max = 0
        self._timing_monitor_publisher_retention_age_unlink_ns_total = 0
        self._timing_monitor_publisher_retention_age_unlink_ns_max = 0
        self._timing_monitor_publisher_retention_cap_unlink_ns_total = 0
        self._timing_monitor_publisher_retention_cap_unlink_ns_max = 0
        self._timing_monitor_publisher_retention_inventory_entries_visited = 0
        self._timing_monitor_publisher_retention_inventory_candidates = 0
        self._timing_monitor_publisher_retention_age_deleted = 0
        self._timing_monitor_publisher_retention_cap_deleted = 0
        self._pending_timing_windows: dict[int, _EventPipelineTimingWindow] = {}
        self._next_timing_snapshot_token = 1
        self._worker: threading.Thread | None = None
        if start:
            self.start()

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._drain,
            name="live-event-pipeline",
            daemon=True,
        )
        self._worker.start()

    def route_for(self, event: LiveEvent) -> EventRoute:
        return self.routes.get(event.event_type, DEFAULT_ROUTE)

    def with_context_ids(self, **kwargs: str | None) -> LiveEventContext:
        self.context = self.context.with_ids(**kwargs)
        return self.context

    @staticmethod
    def _timing_ms(value_ns: int) -> float:
        return round(max(0, int(value_ns)) / 1_000_000.0, 3)

    def health_snapshot(self) -> dict[str, Any]:
        """Return best-effort operational counters for periodic health events."""
        snapshot, _token = self._health_snapshot(consume_timing=False)
        return snapshot

    def consume_timing_snapshot(self) -> tuple[dict[str, Any], int]:
        snapshot, token = self._health_snapshot(consume_timing=True)
        assert token is not None
        return snapshot, token

    def _health_snapshot(
        self, *, consume_timing: bool
    ) -> tuple[dict[str, Any], int | None]:
        try:
            queue_depth = int(self._queue.qsize())
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            queue_depth = None
        try:
            unfinished_tasks = int(self._queue.unfinished_tasks)
        except (AttributeError, TypeError, ValueError):
            unfinished_tasks = None
        try:
            queue_maxsize = int(getattr(self._queue, "maxsize", 0) or 0)
        except (TypeError, ValueError):
            queue_maxsize = None
        with self._state_lock:
            drop_counts = dict(self.drop_counters)
            sink_error_counts = dict(self.sink_error_counters)
            degraded_count = len(self.degraded_events)
            timing_now_ns = time.monotonic_ns()
            timing_window = _EventPipelineTimingWindow(
                started_ns=int(self._timing_window_started_ns),
                ended_ns=int(timing_now_ns),
                processed_count=int(self._timing_processed_count),
                queue_wait_ns_total=int(self._timing_queue_wait_ns_total),
                queue_wait_ns_max=int(self._timing_queue_wait_ns_max),
                worker_service_ns_total=int(self._timing_worker_service_ns_total),
                worker_service_ns_max=int(self._timing_worker_service_ns_max),
                structured_sink_write_count=int(
                    self._timing_structured_sink_write_count
                ),
                structured_sink_service_ns_total=int(
                    self._timing_structured_sink_service_ns_total
                ),
                structured_sink_service_ns_max=int(
                    self._timing_structured_sink_service_ns_max
                ),
                monitor_sink_write_count=int(self._timing_monitor_sink_write_count),
                monitor_sink_service_ns_total=int(
                    self._timing_monitor_sink_service_ns_total
                ),
                monitor_sink_service_ns_max=int(
                    self._timing_monitor_sink_service_ns_max
                ),
                monitor_prepare_ns_total=int(self._timing_monitor_prepare_ns_total),
                monitor_prepare_ns_max=int(self._timing_monitor_prepare_ns_max),
                monitor_publisher_lock_wait_ns_total=int(
                    self._timing_monitor_publisher_lock_wait_ns_total
                ),
                monitor_publisher_lock_wait_ns_max=int(
                    self._timing_monitor_publisher_lock_wait_ns_max
                ),
                monitor_publisher_rotation_ns_total=int(
                    self._timing_monitor_publisher_rotation_ns_total
                ),
                monitor_publisher_rotation_ns_max=int(
                    self._timing_monitor_publisher_rotation_ns_max
                ),
                monitor_publisher_persist_ns_total=int(
                    self._timing_monitor_publisher_persist_ns_total
                ),
                monitor_publisher_persist_ns_max=int(
                    self._timing_monitor_publisher_persist_ns_max
                ),
                monitor_publisher_maintenance_ns_total=int(
                    self._timing_monitor_publisher_maintenance_ns_total
                ),
                monitor_publisher_maintenance_ns_max=int(
                    self._timing_monitor_publisher_maintenance_ns_max
                ),
                monitor_publisher_manifest_checkpoint_count=int(
                    self._timing_monitor_publisher_manifest_checkpoint_count
                ),
                monitor_publisher_manifest_checkpoint_ns_total=int(
                    self._timing_monitor_publisher_manifest_checkpoint_ns_total
                ),
                monitor_publisher_manifest_checkpoint_ns_max=int(
                    self._timing_monitor_publisher_manifest_checkpoint_ns_max
                ),
                monitor_publisher_retention_run_count=int(
                    self._timing_monitor_publisher_retention_run_count
                ),
                monitor_publisher_retention_ns_total=int(
                    self._timing_monitor_publisher_retention_ns_total
                ),
                monitor_publisher_retention_ns_max=int(
                    self._timing_monitor_publisher_retention_ns_max
                ),
                monitor_publisher_retention_thread_cpu_ns_total=int(
                    self._timing_monitor_publisher_retention_thread_cpu_ns_total
                ),
                monitor_publisher_retention_thread_cpu_ns_max=int(
                    self._timing_monitor_publisher_retention_thread_cpu_ns_max
                ),
                monitor_publisher_retention_non_cpu_ns_total=int(
                    self._timing_monitor_publisher_retention_non_cpu_ns_total
                ),
                monitor_publisher_retention_non_cpu_ns_max=int(
                    self._timing_monitor_publisher_retention_non_cpu_ns_max
                ),
                monitor_publisher_retention_inventory_ns_total=int(
                    self._timing_monitor_publisher_retention_inventory_ns_total
                ),
                monitor_publisher_retention_inventory_ns_max=int(
                    self._timing_monitor_publisher_retention_inventory_ns_max
                ),
                monitor_publisher_retention_age_filter_ns_total=int(
                    self._timing_monitor_publisher_retention_age_filter_ns_total
                ),
                monitor_publisher_retention_age_filter_ns_max=int(
                    self._timing_monitor_publisher_retention_age_filter_ns_max
                ),
                monitor_publisher_retention_cap_prune_ns_total=int(
                    self._timing_monitor_publisher_retention_cap_prune_ns_total
                ),
                monitor_publisher_retention_cap_prune_ns_max=int(
                    self._timing_monitor_publisher_retention_cap_prune_ns_max
                ),
                monitor_publisher_retention_age_unlink_ns_total=int(
                    self._timing_monitor_publisher_retention_age_unlink_ns_total
                ),
                monitor_publisher_retention_age_unlink_ns_max=int(
                    self._timing_monitor_publisher_retention_age_unlink_ns_max
                ),
                monitor_publisher_retention_cap_unlink_ns_total=int(
                    self._timing_monitor_publisher_retention_cap_unlink_ns_total
                ),
                monitor_publisher_retention_cap_unlink_ns_max=int(
                    self._timing_monitor_publisher_retention_cap_unlink_ns_max
                ),
                monitor_publisher_retention_inventory_entries_visited=int(
                    self._timing_monitor_publisher_retention_inventory_entries_visited
                ),
                monitor_publisher_retention_inventory_candidates=int(
                    self._timing_monitor_publisher_retention_inventory_candidates
                ),
                monitor_publisher_retention_age_deleted=int(
                    self._timing_monitor_publisher_retention_age_deleted
                ),
                monitor_publisher_retention_cap_deleted=int(
                    self._timing_monitor_publisher_retention_cap_deleted
                ),
            )
            timing_snapshot_token = None
            if consume_timing:
                timing_snapshot_token = int(self._next_timing_snapshot_token)
                self._next_timing_snapshot_token += 1
                self._pending_timing_windows[timing_snapshot_token] = timing_window
                self._timing_window_started_ns = int(timing_now_ns)
                self._timing_processed_count = 0
                self._timing_queue_wait_ns_total = 0
                self._timing_queue_wait_ns_max = 0
                self._timing_worker_service_ns_total = 0
                self._timing_worker_service_ns_max = 0
                self._timing_structured_sink_write_count = 0
                self._timing_structured_sink_service_ns_total = 0
                self._timing_structured_sink_service_ns_max = 0
                self._timing_monitor_sink_write_count = 0
                self._timing_monitor_sink_service_ns_total = 0
                self._timing_monitor_sink_service_ns_max = 0
                self._timing_monitor_prepare_ns_total = 0
                self._timing_monitor_prepare_ns_max = 0
                self._timing_monitor_publisher_lock_wait_ns_total = 0
                self._timing_monitor_publisher_lock_wait_ns_max = 0
                self._timing_monitor_publisher_rotation_ns_total = 0
                self._timing_monitor_publisher_rotation_ns_max = 0
                self._timing_monitor_publisher_persist_ns_total = 0
                self._timing_monitor_publisher_persist_ns_max = 0
                self._timing_monitor_publisher_maintenance_ns_total = 0
                self._timing_monitor_publisher_maintenance_ns_max = 0
                self._timing_monitor_publisher_manifest_checkpoint_count = 0
                self._timing_monitor_publisher_manifest_checkpoint_ns_total = 0
                self._timing_monitor_publisher_manifest_checkpoint_ns_max = 0
                self._timing_monitor_publisher_retention_run_count = 0
                self._timing_monitor_publisher_retention_ns_total = 0
                self._timing_monitor_publisher_retention_ns_max = 0
                self._timing_monitor_publisher_retention_thread_cpu_ns_total = 0
                self._timing_monitor_publisher_retention_thread_cpu_ns_max = 0
                self._timing_monitor_publisher_retention_non_cpu_ns_total = 0
                self._timing_monitor_publisher_retention_non_cpu_ns_max = 0
                self._timing_monitor_publisher_retention_inventory_ns_total = 0
                self._timing_monitor_publisher_retention_inventory_ns_max = 0
                self._timing_monitor_publisher_retention_age_filter_ns_total = 0
                self._timing_monitor_publisher_retention_age_filter_ns_max = 0
                self._timing_monitor_publisher_retention_cap_prune_ns_total = 0
                self._timing_monitor_publisher_retention_cap_prune_ns_max = 0
                self._timing_monitor_publisher_retention_age_unlink_ns_total = 0
                self._timing_monitor_publisher_retention_age_unlink_ns_max = 0
                self._timing_monitor_publisher_retention_cap_unlink_ns_total = 0
                self._timing_monitor_publisher_retention_cap_unlink_ns_max = 0
                self._timing_monitor_publisher_retention_inventory_entries_visited = 0
                self._timing_monitor_publisher_retention_inventory_candidates = 0
                self._timing_monitor_publisher_retention_age_deleted = 0
                self._timing_monitor_publisher_retention_cap_deleted = 0
        snapshot: dict[str, Any] = {
            "event_queue_depth": queue_depth,
            "event_queue_maxsize": queue_maxsize,
            "event_queue_unfinished_tasks": unfinished_tasks,
            "event_dropped_total": int(sum(drop_counts.values())),
            "event_drop_counts": drop_counts,
            "event_sink_error_total": int(sum(sink_error_counts.values())),
            "event_sink_error_counts": sink_error_counts,
            "event_degraded_count": int(degraded_count),
            "event_pipeline_stopping": bool(self._stop.is_set()),
            "event_pipeline_worker_alive": bool(
                self._worker is not None and self._worker.is_alive()
            ),
            "event_pipeline_timing_window_ms": self._timing_ms(
                max(0, timing_window.ended_ns - timing_window.started_ns)
            ),
            "event_pipeline_processed_count": timing_window.processed_count,
            "event_queue_wait_ms_total": self._timing_ms(
                timing_window.queue_wait_ns_total
            ),
            "event_queue_wait_ms_max": self._timing_ms(timing_window.queue_wait_ns_max),
            "event_worker_service_ms_total": self._timing_ms(
                timing_window.worker_service_ns_total
            ),
            "event_worker_service_ms_max": self._timing_ms(
                timing_window.worker_service_ns_max
            ),
            "event_structured_sink_write_count": timing_window.structured_sink_write_count,
            "event_structured_sink_service_ms_total": self._timing_ms(
                timing_window.structured_sink_service_ns_total
            ),
            "event_structured_sink_service_ms_max": self._timing_ms(
                timing_window.structured_sink_service_ns_max
            ),
            "event_monitor_sink_write_count": timing_window.monitor_sink_write_count,
            "event_monitor_sink_service_ms_total": self._timing_ms(
                timing_window.monitor_sink_service_ns_total
            ),
            "event_monitor_sink_service_ms_max": self._timing_ms(
                timing_window.monitor_sink_service_ns_max
            ),
            "event_monitor_prepare_ms_total": self._timing_ms(
                timing_window.monitor_prepare_ns_total
            ),
            "event_monitor_prepare_ms_max": self._timing_ms(
                timing_window.monitor_prepare_ns_max
            ),
            "event_monitor_publisher_lock_wait_ms_total": self._timing_ms(
                timing_window.monitor_publisher_lock_wait_ns_total
            ),
            "event_monitor_publisher_lock_wait_ms_max": self._timing_ms(
                timing_window.monitor_publisher_lock_wait_ns_max
            ),
            "event_monitor_publisher_rotation_ms_total": self._timing_ms(
                timing_window.monitor_publisher_rotation_ns_total
            ),
            "event_monitor_publisher_rotation_ms_max": self._timing_ms(
                timing_window.monitor_publisher_rotation_ns_max
            ),
            "event_monitor_publisher_persist_ms_total": self._timing_ms(
                timing_window.monitor_publisher_persist_ns_total
            ),
            "event_monitor_publisher_persist_ms_max": self._timing_ms(
                timing_window.monitor_publisher_persist_ns_max
            ),
            "event_monitor_publisher_maintenance_ms_total": self._timing_ms(
                timing_window.monitor_publisher_maintenance_ns_total
            ),
            "event_monitor_publisher_maintenance_ms_max": self._timing_ms(
                timing_window.monitor_publisher_maintenance_ns_max
            ),
            "event_monitor_publisher_manifest_checkpoint_count": (
                timing_window.monitor_publisher_manifest_checkpoint_count
            ),
            "event_monitor_publisher_manifest_checkpoint_ms_total": self._timing_ms(
                timing_window.monitor_publisher_manifest_checkpoint_ns_total
            ),
            "event_monitor_publisher_manifest_checkpoint_ms_max": self._timing_ms(
                timing_window.monitor_publisher_manifest_checkpoint_ns_max
            ),
            "event_monitor_publisher_retention_run_count": (
                timing_window.monitor_publisher_retention_run_count
            ),
            "event_monitor_publisher_retention_ms_total": self._timing_ms(
                timing_window.monitor_publisher_retention_ns_total
            ),
            "event_monitor_publisher_retention_ms_max": self._timing_ms(
                timing_window.monitor_publisher_retention_ns_max
            ),
            "event_monitor_publisher_retention_thread_cpu_ms_total": self._timing_ms(
                timing_window.monitor_publisher_retention_thread_cpu_ns_total
            ),
            "event_monitor_publisher_retention_thread_cpu_ms_max": self._timing_ms(
                timing_window.monitor_publisher_retention_thread_cpu_ns_max
            ),
            "event_monitor_publisher_retention_non_cpu_ms_total": self._timing_ms(
                timing_window.monitor_publisher_retention_non_cpu_ns_total
            ),
            "event_monitor_publisher_retention_non_cpu_ms_max": self._timing_ms(
                timing_window.monitor_publisher_retention_non_cpu_ns_max
            ),
            "event_monitor_publisher_retention_inventory_ms_total": self._timing_ms(
                timing_window.monitor_publisher_retention_inventory_ns_total
            ),
            "event_monitor_publisher_retention_inventory_ms_max": self._timing_ms(
                timing_window.monitor_publisher_retention_inventory_ns_max
            ),
            "event_monitor_publisher_retention_age_filter_ms_total": self._timing_ms(
                timing_window.monitor_publisher_retention_age_filter_ns_total
            ),
            "event_monitor_publisher_retention_age_filter_ms_max": self._timing_ms(
                timing_window.monitor_publisher_retention_age_filter_ns_max
            ),
            "event_monitor_publisher_retention_cap_prune_ms_total": self._timing_ms(
                timing_window.monitor_publisher_retention_cap_prune_ns_total
            ),
            "event_monitor_publisher_retention_cap_prune_ms_max": self._timing_ms(
                timing_window.monitor_publisher_retention_cap_prune_ns_max
            ),
            "event_monitor_publisher_retention_age_unlink_ms_total": self._timing_ms(
                timing_window.monitor_publisher_retention_age_unlink_ns_total
            ),
            "event_monitor_publisher_retention_age_unlink_ms_max": self._timing_ms(
                timing_window.monitor_publisher_retention_age_unlink_ns_max
            ),
            "event_monitor_publisher_retention_cap_unlink_ms_total": self._timing_ms(
                timing_window.monitor_publisher_retention_cap_unlink_ns_total
            ),
            "event_monitor_publisher_retention_cap_unlink_ms_max": self._timing_ms(
                timing_window.monitor_publisher_retention_cap_unlink_ns_max
            ),
            "event_monitor_publisher_retention_inventory_entries_visited": (
                timing_window.monitor_publisher_retention_inventory_entries_visited
            ),
            "event_monitor_publisher_retention_inventory_candidates": (
                timing_window.monitor_publisher_retention_inventory_candidates
            ),
            "event_monitor_publisher_retention_age_deleted": (
                timing_window.monitor_publisher_retention_age_deleted
            ),
            "event_monitor_publisher_retention_cap_deleted": (
                timing_window.monitor_publisher_retention_cap_deleted
            ),
        }
        return (
            {key: value for key, value in snapshot.items() if value is not None},
            timing_snapshot_token,
        )

    def confirm_timing_snapshot(self, token: int) -> None:
        with self._state_lock:
            self._pending_timing_windows.pop(int(token), None)

    def restore_timing_snapshot(self, token: int) -> None:
        with self._state_lock:
            pending = self._pending_timing_windows.pop(int(token), None)
            if pending is not None:
                self._restore_timing_window_locked(pending)

    def _restore_timing_window_locked(
        self, pending: _EventPipelineTimingWindow
    ) -> None:
        self._timing_window_started_ns = min(
            int(pending.started_ns), int(self._timing_window_started_ns)
        )
        self._timing_processed_count += int(pending.processed_count)
        self._timing_queue_wait_ns_total += int(pending.queue_wait_ns_total)
        self._timing_queue_wait_ns_max = max(
            self._timing_queue_wait_ns_max, int(pending.queue_wait_ns_max)
        )
        self._timing_worker_service_ns_total += int(pending.worker_service_ns_total)
        self._timing_worker_service_ns_max = max(
            self._timing_worker_service_ns_max, int(pending.worker_service_ns_max)
        )
        self._timing_structured_sink_write_count += int(
            pending.structured_sink_write_count
        )
        self._timing_structured_sink_service_ns_total += int(
            pending.structured_sink_service_ns_total
        )
        self._timing_structured_sink_service_ns_max = max(
            self._timing_structured_sink_service_ns_max,
            int(pending.structured_sink_service_ns_max),
        )
        self._timing_monitor_sink_write_count += int(pending.monitor_sink_write_count)
        self._timing_monitor_sink_service_ns_total += int(
            pending.monitor_sink_service_ns_total
        )
        self._timing_monitor_sink_service_ns_max = max(
            self._timing_monitor_sink_service_ns_max,
            int(pending.monitor_sink_service_ns_max),
        )
        for field_name in (
            "monitor_prepare_ns_total",
            "monitor_publisher_lock_wait_ns_total",
            "monitor_publisher_rotation_ns_total",
            "monitor_publisher_persist_ns_total",
            "monitor_publisher_maintenance_ns_total",
            "monitor_publisher_manifest_checkpoint_count",
            "monitor_publisher_manifest_checkpoint_ns_total",
            "monitor_publisher_retention_run_count",
            "monitor_publisher_retention_ns_total",
            "monitor_publisher_retention_thread_cpu_ns_total",
            "monitor_publisher_retention_non_cpu_ns_total",
            "monitor_publisher_retention_inventory_ns_total",
            "monitor_publisher_retention_age_filter_ns_total",
            "monitor_publisher_retention_cap_prune_ns_total",
            "monitor_publisher_retention_age_unlink_ns_total",
            "monitor_publisher_retention_cap_unlink_ns_total",
            "monitor_publisher_retention_inventory_entries_visited",
            "monitor_publisher_retention_inventory_candidates",
            "monitor_publisher_retention_age_deleted",
            "monitor_publisher_retention_cap_deleted",
        ):
            setattr(
                self,
                f"_timing_{field_name}",
                int(getattr(self, f"_timing_{field_name}"))
                + int(getattr(pending, field_name)),
            )
        for field_name in (
            "monitor_prepare_ns_max",
            "monitor_publisher_lock_wait_ns_max",
            "monitor_publisher_rotation_ns_max",
            "monitor_publisher_persist_ns_max",
            "monitor_publisher_maintenance_ns_max",
            "monitor_publisher_manifest_checkpoint_ns_max",
            "monitor_publisher_retention_ns_max",
            "monitor_publisher_retention_thread_cpu_ns_max",
            "monitor_publisher_retention_non_cpu_ns_max",
            "monitor_publisher_retention_inventory_ns_max",
            "monitor_publisher_retention_age_filter_ns_max",
            "monitor_publisher_retention_cap_prune_ns_max",
            "monitor_publisher_retention_age_unlink_ns_max",
            "monitor_publisher_retention_cap_unlink_ns_max",
        ):
            setattr(
                self,
                f"_timing_{field_name}",
                max(
                    int(getattr(self, f"_timing_{field_name}")),
                    int(getattr(pending, field_name)),
                ),
            )

    def emit(
        self,
        event: LiveEvent | Mapping[str, Any],
        *,
        require_enqueue: bool = False,
        **overrides: Any,
    ) -> LiveEvent | None:
        if isinstance(event, LiveEvent):
            live_event = event
        else:
            live_event = LiveEvent(**dict(event))
        if overrides:
            live_event = replace(live_event, **overrides)
        live_event = live_event.with_context(self.context)
        route = self.route_for(live_event)
        if (
            route.console
            and self.console_sink is not None
            and _console_sink_event_visible(live_event)
            and self._should_emit_throttled_sink("console", live_event, route)
        ):
            self._write_sink("console", self.console_sink, live_event)
        if (
            route.text
            and self.text_sink is not None
            and _operator_sink_event_visible(live_event)
            and self._should_emit_throttled_sink("text", live_event, route)
        ):
            self._write_sink("text", self.text_sink, live_event)
        enqueued = True
        if route.structured or route.monitor:
            with self._enqueue_lock:
                if self._stop.is_set():
                    enqueued = False
                    with self._state_lock:
                        self.drop_counters[live_event.event_type] += 1
                    self._record_degraded(
                        reason_code=ReasonCodes.SINK_PIPELINE_CLOSING,
                        message=f"live event pipeline closing; dropped {live_event.event_type}",
                        data={"dropped_event_type": live_event.event_type},
                    )
                else:
                    try:
                        self._queue.put_nowait(
                            _QueuedLiveEvent(
                                event=live_event,
                                enqueued_ns=time.monotonic_ns(),
                            )
                        )
                    except queue.Full:
                        enqueued = False
                        with self._state_lock:
                            self.drop_counters[live_event.event_type] += 1
                        self._record_degraded(
                            reason_code=ReasonCodes.QUEUE_FULL,
                            message=f"live event queue full; dropped {live_event.event_type}",
                            data={"dropped_event_type": live_event.event_type},
                        )
        if require_enqueue and not enqueued:
            return None
        return live_event

    def _should_emit_throttled_sink(
        self, sink_name: str, event: LiveEvent, route: EventRoute
    ) -> bool:
        interval_ms = int(route.throttle_interval_ms or 0)
        if interval_ms <= 0:
            return True
        key = (str(sink_name), event.event_type)
        now_ms = int(event.ts_ms)
        with self._state_lock:
            last_ms = self._throttle_last_emit_ms.get(key)
            if last_ms is not None and now_ms - int(last_ms) < interval_ms:
                return False
            self._throttle_last_emit_ms[key] = now_ms
        return True

    def flush(self, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return self._queue.unfinished_tasks == 0

    def close(self, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._enqueue_lock:
            self._stop.set()
        if self._worker is None:
            return self.flush(timeout=max(0.0, deadline - time.monotonic()))
        sentinel_queued = False
        while time.monotonic() < deadline:
            try:
                with self._enqueue_lock:
                    self._queue.put(
                        None,
                        timeout=min(0.05, max(0.0, deadline - time.monotonic())),
                    )
                    sentinel_queued = True
                    break
            except queue.Full:
                continue
        if not sentinel_queued:
            logging.warning("[event] live event pipeline close timed out before sentinel")
            return False
        if self._worker is not None:
            self._worker.join(timeout=max(0.0, deadline - time.monotonic()))
            closed = not self._worker.is_alive()
            if not closed:
                logging.warning("[event] live event pipeline close timed out while draining")
            return closed
        return True

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                service_started_ns = time.monotonic_ns()
                sink_write_timing = _EventPipelineSinkWriteTiming()
                live_event = item.event
                route = self.route_for(live_event)
                if route.structured:
                    for sink in self.structured_sinks:
                        self._write_sink_in_worker(
                            "structured", sink, live_event, sink_write_timing
                        )
                if route.monitor:
                    for sink in self.monitor_sinks:
                        self._write_monitor_sink_in_worker(
                            sink, live_event, sink_write_timing
                        )
            finally:
                if item is not None:
                    service_finished_ns = time.monotonic_ns()
                    queue_wait_ns = max(
                        0, int(service_started_ns) - int(item.enqueued_ns)
                    )
                    service_ns = max(
                        0, int(service_finished_ns) - int(service_started_ns)
                    )
                    with self._state_lock:
                        self._timing_processed_count += 1
                        self._timing_queue_wait_ns_total += queue_wait_ns
                        self._timing_queue_wait_ns_max = max(
                            self._timing_queue_wait_ns_max, queue_wait_ns
                        )
                        self._timing_worker_service_ns_total += service_ns
                        self._timing_worker_service_ns_max = max(
                            self._timing_worker_service_ns_max, service_ns
                        )
                        self._timing_structured_sink_write_count += (
                            sink_write_timing.structured_sink_write_count
                        )
                        self._timing_structured_sink_service_ns_total += (
                            sink_write_timing.structured_sink_service_ns_total
                        )
                        self._timing_structured_sink_service_ns_max = max(
                            self._timing_structured_sink_service_ns_max,
                            sink_write_timing.structured_sink_service_ns_max,
                        )
                        self._timing_monitor_sink_write_count += (
                            sink_write_timing.monitor_sink_write_count
                        )
                        self._timing_monitor_sink_service_ns_total += (
                            sink_write_timing.monitor_sink_service_ns_total
                        )
                        self._timing_monitor_sink_service_ns_max = max(
                            self._timing_monitor_sink_service_ns_max,
                            sink_write_timing.monitor_sink_service_ns_max,
                        )
                        self._timing_monitor_prepare_ns_total += (
                            sink_write_timing.monitor_prepare_ns_total
                        )
                        self._timing_monitor_prepare_ns_max = max(
                            self._timing_monitor_prepare_ns_max,
                            sink_write_timing.monitor_prepare_ns_max,
                        )
                        self._timing_monitor_publisher_lock_wait_ns_total += (
                            sink_write_timing.monitor_publisher_lock_wait_ns_total
                        )
                        self._timing_monitor_publisher_lock_wait_ns_max = max(
                            self._timing_monitor_publisher_lock_wait_ns_max,
                            sink_write_timing.monitor_publisher_lock_wait_ns_max,
                        )
                        self._timing_monitor_publisher_rotation_ns_total += (
                            sink_write_timing.monitor_publisher_rotation_ns_total
                        )
                        self._timing_monitor_publisher_rotation_ns_max = max(
                            self._timing_monitor_publisher_rotation_ns_max,
                            sink_write_timing.monitor_publisher_rotation_ns_max,
                        )
                        self._timing_monitor_publisher_persist_ns_total += (
                            sink_write_timing.monitor_publisher_persist_ns_total
                        )
                        self._timing_monitor_publisher_persist_ns_max = max(
                            self._timing_monitor_publisher_persist_ns_max,
                            sink_write_timing.monitor_publisher_persist_ns_max,
                        )
                        self._timing_monitor_publisher_maintenance_ns_total += (
                            sink_write_timing.monitor_publisher_maintenance_ns_total
                        )
                        self._timing_monitor_publisher_maintenance_ns_max = max(
                            self._timing_monitor_publisher_maintenance_ns_max,
                            sink_write_timing.monitor_publisher_maintenance_ns_max,
                        )
                        self._timing_monitor_publisher_manifest_checkpoint_count += (
                            sink_write_timing.monitor_publisher_manifest_checkpoint_count
                        )
                        self._timing_monitor_publisher_manifest_checkpoint_ns_total += (
                            sink_write_timing.monitor_publisher_manifest_checkpoint_ns_total
                        )
                        self._timing_monitor_publisher_manifest_checkpoint_ns_max = max(
                            self._timing_monitor_publisher_manifest_checkpoint_ns_max,
                            sink_write_timing.monitor_publisher_manifest_checkpoint_ns_max,
                        )
                        self._timing_monitor_publisher_retention_run_count += (
                            sink_write_timing.monitor_publisher_retention_run_count
                        )
                        self._timing_monitor_publisher_retention_ns_total += (
                            sink_write_timing.monitor_publisher_retention_ns_total
                        )
                        self._timing_monitor_publisher_retention_ns_max = max(
                            self._timing_monitor_publisher_retention_ns_max,
                            sink_write_timing.monitor_publisher_retention_ns_max,
                        )
                        for field_name in (
                            "monitor_publisher_retention_thread_cpu_ns_total",
                            "monitor_publisher_retention_non_cpu_ns_total",
                            "monitor_publisher_retention_inventory_ns_total",
                            "monitor_publisher_retention_age_filter_ns_total",
                            "monitor_publisher_retention_cap_prune_ns_total",
                            "monitor_publisher_retention_age_unlink_ns_total",
                            "monitor_publisher_retention_cap_unlink_ns_total",
                            "monitor_publisher_retention_inventory_entries_visited",
                            "monitor_publisher_retention_inventory_candidates",
                            "monitor_publisher_retention_age_deleted",
                            "monitor_publisher_retention_cap_deleted",
                        ):
                            timing_field = f"_timing_{field_name}"
                            setattr(
                                self,
                                timing_field,
                                int(getattr(self, timing_field))
                                + int(getattr(sink_write_timing, field_name)),
                            )
                        for field_name in (
                            "monitor_publisher_retention_thread_cpu_ns_max",
                            "monitor_publisher_retention_non_cpu_ns_max",
                            "monitor_publisher_retention_inventory_ns_max",
                            "monitor_publisher_retention_age_filter_ns_max",
                            "monitor_publisher_retention_cap_prune_ns_max",
                            "monitor_publisher_retention_age_unlink_ns_max",
                            "monitor_publisher_retention_cap_unlink_ns_max",
                        ):
                            timing_field = f"_timing_{field_name}"
                            setattr(
                                self,
                                timing_field,
                                max(
                                    int(getattr(self, timing_field)),
                                    int(getattr(sink_write_timing, field_name)),
                                ),
                            )
                self._queue.task_done()

    def _write_sink(self, name: str, sink: LiveEventSink, event: LiveEvent) -> Any:
        try:
            return sink.write(event)
        except Exception as exc:
            self._handle_sink_failure(name, exc)
            return None

    def _write_sink_in_worker(
        self,
        name: str,
        sink: LiveEventSink,
        event: LiveEvent,
        sink_write_timing: _EventPipelineSinkWriteTiming,
    ) -> Any:
        sink_started_ns = time.monotonic_ns()
        try:
            try:
                return sink.write(event)
            finally:
                sink_write_timing.record(
                    name, time.monotonic_ns() - sink_started_ns
                )
        except Exception as exc:
            self._handle_sink_failure(name, exc, sink_write_timing=sink_write_timing)
            return None

    def _write_monitor_sink_in_worker(
        self,
        sink: LiveEventSink,
        event: LiveEvent,
        sink_write_timing: _EventPipelineSinkWriteTiming,
    ) -> Any:
        if not isinstance(sink, MonitorEventSink):
            return self._write_sink_in_worker("monitor", sink, event, sink_write_timing)

        sink_started_ns = time.monotonic_ns()
        try:
            result, phase_timing = sink._write_with_timing(event)
            sink_write_timing.record_monitor_phase_timing(phase_timing)
            if result is None:
                raise RuntimeError(
                    f"monitor publisher returned None for {event.event_type}"
                )
            return result
        except _MonitorEventPrepareError as exc:
            sink_write_timing.record_monitor_phase_timing(exc.timing)
            self._handle_sink_failure("monitor", exc.error, sink_write_timing=sink_write_timing)
            return None
        except Exception as exc:
            self._handle_sink_failure("monitor", exc, sink_write_timing=sink_write_timing)
            return None
        finally:
            sink_write_timing.record("monitor", time.monotonic_ns() - sink_started_ns)

    def _handle_sink_failure(
        self,
        name: str,
        exc: Exception,
        *,
        sink_write_timing: _EventPipelineSinkWriteTiming | None = None,
    ) -> None:
        with self._state_lock:
            self.sink_error_counters[name] += 1
        self._record_degraded(
            reason_code=sink_failed_reason_code(name),
            message=f"{name} sink failed: {type(exc).__name__}",
            data={"sink": name, "error_type": type(exc).__name__},
            sink_write_timing=sink_write_timing,
        )

    def _record_degraded(
        self,
        *,
        reason_code: str,
        message: str,
        data: Mapping[str, Any] | None = None,
        sink_write_timing: _EventPipelineSinkWriteTiming | None = None,
    ) -> None:
        degraded = LiveEvent(
            EventTypes.SINK_DEGRADED,
            level="warning",
            source="live",
            component="event_bus",
            tags=(EventTags.LOGGING, EventTags.SINK),
            status="degraded",
            reason_code=reason_code,
            message=message,
            data=data or {},
        ).with_context(self.context)
        with self._state_lock:
            self.degraded_events.append(degraded)
        logging.warning("[event] %s | reason=%s", message, reason_code)
        if self.console_sink is not None:
            try:
                self.console_sink.write(degraded)
            except Exception:
                with self._state_lock:
                    self.sink_error_counters["console"] += 1
        if reason_code != "monitor_sink_failed":
            for sink in self.monitor_sinks:
                sink_started_ns = (
                    time.monotonic_ns() if sink_write_timing is not None else 0
                )
                try:
                    if isinstance(sink, MonitorEventSink):
                        result, phase_timing = sink._write_with_timing(degraded)
                        if sink_write_timing is not None:
                            sink_write_timing.record_monitor_phase_timing(phase_timing)
                        if result is None:
                            raise RuntimeError(
                                f"monitor publisher returned None for {degraded.event_type}"
                            )
                    else:
                        sink.write(degraded)
                except _MonitorEventPrepareError as exc:
                    if sink_write_timing is not None:
                        sink_write_timing.record_monitor_phase_timing(exc.timing)
                    with self._state_lock:
                        self.sink_error_counters["monitor"] += 1
                    logging.warning(
                        "[event] failed to emit sink.degraded to monitor: %s",
                        type(exc.error).__name__,
                    )
                except Exception as exc:
                    with self._state_lock:
                        self.sink_error_counters["monitor"] += 1
                    logging.warning(
                        "[event] failed to emit sink.degraded to monitor: %s",
                        type(exc).__name__,
                    )
                finally:
                    if sink_write_timing is not None:
                        sink_write_timing.record(
                            "monitor", time.monotonic_ns() - sink_started_ns
                        )


def emit_event(
    bot: Any,
    event: LiveEvent | Mapping[str, Any],
    *,
    require_enqueue: bool = False,
    **overrides: Any,
) -> Any:
    pipeline = getattr(bot, "_live_event_pipeline", None)
    if pipeline is None:
        pipeline = getattr(bot, "live_event_pipeline", None)
    if pipeline is None or not callable(getattr(pipeline, "emit", None)):
        return None
    try:
        return pipeline.emit(event, require_enqueue=require_enqueue, **overrides)
    except Exception as exc:
        logging.debug(
            "[event] failed to emit %s: %s",
            getattr(event, "event_type", event),
            exc,
        )
        return None
