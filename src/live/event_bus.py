from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import logging
import queue
import re
import threading
import time
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
LIVE_EVENT_DEBUG_PROFILES = (
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
    "candle": "candles",
    "emas": "ema",
    "ema-readiness": "ema",
    "ema_readiness": "ema",
    "remote": "remote_calls",
    "remote_call": "remote_calls",
    "remote-call": "remote_calls",
    "remote-calls": "remote_calls",
}


class EventTypes:
    BOT_STARTED = "bot.started"
    BOT_READY = "bot.ready"
    BOT_STARTUP_TIMING = "bot.startup_timing"
    BOT_STOPPING = "bot.stopping"
    BOT_SHUTDOWN_STAGE = "bot.shutdown.stage"
    BOT_STOPPED = "bot.stopped"
    HEALTH_SUMMARY = "health.summary"
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
    EMA_BUNDLE_STARTED = "ema.bundle.started"
    EMA_BUNDLE_COMPLETED = "ema.bundle.completed"
    EMA_FALLBACK_USED = "ema.fallback_used"
    EMA_UNAVAILABLE = "ema.unavailable"
    CANDLE_COVERAGE_CHECKED = "candle.coverage_checked"
    CANDLE_TAIL_PROJECTED = "candle.tail_projected"
    CACHE_LOAD_COMPLETED = "cache.load.completed"
    CACHE_FLUSH_COMPLETED = "cache.flush.completed"
    CACHE_WARMUP_DECISION = "cache.warmup_decision"
    EXCHANGE_TIME_SYNC = "exchange.time_sync"
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
    EXECUTION_CREATE_SUCCEEDED = "execution.create_succeeded"
    EXECUTION_CREATE_FAILED = "execution.create_failed"
    EXECUTION_CREATE_REJECTED = "execution.create_rejected"
    EXECUTION_CREATE_DEFERRED = "execution.create_deferred"
    EXECUTION_CREATE_SKIPPED = "execution.create_skipped"
    ENTRY_INITIAL_DISTANCE_GATE_BLOCKED = "entry.initial_distance_gate_blocked"
    ENTRY_INITIAL_DISTANCE_GATE_CLEARED = "entry.initial_distance_gate_cleared"
    ENTRY_MIN_EFFECTIVE_COST_BLOCKED = "entry.min_effective_cost_blocked"
    EXECUTION_CANCEL_SENT = "execution.cancel_sent"
    EXECUTION_CANCEL_SUCCEEDED = "execution.cancel_succeeded"
    EXECUTION_CANCEL_FAILED = "execution.cancel_failed"
    EXECUTION_CANCEL_AMBIGUOUS_TERMINAL = "execution.cancel_ambiguous_terminal"
    EXECUTION_AMBIGUOUS = "execution.ambiguous"
    EXECUTION_CONFIRMATION_REQUESTED = "execution.confirmation_requested"
    EXECUTION_CONFIRMATION_SATISFIED = "execution.confirmation_satisfied"
    EXECUTION_CONFIRMATION_TIMEOUT = "execution.confirmation_timeout"
    FILLS_REFRESH_SUMMARY = "fills.refresh_summary"
    FILL_INGESTED = "fill.ingested"
    POSITION_CHANGED = "position.changed"
    BALANCE_CHANGED = "balance.changed"
    RISK_MODE_CHANGED = "risk.mode_changed"
    HSL_TRANSITION = "hsl.transition"
    HSL_STATUS = "hsl.status"
    HSL_REPLAY_STARTED = "hsl.replay.started"
    HSL_REPLAY_PROGRESS = "hsl.replay.progress"
    HSL_REPLAY_COMPLETED = "hsl.replay.completed"
    HSL_REPLAY_FAILED = "hsl.replay.failed"
    HSL_RED_TRIGGERED = "hsl.red_triggered"
    HSL_COOLDOWN_STARTED = "hsl.cooldown_started"
    HSL_COOLDOWN_ENDED = "hsl.cooldown_ended"
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
    UNAVAILABLE = "unavailable"
    UNSTUCK = "unstuck"
    WARMUP = "warmup"
    WAVE = "wave"


class ReasonCodes:
    AUTHORITATIVE_CONFIRMATION = "authoritative_confirmation"
    AUTHORITATIVE_CONFIRMATION_TIMEOUT = "authoritative_confirmation_timeout"
    BALANCE_CHANGED = "balance_changed"
    CANDLE_DISK_FLUSH_COMPLETED = "candle_disk_flush_completed"
    CANDLE_DISK_LOAD_COMPLETED = "candle_disk_load_completed"
    EMA_FALLBACK_USED = "ema_fallback_used"
    EXCHANGE_ACKNOWLEDGED = "exchange_acknowledged"
    EXCHANGE_EXCEPTION = "exchange_exception"
    EXCHANGE_TIME_SYNC = "exchange_time_sync"
    EXCHANGE_TIME_SYNC_UNAVAILABLE = "exchange_time_sync_unavailable"
    EXECUTION_LOOP_ERROR_BURST = "execution_loop_error_burst"
    FILL_CACHE_DOCTOR_REPORT = "fill_cache_doctor_report"
    FILL_CACHE_QUARANTINED = "fill_cache_quarantined"
    FILL_CACHE_READY = "fill_cache_ready"
    FILL_CACHE_REBUILD_STARTED = "fill_cache_rebuild_started"
    INITIAL_ENTRY_DISTANCE_GATE = "initial_entry_distance_gate"
    LENGTH_MISMATCH = "length_mismatch"
    LIMIT_ORDER_CREATE_MARKET_DISTANCE = "limit_order_create_market_distance"
    LOW_BALANCE = "low_balance"
    MIN_EFFECTIVE_COST_BLOCKED = "min_effective_cost_blocked"
    NEW_FILL = "new_fill"
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
    RECENT_EXECUTION = "recent_execution"
    REMOTE_FETCH = "remote_fetch"
    RISK_ENTRY_COOLDOWN_POSITION_DELTA = "entry_cooldown_position_delta"
    REQUIRED_CANDLE_DISK_COVERAGE = "required_candle_disk_coverage"
    REQUIRED_EMA_UNAVAILABLE = "required_ema_unavailable"
    HSL_BALANCE_OVERRIDE_ACCOUNT_LEVEL_REPLAY_UNSAFE = (
        "hsl_balance_override_account_level_replay_unsafe"
    )
    RUST_OUTPUT_ACTIONS = "rust_output_actions"
    SINK_PIPELINE_CLOSING = "pipeline_closing"
    SNAPSHOT_SYMBOL_STATE = "snapshot_symbol_state"
    STARTUP_PHASE_READY = "startup_phase_ready"
    STAGED_REFRESH_PROGRESS = "staged_refresh_progress"
    STAGED_REFRESH_TIMING = "staged_refresh_timing"
    STATE_CHANGE_DETECTED = "state_change_detected"
    SUBMITTED_TO_EXCHANGE = "submitted_to_exchange"
    REALIZED_LOSS_GATE_BLOCKED = "realized_loss_gate_blocked"
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


PHASE1_EVENT_TYPES = {
    EventTypes.BOT_STARTED,
    EventTypes.BOT_READY,
    EventTypes.BOT_STARTUP_TIMING,
    EventTypes.BOT_STOPPING,
    EventTypes.BOT_SHUTDOWN_STAGE,
    EventTypes.BOT_STOPPED,
    EventTypes.HEALTH_SUMMARY,
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
    EventTypes.EMA_BUNDLE_STARTED,
    EventTypes.EMA_BUNDLE_COMPLETED,
    EventTypes.EMA_FALLBACK_USED,
    EventTypes.EMA_UNAVAILABLE,
    EventTypes.CANDLE_COVERAGE_CHECKED,
    EventTypes.CANDLE_TAIL_PROJECTED,
    EventTypes.CACHE_LOAD_COMPLETED,
    EventTypes.CACHE_FLUSH_COMPLETED,
    EventTypes.CACHE_WARMUP_DECISION,
    EventTypes.EXCHANGE_TIME_SYNC,
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
    EventTypes.EXECUTION_CREATE_SUCCEEDED,
    EventTypes.EXECUTION_CREATE_FAILED,
    EventTypes.EXECUTION_CREATE_REJECTED,
    EventTypes.EXECUTION_CREATE_DEFERRED,
    EventTypes.EXECUTION_CREATE_SKIPPED,
    EventTypes.ENTRY_INITIAL_DISTANCE_GATE_BLOCKED,
    EventTypes.ENTRY_INITIAL_DISTANCE_GATE_CLEARED,
    EventTypes.ENTRY_MIN_EFFECTIVE_COST_BLOCKED,
    EventTypes.EXECUTION_CANCEL_SENT,
    EventTypes.EXECUTION_CANCEL_SUCCEEDED,
    EventTypes.EXECUTION_CANCEL_FAILED,
    EventTypes.EXECUTION_CANCEL_AMBIGUOUS_TERMINAL,
    EventTypes.EXECUTION_AMBIGUOUS,
    EventTypes.EXECUTION_CONFIRMATION_REQUESTED,
    EventTypes.EXECUTION_CONFIRMATION_SATISFIED,
    EventTypes.EXECUTION_CONFIRMATION_TIMEOUT,
    EventTypes.FILLS_REFRESH_SUMMARY,
    EventTypes.FILL_INGESTED,
    EventTypes.POSITION_CHANGED,
    EventTypes.BALANCE_CHANGED,
    EventTypes.RISK_MODE_CHANGED,
    EventTypes.HSL_TRANSITION,
    EventTypes.HSL_STATUS,
    EventTypes.HSL_REPLAY_STARTED,
    EventTypes.HSL_REPLAY_PROGRESS,
    EventTypes.HSL_REPLAY_COMPLETED,
    EventTypes.HSL_REPLAY_FAILED,
    EventTypes.HSL_RED_TRIGGERED,
    EventTypes.HSL_COOLDOWN_STARTED,
    EventTypes.HSL_COOLDOWN_ENDED,
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
    EventTypes.BOT_READY: EventRoute(console=True, text=True),
    EventTypes.BOT_STARTUP_TIMING: EventRoute(console=False, text=False),
    EventTypes.BOT_STOPPING: EventRoute(console=True, text=True),
    EventTypes.BOT_SHUTDOWN_STAGE: EventRoute(console=True, text=True),
    EventTypes.BOT_STOPPED: EventRoute(console=True, text=True),
    EventTypes.HEALTH_SUMMARY: EventRoute(console=False, text=False),
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
    EventTypes.FORAGER_SELECTION: EventRoute(console=False, text=False),
    EventTypes.FORAGER_FEATURE_UNAVAILABLE: EventRoute(console=False, text=False),
    EventTypes.EMA_BUNDLE_STARTED: EventRoute(console=False, text=False),
    EventTypes.EMA_BUNDLE_COMPLETED: EventRoute(console=False, text=False),
    EventTypes.EMA_FALLBACK_USED: EventRoute(console=False, text=False),
    EventTypes.EMA_UNAVAILABLE: EventRoute(console=False, text=False),
    EventTypes.CANDLE_COVERAGE_CHECKED: EventRoute(console=False, text=False),
    EventTypes.CANDLE_TAIL_PROJECTED: EventRoute(console=False, text=False),
    EventTypes.CACHE_LOAD_COMPLETED: EventRoute(console=False, text=False),
    EventTypes.CACHE_FLUSH_COMPLETED: EventRoute(console=False, text=False),
    EventTypes.CACHE_WARMUP_DECISION: EventRoute(console=False, text=False),
    EventTypes.EXCHANGE_TIME_SYNC: EventRoute(console=False, text=False),
    EventTypes.REMOTE_CALL_STARTED: EventRoute(console=False),
    EventTypes.REMOTE_CALL_SUCCEEDED: EventRoute(console=False),
    EventTypes.REMOTE_CALL_FAILED: EventRoute(console=False),
    EventTypes.REMOTE_CALL_THROTTLED: EventRoute(console=False),
    EventTypes.RUST_ORCHESTRATOR_CALLED: EventRoute(console=False),
    EventTypes.RUST_ORCHESTRATOR_RETURNED: EventRoute(
        console=True, text=True, throttle_interval_ms=60_000
    ),
    EventTypes.ACTION_PLANNED: EventRoute(console=False, text=False),
    EventTypes.ORDER_WAVE_STARTED: EventRoute(console=False),
    EventTypes.ORDER_WAVE_COMPLETED: EventRoute(console=True, text=True),
    EventTypes.EXECUTION_CREATE_SENT: EventRoute(console=False),
    EventTypes.EXECUTION_CREATE_SUCCEEDED: EventRoute(console=True, text=True),
    EventTypes.EXECUTION_CREATE_FAILED: EventRoute(console=True, text=True),
    EventTypes.EXECUTION_CREATE_REJECTED: EventRoute(console=True, text=True),
    EventTypes.EXECUTION_CREATE_DEFERRED: EventRoute(console=False, text=False),
    EventTypes.EXECUTION_CREATE_SKIPPED: EventRoute(console=False, text=False),
    EventTypes.ENTRY_INITIAL_DISTANCE_GATE_BLOCKED: EventRoute(
        console=False, text=False
    ),
    EventTypes.ENTRY_INITIAL_DISTANCE_GATE_CLEARED: EventRoute(
        console=False, text=False
    ),
    EventTypes.ENTRY_MIN_EFFECTIVE_COST_BLOCKED: EventRoute(console=False, text=False),
    EventTypes.EXECUTION_CANCEL_SENT: EventRoute(console=False),
    EventTypes.EXECUTION_CANCEL_SUCCEEDED: EventRoute(console=True, text=True),
    EventTypes.EXECUTION_CANCEL_FAILED: EventRoute(console=True, text=True),
    EventTypes.EXECUTION_CANCEL_AMBIGUOUS_TERMINAL: EventRoute(console=True, text=True),
    EventTypes.EXECUTION_AMBIGUOUS: EventRoute(console=True, text=True),
    EventTypes.EXECUTION_CONFIRMATION_REQUESTED: EventRoute(console=False),
    EventTypes.EXECUTION_CONFIRMATION_SATISFIED: EventRoute(console=True, text=True),
    EventTypes.EXECUTION_CONFIRMATION_TIMEOUT: EventRoute(console=True, text=True),
    EventTypes.FILLS_REFRESH_SUMMARY: EventRoute(console=False, text=False),
    EventTypes.FILL_INGESTED: EventRoute(console=False, text=False),
    EventTypes.POSITION_CHANGED: EventRoute(console=False, text=False),
    EventTypes.BALANCE_CHANGED: EventRoute(console=False, text=False),
    EventTypes.RISK_MODE_CHANGED: EventRoute(console=False, text=False),
    EventTypes.HSL_TRANSITION: EventRoute(console=False, text=False),
    EventTypes.HSL_STATUS: EventRoute(console=False, text=False),
    EventTypes.HSL_REPLAY_STARTED: EventRoute(console=False, text=False),
    EventTypes.HSL_REPLAY_PROGRESS: EventRoute(console=False, text=False),
    EventTypes.HSL_REPLAY_COMPLETED: EventRoute(console=False, text=False),
    EventTypes.HSL_REPLAY_FAILED: EventRoute(console=False, text=False),
    EventTypes.HSL_RED_TRIGGERED: EventRoute(console=False, text=False),
    EventTypes.HSL_COOLDOWN_STARTED: EventRoute(console=False, text=False),
    EventTypes.HSL_COOLDOWN_ENDED: EventRoute(console=False, text=False),
    EventTypes.UNSTUCK_STATUS: EventRoute(console=False, text=False),
    EventTypes.UNSTUCK_SELECTION: EventRoute(console=False, text=False),
    EventTypes.RISK_ENTRY_COOLDOWN_DELTA_ANCHORED: EventRoute(
        console=False, text=False
    ),
    EventTypes.REALIZED_LOSS_GATE_BLOCKED: EventRoute(console=False, text=False),
    EventTypes.SINK_DEGRADED: EventRoute(console=True, text=True),
}


class LiveEventSink(Protocol):
    def write(self, event: LiveEvent) -> Any:
        ...


class MonitorEventSink:
    def __init__(self, publisher: Any):
        self.publisher = publisher

    def write(self, event: LiveEvent) -> Any:
        kind, tags, payload = event.to_monitor_event()
        result = self.publisher.record_event(
            kind,
            tags,
            payload,
            ts=event.ts_ms,
            symbol=event.symbol,
            pside=event.pside,
        )
        if result is None:
            raise RuntimeError(f"monitor publisher returned None for {kind}")
        return result


class ListEventSink:
    def __init__(self) -> None:
        self.events: list[LiveEvent] = []

    def write(self, event: LiveEvent) -> LiveEvent:
        self.events.append(event)
        return event


_CONSOLE_EVENT_TAGS = {
    EventTypes.BOT_STARTED: "bot",
    EventTypes.BOT_READY: "bot",
    EventTypes.BOT_STOPPING: "bot",
    EventTypes.BOT_SHUTDOWN_STAGE: "shutdown",
    EventTypes.BOT_STOPPED: "bot",
    EventTypes.CYCLE_STARTED: "cycle",
    EventTypes.CYCLE_COMPLETED: "cycle",
    EventTypes.CYCLE_DEGRADED: "cycle",
    EventTypes.PLANNING_UNAVAILABLE: "gate",
    EventTypes.RUST_ORCHESTRATOR_RETURNED: "rust",
    EventTypes.ORDER_WAVE_COMPLETED: "execute",
    EventTypes.EXECUTION_CREATE_SUCCEEDED: "order",
    EventTypes.EXECUTION_CREATE_FAILED: "order",
    EventTypes.EXECUTION_CREATE_REJECTED: "order",
    EventTypes.EXECUTION_CREATE_DEFERRED: "gate",
    EventTypes.EXECUTION_CREATE_SKIPPED: "gate",
    EventTypes.EXECUTION_CANCEL_SUCCEEDED: "order",
    EventTypes.EXECUTION_CANCEL_FAILED: "order",
    EventTypes.EXECUTION_CANCEL_AMBIGUOUS_TERMINAL: "order",
    EventTypes.EXECUTION_AMBIGUOUS: "order",
    EventTypes.EXECUTION_CONFIRMATION_SATISFIED: "execute",
    EventTypes.EXECUTION_CONFIRMATION_TIMEOUT: "execute",
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


def _console_data_summary(event: LiveEvent) -> list[str]:
    if event.event_type == EventTypes.ORDER_WAVE_COMPLETED:
        return _console_order_wave_summary(event)
    if event.event_type in {
        EventTypes.EXECUTION_CREATE_DEFERRED,
        EventTypes.EXECUTION_CREATE_SKIPPED,
    }:
        return _console_create_filter_summary(event)
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
    return []


def format_console_event(event: LiveEvent) -> str:
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
        self._queue: queue.Queue[LiveEvent | None] = queue.Queue(maxsize=queue_maxsize)
        self._stop = threading.Event()
        self._enqueue_lock = threading.RLock()
        self._state_lock = threading.RLock()
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

    def health_snapshot(self) -> dict[str, Any]:
        """Return best-effort operational counters for periodic health events."""
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
        }
        return {key: value for key, value in snapshot.items() if value is not None}

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
            and self._should_emit_throttled_sink("console", live_event, route)
        ):
            self._write_sink("console", self.console_sink, live_event)
        if (
            route.text
            and self.text_sink is not None
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
                        self._queue.put_nowait(live_event)
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
                route = self.route_for(item)
                if route.structured:
                    for sink in self.structured_sinks:
                        self._write_sink("structured", sink, item)
                if route.monitor:
                    for sink in self.monitor_sinks:
                        self._write_sink("monitor", sink, item)
            finally:
                self._queue.task_done()

    def _write_sink(self, name: str, sink: LiveEventSink, event: LiveEvent) -> Any:
        try:
            return sink.write(event)
        except Exception as exc:
            with self._state_lock:
                self.sink_error_counters[name] += 1
            self._record_degraded(
                reason_code=sink_failed_reason_code(name),
                message=f"{name} sink failed: {type(exc).__name__}",
                data={"sink": name, "error_type": type(exc).__name__, "error": str(exc)},
            )
            return None

    def _record_degraded(
        self,
        *,
        reason_code: str,
        message: str,
        data: Mapping[str, Any] | None = None,
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
                try:
                    sink.write(degraded)
                except Exception as exc:
                    with self._state_lock:
                        self.sink_error_counters["monitor"] += 1
                    logging.warning(
                        "[event] failed to emit sink.degraded to monitor: %s",
                        type(exc).__name__,
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
