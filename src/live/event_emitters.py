from __future__ import annotations

import logging
import hashlib
import math
import re
from collections import Counter
from typing import Any

from live.event_bus import (
    EventTags,
    EventTypes,
    LiveEvent,
    ReasonCodes,
    authoritative_reason_code,
    emit_event,
    live_event_debug_profile_enabled,
    startup_phase_readiness_contract,
    utc_ms,
)


def current_live_event_cycle_id(bot: Any) -> str | None:
    return getattr(bot, "_live_event_current_cycle_id", None)


def set_live_event_context_ids(bot: Any, **kwargs: str | None) -> None:
    pipeline = getattr(bot, "_live_event_pipeline", None)
    if pipeline is None or not callable(getattr(pipeline, "with_context_ids", None)):
        return
    try:
        pipeline.with_context_ids(**kwargs)
    except Exception as exc:
        logging.debug("[event] failed updating live event context: %s", exc)


def next_live_event_remote_call_id(bot: Any, prefix: str = "rc") -> str:
    bot._live_event_remote_call_seq = (
        int(getattr(bot, "_live_event_remote_call_seq", 0) or 0) + 1
    )
    return f"{prefix}_{int(bot._live_event_remote_call_seq)}"


_REMOTE_CALL_DEFAULT_MAP_MAX = 2048
_REMOTE_CALL_PER_KEY_MAX = 8
_REMOTE_CALL_NONTERMINAL_STAGES = {"progress"}
_SENSITIVE_VALUE_RE = re.compile(
    r"(?i)\b(api[-_]?key|apikey|secret|token|signature|password|passphrase|authorization|auth|cookie)"
    r"(\s*(?:[:=]|\s)\s*)([^\s,;&]+)"
)
_AUTH_HEADER_RE = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]+")
_EXCHANGE_CONFIG_EVENT_SYMBOL_RE = re.compile(r"[A-Za-z0-9_./:-]{1,160}")
_EXCHANGE_CONFIG_EVENT_RESPONSE_CODE_RE = re.compile(r"-?[0-9]{1,12}")
_EXCHANGE_CONFIG_EVENT_ERROR_TYPE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,79}")
_EXCHANGE_CONFIG_EVENT_OUTCOMES = {"confirmed", "unchanged", "failed"}
_LIVE_EVENT_LEVELS = {"debug", "info", "warning", "error"}
_MEMORY_SNAPSHOT_MAX_BYTES = (1 << 63) - 1
_MEMORY_SNAPSHOT_MAX_COUNT = 1_000_000_000
_MEMORY_SNAPSHOT_MAX_DELTA_PCT = 1_000_000.0
_MEMORY_SNAPSHOT_SYMBOL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,95}")
_MEMORY_SNAPSHOT_TIMEFRAME_RE = re.compile(r"[1-9][0-9]{0,5}[smhdwM]")
_MEMORY_SNAPSHOT_TASK_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,95}")
_MEMORY_SNAPSHOT_SENSITIVE_TASK_RE = re.compile(
    r"(?i)(?:api(?:_?key)?|secret|token|password|authorization|cookie)"
)
_EXECUTION_LOOP_ERROR_STATUS_RE = re.compile(r"[0-9]{1,3}")
_EXECUTION_LOOP_ERROR_CODE_RE = re.compile(r"-?[A-Za-z0-9][A-Za-z0-9_-]{0,47}")
_EXECUTION_LOOP_ERROR_ENDPOINT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,47}")
_EXCHANGE_TIME_SYNC_SOURCE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,79}")
_EXCHANGE_TIME_SYNC_CLIENT_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]{0,15}:[A-Za-z0-9?.+\-]+->[A-Za-z0-9?.+\-]+"
    r"|[A-Za-z_][A-Za-z0-9_]{0,15}:[A-Za-z_][A-Za-z0-9_]{0,63}"
)
_EXCHANGE_TIME_SYNC_ERROR_TYPE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,79}")
_EXCHANGE_TIME_SYNC_CLIENT_SAMPLE_LIMIT = 8


def _sanitize_remote_text(value: Any, *, max_len: int = 500) -> str:
    text = str(value)
    text = _SENSITIVE_VALUE_RE.sub(r"\1\2[redacted]", text)
    text = _AUTH_HEADER_RE.sub(r"\1 [redacted]", text)
    if len(text) > max_len:
        text = f"{text[:max_len]}...<truncated>"
    return text


def _sanitize_remote_url(value: Any) -> tuple[str, str | None]:
    raw = str(value)
    raw_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return "[redacted-url]", raw_hash


def _sanitize_remote_fetch_payload(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    data.pop("error", None)
    data.pop("error_repr", None)
    if data.get("url") is not None:
        url, url_hash = _sanitize_remote_url(data["url"])
        data["url"] = url
        if url_hash is not None:
            data["url_hash"] = url_hash
    return data


def _bounded_exchange_config_event_field(
    value: Any,
    pattern: re.Pattern[str],
) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if pattern.fullmatch(text) else None


def _bounded_execution_loop_error_field(
    value: Any, pattern: re.Pattern[str], *, fallback: str = "-"
) -> str:
    if value is None:
        return fallback
    try:
        text = str(value).strip()
    except Exception:
        return fallback
    return text if pattern.fullmatch(text) else fallback


def _bounded_exchange_time_sync_field(
    value: Any, pattern: re.Pattern[str], *, fallback: str = "unknown"
) -> str:
    try:
        text = str(value)
    except Exception:
        return fallback
    return text if pattern.fullmatch(text) else fallback


def _remote_fetch_payload_key(payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        payload.get("kind"),
        payload.get("symbol"),
        payload.get("tf") or payload.get("timeframe"),
        payload.get("since_ts"),
        payload.get("url"),
    )


def _mapping_key_sample(value: Any, *, limit: int = 32) -> list[str]:
    if not isinstance(value, dict):
        return []
    return sorted(str(key) for key in value)[:limit]


def _candle_debug_payload(
    data: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
    timeframe_ms: int | None = None,
    limit: int = 32,
) -> dict[str, Any]:
    debug: dict[str, Any] = {"data_keys": _mapping_key_sample(data, limit=limit)}
    if isinstance(context, dict):
        debug["context_keys"] = _mapping_key_sample(context, limit=limit)
    if isinstance(report, dict):
        debug["report_keys"] = _mapping_key_sample(report, limit=limit)
    if timeframe_ms is not None:
        debug["timeframe_ms"] = int(max(1, int(timeframe_ms)))
    for key in (
        "context",
        "timeframe",
        "coverage_ok",
        "missing_span_count",
        "missing_candles",
        "loaded_rows",
        "tail_gap_age_ms",
        "tail_gap_candles",
        "max_tail_gap_ms",
    ):
        if key in data:
            debug[key] = data.get(key)
    start_ts = _safe_int(data.get("start_ts"))
    end_ts = _safe_int(data.get("end_ts"))
    if start_ts is not None and end_ts is not None:
        debug["window_ms"] = int(max(0, int(end_ts) - int(start_ts)))
    if isinstance(report, dict):
        raw_spans = report.get("missing_spans") or []
        try:
            debug["raw_missing_span_count"] = len(raw_spans)
        except TypeError:
            debug["raw_missing_span_count"] = 0
    return debug


def _remote_call_debug_payload(
    data: dict[str, Any],
    *,
    matched_start: bool | None = None,
    limit: int = 32,
) -> dict[str, Any]:
    debug: dict[str, Any] = {
        "data_keys": _mapping_key_sample(data, limit=limit),
    }
    if isinstance(data.get("params"), dict):
        debug["param_keys"] = _mapping_key_sample(data.get("params"), limit=limit)
    if matched_start is not None:
        debug["matched_start"] = bool(matched_start)
    for key in (
        "kind",
        "surface",
        "stage",
        "tf",
        "timeframe",
        "since_ts",
        "end_ts",
        "limit",
        "rows",
        "first_ts",
        "last_ts",
        "elapsed_ms",
        "state_epoch",
        "url_hash",
        "error_type",
    ):
        if key in data:
            debug[key] = data.get(key)
    if isinstance(data.get("pending_confirmations"), list):
        debug["pending_confirmation_count"] = len(data["pending_confirmations"])
    return debug


def _bounded_str_values(value: Any, *, limit: int = 32) -> list[str]:
    try:
        values = sorted(str(item) for item in value)
    except TypeError:
        return [str(value)]
    return values[:limit]


def _execution_wave_counts(wave: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in (
        "planned_cancel",
        "planned_create",
        "cancel_posted",
        "create_posted",
        "skipped_cancel",
        "deferred_create",
        "skipped_create",
    ):
        if key in wave:
            try:
                count = int(wave.get(key) or 0)
            except (TypeError, ValueError):
                continue
            if count:
                counts[key] = count
    return counts


def _execution_debug_payload(
    data: dict[str, Any],
    *,
    event_type: str,
    action: str | None = None,
    order: dict | None = None,
    result: dict | BaseException | None = None,
    extra: dict | None = None,
    wave: dict | None = None,
    surfaces: Any = None,
    confirmations: dict | None = None,
    fresh_surfaces: Any = None,
    limit: int = 32,
) -> dict[str, Any]:
    debug: dict[str, Any] = {
        "event_type": str(event_type),
        "data_keys": _mapping_key_sample(data, limit=limit),
    }
    if action is not None:
        debug["action"] = str(action)
    if isinstance(order, dict):
        debug["order_keys"] = _mapping_key_sample(order, limit=limit)
        debug["has_client_order_id"] = bool(
            order.get("custom_id") or order.get("clientOrderId")
        )
        debug["has_exchange_order_id"] = bool(order.get("id") or order.get("order_id"))
    if isinstance(result, dict):
        debug["result_keys"] = _mapping_key_sample(result, limit=limit)
        debug["has_result_order_id"] = bool(result.get("id") or result.get("order_id"))
        debug["has_result_client_order_id"] = bool(
            result.get("custom_id") or result.get("clientOrderId")
        )
        if result.get("status") is not None:
            debug["result_status"] = str(result.get("status"))
    elif isinstance(result, BaseException):
        debug["result_error_type"] = type(result).__name__
    if isinstance(extra, dict):
        debug["extra_keys"] = _mapping_key_sample(extra, limit=limit)
    if isinstance(wave, dict):
        debug["wave_keys"] = _mapping_key_sample(wave, limit=limit)
        counts = _execution_wave_counts(wave)
        if counts:
            debug["wave_counts"] = counts
    if surfaces is not None:
        debug["surfaces"] = _bounded_str_values(surfaces, limit=limit)
    if isinstance(confirmations, dict):
        debug["confirmation_surfaces"] = _mapping_key_sample(confirmations, limit=limit)
        debug["confirmation_count"] = len(confirmations)
    if fresh_surfaces is not None:
        debug["fresh_surfaces"] = _bounded_str_values(fresh_surfaces, limit=limit)
    return {key: value for key, value in debug.items() if value not in (None, [], {})}


def _add_execution_debug_profile(
    bot: Any,
    data: dict[str, Any],
    *,
    event_type: str,
    action: str | None = None,
    order: dict | None = None,
    result: dict | BaseException | None = None,
    extra: dict | None = None,
    wave: dict | None = None,
    surfaces: Any = None,
    confirmations: dict | None = None,
    fresh_surfaces: Any = None,
) -> None:
    if not live_event_debug_profile_enabled(bot, "execution"):
        return
    try:
        data["debug_profile"] = "execution"
        data["debug"] = _execution_debug_payload(
            data,
            event_type=event_type,
            action=action,
            order=order,
            result=result,
            extra=extra,
            wave=wave,
            surfaces=surfaces,
            confirmations=confirmations,
            fresh_surfaces=fresh_surfaces,
        )
    except Exception as exc:
        data.pop("debug_profile", None)
        data.pop("debug", None)
        logging.debug(
            "[event] failed to build execution debug payload type=%s: %s",
            event_type,
            exc,
        )


def _remote_call_map_max(bot: Any) -> int:
    try:
        raw = getattr(bot, "_live_event_remote_call_map_max", None)
        if raw is None:
            return _REMOTE_CALL_DEFAULT_MAP_MAX
        return max(1, int(raw))
    except Exception:
        return _REMOTE_CALL_DEFAULT_MAP_MAX


def _remote_call_map_add(
    bot: Any,
    call_map: dict[Any, Any],
    key: tuple[Any, ...],
    value: tuple[str, str | None],
) -> None:
    existing = call_map.get(key)
    if existing is None:
        entries: list[tuple[str, str | None]] = []
    elif isinstance(existing, list):
        entries = list(existing)
    else:
        entries = [existing]
    entries.append(value)
    call_map[key] = entries[-_REMOTE_CALL_PER_KEY_MAX:]
    while len(call_map) > _remote_call_map_max(bot):
        oldest_key = next(iter(call_map))
        call_map.pop(oldest_key, None)


def _remote_call_map_pop(
    call_map: dict[Any, Any],
    key: tuple[Any, ...],
) -> tuple[str | None, str | None, bool]:
    existing = call_map.get(key)
    if existing is None:
        return None, None, False
    if isinstance(existing, list):
        if not existing:
            call_map.pop(key, None)
            return None, None, False
        remote_call_id, remote_call_group_id = existing.pop(0)
        if existing:
            call_map[key] = existing
        else:
            call_map.pop(key, None)
        return remote_call_id, remote_call_group_id, True
    call_map.pop(key, None)
    remote_call_id, remote_call_group_id = existing
    return remote_call_id, remote_call_group_id, True


def emit_candle_remote_fetch_event(bot: Any, payload: dict[str, Any]) -> Any:
    """Translate CandlestickManager remote-fetch callbacks into LiveEvents."""
    if not isinstance(payload, dict):
        return None
    stage = str(payload.get("stage") or "").lower()
    if not stage:
        return None
    if stage in _REMOTE_CALL_NONTERMINAL_STAGES:
        return None
    key = _remote_fetch_payload_key(payload)
    call_map = getattr(bot, "_live_event_remote_call_ids", None)
    if not isinstance(call_map, dict):
        call_map = {}
        bot._live_event_remote_call_ids = call_map
    if stage == "start":
        remote_call_id = next_live_event_remote_call_id(bot, "rcc")
        cycle_id = current_live_event_cycle_id(bot)
        remote_call_group_id = f"{cycle_id}:candles" if cycle_id else None
        _remote_call_map_add(bot, call_map, key, (remote_call_id, remote_call_group_id))
        event_type = EventTypes.REMOTE_CALL_STARTED
        status = "started"
        level = "debug"
    else:
        remote_call_id, remote_call_group_id, matched_start = _remote_call_map_pop(
            call_map, key
        )
        cycle_id = current_live_event_cycle_id(bot)
        if stage == "error":
            event_type = EventTypes.REMOTE_CALL_FAILED
            status = "failed"
            level = "warning"
        elif stage in {"throttled", "rate_limited"}:
            event_type = EventTypes.REMOTE_CALL_THROTTLED
            status = "deferred"
            level = "debug"
        else:
            event_type = EventTypes.REMOTE_CALL_SUCCEEDED
            status = "skipped" if stage in {"not_found", "missing"} else "succeeded"
            level = "debug"
    data = _sanitize_remote_fetch_payload(payload)
    if stage != "start" and not matched_start:
        data["orphan_result"] = True
    if live_event_debug_profile_enabled(bot, "remote_calls"):
        data["debug_profile"] = "remote_calls"
        data["debug"] = _remote_call_debug_payload(
            data,
            matched_start=matched_start if stage != "start" else None,
        )
    return bot._emit_live_event(
        event_type,
        level=level,
        component="candles.remote_fetch",
        tags=(EventTags.REMOTE_CALL, EventTags.CANDLE),
        cycle_id=cycle_id,
        remote_call_id=remote_call_id,
        remote_call_group_id=remote_call_group_id,
        symbol=str(payload.get("symbol")) if payload.get("symbol") is not None else None,
        status=status,
        reason_code=str(payload.get("kind") or ReasonCodes.REMOTE_FETCH),
        data=data,
    )


def _authoritative_result_summary(surface: str, result: Any) -> dict[str, Any]:
    if surface == "balance":
        try:
            raw, normalized = result
            return {
                "has_raw_payload": raw is not None,
                "balance": round(float(normalized), 12),
            }
        except Exception:
            return {"result_type": type(result).__name__}
    if surface == "positions":
        try:
            _raw, positions = result
            return {"count": len(positions or [])}
        except Exception:
            return {"result_type": type(result).__name__}
    if surface == "open_orders":
        try:
            return {"count": len(result or [])}
        except Exception:
            return {"result_type": type(result).__name__}
    if surface == "fills":
        return {"ok": bool(result)}
    return {"result_type": type(result).__name__}


def _emit_authoritative_remote_call_event_unchecked(
    bot: Any,
    *,
    surface: str,
    stage: str,
    started_ms: int,
    elapsed_ms: int | None = None,
    remote_call_id: str | None = None,
    result: Any = None,
    error: BaseException | None = None,
) -> str | None:
    """Emit remote-call events for staged authoritative account-state fetches."""
    emit = getattr(bot, "_emit_live_event", None)
    if not callable(emit):
        return remote_call_id
    stage = str(stage or "").lower()
    surface = str(surface or "unknown")
    cycle_id = current_live_event_cycle_id(bot)
    authoritative_epoch = int(getattr(bot, "_authoritative_refresh_epoch", 0) or 0)
    remote_call_group_id = (
        f"{cycle_id}:authoritative" if cycle_id else f"auth_{authoritative_epoch}:authoritative"
    )
    if stage == "start":
        remote_call_id = next_live_event_remote_call_id(bot, "rca")
        event_type = EventTypes.REMOTE_CALL_STARTED
        status = "started"
        level = "debug"
    elif error is not None:
        event_type = EventTypes.REMOTE_CALL_FAILED
        status = "failed"
        level = "warning"
    else:
        event_type = EventTypes.REMOTE_CALL_SUCCEEDED
        status = "succeeded"
        level = "debug"
    data: dict[str, Any] = {
        "kind": "authoritative_state_fetch",
        "surface": surface,
        "stage": stage,
        "started_ms": int(started_ms),
        "state_epoch": authoritative_epoch,
        "pending_confirmations": sorted(
            str(item)
            for item in (getattr(bot, "_authoritative_pending_confirmations", {}) or {})
        ),
    }
    if elapsed_ms is not None:
        data["elapsed_ms"] = int(elapsed_ms)
    if error is not None:
        data["error_type"] = type(error).__name__
        data["error"] = _sanitize_remote_text(error, max_len=500)
        data["error_repr"] = _sanitize_remote_text(repr(error), max_len=500)
    elif stage != "start":
        data.update(_authoritative_result_summary(surface, result))
    if live_event_debug_profile_enabled(bot, "remote_calls"):
        data["debug_profile"] = "remote_calls"
        data["debug"] = _remote_call_debug_payload(data)
    emit(
        event_type,
        level=level,
        component="state.authoritative_fetch",
        tags=(EventTags.REMOTE_CALL, EventTags.STATE, EventTags.AUTHORITATIVE, surface),
        cycle_id=cycle_id,
        remote_call_id=remote_call_id,
        remote_call_group_id=remote_call_group_id,
        status=status,
        reason_code=authoritative_reason_code(surface),
        data=data,
    )
    return remote_call_id


def emit_authoritative_remote_call_event(
    bot: Any,
    *,
    surface: str,
    stage: str,
    started_ms: int,
    elapsed_ms: int | None = None,
    remote_call_id: str | None = None,
    result: Any = None,
    error: BaseException | None = None,
) -> str | None:
    """Best-effort authoritative fetch event emission.

    This function is called from account-critical refresh paths. Event
    construction, sanitization, and sink emission must never prevent the
    underlying exchange fetch from running or mask its original exception.
    """
    try:
        return _emit_authoritative_remote_call_event_unchecked(
            bot,
            surface=surface,
            stage=stage,
            started_ms=started_ms,
            elapsed_ms=elapsed_ms,
            remote_call_id=remote_call_id,
            result=result,
            error=error,
        )
    except Exception as exc:
        logging.debug(
            "[event] failed to emit authoritative remote call event surface=%s stage=%s: %s",
            surface,
            stage,
            exc,
        )
        return remote_call_id


def _emit_exchange_time_sync_event_unchecked(
    bot: Any,
    *,
    source: str,
    error: BaseException,
    synced_clients: list[str] | tuple[str, ...],
    failed_clients: list[str] | tuple[str, ...],
    hook_available: bool,
) -> None:
    synced = [
        _bounded_exchange_time_sync_field(item, _EXCHANGE_TIME_SYNC_CLIENT_RE)
        for item in list(synced_clients or [])[:_EXCHANGE_TIME_SYNC_CLIENT_SAMPLE_LIMIT]
    ]
    failed = [
        _bounded_exchange_time_sync_field(item, _EXCHANGE_TIME_SYNC_CLIENT_RE)
        for item in list(failed_clients or [])[:_EXCHANGE_TIME_SYNC_CLIENT_SAMPLE_LIMIT]
    ]
    recovered = bool(synced)
    if not hook_available:
        status = "skipped"
        reason_code = ReasonCodes.EXCHANGE_TIME_SYNC_UNAVAILABLE
        level = "warning"
    elif recovered and not failed:
        status = "succeeded"
        reason_code = ReasonCodes.EXCHANGE_TIME_SYNC
        level = "debug"
    else:
        status = "degraded"
        reason_code = ReasonCodes.EXCHANGE_TIME_SYNC
        level = "warning"
    _safe_emit(
        bot,
        EventTypes.EXCHANGE_TIME_SYNC,
        level=level,
        component="exchange.time_sync",
        tags=(EventTags.EXCHANGE, EventTags.TIME_SYNC),
        cycle_id=current_live_event_cycle_id(bot),
        status=status,
        reason_code=reason_code,
        data={
            "source": _bounded_exchange_time_sync_field(
                source, _EXCHANGE_TIME_SYNC_SOURCE_RE
            ),
            "error_type": _bounded_exchange_time_sync_field(
                type(error).__name__, _EXCHANGE_TIME_SYNC_ERROR_TYPE_RE
            ),
            "hook_available": bool(hook_available),
            "recovered": recovered,
            "synced_clients": synced,
            "failed_clients": failed,
            "synced_count": len(synced_clients or []),
            "failed_count": len(failed_clients or []),
        },
    )


def emit_exchange_time_sync_event(bot: Any, *args: Any, **kwargs: Any) -> None:
    """Best-effort structured visibility for CCXT timestamp/nonce recovery."""
    try:
        _emit_exchange_time_sync_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug("[event] failed to emit exchange time-sync event: %s", exc)


def _emit_websocket_reconnect_event_unchecked(
    bot: Any,
    *,
    reconnect_no: int,
    retry_delay_s: float,
    reason: str,
    warning_visible: bool,
    traceback_emitted: bool,
    exc: BaseException | None = None,
    rate_limited: bool = False,
) -> None:
    reason_value = str(reason or "connection_lost")
    if reason_value not in {"connection_lost", "rate_limited", "time_sync"}:
        reason_value = "other"
    data = {
        "reconnect_no": max(0, int(reconnect_no or 0)),
        "retry_delay_ms": max(0, int(round(float(retry_delay_s or 0.0) * 1000))),
        "reason": reason_value,
        "rate_limited": bool(rate_limited),
        "warning_visible": bool(warning_visible),
        "traceback_emitted": bool(traceback_emitted),
    }
    if exc is not None:
        data["error_type"] = type(exc).__name__
    _safe_emit(
        bot,
        EventTypes.WEBSOCKET_RECONNECT,
        level="warning" if warning_visible else "debug",
        component="exchange.websocket",
        tags=(EventTags.EXCHANGE, EventTags.WEBSOCKET),
        cycle_id=current_live_event_cycle_id(bot),
        status="degraded",
        reason_code=ReasonCodes.WEBSOCKET_RECONNECT,
        data=data,
    )


def emit_websocket_reconnect_event(bot: Any, *args: Any, **kwargs: Any) -> None:
    """Best-effort bounded visibility for websocket reconnect attempts."""
    try:
        _emit_websocket_reconnect_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug("[event] failed to emit websocket reconnect event: %s", exc)


def _emit_exchange_config_refresh_event_unchecked(
    bot: Any,
    *,
    context: str,
    operation: str,
    status: str,
    started_ms: int | None = None,
    elapsed_ms: int | None = None,
    error: BaseException | None = None,
    symbol: str | None = None,
    outcome: str | None = None,
    response_code: str | int | None = None,
    error_type: str | None = None,
    level: str | None = None,
) -> None:
    status_value = str(status or "unknown")
    failed = status_value == "failed"
    default_level = "warning" if failed else "debug"
    symbol_value = _bounded_exchange_config_event_field(
        symbol, _EXCHANGE_CONFIG_EVENT_SYMBOL_RE
    )
    outcome_value = (
        outcome
        if isinstance(outcome, str) and outcome in _EXCHANGE_CONFIG_EVENT_OUTCOMES
        else None
    )
    data: dict[str, Any] = {
        "context": str(context or "unknown"),
        "operation": str(operation or "unknown"),
        "started_ms": int(started_ms) if started_ms is not None else None,
        "elapsed_ms": int(elapsed_ms) if elapsed_ms is not None else None,
        "outcome": outcome_value,
        "response_code": _bounded_exchange_config_event_field(
            response_code, _EXCHANGE_CONFIG_EVENT_RESPONSE_CODE_RE
        ),
    }
    if error is not None:
        data["error_type"] = type(error).__name__
        data["error"] = _sanitize_remote_text(error, max_len=500)
    elif error_type is not None:
        data["error_type"] = _bounded_exchange_config_event_field(
            error_type, _EXCHANGE_CONFIG_EVENT_ERROR_TYPE_RE
        )
    _safe_emit(
        bot,
        EventTypes.EXCHANGE_CONFIG_REFRESH,
        level=level if level in _LIVE_EVENT_LEVELS else default_level,
        component="exchange.config_refresh",
        tags=(EventTags.EXCHANGE,),
        symbol=symbol_value,
        cycle_id=current_live_event_cycle_id(bot),
        status=status_value,
        reason_code=(
            ReasonCodes.EXCHANGE_CONFIG_REFRESH_FAILED
            if failed
            else ReasonCodes.EXCHANGE_CONFIG_REFRESH
        ),
        data={key: value for key, value in data.items() if value is not None},
    )


def emit_exchange_config_refresh_event(
    bot: Any, *args: Any, **kwargs: Any
) -> None:
    """Best-effort structured visibility for periodic exchange config refresh."""
    try:
        _emit_exchange_config_refresh_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug("[event] failed to emit exchange config-refresh event: %s", exc)


def begin_live_event_cycle(bot: Any, *, loop_start_ms: int) -> str:
    bot._live_event_cycle_seq = int(getattr(bot, "_live_event_cycle_seq", 0) or 0) + 1
    cycle_id = f"cy_{int(bot._live_event_cycle_seq)}"
    bot._live_event_current_cycle_id = cycle_id
    bot._emit_live_event(
        EventTypes.CYCLE_STARTED,
        level="debug",
        component="execution_loop",
        tags=(EventTags.CYCLE, EventTags.EXECUTION),
        cycle_id=cycle_id,
        status="started",
        data={
            "loop_start_ms": int(loop_start_ms),
            "authoritative_epoch": int(
                getattr(bot, "_authoritative_refresh_epoch", 0) or 0
            ),
        },
    )
    return cycle_id


def emit_live_cycle_completed(
    bot: Any,
    *,
    cycle_id: str | None,
    loop_start_ms: int,
    timings_ms: dict[str, int],
) -> None:
    if not cycle_id:
        return
    elapsed_ms = max(0, int(utc_ms()) - int(loop_start_ms))
    bot._emit_live_event(
        EventTypes.CYCLE_COMPLETED,
        level="debug",
        component="execution_loop",
        tags=(EventTags.CYCLE, EventTags.EXECUTION),
        cycle_id=cycle_id,
        status="succeeded",
        data={
            "elapsed_ms": elapsed_ms,
            "timings_ms": dict(timings_ms or {}),
            "authoritative_epoch": int(
                getattr(bot, "_authoritative_refresh_epoch", 0) or 0
            ),
            "orders_changed": bool(getattr(bot, "execution_scheduled", False)),
        },
    )
    if getattr(bot, "_live_event_current_cycle_id", None) == cycle_id:
        bot._live_event_current_cycle_id = None


def emit_live_cycle_degraded(
    bot: Any,
    *,
    cycle_id: str | None,
    reason_code: str,
    data: dict | None = None,
    level: str = "debug",
    terminal: bool = True,
) -> None:
    if not cycle_id:
        return
    payload = dict(data or {})
    payload.setdefault(
        "authoritative_epoch", int(getattr(bot, "_authoritative_refresh_epoch", 0) or 0)
    )
    bot._emit_live_event(
        EventTypes.CYCLE_DEGRADED,
        level=level,
        component="execution_loop",
        tags=(EventTags.CYCLE, EventTags.EXECUTION, EventTags.DEGRADED),
        cycle_id=cycle_id,
        status="degraded",
        reason_code=reason_code,
        data=payload,
    )
    if terminal and getattr(bot, "_live_event_current_cycle_id", None) == cycle_id:
        bot._live_event_current_cycle_id = None


def emit_live_event(
    bot: Any,
    event_type: str,
    *,
    level: str = "info",
    component: str | None = None,
    tags: tuple[str, ...] | list[str] = (),
    cycle_id: str | None = None,
    snapshot_id: str | None = None,
    plan_id: str | None = None,
    action_id: str | None = None,
    order_wave_id: str | None = None,
    remote_call_id: str | None = None,
    remote_call_group_id: str | None = None,
    symbol: str | None = None,
    pside: str | None = None,
    side: str | None = None,
    order_id: str | None = None,
    client_order_id: str | None = None,
    status: str | None = None,
    reason_code: str | None = None,
    message: str | None = None,
    data: dict | None = None,
    raw_ref: str | None = None,
    raw_hash: str | None = None,
    require_enqueue: bool = False,
):
    return emit_event(
        bot,
        LiveEvent(
            event_type,
            level=level,
            source="live",
            component=component,
            tags=tuple(tags or ()),
            exchange=getattr(bot, "exchange", None),
            user=getattr(bot, "user", None),
            bot_id=getattr(bot, "bot_id", None),
            cycle_id=cycle_id,
            snapshot_id=snapshot_id,
            plan_id=plan_id,
            action_id=action_id,
            order_wave_id=order_wave_id,
            remote_call_id=remote_call_id,
            remote_call_group_id=remote_call_group_id,
            symbol=symbol,
            pside=pside,
            side=side,
            order_id=order_id,
            client_order_id=client_order_id,
            status=status,
            reason_code=reason_code,
            message=message,
            data=data or {},
            raw_ref=raw_ref,
            raw_hash=raw_hash,
        ),
        require_enqueue=require_enqueue,
    )


def emit_planning_defer_summary_event(
    bot: Any,
    *,
    reason_code: str,
    count: int,
    window_s: int,
    symbols: list[str] | tuple[str, ...] | set[str],
    details: dict | None = None,
) -> None:
    try:
        details = dict(details or {})
        invalid = details.get("invalid") if isinstance(details.get("invalid"), dict) else {}
        all_symbols = sorted(str(symbol) for symbol in (symbols or []) if symbol)
        symbol_limit = 32
        bot._emit_live_event(
            EventTypes.PLANNING_DEFER_SUMMARY,
            level="info",
            component="planning_gates",
            tags=(
                EventTags.PLANNING,
                EventTags.GATE,
                EventTags.DEFER,
                EventTags.SUMMARY,
            ),
            cycle_id=bot._current_live_event_cycle_id(),
            status="deferred",
            reason_code=str(reason_code),
            message="staged planning defer summary",
            data={
                "count": int(count),
                "window_s": int(window_s),
                "symbols": all_symbols[:symbol_limit],
                "symbols_count": len(all_symbols),
                "symbols_truncated": len(all_symbols) > symbol_limit,
                "missing": sorted(str(item) for item in details.get("missing") or []),
                "required": sorted(str(item) for item in details.get("required") or []),
                "context": str(details.get("context") or "planning"),
                "epoch": int(details.get("epoch", 0) or 0),
                "invalid_surfaces": sorted(str(surface) for surface in invalid),
                "will_retry": "automatic",
            },
        )
    except Exception as exc:
        logging.debug("[event] failed to emit planning defer summary event: %s", exc)


def emit_planning_symbol_state_event(
    bot: Any,
    availability: Any,
    *,
    context: str,
    sample_limit: int = 32,
) -> None:
    try:
        records = tuple(getattr(availability, "records", ()) or ())
        summary = dict(availability.summary())
        unavailable = [
            record
            for record in records
            if str(getattr(record, "status", "")) == "unavailable"
        ]
        reason_counts = Counter(
            str(getattr(record, "reason_code", "") or "unknown")
            for record in unavailable
        )
        order_class_counts = Counter(
            str(getattr(record, "order_class", "") or "unknown")
            for record in unavailable
        )
        surface_counts: Counter[str] = Counter()
        for record in unavailable:
            surface_counts.update(
                str(surface)
                for surface in getattr(record, "unavailable_surfaces", ()) or ()
            )
        symbols = sorted(
            {
                str(getattr(record, "symbol", ""))
                for record in unavailable
                if getattr(record, "symbol", None)
            }
        )

        samples = []
        for record in unavailable[: max(0, int(sample_limit))]:
            samples.append(
                {
                    "symbol": str(getattr(record, "symbol", "")),
                    "pside": str(getattr(record, "position_side", "")),
                    "order_class": str(getattr(record, "order_class", "")),
                    "reason_code": str(getattr(record, "reason_code", "") or "unknown"),
                    "unavailable_surfaces": [
                        str(surface)
                        for surface in getattr(record, "unavailable_surfaces", ()) or ()
                    ],
                    "required_surfaces": [
                        str(surface)
                        for surface in getattr(record, "required_surfaces", ()) or ()
                    ],
                }
            )

        bot._emit_live_event(
            EventTypes.PLANNING_SYMBOL_STATE,
            level="debug",
            component="planning_availability",
            tags=(EventTags.PLANNING, EventTags.SNAPSHOT, EventTags.AVAILABILITY),
            cycle_id=current_live_event_cycle_id(bot),
            snapshot_id=str(getattr(availability, "snapshot_id", "") or ""),
            status="succeeded",
            reason_code=ReasonCodes.SNAPSHOT_SYMBOL_STATE,
            data={
                "context": str(context),
                "summary": summary,
                "unavailable_count": len(unavailable),
                "unavailable_by_reason": dict(sorted(reason_counts.items())),
                "unavailable_by_order_class": dict(sorted(order_class_counts.items())),
                "unavailable_by_surface": dict(sorted(surface_counts.items())),
                "unavailable_symbols": symbols[:32],
                "unavailable_symbols_count": len(symbols),
                "unavailable_symbols_truncated": len(symbols) > 32,
                "records_sample": samples,
                "records_sample_count": len(samples),
                "records_truncated": len(unavailable) > len(samples),
            },
        )
    except Exception as exc:
        logging.debug("[event] failed to emit planning symbol state event: %s", exc)


def emit_order_wave_completed_event(
    bot: Any, wave: dict | None, *, elapsed_ms: int, level: str = "info"
) -> None:
    if not wave:
        return
    planned_cancel = int(wave.get("planned_cancel", 0) or 0)
    planned_create = int(wave.get("planned_create", 0) or 0)
    cancel_posted = int(wave.get("cancel_posted", 0) or 0)
    create_posted = int(wave.get("create_posted", 0) or 0)
    skipped_cancel = int(wave.get("skipped_cancel", 0) or 0)
    deferred_create = int(wave.get("deferred_create", 0) or 0)
    skipped_create = int(wave.get("skipped_create", 0) or 0)
    status = "succeeded"
    reason_code = None
    if deferred_create:
        status = "deferred"
        reason_code = "create_deferred"
    elif skipped_cancel or skipped_create:
        status = "skipped"
        reason_code = "order_filtered"
    data = {
        "id": int(wave.get("id", 0) or 0),
        "elapsed_ms": int(elapsed_ms),
        "planned_cancel": planned_cancel,
        "planned_create": planned_create,
        "cancel_posted": cancel_posted,
        "create_posted": create_posted,
        "cancel_ms": wave.get("cancel_ms"),
        "create_ms": wave.get("create_ms"),
        "skipped_cancel": skipped_cancel,
        "deferred_create": deferred_create,
        "skipped_create": skipped_create,
        "symbols": list(wave.get("symbols") or []),
        "requested_confirmations": dict(wave.get("requested_confirmations") or {}),
    }
    _add_execution_debug_profile(
        bot,
        data,
        event_type=EventTypes.ORDER_WAVE_COMPLETED,
        wave=wave,
    )
    bot._emit_live_event(
        EventTypes.ORDER_WAVE_COMPLETED,
        level=str(level).lower(),
        component="order_wave",
        tags=(EventTags.ORDER, EventTags.WAVE, EventTags.EXECUTION),
        cycle_id=bot._current_live_event_cycle_id(),
        order_wave_id=str(wave.get("event_id") or f"ow_{wave.get('id', '')}"),
        status=status,
        reason_code=reason_code,
        data=data,
    )


def emit_order_wave_started_event(bot: Any, wave: dict | None) -> None:
    if not wave:
        return
    try:
        data = {
            "id": int(wave.get("id", 0) or 0),
            "planned_cancel": int(wave.get("planned_cancel", 0) or 0),
            "planned_create": int(wave.get("planned_create", 0) or 0),
            "symbols": list(wave.get("symbols") or []),
        }
        _add_execution_debug_profile(
            bot,
            data,
            event_type=EventTypes.ORDER_WAVE_STARTED,
            wave=wave,
        )
        bot._emit_live_event(
            EventTypes.ORDER_WAVE_STARTED,
            level="debug",
            component="order_wave",
            tags=(EventTags.ORDER, EventTags.WAVE, EventTags.EXECUTION),
            cycle_id=bot._current_live_event_cycle_id(),
            order_wave_id=str(wave.get("event_id") or f"ow_{wave.get('id', '')}"),
            status="started",
            data=data,
        )
    except Exception as exc:
        logging.debug("[event] failed to emit order wave started event: %s", exc)


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _safe_finite_float(value: Any) -> float | None:
    number = _safe_float(value)
    return number if number is not None and math.isfinite(number) else None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bounded_memory_snapshot_int(value: Any, *, maximum: int) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0:
        return None
    return min(int(number), maximum)


def _bounded_memory_snapshot_delta(value: Any) -> float | None:
    number = _safe_finite_float(value)
    if number is None:
        return None
    return round(
        max(-_MEMORY_SNAPSHOT_MAX_DELTA_PCT, min(number, _MEMORY_SNAPSHOT_MAX_DELTA_PCT)),
        6,
    )


def _bounded_memory_snapshot_label(
    value: Any, pattern: re.Pattern[str], *, reject_sensitive: bool = False
) -> str:
    if not isinstance(value, str):
        return "unknown"
    if (
        "://" in value
        or "//" in value
        or not pattern.fullmatch(value)
        or (reject_sensitive and _MEMORY_SNAPSHOT_SENSITIVE_TASK_RE.search(value))
    ):
        return "unknown"
    return value


def _memory_snapshot_cache_samples(value: Any, *, timeframe: bool) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    samples: list[dict[str, Any]] = []
    for raw in value[:3]:
        if not isinstance(raw, (list, tuple)):
            continue
        if timeframe and len(raw) == 3 and isinstance(raw[0], tuple) and len(raw[0]) == 2:
            symbol_raw, timeframe_raw = raw[0]
            bytes_raw, candles_raw = raw[1:]
        elif timeframe and len(raw) == 4:
            symbol_raw, timeframe_raw, bytes_raw, candles_raw = raw
        elif not timeframe and len(raw) == 3:
            symbol_raw, bytes_raw, candles_raw = raw
            timeframe_raw = None
        else:
            continue
        symbol = _bounded_memory_snapshot_label(symbol_raw, _MEMORY_SNAPSHOT_SYMBOL_RE)
        sample: dict[str, Any] = {"symbol": symbol}
        if timeframe:
            sample["timeframe"] = _bounded_memory_snapshot_label(
                timeframe_raw, _MEMORY_SNAPSHOT_TIMEFRAME_RE
            )
        bytes_ = _bounded_memory_snapshot_int(
            bytes_raw, maximum=_MEMORY_SNAPSHOT_MAX_BYTES
        )
        candles = _bounded_memory_snapshot_int(
            candles_raw, maximum=_MEMORY_SNAPSHOT_MAX_COUNT
        )
        if bytes_ is None or candles is None:
            continue
        sample["bytes"] = bytes_
        sample["candles"] = candles
        samples.append(sample)
    return samples


def _memory_snapshot_task_samples(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    samples: list[dict[str, Any]] = []
    for raw in value[:4]:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            continue
        count = _bounded_memory_snapshot_int(
            raw[1], maximum=_MEMORY_SNAPSHOT_MAX_COUNT
        )
        if count is None:
            continue
        samples.append(
            {
                "name": _bounded_memory_snapshot_label(
                    raw[0], _MEMORY_SNAPSHOT_TASK_NAME_RE, reject_sensitive=True
                ),
                "count": count,
            }
        )
    return samples


def emit_memory_snapshot_event(
    bot: Any,
    *,
    rss_bytes: Any,
    rss_delta_pct: Any = None,
    cache_bytes: Any = None,
    cache_candles: Any = None,
    cache_symbols: Any = None,
    cache_samples: Any = None,
    timeframe_cache_bytes: Any = None,
    timeframe_cache_ranges: Any = None,
    timeframe_cache_samples: Any = None,
    task_total: Any = None,
    task_pending: Any = None,
    task_samples: Any = None,
) -> Any:
    """Emit a bounded, observability-only memory snapshot event."""
    try:
        emit = getattr(bot, "_emit_live_event", None)
        if not callable(emit):
            return None
        data: dict[str, Any] = {}
        bounded_rss = _bounded_memory_snapshot_int(
            rss_bytes, maximum=_MEMORY_SNAPSHOT_MAX_BYTES
        )
        if bounded_rss is None:
            return None
        data["rss_bytes"] = bounded_rss
        delta = _bounded_memory_snapshot_delta(rss_delta_pct)
        if delta is not None:
            data["rss_delta_pct"] = delta
        cache: dict[str, Any] = {}
        for key, raw_value, maximum in (
            ("bytes", cache_bytes, _MEMORY_SNAPSHOT_MAX_BYTES),
            ("candles", cache_candles, _MEMORY_SNAPSHOT_MAX_COUNT),
            ("symbols", cache_symbols, _MEMORY_SNAPSHOT_MAX_COUNT),
        ):
            bounded = _bounded_memory_snapshot_int(raw_value, maximum=maximum)
            if bounded is not None:
                cache[key] = bounded
        samples = _memory_snapshot_cache_samples(cache_samples, timeframe=False)
        if samples:
            cache["samples"] = samples
        if cache:
            data["cache"] = cache
        timeframe_cache: dict[str, Any] = {}
        for key, raw_value, maximum in (
            ("bytes", timeframe_cache_bytes, _MEMORY_SNAPSHOT_MAX_BYTES),
            ("ranges", timeframe_cache_ranges, _MEMORY_SNAPSHOT_MAX_COUNT),
        ):
            bounded = _bounded_memory_snapshot_int(raw_value, maximum=maximum)
            if bounded is not None:
                timeframe_cache[key] = bounded
        samples = _memory_snapshot_cache_samples(timeframe_cache_samples, timeframe=True)
        if samples:
            timeframe_cache["samples"] = samples
        if timeframe_cache:
            data["timeframe_cache"] = timeframe_cache
        tasks: dict[str, Any] = {}
        for key, raw_value in (("total", task_total), ("pending", task_pending)):
            bounded = _bounded_memory_snapshot_int(
                raw_value, maximum=_MEMORY_SNAPSHOT_MAX_COUNT
            )
            if bounded is not None:
                tasks[key] = bounded
        samples = _memory_snapshot_task_samples(task_samples)
        if samples:
            tasks["samples"] = samples
        if tasks:
            data["tasks"] = tasks
        return emit(
            EventTypes.RESOURCE_MEMORY_SNAPSHOT,
            level="info",
            component="resource.memory",
            tags=(EventTags.RESOURCE, EventTags.MEMORY),
            cycle_id=current_live_event_cycle_id(bot),
            status="succeeded",
            reason_code=ReasonCodes.MEMORY_SNAPSHOT,
            data=data,
        )
    except Exception as exc:
        logging.debug("[event] failed to emit memory snapshot event: %s", exc)
        return None


def _rust_symbol_name(
    idx_to_symbol: dict[int, str] | None, symbol_idx: int | None
) -> str | None:
    if symbol_idx is None or not isinstance(idx_to_symbol, dict):
        return None
    value = idx_to_symbol.get(int(symbol_idx))
    return str(value) if value is not None else None


def rust_input_symbol_debug_sample(
    input_symbols: Any,
    *,
    idx_to_symbol: dict[int, str] | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    """Return a bounded, non-raw sample of Rust orchestrator symbol inputs."""
    symbols = (
        list(input_symbols or []) if isinstance(input_symbols, (list, tuple)) else []
    )
    sample: list[dict[str, Any]] = []
    for item in symbols[: max(0, int(limit))]:
        if not isinstance(item, dict):
            continue
        symbol_idx = _safe_int(item.get("symbol_idx"))
        order_book = (
            item.get("order_book") if isinstance(item.get("order_book"), dict) else {}
        )
        emas = item.get("emas") if isinstance(item.get("emas"), dict) else {}
        m1 = emas.get("m1") if isinstance(emas.get("m1"), dict) else {}
        h1 = emas.get("h1") if isinstance(emas.get("h1"), dict) else {}
        active_psides: list[str] = []
        for pside in ("long", "short"):
            side_cfg = item.get(pside) if isinstance(item.get(pside), dict) else {}
            position = (
                side_cfg.get("position")
                if isinstance(side_cfg.get("position"), dict)
                else {}
            )
            size = _safe_float(position.get("size"))
            if size is not None and abs(size) > 0.0:
                active_psides.append(pside)
        payload = {
            "symbol_idx": symbol_idx,
            "symbol": _rust_symbol_name(idx_to_symbol, symbol_idx),
            "tradable": bool(item.get("tradable")),
            "has_bid": order_book.get("bid") is not None,
            "has_ask": order_book.get("ask") is not None,
            "effective_min_cost": _safe_float(item.get("effective_min_cost")),
            "m1_close_ema_count": len(m1.get("close") or []),
            "m1_log_range_ema_count": len(m1.get("log_range") or []),
            "m1_volume_ema_count": len(m1.get("volume") or []),
            "h1_log_range_ema_count": len(h1.get("log_range") or []),
            "active_psides": active_psides,
        }
        sample.append(
            {key: value for key, value in payload.items() if value is not None}
        )
    return {
        "count": len(symbols),
        "sample": sample,
        "truncated": max(0, len(symbols) - len(sample)),
    }


def rust_output_order_debug_sample(
    orders: Any,
    *,
    idx_to_symbol: dict[int, str] | None = None,
    limit: int = 12,
) -> dict[str, Any]:
    """Return a bounded sample of Rust output orders for diagnostics."""
    values = list(orders or []) if isinstance(orders, (list, tuple)) else []
    sample: list[dict[str, Any]] = []
    for item in values[: max(0, int(limit))]:
        if not isinstance(item, dict):
            continue
        symbol_idx = _safe_int(item.get("symbol_idx"))
        payload: dict[str, Any] = {
            "symbol_idx": symbol_idx,
            "symbol": _rust_symbol_name(idx_to_symbol, symbol_idx),
            "order_type": (
                str(item.get("order_type"))
                if item.get("order_type") is not None
                else None
            ),
            "execution_type": (
                str(item.get("execution_type"))
                if item.get("execution_type") is not None
                else None
            ),
            "side": str(item.get("side")) if item.get("side") is not None else None,
            "pside": (
                str(item.get("pside") or item.get("position_side"))
                if item.get("pside") is not None
                or item.get("position_side") is not None
                else None
            ),
            "qty": _safe_float(item.get("qty")),
            "price": _safe_float(item.get("price")),
            "reduce_only": bool(item.get("reduce_only"))
            if item.get("reduce_only") is not None
            else None,
        }
        sample.append(
            {key: value for key, value in payload.items() if value is not None}
        )
    return {
        "count": len(values),
        "sample": sample,
        "truncated": max(0, len(values) - len(sample)),
    }


def _best_effort_rust_input_symbol_debug_sample(
    input_symbols: Any,
    *,
    idx_to_symbol: dict[int, str] | None = None,
) -> dict[str, Any] | None:
    try:
        return rust_input_symbol_debug_sample(input_symbols, idx_to_symbol=idx_to_symbol)
    except Exception as exc:
        logging.debug("[event] failed to build rust input debug sample: %s", exc)
        return None


def _best_effort_rust_output_order_debug_sample(
    orders: Any,
    *,
    idx_to_symbol: dict[int, str] | None = None,
) -> dict[str, Any] | None:
    try:
        return rust_output_order_debug_sample(orders, idx_to_symbol=idx_to_symbol)
    except Exception as exc:
        logging.debug("[event] failed to build rust output debug sample: %s", exc)
        return None


def _startup_debug_payload(data: dict[str, Any], *, limit: int = 32) -> dict[str, Any]:
    debug: dict[str, Any] = {"data_keys": _mapping_key_sample(data, limit=limit)}
    phase = data.get("phase")
    if phase is not None:
        debug["phase"] = str(phase)
    for key in ("elapsed_ms", "since_previous_ms"):
        value = _safe_int(data.get(key))
        if value is not None:
            debug[key] = max(0, int(value))
    if "details" in data:
        details = str(data.get("details") or "")
        debug["details_present"] = bool(details)
        debug["details_len"] = len(details)
    return debug


def _emit_startup_timing_event_unchecked(
    bot: Any,
    *,
    phase: str,
    elapsed_ms: int,
    since_previous_ms: int,
    details: str = "",
) -> None:
    elapsed = _safe_int(elapsed_ms)
    since_previous = _safe_int(since_previous_ms)
    phase_text = str(phase)
    data: dict[str, Any] = {
        "phase": phase_text,
        "elapsed_ms": max(0, int(elapsed)) if elapsed is not None else 0,
        "since_previous_ms": (
            max(0, int(since_previous)) if since_previous is not None else 0
        ),
    }
    readiness_contract = startup_phase_readiness_contract(phase_text)
    if readiness_contract is not None:
        data.update(readiness_contract)
    if details:
        data["details"] = str(details)
    if live_event_debug_profile_enabled(bot, "startup"):
        data["debug_profile"] = "startup"
        data["debug"] = _startup_debug_payload(data)
    _safe_emit(
        bot,
        EventTypes.BOT_STARTUP_TIMING,
        level="info",
        component="bot.startup",
        tags=("bot", "startup", "timing"),
        cycle_id=current_live_event_cycle_id(bot),
        status="succeeded",
        reason_code=ReasonCodes.STARTUP_PHASE_READY,
        data=data,
    )


def emit_startup_timing_event(bot: Any, *args: Any, **kwargs: Any) -> None:
    try:
        _emit_startup_timing_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug(
            "[event] failed to emit %s: %s",
            EventTypes.BOT_STARTUP_TIMING,
            exc,
        )


def _emit_health_summary_event_unchecked(
    bot: Any,
    payload: dict[str, Any],
) -> Any:
    data = dict(payload or {})
    return _safe_emit(
        bot,
        EventTypes.HEALTH_SUMMARY,
        level="info",
        component="bot.health",
        tags=(EventTags.HEALTH, EventTags.RESOURCE),
        cycle_id=current_live_event_cycle_id(bot),
        status="succeeded",
        reason_code=ReasonCodes.PERIODIC_HEALTH_SUMMARY,
        data=data,
        require_enqueue=True,
    )


def emit_health_summary_event(bot: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return _emit_health_summary_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug("[event] failed to emit health summary event: %s", exc)
        return None


def _emit_market_snapshot_diagnostic_skipped_event_unchecked(
    bot: Any,
    *,
    context: str,
    error: BaseException,
) -> None:
    _safe_emit(
        bot,
        EventTypes.MARKET_SNAPSHOT_DIAGNOSTIC_SKIPPED,
        level="warning",
        component="market.snapshot",
        tags=(EventTags.MARKET, EventTags.SNAPSHOT, EventTags.DEGRADED),
        cycle_id=current_live_event_cycle_id(bot),
        status="skipped",
        reason_code=ReasonCodes.MARKET_SNAPSHOT_DIAGNOSTIC_SKIPPED,
        data={
            "context": str(context),
            "error_type": type(error).__name__,
            "error": _sanitize_remote_text(error, max_len=500),
        },
    )


def emit_market_snapshot_diagnostic_skipped_event(
    bot: Any, *args: Any, **kwargs: Any
) -> None:
    try:
        _emit_market_snapshot_diagnostic_skipped_event_unchecked(
            bot, *args, **kwargs
        )
    except Exception as exc:
        logging.debug(
            "[event] failed to emit market snapshot diagnostic skipped event: %s",
            exc,
        )


def _safe_int_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, raw in sorted(value.items(), key=lambda item: str(item[0])):
        number = _safe_int(raw)
        if number is None:
            continue
        out[str(key)] = max(0, int(number))
    return out


def _sorted_str_list(value: Any) -> list[str]:
    try:
        return sorted(str(item) for item in value)
    except Exception:
        return []


def _forager_debug_payload(data: dict[str, Any], *, limit: int = 32) -> dict[str, Any]:
    debug: dict[str, Any] = {"data_keys": _mapping_key_sample(data, limit=limit)}
    for key in (
        "candidate_count",
        "eligible_count",
        "selected_count",
        "incumbent_count",
        "max_n_positions",
        "slots_to_fill",
        "feature_unavailable_count",
        "volatility_dropped_count",
        "hysteresis_event_count",
        "fetch_budget",
    ):
        value = _safe_int(data.get(key))
        if value is not None:
            debug[key] = max(0, int(value))
    for key in ("slots_open",):
        if key in data:
            debug[key] = bool(data.get(key))
    unavailable = data.get("unavailable")
    if isinstance(unavailable, dict):
        unavailable_count = _safe_int(unavailable.get("count"))
        sample = unavailable.get("sample")
        debug["unavailable_count"] = max(
            0, int(unavailable_count) if unavailable_count is not None else 0
        )
        debug["unavailable_sample_count"] = len(sample) if isinstance(sample, list) else 0
        debug["unavailable_truncated"] = bool(unavailable.get("truncated"))
    top_scores = data.get("top_scores")
    if isinstance(top_scores, list):
        debug["top_scores_count"] = len(top_scores)
        score_keys: set[str] = set()
        for item in top_scores[:limit]:
            if isinstance(item, dict):
                score_keys.update(str(key) for key in item)
        if score_keys:
            debug["top_score_keys"] = sorted(score_keys)[:limit]
    return debug


def _state_refresh_debug_payload(
    data: dict[str, Any], *, limit: int = 32
) -> dict[str, Any]:
    debug: dict[str, Any] = {"data_keys": _mapping_key_sample(data, limit=limit)}
    plan = data.get("plan")
    if isinstance(plan, list):
        debug["plan_count"] = len(plan)
        debug["plan_sample"] = [str(item) for item in plan[:limit]]
    pending = data.get("pending")
    if isinstance(pending, list):
        debug["pending_count"] = len(pending)
        debug["pending_sample"] = [str(item) for item in pending[:limit]]
    epoch_changed = data.get("epoch_changed")
    if isinstance(epoch_changed, list):
        debug["epoch_changed_count"] = len(epoch_changed)
        debug["epoch_changed_sample"] = [str(item) for item in epoch_changed[:limit]]
    for key in (
        "timings_ms",
        "completed_timings_ms",
        "surfaces_ms",
    ):
        value = data.get(key)
        if isinstance(value, dict):
            keys = _mapping_key_sample(value, limit=limit)
            debug[f"{key}_count"] = len(value)
            debug[f"{key}_keys"] = keys
            if key in {"timings_ms", "completed_timings_ms"} and value:
                slowest_key, slowest_value = max(
                    value.items(),
                    key=lambda item: int(_safe_int(item[1]) or 0),
                )
                debug[f"{key}_slowest"] = {
                    "surface": str(slowest_key),
                    "elapsed_ms": max(0, int(_safe_int(slowest_value) or 0)),
                }
    for key in (
        "wall_ms",
        "surface_sum_ms",
        "surface_max_ms",
        "residual_ms",
        "elapsed_ms",
        "count",
        "since_ms",
    ):
        value = _safe_int(data.get(key))
        if value is not None:
            debug[key] = max(0, int(value))
    threshold = _safe_float(data.get("threshold_s"))
    if threshold is not None:
        debug["threshold_s"] = max(0.0, float(threshold))
    for key in (
        "parallel",
        "pending_confirmations",
        "meaningful_change",
        "unusual_plan",
        "repeated",
        "summary",
    ):
        if key in data:
            debug[key] = bool(data.get(key))
    return debug


def _emit_state_refresh_timing_event_unchecked(
    bot: Any,
    *,
    plan: Any,
    timings_ms: dict[str, int],
    wall_ms: int,
    sum_ms: int,
    max_surface_ms: int,
    residual_ms: int,
    pending_confirmations: bool,
    meaningful_change: bool,
    unusual_plan: bool,
    epoch_changed: Any = None,
    level: str = "debug",
) -> None:
    data = {
        "plan": _sorted_str_list(plan),
        "timings_ms": _safe_int_map(timings_ms),
        "wall_ms": max(0, int(_safe_int(wall_ms) or 0)),
        "surface_sum_ms": max(0, int(_safe_int(sum_ms) or 0)),
        "surface_max_ms": max(0, int(_safe_int(max_surface_ms) or 0)),
        "residual_ms": max(0, int(_safe_int(residual_ms) or 0)),
        "parallel": len(timings_ms or {}) > 1,
        "pending_confirmations": bool(pending_confirmations),
        "meaningful_change": bool(meaningful_change),
        "unusual_plan": bool(unusual_plan),
        "epoch_changed": _sorted_str_list(epoch_changed or []),
    }
    if live_event_debug_profile_enabled(bot, "state"):
        data["debug"] = _state_refresh_debug_payload(data)
        data["debug_profile"] = "state"
    _safe_emit(
        bot,
        EventTypes.STATE_REFRESH_TIMING,
        level=str(level or "debug"),
        component="state.refresh",
        tags=(EventTags.STATE, EventTags.REFRESH, EventTags.ACCOUNT),
        cycle_id=current_live_event_cycle_id(bot),
        status="succeeded",
        reason_code=ReasonCodes.STAGED_REFRESH_TIMING,
        data=data,
    )


def emit_state_refresh_timing_event(bot: Any, *args: Any, **kwargs: Any) -> None:
    try:
        _emit_state_refresh_timing_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug("[event] failed to emit state refresh timing event: %s", exc)


def _stat_summary_payload(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    count = max(0, int(_safe_int(value.get("count")) or 0))
    total = max(0, int(_safe_int(value.get("sum")) or 0))
    mean = int(round(total / max(1, count))) if count else 0
    return {
        "count": count,
        "min": max(0, int(_safe_int(value.get("min")) or 0)),
        "mean": max(0, mean),
        "max": max(0, int(_safe_int(value.get("max")) or 0)),
    }


def _emit_state_refresh_timing_summary_event_unchecked(
    bot: Any,
    *,
    plan: Any,
    count: int,
    since_ms: int,
    wall: dict[str, int],
    surface_sum: dict[str, int],
    surface_max: dict[str, int],
    residual: dict[str, int],
    surfaces: dict[str, dict[str, int]],
) -> None:
    data = {
        "summary": True,
        "plan": _sorted_str_list(plan),
        "count": max(0, int(_safe_int(count) or 0)),
        "since_ms": max(0, int(_safe_int(since_ms) or 0)),
        "wall_ms": _stat_summary_payload(wall),
        "surface_sum_ms": _stat_summary_payload(surface_sum),
        "surface_max_ms": _stat_summary_payload(surface_max),
        "residual_ms": _stat_summary_payload(residual),
        "surfaces_ms": {
            str(surface): _stat_summary_payload(stats)
            for surface, stats in sorted(
                (surfaces or {}).items(), key=lambda item: str(item[0])
            )
        },
    }
    if live_event_debug_profile_enabled(bot, "state"):
        data["debug"] = _state_refresh_debug_payload(data)
        data["debug_profile"] = "state"
    _safe_emit(
        bot,
        EventTypes.STATE_REFRESH_TIMING,
        level="info",
        component="state.refresh",
        tags=(EventTags.STATE, EventTags.REFRESH, EventTags.ACCOUNT, EventTags.SUMMARY),
        cycle_id=current_live_event_cycle_id(bot),
        status="succeeded",
        reason_code=ReasonCodes.STAGED_REFRESH_TIMING,
        data=data,
    )


def emit_state_refresh_timing_summary_event(
    bot: Any, *args: Any, **kwargs: Any
) -> None:
    try:
        _emit_state_refresh_timing_summary_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug(
            "[event] failed to emit state refresh timing summary event: %s", exc
        )


def _emit_state_refresh_progress_event_unchecked(
    bot: Any,
    *,
    plan: Any,
    pending: Any,
    elapsed_ms: int,
    completed_timings_ms: dict[str, int] | None = None,
    threshold_s: float | None = None,
    repeated: bool = False,
    level: str = "info",
) -> None:
    data: dict[str, Any] = {
        "plan": _sorted_str_list(plan),
        "pending": _sorted_str_list(pending),
        "elapsed_ms": max(0, int(_safe_int(elapsed_ms) or 0)),
        "completed_timings_ms": _safe_int_map(completed_timings_ms or {}),
        "repeated": bool(repeated),
    }
    threshold = _safe_float(threshold_s)
    if threshold is not None:
        data["threshold_s"] = max(0.0, float(threshold))
    if live_event_debug_profile_enabled(bot, "state"):
        data["debug"] = _state_refresh_debug_payload(data)
        data["debug_profile"] = "state"
    _safe_emit(
        bot,
        EventTypes.STATE_REFRESH_PROGRESS,
        level=str(level or "info"),
        component="state.refresh",
        tags=(EventTags.STATE, EventTags.REFRESH, EventTags.TIMEOUT),
        cycle_id=current_live_event_cycle_id(bot),
        status="degraded",
        reason_code=ReasonCodes.STAGED_REFRESH_PROGRESS,
        data=data,
    )


def emit_state_refresh_progress_event(bot: Any, *args: Any, **kwargs: Any) -> None:
    try:
        _emit_state_refresh_progress_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug("[event] failed to emit state refresh progress event: %s", exc)


def _emit_execution_loop_error_burst_event_unchecked(
    bot: Any,
    *,
    count: int,
    window_s: int,
    endpoints: Any,
    latest_fields: dict[str, Any],
) -> None:
    try:
        top_endpoints = [
            {
                "endpoint": _bounded_execution_loop_error_field(
                    name, _EXECUTION_LOOP_ERROR_ENDPOINT_RE, fallback="unknown"
                ),
                "count": max(0, int(n)),
            }
            for name, n in endpoints.most_common(5)
        ]
    except Exception:
        top_endpoints = []
    latest = dict(latest_fields or {})
    data: dict[str, Any] = {
        "count": max(0, int(count)),
        "window_s": max(0, int(window_s)),
        "top_endpoints": top_endpoints,
        "latest_error_type": _bounded_execution_loop_error_field(
            latest.get("error_type"), _EXCHANGE_CONFIG_EVENT_ERROR_TYPE_RE
        ),
        "latest_status": _bounded_execution_loop_error_field(
            latest.get("status"), _EXECUTION_LOOP_ERROR_STATUS_RE
        ),
        "latest_code": _bounded_execution_loop_error_field(
            latest.get("code"), _EXECUTION_LOOP_ERROR_CODE_RE
        ),
        "latest_endpoint": _bounded_execution_loop_error_field(
            latest.get("endpoint"),
            _EXECUTION_LOOP_ERROR_ENDPOINT_RE,
            fallback="unknown",
        ),
    }
    _safe_emit(
        bot,
        EventTypes.HEALTH_SUMMARY,
        level="warning",
        component="execution_loop",
        tags=(EventTags.HEALTH, EventTags.EXECUTION, EventTags.SUMMARY),
        cycle_id=current_live_event_cycle_id(bot),
        status="degraded",
        reason_code=ReasonCodes.EXECUTION_LOOP_ERROR_BURST,
        data=data,
    )


def emit_execution_loop_error_burst_event(bot: Any, *args: Any, **kwargs: Any) -> None:
    try:
        _emit_execution_loop_error_burst_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug("[event] failed to emit execution loop error burst event: %s", exc)


def _symbol_sample(symbols: Any, *, limit: int = 12) -> dict[str, Any]:
    if symbols is None:
        values = []
    else:
        try:
            values = sorted({str(symbol) for symbol in symbols})
        except TypeError:
            values = [str(symbols)]
    return {
        "count": len(values),
        "sample": values[:limit],
        "truncated": max(0, len(values) - limit),
    }


def _span_sample(spans: Any, *, limit: int = 8) -> list[float]:
    out: list[float] = []
    values = []
    for raw_span in spans or []:
        span = _safe_float(raw_span)
        if span is not None:
            values.append(span)
    iterable = sorted(values)
    for span in iterable:
        if span == span:
            out.append(span)
        if len(out) >= limit:
            break
    return out


def _ema_map_summary(values: dict[str, dict[float, float]] | None) -> dict[str, int]:
    mapping = values or {}
    return {
        "symbols": len(mapping),
        "values": sum(len(item or {}) for item in mapping.values()),
    }


def _fallback_examples(
    values: dict[str, list[tuple[Any, ...]]] | None,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for symbol, items in sorted((values or {}).items())[:limit]:
        spans: list[float] = []
        ages: list[int] = []
        counts: list[int] = []
        metrics: set[str] = set()
        reason = None
        for item in items or []:
            if not isinstance(item, (list, tuple)):
                continue
            if len(item) >= 1 and isinstance(item[0], str):
                metrics.add(str(item[0]))
                if len(item) >= 2:
                    span = _safe_float(item[1])
                    if span is not None:
                        spans.append(span)
                if len(item) >= 3:
                    age = _safe_int(item[2])
                    if age is not None:
                        ages.append(age)
            else:
                if len(item) >= 1:
                    span = _safe_float(item[0])
                    if span is not None:
                        spans.append(span)
                if len(item) >= 2:
                    age = _safe_int(item[1])
                    if age is not None:
                        ages.append(age)
                if len(item) >= 3:
                    count = _safe_int(item[2])
                    if count is not None:
                        counts.append(count)
                if len(item) >= 4 and item[3] is not None:
                    reason = str(item[3])[:160]
        example: dict[str, Any] = {
            "symbol": str(symbol),
            "count": len(items or []),
            "spans": _span_sample(spans),
        }
        if metrics:
            example["metrics"] = sorted(metrics)
        if ages:
            example["max_age_ms"] = max(ages)
        if counts:
            example["max_fallbacks"] = max(counts)
        if reason:
            example["reason"] = reason
        examples.append(example)
    return examples


def _candidate_unavailable_summary(
    values: dict[str, list[tuple[str, str, str]]] | None,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for reason, items in sorted((values or {}).items())[:limit]:
        symbols = sorted({str(symbol) for symbol, _error_type, _error in items or []})
        error_types = sorted(
            {str(error_type) for _symbol, error_type, _error in items or []}
        )
        example_error = next(
            (str(error) for _symbol, _error_type, error in items or [] if error),
            "",
        )
        out.append(
            {
                "reason": str(reason),
                "symbols": _symbol_sample(symbols),
                "error_types": error_types[:4],
                "example_error": example_error[:160] if example_error else None,
            }
        )
    return out


_EMA_TYPE_RE = re.compile(
    r"\b(?:missing\s+required\s+)?"
    r"(?P<ema_type>m1_close|m1_volume|m1_log_range|h1_log_range)\s+EMA\b",
    re.IGNORECASE,
)
_EMA_SPAN_REASON_RE = re.compile(
    r"\bspan=(?P<span>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"\s+reason=(?P<reason>[^;|]+)",
    re.IGNORECASE,
)


def _candidate_detail_tuple(item: Any) -> tuple[str, str, str]:
    if not isinstance(item, (list, tuple)):
        return str(item), "", ""
    symbol = str(item[0]) if len(item) >= 1 else ""
    error_type = str(item[1]) if len(item) >= 2 else ""
    error = str(item[2]) if len(item) >= 3 else ""
    return symbol, error_type, error


def _ema_unavailable_debug_summary(
    candidate_values: dict[str, list[tuple[str, str, str]]] | None,
    ema_unavailable_reasons: dict[str, list[str]] | None,
    *,
    limit: int = 8,
) -> dict[str, Any]:
    candidate_groups: list[dict[str, Any]] = []
    for reason, raw_items in sorted((candidate_values or {}).items())[:limit]:
        symbols: set[str] = set()
        error_types: set[str] = set()
        ema_type_counts: Counter[str] = Counter()
        spans: list[float] = []
        inner_reason_counts: Counter[str] = Counter()
        for raw_item in raw_items or []:
            symbol, error_type, error = _candidate_detail_tuple(raw_item)
            if symbol:
                symbols.add(symbol)
            if error_type:
                error_types.add(error_type)
            for match in _EMA_TYPE_RE.finditer(error or ""):
                ema_type_counts[match.group("ema_type").lower()] += 1
            for match in _EMA_SPAN_REASON_RE.finditer(error or ""):
                span = _safe_float(match.group("span"))
                if span is not None:
                    spans.append(span)
                inner_reason = str(match.group("reason") or "").strip()
                if inner_reason:
                    inner_reason_counts[inner_reason[:120]] += 1
        group: dict[str, Any] = {
            "reason": str(reason),
            "symbols": _symbol_sample(symbols),
            "error_types": sorted(error_types)[:4],
        }
        if ema_type_counts:
            group["ema_types"] = [
                {"ema_type": key, "count": int(count)}
                for key, count in sorted(ema_type_counts.items())
            ][:limit]
        if spans:
            group["spans"] = _span_sample(spans)
        if inner_reason_counts:
            group["inner_reasons"] = [
                {"reason": key, "count": int(count)}
                for key, count in sorted(
                    inner_reason_counts.items(), key=lambda kv: (-kv[1], kv[0])
                )[:limit]
            ]
        candidate_groups.append(group)

    unavailable_groups = [
        {
            "reason": str(reason),
            "symbols": _symbol_sample(symbols or ()),
        }
        for reason, symbols in sorted((ema_unavailable_reasons or {}).items())[:limit]
    ]
    return {
        "candidate_groups": candidate_groups,
        "unavailable_groups": unavailable_groups,
    }


def _reason_symbol_summary(
    values: dict[str, list[str]] | None,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for reason, symbols in sorted((values or {}).items())[:limit]:
        out.append({"reason": str(reason), "symbols": _symbol_sample(symbols)})
    return out


def _forager_top_score_sample(
    top_scores: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in list(top_scores or [])[:limit]:
        if not isinstance(item, dict):
            continue
        payload: dict[str, Any] = {
            "symbol": str(item.get("symbol")) if item.get("symbol") is not None else None,
            "rank": _safe_int(item.get("rank")),
            "score": _safe_float(item.get("score")),
            "selected": bool(item.get("selected")),
            "incumbent": bool(item.get("incumbent")),
        }
        for key in (
            "volume_component",
            "ema_readiness_component",
            "volatility_component",
        ):
            value = _safe_float(item.get(key))
            if value is not None:
                payload[key] = value
        out.append({key: value for key, value in payload.items() if value is not None})
    return out


def _safe_emit(bot: Any, event_type: str, **kwargs: Any) -> Any:
    try:
        return bot._emit_live_event(event_type, **kwargs)
    except Exception as exc:
        logging.debug("[event] failed to emit %s: %s", event_type, exc)
        return None


def _emit_rust_orchestrator_called_event_unchecked(
    bot: Any,
    *,
    rust_call_id: str | None,
    input_hash: str,
    symbol_count: int,
    tradable_count: int,
    ema_unavailable_count: int,
    trailing_unavailable_count: int,
    hedge_mode: bool,
    strategy_kind: str | None,
    input_symbol_sample: dict[str, Any] | None = None,
    input_symbols: Any = None,
    idx_to_symbol: dict[int, str] | None = None,
) -> None:
    data = {
        "symbol_count": int(symbol_count),
        "tradable_count": int(tradable_count),
        "ema_unavailable_count": int(ema_unavailable_count),
        "trailing_unavailable_count": int(trailing_unavailable_count),
        "hedge_mode": bool(hedge_mode),
        "strategy_kind": strategy_kind,
        "input_hash": str(input_hash),
    }
    if input_symbol_sample is None and live_event_debug_profile_enabled(bot, "rust"):
        input_symbol_sample = _best_effort_rust_input_symbol_debug_sample(
            input_symbols,
            idx_to_symbol=idx_to_symbol,
        )
    if input_symbol_sample is not None:
        data["debug_profile"] = "rust"
        data["input_symbol_sample"] = input_symbol_sample
    bot._emit_live_event(
        EventTypes.RUST_ORCHESTRATOR_CALLED,
        level="debug",
        component="rust_orchestrator",
        tags=("planning", "rust", "orchestrator"),
        cycle_id=current_live_event_cycle_id(bot),
        remote_call_id=rust_call_id,
        status="started",
        raw_hash=str(input_hash),
        data=data,
    )


def emit_rust_orchestrator_called_event(bot: Any, *args: Any, **kwargs: Any) -> None:
    try:
        _emit_rust_orchestrator_called_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug(
            "[event] failed to emit %s: %s",
            EventTypes.RUST_ORCHESTRATOR_CALLED,
            exc,
        )


def _emit_rust_orchestrator_returned_event_unchecked(
    bot: Any,
    *,
    rust_call_id: str | None,
    status: str,
    input_hash: str,
    elapsed_ms: int,
    output_hash: str | None = None,
    order_count: int | None = None,
    diagnostics: Any = None,
    error: BaseException | None = None,
    output_order_sample: dict[str, Any] | None = None,
    orders: Any = None,
    idx_to_symbol: dict[int, str] | None = None,
) -> None:
    event_status = str(status or "succeeded").lower()
    failed = event_status == "failed" or error is not None
    data: dict[str, Any] = {
        "elapsed_ms": int(max(0, int(elapsed_ms))),
        "input_hash": str(input_hash),
    }
    raw_hash = str(input_hash)
    level = "error" if failed else "debug"
    tags = ["planning", "rust", "orchestrator"]
    reason_code = None
    if failed:
        err = error or RuntimeError("rust orchestrator failed")
        tags.append("error")
        reason_code = type(err).__name__
        data.update(
            {
                "error_type": type(err).__name__,
                "error": _sanitize_remote_text(err, max_len=500),
            }
        )
    else:
        if output_hash is not None:
            raw_hash = str(output_hash)
            data["output_hash"] = str(output_hash)
        data["order_count"] = int(order_count or 0)
        data["diagnostic_keys"] = (
            sorted(diagnostics) if isinstance(diagnostics, dict) else []
        )
        if output_order_sample is None and live_event_debug_profile_enabled(bot, "rust"):
            output_order_sample = _best_effort_rust_output_order_debug_sample(
                orders,
                idx_to_symbol=idx_to_symbol,
            )
        if output_order_sample is not None:
            data["debug_profile"] = "rust"
            data["output_order_sample"] = output_order_sample

    bot._emit_live_event(
        EventTypes.RUST_ORCHESTRATOR_RETURNED,
        level=level,
        component="rust_orchestrator",
        tags=tuple(tags),
        cycle_id=current_live_event_cycle_id(bot),
        remote_call_id=rust_call_id,
        status="failed" if failed else "succeeded",
        reason_code=reason_code,
        raw_hash=raw_hash,
        data=data,
    )


def emit_rust_orchestrator_returned_event(bot: Any, *args: Any, **kwargs: Any) -> None:
    try:
        _emit_rust_orchestrator_returned_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug(
            "[event] failed to emit %s: %s",
            EventTypes.RUST_ORCHESTRATOR_RETURNED,
            exc,
        )


def _emit_forager_feature_unavailable_event_unchecked(
    bot: Any,
    *,
    pside: str,
    symbols: list[str] | tuple[str, ...] | set[str],
    candidate_count: int,
    volume_count: int,
    log_range_count: int,
    max_age_ms: int | None,
    fetch_budget: int | None,
) -> None:
    if not symbols:
        return
    data: dict[str, Any] = {
        "candidate_count": int(candidate_count),
        "unavailable": _symbol_sample(symbols),
        "volume_count": int(volume_count),
        "log_range_count": int(log_range_count),
        "max_age_ms": int(max_age_ms) if max_age_ms is not None else None,
        "fetch_budget": int(fetch_budget) if fetch_budget is not None else None,
    }
    if live_event_debug_profile_enabled(bot, "forager"):
        data["debug"] = _forager_debug_payload(data)
        data["debug_profile"] = "forager"
    _safe_emit(
        bot,
        EventTypes.FORAGER_FEATURE_UNAVAILABLE,
        level="debug",
        component="forager.selection",
        tags=(EventTags.FORAGER, EventTags.SELECTION, EventTags.EMA),
        cycle_id=current_live_event_cycle_id(bot),
        pside=str(pside),
        status="skipped",
        reason_code=ReasonCodes.RANKING_FEATURES_UNAVAILABLE,
        data=data,
    )


def emit_forager_feature_unavailable_event(bot: Any, *args: Any, **kwargs: Any) -> None:
    try:
        _emit_forager_feature_unavailable_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug(
            "[event] failed to emit %s: %s",
            EventTypes.FORAGER_FEATURE_UNAVAILABLE,
            exc,
        )


def _emit_forager_eligibility_changed_event_unchecked(
    bot: Any,
    *,
    source: str,
    list_kind: str,
    operation: str,
    changes: dict[str, set[str]],
) -> None:
    if source not in {"config_sources", "live_value"}:
        raise ValueError(f"unsupported forager eligibility source {source!r}")
    if list_kind not in {"approved_coins", "ignored_coins"}:
        raise ValueError(f"unsupported forager eligibility list kind {list_kind!r}")
    if operation not in {"added", "removed"}:
        raise ValueError(f"unsupported forager eligibility operation {operation!r}")
    pside_changes = []
    for pside in ("long", "short"):
        symbols = changes.get(pside)
        if not symbols:
            continue
        ordered_symbols = sorted(str(symbol) for symbol in symbols)
        pside_changes.append(
            {
                "pside": pside,
                "count": len(ordered_symbols),
                "symbols": ordered_symbols[:12],
            }
        )
    if not pside_changes:
        return
    _safe_emit(
        bot,
        EventTypes.FORAGER_ELIGIBILITY_CHANGED,
        level="info",
        component="forager.eligibility",
        tags=(EventTags.FORAGER, EventTags.REFRESH),
        cycle_id=current_live_event_cycle_id(bot),
        status="succeeded",
        reason_code=ReasonCodes.FORAGER_ELIGIBILITY_MEMBERSHIP_CHANGED,
        data={
            "source": source,
            "list_kind": list_kind,
            "operation": operation,
            "changes": pside_changes,
        },
    )


def emit_forager_eligibility_changed_event(
    bot: Any, *args: Any, **kwargs: Any
) -> None:
    try:
        _emit_forager_eligibility_changed_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug(
            "[event] failed to emit %s: %s",
            EventTypes.FORAGER_ELIGIBILITY_CHANGED,
            exc,
        )


_MARKET_COMPATIBILITY_SAMPLE_LIMIT = 12
_MARKET_COMPATIBILITY_SYMBOL_MAX_LEN = 160
_STOCK_PERP_PREFIXES = ("xyz:", "xyz-")
_HIP3_ACCOUNT_MODE_FLUSH_TIMEOUT_S = 0.1


def _bounded_market_symbol_summary(symbols: Any) -> dict[str, Any]:
    raw_symbols = sorted({str(symbol) for symbol in (symbols or []) if symbol})
    safe_symbols = [
        _sanitize_remote_text(
            symbol,
            max_len=_MARKET_COMPATIBILITY_SYMBOL_MAX_LEN,
        )
        for symbol in raw_symbols[:_MARKET_COMPATIBILITY_SAMPLE_LIMIT]
    ]
    return {
        "count": len(raw_symbols),
        "sample": safe_symbols,
        "truncated": len(raw_symbols) > _MARKET_COMPATIBILITY_SAMPLE_LIMIT,
    }


def _flush_live_event_pipeline_after_terminal_emit(bot: Any) -> None:
    pipeline = getattr(bot, "_live_event_pipeline", None)
    flush = getattr(pipeline, "flush", None)
    if not callable(flush):
        return
    try:
        flush(timeout=_HIP3_ACCOUNT_MODE_FLUSH_TIMEOUT_S)
    except Exception as exc:
        logging.debug(
            "[event] terminal market compatibility flush failed: %s",
            type(exc).__name__,
        )


def _emit_hip3_account_mode_unsupported_event_unchecked(
    bot: Any,
    *,
    account_abstraction: Any,
    action: str,
    approved_symbols: Any,
    position_symbols: Any,
    open_order_symbols: Any,
    isolated_only_symbols: Any,
    live_isolated_symbols: Any,
) -> bool:
    emitted = _safe_emit(
        bot,
        EventTypes.CONFIG_MARKET_COMPATIBILITY,
        level="error",
        component="config.market_compatibility",
        tags=(
            EventTags.MARKET,
            EventTags.ACCOUNT,
            EventTags.MODE,
            EventTags.AVAILABILITY,
        ),
        cycle_id=current_live_event_cycle_id(bot),
        status="failed",
        reason_code=ReasonCodes.CONFIG_HIP3_ACCOUNT_MODE_UNSUPPORTED,
        data={
            "account_abstraction": _sanitize_remote_text(
                account_abstraction or "unknown",
                max_len=96,
            ),
            "action": _sanitize_remote_text(action, max_len=96),
            "approved_symbols": _bounded_market_symbol_summary(approved_symbols),
            "position_symbols": _bounded_market_symbol_summary(position_symbols),
            "open_order_symbols": _bounded_market_symbol_summary(open_order_symbols),
            "isolated_only_symbols": _bounded_market_symbol_summary(isolated_only_symbols),
            "live_isolated_symbols": _bounded_market_symbol_summary(live_isolated_symbols),
        },
        require_enqueue=True,
    )
    if emitted is not None:
        _flush_live_event_pipeline_after_terminal_emit(bot)
    return emitted is not None


def emit_hip3_account_mode_unsupported_event(
    bot: Any, *args: Any, **kwargs: Any
) -> bool:
    """Best-effort terminal visibility for unsupported Hyperliquid HIP-3 account state."""
    try:
        return _emit_hip3_account_mode_unsupported_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug(
            "[event] failed to emit HIP-3 account-mode compatibility event: %s",
            type(exc).__name__,
        )
        return False


def _market_compatibility_reason(symbol: str, exchange: str) -> str:
    base = symbol.split("/", 1)[0].casefold()
    if not base.startswith(_STOCK_PERP_PREFIXES):
        return ReasonCodes.CONFIG_MARKET_UNSUPPORTED
    if exchange.casefold() != "hyperliquid":
        return ReasonCodes.CONFIG_STOCK_PERP_WRONG_EXCHANGE
    return ReasonCodes.CONFIG_STOCK_PERP_UNAVAILABLE_MARKET


def _emit_config_market_compatibility_event_unchecked(
    bot: Any,
    *,
    list_kind: str,
    pside: str,
    skipped_symbols: set[str],
) -> bool:
    if list_kind not in {"approved_coins", "ignored_coins"}:
        raise ValueError(f"unsupported configured market list kind {list_kind!r}")
    if pside not in {"long", "short"}:
        raise ValueError(f"unsupported configured market pside {pside!r}")
    symbols = sorted({str(symbol) for symbol in skipped_symbols if symbol})
    if not symbols:
        return False
    safe_symbols = {
        symbol: _sanitize_remote_text(
            symbol,
            max_len=_MARKET_COMPATIBILITY_SYMBOL_MAX_LEN,
        )
        for symbol in symbols
    }
    exchange = str(getattr(bot, "exchange", "") or "")
    by_reason: dict[str, list[str]] = {}
    for symbol in symbols:
        reason_code = _market_compatibility_reason(symbol, exchange)
        by_reason.setdefault(reason_code, []).append(symbol)
    reason_samples = [
        {
            "reason_code": reason_code,
            "count": len(reason_symbols),
            "symbols": [
                safe_symbols[symbol]
                for symbol in reason_symbols[:_MARKET_COMPATIBILITY_SAMPLE_LIMIT]
            ],
            "symbols_truncated": (
                len(reason_symbols) > _MARKET_COMPATIBILITY_SAMPLE_LIMIT
            ),
        }
        for reason_code, reason_symbols in sorted(by_reason.items())
    ]
    event_reason = (
        next(iter(by_reason))
        if len(by_reason) == 1
        else ReasonCodes.CONFIG_MARKET_UNSUPPORTED
    )
    return (
        _safe_emit(
            bot,
            EventTypes.CONFIG_MARKET_COMPATIBILITY,
            level="info",
            component="config.market_compatibility",
            tags=(EventTags.MARKET, EventTags.AVAILABILITY),
            cycle_id=current_live_event_cycle_id(bot),
            pside=pside,
            status="degraded" if list_kind == "approved_coins" else "skipped",
            reason_code=event_reason,
            data={
                "list_kind": list_kind,
                "skipped_count": len(symbols),
                "skipped_symbols": [
                    safe_symbols[symbol]
                    for symbol in symbols[:_MARKET_COMPATIBILITY_SAMPLE_LIMIT]
                ],
                "skipped_symbols_truncated": (
                    len(symbols) > _MARKET_COMPATIBILITY_SAMPLE_LIMIT
                ),
                "reason_counts": {
                    reason_code: len(reason_symbols)
                    for reason_code, reason_symbols in sorted(by_reason.items())
                },
                "reason_samples": reason_samples,
            },
            require_enqueue=True,
        )
        is not None
    )


def emit_config_market_compatibility_event(
    bot: Any, *args: Any, **kwargs: Any
) -> bool:
    """Best-effort visibility for configured symbols skipped before eligibility."""
    try:
        return _emit_config_market_compatibility_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug(
            "[event] failed to emit %s: %s",
            EventTypes.CONFIG_MARKET_COMPATIBILITY,
            exc,
        )
        return False


def _emit_isolated_only_market_blocked_event_unchecked(
    bot: Any,
    *,
    pside: str,
    blocked_symbols: set[str],
) -> bool:
    if pside not in {"long", "short"}:
        raise ValueError(f"unsupported configured market pside {pside!r}")
    symbols = sorted({str(symbol) for symbol in blocked_symbols if symbol})
    if not symbols:
        return False
    safe_symbols = {
        symbol: _sanitize_remote_text(
            symbol,
            max_len=_MARKET_COMPATIBILITY_SYMBOL_MAX_LEN,
        )
        for symbol in symbols
    }
    return (
        _safe_emit(
            bot,
            EventTypes.CONFIG_MARKET_COMPATIBILITY,
            level="info",
            component="config.market_compatibility",
            tags=(EventTags.MARKET, EventTags.MODE, EventTags.AVAILABILITY),
            cycle_id=current_live_event_cycle_id(bot),
            pside=pside,
            status="degraded",
            reason_code=ReasonCodes.CONFIG_ISOLATED_ONLY_MARKET_BLOCKED,
            data={
                "action": "initial_entries_blocked",
                "margin_mode_preference": "cross",
                "capability": "isolated_only",
                "blocked_count": len(symbols),
                "blocked_symbols": [
                    safe_symbols[symbol]
                    for symbol in symbols[:_MARKET_COMPATIBILITY_SAMPLE_LIMIT]
                ],
                "blocked_symbols_truncated": (
                    len(symbols) > _MARKET_COMPATIBILITY_SAMPLE_LIMIT
                ),
            },
            require_enqueue=True,
        )
        is not None
    )


def emit_isolated_only_market_blocked_event(
    bot: Any, *args: Any, **kwargs: Any
) -> bool:
    """Best-effort visibility for isolated-only symbols blocked from new entries."""
    try:
        return _emit_isolated_only_market_blocked_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug(
            "[event] failed to emit isolated-only market compatibility event: %s",
            type(exc).__name__,
        )
        return False


def _emit_forager_selection_event_unchecked(
    bot: Any,
    *,
    pside: str,
    candidate_count: int,
    eligible_count: int,
    selected_symbols: list[str] | tuple[str, ...],
    slots_open: bool,
    max_n_positions: int | None,
    clip_pct: float | None,
    volatility_drop_pct: float | None,
    max_age_ms: int | None,
    fetch_budget: int | None,
    incumbent_symbols: list[str] | tuple[str, ...] | None = None,
    slots_to_fill: int | None = None,
    score_hysteresis_pct: float | None = None,
    top_scores: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    hysteresis_event_count: int = 0,
    source: str = "python_filter",
    feature_unavailable_count: int = 0,
    volatility_dropped_count: int = 0,
    reason_code: str = "selected",
    status: str = "succeeded",
) -> None:
    selected = [str(symbol) for symbol in selected_symbols or []]
    incumbent = [str(symbol) for symbol in incumbent_symbols or []]
    data: dict[str, Any] = {
        "candidate_count": int(candidate_count),
        "eligible_count": int(eligible_count),
        "selected_count": len(selected),
        "selected_symbols": selected[:12],
        "incumbent_count": len(incumbent),
        "incumbent_symbols": incumbent[:12],
        "slots_open": bool(slots_open),
        "max_n_positions": int(max_n_positions) if max_n_positions is not None else None,
        "slots_to_fill": int(slots_to_fill) if slots_to_fill is not None else None,
        "clip_pct": float(clip_pct) if clip_pct is not None else None,
        "volatility_drop_pct": (
            float(volatility_drop_pct)
            if volatility_drop_pct is not None
            else None
        ),
        "score_hysteresis_pct": (
            float(score_hysteresis_pct)
            if score_hysteresis_pct is not None
            else None
        ),
        "max_age_ms": int(max_age_ms) if max_age_ms is not None else None,
        "fetch_budget": int(fetch_budget) if fetch_budget is not None else None,
        "source": str(source),
        "feature_unavailable_count": int(feature_unavailable_count),
        "volatility_dropped_count": int(volatility_dropped_count),
        "hysteresis_event_count": int(hysteresis_event_count),
        "top_scores": _forager_top_score_sample(top_scores),
    }
    if live_event_debug_profile_enabled(bot, "forager"):
        data["debug"] = _forager_debug_payload(data)
        data["debug_profile"] = "forager"
    _safe_emit(
        bot,
        EventTypes.FORAGER_SELECTION,
        level="info",
        component="forager.selection",
        tags=(EventTags.FORAGER, EventTags.SELECTION),
        cycle_id=current_live_event_cycle_id(bot),
        pside=str(pside),
        status=status,
        reason_code=reason_code,
        data=data,
    )


def emit_forager_selection_event(bot: Any, *args: Any, **kwargs: Any) -> None:
    try:
        _emit_forager_selection_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug(
            "[event] failed to emit %s: %s",
            EventTypes.FORAGER_SELECTION,
            exc,
        )


def _emit_candle_tail_projected_event_unchecked(
    bot: Any,
    *,
    symbol: str,
    context: dict[str, Any] | None,
    reason_code: str = ReasonCodes.OPEN_TAIL_PROJECTION,
) -> None:
    ctx = dict(context or {})
    data: dict[str, Any] = {"timeframe": str(ctx.get("timeframe") or "1m")}
    for key in (
        "latest_expected_ts",
        "last_cached_ts",
        "tail_gap_age_ms",
        "tail_gap_candles",
        "missing_candles",
        "max_tail_gap_ms",
    ):
        value = _safe_int(ctx.get(key))
        if value is not None:
            data[key] = value
    reason = ctx.get("reason")
    if reason is not None:
        data["projection_reason"] = str(reason)
    if live_event_debug_profile_enabled(bot, "candles"):
        data["debug_profile"] = "candles"
        data["debug"] = _candle_debug_payload(data, context=ctx)
    _safe_emit(
        bot,
        EventTypes.CANDLE_TAIL_PROJECTED,
        level="debug",
        component="candle.tail_projection",
        tags=(EventTags.CANDLE, EventTags.TAIL, EventTags.EMA),
        cycle_id=current_live_event_cycle_id(bot),
        symbol=str(symbol),
        status="recovered",
        reason_code=str(reason_code),
        data=data,
    )


def emit_candle_tail_projected_event(bot: Any, *args: Any, **kwargs: Any) -> None:
    try:
        _emit_candle_tail_projected_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug(
            "[event] failed to emit %s: %s",
            EventTypes.CANDLE_TAIL_PROJECTED,
            exc,
        )


def _missing_span_preview(
    spans: Any,
    *,
    timeframe_ms: int,
    limit: int = 3,
) -> list[dict[str, int]]:
    preview: list[dict[str, int]] = []
    try:
        iterable = list(spans or [])
    except TypeError:
        iterable = []
    step_ms = max(1, int(timeframe_ms or 1))
    for item in iterable[: max(0, int(limit))]:
        try:
            start_ts, end_ts = item
        except Exception:
            continue
        start_int = _safe_int(start_ts)
        end_int = _safe_int(end_ts)
        if start_int is None or end_int is None:
            continue
        preview.append(
            {
                "start_ts": int(start_int),
                "end_ts": int(end_int),
                "candles": int(max(0, (int(end_int) - int(start_int)) // step_ms) + 1),
            }
        )
    return preview


def _timeframe_ms(timeframe: str) -> int:
    text = str(timeframe or "1m").strip().lower()
    try:
        unit = text[-1]
        value = int(float(text[:-1] or "1"))
    except Exception:
        return 60_000
    if unit == "s":
        return max(1, value * 1_000)
    if unit == "m":
        return max(1, value * 60_000)
    if unit == "h":
        return max(1, value * 3_600_000)
    if unit == "d":
        return max(1, value * 86_400_000)
    return 60_000


def _emit_candle_coverage_checked_event_unchecked(
    bot: Any,
    *,
    symbol: str,
    timeframe: str,
    start_ts: int,
    end_ts: int,
    report: dict[str, Any] | None,
    context: str = "required_disk_audit",
    required: bool = True,
) -> None:
    report_in = dict(report or {})
    tf = str(report_in.get("timeframe") or timeframe or "1m")
    ok = bool(report_in.get("ok", False))
    missing_spans = report_in.get("missing_spans") or []
    try:
        missing_span_count = len(missing_spans)
    except Exception:
        missing_span_count = 0
    tf_ms = _timeframe_ms(tf)
    data: dict[str, Any] = {
        "context": str(context),
        "timeframe": tf,
        "required": bool(required),
        "coverage_ok": ok,
        "missing_span_count": int(missing_span_count),
        "missing_spans_preview": _missing_span_preview(
            missing_spans,
            timeframe_ms=tf_ms,
        ),
    }
    for key, value in (
        ("start_ts", start_ts),
        ("end_ts", end_ts),
        ("missing_candles", report_in.get("missing_candles")),
        ("loaded_rows", report_in.get("loaded_rows")),
    ):
        safe = _safe_int(value)
        if safe is not None:
            data[key] = int(safe)
    if live_event_debug_profile_enabled(bot, "candles"):
        data["debug_profile"] = "candles"
        data["debug"] = _candle_debug_payload(
            data,
            report=report_in,
            timeframe_ms=tf_ms,
        )
    _safe_emit(
        bot,
        EventTypes.CANDLE_COVERAGE_CHECKED,
        level="debug" if ok or not required else "warning",
        component="candle.coverage",
        tags=(EventTags.CANDLE, EventTags.COVERAGE, EventTags.CACHE),
        cycle_id=current_live_event_cycle_id(bot),
        symbol=str(symbol),
        status="succeeded" if ok else ("degraded" if required else "skipped"),
        reason_code=ReasonCodes.REQUIRED_CANDLE_DISK_COVERAGE,
        data=data,
    )


def emit_candle_coverage_checked_event(bot: Any, *args: Any, **kwargs: Any) -> None:
    try:
        _emit_candle_coverage_checked_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug(
            "[event] failed to emit %s: %s",
            EventTypes.CANDLE_COVERAGE_CHECKED,
            exc,
        )


def _reason_counts(data: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in dict(data or {}).items():
        count = _safe_int(value)
        if count is not None:
            out[str(key)] = int(count)
    return dict(sorted(out.items()))


def _cache_debug_payload(
    payload: dict[str, Any],
    *,
    data: dict[str, Any],
    event_kind: str,
    limit: int = 32,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "event_kind": str(event_kind),
        "payload_keys": sorted(str(key) for key in payload)[:limit],
        "data_keys": sorted(str(key) for key in data)[:limit],
    }
    numeric_keys: list[str] = []
    nonzero_numeric_keys: list[str] = []
    for key in sorted(data):
        parsed = _safe_int(data.get(key))
        if parsed is None:
            continue
        numeric_keys.append(str(key))
        if parsed != 0:
            nonzero_numeric_keys.append(str(key))
    if numeric_keys:
        out["numeric_keys"] = numeric_keys[:limit]
    if nonzero_numeric_keys:
        out["nonzero_numeric_keys"] = nonzero_numeric_keys[:limit]
    source_days = data.get("source_days")
    if isinstance(source_days, dict) and source_days:
        out["source_day_sources"] = sorted(str(key) for key in source_days)[:limit]
        out["source_day_total"] = int(
            sum(
                int(value)
                for value in source_days.values()
                if _safe_int(value) is not None
            )
        )
    reason_counts = data.get("reason_counts")
    if isinstance(reason_counts, dict) and reason_counts:
        out["reason_count_keys"] = sorted(str(key) for key in reason_counts)[:limit]
        out["reason_count_total"] = int(
            sum(
                int(value)
                for value in reason_counts.values()
                if _safe_int(value) is not None
            )
        )
    return {key: value for key, value in out.items() if value not in (None, {}, [])}


def _emit_cache_load_completed_event_unchecked(
    bot: Any,
    payload: dict[str, Any],
) -> None:
    data_in = dict(payload or {})
    symbol = data_in.get("symbol")
    timeframe = str(data_in.get("timeframe") or data_in.get("tf") or "1m")
    data: dict[str, Any] = {"timeframe": timeframe}
    for key in (
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
    ):
        value = _safe_int(data_in.get(key))
        if value is not None:
            data[key] = value
    source_days = data_in.get("source_days")
    if isinstance(source_days, dict):
        clean_source_days: dict[str, int] = {}
        for key, value in source_days.items():
            count = _safe_int(value)
            if count is not None:
                clean_source_days[str(key)] = int(count)
        if clean_source_days:
            data["source_days"] = dict(sorted(clean_source_days.items()))
    if live_event_debug_profile_enabled(bot, "cache"):
        data["debug_profile"] = "cache"
        data["debug"] = _cache_debug_payload(
            data_in,
            data=data,
            event_kind="load_completed",
        )
    _safe_emit(
        bot,
        EventTypes.CACHE_LOAD_COMPLETED,
        level="debug",
        component="cache.candles",
        tags=(EventTags.CACHE, EventTags.CANDLE, EventTags.LOAD),
        cycle_id=current_live_event_cycle_id(bot),
        symbol=str(symbol) if symbol is not None else None,
        status="succeeded",
        reason_code=ReasonCodes.CANDLE_DISK_LOAD_COMPLETED,
        data=data,
    )


def emit_cache_load_completed_event(bot: Any, *args: Any, **kwargs: Any) -> None:
    try:
        _emit_cache_load_completed_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug(
            "[event] failed to emit %s: %s",
            EventTypes.CACHE_LOAD_COMPLETED,
            exc,
        )


def _emit_cache_flush_completed_event_unchecked(
    bot: Any,
    payload: dict[str, Any],
) -> None:
    data_in = dict(payload or {})
    symbol = data_in.get("symbol")
    timeframe = str(data_in.get("timeframe") or data_in.get("tf") or "1m")
    data: dict[str, Any] = {"timeframe": timeframe}
    for key in (
        "persisted_rows",
        "persisted_start_ts",
        "persisted_end_ts",
        "suppressed_count",
        "suppressed_rows",
    ):
        value = _safe_int(data_in.get(key))
        if value is not None:
            data[key] = value
    if live_event_debug_profile_enabled(bot, "cache"):
        data["debug_profile"] = "cache"
        data["debug"] = _cache_debug_payload(
            data_in,
            data=data,
            event_kind="flush_completed",
        )
    _safe_emit(
        bot,
        EventTypes.CACHE_FLUSH_COMPLETED,
        level="debug",
        component="cache.candles",
        tags=(EventTags.CACHE, EventTags.CANDLE, EventTags.FLUSH),
        cycle_id=current_live_event_cycle_id(bot),
        symbol=str(symbol) if symbol is not None else None,
        status="succeeded",
        reason_code=ReasonCodes.CANDLE_DISK_FLUSH_COMPLETED,
        data=data,
    )


def emit_cache_flush_completed_event(bot: Any, *args: Any, **kwargs: Any) -> None:
    try:
        _emit_cache_flush_completed_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug(
            "[event] failed to emit %s: %s",
            EventTypes.CACHE_FLUSH_COMPLETED,
            exc,
        )


def _emit_cache_warmup_decision_event_unchecked(
    bot: Any,
    *,
    context: str,
    timeframe: str = "1m",
    symbol_count: int,
    reused_count: int,
    cold_count: int,
    reason_counts: dict[str, int] | None = None,
    elapsed_ms: int | None = None,
    concurrency: int | None = None,
    ttl_ms: int | None = None,
    window_min_candles: int | None = None,
    window_max_candles: int | None = None,
) -> None:
    reused = max(0, int(reused_count))
    cold = max(0, int(cold_count))
    data: dict[str, Any] = {
        "context": str(context),
        "timeframe": str(timeframe or "1m"),
        "symbol_count": max(0, int(symbol_count)),
        "reused_count": reused,
        "cold_count": cold,
        "cold_path_required": cold > 0,
        "reason_counts": _reason_counts(reason_counts),
    }
    for key, value in (
        ("elapsed_ms", elapsed_ms),
        ("concurrency", concurrency),
        ("ttl_ms", ttl_ms),
        ("window_min_candles", window_min_candles),
        ("window_max_candles", window_max_candles),
    ):
        safe = _safe_int(value)
        if safe is not None:
            data[key] = safe
    if live_event_debug_profile_enabled(bot, "cache"):
        data["debug_profile"] = "cache"
        data["debug"] = _cache_debug_payload(
            {},
            data=data,
            event_kind="warmup_decision",
        )
    _safe_emit(
        bot,
        EventTypes.CACHE_WARMUP_DECISION,
        level="debug",
        component="cache.warmup",
        tags=(EventTags.CACHE, EventTags.WARMUP, EventTags.CANDLE),
        cycle_id=current_live_event_cycle_id(bot),
        status="succeeded",
        reason_code=ReasonCodes.WARMUP_CACHE_DECISION,
        data=data,
    )


def emit_cache_warmup_decision_event(bot: Any, *args: Any, **kwargs: Any) -> None:
    try:
        _emit_cache_warmup_decision_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug(
            "[event] failed to emit %s: %s",
            EventTypes.CACHE_WARMUP_DECISION,
            exc,
        )


def _mode_counts(modes: dict[str, dict[str, str]] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol_modes in (modes or {}).values():
        if not isinstance(symbol_modes, dict):
            continue
        for mode in symbol_modes.values():
            key = str(mode or "")
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _emit_ema_bundle_started_event_unchecked(
    bot: Any,
    *,
    symbols: list[str] | tuple[str, ...],
    modes: dict[str, dict[str, str]] | None = None,
) -> None:
    _safe_emit(
        bot,
        EventTypes.EMA_BUNDLE_STARTED,
        level="debug",
        component="ema.bundle",
        tags=(EventTags.EMA, EventTags.BUNDLE),
        cycle_id=current_live_event_cycle_id(bot),
        status="started",
        reason_code="orchestrator_ema_bundle",
        data={
            "symbol_count": len(symbols or []),
            "symbols": _symbol_sample(symbols or ()),
            "mode_counts": _mode_counts(modes),
        },
    )


def emit_ema_bundle_started_event(bot: Any, *args: Any, **kwargs: Any) -> None:
    try:
        _emit_ema_bundle_started_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug(
            "[event] failed to emit %s: %s",
            EventTypes.EMA_BUNDLE_STARTED,
            exc,
        )


def _emit_ema_bundle_completed_event_unchecked(
    bot: Any,
    *,
    symbols: list[str] | tuple[str, ...],
    m1_close_emas: dict[str, dict[float, float]],
    m1_volume_emas: dict[str, dict[float, float]],
    m1_log_range_emas: dict[str, dict[float, float]],
    h1_log_range_emas: dict[str, dict[float, float]],
    cache_only_symbols: set[str] | None = None,
    projection_contexts: dict[str, dict] | None = None,
) -> None:
    _safe_emit(
        bot,
        EventTypes.EMA_BUNDLE_COMPLETED,
        level="debug",
        component="ema.bundle",
        tags=(EventTags.EMA, EventTags.BUNDLE),
        cycle_id=current_live_event_cycle_id(bot),
        status="succeeded",
        data={
            "symbol_count": len(symbols or []),
            "cache_only": _symbol_sample(cache_only_symbols or set()),
            "projection_contexts": _symbol_sample((projection_contexts or {}).keys()),
            "m1_close": _ema_map_summary(m1_close_emas),
            "m1_volume": _ema_map_summary(m1_volume_emas),
            "m1_log_range": _ema_map_summary(m1_log_range_emas),
            "h1_log_range": _ema_map_summary(h1_log_range_emas),
        },
    )


def emit_ema_bundle_completed_event(bot: Any, *args: Any, **kwargs: Any) -> None:
    try:
        _emit_ema_bundle_completed_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug(
            "[event] failed to emit %s: %s",
            EventTypes.EMA_BUNDLE_COMPLETED,
            exc,
        )


def _emit_ema_fallback_used_event_unchecked(
    bot: Any,
    *,
    close_ema_recoveries: dict[str, list[tuple[float, int]]] | None = None,
    close_ema_fallbacks: dict[str, list[tuple[float, int, int, str]]] | None = None,
    forager_cached_ema_fallbacks: dict[str, list[tuple[str, float, int]]] | None = None,
) -> None:
    recovered_count = sum(len(items) for items in (close_ema_recoveries or {}).values())
    close_fallback_count = sum(len(items) for items in (close_ema_fallbacks or {}).values())
    forager_count = sum(len(items) for items in (forager_cached_ema_fallbacks or {}).values())
    if not (recovered_count or close_fallback_count or forager_count):
        return
    level = "warning" if close_fallback_count else "debug"
    _safe_emit(
        bot,
        EventTypes.EMA_FALLBACK_USED,
        level=level,
        component="ema.bundle",
        tags=(EventTags.EMA, EventTags.FALLBACK),
        cycle_id=current_live_event_cycle_id(bot),
        status="recovered",
        reason_code=ReasonCodes.EMA_FALLBACK_USED,
        data={
            "close_recovered_count": int(recovered_count),
            "close_recovered_symbols": _symbol_sample((close_ema_recoveries or {}).keys()),
            "close_fallback_count": int(close_fallback_count),
            "close_fallback_symbols": _symbol_sample((close_ema_fallbacks or {}).keys()),
            "forager_cached_fallback_count": int(forager_count),
            "forager_cached_fallback_symbols": _symbol_sample(
                (forager_cached_ema_fallbacks or {}).keys()
            ),
            "examples": {
                "close_recovered": _fallback_examples(close_ema_recoveries),
                "close_fallback": _fallback_examples(close_ema_fallbacks),
                "forager_cached": _fallback_examples(forager_cached_ema_fallbacks),
            },
        },
    )


def emit_ema_fallback_used_event(bot: Any, *args: Any, **kwargs: Any) -> None:
    try:
        _emit_ema_fallback_used_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug(
            "[event] failed to emit %s: %s",
            EventTypes.EMA_FALLBACK_USED,
            exc,
        )


def _emit_ema_unavailable_event_unchecked(
    bot: Any,
    *,
    optional_ema_drops: dict[tuple[str, str], list[tuple[str, float]]] | None = None,
    candidate_ema_unavailable_details: dict[str, list[tuple[str, str, str]]] | None = None,
    ema_unavailable_reasons: dict[str, list[str]] | None = None,
) -> None:
    optional_count = sum(len(items) for items in (optional_ema_drops or {}).values())
    candidate_symbols = {
        str(symbol)
        for items in (candidate_ema_unavailable_details or {}).values()
        for symbol, _error_type, _error in items
    }
    unavailable_symbols = {
        str(symbol)
        for items in (ema_unavailable_reasons or {}).values()
        for symbol in items
    }
    if not (optional_count or candidate_symbols or unavailable_symbols):
        return
    level = "warning" if candidate_symbols else "debug"
    status = "degraded" if candidate_symbols or unavailable_symbols else "skipped"
    reason_code = (
        ReasonCodes.REQUIRED_EMA_UNAVAILABLE
        if candidate_symbols or unavailable_symbols
        else ReasonCodes.OPTIONAL_EMA_DROPPED
    )
    optional_summary = []
    for (ema_type, reason), items in sorted((optional_ema_drops or {}).items())[:8]:
        optional_summary.append(
            {
                "ema_type": str(ema_type),
                "reason": str(reason)[:160],
                "symbols": _symbol_sample(symbol for symbol, _span in items),
                "spans": _span_sample(span for _symbol, span in items),
            }
        )
    data = {
        "optional_drop_count": int(optional_count),
        "optional_drop_groups": optional_summary,
        "candidate_unavailable": _symbol_sample(candidate_symbols),
        "candidate_unavailable_groups": _candidate_unavailable_summary(
            candidate_ema_unavailable_details
        ),
        "unavailable": _symbol_sample(unavailable_symbols),
        "unavailable_reasons": _reason_symbol_summary(ema_unavailable_reasons),
    }
    if live_event_debug_profile_enabled(bot, "ema"):
        data["debug_profile"] = "ema"
        data["debug"] = _ema_unavailable_debug_summary(
            candidate_ema_unavailable_details,
            ema_unavailable_reasons,
        )
    _safe_emit(
        bot,
        EventTypes.EMA_UNAVAILABLE,
        level=level,
        component="ema.bundle",
        tags=(EventTags.EMA, EventTags.UNAVAILABLE),
        cycle_id=current_live_event_cycle_id(bot),
        status=status,
        reason_code=reason_code,
        data=data,
    )


def emit_ema_unavailable_event(bot: Any, *args: Any, **kwargs: Any) -> None:
    try:
        _emit_ema_unavailable_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug(
            "[event] failed to emit %s: %s",
            EventTypes.EMA_UNAVAILABLE,
            exc,
        )


def _short_order_id(value: Any, *, max_len: int = 32) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= max_len:
        return text
    keep = max(4, (max_len - 3) // 2)
    return f"{text[:keep]}...{text[-keep:]}"


def _event_fingerprint(value: Any) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _planned_order_pside(order_type: str) -> str | None:
    text = str(order_type or "")
    if text.endswith("_long") or "_long_" in text:
        return "long"
    if text.endswith("_short") or "_short_" in text:
        return "short"
    return None


def _planned_action_summary(
    orders: Any,
    idx_to_symbol: dict[int, str] | None,
    *,
    limit: int = 16,
) -> dict[str, Any]:
    rows = list(orders or []) if isinstance(orders, (list, tuple)) else []
    symbols_by_idx = idx_to_symbol or {}
    by_order_type: Counter[str] = Counter()
    by_pside: Counter[str] = Counter()
    by_execution_type: Counter[str] = Counter()
    symbols: set[str] = set()
    sample: list[dict[str, Any]] = []
    for index, order in enumerate(rows):
        if not isinstance(order, dict):
            by_order_type[type(order).__name__] += 1
            continue
        order_type = str(order.get("order_type") or "unknown")
        execution_type = str(order.get("execution_type") or "limit")
        pside = _planned_order_pside(order_type)
        by_order_type[order_type] += 1
        by_execution_type[execution_type] += 1
        if pside:
            by_pside[pside] += 1
        symbol = None
        symbol_idx = _safe_int(order.get("symbol_idx"))
        if symbol_idx is not None:
            symbol = symbols_by_idx.get(symbol_idx)
        if symbol:
            symbols.add(str(symbol))
        if len(sample) >= max(0, int(limit)):
            continue
        item: dict[str, Any] = {
            "index": int(index),
            "symbol": str(symbol) if symbol else None,
            "symbol_idx": symbol_idx,
            "pside": pside,
            "order_type": order_type,
            "execution_type": execution_type,
        }
        qty = _safe_float(order.get("qty"))
        price = _safe_float(order.get("price"))
        if qty is not None:
            item["qty"] = qty
        if price is not None:
            item["price"] = price
        sample.append({key: value for key, value in item.items() if value is not None})
    return {
        "order_count": len(rows),
        "by_order_type": dict(sorted(by_order_type.items())),
        "by_pside": dict(sorted(by_pside.items())),
        "by_execution_type": dict(sorted(by_execution_type.items())),
        "symbols": _symbol_sample(symbols),
        "orders_sample": sample,
        "orders_sample_count": len(sample),
        "orders_truncated": len(rows) > len(sample),
    }


def emit_action_planned_event(
    bot: Any,
    *,
    orders: Any,
    idx_to_symbol: dict[int, str] | None = None,
    output_hash: str | None = None,
    remote_call_id: str | None = None,
) -> None:
    try:
        data = _planned_action_summary(orders, idx_to_symbol)
        if output_hash:
            data["output_hash"] = str(output_hash)
        bot._emit_live_event(
            EventTypes.ACTION_PLANNED,
            level="debug",
            component="action_planner",
            tags=(EventTags.PLANNING, EventTags.ACTION, EventTags.RUST),
            cycle_id=current_live_event_cycle_id(bot),
            remote_call_id=remote_call_id,
            status="succeeded",
            reason_code=ReasonCodes.RUST_OUTPUT_ACTIONS,
            raw_hash=output_hash,
            data=data,
        )
    except Exception as exc:
        logging.debug(
            "[event] failed to emit %s: %s",
            EventTypes.ACTION_PLANNED,
            exc,
        )


def emit_initial_entry_distance_gate_event(
    bot: Any,
    *,
    event_type: str,
    status: str,
    action: str,
    order: dict,
    market_price: float,
    signed_dist: float,
    threshold: float,
    tolerance: float | None = None,
    operator_visible: bool = True,
    active_count: int | None = None,
    suppressed_count: int | None = None,
) -> None:
    try:
        symbol = str(order.get("symbol") or "")
        pside = str(order.get("position_side") or "")
        side = str(order.get("side") or "")
        data = {
            "qty": _safe_float(order.get("qty")),
            "price": _safe_float(order.get("price")),
            "market_price": _safe_float(market_price),
            "distance_pct": _safe_float(float(signed_dist) * 100.0),
            "threshold_pct": _safe_float(float(threshold) * 100.0),
            "action": str(action),
            "operator_visible": bool(operator_visible),
            "order_type": str(
                order.get("pb_order_type") or order.get("type") or "unknown"
            ),
        }
        if tolerance is not None:
            data["tolerance_pct"] = _safe_float(float(tolerance) * 100.0)
        if active_count is not None:
            data["active_count"] = max(0, min(int(active_count), 999))
        if suppressed_count is not None:
            data["suppressed_count"] = max(0, min(int(suppressed_count), 999))
        bot._emit_live_event(
            event_type,
            level="info",
            component="entry.initial_distance_gate",
            tags=(EventTags.ORDER, EventTags.GATE, EventTags.ACTION),
            cycle_id=current_live_event_cycle_id(bot),
            symbol=symbol,
            pside=pside,
            side=side,
            status=status,
            reason_code=ReasonCodes.INITIAL_ENTRY_DISTANCE_GATE,
            data={key: value for key, value in data.items() if value is not None},
        )
    except Exception as exc:
        logging.debug(
            "[event] failed to emit initial entry distance gate event type=%s symbol=%s: %s",
            event_type,
            order.get("symbol") if isinstance(order, dict) else None,
            exc,
        )


def _order_event_data(order: dict | None, *, index: int | None = None) -> dict[str, Any]:
    if not isinstance(order, dict):
        return {"order_type": type(order).__name__}
    data: dict[str, Any] = {
        "index": index,
        "pb_order_type": order.get("pb_order_type"),
        "order_type": str(order.get("type") or order.get("execution_type") or "limit"),
        "context": order.get("_context"),
        "reason": order.get("_reason"),
        "reduce_only": bool(order.get("reduce_only") or order.get("reduceOnly")),
    }
    price = _safe_float(order.get("price"))
    qty = _safe_float(order.get("qty"))
    if price is not None:
        data["price"] = price
    if qty is not None:
        data["qty"] = qty
    custom_id = order.get("custom_id") or order.get("clientOrderId")
    if custom_id is not None:
        data["client_order_id_short"] = _short_order_id(custom_id)
    exchange_id = order.get("id") or order.get("order_id")
    if exchange_id is not None:
        data["order_id_short"] = _short_order_id(exchange_id)
    delta = order.get("_delta")
    if isinstance(delta, dict):
        compact_delta = {
            key: delta.get(key)
            for key in ("price_pct_diff", "qty_pct_diff")
            if delta.get(key) is not None
        }
        if compact_delta:
            data["delta"] = compact_delta
    return {key: value for key, value in data.items() if value is not None}


def _execution_event_ids(
    wave: dict | None,
    *,
    action: str,
    index: int | None,
) -> tuple[str | None, str | None]:
    order_wave_id = None
    action_id = None
    if isinstance(wave, dict):
        order_wave_id = str(wave.get("event_id") or f"ow_{wave.get('id', '')}")
        if index is not None:
            action_id = f"{order_wave_id}:{action}:{int(index)}"
    return order_wave_id, action_id


def _execution_connector_call_context(
    bot: Any,
    order: dict | None,
    *,
    action: str,
) -> tuple[dict | None, int | None]:
    wave = getattr(bot, "_order_wave_in_progress", None)
    index = None
    context = getattr(bot, "_execution_connector_call_context", None)
    if isinstance(context, dict) and context.get("action") == action:
        if isinstance(context.get("wave"), dict):
            wave = context["wave"]
        for candidate_index, candidate in enumerate(context.get("orders") or []):
            if candidate is order:
                index = candidate_index
                break
    return wave, index


def emit_execution_connector_call_started_event(
    bot: Any,
    *,
    order: dict | None,
    action: str,
    connector_route: str,
) -> None:
    """Emit bounded evidence immediately before a concrete connector call."""
    try:
        event_types = {
            "create": EventTypes.EXECUTION_CREATE_CONNECTOR_CALL_STARTED,
            "cancel": EventTypes.EXECUTION_CANCEL_CONNECTOR_CALL_STARTED,
        }
        connector_methods = {
            "create": "cca.create_order",
            "cancel": "cca.cancel_order",
        }
        allowed_routes = {"base", "hyperliquid", "okx"}
        if action not in event_types:
            raise ValueError(f"unsupported connector action: {action}")
        if connector_route not in allowed_routes:
            raise ValueError(f"unsupported connector route: {connector_route}")

        wave, index = _execution_connector_call_context(
            bot,
            order,
            action=action,
        )
        order_wave_id, action_id = _execution_event_ids(
            wave,
            action=action,
            index=index,
        )
        order_data = _order_event_data(order, index=index)
        data: dict[str, Any] = {
            "action": action,
            "connector_method": connector_methods[action],
            "connector_route": connector_route,
        }
        if index is not None:
            data["index"] = int(index)
        for key in ("pb_order_type", "order_type"):
            if order_data.get(key) is not None:
                data[key] = str(order_data[key])[:64]
        if "reduce_only" in order_data:
            data["reduce_only"] = bool(order_data["reduce_only"])
        for key in ("client_order_id_short", "order_id_short"):
            if order_data.get(key) is not None:
                data[key] = str(order_data[key])[:64]
        for key in ("price", "qty"):
            if (safe_value := _safe_finite_float(order_data.get(key))) is not None:
                data[key] = safe_value
        delta = order_data.get("delta")
        if isinstance(delta, dict):
            safe_delta = {
                key: safe_value
                for key in ("price_pct_diff", "qty_pct_diff")
                if (safe_value := _safe_finite_float(delta.get(key))) is not None
            }
            if safe_delta:
                data["delta"] = safe_delta
        _safe_emit(
            bot,
            event_types[action],
            level="debug",
            component="execution.connector_call",
            tags=(EventTags.EXECUTION, EventTags.ORDER, action),
            cycle_id=current_live_event_cycle_id(bot),
            order_wave_id=order_wave_id,
            action_id=action_id,
            symbol=(
                str(order.get("symbol"))
                if isinstance(order, dict) and order.get("symbol")
                else None
            ),
            pside=(
                str(order.get("position_side"))
                if isinstance(order, dict) and order.get("position_side")
                else None
            ),
            side=(
                str(order.get("side"))
                if isinstance(order, dict) and order.get("side")
                else None
            ),
            order_id=(
                str(order.get("id") or order.get("order_id"))
                if isinstance(order, dict)
                and (order.get("id") or order.get("order_id"))
                else None
            ),
            client_order_id=(
                str(order.get("custom_id") or order.get("clientOrderId"))
                if isinstance(order, dict)
                and (order.get("custom_id") or order.get("clientOrderId"))
                else None
            ),
            status="started",
            reason_code=ReasonCodes.CONNECTOR_CALL_STARTED,
            data=data,
        )
    except Exception as exc:
        logging.debug(
            "[event] failed to emit connector call event action=%s route=%s error_type=%s",
            action,
            connector_route,
            type(exc).__name__,
        )


def emit_execution_create_filter_event(
    bot: Any,
    *,
    event_type: str,
    status: str,
    reason_code: str,
    order_count: int,
    symbols: Any,
    wave: dict | None = None,
    level: str = "debug",
    message: str | None = None,
    data: dict | None = None,
) -> None:
    """Emit a bounded event for create orders filtered before exchange write."""
    try:
        order_wave_id, _action_id = _execution_event_ids(
            wave, action="create", index=None
        )
        try:
            symbol_values = sorted({str(symbol) for symbol in symbols or [] if symbol})
        except TypeError:
            symbol_values = [str(symbols)]
        payload = dict(data or {})
        payload.update(
            {
                "order_count": max(0, int(order_count)),
                "symbols": symbol_values[:12],
                "symbols_count": len(symbol_values),
                "symbols_truncated": len(symbol_values) > 12,
            }
        )
        _add_execution_debug_profile(
            bot,
            payload,
            event_type=event_type,
            action="create_filter",
            extra=data,
            wave=wave,
        )
        bot._emit_live_event(
            event_type,
            level=level,
            component="execution.create_filter",
            tags=(EventTags.EXECUTION, EventTags.ORDER, EventTags.GATE, "create"),
            cycle_id=bot._current_live_event_cycle_id(),
            order_wave_id=order_wave_id,
            status=status,
            reason_code=reason_code,
            message=message,
            data={key: value for key, value in payload.items() if value is not None},
        )
    except Exception as exc:
        logging.debug(
            "[event] failed to emit execution create filter event type=%s status=%s reason=%s: %s",
            event_type,
            status,
            reason_code,
            exc,
        )


def emit_initial_entry_eligibility_event(
    bot: Any,
    *,
    data: dict[str, Any],
    wave: dict | None = None,
) -> None:
    """Emit bounded cycle-local fresh-entry eligibility evidence."""
    try:
        order_wave_id, _action_id = _execution_event_ids(
            wave,
            action="create",
            index=None,
        )
        _safe_emit(
            bot,
            EventTypes.ENTRY_INITIAL_ELIGIBILITY,
            level="debug",
            component="entry.initial_eligibility",
            tags=(EventTags.ENTRY, EventTags.AVAILABILITY, EventTags.EXECUTION),
            cycle_id=current_live_event_cycle_id(bot),
            order_wave_id=order_wave_id,
            status="succeeded",
            reason_code=ReasonCodes.FRESH_ENTRY_ELIGIBILITY,
            data=dict(data),
        )
    except Exception as exc:
        logging.debug(
            "[event] failed to emit %s: %s",
            EventTypes.ENTRY_INITIAL_ELIGIBILITY,
            exc,
        )


def emit_execution_order_event(
    bot: Any,
    *,
    event_type: str,
    order: dict | None,
    action: str,
    status: str,
    reason_code: str | None = None,
    level: str = "debug",
    index: int | None = None,
    wave: dict | None = None,
    result: dict | BaseException | None = None,
    error: BaseException | None = None,
    extra: dict | None = None,
) -> None:
    try:
        order_wave_id, action_id = _execution_event_ids(wave, action=action, index=index)
        data = _order_event_data(order, index=index)
        if isinstance(result, dict):
            data.update(
                {
                    "result_status": result.get("status"),
                    "result_order_id_short": _short_order_id(
                        result.get("id") or result.get("order_id")
                    ),
                    "result_client_order_id_short": _short_order_id(
                        result.get("custom_id") or result.get("clientOrderId")
                    ),
                }
            )
        elif isinstance(result, BaseException):
            error = result
        if error is not None:
            data["error_type"] = type(error).__name__
            data["error"] = _sanitize_remote_text(error, max_len=500)
        if extra:
            data.update(extra)
        _add_execution_debug_profile(
            bot,
            data,
            event_type=event_type,
            action=action,
            order=order,
            result=result,
            extra=extra,
            wave=wave,
        )
        bot._emit_live_event(
            event_type,
            level=level,
            component="execution.order_write",
            tags=(EventTags.EXECUTION, EventTags.ORDER, action),
            cycle_id=bot._current_live_event_cycle_id(),
            order_wave_id=order_wave_id,
            action_id=action_id,
            symbol=str(order.get("symbol")) if isinstance(order, dict) and order.get("symbol") else None,
            pside=str(order.get("position_side")) if isinstance(order, dict) and order.get("position_side") else None,
            side=str(order.get("side")) if isinstance(order, dict) and order.get("side") else None,
            order_id=str((result or {}).get("id")) if isinstance(result, dict) and (result or {}).get("id") else None,
            client_order_id=str(order.get("custom_id")) if isinstance(order, dict) and order.get("custom_id") else None,
            status=status,
            reason_code=reason_code,
            data={key: value for key, value in data.items() if value is not None},
        )
    except Exception as exc:
        logging.debug(
            "[event] failed to emit execution order event type=%s action=%s status=%s: %s",
            event_type,
            action,
            status,
            exc,
        )


def emit_execution_confirmation_requested_event(
    bot: Any,
    *,
    surfaces: set[str],
    target_epoch: int,
    wave: dict | None = None,
    min_epoch: int | None = None,
) -> None:
    try:
        order_wave_id = (
            str(wave.get("event_id") or f"ow_{wave.get('id', '')}")
            if isinstance(wave, dict)
            else None
        )
        data = {
            "surfaces": sorted(str(surface) for surface in surfaces),
            "target_epoch": int(target_epoch),
            "current_epoch": int(getattr(bot, "_authoritative_refresh_epoch", 0) or 0),
            "min_epoch": int(min_epoch) if min_epoch is not None else None,
        }
        _add_execution_debug_profile(
            bot,
            data,
            event_type=EventTypes.EXECUTION_CONFIRMATION_REQUESTED,
            wave=wave,
            surfaces=surfaces,
        )
        bot._emit_live_event(
            EventTypes.EXECUTION_CONFIRMATION_REQUESTED,
            level="debug",
            component="execution.confirmation",
            tags=(EventTags.EXECUTION, EventTags.CONFIRMATION),
            cycle_id=bot._current_live_event_cycle_id(),
            order_wave_id=order_wave_id,
            status="started",
            reason_code=ReasonCodes.AUTHORITATIVE_CONFIRMATION,
            data=data,
        )
    except Exception as exc:
        logging.debug("[event] failed to emit confirmation requested event: %s", exc)


def emit_execution_confirmation_satisfied_event(
    bot: Any,
    *,
    wave: dict,
    confirmations: dict,
    current_epoch: int,
    fresh_surfaces: set[str],
    changed_surfaces: list[str],
    elapsed_ms: int,
    confirm_ms: int,
    level: str = "debug",
) -> None:
    try:
        data = {
            "id": int(wave.get("id", 0) or 0),
            "elapsed_ms": int(elapsed_ms),
            "confirm_ms": int(confirm_ms),
            "current_epoch": int(current_epoch),
            "confirmations": {
                str(surface): int(epoch)
                for surface, epoch in dict(confirmations or {}).items()
            },
            "fresh_surfaces": sorted(str(surface) for surface in fresh_surfaces),
            "changed_surfaces": list(changed_surfaces or []),
            "planned_cancel": int(wave.get("planned_cancel", 0) or 0),
            "planned_create": int(wave.get("planned_create", 0) or 0),
            "cancel_posted": int(wave.get("cancel_posted", 0) or 0),
            "create_posted": int(wave.get("create_posted", 0) or 0),
            "symbols": list(wave.get("symbols") or []),
        }
        _add_execution_debug_profile(
            bot,
            data,
            event_type=EventTypes.EXECUTION_CONFIRMATION_SATISFIED,
            wave=wave,
            confirmations=confirmations,
            fresh_surfaces=fresh_surfaces,
        )
        bot._emit_live_event(
            EventTypes.EXECUTION_CONFIRMATION_SATISFIED,
            level=str(level).lower(),
            component="execution.confirmation",
            tags=(EventTags.EXECUTION, EventTags.CONFIRMATION),
            cycle_id=bot._current_live_event_cycle_id(),
            order_wave_id=str(wave.get("event_id") or f"ow_{wave.get('id', '')}"),
            status="succeeded",
            reason_code=ReasonCodes.AUTHORITATIVE_CONFIRMATION,
            data=data,
        )
    except Exception as exc:
        logging.debug("[event] failed to emit confirmation satisfied event: %s", exc)


def emit_execution_confirmation_timeout_event(
    bot: Any,
    *,
    wave: dict,
    confirmations: dict,
    current_epoch: int,
    fresh_surfaces: set[str],
    elapsed_ms: int,
    confirm_ms: int,
    timeout_ms: int,
    level: str = "warning",
) -> None:
    try:
        data = {
            "id": int(wave.get("id", 0) or 0),
            "elapsed_ms": int(elapsed_ms),
            "confirm_ms": int(confirm_ms),
            "timeout_ms": int(timeout_ms),
            "current_epoch": int(current_epoch),
            "confirmations": {
                str(surface): int(epoch)
                for surface, epoch in dict(confirmations or {}).items()
            },
            "fresh_surfaces": sorted(str(surface) for surface in fresh_surfaces),
            "pending_surfaces": sorted(
                str(surface)
                for surface, epoch in dict(confirmations or {}).items()
                if surface not in fresh_surfaces or current_epoch < int(epoch)
            ),
            "planned_cancel": int(wave.get("planned_cancel", 0) or 0),
            "planned_create": int(wave.get("planned_create", 0) or 0),
            "cancel_posted": int(wave.get("cancel_posted", 0) or 0),
            "create_posted": int(wave.get("create_posted", 0) or 0),
            "symbols": list(wave.get("symbols") or []),
        }
        _add_execution_debug_profile(
            bot,
            data,
            event_type=EventTypes.EXECUTION_CONFIRMATION_TIMEOUT,
            wave=wave,
            confirmations=confirmations,
            fresh_surfaces=fresh_surfaces,
        )
        bot._emit_live_event(
            EventTypes.EXECUTION_CONFIRMATION_TIMEOUT,
            level=str(level).lower(),
            component="execution.confirmation",
            tags=(EventTags.EXECUTION, EventTags.CONFIRMATION, EventTags.TIMEOUT),
            cycle_id=bot._current_live_event_cycle_id(),
            order_wave_id=str(wave.get("event_id") or f"ow_{wave.get('id', '')}"),
            status="degraded",
            reason_code=ReasonCodes.AUTHORITATIVE_CONFIRMATION_TIMEOUT,
            data=data,
        )
    except Exception as exc:
        logging.debug("[event] failed to emit confirmation timeout event: %s", exc)


def emit_risk_mode_changed_event(
    bot: Any,
    *,
    pside: str,
    source: str,
    action: str,
    previous_mode: str | None = None,
    mode: str | None = None,
    symbol: str | None = None,
    symbols: Any = None,
    previous_modes: dict[str, str] | None = None,
    modes: dict[str, str] | None = None,
    reason_code: str | None = None,
) -> None:
    try:
        source_name = str(source or "risk").strip().lower() or "risk"
        action_name = str(action or "changed").strip().lower() or "changed"
        data: dict[str, Any] = {
            "source": source_name,
            "action": action_name,
        }
        if previous_mode is not None:
            data["previous_mode"] = str(previous_mode)
        if mode is not None:
            data["mode"] = str(mode)
        if symbols is not None:
            data["symbols"] = _symbol_sample(symbols)
        if previous_modes is not None:
            data["previous_mode_counts"] = dict(
                sorted(Counter(str(value) for value in previous_modes.values()).items())
            )
        if modes is not None:
            data["mode_counts"] = dict(
                sorted(Counter(str(value) for value in modes.values()).items())
            )
        bot._emit_live_event(
            EventTypes.RISK_MODE_CHANGED,
            level="info",
            component=f"risk.{source_name}.mode",
            tags=(EventTags.RISK, EventTags.MODE, source_name),
            cycle_id=bot._current_live_event_cycle_id(),
            symbol=symbol,
            pside=pside,
            status="succeeded",
            reason_code=reason_code or f"{source_name}_{action_name}",
            data={key: value for key, value in data.items() if value is not None},
        )
    except Exception as exc:
        logging.debug(
            "[event] failed to emit risk mode changed event pside=%s symbol=%s: %s",
            pside,
            symbol,
            exc,
        )


def emit_realized_loss_gate_blocked_event(
    bot: Any,
    *,
    symbol: str,
    pside: str,
    order_type: str,
    qty: float,
    price: float,
    projected_pnl: float,
    projected_balance: float,
    balance_floor: float,
    max_realized_loss_pct: float,
) -> None:
    try:
        bot._emit_live_event(
            EventTypes.REALIZED_LOSS_GATE_BLOCKED,
            level="warning",
            component="risk.realized_loss_gate",
            tags=(EventTags.RISK, EventTags.ORDER, EventTags.GATE),
            cycle_id=bot._current_live_event_cycle_id(),
            symbol=str(symbol),
            pside=str(pside),
            status="deferred",
            reason_code=ReasonCodes.REALIZED_LOSS_GATE_BLOCKED,
            data={
                "order_type": str(order_type),
                "qty": _safe_float(qty),
                "price": _safe_float(price),
                "projected_pnl": _safe_float(projected_pnl),
                "projected_balance_after": _safe_float(projected_balance),
                "balance_floor": _safe_float(balance_floor),
                "max_realized_loss_pct": _safe_float(max_realized_loss_pct),
            },
        )
    except Exception as exc:
        logging.debug(
            "[event] failed to emit realized loss gate event pside=%s symbol=%s: %s",
            pside,
            symbol,
            exc,
        )


def emit_entry_cooldown_delta_anchored_event(
    bot: Any,
    *,
    symbol: str,
    pside: str,
    previous_abs_size: float,
    current_abs_size: float,
    qty_step: float,
    epsilon: float,
    anchor_ts_ms: int,
    text_log_emitted: bool,
) -> None:
    try:
        bot._emit_live_event(
            EventTypes.RISK_ENTRY_COOLDOWN_DELTA_ANCHORED,
            level="warning",
            component="risk.entry_cooldown",
            tags=(EventTags.RISK, EventTags.GATE, EventTags.FALLBACK),
            cycle_id=bot._current_live_event_cycle_id(),
            symbol=str(symbol),
            pside=str(pside),
            status="deferred",
            reason_code=ReasonCodes.RISK_ENTRY_COOLDOWN_POSITION_DELTA,
            data={
                "previous_abs_size": _safe_float(previous_abs_size),
                "current_abs_size": _safe_float(current_abs_size),
                "qty_step": _safe_float(qty_step),
                "epsilon": _safe_float(epsilon),
                "anchor_ts_ms": int(anchor_ts_ms),
                "fallback_source": "exchange_position_delta",
                "text_log_emitted": bool(text_log_emitted),
            },
        )
    except Exception as exc:
        logging.debug(
            "[event] failed to emit entry cooldown delta event pside=%s symbol=%s: %s",
            pside,
            symbol,
            exc,
        )


def emit_entry_min_effective_cost_blocked_event(
    bot: Any,
    *,
    symbol: str,
    pside: str,
    projected_initial_cost: float,
    effective_min_cost: float,
    balance: float,
    effective_limit: float,
    entry_initial_qty_pct: float,
) -> None:
    try:
        bot._emit_live_event(
            EventTypes.ENTRY_MIN_EFFECTIVE_COST_BLOCKED,
            level="info",
            component="entry.min_effective_cost",
            tags=(EventTags.ORDER, EventTags.GATE),
            cycle_id=bot._current_live_event_cycle_id(),
            symbol=str(symbol),
            pside=str(pside),
            status="skipped",
            reason_code=ReasonCodes.MIN_EFFECTIVE_COST_BLOCKED,
            data={
                "projected_initial_cost": _safe_float(projected_initial_cost),
                "effective_min_cost": _safe_float(effective_min_cost),
                "balance": _safe_float(balance),
                "effective_limit": _safe_float(effective_limit),
                "entry_initial_qty_pct": _safe_float(entry_initial_qty_pct),
                "action": "skip_create",
            },
        )
    except Exception as exc:
        logging.debug(
            "[event] failed to emit min effective cost event pside=%s symbol=%s: %s",
            pside,
            symbol,
            exc,
        )


def _unstuck_status_side_summary(
    info: dict[str, Any],
    *,
    override_limit: int = 8,
) -> dict[str, Any]:
    status = str(info.get("status") or "unknown")
    out: dict[str, Any] = {"status": status}
    for key in ("allowance", "peak", "pct_from_peak", "loss_allowance_pct"):
        value = _safe_float(info.get(key))
        if value is not None:
            out[key] = value
    allowance = _safe_float(info.get("allowance"))
    if allowance is not None:
        out["over_budget"] = allowance < 0.0
    override_pcts = info.get("override_loss_allowance_pcts")
    if isinstance(override_pcts, dict):
        items = sorted(
            (str(symbol), _safe_float(pct)) for symbol, pct in override_pcts.items()
        )
        clean_items = [(symbol, pct) for symbol, pct in items if pct is not None]
        if clean_items:
            out["override_loss_allowance_pct_count"] = len(clean_items)
            out["override_loss_allowance_pcts"] = {
                symbol: pct for symbol, pct in clean_items[:override_limit]
            }
            out["override_loss_allowance_pcts_truncated"] = max(
                0, len(clean_items) - override_limit
            )
    override_allowances = info.get("override_allowances")
    if isinstance(override_allowances, dict):
        items = sorted(
            (str(symbol), _safe_float(allowance))
            for symbol, allowance in override_allowances.items()
        )
        clean_items = [
            (symbol, allowance) for symbol, allowance in items if allowance is not None
        ]
        if clean_items:
            out["override_allowance_count"] = len(clean_items)
            out["override_allowances"] = {
                symbol: allowance for symbol, allowance in clean_items[:override_limit]
            }
            out["override_allowances_truncated"] = max(0, len(clean_items) - override_limit)
    for key in (
        "next_symbol",
        "next_target_price",
        "next_target_distance_ratio",
        "next_unstuck_trigger_distance_ratio",
    ):
        value = info.get(key)
        if value is None:
            continue
        if key == "next_symbol":
            out[key] = str(value)
            continue
        number = _safe_float(value)
        if number is not None:
            out[key] = number
    return out


def _trailing_threshold_projection(
    *,
    kind: str,
    pside: str,
    threshold_price: float | None,
    retracement_pct: float | None,
) -> float | None:
    if threshold_price is None or retracement_pct is None:
        return None
    if threshold_price <= 0.0 or retracement_pct < 0.0:
        return None
    if (kind == "entry" and pside == "long") or (kind == "close" and pside == "short"):
        return threshold_price * (1.0 + retracement_pct)
    if (kind == "entry" and pside == "short") or (kind == "close" and pside == "long"):
        return threshold_price * (1.0 - retracement_pct)
    return None


def _trailing_status_summary(payload: dict[str, Any]) -> dict[str, Any]:
    kind = str(payload.get("kind") or "unknown")
    pside = str(payload.get("pside") or "unknown")
    threshold_price = _safe_float(payload.get("threshold_price"))
    retracement_pct = _safe_float(payload.get("retracement_pct"))
    out: dict[str, Any] = {
        "kind": kind,
        "diagnostics_supported": bool(payload.get("diagnostics_supported", True)),
        "strategy_kind": str(payload.get("strategy_kind") or "") or None,
        "trailing_status": str(payload.get("status") or payload.get("trailing_status") or "unknown"),
        "selected_mode": str(payload.get("selected_mode") or "") or None,
        "order_type": str(payload.get("order_type") or "") or None,
        "triggered": bool(payload.get("triggered", False)),
        "threshold_met": bool(payload.get("threshold_met", False))
        if "threshold_met" in payload
        else None,
        "retracement_met": bool(payload.get("retracement_met", False))
        if "retracement_met" in payload
        else None,
        "threshold_pct": _safe_float(payload.get("threshold_pct")),
        "threshold_price": threshold_price,
        "retracement_pct": retracement_pct,
        "retracement_price": _safe_float(payload.get("retracement_price")),
        "threshold_projection_retracement_price": _trailing_threshold_projection(
            kind=kind,
            pside=pside,
            threshold_price=threshold_price,
            retracement_pct=retracement_pct,
        ),
        "current_price": _safe_float(payload.get("current_price")),
        "position_price": _safe_float(payload.get("position_price")),
        "position_size": _safe_float(payload.get("position_size")),
        "current_vs_threshold_ratio": _safe_float(
            payload.get("current_vs_threshold_ratio")
        ),
        "current_vs_retracement_ratio": _safe_float(
            payload.get("current_vs_retracement_ratio")
        ),
        "unsupported_reason": str(payload.get("unsupported_reason") or "") or None,
        "changed": bool(payload.get("changed", False)),
        "operator_visible": bool(payload.get("operator_visible", True)),
    }
    return {key: value for key, value in out.items() if value is not None}


def emit_trailing_status_event(
    bot: Any,
    *,
    symbol: str,
    pside: str,
    kind: str,
    payload: dict[str, Any],
    changed: bool,
    operator_visible: bool,
) -> None:
    try:
        data = _trailing_status_summary(
            {
                **dict(payload or {}),
                "kind": str(kind),
                "pside": str(pside),
                "changed": bool(changed),
                "operator_visible": bool(operator_visible),
            }
        )
        bot._emit_live_event(
            EventTypes.TRAILING_STATUS,
            level="info",
            component="risk.trailing.status",
            tags=(EventTags.RISK, EventTags.TRAILING, EventTags.POSITION),
            cycle_id=bot._current_live_event_cycle_id(),
            symbol=str(symbol),
            pside=str(pside),
            status="succeeded",
            reason_code=ReasonCodes.TRAILING_STATUS,
            data=data,
        )
    except Exception as exc:
        logging.debug(
            "[event] failed to emit trailing status event symbol=%s pside=%s kind=%s: %s",
            symbol,
            pside,
            kind,
            exc,
        )


def emit_unstuck_status_event(
    bot: Any,
    *,
    side_statuses: dict[str, dict[str, Any]],
    changed: bool,
    operator_visible: bool,
) -> None:
    try:
        sides = {
            str(pside): _unstuck_status_side_summary(dict(info or {}))
            for pside, info in sorted((side_statuses or {}).items())
        }
        status_counts = dict(
            sorted(
                Counter(
                    str(info.get("status") or "unknown") for info in sides.values()
                ).items()
            )
        )
        over_budget_sides = sorted(
            pside for pside, info in sides.items() if bool(info.get("over_budget"))
        )
        bot._emit_live_event(
            EventTypes.UNSTUCK_STATUS,
            level="info",
            component="risk.unstuck.status",
            tags=(EventTags.RISK, EventTags.UNSTUCK, EventTags.SUMMARY),
            cycle_id=bot._current_live_event_cycle_id(),
            status="succeeded",
            reason_code=ReasonCodes.UNSTUCK_STATUS,
            data={
                "changed": bool(changed),
                "operator_visible": bool(operator_visible),
                "status_counts": status_counts,
                "over_budget_sides": over_budget_sides,
                "sides": sides,
            },
        )
    except Exception as exc:
        logging.debug("[event] failed to emit unstuck status event: %s", exc)


def emit_unstuck_selection_event(
    bot: Any,
    *,
    symbol: str,
    pside: str,
    entry_price: float,
    current_price: float,
    allowance: float,
    changed: bool,
) -> None:
    try:
        entry = _safe_float(entry_price)
        current = _safe_float(current_price)
        price_diff_pct = None
        if entry is not None and current is not None and entry > 0.0 and current > 0.0:
            price_diff_pct = (current / entry - 1.0) * 100.0
        data = {
            "changed": bool(changed),
            "entry_price": entry,
            "current_price": current,
            "price_diff_pct": price_diff_pct,
            "allowance": _safe_float(allowance),
        }
        bot._emit_live_event(
            EventTypes.UNSTUCK_SELECTION,
            level="info",
            component="risk.unstuck.selection",
            tags=(EventTags.RISK, EventTags.UNSTUCK, EventTags.SELECTION),
            cycle_id=bot._current_live_event_cycle_id(),
            symbol=str(symbol),
            pside=str(pside),
            status="succeeded",
            reason_code=ReasonCodes.UNSTUCK_SELECTION,
            data={key: value for key, value in data.items() if value is not None},
        )
    except Exception as exc:
        logging.debug(
            "[event] failed to emit unstuck selection event pside=%s symbol=%s: %s",
            pside,
            symbol,
            exc,
        )


def emit_balance_changed_event(
    bot: Any,
    *,
    previous_balance_raw: float,
    balance_raw: float,
    previous_balance_snapped: float,
    balance_snapped: float,
    equity: float,
    source: str,
) -> None:
    try:
        raw_delta = float(balance_raw) - float(previous_balance_raw)
        snapped_delta = float(balance_snapped) - float(previous_balance_snapped)
        bot._emit_live_event(
            EventTypes.BALANCE_CHANGED,
            level="info",
            component="account.balance",
            tags=(EventTags.ACCOUNT, EventTags.BALANCE),
            cycle_id=bot._current_live_event_cycle_id(),
            status="succeeded",
            reason_code=ReasonCodes.BALANCE_CHANGED,
            data={
                "previous_balance_raw": float(previous_balance_raw),
                "balance_raw": float(balance_raw),
                "balance_raw_delta": raw_delta,
                "previous_balance_snapped": float(previous_balance_snapped),
                "balance_snapped": float(balance_snapped),
                "balance_snapped_delta": snapped_delta,
                "equity": float(equity),
                "source": str(source),
            },
        )
    except Exception as exc:
        logging.debug("[event] failed to emit balance changed event: %s", exc)


def _fill_coverage_summary(status: Any) -> dict[str, Any]:
    if not isinstance(status, dict):
        return {}
    data: dict[str, Any] = {}
    if "ready" in status:
        data["ready"] = bool(status.get("ready"))
    for key in ("reason", "history_scope", "gap_reason"):
        value = status.get(key)
        if value is not None:
            data[key] = str(value)[:160]
    for key in ("covered_start_ms", "oldest_event_ts", "gap_start_ts", "gap_end_ts"):
        value = _safe_int(status.get(key))
        if value is not None:
            data[key] = value
    return data


def _fill_refresh_debug_payload(
    *,
    coverage_before: Any = None,
    coverage_after: Any = None,
    event_count_before: int | None = None,
    event_count_after: int | None = None,
    new_count: int | None = None,
    enriched_count: int | None = None,
    pending_pnl_count: int | None = None,
) -> dict[str, Any]:
    before_count = _safe_int(event_count_before)
    after_count = _safe_int(event_count_after)
    new_value = _safe_int(new_count)
    enriched_value = _safe_int(enriched_count)
    pending_value = _safe_int(pending_pnl_count)
    data: dict[str, Any] = {
        "coverage_before_keys": _mapping_key_sample(coverage_before),
        "coverage_after_keys": _mapping_key_sample(coverage_after),
        "event_count_before": before_count,
        "event_count_after": after_count,
        "new_count": new_value,
        "enriched_count": enriched_value,
        "pending_pnl_count": pending_value,
    }
    if before_count is not None and after_count is not None:
        data["event_count_delta"] = int(after_count) - int(before_count)
    if isinstance(coverage_before, dict):
        data["coverage_before_ready"] = bool(coverage_before.get("ready", False))
        reason = coverage_before.get("reason")
        if reason is not None:
            data["coverage_before_reason"] = str(reason)[:160]
    if isinstance(coverage_after, dict):
        data["coverage_after_ready"] = bool(coverage_after.get("ready", False))
        reason = coverage_after.get("reason")
        if reason is not None:
            data["coverage_after_reason"] = str(reason)[:160]
    if isinstance(coverage_before, dict) and isinstance(coverage_after, dict):
        before_ready = bool(coverage_before.get("ready", False))
        after_ready = bool(coverage_after.get("ready", False))
        if before_ready != after_ready:
            data["coverage_ready_transition"] = f"{before_ready}->{after_ready}"
    return {key: value for key, value in data.items() if value not in (None, [], {})}


def _best_effort_fill_refresh_debug_payload(**kwargs: Any) -> dict[str, Any] | None:
    try:
        return _fill_refresh_debug_payload(**kwargs)
    except Exception as exc:
        logging.debug("[event] failed to build fill refresh debug payload: %s", exc)
        return None


def _fill_ingested_debug_payload(
    event: Any,
    *,
    payload: dict | None = None,
) -> dict[str, Any]:
    fill_payload = payload if isinstance(payload, dict) else {}
    source_ids = list(getattr(event, "source_ids", []) or [])
    data: dict[str, Any] = {
        "payload_keys": _mapping_key_sample(fill_payload),
        "payload_key_count": len(fill_payload),
        "source_ids_count": len(source_ids),
        "has_client_order_id": bool(getattr(event, "client_order_id", None)),
        "has_fee": getattr(event, "fee", None) is not None,
        "has_fee_paid": getattr(event, "fee_paid", None) is not None,
        "has_pnl": getattr(event, "pnl", None) is not None,
        "pnl_status": str(getattr(event, "pnl_status", "") or "")[:80] or None,
    }
    return {key: value for key, value in data.items() if value not in (None, [], {})}


def _best_effort_fill_ingested_debug_payload(
    event: Any,
    *,
    payload: dict | None = None,
) -> dict[str, Any] | None:
    try:
        return _fill_ingested_debug_payload(event, payload=payload)
    except Exception as exc:
        logging.debug("[event] failed to build fill ingested debug payload: %s", exc)
        return None


def _emit_fills_refresh_summary_event_unchecked(
    bot: Any,
    *,
    source: str,
    refresh_mode: str,
    status: str,
    reason_code: str,
    elapsed_ms: int,
    lookback: Any = None,
    history_scope: str | None = None,
    event_count_before: int | None = None,
    event_count_after: int | None = None,
    new_count: int | None = None,
    enriched_count: int | None = None,
    pending_pnl_count: int | None = None,
    coverage_before: dict[str, Any] | None = None,
    coverage_after: dict[str, Any] | None = None,
    overlap_minutes: float | None = None,
    retry_count: int | None = None,
    next_retry_in_ms: int | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
    doctor_mode: str | None = None,
    doctor_action: str | None = None,
    auto_repair: bool | None = None,
    anomaly_events: int | None = None,
    repaired: bool | None = None,
    degraded_events_after: int | None = None,
    legacy_files_quarantined: int | None = None,
    quarantine_created: bool | None = None,
    quarantine_reason: str | None = None,
    error: BaseException | None = None,
    level: str = "debug",
) -> None:
    coverage_before_summary = _fill_coverage_summary(coverage_before)
    coverage_after_summary = _fill_coverage_summary(coverage_after)
    data: dict[str, Any] = {
        "source": str(source),
        "refresh_mode": str(refresh_mode),
        "elapsed_ms": max(0, int(elapsed_ms)),
        "lookback": str(lookback) if lookback is not None else None,
        "history_scope": str(history_scope) if history_scope is not None else None,
        "event_count_before": _safe_int(event_count_before),
        "event_count_after": _safe_int(event_count_after),
        "new_count": _safe_int(new_count),
        "enriched_count": _safe_int(enriched_count),
        "pending_pnl_count": _safe_int(pending_pnl_count),
    }
    if coverage_before_summary:
        data["coverage_before"] = coverage_before_summary
        if "ready" in coverage_before_summary:
            data["coverage_ready_before"] = coverage_before_summary["ready"]
        if "reason" in coverage_before_summary:
            data["coverage_reason_before"] = coverage_before_summary["reason"]
    if coverage_after_summary:
        data["coverage_after"] = coverage_after_summary
        if "ready" in coverage_after_summary:
            data["coverage_ready_after"] = coverage_after_summary["ready"]
        if "reason" in coverage_after_summary:
            data["coverage_reason_after"] = coverage_after_summary["reason"]
    overlap = _safe_float(overlap_minutes)
    if overlap is not None:
        data["overlap_minutes"] = overlap
    retry_value = _safe_int(retry_count)
    if retry_value is not None:
        data["retry_count"] = retry_value
    next_retry_value = _safe_int(next_retry_in_ms)
    if next_retry_value is not None:
        data["next_retry_in_ms"] = max(0, next_retry_value)
    start_value = _safe_int(start_ms)
    if start_value is not None:
        data["start_ms"] = start_value
    end_value = _safe_int(end_ms)
    if end_value is not None:
        data["end_ms"] = end_value
    if doctor_mode is not None:
        data["doctor_mode"] = str(doctor_mode)
    if doctor_action is not None:
        data["doctor_action"] = str(doctor_action)
    if auto_repair is not None:
        data["auto_repair"] = bool(auto_repair)
    anomaly_value = _safe_int(anomaly_events)
    if anomaly_value is not None:
        data["anomaly_events"] = anomaly_value
    if repaired is not None:
        data["repaired"] = bool(repaired)
    degraded_value = _safe_int(degraded_events_after)
    if degraded_value is not None:
        data["degraded_events_after"] = degraded_value
    quarantined_value = _safe_int(legacy_files_quarantined)
    if quarantined_value is not None:
        data["legacy_files_quarantined"] = quarantined_value
    if quarantine_created is not None:
        data["quarantine_created"] = bool(quarantine_created)
    if quarantine_reason is not None:
        data["quarantine_reason"] = str(quarantine_reason)
    if error is not None:
        data["error_type"] = type(error).__name__
        data["error"] = _sanitize_remote_text(error, max_len=500)
    if live_event_debug_profile_enabled(bot, "fills"):
        debug = _best_effort_fill_refresh_debug_payload(
            coverage_before=coverage_before,
            coverage_after=coverage_after,
            event_count_before=event_count_before,
            event_count_after=event_count_after,
            new_count=new_count,
            enriched_count=enriched_count,
            pending_pnl_count=pending_pnl_count,
        )
        if debug:
            data["debug_profile"] = "fills"
            data["debug"] = debug
    _safe_emit(
        bot,
        EventTypes.FILLS_REFRESH_SUMMARY,
        level=str(level).lower(),
        component="fills.refresh",
        tags=(EventTags.FILLS, EventTags.REFRESH, EventTags.COVERAGE),
        cycle_id=current_live_event_cycle_id(bot),
        status=str(status),
        reason_code=str(reason_code),
        data={key: value for key, value in data.items() if value is not None},
    )


def emit_fills_refresh_summary_event(bot: Any, *args: Any, **kwargs: Any) -> None:
    try:
        _emit_fills_refresh_summary_event_unchecked(bot, *args, **kwargs)
    except Exception as exc:
        logging.debug("[event] failed to emit fills refresh summary event: %s", exc)


def emit_fill_ingested_event(
    bot: Any,
    event: Any,
    *,
    payload: dict | None = None,
    operator_visible: bool = True,
) -> None:
    try:
        fill_payload = dict(payload or {})
        fill_id = getattr(event, "id", None)
        client_order_id = getattr(event, "client_order_id", None)
        source_ids = list(getattr(event, "source_ids", []) or [])
        data = {
            "fill_id_hash": _event_fingerprint(fill_id),
            "client_order_id_short": _short_order_id(client_order_id),
            "timestamp": int(getattr(event, "timestamp", 0) or 0),
            "qty": _safe_float(getattr(event, "qty", None)),
            "price": _safe_float(getattr(event, "price", None)),
            "pnl": _safe_float(getattr(event, "pnl", None)),
            "fee": _safe_float(getattr(event, "fee", None)),
            "pb_order_type": str(getattr(event, "pb_order_type", "") or "").lower(),
            "pnl_status": str(
                getattr(event, "pnl_status", "complete") or "complete"
            ).lower()[:80],
            "operator_visible": bool(operator_visible),
            "source_ids_count": len(source_ids),
        }
        for key in ("qty", "price", "pnl", "fee", "pb_order_type", "timestamp"):
            if key in fill_payload and data.get(key) is None:
                data[key] = fill_payload.get(key)
        if live_event_debug_profile_enabled(bot, "fills"):
            debug = _best_effort_fill_ingested_debug_payload(event, payload=fill_payload)
            if debug:
                data["debug_profile"] = "fills"
                data["debug"] = debug
        bot._emit_live_event(
            EventTypes.FILL_INGESTED,
            level="info",
            component="fills.ingest",
            tags=(EventTags.FILL, EventTags.ORDER),
            cycle_id=bot._current_live_event_cycle_id(),
            symbol=getattr(event, "symbol", None),
            pside=str(getattr(event, "position_side", "") or "").lower() or None,
            side=str(getattr(event, "side", "") or "").lower() or None,
            client_order_id=str(client_order_id) if client_order_id else None,
            status="succeeded",
            reason_code=ReasonCodes.NEW_FILL,
            data={key: value for key, value in data.items() if value is not None},
        )
    except Exception as exc:
        logging.debug("[event] failed to emit fill ingested event: %s", exc)


def emit_fills_ingested_summary_event(
    bot: Any,
    *,
    count: int,
    known_net_realized_pnl: float,
    known_pnl_count: int,
    pending_pnl_count: int,
) -> None:
    try:
        bot._emit_live_event(
            EventTypes.FILLS_INGESTED_SUMMARY,
            level="info",
            component="fills.ingest",
            tags=(EventTags.FILLS, EventTags.FILL, EventTags.SUMMARY),
            cycle_id=bot._current_live_event_cycle_id(),
            status="succeeded",
            reason_code=ReasonCodes.NEW_FILL_BATCH,
            data={
                "count": max(0, int(count)),
                "known_net_realized_pnl": _safe_finite_float(known_net_realized_pnl),
                "known_pnl_count": max(0, int(known_pnl_count)),
                "pending_pnl_count": max(0, int(pending_pnl_count)),
            },
        )
    except Exception as exc:
        logging.debug("[event] failed to emit fills ingested summary event: %s", exc)


def emit_position_changed_event(
    bot: Any,
    *,
    symbol: str,
    pside: str,
    action: str,
    old: dict,
    new: dict,
    wallet_exposure: float,
    wel_ratio: float,
    wele_ratio: float,
    twel_ratio: float,
    price_action_distance: float,
    upnl: float,
    last_price: float | None,
) -> None:
    try:
        old_size = _safe_float(old.get("size")) or 0.0
        new_size = _safe_float(new.get("size")) or 0.0
        old_price = _safe_float(old.get("price")) or 0.0
        new_price = _safe_float(new.get("price")) or 0.0
        status = "succeeded"
        reason_code = str(action).strip().replace(" ", "_") or "position_changed"
        bot._emit_live_event(
            EventTypes.POSITION_CHANGED,
            level="info",
            component="account.position",
            tags=(EventTags.ACCOUNT, EventTags.POSITION),
            cycle_id=bot._current_live_event_cycle_id(),
            symbol=symbol,
            pside=pside,
            status=status,
            reason_code=reason_code,
            data={
                "action": str(action).strip(),
                "old_size": old_size,
                "new_size": new_size,
                "size_delta": new_size - old_size,
                "old_price": old_price,
                "new_price": new_price,
                "last_price": _safe_float(last_price),
                "wallet_exposure": float(wallet_exposure),
                "wel_ratio": float(wel_ratio),
                "wele_ratio": float(wele_ratio),
                "twel_ratio": float(twel_ratio),
                "price_action_distance": float(price_action_distance),
                "upnl": float(upnl),
            },
        )
    except Exception as exc:
        logging.debug("[event] failed to emit position changed event: %s", exc)
