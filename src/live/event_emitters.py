from __future__ import annotations

import logging
import hashlib
import re
from urllib.parse import urlsplit, urlunsplit
from typing import Any

from live.event_bus import EventTypes, LiveEvent, emit_event, utc_ms


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
    try:
        parsed = urlsplit(raw)
    except Exception:
        return _sanitize_remote_text(raw, max_len=500), raw_hash
    if parsed.query or parsed.fragment:
        sanitized = urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                "[redacted]" if parsed.query else "",
                "[redacted]" if parsed.fragment else "",
            )
        )
    else:
        sanitized = raw
    return _sanitize_remote_text(sanitized, max_len=500), raw_hash


def _sanitize_remote_fetch_payload(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    for key in ("error", "error_repr"):
        if data.get(key) is not None:
            data[key] = _sanitize_remote_text(data[key], max_len=500)
    if data.get("url") is not None:
        url, url_hash = _sanitize_remote_url(data["url"])
        data["url"] = url
        if url_hash is not None:
            data["url_hash"] = url_hash
    return data


def _remote_fetch_payload_key(payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        payload.get("kind"),
        payload.get("symbol"),
        payload.get("tf") or payload.get("timeframe"),
        payload.get("since_ts"),
        payload.get("url"),
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
    return bot._emit_live_event(
        event_type,
        level=level,
        component="candles.remote_fetch",
        tags=("remote_call", "candle"),
        cycle_id=cycle_id,
        remote_call_id=remote_call_id,
        remote_call_group_id=remote_call_group_id,
        symbol=str(payload.get("symbol")) if payload.get("symbol") is not None else None,
        status=status,
        reason_code=str(payload.get("kind") or "remote_fetch"),
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
    emit(
        event_type,
        level=level,
        component="state.authoritative_fetch",
        tags=("remote_call", "state", "authoritative", surface),
        cycle_id=cycle_id,
        remote_call_id=remote_call_id,
        remote_call_group_id=remote_call_group_id,
        status=status,
        reason_code=f"authoritative_{surface}",
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


def begin_live_event_cycle(bot: Any, *, loop_start_ms: int) -> str:
    bot._live_event_cycle_seq = int(getattr(bot, "_live_event_cycle_seq", 0) or 0) + 1
    cycle_id = f"cy_{int(bot._live_event_cycle_seq)}"
    bot._live_event_current_cycle_id = cycle_id
    bot._emit_live_event(
        EventTypes.CYCLE_STARTED,
        level="debug",
        component="execution_loop",
        tags=("cycle", "execution"),
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
        tags=("cycle", "execution"),
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
        tags=("cycle", "execution", "degraded"),
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
    bot._emit_live_event(
        EventTypes.ORDER_WAVE_COMPLETED,
        level=str(level).lower(),
        component="order_wave",
        tags=("order", "wave", "execution"),
        cycle_id=bot._current_live_event_cycle_id(),
        order_wave_id=str(wave.get("event_id") or f"ow_{wave.get('id', '')}"),
        status=status,
        reason_code=reason_code,
        data={
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
            "requested_confirmations": dict(
                wave.get("requested_confirmations") or {}
            ),
        },
    )
