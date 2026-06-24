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
            tags=("planning", "gate", "defer", "summary"),
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


def emit_order_wave_started_event(bot: Any, wave: dict | None) -> None:
    if not wave:
        return
    try:
        bot._emit_live_event(
            EventTypes.ORDER_WAVE_STARTED,
            level="debug",
            component="order_wave",
            tags=("order", "wave", "execution"),
            cycle_id=bot._current_live_event_cycle_id(),
            order_wave_id=str(wave.get("event_id") or f"ow_{wave.get('id', '')}"),
            status="started",
            data={
                "id": int(wave.get("id", 0) or 0),
                "planned_cancel": int(wave.get("planned_cancel", 0) or 0),
                "planned_create": int(wave.get("planned_create", 0) or 0),
                "symbols": list(wave.get("symbols") or []),
            },
        )
    except Exception as exc:
        logging.debug("[event] failed to emit order wave started event: %s", exc)


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
    _safe_emit(
        bot,
        EventTypes.FORAGER_FEATURE_UNAVAILABLE,
        level="debug",
        component="forager.selection",
        tags=("forager", "selection", "ema"),
        cycle_id=current_live_event_cycle_id(bot),
        pside=str(pside),
        status="skipped",
        reason_code="ranking_features_unavailable",
        data={
            "candidate_count": int(candidate_count),
            "unavailable": _symbol_sample(symbols),
            "volume_count": int(volume_count),
            "log_range_count": int(log_range_count),
            "max_age_ms": int(max_age_ms) if max_age_ms is not None else None,
            "fetch_budget": int(fetch_budget) if fetch_budget is not None else None,
        },
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
    _safe_emit(
        bot,
        EventTypes.FORAGER_SELECTION,
        level="debug",
        component="forager.selection",
        tags=("forager", "selection"),
        cycle_id=current_live_event_cycle_id(bot),
        pside=str(pside),
        status=status,
        reason_code=reason_code,
        data={
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
        },
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
        tags=("ema", "bundle"),
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
        tags=("ema", "fallback"),
        cycle_id=current_live_event_cycle_id(bot),
        status="recovered",
        reason_code="ema_fallback_used",
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
        "required_ema_unavailable"
        if candidate_symbols or unavailable_symbols
        else "optional_ema_dropped"
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
    _safe_emit(
        bot,
        EventTypes.EMA_UNAVAILABLE,
        level=level,
        component="ema.bundle",
        tags=("ema", "unavailable"),
        cycle_id=current_live_event_cycle_id(bot),
        status=status,
        reason_code=reason_code,
        data={
            "optional_drop_count": int(optional_count),
            "optional_drop_groups": optional_summary,
            "candidate_unavailable": _symbol_sample(candidate_symbols),
            "candidate_unavailable_groups": _candidate_unavailable_summary(
                candidate_ema_unavailable_details
            ),
            "unavailable": _symbol_sample(unavailable_symbols),
            "unavailable_reasons": _reason_symbol_summary(ema_unavailable_reasons),
        },
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
        bot._emit_live_event(
            event_type,
            level=level,
            component="execution.order_write",
            tags=("execution", "order", action),
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
        bot._emit_live_event(
            EventTypes.EXECUTION_CONFIRMATION_REQUESTED,
            level="debug",
            component="execution.confirmation",
            tags=("execution", "confirmation"),
            cycle_id=bot._current_live_event_cycle_id(),
            order_wave_id=order_wave_id,
            status="started",
            reason_code="authoritative_confirmation",
            data={
                "surfaces": sorted(str(surface) for surface in surfaces),
                "target_epoch": int(target_epoch),
                "current_epoch": int(getattr(bot, "_authoritative_refresh_epoch", 0) or 0),
                "min_epoch": int(min_epoch) if min_epoch is not None else None,
            },
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
        bot._emit_live_event(
            EventTypes.EXECUTION_CONFIRMATION_SATISFIED,
            level=str(level).lower(),
            component="execution.confirmation",
            tags=("execution", "confirmation"),
            cycle_id=bot._current_live_event_cycle_id(),
            order_wave_id=str(wave.get("event_id") or f"ow_{wave.get('id', '')}"),
            status="succeeded",
            reason_code="authoritative_confirmation",
            data={
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
            },
        )
    except Exception as exc:
        logging.debug("[event] failed to emit confirmation satisfied event: %s", exc)


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
            tags=("account", "balance"),
            cycle_id=bot._current_live_event_cycle_id(),
            status="succeeded",
            reason_code="balance_changed",
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


def emit_fill_ingested_event(bot: Any, event: Any, *, payload: dict | None = None) -> None:
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
            "source_ids_count": len(source_ids),
        }
        for key in ("qty", "price", "pnl", "fee", "pb_order_type", "timestamp"):
            if key in fill_payload and data.get(key) is None:
                data[key] = fill_payload.get(key)
        bot._emit_live_event(
            EventTypes.FILL_INGESTED,
            level="info",
            component="fills.ingest",
            tags=("fill", "order"),
            cycle_id=bot._current_live_event_cycle_id(),
            symbol=getattr(event, "symbol", None),
            pside=str(getattr(event, "position_side", "") or "").lower() or None,
            side=str(getattr(event, "side", "") or "").lower() or None,
            client_order_id=str(client_order_id) if client_order_id else None,
            status="succeeded",
            reason_code="new_fill",
            data={key: value for key, value in data.items() if value is not None},
        )
    except Exception as exc:
        logging.debug("[event] failed to emit fill ingested event: %s", exc)


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
            tags=("account", "position"),
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
