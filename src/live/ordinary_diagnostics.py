"""Bounded, passive observations of ordinary plans and account confirmation waits."""
from copy import deepcopy
import logging
import re

from live.diagnostic_safety import bounded_exception_type

MODES = {"normal", "manual", "panic", "tp_only", "graceful_stop"}
SURFACES = ("balance", "positions", "open_orders", "fills")
ROW_LIMIT = 128
_HSL_REASONS = {
    "current_balance_unavailable", "current_positions_unavailable",
    "current_open_orders_unavailable", "current_mark_unavailable",
    "fills_unavailable", "fill_history_unavailable", "prices_unavailable",
    "continuity_unavailable", "price_history_unavailable", "unavailable",
    "current_position_unavailable", "current_contract_metadata_unavailable",
    "causal_position_mark_unavailable", "causal_scope_inventory_unavailable",
    "causal_position_snapshot_precedes_fill",
}


def _failed(stage, exc):
    try:
        logging.debug("[monitor] %s unavailable error_type=%s", stage, bounded_exception_type(exc))
    except Exception:
        # Optional diagnostics must survive an unavailable logging sink too.
        pass


def _reason(value):
    # These are classifications, never arbitrary native/exchange exception text.
    parts = str(value or "").split(",")
    return ",".join(parts[:4]) if parts and all(p in _HSL_REASONS for p in parts) else "unavailable"


def observe_plan(bot, diagnostics, idx_to_symbol, *, now_ms):
    """Called after validation of the native result; never refresh or reevaluate risk."""
    try:
        from live.hsl_live import matches

        wave = getattr(bot, "_hsl_planning_wave", None)
        rows, counts = [], {mode: 0 for mode in sorted(MODES)}
        held = {(symbol, side) for symbol, sides in getattr(bot, "positions", {}).items()
                for side in ("long", "short") if sides.get(side, {}).get("size", 0)}
        for state in (diagnostics or {}).get("symbol_states", []):
            symbol = idx_to_symbol.get(state["symbol_idx"])
            if symbol is None:
                continue
            if not isinstance(symbol, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,32}(?:/[A-Za-z0-9_.-]{1,16})?(?::[A-Za-z0-9_.-]{1,16})?", symbol):
                raise ValueError("invalid diagnostic symbol")
            for side in ("long", "short"):
                native = state[side]
                mode = native["effective_mode"]
                if mode not in MODES or native["input_mode"] not in MODES | {None}:
                    raise ValueError("invalid diagnostic mode")
                if type(native["active"]) is not bool or type(native["allow_initial"]) is not bool:
                    raise ValueError("invalid diagnostic flags")
                counts[mode] += 1
                action = wave.permission(symbol, side)[0] if wave is not None else None
                if action not in {None, "normal", "panic", "halted", "unavailable"}:
                    raise ValueError("invalid diagnostic permission")
                cause = next((_reason(item.reason) for item in wave.unavailable
                              if matches(item.scope, symbol, side)), None) if wave is not None else None
                rows.append(dict(symbol=symbol, pside=side, held=(symbol, side) in held,
                    input_mode=native["input_mode"], effective_mode=mode,
                    active=native["active"], allow_initial=native["allow_initial"],
                    hsl_permission=action, hsl_unavailable_reason=cause))
        rows.sort(key=lambda r: (not r["held"], r["effective_mode"] == "normal", r["symbol"], r["pside"]))
        max_age_getter = getattr(bot, "_live_market_snapshot_max_age_ms", None)
        max_age = max_age_getter() if callable(max_age_getter) else None
        captured = wave.captured_ms if wave is not None else now_ms
        stamps = ([captured, wave.position_observed_ms, *wave.mark_observed_ms]
                  if wave is not None else [captured])
        expires = min(stamps) + max_age if type(max_age) is int and max_age > 0 else None
        bot._monitor_ordinary_observation = dict(
            source="rust_orchestrator", authority="observation_only",
            stage="computed_native_plan",
            captured_at_ms=captured, completed_at_ms=now_ms,
            input_expires_at_ms=expires,
            account_generation=(wave.generation if wave is not None else
                                getattr(bot, "_account_invalidation_generation", 0)),
            mode_counts=counts, scope_count=len(rows), scopes=rows[:ROW_LIMIT],
            omitted_scopes=max(0, len(rows) - ROW_LIMIT),
            restricted_held_count=sum(r["held"] and r["effective_mode"] != "normal" for r in rows),
        ) if rows else None
        bot._monitor_ordinary_diagnostic_failed = False
    except Exception as exc:
        bot._monitor_ordinary_observation = None
        bot._monitor_ordinary_diagnostic_failed = True
        _failed("ordinary observation", exc)


def observe_create_filter(bot, *, event_type, reason_code, order_count, wave, data, now_ms):
    """Retain the last diagnostic outcome independently of optional event sinks."""
    try:
        from live.event_bus import EventTypes, ReasonCodes
        if event_type not in {EventTypes.EXECUTION_CREATE_DEFERRED,
                EventTypes.EXECUTION_CREATE_SKIPPED, EventTypes.EXECUTION_CANCEL_FIRST_BARRIER}:
            return
        reasons = {value for key, value in vars(ReasonCodes).items() if key.isupper()}
        reason = reason_code if reason_code in reasons else "unavailable"
        record = dict(observed_at_ms=now_ms, reason=reason,
            order_count=min(999999, max(0, int(order_count))),
            wave_id=min(999999999, max(0, int((wave or {}).get("id", 0)))),
            required_surfaces=[name for name in SURFACES if name in (data or {}).get("required_surfaces", [])])
        bot._monitor_last_create_filter = record
        if event_type == EventTypes.EXECUTION_CANCEL_FIRST_BARRIER:
            bot._monitor_last_cancel_first_barrier = record
    except Exception as exc:
        _failed("create filter observation", exc)


def snapshot(bot, *, now_ms):
    """Project retained facts, explicitly separating current waits from old outcomes."""
    try:
        pending = getattr(bot, "_authoritative_pending_confirmations", {}) or {}
        ledger = getattr(getattr(bot, "freshness_ledger", None), "surfaces", {})
        confirmations = []
        for name in SURFACES:
            requested = pending.get(name, 0)
            epoch = getattr(ledger.get(name), "epoch", 0)
            if type(requested) is int and requested > epoch:
                confirmations.append(dict(surface=name, requested_epoch=requested, current_epoch=epoch))
        result = dict(schema_version=1, authority="observation_only",
                      observation_status=("diagnostic_unavailable" if getattr(bot, "_monitor_ordinary_diagnostic_failed", False)
                                          else "not_evaluated"), scopes=[], pending_confirmations=confirmations)
        observed = getattr(bot, "_monitor_ordinary_observation", None)
        if observed:
            result.update(deepcopy(observed))
            age = now_ms - observed["captured_at_ms"]
            reasons = []
            if age < 0:
                reasons.append("clock_before_capture")
            if observed["input_expires_at_ms"] is None:
                reasons.append("input_age_unknown")
            elif now_ms > observed["input_expires_at_ms"]:
                reasons.append("input_expired")
            if observed["account_generation"] != getattr(bot, "_account_invalidation_generation", 0):
                reasons.append("account_generation_changed")
            if confirmations:
                reasons.append("account_confirmation_pending")
            result.update(age_ms=max(0, age), stale_reasons=reasons,
                          observation_status="stale" if reasons else "current")
        for field, attr in (("last_create_filter", "_monitor_last_create_filter"),
                            ("last_cancel_first_barrier", "_monitor_last_cancel_first_barrier")):
            record = getattr(bot, attr, None)
            if record is not None:
                result[field] = dict(record, age_ms=max(0, now_ms - record["observed_at_ms"]),
                                     status="last_observed_outcome")
        return result
    except Exception as exc:
        _failed("ordinary snapshot", exc)
        return dict(schema_version=1, authority="observation_only", observation_status="diagnostic_unavailable", scopes=[])
