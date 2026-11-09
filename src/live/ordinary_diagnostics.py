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


def describe_ordinary_admission(bot, *, now_ms, handoff, reason, create_count, cancel_count):
    """Describe the already-read account clocks at a guarded handoff."""
    ledger = bot._ensure_freshness_ledger()
    pending = getattr(bot, "_authoritative_pending_confirmations", {})
    max_age_ms = bot._live_market_snapshot_max_age_ms()
    receipts = []
    for surface in ("balance", "positions", "open_orders"):
        state = ledger.surfaces[surface]
        age_ms = now_ms - state.updated_ms
        if state.updated_ms <= 0 or age_ms < 0:
            age_ms, clock = None, "invalid"
        else:
            clock = "stale" if age_ms > max_age_ms else "fresh"
        receipts.append(dict(
            surface=surface,
            age_ms=age_ms,
            clock_status=clock,
            confirmation_pending=int(pending.get(surface, 0)) > state.epoch,
        ))
    return dict(
        handoff=handoff,
        reason=reason,
        create_count=create_count,
        cancel_count=cancel_count,
        max_age_ms=max_age_ms,
        receipts=receipts,
    )


def _input_receipts(bot, wave, symbols, *, now_ms, account_max_age_ms):
    """Project the immutable inputs consumed by this computation, never live caches."""
    from live.planning_snapshot import PlanningSnapshot

    snapshot = getattr(bot, "_current_planning_snapshot", None)
    detail = dict(account_max_age_ms=(account_max_age_ms if type(account_max_age_ms) is int and account_max_age_ms >= 0 else None),
                  account_observed_at_ms={}, account_generation=None,
                  required_market_symbol_count=len(symbols), market_receipt_count=0,
                  market_max_age_ms=None, market_oldest_observed_at_ms=None,
                  planning_captured_at_ms=None,
                  hsl_inputs_required=bool(wave is not None and (wave.decisions or wave.unavailable)))
    problems, deadlines = set(), []

    def expiry(stamp, ttl, latest, problem):
        if (type(stamp) is not int or not 0 < stamp <= latest
                or type(ttl) is not int or ttl < 0):
            problems.add(problem)
            return None
        deadlines.append(stamp + ttl)
        return stamp + ttl

    if not isinstance(snapshot, PlanningSnapshot):
        problems.add("planning_snapshot_unavailable")
    else:
        if type(snapshot.account_invalidation_generation) is int and snapshot.account_invalidation_generation >= 0:
            detail["account_generation"] = snapshot.account_invalidation_generation
        else:
            problems.add("planning_account_generation_unavailable")
        snapshot_time = snapshot.ts_ms
        if type(snapshot_time) is not int or not 0 < snapshot_time <= now_ms:
            problems.add("planning_capture_time_unavailable")
            snapshot_time = 0
        else:
            detail["planning_captured_at_ms"] = snapshot_time
        if set(snapshot.symbols) != symbols or len(snapshot.symbols) != len(symbols):
            problems.add("planning_symbol_cohort_mismatch")
        if wave is not None and snapshot.account_invalidation_generation != wave.generation:
            problems.add("planning_account_generation_mismatch")
        surfaces = {row.name: row for row in snapshot.surfaces}
        if len(surfaces) != len(snapshot.surfaces):
            problems.add("duplicate_surface_receipt")
        for name in ("balance", "positions", "open_orders"):
            row = surfaces.get(name)
            stamp = row.updated_ms if row is not None else None
            detail["account_observed_at_ms"][name] = stamp if type(stamp) is int and stamp >= 0 else None
            if name not in snapshot.required_surfaces:
                problems.add("account_surface_not_declared")
            expiry(stamp, account_max_age_ms, snapshot_time, name + "_receipt_unavailable")
        # Fill/candle readiness is generation/coverage based, not the live quote TTL.
        for name in snapshot.required_surfaces:
            row = surfaces.get(name)
            if (row is None or type(row.epoch) is not int or type(row.min_epoch) is not int
                    or row.epoch < row.min_epoch):
                problems.add("required_surface_epoch_unavailable")
        quotes = {row.symbol: row for row in snapshot.market_snapshots}
        detail["market_receipt_count"] = len(quotes)
        detail["market_max_age_ms"] = (snapshot.market_snapshot_max_age_ms
            if type(snapshot.market_snapshot_max_age_ms) is int and snapshot.market_snapshot_max_age_ms >= 0 else None)
        missing = symbols - set(quotes)
        detail["missing_market_symbol_count"] = len(missing)
        detail["missing_market_symbols"] = sorted(missing)[:3]
        if len(quotes) != len(snapshot.market_snapshots) or set(quotes) != symbols:
            problems.add("market_receipt_cohort_mismatch")
        quote_times = []
        for symbol in symbols:
            row = quotes.get(symbol)
            stamp = row.fetched_ms if row is not None else None
            if expiry(stamp, snapshot.market_snapshot_max_age_ms, snapshot_time,
                      "market_receipt_unavailable") is not None:
                quote_times.append(stamp)
        detail["market_oldest_observed_at_ms"] = min(quote_times, default=None)
    if detail["hsl_inputs_required"]:
        if type(wave.captured_ms) is not int or not 0 < wave.captured_ms <= now_ms:
            problems.add("hsl_capture_time_unavailable")
        for stamp in (wave.captured_ms, wave.position_observed_ms,
                      wave.balance_observed_ms, *wave.mark_observed_ms):
            expiry(stamp, account_max_age_ms, wave.captured_ms, "hsl_receipt_unavailable")
    detail["problems"] = sorted(problems)
    return (min(deadlines) if deadlines and not problems else None), detail


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
            if not isinstance(symbol, str) or not re.fullmatch(r"(?:[A-Za-z0-9_.-]{1,16}:)?[A-Za-z0-9_.-]{1,32}(?:/[A-Za-z0-9_.-]{1,16})?(?::[A-Za-z0-9_.-]{1,16})?", symbol):
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
        expires, receipts = _input_receipts(bot, wave, set(idx_to_symbol.values()),
                                           now_ms=now_ms, account_max_age_ms=max_age)
        balance_at = getattr(wave, "balance_observed_ms", None) if receipts["hsl_inputs_required"] else None
        bot._monitor_ordinary_observation = dict(
            source="rust_orchestrator", authority="observation_only",
            stage="computed_native_plan",
            captured_at_ms=captured, completed_at_ms=now_ms,
            input_expires_at_ms=expires,
            balance_observed_at_ms=balance_at,
            input_receipts=receipts,
            account_generation=(receipts["account_generation"] if receipts["account_generation"] is not None else
                                wave.generation if wave is not None else None),
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
