"""Observe delayed close replacements from admitted plans and exchange facts."""

from dataclasses import dataclass
import logging
import math

from live.console_health import token
from live.event_bus import EventTags, EventTypes
from live.events import DiagnosticEvent, emit_diagnostic_event, run_diagnostic_step
from live.freshness import ACCOUNT_SURFACES
from live.reconciler import extract_order_remaining_qty


_CLOSE_DELAY_MS = 60_000
_MAX_TRACKED_SCOPES = 64
_SURFACES = {"positions", "open_orders"}
_CLOSE_SIDES = {"long": "sell", "short": "buy"}


@dataclass
class _Replacement:
    canceled_id: str
    canceled_ms: int
    canceled_epoch: int
    previous_defer: dict | None
    cancel_wave_id: int | None
    account_confirmed_ms: int | None = None
    first_plan_ms: int | None = None
    planned_order: dict | None = None
    create_admitted_ms: int | None = None
    last_missing_surfaces: tuple[str, ...] = ()
    last_create_defer: str | None = None
    sent: bool = False
    warned: bool = False
    observed_uncovered: bool = False


def _wave_id(wave):
    value = wave.get("id") if wave is not None else None
    return value if type(value) is int and value > 0 else None


def _observe_missing_surfaces(bot, replacement):
    defer = getattr(bot, "_console_ordinary_last_defer", None)
    if defer and defer is not replacement.previous_defer:
        missing = tuple(
            name
            for name in defer.get("missing_surfaces") or ()
            if isinstance(name, str) and name in ACCOUNT_SURFACES
        )
        if missing:
            replacement.last_missing_surfaces = missing


def _emit_timing(
    bot, scope, replacement, outcome, now_ms, *, create_wave=None, action_index=None
):
    if now_ms < replacement.canceled_ms:
        return
    _observe_missing_surfaces(bot, replacement)
    elapsed_ms = now_ms - replacement.canceled_ms
    payload = {
        "outcome": outcome,
        "elapsed_ms": elapsed_ms,
    }
    for name, value in (
        ("cancel_wave_id", replacement.cancel_wave_id),
        ("create_wave_id", _wave_id(create_wave)),
        ("create_action_index", action_index),
        ("last_create_defer", replacement.last_create_defer),
        ("last_missing_surfaces", list(replacement.last_missing_surfaces) or None),
    ):
        if value is not None:
            payload[name] = value
    for name, value in (
        ("account_confirmed_ms", replacement.account_confirmed_ms),
        ("first_plan_ms", replacement.first_plan_ms),
        ("create_admitted_ms", replacement.create_admitted_ms if outcome == "create_sent" else None),
    ):
        if value is not None and 0 <= value <= elapsed_ms:
            payload[name] = value

    run_diagnostic_step(
        "emit close replacement timing",
        lambda: emit_diagnostic_event(
            bot,
            DiagnosticEvent.build(
                EventTypes.ORDER_CLOSE_REPLACEMENT_TIMING,
                (EventTags.ORDER, EventTags.EXECUTION),
                payload,
                ts_ms=now_ms,
                symbol=token(scope[0], 48),
                pside=scope[1],
            ),
        ),
    )


def _close_scope(order):
    symbol = order.get("symbol")
    pside = order.get("position_side")
    close_side = _CLOSE_SIDES.get(pside)
    if (
        close_side is not None and close_side == order.get("side")
        and isinstance(symbol, str) and symbol and "://" not in symbol
    ):
        return symbol, pside
    return None


def _ordinary_close(order):
    return (
        order.get("reduce_only") is True
        and order.get("type", "limit") == "limit"
        and order.get("execution_priority", "ordinary") == "ordinary"
    )


def _ordinary_closes_by_scope(creates):
    close_creates = {}
    for order in creates:
        if _ordinary_close(order) and (scope := _close_scope(order)) is not None:
            close_creates.setdefault(scope, []).append(order)
    return close_creates


def _observe_first_plan(bot, replacement, now_ms):
    if replacement.first_plan_ms is None:
        replacement.first_plan_ms = now_ms - replacement.canceled_ms
    _observe_missing_surfaces(bot, replacement)


def observe_reconciled_plan(bot, creates, now_ms):
    """Time the validated plan before it waits for the execution owner."""
    tracked = getattr(bot, "_console_close_replacements", {})
    close_creates = _ordinary_closes_by_scope(creates)
    for scope, replacement in tracked.items():
        if (
            not replacement.sent
            and len(close_creates.get(scope, ())) == 1
            and now_ms >= replacement.canceled_ms
        ):
            _observe_first_plan(bot, replacement, now_ms)


def observe_plan(bot, wave, cancels, creates, now_ms):
    """Retain exact one-to-one replacement evidence from the admitted ordinary plan."""
    tracked = getattr(bot, "_console_close_replacements", {})
    if not tracked and (not cancels or not creates):
        return

    close_creates = _ordinary_closes_by_scope(creates)
    for scope, replacement in list(tracked.items()):
        if replacement.observed_uncovered and scope not in close_creates:
            if replacement.warned:
                _log_resolution(scope, replacement, now_ms, "latest plan has no close to submit")
            del tracked[scope]
            if not replacement.sent:
                _emit_timing(bot, scope, replacement, "no_close_in_plan", now_ms)
            continue
        if replacement.sent:
            continue
        replacement.planned_order = None
        replacement.create_admitted_ms = None
        matches = close_creates.get(scope, ())
        if len(matches) == 1 and now_ms >= replacement.canceled_ms:
            replacement.planned_order = matches[0]
            _observe_first_plan(bot, replacement, now_ms)
    if wave is None:
        return

    candidates = {}
    for order in cancels:
        if not _ordinary_close(order) or order.get("_context") != "replace":
            continue
        scope = _close_scope(order)
        matches = close_creates.get(scope, ())
        if len(matches) != 1 or matches[0].get("_context") != "replace":
            continue
        delta = order.get("_delta")
        if not delta or delta != matches[0].get("_delta"):
            continue
        order_id = order.get("id")
        if not order_id:
            continue
        resting = [
            row for row in getattr(bot, "open_orders", {}).get(scope[0], ())
            if _close_scope(row) == scope
        ]
        if len(resting) == 1 and str(resting[0].get("id")) == str(order_id):
            candidates[id(order)] = (scope, str(order_id))
    if candidates:
        wave["close_replacement_candidates"] = candidates


def observe_cancel_ack(bot, wave, order, now_ms):
    """Start at the unambiguous cancel acknowledgment for the tracked close."""
    candidate = wave.get("close_replacement_candidates", {}).get(id(order)) if wave else None
    if candidate is None:
        return
    scope, order_id = candidate
    tracked = getattr(bot, "_console_close_replacements", None)
    if tracked is None:
        tracked = bot._console_close_replacements = {}
    if len(tracked) >= _MAX_TRACKED_SCOPES or scope in tracked:
        return
    tracked[scope] = _Replacement(
        canceled_id=order_id,
        canceled_ms=now_ms,
        canceled_epoch=bot.freshness_ledger.epoch,
        previous_defer=getattr(bot, "_console_ordinary_last_defer", None),
        cancel_wave_id=_wave_id(wave),
    )


def observe_confirmation(bot, now_ms):
    """Record when all four post-cancel account surfaces have been confirmed."""
    ledger = bot.freshness_ledger
    pending = getattr(bot, "_authoritative_pending_confirmations", {})
    epochs = {surface: ledger.surface_epoch(surface) for surface in ACCOUNT_SURFACES}
    if any(epoch < pending.get(surface, 0) for surface, epoch in epochs.items()):
        return
    oldest_epoch = min(epochs.values())
    for replacement in getattr(bot, "_console_close_replacements", {}).values():
        if (
            not replacement.sent
            and replacement.account_confirmed_ms is None
            and oldest_epoch > replacement.canceled_epoch
            and now_ms >= replacement.canceled_ms
        ):
            replacement.account_confirmed_ms = now_ms - replacement.canceled_ms
            _observe_missing_surfaces(bot, replacement)


def observe_admission(bot, orders, now_ms):
    """Record executor admission for the specific planned replacement order."""
    tracked = getattr(bot, "_console_close_replacements", {})
    for order in orders:
        replacement = tracked.get(_close_scope(order))
        if (
            replacement is not None
            and replacement.planned_order is order
            and replacement.first_plan_ms <= now_ms - replacement.canceled_ms
        ):
            replacement.create_admitted_ms = now_ms - replacement.canceled_ms


def observe_create_sent(bot, wave, order, now_ms, *, action_index=None):
    """Publish timing once the specific replacement enters guarded transport."""
    scope = _close_scope(order)
    replacement = getattr(bot, "_console_close_replacements", {}).get(scope)
    if (
        replacement is None
        or replacement.sent
        or replacement.planned_order is not order
        or replacement.create_admitted_ms is None
        or replacement.create_admitted_ms > now_ms - replacement.canceled_ms
    ):
        return
    replacement.sent = True
    replacement.planned_order = None
    _emit_timing(
        bot, scope, replacement, "create_sent", now_ms,
        create_wave=wave, action_index=action_index,
    )


def observe_create_defer(bot, orders, reason):
    """Attach a known create-side guard to the same symbol and position side."""
    tracked = getattr(bot, "_console_close_replacements", {})
    for order in orders:
        if _ordinary_close(order) and (scope := _close_scope(order)) in tracked:
            tracked[scope].last_create_defer = reason


def _log_resolution(scope, replacement, now_ms, description):
    logging.info(
        "[order] close replacement resolved | %s %s | elapsed=%.1fs | %s",
        token(scope[0], 48), scope[1], (now_ms - replacement.canceled_ms) / 1000, description,
    )


def observe_account(bot, refreshed_surfaces, now_ms):
    """Use only a committed post-cancel positions and open-orders cohort."""
    tracked = getattr(bot, "_console_close_replacements", None)
    if not tracked or not _SURFACES <= refreshed_surfaces:
        return
    ledger = bot.freshness_ledger
    pending = getattr(bot, "_authoritative_pending_confirmations", {})
    if not _SURFACES <= ledger.surfaces_at_epoch() or any(
        pending.get(surface, 0) > ledger.epoch for surface in _SURFACES
    ):
        return

    for scope, replacement in list(tracked.items()):
        if ledger.epoch <= replacement.canceled_epoch or now_ms < replacement.canceled_ms:
            continue
        symbol, pside = scope
        position_qty = abs(float(bot.positions.get(symbol, {}).get(pside, {}).get("size", 0)))
        if not math.isfinite(position_qty):
            continue
        if position_qty == 0:
            if replacement.warned:
                _log_resolution(scope, replacement, now_ms, "position closed")
            del tracked[scope]
            if not replacement.sent:
                _emit_timing(bot, scope, replacement, "position_closed", now_ms)
            continue

        resting = [order for order in bot.fetched_open_orders if _close_scope(order) == scope]
        if resting:
            replacement.observed_uncovered = False
            # An old order still visible after cancel proves neither the gap nor recovery.
            remaining = [
                extract_order_remaining_qty(row)
                for row in resting
                if row.get("id") and str(row["id"]) != replacement.canceled_id
                and bot._canonical_open_order_reduce_only(row) is True
            ]
            if any(qty is not None and qty > 0 for qty in remaining):
                if replacement.warned:
                    description = "resting close observed"
                    if all(qty is not None for qty in remaining):
                        description += f" | resting_close_qty={sum(remaining):g}/{position_qty:g}"
                    _log_resolution(scope, replacement, now_ms, description)
                del tracked[scope]
                if not replacement.sent:
                    _emit_timing(bot, scope, replacement, "resting_close_observed", now_ms)
            continue

        replacement.observed_uncovered = True
        elapsed_ms = now_ms - replacement.canceled_ms
        if replacement.warned or elapsed_ms < _CLOSE_DELAY_MS:
            continue
        defer = getattr(bot, "_console_ordinary_last_defer", None)
        blockers = []
        if defer and defer is not replacement.previous_defer:
            missing = defer.get("missing_surfaces", ())
            last_defer = token(",".join(missing) or defer.get("reason"), 32)
            blockers.append(f"last_planning_defer={last_defer}")
        if replacement.last_create_defer:
            blockers.append(f"last_create_defer={replacement.last_create_defer}")
        logging.warning(
            "[order] close replacement delayed | %s %s | elapsed=%.1fs | "
            "resting_close_qty=0/%g | %s",
            token(symbol, 32), pside, elapsed_ms / 1000, position_qty,
            " ".join(blockers) or "blocker=unreported",
        )
        replacement.warned = True
        if not replacement.sent:
            _emit_timing(bot, scope, replacement, "still_waiting", now_ms)
