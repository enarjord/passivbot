"""Observe delayed close replacements from admitted plans and exchange facts."""

from dataclasses import dataclass
import logging
import math

from live.console_health import token
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
    last_create_defer: str | None = None
    warned: bool = False
    observed_uncovered: bool = False


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


def observe_plan(bot, wave, cancels, creates, now_ms):
    """Retain exact one-to-one replacement evidence from the admitted ordinary plan."""
    tracked = getattr(bot, "_console_close_replacements", {})
    if not tracked and (not cancels or not creates):
        return

    close_creates = {}
    for order in creates:
        if _ordinary_close(order) and (scope := _close_scope(order)) is not None:
            close_creates.setdefault(scope, []).append(order)
    for scope, replacement in list(tracked.items()):
        if replacement.observed_uncovered and scope not in close_creates:
            if replacement.warned:
                _log_resolution(scope, replacement, now_ms, "latest plan has no close to submit")
            del tracked[scope]
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
