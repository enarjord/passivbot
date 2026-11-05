"""Bounded presentation of Rust unstuck decisions; never an execution authority."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import logging
import math

from live.console_health import token
from utils import utc_ms

OBSERVATION_MS = 5 * 60_000
REMINDER_MS = 20 * 60_000
STALE_MS = 5 * 60_000
# Presentation priority only: Rust has already classified every held position.
REASONS = (
    "selected",
    "loss_gate_blocked",
    "superseded",
    "finalization_filtered",
    "inputs_unavailable",
    "waiting_ema",
    "eligible",
    "allowance_exhausted",
    "below_exposure_trigger",
    "mode_blocked",
    "planning_disabled",
    "disabled",
)
NUMBERS = (
    "allowance",
    "exposure_ratio",
    "exposure_trigger_ratio",
    "current_price",
    "ema_trigger_price",
    "qty",
    "price",
    "estimated_pnl",
)


def observe(bot, diagnostics, idx_to_symbol):
    """Copy only bounded public-schema fields from the completed native plan."""
    bot._unstuck_native_observation = None
    rows = diagnostics.get("unstuck")
    if not isinstance(rows, list):
        bot._unstuck_native_observation = None
        return
    sides = {}
    for pside in ("long", "short"):
        items = []
        for row in rows:
            if row.get("pside") != pside:
                continue
            reason = row["reason"]
            if reason not in REASONS:
                raise ValueError("unknown native unstuck reason")
            item = {
                "symbol": idx_to_symbol[row["symbol_idx"]],
                "reason": reason,
                "proposed": bool(row["proposed"]),
            }
            for key in NUMBERS:
                value = row.get(key)
                if value is not None:
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError("non-finite native unstuck diagnostic")
                    item[key] = value
            items.append(item)
        counts = dict(sorted(Counter(item["reason"] for item in items).items()))

        # Prefer the nearest EMA gate within the same native reason, then stable identity.
        def rank(item):
            price, trigger = item.get("current_price"), item.get("ema_trigger_price")
            distance = abs(trigger / price - 1) if price and trigger else math.inf
            return REASONS.index(item["reason"]), distance, item["symbol"]

        items.sort(key=rank)
        allowances = [item["allowance"] for item in items if "allowance" in item]
        info = {
            "native": True,
            "status": items[0]["reason"] if items else "no_positions",
            "held_count": len(items),
            "eligible_count": sum(
                item["proposed"] or item["reason"] == "eligible" for item in items
            ),
            "reason_counts": counts,
            "samples": items[:3],
            "samples_omitted": max(0, len(items) - 3),
            "selected_symbol": next(
                (item["symbol"] for item in items if item["reason"] == "selected"), None
            ),
        }
        if allowances:
            info.update(allowance=min(allowances), allowance_max=max(allowances))
        sides[pside] = info
    bot._unstuck_native_observation = {"sides": sides, "observed_ms": utc_ms()}


def snapshot(bot, now_ms):
    observation = getattr(bot, "_unstuck_native_observation", None)
    if observation is None:
        return {
            side: {"native": True, "status": "awaiting_plan", "samples": []}
            for side in ("long", "short")
        }
    sides = deepcopy(observation["sides"])
    age = max(0, now_ms - observation["observed_ms"])
    for info in sides.values():
        info["plan_age_ms"] = age
        if age >= STALE_MS:
            info["status"] = "stale_plan"
            info["selected_symbol"] = None
        info["has_open_order"] = bool(bot.has_open_unstuck_order())
    return sides


def _semantic(info):
    return (
        info["status"],
        tuple(sorted(info.get("reason_counts", {}).items())),
        info.get("has_open_order"),
        tuple(
            None if info.get(k) is None else info[k] > 0
            for k in ("allowance", "allowance_max")
        ),
    )


def _identity(info):
    samples = info.get("samples", [])
    return info.get("selected_symbol"), samples[0]["symbol"] if samples else None


def _material(previous, current):
    for key in ("allowance", "allowance_max"):
        a, b = previous.get(key), current.get(key)
        if a is None or b is None:
            if a != b:
                return True
        elif (a > 0) != (b > 0) or abs(a - b) >= max(abs(a), 1e-12) * 0.05:
            return True
    return False


def maybe_log_status(bot):
    """Initial/transition INFO, five-minute structured observations, 20-minute reminders."""
    now = utc_ms()
    sides = snapshot(bot, now)
    previous = getattr(bot, "_unstuck_visible_sides", None)
    last_info = getattr(bot, "_unstuck_visible_ms", None)
    elapsed = math.inf if last_info is None else now - last_info
    semantic = previous is None or any(
        _semantic(sides[s]) != _semantic(previous[s]) for s in sides
    )
    detail = previous is None or any(
        _identity(sides[s]) != _identity(previous[s])
        or _material(previous[s], sides[s])
        for s in sides
    )
    changed = semantic or (detail and elapsed >= OBSERVATION_MS)
    visible = changed or elapsed >= REMINDER_MS
    last_event = getattr(bot, "_unstuck_observation_ms", None)
    if not visible and last_event is not None and now - last_event < OBSERVATION_MS:
        return
    legacy_delivered = visible and not bot._live_event_console_available()
    if legacy_delivered:
        for line in format_sides(sides):
            logging.info("%s", line)
    emitted = bot._emit_unstuck_status_event(
        side_statuses=sides, changed=changed, operator_visible=visible
    )
    if emitted is not None or legacy_delivered:
        bot._unstuck_observation_ms = now
    if visible and (legacy_delivered or emitted is not None):
        bot._unstuck_visible_sides = sides
        bot._unstuck_visible_ms = now


def format_sides(sides):
    """One compact record per side, with a second record for native proposal details."""
    lines = []

    def number(value):
        return "?" if value is None else f"{float(value):.5g}"

    for side in ("long", "short"):
        info = sides.get(side, {})
        status = info.get("status", "unavailable")
        line = f"[unstuck] {side} {token(status, 28)}"
        allowance = info.get("allowance")
        if allowance is not None:
            line += f" allowance={number(allowance)}"
            if info.get("allowance_max") != allowance:
                line += f"..{number(info.get('allowance_max'))}"
            line += " quote"
        if "held_count" in info:
            line += f" eligible={info['eligible_count']}/{info['held_count']}"
        line += f" selected={token(info.get('selected_symbol') or 'none', 18)}"
        if "has_open_order" in info:
            line += f" open_any={'yes' if info['has_open_order'] else 'no'}"
        if "plan_age_ms" in info:
            line += f" plan_age={number(info['plan_age_ms'] / 1000)}s"
        lines.append(line)
        samples = info.get("samples", [])
        if samples and status not in {"stale_plan", "awaiting_plan"}:
            item = samples[0]
            line = f"[unstuck] {side} watch={token(item['symbol'], 18)} {token(item['reason'], 28)}"
            if item.get("proposed"):
                line += f" qty={number(item.get('qty'))} price={number(item.get('price'))} est_pnl={number(item.get('estimated_pnl'))}"
            else:
                if "exposure_ratio" in item:
                    line += f" WE/eWEL={number(item['exposure_ratio'])} trigger={number(item.get('exposure_trigger_ratio'))}"
                if "ema_trigger_price" in item:
                    line += f" price={number(item.get('current_price'))} ema_trigger={number(item['ema_trigger_price'])}"
            lines.append(line)
    return lines
