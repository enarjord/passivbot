"""Observe sustained forager exclusions and free slots without a working entry."""

from config.access import get_optional_config_value

EXCLUSION_WARNING_CYCLES = 10
DEFAULT_NO_ENTRIES_WARNING_MINUTES = 15.0


def observe(
    bot,
    selections,
    *,
    now_ms: int,
    active_entry_cooldowns: set[tuple[int, str]] | None = None,
) -> dict[str, dict]:
    """Track consecutive completed planning cycles without affecting trading readiness."""
    active_entry_cooldowns = active_entry_cooldowns or set()
    previous = getattr(bot, "_forager_health_state", {})
    threshold_ms = (
        get_optional_config_value(
            getattr(bot, "config", {}),
            "logging.forager_no_entries_warning_minutes",
            DEFAULT_NO_ENTRIES_WARNING_MINUTES,
        )
        * 60_000
    )
    positions = getattr(bot, "positions", {})
    open_orders = getattr(bot, "open_orders", {})
    state = {}
    for pside, selection in selections.items():
        prior = previous.get(pside, {})
        prior_streaks = prior.get("exclusion_streaks", {})
        exclusions = selection["excluded_candidates"]
        streaks = {}
        persistent = []
        for row in exclusions:
            key = (row["symbol"], row["feature"], row["reason"])
            streak = prior_streaks.get(key, 0) + 1
            streaks[key] = streak
            if streak == EXCLUSION_WARNING_CYCLES:
                persistent.append(row)
        held = {
            symbol for symbol, sides in positions.items()
            if sides.get(pside, {}).get("size", 0.0) != 0.0
        }
        entry_side = "buy" if pside == "long" else "sell"
        working_entry = any(
            order.get("position_side") == pside
            and order.get("side") == entry_side
            and not order.get("reduce_only", order.get("reduceOnly", False))
            for symbol, orders in open_orders.items()
            if symbol not in held
            for order in orders
        )
        opened_position = bool(held - prior.get("held", held))
        selected = selection["selected_symbol_indices"]
        cooling_down = bool(selected) and all(
            (idx, pside) in active_entry_cooldowns for idx in selected
        )
        stalled = (
            threshold_ms > 0
            and selection["forager_mode"]
            and selection["slots_to_fill"] > 0
            and selection["candidate_count"] > 0
            and not selection["deferral_reason"]
            and not cooling_down
            and not working_entry
        )
        since_ms = prior.get("since_ms")
        last_warning_ms = prior.get("last_warning_ms")
        if not stalled or opened_position:
            since_ms = last_warning_ms = None
        if stalled and since_ms is None:
            since_ms = now_ms
        stall_ms = now_ms - since_ms if since_ms is not None else 0
        warning = stalled and stall_ms >= threshold_ms
        warning_due = warning and (
            last_warning_ms is None or now_ms - last_warning_ms >= threshold_ms
        )
        state[pside] = {
            "held": held,
            "exclusion_streaks": streaks,
            "persistent_exclusions": persistent,
            "since_ms": since_ms,
            "last_warning_ms": now_ms if warning_due else last_warning_ms,
            "stall_ms": stall_ms,
            "stall_warning": warning,
            "stall_warning_due": warning_due,
        }
    bot._forager_health_state = state
    return state
