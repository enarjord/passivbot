from __future__ import annotations

import asyncio
import logging
import sys
from typing import Iterable

from live.freshness import ACCOUNT_SURFACES, LIVE_STATE_SURFACES
from live.market_snapshot import MarketSnapshot
from live.planning_snapshot import PlanningSnapshot
from utils import utc_ms


def _utc_ms() -> int:
    passivbot_module = sys.modules.get("passivbot")
    time_fn = getattr(passivbot_module, "utc_ms", None)
    if callable(time_fn):
        return int(time_fn())
    return int(utc_ms())


def staged_planner_required_surfaces(
    bot, *, include_market_snapshot: bool = True
) -> frozenset[str]:
    """Return live input surfaces required before staged order planning may proceed."""
    del bot
    surfaces = set(LIVE_STATE_SURFACES)
    if not include_market_snapshot:
        surfaces.discard("market_snapshot")
    return frozenset(surfaces)


def staged_planner_surface_min_epochs(
    bot, required: set[str] | frozenset[str]
) -> dict[str, int]:
    """Return minimum acceptable ledger epoch per staged planner input surface."""
    current_epoch = int(getattr(bot, "_authoritative_refresh_epoch", 0) or 0)
    pending = dict(getattr(bot, "_authoritative_pending_confirmations", {}) or {})
    min_epochs: dict[str, int] = {}
    for surface in required:
        if surface in ACCOUNT_SURFACES:
            min_epochs[surface] = max(1, int(pending.get(surface, 0) or 0))
        elif surface in {"completed_candles", "market_snapshot"}:
            min_epochs[surface] = current_epoch
        else:
            min_epochs[surface] = current_epoch
    return min_epochs


def staged_planner_precondition_state(
    bot,
    *,
    include_market_snapshot: bool = True,
    symbols: Iterable[str] | None = None,
) -> tuple[bool, dict]:
    """Return staged planner input-completeness state for the current planning pass."""
    ledger = bot._ensure_freshness_ledger()
    required = bot._staged_planner_required_surfaces(
        include_market_snapshot=include_market_snapshot
    )
    min_epochs = bot._staged_planner_surface_min_epochs(required)
    missing = sorted(
        surface
        for surface in required
        if ledger.surface_epoch(surface) < int(min_epochs.get(surface, 0) or 0)
    )
    invalid: dict[str, list] = {}
    if "completed_candles" in required and "completed_candles" not in missing:
        expected_symbols = tuple(bot._urgent_active_candle_symbols())
        candle_check_ms = ledger.surface_updated_ms(
            "completed_candles"
        ) or bot._completed_candle_health_now_ms()
        signature, candle_missing = bot._completed_candle_freshness_signature(
            expected_symbols, now_ms=candle_check_ms
        )
        stamped_signature = ledger.surface_signature("completed_candles")
        if candle_missing or stamped_signature != signature:
            missing.append("completed_candles")
            invalid["completed_candles"] = candle_missing or (
                bot._completed_candle_signature_mismatch_details(
                    expected_symbols=expected_symbols,
                    expected_signature=signature,
                    stamped_signature=stamped_signature,
                )
            )
    if (
        include_market_snapshot
        and "market_snapshot" in required
        and "market_snapshot" not in missing
    ):
        expected_market_symbols = tuple(
            sorted(dict.fromkeys(str(symbol) for symbol in (symbols or []) if symbol))
        )
        if expected_market_symbols:
            snapshot_invalid = bot._market_snapshot_signature_invalid(
                expected_market_symbols
            )
            if snapshot_invalid:
                missing.append("market_snapshot")
                invalid["market_snapshot"] = snapshot_invalid
    return not missing, {
        "missing": sorted(set(missing)),
        "required": sorted(required),
        "epoch": int(getattr(bot, "_authoritative_refresh_epoch", 0) or 0),
        "min_epochs": min_epochs,
        "invalid": invalid,
    }


def assert_staged_planner_preconditions(
    bot,
    *,
    include_market_snapshot: bool = True,
    context: str = "planning",
    symbols: Iterable[str] | None = None,
) -> None:
    """Hard-fail before Rust planning if staged live inputs are incomplete."""
    ok, details = bot._staged_planner_precondition_state(
        include_market_snapshot=include_market_snapshot,
        symbols=symbols,
    )
    if ok:
        return
    missing = ",".join(details["missing"])
    required = ",".join(details["required"])
    raise RuntimeError(
        f"staged planner precondition failed before {context}: "
        f"missing current-epoch surfaces={missing} epoch={details['epoch']} "
        f"required={required}"
    )


def staged_execution_ready_state(
    bot,
    *,
    include_market_snapshot: bool = True,
    context: str = "planning",
    symbols: Iterable[str] | None = None,
) -> tuple[bool, dict]:
    """Return whether live execution may proceed without raising on transient staleness."""
    ok, details = bot._staged_planner_precondition_state(
        include_market_snapshot=include_market_snapshot,
        symbols=symbols,
    )
    details = dict(details)
    details["context"] = context
    details["defer_reason"] = "staged_planner_inputs_not_fresh"
    return ok, details


def format_staged_execution_defer_message(bot, details: dict) -> str:
    """Return an operator-facing staged-defer summary."""
    missing = tuple(details.get("missing", ()))
    invalid = details.get("invalid") or {}
    context = str(details.get("context") or "planning")
    retry_note = "will_retry=automatic"
    scope_note = "scope=planner_cycle"
    headline = "staged planning deferred"
    dependency = ",".join(missing) if missing else "unknown"
    extra_parts = []
    if "completed_candles" in missing:
        headline = "staged planning deferred: completed candle target changed or missing"
        dependency = "completed_candles"
        extra_parts.append("action=refresh_candles_then_retry")
    elif missing:
        extra_parts.append("action=refresh_required_surfaces_then_retry")
    parts = [
        headline,
        f"context={context}",
        f"dependency={dependency}",
        scope_note,
        retry_note,
        f"epoch={int(details.get('epoch', 0) or 0)}",
    ]
    parts.extend(extra_parts)
    if invalid:
        summaries = []
        for surface, items in invalid.items():
            if not isinstance(items, list) or not items:
                summaries.append(str(surface))
                continue
            first = items[0]
            if not isinstance(first, dict):
                summaries.append(f"{surface}:{type(first).__name__}")
                continue
            reason = first.get("reason") or "invalid"
            if reason == "signature_mismatch":
                mismatch_type = str(first.get("mismatch_type") or "unknown")
                changed_symbols = list(first.get("changed_symbols") or [])
                missing_symbols = list(first.get("missing_symbols") or [])
                extra_symbols = list(first.get("extra_symbols") or [])
                symbol_bits = []
                if changed_symbols:
                    symbol_bits.append(
                        "changed="
                        + "|".join(bot._log_symbol(sym) for sym in changed_symbols[:4])
                    )
                if missing_symbols:
                    symbol_bits.append(
                        "missing="
                        + "|".join(bot._log_symbol(sym) for sym in missing_symbols[:4])
                    )
                if extra_symbols:
                    symbol_bits.append(
                        "extra="
                        + "|".join(bot._log_symbol(sym) for sym in extra_symbols[:4])
                    )
                summaries.append(
                    f"{surface}:{mismatch_type}"
                    f" expected={int(first.get('expected_count') or 0)}"
                    f" stamped={int(first.get('stamped_count') or 0)}"
                    f" changed={int(first.get('changed_count') or 0)}"
                    + (f" {' '.join(symbol_bits)}" if symbol_bits else "")
                )
            else:
                symbol = first.get("symbol")
                suffix = f":{bot._log_symbol(symbol)}" if symbol else ""
                summaries.append(f"{surface}:{reason}{suffix}")
        if summaries:
            parts.append("details=" + ",".join(summaries[:4]))
    return " | ".join(parts)


def handle_staged_execution_precondition_error(
    bot, exc: RuntimeError
) -> tuple[bool, dict]:
    """Classify staged precondition failures as safe live-loop defers."""
    message = str(exc)
    transient_prefixes = (
        "staged planner precondition failed",
        "planning snapshot invalid before capture",
    )
    if not message.startswith(transient_prefixes):
        return False, {}
    ok, details = bot._staged_execution_ready_state(
        include_market_snapshot=True,
        context="rust order calculation",
    )
    if ok:
        details["missing"] = []
    details["exception"] = message
    return True, details


def log_staged_execution_defer(bot, details: dict) -> None:
    """Log a throttled non-trading defer while staged inputs settle."""
    missing = tuple(details.get("missing", ()))
    required = tuple(details.get("required", ()))
    context = str(details.get("context") or "planning")
    invalid = details.get("invalid") or {}
    routine_candle_target = bot._is_routine_completed_candle_target_defer(details)
    if routine_candle_target:
        bot._record_routine_completed_candle_defer(details)
    log_key = (context, missing, required, tuple(sorted(invalid)))
    now_ms = _utc_ms()
    last_log_ms = int(getattr(bot, "_staged_execution_defer_last_log_ms", 0) or 0)
    throttle_ms = 60_000 if routine_candle_target else 15_000
    if (
        log_key == getattr(bot, "_staged_execution_defer_last_log_key", None)
        and now_ms - last_log_ms < throttle_ms
    ):
        return
    bot._staged_execution_defer_last_log_key = log_key
    bot._staged_execution_defer_last_log_ms = now_ms
    logging.log(
        logging.DEBUG if routine_candle_target else logging.INFO,
        "[state] %s",
        bot._format_staged_execution_defer_message(details),
    )
    if invalid:
        logging.debug("[state] staged execution deferred details | invalid=%s", invalid)


def is_routine_completed_candle_target_defer(bot, details: dict) -> bool:
    """Return true for self-recovering minute-boundary candle target changes."""
    del bot
    missing = set(details.get("missing") or ())
    if missing != {"completed_candles"}:
        return False
    invalid = details.get("invalid") or {}
    items = invalid.get("completed_candles") if isinstance(invalid, dict) else None
    if not isinstance(items, list) or not items:
        return False
    first = items[0]
    return (
        isinstance(first, dict)
        and first.get("reason") == "signature_mismatch"
        and first.get("mismatch_type") == "completed_candle_target_changed"
    )


def record_routine_completed_candle_defer(bot, details: dict) -> None:
    """Aggregate routine completed-candle target defers into periodic INFO summaries."""
    try:
        now_ms = _utc_ms()
        state = getattr(bot, "_routine_completed_candle_defer_summary", None)
        if not isinstance(state, dict):
            state = {
                "window_start_ms": now_ms,
                "last_log_ms": 0,
                "count": 0,
                "symbols": set(),
            }
        invalid = details.get("invalid") or {}
        items = invalid.get("completed_candles") if isinstance(invalid, dict) else []
        symbols: set[str] = set()
        if isinstance(items, list) and items and isinstance(items[0], dict):
            first = items[0]
            for key in ("changed_symbols", "missing_symbols", "extra_symbols"):
                for symbol in first.get(key) or []:
                    if symbol:
                        symbols.add(str(symbol))
        state["count"] = int(state.get("count", 0) or 0) + 1
        state_symbols = state.get("symbols")
        if not isinstance(state_symbols, set):
            state_symbols = set(state_symbols or [])
        state_symbols.update(symbols)
        state["symbols"] = state_symbols
        state["last_seen_ms"] = now_ms
        window_start_raw = state.get("window_start_ms", now_ms)
        window_start_ms = (
            int(window_start_raw) if window_start_raw is not None else now_ms
        )
        last_log_ms = int(state.get("last_log_ms", 0) or 0)
        should_log = (
            int(state["count"]) >= 20 and now_ms - last_log_ms >= 30 * 60_000
        ) or now_ms - window_start_ms >= 30 * 60_000
        bot._routine_completed_candle_defer_summary = state
        if not should_log:
            return
        window_s = max(1, int((now_ms - window_start_ms) / 1000))
        logging.info(
            "[state] staged planning deferred summary | reason=completed_candle_target_changed count=%d window=%ds symbols=%s will_retry=automatic action=refresh_candles_then_retry",
            int(state.get("count", 0) or 0),
            window_s,
            bot._log_symbols(tuple(sorted(state_symbols)), limit=8),
        )
        bot._routine_completed_candle_defer_summary = {
            "window_start_ms": now_ms,
            "last_log_ms": now_ms,
            "count": 0,
            "symbols": set(),
        }
    except Exception as exc:
        logging.debug(
            "[state] staged defer summary log failed | error_type=%s error=%s",
            type(exc).__name__,
            exc,
        )


async def defer_staged_execution_cycle(bot, details: dict, loop_start_ms: int) -> None:
    """Skip trading for this loop while staged planner inputs settle."""
    bot._log_staged_execution_defer(details)
    bot._last_loop_duration_ms = _utc_ms() - loop_start_ms
    bot._maybe_log_health_summary()
    bot._maybe_log_unstuck_status()
    bot._set_log_silence_watchdog_context(phase="runtime", stage="flush_snapshot")
    await bot._monitor_flush_snapshot()
    bot._set_log_silence_watchdog_context(
        phase="runtime", stage="staged_precondition_delay"
    )
    await asyncio.sleep(bot._authoritative_confirmation_retry_delay_seconds(details=details))


def build_staged_planning_snapshot(
    bot, symbols: Iterable[str], market_snapshots: dict[str, MarketSnapshot]
) -> PlanningSnapshot:
    """Capture and validate the exact staged data set handed to Rust."""
    ordered_symbols = tuple(
        sorted(dict.fromkeys(str(symbol) for symbol in symbols if symbol))
    )
    ok, details = bot._staged_planner_precondition_state(
        include_market_snapshot=True, symbols=ordered_symbols
    )
    if not ok:
        raise RuntimeError(f"planning snapshot invalid before capture: {details}")
    ledger = bot._ensure_freshness_ledger()
    required = bot._staged_planner_required_surfaces(include_market_snapshot=True)
    min_epochs = bot._staged_planner_surface_min_epochs(required)
    snapshot = PlanningSnapshot.capture(
        ts_ms=_utc_ms(),
        exchange=str(getattr(bot, "exchange", "")),
        user=str(bot.config_get(["live", "user"]) or ""),
        ledger=ledger,
        required_surfaces=required,
        min_epochs=min_epochs,
        symbols=ordered_symbols,
        market_snapshots=market_snapshots,
        market_snapshot_max_age_ms=bot._live_market_snapshot_max_age_ms(),
    )
    snapshot.raise_if_invalid(now_ms=_utc_ms(), context="rust order calculation")
    return snapshot


def build_protective_planning_snapshot(
    bot, symbols: Iterable[str], market_snapshots: dict[str, MarketSnapshot]
) -> PlanningSnapshot:
    """Capture the reduced live data contract used for protective panic execution."""
    ordered_symbols = tuple(
        sorted(dict.fromkeys(str(symbol) for symbol in symbols if symbol))
    )
    required = frozenset({"balance", "positions", "open_orders", "market_snapshot"})
    ledger = bot._ensure_freshness_ledger()
    current_epoch = int(getattr(bot, "_authoritative_refresh_epoch", 0) or 0)
    required_epoch = max(1, current_epoch)
    min_epochs = {surface: required_epoch for surface in required}
    missing = sorted(
        surface
        for surface in required
        if ledger.surface_epoch(surface) < int(min_epochs.get(surface, 0) or 0)
    )
    invalid: dict[str, list] = {}
    if "market_snapshot" not in missing and ordered_symbols:
        market_invalid = bot._market_snapshot_signature_invalid(ordered_symbols)
        if market_invalid:
            missing.append("market_snapshot")
            invalid["market_snapshot"] = market_invalid
    if missing:
        raise RuntimeError(
            "protective planning snapshot invalid before capture: "
            f"missing current-epoch surfaces={sorted(set(missing))} "
            f"epoch={current_epoch} "
            f"required={sorted(required)}"
            f" invalid={invalid}"
        )
    snapshot = PlanningSnapshot.capture(
        ts_ms=_utc_ms(),
        exchange=str(getattr(bot, "exchange", "")),
        user=str(bot.config_get(["live", "user"]) or ""),
        ledger=ledger,
        required_surfaces=required,
        min_epochs=min_epochs,
        symbols=ordered_symbols,
        market_snapshots=market_snapshots,
        market_snapshot_max_age_ms=bot._live_market_snapshot_max_age_ms(),
    )
    snapshot.raise_if_invalid(
        now_ms=_utc_ms(), context="protective panic order calculation"
    )
    return snapshot


def current_planning_snapshot_invalid_for_creations(
    bot, symbols: Iterable[str]
) -> list[dict]:
    """Return reasons the current staged planning snapshot is unsafe for creations."""
    snapshot = getattr(bot, "_current_planning_snapshot", None)
    ordered_symbols = tuple(
        sorted(dict.fromkeys(str(symbol) for symbol in symbols if symbol))
    )
    if snapshot is None:
        return [
            {
                "surface": "planning_snapshot",
                "reason": "missing_current_snapshot",
                "symbols": list(ordered_symbols),
            }
        ]
    invalid = list(snapshot.invalid_details(now_ms=_utc_ms()))
    snapshot_symbols = set(snapshot.symbols)
    missing_order_symbols = [
        symbol for symbol in ordered_symbols if symbol not in snapshot_symbols
    ]
    if missing_order_symbols:
        invalid.append(
            {
                "surface": "planning_snapshot",
                "reason": "creation_symbols_not_in_snapshot",
                "symbols": missing_order_symbols,
            }
        )
    return invalid
