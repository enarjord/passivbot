"""Live readiness for authoritative inputs that cannot yet support risk evaluation.

This module schedules retries; it never supplies substitute inputs or strategy intent.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from time import monotonic

import numpy as np

from config.access import require_live_value
from passivbot_exceptions import FatalBotException, RestartBotException
from ccxt.base.errors import NetworkError, OrderNotFound
from live.diagnostic_safety import bounded_traceback_detail, bounded_exception_type
from live.state_refresh import AuthoritativeSurfaceUnavailable
from live.event_bus import EventTypes, ReasonCodes, LiveEvent, emit_event, format_console_event


class RiskInputUnavailable(RuntimeError):
    """An authoritative input cannot support live risk evaluation."""

    def __init__(self, reason: str, **details):
        self.reason = reason
        self.details = details
        super().__init__(reason)


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def validate_current_balances(bot):
    validate_balances(bot.get_raw_balance(), bot.get_hysteresis_snapped_balance())


def validate_balances(raw, sizing):
    if not all(math.isfinite(value) and value > 0.0 for value in (raw, sizing)):
        raise RiskInputUnavailable(
            ReasonCodes.CURRENT_BALANCE_UNAVAILABLE,
            balance_raw=_number(raw),
            balance=_number(sizing),
        )


def validate_history_balances(timestamps, balances, *, current_balance):
    """Validate numeric values after the producer's shape/type checks."""
    values = np.asarray(balances, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.int64)
    if values.ndim != 1 or times.ndim != 1 or values.shape != times.shape:
        raise ValueError("HSL replay balance and timestamp arrays must be matching vectors")
    invalid = np.flatnonzero(~np.isfinite(values) | (values <= 0.0))
    if invalid.size:
        first = int(invalid[0])
        raise RiskInputUnavailable(
            ReasonCodes.HSL_HISTORY_BALANCE_UNAVAILABLE,
            balance_raw=_number(current_balance),
            first_invalid_timestamp_ms=int(times[first]),
            first_invalid_balance=_number(values[first]),
            invalid_rows=int(invalid.size),
            replay_start_ms=int(times[0]),
            replay_end_ms=int(times[-1]),
        )


def validate_history_rows(rows, *, current_balance):
    validate_history_balances(
        [row["timestamp"] for row in rows],
        [row["balance"] for row in rows],
        current_balance=current_balance,
    )


@dataclass
class RecoveryState:
    reason: str = ""
    attempts: int = 0
    retry_at: float = 0.0
    max_attempts: int = 10
    blocked_since: float = 0.0
    protective_exit_pending: bool = False


def _emit(bot, *, reason, status, details, level, exc=None):
    message = " ".join(f"{key}={value}" for key, value in details.items())
    if status != "succeeded":
        requirement = (
            "Required HSL episode evidence must prove fill order and current position."
            if reason == ReasonCodes.HSL_EPISODE_EVIDENCE_UNAVAILABLE
            else "Required reconstructed HSL balances must be finite and positive."
            if reason == ReasonCodes.HSL_HISTORY_BALANCE_UNAVAILABLE
            else "Current raw and sizing balances must be finite and positive."
        )
        message = f"{requirement} {message}"
    data = dict(details)
    if exc is not None:
        data["traceback"] = bounded_traceback_detail(exc)
    event = LiveEvent(
        EventTypes.RISK_INPUT_STATUS,
        level=level,
        source="live",
        component="risk_input_recovery",
        tags=("risk", "readiness"),
        exchange=getattr(bot, "exchange", None),
        user=getattr(bot, "user", None),
        bot_id=getattr(bot, "bot_id", None),
        status=status,
        reason_code=reason,
        message=message,
        data=data,
    )
    emitted = emit_event(bot, event)
    pipeline = getattr(bot, "_live_event_pipeline", None)
    if not (
        emitted is not None
        and getattr(bot, "live_event_console_enabled", False)
        and callable(getattr(pipeline, "emit", None))
        and getattr(pipeline, "console_sink", None) is not None
    ):
        logging.log(
            getattr(logging, level.upper()),
            format_console_event(event),
        )

    if exc is not None:
        trace = data["traceback"]
        lines = ["Risk input traceback (bounded frames; no locals or raw exception text):"]
        for item in trace["exceptions"]:
            lines.append(f"  {item['relation']}: {item['error_type']}")
            for frame in item["frames"]:
                lines.append(f"    {frame['file']}:{frame['line']} in {frame['function']}")
        if trace["truncated"]:
            lines.append("  ... traceback truncated")
        logging.log(getattr(logging, level.upper()), "%s", "\n".join(lines))


def defer(bot, exc):
    """Count failed attempts, never readiness polls or changing failure reasons."""
    now = monotonic()
    state = getattr(bot, "_risk_input_recovery", None)
    if state is None:
        state = RecoveryState(
            max_attempts=require_live_value(bot.config, "risk_input_max_attempts"),
            blocked_since=now,
        )
        bot._risk_input_recovery = state
    elif now < state.retry_at:
        return
    state.reason = exc.reason
    state.attempts += 1
    exhausted = state.attempts >= state.max_attempts
    hsl_enabled = bot._equity_hard_stop_enabled()
    # History repair must never terminate the owner of live HSL protection.
    # Commit to reducing exposed HSL scopes before accepting recovery, even if
    # the next historical read succeeds. No approximate risk sample is invented.
    state.protective_exit_pending |= hsl_enabled and bool(_unready_hsl_targets(bot))
    cap = 300.0 if exc.reason == ReasonCodes.HSL_HISTORY_BALANCE_UNAVAILABLE else 60.0
    delay = 0.0 if exhausted and not hsl_enabled else min(cap, 5.0 * 2 ** min(state.attempts - 1, 6))
    state.retry_at = now + delay
    _emit(bot, reason=exc.reason, status="failed" if exhausted else "deferred",
          level="error" if exhausted else "warning", details={
        **exc.details,
        "retry_count": state.attempts,
        "max_attempts": state.max_attempts,
        "blocked_seconds": max(0.0, now - state.blocked_since),
        "retry_delay_seconds": delay,
        "action": ("protective_exit_and_retry" if hsl_enabled else
                   "stop_without_restart" if exhausted else "block_ordinary_trading_and_retry"),
    }, exc=exc if state.attempts in (1, state.max_attempts) else None)
    if exhausted and not hsl_enabled:
        raise FatalBotException(
            f"Risk input recovery exhausted after {state.attempts}/{state.max_attempts} "
            f"failed attempts: {exc.reason}; stopping without automatic restart"
        ) from exc


def defer_episode_evidence(bot, exc):
    if exc.surface != "hsl_episode_boundaries":
        raise exc
    # Never emit the free-form exception reason: legacy producers may include
    # arbitrary payload text. Structured producers supply bounded cause/scope.
    details = getattr(exc, "details", {"cause": "required_boundary_unavailable"})
    unavailable = RiskInputUnavailable(ReasonCodes.HSL_EPISODE_EVIDENCE_UNAVAILABLE, **details)
    unavailable.__cause__ = exc
    defer(bot, unavailable)


def defer_authoritative_hsl(bot):
    """Route historical account-readiness blockers through the same exit policy."""
    reason = getattr(bot, "_last_authoritative_block_reason", None)
    if not bot._equity_hard_stop_enabled() or reason not in {
        "pending_pnl", "degraded_pnl", "fill_history_coverage", "balance_consistency_check",
        "fills_unavailable",
    }:
        return False
    defer(bot, RiskInputUnavailable(ReasonCodes.HSL_EPISODE_EVIDENCE_UNAVAILABLE, cause=reason))
    return True


def mark_ready(bot):
    """Reset only after the owner completes its risk-consuming operation."""
    state = getattr(bot, "_risk_input_recovery", None)
    if state is not None:
        _emit(bot, reason=state.reason, status="succeeded", level="info", details={
            "retry_count": state.attempts,
            "max_attempts": state.max_attempts,
            "action": "resume_readiness_checks",
        })
        bot._risk_input_recovery = None


async def ensure_ready(bot, *, startup=False):
    """Run only on a fresh authoritative account/history cohort."""
    state = getattr(bot, "_risk_input_recovery", None)
    if state is not None and bot._equity_hard_stop_enabled():
        # The fresh cohort may contain new exposure since the previous flat
        # confirmation, including on the same pass that history recovers.
        state.protective_exit_pending |= bool(_unready_hsl_targets(bot))
        if state.protective_exit_pending:
            return False
    try:
        validate_current_balances(bot)
    except RiskInputUnavailable as exc:
        defer(bot, exc)
        return False
    if state is not None and monotonic() < state.retry_at:
        return False
    try:
        if bot._equity_hard_stop_enabled():
            if startup:
                if bot._equity_hard_stop_signal_mode() == "coin":
                    await bot._equity_hard_stop_start_coin_history_replay()
                else:
                    await bot._equity_hard_stop_initialize_from_history()
            else:
                await bot._equity_hard_stop_check()
    except RiskInputUnavailable as exc:
        defer(bot, exc)
        return False
    except AuthoritativeSurfaceUnavailable as exc:
        defer_episode_evidence(bot, exc)
        return False
    return True


def _unready_hsl_targets(bot):
    targets = {}
    positions = getattr(bot, "positions", {})
    open_orders = getattr(bot, "open_orders", {})
    coin_mode = bot._equity_hard_stop_signal_mode() == "coin"
    for symbol in sorted(set(positions) | set(open_orders)):
        for pside in ("long", "short"):
            enabled = (bot._equity_hard_stop_enabled(pside, symbol=symbol) if coin_mode
                       else bot._equity_hard_stop_enabled(pside))
            if not enabled:
                continue
            # Proven halted scopes keep their existing cooldown/manual policy.
            # Their independent protection owner remains responsible for them.
            if coin_mode:
                scope = getattr(bot, "_equity_hard_stop_coin", {}).get(pside, {}).get(symbol, {})
                ready = (getattr(bot, "_equity_hard_stop_coin_initialized", False)
                         or (pside, symbol) in getattr(bot, "_equity_hard_stop_coin_replay_ready_pairs", set()))
            else:
                scope = bot._hsl_state(pside)
                ready = True
            if ready and scope.get("halted", False):
                continue
            position = positions.get(symbol, {}).get(pside)
            has_position = position is not None and float(position["size"]) != 0.0
            has_orders = any(order["position_side"] == pside
                             for order in open_orders.get(symbol, []))
            if has_position or has_orders:
                targets.setdefault(symbol, set()).add(pside)
    return targets


def _report_protective_unavailability(bot, exc):
    if isinstance(exc, AuthoritativeSurfaceUnavailable) and exc.surface == "hsl_episode_boundaries":
        defer_episode_evidence(bot, exc)
    else:
        logging.warning(
            "[risk] HSL protection deferred; retaining exit commitment | error_type=%s",
            bounded_exception_type(exc),
        )


async def protect_unready_hsl(bot):
    """Close HSL-managed exposure with fresh account state, independently of history.

    Rust's existing panic planner owns sizing and execution type. A successful
    submission is not flatness: retain the exit until another authoritative
    snapshot confirms both positions and resting orders are gone.
    """
    state = getattr(bot, "_risk_input_recovery", None)
    if state is None:
        return False
    try:
        if await bot.refresh_protective_authoritative_state():
            validate_current_balances(bot)
            targets = _unready_hsl_targets(bot)
            if not targets:
                state.protective_exit_pending = False
                return False
            state.protective_exit_pending = True
            to_cancel, to_create = await bot.calc_protective_panic_orders_to_cancel_and_create(
                target_psides_by_symbol=targets,
            )
            await bot.execute_order_plan_to_exchange(to_cancel, to_create, configure_creations=False)
    except RiskInputUnavailable as exc:
        defer(bot, exc)
    except (AuthoritativeSurfaceUnavailable, NetworkError, OrderNotFound, OSError, RestartBotException) as exc:
        # A transient reader/connector failure cannot surrender protection to
        # generic full-bot restart handling. Producer/config defects remain strict.
        _report_protective_unavailability(bot, exc)
    await bot._sleep_unless_shutdown(
        float(bot.live_value("execution_delay_seconds")), stage="risk_input_protective_exit",
    )
    return True


async def protect_and_wait(bot, *, cycle_id=None, loop_timings_ms=None):
    unready_protected = False
    if bot._equity_hard_stop_enabled():
        try:
            unready_protected = await protect_unready_hsl(bot)
        except RiskInputUnavailable as exc:
            defer(bot, exc)
        except (AuthoritativeSurfaceUnavailable, NetworkError, OrderNotFound, OSError, RestartBotException) as exc:
            _report_protective_unavailability(bot, exc)
    # The existing panic planner also needs positive current balances. Do not
    # feed it stale/fabricated denominators when the account itself is invalid.
    try:
        validate_current_balances(bot)
    except RiskInputUnavailable:
        current_ready = False
    else:
        current_ready = True
    if current_ready and bot._equity_hard_stop_enabled():
        protected = False
        try:
            protected = await bot._run_halted_hsl_protection_if_active()
        except RiskInputUnavailable as exc:
            defer(bot, exc)
        except (AuthoritativeSurfaceUnavailable, NetworkError, OrderNotFound, OSError, RestartBotException) as exc:
            _report_protective_unavailability(bot, exc)
        try:
            # One wave lets flat RED scopes finalize without a persistent RED
            # supervisor monopolizing recovery of other exposed scopes.
            protected |= await bot._run_latched_hsl_supervisor_if_active(
                cycle_id=cycle_id, loop_timings_ms=loop_timings_ms or {}, single_pass=True,
            )
        except RiskInputUnavailable as exc:
            defer(bot, exc)
        except (AuthoritativeSurfaceUnavailable, NetworkError, OrderNotFound, OSError, RestartBotException) as exc:
            _report_protective_unavailability(bot, exc)
        if protected or unready_protected:
            await bot._monitor_flush_snapshot()
            return
    await bot._monitor_flush_snapshot()
    if not unready_protected:
        await bot._sleep_unless_shutdown(5.0, stage="risk_inputs_waiting")


async def protect_before_history_refresh(bot, *, cycle_id=None, loop_timings_ms=None):
    """Keep the full fill/history cohort behind its deadline while exits continue."""
    state = getattr(bot, "_risk_input_recovery", None)
    if state is None or not bot._equity_hard_stop_enabled():
        return False
    if not state.protective_exit_pending and monotonic() >= state.retry_at:
        return False
    await protect_and_wait(bot, cycle_id=cycle_id, loop_timings_ms=loop_timings_ms)
    return True


async def wait_for_startup(bot):
    # Maintainers do not exist yet; this owner refreshes account/fill inputs.
    while not bot.stop_signal_received:
        if await protect_before_history_refresh(bot):
            continue
        authoritative_ready = await bot.refresh_authoritative_state()
        if bot.stop_signal_received:
            return
        if not authoritative_ready:
            defer_authoritative_hsl(bot)
        if authoritative_ready and await ensure_ready(bot, startup=True):
            mark_ready(bot)
            return
        await protect_and_wait(bot)
