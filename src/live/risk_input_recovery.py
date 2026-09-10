"""Live readiness for valid balance observations that cannot yet support risk math.

This module schedules retries; it never supplies substitute balances or strategy intent.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from time import monotonic

import numpy as np

from live.event_bus import EventTypes, ReasonCodes, LiveEvent, emit_event, format_console_event


class RiskInputUnavailable(RuntimeError):
    """A numeric balance observation cannot support live risk evaluation."""

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
    warning_at: float = 0.0


def _emit(bot, *, reason, status, details, level):
    message = " ".join(f"{key}={value}" for key, value in details.items())
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
        data=details,
    )
    emit_event(bot, event)
    pipeline = getattr(bot, "_live_event_pipeline", None)
    if not (
        getattr(bot, "live_event_console_enabled", False)
        and callable(getattr(pipeline, "emit", None))
        and getattr(pipeline, "console_sink", None) is not None
    ):
        logging.log(
            logging.WARNING if level == "warning" else logging.INFO,
            format_console_event(event),
        )


def defer(bot, exc):
    now = monotonic()
    state = getattr(bot, "_risk_input_recovery", None)
    changed = state is None or state.reason != exc.reason
    if changed:
        state = RecoveryState(reason=exc.reason)
        bot._risk_input_recovery = state
    state.attempts += 1
    cap = 300.0 if exc.reason == ReasonCodes.HSL_HISTORY_BALANCE_UNAVAILABLE else 60.0
    delay = min(cap, 5.0 * 2 ** min(state.attempts - 1, 6))
    state.retry_at = now + delay
    if changed or now >= state.warning_at:
        state.warning_at = now + 300.0
        _emit(bot, reason=exc.reason, status="deferred", level="warning", details={
            **exc.details,
            "retry_count": state.attempts,
            "retry_delay_seconds": delay,
            "action": "block_ordinary_trading_and_retry",
        })


async def ensure_ready(bot, *, startup=False):
    """Run only on a fresh authoritative account/history cohort."""
    state = getattr(bot, "_risk_input_recovery", None)
    try:
        validate_current_balances(bot)
    except RiskInputUnavailable as exc:
        if state is None or state.reason != exc.reason or monotonic() >= state.retry_at:
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
    if state is not None:
        _emit(bot, reason=state.reason, status="succeeded", level="info", details={
            "retry_count": state.attempts,
            "action": "resume_readiness_checks",
        })
        bot._risk_input_recovery = None
    return True


async def protect_and_wait(bot, *, cycle_id=None, loop_timings_ms=None):
    # The existing panic planner also needs positive current balances. Do not
    # feed it stale/fabricated denominators when the account itself is invalid.
    try:
        validate_current_balances(bot)
    except RiskInputUnavailable:
        current_ready = False
    else:
        current_ready = True
    if current_ready and bot._equity_hard_stop_enabled():
        try:
            await bot._run_halted_hsl_protection_if_active()
            await bot._run_latched_hsl_supervisor_if_active(
                cycle_id=cycle_id, loop_timings_ms=loop_timings_ms or {},
            )
        except RiskInputUnavailable as exc:
            # A protective cooldown restart may itself require replay.
            state = getattr(bot, "_risk_input_recovery", None)
            if state is None or state.reason != exc.reason:
                defer(bot, exc)
    await bot._monitor_flush_snapshot()
    await bot._sleep_unless_shutdown(5.0, stage="risk_inputs_waiting")


async def wait_for_startup(bot):
    # Maintainers do not exist yet; this owner refreshes account/fill inputs.
    authoritative_ready = await bot.refresh_authoritative_state()
    while not bot.stop_signal_received:
        if authoritative_ready and await ensure_ready(bot, startup=True):
            return
        await protect_and_wait(bot)
        if bot.stop_signal_received:
            return
        authoritative_ready = await bot.refresh_authoritative_state()
