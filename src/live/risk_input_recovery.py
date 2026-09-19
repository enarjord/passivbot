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
from live import hsl_protection
from passivbot_exceptions import FatalBotException, RestartBotException
from ccxt.base.errors import NetworkError, OrderNotFound
from live.diagnostic_safety import bounded_traceback_detail, bounded_exception_type
from live.state_refresh import AuthoritativeSurfaceUnavailable
from live.market_snapshot import MarketSnapshotUnavailable
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


def validate_emergency_balance(bot):
    raw = bot.get_raw_balance()
    if not math.isfinite(raw) or raw <= 0.0:
        raise RiskInputUnavailable(ReasonCodes.CURRENT_BALANCE_UNAVAILABLE, balance_raw=_number(raw))


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
    normal_evaluation_succeeded: bool = False


def _emit(bot, *, reason, status, details, level, exc=None):
    message = " ".join(f"{key}={value}" for key, value in details.items())
    if status != "succeeded":
        requirement = (
            "HSL episode reconstruction is unavailable; scoped protection remains active."
            if reason == ReasonCodes.HSL_EPISODE_EVIDENCE_UNAVAILABLE
            else "Required reconstructed HSL balances must be finite and positive."
            if reason == ReasonCodes.HSL_HISTORY_BALANCE_UNAVAILABLE
            else "HSL signal evaluation is unavailable; scoped protection remains active."
            if reason == ReasonCodes.HSL_SIGNAL_UNAVAILABLE
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
    retry_due = now >= state.retry_at
    hsl_enabled = bot._equity_hard_stop_enabled()
    new_scope = False
    if hsl_enabled:
        health = hsl_protection.manager(bot)
        for scope in hsl_protection.affected_scopes(bot, _unready_hsl_targets(bot), exc.details):
            new_scope |= scope not in health.scopes or health.scopes[scope].unavailable_since_ms is None
            health.unavailable(scope, now_ms=int(bot.get_exchange_time()),
                reason=str(exc.details.get("cause") or exc.reason), grace_ms=hsl_protection.grace_ms(bot))
        state.protective_exit_pending = bool(health.pending_exits())
        state.normal_evaluation_succeeded = False
    if not retry_due:
        if new_scope:
            _emit(bot, reason=exc.reason, status="deferred", level="warning",
                  details={**exc.details, "action": "new_scope_protection_grace"})
        return
    state.reason = exc.reason
    state.attempts += 1
    exhausted = state.attempts >= state.max_attempts
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
        "action": ("protective_exit_and_retry" if hsl_enabled and state.protective_exit_pending else
                   "evaluate_emergency_after_grace_and_retry" if hsl_enabled else
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
    if state is not None and bot._equity_hard_stop_enabled():
        health = hsl_protection.manager(bot)
        if (not state.normal_evaluation_succeeded or health.pending_exits()
                or any(item.unavailable_since_ms is not None or item.exit_confirmed_flat
                       for item in health.scopes.values())):
            return
    if state is not None:
        _emit(bot, reason=state.reason, status="succeeded", level="info", details={
            "retry_count": state.attempts,
            "max_attempts": state.max_attempts,
            "action": "resume_readiness_checks",
        })
        bot._risk_input_recovery = None


async def ensure_ready(bot, *, startup=False):
    """Evaluate HSL separately from the current inputs needed by ordinary strategy.

    The caller supplies a fresh authoritative cohort. A historical HSL-only
    failure therefore need not block martingale adds while the grace/fallback
    owner remains active. Other input consumers keep their own strict gates.
    """
    enabled = bot._equity_hard_stop_enabled()
    health = hsl_protection.manager(bot)
    hsl_protection.reconcile_config(bot)
    state = getattr(bot, "_risk_input_recovery", None)
    if health is not None and health.pending_exits():
        if state is None:
            state = RecoveryState(reason="restored_protective_exit", protective_exit_pending=True,
                                  max_attempts=require_live_value(bot.config, "risk_input_max_attempts"),
                                  blocked_since=monotonic())
            bot._risk_input_recovery = state
        return False
    try:
        validate_current_balances(bot)
    except RiskInputUnavailable as exc:
        defer(bot, exc)
        return False
    if not enabled:
        return state is None or monotonic() >= state.retry_at
    health.evaluated.clear()
    bot._hsl_readiness_excluded_pairs = set()
    failure_seen = False
    retry_due = state is None or monotonic() >= state.retry_at
    coin_initialized = (bot._equity_hard_stop_signal_mode() == "coin"
                        and (getattr(bot, "_equity_hard_stop_coin_initialized", False)
                             or getattr(bot, "_equity_hard_stop_coin_protective_ready", False)))
    pside_initialized = (bot._equity_hard_stop_signal_mode() == "pside"
                         and all(not bot._equity_hard_stop_enabled(side)
                                 or bot._equity_hard_stop_runtime_initialized(side)
                                 for side in ("long", "short")))
    if state is not None:
        for scope in hsl_protection.affected_scopes(bot, _unready_hsl_targets(bot), {}):
            if scope not in health.scopes:
                health.unavailable(scope, now_ms=int(bot.get_exchange_time()),
                    reason="current_scope_evaluation_pending", grace_ms=hsl_protection.grace_ms(bot))
    if not retry_due and (coin_initialized or pside_initialized):
        bot._hsl_readiness_excluded_pairs = {
            (scope.pside, scope.symbol or None) for scope, item in health.scopes.items()
            if item.unavailable_since_ms is not None
        }
    fills = getattr(bot, "_pnls_manager", None)
    missing_history = (fills is None or not fills.get_events()) and any(
        hsl_protection.has_exposure(bot, scope)
        for scope in hsl_protection.affected_scopes(bot, _unready_hsl_targets(bot), {})
    )
    if missing_history:
        failure_seen = True
        defer(bot, RiskInputUnavailable(ReasonCodes.HSL_SIGNAL_UNAVAILABLE,
                                       cause="missing_execution_history"))
    try:
        if not missing_history and (retry_due or coin_initialized or pside_initialized):
            unified_hold_pending = False
            for scope, item in list(health.scopes.items()):
                if not item.exit_confirmed_flat:
                    continue
                # Reopening belongs to the canonical replay/cooldown policy,
                # never to a successful fetch or an isolated GREEN sample.
                from passivbot_hsl import _equity_hard_stop_replay_live_restart
                if await _equity_hard_stop_replay_live_restart(bot, scope.pside, scope.symbol or None):
                    health.evaluated_successfully(scope, now_ms=int(bot.get_exchange_time()))
                    health.release_flat_hold(scope)
                else:
                    failure_seen = True
                    defer(bot, RiskInputUnavailable(ReasonCodes.HSL_SIGNAL_UNAVAILABLE,
                        cause="emergency_stop_replay_pending", pside=scope.pside, symbol=scope.symbol or None))
                    bot._hsl_readiness_excluded_pairs.add((scope.pside, scope.symbol or None))
                    unified_hold_pending |= scope.mode == "unified"
            if startup and retry_due:
                if bot._equity_hard_stop_signal_mode() == "coin":
                    await bot._equity_hard_stop_start_coin_history_replay()
                else:
                    await bot._equity_hard_stop_initialize_from_history()
            # Retry once per attributable scope so an unavailable coin cannot
            # prevent evaluation of independently ready coins later in the loop.
            while not unified_hold_pending:
                try:
                    await bot._equity_hard_stop_check()
                    break
                except (AuthoritativeSurfaceUnavailable, RiskInputUnavailable) as exc:
                    if isinstance(exc, RiskInputUnavailable):
                        defer(bot, exc)
                    else:
                        if exc.surface != "hsl_episode_boundaries":
                            raise
                        defer_episode_evidence(bot, exc)
                    failure_seen = True
                    details = getattr(exc, "details", {})
                    pair = (details.get("pside"), details.get("symbol"))
                    attributable = ((coin_initialized and bool(pair[1])) or pside_initialized)
                    if pside_initialized:
                        pair = (pair[0], None)
                    if (not attributable or pair[0] not in {"long", "short"}
                            or pair in bot._hsl_readiness_excluded_pairs):
                        break
                    bot._hsl_readiness_excluded_pairs.add(pair)
    except RiskInputUnavailable as exc:
        failure_seen = True
        defer(bot, exc)
    except AuthoritativeSurfaceUnavailable as exc:
        failure_seen = True
        defer_episode_evidence(bot, exc)
    except (MarketSnapshotUnavailable, NetworkError, TimeoutError) as exc:
        failure_seen = True
        defer(bot, RiskInputUnavailable(ReasonCodes.HSL_SIGNAL_UNAVAILABLE,
                                       cause=bounded_exception_type(exc)))
    finally:
        bot._hsl_readiness_excluded_pairs = set()
    state = getattr(bot, "_risk_input_recovery", None)
    if state is not None:
        state.normal_evaluation_succeeded = retry_due and not failure_seen
        await hsl_protection.evaluate_emergency(bot, _unready_hsl_targets(bot))
        state.protective_exit_pending = bool(health.pending_exits())
        if state.protective_exit_pending:
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
    """Run emergency evaluation after grace, then retain exits until confirmed flat."""
    state = getattr(bot, "_risk_input_recovery", None)
    if state is None:
        return False
    health = hsl_protection.manager(bot)
    hsl_protection.reconcile_config(bot)
    had_pending = bool(health.pending_exits())
    try:
        pending = health.pending_exits()
        # An existing exit needs no balance. A new emergency decision does.
        if await bot.refresh_protective_authoritative_state(require_balance=not bool(pending)):
            candidates = _unready_hsl_targets(bot)
            for scope in hsl_protection.affected_scopes(bot, candidates, {}):
                if scope not in health.scopes:
                    health.unavailable(scope, now_ms=int(bot.get_exchange_time()),
                        reason="current_scope_evaluation_pending", grace_ms=hsl_protection.grace_ms(bot))
            if not pending:
                validate_emergency_balance(bot)
                await hsl_protection.evaluate_emergency(bot, candidates)
                pending = health.pending_exits()
            for scope in pending:
                if not hsl_protection.has_exposure(bot, scope) and not hsl_protection.has_orders(bot, scope):
                    health.confirm_flat(scope, now_ms=int(bot.get_exchange_time()))
            pending = health.pending_exits()
            state.protective_exit_pending = bool(pending)
            targets = hsl_protection.targets_for_scopes(bot, pending, candidates)
            if targets:
                to_cancel, to_create = await bot.calc_protective_panic_orders_to_cancel_and_create(
                    target_psides_by_symbol=targets,
                )
                await bot.execute_order_plan_to_exchange(to_cancel, to_create, configure_creations=False)
                for scope in pending:
                    health.scopes[scope].execution_blocked = ""
    except RiskInputUnavailable as exc:
        defer(bot, exc)
    except (AuthoritativeSurfaceUnavailable, NetworkError, OrderNotFound, OSError, RestartBotException) as exc:
        for scope in health.pending_exits():
            health.scopes[scope].execution_blocked = bounded_exception_type(exc)
        _report_protective_unavailability(bot, exc)
    # This pass is independent of the close wave above: a symbol-specific
    # cancellation, quote, or submission outage cannot starve another scope.
    now = int(bot.get_exchange_time())
    if had_pending and any(item.unavailable_since_ms is not None
                           and not item.exit_committed and not item.exit_confirmed_flat
                           and now - item.unavailable_since_ms >= hsl_protection.grace_ms(bot)
                           for item in health.scopes.values()):
        try:
            if await bot.refresh_protective_authoritative_state(require_balance=True):
                validate_emergency_balance(bot)
                await hsl_protection.evaluate_emergency(bot, _unready_hsl_targets(bot))
                state.protective_exit_pending = bool(health.pending_exits())
        except RiskInputUnavailable as exc:
            defer(bot, exc)
        except (AuthoritativeSurfaceUnavailable, NetworkError, OrderNotFound, OSError, RestartBotException) as exc:
            _report_protective_unavailability(bot, exc)
    return state.protective_exit_pending


async def protect_and_wait(bot, *, cycle_id=None, loop_timings_ms=None):
    unready_protected = False
    if bot._equity_hard_stop_enabled():
        try:
            unready_protected = await protect_unready_hsl(bot)
        except RiskInputUnavailable as exc:
            defer(bot, exc)
        except (AuthoritativeSurfaceUnavailable, NetworkError, OrderNotFound, OSError, RestartBotException) as exc:
            _report_protective_unavailability(bot, exc)
    protected = False
    if bot._equity_hard_stop_enabled():
        try:
            protected = await bot._run_halted_hsl_protection_if_active(pace=False)
        except RiskInputUnavailable as exc:
            protected = True
            defer(bot, exc)
        except (AuthoritativeSurfaceUnavailable, NetworkError, OrderNotFound, OSError, RestartBotException) as exc:
            protected = True
            _report_protective_unavailability(bot, exc)
        try:
            # One wave lets flat RED scopes finalize without a persistent RED
            # supervisor monopolizing recovery of other exposed scopes.
            protected |= await bot._run_latched_hsl_supervisor_if_active(
                cycle_id=cycle_id, loop_timings_ms=loop_timings_ms or {}, single_pass=True,
            )
        except RiskInputUnavailable as exc:
            protected = True
            defer(bot, exc)
        except (AuthoritativeSurfaceUnavailable, NetworkError, OrderNotFound, OSError, RestartBotException) as exc:
            protected = True
            _report_protective_unavailability(bot, exc)
    await bot._monitor_flush_snapshot()
    if not bot.stop_signal_received:
        if protected or unready_protected:
            await bot._sleep_unless_shutdown(
                float(bot.live_value("execution_delay_seconds")), stage="risk_input_protective_exit",
            )
        else:
            await bot._sleep_unless_shutdown(5.0, stage="risk_inputs_waiting")


async def protect_before_history_refresh(bot, *, cycle_id=None, loop_timings_ms=None):
    """Keep the full fill/history cohort behind its deadline while exits continue."""
    state = getattr(bot, "_risk_input_recovery", None)
    if state is None or not bot._equity_hard_stop_enabled():
        return False
    if not state.protective_exit_pending:
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
