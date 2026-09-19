"""Scoped HSL signal health and the durable clock for its emergency grace period.

No drawdown/EMA is synthesized here. Rust owns emergency threshold decisions;
this journal records only availability time and a committed protective exit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import math
import os
from pathlib import Path

from config.access import require_live_value
from live.market_snapshot import MarketSnapshotUnavailable
from live.diagnostic_safety import bounded_exception_type
from ccxt.base.errors import NetworkError


@dataclass(frozen=True, order=True)
class Scope:
    mode: str
    pside: str
    symbol: str = ""


@dataclass
class Health:
    status: str = "usable"
    reason: str = ""
    unavailable_since_ms: int | None = None
    last_evaluated_ms: int | None = None
    emergency_active: bool = False
    exit_committed: bool = False
    exit_confirmed_flat: bool = False
    exit_started_ms: int | None = None
    exit_flat_ms: int | None = None
    execution_blocked: str = ""
    budget: float | None = None
    drawdown_raw: float | None = None
    realized_loss: float | None = None


class ProtectionHealth:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.scopes: dict[Scope, Health] = {}
        self.evaluated: set[Scope] = set()
        self.journal_invalid = False
        self.durable = True
        if path is not None:
            self._load()

    def _load(self):
        try:
            payload = json.loads(self.path.read_text())
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            self._invalid_journal()
            return
        try:
            if type(payload["version"]) is not int or payload["version"] != 1 or not isinstance(payload["scopes"], list):
                raise ValueError("invalid protection journal")
            unverified_scopes = payload.get("unverified_scopes", False)
            if type(unverified_scopes) is not bool:
                raise ValueError("invalid journal recovery state")
            loaded = {}
            for row in payload["scopes"]:
                scope = Scope(**row["scope"])
                health = Health(**row["health"])
                if (scope.mode not in {"coin", "pside", "unified"}
                        or scope.pside not in {"long", "short"}
                        or not isinstance(scope.symbol, str)
                        or (scope.mode == "coin") != bool(scope.symbol)
                        or scope in loaded
                        or health.status not in {"usable", "degraded", "unavailable"}
                        or any(type(getattr(health, field)) is not bool for field in
                               ("exit_committed", "exit_confirmed_flat", "emergency_active"))
                        or not isinstance(health.execution_blocked, str)
                        or (health.exit_committed and health.exit_confirmed_flat)
                        or not isinstance(health.reason, str)):
                    raise ValueError("invalid protection scope")
                for stamp in (health.unavailable_since_ms, health.last_evaluated_ms,
                              health.exit_started_ms, health.exit_flat_ms):
                    if stamp is not None and (type(stamp) is not int or stamp < 0):
                        raise ValueError("invalid protection timestamp")
                if ((health.exit_committed or health.exit_confirmed_flat)
                        and health.exit_started_ms is None
                        or health.exit_committed and health.exit_flat_ms is not None
                        or health.exit_confirmed_flat and health.exit_flat_ms is None
                        or health.exit_started_ms is not None and not health.exit_committed
                           and health.exit_flat_ms is None
                        or health.exit_flat_ms is not None and (
                            health.exit_started_ms is None or health.exit_flat_ms < health.exit_started_ms)):
                    raise ValueError("incomplete emergency provenance")
                for value in (health.budget, health.drawdown_raw, health.realized_loss):
                    if value is not None and (isinstance(value, bool)
                            or not isinstance(value, (int, float)) or not math.isfinite(value)):
                        raise ValueError("invalid protection metric")
                loaded[scope] = health
            self.scopes = loaded
            self.journal_invalid = unverified_scopes
        except (KeyError, TypeError, ValueError):
            self._invalid_journal()

    def _invalid_journal(self):
        self.journal_invalid = True
        self.durable = False
        logging.error("[risk] HSL protection journal unavailable; an unavailable signal will not receive renewed grace")

    def save(self):
        if self.path is None:
            return
        # Healthy scopes need no persisted control record. A confirmed evaluation
        # clears an outage; a submitted order does not clear an exit commitment.
        rows = [{"scope": asdict(scope), "health": asdict(health)}
                for scope, health in sorted(self.scopes.items())
                if health.unavailable_since_ms is not None or health.exit_committed
                or health.exit_confirmed_flat or health.exit_started_ms is not None
                or self.journal_invalid and health.last_evaluated_ms is not None]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            with temporary.open("w") as handle:
                json.dump({"version": 1, "unverified_scopes": self.journal_invalid, "scopes": rows}, handle, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            self.durable = True
        except OSError:
            self.durable = False
            logging.error("[risk] HSL protection journal write failed; protection continues but restart continuity is not durable")

    def unavailable(self, scope: Scope, *, now_ms: int, reason: str, grace_ms: int):
        health = self.scopes.setdefault(scope, Health())
        changed = health.unavailable_since_ms is None or health.reason != reason
        if health.unavailable_since_ms is None:
            health.unavailable_since_ms = (
                max(0, now_ms - grace_ms)
                if self.journal_invalid and health.last_evaluated_ms is None else now_ms
            )
        elif health.unavailable_since_ms > now_ms:
            # Clock rollback must not grant a new window on every restart.
            health.unavailable_since_ms = max(0, now_ms - grace_ms)
            changed = True
        health.status = "unavailable"
        health.reason = reason
        if changed:
            self.save()
        return health

    def evaluated_successfully(self, scope: Scope, *, now_ms: int, degraded_reason: str = ""):
        health = self.scopes.setdefault(scope, Health())
        changed = (health.unavailable_since_ms is not None
                   or self.journal_invalid and health.last_evaluated_ms is None)
        status = "degraded" if degraded_reason else "usable"
        if (health.status, health.reason) != (status, degraded_reason):
            log = logging.warning if degraded_reason else logging.info
            log("[risk] HSL evaluation quality | mode=%s pside=%s symbol=%s status=%s reason=%s",
                scope.mode, scope.pside, scope.symbol or "all", status, degraded_reason or "normal_evaluation")
        health.status = status
        health.reason = degraded_reason
        health.last_evaluated_ms = now_ms
        health.unavailable_since_ms = None
        health.emergency_active = False
        if not health.exit_committed:
            health.execution_blocked = ""
        self.evaluated.add(scope)
        if changed:
            self.save()

    def confirm_flat(self, scope: Scope, *, now_ms: int):
        health = self.scopes[scope]
        if health.exit_committed:
            health.exit_committed = False
            health.exit_confirmed_flat = True
            health.exit_flat_ms = now_ms
            health.execution_blocked = ""
            self.save()

    def pending_exits(self):
        return {scope for scope, health in self.scopes.items() if health.exit_committed}

    def release_flat_hold(self, scope: Scope):
        """Only completed canonical replay may take over cooldown/reopening policy."""
        self.scopes[scope].exit_confirmed_flat = False
        self.save()

    def payload(self, now_ms: int, grace_ms: int):
        return [{**asdict(scope), **asdict(health),
                 "unavailable_seconds": (None if health.unavailable_since_ms is None else
                     max(0, now_ms - health.unavailable_since_ms) / 1000.0),
                 "grace_remaining_seconds": (None if health.unavailable_since_ms is None else
                     max(0, grace_ms - max(0, now_ms - health.unavailable_since_ms)) / 1000.0),
                 "journal_durable": self.durable}
                for scope, health in sorted(self.scopes.items())]


def grace_ms(bot):
    value = require_live_value(bot.config, "hsl_unavailable_grace_seconds")
    if isinstance(value, bool) or not math.isfinite(value) or not 0.0 <= value < 2**64 / 1000:
        raise ValueError("hsl_unavailable_grace_seconds must be finite and fit unsigned milliseconds")
    return int(value * 1000.0)


def manager(bot):
    result = getattr(bot, "_hsl_protection_health", None)
    if result is None:
        path = getattr(bot, "_hsl_protection_journal_path", None)
        if path is None and getattr(bot, "exchange", None) and getattr(bot, "user", None):
            path = Path("caches/equity_hard_stop") / bot.exchange / f"{bot.user}_protection.json"
        result = ProtectionHealth(Path(path) if path is not None else None)
        bot._hsl_protection_health = result
    return result


def reconcile_config(bot):
    """A deliberate change of HSL scope/enablement retires its old recovery policy."""
    health = manager(bot)
    if not health.scopes:
        return
    mode = bot._equity_hard_stop_signal_mode()
    obsolete = [scope for scope in health.scopes
                if scope.mode != mode or not bot._equity_hard_stop_enabled(
                    scope.pside, **({"symbol": scope.symbol} if scope.symbol else {}))]
    for scope in obsolete:
        logging.warning("[risk] HSL protection scope retired after configuration change | mode=%s pside=%s symbol=%s",
                        scope.mode, scope.pside, scope.symbol or "all")
        del health.scopes[scope]
    if obsolete:
        health.save()


def scope_for(bot, pside, symbol=None):
    mode = bot._equity_hard_stop_signal_mode()
    return Scope(mode, pside, str(symbol) if mode == "coin" and symbol is not None else "")


def record_evaluation(bot, pside, symbol=None, *, degraded_reason=""):
    health = getattr(bot, "_hsl_protection_health", None)
    if health is not None:
        health.evaluated_successfully(scope_for(bot, pside, symbol),
            now_ms=int(bot.get_exchange_time()), degraded_reason=degraded_reason)


def affected_scopes(bot, targets, details):
    """Attribute known symbol/side failures narrowly; account failures affect all targets."""
    mode = bot._equity_hard_stop_signal_mode()
    side = details.get("pside")
    symbol = details.get("symbol")
    if mode == "coin" and side in {"long", "short"} and symbol:
        return {Scope(mode, side, str(symbol))}
    if mode == "coin":
        return {Scope(mode, pside, symbol) for symbol, psides in targets.items()
                for pside in psides if side is None or pside == side}
    target_sides = {pside for psides in targets.values() for pside in psides}
    return {Scope(mode, pside) for pside in ("long", "short")
            if bot._equity_hard_stop_enabled(pside)
            and pside in target_sides
            and (mode == "unified" or side is None or pside == side)}


def _scope_matches(scope, pside, symbol):
    return ((scope.mode == "unified" or scope.pside == pside)
            and (not scope.symbol or scope.symbol == symbol))


def targets_for_scopes(bot, scopes, candidates):
    return {symbol: selected for symbol, psides in candidates.items()
            if (selected := {pside for pside in psides
                if any(_scope_matches(scope, pside, symbol)
                       for scope in scopes)})}


def has_exposure(bot, scope):
    sides = ("long", "short") if scope.mode == "unified" else (scope.pside,)
    return any(float(position.get(side, {}).get("size", 0.0)) != 0.0
               for symbol, position in bot.positions.items()
               if not scope.symbol or symbol == scope.symbol
               for side in sides)


def has_orders(bot, scope):
    return any(scope.mode == "unified" or order.get("position_side") == scope.pside
               for symbol, orders in bot.open_orders.items()
               if not scope.symbol or symbol == scope.symbol
               for order in orders)


async def evaluate_emergency(bot, candidates):
    """Evaluate unavailable scopes with current account and quote evidence only.

    A historical fallback never clears the outage clock. Successful normal (or
    explicitly bounded degraded) evaluation is the sole reset authority.
    """
    import passivbot_rust as pbr
    health_manager = manager(bot)
    now = int(bot.get_exchange_time())
    delay = grace_ms(bot)
    active = affected_scopes(bot, candidates, {})
    for scope, health in list(health_manager.scopes.items()):
        if scope not in active or health.unavailable_since_ms is None or health.exit_committed:
            continue
        elapsed = max(0, now - health.unavailable_since_ms)
        if elapsed < delay:
            continue
        try:
            if scope.mode == "unified":
                upnl = float(await bot._calc_upnl_sum_strict())
            else:
                upnl = float(await bot._calc_upnl_sum_strict(scope.pside, scope.symbol or None))
        except (MarketSnapshotUnavailable, NetworkError, TimeoutError) as exc:
            health.execution_blocked = bounded_exception_type(exc)
            logging.warning("[risk] HSL emergency evaluation waiting for current quotes | pside=%s symbol=%s error_type=%s",
                            scope.pside, scope.symbol or "all", health.execution_blocked)
            continue
        health.execution_blocked = ""
        divisor = (int(round(float(bot.bot_value(scope.pside, "n_positions"))))
                   if scope.mode == "coin" else 1)
        cfg = bot._equity_hard_stop_config(scope.pside, scope.symbol or None)
        # Nonzero exchange exposure itself proves executions took place. A valid
        # new flat account does not satisfy this severe missing-history condition.
        fills = getattr(bot, "_pnls_manager", None)
        missing_history = has_exposure(bot, scope) and (fills is None or not fills.get_events())
        from passivbot_hsl import _equity_hard_stop_emergency_realized_loss
        health.realized_loss = (_equity_hard_stop_emergency_realized_loss(bot, scope.pside, scope.symbol, now)
                                if scope.mode == "coin" else None)
        result = pbr.hsl_emergency_signal(
            True, float(bot.get_raw_balance()), divisor, upnl,
            float(cfg["red_threshold"]), elapsed, delay, missing_history, health.realized_loss,
        )
        if (not isinstance(result, tuple) or len(result) != 4
                or type(result[2]) is not bool or type(result[3]) is not bool
                or not all(type(value) in (int, float) and math.isfinite(value) for value in result[:2])
                or result[0] <= 0.0 or result[1] < 0.0
                or result[2] != (elapsed >= delay) or (result[3] and not result[2])):
            from passivbot_exceptions import FatalBotException
            raise FatalBotException("malformed Rust emergency HSL result")
        health.budget, health.drawdown_raw, health.emergency_active, trigger = result
        if trigger:
            health.exit_committed = True
            health.exit_confirmed_flat = False
            health.exit_started_ms = now
            health.exit_flat_ms = None
            health_manager.save()
            logging.critical(
                "[risk] HSL emergency exit committed | mode=%s pside=%s symbol=%s "
                "unavailable_seconds=%.3f drawdown_raw=%.6f threshold=%.6f missing_history=%s",
                scope.mode, scope.pside, scope.symbol or "all", elapsed / 1000.0,
                health.drawdown_raw, float(cfg["red_threshold"]), missing_history,
            )


def holds_after_emergency_exit(bot, pside, symbol):
    health = getattr(bot, "_hsl_protection_health", None)
    if health is None:
        return False
    return any(_scope_matches(scope, pside, symbol)
               and (state.exit_committed or state.exit_confirmed_flat)
               for scope, state in health.scopes.items())


def emergency_stop_applies(bot, pside, symbol, fill_timestamp_ms):
    """Correlate an actual panic flatten with a recorded emergency close window.

    Replay still owns fill ordering, the actual flatten timestamp and all normal
    cooldown/restart calculations. This fact explains why the close was issued
    even when reconstructed EMA did not cross RED.
    """
    health = getattr(bot, "_hsl_protection_health", None)
    if health is None:
        return False
    return any(_scope_matches(scope, pside, symbol) and state.exit_started_ms is not None
               and state.exit_started_ms <= fill_timestamp_ms
               and (state.exit_flat_ms is None or fill_timestamp_ms <= state.exit_flat_ms)
               for scope, state in health.scopes.items())


def allows_held_entries(bot, pside, symbol):
    health = getattr(bot, "_hsl_protection_health", None)
    if health is None:
        return False
    state = health.scopes.get(scope_for(bot, pside, symbol))
    return (state is not None and not holds_after_emergency_exit(bot, pside, symbol)
            and float(bot.positions.get(symbol, {}).get(pside, {}).get("size", 0.0)) != 0.0)
