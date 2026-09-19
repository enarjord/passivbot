"""Scoped HSL signal health and the durable clock for its emergency grace period.

No drawdown/EMA is synthesized here. Rust owns emergency threshold decisions;
this journal records only availability time and a committed protective exit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import asyncio
from time import monotonic
import json
import logging
import math
import os
from pathlib import Path

from config.access import require_live_value
from fill_events_manager import FillEventCacheContractError
from live.market_snapshot import MarketSnapshotUnavailable
from live.diagnostic_safety import bounded_exception_type
from ccxt.base.errors import NetworkError
from live.state_refresh import AuthoritativeSurfaceUnavailable


_EMERGENCY_FILL_REFRESH_TIMEOUT_SECONDS = 5.0
_EMERGENCY_FILL_REFRESH_RETRY_SECONDS = 10.0
_EMERGENCY_QUOTE_TIMEOUT_SECONDS = 5.0


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
    degraded_evaluations: int = 0


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
                        or health.status not in {"usable", "degraded", "unavailable", "inactive"}
                        or any(type(getattr(health, field)) is not bool for field in
                               ("exit_committed", "exit_confirmed_flat", "emergency_active"))
                        or not isinstance(health.execution_blocked, str)
                        or (health.exit_committed and health.exit_confirmed_flat)
                        or not isinstance(health.reason, str)
                        or type(health.degraded_evaluations) is not int or health.degraded_evaluations < 0):
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
        health.degraded_evaluations = 0
        health.reason = reason
        if changed:
            self.save()
        return health

    def evaluated_successfully(self, scope: Scope, *, now_ms: int, degraded_reason: str = ""):
        health = self.scopes.setdefault(scope, Health())
        changed = (health.unavailable_since_ms is not None
                   or self.journal_invalid and health.last_evaluated_ms is None)
        status = "degraded" if degraded_reason else "usable"
        health.degraded_evaluations = health.degraded_evaluations + 1 if degraded_reason else 0
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
    changed = bool(obsolete)
    for scope, item in list(health.scopes.items()):
        if (scope.mode == "coin" and not signal_scope_enabled(bot, scope.pside, scope.symbol)
                and not item.exit_committed and not item.exit_confirmed_flat):
            if item.exit_started_ms is None:
                del health.scopes[scope]
                changed = True
            elif item.status != "inactive" or item.unavailable_since_ms is not None:
                # Retain completed emergency provenance for canonical replay,
                # but inactive time must not consume a later outage's grace.
                item.status = "inactive"
                item.reason = "inactive_scope"
                item.unavailable_since_ms = None
                item.emergency_active = False
                item.execution_blocked = ""
                changed = True
    if changed:
        health.save()


def scope_for(bot, pside, symbol=None):
    mode = bot._equity_hard_stop_signal_mode()
    return Scope(mode, pside, str(symbol) if mode == "coin" and symbol is not None else "")


def record_evaluation(bot, pside, symbol=None, *, degraded_reason=""):
    health = getattr(bot, "_hsl_protection_health", None)
    if health is not None:
        health.evaluated_successfully(scope_for(bot, pside, symbol),
            now_ms=int(bot.get_exchange_time()), degraded_reason=degraded_reason)


def signal_scope_enabled(bot, pside, symbol=None):
    """Use the normal signal's activity contract for new emergency decisions."""
    if bot._equity_hard_stop_signal_mode() == "coin":
        from passivbot_hsl import _equity_hard_stop_coin_active_pside
        return _equity_hard_stop_coin_active_pside(bot, pside, symbol)
    return bot._equity_hard_stop_enabled(pside)


def affected_scopes(bot, targets, details):
    """Attribute known symbol/side failures narrowly; account failures affect all targets."""
    mode = bot._equity_hard_stop_signal_mode()
    side = details.get("pside")
    symbol = details.get("symbol")
    if mode == "coin" and side in {"long", "short"} and symbol:
        return ({Scope(mode, side, str(symbol))}
                if signal_scope_enabled(bot, side, symbol) else set())
    if mode == "coin":
        return {Scope(mode, pside, symbol) for symbol, psides in targets.items()
                for pside in psides if (side is None or pside == side)
                and signal_scope_enabled(bot, pside, symbol)}
    target_sides = {pside for psides in targets.values() for pside in psides}
    return {Scope(mode, pside) for pside in ("long", "short")
            if signal_scope_enabled(bot, pside)
            and pside in target_sides
            and (mode == "unified" or side is None or pside == side)}


def _scope_matches(scope, pside, symbol):
    return ((scope.mode == "unified" or scope.pside == pside)
            and (not scope.symbol or scope.symbol == symbol))


def targets_for_scopes(bot, scopes, candidates=None):
    # Commitments own remaining exposure/orders even if sizing is later disabled.
    # Only explicit HSL disablement/mode changes retire them in reconcile_config.
    if candidates is None:
        candidates = {
            symbol: {side for side in ("long", "short")
                     if float(bot.positions.get(symbol, {}).get(side, {}).get("size", 0.0)) != 0.0
                     or any(order["position_side"] == side for order in bot.open_orders.get(symbol, []))}
            for symbol in set(bot.positions) | set(bot.open_orders)
        }
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


def emergency_evidence_needs_confirmation(bot):
    """A newly discovered fill can invalidate both cost basis and cash balance."""
    pending = getattr(bot, "_authoritative_pending_confirmations", {}) or {}
    ledger = getattr(bot, "freshness_ledger", None)
    return any(int(pending.get(surface, 0) or 0) > max(0,
        ledger.surfaces[surface].epoch if ledger is not None else 0
    ) for surface in ("positions", "balance"))


def fill_tail_matches_positions(bot):
    """The fill request must have started after this exact position observation."""
    ledger = getattr(bot, "freshness_ledger", None)
    return bool(
        ledger is not None and ledger.epoch > 0
        and "positions" in ledger.surfaces_at_epoch()
        and not emergency_evidence_needs_confirmation(bot)
        and getattr(bot, "_hsl_fill_tail_observation", None)
        == (ledger.epoch, ledger.surfaces["positions"].revision)
    )


async def evaluate_emergency(bot, candidates, *, refresh_fill_tail=True):
    """Evaluate unavailable scopes with current account and quote evidence only.

    A historical fallback never clears the outage clock. Successful normal (or
    explicitly bounded degraded) evaluation is the sole reset authority.
    """
    import passivbot_rust as pbr
    health_manager = manager(bot)
    reconcile_config(bot)
    now = int(bot.get_exchange_time())
    delay = grace_ms(bot)
    active = affected_scopes(bot, candidates, {})
    previous_pending = health_manager.pending_exits()
    needs_fill_tail = False
    for scope, health in list(health_manager.scopes.items()):
        if scope not in active or health.unavailable_since_ms is None or health.exit_committed:
            continue
        elapsed = max(0, now - health.unavailable_since_ms)
        if elapsed < delay:
            continue
        try:
            if scope.mode == "unified":
                upnl = float(await asyncio.wait_for(bot._calc_upnl_sum_strict(),
                    timeout=_EMERGENCY_QUOTE_TIMEOUT_SECONDS))
            else:
                upnl = float(await asyncio.wait_for(
                    bot._calc_upnl_sum_strict(scope.pside, scope.symbol or None),
                    timeout=_EMERGENCY_QUOTE_TIMEOUT_SECONDS))
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
        needs_fill_tail |= (scope.mode == "coin" and health.realized_loss is None
                            and has_exposure(bot, scope))
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

    # Raw-current-input RED always wins immediately. Optional historical
    # enrichment must neither postpone a newly committed close nor become an
    # unbounded prerequisite for evaluating current loss.
    update_fills = getattr(bot, "update_pnls", None)
    if (refresh_fill_tail and needs_fill_tail and callable(update_fills)
            and not (health_manager.pending_exits() - previous_pending)
            and monotonic() >= getattr(bot, "_hsl_emergency_fill_retry_at", 0.0)):
        bot._hsl_emergency_fill_retry_at = monotonic() + _EMERGENCY_FILL_REFRESH_RETRY_SECONDS
        try:
            await asyncio.wait_for(update_fills(source="hsl_emergency"),
                                   timeout=_EMERGENCY_FILL_REFRESH_TIMEOUT_SECONDS)
        except (TimeoutError, NetworkError, AuthoritativeSurfaceUnavailable, FillEventCacheContractError) as exc:
            bot._hsl_fill_tail_observation = None
            logging.warning("[risk] optional emergency fill-tail refresh unavailable; raw-UPNL protection remains active | error_type=%s",
                            bounded_exception_type(exc))
        else:
            # New fills may have changed cost basis even at unchanged net size.
            # Let the account owner confirm that change before using this tail.
            if not emergency_evidence_needs_confirmation(bot):
                await evaluate_emergency(bot, candidates, refresh_fill_tail=False)


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
