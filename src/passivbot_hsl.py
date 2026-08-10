from __future__ import annotations

import asyncio
import json
import logging
import bisect
import math
import os
import time
import traceback
from collections import deque
from typing import Any, Iterable, Optional

import passivbot_rust as pbr

from candlestick_manager import candle_range_has_full_coverage

from config.access import (
    get_optional_live_value,
    require_live_value,
)
from config.coerce import (
    normalize_hsl_cooldown_position_policy,
    normalize_hsl_restart_after_red_policy,
    normalize_hsl_signal_mode,
)
from config.pnl_lookback import parse_pnls_max_lookback_days
from fill_events_manager import signed_fee_paid_from_payload
from live.diagnostic_safety import bounded_exception_type as _bounded_hsl_exception_type
from live.event_bus import EventTypes, ReasonCodes, live_event_debug_profile_enabled
from passivbot_exceptions import FatalBotException, RestartBotException
from utils import make_get_filepath


_HSL_RISKS_DOC = "docs/equity_hard_stop_loss_risks.md"
_HSL_REPLAY_INTERVAL_MS = 60_000
_HSL_COIN_REPLAY_STARTUP_YIELD_ROWS = 1_000
_HSL_COIN_REPLAY_BACKGROUND_YIELD_ROWS = 100
_HSL_COIN_REPLAY_BACKGROUND_YIELD_SLEEP_S = 0.01
_HSL_FLATTEN_FILL_REFRESH_INTERVAL_MS = 5_000
def _hsl_flat_epsilon(qty_step: Any = 0.0) -> float:
    try:
        step = abs(float(qty_step or 0.0))
    except (TypeError, ValueError):
        step = 0.0
    if not math.isfinite(step):
        step = 0.0
    return max(1e-12, step * 0.5)


def _hsl_qty_step_for_symbol(self: Any, symbol: Any) -> float:
    qty_steps = getattr(self, "qty_steps", None)
    if not isinstance(qty_steps, dict):
        return 0.0
    try:
        value = qty_steps.get(symbol)
        if value is None:
            value = qty_steps.get(str(symbol))
        step = abs(float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0
    return step if math.isfinite(step) else 0.0


def _hsl_key_sample(value: Any, *, limit: int = 32) -> list[str]:
    if not isinstance(value, dict):
        return []
    return sorted(str(key) for key in value)[: max(0, int(limit))]


def _hsl_event_state_snapshot(
    self,
    *,
    pside: str | None,
    symbol: str | None = None,
) -> dict[str, Any]:
    if not pside:
        return {}
    state = None
    if symbol:
        coin_states = getattr(self, "_equity_hard_stop_coin", None)
        if isinstance(coin_states, dict):
            pside_states = coin_states.get(pside)
            if isinstance(pside_states, dict):
                state = pside_states.get(symbol)
    if state is None:
        states = getattr(self, "_equity_hard_stop", None)
        if isinstance(states, dict):
            state = states.get(pside)
    if not isinstance(state, dict):
        return {}
    cooldown_until_ms = state.get("cooldown_until_ms")
    return {
        "halted": bool(state.get("halted")),
        "no_restart_latched": bool(state.get("no_restart_latched")),
        "cooldown_until_present": cooldown_until_ms is not None,
        "pending_red": state.get("pending_red_since_ms") is not None,
        "has_pending_stop_event": state.get("pending_stop_event") is not None,
        "has_last_stop_event": state.get("last_stop_event") is not None,
        "red_trigger_event_emitted": bool(state.get("red_trigger_event_emitted")),
        "cooldown_intervention_active": bool(state.get("cooldown_intervention_active")),
        "cooldown_repanic_reset_pending": bool(
            state.get("cooldown_repanic_reset_pending")
        ),
        "cooldown_unresolved_residue": bool(state.get("cooldown_unresolved_residue")),
        "pnl_reset_timestamp_present": state.get("pnl_reset_timestamp_ms") is not None,
    }


def _hsl_debug_payload(
    self,
    *,
    event_type: str,
    data: dict,
    pside: str | None,
    symbol: str | None = None,
    status: str | None = None,
    reason_code: str | None = None,
) -> dict[str, Any]:
    debug: dict[str, Any] = {
        "event_type": str(event_type),
        "data_keys": _hsl_key_sample(data),
    }
    if status is not None:
        debug["status"] = str(status)
    if reason_code is not None:
        debug["reason_code"] = str(reason_code)
    for key in ("signal_mode", "tier"):
        if data.get(key) is not None:
            debug[key] = str(data[key])
    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        debug["metrics_keys"] = _hsl_key_sample(metrics)
    state_snapshot = _hsl_event_state_snapshot(self, pside=pside, symbol=symbol)
    if state_snapshot:
        debug["state"] = state_snapshot
    return {key: value for key, value in debug.items() if value not in (None, [], {})}


def _best_effort_hsl_debug_payload(
    self,
    *,
    event_type: str,
    data: dict,
    pside: str | None,
    symbol: str | None = None,
    status: str | None = None,
    reason_code: str | None = None,
) -> dict[str, Any] | None:
    try:
        return _hsl_debug_payload(
            self,
            event_type=event_type,
            data=data,
            pside=pside,
            symbol=symbol,
            status=status,
            reason_code=reason_code,
        )
    except Exception as exc:
        logging.debug(
            "[event] failed to build HSL debug payload type=%s: %s",
            event_type,
            _bounded_hsl_exception_type(exc),
        )
        return None


def _hsl_event_data(metrics: dict | None = None, extra: dict | None = None) -> dict[str, Any]:
    data = dict(metrics or {})
    data.pop("changed", None)
    if extra:
        data.update(extra)
    return data


def _emit_hsl_event(
    self,
    event_type: str,
    tags: tuple[str, ...],
    data: dict,
    *,
    pside: str | None,
    symbol: str | None = None,
    ts: int | None = None,
    level: str = "info",
    status: str | None = None,
    reason_code: str | None = None,
) -> None:
    live_event_delivered = False
    event_data = dict(data or {})
    if live_event_debug_profile_enabled(self, "hsl"):
        debug = _best_effort_hsl_debug_payload(
            self,
            event_type=event_type,
            data=event_data,
            pside=pside,
            symbol=symbol,
            status=status,
            reason_code=reason_code,
        )
        if debug:
            event_data["debug_profile"] = "hsl"
            event_data["debug"] = debug
    try:
        emit = getattr(self, "_emit_live_event", None)
        pipeline = getattr(self, "_live_event_pipeline", None)
        if callable(emit) and pipeline is not None:
            live_event_delivered = emit(
                event_type,
                level=level,
                component="risk.hsl",
                tags=tags,
                cycle_id=getattr(self, "_live_event_current_cycle_id", None),
                symbol=symbol,
                pside=pside,
                status=status,
                reason_code=reason_code,
                data=event_data,
            ) is not None
    except Exception as exc:
        logging.debug(
            "[event] failed to emit HSL live event type=%s: %s",
            event_type,
            _bounded_hsl_exception_type(exc),
        )
    if live_event_delivered:
        return
    try:
        record = getattr(self, "_monitor_record_event", None)
        if callable(record):
            record(
                event_type,
                tags,
                event_data,
                pside=pside,
                symbol=symbol,
                ts=ts,
            )
    except Exception as exc:
        logging.debug(
            "[event] failed to emit HSL legacy monitor event type=%s: %s",
            event_type,
            _bounded_hsl_exception_type(exc),
        )


def _emit_runtime_forced_mode_changed_event(
    self,
    *,
    pside: str,
    action: str,
    previous_mode: str | None = None,
    mode: str | None = None,
    symbol: str | None = None,
    symbols: Any = None,
    previous_modes: dict[str, str] | None = None,
    modes: dict[str, str] | None = None,
    reason_code: str | None = None,
) -> None:
    try:
        emit = getattr(self, "_emit_risk_mode_changed_event", None)
        if callable(emit):
            emit(
                pside=pside,
                source="hsl",
                action=action,
                previous_mode=previous_mode,
                mode=mode,
                symbol=symbol,
                symbols=symbols,
                previous_modes=previous_modes,
                modes=modes,
                reason_code=reason_code,
            )
    except Exception as exc:
        logging.debug(
            "[event] failed to emit HSL runtime forced mode event pside=%s symbol=%s: %s",
            pside,
            symbol,
            _bounded_hsl_exception_type(exc),
        )


def _emit_hsl_red_triggered_once(
    self,
    state: dict,
    data: dict,
    *,
    pside: str,
    symbol: str | None = None,
    ts: int | None = None,
    reason_code: str = "red_threshold_crossed",
) -> None:
    if state.get("red_trigger_event_emitted"):
        return
    no_exchange_close_needed = bool(data.get("no_exchange_close_needed"))
    _emit_hsl_event(
        self,
        "hsl.red_triggered",
        ("hsl", "risk", "red"),
        data,
        pside=pside,
        symbol=symbol,
        ts=ts,
        level="info" if no_exchange_close_needed else "critical",
        status="succeeded" if no_exchange_close_needed else "degraded",
        reason_code=reason_code,
    )
    state["red_trigger_event_emitted"] = True


def _emit_hsl_red_finalized_without_order(
    self,
    stop_event: dict,
    *,
    pside: str,
    symbol: str | None,
    stop_ts_ms: int,
    stop_event_anchor_source: str,
    stop_event_anchor_fallback_used: bool,
    cooldown_until_ms: int | None,
    flat_confirmations: int | None,
    position_count: int | None = None,
    entry_orders: int | None = None,
    nonpanic_close_orders: int | None = None,
) -> None:
    data: dict[str, Any] = {
        "reason": "red_finalized_without_exchange_order",
        "no_exchange_close_needed": True,
        "exchange_close_order_submitted": False,
        "panic_order_submitted_count": 0,
        "stop_event_timestamp_ms": int(stop_ts_ms),
        "stop_event_anchor_source": str(stop_event_anchor_source),
        "stop_event_anchor_timestamp_ms": int(stop_ts_ms),
        "stop_event_anchor_fallback_used": bool(stop_event_anchor_fallback_used),
        "cooldown_until_ms": None
        if cooldown_until_ms is None
        else int(cooldown_until_ms),
    }
    if symbol is not None:
        data["symbol_position_open"] = False
    if position_count is not None:
        data["position_count"] = int(position_count)
    if entry_orders is not None:
        data["entry_orders"] = int(entry_orders)
    if nonpanic_close_orders is not None:
        data["nonpanic_close_orders"] = int(nonpanic_close_orders)
    if flat_confirmations is not None:
        data["flat_confirmations"] = int(flat_confirmations)
    for key in (
        "signal_mode",
        "tier",
        "drawdown_raw",
        "drawdown_ema",
        "drawdown_score",
        "red_threshold",
    ):
        if key in stop_event:
            data[key] = stop_event[key]
    _emit_hsl_event(
        self,
        EventTypes.HSL_RED_FINALIZED_WITHOUT_ORDER,
        ("hsl", "risk", "red"),
        data,
        pside=pside,
        symbol=symbol,
        ts=stop_ts_ms,
        level="info",
        status="succeeded",
        reason_code=ReasonCodes.HSL_RED_FINALIZED_WITHOUT_EXCHANGE_ORDER,
    )


def _emit_hsl_replay_event(
    self,
    event_type: str,
    data: dict[str, Any],
    *,
    pside: str | None = None,
    symbol: str | None = None,
    level: str = "debug",
    status: str | None = None,
    reason_code: str | None = None,
) -> None:
    try:
        _emit_hsl_event(
            self,
            event_type,
            ("hsl", "risk", "replay"),
            data,
            pside=pside,
            symbol=symbol,
            level=level,
            status=status,
            reason_code=reason_code,
        )
    except Exception as exc:
        logging.debug(
            "[event] failed to emit HSL replay event type=%s: %s",
            event_type,
            _bounded_hsl_exception_type(exc),
        )


def _calc_hsl_pnl(position_side, entry_price, close_price, qty, c_mult):
    if isinstance(position_side, str):
        if position_side == "long":
            return pbr.calc_pnl_long(entry_price, close_price, qty, c_mult)
        return pbr.calc_pnl_short(entry_price, close_price, qty, c_mult)
    return pbr.calc_pnl_long(entry_price, close_price, qty, c_mult)


def _hsl_psides(self) -> tuple[str, str]:
    return ("long", "short")


def _hsl_state(self, pside: str) -> dict[str, Any]:
    return self._equity_hard_stop[pside]


def _equity_hard_stop_make_state(self) -> dict[str, Any]:
    return {
        "runtime": pbr.EquityHardStopRuntime(),
        "strategy_pnl_peak": pbr.EquityHardStopRollingPeak(),
        "no_restart_peak_strategy_equity": 0.0,
        "halted": False,
        "no_restart_latched": False,
        "last_metrics": None,
        "last_red_progress": None,
        "red_flat_confirmations": 0,
        "pending_red_since_ms": None,
        "cooldown_until_ms": None,
        "pending_stop_event": None,
        "last_stop_event": None,
        "red_trigger_event_emitted": False,
        "last_raw_red_pending_event_ms": 0,
        "last_status_log_ms": 0,
        "last_cooldown_log_ms": 0,
        "cooldown_intervention_active": False,
        "cooldown_repanic_reset_pending": False,
        "cooldown_repanic_since_ms": None,
        "cooldown_repanic_start_sizes": None,
        "last_cooldown_intervention_log_ms": 0,
        "last_missing_flatten_fill_log_ms": 0,
        "last_missing_flatten_fill_refresh_ms": 0,
        "cooldown_unresolved_residue": False,
        "pnl_reset_timestamp_ms": None,
    }


def _hsl_coin_state(self, pside: str, symbol: str) -> dict[str, Any]:
    states = getattr(self, "_equity_hard_stop_coin", None)
    if states is None:
        self._equity_hard_stop_coin = {"long": {}, "short": {}}
        states = self._equity_hard_stop_coin
    pside_states = states.setdefault(pside, {})
    if symbol not in pside_states:
        pside_states[symbol] = self._equity_hard_stop_make_state()
    return pside_states[symbol]


def _equity_hard_stop_coin_active_pside(
    self, pside: str, symbol: Optional[str] = None
) -> bool:
    if not self._equity_hard_stop_enabled(pside, symbol=symbol):
        return False
    n_positions_raw = float(self.bot_value(pside, "n_positions"))
    if not math.isfinite(n_positions_raw) or n_positions_raw < 0.0:
        raise ValueError(
            f"coin HSL n_positions must be finite and >= 0 for {pside}, got {n_positions_raw}"
        )
    n_positions = int(round(n_positions_raw))
    if n_positions <= 0:
        if n_positions_raw == 0.0:
            return False
        raise ValueError(
            f"coin HSL n_positions must round to > 0 for {pside}, got {n_positions_raw}"
        )
    total_wallet_exposure_limit = float(self.bot_value(pside, "total_wallet_exposure_limit"))
    if not math.isfinite(total_wallet_exposure_limit) or total_wallet_exposure_limit < 0.0:
        raise ValueError(
            "coin HSL total_wallet_exposure_limit must be finite and >= 0 for "
            f"{pside}, got {total_wallet_exposure_limit}"
        )
    return total_wallet_exposure_limit > 0.0


def _format_hsl_startup_config(
    pside: str,
    *,
    red_threshold: float,
    ema_span_minutes: float,
    cooldown_minutes_after_red: float,
    no_restart_drawdown_threshold: float,
    signal_mode: str,
    ratio_yellow: float,
    ratio_orange: float,
    orange_tier_mode: str,
    panic_close_order_type: str,
    restart_after_red_policy: str,
) -> str:
    """Return the bounded operator-facing HSL startup configuration summary."""
    return (
        f"[risk] HSL[{pside}] on | red={red_threshold:.6g} ema={ema_span_minutes:.6g} "
        f"cd={cooldown_minutes_after_red:.6g} no-r={no_restart_drawdown_threshold:.6g} "
        f"mode={signal_mode} tiers={ratio_yellow:.6g}/{ratio_orange:.6g} "
        f"orange={orange_tier_mode} panic={panic_close_order_type} "
        f"restart={restart_after_red_policy}"
    )


def _parse_hsl_config(self) -> dict[str, dict[str, Any]]:
    signal_mode = self._equity_hard_stop_signal_mode()
    out = {}
    for pside in self._hsl_psides():
        tier_ratios_raw = self.bot_value(pside, "hsl_tier_ratios")
        if not isinstance(tier_ratios_raw, dict):
            raise TypeError(
                f"bot.{pside}.hsl_tier_ratios must be a dict, got {type(tier_ratios_raw).__name__}"
            )
        enabled = bool(self.bot_value(pside, "hsl_enabled"))
        red_threshold = float(self.bot_value(pside, "hsl_red_threshold"))
        ema_span_minutes = float(self.bot_value(pside, "hsl_ema_span_minutes"))
        cooldown_minutes_after_red = float(self.bot_value(pside, "hsl_cooldown_minutes_after_red"))
        no_restart_drawdown_threshold = float(
            self.bot_value(pside, "hsl_no_restart_drawdown_threshold")
        )
        ratio_yellow = float(self.bot_value(pside, "hsl_tier_ratios.yellow"))
        ratio_orange = float(self.bot_value(pside, "hsl_tier_ratios.orange"))
        orange_tier_mode = str(self.bot_value(pside, "hsl_orange_tier_mode"))
        panic_close_order_type = str(self.bot_value(pside, "hsl_panic_close_order_type"))
        restart_after_red_policy = normalize_hsl_restart_after_red_policy(
            self.bot_value(pside, "hsl_restart_after_red_policy"),
            path=f"bot.{pside}.hsl_restart_after_red_policy",
        )

        if enabled and red_threshold <= 0.0:
            raise ValueError(f"bot.{pside}.hsl_red_threshold must be > 0.0 when enabled")
        if enabled and ema_span_minutes <= 0.0:
            raise ValueError(f"bot.{pside}.hsl_ema_span_minutes must be > 0.0 when enabled")
        if cooldown_minutes_after_red < 0.0:
            raise ValueError(f"bot.{pside}.hsl_cooldown_minutes_after_red must be >= 0.0")
        if no_restart_drawdown_threshold < red_threshold:
            logging.info(
                "[config] clamped bot.%s.hsl_no_restart_drawdown_threshold %.6f -> %.6f to match hsl_red_threshold",
                pside,
                no_restart_drawdown_threshold,
                red_threshold,
            )
            no_restart_drawdown_threshold = red_threshold
        if not (red_threshold <= no_restart_drawdown_threshold <= 1.0):
            raise ValueError(
                f"bot.{pside}.hsl_no_restart_drawdown_threshold must satisfy "
                "hsl_red_threshold <= hsl_no_restart_drawdown_threshold <= 1.0"
            )
        if not (0.0 < ratio_yellow < ratio_orange < 1.0):
            raise ValueError(f"bot.{pside}.hsl_tier_ratios must satisfy 0 < yellow < orange < 1")
        if orange_tier_mode not in {"graceful_stop", "tp_only_with_active_entry_cancellation"}:
            raise ValueError(
                f"bot.{pside}.hsl_orange_tier_mode must be one of "
                "{graceful_stop, tp_only_with_active_entry_cancellation}"
            )
        if panic_close_order_type not in {"market", "limit"}:
            raise ValueError(
                f"bot.{pside}.hsl_panic_close_order_type must be one of {{market, limit}}"
            )

        out[pside] = {
            "enabled": enabled,
            "red_threshold": red_threshold,
            "ema_span_minutes": ema_span_minutes,
            "cooldown_minutes_after_red": cooldown_minutes_after_red,
            "no_restart_drawdown_threshold": no_restart_drawdown_threshold,
            "tier_ratios": {"yellow": ratio_yellow, "orange": ratio_orange},
            "orange_tier_mode": orange_tier_mode,
            "panic_close_order_type": panic_close_order_type,
            "restart_after_red_policy": restart_after_red_policy,
        }
        if enabled:
            logging.warning(
                "[risk] HSL[%s] enabled; review %s. Deposits, withdrawals, "
                "balance overrides, and HSL mode/budget/threshold "
                "changes can reinterpret reconstructed history.",
                pside,
                _HSL_RISKS_DOC,
            )
            live_cfg = self.config.get("live") if isinstance(self.config, dict) else None
            if isinstance(live_cfg, dict) and bool(
                live_cfg.get("hsl_accept_incomplete_history", False)
            ):
                logging.critical(
                    "[risk] HSL[%s] hsl_accept_incomplete_history OVERRIDE is "
                    "active: HSL evidence incomplete; panic/cooldown/no-restart "
                    "may be wrong. This is a dangerous per-run flag - do not "
                    "persist it in config files.",
                    pside,
                )
            logging.info(
                _format_hsl_startup_config(
                    pside,
                    red_threshold=red_threshold,
                    ema_span_minutes=ema_span_minutes,
                    cooldown_minutes_after_red=cooldown_minutes_after_red,
                    no_restart_drawdown_threshold=no_restart_drawdown_threshold,
                    signal_mode=signal_mode,
                    ratio_yellow=ratio_yellow,
                    ratio_orange=ratio_orange,
                    orange_tier_mode=orange_tier_mode,
                    panic_close_order_type=panic_close_order_type,
                    restart_after_red_policy=restart_after_red_policy,
                )
            )
    return out


def _equity_hard_stop_no_restart_latched(
    cfg: dict[str, Any], drawdown_raw: float, drawdown_ema: float
) -> bool:
    """Shared live/backtest no-restart trigger, owned by Rust.

    Contract (fable audit plan, clarified 2026-07-06): the permanent halt is
    conservative and trips on max(drawdown_raw, drawdown_ema), catching either
    catastrophic instantaneous damage or sustained smoothed damage.
    """
    policy = normalize_hsl_restart_after_red_policy(
        cfg.get("restart_after_red_policy", "threshold"),
        path="hsl.restart_after_red_policy",
    )
    return bool(
        pbr.hsl_no_restart_triggered(
            policy,
            float(drawdown_raw),
            float(drawdown_ema),
            float(cfg["no_restart_drawdown_threshold"]),
        )
    )


def _equity_hard_stop_config(
    self, pside: str, symbol: Optional[str] = None
) -> dict[str, Any]:
    global_cfg = self.hsl[pside]
    if symbol is None:
        return global_cfg
    coin_overrides = getattr(self, "coin_overrides", {}) or {}
    if symbol not in coin_overrides or not callable(getattr(self, "bp", None)):
        return global_cfg
    if self._equity_hard_stop_signal_mode() != "coin":
        return global_cfg
    tier_ratios = dict(global_cfg["tier_ratios"])
    override_ratios = self.bp(pside, "hsl_tier_ratios", symbol)
    if isinstance(override_ratios, dict):
        tier_ratios.update(override_ratios)
    return {
        "cooldown_minutes_after_red": float(
            self.bp(pside, "hsl_cooldown_minutes_after_red", symbol)
        ),
        "ema_span_minutes": float(self.bp(pside, "hsl_ema_span_minutes", symbol)),
        "enabled": bool(self.bp(pside, "hsl_enabled", symbol)),
        "no_restart_drawdown_threshold": float(
            self.bp(pside, "hsl_no_restart_drawdown_threshold", symbol)
        ),
        "orange_tier_mode": str(self.bp(pside, "hsl_orange_tier_mode", symbol)),
        "panic_close_order_type": str(
            self.bp(pside, "hsl_panic_close_order_type", symbol)
        ),
        "red_threshold": float(self.bp(pside, "hsl_red_threshold", symbol)),
        "restart_after_red_policy": normalize_hsl_restart_after_red_policy(
            self.bp(pside, "hsl_restart_after_red_policy", symbol),
            path=f"coin HSL {symbol} {pside}.restart_after_red_policy",
        ),
        "tier_ratios": tier_ratios,
    }


def _equity_hard_stop_enabled(
    self, pside: Optional[str] = None, *, symbol: Optional[str] = None
) -> bool:
    if not hasattr(self, "hsl") or not isinstance(self.hsl, dict):
        legacy_cfg = getattr(self, "equity_hard_stop_loss", None)
        enabled = bool(isinstance(legacy_cfg, dict) and legacy_cfg.get("enabled", False))
        if pside is None:
            return enabled
        return enabled
    if self._equity_hard_stop_signal_mode() != "coin":
        if pside is None:
            return any(bool(self.hsl[x]["enabled"]) for x in ("long", "short"))
        return bool(self.hsl[pside]["enabled"])
    if symbol is not None:
        if pside is None:
            return any(
                bool(_equity_hard_stop_config(self, side, symbol)["enabled"])
                for side in ("long", "short")
            )
        return bool(_equity_hard_stop_config(self, pside, symbol)["enabled"])
    symbols = tuple((getattr(self, "coin_overrides", {}) or {}).keys())
    if pside is None:
        return any(
            bool(self.hsl[side]["enabled"])
            or any(
                bool(_equity_hard_stop_config(self, side, coin)["enabled"])
                for coin in symbols
            )
            for side in ("long", "short")
        )
    return bool(self.hsl[pside]["enabled"]) or any(
        bool(_equity_hard_stop_config(self, pside, coin)["enabled"])
        for coin in symbols
    )


def _equity_hard_stop_signal_mode(self) -> str:
    config = getattr(self, "config", {})
    return normalize_hsl_signal_mode(require_live_value(config, "hsl_signal_mode"))


def _equity_hard_stop_balance_override_active(self) -> bool:
    return getattr(self, "balance_override", None) is not None


def _equity_hard_stop_validate_balance_source_for_history_replay(self) -> None:
    signal_mode = self._equity_hard_stop_signal_mode()
    if signal_mode == "coin":
        return
    if not self._equity_hard_stop_balance_override_active():
        return
    enabled_psides = [
        pside for pside in self._hsl_psides() if self._equity_hard_stop_enabled(pside)
    ]
    if not enabled_psides:
        return
    _emit_hsl_replay_event(
        self,
        EventTypes.HSL_REPLAY_FAILED,
        {
            "signal_mode": signal_mode,
            "balance_override_active": True,
            "enabled_psides": enabled_psides,
        },
        level="critical",
        status="failed",
        reason_code=ReasonCodes.HSL_BALANCE_OVERRIDE_ACCOUNT_LEVEL_REPLAY_UNSAFE,
    )
    raise RuntimeError(
        "HSL equity history replay is unsafe with balance_override for "
        f"signal_mode={signal_mode!r} enabled_psides={','.join(enabled_psides)}. "
        "Unified/pside HSL reconstructs historical drawdown from current balance "
        "minus realized PnL; with a balance override this can create a synthetic "
        "peak and false RED panic. Remove the balance override, use "
        "hsl_signal_mode='coin', disable HSL, or initialize an explicit HSL "
        "baseline/checkpoint before live trading."
    )


def _equity_hard_stop_cooldown_position_policy(self) -> str:
    config = getattr(self, "config", {})
    return normalize_hsl_cooldown_position_policy(
        get_optional_live_value(
            config,
            "hsl_position_during_cooldown_policy",
            "panic",
        )
    )


def _equity_hard_stop_format_remaining_time(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    days, rem = divmod(total, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes or hours or days:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return "".join(parts)


def _equity_hard_stop_replay_marker_confirms_red(metrics: dict) -> bool:
    try:
        drawdown_score = float(metrics["drawdown_score"])
        red_threshold = float(metrics["red_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("HSL replay panic marker confirmation metrics are incomplete") from exc
    threshold = red_threshold - 1e-12
    return (
        str(metrics.get("tier")) == "red"
        or drawdown_score >= threshold
    )


def _equity_hard_stop_infer_replay_contract(
    self, pside: str, fill_events: list[dict], now_ms: int
) -> dict[str, Any]:
    policy = self._equity_hard_stop_cooldown_position_policy()
    cooldown_minutes = float(self.hsl[pside]["cooldown_minutes_after_red"])
    cooldown_ms = int(round(cooldown_minutes * 60_000.0)) if cooldown_minutes > 0.0 else 0
    current_symbols = self._equity_hard_stop_position_symbols(pside)
    pos_now = bool(current_symbols)
    panic_events = [
        evt
        for evt in fill_events
        if isinstance(evt, dict)
        and str(evt.get("pside") or "").lower() == pside
        and "panic" in str(evt.get("pb_order_type") or "")
    ]
    latest_panic_ts = int(panic_events[-1]["timestamp"]) if panic_events else None
    cooldown_until_ms = (
        None
        if latest_panic_ts is None or cooldown_ms <= 0
        else int(latest_panic_ts + cooldown_ms)
    )
    intervention_entry_ts = None
    if latest_panic_ts is not None:
        for evt in fill_events:
            if not isinstance(evt, dict):
                continue
            if str(evt.get("pside") or "").lower() != pside:
                continue
            evt_ts = int(evt["timestamp"])
            if evt_ts <= latest_panic_ts:
                continue
            if cooldown_until_ms is not None and evt_ts >= cooldown_until_ms:
                break
            if evt.get("action") == "increase" and "panic" not in str(evt.get("pb_order_type") or ""):
                intervention_entry_ts = evt_ts
                break
    active_cooldown_now = cooldown_until_ms is not None and now_ms < cooldown_until_ms
    unresolved_residue = bool(active_cooldown_now and pos_now and intervention_entry_ts is None)
    intervention_active = bool(active_cooldown_now and pos_now and intervention_entry_ts is not None)
    replay_reset_boundary_ts = None
    if latest_panic_ts is not None:
        replay_reset_boundary_ts = latest_panic_ts
    if policy == "normal" and intervention_entry_ts is not None:
        replay_reset_boundary_ts = intervention_entry_ts
    return {
        "policy": policy,
        "latest_panic_ts": latest_panic_ts,
        "cooldown_until_ms": cooldown_until_ms,
        "intervention_entry_ts": intervention_entry_ts,
        "replay_reset_boundary_ts": replay_reset_boundary_ts,
        "active_cooldown_now": active_cooldown_now,
        "pos_now": pos_now,
        "current_symbols": current_symbols,
        "intervention_active": intervention_active,
        "unresolved_residue": unresolved_residue,
    }


def _equity_hard_stop_infer_coin_replay_contract(
    self, pside: str, symbol: str, fill_events: list[dict], now_ms: int
) -> dict[str, Any]:
    policy = self._equity_hard_stop_cooldown_position_policy()
    cooldown_minutes = float(
        _equity_hard_stop_config(self, pside, symbol)["cooldown_minutes_after_red"]
    )
    cooldown_ms = int(round(cooldown_minutes * 60_000.0)) if cooldown_minutes > 0.0 else 0
    pos_now = self._equity_hard_stop_has_open_position_symbol(pside, symbol)
    panic_events = [
        evt
        for evt in fill_events
        if isinstance(evt, dict)
        and str(evt.get("pside") or evt.get("position_side") or "").lower() == pside
        and _equity_hard_stop_fill_symbol(evt) == symbol
        and "panic" in str(evt.get("pb_order_type") or "")
    ]
    latest_panic_ts = int(panic_events[-1]["timestamp"]) if panic_events else None
    cooldown_until_ms = (
        None
        if latest_panic_ts is None or cooldown_ms <= 0
        else int(latest_panic_ts + cooldown_ms)
    )
    intervention_entry_ts = None
    if latest_panic_ts is not None:
        for evt in fill_events:
            if not isinstance(evt, dict):
                continue
            if str(evt.get("pside") or evt.get("position_side") or "").lower() != pside:
                continue
            if _equity_hard_stop_fill_symbol(evt) != symbol:
                continue
            evt_ts = int(evt["timestamp"])
            if evt_ts <= latest_panic_ts:
                continue
            if cooldown_until_ms is not None and evt_ts >= cooldown_until_ms:
                break
            if evt.get("action") == "increase" and "panic" not in str(evt.get("pb_order_type") or ""):
                intervention_entry_ts = evt_ts
                break
    active_cooldown_now = cooldown_until_ms is not None and now_ms < cooldown_until_ms
    unresolved_residue = bool(active_cooldown_now and pos_now and intervention_entry_ts is None)
    intervention_active = bool(active_cooldown_now and pos_now and intervention_entry_ts is not None)
    replay_reset_boundary_ts = None
    if latest_panic_ts is not None:
        replay_reset_boundary_ts = latest_panic_ts
    if policy == "normal" and intervention_entry_ts is not None:
        replay_reset_boundary_ts = intervention_entry_ts
    return {
        "policy": policy,
        "latest_panic_ts": latest_panic_ts,
        "cooldown_until_ms": cooldown_until_ms,
        "intervention_entry_ts": intervention_entry_ts,
        "replay_reset_boundary_ts": replay_reset_boundary_ts,
        "active_cooldown_now": active_cooldown_now,
        "pos_now": pos_now,
        "symbol": symbol,
        "intervention_active": intervention_active,
        "unresolved_residue": unresolved_residue,
    }


def _equity_hard_stop_halted_mode(self, pside: str, symbol: str | None) -> str:
    state = self._hsl_state(pside)
    policy = self._equity_hard_stop_cooldown_position_policy()
    size = 0.0
    if symbol is not None:
        size = float(self.positions.get(symbol, {}).get(pside, {}).get("size", 0.0) or 0.0)
    if state.get("cooldown_unresolved_residue", False):
        return "panic" if size != 0.0 else "graceful_stop"
    if policy == "panic":
        return "panic" if size != 0.0 else "graceful_stop"
    if policy == "manual":
        return "manual" if size != 0.0 else "graceful_stop"
    if policy == "tp_only":
        return "tp_only" if size != 0.0 else "graceful_stop"
    if policy in {"normal", "graceful_stop"}:
        return "graceful_stop"
    return "graceful_stop"


def _equity_hard_stop_panic_close_order_type(
    self, pside: str, symbol: Optional[str] = None
) -> str:
    hsl_cfg = getattr(self, "hsl", None)
    if isinstance(hsl_cfg, dict) and pside in hsl_cfg and isinstance(hsl_cfg[pside], dict):
        return str(
            _equity_hard_stop_config(self, pside, symbol).get(
                "panic_close_order_type", "market"
            )
        )
    legacy_cfg = getattr(self, "equity_hard_stop_loss", None)
    if isinstance(legacy_cfg, dict):
        return str(legacy_cfg.get("panic_close_order_type", "market"))
    return "market"


def _equity_hard_stop_signal_values(
    self,
    pside: str,
    *,
    realized_pnl_total: float,
    realized_pnl_pside: float,
    unrealized_pnl_pside: float,
    unrealized_pnl_total: Optional[float] = None,
) -> tuple[str, float, float]:
    signal_mode = self._equity_hard_stop_signal_mode()
    if signal_mode == "pside":
        return signal_mode, float(realized_pnl_pside), float(unrealized_pnl_pside)
    if unrealized_pnl_total is None:
        raise ValueError(f"HSL[{pside}] unified signal mode requires unrealized_pnl_total sample input")
    if not math.isfinite(unrealized_pnl_total):
        raise ValueError(f"unrealized_pnl_total must be finite, got {unrealized_pnl_total}")
    return signal_mode, float(realized_pnl_total), float(unrealized_pnl_total)


def _equity_hard_stop_latch_path(self, pside: str, symbol: Optional[str] = None) -> str:
    if symbol:
        safe_symbol = str(symbol).replace("/", "_").replace(":", "_")
        return make_get_filepath(
            f"caches/equity_hard_stop/{self.exchange}/{self.user}_{pside}_{safe_symbol}.json"
        )
    return make_get_filepath(f"caches/equity_hard_stop/{self.exchange}/{self.user}_{pside}.json")


def _equity_hard_stop_write_latch(self, pside: str, metrics: dict, symbol: Optional[str] = None) -> str:
    path = self._equity_hard_stop_latch_path(pside, symbol=symbol)
    payload = dict(metrics)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)
    return path


def _equity_hard_stop_remove_latch_file(self, pside: str, symbol: Optional[str] = None) -> None:
    path = self._equity_hard_stop_latch_path(pside, symbol=symbol)
    if os.path.isfile(path):
        os.remove(path)


def _equity_hard_stop_reset_state(self) -> None:
    for pside in self._hsl_psides():
        state = self._hsl_state(pside)
        state["runtime"].reset()
        state["strategy_pnl_peak"].reset()
        state["no_restart_peak_strategy_equity"] = 0.0
        state["halted"] = False
        state["no_restart_latched"] = False
        state["last_metrics"] = None
        state["last_red_progress"] = None
        state["red_flat_confirmations"] = 0
        state["pending_red_since_ms"] = None
        state["cooldown_until_ms"] = None
        state["pending_stop_event"] = None
        state["last_stop_event"] = None
        state["red_trigger_event_emitted"] = False
        state["last_raw_red_pending_event_ms"] = 0
        state["last_status_log_ms"] = 0
        state["last_cooldown_log_ms"] = 0
        state["cooldown_intervention_active"] = False
        state["cooldown_repanic_reset_pending"] = False
        state["cooldown_repanic_since_ms"] = None
        state["cooldown_repanic_start_sizes"] = None
        state["last_cooldown_intervention_log_ms"] = 0
        state["last_missing_flatten_fill_log_ms"] = 0
        state["last_missing_flatten_fill_refresh_ms"] = 0
        state["cooldown_unresolved_residue"] = False
    self._runtime_forced_modes = {"long": {}, "short": {}}


def _equity_hard_stop_runtime_initialized(self, pside: str) -> bool:
    return bool(self._hsl_state(pside)["runtime"].initialized())


def _equity_hard_stop_runtime_red_latched(self, pside: str) -> bool:
    return bool(self._hsl_state(pside)["runtime"].red_latched())


def _equity_hard_stop_runtime_tier(self, pside: str) -> str:
    return str(self._hsl_state(pside)["runtime"].tier())


def _equity_hard_stop_fill_pside_optional(fill: Any) -> Optional[str]:
    if isinstance(fill, dict):
        raw = fill.get("position_side", fill.get("pside"))
    else:
        raw = getattr(fill, "position_side", getattr(fill, "pside", None))
    out = str(raw).lower()
    return out if out in {"long", "short"} else None


def _equity_hard_stop_fill_pside(fill: Any) -> str:
    return _equity_hard_stop_fill_pside_optional(fill) or "long"


def _equity_hard_stop_fill_symbol(fill: Any) -> str:
    if isinstance(fill, dict):
        raw = fill.get("symbol", fill.get("coin", ""))
    else:
        raw = getattr(fill, "symbol", getattr(fill, "coin", ""))
    return str(raw)


def _equity_hard_stop_fill_timestamp_ms(fill: Any) -> int:
    if isinstance(fill, dict):
        raw = fill.get("timestamp", fill.get("timestamp_ms", 0))
    else:
        raw = getattr(fill, "timestamp", getattr(fill, "timestamp_ms", 0))
    return int(raw or 0)


def _equity_hard_stop_event_value(fill: Any, key: str, default: Any = None) -> Any:
    if isinstance(fill, dict):
        return fill.get(key, default)
    return getattr(fill, key, default)


def _equity_hard_stop_fill_action(fill: Any) -> str:
    """Return an explicit or unambiguously inferred position-size action."""
    explicit = _equity_hard_stop_event_value(fill, "action")
    if explicit is not None and str(explicit).strip():
        action = str(explicit).lower()
        return action if action in {"increase", "decrease"} else ""

    raw_pside = _equity_hard_stop_event_value(
        fill,
        "position_side",
        _equity_hard_stop_event_value(fill, "pside"),
    )
    pside = str(raw_pside or "").lower()
    side = str(_equity_hard_stop_event_value(fill, "side", "") or "").lower()
    if pside not in {"long", "short"} or side not in {"buy", "sell"}:
        return ""
    if pside == "long":
        return "increase" if side == "buy" else "decrease"
    return "increase" if side == "sell" else "decrease"


def _equity_hard_stop_latest_panic_fill_timestamp_ms(
    self,
    pside: str,
    *,
    symbol: Optional[str] = None,
    since_ms: Optional[int] = None,
    fallback_ms: Optional[int] = None,
) -> int:
    latest_ts = _equity_hard_stop_latest_panic_fill_timestamp_optional_ms(
        self,
        pside,
        symbol=symbol,
        since_ms=since_ms,
    )
    if latest_ts is not None:
        return int(latest_ts)
    if fallback_ms is not None:
        return int(fallback_ms)
    return int(self.get_exchange_time())


def _equity_hard_stop_latest_flatten_fill_timestamp_optional_ms(
    self,
    pside: str,
    *,
    symbol: Optional[str] = None,
    since_ms: Optional[int] = None,
    replay_start_sizes: Optional[dict[str, float]] = None,
) -> Optional[int]:
    """Return fill evidence for the scope's transition to flat.

    B2.1 contract: the episode ends (and cooldown anchors) at the fill that
    makes the scoped position set flat, by any means - not only bot-emitted
    panic fills. Ordinary finalization uses the latest episode-bounded fill
    once authoritative state is flat. Cooldown re-panic finalization supplies
    ``replay_start_sizes`` to prove the exact fill that makes the scope flat.
    """
    candidates: list[Any] = []
    if self._pnls_manager is not None:
        for event in self._pnls_manager.get_events():
            if _equity_hard_stop_fill_pside(event) != pside:
                continue
            if symbol is not None and _equity_hard_stop_fill_symbol(event) != symbol:
                continue
            event_ts = _equity_hard_stop_fill_timestamp_ms(event)
            if since_ms is not None and event_ts < int(since_ms):
                continue
            candidates.append(event)
    if replay_start_sizes is None:
        return max(
            (_equity_hard_stop_fill_timestamp_ms(event) for event in candidates),
            default=None,
        )

    # Cooldown re-panic finalization starts from an authoritative non-flat
    # position snapshot. Replay later scoped fills from those sizes and accept
    # only the first fill that actually makes the whole scope flat. This keeps
    # cached intervention entries and partial closes from masquerading as the
    # episode-ending fill while the real close is still unavailable.
    flat_epsilon = _hsl_flat_epsilon(0.0)
    running_sizes = {
        str(replay_symbol): abs(float(size))
        for replay_symbol, size in replay_start_sizes.items()
        if math.isfinite(float(size)) and abs(float(size)) > flat_epsilon
    }
    if not running_sizes:
        return None
    for event in sorted(candidates, key=_equity_hard_stop_fill_timestamp_ms):
        event_symbol = _equity_hard_stop_fill_symbol(event)
        action = _equity_hard_stop_fill_action(event)
        qty = _equity_hard_stop_fill_replay_qty(event)
        if not event_symbol or action not in {"increase", "decrease"}:
            return None
        if qty is None or qty <= 0.0:
            return None
        if action == "increase":
            running_sizes[event_symbol] = (
                running_sizes.get(event_symbol, 0.0) + float(qty)
            )
        else:
            running_sizes[event_symbol] = max(
                0.0, running_sizes.get(event_symbol, 0.0) - float(qty)
            )
        if not any(size > flat_epsilon for size in running_sizes.values()):
            return _equity_hard_stop_fill_timestamp_ms(event)
    return None


def _equity_hard_stop_defer_missing_flatten_fill(
    self,
    pside: str,
    now_ms: int,
    *,
    symbol: Optional[str] = None,
) -> None:
    """Keep the scope protective until its episode-end fill is available."""
    state = (
        self._hsl_coin_state(pside, symbol)
        if symbol is not None
        else self._hsl_state(pside)
    )
    state["pending_stop_event"] = None
    state["red_flat_confirmations"] = 0
    last_log_ms = int(state.get("last_missing_flatten_fill_log_ms", 0) or 0)
    if (
        last_log_ms != 0
        and int(now_ms) - last_log_ms < self._equity_hard_stop_cooldown_log_interval_ms
    ):
        return
    state["last_missing_flatten_fill_log_ms"] = int(now_ms)
    scope = f"{pside}:{symbol}" if symbol is not None else pside
    logging.error(
        "[risk] HSL[%s] scope is flat but its flatten-fill evidence is unavailable; "
        "episode finalization and cooldown anchoring deferred",
        scope,
    )


async def _equity_hard_stop_flatten_fill_timestamp_with_refresh(
    self,
    pside: str,
    now_ms: int,
    *,
    symbol: Optional[str] = None,
    since_ms: Optional[int],
    replay_start_sizes: Optional[dict[str, float]] = None,
) -> Optional[int]:
    """Return episode-bounded flatten evidence, refreshing fills when absent.

    RED supervision deliberately uses a small protective state refresh which
    excludes fill history.  Once the exchange reports the scope flat, perform
    a rate-limited refresh through the normal single-flight fill path so a
    just-arrived closing fill cannot leave the awaited supervisor stuck.
    """
    if since_ms is None:
        self._equity_hard_stop_defer_missing_flatten_fill(
            pside, now_ms, symbol=symbol
        )
        return None
    stop_ts_ms = self._equity_hard_stop_latest_flatten_fill_timestamp_optional_ms(
        pside,
        symbol=symbol,
        since_ms=int(since_ms),
        replay_start_sizes=replay_start_sizes,
    )
    if stop_ts_ms is not None:
        return int(stop_ts_ms)
    state = (
        self._hsl_coin_state(pside, symbol)
        if symbol is not None
        else self._hsl_state(pside)
    )
    refresh_clock_ms = int(time.monotonic() * 1000.0)
    last_refresh_ms = int(
        state.get("last_missing_flatten_fill_refresh_ms", 0) or 0
    )
    if last_refresh_ms == 0 or refresh_clock_ms - last_refresh_ms >= int(
        _HSL_FLATTEN_FILL_REFRESH_INTERVAL_MS
    ):
        state["last_missing_flatten_fill_refresh_ms"] = refresh_clock_ms
        refresh_ready = False
        try:
            refresh_ready = bool(
                await self.update_pnls(
                    source="hsl_flatten_confirmation",
                    since_ms=int(since_ms),
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logging.warning(
                "[risk] HSL[%s%s] flatten-fill refresh failed; keeping scope protective | "
                "error_type=%s",
                pside,
                f":{symbol}" if symbol is not None else "",
                _bounded_hsl_exception_type(exc),
            )
        if refresh_ready:
            stop_ts_ms = self._equity_hard_stop_latest_flatten_fill_timestamp_optional_ms(
                pside,
                symbol=symbol,
                since_ms=int(since_ms),
                replay_start_sizes=replay_start_sizes,
            )
    if stop_ts_ms is None:
        self._equity_hard_stop_defer_missing_flatten_fill(
            pside, now_ms, symbol=symbol
        )
        return None
    state["last_missing_flatten_fill_log_ms"] = 0
    state["last_missing_flatten_fill_refresh_ms"] = 0
    return int(stop_ts_ms)


def _equity_hard_stop_latest_panic_fill_timestamp_optional_ms(
    self,
    pside: str,
    *,
    symbol: Optional[str] = None,
    since_ms: Optional[int] = None,
) -> Optional[int]:
    latest_ts: Optional[int] = None
    if self._pnls_manager is not None:
        for event in self._pnls_manager.get_events():
            if _equity_hard_stop_fill_pside(event) != pside:
                continue
            if symbol is not None and _equity_hard_stop_fill_symbol(event) != symbol:
                continue
            pb_type = str(_equity_hard_stop_event_value(event, "pb_order_type", "") or "").lower()
            if "panic" not in pb_type or "close" not in pb_type:
                continue
            event_ts = _equity_hard_stop_fill_timestamp_ms(event)
            if since_ms is not None and event_ts < int(since_ms):
                continue
            latest_ts = event_ts if latest_ts is None else max(latest_ts, event_ts)
    return latest_ts


async def _calc_upnl_sum_strict(self, pside: Optional[str] = None, symbol: Optional[str] = None) -> float:
    if not self.fetched_positions:
        return 0.0
    symbols = {
        x["symbol"]
        for x in self.fetched_positions
        if (pside is None or x["position_side"] == pside)
        and (symbol is None or x["symbol"] == symbol)
    }
    if not symbols:
        return 0.0
    if hasattr(self, "_get_live_last_prices"):
        last_prices = await self._get_live_last_prices(
            symbols, max_age_ms=60_000, context="hard_stop_upnl"
        )
    else:
        last_prices = await self.cm.get_last_prices(symbols, max_age_ms=60_000)
    upnl_sum = 0.0
    for elm in self.fetched_positions:
        if pside is not None and elm["position_side"] != pside:
            continue
        if symbol is not None and elm["symbol"] != symbol:
            continue
        pos_symbol = elm["symbol"]
        if pos_symbol not in last_prices:
            raise RuntimeError(f"missing last price for {pos_symbol} while evaluating hard stop")
        upnl = _calc_hsl_pnl(
            elm["position_side"],
            elm["price"],
            last_prices[pos_symbol],
            elm["size"],
            self.c_mults[pos_symbol],
        )
        if not math.isfinite(upnl):
            raise RuntimeError(
                f"non-finite upnl for {pos_symbol} {elm['position_side']} while evaluating hard stop"
            )
        upnl_sum += upnl
    return upnl_sum


def _equity_hard_stop_fee_cost(fill: Any) -> float:
    if fill is None:
        return 0.0
    if isinstance(fill, dict):
        return signed_fee_paid_from_payload(fill)
    fee_paid = getattr(fill, "fee_paid", None)
    if fee_paid is not None:
        return float(fee_paid or 0.0)
    return signed_fee_paid_from_payload({"fees": getattr(fill, "fees", None)})


def _get_exchange_fee_rates(self, symbol: str) -> tuple[float, float]:
    market = self.markets_dict[symbol]
    maker_fee = market.get("maker_fee")
    if maker_fee is None:
        maker_fee = market.get("maker")
    taker_fee = market.get("taker_fee")
    if taker_fee is None:
        taker_fee = market.get("taker")
    if maker_fee is None:
        raise ValueError(f"missing maker_fee for {symbol}")
    if taker_fee is None:
        raise ValueError(f"missing taker_fee for {symbol}")
    maker_fee = float(maker_fee)
    taker_fee = float(taker_fee)
    if not math.isfinite(maker_fee):
        raise ValueError(f"maker_fee must be finite for {symbol}, got {maker_fee}")
    if not math.isfinite(taker_fee):
        raise ValueError(f"taker_fee must be finite for {symbol}, got {taker_fee}")
    return maker_fee, taker_fee


def _orchestrator_exchange_params(self, symbol: str) -> dict:
    maker_fee, taker_fee = self._get_exchange_fee_rates(symbol)
    return {
        "qty_step": float(self.qty_steps[symbol]),
        "price_step": float(self.price_steps[symbol]),
        "min_qty": float(self.min_qtys[symbol]),
        "min_cost": float(self.min_costs[symbol]),
        "c_mult": float(self.c_mults[symbol]),
        "maker_fee": float(maker_fee),
        "taker_fee": float(taker_fee),
    }


def _equity_hard_stop_coverage_allow_incomplete(
    self, pside: str, symbol: Optional[str] = None
) -> bool:
    """B2.1 incomplete-history policy (#1122).

    Coverage-category failures on HSL PnL inputs may be waived only when:
    - the explicit per-run operator override is active
      (`live.hsl_accept_incomplete_history`), or
    - coin `restart_after_red_policy=always` AND canonical HSL readiness proves
      every held episode plus the flat-scope cooldown horizon (the `always`
      policy ignores older no-restart evidence). `threshold`/`never` always
      require full configured lookback coverage; pside/unified scopes stay
      strict.
    """
    config = getattr(self, "config", None)
    if isinstance(config, dict):
        live_cfg = config.get("live")
        if isinstance(live_cfg, dict) and bool(
            live_cfg.get("hsl_accept_incomplete_history", False)
        ):
            return True
    if symbol is None or self._pnls_manager is None:
        return False
    hsl_cfg = getattr(self, "hsl", None)
    if not isinstance(hsl_cfg, dict) or pside not in hsl_cfg:
        return False
    effective_hsl_cfg = _equity_hard_stop_config(self, pside, symbol)
    policy = normalize_hsl_restart_after_red_policy(
        effective_hsl_cfg.get("restart_after_red_policy", "threshold"),
        path="hsl.restart_after_red_policy",
    )
    if policy != "always":
        return False
    now_ms = int(self.get_exchange_time())
    lookback = parse_pnls_max_lookback_days(
        self.live_value("pnls_max_lookback_days"),
        field_name="live.pnls_max_lookback_days",
    )
    required, start_ms = self._equity_hard_stop_required_fill_history_start_ms(
        now_ms,
        pnl_start_ms=lookback.fill_cache_age_limit_ms(now_ms),
    )
    if not required:
        return True
    coverage = self._fill_history_coverage_status(
        start_ms=start_ms,
        end_ms=now_ms,
    )
    return bool(coverage.get("ready", False))


def _equity_hard_stop_realized_pnl_now(self, pside: Optional[str] = None) -> float:
    if self._pnls_manager is None:
        return 0.0
    realized = 0.0
    start_ms = self._pnls_lookback_start_ms()
    events = (
        self._pnls_manager.get_events()
        if start_ms is None
        else self._pnls_manager.get_events(start_ms=start_ms)
    )
    if pside is not None:
        events = [
            event for event in events if _equity_hard_stop_fill_pside(event) == pside
        ]
    self._assert_pnl_history_safe_for_risk(
        events,
        context="equity hard stop realized PnL",
        start_ms=start_ms,
        allow_incomplete=self._equity_hard_stop_coverage_allow_incomplete(
            pside if pside is not None else "long"
        ),
    )
    for event in events:
        realized += float(getattr(event, "pnl", 0.0) or 0.0)
        realized += _equity_hard_stop_fee_cost(event)
    return realized


def _equity_hard_stop_coin_realized_pnl_peak_last(
    self, pside: str, symbol: str, timestamp_ms: int, reset_timestamp_ms: Optional[int] = None
) -> tuple[float, float]:
    if self._pnls_manager is None:
        return 0.0, 0.0
    lookback_ms = self._equity_hard_stop_lookback_ms()
    start_ms = None if lookback_ms is None else int(timestamp_ms) - int(lookback_ms)
    if reset_timestamp_ms is not None:
        start_ms = int(reset_timestamp_ms) if start_ms is None else max(start_ms, int(reset_timestamp_ms))
    events = []
    for event in self._pnls_manager.get_events():
        if _equity_hard_stop_fill_pside(event) != pside:
            continue
        if _equity_hard_stop_fill_symbol(event) != symbol:
            continue
        event_ts = _equity_hard_stop_fill_timestamp_ms(event)
        if start_ms is not None and event_ts < start_ms:
            continue
        events.append(event)
    self._assert_pnl_history_safe_for_risk(
        events,
        context="coin HSL realized PnL",
        start_ms=start_ms,
        allow_incomplete=self._equity_hard_stop_coverage_allow_incomplete(
            pside, symbol
        ),
    )
    events.sort(key=_equity_hard_stop_fill_timestamp_ms)
    current = 0.0
    peak = 0.0
    for event in events:
        current += float(_equity_hard_stop_event_value(event, "pnl", 0.0) or 0.0)
        current += _equity_hard_stop_fee_cost(event)
        peak = max(peak, current)
    return float(peak), float(current)


def _equity_hard_stop_lookback_ms(self) -> int | None:
    lookback = parse_pnls_max_lookback_days(
        require_live_value(self.config, "pnls_max_lookback_days"),
        field_name="live.pnls_max_lookback_days",
    )
    return lookback.hsl_window_ms()


def _equity_hard_stop_apply_sample(
    self,
    pside: str,
    timestamp_ms: int,
    balance: float,
    realized_pnl_total: float,
    realized_pnl_pside: float,
    unrealized_pnl_pside: float,
    unrealized_pnl_total: Optional[float] = None,
    *,
    latch_red: bool = True,
) -> dict:
    if not math.isfinite(balance) or balance <= 0.0:
        raise ValueError(f"balance must be finite and > 0, got {balance}")
    if not math.isfinite(realized_pnl_total):
        raise ValueError(f"realized_pnl_total must be finite, got {realized_pnl_total}")
    if not math.isfinite(realized_pnl_pside):
        raise ValueError(f"realized_pnl_pside must be finite, got {realized_pnl_pside}")
    if not math.isfinite(unrealized_pnl_pside):
        raise ValueError(f"unrealized_pnl_pside must be finite, got {unrealized_pnl_pside}")

    state = self._hsl_state(pside)
    last_metrics = state["last_metrics"]
    current_minute = int(timestamp_ms) // 60_000

    signal_mode, realized_pnl_signal, unrealized_pnl_signal = self._equity_hard_stop_signal_values(
        pside,
        realized_pnl_total=realized_pnl_total,
        realized_pnl_pside=realized_pnl_pside,
        unrealized_pnl_pside=unrealized_pnl_pside,
        unrealized_pnl_total=unrealized_pnl_total,
    )
    if last_metrics is not None and int(last_metrics["timestamp_ms"]) // 60_000 == current_minute:
        same_inputs = (
            str(last_metrics.get("signal_mode")) == str(signal_mode)
            and float(last_metrics.get("balance", 0.0)) == float(balance)
            and float(last_metrics.get("realized_pnl_total", 0.0)) == float(realized_pnl_total)
            and float(last_metrics.get("realized_pnl", 0.0)) == float(realized_pnl_signal)
            and float(last_metrics.get("unrealized_pnl", 0.0)) == float(unrealized_pnl_signal)
        )
        needs_latching_replay_red = (
            bool(latch_red)
            and str(last_metrics.get("tier")) == "red"
            and not self._equity_hard_stop_runtime_red_latched(pside)
        )
        if same_inputs and not needs_latching_replay_red:
            cached = dict(last_metrics)
            cached["changed"] = False
            cached["elapsed_minutes"] = 0
            state["last_metrics"] = cached
            return cached
    cfg = self.hsl[pside]
    lookback_ms = self._equity_hard_stop_lookback_ms()
    prev_tier = self._equity_hard_stop_runtime_tier(pside)
    red_threshold = float(cfg["red_threshold"])
    ratio_yellow = float(cfg["tier_ratios"]["yellow"])
    ratio_orange = float(cfg["tier_ratios"]["orange"])
    ema_span_minutes = float(cfg["ema_span_minutes"])
    strategy_pnl = realized_pnl_signal + unrealized_pnl_signal
    peak_strategy_pnl = float(
        state["strategy_pnl_peak"].update(
            int(timestamp_ms),
            float(strategy_pnl),
            int(lookback_ms) if lookback_ms is not None else (2**64 - 1),
        )
    )
    baseline_balance = balance - realized_pnl_total
    strategy_equity = max(float(baseline_balance + strategy_pnl), 1e-12)
    peak_strategy_equity = max(
        float(strategy_equity),
        float(max(baseline_balance + peak_strategy_pnl, 1e-12)),
    )
    step = state["runtime"].apply_sample(
        timestamp_ms=int(timestamp_ms),
        equity=float(strategy_equity),
        peak_strategy_equity=float(peak_strategy_equity),
        red_threshold=red_threshold,
        ema_span_minutes=ema_span_minutes,
        tier_ratio_yellow=ratio_yellow,
        tier_ratio_orange=ratio_orange,
        latch_red=bool(latch_red),
    )
    if not isinstance(step, dict):
        raise TypeError(
            "passivbot_rust.EquityHardStopRuntime.apply_sample() must return a dict, "
            f"got {type(step).__name__}"
        )

    metrics = {
        "pside": pside,
        "signal_mode": signal_mode,
        "timestamp_ms": int(timestamp_ms),
        "balance": float(balance),
        "realized_pnl_total": float(realized_pnl_total),
        "realized_pnl": float(realized_pnl_signal),
        "unrealized_pnl": float(unrealized_pnl_signal),
        "strategy_pnl": float(strategy_pnl),
        "peak_strategy_pnl": float(peak_strategy_pnl),
        "baseline_balance": float(baseline_balance),
        "strategy_equity": float(strategy_equity),
        "equity": float(strategy_equity),
        "peak_strategy_equity": float(step["peak_strategy_equity"]),
        "rolling_peak_strategy_equity": float(step["rolling_peak_strategy_equity"]),
        "drawdown_raw": float(step["drawdown_raw"]),
        "drawdown_ema": float(step["drawdown_ema"]),
        "drawdown_score": float(step["drawdown_score"]),
        "red_threshold": red_threshold,
        "tier": str(step["tier"]),
        "red_active_now": bool(step["red_active_now"]),
        "red_seen_in_episode": bool(step["red_seen_in_episode"]),
        "changed": bool(step["changed"]) or str(step["tier"]) != prev_tier,
        "alpha": float(step["alpha"]),
        "elapsed_minutes": int(step["elapsed_minutes"]),
    }
    state["last_metrics"] = metrics
    return metrics


def _equity_hard_stop_apply_coin_sample(
    self,
    pside: str,
    symbol: str,
    timestamp_ms: int,
    balance: float,
    current_upnl: float,
    *,
    latch_red: bool = True,
) -> dict:
    peak_realized, last_realized = self._equity_hard_stop_coin_realized_pnl_peak_last(
        pside,
        symbol,
        int(timestamp_ms),
        reset_timestamp_ms=self._hsl_coin_state(pside, symbol).get("pnl_reset_timestamp_ms"),
    )
    return self._equity_hard_stop_apply_coin_metrics_sample(
        pside,
        symbol,
        timestamp_ms,
        balance,
        peak_realized,
        last_realized,
        current_upnl,
        latch_red=latch_red,
    )


def _equity_hard_stop_apply_coin_metrics_sample(
    self,
    pside: str,
    symbol: str,
    timestamp_ms: int,
    balance: float,
    peak_realized: float,
    last_realized: float,
    current_upnl: float,
    *,
    latch_red: bool = True,
) -> dict:
    if not math.isfinite(balance) or balance <= 0.0:
        raise ValueError(f"balance must be finite and > 0, got {balance}")
    if not math.isfinite(peak_realized):
        raise ValueError(f"peak_realized must be finite, got {peak_realized}")
    if not math.isfinite(last_realized):
        raise ValueError(f"last_realized must be finite, got {last_realized}")
    if not math.isfinite(current_upnl):
        raise ValueError(f"current_upnl must be finite, got {current_upnl}")
    state = self._hsl_coin_state(pside, symbol)
    last_metrics = state["last_metrics"]
    current_minute = int(timestamp_ms) // 60_000
    cfg = _equity_hard_stop_config(self, pside, symbol)
    red_threshold = float(cfg["red_threshold"])
    ratio_yellow = float(cfg["tier_ratios"]["yellow"])
    ratio_orange = float(cfg["tier_ratios"]["orange"])
    ema_span_minutes = float(cfg["ema_span_minutes"])
    n_positions_raw = float(self.bot_value(pside, "n_positions"))
    if not math.isfinite(n_positions_raw) or n_positions_raw <= 0.0:
        raise ValueError(
            f"coin HSL n_positions must be finite and > 0 for {symbol} {pside}, "
            f"got {n_positions_raw}"
        )
    n_positions = int(round(n_positions_raw))
    if n_positions <= 0:
        raise ValueError(
            f"coin HSL n_positions must round to > 0 for {symbol} {pside}, got {n_positions_raw}"
        )
    signal = pbr.hsl_coin_drawdown_signal(
        balance=float(balance),
        n_positions=n_positions,
        peak_realized=float(peak_realized),
        last_realized=float(last_realized),
        current_upnl=float(current_upnl),
    )
    if not isinstance(signal, dict):
        raise TypeError(
            "passivbot_rust.hsl_coin_drawdown_signal() must return a dict, "
            f"got {type(signal).__name__}"
        )
    slot_budget = float(signal["slot_budget"])
    drawdown_usd = float(signal["drawdown_usd"])
    drawdown_ratio = float(signal["drawdown_raw"])
    if last_metrics is not None and int(last_metrics["timestamp_ms"]) // 60_000 == current_minute:
        same_inputs = (
            str(last_metrics.get("signal_mode")) == "coin"
            and str(last_metrics.get("symbol")) == symbol
            and float(last_metrics.get("balance", 0.0)) == float(balance)
            and float(last_metrics.get("peak_realized_pnl", 0.0)) == float(peak_realized)
            and float(last_metrics.get("realized_pnl", 0.0)) == float(last_realized)
            and float(last_metrics.get("unrealized_pnl", 0.0)) == float(current_upnl)
        )
        needs_latching_replay_red = (
            bool(latch_red)
            and str(last_metrics.get("tier")) == "red"
            and not bool(state["runtime"].red_latched())
        )
        if same_inputs and not needs_latching_replay_red:
            cached = dict(last_metrics)
            cached["changed"] = False
            cached["elapsed_minutes"] = 0
            state["last_metrics"] = cached
            return cached
    prev_tier = str(state["runtime"].tier())
    synthetic_equity = max(1.0 - drawdown_ratio, 1e-12)
    step = state["runtime"].apply_sample(
        timestamp_ms=int(timestamp_ms),
        equity=float(synthetic_equity),
        peak_strategy_equity=1.0,
        red_threshold=red_threshold,
        ema_span_minutes=ema_span_minutes,
        tier_ratio_yellow=ratio_yellow,
        tier_ratio_orange=ratio_orange,
        latch_red=bool(latch_red),
    )
    if not isinstance(step, dict):
        raise TypeError(
            "passivbot_rust.EquityHardStopRuntime.apply_sample() must return a dict, "
            f"got {type(step).__name__}"
        )
    metrics = {
        "pside": pside,
        "symbol": symbol,
        "signal_mode": "coin",
        "timestamp_ms": int(timestamp_ms),
        "balance": float(balance),
        "slot_budget": float(slot_budget),
        "peak_realized_pnl": float(peak_realized),
        "realized_pnl": float(last_realized),
        "unrealized_pnl": float(current_upnl),
        "strategy_pnl": float(last_realized + current_upnl),
        "peak_strategy_pnl": float(peak_realized),
        "baseline_balance": float(balance),
        "strategy_equity": float(synthetic_equity),
        "equity": float(synthetic_equity),
        "peak_strategy_equity": 1.0,
        "rolling_peak_strategy_equity": 1.0,
        "drawdown_usd": float(drawdown_usd),
        "drawdown_raw": float(step["drawdown_raw"]),
        "drawdown_ema": float(step["drawdown_ema"]),
        "drawdown_score": float(step["drawdown_score"]),
        "red_threshold": red_threshold,
        "tier_ratio_yellow": ratio_yellow,
        "tier_ratio_orange": ratio_orange,
        "tier": str(step["tier"]),
        "red_active_now": bool(step["red_active_now"]),
        "red_seen_in_episode": bool(step["red_seen_in_episode"]),
        "changed": bool(step["changed"]) or str(step["tier"]) != prev_tier,
        "alpha": float(step["alpha"]),
        "elapsed_minutes": int(step["elapsed_minutes"]),
    }
    state["last_metrics"] = metrics
    return metrics


def _equity_hard_stop_history_coin_value(
    row: dict,
    key: str,
    symbol: str,
    pside: str,
    *,
    require_key: bool = False,
    require_value: bool = False,
) -> float:
    if key not in row or row[key] is None:
        if require_key or require_value:
            raise ValueError(
                f"get_balance_equity_history()['timeline'][] missing required coin HSL key: {key}"
            )
        return 0.0
    by_symbol = row[key]
    if not isinstance(by_symbol, dict):
        raise TypeError(
            f"get_balance_equity_history()['timeline'][]['{key}'] must be a dict, "
            f"got {type(by_symbol).__name__}"
        )
    if symbol not in by_symbol or by_symbol[symbol] is None:
        if require_value:
            raise ValueError(
                f"get_balance_equity_history()['timeline'][]['{key}'] missing required "
                f"coin HSL symbol: {symbol}"
            )
        return 0.0
    by_pside = by_symbol[symbol]
    if not isinstance(by_pside, dict):
        raise TypeError(
            f"get_balance_equity_history()['timeline'][]['{key}'][{symbol!r}] must be a dict, "
            f"got {type(by_pside).__name__}"
        )
    if pside not in by_pside or by_pside[pside] is None:
        if require_value:
            raise ValueError(
                f"get_balance_equity_history()['timeline'][]['{key}'][{symbol!r}] missing "
                f"required coin HSL pside: {pside}"
            )
        return 0.0
    value = float(by_pside[pside])
    if not math.isfinite(value):
        raise ValueError(
            f"get_balance_equity_history()['timeline'][]['{key}'][{symbol!r}][{pside!r}] "
            f"must be finite, got {value}"
        )
    return value


def _equity_hard_stop_history_coin_has_value(row: dict, key: str, symbol: str, pside: str) -> bool:
    if key not in row or row[key] is None:
        return False
    by_symbol = row[key]
    if not isinstance(by_symbol, dict):
        raise TypeError(
            f"get_balance_equity_history()['timeline'][]['{key}'] must be a dict, "
            f"got {type(by_symbol).__name__}"
        )
    if symbol not in by_symbol or by_symbol[symbol] is None:
        return False
    by_pside = by_symbol[symbol]
    if not isinstance(by_pside, dict):
        raise TypeError(
            f"get_balance_equity_history()['timeline'][]['{key}'][{symbol!r}] must be a dict, "
            f"got {type(by_pside).__name__}"
        )
    return pside in by_pside and by_pside[pside] is not None


def _equity_hard_stop_fill_replay_qty(fill: Any) -> Optional[float]:
    for key in ("qty", "amount", "size", "contracts"):
        raw = _equity_hard_stop_event_value(fill, key)
        if raw is None:
            continue
        try:
            qty = abs(float(raw))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(qty):
            return None
        return qty
    return None


def _equity_hard_stop_index_coin_fill_events(
    fill_events: list[Any],
) -> dict[tuple[str, str], list[Any]]:
    """Group fills by coin HSL scope without changing their source order."""
    indexed: dict[tuple[str, str], list[Any]] = {}
    for event in fill_events:
        pside_alias = _equity_hard_stop_event_value(event, "pside")
        position_side_alias = _equity_hard_stop_event_value(event, "position_side")
        if (
            pside_alias is not None
            and position_side_alias is not None
            and str(pside_alias).lower() != str(position_side_alias).lower()
        ):
            raise ValueError(
                "coin HSL fill has conflicting pside aliases: "
                f"pside={pside_alias!r} position_side={position_side_alias!r}"
            )
        pair = (
            _equity_hard_stop_fill_pside(event),
            _equity_hard_stop_fill_symbol(event),
        )
        indexed.setdefault(pair, []).append(event)
    return indexed


def _hsl_compact_sparse_replay_indices(
    timestamps: Any,
    balances: Any,
    realized_values: Any,
    unrealized_values: Any,
    *,
    lookback_ms: Optional[int],
    boundary_timestamps: tuple[int, ...] = (),
) -> Any:
    """Select exact change-point rows for compact coin-HSL metric stepping.

    Rust advances EMA state across elapsed constant-input minutes exactly. Run
    endpoints preserve the dense rolling-window timestamp semantics; expiry
    boundaries preserve changes caused only by the configured lookback.
    """
    import numpy as np

    ts = np.asarray(timestamps, dtype=np.int64)
    balance = np.asarray(balances, dtype=np.float64)
    if ts.ndim != 1 or balance.ndim != 1 or len(ts) != len(balance):
        raise ValueError("compact sparse replay timestamps/balances must be equal 1D arrays")
    row_count = len(ts)
    if row_count == 0:
        return np.empty(0, dtype=np.int64)

    selected = np.zeros(row_count, dtype=bool)

    def mark_run_boundaries(values: Any) -> Any:
        if values is None:
            return np.empty(0, dtype=np.int64)
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1 or len(arr) != row_count:
            raise ValueError("compact sparse replay values must match timestamp length")
        if row_count == 1:
            starts = np.array([0], dtype=np.int64)
        else:
            finite = np.isfinite(arr)
            same = (finite[1:] == finite[:-1]) & (
                (~finite[1:]) | (arr[1:] == arr[:-1])
            )
            starts = np.concatenate(
                (
                    np.array([0], dtype=np.int64),
                    np.flatnonzero(~same).astype(np.int64) + 1,
                )
            )
        ends = np.concatenate((starts[1:] - 1, np.array([row_count - 1])))
        selected[starts] = True
        selected[ends] = True
        return ends

    mark_run_boundaries(balance)
    realized_run_ends = mark_run_boundaries(realized_values)
    mark_run_boundaries(unrealized_values)

    if lookback_ms is not None:
        lookback_ms = int(lookback_ms)
        for run_end in realized_run_ends:
            expiry_idx = int(
                np.searchsorted(
                    ts,
                    int(ts[int(run_end)]) + lookback_ms,
                    side="right",
                )
            )
            if expiry_idx < row_count:
                selected[expiry_idx] = True
                if expiry_idx > 0:
                    selected[expiry_idx - 1] = True

    for boundary_ts in boundary_timestamps:
        boundary_idx = int(np.searchsorted(ts, int(boundary_ts), side="left"))
        if boundary_idx < row_count:
            selected[boundary_idx] = True
        if boundary_idx > 0:
            selected[boundary_idx - 1] = True

    selected[-1] = True
    return np.flatnonzero(selected).astype(np.int64)


def _equity_hard_stop_coin_replay_events(
    fill_events: list[Any], pside: str, symbol: str, *, qty_step: float = 0.0
) -> tuple[list[tuple[int, str, float, float]], bool]:
    replay_events: list[tuple[int, str, float, float]] = []
    ambiguous = False
    replay_size = 0.0
    flat_epsilon = _hsl_flat_epsilon(qty_step)
    for event in fill_events:
        if _equity_hard_stop_fill_pside(event) != pside:
            continue
        if _equity_hard_stop_fill_symbol(event) != symbol:
            continue
        action = _equity_hard_stop_fill_action(event)
        qty = _equity_hard_stop_fill_replay_qty(event)
        if action not in {"increase", "decrease"} or qty is None or qty <= 0.0:
            ambiguous = True
            continue
        try:
            realized_delta = float(
                _equity_hard_stop_event_value(event, "pnl", 0.0) or 0.0
            ) + float(_equity_hard_stop_fee_cost(event))
        except (TypeError, ValueError):
            ambiguous = True
            continue
        if not math.isfinite(realized_delta):
            ambiguous = True
            continue
        replay_events.append(
            (
                _equity_hard_stop_fill_timestamp_ms(event),
                action,
                float(qty),
                realized_delta,
            )
        )
    replay_events.sort(key=lambda item: item[0])
    for _event_ts, action, qty, _realized_delta in replay_events:
        if action == "increase":
            replay_size += qty
        else:
            if qty > replay_size + flat_epsilon:
                ambiguous = True
            replay_size = max(0.0, replay_size - qty)
    return replay_events, ambiguous


def _equity_hard_stop_coin_bounded_required_replay_start_ts(
    self,
    pside: str,
    symbol: str,
    fill_events: list[Any],
) -> Optional[int]:
    """Return the earliest episode still relevant to an open ``always`` scope.

    Coin HSL resets at every proven flatten. With ``restart_after_red_policy=always``,
    episodes before the current one matter only while a possible RED cooldown can
    chain across the intervening flat periods. Walk proven fill-derived episodes
    backward from the authoritative current position and retain preceding episodes
    only while each flatten falls inside the next episode's cooldown horizon.

    ``None`` means the current episode boundary is not provable. Callers must then
    preserve the strict full-lookback replay rather than guessing a boundary.
    """
    qty_step = _hsl_qty_step_for_symbol(self, symbol)
    flat_epsilon = _hsl_flat_epsilon(qty_step)
    current_size = abs(
        float(
            (self.positions or {})
            .get(symbol, {})
            .get(pside, {})
            .get("size", 0.0)
            or 0.0
        )
    )
    if current_size <= flat_epsilon:
        return None
    replay_events, ambiguous = _equity_hard_stop_coin_replay_events(
        fill_events,
        pside,
        symbol,
        qty_step=qty_step,
    )
    if ambiguous:
        return None

    episodes: list[tuple[int, Optional[int]]] = []
    replay_size = 0.0
    episode_start_ts: Optional[int] = None
    for event_ts, action, qty, _realized_delta in replay_events:
        was_flat = replay_size <= flat_epsilon
        if action == "increase":
            replay_size += qty
            if was_flat and replay_size > flat_epsilon:
                episode_start_ts = int(event_ts)
        else:
            replay_size = max(0.0, replay_size - qty)
            if not was_flat and replay_size <= flat_epsilon:
                if episode_start_ts is None:
                    return None
                episodes.append((int(episode_start_ts), int(event_ts)))
                episode_start_ts = None

    size_tolerance = max(flat_epsilon, abs(current_size) * 1e-12, 1e-12)
    if abs(replay_size - current_size) > size_tolerance:
        return None
    if episode_start_ts is None:
        return None

    required_start_ts = int(episode_start_ts)
    cooldown_minutes = float(
        _equity_hard_stop_config(self, pside, symbol)["cooldown_minutes_after_red"]
    )
    cooldown_ms = (
        int(round(cooldown_minutes * 60_000.0))
        if cooldown_minutes > 0.0
        else 0
    )
    if cooldown_ms <= 0:
        return required_start_ts

    next_episode_start_ts = required_start_ts
    for previous_start_ts, previous_flatten_ts in reversed(episodes):
        if previous_flatten_ts is None:
            return None
        if int(previous_flatten_ts) + cooldown_ms <= next_episode_start_ts:
            break
        required_start_ts = int(previous_start_ts)
        next_episode_start_ts = int(previous_start_ts)
    return required_start_ts


def _equity_hard_stop_required_fill_history_scope(
    self,
    now_ms: int,
    *,
    pnl_start_ms: Optional[int],
) -> tuple[bool, Optional[int], Optional[dict[tuple[str, str], int]]]:
    """Return aggregate coverage and exact coin-PnL boundaries.

    The pair map is present only when coin ``always`` scopes prove bounded
    current episodes. ``None`` means every PnL event in the aggregate window
    remains relevant.
    """
    if not self._equity_hard_stop_enabled():
        return False, None, {}
    if self._equity_hard_stop_signal_mode() != "coin":
        return True, pnl_start_ms, None
    manager = getattr(self, "_pnls_manager", None)
    if manager is None:
        return True, pnl_start_ms, None

    fill_events = [
        event
        for event in manager.get_events()
        if pnl_start_ms is None
        or _equity_hard_stop_fill_timestamp_ms(event) >= int(pnl_start_ms)
    ]
    if any(_equity_hard_stop_fill_pside_optional(event) is None for event in fill_events):
        return True, pnl_start_ms, None
    events_by_pair = _equity_hard_stop_index_coin_fill_events(fill_events)
    floor_ms = 0 if pnl_start_ms is None else max(0, int(pnl_start_ms))

    def clamp(value: int) -> int:
        return max(floor_ms, int(value))

    required_starts: list[int] = []
    max_cooldown_ms_by_pside: dict[str, int] = {}
    held_pairs: set[tuple[str, str]] = set()
    required_start_by_pair: dict[tuple[str, str], int] = {}
    override_symbols = sorted((getattr(self, "coin_overrides", {}) or {}).keys())

    for pside in self._hsl_psides():
        for scope_symbol in [None, *override_symbols]:
            cfg = _equity_hard_stop_config(self, pside, scope_symbol)
            if not bool(cfg["enabled"]):
                continue
            policy = normalize_hsl_restart_after_red_policy(
                cfg.get("restart_after_red_policy", "threshold"),
                path=(
                    f"hsl.{pside}.restart_after_red_policy"
                    if scope_symbol is None
                    else f"coin HSL {scope_symbol} {pside}.restart_after_red_policy"
                ),
            )
            if policy != "always":
                return True, pnl_start_ms, None
            cooldown_ms = max(
                0,
                int(round(float(cfg["cooldown_minutes_after_red"]) * 60_000.0)),
            )
            if cooldown_ms > 0:
                max_cooldown_ms_by_pside[pside] = max(
                    cooldown_ms,
                    max_cooldown_ms_by_pside.get(pside, 0),
                )
                required_starts.append(clamp(int(now_ms) - cooldown_ms))
        for symbol in sorted((self.positions or {}).keys()):
            symbol = str(symbol)
            if not self._equity_hard_stop_enabled(pside, symbol=symbol):
                continue
            if not self._equity_hard_stop_has_open_position_symbol(pside, symbol):
                continue
            pair = (pside, symbol)
            held_pairs.add(pair)
            bounded_start = _equity_hard_stop_coin_bounded_required_replay_start_ts(
                self,
                pside,
                symbol,
                events_by_pair.get(pair, []),
            )
            if bounded_start is None:
                return True, pnl_start_ms, None
            required_start_by_pair[pair] = clamp(bounded_start)
            required_starts.append(required_start_by_pair[pair])

    for event in fill_events:
        pside = _equity_hard_stop_fill_pside(event)
        if pside not in max_cooldown_ms_by_pside:
            continue
        symbol = _equity_hard_stop_fill_symbol(event)
        if not symbol:
            if _equity_hard_stop_fill_timestamp_ms(event) >= clamp(
                int(now_ms) - max_cooldown_ms_by_pside[pside]
            ):
                return True, pnl_start_ms, None
            continue
        cfg = _equity_hard_stop_config(self, pside, symbol)
        if not bool(cfg["enabled"]):
            continue
        cooldown_ms = max(
            0,
            int(round(float(cfg["cooldown_minutes_after_red"]) * 60_000.0)),
        )
        if cooldown_ms <= 0 or _equity_hard_stop_fill_timestamp_ms(event) < clamp(
            int(now_ms) - cooldown_ms
        ):
            continue
        if (pside, symbol) not in held_pairs:
            return True, pnl_start_ms, None

    if not required_starts:
        return False, None, {}
    return True, min(required_starts), required_start_by_pair


def _equity_hard_stop_required_fill_history_start_ms(
    self,
    now_ms: int,
    *,
    pnl_start_ms: Optional[int],
) -> tuple[bool, Optional[int]]:
    """Return the earliest fill needed by enabled HSL consumers."""
    required, start_ms, _pair_starts = _equity_hard_stop_required_fill_history_scope(
        self,
        now_ms,
        pnl_start_ms=pnl_start_ms,
    )
    return required, start_ms


def _equity_hard_stop_required_pnl_events(
    self,
    events: list[Any],
    now_ms: int,
    *,
    pnl_start_ms: Optional[int],
) -> list[Any]:
    """Select only PnL rows consumed by the canonical HSL replay scope."""
    required, start_ms, pair_starts = _equity_hard_stop_required_fill_history_scope(
        self,
        now_ms,
        pnl_start_ms=pnl_start_ms,
    )
    if not required:
        return []
    if pair_starts is None:
        if start_ms is None:
            return list(events)
        return [
            event
            for event in events
            if _equity_hard_stop_fill_timestamp_ms(event) >= int(start_ms)
        ]
    out: list[Any] = []
    for event in events:
        pside = _equity_hard_stop_fill_pside_optional(event)
        symbol = _equity_hard_stop_fill_symbol(event)
        pair_start_ms = pair_starts.get((pside, symbol)) if pside is not None else None
        if pair_start_ms is not None and _equity_hard_stop_fill_timestamp_ms(
            event
        ) >= int(pair_start_ms):
            out.append(event)
    return out


def _equity_hard_stop_coin_replay_size_at(
    replay_events: list[tuple[int, str, float, float]], row_ts_ms: int
) -> float:
    boundary_ts_ms = int(row_ts_ms) + 60_000
    size = 0.0
    for event_ts, action, qty, _realized_delta in replay_events:
        if int(event_ts) >= boundary_ts_ms:
            break
        if action == "increase":
            size += qty
        else:
            size = max(0.0, size - qty)
    return float(size)


def _equity_hard_stop_symbol_supported_for_coin_replay(self, symbol: str) -> bool:
    if symbol in (self.positions or {}):
        return True
    c_mults = getattr(self, "c_mults", None)
    if isinstance(c_mults, dict) and c_mults:
        return symbol in c_mults
    return True


def _equity_hard_stop_activate_coin_red_from_metrics(
    self,
    pside: str,
    symbol: str,
    metrics: dict,
) -> None:
    state = self._hsl_coin_state(pside, symbol)
    if state["pending_red_since_ms"] is None:
        state["pending_red_since_ms"] = int(metrics["timestamp_ms"])
    state["pending_stop_event"] = None
    self._equity_hard_stop_set_coin_runtime_forced_mode(pside, symbol, "panic")


def _equity_hard_stop_prime_coin_runtime_for_replay(
    self, pside: str, symbol: str, first_sample_ts_ms: int
) -> None:
    state = self._hsl_coin_state(pside, symbol)
    if state["runtime"].initialized():
        return
    cfg = _equity_hard_stop_config(self, pside, symbol)
    baseline_ts_ms = max(0, int(first_sample_ts_ms) - 60_000)
    step = state["runtime"].apply_sample(
        timestamp_ms=baseline_ts_ms,
        equity=1.0,
        peak_strategy_equity=1.0,
        red_threshold=float(cfg["red_threshold"]),
        ema_span_minutes=float(cfg["ema_span_minutes"]),
        tier_ratio_yellow=float(cfg["tier_ratios"]["yellow"]),
        tier_ratio_orange=float(cfg["tier_ratios"]["orange"]),
    )
    if not isinstance(step, dict):
        raise TypeError(
            "passivbot_rust.EquityHardStopRuntime.apply_sample() must return a dict, "
            f"got {type(step).__name__}"
        )


def _equity_hard_stop_log_transition(self, pside: str, metrics: dict, prev_tier: str) -> None:
    label = pside
    if metrics["signal_mode"] == "coin":
        label = f"{pside}:{metrics['symbol']}"
    cfg = _equity_hard_stop_config(self, pside, metrics.get("symbol"))
    logging.info(
        "[risk] HSL[%s] tier transition %s -> %s | balance=%.6f strategy_equity=%.6f "
        "peak_strategy_equity=%.6f drawdown_raw=%.6f drawdown_ema=%.6f drawdown_score=%.6f "
        "strategy_pnl=%.6f peak_strategy_pnl=%.6f "
        "red_threshold=%.6f yellow=%.3f orange=%.3f",
        label,
        prev_tier,
        metrics["tier"],
        metrics["balance"],
        metrics["strategy_equity"],
        metrics["peak_strategy_equity"],
        metrics["drawdown_raw"],
        metrics["drawdown_ema"],
        metrics["drawdown_score"],
        metrics["strategy_pnl"],
        metrics["peak_strategy_pnl"],
        metrics["red_threshold"],
        float(metrics.get("tier_ratio_yellow", cfg["tier_ratios"]["yellow"])),
        float(metrics.get("tier_ratio_orange", cfg["tier_ratios"]["orange"])),
    )
    _emit_hsl_event(
        self,
        "hsl.transition",
        ("hsl", "risk", "transition"),
        _hsl_event_data(
            metrics,
            {
                "previous_tier": prev_tier,
                "metrics": dict(metrics),
            },
        ),
        pside=pside,
        symbol=metrics.get("symbol") if metrics.get("signal_mode") == "coin" else None,
        ts=int(metrics.get("timestamp_ms", 0) or 0) or None,
        status="succeeded",
        reason_code=f"{prev_tier}_to_{metrics['tier']}",
    )


def _equity_hard_stop_maybe_emit_raw_red_pending(
    self,
    pside: str,
    metrics: dict,
    *,
    symbol: Optional[str] = None,
) -> None:
    """Emit a bounded diagnostic when raw HSL drawdown is red before EMA confirms."""
    try:
        red_threshold = float(metrics["red_threshold"])
        drawdown_raw = float(metrics["drawdown_raw"])
        drawdown_ema = float(metrics["drawdown_ema"])
        drawdown_score = float(metrics["drawdown_score"])
        if drawdown_raw < red_threshold or drawdown_score >= red_threshold:
            return
        state = self._hsl_coin_state(pside, symbol) if symbol else self._hsl_state(pside)
        now_ms = int(metrics["timestamp_ms"])
        last_event_ms = int(state.get("last_raw_red_pending_event_ms", 0) or 0)
        interval_ms = int(
            getattr(
                self,
                "_equity_hard_stop_status_log_interval_ms",
                15 * 60 * 1000,
            )
            or 0
        )
        if last_event_ms and interval_ms > 0 and now_ms - last_event_ms < interval_ms:
            return
        state["last_raw_red_pending_event_ms"] = now_ms
        _emit_hsl_event(
            self,
            EventTypes.HSL_RAW_RED_PENDING,
            ("hsl", "risk", "red", "pending"),
            {
                "signal_mode": str(metrics.get("signal_mode") or ""),
                "tier": str(metrics.get("tier") or ""),
                "timestamp_ms": now_ms,
                "red_threshold": red_threshold,
                "drawdown_raw": drawdown_raw,
                "drawdown_ema": drawdown_ema,
                "drawdown_score": drawdown_score,
                "dist_to_red": max(0.0, red_threshold - drawdown_score),
                "raw_excess": max(0.0, drawdown_raw - red_threshold),
                "ema_gap_to_red": max(0.0, red_threshold - drawdown_ema),
                "elapsed_minutes": int(metrics.get("elapsed_minutes", 0) or 0),
                "balance_override_active": bool(
                    getattr(self, "balance_override", None) is not None
                ),
            },
            pside=pside,
            symbol=symbol,
            ts=now_ms,
            level="warning",
            status="degraded",
            reason_code=ReasonCodes.HSL_RAW_RED_PENDING_EMA_CONFIRMATION,
        )
    except Exception as exc:
        logging.debug(
            "[event] failed to emit HSL raw-red pending event pside=%s symbol=%s: %s",
            pside,
            symbol,
            _bounded_hsl_exception_type(exc),
        )


def _equity_hard_stop_build_latch_payload(
    self,
    pside: str,
    *,
    symbol: Optional[str] = None,
    stop_event_timestamp_ms: int,
    balance: Optional[float] = None,
    realized_pnl_total: Optional[float] = None,
    realized_pnl: Optional[float] = None,
    unrealized_pnl: Optional[float] = None,
    strategy_pnl: Optional[float] = None,
    peak_strategy_pnl: Optional[float] = None,
    strategy_equity: float,
    peak_strategy_equity: float,
    trigger_peak_strategy_equity: float,
    drawdown_raw: float,
    drawdown_ema: float,
    drawdown_score: float,
    no_restart_latched: bool,
    cooldown_until_ms: Optional[int],
    no_restart_peak_strategy_equity: Optional[float] = None,
    no_restart_drawdown_raw: Optional[float] = None,
) -> dict:
    cfg = _equity_hard_stop_config(self, pside, symbol)
    return {
        "triggered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "exchange": str(self.exchange),
        "user": str(self.user),
        "position_side": pside,
        "symbol": None if symbol is None else str(symbol),
        "signal_mode": self._equity_hard_stop_signal_mode(),
        "tier": "red",
        "red_threshold": float(cfg["red_threshold"]),
        "ema_span_minutes": float(cfg["ema_span_minutes"]),
        "cooldown_minutes_after_red": float(cfg["cooldown_minutes_after_red"]),
        "no_restart_drawdown_threshold": float(cfg["no_restart_drawdown_threshold"]),
        "tier_ratios": {
            "yellow": float(cfg["tier_ratios"]["yellow"]),
            "orange": float(cfg["tier_ratios"]["orange"]),
        },
        "orange_tier_mode": str(cfg["orange_tier_mode"]),
        "panic_close_order_type": str(cfg["panic_close_order_type"]),
        "stop_event_timestamp_ms": int(stop_event_timestamp_ms),
        "balance": None if balance is None else float(balance),
        "realized_pnl_total": None if realized_pnl_total is None else float(realized_pnl_total),
        "realized_pnl": None if realized_pnl is None else float(realized_pnl),
        "unrealized_pnl": None if unrealized_pnl is None else float(unrealized_pnl),
        "strategy_pnl": None if strategy_pnl is None else float(strategy_pnl),
        "peak_strategy_pnl": None if peak_strategy_pnl is None else float(peak_strategy_pnl),
        "strategy_equity": float(strategy_equity),
        "equity": float(strategy_equity),
        "peak_strategy_equity": float(peak_strategy_equity),
        "trigger_peak_strategy_equity": float(trigger_peak_strategy_equity),
        "drawdown_raw": float(drawdown_raw),
        "no_restart_peak_strategy_equity": float(
            peak_strategy_equity
            if no_restart_peak_strategy_equity is None
            else no_restart_peak_strategy_equity
        ),
        "no_restart_drawdown_raw": float(
            drawdown_raw if no_restart_drawdown_raw is None else no_restart_drawdown_raw
        ),
        "drawdown_ema": float(drawdown_ema),
        "drawdown_score": float(drawdown_score),
        "no_restart_latched": bool(no_restart_latched),
        "auto_restart_eligible": bool(
            (not no_restart_latched) and float(cfg["cooldown_minutes_after_red"]) > 0.0
        ),
        "cooldown_until_ms": None if cooldown_until_ms is None else int(cooldown_until_ms),
    }


def _equity_hard_stop_red_episode_finalization(
    self,
    pside: str,
    stop_event: dict,
    stop_event_timestamp_ms: int,
    *,
    symbol: Optional[str] = None,
) -> dict[str, Any]:
    """Apply Rust-owned post-episode restart/cooldown policy math."""
    state = self._hsl_coin_state(pside, symbol) if symbol else self._hsl_state(pside)
    cfg = _equity_hard_stop_config(self, pside, symbol)
    policy = normalize_hsl_restart_after_red_policy(
        cfg.get("restart_after_red_policy", "threshold"),
        path="hsl.restart_after_red_policy",
    )
    result = pbr.hsl_red_episode_finalization(
        restart_after_red_policy=policy,
        stop_timestamp_ms=int(stop_event_timestamp_ms),
        stop_equity=float(stop_event["equity"]),
        stop_peak_strategy_equity=float(stop_event["peak_strategy_equity"]),
        previous_no_restart_peak_strategy_equity=float(
            state.get("no_restart_peak_strategy_equity", 0.0) or 0.0
        ),
        drawdown_ema=float(stop_event["drawdown_ema"]),
        red_threshold=float(cfg["red_threshold"]),
        no_restart_drawdown_threshold=float(cfg["no_restart_drawdown_threshold"]),
        cooldown_minutes_after_red=float(cfg["cooldown_minutes_after_red"]),
    )
    if not isinstance(result, dict):
        raise TypeError(
            "passivbot_rust.hsl_red_episode_finalization() must return a dict, "
            f"got {type(result).__name__}"
        )
    state["no_restart_peak_strategy_equity"] = float(
        result["no_restart_peak_strategy_equity"]
    )
    return result


async def _equity_hard_stop_compute_stop_event(self, pside: str, stop_event_ts_ms: int) -> dict:
    state = self._hsl_state(pside)
    balance = float(self.get_raw_balance())
    realized_pnl_total = float(self._equity_hard_stop_realized_pnl_now())
    realized_pnl_pside = float(self._equity_hard_stop_realized_pnl_now(pside))
    signal_mode = self._equity_hard_stop_signal_mode()
    unrealized_pnl_total = float(await self._calc_upnl_sum_strict()) if signal_mode == "unified" else None
    unrealized_pnl_pside = float(await self._calc_upnl_sum_strict(pside))
    _, realized_pnl, unrealized_pnl = self._equity_hard_stop_signal_values(
        pside,
        realized_pnl_total=realized_pnl_total,
        realized_pnl_pside=realized_pnl_pside,
        unrealized_pnl_pside=unrealized_pnl_pside,
        unrealized_pnl_total=unrealized_pnl_total,
    )
    strategy_pnl = realized_pnl + unrealized_pnl
    peak_strategy_pnl = float(
        max(strategy_pnl, (state["last_metrics"] or {}).get("peak_strategy_pnl", strategy_pnl))
    )
    baseline_balance = float(balance - realized_pnl_total)
    strategy_equity = float(max(baseline_balance + strategy_pnl, 1e-12))
    trigger_peak_strategy_equity = float(state["runtime"].peak_strategy_equity())
    peak_strategy_equity = float(max(strategy_equity, baseline_balance + peak_strategy_pnl, 1e-12))
    if not math.isfinite(trigger_peak_strategy_equity) or trigger_peak_strategy_equity <= 0.0:
        raise RuntimeError(
            f"invalid HSL[{pside}] trigger_peak_strategy_equity at stop finalization: {trigger_peak_strategy_equity}"
        )
    if not math.isfinite(peak_strategy_equity) or peak_strategy_equity <= 0.0:
        raise RuntimeError(
            f"invalid HSL[{pside}] rolling peak_strategy_equity at stop finalization: {peak_strategy_equity}"
        )
    drawdown_ema = float(state["runtime"].drawdown_ema())
    drawdown_raw = max(0.0, 1.0 - strategy_equity / max(peak_strategy_equity, 1e-12))
    return {
        "position_side": pside,
        "signal_mode": signal_mode,
        "stop_event_timestamp_ms": int(stop_event_ts_ms),
        "balance": balance,
        "realized_pnl_total": realized_pnl_total,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "strategy_pnl": strategy_pnl,
        "peak_strategy_pnl": peak_strategy_pnl,
        "strategy_equity": strategy_equity,
        "equity": strategy_equity,
        "peak_strategy_equity": peak_strategy_equity,
        "trigger_peak_strategy_equity": trigger_peak_strategy_equity,
        "drawdown_raw": drawdown_raw,
        "drawdown_ema": drawdown_ema,
        "drawdown_score": min(drawdown_raw, drawdown_ema),
    }


async def _equity_hard_stop_compute_coin_stop_event(
    self, pside: str, symbol: str, stop_event_ts_ms: int
) -> dict:
    state = self._hsl_coin_state(pside, symbol)
    metrics = state["last_metrics"]
    metrics_ts_ms = int((metrics or {}).get("timestamp_ms", 0) or 0)
    if metrics is None or metrics_ts_ms < int(stop_event_ts_ms):
        metrics = self._equity_hard_stop_apply_coin_sample(
            pside,
            symbol,
            int(stop_event_ts_ms),
            float(self.get_raw_balance()),
            float(await self._calc_upnl_sum_strict(pside, symbol)),
        )
    # If fill refresh lagged behind position refresh, the scope may already
    # have a newer flat sample. Reuse it while retaining the fill timestamp as
    # the episode/cooldown boundary; Rust runtimes reject time travel.
    return {
        "position_side": pside,
        "symbol": symbol,
        "signal_mode": "coin",
        "stop_event_timestamp_ms": int(stop_event_ts_ms),
        "balance": float(metrics["balance"]),
        "slot_budget": float(metrics["slot_budget"]),
        "realized_pnl_total": None,
        "realized_pnl": float(metrics["realized_pnl"]),
        "peak_realized_pnl": float(metrics["peak_realized_pnl"]),
        "unrealized_pnl": float(metrics["unrealized_pnl"]),
        "strategy_pnl": float(metrics["strategy_pnl"]),
        "peak_strategy_pnl": float(metrics["peak_strategy_pnl"]),
        "strategy_equity": float(metrics["strategy_equity"]),
        "equity": float(metrics["equity"]),
        "peak_strategy_equity": float(metrics["peak_strategy_equity"]),
        "trigger_peak_strategy_equity": float(metrics["peak_strategy_equity"]),
        "drawdown_raw": float(metrics["drawdown_raw"]),
        "drawdown_ema": float(metrics["drawdown_ema"]),
        "drawdown_score": float(metrics["drawdown_score"]),
    }


def _equity_hard_stop_log_cooldown_status(self, pside: str, now_ms: int) -> None:
    state = self._hsl_state(pside)
    cooldown_until_ms = state["cooldown_until_ms"]
    if cooldown_until_ms is None or now_ms >= cooldown_until_ms:
        return
    if (
        state["last_cooldown_log_ms"] != 0
        and now_ms - state["last_cooldown_log_ms"] < self._equity_hard_stop_cooldown_log_interval_ms
    ):
        return
    state["last_cooldown_log_ms"] = now_ms
    remaining_seconds = max(0.0, (cooldown_until_ms - now_ms) / 1000.0)
    logging.info(
        "[risk] HSL[%s] RED cooldown active | remaining_time=%s",
        pside,
        _equity_hard_stop_format_remaining_time(remaining_seconds),
    )
    _emit_hsl_event(
        self,
        "hsl.status",
        ("hsl", "risk", "status"),
        {
            "tier": "red",
            "cooldown_until_ms": int(cooldown_until_ms),
            "cooldown_remaining_seconds": float(remaining_seconds),
        },
        pside=pside,
        ts=now_ms,
        status="degraded",
        reason_code="cooldown_active",
    )


def _equity_hard_stop_position_symbols(self, pside: str) -> list[str]:
    out = []
    for symbol, position in self.positions.items():
        size = float(position.get(pside, {}).get("size", 0.0) or 0.0)
        if size != 0.0:
            out.append(symbol)
    return sorted(out)


async def _equity_hard_stop_refresh_cooldown_after_repanic(
    self, pside: str, now_ms: int
) -> bool:
    state = self._hsl_state(pside)
    cooldown_minutes = float(self.hsl[pside]["cooldown_minutes_after_red"])
    cooldown_ms = max(0, int(round(cooldown_minutes * 60_000.0))) if cooldown_minutes > 0.0 else 0
    repanic_since_ms = state.get("cooldown_repanic_since_ms")
    stop_ts_ms = await self._equity_hard_stop_flatten_fill_timestamp_with_refresh(
        pside,
        now_ms,
        since_ms=repanic_since_ms,
        replay_start_sizes=state.get("cooldown_repanic_start_sizes") or {},
    )
    if stop_ts_ms is None:
        return False
    state["last_missing_flatten_fill_log_ms"] = 0
    cooldown_until_ms = stop_ts_ms + cooldown_ms if cooldown_ms > 0 else None
    stop_event = await self._equity_hard_stop_compute_stop_event(pside, stop_ts_ms)
    payload = self._equity_hard_stop_build_latch_payload(
        pside,
        stop_event_timestamp_ms=stop_ts_ms,
        balance=float(stop_event["balance"]),
        realized_pnl_total=float(stop_event["realized_pnl_total"]),
        realized_pnl=float(stop_event["realized_pnl"]),
        unrealized_pnl=float(stop_event["unrealized_pnl"]),
        strategy_pnl=float(stop_event["strategy_pnl"]),
        peak_strategy_pnl=float(stop_event["peak_strategy_pnl"]),
        strategy_equity=float(stop_event["strategy_equity"]),
        peak_strategy_equity=float(stop_event["peak_strategy_equity"]),
        trigger_peak_strategy_equity=float(stop_event["trigger_peak_strategy_equity"]),
        drawdown_raw=float(stop_event["drawdown_raw"]),
        drawdown_ema=float(stop_event["drawdown_ema"]),
        drawdown_score=float(stop_event["drawdown_score"]),
        no_restart_latched=False,
        cooldown_until_ms=cooldown_until_ms,
    )
    state["last_stop_event"] = payload
    state["cooldown_until_ms"] = cooldown_until_ms
    state["cooldown_intervention_active"] = False
    state["cooldown_repanic_reset_pending"] = False
    state["cooldown_repanic_since_ms"] = None
    state["cooldown_repanic_start_sizes"] = None
    state["last_cooldown_intervention_log_ms"] = 0
    state["last_missing_flatten_fill_log_ms"] = 0
    state["last_missing_flatten_fill_refresh_ms"] = 0
    state["cooldown_unresolved_residue"] = False
    latch_path = self._equity_hard_stop_write_latch(pside, payload)
    logging.critical(
        "[risk] HSL[%s] cooldown violation repanic flattened; cooldown reset from flat_ts=%s to cooldown_until_ms=%s latch=%s",
        pside,
        stop_ts_ms,
        cooldown_until_ms if cooldown_until_ms is not None else "none",
        latch_path,
    )
    if cooldown_until_ms is not None:
        _emit_hsl_event(
            self,
            "hsl.cooldown_started",
            ("hsl", "risk", "cooldown"),
            {
                "reason": "repanic_reset",
                "cooldown_until_ms": int(cooldown_until_ms),
                "latch_path": str(latch_path),
            },
            pside=pside,
            ts=stop_ts_ms,
            status="started",
            reason_code="repanic_reset",
        )
    return True


async def _equity_hard_stop_refresh_coin_cooldown_after_repanic(
    self, pside: str, symbol: str, now_ms: int
) -> bool:
    state = self._hsl_coin_state(pside, symbol)
    cooldown_minutes = float(
        _equity_hard_stop_config(self, pside, symbol)["cooldown_minutes_after_red"]
    )
    cooldown_ms = max(0, int(round(cooldown_minutes * 60_000.0))) if cooldown_minutes > 0.0 else 0
    repanic_since_ms = state.get("cooldown_repanic_since_ms")
    stop_ts_ms = await self._equity_hard_stop_flatten_fill_timestamp_with_refresh(
        pside,
        now_ms,
        symbol=symbol,
        since_ms=repanic_since_ms,
        replay_start_sizes=state.get("cooldown_repanic_start_sizes") or {},
    )
    if stop_ts_ms is None:
        return False
    state["last_missing_flatten_fill_log_ms"] = 0
    cooldown_until_ms = stop_ts_ms + cooldown_ms if cooldown_ms > 0 else None
    stop_event = await self._equity_hard_stop_compute_coin_stop_event(
        pside, symbol, stop_ts_ms
    )
    payload = self._equity_hard_stop_build_latch_payload(
        pside,
        symbol=symbol,
        stop_event_timestamp_ms=stop_ts_ms,
        balance=float(stop_event["balance"]),
        realized_pnl_total=stop_event.get("realized_pnl_total"),
        realized_pnl=float(stop_event["realized_pnl"]),
        unrealized_pnl=float(stop_event["unrealized_pnl"]),
        strategy_pnl=float(stop_event["strategy_pnl"]),
        peak_strategy_pnl=float(stop_event["peak_strategy_pnl"]),
        strategy_equity=float(stop_event["strategy_equity"]),
        peak_strategy_equity=float(stop_event["peak_strategy_equity"]),
        trigger_peak_strategy_equity=float(stop_event["trigger_peak_strategy_equity"]),
        drawdown_raw=float(stop_event["drawdown_raw"]),
        drawdown_ema=float(stop_event["drawdown_ema"]),
        drawdown_score=float(stop_event["drawdown_score"]),
        no_restart_latched=False,
        cooldown_until_ms=cooldown_until_ms,
    )
    state["last_stop_event"] = payload
    state["cooldown_until_ms"] = cooldown_until_ms
    state["cooldown_intervention_active"] = False
    state["cooldown_repanic_reset_pending"] = False
    state["cooldown_repanic_since_ms"] = None
    state["cooldown_repanic_start_sizes"] = None
    state["last_cooldown_intervention_log_ms"] = 0
    state["cooldown_unresolved_residue"] = False
    state["pending_stop_event"] = None
    state["red_flat_confirmations"] = 0
    state["pnl_reset_timestamp_ms"] = int(stop_ts_ms) + 1
    latch_path = self._equity_hard_stop_write_latch(pside, payload, symbol=symbol)
    logging.critical(
        "[risk] HSL[%s:%s] cooldown violation repanic flattened; cooldown reset from flat_ts=%s "
        "to cooldown_until_ms=%s latch=%s",
        pside,
        symbol,
        stop_ts_ms,
        cooldown_until_ms if cooldown_until_ms is not None else "none",
        latch_path,
    )
    if cooldown_until_ms is not None:
        self._equity_hard_stop_set_coin_runtime_forced_mode(pside, symbol, "graceful_stop")
        _emit_hsl_event(
            self,
            "hsl.cooldown_started",
            ("hsl", "risk", "cooldown"),
            {
                "reason": "coin_repanic_reset",
                "symbol": symbol,
                "cooldown_until_ms": int(cooldown_until_ms),
                "latch_path": str(latch_path),
            },
            pside=pside,
            symbol=symbol,
            ts=stop_ts_ms,
            status="started",
            reason_code="coin_repanic_reset",
        )
    return True


async def _equity_hard_stop_handle_position_during_cooldown(self, pside: str, now_ms: int) -> bool:
    state = self._hsl_state(pside)
    if not state["halted"] or state["no_restart_latched"]:
        return False
    cooldown_until_ms = state["cooldown_until_ms"]
    repanic_pending = bool(state["cooldown_repanic_reset_pending"])
    if (
        cooldown_until_ms is None or now_ms >= cooldown_until_ms
    ) and not repanic_pending:
        return False

    symbols = self._equity_hard_stop_position_symbols(pside)
    policy = self._equity_hard_stop_cooldown_position_policy()
    if not symbols:
        if state["cooldown_repanic_reset_pending"]:
            await self._equity_hard_stop_refresh_cooldown_after_repanic(pside, now_ms)
            return True
        if state["cooldown_intervention_active"]:
            logging.info(
                "[risk] HSL[%s] cooldown intervention ended flat; policy=%s original_cooldown_until_ms=%s",
                pside,
                policy,
                cooldown_until_ms,
            )
        state["cooldown_intervention_active"] = False
        state["cooldown_repanic_reset_pending"] = False
        state["cooldown_repanic_since_ms"] = None
        state["cooldown_repanic_start_sizes"] = None
        state["last_cooldown_intervention_log_ms"] = 0
        state["cooldown_unresolved_residue"] = False
        return False

    should_log = (
        not state["cooldown_intervention_active"]
        or state["last_cooldown_intervention_log_ms"] == 0
        or now_ms - state["last_cooldown_intervention_log_ms"] >= self._equity_hard_stop_cooldown_log_interval_ms
    )
    if should_log:
        logging.critical(
            "[risk] HSL[%s] detected non-flat position during RED cooldown | policy=%s symbols=%s cooldown_until_ms=%s",
            pside,
            policy,
            ",".join(symbols),
            cooldown_until_ms,
        )
        state["last_cooldown_intervention_log_ms"] = now_ms
    if bool(state["cooldown_unresolved_residue"]):
        return False
    state["cooldown_intervention_active"] = True

    if policy == "normal":
        self._equity_hard_stop_reset_after_restart(pside)
        self._equity_hard_stop_remove_latch_file(pside)
        logging.critical(
            "[risk] HSL[%s] operator override during RED cooldown: resumed normal operation and reset drawdown tracker",
            pside,
        )
        return True

    if policy == "panic":
        if not state["cooldown_repanic_reset_pending"]:
            # Exchange fills may share the millisecond of the authoritative
            # non-flat snapshot. Include that boundary and let position-size
            # replay prove which fill actually flattened the scope.
            state["cooldown_repanic_since_ms"] = int(now_ms)
            state["cooldown_repanic_start_sizes"] = {
                position_symbol: abs(
                    float(position.get(pside, {}).get("size", 0.0) or 0.0)
                )
                for position_symbol, position in self.positions.items()
                if float(position.get(pside, {}).get("size", 0.0) or 0.0)
                != 0.0
            }
        state["cooldown_repanic_reset_pending"] = True
    else:
        state["cooldown_repanic_reset_pending"] = False
        state["cooldown_repanic_since_ms"] = None
        state["cooldown_repanic_start_sizes"] = None
    return False


async def _equity_hard_stop_handle_coin_position_during_cooldown(
    self, pside: str, symbol: str, now_ms: int
) -> bool:
    state = self._hsl_coin_state(pside, symbol)
    if not state["halted"] or state["no_restart_latched"]:
        return False
    cooldown_until_ms = state["cooldown_until_ms"]
    repanic_pending = bool(state["cooldown_repanic_reset_pending"])
    if (
        cooldown_until_ms is None or now_ms >= cooldown_until_ms
    ) and not repanic_pending:
        return False

    has_position = self._equity_hard_stop_has_open_position_symbol(pside, symbol)
    policy = self._equity_hard_stop_cooldown_position_policy()
    if not has_position:
        if state["cooldown_repanic_reset_pending"]:
            await self._equity_hard_stop_refresh_coin_cooldown_after_repanic(pside, symbol, now_ms)
            return True
        if state["cooldown_intervention_active"]:
            logging.info(
                "[risk] HSL[%s:%s] cooldown intervention ended flat; policy=%s "
                "original_cooldown_until_ms=%s",
                pside,
                symbol,
                policy,
                cooldown_until_ms,
            )
        state["cooldown_intervention_active"] = False
        state["cooldown_repanic_reset_pending"] = False
        state["cooldown_repanic_since_ms"] = None
        state["cooldown_repanic_start_sizes"] = None
        state["last_cooldown_intervention_log_ms"] = 0
        state["cooldown_unresolved_residue"] = False
        return False

    should_log = (
        not state["cooldown_intervention_active"]
        or state["last_cooldown_intervention_log_ms"] == 0
        or now_ms - state["last_cooldown_intervention_log_ms"] >= self._equity_hard_stop_cooldown_log_interval_ms
    )
    if should_log:
        logging.critical(
            "[risk] HSL[%s:%s] detected non-flat position during RED cooldown | "
            "policy=%s cooldown_until_ms=%s",
            pside,
            symbol,
            policy,
            cooldown_until_ms,
        )
        state["last_cooldown_intervention_log_ms"] = now_ms
    if bool(state["cooldown_unresolved_residue"]):
        return False
    state["cooldown_intervention_active"] = True

    if policy == "normal":
        self._equity_hard_stop_reset_coin_after_restart(pside, symbol)
        self._equity_hard_stop_remove_latch_file(pside, symbol=symbol)
        logging.critical(
            "[risk] HSL[%s:%s] operator override during RED cooldown: resumed normal operation "
            "and reset drawdown tracker",
            pside,
            symbol,
        )
        return True

    if policy == "panic":
        if not state["cooldown_repanic_reset_pending"]:
            # See the pside path above: the flattening fill may carry the same
            # exchange timestamp as this intervention check.
            state["cooldown_repanic_since_ms"] = int(now_ms)
            state["cooldown_repanic_start_sizes"] = {
                symbol: abs(
                    float(
                        self.positions.get(symbol, {})
                        .get(pside, {})
                        .get("size", 0.0)
                        or 0.0
                    )
                )
            }
        state["cooldown_repanic_reset_pending"] = True
        self._equity_hard_stop_set_coin_runtime_forced_mode(pside, symbol, "panic")
    elif policy == "manual":
        self._equity_hard_stop_set_coin_runtime_forced_mode(pside, symbol, "manual")
    elif policy == "tp_only":
        self._equity_hard_stop_set_coin_runtime_forced_mode(
            pside, symbol, "tp_only_with_active_entry_cancellation"
        )
    else:
        self._equity_hard_stop_set_coin_runtime_forced_mode(pside, symbol, "graceful_stop")
    return False


def _equity_hard_stop_reset_after_restart(self, pside: str) -> None:
    state = self._hsl_state(pside)
    state["runtime"].reset()
    state["strategy_pnl_peak"].reset()
    state["halted"] = False
    state["no_restart_latched"] = False
    state["last_metrics"] = None
    state["last_red_progress"] = None
    state["red_flat_confirmations"] = 0
    state["pending_red_since_ms"] = None
    state["cooldown_until_ms"] = None
    state["pending_stop_event"] = None
    state["red_trigger_event_emitted"] = False
    state["last_raw_red_pending_event_ms"] = 0
    state["last_status_log_ms"] = 0
    state["last_cooldown_log_ms"] = 0
    state["cooldown_intervention_active"] = False
    state["cooldown_repanic_reset_pending"] = False
    state["cooldown_repanic_since_ms"] = None
    state["cooldown_repanic_start_sizes"] = None
    state["last_cooldown_intervention_log_ms"] = 0
    state["last_missing_flatten_fill_log_ms"] = 0
    state["last_missing_flatten_fill_refresh_ms"] = 0
    state["cooldown_unresolved_residue"] = False
    self._equity_hard_stop_clear_runtime_forced_modes(pside)


def _equity_hard_stop_replay_from_boundary(
    self, pside: str, timeline: list[dict], signal_mode: str, boundary_ts: int, end_ts: int
) -> int:
    n_rows = 0
    boundary_minute_ts = int(math.floor(int(boundary_ts) / 60_000.0) * 60_000)
    for row in timeline:
        if not isinstance(row, dict):
            continue
        required = ("timestamp", "balance", "realized_pnl")
        if signal_mode == "pside":
            required += (f"realized_pnl_{pside}", f"unrealized_pnl_{pside}")
        else:
            required += ("unrealized_pnl_long", "unrealized_pnl_short")
        if any(key not in row for key in required):
            continue
        ts = int(row["timestamp"])
        if ts < boundary_minute_ts:
            continue
        if ts > int(end_ts):
            break
        row_upnl_total = (
            float(row["unrealized_pnl_long"]) + float(row["unrealized_pnl_short"])
            if signal_mode == "unified"
            else None
        )
        row_realized_pside = float(row[f"realized_pnl_{pside}"]) if signal_mode == "pside" else 0.0
        row_unrealized_pside = (
            float(row[f"unrealized_pnl_{pside}"]) if signal_mode == "pside" else 0.0
        )
        self._equity_hard_stop_apply_sample(
            pside,
            ts,
            float(row["balance"]),
            float(row["realized_pnl"]),
            row_realized_pside,
            row_unrealized_pside,
            unrealized_pnl_total=row_upnl_total,
            latch_red=False,
        )
        n_rows += 1
    return n_rows


def _equity_hard_stop_refresh_halted_runtime_forced_modes(self) -> None:
    symbols = set(self.positions.keys()) | set(self.open_orders.keys()) | set(self.active_symbols)
    for pside in self._hsl_psides():
        if not self._equity_hard_stop_enabled(pside):
            self._equity_hard_stop_clear_runtime_forced_modes(pside)
            continue
        state = self._hsl_state(pside)
        if self._equity_hard_stop_runtime_red_latched(pside) and not state["halted"]:
            # B2.1 red split: a latched episode whose CURRENT sample has
            # recovered stays entry-blocked without panic authorization; the
            # centralized refresher must not overwrite the paused state back
            # to panic. Unknown/missing sample stays protective (panic).
            last_metrics = state.get("last_metrics")
            if last_metrics is not None and not bool(
                last_metrics.get("red_active_now", True)
            ):
                self._equity_hard_stop_set_red_paused_runtime_forced_modes(pside)
            else:
                self._equity_hard_stop_set_red_runtime_forced_modes(pside)
            continue
        if not state["halted"]:
            self._equity_hard_stop_clear_runtime_forced_modes(pside)
            continue
        previous = dict(getattr(self, "_runtime_forced_modes", {}).get(pside, {}) or {})
        forced = {}
        for symbol in symbols:
            forced[symbol] = self._equity_hard_stop_halted_mode(pside, symbol)
        self._runtime_forced_modes[pside] = forced
        if previous != forced:
            _emit_runtime_forced_mode_changed_event(
                self,
                pside=pside,
                action="replace",
                symbols=forced.keys(),
                previous_modes=previous,
                modes=forced,
                reason_code="hsl_halted_runtime_forced_modes",
            )


def _hsl_coin_replay_candidate_batches(
    active_pairs: Iterable[tuple[str, str]],
    held_pairs: Iterable[tuple[str, str]],
    cooldown_pairs: Iterable[tuple[str, str]],
) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Freeze coin replay candidates into protective-priority batches."""
    frozen = tuple(active_pairs)
    held = set(held_pairs)
    cooldown = set(cooldown_pairs) - held
    return (
        tuple(pair for pair in frozen if pair in held),
        tuple(pair for pair in frozen if pair in cooldown),
        tuple(pair for pair in frozen if pair not in held and pair not in cooldown),
    )


async def _equity_hard_stop_initialize_from_history(self) -> None:
    if not self._equity_hard_stop_enabled():
        return
    self._equity_hard_stop_validate_balance_source_for_history_replay()
    prev_phase = getattr(self, "_log_silence_watchdog_phase", "runtime")
    prev_stage = getattr(self, "_log_silence_watchdog_stage", "idle")
    if hasattr(self, "_set_log_silence_watchdog_context"):
        self._set_log_silence_watchdog_context(
            phase=prev_phase, stage="equity_hard_stop_initialize_from_history"
        )
    try:
        self._equity_hard_stop_reset_state()
        signal_mode = self._equity_hard_stop_signal_mode()
        if signal_mode not in ("unified", "pside"):
            raise ValueError(
                "HSL initialize_from_history requires signal_mode unified or pside, "
                f"got {signal_mode!r}; coin mode must use the coin history initializer"
            )
        lookback = parse_pnls_max_lookback_days(
            self.live_value("pnls_max_lookback_days"),
            field_name="live.pnls_max_lookback_days",
        )
        logging.info(
            "[risk] HSL history replay starting | lookback_days=%s signal_mode=%s",
            lookback.display_value,
            signal_mode,
        )
        history = await self.get_balance_equity_history(
            current_balance=self.get_raw_balance(),
            hsl_replay_signal_mode=signal_mode,
        )
        if "timeline" not in history:
            raise ValueError("get_balance_equity_history() missing required key: timeline")
        timeline = history["timeline"]
        if not isinstance(timeline, list):
            raise TypeError(
                f"get_balance_equity_history()['timeline'] must be a list, got {type(timeline).__name__}"
            )
        panic_flatten_events = history["panic_flatten_events"] if "panic_flatten_events" in history else []
        if panic_flatten_events is None:
            panic_flatten_events = []
        if not isinstance(panic_flatten_events, list):
            raise TypeError(
                "get_balance_equity_history()['panic_flatten_events'] must be a list, "
                f"got {type(panic_flatten_events).__name__}"
            )
        fill_events = history["fill_events"] if "fill_events" in history else []
        if fill_events is None:
            fill_events = []
        if not isinstance(fill_events, list):
            raise TypeError(
                f"get_balance_equity_history()['fill_events'] must be a list, got {type(fill_events).__name__}"
            )
        panic_flatten_events_by_key = {}
        for item in panic_flatten_events:
            if not isinstance(item, dict):
                continue
            pside = str(item.get("pside") or "").lower()
            if pside not in self._hsl_psides():
                continue
            minute_ts = item.get("minute_timestamp")
            stop_ts = item.get("timestamp")
            if minute_ts is None or stop_ts is None:
                continue
            key = (pside, int(minute_ts))
            marker = {
                "timestamp": int(stop_ts),
                "minute_timestamp": int(minute_ts),
                "pside": pside,
                "symbol": str(item.get("symbol") or ""),
            }
            prev = panic_flatten_events_by_key.get(key)
            if prev is None or marker["timestamp"] >= prev["timestamp"]:
                panic_flatten_events_by_key[key] = marker
        now_ms = int(self.get_exchange_time())
        replay_contracts = {
            pside: self._equity_hard_stop_infer_replay_contract(pside, fill_events, now_ms)
            for pside in self._hsl_psides()
        }
        current_balance = float(self.get_raw_balance())
        current_realized_total = float(self._equity_hard_stop_realized_pnl_now())
        current_upnl_by_pside = {
            pside: float(await self._calc_upnl_sum_strict(pside)) for pside in self._hsl_psides()
        }
        current_upnl_total = float(sum(current_upnl_by_pside.values()))
        n_rows = {pside: 0 for pside in self._hsl_psides()}
        for pside in self._hsl_psides():
            if not self._equity_hard_stop_enabled(pside):
                continue
            state = self._hsl_state(pside)
            contract = replay_contracts[pside]
            if contract["intervention_entry_ts"] is not None and contract["policy"] == "normal":
                self._equity_hard_stop_reset_after_restart(pside)
                n_rows[pside] = self._equity_hard_stop_replay_from_boundary(
                    pside,
                    timeline,
                    signal_mode,
                    int(contract["intervention_entry_ts"]),
                    now_ms,
                )
                self._equity_hard_stop_remove_latch_file(pside)
                logging.critical(
                    "[risk] HSL[%s] reconstructed operator override during RED cooldown from exchange-derived history | entry_ts=%s policy=normal",
                    pside,
                    int(contract["intervention_entry_ts"]),
                )
                current_metrics = self._equity_hard_stop_apply_sample(
                    pside,
                    now_ms,
                    current_balance,
                    current_realized_total,
                    float(self._equity_hard_stop_realized_pnl_now(pside)),
                    current_upnl_by_pside[pside],
                    unrealized_pnl_total=current_upnl_total,
                )
                logging.info(
                    "[risk] HSL[%s] initialized from equity history | rows=%d tier=%s strategy_equity=%.6f peak_strategy_equity=%.6f rolling_peak_strategy_equity=%.6f drawdown_raw=%.6f drawdown_ema=%.6f drawdown_score=%.6f",
                    pside,
                    n_rows[pside],
                    current_metrics["tier"],
                    current_metrics["strategy_equity"],
                    current_metrics["peak_strategy_equity"],
                    current_metrics["rolling_peak_strategy_equity"],
                    current_metrics["drawdown_raw"],
                    current_metrics["drawdown_ema"],
                    current_metrics["drawdown_score"],
                )
                if current_metrics["tier"] == "red":
                    state["pending_red_since_ms"] = int(current_metrics["timestamp_ms"])
                continue
            ignored_panic_marker_timestamps: set[int] = set()
            scope_flat_key = "is_flat" if signal_mode == "unified" else f"is_flat_{pside}"
            scope_fill_ts = sorted(
                _equity_hard_stop_fill_timestamp_ms(fill)
                for fill in fill_events
                if signal_mode == "unified"
                or _equity_hard_stop_fill_pside(fill) == pside
            )
            scope_was_nonflat = False
            prev_recorded_ts: Optional[int] = None
            for row in timeline:
                if not isinstance(row, dict):
                    continue
                required = ("timestamp", "balance", "realized_pnl")
                if signal_mode == "pside":
                    required += (f"realized_pnl_{pside}", f"unrealized_pnl_{pside}")
                else:
                    required += ("unrealized_pnl_long", "unrealized_pnl_short")
                if any(key not in row for key in required):
                    continue
                row_upnl_total = (
                    float(row["unrealized_pnl_long"]) + float(row["unrealized_pnl_short"])
                    if signal_mode == "unified"
                    else None
                )
                row_realized_pside = float(row[f"realized_pnl_{pside}"]) if signal_mode == "pside" else 0.0
                row_unrealized_pside = float(row[f"unrealized_pnl_{pside}"]) if signal_mode == "pside" else 0.0
                ts = int(row["timestamp"])
                if ts > now_ms:
                    break
                if state["halted"] and not state["no_restart_latched"]:
                    cooldown_until_ms = state["cooldown_until_ms"]
                    if cooldown_until_ms is not None and ts >= cooldown_until_ms:
                        self._equity_hard_stop_reset_after_restart(pside)
                    elif cooldown_until_ms is not None and ts < cooldown_until_ms:
                        continue
                current_metrics = self._equity_hard_stop_apply_sample(
                    pside,
                    int(ts),
                    float(row["balance"]),
                    float(row["realized_pnl"]),
                    row_realized_pside,
                    row_unrealized_pside,
                    unrealized_pnl_total=row_upnl_total,
                    latch_red=False,
                )
                n_rows[pside] += 1
                row_flat = row.get(scope_flat_key)
                scope_flattened_this_row = (
                    isinstance(row_flat, bool) and row_flat and scope_was_nonflat
                )
                if isinstance(row_flat, bool):
                    scope_was_nonflat = not row_flat
                row_prev_recorded_ts = prev_recorded_ts
                prev_recorded_ts = ts
                panic_flatten_marker = panic_flatten_events_by_key.get((pside, ts))
                stop_ts: Optional[int] = None
                stop_source = "panic_fill_flatten"
                if panic_flatten_marker is not None:
                    marker_ts = int(panic_flatten_marker["timestamp"])
                    if not _equity_hard_stop_replay_marker_confirms_red(current_metrics):
                        ignored_panic_marker_timestamps.add(marker_ts)
                        logging.warning(
                            "[risk] HSL[%s] ignored historical panic marker without reconstructed RED | "
                            "stop_ts=%s drawdown_raw=%.6f drawdown_ema=%.6f drawdown_score=%.6f "
                            "red_threshold=%.6f source=panic_fill_flatten",
                            pside,
                            marker_ts,
                            float(current_metrics["drawdown_raw"]),
                            float(current_metrics["drawdown_ema"]),
                            float(current_metrics["drawdown_score"]),
                            float(current_metrics["red_threshold"]),
                        )
                        continue
                    stop_ts = marker_ts
                elif scope_flattened_this_row:
                    if bool(current_metrics.get("red_seen_in_episode")):
                        # B2.1: the episode crossed RED and ended by an ordinary
                        # (non-panic) scope-flattening fill; cooldown/no-restart
                        # evidence is canonical from the reconstructed episode.
                        # Anchor at the latest scope fill inside the flatten
                        # window, falling back to the row minute.
                        boundary = ts + 60_000
                        window_start = (
                            row_prev_recorded_ts
                            if row_prev_recorded_ts is not None
                            else ts - 60_000
                        )
                        idx = bisect.bisect_left(scope_fill_ts, boundary)
                        anchor = None
                        if idx > 0 and scope_fill_ts[idx - 1] > window_start:
                            anchor = int(scope_fill_ts[idx - 1])
                        stop_ts = anchor if anchor is not None else int(ts)
                        stop_source = "red_episode_flatten"
                    else:
                        # Ordinary flatten of a RED-free episode: plain episode
                        # reset with no stop accounting.
                        self._equity_hard_stop_reset_after_restart(pside)
                        state = self._hsl_state(pside)
                        logging.info(
                            "[risk] HSL[%s] replay reset current episode after flat row | ts=%s",
                            pside,
                            int(ts),
                        )
                        continue
                if stop_ts is None:
                    continue
                if True:
                    state["pending_red_since_ms"] = int(ts)
                    stop_drawdown_raw = float(current_metrics["drawdown_raw"])
                    finalization = self._equity_hard_stop_red_episode_finalization(
                        pside,
                        {
                            "equity": float(current_metrics["strategy_equity"]),
                            "peak_strategy_equity": float(current_metrics["peak_strategy_equity"]),
                            "drawdown_ema": float(current_metrics["drawdown_ema"]),
                        },
                        stop_ts,
                    )
                    no_restart_peak_strategy_equity = float(
                        finalization["no_restart_peak_strategy_equity"]
                    )
                    no_restart_drawdown_raw = float(
                        finalization["no_restart_drawdown_raw"]
                    )
                    no_restart_latched = bool(finalization["no_restart_latched"])
                    cooldown_until_ms = finalization["cooldown_until_ms"]
                    payload = self._equity_hard_stop_build_latch_payload(
                        pside,
                        stop_event_timestamp_ms=stop_ts,
                        balance=float(row["balance"]),
                        realized_pnl_total=float(row["realized_pnl"]),
                        realized_pnl=float(row[f"realized_pnl_{pside}"]),
                        unrealized_pnl=float(row[f"unrealized_pnl_{pside}"]),
                        strategy_pnl=float(current_metrics["strategy_pnl"]),
                        peak_strategy_pnl=float(current_metrics["peak_strategy_pnl"]),
                        strategy_equity=float(current_metrics["strategy_equity"]),
                        peak_strategy_equity=float(current_metrics["peak_strategy_equity"]),
                        trigger_peak_strategy_equity=float(current_metrics["peak_strategy_equity"]),
                        drawdown_raw=float(current_metrics["drawdown_raw"]),
                        drawdown_ema=float(current_metrics["drawdown_ema"]),
                        drawdown_score=float(current_metrics["drawdown_score"]),
                        no_restart_latched=no_restart_latched,
                        cooldown_until_ms=cooldown_until_ms,
                        no_restart_peak_strategy_equity=no_restart_peak_strategy_equity,
                        no_restart_drawdown_raw=no_restart_drawdown_raw,
                    )
                    state["last_stop_event"] = payload
                    state["halted"] = True
                    state["no_restart_latched"] = no_restart_latched
                    state["cooldown_until_ms"] = cooldown_until_ms
                    state["pending_red_since_ms"] = None
                    latch_path = self._equity_hard_stop_write_latch(pside, payload)
                    logging.critical(
                        "[risk] HSL[%s] replay found finalized RED stop in exchange-derived history | "
                        "stop_ts=%s drawdown_raw=%.6f no_restart_drawdown_raw=%.6f "
                        "no_restart_latched=%s cooldown_until_ms=%s diagnostic=%s "
                        "source=%s",
                        pside,
                        stop_ts,
                        stop_drawdown_raw,
                        no_restart_drawdown_raw,
                        state["no_restart_latched"],
                        cooldown_until_ms if cooldown_until_ms is not None else "none",
                        latch_path,
                        stop_source,
                    )
                    if state["no_restart_latched"]:
                        break
                    continue
            if (
                not state["halted"]
                and contract["latest_panic_ts"] is not None
                and int(contract["latest_panic_ts"]) not in ignored_panic_marker_timestamps
                and contract["active_cooldown_now"]
                and not state["no_restart_latched"]
            ):
                state["halted"] = True
                state["cooldown_until_ms"] = contract["cooldown_until_ms"]
                state["cooldown_intervention_active"] = bool(contract["intervention_active"])
                state["cooldown_unresolved_residue"] = bool(contract["unresolved_residue"])
                if state["last_stop_event"] is None:
                    state["last_stop_event"] = {
                        "stop_event_timestamp_ms": int(contract["latest_panic_ts"]),
                        "cooldown_until_ms": contract["cooldown_until_ms"],
                        "no_restart_latched": False,
                    }
            if state["halted"] and not state["no_restart_latched"]:
                cooldown_until_ms = state["cooldown_until_ms"]
                if cooldown_until_ms is not None and now_ms >= cooldown_until_ms:
                    self._equity_hard_stop_reset_after_restart(pside)
                    self._equity_hard_stop_remove_latch_file(pside)
                    logging.info("[risk] HSL[%s] replayed cooldown already elapsed; resumed", pside)
                elif cooldown_until_ms is not None:
                    reason = (
                        " unresolved_panic_residue"
                        if state["cooldown_unresolved_residue"]
                        else (
                            f" intervention_policy={contract['policy']}"
                            if contract["intervention_active"]
                            else ""
                        )
                    )
                    logging.critical(
                        "[risk] HSL[%s] reconstructed active RED cooldown from exchange-derived history | remaining_time=%s%s",
                        pside,
                        _equity_hard_stop_format_remaining_time(
                            (cooldown_until_ms - now_ms) / 1000.0
                        ),
                        reason,
                    )
            if state["halted"]:
                continue
            current_metrics = self._equity_hard_stop_apply_sample(
                pside,
                now_ms,
                current_balance,
                current_realized_total,
                float(self._equity_hard_stop_realized_pnl_now(pside)),
                current_upnl_by_pside[pside],
                unrealized_pnl_total=current_upnl_total,
            )
            logging.info(
                "[risk] HSL[%s] initialized from equity history | rows=%d tier=%s strategy_equity=%.6f peak_strategy_equity=%.6f rolling_peak_strategy_equity=%.6f drawdown_raw=%.6f drawdown_ema=%.6f drawdown_score=%.6f",
                pside,
                n_rows[pside],
                current_metrics["tier"],
                current_metrics["strategy_equity"],
                current_metrics["peak_strategy_equity"],
                current_metrics["rolling_peak_strategy_equity"],
                current_metrics["drawdown_raw"],
                current_metrics["drawdown_ema"],
                current_metrics["drawdown_score"],
            )
            if current_metrics["tier"] == "red":
                state["pending_red_since_ms"] = int(current_metrics["timestamp_ms"])
        self._equity_hard_stop_refresh_halted_runtime_forced_modes()
    finally:
        if hasattr(self, "_set_log_silence_watchdog_context"):
            self._set_log_silence_watchdog_context(phase=prev_phase, stage=prev_stage)


async def _equity_hard_stop_initialize_coin_from_history(self) -> None:
    if not self._equity_hard_stop_enabled() or self._equity_hard_stop_signal_mode() != "coin":
        return
    prev_phase = getattr(self, "_log_silence_watchdog_phase", "runtime")
    prev_stage = getattr(self, "_log_silence_watchdog_stage", "idle")
    raise_if_shutdown = getattr(self, "_raise_if_shutdown_requested", None)
    initialization_started_s = time.monotonic()
    watchdog_context_restored = False
    protective_ready_elapsed_s: Optional[float] = None
    ready_event = getattr(self, "_equity_hard_stop_coin_replay_ready_event", None)
    if ready_event is None:
        ready_event = asyncio.Event()
        self._equity_hard_stop_coin_replay_ready_event = ready_event
    self._equity_hard_stop_coin_protective_ready = False
    self._equity_hard_stop_coin_replay_ready_pairs = set()
    self._equity_hard_stop_coin_replay_pending_pairs = set()
    self._equity_hard_stop_coin_replay_failure = None

    def check_shutdown(stage: str) -> None:
        if callable(raise_if_shutdown):
            raise_if_shutdown(stage)
            return
        if getattr(self, "stop_signal_received", False) or getattr(
            self, "_shutdown_in_progress", False
        ):
            raise asyncio.CancelledError(f"shutdown requested during {stage}")

    if hasattr(self, "_set_log_silence_watchdog_context"):
        self._set_log_silence_watchdog_context(
            phase=prev_phase, stage="equity_hard_stop_initialize_coin_from_history"
        )
    try:
        check_shutdown("hsl_coin_history_replay_start")
        self._equity_hard_stop_coin = {"long": {}, "short": {}}
        self._runtime_forced_modes = {"long": {}, "short": {}}
        lookback = parse_pnls_max_lookback_days(
            self.live_value("pnls_max_lookback_days"),
            field_name="live.pnls_max_lookback_days",
        )
        logging.info(
            "[risk] HSL coin history reconstruction starting | lookback_days=%s",
            lookback.display_value,
        )
        _emit_hsl_replay_event(
            self,
            "hsl.replay.started",
            {
                "signal_mode": "coin",
                "lookback_days": lookback.display_value,
            },
            status="started",
            reason_code="coin_history_replay",
        )
        now_ms = int(self.get_exchange_time())
        configured_start_ms = lookback.balance_history_start_ms(now_ms)
        history_required, replay_start_ms = (
            self._equity_hard_stop_required_fill_history_start_ms(
                now_ms,
                pnl_start_ms=configured_start_ms,
            )
        )
        if not history_required:
            replay_start_ms = now_ms
        history_fetch_started_s = time.monotonic()
        history = await self.get_balance_equity_history(
            current_balance=self.get_raw_balance(),
            hsl_replay_signal_mode="coin",
            hsl_coin_compact_replay=True,
            hsl_replay_start_ms=replay_start_ms,
        )
        history_loaded_s = time.monotonic()
        history_fetch_elapsed_s = max(0.0, history_loaded_s - history_fetch_started_s)
        check_shutdown("hsl_coin_history_replay_history_loaded")
        panic_flatten_events = history["panic_flatten_events"] if "panic_flatten_events" in history else []
        if panic_flatten_events is None:
            panic_flatten_events = []
        if not isinstance(panic_flatten_events, list):
            raise TypeError(
                "get_balance_equity_history()['panic_flatten_events'] must be a list, "
                f"got {type(panic_flatten_events).__name__}"
            )
        fill_events = history["fill_events"] if "fill_events" in history else []
        if fill_events is None:
            fill_events = []
        if not isinstance(fill_events, list):
            raise TypeError(
                f"get_balance_equity_history()['fill_events'] must be a list, got {type(fill_events).__name__}"
            )
        fill_events_by_pair = _equity_hard_stop_index_coin_fill_events(fill_events)

        compact_replay = history.get("hsl_coin_compact_replay")
        timeline = history.get("timeline")
        if compact_replay is None:
            if timeline is None:
                raise ValueError(
                    "get_balance_equity_history() missing required coin replay payload: "
                    "timeline or hsl_coin_compact_replay"
                )
            if not isinstance(timeline, list):
                raise TypeError(
                    "get_balance_equity_history()['timeline'] must be a list, "
                    f"got {type(timeline).__name__}"
                )
        else:
            import numpy as np

            if not isinstance(compact_replay, dict):
                raise TypeError(
                    "get_balance_equity_history()['hsl_coin_compact_replay'] must be a dict, "
                    f"got {type(compact_replay).__name__}"
                )
            required_compact_fields = {
                "timestamps",
                "balances",
                "realized_pnl",
                "pair_values",
            }
            actual_compact_fields = set(compact_replay)
            if actual_compact_fields != required_compact_fields:
                raise ValueError(
                    "compact coin HSL replay fields mismatch: "
                    f"missing={sorted(required_compact_fields - actual_compact_fields)} "
                    f"extra={sorted(actual_compact_fields - required_compact_fields)}"
                )
            compact_timestamps = np.asarray(compact_replay["timestamps"], dtype=np.int64)
            compact_balances = np.asarray(compact_replay["balances"], dtype=np.float64)
            compact_realized_pnl = np.asarray(
                compact_replay["realized_pnl"], dtype=np.float64
            )
            compact_pair_values = compact_replay["pair_values"]
            if not isinstance(compact_pair_values, dict):
                raise TypeError(
                    "compact coin HSL replay pair_values must be a dict, "
                    f"got {type(compact_pair_values).__name__}"
                )
            compact_len = len(compact_timestamps)
            if compact_len == 0:
                raise ValueError("compact coin HSL replay must contain at least one row")
            if len(compact_balances) != compact_len or len(compact_realized_pnl) != compact_len:
                raise ValueError("compact coin HSL replay account arrays must have equal lengths")
            if compact_len > 1 and not bool(
                np.all(np.diff(compact_timestamps) == _HSL_REPLAY_INTERVAL_MS)
            ):
                raise ValueError(
                    "compact coin HSL replay timestamps must be contiguous 1m samples"
                )
            if not bool(np.all(np.isfinite(compact_balances))) or bool(
                np.any(compact_balances <= 0.0)
            ):
                raise ValueError(
                    "compact coin HSL replay balances must be finite and > 0"
                )
            if not bool(np.all(np.isfinite(compact_realized_pnl))):
                raise ValueError("compact coin HSL replay realized_pnl must be finite")
            normalized_pair_values: dict[tuple[str, str], dict[str, Any]] = {}
            for pair, values in compact_pair_values.items():
                if (
                    not isinstance(pair, tuple)
                    or len(pair) != 2
                    or str(pair[0]) not in self._hsl_psides()
                    or not str(pair[1])
                ):
                    raise ValueError(f"invalid compact coin HSL replay pair key: {pair!r}")
                if not isinstance(values, dict) or set(values) != {
                    "realized_pnl",
                    "unrealized_pnl",
                }:
                    raise ValueError(
                        f"invalid compact coin HSL replay values for {pair!r}"
                    )
                realized_values = np.asarray(values["realized_pnl"], dtype=np.float64)
                unrealized_values = np.asarray(values["unrealized_pnl"], dtype=np.float64)
                if len(realized_values) != compact_len or len(unrealized_values) != compact_len:
                    raise ValueError(
                        f"compact coin HSL replay pair arrays must match account length for {pair!r}"
                    )
                if bool(np.any(np.isinf(realized_values))) or bool(
                    np.any(np.isinf(unrealized_values))
                ):
                    raise ValueError(
                        f"compact coin HSL replay pair values must be finite or NaN for {pair!r}"
                    )
                normalized_pair_values[(str(pair[0]), str(pair[1]))] = {
                    "realized_pnl": realized_values,
                    "unrealized_pnl": unrealized_values,
                }
            compact_pair_values = normalized_pair_values

        now_ms = int(self.get_exchange_time())
        lookback_ms = self._equity_hard_stop_lookback_ms()
        lookback_start_ms = None if lookback_ms is None else now_ms - int(lookback_ms)
        symbols = set(self.positions.keys())
        current_position_pairs: set[tuple[str, str]] = set()
        required_replay_pairs: set[tuple[str, str]] = set()
        required_replay_start_ts: dict[tuple[str, str], int] = {}
        panic_replay_pairs: set[tuple[str, str]] = set()
        skipped_unsupported_symbols: set[str] = set()

        def remember_required_replay_start(
            pside: str, symbol: str, ts_ms: int
        ) -> None:
            replay_ts = int(math.floor(int(ts_ms) / 60_000) * 60_000)
            key = (pside, symbol)
            prev = required_replay_start_ts.get(key)
            if prev is None or replay_ts < prev:
                required_replay_start_ts[key] = replay_ts

        for symbol, slots in (self.positions or {}).items():
            if not isinstance(slots, dict):
                continue
            for pside in self._hsl_psides():
                if self._equity_hard_stop_has_open_position_symbol(pside, str(symbol)):
                    symbols.add(str(symbol))
                    current_position_pairs.add((pside, str(symbol)))
                    required_replay_pairs.add((pside, str(symbol)))
        for event in fill_events:
            ts = _equity_hard_stop_fill_timestamp_ms(event)
            if lookback_start_ms is not None and ts < lookback_start_ms:
                continue
            symbol = _equity_hard_stop_fill_symbol(event)
            if symbol:
                if not self._equity_hard_stop_symbol_supported_for_coin_replay(symbol):
                    skipped_unsupported_symbols.add(symbol)
                    continue
                pside = _equity_hard_stop_fill_pside(event)
                symbols.add(symbol)
        bounded_held_replay_starts: dict[tuple[str, str], int] = {}
        for pside, symbol in sorted(current_position_pairs):
            pair_fill_events = fill_events_by_pair.get((pside, symbol), [])
            restart_policy = normalize_hsl_restart_after_red_policy(
                _equity_hard_stop_config(self, pside, symbol).get(
                    "restart_after_red_policy", "threshold"
                ),
                path=f"hsl.{pside}.restart_after_red_policy",
            )
            bounded_start_ts = None
            if restart_policy == "always":
                bounded_start_ts = (
                    _equity_hard_stop_coin_bounded_required_replay_start_ts(
                        self,
                        pside,
                        symbol,
                        pair_fill_events,
                    )
                )
            if bounded_start_ts is not None:
                replay_ts = int(
                    math.floor(int(bounded_start_ts) / 60_000) * 60_000
                )
                required_replay_start_ts[(pside, symbol)] = replay_ts
                bounded_held_replay_starts[(pside, symbol)] = replay_ts
            else:
                for event in pair_fill_events:
                    remember_required_replay_start(
                        pside,
                        symbol,
                        _equity_hard_stop_fill_timestamp_ms(event),
                    )
        if compact_replay is not None:
            for _pside, symbol in compact_pair_values:
                if not self._equity_hard_stop_symbol_supported_for_coin_replay(symbol):
                    skipped_unsupported_symbols.add(symbol)
                    continue
                symbols.add(symbol)
        else:
            for row in timeline:
                if not isinstance(row, dict):
                    continue
                for key in (
                    "realized_pnl_by_coin_pside",
                    "unrealized_pnl_by_coin_pside",
                ):
                    if key not in row or row[key] is None:
                        continue
                    if not isinstance(row[key], dict):
                        raise TypeError(
                            f"get_balance_equity_history()['timeline'][]['{key}'] must be a dict, "
                            f"got {type(row[key]).__name__}"
                        )
                    for symbol in row[key].keys():
                        symbol = str(symbol)
                        if not symbol:
                            continue
                        if not self._equity_hard_stop_symbol_supported_for_coin_replay(
                            symbol
                        ):
                            skipped_unsupported_symbols.add(symbol)
                            continue
                        symbols.add(symbol)

        latest_panic_by_coin_minute: dict[tuple[str, str, int], dict[str, Any]] = {}
        for item in panic_flatten_events:
            if not isinstance(item, dict):
                continue
            pside = str(item.get("pside") or "").lower()
            symbol = str(item.get("symbol") or "")
            stop_ts = item.get("timestamp")
            minute_ts = item.get("minute_timestamp")
            if pside not in self._hsl_psides() or not symbol or stop_ts is None or minute_ts is None:
                continue
            stop_ts = int(stop_ts)
            minute_ts = int(minute_ts)
            if lookback_start_ms is not None and stop_ts < lookback_start_ms:
                continue
            if not self._equity_hard_stop_symbol_supported_for_coin_replay(symbol):
                skipped_unsupported_symbols.add(symbol)
                continue
            symbols.add(symbol)
            panic_replay_pairs.add((pside, symbol))
            required_replay_pairs.add((pside, symbol))
            bounded_start_ts = bounded_held_replay_starts.get((pside, symbol))
            if bounded_start_ts is None or minute_ts >= bounded_start_ts:
                remember_required_replay_start(pside, symbol, minute_ts)
            key = (pside, symbol, minute_ts)
            prev = latest_panic_by_coin_minute.get(key)
            if prev is None or stop_ts >= int(prev["timestamp"]):
                latest_panic_by_coin_minute[key] = {
                    "timestamp": stop_ts,
                    "minute_timestamp": minute_ts,
                    "pside": pside,
                    "symbol": symbol,
                }

        if skipped_unsupported_symbols:
            logging.warning(
                "[risk] HSL coin history skipping unsupported historical symbols "
                "with no current position | symbols=%s",
                ",".join(sorted(skipped_unsupported_symbols)),
            )

        timeline_rows: list[dict[str, Any]] = []
        if compact_replay is None:
            for row in timeline:
                if not isinstance(row, dict):
                    continue
                if "timestamp" not in row or "balance" not in row:
                    continue
                ts = int(row["timestamp"])
                if ts > now_ms:
                    break
                timeline_rows.append(row)
            replay_row_count = len(timeline_rows)
        else:
            replay_row_count = int(np.searchsorted(compact_timestamps, now_ms, side="right"))

        balance = float(self.get_raw_balance())
        rows = 0
        total_scanned_rows = 0
        replay_started_s = time.monotonic()
        pre_replay_elapsed_s = max(0.0, replay_started_s - history_loaded_s)
        last_progress_event_s = replay_started_s
        last_progress_console_s: float | None = None
        replay_symbols = set(symbols)
        active_pairs = tuple(
            (pside, symbol)
            for pside in self._hsl_psides()
            for symbol in sorted(replay_symbols)
            if self._equity_hard_stop_coin_active_pside(pside, symbol)
        )
        active_pair_set = set(active_pairs)
        active_held_pairs = active_pair_set.intersection(current_position_pairs)
        active_panic_pairs = active_pair_set.intersection(panic_replay_pairs)
        active_required_pairs = active_pair_set.intersection(required_replay_pairs)
        replay_candidate_batches = _hsl_coin_replay_candidate_batches(
            active_pairs,
            active_held_pairs,
            active_panic_pairs,
        )
        self._equity_hard_stop_coin_replay_pending_pairs = set(active_pairs)
        pair_rows_applied: dict[tuple[str, str], int] = {}
        pair_candidate_rows: dict[tuple[str, str], int] = {}
        dense_replay_pairs = 0
        dense_fallback_pairs = 0
        sparse_replay_pairs = 0
        logging.info(
            "[risk] HSL coin history reconstruction loaded | symbols=%d pairs=%d rows=%d fills=%d panic_events=%d",
            len(symbols),
            len(active_pairs),
            replay_row_count,
            len(fill_events),
            len(panic_flatten_events),
        )
        _emit_hsl_replay_event(
            self,
            "hsl.replay.progress",
            {
                "signal_mode": "coin",
                "stage": "loaded",
                "history_format": (
                    "compact" if compact_replay is not None else "timeline"
                ),
                "replay_strategy": (
                    "compact_pending_classification"
                    if compact_replay is not None
                    else "dense_timeline"
                ),
                "symbols": len(symbols),
                "pairs": len(active_pairs),
                "held_pairs": len(active_held_pairs),
                "cooldown_pairs": len(active_panic_pairs),
                "required_pairs": len(active_required_pairs),
                "timeline_rows": replay_row_count,
                "fill_events": len(fill_events),
                "panic_events": len(panic_flatten_events),
                "skipped_unsupported_symbols": len(skipped_unsupported_symbols),
                "history_fetch_elapsed_s": round(history_fetch_elapsed_s, 3),
                "pre_replay_elapsed_s": round(pre_replay_elapsed_s, 3),
                "elapsed_s": round(pre_replay_elapsed_s, 3),
            },
            status="started",
            reason_code="history_loaded",
        )

        def log_replay_progress(
            pair_idx: int,
            pside: str,
            symbol: str,
            applied_rows: int,
            scanned_rows: int,
            pair_started_s: float,
            *,
            force: bool = False,
        ) -> None:
            nonlocal last_progress_event_s, last_progress_console_s
            now_s = time.monotonic()
            if not force and now_s - last_progress_event_s < 15.0:
                return
            last_progress_event_s = now_s
            elapsed_s = max(0.0, now_s - replay_started_s)
            rows_per_second = float(rows) / elapsed_s if elapsed_s > 0.0 else None
            scanned_rows_per_second = (
                float(total_scanned_rows) / elapsed_s if elapsed_s > 0.0 else None
            )
            pair_elapsed_s = max(0.0, now_s - pair_started_s)
            if (
                last_progress_console_s is None
                or now_s - last_progress_console_s >= 30.0
            ):
                last_progress_console_s = now_s
                logging.info(
                    "[risk] HSL coin history reconstruction progress | pair=%d/%d pside=%s symbol=%s applied_rows=%d scanned_rows=%d total_rows=%d total_scanned_rows=%d elapsed=%.1fs",
                    pair_idx,
                    len(active_pairs),
                    pside,
                    symbol,
                    applied_rows,
                    scanned_rows,
                    rows,
                    total_scanned_rows,
                    now_s - replay_started_s,
                )
            _emit_hsl_replay_event(
                self,
                "hsl.replay.progress",
                {
                    "signal_mode": "coin",
                    "stage": "pair_replay",
                    "pair_idx": int(pair_idx),
                    "pairs": len(active_pairs),
                    "held_pairs": len(active_held_pairs),
                    "cooldown_pairs": len(active_panic_pairs),
                    "required_pairs": len(active_required_pairs),
                    "timeline_rows": replay_row_count,
                    "applied_rows": int(applied_rows),
                    "scanned_rows": int(scanned_rows),
                    "candidate_rows": pair_candidate_rows.get((pside, symbol)),
                    "total_applied_rows": int(rows),
                    "total_scanned_rows": int(total_scanned_rows),
                    "rows_per_second": round(rows_per_second, 3)
                    if rows_per_second is not None
                    else None,
                    "scanned_rows_per_second": round(scanned_rows_per_second, 3)
                    if scanned_rows_per_second is not None
                    else None,
                    "is_held_pair": (pside, symbol) in active_held_pairs,
                    "is_cooldown_pair": (pside, symbol) in active_panic_pairs,
                    "pair_elapsed_s": round(pair_elapsed_s, 3),
                    "elapsed_s": round(elapsed_s, 3),
                },
                pside=pside,
                symbol=symbol,
                status="started",
                reason_code="pair_replay_progress",
            )

        def mark_pair_ready(pside: str, symbol: str) -> None:
            pair = (pside, symbol)
            self._equity_hard_stop_coin_replay_pending_pairs.discard(pair)
            self._equity_hard_stop_coin_replay_ready_pairs.add(pair)

        def mark_protective_ready() -> None:
            nonlocal protective_ready_elapsed_s, watchdog_context_restored
            if self._equity_hard_stop_coin_protective_ready:
                return
            self._equity_hard_stop_coin_protective_ready = True
            startup_blocking_elapsed_s = max(
                0.0, time.monotonic() - initialization_started_s
            )
            protective_ready_elapsed_s = startup_blocking_elapsed_s
            _emit_hsl_replay_event(
                self,
                EventTypes.HSL_REPLAY_PROGRESS,
                {
                    "signal_mode": "coin",
                    "stage": "held_protective_ready",
                    "held_pairs": len(active_held_pairs),
                    "ready_pairs": len(
                        self._equity_hard_stop_coin_replay_ready_pairs
                    ),
                    "pending_pairs": len(
                        self._equity_hard_stop_coin_replay_pending_pairs
                    ),
                    "pairs": len(active_pairs),
                    "startup_blocking_elapsed_s": round(
                        startup_blocking_elapsed_s, 3
                    ),
                    "protective_elapsed_s": round(startup_blocking_elapsed_s, 3),
                    "elapsed_s": round(startup_blocking_elapsed_s, 3),
                },
                status="succeeded",
                reason_code=ReasonCodes.HSL_HELD_PROTECTIVE_READY,
            )
            if hasattr(self, "_set_log_silence_watchdog_context"):
                self._set_log_silence_watchdog_context(
                    phase=prev_phase, stage=prev_stage
                )
                watchdog_context_restored = True
            ready_event.set()

        pair_idx = 0
        for batch_idx, replay_candidates in enumerate(replay_candidate_batches):
            for pside, symbol in replay_candidates:
                check_shutdown("hsl_coin_history_replay_pair")
                pair_idx += 1
                pair_started_s = time.monotonic()
                scanned_rows = 0
                if pair_idx == 1:
                    log_replay_progress(
                        pair_idx,
                        pside,
                        symbol,
                        0,
                        scanned_rows,
                        pair_started_s,
                        force=True,
                    )
                state = self._hsl_coin_state(pside, symbol)
                cooldown_minutes = float(
                    _equity_hard_stop_config(self, pside, symbol)[
                        "cooldown_minutes_after_red"
                    ]
                )
                cooldown_ms = (
                    int(round(cooldown_minutes * 60_000.0))
                    if cooldown_minutes > 0.0
                    else 0
                )
                pair_fill_events = fill_events_by_pair.get((pside, symbol), [])
                contract = self._equity_hard_stop_infer_coin_replay_contract(
                    pside, symbol, pair_fill_events, now_ms
                )
                replay_start_boundary_ts = bounded_held_replay_starts.get(
                    (pside, symbol)
                )
                if contract["intervention_entry_ts"] is not None and contract["policy"] == "normal":
                    intervention_boundary_ts = int(contract["intervention_entry_ts"])
                    replay_start_boundary_ts = (
                        intervention_boundary_ts
                        if replay_start_boundary_ts is None
                        else max(replay_start_boundary_ts, intervention_boundary_ts)
                    )
                    self._equity_hard_stop_reset_coin_after_restart(pside, symbol)
                    self._equity_hard_stop_remove_latch_file(pside, symbol=symbol)
                    state = self._hsl_coin_state(pside, symbol)
                    logging.critical(
                        "[risk] HSL[%s:%s] reconstructed operator override during RED cooldown "
                        "from exchange-derived history | entry_ts=%s policy=normal",
                        pside,
                        symbol,
                        replay_start_boundary_ts,
                    )
                window_points: deque[tuple[int, float]] = deque()
                window_max_points: deque[tuple[int, float]] = deque()
                window_base_realized = 0.0
                reset_baseline_realized = 0.0
                applied_rows = 0
                require_coin_timeline_fields = (pside, symbol) in required_replay_pairs
                required_start_ts = required_replay_start_ts.get((pside, symbol))
                seen_coin_timeline_fields = False
                qty_step = _hsl_qty_step_for_symbol(self, symbol)
                replay_events, replay_ambiguous = _equity_hard_stop_coin_replay_events(
                    pair_fill_events,
                    pside,
                    symbol,
                    qty_step=qty_step,
                )
                if (
                    replay_start_boundary_ts is not None
                    and (pside, symbol) in bounded_held_replay_starts
                ):
                    # Pair realized values are cumulative across the replay
                    # record window. Discarded closed episodes must therefore
                    # become the current episode's baseline rather than leaking
                    # their gains/losses into its drawdown state.
                    reset_baseline_realized = sum(
                        float(realized_delta)
                        for event_ts, _action, _qty, realized_delta in replay_events
                        if int(event_ts) < int(replay_start_boundary_ts)
                    )
                pair_uses_dense_replay = (
                    compact_replay is None
                    or replay_ambiguous
                    or (pside, symbol) in active_held_pairs
                )
                if pair_uses_dense_replay:
                    dense_replay_pairs += 1
                    if compact_replay is not None and replay_ambiguous:
                        dense_fallback_pairs += 1
                else:
                    sparse_replay_pairs += 1
                sparse_boundary_timestamps = []
                if required_start_ts is not None:
                    sparse_boundary_timestamps.append(int(required_start_ts))
                if replay_start_boundary_ts is not None:
                    sparse_boundary_timestamps.append(int(replay_start_boundary_ts))
                if contract["cooldown_until_ms"] is not None:
                    sparse_boundary_timestamps.append(int(contract["cooldown_until_ms"]))
                for event_ts, _action, _qty, _realized_delta in replay_events:
                    sparse_boundary_timestamps.append(
                        int(event_ts) // _HSL_REPLAY_INTERVAL_MS
                        * _HSL_REPLAY_INTERVAL_MS
                    )
                    if cooldown_ms > 0:
                        sparse_boundary_timestamps.append(int(event_ts) + cooldown_ms)
                for (
                    marker_pside,
                    marker_symbol,
                    marker_minute_ts,
                ), marker_payload in latest_panic_by_coin_minute.items():
                    if marker_pside != pside or marker_symbol != symbol:
                        continue
                    sparse_boundary_timestamps.append(int(marker_minute_ts))
                    if cooldown_ms > 0:
                        sparse_boundary_timestamps.append(
                            int(marker_payload["timestamp"]) + cooldown_ms
                        )

                def reset_rolling_window() -> None:
                    nonlocal window_base_realized
                    window_points.clear()
                    window_max_points.clear()
                    window_base_realized = 0.0

                def rolling_realized_at(
                    sample_ts: int, sample_abs_realized: float
                ) -> tuple[float, float]:
                    nonlocal window_base_realized
                    last_realized = float(sample_abs_realized) - reset_baseline_realized
                    start_ms = (
                        None
                        if lookback_ms is None
                        else int(sample_ts) - int(lookback_ms)
                    )
                    reset_ts = state["pnl_reset_timestamp_ms"]
                    if reset_ts is not None:
                        start_ms = (
                            int(reset_ts)
                            if start_ms is None
                            else max(start_ms, int(reset_ts))
                        )
                    window_points.append((int(sample_ts), last_realized))
                    while (
                        window_max_points
                        and window_max_points[-1][1] <= last_realized
                    ):
                        window_max_points.pop()
                    window_max_points.append((int(sample_ts), last_realized))
                    if start_ms is not None:
                        while window_points and window_points[0][0] < start_ms:
                            _old_ts, old_value = window_points.popleft()
                            window_base_realized = float(old_value)
                        while (
                            window_max_points
                            and window_max_points[0][0] < start_ms
                        ):
                            window_max_points.popleft()
                    window_last_realized = (
                        float(last_realized) - window_base_realized
                        if window_points
                        else 0.0
                    )
                    peak_realized = max(
                        0.0,
                        (
                            float(window_max_points[0][1]) - window_base_realized
                            if window_max_points
                            else 0.0
                        ),
                    )
                    return peak_realized, window_last_realized

                replay_event_idx = 0
                replay_size = 0.0
                flat_epsilon = _hsl_flat_epsilon(qty_step)
                ignored_panic_marker_timestamps: set[int] = set()

                def replay_transitions_at(
                    row_ts_ms: int,
                ) -> tuple[float, list[tuple[int, float]], float]:
                    nonlocal replay_event_idx, replay_size
                    boundary_ts_ms = int(row_ts_ms) + 60_000
                    realized_delta = 0.0
                    flatten_boundaries: list[tuple[int, float]] = []
                    while replay_event_idx < len(replay_events):
                        event_ts, action, qty, event_realized_delta = replay_events[
                            replay_event_idx
                        ]
                        if int(event_ts) >= boundary_ts_ms:
                            break
                        was_nonflat = replay_size > flat_epsilon
                        realized_delta += float(event_realized_delta)
                        if action == "increase":
                            replay_size += qty
                        else:
                            replay_size = max(0.0, replay_size - qty)
                            if was_nonflat and replay_size <= flat_epsilon:
                                # Preserve every zero crossing in fill order.
                                # The cumulative realized delta lets the row's
                                # aggregate realized value be split at the
                                # exact episode boundary.
                                flatten_boundaries.append(
                                    (int(event_ts), float(realized_delta))
                                )
                        replay_event_idx += 1
                    return (
                        float(replay_size),
                        flatten_boundaries,
                        float(realized_delta),
                    )

                if compact_replay is not None:
                    compact_values = compact_pair_values.get((pside, symbol))
                    compact_realized_values = (
                        None
                        if compact_values is None
                        else compact_values["realized_pnl"]
                    )
                    compact_unrealized_values = (
                        None
                        if compact_values is None
                        else compact_values["unrealized_pnl"]
                    )
                    compact_replay_indices = (
                        np.arange(replay_row_count, dtype=np.int64)
                        if pair_uses_dense_replay
                        else _hsl_compact_sparse_replay_indices(
                            compact_timestamps[:replay_row_count],
                            compact_balances[:replay_row_count],
                            (
                                None
                                if compact_realized_values is None
                                else compact_realized_values[:replay_row_count]
                            ),
                            (
                                None
                                if compact_unrealized_values is None
                                else compact_unrealized_values[:replay_row_count]
                            ),
                            lookback_ms=lookback_ms,
                            boundary_timestamps=tuple(sparse_boundary_timestamps),
                        )
                    )
                    pair_candidate_rows[(pside, symbol)] = int(
                        len(compact_replay_indices)
                    )

                    def iter_replay_rows():
                        for compact_idx in compact_replay_indices:
                            compact_idx = int(compact_idx)
                            realized_value = (
                                math.nan
                                if compact_realized_values is None
                                else float(compact_realized_values[compact_idx])
                            )
                            unrealized_value = (
                                math.nan
                                if compact_unrealized_values is None
                                else float(compact_unrealized_values[compact_idx])
                            )
                            yield (
                                compact_idx + 1,
                                int(compact_timestamps[compact_idx]),
                                float(compact_balances[compact_idx]),
                                float(compact_realized_pnl[compact_idx]),
                                not math.isnan(realized_value),
                                realized_value,
                                not math.isnan(unrealized_value),
                                unrealized_value,
                                None,
                            )

                else:
                    pair_candidate_rows[(pside, symbol)] = int(replay_row_count)

                    def iter_replay_rows():
                        for legacy_idx, row in enumerate(timeline_rows, start=1):
                            has_realized_value = (
                                _equity_hard_stop_history_coin_has_value(
                                    row,
                                    "realized_pnl_by_coin_pside",
                                    symbol,
                                    pside,
                                )
                            )
                            has_unrealized_value = (
                                _equity_hard_stop_history_coin_has_value(
                                    row,
                                    "unrealized_pnl_by_coin_pside",
                                    symbol,
                                    pside,
                                )
                            )
                            yield (
                                legacy_idx,
                                int(row["timestamp"]),
                                float(row["balance"]),
                                (
                                    float(row["realized_pnl"])
                                    if "realized_pnl" in row
                                    else None
                                ),
                                has_realized_value,
                                (
                                    _equity_hard_stop_history_coin_value(
                                        row,
                                        "realized_pnl_by_coin_pside",
                                        symbol,
                                        pside,
                                        require_key=has_realized_value,
                                        require_value=has_realized_value,
                                    )
                                    if has_realized_value
                                    else 0.0
                                ),
                                has_unrealized_value,
                                (
                                    _equity_hard_stop_history_coin_value(
                                        row,
                                        "unrealized_pnl_by_coin_pside",
                                        symbol,
                                        pside,
                                        require_key=has_unrealized_value,
                                        require_value=has_unrealized_value,
                                    )
                                    if has_unrealized_value
                                    else 0.0
                                ),
                                row,
                            )

                background_replay = bool(
                    self._equity_hard_stop_coin_protective_ready
                )
                yield_rows = (
                    _HSL_COIN_REPLAY_BACKGROUND_YIELD_ROWS
                    if background_replay
                    else _HSL_COIN_REPLAY_STARTUP_YIELD_ROWS
                )
                yield_sleep_s = (
                    _HSL_COIN_REPLAY_BACKGROUND_YIELD_SLEEP_S
                    if background_replay
                    else 0.0
                )
                for replay_iteration_idx, (
                    row_idx,
                    ts,
                    row_balance,
                    row_realized_pnl,
                    has_realized,
                    realized_value,
                    has_unrealized,
                    unrealized_value,
                    source_row,
                ) in enumerate(iter_replay_rows(), start=1):
                    scanned_rows += 1
                    total_scanned_rows += 1
                    if replay_iteration_idx % yield_rows == 0:
                        await asyncio.sleep(yield_sleep_s)
                        check_shutdown("hsl_coin_history_replay_rows")
                        log_replay_progress(
                            pair_idx,
                            pside,
                            symbol,
                            applied_rows,
                            scanned_rows,
                            pair_started_s,
                        )
                    if replay_start_boundary_ts is not None and ts < replay_start_boundary_ts:
                        # Advance fill-derived size state while intentionally
                        # discarding pre-boundary episode transitions.
                        replay_transitions_at(ts)
                        continue
                    if state["halted"]:
                        cooldown_until_ms = state["cooldown_until_ms"]
                        if (
                            not state["no_restart_latched"]
                            and cooldown_until_ms is not None
                            and ts >= cooldown_until_ms
                        ):
                            self._equity_hard_stop_reset_coin_after_restart(pside, symbol)
                            self._equity_hard_stop_remove_latch_file(pside, symbol=symbol)
                            state = self._hsl_coin_state(pside, symbol)
                            reset_rolling_window()
                        else:
                            continue
                    row_has_coin_fields = has_realized or has_unrealized
                    require_coin_timeline_value = (
                        (
                            require_coin_timeline_fields
                            and required_start_ts is not None
                            and ts >= required_start_ts
                        )
                        or seen_coin_timeline_fields
                        or row_has_coin_fields
                    )
                    if not require_coin_timeline_value:
                        continue
                    (
                        replay_position_size,
                        flatten_boundaries,
                        row_realized_delta,
                    ) = replay_transitions_at(ts)
                    replay_is_nonflat = replay_position_size > flat_epsilon
                    if not has_realized:
                        if source_row is not None:
                            _equity_hard_stop_history_coin_value(
                                source_row,
                                "realized_pnl_by_coin_pside",
                                symbol,
                                pside,
                                require_key=True,
                                require_value=True,
                            )
                        raise ValueError(
                            "coin HSL replay missing required "
                            "realized_pnl_by_coin_pside value for "
                            f"{pside}:{symbol} at {ts}"
                        )
                    abs_realized = float(realized_value)
                    seen_coin_timeline_fields = True
                    marker = latest_panic_by_coin_minute.get((pside, symbol, ts))
                    metrics = None
                    stop_ts = None
                    stop_source = None
                    stop_abs_realized = abs_realized

                    if flatten_boundaries and not replay_ambiguous:
                        row_start_abs_realized = abs_realized - row_realized_delta
                        for flatten_ts, realized_delta_at_flatten in flatten_boundaries:
                            boundary_abs_realized = (
                                row_start_abs_realized + realized_delta_at_flatten
                            )
                            boundary_balance = (
                                row_balance
                                - row_realized_delta
                                + realized_delta_at_flatten
                            )
                            peak_realized, window_last_realized = rolling_realized_at(
                                flatten_ts, boundary_abs_realized
                            )
                            self._equity_hard_stop_prime_coin_runtime_for_replay(
                                pside, symbol, flatten_ts
                            )
                            metrics = self._equity_hard_stop_apply_coin_metrics_sample(
                                pside,
                                symbol,
                                flatten_ts,
                                boundary_balance,
                                peak_realized,
                                window_last_realized,
                                0.0,
                                latch_red=False,
                            )
                            applied_rows += 1
                            rows += 1
                            pair_rows_applied[(pside, symbol)] = int(applied_rows)
                            boundary_marker = (
                                marker
                                if marker is not None
                                and int(marker["timestamp"]) == int(flatten_ts)
                                else None
                            )
                            if boundary_marker is not None and not (
                                _equity_hard_stop_replay_marker_confirms_red(metrics)
                            ):
                                ignored_panic_marker_timestamps.add(int(flatten_ts))
                                logging.warning(
                                    "[risk] HSL[%s:%s] ignored historical coin panic marker without reconstructed RED | "
                                    "stop_ts=%s drawdown_raw=%.6f drawdown_ema=%.6f drawdown_score=%.6f "
                                    "red_threshold=%.6f source=panic_fill_flatten",
                                    pside,
                                    symbol,
                                    int(flatten_ts),
                                    float(metrics["drawdown_raw"]),
                                    float(metrics["drawdown_ema"]),
                                    float(metrics["drawdown_score"]),
                                    float(metrics["red_threshold"]),
                                )
                                boundary_marker = None
                            if boundary_marker is not None or bool(
                                metrics.get("red_seen_in_episode")
                            ):
                                stop_ts = int(flatten_ts)
                                stop_source = (
                                    "panic_fill_flatten"
                                    if boundary_marker is not None
                                    else "red_episode_flatten"
                                )
                                stop_abs_realized = boundary_abs_realized
                                break

                            # RED-free episodes still end at every zero
                            # crossing. Reset before evaluating a later
                            # re-entry/flatten inside the same replay minute.
                            state["pnl_reset_timestamp_ms"] = int(flatten_ts) + 1
                            reset_baseline_realized = boundary_abs_realized
                            self._equity_hard_stop_reset_coin_after_restart(
                                pside, symbol
                            )
                            state = self._hsl_coin_state(pside, symbol)
                            reset_rolling_window()
                            logging.info(
                                "[risk] HSL[%s:%s] replay reset current episode after flat fill | flat_ts=%s",
                                pside,
                                symbol,
                                int(flatten_ts),
                            )
                        if stop_ts is None:
                            continue
                    else:
                        peak_realized, window_last_realized = rolling_realized_at(
                            ts, abs_realized
                        )
                    if stop_ts is None:
                        if not has_unrealized and require_coin_timeline_value:
                            if replay_ambiguous or replay_position_size > flat_epsilon:
                                if not require_coin_timeline_fields:
                                    continue
                                if source_row is not None:
                                    _equity_hard_stop_history_coin_value(
                                        source_row,
                                        "unrealized_pnl_by_coin_pside",
                                        symbol,
                                        pside,
                                        require_key=True,
                                        require_value=True,
                                    )
                                raise ValueError(
                                    "coin HSL replay missing required "
                                    "unrealized_pnl_by_coin_pside value for "
                                    f"{pside}:{symbol} at {ts}"
                                )
                            current_upnl = 0.0
                        else:
                            current_upnl = float(unrealized_value)
                        self._equity_hard_stop_prime_coin_runtime_for_replay(
                            pside, symbol, ts
                        )
                        metrics = self._equity_hard_stop_apply_coin_metrics_sample(
                            pside,
                            symbol,
                            ts,
                            row_balance,
                            peak_realized,
                            window_last_realized,
                            current_upnl,
                            latch_red=False,
                        )
                        applied_rows += 1
                        rows += 1
                        pair_rows_applied[(pside, symbol)] = int(applied_rows)
                        if marker is None:
                            continue
                        stop_ts = int(marker["timestamp"])
                        stop_source = "panic_fill_flatten"
                        if not _equity_hard_stop_replay_marker_confirms_red(metrics):
                            ignored_panic_marker_timestamps.add(stop_ts)
                            logging.warning(
                                "[risk] HSL[%s:%s] ignored historical coin panic marker without reconstructed RED | "
                                "stop_ts=%s drawdown_raw=%.6f drawdown_ema=%.6f drawdown_score=%.6f "
                                "red_threshold=%.6f source=panic_fill_flatten",
                                pside,
                                symbol,
                                stop_ts,
                                float(metrics["drawdown_raw"]),
                                float(metrics["drawdown_ema"]),
                                float(metrics["drawdown_score"]),
                                float(metrics["red_threshold"]),
                            )
                            continue
                    stop_drawdown_raw = float(metrics["drawdown_raw"])
                    finalization = self._equity_hard_stop_red_episode_finalization(
                        pside,
                        {
                            "equity": float(metrics["equity"]),
                            "peak_strategy_equity": float(metrics["peak_strategy_equity"]),
                            "drawdown_ema": float(metrics["drawdown_ema"]),
                        },
                        stop_ts,
                        symbol=symbol,
                    )
                    no_restart_latched = bool(finalization["no_restart_latched"])
                    cooldown_until_ms = finalization["cooldown_until_ms"]
                    state["last_stop_event"] = self._equity_hard_stop_build_latch_payload(
                        pside,
                        symbol=symbol,
                        stop_event_timestamp_ms=stop_ts,
                        balance=float(metrics["balance"]),
                        realized_pnl_total=row_realized_pnl,
                        realized_pnl=float(metrics["realized_pnl"]),
                        unrealized_pnl=float(metrics["unrealized_pnl"]),
                        strategy_pnl=float(metrics["strategy_pnl"]),
                        peak_strategy_pnl=float(metrics["peak_strategy_pnl"]),
                        strategy_equity=float(metrics["strategy_equity"]),
                        peak_strategy_equity=float(metrics["peak_strategy_equity"]),
                        trigger_peak_strategy_equity=float(metrics["peak_strategy_equity"]),
                        drawdown_raw=stop_drawdown_raw,
                        drawdown_ema=float(metrics["drawdown_ema"]),
                        drawdown_score=float(metrics["drawdown_score"]),
                        no_restart_latched=no_restart_latched,
                        cooldown_until_ms=cooldown_until_ms,
                        no_restart_peak_strategy_equity=float(
                            finalization["no_restart_peak_strategy_equity"]
                        ),
                        no_restart_drawdown_raw=float(
                            finalization["no_restart_drawdown_raw"]
                        ),
                    )
                    state["pnl_reset_timestamp_ms"] = stop_ts + 1
                    state["pending_red_since_ms"] = None
                    reset_baseline_realized = stop_abs_realized
                    reset_rolling_window()
                    if no_restart_latched:
                        state["halted"] = True
                        state["no_restart_latched"] = True
                        state["cooldown_until_ms"] = None
                        self._equity_hard_stop_write_latch(
                            pside, state["last_stop_event"], symbol=symbol
                        )
                        self._equity_hard_stop_set_coin_runtime_forced_mode(
                            pside, symbol, "graceful_stop"
                        )
                        logging.critical(
                            "[risk] HSL[%s:%s] reconstructed terminal coin RED stop from exchange-derived history | "
                            "stop_ts=%s drawdown_raw=%.6f source=%s",
                            pside,
                            symbol,
                            stop_ts,
                            stop_drawdown_raw,
                            stop_source,
                        )
                        break
                    state["halted"] = True
                    state["no_restart_latched"] = False
                    state["cooldown_until_ms"] = cooldown_until_ms
                    self._equity_hard_stop_write_latch(
                        pside, state["last_stop_event"], symbol=symbol
                    )
                    if cooldown_until_ms is not None and now_ms < cooldown_until_ms:
                        self._equity_hard_stop_set_coin_runtime_forced_mode(
                            pside, symbol, "graceful_stop"
                        )
                        logging.critical(
                            "[risk] HSL[%s:%s] reconstructed active coin RED cooldown from exchange-derived history | "
                            "remaining_time=%s source=%s",
                            pside,
                            symbol,
                            _equity_hard_stop_format_remaining_time(
                                (cooldown_until_ms - now_ms) / 1000.0
                            ),
                            stop_source,
                        )
                        continue
                    self._equity_hard_stop_reset_coin_after_restart(pside, symbol)
                    self._equity_hard_stop_remove_latch_file(pside, symbol=symbol)
                    state = self._hsl_coin_state(pside, symbol)
                if (
                    not state["halted"]
                    and contract["latest_panic_ts"] is not None
                    and int(contract["latest_panic_ts"]) not in ignored_panic_marker_timestamps
                    and contract["active_cooldown_now"]
                    and not state["no_restart_latched"]
                    and not (
                        contract["policy"] == "normal"
                        and contract["intervention_entry_ts"] is not None
                    )
                ):
                    state["halted"] = True
                    state["cooldown_until_ms"] = contract["cooldown_until_ms"]
                    state["cooldown_intervention_active"] = bool(contract["intervention_active"])
                    state["cooldown_unresolved_residue"] = bool(contract["unresolved_residue"])
                    if state["last_stop_event"] is None:
                        state["last_stop_event"] = {
                            "stop_event_timestamp_ms": int(contract["latest_panic_ts"]),
                            "cooldown_until_ms": contract["cooldown_until_ms"],
                            "no_restart_latched": False,
                            "symbol": symbol,
                        }
                if state["halted"] and not state["no_restart_latched"]:
                    cooldown_until_ms = state["cooldown_until_ms"]
                    if cooldown_until_ms is not None and now_ms >= cooldown_until_ms:
                        self._equity_hard_stop_reset_coin_after_restart(pside, symbol)
                        self._equity_hard_stop_remove_latch_file(pside, symbol=symbol)
                        state = self._hsl_coin_state(pside, symbol)
                        logging.info(
                            "[risk] HSL[%s:%s] replayed cooldown already elapsed; resumed",
                            pside,
                            symbol,
                        )
                    elif cooldown_until_ms is not None:
                        state["cooldown_intervention_active"] = bool(contract["intervention_active"])
                        state["cooldown_unresolved_residue"] = bool(contract["unresolved_residue"])
                        if state["cooldown_unresolved_residue"]:
                            mode = (
                                "panic"
                                if self._equity_hard_stop_has_open_position_symbol(pside, symbol)
                                else "graceful_stop"
                            )
                        elif contract["intervention_active"] and contract["policy"] == "panic":
                            mode = (
                                "panic"
                                if self._equity_hard_stop_has_open_position_symbol(pside, symbol)
                                else "graceful_stop"
                            )
                        elif contract["intervention_active"] and contract["policy"] == "manual":
                            mode = (
                                "manual"
                                if self._equity_hard_stop_has_open_position_symbol(pside, symbol)
                                else "graceful_stop"
                            )
                        elif contract["intervention_active"] and contract["policy"] == "tp_only":
                            mode = (
                                "tp_only_with_active_entry_cancellation"
                                if self._equity_hard_stop_has_open_position_symbol(pside, symbol)
                                else "graceful_stop"
                            )
                        else:
                            mode = "graceful_stop"
                        self._equity_hard_stop_set_coin_runtime_forced_mode(pside, symbol, mode)
                        reason = (
                            " unresolved_panic_residue"
                            if state["cooldown_unresolved_residue"]
                            else (
                                f" intervention_policy={contract['policy']}"
                                if contract["intervention_active"]
                                else ""
                            )
                        )
                        logging.critical(
                            "[risk] HSL[%s:%s] reconstructed active coin RED cooldown from "
                            "exchange-derived history | remaining_time=%s%s",
                            pside,
                            symbol,
                            _equity_hard_stop_format_remaining_time(
                                (cooldown_until_ms - now_ms) / 1000.0
                            ),
                            reason,
                        )
                if state["halted"]:
                    pair_rows_applied[(pside, symbol)] = int(applied_rows)
                    mark_pair_ready(pside, symbol)
                    log_replay_progress(
                        pair_idx,
                        pside,
                        symbol,
                        applied_rows,
                        scanned_rows,
                        pair_started_s,
                        force=True,
                    )
                    continue
                if applied_rows == 0:
                    self._equity_hard_stop_prime_coin_runtime_for_replay(pside, symbol, now_ms)
                check_shutdown("hsl_coin_history_replay_current_sample")
                current_metrics = self._equity_hard_stop_apply_coin_sample(
                    pside,
                    symbol,
                    now_ms,
                    balance,
                    float(await self._calc_upnl_sum_strict(pside, symbol)),
                )
                if current_metrics["tier"] == "red":
                    self._equity_hard_stop_activate_coin_red_from_metrics(
                        pside,
                        symbol,
                        current_metrics,
                    )
                pair_rows_applied[(pside, symbol)] = int(applied_rows)
                mark_pair_ready(pside, symbol)
                log_replay_progress(
                    pair_idx,
                    pside,
                    symbol,
                    applied_rows,
                    scanned_rows,
                    pair_started_s,
                    force=True,
                )
            if batch_idx == 0:
                mark_protective_ready()
        self._equity_hard_stop_coin_initialized = True
        elapsed_s = max(0.0, time.monotonic() - replay_started_s)
        total_elapsed_s = max(0.0, time.monotonic() - initialization_started_s)
        rows_per_second = float(rows) / elapsed_s if elapsed_s > 0.0 else None
        scanned_rows_per_second = (
            float(total_scanned_rows) / elapsed_s if elapsed_s > 0.0 else None
        )
        skipped_pairs = sum(1 for pair in active_pairs if pair_rows_applied.get(pair, 0) == 0)
        dense_equivalent_rows = int(replay_row_count * len(active_pairs))
        candidate_rows = int(sum(pair_candidate_rows.values()))
        candidate_reduction_pct = (
            0.0
            if dense_equivalent_rows <= 0
            else max(
                0.0,
                (dense_equivalent_rows - candidate_rows)
                / dense_equivalent_rows
                * 100.0,
            )
        )
        replay_strategy = "dense_timeline"
        if compact_replay is not None:
            if sparse_replay_pairs > 0 and dense_replay_pairs > 0:
                replay_strategy = "mixed"
            elif sparse_replay_pairs > 0:
                replay_strategy = "sparse_change_points"
            else:
                replay_strategy = "dense_compact"
        logging.info(
            "[risk] HSL coin history reconstruction completed | rows=%d pairs=%d elapsed=%.1fs",
            rows,
            len(active_pairs),
            elapsed_s,
        )
        _emit_hsl_replay_event(
            self,
            EventTypes.HSL_REPLAY_COMPLETED,
            {
                "signal_mode": "coin",
                "stage": "full_replay",
                "history_format": (
                    "compact" if compact_replay is not None else "timeline"
                ),
                "replay_strategy": replay_strategy,
                "rows": int(rows),
                "applied_rows": int(rows),
                "total_applied_rows": int(rows),
                "total_scanned_rows": int(total_scanned_rows),
                "candidate_rows": candidate_rows,
                "dense_equivalent_rows": dense_equivalent_rows,
                "candidate_reduction_pct": round(candidate_reduction_pct, 3),
                "dense_replay_pairs": int(dense_replay_pairs),
                "dense_fallback_pairs": int(dense_fallback_pairs),
                "sparse_replay_pairs": int(sparse_replay_pairs),
                "pairs": len(active_pairs),
                "held_pairs": len(active_held_pairs),
                "cooldown_pairs": len(active_panic_pairs),
                "required_pairs": len(active_required_pairs),
                "skipped_pairs": int(skipped_pairs),
                "timeline_rows": replay_row_count,
                "fill_events": len(fill_events),
                "panic_events": len(panic_flatten_events),
                "rows_per_second": round(rows_per_second, 3)
                if rows_per_second is not None
                else None,
                "scanned_rows_per_second": round(scanned_rows_per_second, 3)
                if scanned_rows_per_second is not None
                else None,
                "history_fetch_elapsed_s": round(history_fetch_elapsed_s, 3),
                "pre_replay_elapsed_s": round(pre_replay_elapsed_s, 3),
                "replay_loop_elapsed_s": round(elapsed_s, 3),
                "full_elapsed_s": round(total_elapsed_s, 3),
                "startup_blocking_elapsed_s": round(
                    protective_ready_elapsed_s
                    if protective_ready_elapsed_s is not None
                    else total_elapsed_s,
                    3,
                ),
                "protective_elapsed_s": round(
                    protective_ready_elapsed_s
                    if protective_ready_elapsed_s is not None
                    else total_elapsed_s,
                    3,
                ),
                "elapsed_s": round(total_elapsed_s, 3),
            },
            status="succeeded",
            reason_code="coin_history_replay_completed",
        )
    except asyncio.CancelledError:
        self._equity_hard_stop_coin_replay_failure = "shutdown_cancelled"
        ready_event.set()
        _emit_hsl_replay_event(
            self,
            EventTypes.HSL_REPLAY_FAILED,
            {
                "signal_mode": "coin",
                "elapsed_s": round(time.monotonic() - initialization_started_s, 3),
                "history_fetch_elapsed_s": round(history_fetch_elapsed_s, 3)
                if "history_fetch_elapsed_s" in locals()
                else None,
                "pre_replay_elapsed_s": round(pre_replay_elapsed_s, 3)
                if "pre_replay_elapsed_s" in locals()
                else None,
                "replay_loop_elapsed_s": round(time.monotonic() - replay_started_s, 3)
                if "replay_started_s" in locals()
                else None,
            },
            status="failed",
            reason_code="shutdown_cancelled",
        )
        raise
    except Exception as exc:
        self._equity_hard_stop_coin_replay_failure = _bounded_hsl_exception_type(exc)
        ready_event.set()
        _emit_hsl_replay_event(
            self,
            EventTypes.HSL_REPLAY_FAILED,
            {
                "signal_mode": "coin",
                "error_type": _bounded_hsl_exception_type(exc),
                "elapsed_s": round(time.monotonic() - initialization_started_s, 3),
                "history_fetch_elapsed_s": round(history_fetch_elapsed_s, 3)
                if "history_fetch_elapsed_s" in locals()
                else None,
                "pre_replay_elapsed_s": round(pre_replay_elapsed_s, 3)
                if "pre_replay_elapsed_s" in locals()
                else None,
                "replay_loop_elapsed_s": round(time.monotonic() - replay_started_s, 3)
                if "replay_started_s" in locals()
                else None,
            },
            level="warning",
            status="failed",
            reason_code="coin_history_replay_failed",
        )
        raise
    finally:
        if (
            not watchdog_context_restored
            and hasattr(self, "_set_log_silence_watchdog_context")
        ):
            self._set_log_silence_watchdog_context(phase=prev_phase, stage=prev_stage)


async def _equity_hard_stop_start_coin_history_replay(self) -> None:
    if getattr(self, "_equity_hard_stop_coin_initialized", False):
        return
    if getattr(self, "_equity_hard_stop_coin_protective_ready", False):
        return
    task = getattr(self, "_equity_hard_stop_coin_replay_task", None)
    if task is None or task.done():
        ready_event = asyncio.Event()
        self._equity_hard_stop_coin_replay_ready_event = ready_event

        async def run_replay() -> None:
            try:
                await self._equity_hard_stop_initialize_coin_from_history()
            except asyncio.CancelledError:
                raise
            except Exception:
                # The initializer already emitted/stored the failure. After
                # protective readiness, keep held-pair management alive while
                # unreplayed flat pairs remain explicitly entry-blocked.
                if not getattr(
                    self, "_equity_hard_stop_coin_protective_ready", False
                ):
                    raise

        task = asyncio.create_task(run_replay(), name="hsl_coin_replay")
        self._equity_hard_stop_coin_replay_task = task
        if not isinstance(getattr(self, "maintainers", None), dict):
            self.maintainers = {}
        self.maintainers["hsl_coin_replay"] = task
    ready_event = self._equity_hard_stop_coin_replay_ready_event
    await ready_event.wait()
    if not getattr(self, "_equity_hard_stop_coin_protective_ready", False):
        await task
        failure = getattr(self, "_equity_hard_stop_coin_replay_failure", None)
        raise RuntimeError(
            "coin HSL replay failed before held-position protective readiness"
            + (f": {failure}" if failure else "")
        )


def _equity_hard_stop_log_status(self, pside: str, metrics: dict) -> None:
    state = self._hsl_state(pside)
    now_ms = int(metrics["timestamp_ms"])
    if (
        state["last_status_log_ms"] != 0
        and now_ms - state["last_status_log_ms"] < self._equity_hard_stop_status_log_interval_ms
    ):
        return
    state["last_status_log_ms"] = now_ms
    red_threshold = float(metrics["red_threshold"])
    drawdown_score = float(metrics["drawdown_score"])
    dist_to_red = max(0.0, red_threshold - drawdown_score)
    cooldown_remaining = None
    if state["cooldown_until_ms"] is not None:
        cooldown_remaining = _equity_hard_stop_format_remaining_time(
            max(0.0, (state["cooldown_until_ms"] - now_ms) / 1000.0)
        )
    last_red_ts = None
    if state["last_stop_event"] is not None:
        last_red_ts = state["last_stop_event"].get("stop_event_timestamp_ms")
    if last_red_ts is None:
        last_red_ts = state["pending_red_since_ms"]
    logging.info(
        "[risk] HSL[%s] status | tier=%s dist_to_red=%.6f drawdown_raw=%.6f drawdown_ema=%.6f "
        "drawdown_score=%.6f red_threshold=%.6f cooldown_remaining=%s last_red_ts=%s "
        "pending_red_since_ms=%s peak_strategy_equity=%.6f rolling_peak_strategy_equity=%.6f",
        pside,
        metrics["tier"],
        dist_to_red,
        metrics["drawdown_raw"],
        metrics["drawdown_ema"],
        drawdown_score,
        red_threshold,
        cooldown_remaining if cooldown_remaining is not None else "none",
        last_red_ts if last_red_ts is not None else "none",
        state["pending_red_since_ms"] if state["pending_red_since_ms"] is not None else "none",
        metrics["peak_strategy_equity"],
        metrics["rolling_peak_strategy_equity"],
    )
    _emit_hsl_event(
        self,
        "hsl.status",
        ("hsl", "risk", "status"),
        _hsl_event_data(
            metrics,
            {
                "dist_to_red": float(dist_to_red),
                "cooldown_remaining": cooldown_remaining,
                "last_red_ts": last_red_ts,
                "pending_red_since_ms": state["pending_red_since_ms"],
            },
        ),
        pside=pside,
        ts=now_ms,
        status="succeeded",
        reason_code=str(metrics["tier"]),
    )


async def _equity_hard_stop_check(self) -> Optional[dict]:
    if not self._equity_hard_stop_enabled():
        return None
    if self._equity_hard_stop_signal_mode() == "coin":
        if not (
            getattr(self, "_equity_hard_stop_coin_initialized", False)
            or getattr(self, "_equity_hard_stop_coin_protective_ready", False)
        ):
            await self._equity_hard_stop_initialize_coin_from_history()
        return await self._equity_hard_stop_check_coin()
    if not all(
        self._equity_hard_stop_runtime_initialized(pside)
        or not self._equity_hard_stop_enabled(pside)
        for pside in self._hsl_psides()
    ):
        await self._equity_hard_stop_initialize_from_history()
    balance = self.get_raw_balance()
    ts_ms = int(self.get_exchange_time())
    signal_mode = self._equity_hard_stop_signal_mode()
    realized_pnl_total = self._equity_hard_stop_realized_pnl_now()
    unrealized_pnl_by_pside = {
        pside: await self._calc_upnl_sum_strict(pside) for pside in self._hsl_psides()
    }
    unrealized_pnl_total = (
        float(sum(float(v) for v in unrealized_pnl_by_pside.values()))
        if signal_mode == "unified"
        else None
    )
    out = {}
    for pside in self._hsl_psides():
        if not self._equity_hard_stop_enabled(pside):
            continue
        state = self._hsl_state(pside)
        if state["halted"]:
            if await self._equity_hard_stop_handle_position_during_cooldown(pside, ts_ms):
                state = self._hsl_state(pside)
            if state["halted"]:
                cooldown_until_ms = state["cooldown_until_ms"]
                if (
                    not state["no_restart_latched"]
                    and not state["cooldown_repanic_reset_pending"]
                    and cooldown_until_ms is not None
                    and ts_ms >= cooldown_until_ms
                ):
                    self._equity_hard_stop_reset_after_restart(pside)
                    self._equity_hard_stop_remove_latch_file(pside)
                    logging.info("[risk] HSL[%s] RED cooldown elapsed; trading resumed", pside)
                    _emit_hsl_event(
                        self,
                        "hsl.cooldown_ended",
                        ("hsl", "risk", "cooldown"),
                        {"reason": "elapsed"},
                        pside=pside,
                        ts=ts_ms,
                        status="succeeded",
                        reason_code="elapsed",
                    )
                    state = self._hsl_state(pside)
                else:
                    self._equity_hard_stop_log_cooldown_status(pside, ts_ms)
                    continue
        prev_latched = self._equity_hard_stop_runtime_red_latched(pside)
        prev_tier = self._equity_hard_stop_runtime_tier(pside)
        metrics = self._equity_hard_stop_apply_sample(
            pside,
            ts_ms,
            float(balance),
            float(realized_pnl_total),
            float(self._equity_hard_stop_realized_pnl_now(pside)),
            float(unrealized_pnl_by_pside[pside]),
            unrealized_pnl_total=unrealized_pnl_total,
        )
        if metrics["changed"]:
            self._equity_hard_stop_log_transition(pside, metrics, prev_tier)
        self._equity_hard_stop_maybe_emit_raw_red_pending(pside, metrics)
        if metrics["tier"] == "red" and not prev_latched:
            state["pending_red_since_ms"] = int(metrics["timestamp_ms"])
            state["pending_stop_event"] = None
            logging.critical(
                "[risk] HSL[%s] RED triggered | strategy_equity=%.6f peak_strategy_equity=%.6f rolling_peak_strategy_equity=%.6f drawdown_score=%.6f red_threshold=%.6f",
                pside,
                metrics["strategy_equity"],
                metrics["peak_strategy_equity"],
                metrics["rolling_peak_strategy_equity"],
                metrics["drawdown_score"],
                metrics["red_threshold"],
            )
            _emit_hsl_red_triggered_once(
                self,
                state,
                _hsl_event_data(metrics),
                pside=pside,
                ts=int(metrics["timestamp_ms"]),
            )
        elif metrics["tier"] != "red" and not self._equity_hard_stop_runtime_red_latched(
            pside
        ):
            state["pending_red_since_ms"] = None
            state["red_trigger_event_emitted"] = False
        self._equity_hard_stop_log_status(pside, metrics)
        out[pside] = metrics
    self._equity_hard_stop_refresh_halted_runtime_forced_modes()
    return out if out else None


def _equity_hard_stop_coin_symbols(self) -> set[str]:
    symbols = set(self.positions.keys())
    for pside_states in getattr(self, "_equity_hard_stop_coin", {}).values():
        symbols.update(pside_states.keys())
    if self._pnls_manager is not None:
        lookback_ms = self._equity_hard_stop_lookback_ms()
        now_ms = int(self.get_exchange_time())
        start_ms = None if lookback_ms is None else now_ms - int(lookback_ms)
        for event in self._pnls_manager.get_events():
            ts = _equity_hard_stop_fill_timestamp_ms(event)
            if start_ms is not None and ts < start_ms:
                continue
            symbol = _equity_hard_stop_fill_symbol(event)
            if symbol:
                symbols.add(symbol)
    return {str(symbol) for symbol in symbols if symbol}


def _equity_hard_stop_reset_coin_after_restart(self, pside: str, symbol: str) -> None:
    state = self._hsl_coin_state(pside, symbol)
    reset_ts = state.get("pnl_reset_timestamp_ms")
    no_restart_peak_strategy_equity = float(
        state.get("no_restart_peak_strategy_equity", 0.0) or 0.0
    )
    state.clear()
    state.update(self._equity_hard_stop_make_state())
    state["pnl_reset_timestamp_ms"] = reset_ts
    state["no_restart_peak_strategy_equity"] = no_restart_peak_strategy_equity
    self._equity_hard_stop_clear_coin_runtime_forced_mode(pside, symbol)


def _equity_hard_stop_log_coin_cooldown_status(self, pside: str, symbol: str, now_ms: int) -> None:
    state = self._hsl_coin_state(pside, symbol)
    cooldown_until_ms = state["cooldown_until_ms"]
    if cooldown_until_ms is None or now_ms >= cooldown_until_ms:
        return
    if (
        state["last_cooldown_log_ms"] != 0
        and now_ms - state["last_cooldown_log_ms"] < self._equity_hard_stop_cooldown_log_interval_ms
    ):
        return
    state["last_cooldown_log_ms"] = now_ms
    logging.info(
        "[risk] HSL[%s:%s] RED cooldown active | remaining_time=%s",
        pside,
        symbol,
        _equity_hard_stop_format_remaining_time((cooldown_until_ms - now_ms) / 1000.0),
    )
    remaining_seconds = max(0.0, (cooldown_until_ms - now_ms) / 1000.0)
    _emit_hsl_event(
        self,
        "hsl.status",
        ("hsl", "risk", "status"),
        {
            "tier": "red",
            "cooldown_until_ms": int(cooldown_until_ms),
            "cooldown_remaining_seconds": float(remaining_seconds),
        },
        pside=pside,
        symbol=symbol,
        ts=now_ms,
        status="degraded",
        reason_code="cooldown_active",
    )


def _equity_hard_stop_emit_coin_status(self, pside: str, symbol: str, metrics: dict) -> None:
    try:
        state = self._hsl_coin_state(pside, symbol)
        now_ms = int(metrics["timestamp_ms"])
        if (
            state["last_status_log_ms"] != 0
            and now_ms - state["last_status_log_ms"]
            < self._equity_hard_stop_status_log_interval_ms
        ):
            return
        state["last_status_log_ms"] = now_ms
        red_threshold = float(metrics["red_threshold"])
        drawdown_score = float(metrics["drawdown_score"])
        dist_to_red = max(0.0, red_threshold - drawdown_score)
        cooldown_remaining = None
        if state["cooldown_until_ms"] is not None:
            cooldown_remaining = _equity_hard_stop_format_remaining_time(
                max(0.0, (state["cooldown_until_ms"] - now_ms) / 1000.0)
            )
        last_red_ts = None
        if state["last_stop_event"] is not None:
            last_red_ts = state["last_stop_event"].get("stop_event_timestamp_ms")
        if last_red_ts is None:
            last_red_ts = state["pending_red_since_ms"]
        has_open_position = self._equity_hard_stop_has_open_position_symbol(pside, symbol)
        if has_open_position:
            try:
                logging.info(
                    "[risk] HSL[%s:%s] status | tier=%s dist_to_red=%.6f drawdown_raw=%.6f "
                    "drawdown_ema=%.6f drawdown_score=%.6f red_threshold=%.6f "
                    "cooldown_remaining=%s last_red_ts=%s pending_red_since_ms=%s "
                    "slot_budget=%.6f realized_pnl=%.6f peak_realized_pnl=%.6f upnl=%.6f",
                    pside,
                    symbol,
                    metrics["tier"],
                    dist_to_red,
                    metrics["drawdown_raw"],
                    metrics["drawdown_ema"],
                    drawdown_score,
                    red_threshold,
                    cooldown_remaining if cooldown_remaining is not None else "none",
                    last_red_ts if last_red_ts is not None else "none",
                    (
                        state["pending_red_since_ms"]
                        if state["pending_red_since_ms"] is not None
                        else "none"
                    ),
                    metrics["slot_budget"],
                    metrics["realized_pnl"],
                    metrics["peak_realized_pnl"],
                    metrics["unrealized_pnl"],
                )
            except Exception as exc:
                logging.debug(
                    "[event] failed to log HSL coin status symbol=%s pside=%s: %s",
                    symbol,
                    pside,
                    _bounded_hsl_exception_type(exc),
                )
        _emit_hsl_event(
            self,
            "hsl.status",
            ("hsl", "risk", "status"),
            _hsl_event_data(
                metrics,
                {
                    "dist_to_red": float(dist_to_red),
                    "cooldown_remaining": cooldown_remaining,
                    "last_red_ts": last_red_ts,
                    "pending_red_since_ms": state["pending_red_since_ms"],
                    "has_open_position": bool(has_open_position),
                },
            ),
            pside=pside,
            symbol=symbol,
            ts=now_ms,
            status="succeeded",
            reason_code=str(metrics["tier"]),
        )
    except Exception as exc:
        logging.debug(
            "[event] failed to emit HSL coin status symbol=%s pside=%s: %s",
            symbol,
            pside,
            _bounded_hsl_exception_type(exc),
        )


async def _equity_hard_stop_check_coin(self) -> Optional[dict]:
    balance = float(self.get_raw_balance())
    ts_ms = int(self.get_exchange_time())
    out = {}
    symbols = sorted(self._equity_hard_stop_coin_symbols())
    partial_replay = (
        getattr(self, "_equity_hard_stop_coin_protective_ready", False)
        and not getattr(self, "_equity_hard_stop_coin_initialized", False)
    )
    ready_pairs = set(
        getattr(self, "_equity_hard_stop_coin_replay_ready_pairs", set()) or set()
    )
    pending_pairs = set(
        getattr(self, "_equity_hard_stop_coin_replay_pending_pairs", set()) or set()
    )
    if partial_replay:
        newly_held_pending = sorted(
            pair
            for pair in pending_pairs
            if self._equity_hard_stop_has_open_position_symbol(*pair)
        )
        if newly_held_pending:
            rendered = ",".join(f"{pside}:{symbol}" for pside, symbol in newly_held_pending)
            raise RestartBotException(
                "coin HSL replay still pending for newly held pair(s); "
                f"restart required for held-first reconstruction: {rendered}"
            )
    for pside in self._hsl_psides():
        if not self._equity_hard_stop_coin_active_pside(pside):
            continue
        for symbol in symbols:
            if not self._equity_hard_stop_coin_active_pside(pside, symbol):
                continue
            if partial_replay and (pside, symbol) not in ready_pairs:
                continue
            state = self._hsl_coin_state(pside, symbol)
            if state["halted"]:
                if await self._equity_hard_stop_handle_coin_position_during_cooldown(
                    pside, symbol, ts_ms
                ):
                    state = self._hsl_coin_state(pside, symbol)
                if state["halted"]:
                    cooldown_until_ms = state["cooldown_until_ms"]
                    if (
                        not state["no_restart_latched"]
                        and not state["cooldown_repanic_reset_pending"]
                        and cooldown_until_ms is not None
                        and ts_ms >= cooldown_until_ms
                    ):
                        self._equity_hard_stop_reset_coin_after_restart(pside, symbol)
                        self._equity_hard_stop_remove_latch_file(pside, symbol=symbol)
                        logging.info(
                            "[risk] HSL[%s:%s] RED cooldown elapsed; trading resumed",
                            pside,
                            symbol,
                        )
                        _emit_hsl_event(
                            self,
                            "hsl.cooldown_ended",
                            ("hsl", "risk", "cooldown"),
                            {"reason": "elapsed", "symbol": symbol},
                            pside=pside,
                            symbol=symbol,
                            ts=ts_ms,
                            status="succeeded",
                            reason_code="elapsed",
                        )
                        state = self._hsl_coin_state(pside, symbol)
                    else:
                        self._equity_hard_stop_log_coin_cooldown_status(pside, symbol, ts_ms)
                        if not state["cooldown_repanic_reset_pending"]:
                            forced_modes = self._runtime_forced_modes.setdefault(pside, {})
                            if symbol not in forced_modes:
                                self._equity_hard_stop_set_coin_runtime_forced_mode(
                                    pside, symbol, "graceful_stop"
                                )
                        continue
            prev_latched = bool(state["runtime"].red_latched())
            prev_tier = str(state["runtime"].tier())
            metrics = self._equity_hard_stop_apply_coin_sample(
                pside,
                symbol,
                ts_ms,
                balance,
                float(await self._calc_upnl_sum_strict(pside, symbol)),
            )
            if metrics["changed"]:
                self._equity_hard_stop_log_transition(pside, metrics, prev_tier)
            self._equity_hard_stop_maybe_emit_raw_red_pending(
                pside, metrics, symbol=symbol
            )
            if metrics["tier"] == "red" and not prev_latched:
                state["pending_red_since_ms"] = int(metrics["timestamp_ms"])
                state["pending_stop_event"] = None
                self._equity_hard_stop_set_coin_runtime_forced_mode(pside, symbol, "panic")
                logging.critical(
                    "[risk] HSL[%s:%s] RED triggered | drawdown_raw=%.6f drawdown_ema=%.6f drawdown_score=%.6f red_threshold=%.6f slot_budget=%.6f realized_pnl=%.6f peak_realized_pnl=%.6f upnl=%.6f",
                    pside,
                    symbol,
                    metrics["drawdown_raw"],
                    metrics["drawdown_ema"],
                    metrics["drawdown_score"],
                    metrics["red_threshold"],
                    metrics["slot_budget"],
                    metrics["realized_pnl"],
                    metrics["peak_realized_pnl"],
                    metrics["unrealized_pnl"],
                )
                _emit_hsl_red_triggered_once(
                    self,
                    state,
                    _hsl_event_data(metrics),
                    pside=pside,
                    symbol=symbol,
                    ts=int(metrics["timestamp_ms"]),
                )
            elif metrics["tier"] != "red" and not state["runtime"].red_latched():
                state["pending_red_since_ms"] = None
                state["red_trigger_event_emitted"] = False
                if not state["halted"]:
                    self._equity_hard_stop_clear_coin_runtime_forced_mode(pside, symbol)
            if metrics["tier"] == "orange":
                target = str(
                    _equity_hard_stop_config(self, pside, symbol)["orange_tier_mode"]
                )
                if target == "graceful_stop":
                    self._equity_hard_stop_set_coin_runtime_forced_mode(
                        pside, symbol, "graceful_stop"
                    )
                elif target == "tp_only_with_active_entry_cancellation":
                    self._equity_hard_stop_set_coin_runtime_forced_mode(
                        pside, symbol, "tp_only_with_active_entry_cancellation"
                    )
            if (
                state["runtime"].red_latched()
                and not state["halted"]
                and not bool(metrics.get("red_active_now", True))
            ):
                # B2.1 red split: the episode saw RED but the current sample
                # recovered, so panic supervision is disengaged. The regular
                # check path owns the paused episode: keep entries blocked and
                # perform the flat-confirmation/finalization the supervisor
                # would otherwise do, so cooldown/no-restart accounting can
                # never be skipped just because RED recovered before flat.
                self._equity_hard_stop_set_coin_runtime_forced_mode(
                    pside, symbol, "tp_only_with_active_entry_cancellation"
                )
                has_position = self._equity_hard_stop_has_open_position_symbol(
                    pside, symbol
                )
                entry_orders, nonpanic_close_orders = (
                    self._equity_hard_stop_count_blocking_open_orders_symbol(
                        pside, symbol
                    )
                )
                if not has_position and entry_orders == 0 and nonpanic_close_orders == 0:
                    stop_ts_ms = await self._equity_hard_stop_flatten_fill_timestamp_with_refresh(
                        pside,
                        ts_ms,
                        symbol=symbol,
                        since_ms=state.get("pending_red_since_ms"),
                    )
                    if stop_ts_ms is not None:
                        state["pending_stop_event"] = (
                            await self._equity_hard_stop_compute_coin_stop_event(
                                pside, symbol, stop_ts_ms
                            )
                        )
                        state["red_flat_confirmations"] += 1
                else:
                    state["red_flat_confirmations"] = 0
                    state["pending_stop_event"] = None
                if state["red_flat_confirmations"] >= 2:
                    await self._equity_hard_stop_finalize_coin_red_stop(
                        pside,
                        symbol,
                        state["pending_stop_event"],
                        finalized_without_order=True,
                        flat_confirmations=state["red_flat_confirmations"],
                        entry_orders=entry_orders,
                        nonpanic_close_orders=nonpanic_close_orders,
                    )
                    state = self._hsl_coin_state(pside, symbol)
            self._equity_hard_stop_emit_coin_status(pside, symbol, metrics)
            out[f"{pside}:{symbol}"] = metrics
    return out if out else None


def _equity_hard_stop_coin_needs_panic_supervision(
    self, pside: str, symbol: str, state: dict[str, Any]
) -> bool:
    if not self._equity_hard_stop_enabled(pside, symbol=symbol):
        return False
    if state["runtime"].red_latched() and not state["halted"]:
        # B2.1 contract (clarified 2026-07-06): only the CURRENT sample being
        # in RED may authorize new panic orders. A latched tier with a
        # recovered sample keeps the episode's entry blocking (forced-mode
        # downgrade handled by the supervisor) but must not keep submitting
        # panic orders.
        last_metrics = state.get("last_metrics")
        if last_metrics is None:
            # No sample yet against the latched state: stay protective until
            # the next sample proves recovery.
            return True
        return bool(last_metrics.get("red_active_now", True))
    return bool(state["halted"] and state["cooldown_repanic_reset_pending"])


def _equity_hard_stop_coin_red_active(self) -> bool:
    for pside in self._hsl_psides():
        pside_states = getattr(self, "_equity_hard_stop_coin")[pside]
        for symbol, state in pside_states.items():
            if self._equity_hard_stop_coin_needs_panic_supervision(pside, symbol, state):
                return True
    return False


def _equity_hard_stop_set_red_runtime_forced_modes(self, pside: str) -> None:
    previous = dict(getattr(self, "_runtime_forced_modes", {}).get(pside, {}) or {})
    forced = {}
    symbols = set(self.positions.keys()) | set(self.open_orders.keys()) | set(self.active_symbols)
    for symbol in symbols:
        forced[symbol] = "panic"
    self._runtime_forced_modes[pside] = forced
    if previous != forced:
        _emit_runtime_forced_mode_changed_event(
            self,
            pside=pside,
            action="replace",
            symbols=forced.keys(),
            previous_modes=previous,
            modes=forced,
            reason_code="hsl_red_runtime_forced_modes",
        )


def _equity_hard_stop_set_red_paused_runtime_forced_modes(self, pside: str) -> None:
    """Episode entry blocking without panic emission (B2.1 red split).

    Used while a red-seen episode is still open but the current sample is no
    longer in RED: entries stay cancelled/blocked, normal closes may proceed,
    and no new panic orders are authorized.
    """
    previous = dict(getattr(self, "_runtime_forced_modes", {}).get(pside, {}) or {})
    forced = {}
    symbols = set(self.positions.keys()) | set(self.open_orders.keys()) | set(self.active_symbols)
    for symbol in symbols:
        forced[symbol] = "tp_only_with_active_entry_cancellation"
    self._runtime_forced_modes[pside] = forced
    if previous != forced:
        _emit_runtime_forced_mode_changed_event(
            self,
            pside=pside,
            action="replace",
            symbols=forced.keys(),
            previous_modes=previous,
            modes=forced,
            reason_code="hsl_red_paused_runtime_forced_modes",
        )


def _equity_hard_stop_set_coin_runtime_forced_mode(
    self, pside: str, symbol: str, mode: str
) -> None:
    forced_modes = self._runtime_forced_modes.setdefault(pside, {})
    previous = forced_modes.get(symbol)
    forced_modes[symbol] = mode
    if previous != mode:
        _emit_runtime_forced_mode_changed_event(
            self,
            pside=pside,
            symbol=symbol,
            action="set",
            previous_mode=previous,
            mode=mode,
            reason_code="hsl_runtime_forced_mode_set",
        )


def _equity_hard_stop_clear_coin_runtime_forced_mode(self, pside: str, symbol: str) -> None:
    forced_modes = self._runtime_forced_modes.setdefault(pside, {})
    previous = forced_modes.pop(symbol, None)
    if previous is not None:
        _emit_runtime_forced_mode_changed_event(
            self,
            pside=pside,
            symbol=symbol,
            action="clear",
            previous_mode=previous,
            reason_code="hsl_runtime_forced_mode_clear",
        )


def _equity_hard_stop_clear_runtime_forced_modes(self, pside: Optional[str] = None) -> None:
    if pside is None:
        previous_by_pside = {
            side: dict(modes or {})
            for side, modes in (getattr(self, "_runtime_forced_modes", {}) or {}).items()
        }
        self._runtime_forced_modes = {"long": {}, "short": {}}
        for side in self._hsl_psides():
            previous = previous_by_pside.get(side, {})
            if previous:
                _emit_runtime_forced_mode_changed_event(
                    self,
                    pside=side,
                    action="clear_all",
                    symbols=previous.keys(),
                    previous_modes=previous,
                    modes={},
                    reason_code="hsl_runtime_forced_modes_clear_all",
                )
        return
    previous = dict(getattr(self, "_runtime_forced_modes", {}).get(pside, {}) or {})
    self._runtime_forced_modes[pside] = {}
    if previous:
        _emit_runtime_forced_mode_changed_event(
            self,
            pside=pside,
            action="clear_all",
            symbols=previous.keys(),
            previous_modes=previous,
            modes={},
            reason_code="hsl_runtime_forced_modes_clear_all",
        )


def _equity_hard_stop_count_open_positions(self, pside: str) -> int:
    n_positions = 0
    for pos in self.positions.values():
        if float(pos.get(pside, {}).get("size", 0.0) or 0.0) != 0.0:
            n_positions += 1
    return n_positions


def _equity_hard_stop_has_open_position_symbol(self, pside: str, symbol: str) -> bool:
    pos = self.positions.get(symbol, {})
    return float(pos.get(pside, {}).get("size", 0.0) or 0.0) != 0.0


def _equity_hard_stop_count_blocking_open_orders(self, pside: str) -> tuple[int, int]:
    entry_orders = 0
    nonpanic_close_orders = 0
    for orders in self.open_orders.values():
        for order in orders:
            if str(order.get("position_side", "long")).lower() != pside:
                continue
            reduce_only = bool(order.get("reduce_only") or order.get("reduceOnly"))
            if not reduce_only:
                entry_orders += 1
                continue
            pb_type = self._resolve_pb_order_type(order).lower()
            if "panic" not in pb_type:
                nonpanic_close_orders += 1
    return entry_orders, nonpanic_close_orders


def _equity_hard_stop_count_blocking_open_orders_symbol(
    self, pside: str, symbol: str
) -> tuple[int, int]:
    entry_orders = 0
    nonpanic_close_orders = 0
    for order in self.open_orders.get(symbol, []):
        if str(order.get("position_side", "long")).lower() != pside:
            continue
        reduce_only = bool(order.get("reduce_only") or order.get("reduceOnly"))
        if not reduce_only:
            entry_orders += 1
            continue
        pb_type = self._resolve_pb_order_type(order).lower()
        if "panic" not in pb_type:
            nonpanic_close_orders += 1
    return entry_orders, nonpanic_close_orders


def _equity_hard_stop_log_red_progress(
    self,
    pside: str,
    n_positions: int,
    entry_orders: int,
    nonpanic_close_orders: int,
    flat_confirmations: int,
) -> None:
    state = self._hsl_state(pside)
    progress = (n_positions, entry_orders, nonpanic_close_orders, flat_confirmations)
    if progress == state["last_red_progress"]:
        return
    state["last_red_progress"] = progress
    logging.info(
        "[risk] HSL[%s] RED supervisor progress | positions=%d entry_orders=%d "
        "nonpanic_close_orders=%d flat_confirmations=%d/2",
        pside,
        n_positions,
        entry_orders,
        nonpanic_close_orders,
        flat_confirmations,
    )


async def _equity_hard_stop_finalize_red_stop(
    self,
    pside: str,
    stop_event: dict,
    *,
    finalized_without_order: bool = False,
    flat_confirmations: int | None = None,
    position_count: int | None = None,
    entry_orders: int | None = None,
    nonpanic_close_orders: int | None = None,
) -> None:
    state = self._hsl_state(pside)
    cfg = self.hsl[pside]
    stop_ts_ms = int(stop_event["stop_event_timestamp_ms"])
    stop_event_anchor_source = "provided_stop_event"
    stop_event_anchor_fallback_used = False
    no_restart_drawdown_threshold = float(cfg["no_restart_drawdown_threshold"])
    finalization = self._equity_hard_stop_red_episode_finalization(
        pside,
        stop_event,
        stop_ts_ms,
    )
    no_restart_peak_strategy_equity = float(
        finalization["no_restart_peak_strategy_equity"]
    )
    no_restart_drawdown_raw = float(finalization["no_restart_drawdown_raw"])
    no_restart_latched = bool(finalization["no_restart_latched"])
    cooldown_until_ms = finalization["cooldown_until_ms"]
    payload = self._equity_hard_stop_build_latch_payload(
        pside,
        stop_event_timestamp_ms=stop_ts_ms,
        balance=stop_event.get("balance"),
        realized_pnl_total=stop_event.get("realized_pnl_total"),
        realized_pnl=stop_event.get("realized_pnl"),
        unrealized_pnl=stop_event.get("unrealized_pnl"),
        strategy_pnl=stop_event.get("strategy_pnl"),
        peak_strategy_pnl=stop_event.get("peak_strategy_pnl"),
        strategy_equity=float(stop_event["equity"]),
        peak_strategy_equity=float(stop_event["peak_strategy_equity"]),
        trigger_peak_strategy_equity=float(stop_event["trigger_peak_strategy_equity"]),
        drawdown_raw=float(stop_event["drawdown_raw"]),
        drawdown_ema=float(stop_event["drawdown_ema"]),
        drawdown_score=float(stop_event["drawdown_score"]),
        no_restart_latched=no_restart_latched,
        cooldown_until_ms=cooldown_until_ms,
        no_restart_peak_strategy_equity=no_restart_peak_strategy_equity,
        no_restart_drawdown_raw=no_restart_drawdown_raw,
    )
    state["last_stop_event"] = payload
    state["halted"] = True
    state["no_restart_latched"] = no_restart_latched
    state["cooldown_until_ms"] = cooldown_until_ms
    state["pending_stop_event"] = None
    state["red_flat_confirmations"] = 0
    state["pending_red_since_ms"] = None
    latch_path = self._equity_hard_stop_write_latch(pside, payload)
    if finalized_without_order:
        _emit_hsl_red_finalized_without_order(
            self,
            stop_event,
            pside=pside,
            symbol=None,
            stop_ts_ms=stop_ts_ms,
            stop_event_anchor_source=stop_event_anchor_source,
            stop_event_anchor_fallback_used=stop_event_anchor_fallback_used,
            cooldown_until_ms=cooldown_until_ms,
            flat_confirmations=flat_confirmations,
            position_count=position_count,
            entry_orders=entry_orders,
            nonpanic_close_orders=nonpanic_close_orders,
        )
    _emit_hsl_red_triggered_once(
        self,
        state,
        _hsl_event_data(
            stop_event,
            {
                "reason": "red_stop_finalized",
                "cooldown_until_ms": cooldown_until_ms,
                "no_restart_latched": bool(no_restart_latched),
                "no_restart_drawdown_raw": float(no_restart_drawdown_raw),
            },
        ),
        pside=pside,
        ts=stop_ts_ms,
        reason_code="red_stop_finalized",
    )
    self._equity_hard_stop_refresh_halted_runtime_forced_modes()
    if cooldown_until_ms is not None:
        _emit_hsl_event(
            self,
            "hsl.cooldown_started",
            ("hsl", "risk", "cooldown"),
            {
                "reason": "red_stop_finalized",
                "cooldown_until_ms": int(cooldown_until_ms),
                "latch_path": str(latch_path),
                "drawdown_raw": float(stop_event["drawdown_raw"]),
                "no_restart_drawdown_raw": float(no_restart_drawdown_raw),
            },
            pside=pside,
            ts=stop_ts_ms,
            status="started",
            reason_code="red_stop_finalized",
        )
    if no_restart_latched or cooldown_until_ms is None:
        logging.critical(
            "[risk] HSL[%s] RED stop finalized (terminal) | stop_ts=%s strategy_equity=%.6f "
            "peak_strategy_equity=%.6f drawdown_raw=%.6f no_restart_drawdown_raw=%.6f "
            "no_restart_drawdown_threshold=%.6f latch=%s",
            pside,
            stop_ts_ms,
            stop_event["equity"],
            stop_event["peak_strategy_equity"],
            stop_event["drawdown_raw"],
            no_restart_drawdown_raw,
            no_restart_drawdown_threshold,
            latch_path,
        )
        return
    logging.critical(
        "[risk] HSL[%s] RED stop finalized (auto-restart eligible) | stop_ts=%s "
        "drawdown_raw=%.6f no_restart_drawdown_raw=%.6f cooldown_until_ms=%s latch=%s",
        pside,
        stop_ts_ms,
        stop_event["drawdown_raw"],
        no_restart_drawdown_raw,
        cooldown_until_ms,
        latch_path,
    )


async def _equity_hard_stop_finalize_coin_red_stop(
    self,
    pside: str,
    symbol: str,
    stop_event: dict,
    *,
    finalized_without_order: bool = False,
    flat_confirmations: int | None = None,
    entry_orders: int | None = None,
    nonpanic_close_orders: int | None = None,
) -> None:
    state = self._hsl_coin_state(pside, symbol)
    stop_ts_ms = int(stop_event["stop_event_timestamp_ms"])
    stop_event_anchor_source = "provided_stop_event"
    stop_event_anchor_fallback_used = False
    finalization = self._equity_hard_stop_red_episode_finalization(
        pside,
        stop_event,
        stop_ts_ms,
        symbol=symbol,
    )
    no_restart_latched = bool(finalization["no_restart_latched"])
    cooldown_until_ms = finalization["cooldown_until_ms"]
    payload = self._equity_hard_stop_build_latch_payload(
        pside,
        symbol=symbol,
        stop_event_timestamp_ms=stop_ts_ms,
        balance=stop_event.get("balance"),
        realized_pnl_total=stop_event.get("realized_pnl_total"),
        realized_pnl=stop_event.get("realized_pnl"),
        unrealized_pnl=stop_event.get("unrealized_pnl"),
        strategy_pnl=stop_event.get("strategy_pnl"),
        peak_strategy_pnl=stop_event.get("peak_strategy_pnl"),
        strategy_equity=float(stop_event["equity"]),
        peak_strategy_equity=float(stop_event["peak_strategy_equity"]),
        trigger_peak_strategy_equity=float(stop_event["trigger_peak_strategy_equity"]),
        drawdown_raw=float(stop_event["drawdown_raw"]),
        drawdown_ema=float(stop_event["drawdown_ema"]),
        drawdown_score=float(stop_event["drawdown_score"]),
        no_restart_latched=no_restart_latched,
        cooldown_until_ms=cooldown_until_ms,
        no_restart_peak_strategy_equity=float(
            finalization["no_restart_peak_strategy_equity"]
        ),
        no_restart_drawdown_raw=float(finalization["no_restart_drawdown_raw"]),
    )
    state["last_stop_event"] = payload
    state["halted"] = True
    state["no_restart_latched"] = no_restart_latched
    state["cooldown_until_ms"] = cooldown_until_ms
    state["pending_stop_event"] = None
    state["red_flat_confirmations"] = 0
    state["pending_red_since_ms"] = None
    state["pnl_reset_timestamp_ms"] = int(stop_ts_ms) + 1
    latch_path = self._equity_hard_stop_write_latch(pside, payload, symbol=symbol)
    if finalized_without_order:
        _emit_hsl_red_finalized_without_order(
            self,
            stop_event,
            pside=pside,
            symbol=symbol,
            stop_ts_ms=stop_ts_ms,
            stop_event_anchor_source=stop_event_anchor_source,
            stop_event_anchor_fallback_used=stop_event_anchor_fallback_used,
            cooldown_until_ms=cooldown_until_ms,
            flat_confirmations=flat_confirmations,
            position_count=0,
            entry_orders=entry_orders,
            nonpanic_close_orders=nonpanic_close_orders,
        )
    trigger_extra = {
        "reason": "coin_red_stop_finalized",
        "cooldown_until_ms": cooldown_until_ms,
        "no_restart_latched": bool(no_restart_latched),
    }
    if finalized_without_order:
        trigger_extra.update(
            {
                "no_exchange_close_needed": True,
                "exchange_close_order_submitted": False,
                "panic_order_submitted_count": 0,
                "symbol_position_open": False,
                "entry_orders": entry_orders,
                "nonpanic_close_orders": nonpanic_close_orders,
                "flat_confirmations": flat_confirmations,
            }
        )
    _emit_hsl_red_triggered_once(
        self,
        state,
        _hsl_event_data(stop_event, trigger_extra),
        pside=pside,
        symbol=symbol,
        ts=stop_ts_ms,
        reason_code="coin_red_stop_finalized",
    )
    self._equity_hard_stop_clear_coin_runtime_forced_mode(pside, symbol)
    if cooldown_until_ms is not None:
        self._equity_hard_stop_set_coin_runtime_forced_mode(pside, symbol, "graceful_stop")
        _emit_hsl_event(
            self,
            "hsl.cooldown_started",
            ("hsl", "risk", "cooldown"),
            {
                "reason": "coin_red_stop_finalized",
                "symbol": symbol,
                "cooldown_until_ms": int(cooldown_until_ms),
                "latch_path": str(latch_path),
                "drawdown_raw": float(stop_event["drawdown_raw"]),
            },
            pside=pside,
            symbol=symbol,
            ts=stop_ts_ms,
            status="started",
            reason_code="coin_red_stop_finalized",
        )
    logging.critical(
        "[risk] HSL[%s:%s] RED stop finalized | stop_ts=%s drawdown_raw=%.6f cooldown_until_ms=%s no_restart_latched=%s latch=%s",
        pside,
        symbol,
        stop_ts_ms,
        stop_event["drawdown_raw"],
        cooldown_until_ms if cooldown_until_ms is not None else "none",
        no_restart_latched,
        latch_path,
    )


async def _equity_hard_stop_run_red_supervisor(self) -> None:
    if self._equity_hard_stop_supervisor_running:
        return
    self._equity_hard_stop_supervisor_running = True
    for pside in self._hsl_psides():
        state = self._hsl_state(pside)
        state["red_flat_confirmations"] = 0
        state["last_red_progress"] = None
    try:
        logging.critical("[risk] entering HSL RED supervisor loop (panic-close until confirmed flat)")
        while not self.stop_signal_received:
            active_red_psides = [
                pside
                for pside in self._hsl_psides()
                if self._equity_hard_stop_enabled(pside)
                and self._equity_hard_stop_runtime_red_latched(pside)
                and not self._hsl_state(pside)["halted"]
            ]
            if not active_red_psides:
                return
            if not await self.refresh_protective_authoritative_state():
                await asyncio.sleep(0.5)
                continue
            for pside in list(active_red_psides):
                state = self._hsl_state(pside)
                n_positions = self._equity_hard_stop_count_open_positions(pside)
                entry_orders, nonpanic_close_orders = self._equity_hard_stop_count_blocking_open_orders(pside)
                if n_positions == 0 and entry_orders == 0 and nonpanic_close_orders == 0:
                    now_ms = int(self.get_exchange_time())
                    stop_ts_ms = await self._equity_hard_stop_flatten_fill_timestamp_with_refresh(
                        pside,
                        now_ms,
                        since_ms=state.get("pending_red_since_ms"),
                    )
                    if stop_ts_ms is not None:
                        state["pending_stop_event"] = (
                            await self._equity_hard_stop_compute_stop_event(
                                pside, stop_ts_ms
                            )
                        )
                        state["red_flat_confirmations"] += 1
                else:
                    state["red_flat_confirmations"] = 0
                    state["pending_stop_event"] = None
                self._equity_hard_stop_log_red_progress(
                    pside,
                    n_positions,
                    entry_orders,
                    nonpanic_close_orders,
                    state["red_flat_confirmations"],
                )
                if state["red_flat_confirmations"] >= 2:
                    await self._equity_hard_stop_finalize_red_stop(
                        pside,
                        state["pending_stop_event"],
                        finalized_without_order=True,
                        flat_confirmations=state["red_flat_confirmations"],
                        position_count=n_positions,
                        entry_orders=entry_orders,
                        nonpanic_close_orders=nonpanic_close_orders,
                    )
            active_red_psides = [
                pside
                for pside in self._hsl_psides()
                if self._equity_hard_stop_enabled(pside)
                and self._equity_hard_stop_runtime_red_latched(pside)
                and not self._hsl_state(pside)["halted"]
            ]
            if not active_red_psides:
                return
            for pside in active_red_psides:
                # B2.1 contract: refresh the sample so recovery is observable
                # mid-supervision; only red_active_now authorizes continued
                # panic emission for the episode. Any refresh failure keeps
                # panic modes: recovery must be PROVEN by a fresh sample.
                metrics = None
                try:
                    signal_mode = self._equity_hard_stop_signal_mode()
                    upnl_total = (
                        float(await self._calc_upnl_sum_strict())
                        if signal_mode == "unified"
                        else None
                    )
                    metrics = self._equity_hard_stop_apply_sample(
                        pside,
                        int(self.get_exchange_time()),
                        float(self.get_raw_balance()),
                        float(self._equity_hard_stop_realized_pnl_now()),
                        float(self._equity_hard_stop_realized_pnl_now(pside)),
                        float(await self._calc_upnl_sum_strict(pside)),
                        unrealized_pnl_total=upnl_total,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logging.warning(
                        "[risk] HSL[%s] RED supervisor sample refresh failed; "
                        "keeping panic modes (recovery must be proven) | "
                        "error=%s: %s",
                        pside,
                        type(exc).__name__,
                        exc,
                    )
                if metrics is not None and not bool(
                    metrics.get("red_active_now", True)
                ):
                    logging.info(
                        "[risk] HSL[%s] RED no longer active on current sample; "
                        "pausing panic emission for the remainder of the episode "
                        "(entries stay blocked)",
                        pside,
                    )
                    self._equity_hard_stop_set_red_paused_runtime_forced_modes(pside)
                else:
                    self._equity_hard_stop_set_red_runtime_forced_modes(pside)
            self._equity_hard_stop_refresh_halted_runtime_forced_modes()
            try:
                to_cancel, to_create = (
                    await self.calc_protective_panic_orders_to_cancel_and_create()
                )
                await self.execute_order_plan_to_exchange(
                    to_cancel,
                    to_create,
                    configure_creations=False,
                )
            except FatalBotException:
                raise
            except RestartBotException as e:
                logging.error("[risk] RED supervisor ignored restart request: %s", e)
            except Exception as e:
                logging.error("[risk] RED supervisor execute_to_exchange failed: %s", e)
                traceback.print_exc()
            await asyncio.sleep(float(self.live_value("execution_delay_seconds")))
    finally:
        self._equity_hard_stop_supervisor_running = False


async def _equity_hard_stop_run_coin_red_supervisor(self) -> None:
    if self._equity_hard_stop_supervisor_running:
        return
    self._equity_hard_stop_supervisor_running = True
    try:
        logging.critical("[risk] entering HSL coin RED supervisor loop")
        while not self.stop_signal_received:
            active = []
            for pside in self._hsl_psides():
                for symbol, state in getattr(self, "_equity_hard_stop_coin", {}).get(pside, {}).items():
                    if self._equity_hard_stop_coin_needs_panic_supervision(pside, symbol, state):
                        active.append((pside, symbol))
            if not active:
                return
            if not await self.refresh_protective_authoritative_state():
                await asyncio.sleep(0.5)
                continue
            for pside, symbol in list(active):
                state = self._hsl_coin_state(pside, symbol)
                has_position = self._equity_hard_stop_has_open_position_symbol(pside, symbol)
                entry_orders, nonpanic_close_orders = (
                    self._equity_hard_stop_count_blocking_open_orders_symbol(pside, symbol)
                )
                if not has_position and entry_orders == 0 and nonpanic_close_orders == 0:
                    now_ms = int(self.get_exchange_time())
                    flatten_since_ms = state.get("pending_red_since_ms")
                    if state["halted"] and state["cooldown_repanic_reset_pending"]:
                        flatten_since_ms = state.get("cooldown_repanic_since_ms")
                    stop_ts_ms = await self._equity_hard_stop_flatten_fill_timestamp_with_refresh(
                        pside,
                        now_ms,
                        symbol=symbol,
                        since_ms=flatten_since_ms,
                        replay_start_sizes=(
                            state.get("cooldown_repanic_start_sizes") or {}
                            if state["halted"]
                            and state["cooldown_repanic_reset_pending"]
                            else None
                        ),
                    )
                    if stop_ts_ms is not None:
                        if not (
                            state["halted"]
                            and state["cooldown_repanic_reset_pending"]
                        ):
                            state["pending_stop_event"] = (
                                await self._equity_hard_stop_compute_coin_stop_event(
                                    pside, symbol, stop_ts_ms
                                )
                            )
                        state["red_flat_confirmations"] += 1
                else:
                    state["red_flat_confirmations"] = 0
                    state["pending_stop_event"] = None
                if state["red_flat_confirmations"] >= 2:
                    if state["halted"] and state["cooldown_repanic_reset_pending"]:
                        await self._equity_hard_stop_refresh_coin_cooldown_after_repanic(
                            pside, symbol, int(self.get_exchange_time())
                        )
                    else:
                        await self._equity_hard_stop_finalize_coin_red_stop(
                            pside,
                            symbol,
                            state["pending_stop_event"],
                            finalized_without_order=True,
                            flat_confirmations=state["red_flat_confirmations"],
                            entry_orders=entry_orders,
                            nonpanic_close_orders=nonpanic_close_orders,
                        )
                else:
                    # B2.1 contract: refresh the sample so recovery is
                    # observable mid-supervision; only red_active_now may
                    # authorize continued panic emission. A recovered sample
                    # inside a red-seen episode keeps entry blocking without
                    # new panic orders; supervision re-engages if RED
                    # re-activates. Any refresh failure keeps panic modes:
                    # recovery must be PROVEN by a fresh sample.
                    metrics = None
                    try:
                        metrics = self._equity_hard_stop_apply_coin_sample(
                            pside,
                            symbol,
                            int(self.get_exchange_time()),
                            float(self.get_raw_balance()),
                            float(await self._calc_upnl_sum_strict(pside, symbol)),
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logging.warning(
                            "[risk] HSL[%s:%s] RED supervisor sample refresh "
                            "failed; keeping panic mode (recovery must be "
                            "proven) | error=%s: %s",
                            pside,
                            symbol,
                            type(exc).__name__,
                            exc,
                        )
                    if metrics is not None and not bool(
                        metrics.get("red_active_now", True)
                    ):
                        logging.info(
                            "[risk] HSL[%s:%s] RED no longer active on current "
                            "sample; pausing panic emission for the remainder "
                            "of the episode (entries stay blocked)",
                            pside,
                            symbol,
                        )
                        self._equity_hard_stop_set_coin_runtime_forced_mode(
                            pside, symbol, "tp_only_with_active_entry_cancellation"
                        )
                    else:
                        self._equity_hard_stop_set_coin_runtime_forced_mode(
                            pside, symbol, "panic"
                        )
            active = [
                (pside, symbol)
                for pside in self._hsl_psides()
                for symbol, state in getattr(self, "_equity_hard_stop_coin", {}).get(pside, {}).items()
                if self._equity_hard_stop_coin_needs_panic_supervision(pside, symbol, state)
            ]
            if not active:
                return
            try:
                to_cancel, to_create = (
                    await self.calc_protective_panic_orders_to_cancel_and_create()
                )
                await self.execute_order_plan_to_exchange(
                    to_cancel,
                    to_create,
                    configure_creations=False,
                )
            except FatalBotException:
                raise
            except RestartBotException as e:
                logging.error("[risk] coin RED supervisor ignored restart request: %s", e)
            except Exception as e:
                logging.error("[risk] coin RED supervisor execute_to_exchange failed: %s", e)
                traceback.print_exc()
            await asyncio.sleep(float(self.live_value("execution_delay_seconds")))
    finally:
        self._equity_hard_stop_supervisor_running = False


def _apply_equity_hard_stop_orange_overlay(self) -> None:
    if not self._equity_hard_stop_enabled():
        return
    symbols = (
        set(self.PB_modes["long"].keys())
        | set(self.PB_modes["short"].keys())
        | set(self.positions.keys())
        | set(self.open_orders.keys())
    )
    for pside in self._hsl_psides():
        if not self._equity_hard_stop_enabled(pside):
            continue
        if self._hsl_state(pside)["halted"]:
            continue
        if (
            self._equity_hard_stop_runtime_red_latched(pside)
            or self._equity_hard_stop_runtime_tier(pside) != "orange"
        ):
            continue
        orange_mode = str(self.hsl[pside]["orange_tier_mode"])
        for symbol in symbols:
            if symbol not in self.PB_modes[pside]:
                continue
            current_mode = self.PB_modes[pside][symbol]
            if orange_mode == "graceful_stop":
                if current_mode == "normal":
                    self.PB_modes[pside][symbol] = "graceful_stop"
            else:
                if current_mode in ("normal", "graceful_stop"):
                    self.PB_modes[pside][symbol] = "tp_only_with_active_entry_cancellation"
