from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
import traceback
from typing import Any, Optional

import passivbot_rust as pbr

from config_utils import (
    get_optional_live_value,
    normalize_hsl_cooldown_position_policy,
    normalize_hsl_signal_mode,
    require_live_value,
)
from passivbot_exceptions import RestartBotException
from utils import make_get_filepath


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
        }
        if enabled:
            logging.info(
                "[risk] HSL[%s] enabled | red_threshold=%.6f ema_span_minutes=%.3f "
                "cooldown_minutes_after_red=%.3f "
                "no_restart_drawdown_threshold=%.6f signal_mode=%s "
                "yellow_ratio=%.3f orange_ratio=%.3f "
                "orange_mode=%s panic_close=%s",
                pside,
                red_threshold,
                ema_span_minutes,
                cooldown_minutes_after_red,
                no_restart_drawdown_threshold,
                signal_mode,
                ratio_yellow,
                ratio_orange,
                orange_tier_mode,
                panic_close_order_type,
            )
    return out


def _equity_hard_stop_enabled(self, pside: Optional[str] = None) -> bool:
    if not hasattr(self, "hsl") or not isinstance(self.hsl, dict):
        legacy_cfg = getattr(self, "equity_hard_stop_loss", None)
        enabled = bool(isinstance(legacy_cfg, dict) and legacy_cfg.get("enabled", False))
        if pside is None:
            return enabled
        return enabled
    if pside is None:
        return any(bool(self.hsl[x]["enabled"]) for x in ("long", "short"))
    return bool(self.hsl[pside]["enabled"])


def _equity_hard_stop_signal_mode(self) -> str:
    config = getattr(self, "config", {})
    return normalize_hsl_signal_mode(get_optional_live_value(config, "hsl_signal_mode", "pside"))


def _equity_hard_stop_cooldown_position_policy(self) -> str:
    config = getattr(self, "config", {})
    return normalize_hsl_cooldown_position_policy(
        get_optional_live_value(
            config,
            "hsl_position_during_cooldown_policy",
            "repanic_reset_cooldown",
        )
    )


def _equity_hard_stop_halted_mode(self, pside: str, symbol: str | None) -> str:
    policy = self._equity_hard_stop_cooldown_position_policy()
    size = 0.0
    if symbol is not None:
        size = float(self.positions.get(symbol, {}).get(pside, {}).get("size", 0.0) or 0.0)
    if policy in {"repanic_reset_cooldown", "repanic_keep_original_cooldown"}:
        return "panic" if size != 0.0 else "graceful_stop"
    if policy == "graceful_stop_keep_cooldown":
        return "graceful_stop"
    if policy == "manual_quarantine":
        return "manual" if size != 0.0 else "graceful_stop"
    if policy == "resume_normal_reset_drawdown":
        return "normal"
    return "graceful_stop"


def _equity_hard_stop_panic_close_order_type(self, pside: str) -> str:
    hsl_cfg = getattr(self, "hsl", None)
    if isinstance(hsl_cfg, dict) and pside in hsl_cfg and isinstance(hsl_cfg[pside], dict):
        return str(hsl_cfg[pside].get("panic_close_order_type", "market"))
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


def _equity_hard_stop_latch_path(self, pside: str) -> str:
    return make_get_filepath(f"caches/equity_hard_stop/{self.exchange}/{self.user}_{pside}.json")


def _equity_hard_stop_write_latch(self, pside: str, metrics: dict) -> str:
    path = self._equity_hard_stop_latch_path(pside)
    payload = dict(metrics)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)
    return path


def _equity_hard_stop_remove_latch_file(self, pside: str) -> None:
    path = self._equity_hard_stop_latch_path(pside)
    if os.path.isfile(path):
        os.remove(path)


def _equity_hard_stop_reset_state(self) -> None:
    for pside in self._hsl_psides():
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
        state["last_stop_event"] = None
        state["last_status_log_ms"] = 0
        state["last_cooldown_log_ms"] = 0
        state["cooldown_intervention_active"] = False
        state["cooldown_repanic_reset_pending"] = False
        state["last_cooldown_intervention_log_ms"] = 0
    self._runtime_forced_modes = {"long": {}, "short": {}}


def _equity_hard_stop_runtime_initialized(self, pside: str) -> bool:
    return bool(self._hsl_state(pside)["runtime"].initialized())


def _equity_hard_stop_runtime_red_latched(self, pside: str) -> bool:
    return bool(self._hsl_state(pside)["runtime"].red_latched())


def _equity_hard_stop_runtime_tier(self, pside: str) -> str:
    return str(self._hsl_state(pside)["runtime"].tier())


def _equity_hard_stop_fill_pside(fill: Any) -> str:
    if isinstance(fill, dict):
        raw = fill.get("position_side", fill.get("pside", "long"))
    else:
        raw = getattr(fill, "position_side", getattr(fill, "pside", "long"))
    out = str(raw).lower()
    return out if out in {"long", "short"} else "long"


async def _calc_upnl_sum_strict(self, pside: Optional[str] = None) -> float:
    if not self.fetched_positions:
        return 0.0
    symbols = {
        x["symbol"] for x in self.fetched_positions if pside is None or x["position_side"] == pside
    }
    if not symbols:
        return 0.0
    last_prices = await self.cm.get_last_prices(symbols, max_age_ms=60_000)
    upnl_sum = 0.0
    for elm in self.fetched_positions:
        if pside is not None and elm["position_side"] != pside:
            continue
        symbol = elm["symbol"]
        if symbol not in last_prices:
            raise RuntimeError(f"missing last price for {symbol} while evaluating hard stop")
        upnl = _calc_hsl_pnl(
            elm["position_side"],
            elm["price"],
            last_prices[symbol],
            elm["size"],
            self.c_mults[symbol],
        )
        if not math.isfinite(upnl):
            raise RuntimeError(
                f"non-finite upnl for {symbol} {elm['position_side']} while evaluating hard stop"
            )
        upnl_sum += upnl
    return upnl_sum


def _equity_hard_stop_fee_cost(fill: Any) -> float:
    if fill is None:
        return 0.0
    if isinstance(fill, dict):
        fee_obj = fill.get("fee")
        if isinstance(fee_obj, dict):
            return float(fee_obj.get("cost", 0.0) or 0.0)
        if isinstance(fee_obj, (int, float, str)):
            return float(fee_obj or 0.0)
        fees_obj = fill.get("fees")
    else:
        fees_obj = getattr(fill, "fees", None)
    if isinstance(fees_obj, dict):
        return float(fees_obj.get("cost", 0.0) or 0.0)
    if isinstance(fees_obj, (list, tuple)):
        total = 0.0
        for item in fees_obj:
            if isinstance(item, dict):
                total += float(item.get("cost", 0.0) or 0.0)
        return total
    return 0.0


def _get_exchange_fee_rates(self, symbol: str) -> tuple[float, float]:
    market = {}
    try:
        market = self.markets_dict.get(symbol, {}) or {}
    except Exception:
        market = {}
    maker_fee = market.get("maker_fee", market.get("maker", 0.0002))
    taker_fee = market.get("taker_fee", market.get("taker", 0.00055))
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


def _equity_hard_stop_realized_pnl_now(self, pside: Optional[str] = None) -> float:
    if self._pnls_manager is None:
        return 0.0
    realized = 0.0
    for event in self._pnls_manager.get_events():
        if pside is not None and _equity_hard_stop_fill_pside(event) != pside:
            continue
        realized += float(getattr(event, "pnl", 0.0) or 0.0)
        realized += _equity_hard_stop_fee_cost(event)
    return realized


def _equity_hard_stop_lookback_ms(self) -> int:
    lookback_days_raw = float(require_live_value(self.config, "pnls_max_lookback_days"))
    if not math.isfinite(lookback_days_raw):
        raise ValueError("live.pnls_max_lookback_days must be finite for hard-stop rolling-peak logic")
    lookback_days = max(0.0, lookback_days_raw)
    return max(1, int(round(lookback_days * 86_400_000.0)))


def _equity_hard_stop_apply_sample(
    self,
    pside: str,
    timestamp_ms: int,
    balance: float,
    realized_pnl_total: float,
    realized_pnl_pside: float,
    unrealized_pnl_pside: float,
    unrealized_pnl_total: Optional[float] = None,
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
    if last_metrics is not None and int(last_metrics["timestamp_ms"]) // 60_000 == current_minute:
        cached = dict(last_metrics)
        cached["changed"] = False
        cached["elapsed_minutes"] = 0
        state["last_metrics"] = cached
        return cached

    signal_mode, realized_pnl_signal, unrealized_pnl_signal = self._equity_hard_stop_signal_values(
        pside,
        realized_pnl_total=realized_pnl_total,
        realized_pnl_pside=realized_pnl_pside,
        unrealized_pnl_pside=unrealized_pnl_pside,
        unrealized_pnl_total=unrealized_pnl_total,
    )
    cfg = self.hsl[pside]
    lookback_ms = self._equity_hard_stop_lookback_ms()
    prev_tier = self._equity_hard_stop_runtime_tier(pside)
    red_threshold = float(cfg["red_threshold"])
    ratio_yellow = float(cfg["tier_ratios"]["yellow"])
    ratio_orange = float(cfg["tier_ratios"]["orange"])
    ema_span_minutes = float(cfg["ema_span_minutes"])
    strategy_pnl = realized_pnl_signal + unrealized_pnl_signal
    peak_strategy_pnl = float(
        state["strategy_pnl_peak"].update(int(timestamp_ms), float(strategy_pnl), int(lookback_ms))
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
        "changed": bool(step["changed"]) or str(step["tier"]) != prev_tier,
        "alpha": float(step["alpha"]),
        "elapsed_minutes": int(step["elapsed_minutes"]),
    }
    state["last_metrics"] = metrics
    return metrics


def _equity_hard_stop_log_transition(self, pside: str, metrics: dict, prev_tier: str) -> None:
    logging.info(
        "[risk] HSL[%s] tier transition %s -> %s | balance=%.6f strategy_equity=%.6f "
        "peak_strategy_equity=%.6f drawdown_raw=%.6f drawdown_ema=%.6f drawdown_score=%.6f "
        "strategy_pnl=%.6f peak_strategy_pnl=%.6f "
        "red_threshold=%.6f yellow=%.3f orange=%.3f",
        pside,
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
        float(self.hsl[pside]["tier_ratios"]["yellow"]),
        float(self.hsl[pside]["tier_ratios"]["orange"]),
    )
    self._monitor_record_event(
        "hsl.transition",
        ("hsl", "risk", "transition"),
        {"previous_tier": prev_tier, "metrics": dict(metrics)},
        pside=pside,
        ts=int(metrics.get("timestamp_ms", 0) or 0) or None,
    )


def _equity_hard_stop_build_latch_payload(
    self,
    pside: str,
    *,
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
) -> dict:
    cfg = self.hsl[pside]
    return {
        "triggered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "exchange": str(self.exchange),
        "user": str(self.user),
        "position_side": pside,
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
        "drawdown_ema": float(drawdown_ema),
        "drawdown_score": float(drawdown_score),
        "no_restart_latched": bool(no_restart_latched),
        "auto_restart_eligible": bool(
            (not no_restart_latched) and float(cfg["cooldown_minutes_after_red"]) > 0.0
        ),
        "cooldown_until_ms": None if cooldown_until_ms is None else int(cooldown_until_ms),
    }


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
    logging.info("[risk] HSL[%s] RED cooldown active | remaining_seconds=%.1f", pside, remaining_seconds)


def _equity_hard_stop_position_symbols(self, pside: str) -> list[str]:
    out = []
    for symbol, position in self.positions.items():
        size = float(position.get(pside, {}).get("size", 0.0) or 0.0)
        if size != 0.0:
            out.append(symbol)
    return sorted(out)


async def _equity_hard_stop_refresh_cooldown_after_repanic(self, pside: str, now_ms: int) -> None:
    state = self._hsl_state(pside)
    cooldown_minutes = float(self.hsl[pside]["cooldown_minutes_after_red"])
    cooldown_ms = max(0, int(round(cooldown_minutes * 60_000.0))) if cooldown_minutes > 0.0 else 0
    cooldown_until_ms = now_ms + cooldown_ms if cooldown_ms > 0 else None
    stop_event = await self._equity_hard_stop_compute_stop_event(pside, now_ms)
    payload = self._equity_hard_stop_build_latch_payload(
        pside,
        stop_event_timestamp_ms=now_ms,
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
    state["last_cooldown_intervention_log_ms"] = 0
    latch_path = self._equity_hard_stop_write_latch(pside, payload)
    logging.critical(
        "[risk] HSL[%s] cooldown violation repanic flattened; cooldown reset from flat_ts=%s to cooldown_until_ms=%s latch=%s",
        pside,
        now_ms,
        cooldown_until_ms if cooldown_until_ms is not None else "none",
        latch_path,
    )
    if cooldown_until_ms is not None:
        self._monitor_record_event(
            "hsl.cooldown_started",
            ("hsl", "risk", "cooldown"),
            {
                "reason": "repanic_reset",
                "cooldown_until_ms": int(cooldown_until_ms),
                "latch_path": str(latch_path),
            },
            pside=pside,
            ts=now_ms,
        )


async def _equity_hard_stop_handle_position_during_cooldown(self, pside: str, now_ms: int) -> bool:
    state = self._hsl_state(pside)
    if not state["halted"] or state["no_restart_latched"]:
        return False
    cooldown_until_ms = state["cooldown_until_ms"]
    if cooldown_until_ms is None or now_ms >= cooldown_until_ms:
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
        state["last_cooldown_intervention_log_ms"] = 0
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
    state["cooldown_intervention_active"] = True

    if policy == "resume_normal_reset_drawdown":
        self._equity_hard_stop_reset_after_restart(pside)
        self._equity_hard_stop_remove_latch_file(pside)
        logging.critical(
            "[risk] HSL[%s] operator override during RED cooldown: resumed normal operation and reset drawdown tracker",
            pside,
        )
        return True

    state["cooldown_repanic_reset_pending"] = policy == "repanic_reset_cooldown"
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
    state["last_status_log_ms"] = 0
    state["last_cooldown_log_ms"] = 0
    state["cooldown_intervention_active"] = False
    state["cooldown_repanic_reset_pending"] = False
    state["last_cooldown_intervention_log_ms"] = 0
    self._equity_hard_stop_clear_runtime_forced_modes(pside)


def _equity_hard_stop_refresh_halted_runtime_forced_modes(self) -> None:
    symbols = set(self.positions.keys()) | set(self.open_orders.keys()) | set(self.active_symbols)
    for pside in self._hsl_psides():
        if not self._equity_hard_stop_enabled(pside):
            self._equity_hard_stop_clear_runtime_forced_modes(pside)
            continue
        state = self._hsl_state(pside)
        if self._equity_hard_stop_runtime_red_latched(pside) and not state["halted"]:
            self._equity_hard_stop_set_red_runtime_forced_modes(pside)
            continue
        if not state["halted"]:
            self._equity_hard_stop_clear_runtime_forced_modes(pside)
            continue
        forced = {}
        for symbol in symbols:
            forced[symbol] = self._equity_hard_stop_halted_mode(pside, symbol)
        self._runtime_forced_modes[pside] = forced


async def _equity_hard_stop_initialize_from_history(self) -> None:
    if not self._equity_hard_stop_enabled():
        return
    prev_phase = getattr(self, "_log_silence_watchdog_phase", "runtime")
    prev_stage = getattr(self, "_log_silence_watchdog_stage", "idle")
    self._set_log_silence_watchdog_context(phase=prev_phase, stage="equity_hard_stop_initialize_from_history")
    try:
        self._equity_hard_stop_reset_state()
        signal_mode = self._equity_hard_stop_signal_mode()
        logging.info(
            "[risk] HSL history replay starting | lookback_days=%.3f signal_mode=%s",
            float(self.live_value("pnls_max_lookback_days")),
            signal_mode,
        )
        history = await self.get_balance_equity_history(current_balance=self.get_raw_balance())
        if "timeline" not in history:
            raise ValueError("get_balance_equity_history() missing required key: timeline")
        timeline = history["timeline"]
        if not isinstance(timeline, list):
            raise TypeError(
                f"get_balance_equity_history()['timeline'] must be a list, got {type(timeline).__name__}"
            )
        now_ms = int(self.get_exchange_time())
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
            cfg = self.hsl[pside]
            cooldown_minutes = float(cfg["cooldown_minutes_after_red"])
            no_restart_drawdown_threshold = float(cfg["no_restart_drawdown_threshold"])
            cooldown_ms = int(round(cooldown_minutes * 60_000.0)) if cooldown_minutes > 0.0 else 0
            pending_red = False
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
                )
                n_rows[pside] += 1
                if self._equity_hard_stop_runtime_tier(pside) == "red":
                    pending_red = True
                    state["pending_red_since_ms"] = int(ts)
                is_flat = bool(row.get(f"is_flat_{pside}", False))
                if pending_red and is_flat:
                    stop_drawdown_raw = float(current_metrics["drawdown_raw"])
                    cooldown_until_ms = None
                    if stop_drawdown_raw < no_restart_drawdown_threshold and cooldown_ms > 0:
                        cooldown_until_ms = ts + cooldown_ms
                    payload = self._equity_hard_stop_build_latch_payload(
                        pside,
                        stop_event_timestamp_ms=ts,
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
                        no_restart_latched=bool(stop_drawdown_raw >= no_restart_drawdown_threshold),
                        cooldown_until_ms=cooldown_until_ms,
                    )
                    state["last_stop_event"] = payload
                    state["halted"] = True
                    state["no_restart_latched"] = bool(stop_drawdown_raw >= no_restart_drawdown_threshold)
                    state["cooldown_until_ms"] = cooldown_until_ms
                    state["pending_red_since_ms"] = None
                    latch_path = self._equity_hard_stop_write_latch(pside, payload)
                    logging.critical(
                        "[risk] HSL[%s] replay found finalized RED stop in exchange-derived history | stop_ts=%s drawdown_raw=%.6f no_restart_latched=%s cooldown_until_ms=%s diagnostic=%s",
                        pside,
                        ts,
                        stop_drawdown_raw,
                        state["no_restart_latched"],
                        cooldown_until_ms if cooldown_until_ms is not None else "none",
                        latch_path,
                    )
                    pending_red = False
                    if state["no_restart_latched"]:
                        break
            if state["halted"] and not state["no_restart_latched"]:
                cooldown_until_ms = state["cooldown_until_ms"]
                if cooldown_until_ms is not None and now_ms >= cooldown_until_ms:
                    self._equity_hard_stop_reset_after_restart(pside)
                    self._equity_hard_stop_remove_latch_file(pside)
                    logging.info("[risk] HSL[%s] replayed cooldown already elapsed; resumed", pside)
                elif cooldown_until_ms is not None:
                    logging.critical(
                        "[risk] HSL[%s] reconstructed active RED cooldown from exchange-derived history | remaining_seconds=%.1f",
                        pside,
                        (cooldown_until_ms - now_ms) / 1000.0,
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
        self._set_log_silence_watchdog_context(phase=prev_phase, stage=prev_stage)


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
    cooldown_remaining_s = None
    if state["cooldown_until_ms"] is not None:
        cooldown_remaining_s = max(0.0, (state["cooldown_until_ms"] - now_ms) / 1000.0)
    last_red_ts = None
    if state["last_stop_event"] is not None:
        last_red_ts = state["last_stop_event"].get("stop_event_timestamp_ms")
    if last_red_ts is None:
        last_red_ts = state["pending_red_since_ms"]
    logging.info(
        "[risk] HSL[%s] status | tier=%s dist_to_red=%.6f drawdown_raw=%.6f drawdown_ema=%.6f "
        "drawdown_score=%.6f red_threshold=%.6f cooldown_remaining_s=%s last_red_ts=%s "
        "pending_red_since_ms=%s peak_strategy_equity=%.6f rolling_peak_strategy_equity=%.6f",
        pside,
        metrics["tier"],
        dist_to_red,
        metrics["drawdown_raw"],
        metrics["drawdown_ema"],
        drawdown_score,
        red_threshold,
        f"{cooldown_remaining_s:.1f}" if cooldown_remaining_s is not None else "none",
        last_red_ts if last_red_ts is not None else "none",
        state["pending_red_since_ms"] if state["pending_red_since_ms"] is not None else "none",
        metrics["peak_strategy_equity"],
        metrics["rolling_peak_strategy_equity"],
    )


async def _equity_hard_stop_check(self) -> Optional[dict]:
    if not self._equity_hard_stop_enabled():
        return None
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
                    and cooldown_until_ms is not None
                    and ts_ms >= cooldown_until_ms
                ):
                    self._equity_hard_stop_reset_after_restart(pside)
                    self._equity_hard_stop_remove_latch_file(pside)
                    logging.info("[risk] HSL[%s] RED cooldown elapsed; trading resumed", pside)
                    self._monitor_record_event(
                        "hsl.cooldown_ended",
                        ("hsl", "risk", "cooldown"),
                        {"reason": "elapsed"},
                        pside=pside,
                        ts=ts_ms,
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
        if metrics["tier"] == "red" and not prev_latched:
            state["pending_red_since_ms"] = int(metrics["timestamp_ms"])
            logging.critical(
                "[risk] HSL[%s] RED triggered | strategy_equity=%.6f peak_strategy_equity=%.6f rolling_peak_strategy_equity=%.6f drawdown_score=%.6f red_threshold=%.6f",
                pside,
                metrics["strategy_equity"],
                metrics["peak_strategy_equity"],
                metrics["rolling_peak_strategy_equity"],
                metrics["drawdown_score"],
                metrics["red_threshold"],
            )
        elif metrics["tier"] != "red":
            state["pending_red_since_ms"] = None
        self._equity_hard_stop_log_status(pside, metrics)
        out[pside] = metrics
    self._equity_hard_stop_refresh_halted_runtime_forced_modes()
    return out if out else None


def _equity_hard_stop_set_red_runtime_forced_modes(self, pside: str) -> None:
    forced = {}
    symbols = set(self.positions.keys()) | set(self.open_orders.keys()) | set(self.active_symbols)
    for symbol in symbols:
        forced[symbol] = "panic"
    self._runtime_forced_modes[pside] = forced


def _equity_hard_stop_clear_runtime_forced_modes(self, pside: Optional[str] = None) -> None:
    if pside is None:
        self._runtime_forced_modes = {"long": {}, "short": {}}
        return
    self._runtime_forced_modes[pside] = {}


def _equity_hard_stop_count_open_positions(self, pside: str) -> int:
    n_positions = 0
    for pos in self.positions.values():
        if float(pos.get(pside, {}).get("size", 0.0) or 0.0) != 0.0:
            n_positions += 1
    return n_positions


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


async def _equity_hard_stop_finalize_red_stop(self, pside: str, stop_event: Optional[dict] = None) -> None:
    state = self._hsl_state(pside)
    cfg = self.hsl[pside]
    stop_ts_ms = int(self.get_exchange_time())
    if stop_event is None:
        stop_event = await self._equity_hard_stop_compute_stop_event(pside, stop_ts_ms)
    else:
        stop_ts_ms = int(stop_event["stop_event_timestamp_ms"])
    cooldown_minutes = float(cfg["cooldown_minutes_after_red"])
    no_restart_drawdown_threshold = float(cfg["no_restart_drawdown_threshold"])
    no_restart_latched = bool(stop_event["drawdown_raw"] >= no_restart_drawdown_threshold)
    cooldown_ms = int(round(cooldown_minutes * 60_000.0)) if cooldown_minutes > 0.0 else 0
    cooldown_until_ms = None if no_restart_latched or cooldown_ms <= 0 else int(stop_ts_ms + cooldown_ms)
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
    )
    state["last_stop_event"] = payload
    state["halted"] = True
    state["no_restart_latched"] = no_restart_latched
    state["cooldown_until_ms"] = cooldown_until_ms
    state["pending_stop_event"] = None
    state["red_flat_confirmations"] = 0
    state["pending_red_since_ms"] = None
    latch_path = self._equity_hard_stop_write_latch(pside, payload)
    self._equity_hard_stop_refresh_halted_runtime_forced_modes()
    if cooldown_until_ms is not None:
        self._monitor_record_event(
            "hsl.cooldown_started",
            ("hsl", "risk", "cooldown"),
            {
                "reason": "red_stop_finalized",
                "cooldown_until_ms": int(cooldown_until_ms),
                "latch_path": str(latch_path),
                "drawdown_raw": float(stop_event["drawdown_raw"]),
            },
            pside=pside,
            ts=stop_ts_ms,
        )
    if no_restart_latched or cooldown_until_ms is None:
        logging.critical(
            "[risk] HSL[%s] RED stop finalized (terminal) | stop_ts=%s strategy_equity=%.6f "
            "peak_strategy_equity=%.6f drawdown_raw=%.6f "
            "no_restart_drawdown_threshold=%.6f latch=%s",
            pside,
            stop_ts_ms,
            stop_event["equity"],
            stop_event["peak_strategy_equity"],
            stop_event["drawdown_raw"],
            no_restart_drawdown_threshold,
            latch_path,
        )
        return
    logging.critical(
        "[risk] HSL[%s] RED stop finalized (auto-restart eligible) | stop_ts=%s "
        "drawdown_raw=%.6f cooldown_until_ms=%s latch=%s",
        pside,
        stop_ts_ms,
        stop_event["drawdown_raw"],
        cooldown_until_ms,
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
        state["pending_stop_event"] = None
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
            if not await self.update_pos_oos_pnls_ohlcvs():
                await asyncio.sleep(0.5)
                continue
            for pside in list(active_red_psides):
                state = self._hsl_state(pside)
                n_positions = self._equity_hard_stop_count_open_positions(pside)
                entry_orders, nonpanic_close_orders = self._equity_hard_stop_count_blocking_open_orders(pside)
                if n_positions == 0 and entry_orders == 0 and nonpanic_close_orders == 0:
                    if state["red_flat_confirmations"] == 0:
                        state["pending_stop_event"] = await self._equity_hard_stop_compute_stop_event(
                            pside, int(self.get_exchange_time())
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
                    await self._equity_hard_stop_finalize_red_stop(pside, state["pending_stop_event"])
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
                self._equity_hard_stop_set_red_runtime_forced_modes(pside)
            self._equity_hard_stop_refresh_halted_runtime_forced_modes()
            try:
                await self.execute_to_exchange()
            except RestartBotException as e:
                logging.error("[risk] RED supervisor ignored restart request: %s", e)
            except Exception as e:
                logging.error("[risk] RED supervisor execute_to_exchange failed: %s", e)
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
                size = float(self.positions.get(symbol, {}).get(pside, {}).get("size", 0.0) or 0.0)
                if size == 0.0:
                    continue
                if current_mode in ("normal", "graceful_stop"):
                    self.PB_modes[pside][symbol] = "tp_only_with_active_entry_cancellation"
