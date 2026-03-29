from __future__ import annotations

import logging
import math
import os
import sys
from copy import deepcopy
from typing import Any, Iterable, Optional

import numpy as np
import passivbot_rust as pbr

from utils import utc_ms

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

try:
    import resource  # type: ignore
except Exception:
    resource = None


def _get_process_rss_bytes() -> Optional[int]:
    """Return current process RSS in bytes or None if unavailable."""
    try:
        if psutil is not None:
            return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass
    if resource is not None:
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform.startswith("linux"):
                usage = int(usage) * 1024
            else:
                usage = int(usage)
            return int(usage)
        except Exception:
            pass
    return None


def _calc_monitor_pnl(position_side, entry_price, close_price, qty, c_mult):
    """Calculate trade PnL using the same Rust helpers as the live bot."""
    try:
        if isinstance(position_side, str):
            if position_side == "long":
                return pbr.calc_pnl_long(entry_price, close_price, qty, c_mult)
            return pbr.calc_pnl_short(entry_price, close_price, qty, c_mult)
        return pbr.calc_pnl_long(entry_price, close_price, qty, c_mult)
    except Exception:
        raise


def _monitor_record_event(
    self,
    kind: str,
    tags: Iterable[str],
    payload: Optional[dict] = None,
    *,
    symbol: Optional[str] = None,
    pside: Optional[str] = None,
    ts: Optional[int] = None,
) -> Optional[dict]:
    publisher = getattr(self, "monitor_publisher", None)
    if publisher is None:
        return None
    return publisher.record_event(kind, tags, payload, ts=ts, symbol=symbol, pside=pside)


def _monitor_record_error(
    self,
    kind: str,
    error: Exception,
    *,
    tags: Optional[Iterable[str]] = None,
    payload: Optional[dict] = None,
    symbol: Optional[str] = None,
    pside: Optional[str] = None,
    ts: Optional[int] = None,
) -> Optional[dict]:
    publisher = getattr(self, "monitor_publisher", None)
    if publisher is None:
        return None
    return publisher.record_error(
        kind,
        error,
        tags=tags,
        payload=payload,
        ts=ts,
        symbol=symbol,
        pside=pside,
    )


def _monitor_emit_stop(
    self, reason: str, *, ts: Optional[int] = None, payload: Optional[dict] = None
) -> Optional[dict]:
    if getattr(self, "_monitor_stop_emitted", False):
        return None
    self._monitor_stop_emitted = True
    stop_payload = {"reason": str(reason)}
    if payload:
        stop_payload.update(payload)
    return self._monitor_record_event(
        "bot.stop",
        ("bot", "lifecycle", "stop"),
        stop_payload,
        ts=ts,
    )


def _monitor_hsl_payload(self, pside: str) -> dict:
    enabled = self._equity_hard_stop_enabled(pside)
    state = self._hsl_state(pside)
    last_metrics = state.get("last_metrics") or {}
    payload = {
        "enabled": bool(enabled),
        "tier": str(last_metrics.get("tier", "disabled" if not enabled else "unknown")),
        "halted": bool(state.get("halted", False)),
        "no_restart_latched": bool(state.get("no_restart_latched", False)),
        "pending_red_since_ms": state.get("pending_red_since_ms"),
        "cooldown_until_ms": state.get("cooldown_until_ms"),
        "cooldown_intervention_active": bool(state.get("cooldown_intervention_active", False)),
        "cooldown_repanic_reset_pending": bool(state.get("cooldown_repanic_reset_pending", False)),
        "last_metrics": dict(last_metrics) if isinstance(last_metrics, dict) else {},
    }
    return {k: v for k, v in payload.items() if v is not None}


def _monitor_order_payload(self, order: dict, *, source: str) -> dict:
    payload = {
        "id": order.get("id"),
        "custom_id": order.get("custom_id"),
        "side": order.get("side"),
        "position_side": order.get("position_side"),
        "qty": abs(float(order.get("qty", 0.0) or 0.0)),
        "price": float(order.get("price", 0.0) or 0.0),
        "reduce_only": bool(
            order.get("reduce_only")
            or (order.get("position_side") == "long" and order.get("side") == "sell")
            or (order.get("position_side") == "short" and order.get("side") == "buy")
        ),
        "pb_order_type": self._resolve_pb_order_type(order),
        "source": source,
    }
    if order.get("_context"):
        payload["context"] = str(order["_context"])
    if order.get("_reason"):
        payload["reason"] = str(order["_reason"])
    if isinstance(order.get("_delta"), dict):
        payload["delta"] = dict(order["_delta"])
    return {k: v for k, v in payload.items() if v is not None}


def _monitor_fill_payload(self, event) -> dict:
    payload = {
        "id": getattr(event, "id", None),
        "timestamp": int(getattr(event, "timestamp", 0) or 0),
        "symbol": getattr(event, "symbol", None),
        "side": str(getattr(event, "side", "") or "").lower(),
        "position_side": str(getattr(event, "position_side", "") or "").lower(),
        "qty": float(getattr(event, "qty", 0.0) or 0.0),
        "price": float(getattr(event, "price", 0.0) or 0.0),
        "pnl": float(getattr(event, "pnl", 0.0) or 0.0),
        "fee": float(getattr(event, "fee", 0.0) or 0.0),
        "pb_order_type": str(getattr(event, "pb_order_type", "") or "").lower(),
        "client_order_id": getattr(event, "client_order_id", None),
        "source_ids": list(getattr(event, "source_ids", []) or []),
    }
    return {k: v for k, v in payload.items() if v is not None}


def _monitor_record_fill_history(self, event) -> Optional[dict]:
    publisher = getattr(self, "monitor_publisher", None)
    if publisher is None:
        return None
    return publisher.record_fill(
        self._monitor_fill_payload(event),
        symbol=getattr(event, "symbol", None),
        pside=str(getattr(event, "position_side", "") or "").lower() or None,
        ts=int(getattr(event, "timestamp", 0) or 0) or None,
        raw_payload=getattr(event, "raw", None),
    )


def _monitor_record_price_ticks(
    self,
    last_prices: dict[str, float],
    *,
    ts: Optional[int] = None,
    source: Optional[str] = None,
) -> int:
    publisher = getattr(self, "monitor_publisher", None)
    if publisher is None:
        return 0
    emitted = 0
    for symbol in sorted(last_prices):
        try:
            price = float(last_prices[symbol])
        except Exception:
            continue
        if not math.isfinite(price) or price <= 0.0:
            continue
        if publisher.record_price_tick(symbol, price, ts=ts, source=source) is not None:
            emitted += 1
    return emitted


def _monitor_handle_candlestick_persist(
    self,
    symbol: str,
    timeframe: str,
    batch: np.ndarray,
) -> None:
    publisher = getattr(self, "monitor_publisher", None)
    if publisher is None or not getattr(self, "_bot_ready", False):
        return
    timeframe = str(timeframe)
    if timeframe not in ("1m", "1h"):
        return
    if not isinstance(batch, np.ndarray) or batch.size == 0:
        return
    candles = [
        {
            "ts": int(row["ts"]),
            "o": float(row["o"]),
            "h": float(row["h"]),
            "l": float(row["l"]),
            "c": float(row["c"]),
            "bv": float(row["bv"]),
        }
        for row in batch
    ]
    publisher.record_completed_candles(symbol, timeframe, candles)


def _build_health_summary_payload(self, *, now_ms: Optional[int] = None) -> dict:
    now_ms = utc_ms() if now_ms is None else int(now_ms)
    n_long = 0
    n_short = 0
    for _, pos_data in self.positions.items():
        if pos_data.get("long", {}).get("size", 0.0) != 0.0:
            n_long += 1
        if pos_data.get("short", {}).get("size", 0.0) != 0.0:
            n_short += 1
    balance_raw = float(self.get_raw_balance())
    balance_snapped = float(self.get_hysteresis_snapped_balance())
    error_counts = getattr(self, "error_counts", [])
    recent_errors = len([x for x in error_counts if x > now_ms - 1000 * 60 * 60])
    payload = {
        "uptime_ms": max(0, now_ms - self._health_start_ms),
        "last_loop_duration_ms": int(getattr(self, "_last_loop_duration_ms", 0) or 0),
        "positions_long": n_long,
        "positions_short": n_short,
        "balance_raw": balance_raw,
        "balance_snapped": balance_snapped,
        "equity": float(getattr(self, "_monitor_last_equity", balance_raw) or balance_raw),
        "orders_placed": int(self._health_orders_placed),
        "orders_cancelled": int(self._health_orders_cancelled),
        "fills": int(self._health_fills),
        "pnl": float(self._health_pnl),
        "errors_last_hour": recent_errors,
        "ws_reconnects": int(self._health_ws_reconnects),
        "rate_limits": int(self._health_rate_limits),
    }
    rss = _get_process_rss_bytes()
    if rss is not None:
        payload["rss_bytes"] = int(rss)
    return payload


def _monitor_recent_orders_payload(
    self,
    orders: list[dict],
    *,
    limit: int = 20,
) -> list[dict]:
    trimmed = list(orders[-limit:]) if orders else []
    payloads: list[dict] = []
    for order in trimmed:
        try:
            payload = self._monitor_order_payload(order, source=str(order.get("source", "runtime")))
        except Exception:
            continue
        execution_ts = order.get("execution_timestamp")
        if execution_ts is not None:
            try:
                payload["execution_timestamp"] = int(execution_ts)
            except Exception:
                pass
        payloads.append(payload)
    return payloads


def _build_monitor_market_section(self) -> dict[str, dict]:
    symbols = (
        set(getattr(self, "active_symbols", []) or [])
        | set(getattr(self, "positions", {}).keys())
        | set(getattr(self, "open_orders", {}).keys())
        | set(getattr(self, "trailing_prices", {}).keys())
        | set(getattr(self, "effective_min_cost", {}).keys())
    )
    approved_minus_ignored = getattr(self, "approved_coins_minus_ignored_coins", {})
    approved = getattr(self, "approved_coins", {})
    ignored = getattr(self, "ignored_coins", {})
    for pside in ("long", "short"):
        symbols |= set(approved_minus_ignored.get(pside, set()) or set())
        symbols |= set(approved.get(pside, set()) or set())
        symbols |= set(ignored.get(pside, set()) or set())
    current_close_cache = getattr(getattr(self, "cm", None), "_current_close_cache", {})
    runtime_hints = getattr(self, "_monitor_runtime_market_hints", {})
    out: dict[str, dict] = {}
    for symbol in sorted(symbols):
        entry: dict[str, Any] = {
            "active_symbol": symbol in set(getattr(self, "active_symbols", []) or []),
            "tradable": bool(getattr(self, "markets_dict", {}).get(symbol, {}).get("active", True)),
            "approved": {
                "long": symbol in set(approved_minus_ignored.get("long", set()) or set()),
                "short": symbol in set(approved_minus_ignored.get("short", set()) or set()),
            },
            "ignored": {
                "long": symbol in set(ignored.get("long", set()) or set()),
                "short": symbol in set(ignored.get("short", set()) or set()),
            },
            "effective_min_cost": float(
                getattr(self, "effective_min_cost", {}).get(symbol, 0.0) or 0.0
            ),
            "min_cost": float(getattr(self, "min_costs", {}).get(symbol, 0.0) or 0.0),
            "min_qty": float(getattr(self, "min_qtys", {}).get(symbol, 0.0) or 0.0),
            "price_step": float(getattr(self, "price_steps", {}).get(symbol, 0.0) or 0.0),
            "qty_step": float(getattr(self, "qty_steps", {}).get(symbol, 0.0) or 0.0),
            "has_open_orders": bool(getattr(self, "open_orders", {}).get(symbol)),
            "has_position": bool(self.has_position(symbol=symbol)),
        }
        cached = current_close_cache.get(symbol)
        if cached is not None:
            try:
                entry["last_price"] = float(cached[0])
                entry["last_price_ts_ms"] = int(cached[1])
            except Exception:
                pass
        try:
            entry["last_refresh_ms"] = int(self.cm.get_last_refresh_ms(symbol))
            entry["last_final_candle_ts_ms"] = int(self.cm.get_last_final_ts(symbol))
        except Exception:
            pass
        trailing = getattr(self, "trailing_prices", {}).get(symbol, {})
        if trailing:
            entry["trailing"] = {
                pside: {k: float(v) for k, v in trailing.get(pside, {}).items()}
                for pside in ("long", "short")
                if trailing.get(pside)
            }
        hint = runtime_hints.get(symbol, {})
        if isinstance(hint, dict):
            ema_bands = hint.get("ema_bands")
            if isinstance(ema_bands, dict):
                entry["ema_bands"] = deepcopy(ema_bands)
        out[symbol] = entry
    return out


def _build_monitor_forager_section(self) -> dict[str, dict]:
    out: dict[str, dict] = {}
    approved_minus_ignored = getattr(self, "approved_coins_minus_ignored_coins", {})
    approved = getattr(self, "approved_coins", {})
    ignored = getattr(self, "ignored_coins", {})
    for pside in ("long", "short"):
        candidate_universe = sorted(approved_minus_ignored.get(pside, set()) or set())
        approved_symbols = sorted(approved.get(pside, set()) or set())
        ignored_symbols = sorted(ignored.get(pside, set()) or set())
        current_n = int(self.get_current_n_positions(pside))
        max_n = int(self.get_max_n_positions(pside))
        selected_symbols = sorted(
            set(getattr(self, "PB_modes", {}).get(pside, {}).keys())
            | {sym for sym in getattr(self, "positions", {}) if self.has_position(pside, sym)}
        )
        held_symbols = {sym for sym in getattr(self, "positions", {}) if self.has_position(pside, sym)}
        entry_order_symbols = set()
        for symbol in getattr(self, "open_orders", {}):
            for order in self.open_orders.get(symbol, []):
                if str(order.get("position_side", "")) != pside:
                    continue
                side = str(order.get("side", ""))
                reduce_only = (pside == "long" and side == "sell") or (
                    pside == "short" and side == "buy"
                )
                if not reduce_only:
                    entry_order_symbols.add(symbol)
                    break
        pending_symbols = [
            sym for sym in selected_symbols if sym not in held_symbols and sym not in entry_order_symbols
        ]
        out[pside] = {
            "enabled": bool(self.is_pside_enabled(pside)),
            "forager_mode": bool(self.is_forager_mode(pside)),
            "forced_mode": self.live_value(f"forced_mode_{pside}"),
            "slots": {
                "current": current_n,
                "max": max_n,
                "open": max(0, max_n - current_n),
            },
            "approved_symbols": approved_symbols,
            "ignored_symbols": ignored_symbols,
            "candidate_universe": candidate_universe,
            "selected_symbols": selected_symbols,
            "active_symbols": sorted(
                sym
                for sym in set(getattr(self, "active_symbols", []) or [])
                if sym in candidate_universe or sym in selected_symbols
            ),
            "pending_symbols": pending_symbols,
            "next_symbol": pending_symbols[0] if pending_symbols else None,
            "score_weights": dict(self.bot_value(pside, "forager_score_weights") or {}),
            "volume_drop_pct": float(self.bot_value(pside, "forager_volume_drop_pct") or 0.0),
        }
    return out


def _build_monitor_unstuck_section(self) -> dict[str, Any]:
    has_open = bool(self.has_open_unstuck_order())
    allowances_live = self._calc_unstuck_allowances_live(allow_new_unstuck=not has_open)
    out: dict[str, Any] = {
        "has_open_order": has_open,
        "open_orders": [],
        "sides": {},
    }
    runtime_hints = getattr(self, "_monitor_runtime_unstuck_hints", {})
    for symbol in sorted(getattr(self, "open_orders", {})):
        for order in self.open_orders.get(symbol, []):
            try:
                pb_order_type = str(order.get("custom_id", "")).lower()
            except Exception:
                pb_order_type = ""
            if "unstuck" not in pb_order_type and "unstuck" not in str(
                self._resolve_pb_order_type(order)
            ):
                continue
            payload = self._monitor_order_payload(order, source="REST")
            payload["symbol"] = symbol
            out["open_orders"].append(payload)
    for pside in ("long", "short"):
        info = self._calc_unstuck_allowance_for_logging(pside)
        side_payload: dict[str, Any] = {
            "status": info.get("status"),
            "allowance_live": float(allowances_live.get(pside, 0.0) or 0.0),
            "configured_loss_allowance_pct": float(
                self.bot_value(pside, "unstuck_loss_allowance_pct") or 0.0
            ),
            "configured_close_pct": float(self.bot_value(pside, "unstuck_close_pct") or 0.0),
            "configured_threshold": float(self.bot_value(pside, "unstuck_threshold") or 0.0),
        }
        for key in ("allowance", "peak", "pct_from_peak"):
            if key in info:
                side_payload[key] = float(info[key])
        hint = runtime_hints.get(pside, {})
        if isinstance(hint, dict):
            for key in (
                "next_symbol",
                "next_target_price",
                "next_target_distance_ratio",
                "next_unstuck_trigger_distance_ratio",
            ):
                if key in hint and hint[key] is not None:
                    side_payload[key] = hint[key]
            ema_bands = hint.get("ema_bands")
            if isinstance(ema_bands, dict):
                side_payload["ema_bands"] = deepcopy(ema_bands)
        out["sides"][pside] = side_payload
    return out


def _build_monitor_runtime_market_hints(
    self,
    symbols: Iterable[str],
    last_prices: dict[str, float],
    m1_close_emas: dict[str, dict[float, float]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        hint: dict[str, Any] = {}
        per_side: dict[str, dict[str, float]] = {}
        last_price = last_prices.get(symbol)
        for pside in ("long", "short"):
            try:
                span0 = float(self.bp(pside, "ema_span_0", symbol))
                span1 = float(self.bp(pside, "ema_span_1", symbol))
                entry_dist = float(self.bp(pside, "entry_initial_ema_dist", symbol))
                unstuck_ema_dist = float(self.bp(pside, "unstuck_ema_dist", symbol))
            except Exception:
                continue
            if span0 <= 0.0 or span1 <= 0.0:
                continue
            span2 = (span0 * span1) ** 0.5
            emas = m1_close_emas.get(symbol, {})
            ema0 = float(emas.get(span0, 0.0) or 0.0)
            ema1 = float(emas.get(span1, 0.0) or 0.0)
            ema2 = float(emas.get(span2, 0.0) or 0.0)
            if min(ema0, ema1, ema2) <= 0.0:
                continue
            ema_lower = min(ema0, ema1, ema2)
            ema_upper = max(ema0, ema1, ema2)
            side_hint: dict[str, float] = {
                "lower": float(ema_lower),
                "upper": float(ema_upper),
                "entry_trigger_price": float(
                    ema_lower * (1.0 - entry_dist) if pside == "long" else ema_upper * (1.0 + entry_dist)
                ),
                "unstuck_trigger_price": float(
                    ema_upper * (1.0 + unstuck_ema_dist)
                    if pside == "long"
                    else ema_lower * (1.0 - unstuck_ema_dist)
                ),
            }
            if last_price is not None and float(last_price) > 0.0:
                side_hint["last_price"] = float(last_price)
            per_side[pside] = side_hint
        if per_side:
            hint["ema_bands"] = per_side
            out[symbol] = hint
    return out


def _build_monitor_runtime_unstuck_hints(
    self,
    idx_to_symbol: dict[int, str],
    orders: list[dict[str, Any]],
    last_prices: dict[str, float],
    market_hints: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {"long": {}, "short": {}}
    for order in orders:
        order_type_str = str(order.get("order_type", ""))
        if "close_unstuck" not in order_type_str:
            continue
        symbol = idx_to_symbol.get(int(order.get("symbol_idx", -1)))
        if symbol is None:
            continue
        pside = "long" if "long" in order_type_str else "short"
        target_price = float(order.get("price", 0.0) or 0.0)
        current_price = float(last_prices.get(symbol, 0.0) or 0.0)
        distance_ratio = None
        if current_price > 0.0 and target_price > 0.0:
            distance_ratio = float(target_price / current_price - 1.0)
        hint = {
            "next_symbol": symbol,
            "next_target_price": target_price,
            "next_target_distance_ratio": distance_ratio,
        }
        market_hint = market_hints.get(symbol, {})
        if isinstance(market_hint, dict):
            ema_bands = market_hint.get("ema_bands", {})
            if isinstance(ema_bands, dict) and isinstance(ema_bands.get(pside), dict):
                side_ema_bands = deepcopy(ema_bands.get(pside))
                hint["ema_bands"] = side_ema_bands
                trigger_price = float(side_ema_bands.get("unstuck_trigger_price", 0.0) or 0.0)
                if current_price > 0.0 and trigger_price > 0.0:
                    hint["next_unstuck_trigger_distance_ratio"] = float(
                        trigger_price / current_price - 1.0
                    )
        out[pside] = hint
        break
    return out


def _update_monitor_runtime_hints(
    self,
    *,
    symbols: Iterable[str],
    last_prices: dict[str, float],
    m1_close_emas: dict[str, dict[float, float]],
    idx_to_symbol: dict[int, str],
    orders: list[dict[str, Any]],
) -> None:
    market_hints = self._build_monitor_runtime_market_hints(symbols, last_prices, m1_close_emas)
    self._monitor_runtime_market_hints = market_hints
    self._monitor_runtime_unstuck_hints = self._build_monitor_runtime_unstuck_hints(
        idx_to_symbol,
        orders,
        last_prices,
        market_hints,
    )


def _build_monitor_recent_section(self) -> dict[str, Any]:
    return {
        "order_executions": self._monitor_recent_orders_payload(
            getattr(self, "recent_order_executions", [])
        ),
        "order_cancellations": self._monitor_recent_orders_payload(
            getattr(self, "recent_order_cancellations", [])
        ),
    }


def _build_monitor_position_side_payload(
    self,
    symbol: str,
    pside: str,
    pos: dict[str, Any],
    *,
    balance_raw: float,
    last_price: Optional[float],
    total_we_by_pside: dict[str, float],
) -> dict[str, Any]:
    size = float(pos.get("size", 0.0) or 0.0)
    price = float(pos.get("price", 0.0) or 0.0)
    wallet_exposure = 0.0
    if size != 0.0 and balance_raw > 0.0 and symbol in self.c_mults:
        wallet_exposure = float(pbr.qty_to_cost(size, price, self.c_mults[symbol]) / balance_raw)
    wel = float(self.bp(pside, "wallet_exposure_limit", symbol))
    allowance_pct = float(self.bp(pside, "risk_we_excess_allowance_pct", symbol))
    effective_wel = wel * (1.0 + max(0.0, allowance_pct))
    twel = float(self.bot_value(pside, "total_wallet_exposure_limit") or 0.0)

    payload: dict[str, Any] = {
        "size": size,
        "price": price,
        "wallet_exposure": wallet_exposure,
        "wallet_exposure_limit": wel,
        "effective_wallet_exposure_limit": effective_wel,
        "wel_ratio": wallet_exposure / wel if wel > 0.0 else 0.0,
        "wele_ratio": wallet_exposure / effective_wel if effective_wel > 0.0 else 0.0,
        "total_wallet_exposure": float(total_we_by_pside.get(pside, 0.0) or 0.0),
        "total_wallet_exposure_limit": twel,
        "twel_ratio": float(total_we_by_pside.get(pside, 0.0) or 0.0) / twel if twel > 0.0 else 0.0,
    }
    if last_price is None:
        return payload

    payload["last_price"] = float(last_price)
    if size == 0.0 or price == 0.0 or symbol not in self.c_mults:
        payload["price_action_distance"] = 0.0
        payload["upnl"] = 0.0
        return payload
    payload["price_action_distance"] = float(
        pbr.calc_pprice_diff_int(self.pside_int_map[pside], price, last_price)
    )
    payload["upnl"] = float(_calc_monitor_pnl(pside, price, last_price, size, self.c_mults[symbol]))
    return payload


def _build_monitor_positions_section(
    self,
    *,
    balance_raw: float,
    market: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    positions: dict[str, dict[str, Any]] = {}
    total_we_by_pside = {"long": 0.0, "short": 0.0}
    for symbol, pos in self.positions.items():
        for pside in ("long", "short"):
            side_pos = pos.get(pside, {})
            if not isinstance(side_pos, dict):
                continue
            size = float(side_pos.get("size", 0.0) or 0.0)
            price = float(side_pos.get("price", 0.0) or 0.0)
            if size != 0.0 and balance_raw > 0.0 and symbol in self.c_mults:
                total_we_by_pside[pside] += pbr.qty_to_cost(size, price, self.c_mults[symbol]) / balance_raw

    for symbol in sorted(self.positions):
        pos = self.positions[symbol]
        market_entry = market.get(symbol, {}) if isinstance(market.get(symbol), dict) else {}
        last_price_raw = market_entry.get("last_price")
        last_price = float(last_price_raw) if last_price_raw is not None else None
        long_pos = pos.get("long", {}) if isinstance(pos.get("long"), dict) else {}
        short_pos = pos.get("short", {}) if isinstance(pos.get("short"), dict) else {}
        positions[symbol] = {
            "long": self._build_monitor_position_side_payload(
                symbol,
                "long",
                long_pos,
                balance_raw=balance_raw,
                last_price=last_price,
                total_we_by_pside=total_we_by_pside,
            ),
            "short": self._build_monitor_position_side_payload(
                symbol,
                "short",
                short_pos,
                balance_raw=balance_raw,
                last_price=last_price,
                total_we_by_pside=total_we_by_pside,
            ),
        }
    return positions


async def _build_monitor_snapshot(self, *, now_ms: Optional[int] = None) -> dict:
    now_ms = utc_ms() if now_ms is None else int(now_ms)
    balance_raw = float(self.get_raw_balance())
    balance_snapped = float(self.get_hysteresis_snapped_balance())
    equity = float(getattr(self, "_monitor_last_equity", balance_raw) or balance_raw)
    if abs(equity) < 1e-18 and balance_raw != 0.0:
        equity = balance_raw
    account = {
        "balance_raw": balance_raw,
        "balance_snapped": balance_snapped,
        "equity": equity,
    }
    try:
        account["realized_pnl_cumsum"] = {
            "current": float(self._equity_hard_stop_realized_pnl_now()),
        }
    except Exception:
        pass
    market = self._build_monitor_market_section()
    positions = self._build_monitor_positions_section(balance_raw=balance_raw, market=market)

    open_orders: dict[str, list[dict]] = {}
    for symbol in sorted(self.open_orders):
        symbol_orders = []
        for order in self.open_orders.get(symbol, []):
            try:
                symbol_orders.append(
                    {
                        "id": order.get("id"),
                        "custom_id": order.get("custom_id"),
                        "side": order["side"],
                        "position_side": order["position_side"],
                        "qty": abs(float(order["qty"])),
                        "price": float(order["price"]),
                        "reduce_only": bool(
                            (order["position_side"] == "long" and order["side"] == "sell")
                            or (order["position_side"] == "short" and order["side"] == "buy")
                        ),
                        "pb_order_type": self._resolve_pb_order_type(order),
                    }
                )
            except Exception:
                continue
        open_orders[symbol] = symbol_orders

    return {
        "meta": {
            "exchange": self.exchange,
            "user": self.user,
            "quote": self.quote,
            "pid": os.getpid(),
            "bot_start_ts_ms": self.start_time_ms,
            "current_cycle_ts_ms": now_ms,
        },
        "account": account,
        "health": self._build_health_summary_payload(now_ms=now_ms),
        "positions": positions,
        "open_orders": open_orders,
        "modes": {
            "effective": {
                "long": dict(self.PB_modes.get("long", {})),
                "short": dict(self.PB_modes.get("short", {})),
            },
            "runtime_forced": {
                "long": dict(self._runtime_forced_modes.get("long", {})),
                "short": dict(self._runtime_forced_modes.get("short", {})),
            },
        },
        "hsl": {pside: self._monitor_hsl_payload(pside) for pside in ("long", "short")},
        "market": market,
        "forager": self._build_monitor_forager_section(),
        "unstuck": self._build_monitor_unstuck_section(),
        "recent": self._build_monitor_recent_section(),
    }


async def _monitor_flush_snapshot(self, *, force: bool = False, ts: Optional[int] = None) -> bool:
    publisher = getattr(self, "monitor_publisher", None)
    if publisher is None:
        return False
    try:
        snapshot = await self._build_monitor_snapshot(now_ms=ts)
        return publisher.write_snapshot(snapshot, ts=ts, force=force)
    except Exception as exc:
        logging.error("[monitor] failed building monitor snapshot: %s", exc)
        return False
