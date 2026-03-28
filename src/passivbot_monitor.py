from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

import numpy as np

from utils import utc_ms


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
    return self._monitor_record_event("bot.stop", ("bot", "lifecycle", "stop"), stop_payload, ts=ts)


def _monitor_hsl_payload(self) -> dict[str, Any]:
    enabled = bool(self._equity_hard_stop_enabled())
    last_metrics = getattr(self, "_equity_hard_stop_last_metrics", None)
    payload = {
        "enabled": enabled,
        "tier": (
            str(last_metrics.get("tier", "unknown"))
            if isinstance(last_metrics, dict)
            else ("disabled" if not enabled else "unknown")
        ),
        "halted": bool(getattr(self, "_equity_hard_stop_halted", False)),
        "no_restart_latched": bool(getattr(self, "_equity_hard_stop_no_restart_latched", False)),
        "pending_red_since_ms": getattr(self, "_equity_hard_stop_pending_red_since_ms", None),
        "cooldown_until_ms": getattr(self, "_equity_hard_stop_halted_until_ms", None),
        "cooldown_intervention_active": bool(
            getattr(self, "_equity_hard_stop_cooldown_intervention_active", False)
        ),
        "cooldown_repanic_reset_pending": bool(
            getattr(self, "_equity_hard_stop_cooldown_repanic_reset_pending", False)
        ),
        "last_metrics": dict(last_metrics) if isinstance(last_metrics, dict) else {},
        "last_stop_event": getattr(self, "_equity_hard_stop_last_stop_event", None),
    }
    return {k: v for k, v in payload.items() if v is not None}


def _monitor_order_payload(self, order: dict, *, source: str) -> dict[str, Any]:
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


def _monitor_fill_payload(self, event) -> dict[str, Any]:
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
        if not np.isfinite(price) or price <= 0.0:
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


def _build_health_summary_payload(self, *, now_ms: Optional[int] = None) -> dict[str, Any]:
    now_ms = utc_ms() if now_ms is None else int(now_ms)
    n_long = 0
    n_short = 0
    for _, pos_data in getattr(self, "positions", {}).items():
        if pos_data.get("long", {}).get("size", 0.0) != 0.0:
            n_long += 1
        if pos_data.get("short", {}).get("size", 0.0) != 0.0:
            n_short += 1
    balance_raw = float(self.get_raw_balance())
    balance_snapped = float(self.get_hysteresis_snapped_balance())
    error_counts = getattr(self, "error_counts", [])
    recent_errors = len([x for x in error_counts if x > now_ms - 1000 * 60 * 60])
    return {
        "uptime_ms": max(0, now_ms - int(getattr(self, "_health_start_ms", now_ms))),
        "last_loop_duration_ms": int(getattr(self, "_last_loop_duration_ms", 0) or 0),
        "positions_long": n_long,
        "positions_short": n_short,
        "balance_raw": balance_raw,
        "balance_snapped": balance_snapped,
        "equity": float(getattr(self, "_monitor_last_equity", balance_raw) or balance_raw),
        "orders_placed": int(getattr(self, "_health_orders_placed", 0)),
        "orders_cancelled": int(getattr(self, "_health_orders_cancelled", 0)),
        "fills": int(getattr(self, "_health_fills", 0)),
        "pnl": float(getattr(self, "_health_pnl", 0.0)),
        "errors_last_hour": recent_errors,
        "ws_reconnects": int(getattr(self, "_health_ws_reconnects", 0)),
        "rate_limits": int(getattr(self, "_health_rate_limits", 0)),
    }


def _monitor_recent_orders_payload(self, orders: list[dict], *, limit: int = 20) -> list[dict]:
    trimmed = list(orders[-limit:]) if orders else []
    payloads = []
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


def _build_monitor_snapshot(self, *, now_ms: Optional[int] = None) -> dict[str, Any]:
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

    symbols = (
        set(getattr(self, "active_symbols", []) or [])
        | set(getattr(self, "positions", {}).keys())
        | set(getattr(self, "open_orders", {}).keys())
    )
    market: dict[str, dict[str, Any]] = {}
    current_close_cache = getattr(getattr(self, "cm", None), "_current_close_cache", {})
    for symbol in sorted(symbols):
        entry = {
            "active_symbol": symbol in set(getattr(self, "active_symbols", []) or []),
            "tradable": bool(getattr(self, "markets_dict", {}).get(symbol, {}).get("active", True)),
            "effective_min_cost": float(
                getattr(self, "effective_min_cost", {}).get(symbol, 0.0) or 0.0
            ),
            "min_cost": float(getattr(self, "min_costs", {}).get(symbol, 0.0) or 0.0),
            "min_qty": float(getattr(self, "min_qtys", {}).get(symbol, 0.0) or 0.0),
            "price_step": float(getattr(self, "price_steps", {}).get(symbol, 0.0) or 0.0),
            "qty_step": float(getattr(self, "qty_steps", {}).get(symbol, 0.0) or 0.0),
            "c_mult": float(getattr(self, "c_mults", {}).get(symbol, 0.0) or 0.0),
            "has_open_orders": bool(getattr(self, "open_orders", {}).get(symbol)),
            "has_position": bool(getattr(self, "positions", {}).get(symbol)),
        }
        cached = current_close_cache.get(symbol)
        if cached is not None:
            try:
                entry["last_price"] = float(cached[0])
                entry["last_price_ts_ms"] = int(cached[1])
            except Exception:
                pass
        market[symbol] = entry

    positions: dict[str, dict[str, Any]] = {}
    for symbol in sorted(getattr(self, "positions", {})):
        symbol_positions: dict[str, Any] = {}
        for pside in ("long", "short"):
            pos = getattr(self, "positions", {}).get(symbol, {}).get(pside, {})
            if isinstance(pos, dict):
                symbol_positions[pside] = {
                    "size": float(pos.get("size", 0.0) or 0.0),
                    "price": float(pos.get("price", 0.0) or 0.0),
                }
        positions[symbol] = symbol_positions

    open_orders: dict[str, list[dict[str, Any]]] = {}
    for symbol in sorted(getattr(self, "open_orders", {})):
        open_orders[symbol] = []
        for order in getattr(self, "open_orders", {}).get(symbol, []):
            try:
                open_orders[symbol].append(self._monitor_order_payload(order, source="REST"))
            except Exception:
                continue

    return {
        "meta": {
            "exchange": self.exchange,
            "user": self.user,
            "quote": self.quote,
            "pid": getattr(self, "pid", None),
            "bot_start_ts_ms": getattr(self, "start_time_ms", now_ms),
            "current_cycle_ts_ms": now_ms,
        },
        "account": account,
        "health": self._build_health_summary_payload(now_ms=now_ms),
        "positions": positions,
        "open_orders": open_orders,
        "modes": {
            "effective": {
                "long": dict(getattr(self, "PB_modes", {}).get("long", {})),
                "short": dict(getattr(self, "PB_modes", {}).get("short", {})),
            },
            "runtime_forced": {
                "long": dict(getattr(self, "_runtime_forced_modes", {}).get("long", {})),
                "short": dict(getattr(self, "_runtime_forced_modes", {}).get("short", {})),
            },
        },
        "hsl": self._monitor_hsl_payload(),
        "market": market,
        "recent": {
            "order_executions": self._monitor_recent_orders_payload(
                getattr(self, "recent_order_executions", [])
            ),
            "order_cancellations": self._monitor_recent_orders_payload(
                getattr(self, "recent_order_cancellations", [])
            ),
        },
    }


async def _monitor_flush_snapshot(self, *, force: bool = False, ts: Optional[int] = None) -> bool:
    publisher = getattr(self, "monitor_publisher", None)
    if publisher is None:
        return False
    try:
        snapshot = self._build_monitor_snapshot(now_ms=ts)
        return publisher.write_snapshot(snapshot, ts=ts, force=force)
    except Exception as exc:
        logging.error("[monitor] failed building monitor snapshot: %s", exc)
        return False
