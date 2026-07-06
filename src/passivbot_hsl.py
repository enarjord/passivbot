from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Optional

import passivbot_rust as pbr

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
from live.event_bus import EventTypes, ReasonCodes, live_event_debug_profile_enabled
from passivbot_exceptions import RestartBotException
from utils import make_get_filepath


_HSL_RISKS_DOC = "docs/equity_hard_stop_loss_risks.md"
_HSL_REPLAY_MATRIX_INTERVAL_MS = 60_000
_HSL_REPLAY_MATRIX_RAW_FIELDS = ("ts", "price", "psize", "pprice", "pnl", "upnl")
_HSL_REPLAY_ACCOUNT_SERIES_FIELDS = ("ts", "pnl")
_HSL_REPLAY_CACHE_SERIES_KINDS = ("pair_matrix", "account_pnl")
_HSL_REPLAY_CACHE_ACCOUNT_PSIDE = "account"
_HSL_REPLAY_CACHE_ACCOUNT_SYMBOL = "__account__"
_HSL_REPLAY_CACHE_SCHEMA_VERSION = 4
_HSL_REPLAY_CACHE_MATRIX_FILENAME = "hsl_replay_matrix.npz"
_HSL_REPLAY_CACHE_MANIFEST_FILENAME = "hsl_replay_manifest.json"
_HSL_REPLAY_CACHE_REQUIRED_METADATA = (
    "exchange",
    "market_type",
    "user",
    "config_digest",
    "signal_mode",
    "pside",
    "symbol",
    "fill_covered_start_ms",
    "fill_covered_end_ms",
    "fill_history_scope",
    "fill_coverage_proven",
    "candle_covered_start_ms",
    "candle_covered_end_ms",
)
_HSL_REPLAY_CACHE_FILL_HISTORY_SCOPES = ("unknown", "window", "all")


def _hsl_replay_cache_safe_fragment(value: Any) -> str:
    raw = str(value).strip()
    out = []
    for char in raw:
        if char.isalnum() or char in {"-", "_", "."}:
            out.append(char)
        else:
            out.append("_")
    safe = "".join(out).strip("._")
    return safe or "unknown"


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
            exc,
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
        logging.debug("[event] failed to emit HSL live event type=%s: %s", event_type, exc)
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
            exc,
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
            exc,
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
        logging.debug("[event] failed to emit HSL replay event type=%s: %s", event_type, exc)


def _calc_hsl_pnl(position_side, entry_price, close_price, qty, c_mult):
    if isinstance(position_side, str):
        if position_side == "long":
            return pbr.calc_pnl_long(entry_price, close_price, qty, c_mult)
        return pbr.calc_pnl_short(entry_price, close_price, qty, c_mult)
    return pbr.calc_pnl_long(entry_price, close_price, qty, c_mult)


def _finite_hsl_float(value: Any, field_name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"HSL replay matrix field {field_name} must be finite, got {out}")
    return out


def _hsl_replay_matrix_row(
    *,
    pside: str,
    ts: int,
    price: float,
    psize: float,
    pprice: float,
    pnl: float,
    c_mult: float,
) -> dict[str, float | int]:
    """Build one non-authoritative cache row from authoritative candles/fills."""
    if pside not in {"long", "short"}:
        raise ValueError(f"HSL replay matrix pside must be long or short, got {pside!r}")
    ts_int = int(ts)
    if ts_int < 0:
        raise ValueError(f"HSL replay matrix ts must be >= 0, got {ts_int}")
    price_f = _finite_hsl_float(price, "price")
    psize_f = _finite_hsl_float(psize, "psize")
    pprice_f = _finite_hsl_float(pprice, "pprice")
    pnl_f = _finite_hsl_float(pnl, "pnl")
    c_mult_f = _finite_hsl_float(c_mult, "c_mult")
    if price_f <= 0.0:
        raise ValueError(f"HSL replay matrix price must be > 0, got {price_f}")
    if c_mult_f <= 0.0:
        raise ValueError(f"HSL replay matrix c_mult must be > 0, got {c_mult_f}")
    if abs(psize_f) <= 0.0:
        upnl = 0.0
    else:
        if pprice_f <= 0.0:
            raise ValueError(
                f"HSL replay matrix pprice must be > 0 for non-flat rows, got {pprice_f}"
            )
        qty = abs(psize_f)
        upnl = float(_calc_hsl_pnl(pside, pprice_f, price_f, qty, c_mult_f))
    return {
        "ts": ts_int,
        "price": price_f,
        "psize": psize_f,
        "pprice": pprice_f,
        "pnl": pnl_f,
        "upnl": upnl,
    }


def _hsl_replay_account_series_row(*, ts: int, pnl: float) -> dict[str, float | int]:
    """Build one non-authoritative account-level realized-PnL row."""
    ts_int = int(ts)
    if ts_int < 0:
        raise ValueError(f"HSL account series ts must be >= 0, got {ts_int}")
    return {"ts": ts_int, "pnl": _finite_hsl_float(pnl, "pnl")}


def _hsl_replay_account_series_arrays(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    prev_ts: int | None = None
    for row in rows:
        missing = [field for field in _HSL_REPLAY_ACCOUNT_SERIES_FIELDS if field not in row]
        if missing:
            raise ValueError(f"HSL account series row missing fields: {missing}")
        ts = int(row["ts"])
        if ts < 0:
            raise ValueError(f"HSL account series ts must be >= 0, got {ts}")
        if prev_ts is not None and ts - prev_ts != _HSL_REPLAY_MATRIX_INTERVAL_MS:
            raise ValueError(
                "HSL account series rows must be contiguous 1m samples; "
                f"got previous_ts={prev_ts} ts={ts}"
            )
        _finite_hsl_float(row["pnl"], "pnl")
        prev_ts = ts
    return {
        "ts": np.asarray([int(row["ts"]) for row in rows], dtype=np.int64),
        "pnl": np.asarray([float(row["pnl"]) for row in rows], dtype=np.float64),
    }


def _hsl_replay_matrix_derived_series(
    rows: list[dict[str, Any]], *, base_equity: float
) -> list[dict[str, float | int]]:
    """Derive cumulative PnL/equity without treating persisted raw PnL as cumulative."""
    base_equity_f = _finite_hsl_float(base_equity, "base_equity")
    if base_equity_f <= 0.0:
        raise ValueError(f"HSL replay matrix base_equity must be > 0, got {base_equity_f}")
    out: list[dict[str, float | int]] = []
    pnl_cumsum = 0.0
    prev_ts: int | None = None
    for row in rows:
        missing = [field for field in _HSL_REPLAY_MATRIX_RAW_FIELDS if field not in row]
        if missing:
            raise ValueError(f"HSL replay matrix row missing fields: {missing}")
        ts = int(row["ts"])
        if prev_ts is not None and ts - prev_ts != _HSL_REPLAY_MATRIX_INTERVAL_MS:
            raise ValueError(
                "HSL replay matrix rows must be contiguous 1m samples; "
                f"got previous_ts={prev_ts} ts={ts}"
            )
        pnl = _finite_hsl_float(row["pnl"], "pnl")
        upnl = _finite_hsl_float(row["upnl"], "upnl")
        pnl_cumsum += pnl
        out.append(
            {
                "ts": ts,
                "pnl_cumsum": float(pnl_cumsum),
                "upnl": upnl,
                "equity": float(base_equity_f + pnl_cumsum + upnl),
            }
        )
        prev_ts = ts
    return out


def _hsl_replay_matrix_arrays(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    # Reuse the derived-series validation path for raw fields and 1m continuity.
    _hsl_replay_matrix_derived_series(rows, base_equity=1.0)
    for row in rows:
        price = _finite_hsl_float(row["price"], "price")
        psize = _finite_hsl_float(row["psize"], "psize")
        pprice = _finite_hsl_float(row["pprice"], "pprice")
        if price <= 0.0:
            raise ValueError(f"HSL replay matrix price must be > 0, got {price}")
        if abs(psize) > 0.0 and pprice <= 0.0:
            raise ValueError(
                f"HSL replay matrix pprice must be > 0 for non-flat rows, got {pprice}"
            )
    return {
        "ts": np.asarray([int(row["ts"]) for row in rows], dtype=np.int64),
        "price": np.asarray([float(row["price"]) for row in rows], dtype=np.float64),
        "psize": np.asarray([float(row["psize"]) for row in rows], dtype=np.float64),
        "pprice": np.asarray([float(row["pprice"]) for row in rows], dtype=np.float64),
        "pnl": np.asarray([float(row["pnl"]) for row in rows], dtype=np.float64),
        "upnl": np.asarray([float(row["upnl"]) for row in rows], dtype=np.float64),
    }


def _hsl_replay_matrix_derived_arrays(
    arrays: dict[str, Any], *, base_equity: float
) -> dict[str, Any]:
    import numpy as np

    if not isinstance(arrays, dict):
        raise TypeError(f"HSL replay matrix arrays must be a dict, got {type(arrays).__name__}")
    expected = set(_HSL_REPLAY_MATRIX_RAW_FIELDS)
    actual = set(arrays)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "HSL replay matrix arrays must contain exactly the raw fields; "
            f"missing={missing} extra={extra}"
        )
    base_equity_f = _finite_hsl_float(base_equity, "base_equity")
    if base_equity_f <= 0.0:
        raise ValueError(f"HSL replay matrix base_equity must be > 0, got {base_equity_f}")
    reasons = _hsl_replay_cache_array_value_reasons(arrays)
    if reasons:
        raise ValueError("HSL replay matrix arrays invalid: " + ", ".join(reasons))
    ts = np.asarray(arrays["ts"], dtype=np.int64)
    if len(ts) > 1 and not bool(np.all(np.diff(ts) == _HSL_REPLAY_MATRIX_INTERVAL_MS)):
        raise ValueError("HSL replay matrix arrays invalid: timestamp_not_contiguous")
    pnl = np.asarray(arrays["pnl"], dtype=np.float64)
    upnl = np.asarray(arrays["upnl"], dtype=np.float64)
    pnl_cumsum = np.cumsum(pnl, dtype=np.float64)
    return {
        "ts": ts.copy(),
        "pnl_cumsum": pnl_cumsum,
        "upnl": upnl.copy(),
        "equity": base_equity_f + pnl_cumsum + upnl,
    }


def _hsl_replay_timeline_rows_from_cache(
    pair_arrays: dict[tuple[str, str], dict[str, Any]],
    account_arrays: dict[str, Any],
    *,
    current_balance: float,
) -> list[dict[str, Any]]:
    """Synthesize coin-replay timeline rows from persisted cache arrays.

    Pure and unwired: this is the trust-boundary conversion a future reuse
    slice would feed into the existing coin replay loop. The output is an
    explicit coin-replay row contract, not the full unified timeline shape:
    each row carries exactly `timestamp`, `balance`, `realized_pnl`
    (account-level, anchored at the series start like the authoritative
    record-window anchor), `realized_pnl_by_coin_pside`, and
    `unrealized_pnl_by_coin_pside` — the fields the coin replay loop and its
    stop/latch payloads consume. Values must equal the authoritative
    `get_balance_equity_history` timeline for the covered pairs; consumers
    that need additional unified-timeline fields must fail loudly rather than
    assume this shape. Fail-loud on any input that cannot prove equivalence.
    Per-pair realized values are cumulative from each pair's matrix start; the
    replay windowing consumes differences, so a constant anchor offset versus
    the authoritative timeline is harmless by contract.
    """
    import numpy as np

    balance_f = _finite_hsl_float(current_balance, "current_balance")
    if balance_f <= 0.0:
        raise ValueError(f"current_balance must be > 0, got {balance_f}")
    account_reasons = _hsl_replay_cache_array_value_reasons(
        dict(account_arrays), series_kind="account_pnl"
    )
    if account_reasons:
        raise ValueError(
            "HSL cache account arrays invalid: " + ", ".join(account_reasons)
        )
    account_ts = np.asarray(account_arrays["ts"], dtype=np.int64)
    if account_ts.size == 0:
        raise ValueError("HSL cache account arrays must not be empty")
    if account_ts.size > 1 and not bool(
        np.all(np.diff(account_ts) == _HSL_REPLAY_MATRIX_INTERVAL_MS)
    ):
        raise ValueError("HSL cache account arrays invalid: timestamp_not_contiguous")
    account_pnl = np.asarray(account_arrays["pnl"], dtype=np.float64)
    account_cumsum = np.cumsum(account_pnl, dtype=np.float64)
    # Anchor the balance series so its final minute equals the current balance,
    # mirroring how the authoritative replay back-computes its baseline.
    balances = balance_f - (float(account_cumsum[-1]) - account_cumsum)
    if not bool(np.all(np.isfinite(balances))) or bool(np.any(balances <= 0.0)):
        raise ValueError(
            "HSL cache-derived balance series must be finite and > 0 for every minute"
        )
    start_ts = int(account_ts[0])
    end_ts = int(account_ts[-1])
    realized_by_minute: list[dict[str, dict[str, float]]] = [
        {} for _ in range(len(account_ts))
    ]
    unrealized_by_minute: list[dict[str, dict[str, float]]] = [
        {} for _ in range(len(account_ts))
    ]
    for pair, arrays in pair_arrays.items():
        pside, symbol = pair
        if pside not in ("long", "short"):
            raise ValueError(f"HSL cache pair pside must be long or short, got {pside!r}")
        pair_reasons = _hsl_replay_cache_array_value_reasons(
            dict(arrays), series_kind="pair_matrix"
        )
        if pair_reasons:
            raise ValueError(
                f"HSL cache pair arrays invalid for {pside}:{symbol}: "
                + ", ".join(pair_reasons)
            )
        pair_ts = np.asarray(arrays["ts"], dtype=np.int64)
        if pair_ts.size == 0:
            raise ValueError(f"HSL cache pair arrays empty for {pside}:{symbol}")
        if pair_ts.size > 1 and not bool(
            np.all(np.diff(pair_ts) == _HSL_REPLAY_MATRIX_INTERVAL_MS)
        ):
            raise ValueError(
                f"HSL cache pair arrays invalid for {pside}:{symbol}: "
                "timestamp_not_contiguous"
            )
        pair_start = int(pair_ts[0])
        pair_end = int(pair_ts[-1])
        if pair_start < start_ts or pair_end > end_ts:
            raise ValueError(
                f"HSL cache pair series for {pside}:{symbol} spans "
                f"[{pair_start}, {pair_end}] outside account span "
                f"[{start_ts}, {end_ts}]"
            )
        if (pair_start - start_ts) % _HSL_REPLAY_MATRIX_INTERVAL_MS != 0:
            raise ValueError(
                f"HSL cache pair series for {pside}:{symbol} is not aligned to "
                "the account minute grid"
            )
        offset = (pair_start - start_ts) // _HSL_REPLAY_MATRIX_INTERVAL_MS
        pair_cumsum = np.cumsum(
            np.asarray(arrays["pnl"], dtype=np.float64), dtype=np.float64
        )
        pair_upnl = np.asarray(arrays["upnl"], dtype=np.float64)
        for idx in range(len(pair_ts)):
            realized_by_minute[offset + idx].setdefault(symbol, {})[pside] = float(
                pair_cumsum[idx]
            )
            unrealized_by_minute[offset + idx].setdefault(symbol, {})[pside] = float(
                pair_upnl[idx]
            )
    rows: list[dict[str, Any]] = []
    for idx in range(len(account_ts)):
        rows.append(
            {
                "timestamp": int(account_ts[idx]),
                "balance": float(balances[idx]),
                "realized_pnl": float(account_cumsum[idx]),
                "realized_pnl_by_coin_pside": realized_by_minute[idx],
                "unrealized_pnl_by_coin_pside": unrealized_by_minute[idx],
            }
        )
    return rows


def _hsl_replay_rows_from_arrays(
    arrays: dict[str, Any], *, series_kind: str
) -> list[dict[str, Any]]:
    """Inverse of the arrays builders: expand validated arrays into row dicts."""
    fields = _hsl_replay_cache_series_fields(series_kind)
    reasons = _hsl_replay_cache_array_value_reasons(dict(arrays), series_kind=series_kind)
    if reasons:
        raise ValueError(
            f"HSL replay {series_kind} arrays invalid: " + ", ".join(reasons)
        )
    row_count = int(len(arrays["ts"]))
    rows: list[dict[str, Any]] = []
    for idx in range(row_count):
        row: dict[str, Any] = {}
        for field in fields:
            value = arrays[field][idx]
            row[field] = int(value) if field == "ts" else float(value)
        rows.append(row)
    return rows


def _hsl_replay_extension_minutes(
    watermark_ts: int, end_ts: int
) -> list[int]:
    end_minute = int(math.floor(int(end_ts) / _HSL_REPLAY_MATRIX_INTERVAL_MS)) * (
        _HSL_REPLAY_MATRIX_INTERVAL_MS
    )
    if end_minute < int(watermark_ts):
        raise ValueError(
            f"HSL cache extension end {end_ts} precedes watermark {watermark_ts}"
        )
    return list(
        range(
            int(watermark_ts) + _HSL_REPLAY_MATRIX_INTERVAL_MS,
            end_minute + _HSL_REPLAY_MATRIX_INTERVAL_MS,
            _HSL_REPLAY_MATRIX_INTERVAL_MS,
        )
    )


def _hsl_replay_extension_require_fill(fill: Any, watermark_ts: int) -> dict[str, Any]:
    if not isinstance(fill, dict):
        raise TypeError(
            f"HSL cache extension fills must be dicts, got {type(fill).__name__}"
        )
    for key in ("timestamp", "qty", "price", "action", "pnl"):
        if key not in fill:
            raise ValueError(f"HSL cache extension fill missing required key: {key}")
    ts = int(fill["timestamp"])
    if ts < int(watermark_ts) + _HSL_REPLAY_MATRIX_INTERVAL_MS:
        raise ValueError(
            "HSL cache extension fill lies inside the cached window "
            f"(ts={ts}, watermark={watermark_ts}); the cache must be rejected "
            "instead of double-counting fills"
        )
    return fill


def _hsl_replay_extend_pair_rows(
    arrays: dict[str, Any],
    *,
    pside: str,
    symbol: str,
    fills: list[dict[str, Any]],
    closes_by_minute: dict[int, float],
    end_ts: int,
    c_mult: float,
) -> list[dict[str, Any]]:
    """Extend a cached pair matrix from its watermark to `end_ts` (pure).

    Mirrors the authoritative replay's position bookkeeping exactly
    (`_apply_event` in get_balance_equity_history): increases move the
    weighted-average entry price, decreases shrink size and zero the price on
    flat. `fills` must contain only this pair's extracted events strictly
    after the cached window; anything else is rejected fail-loud because it
    would double-count or misattribute cached data.
    """
    if pside not in ("long", "short"):
        raise ValueError(f"HSL cache extension pside must be long or short, got {pside!r}")
    rows = _hsl_replay_rows_from_arrays(arrays, series_kind="pair_matrix")
    if not rows:
        raise ValueError(f"HSL cache extension requires non-empty pair arrays for {symbol}")
    watermark_ts = int(rows[-1]["ts"])
    minutes = _hsl_replay_extension_minutes(watermark_ts, end_ts)
    c_mult_f = _finite_hsl_float(c_mult, "c_mult")
    size = abs(float(rows[-1]["psize"]))
    pprice = float(rows[-1]["pprice"]) if size > 1e-12 else 0.0
    last_price = float(rows[-1]["price"])
    ordered = sorted(
        (_hsl_replay_extension_require_fill(fill, watermark_ts) for fill in fills),
        key=lambda fill: int(fill["timestamp"]),
    )
    for fill in ordered:
        # Pair identity must be explicit: defaulting a stripped fill to the
        # target pair would silently apply it to the wrong (or every) pair.
        if "symbol" not in fill or "pside" not in fill:
            raise ValueError(
                "HSL cache extension fill missing pair identity (symbol/pside) "
                f"for {pside}:{symbol}"
            )
        if str(fill["symbol"]) != symbol or str(fill["pside"]) != pside:
            raise ValueError(
                "HSL cache extension fill does not belong to pair "
                f"{pside}:{symbol}: {fill.get('pside')}:{fill.get('symbol')}"
            )
    if ordered and int(ordered[-1]["timestamp"]) >= (
        (minutes[-1] if minutes else watermark_ts) + _HSL_REPLAY_MATRIX_INTERVAL_MS
    ):
        raise ValueError(
            "HSL cache extension fill lies beyond the extension end "
            f"(ts={ordered[-1]['timestamp']})"
        )
    new_rows: list[dict[str, Any]] = []
    fill_idx = 0
    for minute in minutes:
        boundary = minute + _HSL_REPLAY_MATRIX_INTERVAL_MS
        minute_pnl = 0.0
        while fill_idx < len(ordered) and int(ordered[fill_idx]["timestamp"]) < boundary:
            fill = ordered[fill_idx]
            qty = abs(_finite_hsl_float(fill["qty"], "qty"))
            fill_price = _finite_hsl_float(fill["price"], "price")
            minute_pnl += _finite_hsl_float(fill["pnl"], "pnl") + _finite_hsl_float(
                fill.get("fee", 0.0), "fee"
            )
            if str(fill["action"]) == "increase":
                new_size = size + qty
                if new_size <= 0.0:
                    size, pprice = 0.0, 0.0
                elif size <= 0.0:
                    size, pprice = new_size, fill_price
                else:
                    pprice = max((size * pprice + qty * fill_price) / new_size, 0.0)
                    size = new_size
            else:
                size = max(size - qty, 0.0)
                if size <= 0.0:
                    pprice = 0.0
            fill_idx += 1
        close = closes_by_minute.get(minute)
        if close is not None:
            close_f = _finite_hsl_float(close, "close")
            if close_f <= 0.0:
                raise ValueError(
                    f"HSL cache extension close must be > 0 at {minute}, got {close_f}"
                )
            last_price = close_f
        psize = size if pside == "long" else -size
        new_rows.append(
            _hsl_replay_matrix_row(
                pside=pside,
                ts=int(minute),
                price=last_price,
                psize=psize if size > 1e-12 else 0.0,
                pprice=pprice if size > 1e-12 else 0.0,
                pnl=minute_pnl,
                c_mult=c_mult_f,
            )
        )
    return new_rows


def _hsl_replay_extend_account_rows(
    arrays: dict[str, Any],
    *,
    fills: list[dict[str, Any]],
    end_ts: int,
) -> list[dict[str, Any]]:
    """Extend a cached account pnl series from its watermark to `end_ts` (pure).

    `fills` must contain the extracted events for ALL symbols strictly after
    the cached window; per-minute pnl is the sum of `pnl + fee`, matching the
    authoritative balance accounting.
    """
    rows = _hsl_replay_rows_from_arrays(arrays, series_kind="account_pnl")
    if not rows:
        raise ValueError("HSL cache extension requires non-empty account arrays")
    watermark_ts = int(rows[-1]["ts"])
    minutes = _hsl_replay_extension_minutes(watermark_ts, end_ts)
    ordered = sorted(
        (_hsl_replay_extension_require_fill(fill, watermark_ts) for fill in fills),
        key=lambda fill: int(fill["timestamp"]),
    )
    if ordered and int(ordered[-1]["timestamp"]) >= (
        (minutes[-1] if minutes else watermark_ts) + _HSL_REPLAY_MATRIX_INTERVAL_MS
    ):
        raise ValueError(
            "HSL cache extension fill lies beyond the extension end "
            f"(ts={ordered[-1]['timestamp']})"
        )
    new_rows: list[dict[str, Any]] = []
    fill_idx = 0
    for minute in minutes:
        boundary = minute + _HSL_REPLAY_MATRIX_INTERVAL_MS
        minute_pnl = 0.0
        while fill_idx < len(ordered) and int(ordered[fill_idx]["timestamp"]) < boundary:
            fill = ordered[fill_idx]
            minute_pnl += _finite_hsl_float(fill["pnl"], "pnl") + _finite_hsl_float(
                fill.get("fee", 0.0), "fee"
            )
            fill_idx += 1
        new_rows.append(_hsl_replay_account_series_row(ts=int(minute), pnl=minute_pnl))
    return new_rows


def _hsl_hash_array(array: Any) -> str:
    import numpy as np

    arr = np.ascontiguousarray(np.asarray(array))
    hasher = hashlib.sha256()
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(arr.tobytes(order="C"))
    return hasher.hexdigest()


def _hsl_replay_cache_array_manifest(arrays: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        field: {
            "sha256": _hsl_hash_array(arrays[field]),
            "shape": [int(x) for x in arrays[field].shape],
            "dtype": str(arrays[field].dtype),
        }
        for field in sorted(arrays)
    }


def _hsl_replay_cache_array_dtype_is_valid(field: str, dtype: Any) -> bool:
    import numpy as np

    np_dtype = np.dtype(dtype)
    if np.issubdtype(np_dtype, np.complexfloating):
        return False
    if field == "ts":
        return bool(np.issubdtype(np_dtype, np.integer))
    return bool(np.issubdtype(np_dtype, np.integer) or np.issubdtype(np_dtype, np.floating))


def _hsl_replay_cache_series_fields(series_kind: str) -> tuple[str, ...]:
    if series_kind == "pair_matrix":
        return _HSL_REPLAY_MATRIX_RAW_FIELDS
    if series_kind == "account_pnl":
        return _HSL_REPLAY_ACCOUNT_SERIES_FIELDS
    raise ValueError(
        "HSL replay cache series_kind must be one of "
        f"{_HSL_REPLAY_CACHE_SERIES_KINDS}, got {series_kind!r}"
    )


def _hsl_replay_cache_array_value_reasons(
    arrays: dict[str, Any], *, series_kind: str = "pair_matrix"
) -> list[str]:
    import numpy as np

    fields = _hsl_replay_cache_series_fields(series_kind)
    reasons: list[str] = []
    required = set(fields)
    if set(arrays) != required:
        return reasons
    try:
        row_count = int(len(arrays["ts"]))
    except TypeError:
        return ["array_value_invalid:ts"]
    for field in fields:
        try:
            field_len = int(len(arrays[field]))
        except TypeError:
            reasons.append(f"array_value_invalid:{field}")
            continue
        if field_len != row_count:
            reasons.append(f"array_length_mismatch:{field}")
    if reasons:
        return reasons
    numeric_arrays: dict[str, Any] = {}
    for field in fields:
        arr = np.asarray(arrays[field])
        if arr.ndim != 1 or not _hsl_replay_cache_array_dtype_is_valid(field, arr.dtype):
            reasons.append(f"array_value_invalid:{field}")
            continue
        numeric_arrays[field] = arr
        try:
            if not bool(np.all(np.isfinite(arr))):
                reasons.append(f"array_nonfinite:{field}")
        except (TypeError, ValueError):
            reasons.append(f"array_value_invalid:{field}")
    if "ts" in numeric_arrays and row_count:
        try:
            if int(numeric_arrays["ts"][0]) < 0:
                reasons.append("timestamp_invalid")
        except (TypeError, ValueError, OverflowError):
            reasons.append("timestamp_invalid")
    if "price" in numeric_arrays and row_count and bool(np.any(numeric_arrays["price"] <= 0.0)):
        reasons.append("price_nonpositive")
    if {"psize", "pprice"}.issubset(numeric_arrays) and row_count:
        nonflat = np.abs(numeric_arrays["psize"]) > 0.0
        if bool(np.any(nonflat & (numeric_arrays["pprice"] <= 0.0))):
            reasons.append("nonflat_pprice_nonpositive")
    return reasons


def _normalize_hsl_replay_cache_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise TypeError(f"HSL replay cache metadata must be a dict, got {type(metadata).__name__}")
    missing = [key for key in _HSL_REPLAY_CACHE_REQUIRED_METADATA if metadata.get(key) is None]
    if missing:
        raise ValueError(f"HSL replay cache metadata missing required fields: {missing}")
    out = dict(metadata)
    for key in (
        "exchange",
        "market_type",
        "user",
        "config_digest",
        "signal_mode",
        "pside",
        "symbol",
        "fill_history_scope",
    ):
        out[key] = str(out[key])
    for key in (
        "fill_covered_start_ms",
        "fill_covered_end_ms",
        "candle_covered_start_ms",
        "candle_covered_end_ms",
    ):
        out[key] = int(out[key])
    if out["fill_history_scope"] not in _HSL_REPLAY_CACHE_FILL_HISTORY_SCOPES:
        raise ValueError(
            "HSL replay cache metadata fill_history_scope must be one of "
            f"{_HSL_REPLAY_CACHE_FILL_HISTORY_SCOPES}, got {out['fill_history_scope']!r}"
        )
    if not isinstance(out["fill_coverage_proven"], bool):
        raise ValueError(
            "HSL replay cache metadata fill_coverage_proven must be a bool, "
            f"got {type(out['fill_coverage_proven']).__name__}"
        )
    return out


def _hsl_replay_cache_json_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _hsl_replay_cache_config_digest(self, pside: str) -> str:
    if pside not in {"long", "short"}:
        raise ValueError(f"HSL replay cache pside must be long or short, got {pside!r}")
    hsl_cfg = getattr(self, "hsl", None)
    if not isinstance(hsl_cfg, dict) or pside not in hsl_cfg:
        raise ValueError(f"HSL replay cache requires parsed HSL config for {pside}")
    payload = {
        "schema_version": _HSL_REPLAY_CACHE_SCHEMA_VERSION,
        "signal_mode": self._equity_hard_stop_signal_mode(),
        "cooldown_position_policy": self._equity_hard_stop_cooldown_position_policy(),
        "pnls_max_lookback_days": parse_pnls_max_lookback_days(
            self.live_value("pnls_max_lookback_days"),
            field_name="live.pnls_max_lookback_days",
        ).display_value,
        "pside": pside,
        "hsl": hsl_cfg[pside],
        "n_positions": float(self.bot_value(pside, "n_positions")),
        "total_wallet_exposure_limit": float(
            self.bot_value(pside, "total_wallet_exposure_limit")
        ),
    }
    return _hsl_replay_cache_json_digest(payload)


def _hsl_replay_cache_market_type(self) -> str:
    for attr in ("market_type", "market"):
        value = getattr(self, attr, None)
        if value not in (None, ""):
            return str(value)
    config = getattr(self, "config", {})
    value = get_optional_live_value(config, "market_type", None)
    if value not in (None, ""):
        return str(value)
    return "unknown"


def _hsl_replay_cache_expected_metadata(
    self,
    pside: str,
    symbol: str,
    *,
    fill_covered_start_ms: int,
    fill_covered_end_ms: int,
    fill_history_scope: str,
    fill_coverage_proven: bool,
    candle_covered_start_ms: int,
    candle_covered_end_ms: int,
) -> dict[str, Any]:
    metadata = {
        "exchange": str(self.exchange),
        "market_type": _hsl_replay_cache_market_type(self),
        "user": str(self.user),
        "config_digest": self._hsl_replay_cache_config_digest(pside),
        "signal_mode": self._equity_hard_stop_signal_mode(),
        "pside": str(pside),
        "symbol": str(symbol),
        "fill_covered_start_ms": int(fill_covered_start_ms),
        "fill_covered_end_ms": int(fill_covered_end_ms),
        "fill_history_scope": str(fill_history_scope),
        # Passed through raw so the normalizer rejects non-bool proof values
        # instead of silently blessing truthy garbage as a proven manifest.
        "fill_coverage_proven": fill_coverage_proven,
        "candle_covered_start_ms": int(candle_covered_start_ms),
        "candle_covered_end_ms": int(candle_covered_end_ms),
    }
    return _normalize_hsl_replay_cache_metadata(metadata)


def _hsl_replay_cache_dir(self, pside: str, symbol: str, config_digest: str | None = None) -> str:
    digest = str(config_digest or self._hsl_replay_cache_config_digest(pside))
    if len(digest) < 16:
        raise ValueError("HSL replay cache config digest must be at least 16 characters")
    exchange = _hsl_replay_cache_safe_fragment(getattr(self, "exchange", "unknown"))
    user = _hsl_replay_cache_safe_fragment(getattr(self, "user", "unknown"))
    safe_symbol = _hsl_replay_cache_safe_fragment(symbol)
    return make_get_filepath(
        "caches/equity_hard_stop/"
        f"{exchange}/replay_matrix/{user}/{pside}/{safe_symbol}/{digest[:16]}/"
    )


def _hsl_replay_cache_account_config_digest(self) -> str:
    """Digest over the inputs that change how the account pnl series is built."""
    payload = {
        "schema_version": _HSL_REPLAY_CACHE_SCHEMA_VERSION,
        "series_kind": "account_pnl",
        "pnls_max_lookback_days": parse_pnls_max_lookback_days(
            self.live_value("pnls_max_lookback_days"),
            field_name="live.pnls_max_lookback_days",
        ).display_value,
    }
    return _hsl_replay_cache_json_digest(payload)


def _hsl_replay_cache_account_series_dir(self, config_digest: str | None = None) -> str:
    digest = str(config_digest or self._hsl_replay_cache_account_config_digest())
    if len(digest) < 16:
        raise ValueError("HSL replay cache account digest must be at least 16 characters")
    exchange = _hsl_replay_cache_safe_fragment(getattr(self, "exchange", "unknown"))
    user = _hsl_replay_cache_safe_fragment(getattr(self, "user", "unknown"))
    return make_get_filepath(
        "caches/equity_hard_stop/"
        f"{exchange}/replay_matrix/{user}/{_HSL_REPLAY_CACHE_ACCOUNT_PSIDE}/"
        f"{_HSL_REPLAY_CACHE_ACCOUNT_SYMBOL}/{digest[:16]}/"
    )


def _hsl_replay_cache_account_expected_metadata(
    self,
    *,
    fill_covered_start_ms: int,
    fill_covered_end_ms: int,
    fill_history_scope: str,
    fill_coverage_proven: bool,
    candle_covered_start_ms: int,
    candle_covered_end_ms: int,
) -> dict[str, Any]:
    metadata = {
        "exchange": str(self.exchange),
        "market_type": _hsl_replay_cache_market_type(self),
        "user": str(self.user),
        "config_digest": self._hsl_replay_cache_account_config_digest(),
        "signal_mode": self._equity_hard_stop_signal_mode(),
        "pside": _HSL_REPLAY_CACHE_ACCOUNT_PSIDE,
        "symbol": _HSL_REPLAY_CACHE_ACCOUNT_SYMBOL,
        "fill_covered_start_ms": int(fill_covered_start_ms),
        "fill_covered_end_ms": int(fill_covered_end_ms),
        "fill_history_scope": str(fill_history_scope),
        # Raw pass-through: the normalizer rejects non-bool proof values.
        "fill_coverage_proven": fill_coverage_proven,
        "candle_covered_start_ms": int(candle_covered_start_ms),
        "candle_covered_end_ms": int(candle_covered_end_ms),
    }
    return _normalize_hsl_replay_cache_metadata(metadata)


def _hsl_replay_cache_normalize_panic_events(
    events: Any, *, start_ts: int, end_ts: int
) -> list[dict[str, Any]]:
    """Normalize/validate panic flatten markers persisted with the account series."""
    if not isinstance(events, list):
        raise TypeError(
            f"HSL cache panic_flatten_events must be a list, got {type(events).__name__}"
        )
    out: list[dict[str, Any]] = []
    prev_ts: int | None = None
    for event in events:
        if not isinstance(event, dict):
            raise TypeError(
                "HSL cache panic_flatten_events entries must be dicts, "
                f"got {type(event).__name__}"
            )
        missing = [
            key
            for key in ("timestamp", "minute_timestamp", "pside", "symbol")
            if event.get(key) is None
        ]
        if missing:
            raise ValueError(
                f"HSL cache panic_flatten_events entry missing fields: {missing}"
            )
        ts = int(event["timestamp"])
        minute_ts = int(event["minute_timestamp"])
        pside = str(event["pside"])
        symbol = str(event["symbol"])
        if pside not in ("long", "short"):
            raise ValueError(
                f"HSL cache panic_flatten_events pside must be long or short, got {pside!r}"
            )
        if not symbol:
            raise ValueError("HSL cache panic_flatten_events symbol must be non-empty")
        if minute_ts % _HSL_REPLAY_MATRIX_INTERVAL_MS != 0:
            raise ValueError(
                f"HSL cache panic_flatten_events minute_timestamp {minute_ts} "
                "is not minute-aligned"
            )
        if minute_ts < int(start_ts) or minute_ts > int(end_ts):
            raise ValueError(
                f"HSL cache panic_flatten_events minute_timestamp {minute_ts} lies "
                f"outside the series span [{start_ts}, {end_ts}]"
            )
        if prev_ts is not None and ts < prev_ts:
            raise ValueError(
                "HSL cache panic_flatten_events must be ascending by timestamp"
            )
        prev_ts = ts
        out.append(
            {
                "timestamp": ts,
                "minute_timestamp": minute_ts,
                "pside": pside,
                "symbol": symbol,
            }
        )
    return out


def _write_hsl_replay_matrix_cache(
    cache_dir: str | os.PathLike[str],
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    series_kind: str = "pair_matrix",
    panic_flatten_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    import numpy as np

    fields = _hsl_replay_cache_series_fields(series_kind)
    if series_kind != "account_pnl" and panic_flatten_events is not None:
        raise ValueError(
            "HSL cache panic_flatten_events are account-scoped facts; "
            f"they cannot be persisted with series_kind={series_kind!r}"
        )
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    if series_kind == "account_pnl":
        arrays = _hsl_replay_account_series_arrays(rows)
    else:
        arrays = _hsl_replay_matrix_arrays(rows)
    metadata_norm = _normalize_hsl_replay_cache_metadata(metadata)
    manifest = {
        "schema_version": _HSL_REPLAY_CACHE_SCHEMA_VERSION,
        "series_kind": str(series_kind),
        "matrix_file": _HSL_REPLAY_CACHE_MATRIX_FILENAME,
        "row_count": int(len(rows)),
        "fields": list(fields),
        "interval_ms": _HSL_REPLAY_MATRIX_INTERVAL_MS,
        "start_ts_ms": int(arrays["ts"][0]) if len(rows) else None,
        "end_ts_ms": int(arrays["ts"][-1]) if len(rows) else None,
        "metadata": metadata_norm,
        "arrays": _hsl_replay_cache_array_manifest(arrays),
    }
    if series_kind == "account_pnl":
        if not len(rows):
            raise ValueError(
                "HSL cache account series must be non-empty to scope panic markers"
            )
        manifest["panic_flatten_events"] = _hsl_replay_cache_normalize_panic_events(
            panic_flatten_events if panic_flatten_events is not None else [],
            start_ts=int(arrays["ts"][0]),
            end_ts=int(arrays["ts"][-1]),
        )
    matrix_path = cache_path / _HSL_REPLAY_CACHE_MATRIX_FILENAME
    manifest_path = cache_path / _HSL_REPLAY_CACHE_MANIFEST_FILENAME
    matrix_tmp = matrix_path.with_suffix(matrix_path.suffix + f".{os.getpid()}.tmp")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + f".{os.getpid()}.tmp")
    np.savez(matrix_tmp, **arrays)
    # numpy appends .npz when the provided filename does not already end with it.
    actual_matrix_tmp = matrix_tmp if matrix_tmp.exists() else Path(str(matrix_tmp) + ".npz")
    os.replace(actual_matrix_tmp, matrix_path)
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    return manifest


def _hsl_replay_cache_validation_reasons(
    cache_dir: str | os.PathLike[str],
    *,
    expected_metadata: dict[str, Any] | None = None,
) -> list[str]:
    import numpy as np

    cache_path = Path(cache_dir)
    manifest_path = cache_path / _HSL_REPLAY_CACHE_MANIFEST_FILENAME
    matrix_path = cache_path / _HSL_REPLAY_CACHE_MATRIX_FILENAME
    if not manifest_path.exists():
        return ["manifest_missing"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return ["manifest_unreadable"]
    reasons: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest_not_object"]
    if int(manifest.get("schema_version", 0) or 0) != _HSL_REPLAY_CACHE_SCHEMA_VERSION:
        reasons.append("schema_version_mismatch")
    series_kind = str(manifest.get("series_kind", "") or "")
    if series_kind not in _HSL_REPLAY_CACHE_SERIES_KINDS:
        reasons.append("series_kind_invalid")
        # Field-set checks below need a concrete kind; a wrong-kind manifest is
        # already invalid, so validate the arrays against the pair layout.
        series_kind = "pair_matrix"
    series_fields = _hsl_replay_cache_series_fields(series_kind)
    if list(manifest.get("fields") or []) != list(series_fields):
        # A kind/field-list disagreement means the manifest was tampered with
        # or written by an incompatible writer.
        reasons.append("fields_mismatch")
    if series_kind == "account_pnl":
        if "panic_flatten_events" not in manifest:
            reasons.append("panic_events_missing")
        else:
            try:
                _hsl_replay_cache_normalize_panic_events(
                    manifest.get("panic_flatten_events"),
                    start_ts=int(manifest.get("start_ts_ms") or 0),
                    end_ts=int(manifest.get("end_ts_ms") or 0),
                )
            except (TypeError, ValueError):
                reasons.append("panic_events_invalid")
    elif manifest.get("panic_flatten_events") is not None:
        reasons.append("panic_events_wrong_kind")
    if str(manifest.get("matrix_file", "")) != _HSL_REPLAY_CACHE_MATRIX_FILENAME:
        reasons.append("matrix_filename_mismatch")
    if int(manifest.get("interval_ms", 0) or 0) != _HSL_REPLAY_MATRIX_INTERVAL_MS:
        reasons.append("interval_mismatch")
    if not matrix_path.exists():
        reasons.append("matrix_missing")
        return reasons
    expected_norm = None
    if expected_metadata is not None:
        try:
            expected_norm = _normalize_hsl_replay_cache_metadata(expected_metadata)
        except Exception:
            reasons.append("expected_metadata_invalid")
    metadata = manifest.get("metadata")
    metadata_norm = None
    if not isinstance(metadata, dict):
        reasons.append("metadata_missing")
    else:
        try:
            metadata_norm = _normalize_hsl_replay_cache_metadata(metadata)
        except ValueError:
            missing_fields = [
                key for key in _HSL_REPLAY_CACHE_REQUIRED_METADATA if metadata.get(key) is None
            ]
            if missing_fields:
                reasons.extend(f"metadata_missing_required:{key}" for key in missing_fields)
            else:
                reasons.append("metadata_invalid")
        except Exception:
            reasons.append("metadata_invalid")
    if metadata_norm is not None and expected_norm is not None:
        for key, value in expected_norm.items():
            if metadata_norm.get(key) != value:
                reasons.append(f"metadata_mismatch:{key}")
    try:
        loaded_npz = np.load(matrix_path, allow_pickle=False)
    except Exception:
        reasons.append("matrix_unreadable")
        return reasons
    with loaded_npz as loaded:
        arrays_manifest = manifest.get("arrays")
        if not isinstance(arrays_manifest, dict):
            reasons.append("arrays_manifest_missing")
            arrays_manifest = {}
        loaded_arrays: dict[str, Any] = {}
        for field in series_fields:
            if field not in loaded:
                reasons.append(f"array_missing:{field}")
                continue
            arr = loaded[field]
            loaded_arrays[field] = arr
            entry = arrays_manifest.get(field)
            if not isinstance(entry, dict):
                reasons.append(f"array_manifest_missing:{field}")
                continue
            if str(arr.dtype) != str(entry.get("dtype")):
                reasons.append(f"array_dtype_mismatch:{field}")
            if [int(x) for x in arr.shape] != list(entry.get("shape", [])):
                reasons.append(f"array_shape_mismatch:{field}")
            if _hsl_hash_array(arr) != str(entry.get("sha256")):
                reasons.append(f"array_hash_mismatch:{field}")
        if set(loaded_arrays) == set(series_fields):
            value_reasons = _hsl_replay_cache_array_value_reasons(
                loaded_arrays, series_kind=series_kind
            )
            reasons.extend(value_reasons)
            ts_array = np.asarray(loaded_arrays["ts"])
            ts_valid_for_time_checks = (
                ts_array.ndim == 1
                and _hsl_replay_cache_array_dtype_is_valid("ts", ts_array.dtype)
                and bool(np.all(np.isfinite(ts_array)))
            )
            row_count = int(len(ts_array)) if ts_valid_for_time_checks else None
            if row_count is not None and row_count != int(manifest.get("row_count", -1)):
                reasons.append("row_count_mismatch")
            if row_count:
                diffs = np.diff(ts_array)
                if diffs.size and not bool(np.all(diffs == _HSL_REPLAY_MATRIX_INTERVAL_MS)):
                    reasons.append("timestamp_not_contiguous")
                try:
                    if int(ts_array[0]) != manifest.get("start_ts_ms"):
                        reasons.append("start_ts_mismatch")
                    if int(ts_array[-1]) != manifest.get("end_ts_ms"):
                        reasons.append("end_ts_mismatch")
                except (TypeError, ValueError, OverflowError):
                    reasons.append("timestamp_invalid")
    return reasons


def _load_hsl_replay_matrix_cache(
    cache_dir: str | os.PathLike[str],
    *,
    expected_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    reasons = _hsl_replay_cache_validation_reasons(
        cache_dir,
        expected_metadata=expected_metadata,
    )
    if reasons:
        raise ValueError("HSL replay cache validation failed: " + ", ".join(reasons))
    cache_path = Path(cache_dir)
    manifest = json.loads(
        (cache_path / _HSL_REPLAY_CACHE_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    fields = _hsl_replay_cache_series_fields(str(manifest.get("series_kind", "")))
    with np.load(cache_path / _HSL_REPLAY_CACHE_MATRIX_FILENAME, allow_pickle=False) as loaded:
        arrays = {field: loaded[field].copy() for field in fields}
    return manifest, arrays


def _hsl_replay_cache_status_data(
    *,
    cache_status: str,
    elapsed_s: float,
    reasons: list[str] | None = None,
    manifest: dict[str, Any] | None = None,
    expected_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "cache_status": str(cache_status),
        "schema_version": _HSL_REPLAY_CACHE_SCHEMA_VERSION,
        "matrix_file": _HSL_REPLAY_CACHE_MATRIX_FILENAME,
        "manifest_file": _HSL_REPLAY_CACHE_MANIFEST_FILENAME,
        "elapsed_s": round(max(0.0, float(elapsed_s)), 3),
    }
    metadata = manifest.get("metadata") if isinstance(manifest, dict) else expected_metadata
    if isinstance(metadata, dict):
        for key in ("signal_mode", "pside", "symbol"):
            value = metadata.get(key)
            if value is not None:
                data[key] = str(value)
    if isinstance(manifest, dict):
        for key in ("row_count", "start_ts_ms", "end_ts_ms", "interval_ms"):
            value = manifest.get(key)
            if value is not None:
                data[key] = int(value)
    if reasons:
        bounded_reasons = [str(reason) for reason in reasons[:8]]
        data["reasons"] = bounded_reasons
        data["reason_count"] = int(len(reasons))
        data["reasons_truncated"] = bool(len(reasons) > len(bounded_reasons))
    else:
        data["reason_count"] = 0
        data["reasons_truncated"] = False
    return data


def _try_load_hsl_replay_matrix_cache(
    self,
    cache_dir: str | os.PathLike[str],
    *,
    expected_metadata: dict[str, Any] | None = None,
    pside: str | None = None,
    symbol: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    started_s = time.monotonic()
    try:
        reasons = _hsl_replay_cache_validation_reasons(
            cache_dir,
            expected_metadata=expected_metadata,
        )
    except Exception as exc:
        elapsed_s = time.monotonic() - started_s
        _emit_hsl_replay_event(
            self,
            EventTypes.HSL_REPLAY_CACHE,
            _hsl_replay_cache_status_data(
                cache_status="rejected",
                elapsed_s=elapsed_s,
                reasons=[f"validation_exception:{type(exc).__name__}"],
                expected_metadata=expected_metadata,
            ),
            pside=pside,
            symbol=symbol,
            level="debug",
            status="skipped",
            reason_code=ReasonCodes.HSL_REPLAY_CACHE_REJECTED,
        )
        return None
    if reasons:
        elapsed_s = time.monotonic() - started_s
        cache_status = "miss" if reasons == ["manifest_missing"] else "rejected"
        reason_code = (
            ReasonCodes.HSL_REPLAY_CACHE_MISS
            if cache_status == "miss"
            else ReasonCodes.HSL_REPLAY_CACHE_REJECTED
        )
        _emit_hsl_replay_event(
            self,
            EventTypes.HSL_REPLAY_CACHE,
            _hsl_replay_cache_status_data(
                cache_status=cache_status,
                elapsed_s=elapsed_s,
                reasons=reasons,
                expected_metadata=expected_metadata,
            ),
            pside=pside,
            symbol=symbol,
            level="debug",
            status="skipped",
            reason_code=reason_code,
        )
        return None
    try:
        import numpy as np

        cache_path = Path(cache_dir)
        manifest = json.loads(
            (cache_path / _HSL_REPLAY_CACHE_MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        fields = _hsl_replay_cache_series_fields(str(manifest.get("series_kind", "")))
        with np.load(cache_path / _HSL_REPLAY_CACHE_MATRIX_FILENAME, allow_pickle=False) as loaded:
            arrays = {field: loaded[field].copy() for field in fields}
    except Exception as exc:
        elapsed_s = time.monotonic() - started_s
        _emit_hsl_replay_event(
            self,
            EventTypes.HSL_REPLAY_CACHE,
            _hsl_replay_cache_status_data(
                cache_status="rejected",
                elapsed_s=elapsed_s,
                reasons=[f"load_exception:{type(exc).__name__}"],
                expected_metadata=expected_metadata,
            ),
            pside=pside,
            symbol=symbol,
            level="debug",
            status="skipped",
            reason_code=ReasonCodes.HSL_REPLAY_CACHE_REJECTED,
        )
        return None
    elapsed_s = time.monotonic() - started_s
    _emit_hsl_replay_event(
        self,
        EventTypes.HSL_REPLAY_CACHE,
        _hsl_replay_cache_status_data(
            cache_status="hit",
            elapsed_s=elapsed_s,
            manifest=manifest,
            expected_metadata=expected_metadata,
        ),
        pside=pside,
        symbol=symbol,
        level="debug",
        status="succeeded",
        reason_code=ReasonCodes.HSL_REPLAY_CACHE_HIT,
    )
    return manifest, arrays


def _equity_hard_stop_persist_replay_matrices(self, history: dict[str, Any]) -> int:
    """Persist non-authoritative raw replay matrices after a successful coin replay.

    Write-only cache population: nothing here is read back for trading decisions.
    Per-pair failures are logged and emitted as events but never raised, because
    the cache is a performance aid, not trading state.
    """
    matrices = history.get("hsl_replay_matrices")
    coverage = history.get("hsl_replay_matrix_coverage")
    if not isinstance(matrices, dict) or not matrices:
        return 0
    if not isinstance(coverage, dict):
        logging.warning(
            "[risk] HSL replay cache write skipped: history missing matrix coverage metadata"
        )
        return 0
    written = 0
    for pside in sorted(matrices):
        by_symbol = matrices.get(pside)
        if not isinstance(by_symbol, dict):
            continue
        for symbol in sorted(by_symbol):
            rows = by_symbol[symbol]
            started_s = time.monotonic()
            try:
                config_digest = self._hsl_replay_cache_config_digest(pside)
                cache_dir = self._hsl_replay_cache_dir(
                    pside, symbol, config_digest=config_digest
                )
                metadata = self._hsl_replay_cache_expected_metadata(
                    pside,
                    symbol,
                    fill_covered_start_ms=int(coverage["fill_covered_start_ms"]),
                    fill_covered_end_ms=int(coverage["fill_covered_end_ms"]),
                    fill_history_scope=str(coverage["fill_history_scope"]),
                    fill_coverage_proven=coverage["fill_coverage_proven"],
                    candle_covered_start_ms=int(coverage["candle_covered_start_ms"]),
                    candle_covered_end_ms=int(coverage["candle_covered_end_ms"]),
                )
                manifest = _write_hsl_replay_matrix_cache(cache_dir, rows, metadata)
            except Exception as exc:
                logging.warning(
                    "[risk] HSL[%s:%s] replay cache write failed | rows=%d error=%s: %s",
                    pside,
                    symbol,
                    len(rows) if isinstance(rows, list) else -1,
                    type(exc).__name__,
                    exc,
                )
                _emit_hsl_replay_event(
                    self,
                    EventTypes.HSL_REPLAY_CACHE,
                    _hsl_replay_cache_status_data(
                        cache_status="write_failed",
                        elapsed_s=time.monotonic() - started_s,
                        reasons=[f"write_exception:{type(exc).__name__}"],
                    ),
                    pside=pside,
                    symbol=symbol,
                    level="warning",
                    status="failed",
                    reason_code=ReasonCodes.HSL_REPLAY_CACHE_WRITE_FAILED,
                )
                continue
            written += 1
            _emit_hsl_replay_event(
                self,
                EventTypes.HSL_REPLAY_CACHE,
                _hsl_replay_cache_status_data(
                    cache_status="written",
                    elapsed_s=time.monotonic() - started_s,
                    manifest=manifest,
                ),
                pside=pside,
                symbol=symbol,
                level="debug",
                status="succeeded",
                reason_code=ReasonCodes.HSL_REPLAY_CACHE_WRITTEN,
            )
    account_rows = history.get("hsl_replay_account_series")
    if written and isinstance(account_rows, list) and account_rows:
        # Pair matrices are only reusable together with the account-level pnl
        # series (per-minute slot budgets need the account balance), so persist
        # it alongside them under the same guarded, write-only contract.
        started_s = time.monotonic()
        try:
            cache_dir = self._hsl_replay_cache_account_series_dir()
            metadata = self._hsl_replay_cache_account_expected_metadata(
                fill_covered_start_ms=int(coverage["fill_covered_start_ms"]),
                fill_covered_end_ms=int(coverage["fill_covered_end_ms"]),
                fill_history_scope=str(coverage["fill_history_scope"]),
                fill_coverage_proven=coverage["fill_coverage_proven"],
                candle_covered_start_ms=int(coverage["candle_covered_start_ms"]),
                candle_covered_end_ms=int(coverage["candle_covered_end_ms"]),
            )
            panic_events = history.get("panic_flatten_events")
            manifest = _write_hsl_replay_matrix_cache(
                cache_dir,
                account_rows,
                metadata,
                series_kind="account_pnl",
                panic_flatten_events=(
                    panic_events if isinstance(panic_events, list) else []
                ),
            )
        except Exception as exc:
            logging.warning(
                "[risk] HSL account series replay cache write failed | rows=%d error=%s: %s",
                len(account_rows),
                type(exc).__name__,
                exc,
            )
            _emit_hsl_replay_event(
                self,
                EventTypes.HSL_REPLAY_CACHE,
                _hsl_replay_cache_status_data(
                    cache_status="write_failed",
                    elapsed_s=time.monotonic() - started_s,
                    reasons=[f"write_exception:{type(exc).__name__}"],
                ),
                pside=_HSL_REPLAY_CACHE_ACCOUNT_PSIDE,
                symbol=_HSL_REPLAY_CACHE_ACCOUNT_SYMBOL,
                level="warning",
                status="failed",
                reason_code=ReasonCodes.HSL_REPLAY_CACHE_WRITE_FAILED,
            )
        else:
            written += 1
            _emit_hsl_replay_event(
                self,
                EventTypes.HSL_REPLAY_CACHE,
                _hsl_replay_cache_status_data(
                    cache_status="written",
                    elapsed_s=time.monotonic() - started_s,
                    manifest=manifest,
                ),
                pside=_HSL_REPLAY_CACHE_ACCOUNT_PSIDE,
                symbol=_HSL_REPLAY_CACHE_ACCOUNT_SYMBOL,
                level="debug",
                status="succeeded",
                reason_code=ReasonCodes.HSL_REPLAY_CACHE_WRITTEN,
            )
    return written


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
        "last_cooldown_intervention_log_ms": 0,
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


def _equity_hard_stop_coin_active_pside(self, pside: str) -> bool:
    if not self._equity_hard_stop_enabled(pside):
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
                "balance overrides, HSL mode changes, and HSL budget/threshold "
                "changes can reinterpret reconstructed history.",
                pside,
                _HSL_RISKS_DOC,
            )
            logging.info(
                "[risk] HSL[%s] enabled | red_threshold=%.6f ema_span_minutes=%.3f "
                "cooldown_minutes_after_red=%.3f "
                "no_restart_drawdown_threshold=%.6f signal_mode=%s "
                "yellow_ratio=%.3f orange_ratio=%.3f "
                "orange_mode=%s panic_close=%s restart_after_red=%s",
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
                restart_after_red_policy,
            )
    return out


def _equity_hard_stop_no_restart_latched(cfg: dict[str, Any], drawdown_raw: float) -> bool:
    policy = normalize_hsl_restart_after_red_policy(
        cfg.get("restart_after_red_policy", "threshold"),
        path="hsl.restart_after_red_policy",
    )
    if policy == "always":
        return False
    if policy == "never":
        return True
    return bool(float(drawdown_raw) >= float(cfg["no_restart_drawdown_threshold"]))


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
    cooldown_minutes = float(self.hsl[pside]["cooldown_minutes_after_red"])
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
        state["last_cooldown_intervention_log_ms"] = 0
        state["cooldown_unresolved_residue"] = False
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
    cfg = self.hsl[pside]
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
    # HSL is a drawdown stop, not an exposure scaler. Keep coin-mode live HSL
    # sensitivity anchored to the configured slot count; TWEL/excess allowance
    # must not make the RED threshold tolerate a larger percentage drawdown.
    slot_budget = float(balance) / n_positions
    if not math.isfinite(slot_budget) or slot_budget <= 0.0:
        raise ValueError(
            f"coin HSL slot_budget must be finite and > 0 for {symbol} {pside}, "
            f"got balance={balance} n_positions={n_positions}"
        )
    drawdown_usd = max(0.0, float(peak_realized) - (float(last_realized) + float(current_upnl)))
    drawdown_ratio = drawdown_usd / max(slot_budget, 1e-12)
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
        "tier": str(step["tier"]),
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


def _equity_hard_stop_coin_replay_events(
    fill_events: list[Any], pside: str, symbol: str
) -> tuple[list[tuple[int, str, float]], bool]:
    replay_events: list[tuple[int, str, float]] = []
    ambiguous = False
    replay_size = 0.0
    for event in fill_events:
        if _equity_hard_stop_fill_pside(event) != pside:
            continue
        if _equity_hard_stop_fill_symbol(event) != symbol:
            continue
        action = str(_equity_hard_stop_event_value(event, "action", "") or "").lower()
        qty = _equity_hard_stop_fill_replay_qty(event)
        if action not in {"increase", "decrease"} or qty is None or qty <= 0.0:
            ambiguous = True
            continue
        replay_events.append(
            (_equity_hard_stop_fill_timestamp_ms(event), action, float(qty))
        )
    replay_events.sort(key=lambda item: item[0])
    for _event_ts, action, qty in replay_events:
        if action == "increase":
            replay_size += qty
        else:
            if qty > replay_size + 1e-12:
                ambiguous = True
            replay_size = max(0.0, replay_size - qty)
    return replay_events, ambiguous


def _equity_hard_stop_coin_replay_size_at(
    replay_events: list[tuple[int, str, float]], row_ts_ms: int
) -> float:
    boundary_ts_ms = int(row_ts_ms) + 60_000
    size = 0.0
    for event_ts, action, qty in replay_events:
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
    *,
    realized_pnl_total: Optional[float],
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
    cfg = self.hsl[pside]
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
        float(self.hsl[pside]["tier_ratios"]["yellow"]),
        float(self.hsl[pside]["tier_ratios"]["orange"]),
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
            exc,
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
    cfg = self.hsl[pside]
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


def _equity_hard_stop_record_no_restart_stop(
    self, pside: str, stop_event: dict
) -> tuple[float, float]:
    state = self._hsl_state(pside)
    equity = float(stop_event["equity"])
    peak_strategy_equity = float(stop_event["peak_strategy_equity"])
    if not math.isfinite(equity) or equity <= 0.0:
        raise RuntimeError(f"invalid HSL[{pside}] stop equity for no-restart latch: {equity}")
    if not math.isfinite(peak_strategy_equity) or peak_strategy_equity <= 0.0:
        raise RuntimeError(
            f"invalid HSL[{pside}] stop peak_strategy_equity for no-restart latch: {peak_strategy_equity}"
        )
    no_restart_peak_strategy_equity = max(
        float(state.get("no_restart_peak_strategy_equity", 0.0) or 0.0),
        peak_strategy_equity,
        equity,
    )
    if (
        not math.isfinite(no_restart_peak_strategy_equity)
        or no_restart_peak_strategy_equity <= 0.0
    ):
        raise RuntimeError(
            f"invalid HSL[{pside}] no_restart_peak_strategy_equity: {no_restart_peak_strategy_equity}"
        )
    state["no_restart_peak_strategy_equity"] = no_restart_peak_strategy_equity
    no_restart_drawdown_raw = max(
        0.0, 1.0 - equity / max(no_restart_peak_strategy_equity, 1e-12)
    )
    return float(no_restart_peak_strategy_equity), float(no_restart_drawdown_raw)


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
    if metrics is None or int(metrics.get("timestamp_ms", 0) or 0) != int(stop_event_ts_ms):
        metrics = self._equity_hard_stop_apply_coin_sample(
            pside,
            symbol,
            int(stop_event_ts_ms),
            float(self.get_raw_balance()),
            float(await self._calc_upnl_sum_strict(pside, symbol)),
        )
    return {
        "position_side": pside,
        "symbol": symbol,
        "signal_mode": "coin",
        "stop_event_timestamp_ms": int(stop_event_ts_ms),
        "balance": float(metrics["balance"]),
        "slot_budget": float(metrics["slot_budget"]),
        "realized_pnl_total": float(self._equity_hard_stop_realized_pnl_now()),
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


async def _equity_hard_stop_refresh_cooldown_after_repanic(self, pside: str, now_ms: int) -> None:
    state = self._hsl_state(pside)
    cooldown_minutes = float(self.hsl[pside]["cooldown_minutes_after_red"])
    cooldown_ms = max(0, int(round(cooldown_minutes * 60_000.0))) if cooldown_minutes > 0.0 else 0
    previous_stop_ts = None
    if state.get("last_stop_event") is not None:
        previous_stop_ts = int(state["last_stop_event"]["stop_event_timestamp_ms"]) + 1
    stop_ts_ms = self._equity_hard_stop_latest_panic_fill_timestamp_ms(
        pside,
        since_ms=previous_stop_ts,
        fallback_ms=now_ms,
    )
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
    state["last_cooldown_intervention_log_ms"] = 0
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


async def _equity_hard_stop_refresh_coin_cooldown_after_repanic(
    self, pside: str, symbol: str, now_ms: int
) -> None:
    state = self._hsl_coin_state(pside, symbol)
    cooldown_minutes = float(self.hsl[pside]["cooldown_minutes_after_red"])
    cooldown_ms = max(0, int(round(cooldown_minutes * 60_000.0))) if cooldown_minutes > 0.0 else 0
    previous_stop_ts = None
    if state.get("last_stop_event") is not None:
        previous_stop_ts = int(state["last_stop_event"]["stop_event_timestamp_ms"]) + 1
    stop_ts_ms = self._equity_hard_stop_latest_panic_fill_timestamp_ms(
        pside,
        symbol=symbol,
        since_ms=previous_stop_ts,
        fallback_ms=now_ms,
    )
    cooldown_until_ms = stop_ts_ms + cooldown_ms if cooldown_ms > 0 else None
    stop_event = await self._equity_hard_stop_compute_coin_stop_event(pside, symbol, stop_ts_ms)
    payload = self._equity_hard_stop_build_latch_payload(
        pside,
        symbol=symbol,
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

    state["cooldown_repanic_reset_pending"] = policy == "panic"
    return False


async def _equity_hard_stop_handle_coin_position_during_cooldown(
    self, pside: str, symbol: str, now_ms: int
) -> bool:
    state = self._hsl_coin_state(pside, symbol)
    if not state["halted"] or state["no_restart_latched"]:
        return False
    cooldown_until_ms = state["cooldown_until_ms"]
    if cooldown_until_ms is None or now_ms >= cooldown_until_ms:
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
    state["last_cooldown_intervention_log_ms"] = 0
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
            cfg = self.hsl[pside]
            contract = replay_contracts[pside]
            cooldown_minutes = float(cfg["cooldown_minutes_after_red"])
            cooldown_ms = int(round(cooldown_minutes * 60_000.0)) if cooldown_minutes > 0.0 else 0
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
                panic_flatten_marker = panic_flatten_events_by_key.get((pside, ts))
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
                    state["pending_red_since_ms"] = int(ts)
                    stop_drawdown_raw = float(current_metrics["drawdown_raw"])
                    (
                        no_restart_peak_strategy_equity,
                        no_restart_drawdown_raw,
                    ) = self._equity_hard_stop_record_no_restart_stop(
                        pside,
                        {
                            "equity": float(current_metrics["strategy_equity"]),
                            "peak_strategy_equity": float(current_metrics["peak_strategy_equity"]),
                        },
                    )
                    no_restart_latched = _equity_hard_stop_no_restart_latched(
                        cfg, no_restart_drawdown_raw
                    )
                    cooldown_until_ms = None
                    if not no_restart_latched and cooldown_ms > 0:
                        cooldown_until_ms = marker_ts + cooldown_ms
                    payload = self._equity_hard_stop_build_latch_payload(
                        pside,
                        stop_event_timestamp_ms=marker_ts,
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
                        "source=panic_fill_flatten",
                        pside,
                        marker_ts,
                        stop_drawdown_raw,
                        no_restart_drawdown_raw,
                        state["no_restart_latched"],
                        cooldown_until_ms if cooldown_until_ms is not None else "none",
                        latch_path,
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
        history_fetch_started_s = time.monotonic()
        history = await self.get_balance_equity_history(
            current_balance=self.get_raw_balance(),
            hsl_replay_signal_mode="coin",
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

        if "timeline" not in history:
            raise ValueError("get_balance_equity_history() missing required key: timeline")
        timeline = history["timeline"]
        if not isinstance(timeline, list):
            raise TypeError(
                f"get_balance_equity_history()['timeline'] must be a list, got {type(timeline).__name__}"
            )

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
                if (pside, symbol) in current_position_pairs:
                    required_replay_pairs.add((pside, symbol))
                    remember_required_replay_start(pside, symbol, ts)
        for row in timeline:
            if not isinstance(row, dict):
                continue
            for key in ("realized_pnl_by_coin_pside", "unrealized_pnl_by_coin_pside"):
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
                    if not self._equity_hard_stop_symbol_supported_for_coin_replay(symbol):
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
        for row in timeline:
            if not isinstance(row, dict):
                continue
            if "timestamp" not in row or "balance" not in row:
                continue
            ts = int(row["timestamp"])
            if ts > now_ms:
                break
            timeline_rows.append(row)

        balance = float(self.get_raw_balance())
        rows = 0
        replay_started_s = time.monotonic()
        pre_replay_elapsed_s = max(0.0, replay_started_s - history_loaded_s)
        last_progress_log_s = replay_started_s
        replay_symbols = set(symbols)
        active_pairs = [
            (pside, symbol)
            for pside in self._hsl_psides()
            if self._equity_hard_stop_coin_active_pside(pside)
            for symbol in sorted(replay_symbols)
        ]
        active_pair_set = set(active_pairs)
        active_held_pairs = active_pair_set.intersection(current_position_pairs)
        active_panic_pairs = active_pair_set.intersection(panic_replay_pairs)
        active_required_pairs = active_pair_set.intersection(required_replay_pairs)
        pair_rows_applied: dict[tuple[str, str], int] = {}
        logging.info(
            "[risk] HSL coin history reconstruction loaded | symbols=%d pairs=%d rows=%d fills=%d panic_events=%d",
            len(symbols),
            len(active_pairs),
            len(timeline_rows),
            len(fill_events),
            len(panic_flatten_events),
        )
        _emit_hsl_replay_event(
            self,
            "hsl.replay.progress",
            {
                "signal_mode": "coin",
                "stage": "loaded",
                "symbols": len(symbols),
                "pairs": len(active_pairs),
                "held_pairs": len(active_held_pairs),
                "cooldown_pairs": len(active_panic_pairs),
                "required_pairs": len(active_required_pairs),
                "timeline_rows": len(timeline_rows),
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
            *,
            force: bool = False,
        ) -> None:
            nonlocal last_progress_log_s
            now_s = time.monotonic()
            if not force and now_s - last_progress_log_s < 15.0:
                return
            last_progress_log_s = now_s
            elapsed_s = max(0.0, now_s - replay_started_s)
            rows_per_second = float(rows) / elapsed_s if elapsed_s > 0.0 else None
            logging.info(
                "[risk] HSL coin history reconstruction progress | pair=%d/%d pside=%s symbol=%s applied_rows=%d total_rows=%d elapsed=%.1fs",
                pair_idx,
                len(active_pairs),
                pside,
                symbol,
                applied_rows,
                rows,
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
                    "timeline_rows": len(timeline_rows),
                    "applied_rows": int(applied_rows),
                    "total_applied_rows": int(rows),
                    "rows_per_second": round(rows_per_second, 3)
                    if rows_per_second is not None
                    else None,
                    "is_held_pair": (pside, symbol) in active_held_pairs,
                    "is_cooldown_pair": (pside, symbol) in active_panic_pairs,
                    "elapsed_s": round(elapsed_s, 3),
                },
                pside=pside,
                symbol=symbol,
                status="started",
                reason_code="pair_replay_progress",
            )

        pair_idx = 0
        for pside in self._hsl_psides():
            check_shutdown("hsl_coin_history_replay_pside")
            if not self._equity_hard_stop_coin_active_pside(pside):
                continue
            cfg = self.hsl[pside]
            cooldown_minutes = float(cfg["cooldown_minutes_after_red"])
            cooldown_ms = int(round(cooldown_minutes * 60_000.0)) if cooldown_minutes > 0.0 else 0
            for symbol in sorted(replay_symbols):
                check_shutdown("hsl_coin_history_replay_pair")
                pair_idx += 1
                state = self._hsl_coin_state(pside, symbol)
                contract = self._equity_hard_stop_infer_coin_replay_contract(
                    pside, symbol, fill_events, now_ms
                )
                replay_start_boundary_ts = None
                if contract["intervention_entry_ts"] is not None and contract["policy"] == "normal":
                    replay_start_boundary_ts = int(contract["intervention_entry_ts"])
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
                replay_events, replay_ambiguous = _equity_hard_stop_coin_replay_events(
                    fill_events, pside, symbol
                )

                def reset_rolling_window() -> None:
                    nonlocal window_base_realized
                    window_points.clear()
                    window_max_points.clear()
                    window_base_realized = 0.0

                replay_event_idx = 0
                replay_size = 0.0
                replay_was_nonflat = False
                replay_flattened_at_ms: Optional[int] = None
                ignored_panic_marker_timestamps: set[int] = set()

                def replay_size_at(row_ts_ms: int) -> float:
                    nonlocal replay_event_idx, replay_size, replay_flattened_at_ms
                    boundary_ts_ms = int(row_ts_ms) + 60_000
                    while replay_event_idx < len(replay_events):
                        event_ts, action, qty = replay_events[replay_event_idx]
                        if int(event_ts) >= boundary_ts_ms:
                            break
                        if action == "increase":
                            replay_size += qty
                        else:
                            replay_size = max(0.0, replay_size - qty)
                            if replay_size <= 1e-12:
                                replay_flattened_at_ms = int(event_ts)
                        replay_event_idx += 1
                    return float(replay_size)

                for row_idx, row in enumerate(timeline_rows, start=1):
                    if row_idx % 1000 == 0:
                        await asyncio.sleep(0)
                        check_shutdown("hsl_coin_history_replay_rows")
                        log_replay_progress(pair_idx, pside, symbol, applied_rows)
                    ts = int(row["timestamp"])
                    if replay_start_boundary_ts is not None and ts < replay_start_boundary_ts:
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
                    has_realized = _equity_hard_stop_history_coin_has_value(
                        row, "realized_pnl_by_coin_pside", symbol, pside
                    )
                    has_unrealized = _equity_hard_stop_history_coin_has_value(
                        row, "unrealized_pnl_by_coin_pside", symbol, pside
                    )
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
                    replay_position_size = replay_size_at(ts)
                    abs_realized = _equity_hard_stop_history_coin_value(
                        row,
                        "realized_pnl_by_coin_pside",
                        symbol,
                        pside,
                        require_key=require_coin_timeline_value,
                        require_value=require_coin_timeline_value,
                    )
                    seen_coin_timeline_fields = True
                    last_realized = abs_realized - reset_baseline_realized
                    start_ms = None if lookback_ms is None else ts - int(lookback_ms)
                    reset_ts = state["pnl_reset_timestamp_ms"]
                    if reset_ts is not None:
                        start_ms = int(reset_ts) if start_ms is None else max(start_ms, int(reset_ts))
                    window_points.append((ts, last_realized))
                    while window_max_points and window_max_points[-1][1] <= last_realized:
                        window_max_points.pop()
                    window_max_points.append((ts, last_realized))
                    if start_ms is not None:
                        while window_points and window_points[0][0] < start_ms:
                            _old_ts, old_value = window_points.popleft()
                            window_base_realized = float(old_value)
                        while window_max_points and window_max_points[0][0] < start_ms:
                            window_max_points.popleft()
                    window_last_realized = (
                        float(last_realized) - window_base_realized if window_points else 0.0
                    )
                    peak_realized = max(
                        0.0,
                        (
                            float(window_max_points[0][1]) - window_base_realized
                            if window_max_points
                            else 0.0
                        ),
                    )
                    if not has_unrealized and require_coin_timeline_value:
                        if replay_ambiguous or replay_position_size > 1e-12:
                            if not require_coin_timeline_fields:
                                continue
                            current_upnl = _equity_hard_stop_history_coin_value(
                                row,
                                "unrealized_pnl_by_coin_pside",
                                symbol,
                                pside,
                                require_key=True,
                                require_value=True,
                            )
                        else:
                            current_upnl = 0.0
                    else:
                        current_upnl = _equity_hard_stop_history_coin_value(
                            row,
                            "unrealized_pnl_by_coin_pside",
                            symbol,
                            pside,
                            require_key=require_coin_timeline_value,
                            require_value=require_coin_timeline_value,
                        )
                    self._equity_hard_stop_prime_coin_runtime_for_replay(pside, symbol, ts)
                    marker = latest_panic_by_coin_minute.get((pside, symbol, ts))
                    metrics = self._equity_hard_stop_apply_coin_metrics_sample(
                        pside,
                        symbol,
                        ts,
                        float(row["balance"]),
                        peak_realized,
                        window_last_realized,
                        current_upnl,
                        latch_red=False,
                    )
                    applied_rows += 1
                    rows += 1
                    pair_rows_applied[(pside, symbol)] = int(applied_rows)
                    replay_is_nonflat = replay_position_size > 1e-12
                    replay_flattened_this_row = (
                        not replay_ambiguous
                        and replay_was_nonflat
                        and not replay_is_nonflat
                        and replay_flattened_at_ms is not None
                        and replay_flattened_at_ms < ts + 60_000
                    )
                    if replay_is_nonflat:
                        replay_was_nonflat = True
                    if marker is None:
                        if replay_flattened_this_row:
                            # Ordinary, non-panic flattening ends the current coin episode.
                            # Cooldown/no-restart accounting remains driven by panic markers.
                            reset_ts = int(replay_flattened_at_ms) + 1
                            state["pnl_reset_timestamp_ms"] = reset_ts
                            reset_baseline_realized = abs_realized
                            self._equity_hard_stop_reset_coin_after_restart(pside, symbol)
                            state = self._hsl_coin_state(pside, symbol)
                            reset_rolling_window()
                            replay_was_nonflat = False
                            logging.info(
                                "[risk] HSL[%s:%s] replay reset current episode after flat fill | flat_ts=%s",
                                pside,
                                symbol,
                                int(replay_flattened_at_ms),
                            )
                            replay_flattened_at_ms = None
                        continue
                    stop_ts = int(marker["timestamp"])
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
                    no_restart_latched = _equity_hard_stop_no_restart_latched(
                        cfg, stop_drawdown_raw
                    )
                    cooldown_until_ms = None
                    if not no_restart_latched and cooldown_ms > 0:
                        cooldown_until_ms = stop_ts + cooldown_ms
                    state["last_stop_event"] = self._equity_hard_stop_build_latch_payload(
                        pside,
                        symbol=symbol,
                        stop_event_timestamp_ms=stop_ts,
                        balance=float(metrics["balance"]),
                        realized_pnl_total=float(row["realized_pnl"]) if "realized_pnl" in row else None,
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
                    )
                    state["pnl_reset_timestamp_ms"] = stop_ts + 1
                    state["pending_red_since_ms"] = None
                    reset_baseline_realized = abs_realized
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
                            "stop_ts=%s drawdown_raw=%.6f",
                            pside,
                            symbol,
                            stop_ts,
                            stop_drawdown_raw,
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
                            "remaining_time=%s",
                            pside,
                            symbol,
                            _equity_hard_stop_format_remaining_time(
                                (cooldown_until_ms - now_ms) / 1000.0
                            ),
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
                        realized_pnl_total=float(self._equity_hard_stop_realized_pnl_now()),
                    )
                pair_rows_applied[(pside, symbol)] = int(applied_rows)
                log_replay_progress(pair_idx, pside, symbol, applied_rows)
        self._equity_hard_stop_coin_initialized = True
        elapsed_s = max(0.0, time.monotonic() - replay_started_s)
        total_elapsed_s = max(0.0, time.monotonic() - initialization_started_s)
        rows_per_second = float(rows) / elapsed_s if elapsed_s > 0.0 else None
        skipped_pairs = sum(1 for pair in active_pairs if pair_rows_applied.get(pair, 0) == 0)
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
                "rows": int(rows),
                "applied_rows": int(rows),
                "pairs": len(active_pairs),
                "held_pairs": len(active_held_pairs),
                "cooldown_pairs": len(active_panic_pairs),
                "required_pairs": len(active_required_pairs),
                "skipped_pairs": int(skipped_pairs),
                "timeline_rows": len(timeline_rows),
                "fill_events": len(fill_events),
                "panic_events": len(panic_flatten_events),
                "rows_per_second": round(rows_per_second, 3)
                if rows_per_second is not None
                else None,
                "history_fetch_elapsed_s": round(history_fetch_elapsed_s, 3),
                "pre_replay_elapsed_s": round(pre_replay_elapsed_s, 3),
                "replay_loop_elapsed_s": round(elapsed_s, 3),
                "full_elapsed_s": round(total_elapsed_s, 3),
                "startup_blocking_elapsed_s": round(total_elapsed_s, 3),
                "elapsed_s": round(total_elapsed_s, 3),
            },
            status="succeeded",
            reason_code="coin_history_replay_completed",
        )
        try:
            self._equity_hard_stop_persist_replay_matrices(history)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Cache persistence is a performance aid; a failure here must
            # never invalidate the completed replay.
            logging.warning(
                "[risk] HSL replay cache persistence failed | error=%s: %s",
                type(exc).__name__,
                exc,
            )
    except asyncio.CancelledError:
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
        _emit_hsl_replay_event(
            self,
            EventTypes.HSL_REPLAY_FAILED,
            {
                "signal_mode": "coin",
                "error_type": type(exc).__name__,
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
        if hasattr(self, "_set_log_silence_watchdog_context"):
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
        if not getattr(self, "_equity_hard_stop_coin_initialized", False):
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
        elif metrics["tier"] != "red":
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
    state.clear()
    state.update(self._equity_hard_stop_make_state())
    state["pnl_reset_timestamp_ms"] = reset_ts
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
                    exc,
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
            exc,
        )


async def _equity_hard_stop_check_coin(self) -> Optional[dict]:
    balance = float(self.get_raw_balance())
    ts_ms = int(self.get_exchange_time())
    out = {}
    symbols = sorted(self._equity_hard_stop_coin_symbols())
    for pside in self._hsl_psides():
        if not self._equity_hard_stop_coin_active_pside(pside):
            continue
        for symbol in symbols:
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
            elif metrics["tier"] != "red":
                state["pending_red_since_ms"] = None
                state["red_trigger_event_emitted"] = False
                if not state["halted"]:
                    self._equity_hard_stop_clear_coin_runtime_forced_mode(pside, symbol)
            if metrics["tier"] == "orange":
                target = str(self.hsl[pside]["orange_tier_mode"])
                if target == "graceful_stop":
                    self._equity_hard_stop_set_coin_runtime_forced_mode(
                        pside, symbol, "graceful_stop"
                    )
                elif target == "tp_only_with_active_entry_cancellation":
                    self._equity_hard_stop_set_coin_runtime_forced_mode(
                        pside, symbol, "tp_only_with_active_entry_cancellation"
                    )
            self._equity_hard_stop_emit_coin_status(pside, symbol, metrics)
            out[f"{pside}:{symbol}"] = metrics
    return out if out else None


def _equity_hard_stop_coin_needs_panic_supervision(
    self, pside: str, symbol: str, state: dict[str, Any]
) -> bool:
    if not self._equity_hard_stop_enabled(pside):
        return False
    if state["runtime"].red_latched() and not state["halted"]:
        return True
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
    stop_event: Optional[dict] = None,
    *,
    finalized_without_order: bool = False,
    flat_confirmations: int | None = None,
    position_count: int | None = None,
    entry_orders: int | None = None,
    nonpanic_close_orders: int | None = None,
) -> None:
    state = self._hsl_state(pside)
    cfg = self.hsl[pside]
    stop_ts_ms = int(self.get_exchange_time())
    stop_event_anchor_source = "provided_stop_event"
    stop_event_anchor_fallback_used = False
    if stop_event is None:
        fallback_stop_ts_ms = stop_ts_ms
        latest_panic_ts = self._equity_hard_stop_latest_panic_fill_timestamp_optional_ms(
            pside,
            since_ms=state.get("pending_red_since_ms"),
        )
        if latest_panic_ts is None:
            stop_ts_ms = fallback_stop_ts_ms
            stop_event_anchor_source = "current_time_fallback"
            stop_event_anchor_fallback_used = True
        else:
            stop_ts_ms = int(latest_panic_ts)
            stop_event_anchor_source = "panic_fill"
        stop_event = await self._equity_hard_stop_compute_stop_event(pside, stop_ts_ms)
    else:
        stop_ts_ms = int(stop_event["stop_event_timestamp_ms"])
    cooldown_minutes = float(cfg["cooldown_minutes_after_red"])
    no_restart_drawdown_threshold = float(cfg["no_restart_drawdown_threshold"])
    (
        no_restart_peak_strategy_equity,
        no_restart_drawdown_raw,
    ) = self._equity_hard_stop_record_no_restart_stop(pside, stop_event)
    no_restart_latched = _equity_hard_stop_no_restart_latched(cfg, no_restart_drawdown_raw)
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
    stop_event: Optional[dict] = None,
    *,
    finalized_without_order: bool = False,
    flat_confirmations: int | None = None,
    entry_orders: int | None = None,
    nonpanic_close_orders: int | None = None,
) -> None:
    state = self._hsl_coin_state(pside, symbol)
    cfg = self.hsl[pside]
    stop_ts_ms = int(self.get_exchange_time())
    stop_event_anchor_source = "provided_stop_event"
    stop_event_anchor_fallback_used = False
    if stop_event is None:
        fallback_stop_ts_ms = stop_ts_ms
        latest_panic_ts = self._equity_hard_stop_latest_panic_fill_timestamp_optional_ms(
            pside,
            symbol=symbol,
            since_ms=state.get("pending_red_since_ms"),
        )
        if latest_panic_ts is None:
            stop_ts_ms = fallback_stop_ts_ms
            stop_event_anchor_source = "current_time_fallback"
            stop_event_anchor_fallback_used = True
        else:
            stop_ts_ms = int(latest_panic_ts)
            stop_event_anchor_source = "panic_fill"
        stop_event = await self._equity_hard_stop_compute_coin_stop_event(pside, symbol, stop_ts_ms)
    else:
        stop_ts_ms = int(stop_event["stop_event_timestamp_ms"])
    cooldown_minutes = float(cfg["cooldown_minutes_after_red"])
    no_restart_latched = _equity_hard_stop_no_restart_latched(cfg, stop_event["drawdown_raw"])
    cooldown_ms = int(round(cooldown_minutes * 60_000.0)) if cooldown_minutes > 0.0 else 0
    cooldown_until_ms = None if no_restart_latched or cooldown_ms <= 0 else int(stop_ts_ms + cooldown_ms)
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
                    stop_ts_ms = self._equity_hard_stop_latest_panic_fill_timestamp_ms(
                        pside,
                        since_ms=state.get("pending_red_since_ms"),
                        fallback_ms=int(self.get_exchange_time()),
                    )
                    state["pending_stop_event"] = await self._equity_hard_stop_compute_stop_event(
                        pside, stop_ts_ms
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
                    stop_ts_ms = self._equity_hard_stop_latest_panic_fill_timestamp_ms(
                        pside,
                        symbol=symbol,
                        since_ms=state.get("pending_red_since_ms"),
                        fallback_ms=int(self.get_exchange_time()),
                    )
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
                    self._equity_hard_stop_set_coin_runtime_forced_mode(pside, symbol, "panic")
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
