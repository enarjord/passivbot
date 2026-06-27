from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from procedures import load_user_info
from live.smoke_report import _redact_log_text
from tools.probe_ticker_capabilities import (
    create_exchange,
    is_active_linear_swap,
    resolve_symbols,
    split_csv,
    summarize_order_book,
    summarize_ticker,
    summarize_tickers,
    timed_call as _raw_timed_call,
)
from utils import ts_to_date, utc_ms


ACCOUNT_CRITICAL_ENDPOINTS = {
    "balance": "fetch_balance",
    "positions": "fetch_positions",
    "open_orders": "fetch_open_orders_all",
}


def select_default_symbols(
    markets: dict[str, Any], *, quote: str | None, max_symbols: int
) -> list[str]:
    """Pick a small deterministic swap/contract symbol sample when no symbols are provided."""
    quote_upper = str(quote or "").upper()
    candidates = []
    for symbol, market in markets.items():
        if quote_upper and str(market.get("quote") or "").upper() != quote_upper:
            continue
        if market.get("active") is False:
            continue
        if not is_active_linear_swap(market):
            continue
        candidates.append(str(symbol))
    candidates.sort()
    return candidates[: max(0, int(max_symbols))]


def summarize_market(market: Any) -> dict[str, Any]:
    if not isinstance(market, dict):
        return {"type": type(market).__name__, "valid": False}
    limits = market.get("limits") if isinstance(market.get("limits"), dict) else {}
    precision = market.get("precision") if isinstance(market.get("precision"), dict) else {}
    return {
        "id": market.get("id"),
        "symbol": market.get("symbol"),
        "base": market.get("base"),
        "quote": market.get("quote"),
        "settle": market.get("settle"),
        "type": market.get("type"),
        "active": market.get("active"),
        "swap": market.get("swap"),
        "future": market.get("future"),
        "contract": market.get("contract"),
        "linear": market.get("linear"),
        "inverse": market.get("inverse"),
        "precision": {
            "amount": precision.get("amount"),
            "price": precision.get("price"),
        },
        "limits": {
            "amount_min": (limits.get("amount") or {}).get("min"),
            "cost_min": (limits.get("cost") or {}).get("min"),
        },
    }


def summarize_ohlcvs(ohlcvs: Any) -> dict[str, Any]:
    if not isinstance(ohlcvs, list):
        return {"type": type(ohlcvs).__name__, "count": None, "valid": False}
    observed_at_ms = int(utc_ms())
    current_minute_open = int(observed_at_ms // 60_000 * 60_000)
    rows = []
    for row in ohlcvs[-3:]:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            rows.append({"valid": False, "type": type(row).__name__})
            continue
        try:
            ts = int(row[0])
        except (TypeError, ValueError):
            ts = None
        rows.append(
            {
                "timestamp": ts,
                "datetime": ts_to_date(ts) if ts is not None else None,
                "is_current_incomplete_minute": ts == current_minute_open,
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
            }
        )
    last_ts = rows[-1].get("timestamp") if rows and rows[-1].get("valid", True) else None
    last_age_ms = None
    if last_ts is not None:
        last_age_ms = max(0, int(observed_at_ms) - int(last_ts))
    return {
        "count": len(ohlcvs),
        "observed_at_ms": observed_at_ms,
        "observed_at": ts_to_date(observed_at_ms),
        "last_timestamp": last_ts,
        "last_datetime": ts_to_date(last_ts) if last_ts is not None else None,
        "last_age_ms": last_age_ms,
        "last_age_minutes": round(last_age_ms / 60_000.0, 3) if last_age_ms is not None else None,
        "last_is_current_incomplete_minute": last_ts == current_minute_open,
        "sample_tail": rows,
    }


def summarize_collection(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"type": "dict", "count": len(value), "keys_sample": sorted(map(str, value.keys()))[:20]}
    if isinstance(value, list):
        return {"type": "list", "count": len(value)}
    return {"type": type(value).__name__, "count": None}


def _int_timestamp(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def summarize_my_trades(trades: Any) -> dict[str, Any]:
    """Summarize fill-history sample shape without raw trade/order ids."""
    if not isinstance(trades, list):
        return {"type": type(trades).__name__, "count": None, "valid": False}
    timestamps = []
    symbols = set()
    side_counts: Counter[str] = Counter()
    id_present_count = 0
    order_present_count = 0
    dict_count = 0
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        dict_count += 1
        ts = _int_timestamp(trade.get("timestamp"))
        if ts is not None:
            timestamps.append(ts)
        symbol = trade.get("symbol")
        if symbol is not None:
            symbols.add(str(symbol))
        side = trade.get("side")
        if side is not None:
            side_counts[str(side)] += 1
        if trade.get("id") is not None:
            id_present_count += 1
        if trade.get("order") is not None or trade.get("orderId") is not None:
            order_present_count += 1
    timestamps.sort()
    first_ts = timestamps[0] if timestamps else None
    last_ts = timestamps[-1] if timestamps else None
    return {
        "type": "list",
        "count": len(trades),
        "dict_count": dict_count,
        "timestamp_count": len(timestamps),
        "missing_timestamp_count": max(0, len(trades) - len(timestamps)),
        "first_timestamp": first_ts,
        "first_datetime": ts_to_date(first_ts) if first_ts is not None else None,
        "last_timestamp": last_ts,
        "last_datetime": ts_to_date(last_ts) if last_ts is not None else None,
        "symbol_count": len(symbols),
        "symbols_sample": sorted(symbols)[:8],
        "symbols_truncated": max(0, len(symbols) - 8),
        "side_counts": dict(sorted(side_counts.items())),
        "id_present_count": id_present_count,
        "order_present_count": order_present_count,
        "top_level_keys_sample": sorted(
            {
                str(key)
                for trade in trades
                if isinstance(trade, dict)
                for key in trade.keys()
            }
        )[:24],
    }


def emit_progress(message: str, *, enabled: bool) -> None:
    if enabled:
        print(message, flush=True)


def format_outcome(outcome: dict[str, Any]) -> str:
    elapsed = outcome.get("elapsed_ms")
    elapsed_part = f" elapsed_ms={elapsed}" if elapsed is not None else ""
    if outcome.get("ok"):
        return f"ok{elapsed_part}"
    return (
        f"failed{elapsed_part} error_type={outcome.get('error_type')} "
        f"error={outcome.get('error')}"
    )


def _redact_probe_outcome_error(outcome: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(outcome, dict):
        return outcome
    error = outcome.get("error")
    if not isinstance(error, str):
        return outcome
    out = dict(outcome)
    out["error"] = _redact_log_text(error, max_len=500)
    return out


async def _timed_call(coro) -> dict[str, Any]:
    return _redact_probe_outcome_error(await _raw_timed_call(coro))


def _pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) * 100.0 / float(denominator), 3)


def _elapsed_ms(outcome: Any) -> float | None:
    if not isinstance(outcome, dict):
        return None
    elapsed = outcome.get("elapsed_ms")
    if not isinstance(elapsed, (int, float)):
        return None
    return round(float(elapsed), 3)


def _latency_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "count": len(ordered),
        "min": round(ordered[0], 3),
        "median": round(float(statistics.median(ordered)), 3),
        "p95": round(ordered[p95_index], 3),
        "max": round(ordered[-1], 3),
    }


def summarize_account_critical_probe_health(probe: dict[str, Any]) -> dict[str, Any]:
    """Summarize authenticated account-state endpoint health without raw payloads/errors."""
    include_private = bool(probe.get("include_private", True))
    repeats = probe.get("repeats") if isinstance(probe.get("repeats"), list) else []
    surfaces: dict[str, Any] = {}
    for surface, outcome_key in ACCOUNT_CRITICAL_ENDPOINTS.items():
        total = 0
        succeeded = 0
        failed = 0
        latencies = []
        error_types: Counter[str] = Counter()
        latest: dict[str, Any] | None = None
        for repeat in repeats:
            outcome = repeat.get(outcome_key) if isinstance(repeat, dict) else None
            if outcome is None and not include_private:
                continue
            total += 1
            elapsed_ms = _elapsed_ms(outcome)
            if elapsed_ms is not None:
                latencies.append(elapsed_ms)
            ok = bool(outcome.get("ok")) if isinstance(outcome, dict) else False
            if ok:
                succeeded += 1
                latest = {"ok": True, "elapsed_ms": elapsed_ms}
                continue
            failed += 1
            error_type = (
                str(outcome.get("error_type") or "unknown")
                if isinstance(outcome, dict)
                else "missing_outcome"
            )
            error_types[error_type] += 1
            latest = {"ok": False, "elapsed_ms": elapsed_ms, "error_type": error_type}
        surfaces[surface] = {
            "endpoint": outcome_key,
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "failure_pct": _pct(failed, total),
            "latency_ms": _latency_summary(latencies),
            "error_types": dict(sorted(error_types.items())),
            "latest": latest,
        }

    total = sum(int(surface["total"]) for surface in surfaces.values())
    succeeded = sum(int(surface["succeeded"]) for surface in surfaces.values())
    failed = sum(int(surface["failed"]) for surface in surfaces.values())
    return {
        "enabled": include_private,
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "failure_pct": _pct(failed, total),
        "surfaces": surfaces,
    }


def summarize_account_critical_probe_collection(probes: list[dict[str, Any]]) -> dict[str, Any]:
    users = []
    total = 0
    succeeded = 0
    failed = 0
    for probe in probes:
        summary = probe.get("account_critical_health")
        if not isinstance(summary, dict):
            summary = summarize_account_critical_probe_health(probe)
        user_summary = {
            "user": probe.get("user"),
            "exchange": probe.get("exchange"),
            "enabled": bool(summary.get("enabled")),
            "total": int(summary.get("total") or 0),
            "succeeded": int(summary.get("succeeded") or 0),
            "failed": int(summary.get("failed") or 0),
            "failure_pct": summary.get("failure_pct"),
        }
        users.append(user_summary)
        total += user_summary["total"]
        succeeded += user_summary["succeeded"]
        failed += user_summary["failed"]
    return {
        "users": users,
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "failure_pct": _pct(failed, total),
    }


def _clock_skew_summary(values: list[float]) -> dict[str, Any]:
    summary = _latency_summary(values)
    summary["max_abs"] = (
        round(max((abs(value) for value in values), default=0.0), 3)
        if values
        else None
    )
    return summary


def _round_ms_value(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value), 3)


def summarize_time_sync_probe_health(probe: dict[str, Any]) -> dict[str, Any]:
    """Summarize read-only exchange clock-skew probe results."""
    include_time_sync = bool(probe.get("include_time_sync", True))
    repeats = probe.get("repeats") if isinstance(probe.get("repeats"), list) else []
    total = 0
    succeeded = 0
    failed = 0
    unsupported = 0
    latencies = []
    skews = []
    error_types: Counter[str] = Counter()
    latest: dict[str, Any] | None = None
    for repeat in repeats:
        outcome = repeat.get("fetch_time") if isinstance(repeat, dict) else None
        if outcome is None:
            if include_time_sync:
                unsupported += 1
                latest = {"ok": False, "supported": False, "skipped": True}
            continue
        if not isinstance(outcome, dict):
            total += 1
            failed += 1
            error_types["missing_outcome"] += 1
            latest = {
                "ok": False,
                "elapsed_ms": None,
                "error_type": "missing_outcome",
            }
            continue
        if bool(outcome.get("skipped")) or outcome.get("supported") is False:
            unsupported += 1
            latest = {"ok": False, "supported": False, "skipped": True}
            continue
        total += 1
        elapsed_ms = _elapsed_ms(outcome)
        if elapsed_ms is not None:
            latencies.append(elapsed_ms)
        ok = bool(outcome.get("ok")) if isinstance(outcome, dict) else False
        if ok:
            succeeded += 1
            skew = _round_ms_value(outcome.get("clock_skew_ms"))
            if skew is not None:
                skews.append(skew)
            latest = {
                "ok": True,
                "elapsed_ms": elapsed_ms,
                "clock_skew_ms": skew,
                "abs_clock_skew_ms": round(abs(skew), 3) if skew is not None else None,
            }
            continue
        failed += 1
        error_type = (
            str(outcome.get("error_type") or "unknown")
            if isinstance(outcome, dict)
            else "missing_outcome"
        )
        error_types[error_type] += 1
        latest = {"ok": False, "elapsed_ms": elapsed_ms, "error_type": error_type}
    return {
        "enabled": include_time_sync,
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "unsupported": unsupported,
        "failure_pct": _pct(failed, total),
        "latency_ms": _latency_summary(latencies),
        "clock_skew_ms": _clock_skew_summary(skews),
        "error_types": dict(sorted(error_types.items())),
        "latest": latest,
    }


def summarize_time_sync_probe_collection(probes: list[dict[str, Any]]) -> dict[str, Any]:
    users = []
    total = 0
    succeeded = 0
    failed = 0
    unsupported = 0
    skews = []
    for probe in probes:
        summary = probe.get("time_sync_health")
        if not isinstance(summary, dict):
            summary = summarize_time_sync_probe_health(probe)
        skew_summary = (
            summary.get("clock_skew_ms") if isinstance(summary.get("clock_skew_ms"), dict) else {}
        )
        latest = summary.get("latest") if isinstance(summary.get("latest"), dict) else {}
        latest_skew = _round_ms_value(latest.get("clock_skew_ms"))
        probe_skews: list[float] = []
        repeats = probe.get("repeats") if isinstance(probe.get("repeats"), list) else []
        for repeat in repeats:
            outcome = repeat.get("fetch_time") if isinstance(repeat, dict) else None
            if isinstance(outcome, dict) and outcome.get("ok"):
                skew = _round_ms_value(outcome.get("clock_skew_ms"))
                if skew is not None:
                    probe_skews.append(skew)
        if probe_skews:
            skews.extend(probe_skews)
        elif latest_skew is not None:
            skews.append(latest_skew)
        user_summary = {
            "user": probe.get("user"),
            "exchange": probe.get("exchange"),
            "enabled": bool(summary.get("enabled")),
            "total": int(summary.get("total") or 0),
            "succeeded": int(summary.get("succeeded") or 0),
            "failed": int(summary.get("failed") or 0),
            "unsupported": int(summary.get("unsupported") or 0),
            "failure_pct": summary.get("failure_pct"),
            "latest_clock_skew_ms": latest_skew,
            "max_abs_clock_skew_ms": skew_summary.get("max_abs"),
        }
        users.append(user_summary)
        total += user_summary["total"]
        succeeded += user_summary["succeeded"]
        failed += user_summary["failed"]
        unsupported += user_summary["unsupported"]
    return {
        "users": users,
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "unsupported": unsupported,
        "failure_pct": _pct(failed, total),
        "clock_skew_ms": _clock_skew_summary(skews),
    }


def _freshness_summary(values: list[float]) -> dict[str, Any]:
    return _latency_summary(values)


def summarize_candle_freshness_probe_health(probe: dict[str, Any]) -> dict[str, Any]:
    """Summarize existing 1m OHLCV tail probe freshness without extra exchange calls."""
    include_public = bool(probe.get("include_public", True))
    include_ohlcv = bool(probe.get("include_ohlcv", True))
    enabled = include_public and include_ohlcv
    repeats = probe.get("repeats") if isinstance(probe.get("repeats"), list) else []
    total_symbols = 0
    succeeded_symbols = 0
    failed_symbols = 0
    missing_timestamp_symbols = 0
    current_incomplete_symbols = 0
    latencies = []
    last_ages = []
    error_types: Counter[str] = Counter()
    worst: dict[str, Any] | None = None
    latest: dict[str, Any] | None = None
    for repeat in repeats:
        outcome = repeat.get("fetch_ohlcv_1m_tail") if isinstance(repeat, dict) else None
        if outcome is None:
            continue
        symbols = outcome.get("symbols") if isinstance(outcome, dict) else None
        if not isinstance(symbols, dict):
            total_symbols += 1
            failed_symbols += 1
            error_types["missing_symbols"] += 1
            latest = {"ok": False, "error_type": "missing_symbols"}
            continue
        for symbol, symbol_outcome in sorted(symbols.items()):
            total_symbols += 1
            elapsed_ms = _elapsed_ms(symbol_outcome)
            if elapsed_ms is not None:
                latencies.append(elapsed_ms)
            if not isinstance(symbol_outcome, dict) or not symbol_outcome.get("ok"):
                failed_symbols += 1
                error_type = (
                    str(symbol_outcome.get("error_type") or "unknown")
                    if isinstance(symbol_outcome, dict)
                    else "missing_outcome"
                )
                error_types[error_type] += 1
                latest = {
                    "ok": False,
                    "symbol": str(symbol),
                    "elapsed_ms": elapsed_ms,
                    "error_type": error_type,
                }
                continue
            value = symbol_outcome.get("value")
            if not isinstance(value, dict):
                failed_symbols += 1
                error_types["missing_value"] += 1
                latest = {
                    "ok": False,
                    "symbol": str(symbol),
                    "elapsed_ms": elapsed_ms,
                    "error_type": "missing_value",
                }
                continue
            succeeded_symbols += 1
            last_age_ms = _round_ms_value(value.get("last_age_ms"))
            last_ts = value.get("last_timestamp")
            if last_ts is None:
                missing_timestamp_symbols += 1
            if bool(value.get("last_is_current_incomplete_minute")):
                current_incomplete_symbols += 1
            if last_age_ms is not None:
                last_ages.append(last_age_ms)
                candidate = {
                    "symbol": str(symbol),
                    "last_age_ms": last_age_ms,
                    "last_datetime": value.get("last_datetime"),
                    "last_is_current_incomplete_minute": bool(
                        value.get("last_is_current_incomplete_minute")
                    ),
                }
                if worst is None or last_age_ms > float(worst.get("last_age_ms") or -1.0):
                    worst = candidate
                latest = {"ok": True, **candidate}
            else:
                latest = {
                    "ok": True,
                    "symbol": str(symbol),
                    "last_age_ms": None,
                    "last_datetime": value.get("last_datetime"),
                    "last_is_current_incomplete_minute": bool(
                        value.get("last_is_current_incomplete_minute")
                    ),
                }
    return {
        "enabled": enabled,
        "total_symbols": total_symbols,
        "succeeded_symbols": succeeded_symbols,
        "failed_symbols": failed_symbols,
        "failure_pct": _pct(failed_symbols, total_symbols),
        "missing_timestamp_symbols": missing_timestamp_symbols,
        "current_incomplete_symbols": current_incomplete_symbols,
        "latency_ms": _latency_summary(latencies),
        "last_age_ms": _freshness_summary(last_ages),
        "error_types": dict(sorted(error_types.items())),
        "worst": worst,
        "latest": latest,
    }


def summarize_candle_freshness_probe_collection(probes: list[dict[str, Any]]) -> dict[str, Any]:
    users = []
    total_symbols = 0
    succeeded_symbols = 0
    failed_symbols = 0
    missing_timestamp_symbols = 0
    current_incomplete_symbols = 0
    last_ages = []
    worst: dict[str, Any] | None = None
    for probe in probes:
        summary = probe.get("candle_freshness_health")
        if not isinstance(summary, dict):
            summary = summarize_candle_freshness_probe_health(probe)
        summary_worst = summary.get("worst") if isinstance(summary.get("worst"), dict) else None
        user_summary = {
            "user": probe.get("user"),
            "exchange": probe.get("exchange"),
            "enabled": bool(summary.get("enabled")),
            "total_symbols": int(summary.get("total_symbols") or 0),
            "succeeded_symbols": int(summary.get("succeeded_symbols") or 0),
            "failed_symbols": int(summary.get("failed_symbols") or 0),
            "failure_pct": summary.get("failure_pct"),
            "missing_timestamp_symbols": int(summary.get("missing_timestamp_symbols") or 0),
            "current_incomplete_symbols": int(summary.get("current_incomplete_symbols") or 0),
            "max_last_age_ms": (
                summary["last_age_ms"].get("max")
                if isinstance(summary.get("last_age_ms"), dict)
                else None
            ),
            "worst_symbol": summary_worst.get("symbol") if summary_worst else None,
        }
        users.append(user_summary)
        total_symbols += user_summary["total_symbols"]
        succeeded_symbols += user_summary["succeeded_symbols"]
        failed_symbols += user_summary["failed_symbols"]
        missing_timestamp_symbols += user_summary["missing_timestamp_symbols"]
        current_incomplete_symbols += user_summary["current_incomplete_symbols"]
        repeats = probe.get("repeats") if isinstance(probe.get("repeats"), list) else []
        for repeat in repeats:
            outcome = repeat.get("fetch_ohlcv_1m_tail") if isinstance(repeat, dict) else None
            symbols = outcome.get("symbols") if isinstance(outcome, dict) else None
            if not isinstance(symbols, dict):
                continue
            for symbol, symbol_outcome in sorted(symbols.items()):
                value = symbol_outcome.get("value") if isinstance(symbol_outcome, dict) else None
                if not isinstance(value, dict):
                    continue
                last_age_ms = _round_ms_value(value.get("last_age_ms"))
                if last_age_ms is None:
                    continue
                last_ages.append(last_age_ms)
                candidate = {
                    "user": probe.get("user"),
                    "exchange": probe.get("exchange"),
                    "symbol": str(symbol),
                    "last_age_ms": last_age_ms,
                    "last_datetime": value.get("last_datetime"),
                    "last_is_current_incomplete_minute": bool(
                        value.get("last_is_current_incomplete_minute")
                    ),
                }
                if worst is None or last_age_ms > float(worst.get("last_age_ms") or -1.0):
                    worst = candidate
    return {
        "users": users,
        "total_symbols": total_symbols,
        "succeeded_symbols": succeeded_symbols,
        "failed_symbols": failed_symbols,
        "failure_pct": _pct(failed_symbols, total_symbols),
        "missing_timestamp_symbols": missing_timestamp_symbols,
        "current_incomplete_symbols": current_incomplete_symbols,
        "last_age_ms": _freshness_summary(last_ages),
        "worst": worst,
    }


def summarize_fill_history_probe_health(probe: dict[str, Any]) -> dict[str, Any]:
    """Summarize existing fetch_my_trades probe results without raw fill payloads."""
    enabled = bool(probe.get("include_private", True)) and bool(
        probe.get("include_my_trades", True)
    )
    repeats = probe.get("repeats") if isinstance(probe.get("repeats"), list) else []
    total = 0
    succeeded = 0
    failed = 0
    latencies = []
    trade_counts = []
    error_types: Counter[str] = Counter()
    latest: dict[str, Any] | None = None
    newest_trade: dict[str, Any] | None = None
    for repeat in repeats:
        outcome = (
            repeat.get("fetch_my_trades_first_symbol")
            if isinstance(repeat, dict)
            else None
        )
        if outcome is None:
            continue
        total += 1
        elapsed_ms = _elapsed_ms(outcome)
        if elapsed_ms is not None:
            latencies.append(elapsed_ms)
        if not isinstance(outcome, dict) or not outcome.get("ok"):
            failed += 1
            error_type = (
                str(outcome.get("error_type") or "unknown")
                if isinstance(outcome, dict)
                else "missing_outcome"
            )
            error_types[error_type] += 1
            latest = {
                "ok": False,
                "symbol": outcome.get("symbol") if isinstance(outcome, dict) else None,
                "elapsed_ms": elapsed_ms,
                "error_type": error_type,
                "call_count": int(outcome.get("call_count") or 0)
                if isinstance(outcome, dict)
                else 0,
                "requested_pages": int(outcome.get("requested_pages") or 0)
                if isinstance(outcome, dict)
                else 0,
                "page_count": int(outcome.get("page_count") or 0)
                if isinstance(outcome, dict)
                else 0,
                "terminal_reason": outcome.get("terminal_reason")
                if isinstance(outcome, dict)
                else None,
            }
            continue
        value = outcome.get("value")
        if not isinstance(value, dict):
            failed += 1
            error_types["missing_value"] += 1
            latest = {
                "ok": False,
                "symbol": outcome.get("symbol"),
                "elapsed_ms": elapsed_ms,
                "error_type": "missing_value",
            }
            continue
        succeeded += 1
        trade_count = int(value.get("count") or 0)
        trade_counts.append(float(trade_count))
        last_ts = _int_timestamp(value.get("last_timestamp"))
        candidate = {
            "ok": True,
            "symbol": outcome.get("symbol"),
            "elapsed_ms": elapsed_ms,
            "call_count": int(outcome.get("call_count") or 0),
            "requested_pages": int(outcome.get("requested_pages") or 0),
            "page_limit": int(outcome.get("page_limit") or 0),
            "page_count": int(outcome.get("page_count") or 0),
            "terminal_reason": outcome.get("terminal_reason"),
            "trade_count": trade_count,
            "timestamp_count": int(value.get("timestamp_count") or 0),
            "missing_timestamp_count": int(value.get("missing_timestamp_count") or 0),
            "first_timestamp": _int_timestamp(value.get("first_timestamp")),
            "first_datetime": value.get("first_datetime"),
            "last_timestamp": last_ts,
            "last_datetime": value.get("last_datetime"),
            "symbol_count": int(value.get("symbol_count") or 0),
            "side_counts": (
                value.get("side_counts") if isinstance(value.get("side_counts"), dict) else {}
            ),
            "id_present_count": int(value.get("id_present_count") or 0),
            "order_present_count": int(value.get("order_present_count") or 0),
        }
        latest = candidate
        if last_ts is not None and (
            newest_trade is None
            or last_ts > int(newest_trade.get("last_timestamp") or -1)
        ):
            newest_trade = candidate
    return {
        "enabled": enabled,
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "failure_pct": _pct(failed, total),
        "latency_ms": _latency_summary(latencies),
        "trade_count": _latency_summary(trade_counts),
        "error_types": dict(sorted(error_types.items())),
        "latest": latest,
        "newest_trade": newest_trade,
    }


def summarize_fill_history_probe_collection(probes: list[dict[str, Any]]) -> dict[str, Any]:
    users = []
    total = 0
    succeeded = 0
    failed = 0
    trade_counts = []
    newest_trade: dict[str, Any] | None = None
    for probe in probes:
        summary = probe.get("fill_history_health")
        if not isinstance(summary, dict):
            summary = summarize_fill_history_probe_health(probe)
        latest = summary.get("latest") if isinstance(summary.get("latest"), dict) else {}
        summary_newest = (
            summary.get("newest_trade")
            if isinstance(summary.get("newest_trade"), dict)
            else None
        )
        user_summary = {
            "user": probe.get("user"),
            "exchange": probe.get("exchange"),
            "enabled": bool(summary.get("enabled")),
            "total": int(summary.get("total") or 0),
            "succeeded": int(summary.get("succeeded") or 0),
            "failed": int(summary.get("failed") or 0),
            "failure_pct": summary.get("failure_pct"),
            "latest_symbol": latest.get("symbol"),
            "latest_trade_count": int(latest.get("trade_count") or 0),
            "latest_page_count": int(latest.get("page_count") or 0),
            "latest_call_count": int(latest.get("call_count") or 0),
            "latest_terminal_reason": latest.get("terminal_reason"),
            "newest_trade_timestamp": (
                summary_newest.get("last_timestamp") if summary_newest else None
            ),
            "newest_trade_datetime": (
                summary_newest.get("last_datetime") if summary_newest else None
            ),
        }
        users.append(user_summary)
        total += user_summary["total"]
        succeeded += user_summary["succeeded"]
        failed += user_summary["failed"]
        if user_summary["total"] > 0 and latest:
            trade_counts.append(float(user_summary["latest_trade_count"]))
        if summary_newest is not None:
            ts = _int_timestamp(summary_newest.get("last_timestamp"))
            if ts is not None and (
                newest_trade is None
                or ts > int(newest_trade.get("last_timestamp") or -1)
            ):
                newest_trade = {
                    "user": probe.get("user"),
                    "exchange": probe.get("exchange"),
                    "symbol": summary_newest.get("symbol"),
                    "last_timestamp": ts,
                    "last_datetime": summary_newest.get("last_datetime"),
                    "trade_count": int(summary_newest.get("trade_count") or 0),
                }
    return {
        "users": users,
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "failure_pct": _pct(failed, total),
        "trade_count": _latency_summary(trade_counts),
        "newest_trade": newest_trade,
    }


def _non_negative_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        return None
    return round(parsed, 3)


def _probe_outcome_call_count(outcome: Any) -> int:
    if not isinstance(outcome, dict) or bool(outcome.get("skipped")):
        return 0
    call_count = outcome.get("call_count")
    if isinstance(call_count, int) and not isinstance(call_count, bool) and call_count >= 0:
        return int(call_count)
    return 1


def _symbol_outcome_call_count(outcome: Any) -> int:
    if not isinstance(outcome, dict) or bool(outcome.get("skipped")):
        return 0
    symbols = outcome.get("symbols")
    if isinstance(symbols, dict):
        return len(symbols)
    return _probe_outcome_call_count(outcome)


def _open_orders_call_count(outcome: Any) -> int:
    if not isinstance(outcome, dict) or bool(outcome.get("skipped")):
        return 0
    attempts = outcome.get("attempts")
    if not isinstance(attempts, dict):
        return _probe_outcome_call_count(outcome)
    count = 0
    if isinstance(attempts.get("all_symbols"), dict):
        count += 1
    symbol_attempt = attempts.get("symbol")
    if isinstance(symbol_attempt, dict) and isinstance(symbol_attempt.get("outcome"), dict):
        count += 1
    return count


def _add_rate_limit_count(
    counters: dict[str, Any],
    *,
    endpoint: str,
    category: str,
    count: int,
    concurrent: bool = False,
) -> None:
    if count <= 0:
        return
    counters["total"] += int(count)
    counters[category] += int(count)
    counters["endpoint_counts"][endpoint] += int(count)
    if concurrent:
        counters["concurrent_request_count"] += int(count)
        counters["concurrent_endpoint_counts"][endpoint] += int(count)


def _new_rate_limit_counters() -> dict[str, Any]:
    return {
        "total": 0,
        "market_metadata": 0,
        "time_sync": 0,
        "public": 0,
        "private": 0,
        "concurrent_request_count": 0,
        "endpoint_counts": Counter(),
        "concurrent_endpoint_counts": Counter(),
    }


def _rate_limit_counts_for_repeat(repeat: dict[str, Any]) -> dict[str, Any]:
    counters = _new_rate_limit_counters()
    _add_rate_limit_count(
        counters,
        endpoint="fetch_time",
        category="time_sync",
        count=_probe_outcome_call_count(repeat.get("fetch_time")),
    )
    for endpoint in ("fetch_tickers_all", "fetch_tickers_symbols"):
        _add_rate_limit_count(
            counters,
            endpoint=endpoint,
            category="public",
            count=_probe_outcome_call_count(repeat.get(endpoint)),
        )
    for endpoint in (
        "fetch_ticker_sequential",
        "fetch_order_book_sequential",
        "fetch_ohlcv_1m_tail",
    ):
        _add_rate_limit_count(
            counters,
            endpoint=endpoint,
            category="public",
            count=_symbol_outcome_call_count(repeat.get(endpoint)),
        )
    for endpoint in ("fetch_ticker_concurrent", "fetch_order_book_concurrent"):
        _add_rate_limit_count(
            counters,
            endpoint=endpoint,
            category="public",
            count=_symbol_outcome_call_count(repeat.get(endpoint)),
            concurrent=True,
        )
    for endpoint in ("fetch_bids_asks_all", "fetch_bids_asks_symbols"):
        _add_rate_limit_count(
            counters,
            endpoint=endpoint,
            category="public",
            count=_probe_outcome_call_count(repeat.get(endpoint)),
        )
    for endpoint in ("fetch_balance", "fetch_positions", "fetch_my_trades_first_symbol"):
        _add_rate_limit_count(
            counters,
            endpoint=endpoint,
            category="private",
            count=_probe_outcome_call_count(repeat.get(endpoint)),
        )
    _add_rate_limit_count(
        counters,
        endpoint="fetch_open_orders",
        category="private",
        count=_open_orders_call_count(repeat.get("fetch_open_orders_all")),
    )
    return counters


def _merge_rate_limit_counters(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in (
        "total",
        "market_metadata",
        "time_sync",
        "public",
        "private",
        "concurrent_request_count",
    ):
        target[key] += int(source[key])
    target["endpoint_counts"].update(source["endpoint_counts"])
    target["concurrent_endpoint_counts"].update(source["concurrent_endpoint_counts"])


def _counter_to_sorted_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def summarize_rate_limit_probe_health(probe: dict[str, Any]) -> dict[str, Any]:
    """Estimate read-only probe request pressure from already-recorded outcomes."""
    counters = _new_rate_limit_counters()
    repeat_totals = []
    load_markets_count = _probe_outcome_call_count(probe.get("load_markets"))
    _add_rate_limit_count(
        counters,
        endpoint="load_markets",
        category="market_metadata",
        count=load_markets_count,
    )
    repeats = probe.get("repeats") if isinstance(probe.get("repeats"), list) else []
    for repeat in repeats:
        if not isinstance(repeat, dict):
            continue
        repeat_counters = _rate_limit_counts_for_repeat(repeat)
        repeat_totals.append(float(repeat_counters["total"]))
        _merge_rate_limit_counters(counters, repeat_counters)
    exchange_rate_limit_ms = _non_negative_float(probe.get("exchange_rate_limit_ms"))
    estimated_min_serial_ms = (
        round(counters["total"] * exchange_rate_limit_ms, 3)
        if exchange_rate_limit_ms is not None
        else None
    )
    notes = []
    if probe.get("exchange_enable_rate_limit") is False:
        notes.append("ccxt_rate_limit_disabled")
    if counters["concurrent_request_count"] > 0:
        notes.append("contains_concurrent_batches")
    if counters["private"] > 0:
        notes.append("contains_authenticated_calls")
    return {
        "exchange_rate_limit_ms": exchange_rate_limit_ms,
        "exchange_enable_rate_limit": probe.get("exchange_enable_rate_limit"),
        "symbol_count": len(probe.get("symbols")) if isinstance(probe.get("symbols"), list) else 0,
        "repeat_count": len(repeats),
        "configured_sleep_between_seconds": _non_negative_float(
            probe.get("sleep_between_seconds")
        ),
        "observed_call_count": int(counters["total"]),
        "market_metadata_call_count": int(counters["market_metadata"]),
        "time_sync_call_count": int(counters["time_sync"]),
        "public_call_count": int(counters["public"]),
        "private_call_count": int(counters["private"]),
        "concurrent_request_count": int(counters["concurrent_request_count"]),
        "estimated_min_serial_ms": estimated_min_serial_ms,
        "estimated_min_serial_seconds": (
            round(estimated_min_serial_ms / 1000.0, 3)
            if estimated_min_serial_ms is not None
            else None
        ),
        "calls_per_repeat": _latency_summary(repeat_totals),
        "endpoint_counts": _counter_to_sorted_dict(counters["endpoint_counts"]),
        "concurrent_endpoint_counts": _counter_to_sorted_dict(
            counters["concurrent_endpoint_counts"]
        ),
        "notes": notes,
    }


def summarize_rate_limit_probe_collection(probes: list[dict[str, Any]]) -> dict[str, Any]:
    users = []
    totals = _new_rate_limit_counters()
    rate_limits = []
    estimated_serials = []
    for probe in probes:
        summary = probe.get("rate_limit_health")
        if not isinstance(summary, dict):
            summary = summarize_rate_limit_probe_health(probe)
        observed_call_count = int(summary.get("observed_call_count") or 0)
        user_summary = {
            "user": probe.get("user"),
            "exchange": probe.get("exchange"),
            "exchange_rate_limit_ms": summary.get("exchange_rate_limit_ms"),
            "exchange_enable_rate_limit": summary.get("exchange_enable_rate_limit"),
            "observed_call_count": observed_call_count,
            "private_call_count": int(summary.get("private_call_count") or 0),
            "public_call_count": int(summary.get("public_call_count") or 0),
            "concurrent_request_count": int(summary.get("concurrent_request_count") or 0),
            "estimated_min_serial_ms": summary.get("estimated_min_serial_ms"),
            "notes": summary.get("notes") if isinstance(summary.get("notes"), list) else [],
        }
        users.append(user_summary)
        for key, counter_key in (
            ("observed_call_count", "total"),
            ("private_call_count", "private"),
            ("public_call_count", "public"),
            ("concurrent_request_count", "concurrent_request_count"),
        ):
            totals[counter_key] += int(user_summary[key])
        endpoint_counts = summary.get("endpoint_counts")
        if isinstance(endpoint_counts, dict):
            totals["endpoint_counts"].update(
                {str(key): int(value) for key, value in endpoint_counts.items()}
            )
        concurrent_counts = summary.get("concurrent_endpoint_counts")
        if isinstance(concurrent_counts, dict):
            totals["concurrent_endpoint_counts"].update(
                {str(key): int(value) for key, value in concurrent_counts.items()}
            )
        rate_limit = _non_negative_float(summary.get("exchange_rate_limit_ms"))
        if rate_limit is not None:
            rate_limits.append(rate_limit)
        serial_ms = _non_negative_float(summary.get("estimated_min_serial_ms"))
        if serial_ms is not None:
            estimated_serials.append(serial_ms)
    return {
        "users": users,
        "observed_call_count": int(totals["total"]),
        "public_call_count": int(totals["public"]),
        "private_call_count": int(totals["private"]),
        "concurrent_request_count": int(totals["concurrent_request_count"]),
        "exchange_rate_limit_ms": _latency_summary(rate_limits),
        "estimated_min_serial_ms": _latency_summary(estimated_serials),
        "endpoint_counts": _counter_to_sorted_dict(totals["endpoint_counts"]),
        "concurrent_endpoint_counts": _counter_to_sorted_dict(
            totals["concurrent_endpoint_counts"]
        ),
    }


async def _timed_load_markets(exchange) -> tuple[dict[str, Any], dict[str, Any]]:
    outcome = await _timed_call(exchange.load_markets())
    markets = outcome["value"] if outcome["ok"] and isinstance(outcome["value"], dict) else {}
    if outcome["ok"]:
        outcome["value"] = {
            "count": len(markets) if isinstance(markets, dict) else None,
            "type": type(markets).__name__,
        }
    return outcome, markets


async def _fetch_exchange_time_sync(exchange) -> dict[str, Any]:
    method = getattr(exchange, "fetch_time", None)
    has = getattr(exchange, "has", None)
    supported = isinstance(has, dict) and has.get("fetchTime") is True
    if method is None or not supported:
        return {
            "ok": False,
            "supported": False,
            "skipped": True,
            "error_type": "UnsupportedEndpoint",
            "error": "fetch_time unavailable",
        }
    local_before_ms = int(utc_ms())
    outcome = await _timed_call(method())
    local_after_ms = int(utc_ms())
    local_midpoint_ms = int(round((local_before_ms + local_after_ms) / 2.0))
    outcome["supported"] = True
    outcome["local_before_ms"] = local_before_ms
    outcome["local_after_ms"] = local_after_ms
    outcome["local_midpoint_ms"] = local_midpoint_ms
    if not outcome["ok"] and outcome.get("error_type") == "NotSupported":
        outcome["supported"] = False
        outcome["skipped"] = True
        return outcome
    if outcome["ok"]:
        try:
            exchange_time_ms = int(outcome.pop("value"))
        except (TypeError, ValueError):
            outcome.update(
                {
                    "ok": False,
                    "error_type": "ValueError",
                    "error": "fetch_time returned non-integer timestamp",
                }
            )
        else:
            outcome["exchange_time_ms"] = exchange_time_ms
            outcome["exchange_datetime"] = ts_to_date(exchange_time_ms)
            outcome["clock_skew_ms"] = int(exchange_time_ms - local_midpoint_ms)
            outcome["abs_clock_skew_ms"] = abs(int(outcome["clock_skew_ms"]))
    return outcome


async def _fetch_ticker_sequential(exchange, symbols: list[str]) -> dict[str, Any]:
    started_ms = utc_ms()
    results = {}
    for symbol in symbols:
        outcome = await _timed_call(exchange.fetch_ticker(symbol))
        if outcome["ok"]:
            outcome["value"] = summarize_ticker(outcome["value"])
        results[symbol] = outcome
    elapsed_ms = int(max(0, utc_ms() - started_ms))
    return {
        "ok": all(bool(result["ok"]) for result in results.values()),
        "elapsed_ms": elapsed_ms,
        "symbols": results,
    }


async def _fetch_ticker_concurrent(exchange, symbols: list[str]) -> dict[str, Any]:
    async def _one(symbol: str) -> tuple[str, dict[str, Any]]:
        outcome = await _timed_call(exchange.fetch_ticker(symbol))
        if outcome["ok"]:
            outcome["value"] = summarize_ticker(outcome["value"])
        return symbol, outcome

    started_ms = utc_ms()
    pairs = await asyncio.gather(*[_one(symbol) for symbol in symbols])
    elapsed_ms = int(max(0, utc_ms() - started_ms))
    results = dict(pairs)
    return {
        "ok": all(bool(result["ok"]) for result in results.values()),
        "elapsed_ms": elapsed_ms,
        "symbols": results,
    }


async def _fetch_order_books_sequential(exchange, symbols: list[str]) -> dict[str, Any]:
    started_ms = utc_ms()
    results = {}
    for symbol in symbols:
        outcome = await _timed_call(exchange.fetch_order_book(symbol, limit=5))
        if outcome["ok"]:
            outcome["value"] = summarize_order_book(outcome["value"])
        results[symbol] = outcome
    return {
        "ok": all(bool(result["ok"]) for result in results.values()),
        "elapsed_ms": int(max(0, utc_ms() - started_ms)),
        "symbols": results,
    }


async def _fetch_order_books_concurrent(exchange, symbols: list[str]) -> dict[str, Any]:
    async def _one(symbol: str) -> tuple[str, dict[str, Any]]:
        outcome = await _timed_call(exchange.fetch_order_book(symbol, limit=5))
        if outcome["ok"]:
            outcome["value"] = summarize_order_book(outcome["value"])
        return symbol, outcome

    started_ms = utc_ms()
    pairs = await asyncio.gather(*[_one(symbol) for symbol in symbols])
    results = dict(pairs)
    return {
        "ok": all(bool(result["ok"]) for result in results.values()),
        "elapsed_ms": int(max(0, utc_ms() - started_ms)),
        "symbols": results,
    }


async def _fetch_ohlcvs(exchange, symbols: list[str]) -> dict[str, Any]:
    started_ms = utc_ms()
    results = {}
    for symbol in symbols:
        outcome = await _timed_call(exchange.fetch_ohlcv(symbol, timeframe="1m", limit=3))
        if outcome["ok"]:
            outcome["value"] = summarize_ohlcvs(outcome["value"])
        results[symbol] = outcome
    return {
        "ok": all(bool(result["ok"]) for result in results.values()),
        "elapsed_ms": int(max(0, utc_ms() - started_ms)),
        "symbols": results,
    }


async def _timed_method(exchange, method_name: str, *args, summary=None) -> dict[str, Any]:
    method = getattr(exchange, method_name, None)
    if method is None:
        return {"ok": False, "elapsed_ms": 0, "error_type": "AttributeError", "error": method_name}
    outcome = await _timed_call(method(*args))
    if outcome["ok"] and summary is not None:
        outcome["value"] = summary(outcome["value"])
    return outcome


async def _fetch_my_trades_sample(
    exchange,
    symbol: str,
    *,
    pages: int,
    page_limit: int,
) -> dict[str, Any]:
    requested_pages = max(1, int(pages))
    limit = max(1, int(page_limit))
    started_ms = int(utc_ms())
    page_summaries: list[dict[str, Any]] = []
    combined_trades: list[Any] = []
    since: int | None = None
    terminal_reason = "requested_pages_exhausted"
    call_count = 0
    for page_index in range(requested_pages):
        outcome = await _timed_call(exchange.fetch_my_trades(symbol, since, limit))
        call_count += 1
        page_summary: dict[str, Any] = {
            "page_index": page_index,
            "since": since,
            "limit": limit,
            "ok": bool(outcome.get("ok")),
            "elapsed_ms": _elapsed_ms(outcome),
        }
        if not outcome.get("ok"):
            page_summary["error_type"] = str(outcome.get("error_type") or "unknown")
            page_summaries.append(page_summary)
            elapsed_ms = int(max(0, utc_ms() - started_ms))
            return {
                "ok": False,
                "elapsed_ms": elapsed_ms,
                "symbol": symbol,
                "call_count": call_count,
                "requested_pages": requested_pages,
                "page_limit": limit,
                "page_count": len(page_summaries),
                "pages": page_summaries,
                "terminal_reason": "page_failed",
                "error_type": outcome.get("error_type"),
                "error": outcome.get("error"),
                "value": summarize_my_trades(combined_trades),
            }
        trades = outcome.get("value")
        if not isinstance(trades, list):
            page_summary["ok"] = False
            page_summary["error_type"] = "ValueError"
            page_summaries.append(page_summary)
            elapsed_ms = int(max(0, utc_ms() - started_ms))
            return {
                "ok": False,
                "elapsed_ms": elapsed_ms,
                "symbol": symbol,
                "call_count": call_count,
                "requested_pages": requested_pages,
                "page_limit": limit,
                "page_count": len(page_summaries),
                "pages": page_summaries,
                "terminal_reason": "invalid_page_shape",
                "error_type": "ValueError",
                "error": "fetch_my_trades returned non-list page",
                "value": summarize_my_trades(combined_trades),
            }
        page_value = summarize_my_trades(trades)
        page_summary.update(
            {
                "trade_count": int(page_value.get("count") or 0),
                "timestamp_count": int(page_value.get("timestamp_count") or 0),
                "first_timestamp": page_value.get("first_timestamp"),
                "first_datetime": page_value.get("first_datetime"),
                "last_timestamp": page_value.get("last_timestamp"),
                "last_datetime": page_value.get("last_datetime"),
            }
        )
        page_summaries.append(page_summary)
        combined_trades.extend(trades)
        last_ts = _int_timestamp(page_value.get("last_timestamp"))
        if len(trades) < limit:
            terminal_reason = "short_page"
            break
        if last_ts is None:
            terminal_reason = "missing_page_timestamp"
            break
        next_since = last_ts + 1
        if since is not None and next_since <= since:
            terminal_reason = "non_advancing_timestamp"
            break
        since = next_since
    elapsed_ms = int(max(0, utc_ms() - started_ms))
    return {
        "ok": True,
        "elapsed_ms": elapsed_ms,
        "symbol": symbol,
        "call_count": call_count,
        "requested_pages": requested_pages,
        "page_limit": limit,
        "page_count": len(page_summaries),
        "pages": page_summaries,
        "terminal_reason": terminal_reason,
        "value": summarize_my_trades(combined_trades),
    }


def _with_present_fields(out: dict[str, Any], **fields) -> dict[str, Any]:
    for key, value in fields.items():
        if value is not None:
            out[key] = value
    return out


async def _fetch_open_orders_account_critical(exchange, symbols: list[str]) -> dict[str, Any]:
    started_ms = utc_ms()
    all_symbols_outcome = await _timed_method(
        exchange, "fetch_open_orders", summary=summarize_collection
    )
    attempts: dict[str, Any] = {"all_symbols": all_symbols_outcome}
    if all_symbols_outcome["ok"]:
        return _with_present_fields(
            dict(all_symbols_outcome),
            mode="all_symbols",
            attempts=attempts,
        )
    if not symbols:
        return _with_present_fields(
            dict(all_symbols_outcome),
            mode="all_symbols_failed",
            attempts=attempts,
        )

    fallback_symbol = symbols[0]
    symbol_outcome = await _timed_method(
        exchange,
        "fetch_open_orders",
        fallback_symbol,
        summary=summarize_collection,
    )
    attempts["symbol"] = {"symbol": fallback_symbol, "outcome": symbol_outcome}
    elapsed_ms = int(max(0, utc_ms() - started_ms))
    if symbol_outcome["ok"]:
        return {
            "ok": True,
            "elapsed_ms": elapsed_ms,
            "mode": "symbol_fallback",
            "fallback_symbol": fallback_symbol,
            "attempts": attempts,
            "value": symbol_outcome.get("value"),
        }
    return _with_present_fields(
        {
            "ok": False,
            "elapsed_ms": elapsed_ms,
            "mode": "all_symbols_and_symbol_failed",
            "fallback_symbol": fallback_symbol,
            "attempts": attempts,
            "error_type": symbol_outcome.get("error_type") or all_symbols_outcome.get("error_type"),
            "error": symbol_outcome.get("error") or all_symbols_outcome.get("error"),
        }
    )


async def probe_exchange_ticker_endpoints(
    exchange,
    *,
    user: str,
    user_info: dict[str, Any],
    symbols: list[str],
    coins: list[str],
    quote: str | None,
    max_symbols: int,
    repeats: int,
    sleep_between_seconds: float,
    include_private: bool = True,
    include_public: bool = True,
    include_order_book: bool = True,
    include_ohlcv: bool = True,
    include_my_trades: bool = True,
    include_time_sync: bool = True,
    fill_history_pages: int = 1,
    fill_history_page_limit: int = 10,
    progress: bool = False,
) -> dict[str, Any]:
    exchange_id = getattr(exchange, "id", str(user_info.get("exchange") or ""))
    emit_progress(f"[{user}] loading markets | exchange={exchange_id}", enabled=progress)
    load_markets, markets = await _timed_load_markets(exchange)
    emit_progress(f"[{user}] load_markets {format_outcome(load_markets)}", enabled=progress)
    resolved_symbols = list(dict.fromkeys(symbols))
    if coins:
        emit_progress(f"[{user}] resolving coins | coins={','.join(coins)}", enabled=progress)
        resolved_symbols.extend(await resolve_symbols(exchange, coins, quote=quote or user_info.get("quote")))
        resolved_symbols = list(dict.fromkeys(resolved_symbols))
    if not resolved_symbols:
        resolved_symbols = select_default_symbols(
            markets, quote=quote or user_info.get("quote"), max_symbols=max_symbols
        )
    if not resolved_symbols and (include_public or include_my_trades):
        raise ValueError(f"user {user!r}: no symbols resolved for ticker endpoint probe")

    out: dict[str, Any] = {
        "user": user,
        "exchange": exchange_id,
        "quote": quote or user_info.get("quote"),
        "symbols": resolved_symbols,
        "include_private": bool(include_private),
        "include_public": bool(include_public),
        "include_ohlcv": bool(include_ohlcv),
        "include_my_trades": bool(include_my_trades),
        "include_time_sync": bool(include_time_sync),
        "fill_history_pages": max(1, int(fill_history_pages)),
        "fill_history_page_limit": max(1, int(fill_history_page_limit)),
        "sleep_between_seconds": float(sleep_between_seconds),
        "exchange_rate_limit_ms": getattr(exchange, "rateLimit", None),
        "exchange_enable_rate_limit": getattr(exchange, "enableRateLimit", None),
        "market_count": len(markets) if isinstance(markets, dict) else None,
        "has": {
            "fetchBalance": bool(getattr(exchange, "has", {}).get("fetchBalance")),
            "fetchBidsAsks": bool(getattr(exchange, "has", {}).get("fetchBidsAsks")),
            "fetchMyTrades": bool(getattr(exchange, "has", {}).get("fetchMyTrades")),
            "fetchOHLCV": bool(getattr(exchange, "has", {}).get("fetchOHLCV")),
            "fetchOpenOrders": bool(getattr(exchange, "has", {}).get("fetchOpenOrders")),
            "fetchOrderBook": bool(getattr(exchange, "has", {}).get("fetchOrderBook")),
            "fetchPositions": bool(getattr(exchange, "has", {}).get("fetchPositions")),
            "fetchTime": bool(getattr(exchange, "has", {}).get("fetchTime")),
            "fetchTicker": bool(getattr(exchange, "has", {}).get("fetchTicker")),
            "fetchTickers": bool(getattr(exchange, "has", {}).get("fetchTickers")),
        },
        "load_markets": load_markets,
        "markets": {symbol: summarize_market(markets.get(symbol)) for symbol in resolved_symbols},
        "repeats": [],
    }
    emit_progress(
        f"[{user}] selected symbols | count={len(resolved_symbols)} "
        f"symbols={','.join(resolved_symbols)}",
        enabled=progress,
    )

    for idx in range(max(1, int(repeats))):
        repeat_label = f"[{user}] repeat {idx + 1}/{max(1, int(repeats))}"
        emit_progress(f"{repeat_label} starting", enabled=progress)
        repeat: dict[str, Any] = {
            "index": idx,
            "started_at_ms": int(utc_ms()),
            "started_at": ts_to_date(utc_ms()),
        }

        if include_time_sync:
            emit_progress(f"{repeat_label} fetch_time", enabled=progress)
            repeat["fetch_time"] = await _fetch_exchange_time_sync(exchange)
            emit_progress(
                f"{repeat_label} fetch_time {format_outcome(repeat['fetch_time'])}",
                enabled=progress,
            )

        if include_public:
            emit_progress(f"{repeat_label} fetch_tickers()", enabled=progress)
            all_outcome = await _timed_call(exchange.fetch_tickers())
            if all_outcome["ok"]:
                all_outcome["value"] = summarize_tickers(all_outcome["value"], resolved_symbols)
            repeat["fetch_tickers_all"] = all_outcome
            emit_progress(
                f"{repeat_label} fetch_tickers() {format_outcome(all_outcome)}",
                enabled=progress,
            )

            emit_progress(f"{repeat_label} fetch_tickers(symbols)", enabled=progress)
            listed_outcome = await _timed_call(exchange.fetch_tickers(resolved_symbols))
            if listed_outcome["ok"]:
                listed_outcome["value"] = summarize_tickers(
                    listed_outcome["value"], resolved_symbols
                )
            repeat["fetch_tickers_symbols"] = listed_outcome
            emit_progress(
                f"{repeat_label} fetch_tickers(symbols) {format_outcome(listed_outcome)}",
                enabled=progress,
            )

            emit_progress(f"{repeat_label} fetch_ticker sequential", enabled=progress)
            repeat["fetch_ticker_sequential"] = await _fetch_ticker_sequential(
                exchange, resolved_symbols
            )
            emit_progress(
                f"{repeat_label} fetch_ticker sequential "
                f"{format_outcome(repeat['fetch_ticker_sequential'])}",
                enabled=progress,
            )
            emit_progress(f"{repeat_label} fetch_ticker concurrent", enabled=progress)
            repeat["fetch_ticker_concurrent"] = await _fetch_ticker_concurrent(
                exchange, resolved_symbols
            )
            emit_progress(
                f"{repeat_label} fetch_ticker concurrent "
                f"{format_outcome(repeat['fetch_ticker_concurrent'])}",
                enabled=progress,
            )
            emit_progress(f"{repeat_label} fetch_bids_asks()", enabled=progress)
            bids_asks_all = await _timed_method(
                exchange,
                "fetch_bids_asks",
                summary=lambda value: summarize_tickers(value, resolved_symbols),
            )
            repeat["fetch_bids_asks_all"] = bids_asks_all
            emit_progress(
                f"{repeat_label} fetch_bids_asks() {format_outcome(bids_asks_all)}",
                enabled=progress,
            )

            emit_progress(f"{repeat_label} fetch_bids_asks(symbols)", enabled=progress)
            bids_asks_symbols = await _timed_method(
                exchange,
                "fetch_bids_asks",
                resolved_symbols,
                summary=lambda value: summarize_tickers(value, resolved_symbols),
            )
            repeat["fetch_bids_asks_symbols"] = bids_asks_symbols
            emit_progress(
                f"{repeat_label} fetch_bids_asks(symbols) {format_outcome(bids_asks_symbols)}",
                enabled=progress,
            )

        if include_public and include_order_book:
            emit_progress(f"{repeat_label} fetch_order_book sequential", enabled=progress)
            repeat["fetch_order_book_sequential"] = await _fetch_order_books_sequential(
                exchange, resolved_symbols
            )
            emit_progress(
                f"{repeat_label} fetch_order_book sequential "
                f"{format_outcome(repeat['fetch_order_book_sequential'])}",
                enabled=progress,
            )
            emit_progress(f"{repeat_label} fetch_order_book concurrent", enabled=progress)
            repeat["fetch_order_book_concurrent"] = await _fetch_order_books_concurrent(
                exchange, resolved_symbols
            )
            emit_progress(
                f"{repeat_label} fetch_order_book concurrent "
                f"{format_outcome(repeat['fetch_order_book_concurrent'])}",
                enabled=progress,
            )
        if include_public and include_ohlcv:
            emit_progress(f"{repeat_label} fetch_ohlcv 1m tail", enabled=progress)
            repeat["fetch_ohlcv_1m_tail"] = await _fetch_ohlcvs(exchange, resolved_symbols)
            emit_progress(
                f"{repeat_label} fetch_ohlcv 1m tail "
                f"{format_outcome(repeat['fetch_ohlcv_1m_tail'])}",
                enabled=progress,
            )
        if include_private:
            emit_progress(f"{repeat_label} fetch_balance", enabled=progress)
            repeat["fetch_balance"] = await _timed_method(
                exchange, "fetch_balance", summary=summarize_collection
            )
            emit_progress(
                f"{repeat_label} fetch_balance {format_outcome(repeat['fetch_balance'])}",
                enabled=progress,
            )
            emit_progress(f"{repeat_label} fetch_positions", enabled=progress)
            repeat["fetch_positions"] = await _timed_method(
                exchange, "fetch_positions", summary=summarize_collection
            )
            emit_progress(
                f"{repeat_label} fetch_positions {format_outcome(repeat['fetch_positions'])}",
                enabled=progress,
            )
            emit_progress(f"{repeat_label} fetch_open_orders()", enabled=progress)
            repeat["fetch_open_orders_all"] = await _fetch_open_orders_account_critical(
                exchange, resolved_symbols
            )
            emit_progress(
                f"{repeat_label} fetch_open_orders() "
                f"{format_outcome(repeat['fetch_open_orders_all'])}",
                enabled=progress,
            )
            if include_my_trades:
                emit_progress(
                    f"{repeat_label} fetch_my_trades(first symbol) "
                    f"pages={max(1, int(fill_history_pages))}",
                    enabled=progress,
                )
                repeat["fetch_my_trades_first_symbol"] = await _fetch_my_trades_sample(
                    exchange,
                    resolved_symbols[0],
                    pages=max(1, int(fill_history_pages)),
                    page_limit=max(1, int(fill_history_page_limit)),
                )
                emit_progress(
                    f"{repeat_label} fetch_my_trades(first symbol) "
                    f"{format_outcome(repeat['fetch_my_trades_first_symbol'])}",
                    enabled=progress,
                )
        out["repeats"].append(repeat)
        emit_progress(f"{repeat_label} complete", enabled=progress)
        if idx + 1 < max(1, int(repeats)) and sleep_between_seconds > 0:
            emit_progress(
                f"{repeat_label} sleeping {float(sleep_between_seconds):.1f}s before next repeat",
                enabled=progress,
            )
            await asyncio.sleep(float(sleep_between_seconds))

    out["account_critical_health"] = summarize_account_critical_probe_health(out)
    out["time_sync_health"] = summarize_time_sync_probe_health(out)
    out["candle_freshness_health"] = summarize_candle_freshness_probe_health(out)
    out["fill_history_health"] = summarize_fill_history_probe_health(out)
    out["rate_limit_health"] = summarize_rate_limit_probe_health(out)
    return out


async def probe_user_ticker_endpoints(
    user: str,
    *,
    api_keys_path: str,
    symbols: list[str],
    coins: list[str],
    quote: str | None,
    max_symbols: int,
    repeats: int,
    sleep_between_seconds: float,
    include_private: bool = True,
    include_public: bool = True,
    include_order_book: bool = True,
    include_ohlcv: bool = True,
    include_my_trades: bool = True,
    include_time_sync: bool = True,
    fill_history_pages: int = 1,
    fill_history_page_limit: int = 10,
    progress: bool = False,
) -> dict[str, Any]:
    emit_progress(f"[{user}] loading api key config", enabled=progress)
    user_info = load_user_info(user, api_keys_path=api_keys_path)
    exchange_id = str(user_info.get("exchange") or "").lower()
    if not exchange_id:
        raise ValueError(f"user {user!r}: missing exchange in api keys")
    emit_progress(f"[{user}] creating exchange session | exchange={exchange_id}", enabled=progress)
    exchange = create_exchange(exchange_id, user_info)
    try:
        return await probe_exchange_ticker_endpoints(
            exchange,
            user=user,
            user_info=user_info,
            symbols=symbols,
            coins=coins,
            quote=quote,
            max_symbols=max_symbols,
            repeats=repeats,
            sleep_between_seconds=sleep_between_seconds,
            include_private=include_private,
            include_public=include_public,
            include_order_book=include_order_book,
            include_ohlcv=include_ohlcv,
            include_my_trades=include_my_trades,
            include_time_sync=include_time_sync,
            fill_history_pages=fill_history_pages,
            fill_history_page_limit=fill_history_page_limit,
            progress=progress,
        )
    finally:
        emit_progress(f"[{user}] closing exchange session", enabled=progress)
        await exchange.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only multi-user CCXT ticker endpoint latency probe."
    )
    parser.add_argument("--users", required=True, help="comma-separated user keys in api-keys.json")
    parser.add_argument("--api-keys", default="api-keys.json", help="path to api-keys.json")
    parser.add_argument("--symbols", help="comma-separated CCXT symbols to probe for every user")
    parser.add_argument("--coins", help="comma-separated base coins to resolve to swap symbols")
    parser.add_argument("--quote", help="quote currency for --coins or default symbol selection")
    parser.add_argument("--max-symbols", type=int, default=5, help="default symbol sample size")
    parser.add_argument("--repeats", type=int, default=1, help="number of probe rounds per user")
    parser.add_argument(
        "--sleep-between-seconds",
        type=float,
        default=0.0,
        help="pause between repeated probe rounds for the same user",
    )
    parser.add_argument(
        "--public-only",
        action="store_true",
        help="skip authenticated read-only account-state endpoint probes",
    )
    parser.add_argument(
        "--account-only",
        action="store_true",
        help="probe only authenticated balance, positions, and open-orders endpoints",
    )
    parser.add_argument("--skip-order-book", action="store_true", help="skip order book probes")
    parser.add_argument("--skip-ohlcv", action="store_true", help="skip 1m OHLCV tail probes")
    parser.add_argument(
        "--skip-my-trades",
        action="store_true",
        help="skip authenticated fetch_my_trades probe",
    )
    parser.add_argument(
        "--fill-history-pages",
        type=int,
        default=1,
        help=(
            "bounded first-symbol fetch_my_trades page sample; default 1 preserves "
            "the low-impact single-call probe"
        ),
    )
    parser.add_argument(
        "--fill-history-page-limit",
        type=int,
        default=10,
        help="per-page fetch_my_trades limit for the bounded pagination sample",
    )
    parser.add_argument(
        "--skip-time-sync",
        action="store_true",
        help="skip read-only fetch_time clock-skew probe",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    parser.add_argument("--out", help="output JSON path; default is tmp/ccxt_ticker_probe_<ts>.json")
    parser.add_argument("--json", action="store_true", help="also print JSON to stdout")
    return parser


async def async_main() -> int:
    args = build_parser().parse_args()
    if args.public_only and args.account_only:
        raise ValueError("--account-only cannot be combined with --public-only")
    users = split_csv(args.users)
    if not users:
        raise ValueError("provide at least one user via --users")
    symbols = split_csv(args.symbols)
    coins = split_csv(args.coins)
    started_ms = int(utc_ms())
    result: dict[str, Any] = {
        "generated_at_ms": started_ms,
        "generated_at": ts_to_date(started_ms),
        "api_keys_path": args.api_keys,
        "users_requested": users,
        "symbols_requested": symbols,
        "coins_requested": coins,
        "quote": args.quote,
        "max_symbols": int(args.max_symbols),
        "repeats": int(args.repeats),
        "sleep_between_seconds": float(args.sleep_between_seconds),
        "account_only": bool(args.account_only),
        "include_private": not bool(args.public_only),
        "include_public": not bool(args.account_only),
        "include_order_book": not bool(args.skip_order_book) and not bool(args.account_only),
        "include_ohlcv": not bool(args.skip_ohlcv) and not bool(args.account_only),
        "include_my_trades": not bool(args.skip_my_trades) and not bool(args.account_only),
        "include_time_sync": not bool(args.skip_time_sync),
        "fill_history_pages": max(1, int(args.fill_history_pages)),
        "fill_history_page_limit": max(1, int(args.fill_history_page_limit)),
        "probes": [],
    }
    for user in users:
        emit_progress(f"[{user}] probe start", enabled=not bool(args.quiet))
        result["probes"].append(
            await probe_user_ticker_endpoints(
                user,
                api_keys_path=args.api_keys,
                symbols=symbols,
                coins=coins,
                quote=args.quote,
                max_symbols=int(args.max_symbols),
                repeats=int(args.repeats),
                sleep_between_seconds=float(args.sleep_between_seconds),
                include_private=not bool(args.public_only),
                include_public=not bool(args.account_only),
                include_order_book=not bool(args.skip_order_book) and not bool(args.account_only),
                include_ohlcv=not bool(args.skip_ohlcv) and not bool(args.account_only),
                include_my_trades=not bool(args.skip_my_trades) and not bool(args.account_only),
                include_time_sync=not bool(args.skip_time_sync),
                fill_history_pages=max(1, int(args.fill_history_pages)),
                fill_history_page_limit=max(1, int(args.fill_history_page_limit)),
                progress=not bool(args.quiet),
            )
        )
        emit_progress(f"[{user}] probe complete", enabled=not bool(args.quiet))

    result["account_critical_health"] = summarize_account_critical_probe_collection(
        result["probes"]
    )
    result["time_sync_health"] = summarize_time_sync_probe_collection(result["probes"])
    result["candle_freshness_health"] = summarize_candle_freshness_probe_collection(
        result["probes"]
    )
    result["fill_history_health"] = summarize_fill_history_probe_collection(
        result["probes"]
    )
    result["rate_limit_health"] = summarize_rate_limit_probe_collection(
        result["probes"]
    )
    out_path = Path(args.out) if args.out else Path("tmp") / f"ccxt_ticker_probe_{started_ms}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {out_path}")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def main() -> int:
    return asyncio.run(async_main())
