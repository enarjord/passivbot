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
    current_minute_open = int(utc_ms() // 60_000 * 60_000)
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
    return {
        "count": len(ohlcvs),
        "last_timestamp": last_ts,
        "last_datetime": ts_to_date(last_ts) if last_ts is not None else None,
        "last_is_current_incomplete_minute": last_ts == current_minute_open,
        "sample_tail": rows,
    }


def summarize_collection(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"type": "dict", "count": len(value), "keys_sample": sorted(map(str, value.keys()))[:20]}
    if isinstance(value, list):
        return {"type": "list", "count": len(value)}
    return {"type": type(value).__name__, "count": None}


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


async def _timed_load_markets(exchange) -> tuple[dict[str, Any], dict[str, Any]]:
    outcome = await _timed_call(exchange.load_markets())
    markets = outcome["value"] if outcome["ok"] and isinstance(outcome["value"], dict) else {}
    if outcome["ok"]:
        outcome["value"] = {
            "count": len(markets) if isinstance(markets, dict) else None,
            "type": type(markets).__name__,
        }
    return outcome, markets


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
        "include_my_trades": bool(include_my_trades),
        "market_count": len(markets) if isinstance(markets, dict) else None,
        "has": {
            "fetchBalance": bool(getattr(exchange, "has", {}).get("fetchBalance")),
            "fetchBidsAsks": bool(getattr(exchange, "has", {}).get("fetchBidsAsks")),
            "fetchMyTrades": bool(getattr(exchange, "has", {}).get("fetchMyTrades")),
            "fetchOHLCV": bool(getattr(exchange, "has", {}).get("fetchOHLCV")),
            "fetchOpenOrders": bool(getattr(exchange, "has", {}).get("fetchOpenOrders")),
            "fetchOrderBook": bool(getattr(exchange, "has", {}).get("fetchOrderBook")),
            "fetchPositions": bool(getattr(exchange, "has", {}).get("fetchPositions")),
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
                emit_progress(f"{repeat_label} fetch_my_trades(first symbol)", enabled=progress)
                repeat["fetch_my_trades_first_symbol"] = await _timed_method(
                    exchange,
                    "fetch_my_trades",
                    resolved_symbols[0],
                    None,
                    10,
                    summary=summarize_collection,
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
                progress=not bool(args.quiet),
            )
        )
        emit_progress(f"[{user}] probe complete", enabled=not bool(args.quiet))

    result["account_critical_health"] = summarize_account_critical_probe_collection(
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
