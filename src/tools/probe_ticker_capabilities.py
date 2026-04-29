from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

import ccxt.async_support as ccxt_async

from procedures import load_user_info


PRICE_FIELDS = ("last", "bid", "ask")


def split_csv(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def build_ccxt_config(user_info: dict[str, Any]) -> dict[str, Any]:
    passivbot_fields = {"exchange", "options", "quote", "is_vault"}
    config = {k: v for k, v in user_info.items() if k not in passivbot_fields and v != ""}
    config["enableRateLimit"] = True
    config.setdefault("timeout", 30_000)
    legacy_mappings = {
        "key": "apiKey",
        "api_key": "apiKey",
        "wallet": "walletAddress",
        "private_key": "privateKey",
        "passphrase": "password",
        "wallet_address": "walletAddress",
    }
    for old_name, new_name in legacy_mappings.items():
        if old_name in config and new_name not in config:
            config[new_name] = config.pop(old_name)
    return config


def create_exchange(exchange_id: str, user_info: dict[str, Any] | None = None):
    exchange_class = getattr(ccxt_async, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"exchange {exchange_id!r} not found in ccxt.async_support")
    user_info = user_info or {}
    options = dict(user_info.get("options", {}) if isinstance(user_info.get("options"), dict) else {})
    options["defaultType"] = "swap"
    config = build_ccxt_config(user_info)
    config["options"] = dict(options)
    session = exchange_class(config)
    session.options.update(options)
    session.options["defaultType"] = "swap"
    if exchange_id == "hyperliquid":
        session.options.setdefault(
            "fetchMarkets",
            {"types": ["swap", "hip3"], "hip3": {"dex": ["xyz"]}},
        )
    return session


def is_active_linear_swap(market: Any) -> bool:
    if not isinstance(market, dict):
        return False
    return bool(
        market.get("active") is not False
        and market.get("swap") is True
        and market.get("linear") is True
    )


def summarize_ticker(ticker: Any) -> dict[str, Any]:
    if not isinstance(ticker, dict):
        return {"type": type(ticker).__name__, "valid": False}
    out = {
        "symbol": ticker.get("symbol"),
        "timestamp": ticker.get("timestamp"),
        "datetime": ticker.get("datetime"),
        "has_last": ticker.get("last") is not None,
        "has_bid": ticker.get("bid") is not None,
        "has_ask": ticker.get("ask") is not None,
        "last": ticker.get("last"),
        "bid": ticker.get("bid"),
        "ask": ticker.get("ask"),
        "top_level_keys": sorted(str(k) for k in ticker.keys()),
    }
    info = ticker.get("info")
    out["info_keys"] = sorted(str(k) for k in info.keys()) if isinstance(info, dict) else []
    return out


def summarize_tickers(tickers: Any, requested_symbols: list[str]) -> dict[str, Any]:
    if not isinstance(tickers, dict):
        return {"type": type(tickers).__name__, "count": None, "hits": {}, "samples": {}}
    hits = {symbol: symbol in tickers for symbol in requested_symbols}
    samples = {}
    for symbol in requested_symbols:
        if symbol in tickers:
            samples[symbol] = summarize_ticker(tickers[symbol])
    return {
        "count": len(tickers),
        "hits": hits,
        "samples": samples,
    }


def summarize_order_book(order_book: Any) -> dict[str, Any]:
    if not isinstance(order_book, dict):
        return {"type": type(order_book).__name__, "valid": False}
    bids = order_book.get("bids") or []
    asks = order_book.get("asks") or []
    best_bid = bids[0] if bids else None
    best_ask = asks[0] if asks else None
    return {
        "timestamp": order_book.get("timestamp"),
        "datetime": order_book.get("datetime"),
        "has_bid": bool(best_bid),
        "has_ask": bool(best_ask),
        "bid": best_bid[0] if best_bid else None,
        "bid_qty": best_bid[1] if best_bid and len(best_bid) > 1 else None,
        "ask": best_ask[0] if best_ask else None,
        "ask_qty": best_ask[1] if best_ask and len(best_ask) > 1 else None,
    }


async def timed_call(coro) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        value = await coro
    except Exception as exc:
        return {
            "ok": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "ok": True,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "value": value,
    }


async def resolve_symbols(exchange, coins: list[str], quote: str | None = None) -> list[str]:
    if not coins:
        return []
    markets = await exchange.load_markets()
    quote_upper = str(quote or "").upper()
    resolved = []
    for coin in coins:
        coin_upper = coin.upper()
        candidates = []
        for symbol, market in markets.items():
            if str(market.get("base") or "").upper() != coin_upper:
                continue
            if quote_upper and str(market.get("quote") or "").upper() != quote_upper:
                continue
            if not is_active_linear_swap(market):
                continue
            candidates.append((symbol, market))
        if not candidates:
            raise ValueError(f"could not resolve active linear swap symbol for coin {coin!r} quote={quote!r}")
        candidates.sort(key=lambda item: item[0])
        resolved.append(candidates[0][0])
    return resolved


async def probe_ticker_capabilities(
    exchange,
    symbols: list[str],
    *,
    probe_all: bool,
    probe_order_book: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exchange": getattr(exchange, "id", type(exchange).__name__),
        "symbols": symbols,
        "has": {
            "fetchTicker": bool(getattr(exchange, "has", {}).get("fetchTicker")),
            "fetchTickers": bool(getattr(exchange, "has", {}).get("fetchTickers")),
            "fetchOrderBook": bool(getattr(exchange, "has", {}).get("fetchOrderBook")),
        },
        "fetch_ticker": {},
        "fetch_tickers_symbols": None,
        "fetch_tickers_all": None,
        "fetch_order_book": {},
    }
    for symbol in symbols:
        outcome = await timed_call(exchange.fetch_ticker(symbol))
        if outcome["ok"]:
            outcome["value"] = summarize_ticker(outcome["value"])
        result["fetch_ticker"][symbol] = outcome

    if symbols:
        outcome = await timed_call(exchange.fetch_tickers(symbols))
        if outcome["ok"]:
            outcome["value"] = summarize_tickers(outcome["value"], symbols)
        result["fetch_tickers_symbols"] = outcome

    if probe_all:
        outcome = await timed_call(exchange.fetch_tickers())
        if outcome["ok"]:
            outcome["value"] = summarize_tickers(outcome["value"], symbols)
        result["fetch_tickers_all"] = outcome

    if probe_order_book:
        for symbol in symbols:
            outcome = await timed_call(exchange.fetch_order_book(symbol, limit=5))
            if outcome["ok"]:
                outcome["value"] = summarize_order_book(outcome["value"])
            result["fetch_order_book"][symbol] = outcome

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only probe for exchange ticker and top-of-book capabilities."
    )
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--user", help="user key in api-keys.json")
    identity.add_argument("--exchange", help="CCXT exchange id for public unauthenticated probing")
    parser.add_argument("--api-keys", default="api-keys.json", help="path to api-keys.json")
    parser.add_argument("--symbols", help="comma-separated CCXT symbols to probe")
    parser.add_argument("--coins", help="comma-separated base coins to resolve to swap symbols")
    parser.add_argument("--quote", help="quote currency for --coins symbol resolution")
    parser.add_argument(
        "--all",
        action="store_true",
        help="also probe fetch_tickers() without a symbol list",
    )
    parser.add_argument(
        "--order-book",
        action="store_true",
        help="also probe fetch_order_book(symbol, limit=5)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    return parser


async def async_main() -> int:
    args = build_parser().parse_args()
    user_info = load_user_info(args.user, api_keys_path=args.api_keys) if args.user else {}
    exchange_id = str(user_info.get("exchange") or args.exchange or "").lower()
    if not exchange_id:
        raise ValueError("missing exchange id")
    exchange = create_exchange(exchange_id, user_info)
    try:
        symbols = split_csv(args.symbols)
        coins = split_csv(args.coins)
        if coins:
            symbols.extend(await resolve_symbols(exchange, coins, quote=args.quote or user_info.get("quote")))
        symbols = list(dict.fromkeys(symbols))
        if not symbols and not args.all:
            raise ValueError("provide --symbols, --coins, or --all")
        result = await probe_ticker_capabilities(
            exchange,
            symbols,
            probe_all=bool(args.all),
            probe_order_book=bool(args.order_book),
        )
    finally:
        await exchange.close()

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    else:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
