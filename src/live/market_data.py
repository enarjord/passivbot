from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from typing import Iterable

from config.access import get_optional_live_value
from live.market_snapshot import MarketSnapshot
from utils import utc_ms


def _utc_ms() -> int:
    passivbot_module = sys.modules.get("passivbot")
    time_fn = getattr(passivbot_module, "utc_ms", None)
    if callable(time_fn):
        return int(time_fn())
    return int(utc_ms())


def market_snapshot_ticker_strategy(bot) -> str:
    """Choose the cheapest safe ticker endpoint shape for market snapshots."""
    explicit = get_optional_live_value(
        bot.config, "market_snapshot_ticker_strategy", None
    )
    if explicit is not None:
        mode = str(explicit).lower()
        if mode == "auto":
            explicit = None
        elif mode in {"bulk", "symbols"}:
            return mode
        else:
            logging.warning(
                "[market] invalid live.market_snapshot_ticker_strategy=%r; using exchange default",
                explicit,
            )
    if str(getattr(bot, "exchange", "") or "").lower() == "bitget":
        return "symbols"
    if str(getattr(bot, "exchange", "") or "").lower() == "hyperliquid":
        return "symbols"
    if str(getattr(bot, "exchange", "") or "").lower() == "kucoin":
        return "symbols"
    return "bulk"


async def filter_fresh_market_snapshot_creations(
    bot, orders: list[dict]
) -> list[dict]:
    """Block staged order creations unless live market snapshots are still fresh."""
    if not orders:
        return orders
    symbols = sorted({str(order["symbol"]) for order in orders if order.get("symbol")})
    if not symbols:
        return orders
    planning_snapshot_invalid = bot._current_planning_snapshot_invalid_for_creations(
        symbols
    )
    if planning_snapshot_invalid:
        refreshable_reasons = {
            ("market_snapshot", "snapshot_too_old"),
        }
        if not all(
            isinstance(item, dict)
            and (str(item.get("surface")), str(item.get("reason")))
            in refreshable_reasons
            for item in planning_snapshot_invalid
        ):
            logging.warning(
                "[market] skipping order creation; planning snapshot invalid before create | symbols=%s details=%s",
                bot._log_symbols(symbols, limit=12),
                bot._log_compact_symbol_payload(planning_snapshot_invalid[:8]),
            )
            return []
        logging.info(
            "[market] refreshing stale planning market snapshot before create | symbols=%s stale=%s",
            bot._log_symbols(symbols, limit=12),
            len(planning_snapshot_invalid),
        )
    try:
        snapshots = await bot._get_live_market_snapshots(
            symbols,
            max_age_ms=bot._live_market_snapshot_max_age_ms(),
            context="pre_create",
            allow_completed_candle_fallback=False,
        )
        bot._record_market_snapshot_surface(symbols, snapshots)
        invalid = bot._market_snapshot_signature_invalid(symbols)
    except Exception as exc:
        logging.warning(
            "[market] skipping order creation; failed pre-create market snapshot refresh | symbols=%s error_type=%s error=%s",
            bot._log_symbols(symbols, limit=12),
            type(exc).__name__,
            exc,
        )
        return []
    if invalid:
        logging.warning(
            "[market] skipping order creation; stale pre-create market snapshots | symbols=%s details=%s",
            bot._log_symbols(symbols, limit=12),
            bot._log_compact_symbol_payload(invalid[:8]),
        )
        return []
    return orders


async def get_live_market_snapshots(
    bot,
    symbols: Iterable[str],
    *,
    max_age_ms: int = 10_000,
    context: str = "live",
    allow_completed_candle_fallback: bool = False,
) -> dict[str, MarketSnapshot]:
    """Return live market snapshots without using in-progress candles."""
    ordered_symbols = list(dict.fromkeys(str(s) for s in symbols if s))
    if not ordered_symbols:
        return {}
    provider = getattr(bot, "market_snapshot_provider", None)
    snapshots: dict[str, MarketSnapshot] = {}
    if provider is not None:
        try:
            snapshots = await provider.get_snapshots(
                ordered_symbols, max_age_ms=max_age_ms
            )
        except RuntimeError as exc:
            if str(getattr(bot, "exchange", "") or "").lower() != "hyperliquid":
                raise
            logging.debug(
                "[market] hyperliquid primary ticker snapshot path failed; trying explicit fallback | context=%s symbols=%s error=%s",
                context,
                len(ordered_symbols),
                exc,
            )
            snapshots = {}

    missing = [
        symbol
        for symbol in ordered_symbols
        if symbol not in snapshots or not snapshots[symbol].is_valid()
    ]
    if missing and str(getattr(bot, "exchange", "") or "").lower() == "hyperliquid":
        try:
            fetched = await bot.cca.fetch(
                bot._hl_info_url(),
                method="POST",
                headers={"Content-Type": "application/json"},
                body=json.dumps({"type": "allMids"}),
            )
            coin_to_sym = {v: k for k, v in bot.symbol_ids.items()} if bot.symbol_ids else {}
            fetched_ms = _utc_ms()
            for coin, mid_str in fetched.items():
                sym = coin_to_sym.get(coin)
                if sym not in missing:
                    continue
                price = float(mid_str)
                snap = MarketSnapshot(
                    symbol=sym,
                    bid=price,
                    ask=price,
                    last=price,
                    fetched_ms=fetched_ms,
                    source="hyperliquid_all_mids",
                )
                if snap.is_valid():
                    snapshots[sym] = snap
        except Exception as exc:
            logging.debug(
                "[market] hyperliquid allMids snapshot failed | context=%s symbols=%s error_type=%s error=%s",
                context,
                len(missing),
                type(exc).__name__,
                exc,
            )
        missing = [
            symbol
            for symbol in ordered_symbols
            if symbol not in snapshots or not snapshots[symbol].is_valid()
        ]
        if missing and hasattr(bot, "fetch_tickers_for_symbols"):
            try:
                fetched_symbol_tickers = await bot.fetch_tickers_for_symbols(missing)
                fetched_ms = _utc_ms()
                for symbol, ticker in fetched_symbol_tickers.items():
                    if symbol not in missing or not isinstance(ticker, dict):
                        continue
                    try:
                        last = float(ticker.get("last") or ticker.get("close"))
                        bid = float(ticker.get("bid"))
                        ask = float(ticker.get("ask"))
                    except (TypeError, ValueError):
                        continue
                    source = ticker.get("source")
                    if not (last > 0.0 and bid > 0.0 and ask > 0.0):
                        continue
                    snap = MarketSnapshot(
                        symbol=symbol,
                        bid=bid,
                        ask=ask,
                        last=last,
                        fetched_ms=fetched_ms,
                        source=str(source or "hyperliquid_symbol_tickers"),
                    )
                    if snap.is_valid():
                        snapshots[symbol] = snap
            except Exception as exc:
                logging.debug(
                    "[market] hyperliquid symbol ticker snapshot failed | context=%s symbols=%s error_type=%s error=%s",
                    context,
                    len(missing),
                    type(exc).__name__,
                    exc,
                )
            missing = [
                symbol
                for symbol in ordered_symbols
                if symbol not in snapshots or not snapshots[symbol].is_valid()
            ]

    if missing and allow_completed_candle_fallback:
        completed_prices = await bot.cm.get_last_prices(missing, max_age_ms=max_age_ms)
        now_ms = _utc_ms()
        for symbol in missing:
            raw = completed_prices.get(symbol)
            try:
                price = float(raw)
            except (TypeError, ValueError):
                continue
            snap = MarketSnapshot(
                symbol=symbol,
                bid=price,
                ask=price,
                last=price,
                fetched_ms=now_ms,
                source="completed_candle_fallback",
            )
            if snap.is_valid():
                snapshots[symbol] = snap

    missing = [
        symbol
        for symbol in ordered_symbols
        if symbol not in snapshots or not snapshots[symbol].is_valid()
    ]
    if missing:
        raise RuntimeError(
            f"missing live market snapshots for {context}: {bot._log_symbols(missing, limit=12)}"
        )
    return {symbol: snapshots[symbol] for symbol in ordered_symbols}


async def get_orchestrator_market_snapshots(
    bot, symbols: list[str]
) -> dict[str, MarketSnapshot]:
    """Return current bid/ask/last snapshots for orchestrator planning."""
    fetch_ttl_ms = bot._live_market_snapshot_fetch_max_age_ms()
    provider = getattr(bot, "market_snapshot_provider", None)
    snapshots: dict[str, MarketSnapshot] = {}
    if provider is not None:
        logging.debug(
            "[state] staged orchestrator requesting market snapshots | symbols=%s | fetch_ttl=%sms",
            len(symbols),
            fetch_ttl_ms,
        )
        try:
            snapshots = await provider.get_snapshots(symbols, max_age_ms=fetch_ttl_ms)
        except RuntimeError as exc:
            if str(getattr(bot, "exchange", "") or "").lower() != "hyperliquid":
                raise
            logging.debug(
                "[state] staged hyperliquid primary market snapshots failed; trying explicit fallback | symbols=%s error=%s",
                len(symbols),
                exc,
            )
            snapshots = {}
    invalid = [
        symbol
        for symbol in symbols
        if symbol not in snapshots or not snapshots[symbol].is_valid()
    ]
    if invalid:
        suffix = (
            " | attempting hyperliquid fallback"
            if str(getattr(bot, "exchange", "") or "").lower() == "hyperliquid"
            else ""
        )
        logging.debug(
            "[state] staged bulk market snapshots incomplete | symbols=%s | missing=%s%s",
            len(symbols),
            bot._log_symbols(invalid, limit=12),
            suffix,
        )
        if str(getattr(bot, "exchange", "") or "").lower() == "hyperliquid":
            try:
                snapshots = await bot._get_live_market_snapshots(
                    symbols,
                    max_age_ms=fetch_ttl_ms,
                    context="orchestrator",
                    allow_completed_candle_fallback=False,
                )
                bot._record_market_snapshot_surface(symbols, snapshots)
                sources = Counter(snap.source for snap in snapshots.values())
                logging.debug(
                    "[state] staged market snapshots ready | symbols=%s | ok=%s | invalid=0 | sources=%s",
                    len(symbols),
                    len(symbols),
                    ",".join(f"{k}:{v}" for k, v in sorted(sources.items())),
                )
                return snapshots
            except RuntimeError as exc:
                raise RuntimeError(
                    "staged market snapshots incomplete after hyperliquid fallback "
                    f"| missing={bot._log_symbols(invalid, limit=12)} "
                    f"| fallback_error={type(exc).__name__}: {exc}"
                ) from exc
        raise RuntimeError(
            "staged market snapshots incomplete "
            f"| exchange={getattr(bot, 'exchange', '')} "
            f"| symbols={len(symbols)} "
            f"| missing={bot._log_symbols(invalid, limit=12)}"
        )
    sources = Counter(snap.source for snap in snapshots.values())
    logging.debug(
        "[state] staged market snapshots ready | symbols=%s | ok=%s | invalid=0 | sources=%s",
        len(symbols),
        len(symbols),
        ",".join(f"{k}:{v}" for k, v in sorted(sources.items())),
    )
    bot._record_market_snapshot_surface(symbols, snapshots)
    return snapshots


def live_market_snapshot_max_age_ms(bot) -> int:
    del bot
    return 10_000


def live_market_snapshot_fetch_max_age_ms(bot) -> int:
    """Use a stricter fetch TTL than the hard safety TTL to leave planning headroom."""
    max_age_ms = int(bot._live_market_snapshot_max_age_ms())
    return max(1_000, min(max_age_ms, int(max_age_ms * 0.5)))


def market_snapshot_signature(
    bot, symbols: Iterable[str], snapshots: dict[str, MarketSnapshot]
) -> tuple:
    del bot
    expected = tuple(sorted(dict.fromkeys(str(symbol) for symbol in symbols if symbol)))
    return tuple(
        sorted(
            (
                symbol,
                round(float(snapshots[symbol].bid), 12),
                round(float(snapshots[symbol].ask), 12),
                round(float(snapshots[symbol].last), 12),
                int(snapshots[symbol].fetched_ms),
                snapshots[symbol].source,
            )
            for symbol in expected
            if symbol in snapshots and snapshots[symbol].is_valid()
        )
    )


def record_market_snapshot_surface(
    bot, symbols: Iterable[str], snapshots: dict[str, MarketSnapshot]
) -> None:
    bot._ensure_freshness_ledger().stamp(
        "market_snapshot",
        bot._market_snapshot_signature(symbols, snapshots),
        now_ms=_utc_ms(),
        epoch=int(getattr(bot, "_authoritative_refresh_epoch", 0) or 0),
    )


def market_snapshot_signature_invalid(bot, symbols: Iterable[str]) -> list[dict]:
    expected = tuple(sorted(dict.fromkeys(str(symbol) for symbol in symbols if symbol)))
    if not expected:
        return []
    ledger = bot._ensure_freshness_ledger()
    signature = ledger.surface_signature("market_snapshot")
    if not isinstance(signature, tuple):
        return [{"reason": "missing_signature", "symbols": list(expected)}]
    by_symbol = {}
    for item in signature:
        if not isinstance(item, (list, tuple)) or len(item) < 6:
            continue
        by_symbol[str(item[0])] = item
    now = _utc_ms()
    max_age_ms = bot._live_market_snapshot_max_age_ms()
    invalid = []
    for symbol in expected:
        item = by_symbol.get(symbol)
        if item is None:
            invalid.append({"symbol": symbol, "reason": "missing"})
            continue
        fetched_ms = int(item[4])
        age_ms = int(now - fetched_ms)
        if age_ms > max_age_ms:
            invalid.append(
                {
                    "symbol": symbol,
                    "reason": "stale",
                    "age_ms": age_ms,
                    "max_age_ms": max_age_ms,
                }
            )
    return invalid
