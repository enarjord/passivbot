from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Optional

from utils import utc_ms


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    bid: float
    ask: float
    last: float
    fetched_ms: int
    source: str
    exchange_timestamp_ms: Optional[int] = None

    def is_valid(self) -> bool:
        return (
            math.isfinite(float(self.bid))
            and math.isfinite(float(self.ask))
            and math.isfinite(float(self.last))
            and float(self.bid) > 0.0
            and float(self.ask) > 0.0
            and float(self.last) > 0.0
        )


TickerFetcher = Callable[[], Awaitable[dict[str, Any]]]
SymbolTickerFetcher = Callable[[list[str]], Awaitable[dict[str, Any]]]
CacheSink = Callable[[str, float, int], None]


class MarketSnapshotProvider:
    """Fetch and cache live bid/ask/last snapshots independently of candle state."""

    def __init__(
        self,
        *,
        exchange_name: str,
        fetch_tickers: Optional[TickerFetcher],
        fetch_tickers_for_symbols: Optional[SymbolTickerFetcher] = None,
        ticker_strategy: str = "bulk",
        cache_sink: Optional[CacheSink] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.exchange_name = str(exchange_name or "").lower()
        self._fetch_tickers = fetch_tickers
        self._fetch_tickers_for_symbols = fetch_tickers_for_symbols
        self._ticker_strategy = str(ticker_strategy or "bulk").lower()
        if self._ticker_strategy not in {"bulk", "symbols"}:
            self._ticker_strategy = "bulk"
        self._cache_sink = cache_sink
        self._log = logger or logging.getLogger("passivbot.market_snapshot")
        self._cache: dict[str, MarketSnapshot] = {}
        self._fetch_task: Optional[asyncio.Task] = None
        self._symbol_fetch_tasks: dict[tuple[str, ...], asyncio.Task] = {}

    def get_cached(self, symbol: str, *, now_ms: int, max_age_ms: int) -> Optional[MarketSnapshot]:
        snap = self._cache.get(symbol)
        if snap is None:
            return None
        if int(now_ms) - int(snap.fetched_ms) > int(max_age_ms):
            return None
        return snap if snap.is_valid() else None

    async def get_snapshots(
        self, symbols: Iterable[str], *, max_age_ms: int = 10_000
    ) -> dict[str, MarketSnapshot]:
        ordered_symbols = list(dict.fromkeys(str(s) for s in symbols if s))
        if not ordered_symbols:
            return {}

        now = utc_ms()
        out: dict[str, MarketSnapshot] = {}
        for symbol in ordered_symbols:
            snap = self.get_cached(symbol, now_ms=now, max_age_ms=max_age_ms)
            if snap is not None:
                out[symbol] = snap

        missing = [s for s in ordered_symbols if s not in out]
        if not missing or (
            self._fetch_tickers is None and self._fetch_tickers_for_symbols is None
        ):
            return out

        try:
            fetched, source = await self._fetch_tickers_for_missing(missing)
        except Exception as exc:
            self._log.debug(
                "[market] ticker snapshot fetch failed | exchange=%s symbols=%s error_type=%s error=%s",
                self.exchange_name,
                len(missing),
                type(exc).__name__,
                exc,
            )
            return out

        fetched_ms = utc_ms()
        if not isinstance(fetched, dict):
            self._log.debug(
                "[market] ticker snapshot fetch returned non-dict | exchange=%s type=%s",
                self.exchange_name,
                type(fetched).__name__,
            )
            return out

        cached = 0
        for raw_symbol, ticker in fetched.items():
            symbol = str(raw_symbol)
            snap = self._snapshot_from_ticker(symbol, ticker, fetched_ms=fetched_ms, source=source)
            if snap is None:
                continue
            self._cache[symbol] = snap
            cached += 1
            if self._cache_sink is not None:
                try:
                    self._cache_sink(symbol, float(snap.last), int(snap.fetched_ms))
                except Exception as exc:
                    self._log.debug(
                        "[market] snapshot cache sink failed | symbol=%s error_type=%s error=%s",
                        symbol,
                        type(exc).__name__,
                        exc,
                    )

        hits = 0
        for symbol in missing:
            snap = self.get_cached(symbol, now_ms=fetched_ms, max_age_ms=max_age_ms)
            if snap is None:
                continue
            out[symbol] = snap
            hits += 1

        self._log.debug(
            "[market] ticker snapshots ready | exchange=%s requested=%s hits=%s misses=%s cached=%s source=%s",
            self.exchange_name,
            len(ordered_symbols),
            hits,
            max(0, len(missing) - hits),
            cached,
            source,
        )
        return out

    async def _fetch_tickers_for_missing(self, missing: list[str]) -> tuple[dict[str, Any], str]:
        if self._ticker_strategy == "symbols" and self._fetch_tickers_for_symbols is not None:
            fetched = await self._fetch_tickers_for_symbols_shared(missing)
            return fetched, "fetch_tickers_symbols"
        if self._fetch_tickers is not None:
            fetched = await self._fetch_tickers_shared()
            return fetched, "fetch_tickers"
        fetched = await self._fetch_tickers_for_symbols_shared(missing)
        return fetched, "fetch_tickers_symbols"

    async def _fetch_tickers_shared(self) -> dict[str, Any]:
        if self._fetch_tickers is None:
            return {}
        task = self._fetch_task
        if task is None or task.done():
            task = asyncio.create_task(self._fetch_tickers())
            self._fetch_task = task
        try:
            return await task
        finally:
            if self._fetch_task is task and task.done():
                self._fetch_task = None

    async def _fetch_tickers_for_symbols_shared(self, symbols: list[str]) -> dict[str, Any]:
        if self._fetch_tickers_for_symbols is None:
            return {}
        key = tuple(dict.fromkeys(str(symbol) for symbol in symbols if symbol))
        task = self._symbol_fetch_tasks.get(key)
        if task is None or task.done():
            task = asyncio.create_task(self._fetch_tickers_for_symbols(list(key)))
            self._symbol_fetch_tasks[key] = task
        try:
            return await task
        finally:
            if self._symbol_fetch_tasks.get(key) is task and task.done():
                self._symbol_fetch_tasks.pop(key, None)

    @staticmethod
    def _coerce_positive(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(out) or out <= 0.0:
            return None
        return out

    def _snapshot_from_ticker(
        self, symbol: str, ticker: Any, *, fetched_ms: int, source: str = "fetch_tickers"
    ) -> Optional[MarketSnapshot]:
        if not isinstance(ticker, dict):
            return None
        last = self._coerce_positive(ticker.get("last"))
        if last is None:
            last = self._coerce_positive(ticker.get("close"))
        bid = self._coerce_positive(ticker.get("bid"))
        ask = self._coerce_positive(ticker.get("ask"))
        if last is None:
            last = bid or ask
        if bid is None:
            bid = last
        if ask is None:
            ask = last
        if last is None or bid is None or ask is None:
            return None
        exchange_ts = None
        try:
            raw_ts = ticker.get("timestamp")
            if raw_ts is not None:
                exchange_ts = int(raw_ts)
        except Exception:
            exchange_ts = None
        snap = MarketSnapshot(
            symbol=symbol,
            bid=float(bid),
            ask=float(ask),
            last=float(last),
            fetched_ms=int(fetched_ms),
            source=str(source),
            exchange_timestamp_ms=exchange_ts,
        )
        return snap if snap.is_valid() else None
