"""Fill events management module.

Provides a reusable manager that keeps local cache of canonicalised fill events,
fetches fresh data from the exchange when requested, and exposes convenient query
APIs (PnL summaries, cumulative PnL, last fill timestamps, etc.).

Currently implements a Bitget fetcher; the design is extensible to other
exchanges.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ccxt.base.errors import RateLimitExceeded

try:
    from utils import ts_to_date  # type: ignore
except ImportError:  # pragma: no cover - fallback for package-relative execution
    from .utils import ts_to_date

from config_utils import format_config, load_config
from logging_setup import configure_logging
from procedures import load_user_info

logger = logging.getLogger(__name__)


def _format_ms(ts: Optional[int]) -> str:
    if ts is None:
        return "None"
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _day_key(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _merge_fee_lists(
    fees_a: Optional[Sequence], fees_b: Optional[Sequence]
) -> Optional[List[Dict[str, object]]]:
    def to_list(fees):
        if not fees:
            return []
        if isinstance(fees, dict):
            return [fees]
        return list(fees)

    merged: Dict[str, Dict[str, object]] = {}
    for entry in to_list(fees_a) + to_list(fees_b):
        if not isinstance(entry, dict):
            continue
        currency = str(entry.get("currency") or entry.get("code") or "")
        if currency not in merged:
            merged[currency] = dict(entry)
            try:
                merged[currency]["cost"] = float(entry.get("cost", 0.0))
            except Exception:
                merged[currency]["cost"] = 0.0
        else:
            try:
                merged[currency]["cost"] += float(entry.get("cost", 0.0))
            except Exception:
                pass
    if not merged:
        return None
    return [dict(value) for value in merged.values()]


def _coalesce_events(events: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Group events sharing timestamp/symbol/pb_type/side/position."""
    aggregated: Dict[Tuple, Dict[str, object]] = {}
    order: List[Tuple] = []
    for ev in events:
        key = (
            ev.get("timestamp"),
            ev.get("symbol"),
            ev.get("pb_order_type"),
            ev.get("side"),
            ev.get("position_side"),
        )
        if key not in aggregated:
            aggregated[key] = dict(ev)
            aggregated[key]["id"] = str(ev.get("id", ""))
            aggregated[key]["qty"] = float(ev.get("qty", 0.0))
            aggregated[key]["pnl"] = float(ev.get("pnl", 0.0))
            aggregated[key]["fees"] = _merge_fee_lists(ev.get("fees"), None)
            aggregated[key]["_price_numerator"] = float(ev.get("price", 0.0)) * float(
                ev.get("qty", 0.0)
            )
            order.append(key)
        else:
            agg = aggregated[key]
            agg["id"] = f"{agg['id']}+{ev.get('id', '')}".strip("+")
            agg["qty"] = float(agg.get("qty", 0.0)) + float(ev.get("qty", 0.0))
            agg["pnl"] = float(agg.get("pnl", 0.0)) + float(ev.get("pnl", 0.0))
            agg["fees"] = _merge_fee_lists(agg.get("fees"), ev.get("fees"))
            agg["_price_numerator"] = float(agg.get("_price_numerator", 0.0)) + float(
                ev.get("price", 0.0)
            ) * float(ev.get("qty", 0.0))
            if not agg.get("client_order_id") and ev.get("client_order_id"):
                agg["client_order_id"] = ev.get("client_order_id")
            if not agg.get("pb_order_type"):
                agg["pb_order_type"] = ev.get("pb_order_type")
    coalesced: List[Dict[str, object]] = []
    for key in order:
        agg = aggregated[key]
        qty = float(agg.get("qty", 0.0))
        price_numerator = float(agg.get("_price_numerator", 0.0))
        if qty > 0:
            agg["price"] = price_numerator / qty
        agg.pop("_price_numerator", None)
        fees = agg.get("fees")
        if isinstance(fees, list) and len(fees) == 1:
            agg["fees"] = fees[0]
        coalesced.append(agg)
    return coalesced


def _check_pagination_progress(
    previous: Optional[Tuple[Tuple[str, object], ...]],
    params: Dict[str, object],
    context: str,
) -> Optional[Tuple[Tuple[str, object], ...]]:
    params_key = tuple(sorted(params.items()))
    if previous == params_key:
        logger.warning(
            "%s: repeated params detected; aborting pagination (%s)",
            context,
            dict(params),
        )
        return None
    logger.debug("%s: fetching with params %s", context, dict(params))
    return params_key


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FillEvent:
    """Canonical representation of a single fill event."""

    id: str
    timestamp: int
    datetime: str
    symbol: str
    side: str
    qty: float
    price: float
    pnl: float
    fees: Optional[Sequence]
    pb_order_type: str
    position_side: str
    client_order_id: str

    @property
    def key(self) -> str:
        return self.id

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "datetime": self.datetime,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "price": self.price,
            "pnl": self.pnl,
            "fees": self.fees,
            "pb_order_type": self.pb_order_type,
            "position_side": self.position_side,
            "client_order_id": self.client_order_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "FillEvent":
        required = [
            "id",
            "timestamp",
            "symbol",
            "side",
            "qty",
            "price",
            "pnl",
            "pb_order_type",
            "position_side",
            "client_order_id",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Fill event missing required keys: {missing}")
        return cls(
            id=str(data["id"]),
            timestamp=int(data["timestamp"]),
            datetime=str(data.get("datetime") or ts_to_date(int(data["timestamp"]))),
            symbol=str(data["symbol"]),
            side=str(data["side"]).lower(),
            qty=float(data["qty"]),
            price=float(data["price"]),
            pnl=float(data["pnl"]),
            fees=data.get("fees"),
            pb_order_type=str(data["pb_order_type"]),
            position_side=str(data["position_side"]).lower(),
            client_order_id=str(data["client_order_id"]),
        )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class FillEventCache:
    """JSON cache storing fills split by UTC day."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self) -> List[FillEvent]:
        files = sorted(self.root.glob("*.json"))
        events: List[FillEvent] = []
        for path in files:
            try:
                with path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh) or []
            except Exception as exc:
                logger.warning("FillEventCache.load: failed to read %s (%s)", path, exc)
                continue
            for raw in payload:
                try:
                    events.append(FillEvent.from_dict(raw))
                except Exception:
                    logger.debug("FillEventCache.load: skipping malformed record in %s", path)
        events.sort(key=lambda ev: ev.timestamp)
        logger.info(
            "FillEventCache.load: loaded %d events from %d files in %s",
            len(events),
            len(files),
            self.root,
        )
        return events

    def save(self, events: Sequence[FillEvent]) -> None:
        day_map: Dict[str, List[FillEvent]] = defaultdict(list)
        for event in events:
            day_map[_day_key(event.timestamp)].append(event)
        for day in day_map:
            day_map[day].sort(key=lambda ev: ev.timestamp)
        self.save_days(day_map)

    def save_days(self, day_events: Dict[str, Sequence[FillEvent]]) -> None:
        for day, events in day_events.items():
            path = self.root / f"{day}.json"
            payload = [event.to_dict() for event in sorted(events, key=lambda ev: ev.timestamp)]
            if path.exists():
                try:
                    with path.open("r", encoding="utf-8") as fh:
                        current = json.load(fh)
                except Exception:
                    current = None
                if current == payload:
                    logger.debug("FillEventCache.save_days: %s unchanged", path.name)
                    continue
            tmp_path = path.with_suffix(".tmp")
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp_path, path)
            logger.info(
                "FillEventCache.save_days: wrote %d events to %s",
                len(payload),
                path,
            )


# ---------------------------------------------------------------------------
# Fetcher infrastructure
# ---------------------------------------------------------------------------


class BaseFetcher:
    """Abstract interface for exchange-specific fill fetchers."""

    async def fetch(
        self,
        since_ms: Optional[int],
        until_ms: Optional[int],
        detail_cache: Dict[str, Tuple[str, str]],
        on_batch: Optional[Callable[[List[Dict[str, object]]], None]] = None,
    ) -> List[Dict[str, object]]:
        raise NotImplementedError


class BitgetFetcher(BaseFetcher):
    """Fetches and enriches fill events from Bitget."""

    def __init__(
        self,
        api,
        *,
        product_type: str = "USDT-FUTURES",
        history_limit: int = 100,
        detail_calls_per_minute: int = 120,
        detail_concurrency: int = 10,
        now_func: Optional[Callable[[], int]] = None,
        symbol_resolver: Optional[Callable[[Optional[str]], str]] = None,
    ) -> None:
        self.api = api
        self.product_type = product_type
        self.history_limit = history_limit
        self.detail_calls_per_minute = max(1, detail_calls_per_minute)
        self._detail_call_timestamps: deque[int] = deque()
        self.detail_concurrency = max(1, detail_concurrency)
        self._rate_lock = asyncio.Lock()
        self._now_func = now_func or (lambda: int(datetime.now(tz=timezone.utc).timestamp() * 1000))
        if symbol_resolver is None:
            raise ValueError("BitgetFetcher requires a symbol_resolver callable")
        self._symbol_resolver = symbol_resolver

    async def fetch(
        self,
        since_ms: Optional[int],
        until_ms: Optional[int],
        detail_cache: Dict[str, Tuple[str, str]],
        on_batch: Optional[Callable[[List[Dict[str, object]]], None]] = None,
    ) -> List[Dict[str, object]]:
        buffer_step_ms = 24 * 60 * 60 * 1000
        end_time = int(until_ms) if until_ms is not None else self._now_func() + buffer_step_ms
        params: Dict[str, object] = {
            "productType": self.product_type,
            "limit": self.history_limit,
            "endTime": end_time,
        }
        events: Dict[str, Dict[str, object]] = {}

        detail_hits = 0
        detail_fetches = 0
        max_fetches = 400
        fetch_count = 0

        logger.info(
            "BitgetFetcher.fetch: start (since=%s, until=%s, limit=%d)",
            _format_ms(since_ms),
            _format_ms(until_ms),
            self.history_limit,
        )

        while True:
            if fetch_count >= max_fetches:
                logger.warning(
                    "BitgetFetcher.fetch: reached maximum pagination depth (%d)",
                    max_fetches,
                )
                break
            fetch_count += 1
            payload = await self.api.private_mix_get_v2_mix_order_fill_history(dict(params))
            fill_list = payload.get("data", {}).get("fillList") or []
            if not fill_list:
                if since_ms is None:
                    logger.debug("BitgetFetcher.fetch: empty batch without start bound; stopping")
                    break
                end_param = int(params.get("endTime", self._now_func()))
                if end_param <= since_ms:
                    logger.debug(
                        "BitgetFetcher.fetch: empty batch and cursor reached start; stopping"
                    )
                    break
                new_end_time = max(since_ms, end_param - buffer_step_ms)
                if new_end_time == end_param:
                    new_end_time = max(since_ms, end_param - 1)
                params["endTime"] = new_end_time
                logger.debug(
                    "BitgetFetcher.fetch: empty batch, continuing with endTime=%s",
                    _format_ms(params["endTime"]),
                )
                continue
            logger.debug(
                "BitgetFetcher.fetch: received batch size=%d endTime=%s",
                len(fill_list),
                params.get("endTime"),
            )
            batch_ids: List[str] = []
            pending_tasks: List[asyncio.Task[int]] = []
            for raw in fill_list:
                event = self._normalize_fill(raw)
                event_id = event["id"]
                if not event_id:
                    continue
                batch_ids.append(event_id)
                if event_id in detail_cache:
                    client_oid, pb_type = detail_cache[event_id]
                    event["client_order_id"] = client_oid
                    event["pb_order_type"] = pb_type
                    detail_hits += 1
                if not event.get("client_order_id"):
                    pending_tasks.append(
                        asyncio.create_task(self._enrich_with_details(event, detail_cache))
                    )
                    if len(pending_tasks) >= self.detail_concurrency:
                        detail_fetches += await self._flush_detail_tasks(pending_tasks)
                events[event_id] = event
            detail_fetches += await self._flush_detail_tasks(pending_tasks)
            if on_batch:
                batch_events = [
                    dict(events[event_id])
                    for event_id in batch_ids
                    if events[event_id].get("client_order_id")
                ]
                if batch_events:
                    on_batch(batch_events)
            oldest = min(int(raw["cTime"]) for raw in fill_list)
            if len(fill_list) < self.history_limit:
                if since_ms is None:
                    logger.debug(
                        "BitgetFetcher.fetch: short batch size=%d without start bound; stopping",
                        len(fill_list),
                    )
                    break
                end_param = int(params.get("endTime", oldest))
                if end_param - since_ms < buffer_step_ms:
                    logger.debug(
                        "BitgetFetcher.fetch: short batch size=%d close to requested start; stopping",
                        len(fill_list),
                    )
                    break
                new_end_time = max(since_ms, min(end_param, oldest) - 1)
                if new_end_time <= since_ms:
                    logger.debug(
                        "BitgetFetcher.fetch: rewound endTime to start boundary; stopping",
                    )
                    break
                params["endTime"] = new_end_time
                logger.debug(
                    "BitgetFetcher.fetch: short batch size=%d, continuing with endTime=%s",
                    len(fill_list),
                    _format_ms(params["endTime"]),
                )
                continue
            first_ts = min(ev["timestamp"] for ev in events.values()) if events else None
            if since_ms is not None and first_ts is not None and first_ts <= since_ms:
                break
            params["endTime"] = max(since_ms or oldest, oldest - 1)

        ordered = sorted(events.values(), key=lambda ev: ev["timestamp"])
        if since_ms is not None:
            ordered = [ev for ev in ordered if ev["timestamp"] >= since_ms]
        if until_ms is not None:
            ordered = [ev for ev in ordered if ev["timestamp"] <= until_ms]
        logger.info(
            "BitgetFetcher.fetch: done (events=%d, detail_cache_hits=%d, detail_fetches=%d)",
            len(ordered),
            detail_hits,
            detail_fetches,
        )
        return ordered

    async def _enrich_with_details(
        self,
        event: Dict[str, object],
        cache: Dict[str, Tuple[str, str]],
    ) -> int:
        if not event.get("order_id"):
            return 0
        logger.debug(
            "BitgetFetcher._enrich_with_details: fetching detail for order %s %s",
            event["order_id"],
            event.get("datetime"),
        )
        await self._respect_rate_limit()
        order_details = await self.api.private_mix_get_v2_mix_order_detail(
            {
                "productType": self.product_type,
                "orderId": event["order_id"],
                "symbol": event["symbol_external"],
            }
        )
        client_oid = (
            order_details.get("data", {}).get("clientOid")
            if isinstance(order_details, dict)
            else None
        )
        if client_oid:
            pb_type = custom_id_to_snake(client_oid)
            event["client_order_id"] = client_oid
            event["pb_order_type"] = pb_type
            cache[event["id"]] = (client_oid, pb_type)
            logger.debug(
                "BitgetFetcher._enrich_with_details: cached clientOid=%s for trade %s, pb_order_type %s",
                client_oid,
                event["id"],
                pb_type,
            )
            return 1
        else:
            logger.debug(
                "BitgetFetcher._enrich_with_details: no clientOid returned for order %s",
                event["order_id"],
            )
            return 1

    async def _respect_rate_limit(self) -> None:
        window_ms = 60_000
        max_calls = self.detail_calls_per_minute
        q = self._detail_call_timestamps
        while True:
            async with self._rate_lock:
                now = self._now_func()
                window_start = now - window_ms
                while q and q[0] <= window_start:
                    q.popleft()
                if len(q) < max_calls:
                    q.append(now)
                    return
                wait_ms = q[0] + window_ms - now
            if wait_ms > 0:
                logger.debug(
                    "BitgetFetcher._respect_rate_limit: sleeping %.3fs to respect %d calls/min",
                    wait_ms / 1000,
                    max_calls,
                )
                await asyncio.sleep(wait_ms / 1000)
            else:
                await asyncio.sleep(0)

    async def _flush_detail_tasks(self, tasks: List[asyncio.Task[int]]) -> int:
        if not tasks:
            return 0
        results = await asyncio.gather(*tasks, return_exceptions=True)
        tasks.clear()
        total = 0
        for res in results:
            if isinstance(res, Exception):
                logger.error(
                    "BitgetFetcher._flush_detail_tasks: detail fetch failed: %s",
                    res,
                )
                continue
            total += res or 0
        return total

    def _normalize_fill(self, raw: Dict[str, object]) -> Dict[str, object]:
        timestamp = int(raw["cTime"])
        side, position_side = deduce_side_pside(raw)
        return {
            "id": raw.get("tradeId"),
            "order_id": raw.get("orderId"),
            "timestamp": timestamp,
            "datetime": ts_to_date(timestamp),
            "symbol": self._resolve_symbol(raw.get("symbol")),
            "symbol_external": raw.get("symbol"),
            "side": side,
            "qty": float(raw.get("baseVolume", 0.0)),
            "price": float(raw.get("price", 0.0)),
            "pnl": float(raw.get("profit", 0.0)),
            "fees": raw.get("feeDetail"),
            "pb_order_type": raw.get("pb_order_type", ""),
            "position_side": position_side,
            "client_order_id": raw.get("client_order_id"),
        }

    def _resolve_symbol(self, market_symbol: Optional[str]) -> str:
        if not market_symbol:
            return ""
        try:
            resolved = self._symbol_resolver(market_symbol)
        except Exception as exc:
            logger.warning(
                "BitgetFetcher._resolve_symbol: resolver failed for %s (%s); using fallback",
                market_symbol,
                exc,
            )
            resolved = None
        if resolved:
            return resolved
        logger.warning(
            "BitgetFetcher._resolve_symbol: unresolved symbol '%s'; falling back to raw value",
            market_symbol,
        )
        return str(market_symbol)


class BinanceFetcher(BaseFetcher):
    """Fetch realised PnL events for Binance by combining income and trade history."""

    def __init__(
        self,
        api,
        *,
        symbol_resolver: Callable[[str], str],
        now_func: Optional[Callable[[], int]] = None,
        positions_provider: Optional[Callable[[], Iterable[str]]] = None,
        open_orders_provider: Optional[Callable[[], Iterable[str]]] = None,
        income_limit: int = 1000,
        trade_limit: int = 1000,
    ) -> None:
        self.api = api
        if symbol_resolver is None:
            raise ValueError("BinanceFetcher requires a symbol_resolver callable")
        self._symbol_resolver = symbol_resolver
        self._positions_provider = positions_provider or (lambda: ())
        self._open_orders_provider = open_orders_provider or (lambda: ())
        self.income_limit = min(1000, max(1, income_limit))  # cap to max 1000
        self._now_func = now_func or (lambda: int(datetime.now(tz=timezone.utc).timestamp() * 1000))
        self.trade_limit = max(1, trade_limit)

    async def fetch(
        self,
        since_ms: Optional[int],
        until_ms: Optional[int],
        detail_cache: Dict[str, Tuple[str, str]],
        on_batch: Optional[Callable[[List[Dict[str, object]]], None]] = None,
    ) -> List[Dict[str, object]]:
        logger.info(
            "BinanceFetcher.fetch: start since=%s until=%s",
            _format_ms(since_ms),
            _format_ms(until_ms),
        )
        income_events = await self._fetch_income(since_ms, until_ms)
        symbol_pool = set(self._collect_symbols(self._positions_provider))
        symbol_pool.update(self._collect_symbols(self._open_orders_provider))
        symbol_pool.update(ev["symbol"] for ev in income_events if ev.get("symbol"))
        if detail_cache is None:
            detail_cache = {}

        trade_events: Dict[str, Dict[str, object]] = {}
        trade_tasks: Dict[str, asyncio.Task[List[Dict[str, object]]]] = {}
        for symbol in sorted(symbol_pool):
            if not symbol:
                continue
            trade_tasks[symbol] = asyncio.create_task(
                self._fetch_symbol_trades(symbol, since_ms, until_ms)
            )
        for symbol, task in trade_tasks.items():
            try:
                trades = await task
            except RateLimitExceeded as exc:  # pragma: no cover - depends on live API
                logger.warning(
                    "BinanceFetcher.fetch: rate-limited fetching trades for %s (%s)", symbol, exc
                )
                trades = []
            except Exception as exc:
                logger.error("BinanceFetcher.fetch: error fetching trades for %s (%s)", symbol, exc)
                trades = []
            for trade in trades:
                event = self._normalize_trade(trade)
                cached = detail_cache.get(event["id"])
                if cached:
                    event.setdefault("client_order_id", cached[0])
                    if cached[1]:
                        event.setdefault("pb_order_type", cached[1])
                trade_events[event["id"]] = event

        merged: Dict[str, Dict[str, object]] = {}
        for ev in income_events:
            merged[ev["id"]] = ev

        def _event_from_trade(trade: Dict[str, object]) -> Dict[str, object]:
            symbol = trade.get("symbol") or self._resolve_symbol(trade.get("info", {}).get("symbol"))
            timestamp = int(trade.get("timestamp") or 0)
            client_oid = trade.get("client_order_id") or ""
            event: Dict[str, object] = {
                "id": str(trade.get("id")),
                "timestamp": timestamp,
                "datetime": ts_to_date(timestamp) if timestamp else "",
                "symbol": symbol or "",
                "side": trade.get("side") or "",
                "qty": float(trade.get("qty") or 0.0),
                "price": float(trade.get("price") or 0.0),
                "pnl": float(trade.get("pnl") or 0.0),
                "fees": trade.get("fees"),
                "pb_order_type": trade.get("pb_order_type") or "",
                "position_side": trade.get("position_side") or "unknown",
                "client_order_id": client_oid,
                "order_id": trade.get("order_id") or "",
                "info": trade.get("info"),
            }
            return event

        def _merge_trade_into_event(event: Dict[str, object], trade: Dict[str, object]) -> None:
            if not event.get("symbol") and trade.get("symbol"):
                event["symbol"] = trade["symbol"]
            if not event.get("side") and trade.get("side"):
                event["side"] = trade["side"]
            if float(event.get("qty", 0.0)) == 0.0 and trade.get("qty") is not None:
                event["qty"] = float(trade.get("qty", 0.0))
            if float(event.get("price", 0.0)) == 0.0 and trade.get("price") is not None:
                event["price"] = float(trade.get("price", 0.0))
            if not event.get("fees") and trade.get("fees"):
                event["fees"] = trade["fees"]
            if (event.get("position_side") in (None, "", "unknown")) and trade.get("position_side"):
                event["position_side"] = trade["position_side"]
            if trade.get("client_order_id"):
                event["client_order_id"] = trade["client_order_id"]
            if trade.get("order_id"):
                event["order_id"] = trade["order_id"]
            if trade.get("info"):
                event["info"] = trade["info"]
            if trade.get("pb_order_type"):
                event["pb_order_type"] = trade["pb_order_type"]

        if trade_events:
            for event_id, trade in trade_events.items():
                if event_id not in merged:
                    merged[event_id] = _event_from_trade(trade)
                event = merged[event_id]
                _merge_trade_into_event(event, trade)

        for event_id, event in merged.items():
            cached = detail_cache.get(event_id)
            if cached:
                client_oid, pb_type = cached
                if client_oid:
                    event["client_order_id"] = client_oid
                if pb_type and pb_type != "unknown":
                    event["pb_order_type"] = pb_type

        enrichment_tasks: List[asyncio.Task[Optional[Tuple[str, str]]]] = []
        enrichment_events: List[Tuple[Dict[str, object], str]] = []
        if merged:
            for event_id, event in merged.items():
                has_client = bool(event.get("client_order_id"))
                has_type = bool(event.get("pb_order_type")) and event["pb_order_type"] != "unknown"
                if has_client and has_type:
                    continue
                trade = trade_events.get(event_id)
                order_id = None
                symbol = None
                if trade:
                    order_id = trade.get("order_id")
                    symbol = trade.get("symbol") or event.get("symbol")
                else:
                    order_id = event.get("order_id")
                    symbol = event.get("symbol")
                if not order_id or not symbol:
                    continue
                enrichment_events.append((event, event_id))
                enrichment_tasks.append(
                    asyncio.create_task(
                        self._enrich_with_order_details(
                            str(order_id),
                            str(symbol),
                        )
                    )
                )
        if enrichment_tasks:
            detail_results = await asyncio.gather(*enrichment_tasks, return_exceptions=True)
            for (event, event_id), res in zip(enrichment_events, detail_results):
                if isinstance(res, Exception):
                    logger.debug(
                        "BinanceFetcher.fetch: fetch_order failed for %s (%s)",
                        event.get("id"),
                        res,
                    )
                    continue
                if not res:
                    continue
                client_oid, pb_type = res
                event["client_order_id"] = client_oid
                if pb_type:
                    event["pb_order_type"] = pb_type
                if event_id:
                    detail_cache[event_id] = (client_oid, pb_type or "")

        for event_id, ev in merged.items():
            client_oid = ev.get("client_order_id")
            if client_oid and not ev.get("pb_order_type"):
                ev["pb_order_type"] = custom_id_to_snake(str(client_oid))
            if not ev.get("pb_order_type"):
                ev["pb_order_type"] = ""
            ev["client_order_id"] = str(client_oid) if client_oid is not None else ""
            if event_id and ev.get("client_order_id"):
                detail_cache[event_id] = (ev["client_order_id"], ev["pb_order_type"])

        ordered = sorted(merged.values(), key=lambda ev: ev["timestamp"])
        if since_ms is not None:
            ordered = [ev for ev in ordered if ev["timestamp"] >= since_ms]
        if until_ms is not None:
            ordered = [ev for ev in ordered if ev["timestamp"] <= until_ms]

        if on_batch and ordered:
            on_batch(ordered)

        logger.info(
            "BinanceFetcher.fetch: done events=%d (symbols=%d)",
            len(ordered),
            len(symbol_pool),
        )
        return ordered

    async def _enrich_with_order_details(
        self,
        order_id: Optional[str],
        symbol: Optional[str],
    ) -> Optional[Tuple[str, str]]:
        if not order_id or not symbol:
            return None
        try:
            detail = await self.api.fetch_order(order_id, symbol)
        except Exception as exc:  # pragma: no cover - live API dependent
            logger.debug(
                "BinanceFetcher._enrich_with_order_details: fetch_order failed for %s (%s)",
                order_id,
                exc,
            )
            return None
        info = detail.get("info") if isinstance(detail, dict) else detail
        if not isinstance(info, dict):
            return None
        client_oid = info.get("clientOrderId") or info.get("clientOrderID")
        if not client_oid:
            return None
        client_oid = str(client_oid)
        return client_oid, custom_id_to_snake(client_oid)

    async def _fetch_income(
        self,
        since_ms: Optional[int],
        until_ms: Optional[int],
    ) -> List[Dict[str, object]]:
        params: Dict[str, object] = {"incomeType": "REALIZED_PNL", "limit": self.income_limit}
        if until_ms is None:
            if since_ms is None:
                logger.debug(f"BinanceFetcher._fetch_income.fapiprivate_get_income params={params}")
                payload = await self.api.fapiprivate_get_income(params=params)
                return sorted(
                    [self._normalize_income(x) for x in payload], key=lambda x: x["timestamp"]
                )
            until_ms = self._now_func() + 1000 * 60 * 60
        week_buffer_ms = 1000 * 60 * 60 * 24 * 6.95
        params["startTime"] = int(since_ms)
        params["endTime"] = int(min(until_ms, since_ms + week_buffer_ms))
        events = []
        previous_key: Optional[Tuple[Tuple[str, object], ...]] = None
        while True:
            key = _check_pagination_progress(
                previous_key,
                params,
                "BinanceFetcher._fetch_income",
            )
            if key is None:
                break
            previous_key = key
            payload = await self.api.fapiprivate_get_income(params=params)
            if payload == []:
                if params["startTime"] + week_buffer_ms >= until_ms:
                    break
                params["startTime"] = int(params["startTime"] + week_buffer_ms)
                params["endTime"] = int(min(until_ms, params["startTime"] + week_buffer_ms))
                continue
            events.extend(
                sorted([self._normalize_income(x) for x in payload], key=lambda x: x["timestamp"])
            )
            params["startTime"] = int(events[-1]["timestamp"]) + 1
            params["endTime"] = int(min(until_ms, params["startTime"] + week_buffer_ms))
            if params["startTime"] > until_ms:
                break
        return events

    async def _fetch_symbol_trades(
        self,
        ccxt_symbol: str,
        since_ms: Optional[int],
        until_ms: Optional[int],
    ) -> List[Dict[str, object]]:
        limit = min(1000, max(1, self.trade_limit))
        try:
            if since_ms is None and until_ms is None:
                return await self.api.fetch_my_trades(ccxt_symbol, limit=limit)

            end_bound = until_ms or (self._now_func() + 60 * 60 * 1000)
            start_bound = since_ms or max(0, end_bound - 7 * 24 * 60 * 60 * 1000)
            week_span = int(7 * 24 * 60 * 60 * 1000 * 0.999)
            params: Dict[str, object] = {}
            fetched: Dict[str, Dict[str, object]] = {}
            previous_key: Optional[Tuple[Tuple[str, object], ...]] = None

            cursor = int(start_bound)
            while cursor <= end_bound:
                window_end = int(min(end_bound, cursor + week_span))
                params["startTime"] = cursor
                params["endTime"] = window_end
                param_key = _check_pagination_progress(
                    previous_key,
                    params,
                    f"BinanceFetcher._fetch_symbol_trades({ccxt_symbol})",
                )
                if param_key is None:
                    break
                previous_key = param_key
                batch = await self.api.fetch_my_trades(
                    ccxt_symbol,
                    limit=limit,
                    params=dict(params),
                )
                if not batch:
                    cursor = window_end + 1
                    continue
                for trade in batch:
                    trade_id = str(
                        trade.get("id")
                        or (trade.get("info") or {}).get("id")
                        or f"{trade.get('order')}-{trade.get('timestamp')}"
                    )
                    fetched[trade_id] = trade
                last_ts = int(
                    batch[-1].get("timestamp")
                    or (batch[-1].get("info") or {}).get("time")
                    or params["endTime"]
                )
                if last_ts >= end_bound or len(batch) < limit:
                    cursor = last_ts + 1
                    if cursor > end_bound:
                        break
                else:
                    cursor = last_ts + 1

            ordered = sorted(
                fetched.values(),
                key=lambda tr: int(tr.get("timestamp") or (tr.get("info") or {}).get("time") or 0),
            )
            return ordered
        except Exception as exc:  # pragma: no cover - depends on live API
            logger.error("BinanceFetcher._fetch_symbol_trades: error %s (%s)", ccxt_symbol, exc)
            return []

    def _normalize_income(self, entry: Dict[str, object]) -> Dict[str, object]:
        trade_id = entry.get("tradeId") or entry.get("id") or f"income-{entry.get('time')}"
        timestamp = int(entry.get("time") or entry.get("timestamp") or 0)
        raw_symbol = entry.get("symbol")
        ccxt_symbol = self._resolve_symbol(raw_symbol)
        pnl = float(entry.get("income") or entry.get("pnl") or 0.0)
        position_side = str(entry.get("positionSide") or entry.get("pside") or "unknown").lower()
        return {
            "id": str(trade_id),
            "timestamp": timestamp,
            "datetime": ts_to_date(timestamp),
            "symbol": ccxt_symbol,
            "side": entry.get("side") or "",
            "qty": 0.0,
            "price": 0.0,
            "pnl": pnl,
            "fees": None,
            "pb_order_type": "",
            "position_side": position_side or "unknown",
            "client_order_id": entry.get("clientOrderId") or "",
        }

    def _normalize_trade(self, trade: Dict[str, object]) -> Dict[str, object]:
        info = trade.get("info") or {}
        trade_id = trade.get("id") or info.get("id")
        timestamp = int(trade.get("timestamp") or info.get("time") or info.get("T") or 0)
        pnl = float(info.get("realizedPnl") or trade.get("pnl") or 0.0)
        position_side = str(
            info.get("positionSide") or trade.get("position_side") or "unknown"
        ).lower()
        fees = trade.get("fees") or trade.get("fee")
        client_order_id = (
            trade.get("clientOrderId")
            or info.get("clientOrderId")
            or info.get("origClientOrderId")
            or info.get("clientOrderID")
            or ""
        )
        symbol = trade.get("symbol")
        if symbol and "/" not in symbol:
            symbol = self._resolve_symbol(symbol)
        order_id = (
            trade.get("order")
            or info.get("orderId")
            or info.get("origClientOrderId")
            or info.get("orderID")
        )
        return {
            "id": str(trade_id),
            "timestamp": timestamp,
            "datetime": ts_to_date(timestamp),
            "symbol": symbol or "",
            "side": trade.get("side") or "",
            "qty": float(trade.get("amount") or trade.get("qty") or 0.0),
            "price": float(trade.get("price") or 0.0),
            "pnl": pnl,
            "fees": fees,
            "pb_order_type": "",
            "position_side": position_side or "unknown",
            "client_order_id": client_order_id,
            "order_id": str(order_id) if order_id else "",
            "info": info,
        }

    def _collect_symbols(self, provider: Callable[[], Iterable[str]]) -> List[str]:
        try:
            items = provider() or []
        except Exception as exc:
            logger.warning("BinanceFetcher._collect_symbols: provider failed (%s)", exc)
            return []
        symbols: List[str] = []
        for raw in items:
            normalized = self._resolve_symbol(raw)
            if normalized:
                symbols.append(normalized)
        return symbols

    def _resolve_symbol(self, value: Optional[str]) -> str:
        if not value:
            return ""
        try:
            resolved = self._symbol_resolver(value)
            if resolved:
                return resolved
        except Exception as exc:
            logger.warning("BinanceFetcher._resolve_symbol: resolver failed for %s (%s)", value, exc)
        return str(value)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class FillEventsManager:
    """High-level interface around cached/fetched fill events."""

    def __init__(
        self,
        *,
        exchange: str,
        user: str,
        fetcher: BaseFetcher,
        cache_path: Path,
    ) -> None:
        self.exchange = exchange
        self.user = user
        self.fetcher = fetcher
        self.cache = FillEventCache(cache_path)
        self._events: List[FillEvent] = []
        self._loaded = False
        self._lock = asyncio.Lock()

    async def ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._lock:
            if self._loaded:
                return
            cached = self.cache.load()
            self._events = list(cached)
            logger.info(
                "FillEventsManager.ensure_loaded: loaded %d cached events",
                len(self._events),
            )
            self._loaded = True

    async def refresh(
        self,
        *,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> None:
        await self.ensure_loaded()
        requested_start = start_ms
        logger.info(
            "FillEventsManager.refresh: start=%s end=%s current_cache=%d (requested_start=%s)",
            _format_ms(start_ms),
            _format_ms(end_ms),
            len(self._events),
            _format_ms(requested_start),
        )
        detail_cache = {
            ev.id: (ev.client_order_id, ev.pb_order_type) for ev in self._events if ev.client_order_id
        }
        updated_map: Dict[str, FillEvent] = {ev.id: ev for ev in self._events}
        added_ids: set[str] = set()

        def handle_batch(batch: List[Dict[str, object]]) -> None:
            days_touched: set[str] = set()
            for raw in batch:
                try:
                    event = FillEvent.from_dict(raw)
                except ValueError as exc:
                    logger.warning(
                        "FillEventsManager.refresh: skipping malformed event %s (error=%s)",
                        raw.get("id"),
                        exc,
                    )
                    continue
                prev = updated_map.get(event.id)
                if prev is not None and event.timestamp < prev.timestamp:
                    continue
                updated_map[event.id] = event
                if prev is None:
                    added_ids.add(event.id)
                day = _day_key(event.timestamp)
                days_touched.add(day)
            if not days_touched:
                return
            day_payload = self._events_for_days(updated_map.values(), days_touched)
            self.cache.save_days(day_payload)
            days_list = sorted(days_touched)
            preview = ", ".join(days_list[:5])
            if len(days_list) > 5:
                preview += ", ..."
            logger.info(
                "FillEventsManager.refresh: persisted %d day files (%s)",
                len(day_payload),
                preview,
            )

        await self.fetcher.fetch(start_ms, end_ms, detail_cache, on_batch=handle_batch)

        self._events = sorted(updated_map.values(), key=lambda ev: ev.timestamp)
        logger.info(
            "FillEventsManager.refresh: merged events=%d (added=%d)",
            len(self._events),
            len(added_ids),
        )

    async def refresh_latest(self, *, overlap: int = 20) -> None:
        """Fetch only the most recent fills, overlapping by `overlap` events."""
        await self.ensure_loaded()
        if not self._events:
            logger.info("FillEventsManager.refresh_latest: cache empty, falling back to full refresh")
        start_ms = None
        if self._events:
            idx = max(0, len(self._events) - overlap)
            start_ms = self._events[idx].timestamp
        await self.refresh(start_ms=start_ms, end_ms=None)

    async def refresh_range(
        self,
        start_ms: int,
        end_ms: Optional[int],
        *,
        gap_hours: float = 12.0,
        overlap: int = 20,
    ) -> None:
        """Fill missing data between `start_ms` and `end_ms` using gap heuristics."""
        await self.ensure_loaded()
        gap_ms = max(1, int(gap_hours * 60 * 60 * 1000))
        intervals: List[Tuple[int, int]] = []

        if not self._events:
            logger.info("FillEventsManager.refresh_range: cache empty, refreshing entire interval")
            await self.refresh(start_ms=start_ms, end_ms=end_ms)
            await self.refresh_latest(overlap=overlap)
            return

        events_sorted = self._events
        earliest = events_sorted[0].timestamp
        latest = events_sorted[-1].timestamp

        if start_ms < earliest:
            upper = earliest if end_ms is None else min(earliest, end_ms)
            if start_ms < upper:
                intervals.append((start_ms, upper))

        prev_ts = events_sorted[0].timestamp
        for ev in events_sorted[1:]:
            cur_ts = ev.timestamp
            if end_ms is not None and prev_ts >= end_ms:
                break
            gap = cur_ts - prev_ts
            if gap >= gap_ms:
                gap_start = max(prev_ts, start_ms)
                gap_end = cur_ts if end_ms is None else min(cur_ts, end_ms)
                if gap_start < gap_end:
                    intervals.append((gap_start, gap_end))
            prev_ts = cur_ts

        if end_ms is not None and latest < end_ms:
            lower = max(latest, start_ms)
            if lower < end_ms:
                intervals.append((lower, end_ms))

        merged = self._merge_intervals(intervals)
        if merged:
            logger.info(
                "FillEventsManager.refresh_range: refreshing %d intervals: %s",
                len(merged),
                ", ".join(f"{_format_ms(start)} → {_format_ms(end)}" for start, end in merged),
            )
        else:
            logger.info("FillEventsManager.refresh_range: no gaps detected in requested interval")

        for start, end in merged:
            await self.refresh(start_ms=start, end_ms=end)

        await self.refresh_latest(overlap=overlap)

    def get_events(
        self,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        symbol: Optional[str] = None,
    ) -> List[FillEvent]:
        events = self._events
        if start_ms is not None:
            events = [ev for ev in events if ev.timestamp >= start_ms]
        if end_ms is not None:
            events = [ev for ev in events if ev.timestamp <= end_ms]
        if symbol:
            events = [ev for ev in events if ev.symbol == symbol]
        return list(events)

    def get_pnl_sum(
        self,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        symbol: Optional[str] = None,
    ) -> float:
        events = self.get_events(start_ms, end_ms, symbol)
        return float(sum(ev.pnl for ev in events))

    def get_pnl_cumsum(
        self,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        symbol: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        events = self.get_events(start_ms, end_ms, symbol)
        total = 0.0
        result = []
        for ev in events:
            total += ev.pnl
            result.append((ev.timestamp, total))
        return result

    def get_last_timestamp(self, symbol: Optional[str] = None) -> Optional[int]:
        events = self._events
        if symbol:
            events = [ev for ev in events if ev.symbol == symbol]
        if not events:
            return None
        return max(ev.timestamp for ev in events)

    def reconstruct_positions(
        self, current_positions: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        positions: Dict[str, float] = dict(current_positions or {})
        for ev in self._events:
            sign = 1 if ev.side == "buy" else -1
            key = f"{ev.symbol}:{ev.position_side}"
            positions[key] = positions.get(key, 0.0) + sign * ev.qty
        return positions

    def reconstruct_equity_curve(self, starting_equity: float = 0.0) -> List[Tuple[int, float]]:
        total = starting_equity
        points: List[Tuple[int, float]] = []
        for ev in self._events:
            total += ev.pnl
            points.append((ev.timestamp, total))
        return points

    @staticmethod
    def _events_for_days(
        events: Iterable[FillEvent], days: Iterable[str]
    ) -> Dict[str, List[FillEvent]]:
        target = {day: [] for day in days}
        for event in events:
            day = _day_key(event.timestamp)
            if day in target:
                target[day].append(event)
        for day_events in target.values():
            day_events.sort(key=lambda ev: ev.timestamp)
        return target

    @staticmethod
    def _merge_intervals(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
        cleaned = [(int(start), int(end)) for start, end in intervals if end > start]
        if not cleaned:
            return []
        cleaned.sort(key=lambda x: x[0])
        merged: List[Tuple[int, int]] = []
        cur_start, cur_end = cleaned[0]
        for start, end in cleaned[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        merged.append((cur_start, cur_end))
        return merged


class BybitFetcher(BaseFetcher):
    """Fetches fill events from Bybit using trades + positions history."""

    def __init__(
        self,
        api,
        *,
        category: str = "linear",
        trade_limit: int = 100,
        position_limit: int = 100,
        overlap_days: float = 3.0,
        max_span_days: float = 6.5,
    ) -> None:
        self.api = api
        self.category = category
        self.trade_limit = max(1, min(trade_limit, 100))
        self.position_limit = max(1, min(position_limit, 100))
        self._default_span_ms = int(overlap_days * 24 * 60 * 60 * 1000)
        self._max_span_ms = int(max_span_days * 24 * 60 * 60 * 1000)

    async def fetch(
        self,
        since_ms: Optional[int],
        until_ms: Optional[int],
        detail_cache: Dict[str, Tuple[str, str]],
        on_batch: Optional[Callable[[List[Dict[str, object]]], None]] = None,
    ) -> List[Dict[str, object]]:
        end_ms = until_ms or (self._now_ms() + 60 * 60 * 1000)
        start_ms = since_ms or max(0, end_ms - self._default_span_ms)

        trades = await self._fetch_my_trades(start_ms, end_ms)
        positions = await self._fetch_positions_history(start_ms, end_ms)

        events = self._combine(trades, positions, detail_cache)
        events = [
            ev
            for ev in events
            if (since_ms is None or ev["timestamp"] >= since_ms)
            and (until_ms is None or ev["timestamp"] <= until_ms)
        ]
        events.sort(key=lambda ev: ev["timestamp"])
        events = _coalesce_events(events)

        if on_batch and events:
            day_map = defaultdict(list)
            for ev in events:
                day_map[_day_key(ev["timestamp"])].append(ev)
            for day in sorted(day_map):
                on_batch(day_map[day])

        logger.info(
            "BybitFetcher.fetch: done (events=%d, trades=%d, positions=%d)",
            len(events),
            len(trades),
            len(positions),
        )
        return events

    async def _fetch_my_trades(self, start_ms: int, end_ms: int) -> List[Dict[str, object]]:
        params = {
            "type": "swap",
            "subType": self.category,
            "limit": self.trade_limit,
            "endTime": int(end_ms),
        }
        results: List[Dict[str, object]] = []
        max_fetches = 200
        fetch_count = 0
        prev_params = None
        while True:
            new_key = _check_pagination_progress(
                prev_params,
                params,
                "BybitFetcher._fetch_my_trades",
            )
            if new_key is None:
                break
            prev_params = new_key
            fetch_count += 1
            batch = await self.api.fetch_my_trades(params=params)
            if not batch:
                break
            batch.sort(key=lambda x: x["timestamp"])
            results.extend(batch)
            if len(batch) < self.trade_limit:
                if params["endTime"] - start_ms < self._max_span_ms:
                    break
                params["endTime"] = max(start_ms, params["endTime"] - self._max_span_ms)
                continue
            first_ts = batch[0]["timestamp"]
            if first_ts <= start_ms:
                break
            if params["endTime"] == first_ts:
                break
            params["endTime"] = int(first_ts)
            if fetch_count >= max_fetches:
                logger.warning("BybitFetcher._fetch_my_trades: max fetches reached")
                break
        ordered = sorted(
            results,
            key=lambda x: int(x.get("info", {}).get("updatedTime") or x.get("timestamp") or 0),
        )
        return ordered

    async def _fetch_positions_history(self, start_ms: int, end_ms: int) -> List[Dict[str, object]]:
        params = {
            "limit": self.position_limit,
            "endTime": int(end_ms),
        }
        results: List[Dict[str, object]] = []
        max_fetches = 200
        fetch_count = 0
        prev_params = None
        while True:
            new_key = _check_pagination_progress(
                prev_params,
                params,
                "BybitFetcher._fetch_positions_history",
            )
            if new_key is None:
                break
            prev_params = new_key
            fetch_count += 1
            batch = await self.api.fetch_positions_history(params=params)
            if not batch:
                break
            batch.sort(key=lambda x: x["timestamp"])
            results.extend(batch)
            if len(batch) < self.position_limit:
                if params["endTime"] - start_ms < self._max_span_ms:
                    break
                params["endTime"] = max(start_ms, params["endTime"] - self._max_span_ms)
                continue
            first_ts = batch[0]["timestamp"]
            if first_ts <= start_ms:
                break
            if params["endTime"] == first_ts:
                break
            params["endTime"] = int(first_ts)
            if fetch_count >= max_fetches:
                logger.warning("BybitFetcher._fetch_positions_history: max fetches reached")
                break
        return results

    def _combine(
        self,
        trades: List[Dict[str, object]],
        positions: List[Dict[str, object]],
        detail_cache: Dict[str, Tuple[str, str]],
    ) -> List[Dict[str, object]]:
        pnls: Dict[str, float] = defaultdict(float)
        symbol_realized: Dict[str, float] = defaultdict(float)
        symbol_closed_qty: Dict[str, float] = defaultdict(float)
        for entry in positions:
            order_id = str(entry.get("info", {}).get("orderId", entry.get("orderId", "")))
            if not order_id:
                continue
            pnl = float(entry.get("realizedPnl") or entry.get("info", {}).get("closedPnl") or 0.0)
            pnls[order_id] += pnl
            symbol = entry.get("symbol")
            if symbol:
                symbol_realized[symbol] += pnl
                closed = float(
                    entry.get("info", {}).get("closedSize") or entry.get("contracts") or 0.0
                )
                symbol_closed_qty[symbol] += closed

        order_total_qty: Dict[str, float] = defaultdict(float)
        symbol_order_qty: Dict[str, float] = defaultdict(float)
        symbol_unknown_trade_qty: Dict[str, float] = defaultdict(float)
        for trade in trades:
            order_id = str(trade.get("info", {}).get("orderId", trade.get("order")))
            qty = abs(float(trade.get("amount") or trade.get("info", {}).get("execQty") or 0.0))
            symbol = trade.get("symbol") or trade.get("info", {}).get("symbol")
            if order_id and order_id in pnls:
                order_total_qty[order_id] += qty
                if symbol:
                    symbol_order_qty[symbol] += qty
            elif symbol:
                symbol_unknown_trade_qty[symbol] += qty

        order_remaining_qty = dict(order_total_qty)
        order_remaining_pnl = dict(pnls)
        symbol_remaining_pnl = dict(symbol_realized)
        symbol_remaining_qty: Dict[str, float] = {}
        for sym, closed in symbol_closed_qty.items():
            remaining = max(closed - symbol_order_qty.get(sym, 0.0), 0.0)
            symbol_remaining_qty[sym] = remaining
        for sym, qty in symbol_unknown_trade_qty.items():
            symbol_remaining_qty[sym] = symbol_remaining_qty.get(sym, 0.0) + qty

        events: List[Dict[str, object]] = []
        for trade in trades:
            event = self._normalize_trade(trade)
            order_id = event.get("order_id")
            cache_entry = detail_cache.get(event["id"])
            allocated = False
            if order_id and order_id in order_remaining_pnl and order_remaining_qty[order_id] > 0:
                remaining_qty = order_remaining_qty[order_id]
                remaining_pnl = order_remaining_pnl[order_id]
                qty = abs(event["qty"])
                if remaining_qty <= qty * 1.0000001:
                    event["pnl"] = remaining_pnl
                else:
                    event["pnl"] = remaining_pnl * (qty / remaining_qty)
                order_remaining_qty[order_id] = max(0.0, remaining_qty - qty)
                order_remaining_pnl[order_id] = remaining_pnl - event["pnl"]
                symbol_remaining_pnl[event["symbol"]] = (
                    symbol_remaining_pnl.get(event["symbol"], 0.0) - event["pnl"]
                )
                symbol_remaining_qty[event["symbol"]] = max(
                    0.0, symbol_remaining_qty.get(event["symbol"], 0.0) - qty
                )
                allocated = True
            if cache_entry:
                event["client_order_id"], event["pb_order_type"] = cache_entry
                if not event["pb_order_type"]:
                    event["pb_order_type"] = "unknown"
            elif event["client_order_id"]:
                pb_type = custom_id_to_snake(event["client_order_id"])
                event["pb_order_type"] = pb_type or "unknown"
            else:
                event["pb_order_type"] = "unknown"
            if (
                event["pb_order_type"] == "unknown"
                and not allocated
                and abs(event.get("pnl", 0.0)) < 1e-12
            ):
                symbol = event["symbol"]
                remaining_symbol_qty = symbol_remaining_qty.get(symbol, 0.0)
                remaining_symbol_pnl = symbol_remaining_pnl.get(symbol, 0.0)
                qty = abs(event["qty"])
                if remaining_symbol_qty > 0:
                    if remaining_symbol_qty <= qty * 1.0000001:
                        event["pnl"] = remaining_symbol_pnl
                    else:
                        event["pnl"] = remaining_symbol_pnl * (qty / remaining_symbol_qty)
                    symbol_remaining_qty[symbol] = max(0.0, remaining_symbol_qty - qty)
                    symbol_remaining_pnl[symbol] = remaining_symbol_pnl - event["pnl"]
                else:
                    event["pnl"] = remaining_symbol_pnl
            events.append(event)

        remaining_orders = [k for k, v in order_remaining_pnl.items() if abs(v) > 1e-6]
        if remaining_orders:
            logger.warning(
                "BybitFetcher._combine: residual PnL for orders %s (values=%s)",
                remaining_orders,
                [order_remaining_pnl[k] for k in remaining_orders],
            )
        remaining_symbols = [k for k, v in symbol_remaining_pnl.items() if abs(v) > 1e-6]
        if remaining_symbols:
            logger.debug(
                "BybitFetcher._combine: remaining symbol-level PnL after distribution %s",
                {k: symbol_remaining_pnl[k] for k in remaining_symbols},
            )
        return events

    @staticmethod
    def _normalize_trade(trade: Dict[str, object]) -> Dict[str, object]:
        info = trade.get("info", {})
        order_id = str(info.get("orderId", trade.get("order")))
        trade_id = str(trade.get("id") or info.get("execId") or order_id)
        timestamp = int(trade.get("timestamp") or info.get("execTime", 0))
        qty = float(trade.get("amount") or info.get("execQty", 0.0))
        side = str(trade.get("side") or info.get("side", "")).lower()
        price = float(trade.get("price") or info.get("execPrice", 0.0))
        closed_size = float(info.get("closedSize") or info.get("closeSize") or 0.0)
        position_side = BybitFetcher._determine_position_side(side, closed_size)
        pnl = float(trade.get("pnl") or 0.0)
        client_order_id = info.get("orderLinkId") or trade.get("clientOrderId")
        fee = trade.get("fee")
        symbol = trade.get("symbol") or info.get("symbol")

        return {
            "id": trade_id,
            "order_id": order_id,
            "timestamp": timestamp,
            "datetime": ts_to_date(timestamp),
            "symbol": symbol,
            "side": side,
            "qty": abs(qty),
            "price": price,
            "pnl": pnl,
            "fees": fee,
            "pb_order_type": "",
            "position_side": position_side,
            "client_order_id": client_order_id or "",
        }

    @staticmethod
    def _determine_position_side(side: str, closed_size: float) -> str:
        if side == "buy":
            return "short" if closed_size else "long"
        if side == "sell":
            return "long" if closed_size else "short"
        return "long"

    @staticmethod
    def _now_ms() -> int:
        return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


class HyperliquidFetcher(BaseFetcher):
    """Fetches fill events via ccxt.fetch_my_trades for Hyperliquid."""

    def __init__(
        self,
        api,
        *,
        trade_limit: int = 500,
        symbol_resolver: Optional[Callable[[Optional[str]], str]] = None,
    ) -> None:
        self.api = api
        self.trade_limit = max(1, trade_limit)
        self._symbol_resolver = symbol_resolver

    async def fetch(
        self,
        since_ms: Optional[int],
        until_ms: Optional[int],
        detail_cache: Dict[str, Tuple[str, str]],
        on_batch: Optional[Callable[[List[Dict[str, object]]], None]] = None,
    ) -> List[Dict[str, object]]:
        params: Dict[str, object] = {"limit": self.trade_limit}
        if since_ms is not None:
            params["since"] = int(since_ms)

        collected: Dict[str, Dict[str, object]] = {}
        max_fetches = 200
        fetch_count = 0

        prev_params = None
        while True:
            check_params = dict(params)
            check_params["_page"] = fetch_count
            new_key = _check_pagination_progress(
                prev_params,
                check_params,
                "HyperliquidFetcher.fetch",
            )
            if new_key is None:
                break
            prev_params = new_key
            try:
                trades = await self.api.fetch_my_trades(params=params)
            except RateLimitExceeded as exc:
                logger.debug(
                    "HyperliquidFetcher.fetch: rate limit exceeded, sleeping briefly (%s)",
                    exc,
                )
                await asyncio.sleep(1.0)
                continue
            fetch_count += 1
            if not trades:
                break
            before_count = len(collected)
            for trade in trades:
                event = self._normalize_trade(trade)
                ts = event["timestamp"]
                if since_ms is not None and ts < since_ms:
                    continue
                if until_ms is not None and ts > until_ms:
                    continue
                collected[event["id"]] = event
            added = len(collected) - before_count
            if len(trades) < self.trade_limit:
                break
            last_ts = int(
                trades[-1].get("timestamp")
                or trades[-1].get("info", {}).get("time")
                or trades[-1].get("info", {}).get("updatedTime")
                or 0
            )
            if last_ts <= 0:
                break
            if until_ms is not None and last_ts >= until_ms:
                break
            if added <= 0:
                logger.debug(
                    "HyperliquidFetcher.fetch: no new trades added on page (last_ts=%s), stopping",
                    last_ts,
                )
                break
            params["since"] = last_ts
            if fetch_count >= max_fetches:
                logger.warning(
                    "HyperliquidFetcher.fetch: reached maximum pagination depth (%d)",
                    max_fetches,
                )
                break

        events = sorted(collected.values(), key=lambda ev: ev["timestamp"])
        events = _coalesce_events(events)

        for event in events:
            cache_entry = detail_cache.get(event["id"])
            if cache_entry:
                event["client_order_id"], event["pb_order_type"] = cache_entry
            elif event["client_order_id"]:
                event["pb_order_type"] = custom_id_to_snake(event["client_order_id"])
            else:
                event["pb_order_type"] = "unknown"
            if not event["pb_order_type"]:
                event["pb_order_type"] = "unknown"

        if on_batch and events:
            on_batch(events)

        return events

    def _normalize_trade(self, trade: Dict[str, object]) -> Dict[str, object]:
        info = trade.get("info", {}) or {}
        trade_id = str(trade.get("id") or info.get("tid") or info.get("hash") or info.get("oid"))
        order_id = str(trade.get("order") or info.get("oid") or "")
        timestamp = int(
            trade.get("timestamp")
            or info.get("time")
            or info.get("updatedTime")
            or info.get("tradeTime")
            or 0
        )
        symbol_raw = trade.get("symbol") or info.get("symbol") or info.get("coin")
        if self._symbol_resolver:
            try:
                symbol = self._symbol_resolver(symbol_raw)
            except Exception:
                symbol = str(symbol_raw) if symbol_raw is not None else ""
        else:
            symbol = str(symbol_raw) if symbol_raw is not None else ""
        side = str(trade.get("side") or info.get("side", "")).lower()
        qty = abs(float(trade.get("amount") or info.get("sz") or 0.0))
        price = float(trade.get("price") or info.get("px") or 0.0)
        pnl = float(trade.get("pnl") or info.get("closedPnl") or 0.0)
        fee = trade.get("fee") or {
            "currency": info.get("feeToken"),
            "cost": info.get("fee"),
        }
        client_order_id = trade.get("clientOrderId") or info.get("cloid") or info.get("clOrdId") or ""
        direction = str(info.get("dir", "")).lower()
        if "short" in direction:
            position_side = "short"
        elif "long" in direction:
            position_side = "long"
        else:
            position_side = "long" if side == "buy" else "short"

        return {
            "id": trade_id,
            "order_id": order_id,
            "timestamp": timestamp,
            "datetime": ts_to_date(timestamp) if timestamp else "",
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "pnl": pnl,
            "fees": fee,
            "pb_order_type": "",
            "position_side": position_side,
            "client_order_id": str(client_order_id or ""),
        }


# ---------------------------------------------------------------------------
# Utilities for Bitget integration
# ---------------------------------------------------------------------------


def custom_id_to_snake(client_oid: str) -> str:
    """Placeholder import shim; real implementation lives in passivbot."""
    try:
        from passivbot import custom_id_to_snake as _real

        return _real(client_oid)
    except Exception:
        return client_oid or ""


def deduce_side_pside(elm: dict) -> Tuple[str, str]:
    """Import helper from exchanges.bitget when available."""
    try:
        from exchanges.bitget import deduce_side_pside as _real

        return _real(elm)
    except Exception:
        side = str(elm.get("side", "buy")).lower()
        return side or "buy", "long"


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


EXCHANGE_BOT_CLASSES: Dict[str, Tuple[str, str]] = {
    "binance": ("exchanges.binance", "BinanceBot"),
    "bitget": ("exchanges.bitget", "BitgetBot"),
    "bybit": ("exchanges.bybit", "BybitBot"),
    "hyperliquid": ("exchanges.hyperliquid", "HyperliquidBot"),
}


def _parse_time_arg(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        ts = int(value)
        if ts < 10**11:
            ts *= 1000
        return ts
    except ValueError:
        pass
    try:
        if value.lower() == "now":
            dt = datetime.now(tz=timezone.utc)
        else:
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        raise ValueError(f"Unable to parse datetime '{value}'")


def _parse_log_level(value: str) -> int:
    mapping = {"warning": 0, "warn": 0, "info": 1, "debug": 2, "trace": 3}
    if value is None:
        return 1
    value = str(value).strip().lower()
    if value in mapping:
        return mapping[value]
    try:
        lvl = int(float(value))
        return max(0, min(3, lvl))
    except Exception:
        return 1


def _extract_symbol_pool(config: dict, override: Optional[List[str]]) -> List[str]:
    if override:
        return sorted({sym for sym in override if sym})
    live = config.get("live", {})
    approved = live.get("approved_coins")
    symbols: List[str] = []
    if isinstance(approved, dict):
        for vals in approved.values():
            if isinstance(vals, list):
                symbols.extend(vals)
    elif isinstance(approved, list):
        symbols.extend(approved)
    return sorted({sym for sym in symbols if sym})


def _symbol_resolver(bot) -> Callable[[Optional[str]], str]:
    def resolver(raw: Optional[str]) -> str:
        if not raw:
            return ""
        if isinstance(raw, str) and "/" in raw:
            return raw
        value = "" if raw is None else str(raw)
        if not value:
            return ""
        # Prefer the bot's coin_to_symbol mapping which handles exchange quirks
        try:
            mapped = bot.coin_to_symbol(value, verbose=False)
            if mapped:
                return mapped
        except Exception:
            pass
        upper = value.upper()
        for quote in ("USDT", "USDC", "USD"):
            if upper.endswith(quote) and len(upper) > len(quote):
                base = upper[: -len(quote)]
                if base:
                    return f"{base}/{quote}:{quote}"
        if ":" in value and "/" not in value:
            base, _, quote = value.partition(":")
            if base and quote:
                return f"{base}/{quote}:{quote}"
        return value

    return resolver


def _build_fetcher_for_bot(bot, symbols: List[str]) -> BaseFetcher:
    exchange = getattr(bot, "exchange", "").lower()
    resolver = _symbol_resolver(bot)
    static_provider = lambda: symbols  # noqa: E731
    if exchange == "binance":
        return BinanceFetcher(
            api=bot.cca,
            symbol_resolver=resolver,
            positions_provider=static_provider,
            open_orders_provider=static_provider,
        )
    if exchange == "bitget":
        return BitgetFetcher(
            api=bot.cca,
            symbol_resolver=lambda value: resolver(value),
        )
    if exchange == "bybit":
        return BybitFetcher(api=bot.cca)
    if exchange == "hyperliquid":
        return HyperliquidFetcher(
            api=bot.cca,
            symbol_resolver=lambda value: resolver(value),
        )
    raise ValueError(f"Unsupported exchange '{exchange}' for fill events CLI")


def _instantiate_bot(config: dict):
    live = config.get("live", {})
    user = str(live.get("user") or "").strip()
    if not user:
        raise ValueError("Config missing live.user to determine bot exchange")
    user_info = load_user_info(user)
    exchange = str(user_info.get("exchange") or "").lower()
    if not exchange:
        raise ValueError(f"User '{user}' has no exchange configured in api-keys.json")
    bot_cls_info = EXCHANGE_BOT_CLASSES.get(exchange)
    if bot_cls_info is None:
        raise ValueError(f"No bot class registered for exchange '{exchange}'")
    module = import_module(bot_cls_info[0])
    bot_cls = getattr(module, bot_cls_info[1])
    return bot_cls(config)


async def _run_cli(args: argparse.Namespace) -> None:
    config = load_config(args.config, verbose=False)
    config = format_config(config, verbose=False)
    live = config.setdefault("live", {})
    if args.user:
        live["user"] = args.user
    bot = _instantiate_bot(config)
    try:
        symbol_pool = _extract_symbol_pool(config, args.symbols)
        fetcher = _build_fetcher_for_bot(bot, symbol_pool)
        cache_root = Path(args.cache_root)
        cache_path = cache_root / bot.exchange / bot.user
        manager = FillEventsManager(
            exchange=bot.exchange,
            user=bot.user,
            fetcher=fetcher,
            cache_path=cache_path,
        )
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        start_ms = _parse_time_arg(args.start) or (
            now_ms - int(args.lookback_days * 24 * 60 * 60 * 1000)
        )
        end_ms = _parse_time_arg(args.end) or now_ms
        if start_ms >= end_ms:
            raise ValueError("start time must be earlier than end time")
        logger.info(
            "fill_events_manager CLI | exchange=%s user=%s start=%s end=%s cache=%s",
            bot.exchange,
            bot.user,
            _format_ms(start_ms),
            _format_ms(end_ms),
            cache_path,
        )
        await manager.refresh_range(start_ms, end_ms)
        events = manager.get_events(start_ms, end_ms)
        logger.info("fill_events_manager CLI: events=%d written to %s", len(events), cache_path)
    finally:
        try:
            await bot.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill events cache refresher")
    parser.add_argument("--config", type=str, default="configs/template.json", help="Config path")
    parser.add_argument("--user", type=str, required=True, help="Live user identifier")
    parser.add_argument("--start", type=str, help="Start datetime (ms or ISO)")
    parser.add_argument("--end", type=str, help="End datetime (ms or ISO)")
    parser.add_argument(
        "--lookback-days",
        type=float,
        default=3.0,
        help="Default lookback window in days when start is omitted",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Logging verbosity (warning/info/debug/trace or 0-3)",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default="caches/fill_events",
        help="Root directory for fill events cache (default: caches/fill_events)",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional explicit symbol list to fetch",
    )
    args = parser.parse_args()
    configure_logging(debug=_parse_log_level(args.log_level))
    try:
        asyncio.run(_run_cli(args))
    except KeyboardInterrupt:
        logger.info("fill_events_manager CLI interrupted by user")


if __name__ == "__main__":
    main()
