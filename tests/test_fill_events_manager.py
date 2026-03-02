import asyncio
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from ccxt.base.errors import RateLimitExceeded

from src.fill_events_manager import (
    BaseFetcher,
    BinanceFetcher,
    BitgetFetcher,
    BybitFetcher,
    GateioFetcher,
    HyperliquidFetcher,
    KucoinFetcher,
    OkxFetcher,
    FillEvent,
    FillEventCache,
    FillEventsManager,
    GAP_REASON_FETCH_FAILED,
    compute_psize_pprice,
    custom_id_to_snake,
    ensure_qty_signage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self, start_ms: int = 0) -> None:
        self._now = start_ms

    def now_ms(self) -> int:
        return self._now

    def advance(self, delta_ms: int) -> None:
        self._now += delta_ms


class _FakeBitgetAPI:
    def __init__(self, batches: List[List[Dict[str, str]]]) -> None:
        self._batches = batches
        self.history_calls: List[Dict[str, object]] = []
        self.detail_calls: List[Dict[str, object]] = []
        self.details_by_order: Dict[str, Dict[str, object]] = {}

    def add_detail(self, order_id: str, client_oid: str) -> None:
        self.details_by_order[order_id] = {
            "data": {
                "clientOid": client_oid,
            }
        }

    async def private_mix_get_v2_mix_order_fill_history(self, params: Dict[str, object]):
        self.history_calls.append(dict(params))
        if not self._batches:
            return {"data": {"fillList": []}}
        return {"data": {"fillList": self._batches.pop(0)}}

    async def private_mix_get_v2_mix_order_detail(self, params: Dict[str, object]):
        self.detail_calls.append(dict(params))
        order_id = params["orderId"]
        return self.details_by_order.get(order_id, {"data": {"clientOid": None}})


class _StaticFetcher(BaseFetcher):
    def __init__(self, events: List[Dict[str, object]]):
        self.events = list(events)
        self.calls: List[Tuple[Optional[int], Optional[int]]] = []

    async def fetch(self, since_ms, until_ms, detail_cache, on_batch=None):
        self.calls.append((since_ms, until_ms))
        payload = [dict(ev) for ev in self.events]
        if on_batch:
            on_batch(payload)
        return payload


class _FakeBybitAPI:
    def __init__(
        self,
        trades_batches: List[List[Dict[str, Any]]],
        positions_batches: List[List[Dict[str, Any]]],
    ) -> None:
        self._trades_batches = list(trades_batches)
        self._positions_batches = list(positions_batches)
        self.trade_calls: List[Dict[str, Any]] = []
        self.position_calls: List[Dict[str, Any]] = []
        self.markets: Dict[str, Dict[str, Any]] = {}  # For symbol resolution

    async def fetch_my_trades(self, params: Dict[str, Any]):
        self.trade_calls.append(dict(params))
        if not self._trades_batches:
            return []
        return self._trades_batches.pop(0)

    async def fetch_positions_history(self, params: Dict[str, Any]):
        self.position_calls.append(dict(params))
        if not self._positions_batches:
            return []
        return self._positions_batches.pop(0)

    async def private_get_v5_position_closed_pnl(self, params: Dict[str, Any]):
        """Return raw Bybit API format for closed-pnl endpoint."""
        self.position_calls.append(dict(params))
        if not self._positions_batches:
            return {"result": {"list": [], "nextPageCursor": ""}}

        # Convert CCXT-like format back to raw Bybit format
        ccxt_batch = self._positions_batches.pop(0)
        raw_list = []
        for pos in ccxt_batch:
            info = pos.get("info", {})
            if info:
                raw_list.append(info)
            else:
                # Construct raw format from CCXT format
                raw_list.append(
                    {
                        "symbol": pos.get("symbol", "").replace("/", "").replace(":", ""),
                        "orderId": pos.get("orderId", ""),
                        "closedPnl": str(pos.get("realizedPnl", 0)),
                        "closedSize": str(pos.get("contracts", 0)),
                        "avgEntryPrice": str(pos.get("entryPrice", 0)),
                        "avgExitPrice": str(pos.get("lastPrice", 0)),
                        "updatedTime": str(pos.get("timestamp", 0)),
                        "createdTime": str(pos.get("timestamp", 0)),
                        "leverage": str(pos.get("leverage", 1)),
                        "side": "Sell" if pos.get("side") == "long" else "Buy",
                        "closeFee": "0",
                        "openFee": "0",
                    }
                )
        return {"result": {"list": raw_list, "nextPageCursor": ""}}


class _ManualFetcher(BaseFetcher):
    def __init__(self, events: List[Dict[str, object]]):
        self.events = events

    async def fetch(self, since_ms, until_ms, detail_cache, on_batch=None):
        if on_batch:
            on_batch(self.events)
        return list(self.events)


class _FakeHyperliquidAPI:
    def __init__(self, batches: List[List[Dict[str, Any]]]) -> None:
        self._batches = list(batches)
        self.calls: List[Dict[str, Any]] = []

    async def fetch_my_trades(self, params: Dict[str, Any]):
        self.calls.append(dict(params))
        if not self._batches:
            return []
        return self._batches.pop(0)


# ---------------------------------------------------------------------------
# Binance fetcher tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_binance_fetcher_merges_trade_details(monkeypatch):
    client_id = "order-0x0005"
    income_events = [
        {
            "id": "trade-1",
            "timestamp": 1_700_000_000_000,
            "datetime": "2023-11-01T00:00:00Z",
            "symbol": "ADA/USDT:USDT",
            "side": "buy",
            "qty": 0.0,
            "price": 0.0,
            "pnl": 1.5,
            "fees": None,
            "pb_order_type": "",
            "position_side": "long",
            "client_order_id": "",
        }
    ]
    trades = [
        {
            "id": "trade-1",
            "timestamp": 1_700_000_000_000,
            "symbol": "ADA/USDT:USDT",
            "side": "buy",
            "amount": 5.0,
            "price": 0.65,
            "fee": {"currency": "USDT", "cost": 0.001},
            "clientOrderId": client_id,
            "order": "123456",
            "info": {"positionSide": "LONG"},
        }
    ]

    fetcher = BinanceFetcher(
        api=object(),
        symbol_resolver=lambda sym: sym or "",
        positions_provider=lambda: ["ADA/USDT:USDT"],
        open_orders_provider=lambda: [],
    )

    async def fake_fetch_income(self, *_args, **_kwargs):
        return [dict(ev) for ev in income_events]

    async def fake_fetch_trades(self, symbol, *_args, **_kwargs):
        return list(trades if symbol == "ADA/USDT:USDT" else [])

    monkeypatch.setattr(
        fetcher,
        "_fetch_income",
        types.MethodType(fake_fetch_income, fetcher),
    )
    monkeypatch.setattr(
        fetcher,
        "_fetch_symbol_trades",
        types.MethodType(fake_fetch_trades, fetcher),
    )

    detail_cache: Dict[str, Tuple[str, str]] = {}
    events = await fetcher.fetch(
        since_ms=income_events[0]["timestamp"],
        until_ms=income_events[0]["timestamp"] + 1,
        detail_cache=detail_cache,
    )
    assert len(events) == 1
    event = events[0]
    assert event["qty"] == pytest.approx(5.0)
    assert event["price"] == pytest.approx(0.65)
    assert event["client_order_id"] == client_id
    assert event["pb_order_type"] == custom_id_to_snake(client_id)
    assert detail_cache["trade-1"][0] == client_id


@pytest.mark.asyncio
async def test_binance_fetcher_enriches_missing_client_ids(monkeypatch):
    income_events = [
        {
            "id": "trade-2",
            "timestamp": 1_700_000_100_000,
            "datetime": "2023-11-02T00:00:00Z",
            "symbol": "BTC/USDT:USDT",
            "side": "sell",
            "qty": 0.0,
            "price": 0.0,
            "pnl": -2.0,
            "fees": None,
            "pb_order_type": "",
            "position_side": "short",
            "client_order_id": "",
        }
    ]
    trades = [
        {
            "id": "trade-2",
            "timestamp": 1_700_000_100_000,
            "symbol": "BTC/USDT:USDT",
            "side": "sell",
            "amount": 0.1,
            "price": 35_000.0,
            "fee": {"currency": "USDT", "cost": 0.003},
            "order": "abc-789",
            "info": {"positionSide": "SHORT"},
        }
    ]
    enrich_calls: List[Tuple[str, str]] = []

    fetcher = BinanceFetcher(
        api=object(),
        symbol_resolver=lambda sym: sym or "",
        positions_provider=lambda: ["ADA/USDT:USDT"],
        open_orders_provider=lambda: [],
    )

    async def fake_fetch_income(self, *_args, **_kwargs):
        return [dict(ev) for ev in income_events]

    async def fake_fetch_trades(self, symbol, *_args, **_kwargs):
        return list(trades if symbol == "BTC/USDT:USDT" else [])

    async def fake_enrich(self, order_id, symbol):
        enrich_calls.append((order_id, symbol))
        return ("clid-0x0010", "entry_initial_normal_long")

    monkeypatch.setattr(
        fetcher,
        "_fetch_income",
        types.MethodType(fake_fetch_income, fetcher),
    )
    monkeypatch.setattr(
        fetcher,
        "_fetch_symbol_trades",
        types.MethodType(fake_fetch_trades, fetcher),
    )
    monkeypatch.setattr(
        fetcher,
        "_enrich_with_order_details",
        types.MethodType(fake_enrich, fetcher),
    )

    detail_cache: Dict[str, Tuple[str, str]] = {}
    events = await fetcher.fetch(
        since_ms=income_events[0]["timestamp"],
        until_ms=income_events[0]["timestamp"] + 1,
        detail_cache=detail_cache,
    )
    assert len(events) == 1
    event = events[0]
    assert enrich_calls == [("abc-789", "BTC/USDT:USDT")]
    assert event["client_order_id"] == "clid-0x0010"
    assert event["pb_order_type"] == "entry_initial_normal_long"
    assert detail_cache["trade-2"] == ("clid-0x0010", "entry_initial_normal_long")


@pytest.mark.asyncio
async def test_binance_fetcher_includes_trade_entries(monkeypatch):
    trades = [
        {
            "id": "entry-1",
            "timestamp": 1_700_000_200_000,
            "symbol": "ADA/USDT:USDT",
            "side": "buy",
            "amount": 5.0,
            "price": 0.65,
            "fee": {"currency": "USDT", "cost": 0.001},
            "clientOrderId": "x-entry-test-0x0007",
            "order": "order-123",
            "info": {"positionSide": "LONG"},
        }
    ]

    fetcher = BinanceFetcher(
        api=object(),
        symbol_resolver=lambda sym: sym or "",
        positions_provider=lambda: ["ADA/USDT:USDT"],
        open_orders_provider=lambda: [],
    )

    async def fake_fetch_income(self, *_args, **_kwargs):
        return []

    async def fake_fetch_trades(self, symbol, *_args, **_kwargs):
        return list(trades if symbol == "ADA/USDT:USDT" else [])

    monkeypatch.setattr(
        fetcher,
        "_fetch_income",
        types.MethodType(fake_fetch_income, fetcher),
    )
    monkeypatch.setattr(
        fetcher,
        "_fetch_symbol_trades",
        types.MethodType(fake_fetch_trades, fetcher),
    )

    detail_cache: Dict[str, Tuple[str, str]] = {}
    events = await fetcher.fetch(
        since_ms=trades[0]["timestamp"] - 1,
        until_ms=trades[0]["timestamp"] + 1,
        detail_cache=detail_cache,
    )

    assert len(events) == 1
    event = events[0]
    assert event["symbol"] == "ADA/USDT:USDT"
    assert event["qty"] == pytest.approx(5.0)
    assert event["pnl"] == pytest.approx(0.0)
    assert event["client_order_id"] == "x-entry-test-0x0007"
    assert event["pb_order_type"] == custom_id_to_snake("x-entry-test-0x0007")


@pytest.mark.asyncio
async def test_binance_fetcher_income_only_events(monkeypatch):
    """Test that income events without matching trades are still included."""
    income_events = [
        {
            "id": "income-only-1",
            "timestamp": 1_700_000_000_000,
            "datetime": "2023-11-01T00:00:00Z",
            "symbol": "SOL/USDT:USDT",
            "side": "",
            "qty": 0.0,
            "price": 0.0,
            "pnl": 5.0,
            "fees": None,
            "pb_order_type": "",
            "position_side": "long",
            "client_order_id": "",
        }
    ]

    fetcher = BinanceFetcher(
        api=object(),
        symbol_resolver=lambda sym: sym or "",
        positions_provider=lambda: [],
        open_orders_provider=lambda: [],
    )

    async def fake_fetch_income(self, *_args, **_kwargs):
        return [dict(ev) for ev in income_events]

    async def fake_fetch_trades(self, symbol, *_args, **_kwargs):
        return []  # No matching trades

    monkeypatch.setattr(
        fetcher,
        "_fetch_income",
        types.MethodType(fake_fetch_income, fetcher),
    )
    monkeypatch.setattr(
        fetcher,
        "_fetch_symbol_trades",
        types.MethodType(fake_fetch_trades, fetcher),
    )

    detail_cache: Dict[str, Tuple[str, str]] = {}
    events = await fetcher.fetch(
        since_ms=income_events[0]["timestamp"] - 1,
        until_ms=income_events[0]["timestamp"] + 1,
        detail_cache=detail_cache,
    )

    assert len(events) == 1
    event = events[0]
    assert event["id"] == "income-only-1"
    assert event["pnl"] == pytest.approx(5.0)
    assert event["position_side"] == "long"


@pytest.mark.asyncio
async def test_binance_fetcher_time_bounds_filtering(monkeypatch):
    """Test that events outside time bounds are filtered."""
    income_events = [
        {
            "id": "before-range",
            "timestamp": 1_700_000_000_000,
            "datetime": "",
            "symbol": "BTC/USDT:USDT",
            "side": "sell",
            "qty": 0.0,
            "price": 0.0,
            "pnl": 1.0,
            "fees": None,
            "pb_order_type": "",
            "position_side": "long",
            "client_order_id": "",
        },
        {
            "id": "in-range",
            "timestamp": 1_700_000_500_000,
            "datetime": "",
            "symbol": "BTC/USDT:USDT",
            "side": "sell",
            "qty": 0.0,
            "price": 0.0,
            "pnl": 2.0,
            "fees": None,
            "pb_order_type": "",
            "position_side": "long",
            "client_order_id": "",
        },
        {
            "id": "after-range",
            "timestamp": 1_700_001_000_000,
            "datetime": "",
            "symbol": "BTC/USDT:USDT",
            "side": "sell",
            "qty": 0.0,
            "price": 0.0,
            "pnl": 3.0,
            "fees": None,
            "pb_order_type": "",
            "position_side": "long",
            "client_order_id": "",
        },
    ]

    fetcher = BinanceFetcher(
        api=object(),
        symbol_resolver=lambda sym: sym or "",
        positions_provider=lambda: [],
        open_orders_provider=lambda: [],
    )

    async def fake_fetch_income(self, *_args, **_kwargs):
        return [dict(ev) for ev in income_events]

    async def fake_fetch_trades(self, symbol, *_args, **_kwargs):
        return []

    monkeypatch.setattr(
        fetcher,
        "_fetch_income",
        types.MethodType(fake_fetch_income, fetcher),
    )
    monkeypatch.setattr(
        fetcher,
        "_fetch_symbol_trades",
        types.MethodType(fake_fetch_trades, fetcher),
    )

    events = await fetcher.fetch(
        since_ms=1_700_000_100_000,
        until_ms=1_700_000_600_000,
        detail_cache={},
    )

    assert len(events) == 1
    assert events[0]["id"] == "in-range"


@pytest.mark.asyncio
async def test_binance_fetcher_position_side_from_trade(monkeypatch):
    """Test position side is captured from trade info."""
    income_events = [
        {
            "id": "trade-ps-1",
            "timestamp": 1_700_000_000_000,
            "datetime": "",
            "symbol": "ETH/USDT:USDT",
            "side": "sell",
            "qty": 0.0,
            "price": 0.0,
            "pnl": 10.0,
            "fees": None,
            "pb_order_type": "",
            "position_side": "unknown",  # Income doesn't have position side
            "client_order_id": "",
        }
    ]
    trades = [
        {
            "id": "trade-ps-1",
            "timestamp": 1_700_000_000_000,
            "symbol": "ETH/USDT:USDT",
            "side": "sell",
            "amount": 1.0,
            "price": 2000.0,
            "fee": {"currency": "USDT", "cost": 0.01},
            "order": "order-123",
            "info": {"positionSide": "SHORT"},  # Trade has position side
        }
    ]

    fetcher = BinanceFetcher(
        api=object(),
        symbol_resolver=lambda sym: sym or "",
        positions_provider=lambda: [],
        open_orders_provider=lambda: [],
    )

    async def fake_fetch_income(self, *_args, **_kwargs):
        return [dict(ev) for ev in income_events]

    async def fake_fetch_trades(self, symbol, *_args, **_kwargs):
        return list(trades if symbol == "ETH/USDT:USDT" else [])

    monkeypatch.setattr(
        fetcher,
        "_fetch_income",
        types.MethodType(fake_fetch_income, fetcher),
    )
    monkeypatch.setattr(
        fetcher,
        "_fetch_symbol_trades",
        types.MethodType(fake_fetch_trades, fetcher),
    )

    events = await fetcher.fetch(
        since_ms=income_events[0]["timestamp"] - 1,
        until_ms=income_events[0]["timestamp"] + 1,
        detail_cache={},
    )

    assert len(events) == 1
    event = events[0]
    # Position side should be taken from trade, normalized to lowercase
    assert event["position_side"] == "short"


@pytest.mark.asyncio
async def test_binance_fetcher_fees_from_trade(monkeypatch):
    """Test fees are captured from trades."""
    trades = [
        {
            "id": "fee-trade-1",
            "timestamp": 1_700_000_000_000,
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "amount": 0.01,
            "price": 40000.0,
            "fee": {"currency": "BNB", "cost": 0.0001},
            "order": "order-456",
            "info": {"positionSide": "LONG", "realizedPnl": "0"},
        }
    ]

    fetcher = BinanceFetcher(
        api=object(),
        symbol_resolver=lambda sym: sym or "",
        positions_provider=lambda: ["BTC/USDT:USDT"],
        open_orders_provider=lambda: [],
    )

    async def fake_fetch_income(self, *_args, **_kwargs):
        return []

    async def fake_fetch_trades(self, symbol, *_args, **_kwargs):
        return list(trades if symbol == "BTC/USDT:USDT" else [])

    monkeypatch.setattr(
        fetcher,
        "_fetch_income",
        types.MethodType(fake_fetch_income, fetcher),
    )
    monkeypatch.setattr(
        fetcher,
        "_fetch_symbol_trades",
        types.MethodType(fake_fetch_trades, fetcher),
    )

    events = await fetcher.fetch(
        since_ms=trades[0]["timestamp"] - 1,
        until_ms=trades[0]["timestamp"] + 1,
        detail_cache={},
    )

    assert len(events) == 1
    event = events[0]
    assert event["fees"] is not None
    assert event["fees"]["cost"] == pytest.approx(0.0001)
    assert event["fees"]["currency"] == "BNB"


@pytest.mark.asyncio
async def test_binance_fetcher_symbol_pool_from_providers(monkeypatch):
    """Test symbol pool is collected from positions, open orders, and income."""
    position_symbols = ["BTC/USDT:USDT"]
    order_symbols = ["ETH/USDT:USDT"]
    income_symbol = "SOL/USDT:USDT"

    income_events = [
        {
            "id": "income-sol-1",
            "timestamp": 1_700_000_000_000,
            "datetime": "",
            "symbol": income_symbol,
            "side": "",
            "qty": 0.0,
            "price": 0.0,
            "pnl": 1.0,
            "fees": None,
            "pb_order_type": "",
            "position_side": "long",
            "client_order_id": "",
        }
    ]

    fetched_symbols: List[str] = []

    fetcher = BinanceFetcher(
        api=object(),
        symbol_resolver=lambda sym: sym or "",
        positions_provider=lambda: position_symbols,
        open_orders_provider=lambda: order_symbols,
    )

    async def fake_fetch_income(self, *_args, **_kwargs):
        return [dict(ev) for ev in income_events]

    async def fake_fetch_trades(self, symbol, *_args, **_kwargs):
        fetched_symbols.append(symbol)
        return []

    monkeypatch.setattr(
        fetcher,
        "_fetch_income",
        types.MethodType(fake_fetch_income, fetcher),
    )
    monkeypatch.setattr(
        fetcher,
        "_fetch_symbol_trades",
        types.MethodType(fake_fetch_trades, fetcher),
    )

    await fetcher.fetch(
        since_ms=income_events[0]["timestamp"] - 1,
        until_ms=income_events[0]["timestamp"] + 1,
        detail_cache={},
    )

    # All three sources should be queried for trades
    assert "BTC/USDT:USDT" in fetched_symbols  # From positions
    assert "ETH/USDT:USDT" in fetched_symbols  # From open orders
    assert income_symbol in fetched_symbols  # From income events


@pytest.mark.asyncio
async def test_binance_fetcher_detail_cache_usage(monkeypatch):
    """Test detail cache is used and populated."""
    trades = [
        {
            "id": "cached-trade-1",
            "timestamp": 1_700_000_000_000,
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "amount": 0.1,
            "price": 50000.0,
            "order": "order-789",
            "info": {"positionSide": "LONG"},
        }
    ]

    fetcher = BinanceFetcher(
        api=object(),
        symbol_resolver=lambda sym: sym or "",
        positions_provider=lambda: ["BTC/USDT:USDT"],
        open_orders_provider=lambda: [],
    )

    async def fake_fetch_income(self, *_args, **_kwargs):
        return []

    async def fake_fetch_trades(self, symbol, *_args, **_kwargs):
        return list(trades if symbol == "BTC/USDT:USDT" else [])

    monkeypatch.setattr(
        fetcher,
        "_fetch_income",
        types.MethodType(fake_fetch_income, fetcher),
    )
    monkeypatch.setattr(
        fetcher,
        "_fetch_symbol_trades",
        types.MethodType(fake_fetch_trades, fetcher),
    )

    # Pre-populated cache
    detail_cache: Dict[str, Tuple[str, str]] = {
        "cached-trade-1": ("cached-client-id", "entry_initial_normal_long"),
    }

    events = await fetcher.fetch(
        since_ms=trades[0]["timestamp"] - 1,
        until_ms=trades[0]["timestamp"] + 1,
        detail_cache=detail_cache,
    )

    assert len(events) == 1
    event = events[0]
    assert event["client_order_id"] == "cached-client-id"
    assert event["pb_order_type"] == "entry_initial_normal_long"


# ---------------------------------------------------------------------------
# FillEvent tests
# ---------------------------------------------------------------------------


def test_fill_event_from_dict_requires_fields():
    with pytest.raises(ValueError):
        FillEvent.from_dict({})


def test_fill_event_from_dict_normalises_datetime():
    data = {
        "id": "t1",
        "timestamp": 1_000,
        "symbol": "BTC/USDT",
        "side": "buy",
        "qty": 1.0,
        "price": 10.0,
        "pnl": 0.5,
        "pb_order_type": "entry",
        "position_side": "long",
        "client_order_id": "cid",
    }
    event = FillEvent.from_dict(data)
    expected = (
        datetime.fromtimestamp(data["timestamp"], tz=timezone.utc).isoformat().replace("+00:00", "")
    )
    assert event.datetime == expected
    assert event.side == "buy"
    assert event.position_side == "long"


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------


def test_fill_event_cache_roundtrip(tmp_path: Path):
    cache_dir = tmp_path / "fills"
    cache = FillEventCache(cache_dir)

    events = [
        FillEvent.from_dict(
            dict(
                id="t1",
                timestamp=1_700_000_000_000,
                datetime="",
                symbol="A/B",
                side="buy",
                qty=1,
                price=2,
                pnl=0.0,
                pb_order_type="entry",
                position_side="long",
                client_order_id="cid1",
            )
        ),
        FillEvent.from_dict(
            dict(
                id="t0",
                timestamp=1_700_086_400_000,
                datetime="",
                symbol="A/B",
                side="sell",
                qty=1,
                price=2,
                pnl=0.0,
                pb_order_type="close",
                position_side="short",
                client_order_id="cid2",
            )
        ),
    ]

    cache.save(events)
    daily_files = sorted(p.name for p in cache_dir.glob("*.json"))
    assert daily_files == ["2023-11-14.json", "2023-11-15.json"]

    loaded = cache.load()
    assert [ev.id for ev in loaded] == ["t1", "t0"]


# ---------------------------------------------------------------------------
# BitgetFetcher tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bitget_fetcher_enriches_and_deduplicates(monkeypatch):
    fills = [
        [
            {
                "tradeId": "tid-1",
                "orderId": "oid-1",
                "cTime": "1000",
                "symbol": "BTCUSDT",
                "side": "buy",
                "baseVolume": "0.1",
                "price": "10",
                "profit": "1",
            },
            {
                "tradeId": "tid-2",
                "orderId": "oid-1",
                "cTime": "1500",
                "symbol": "BTCUSDT",
                "side": "sell",
                "baseVolume": "0.1",
                "price": "11",
                "profit": "-0.5",
            },
        ],
        [],
    ]
    api = _FakeBitgetAPI(fills)
    api.add_detail("oid-1", "0xabc")
    clock = _Clock(start_ms=0)

    async def fake_sleep(seconds: float):
        clock.advance(int(seconds * 1000))

    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda value: value)
    monkeypatch.setattr("src.fill_events_manager.asyncio.sleep", fake_sleep)

    resolver = lambda s: f"{s[:-4]}/USDT:USDT" if s and s.endswith("USDT") else s

    fetcher = BitgetFetcher(
        api,
        history_limit=2,
        detail_calls_per_minute=2,
        now_func=clock.now_ms,
        symbol_resolver=resolver,
    )

    detail_cache = {}
    batches: List[List[Dict[str, object]]] = []
    events = await fetcher.fetch(
        since_ms=500,
        until_ms=2000,
        detail_cache=detail_cache,
        on_batch=batches.append,
    )
    assert [ev["id"] for ev in events] == ["tid-1", "tid-2"]
    assert all(ev["client_order_id"] == "0xabc" for ev in events)
    assert all(ev["pb_order_type"] == "0xabc" for ev in events)
    assert all(ev["symbol"].endswith(":USDT") for ev in events)
    assert api.detail_calls, "Expected detail endpoint to be called"
    assert len(batches) == 1
    assert [ev["id"] for ev in batches[0]] == ["tid-1", "tid-2"]


@pytest.mark.asyncio
async def test_bitget_fetcher_paginates_across_sparse_history(monkeypatch):
    day = 24 * 60 * 60 * 1000
    recent = [
        {
            "tradeId": "tid-new-1",
            "orderId": "oid-new-1",
            "cTime": str(210 * day),
            "symbol": "BTCUSDT",
            "side": "buy",
            "baseVolume": "0.1",
            "price": "10",
            "profit": "0",
        },
        {
            "tradeId": "tid-new-2",
            "orderId": "oid-new-2",
            "cTime": str(209 * day),
            "symbol": "BTCUSDT",
            "side": "sell",
            "baseVolume": "0.1",
            "price": "11",
            "profit": "0",
        },
    ]
    sparse = [
        {
            "tradeId": "tid-gap",
            "orderId": "oid-gap",
            "cTime": str(150 * day),
            "symbol": "BTCUSDT",
            "side": "buy",
            "baseVolume": "0.05",
            "price": "9",
            "profit": "0",
        }
    ]
    older = [
        {
            "tradeId": "tid-old",
            "orderId": "oid-old",
            "cTime": str(20 * day),
            "symbol": "BTCUSDT",
            "side": "sell",
            "baseVolume": "0.2",
            "price": "8",
            "profit": "-1",
        }
    ]
    raw_batches = [recent, sparse, older, []]
    api = _FakeBitgetAPI([list(batch) for batch in raw_batches])

    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda value: value)

    resolver = lambda s: f"{s[:-4]}/USDT:USDT" if s and s.endswith("USDT") else s

    detail_cache = {
        entry["tradeId"]: (f"client-{entry['tradeId']}", f"type-{entry['tradeId']}")
        for bucket in raw_batches
        for entry in bucket
        if entry
    }

    start = 10 * day
    end = 220 * day

    fetcher = BitgetFetcher(
        api,
        history_limit=2,
        detail_calls_per_minute=5,
        now_func=lambda: end,
        symbol_resolver=resolver,
    )

    events = await fetcher.fetch(
        since_ms=start,
        until_ms=end,
        detail_cache=detail_cache,
        on_batch=None,
    )
    ids = [ev["id"] for ev in events]
    assert "tid-old" in ids
    assert len(api.history_calls) >= 3


@pytest.mark.asyncio
async def test_bitget_fetcher_handles_empty_batches(monkeypatch):
    day = 24 * 60 * 60 * 1000
    gap_batches = [
        [],
        [
            {
                "tradeId": "tid-gap-old",
                "orderId": "oid-gap-old",
                "cTime": str(15 * day),
                "symbol": "BTCUSDT",
                "side": "buy",
                "baseVolume": "0.1",
                "price": "10",
                "profit": "0",
            }
        ],
        [],
    ]
    api = _FakeBitgetAPI([list(batch) for batch in gap_batches])
    resolver = lambda s: f"{s[:-4]}/USDT:USDT" if s and s.endswith("USDT") else s
    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda value: value)

    fetcher = BitgetFetcher(
        api,
        history_limit=2,
        detail_calls_per_minute=5,
        now_func=lambda: 20 * day,
        symbol_resolver=resolver,
    )

    events = await fetcher.fetch(
        since_ms=10 * day,
        until_ms=20 * day,
        detail_cache={},
        on_batch=None,
    )

    assert any(ev["id"] == "tid-gap-old" for ev in events)
    assert len(api.history_calls) >= 2


@pytest.mark.asyncio
async def test_bitget_fetcher_reuses_detail_cache(monkeypatch):
    fills = [
        [
            {
                "tradeId": "tid-3",
                "orderId": "oid-3",
                "cTime": "3000",
                "symbol": "ETHUSDT",
                "side": "buy",
                "baseVolume": "1",
                "price": "100",
                "profit": "0",
            }
        ]
    ]
    api = _FakeBitgetAPI(fills)
    clock = _Clock(start_ms=0)

    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda value: value)

    async def noop_sleep(_):
        return None

    monkeypatch.setattr("src.fill_events_manager.asyncio.sleep", noop_sleep)

    resolver = lambda s: f"{s[:-4]}/USDT:USDT" if s and s.endswith("USDT") else s

    fetcher = BitgetFetcher(
        api,
        history_limit=1,
        detail_calls_per_minute=10,
        now_func=clock.now_ms,
        symbol_resolver=resolver,
    )

    batch_log: List[List[Dict[str, object]]] = []
    events = await fetcher.fetch(
        since_ms=None,
        until_ms=None,
        detail_cache={"tid-3": ("cached-client", "cached-type")},
        on_batch=batch_log.append,
    )
    assert len(events) == 1
    assert api.detail_calls == []
    assert events[0]["client_order_id"] == "cached-client"
    assert events[0]["pb_order_type"] == "cached-type"
    assert len(batch_log) == 1
    assert events[0]["symbol"].endswith(":USDT")


@pytest.mark.asyncio
async def test_bybit_fetcher_merges_pnl_and_batches(monkeypatch):
    """Test that close fills get PnL computed from avgEntryPrice."""
    base_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    trades_batches = [
        [
            {
                "id": "trade-1",
                "timestamp": base_ts + 1_000,
                "amount": 0.1,
                "price": 105.0,  # Exit price
                "side": "sell",  # Selling to close long
                "symbol": "BTC/USDT",
                "fee": {"cost": 0.0001, "currency": "USDT"},
                "info": {
                    "orderId": "order-1",
                    "orderLinkId": "0xabc",
                    "closedSize": "0.1",  # This is a close fill
                },
            }
        ]
    ]
    positions_batches = [
        [
            {
                "timestamp": base_ts + 1_500,
                "symbol": "BTC/USDT",
                "realizedPnl": "0.5",
                "info": {
                    "orderId": "order-1",
                    "side": "Sell",
                    "avgEntryPrice": "100.0",  # Entry at 100
                    "avgExitPrice": "105.0",  # Exit at 105
                    "closedSize": "0.1",
                    "closeFee": "0.0001",
                    "openFee": "0.0001",
                    "updatedTime": str(base_ts + 1_500),
                    "createdTime": str(base_ts + 1_000),
                },
            }
        ]
    ]

    api = _FakeBybitAPI(trades_batches, positions_batches)

    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda value: "close_grid_long")

    fetcher = BybitFetcher(api)
    batches: List[List[Dict[str, object]]] = []

    events = await fetcher.fetch(
        since_ms=base_ts,
        until_ms=base_ts + 2_000,
        detail_cache={},
        on_batch=batches.append,
    )

    assert len(events) == 1
    event = events[0]
    assert event["id"] == "trade-1"
    assert event["pb_order_type"] == "close_grid_long"
    # PnL = (105 - 100) * 0.1 - fees = 0.5 - 0.0002 = 0.4998
    assert event["pnl"] == pytest.approx(0.4998, rel=1e-3)
    assert event["client_order_id"] == "0xabc"
    assert event["symbol"] == "BTC/USDT"
    assert batches and batches[0][0]["id"] == "trade-1"


@pytest.mark.asyncio
async def test_bybit_fetcher_uses_detail_cache(monkeypatch):
    base_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    trades_batches = [
        [
            {
                "id": "trade-2",
                "timestamp": base_ts + 10_000,
                "amount": 0.2,
                "price": 120.0,
                "side": "sell",
                "symbol": "ETH/USDT",
                "info": {
                    "orderId": "order-2",
                    "closedSize": "0.1",
                },
            }
        ]
    ]
    positions_batches = [[]]

    api = _FakeBybitAPI(trades_batches, positions_batches)

    cache_entry = {"trade-2": ("cached-id", "cached-type")}

    fetcher = BybitFetcher(api)
    events = await fetcher.fetch(
        since_ms=base_ts,
        until_ms=base_ts + 20_000,
        detail_cache=cache_entry,
    )

    assert len(events) == 1
    event = events[0]
    assert event["client_order_id"] == "cached-id"
    assert event["pb_order_type"] == "cached-type"


@pytest.mark.asyncio
async def test_bybit_fetcher_distributes_pnl_across_fills():
    """Test that multiple close fills from same order each get computed PnL."""
    base_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    # Two fills from same order, closing a long position
    # avgEntryPrice = 11.0, exits at 10.0 and 10.5 -> losses
    trades_batches = [
        [
            {
                "id": "trade-ord-1",
                "timestamp": base_ts,
                "amount": 2.0,
                "price": 10.0,  # Exit price for first fill
                "side": "sell",
                "symbol": "LTC/USDT:USDT",
                "info": {
                    "orderId": "order-dist",
                    "closedSize": "2.0",  # Closes 2 units
                },
            },
            {
                "id": "trade-ord-2",
                "timestamp": base_ts + 1,
                "amount": 3.0,
                "price": 10.5,  # Exit price for second fill
                "side": "sell",
                "symbol": "LTC/USDT:USDT",
                "info": {
                    "orderId": "order-dist",
                    "closedSize": "3.0",  # Closes 3 units
                },
            },
        ]
    ]
    positions_batches = [
        [
            {
                "symbol": "LTC/USDT:USDT",
                "realizedPnl": "-5",
                "timestamp": base_ts + 5,
                "info": {
                    "orderId": "order-dist",
                    "avgEntryPrice": "11.0",  # Entered at 11.0
                    "avgExitPrice": "10.3",  # Weighted avg exit
                    "closedSize": "5",
                    "closeFee": "0.0",
                    "openFee": "0.0",
                    "updatedTime": str(base_ts + 5),
                    "createdTime": str(base_ts),
                },
            }
        ]
    ]

    api = _FakeBybitAPI(trades_batches, positions_batches)

    fetcher = BybitFetcher(api)
    events = await fetcher.fetch(
        since_ms=base_ts - 1_000,
        until_ms=base_ts + 10_000,
        detail_cache={},
    )

    assert len(events) == 2
    pnls = [event["pnl"] for event in events]
    # First fill: (10.0 - 11.0) * 2.0 = -2.0
    # Second fill: (10.5 - 11.0) * 3.0 = -1.5
    assert pnls[0] == pytest.approx(-2.0, rel=1e-3)
    assert pnls[1] == pytest.approx(-1.5, rel=1e-3)

    # Verify raw field includes positions_history data for close fills
    for event in events:
        raw_sources = [r["source"] for r in event["raw"]]
        assert "fetch_my_trades" in raw_sources
        assert "positions_history" in raw_sources
        # Verify positions_history contains expected data
        pos_hist = next(r for r in event["raw"] if r["source"] == "positions_history")
        assert pos_hist["data"]["info"]["avgEntryPrice"] == "11.0"


@pytest.mark.asyncio
async def test_bybit_fetcher_marks_unknown_for_manual():
    base_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    trades_batches = [
        [
            {
                "id": "trade-3",
                "timestamp": base_ts,
                "amount": 0.05,
                "price": 50.0,
                "side": "buy",
                "symbol": "ADA/USDT:USDT",
                "info": {
                    "orderId": "manual-order",
                },
            }
        ]
    ]
    positions_batches: List[List[Dict[str, Any]]] = [[]]

    api = _FakeBybitAPI(trades_batches, positions_batches)

    fetcher = BybitFetcher(api)
    events = await fetcher.fetch(
        since_ms=base_ts - 1_000,
        until_ms=base_ts + 1_000,
        detail_cache={},
    )

    assert len(events) == 1
    event = events[0]
    assert event["client_order_id"] == ""
    assert event["pb_order_type"] == "unknown"


@pytest.mark.asyncio
async def test_bybit_fetcher_no_pnl_without_matching_order():
    """Test that close fills without matching closed-pnl record get zero PnL.

    When a close fill's orderId doesn't match any closed-pnl record,
    we cannot compute PnL (no avgEntryPrice available), so it stays at 0.
    """
    base_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    # Trade with order ID not in positions (no matching closed-pnl)
    trades_batches = [
        [
            {
                "id": "trade-orphan",
                "timestamp": base_ts,
                "amount": 1.0,
                "price": 100.0,
                "side": "sell",
                "symbol": "BTC/USDT:USDT",
                "info": {
                    "orderId": "order-orphan",  # No matching position for this order
                    "closedSize": "1.0",
                },
            }
        ]
    ]
    # Position with different order ID (won't match)
    positions_batches = [
        [
            {
                "symbol": "BTC/USDT:USDT",
                "realizedPnl": "25.0",
                "timestamp": base_ts + 10,
                "info": {
                    "orderId": "order-other",  # Different order ID
                    "avgEntryPrice": "95.0",
                    "closedSize": "1.0",
                },
            }
        ]
    ]

    api = _FakeBybitAPI(trades_batches, positions_batches)

    fetcher = BybitFetcher(api)
    events = await fetcher.fetch(
        since_ms=base_ts - 1_000,
        until_ms=base_ts + 1_000,
        detail_cache={},
    )

    assert len(events) == 1
    event = events[0]
    # No matching closed-pnl record, so PnL stays at 0
    assert event["pnl"] == 0.0


@pytest.mark.asyncio
async def test_bybit_fetcher_position_side_from_closed_size():
    """Test position side is inferred from side + closedSize."""
    base_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    trades_batches = [
        [
            # Entry long (buy, closedSize=0)
            {
                "id": "trade-entry-long",
                "timestamp": base_ts,
                "amount": 1.0,
                "price": 100.0,
                "side": "buy",
                "symbol": "BTC/USDT:USDT",
                "info": {
                    "orderId": "order-1",
                    "closedSize": "0",
                },
            },
            # Close long (sell, closedSize>0)
            {
                "id": "trade-close-long",
                "timestamp": base_ts + 100,
                "amount": 1.0,
                "price": 110.0,
                "side": "sell",
                "symbol": "BTC/USDT:USDT",
                "info": {
                    "orderId": "order-2",
                    "closedSize": "1.0",
                },
            },
            # Entry short (sell, closedSize=0)
            {
                "id": "trade-entry-short",
                "timestamp": base_ts + 200,
                "amount": 1.0,
                "price": 105.0,
                "side": "sell",
                "symbol": "ETH/USDT:USDT",
                "info": {
                    "orderId": "order-3",
                    "closedSize": "0",
                },
            },
            # Close short (buy, closedSize>0)
            {
                "id": "trade-close-short",
                "timestamp": base_ts + 300,
                "amount": 1.0,
                "price": 100.0,
                "side": "buy",
                "symbol": "ETH/USDT:USDT",
                "info": {
                    "orderId": "order-4",
                    "closedSize": "1.0",
                },
            },
        ]
    ]
    positions_batches: List[List[Dict[str, Any]]] = [[]]

    api = _FakeBybitAPI(trades_batches, positions_batches)

    fetcher = BybitFetcher(api)
    events = await fetcher.fetch(
        since_ms=base_ts - 1_000,
        until_ms=base_ts + 1_000,
        detail_cache={},
    )

    events_by_id = {ev["id"]: ev for ev in events}

    assert events_by_id["trade-entry-long"]["position_side"] == "long"
    assert events_by_id["trade-close-long"]["position_side"] == "long"
    assert events_by_id["trade-entry-short"]["position_side"] == "short"
    assert events_by_id["trade-close-short"]["position_side"] == "short"


@pytest.mark.asyncio
async def test_bybit_fetcher_empty_batches():
    """Test handling when trades or positions are empty."""
    base_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    # Empty trades, has positions
    api1 = _FakeBybitAPI(
        [[]],
        [
            [
                {
                    "symbol": "BTC/USDT:USDT",
                    "realizedPnl": "10.0",
                    "timestamp": base_ts,
                    "info": {"orderId": "o1"},
                }
            ]
        ],
    )
    fetcher1 = BybitFetcher(api1)
    events1 = await fetcher1.fetch(since_ms=base_ts - 1000, until_ms=base_ts + 1000, detail_cache={})
    assert len(events1) == 0  # No trades means no events

    # Has trades, empty positions
    api2 = _FakeBybitAPI(
        [
            [
                {
                    "id": "t1",
                    "timestamp": base_ts,
                    "amount": 1.0,
                    "price": 100.0,
                    "side": "buy",
                    "symbol": "BTC/USDT:USDT",
                    "info": {"orderId": "o1"},
                }
            ]
        ],
        [[]],
    )
    fetcher2 = BybitFetcher(api2)
    events2 = await fetcher2.fetch(since_ms=base_ts - 1000, until_ms=base_ts + 1000, detail_cache={})
    assert len(events2) == 1
    assert events2[0]["pnl"] == pytest.approx(0.0)  # No position PnL to assign


@pytest.mark.asyncio
async def test_bybit_fetcher_time_bounds_filtering():
    """Test that events outside time bounds are filtered."""
    base_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    trades_batches = [
        [
            {
                "id": "trade-before",
                "timestamp": base_ts - 5000,
                "amount": 1.0,
                "price": 100.0,
                "side": "buy",
                "symbol": "BTC/USDT:USDT",
                "info": {},
            },
            {
                "id": "trade-in-range",
                "timestamp": base_ts,
                "amount": 1.0,
                "price": 100.0,
                "side": "buy",
                "symbol": "BTC/USDT:USDT",
                "info": {},
            },
            {
                "id": "trade-after",
                "timestamp": base_ts + 5000,
                "amount": 1.0,
                "price": 100.0,
                "side": "buy",
                "symbol": "BTC/USDT:USDT",
                "info": {},
            },
        ]
    ]
    positions_batches: List[List[Dict[str, Any]]] = [[]]

    api = _FakeBybitAPI(trades_batches, positions_batches)

    fetcher = BybitFetcher(api)
    events = await fetcher.fetch(
        since_ms=base_ts - 1000,
        until_ms=base_ts + 1000,
        detail_cache={},
    )

    assert len(events) == 1
    assert events[0]["id"] == "trade-in-range"


@pytest.mark.asyncio
async def test_bybit_fetcher_fees_captured():
    """Test that fees are captured from trades."""
    base_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    trades_batches = [
        [
            {
                "id": "trade-with-fee",
                "timestamp": base_ts,
                "amount": 1.0,
                "price": 50000.0,
                "side": "buy",
                "symbol": "BTC/USDT:USDT",
                "fee": {"currency": "USDT", "cost": 0.05},
                "info": {"orderId": "order-1"},
            }
        ]
    ]
    positions_batches: List[List[Dict[str, Any]]] = [[]]

    api = _FakeBybitAPI(trades_batches, positions_batches)

    fetcher = BybitFetcher(api)
    events = await fetcher.fetch(
        since_ms=base_ts - 1000,
        until_ms=base_ts + 1000,
        detail_cache={},
    )

    assert len(events) == 1
    assert events[0]["fees"] is not None
    assert events[0]["fees"]["cost"] == pytest.approx(0.05)
    assert events[0]["fees"]["currency"] == "USDT"


@pytest.mark.asyncio
async def test_hyperliquid_fetcher_basic(monkeypatch):
    base_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    trades_batches = [
        [
            {
                "id": "tid-hl-1",
                "timestamp": base_ts,
                "symbol": "BTC/USDC:USDC",
                "side": "buy",
                "amount": 0.1,
                "price": 30000.0,
                "info": {
                    "closedPnl": "1.5",
                    "dir": "Close Long",
                    "cloid": "0xabc",
                    "feeToken": "USDC",
                    "fee": "0.5",
                },
            },
            {
                "id": "tid-hl-1b",
                "timestamp": base_ts,
                "symbol": "BTC/USDC:USDC",
                "side": "buy",
                "amount": 0.2,
                "price": 30000.0,
                "info": {
                    "closedPnl": "0.5",
                    "dir": "Close Long",
                    "cloid": "0xabc",
                    "feeToken": "USDC",
                    "fee": "0.2",
                },
            },
        ]
    ]

    api = _FakeHyperliquidAPI(trades_batches)
    fetcher = HyperliquidFetcher(api, symbol_resolver=lambda s: s)
    events = await fetcher.fetch(
        since_ms=base_ts - 1,
        until_ms=base_ts + 10,
        detail_cache={},
    )
    assert len(events) == 1
    event = events[0]
    assert event["id"] == "tid-hl-1+tid-hl-1b"
    assert event["pb_order_type"] == "unknown"
    assert event["pnl"] == pytest.approx(2.0)
    assert event["qty"] == pytest.approx(0.30000000000000004)
    assert event["client_order_id"] == "0xabc"
    assert isinstance(event["fees"], dict)
    assert event["fees"]["cost"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_hyperliquid_fetcher_uses_cache():
    base_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    trades_batches = [
        [
            {
                "id": "tid-hl-2",
                "timestamp": base_ts,
                "symbol": "ETH/USDC:USDC",
                "side": "sell",
                "amount": 0.5,
                "price": 2000.0,
                "info": {"closedPnl": "-2.0", "dir": "Close Short"},
            }
        ]
    ]
    api = _FakeHyperliquidAPI(trades_batches)
    cache = {"tid-hl-2": ("cached-id", "entry_initial_normal_long")}
    fetcher = HyperliquidFetcher(api, symbol_resolver=lambda s: s)
    events = await fetcher.fetch(
        since_ms=None,
        until_ms=None,
        detail_cache=cache,
    )
    assert len(events) == 1
    event = events[0]
    assert event["client_order_id"] == "cached-id"
    assert event["pb_order_type"] == "entry_initial_normal_long"


# ---------------------------------------------------------------------------
# FillEventsManager tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_events() -> List[Dict[str, object]]:
    base_ts = int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp() * 1000)
    return [
        dict(
            id="tid-1",
            timestamp=base_ts,
            datetime="",
            symbol="BTC/USDT",
            side="buy",
            qty=0.1,
            price=10,
            pnl=1.0,
            pb_order_type="entry",
            position_side="long",
            client_order_id="cid-1",
        ),
        dict(
            id="tid-2",
            timestamp=base_ts + 1_000,
            datetime="",
            symbol="ETH/USDT",
            side="sell",
            qty=0.2,
            price=20,
            pnl=-0.5,
            pb_order_type="close",
            position_side="short",
            client_order_id="cid-2",
        ),
    ]


@pytest.mark.asyncio
async def test_manager_refresh_persists_and_queries(tmp_path: Path, sample_events):
    cache_dir = tmp_path / "fills"
    fetcher = _StaticFetcher(sample_events)
    manager = FillEventsManager(
        exchange="bitget",
        user="default",
        fetcher=fetcher,
        cache_path=cache_dir,
    )

    await manager.refresh()
    assert cache_dir.is_dir()
    assert list(cache_dir.glob("*.json"))

    assert pytest.approx(manager.get_pnl_sum()) == 0.5
    cumsum = manager.get_pnl_cumsum()
    assert cumsum[-1][1] == pytest.approx(0.5)
    last_ts = sample_events[-1]["timestamp"]
    assert manager.get_last_timestamp() == last_ts

    positions = manager.reconstruct_positions()
    assert positions["BTC/USDT:long"] == pytest.approx(0.1)
    assert positions["ETH/USDT:short"] == pytest.approx(-0.2)

    equity_curve = manager.reconstruct_equity_curve()
    assert equity_curve[-1][1] == pytest.approx(0.5)

    # Reload to ensure cache is used
    fetcher2 = _StaticFetcher([])
    manager2 = FillEventsManager(
        exchange="bitget",
        user="default",
        fetcher=fetcher2,
        cache_path=cache_dir,
    )
    await manager2.ensure_loaded()
    assert manager2.get_last_timestamp() == last_ts
    assert manager2.get_pnl_sum() == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_manager_refresh_latest_uses_overlap(tmp_path: Path, sample_events):
    cache_dir = tmp_path / "fills_latest"

    batches = [
        [
            dict(sample_events[0]),
            dict(sample_events[1]),
        ],
        [
            dict(
                id="tid-3",
                timestamp=sample_events[1]["timestamp"] + 500,
                datetime="",
                symbol="BTC/USDT",
                side="buy",
                qty=0.05,
                price=11,
                pnl=0.1,
                pb_order_type="entry",
                position_side="long",
                client_order_id="cid-3",
            )
        ],
    ]

    class _SequentialFetcher(BaseFetcher):
        def __init__(self, batches):
            self.batches = batches
            self.calls = []

        async def fetch(self, since_ms, until_ms, detail_cache, on_batch=None):
            self.calls.append((since_ms, until_ms))
            batch = self.batches.pop(0) if self.batches else []
            if on_batch and batch:
                on_batch(batch)
            return batch

    fetcher = _SequentialFetcher(batches.copy())
    manager = FillEventsManager(
        exchange="bitget",
        user="default",
        fetcher=fetcher,
        cache_path=cache_dir,
    )

    await manager.refresh()
    assert len(manager.get_events()) == 2
    assert len(fetcher.calls) == 1

    await manager.refresh_latest()
    assert len(manager.get_events()) == 3
    assert len(fetcher.calls) == 2
    assert fetcher.calls[1][0] == sample_events[0]["timestamp"]


@pytest.mark.asyncio
async def test_manager_refresh_range_detects_gaps(tmp_path: Path):
    cache_dir = tmp_path / "fills_range"
    cache = FillEventCache(cache_dir)
    base = int((datetime.now(timezone.utc) - timedelta(hours=48)).timestamp() * 1000)

    events = [
        FillEvent.from_dict(
            dict(
                id="gap-1",
                timestamp=base,
                datetime="",
                symbol="BTC/USDT",
                side="buy",
                qty=0.1,
                price=10,
                pnl=0.0,
                pb_order_type="entry",
                position_side="long",
                client_order_id="cid-gap-1",
            )
        ),
        FillEvent.from_dict(
            dict(
                id="gap-2",
                timestamp=base + 1_000,
                datetime="",
                symbol="BTC/USDT",
                side="sell",
                qty=0.1,
                price=11,
                pnl=0.0,
                pb_order_type="close",
                position_side="short",
                client_order_id="cid-gap-2",
            )
        ),
        FillEvent.from_dict(
            dict(
                id="gap-3",
                timestamp=base + 16 * 60 * 60 * 1000,
                datetime="",
                symbol="BTC/USDT",
                side="buy",
                qty=0.2,
                price=12,
                pnl=0.0,
                pb_order_type="entry",
                position_side="long",
                client_order_id="cid-gap-3",
            )
        ),
    ]

    cache.save(events)

    class _RecordingFetcher(BaseFetcher):
        def __init__(self):
            self.calls: List[Tuple[Optional[int], Optional[int]]] = []

        async def fetch(self, since_ms, until_ms, detail_cache, on_batch=None):
            self.calls.append((since_ms, until_ms))
            if on_batch:
                on_batch([])
            return []

    fetcher = _RecordingFetcher()
    manager = FillEventsManager(
        exchange="bitget",
        user="default",
        fetcher=fetcher,
        cache_path=cache_dir,
    )

    start_ms = base - int(6 * 60 * 60 * 1000)
    end_ms = base + int(24 * 60 * 60 * 1000)

    await manager.refresh_range(start_ms=start_ms, end_ms=end_ms, gap_hours=12, overlap=1)

    assert len(fetcher.calls) == 3
    assert fetcher.calls[0] == (start_ms, events[0].timestamp)
    assert fetcher.calls[1] == (events[1].timestamp, end_ms)
    assert fetcher.calls[2] == (events[2].timestamp, None)


@pytest.mark.asyncio
async def test_manager_persists_manual_fill(tmp_path: Path):
    cache_dir = tmp_path / "fills_manual"
    manual_event = dict(
        id="manual-trade",
        timestamp=int(datetime.now(timezone.utc).timestamp() * 1000),
        datetime="",
        symbol="ADA/USDT:USDT",
        side="buy",
        qty=0.1,
        price=0.5,
        pnl=0.0,
        pb_order_type="unknown",
        position_side="long",
        client_order_id="",
    )

    fetcher = _ManualFetcher([manual_event])
    manager = FillEventsManager(
        exchange="bitget",
        user="default",
        fetcher=fetcher,
        cache_path=cache_dir,
    )

    await manager.refresh()
    events = manager.get_events()
    assert len(events) == 1
    assert events[0].pb_order_type == "unknown"
    assert events[0].client_order_id == ""


@pytest.mark.asyncio
async def test_hyperliquid_fetcher_raises_after_max_rate_limit_retries(monkeypatch):
    class _RateLimitedHyperliquidAPI:
        def __init__(self):
            self.calls = 0

        async def fetch_my_trades(self, params: Dict[str, Any]):
            self.calls += 1
            raise RateLimitExceeded("429 too many requests")

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr("src.fill_events_manager.asyncio.sleep", _no_sleep)

    api = _RateLimitedHyperliquidAPI()
    fetcher = HyperliquidFetcher(api, symbol_resolver=lambda s: s)
    with pytest.raises(RateLimitExceeded, match="too many consecutive rate-limit retries"):
        await fetcher.fetch(since_ms=None, until_ms=None, detail_cache={})
    assert api.calls == 5


@pytest.mark.asyncio
async def test_manager_refresh_registers_gap_and_reraises_rate_limit(tmp_path: Path):
    cache_dir = tmp_path / "fills_rate_limit_gap"

    class _RateLimitedFetcher(BaseFetcher):
        async def fetch(self, since_ms, until_ms, detail_cache, on_batch=None):
            raise RateLimitExceeded("rate limited")

    manager = FillEventsManager(
        exchange="hyperliquid",
        user="default",
        fetcher=_RateLimitedFetcher(),
        cache_path=cache_dir,
    )

    start_ms = 1_700_000_000_000
    end_ms = start_ms + 60_000
    with pytest.raises(RateLimitExceeded):
        await manager.refresh(start_ms=start_ms, end_ms=end_ms)

    gaps = manager.cache.get_known_gaps()
    assert len(gaps) == 1
    assert gaps[0]["start_ts"] == start_ms
    assert gaps[0]["end_ts"] == end_ms
    assert gaps[0]["reason"] == GAP_REASON_FETCH_FAILED


# ---------------------------------------------------------------------------
# OkxFetcher tests
# ---------------------------------------------------------------------------


class _FakeOkxAPI:
    """Fake OKX API for testing OkxFetcher."""

    def __init__(
        self,
        fills_batches: List[List[Dict[str, Any]]],
        fills_history_batches: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> None:
        self._fills_batches = list(fills_batches)
        self._fills_history_batches = list(fills_history_batches or [])
        self.fills_calls: List[Dict[str, Any]] = []
        self.fills_history_calls: List[Dict[str, Any]] = []

    async def private_get_trade_fills(self, params: Dict[str, Any]):
        self.fills_calls.append(dict(params))
        if not self._fills_batches:
            return {"data": []}
        return {"data": self._fills_batches.pop(0)}

    async def private_get_trade_fills_history(self, params: Dict[str, Any]):
        self.fills_history_calls.append(dict(params))
        if not self._fills_history_batches:
            return {"data": []}
        return {"data": self._fills_history_batches.pop(0)}


def _make_okx_fill(
    trade_id: str,
    ts: int,
    inst_id: str = "BTC-USDT-SWAP",
    side: str = "buy",
    qty: str = "0.1",
    price: str = "50000",
    pnl: str = "0",
    pos_side: str = "long",
    cl_ord_id: str = "",
    bill_id: str = "",
) -> Dict[str, Any]:
    """Helper to create an OKX fill dict."""
    return {
        "tradeId": trade_id,
        "ordId": f"ord-{trade_id}",
        "billId": bill_id or trade_id,
        "ts": str(ts),
        "fillTime": str(ts),
        "instId": inst_id,
        "side": side,
        "fillSz": qty,
        "fillPx": price,
        "fillPnl": pnl,
        "posSide": pos_side,
        "clOrdId": cl_ord_id,
        "feeCcy": "USDT",
        "fee": "-0.01",
    }


@pytest.mark.asyncio
async def test_okx_fetcher_basic_fetch(monkeypatch):
    """Test basic fetch with all required fields present."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    fills = [
        _make_okx_fill(
            trade_id="tid-1",
            ts=now_ms - 1000,
            side="buy",
            qty="0.5",
            price="50000",
            pnl="0",
            pos_side="long",
            cl_ord_id="entry_initial_normal_long",
        ),
        _make_okx_fill(
            trade_id="tid-2",
            ts=now_ms - 500,
            side="sell",
            qty="0.5",
            price="51000",
            pnl="500",
            pos_side="long",
            cl_ord_id="close_grid_long",
        ),
    ]

    api = _FakeOkxAPI([fills])
    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda v: v)

    fetcher = OkxFetcher(api, trade_limit=100)
    detail_cache: Dict[str, Tuple[str, str]] = {}
    batches: List[List[Dict[str, object]]] = []

    events = await fetcher.fetch(
        since_ms=now_ms - 2000,
        until_ms=now_ms,
        detail_cache=detail_cache,
        on_batch=batches.append,
    )

    assert len(events) == 2
    # Check fields
    entry = events[0]
    assert entry["id"] == "tid-1"
    assert entry["symbol"] == "BTC/USDT:USDT"
    assert entry["side"] == "buy"
    assert entry["qty"] == pytest.approx(0.5)
    assert entry["price"] == pytest.approx(50000)
    assert entry["pnl"] == pytest.approx(0)
    assert entry["position_side"] == "long"
    assert entry["client_order_id"] == "entry_initial_normal_long"
    assert entry["pb_order_type"] == "entry_initial_normal_long"

    close = events[1]
    assert close["id"] == "tid-2"
    assert close["pnl"] == pytest.approx(500)
    assert close["pb_order_type"] == "close_grid_long"

    assert len(api.fills_calls) == 1
    assert len(batches) >= 1


@pytest.mark.asyncio
async def test_okx_fetcher_pagination_with_billid_cursor(monkeypatch):
    """Test pagination using billId cursor for backward traversal."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # First batch (most recent)
    batch1 = [
        _make_okx_fill(trade_id="tid-3", ts=now_ms - 100, bill_id="bill-3"),
        _make_okx_fill(trade_id="tid-2", ts=now_ms - 200, bill_id="bill-2"),
    ]
    # Second batch (older)
    batch2 = [
        _make_okx_fill(trade_id="tid-1", ts=now_ms - 300, bill_id="bill-1"),
    ]

    api = _FakeOkxAPI([batch1, batch2, []])
    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda v: v or "unknown")

    fetcher = OkxFetcher(api, trade_limit=2)
    detail_cache: Dict[str, Tuple[str, str]] = {}

    events = await fetcher.fetch(
        since_ms=now_ms - 500,
        until_ms=now_ms,
        detail_cache=detail_cache,
    )

    assert len(events) == 3
    assert [ev["id"] for ev in events] == ["tid-1", "tid-2", "tid-3"]

    # Check pagination cursor was used
    assert len(api.fills_calls) == 2
    # Second call should have 'after' param with billId from oldest fill in first batch
    assert "after" in api.fills_calls[1]
    assert api.fills_calls[1]["after"] == "bill-2"


@pytest.mark.asyncio
async def test_okx_fetcher_dual_endpoint_old_and_recent(monkeypatch):
    """Test using both /fills-history and /fills endpoints for date ranges spanning 3+ days."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    three_days_ms = 3 * 24 * 60 * 60 * 1000

    # Old data (before 3 days ago)
    old_ts = now_ms - three_days_ms - 1000
    old_fills = [
        _make_okx_fill(trade_id="tid-old", ts=old_ts, cl_ord_id="entry_old"),
    ]

    # Recent data (within last 3 days)
    recent_ts = now_ms - 1000
    recent_fills = [
        _make_okx_fill(trade_id="tid-recent", ts=recent_ts, cl_ord_id="entry_recent"),
    ]

    api = _FakeOkxAPI(
        fills_batches=[recent_fills],
        fills_history_batches=[old_fills],
    )
    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda v: v or "unknown")

    fetcher = OkxFetcher(api, trade_limit=100)
    detail_cache: Dict[str, Tuple[str, str]] = {}

    events = await fetcher.fetch(
        since_ms=old_ts - 1000,
        until_ms=now_ms,
        detail_cache=detail_cache,
    )

    assert len(events) == 2
    assert events[0]["id"] == "tid-old"
    assert events[1]["id"] == "tid-recent"

    # Both endpoints should have been called
    assert len(api.fills_history_calls) == 1
    assert len(api.fills_calls) == 1


@pytest.mark.asyncio
async def test_okx_fetcher_net_mode_position_side_inference(monkeypatch):
    """Test net mode position side inference from side + pnl."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    fills = [
        # Opening long: buy with no pnl -> position_side = long
        _make_okx_fill(
            trade_id="tid-open-long",
            ts=now_ms - 3000,
            side="buy",
            pnl="0",
            pos_side="net",
        ),
        # Closing long: sell with pnl -> position_side = long (was holding long)
        _make_okx_fill(
            trade_id="tid-close-long",
            ts=now_ms - 2000,
            side="sell",
            pnl="100",
            pos_side="net",
        ),
        # Opening short: sell with no pnl -> position_side = short
        _make_okx_fill(
            trade_id="tid-open-short",
            ts=now_ms - 1000,
            side="sell",
            pnl="0",
            pos_side="net",
        ),
        # Closing short: buy with pnl -> position_side = short (was holding short)
        _make_okx_fill(
            trade_id="tid-close-short",
            ts=now_ms,
            side="buy",
            pnl="50",
            pos_side="net",
        ),
    ]

    api = _FakeOkxAPI([fills])
    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda v: v or "unknown")

    fetcher = OkxFetcher(api, trade_limit=100)
    detail_cache: Dict[str, Tuple[str, str]] = {}

    events = await fetcher.fetch(
        since_ms=now_ms - 5000,
        until_ms=now_ms + 1000,
        detail_cache=detail_cache,
    )

    assert len(events) == 4

    # Check position side inference for net mode
    open_long = next(ev for ev in events if ev["id"] == "tid-open-long")
    assert open_long["position_side"] == "long"  # buy with no pnl = opening long

    close_long = next(ev for ev in events if ev["id"] == "tid-close-long")
    assert close_long["position_side"] == "long"  # sell with pnl = closing long

    open_short = next(ev for ev in events if ev["id"] == "tid-open-short")
    assert open_short["position_side"] == "short"  # sell with no pnl = opening short

    close_short = next(ev for ev in events if ev["id"] == "tid-close-short")
    assert close_short["position_side"] == "short"  # buy with pnl = closing short


@pytest.mark.asyncio
async def test_okx_fetcher_detail_cache_reuse(monkeypatch):
    """Test that detail cache is used to avoid recomputing pb_order_type."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    fills = [
        _make_okx_fill(
            trade_id="tid-cached",
            ts=now_ms - 1000,
            cl_ord_id="",  # No client order ID in fill
        ),
    ]

    api = _FakeOkxAPI([fills])

    # Pre-populate cache
    detail_cache: Dict[str, Tuple[str, str]] = {
        "tid-cached": ("cached-client-id", "cached-pb-type"),
    }

    fetcher = OkxFetcher(api, trade_limit=100)

    events = await fetcher.fetch(
        since_ms=now_ms - 2000,
        until_ms=now_ms,
        detail_cache=detail_cache,
    )

    assert len(events) == 1
    event = events[0]
    assert event["client_order_id"] == "cached-client-id"
    assert event["pb_order_type"] == "cached-pb-type"


@pytest.mark.asyncio
async def test_okx_fetcher_symbol_normalization(monkeypatch):
    """Test instId to CCXT symbol conversion."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    fills = [
        _make_okx_fill(trade_id="tid-btc", ts=now_ms - 300, inst_id="BTC-USDT-SWAP"),
        _make_okx_fill(trade_id="tid-eth", ts=now_ms - 200, inst_id="ETH-USDT-SWAP"),
        _make_okx_fill(trade_id="tid-sol", ts=now_ms - 100, inst_id="SOL-USDT-SWAP"),
    ]

    api = _FakeOkxAPI([fills])
    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda v: v or "unknown")

    fetcher = OkxFetcher(api, trade_limit=100)

    events = await fetcher.fetch(
        since_ms=now_ms - 1000,
        until_ms=now_ms,
        detail_cache={},
    )

    assert len(events) == 3
    assert events[0]["symbol"] == "BTC/USDT:USDT"
    assert events[1]["symbol"] == "ETH/USDT:USDT"
    assert events[2]["symbol"] == "SOL/USDT:USDT"


@pytest.mark.asyncio
async def test_okx_fetcher_hedge_mode(monkeypatch):
    """Test hedge mode with explicit long/short position sides."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    fills = [
        _make_okx_fill(
            trade_id="tid-long",
            ts=now_ms - 200,
            side="buy",
            pos_side="long",
        ),
        _make_okx_fill(
            trade_id="tid-short",
            ts=now_ms - 100,
            side="sell",
            pos_side="short",
        ),
    ]

    api = _FakeOkxAPI([fills])
    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda v: v or "unknown")

    fetcher = OkxFetcher(api, trade_limit=100)

    events = await fetcher.fetch(
        since_ms=now_ms - 1000,
        until_ms=now_ms,
        detail_cache={},
    )

    assert len(events) == 2
    assert events[0]["position_side"] == "long"
    assert events[1]["position_side"] == "short"


@pytest.mark.asyncio
async def test_okx_fetcher_short_batch_stops_pagination(monkeypatch):
    """Test that pagination stops when batch size is less than trade_limit."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # Single batch smaller than trade_limit - should not paginate
    batch = [
        _make_okx_fill(trade_id="tid-1", ts=now_ms - 100, bill_id="bill-1"),
    ]

    api = _FakeOkxAPI([batch])
    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda v: v or "unknown")

    fetcher = OkxFetcher(api, trade_limit=100)  # trade_limit=100, batch has 1

    events = await fetcher.fetch(
        since_ms=now_ms - 1000,
        until_ms=now_ms,
        detail_cache={},
    )

    assert len(events) == 1
    # Only one API call since batch was smaller than limit
    assert len(api.fills_calls) == 1


@pytest.mark.asyncio
async def test_okx_fetcher_time_bounds_filtering(monkeypatch):
    """Test that events outside time bounds are filtered."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    fills = [
        _make_okx_fill(trade_id="tid-before", ts=now_ms - 5000),  # Before since_ms
        _make_okx_fill(trade_id="tid-in-range", ts=now_ms - 500),  # In range
        _make_okx_fill(trade_id="tid-after", ts=now_ms + 1000),  # After until_ms
    ]

    api = _FakeOkxAPI([fills])
    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda v: v or "unknown")

    fetcher = OkxFetcher(api, trade_limit=100)

    events = await fetcher.fetch(
        since_ms=now_ms - 1000,  # Only include tid-in-range
        until_ms=now_ms,
        detail_cache={},
    )

    assert len(events) == 1
    assert events[0]["id"] == "tid-in-range"


@pytest.mark.asyncio
async def test_okx_fetcher_on_batch_callback(monkeypatch):
    """Test on_batch callback is called with each batch of events."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    batch1 = [
        _make_okx_fill(trade_id="tid-1", ts=now_ms - 200, bill_id="bill-1"),
        _make_okx_fill(trade_id="tid-2", ts=now_ms - 100, bill_id="bill-2"),
    ]

    api = _FakeOkxAPI([batch1, []])
    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda v: v or "unknown")

    fetcher = OkxFetcher(api, trade_limit=2)
    batches: List[List[Dict[str, object]]] = []

    await fetcher.fetch(
        since_ms=now_ms - 1000,
        until_ms=now_ms,
        detail_cache={},
        on_batch=batches.append,
    )

    assert len(batches) >= 1
    # First batch should have both events
    assert len(batches[0]) == 2


# ---------------------------------------------------------------------------
# KucoinFetcher tests
# ---------------------------------------------------------------------------


def test_kucoin_match_pnls_distributes_proportionally():
    """Test that _match_pnls distributes PnL proportionally across multiple fills."""
    # Mock fetcher (we only need the _match_pnls method)
    fetcher = KucoinFetcher(api=None)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # Create closing trades that match a single position close
    closes = [
        {
            "id": "close-1",
            "symbol": "BTC/USDT:USDT",
            "timestamp": now_ms,
            "qty": 80.0,  # 80% of position
        },
        {
            "id": "close-2",
            "symbol": "BTC/USDT:USDT",
            "timestamp": now_ms + 100,
            "qty": 20.0,  # 20% of position
        },
    ]

    # Position history entry with total PnL
    positions = [
        {
            "symbol": "BTC/USDT:USDT",
            "lastUpdateTimestamp": now_ms + 50,  # Between the two closes
            "realizedPnl": 100.0,  # Total PnL to distribute
        },
    ]

    # Events dict that _match_pnls will modify
    events = {
        "close-1": {"id": "close-1", "pnl": 0.0, "symbol": "BTC/USDT:USDT"},
        "close-2": {"id": "close-2", "pnl": 0.0, "symbol": "BTC/USDT:USDT"},
    }

    fetcher._match_pnls(closes, positions, events)

    # PnL should be distributed 80/20
    assert events["close-1"]["pnl"] == pytest.approx(80.0, rel=0.01)
    assert events["close-2"]["pnl"] == pytest.approx(20.0, rel=0.01)


def test_kucoin_match_pnls_handles_unmatched_closes():
    """Test that unmatched closes get PnL set to 0."""
    fetcher = KucoinFetcher(api=None)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # Close trade with no matching position
    closes = [
        {
            "id": "orphan-close",
            "symbol": "ETH/USDT:USDT",
            "timestamp": now_ms,
            "qty": 10.0,
        },
    ]

    # Position history from a different time window (>5 min away)
    positions = [
        {
            "symbol": "ETH/USDT:USDT",
            "lastUpdateTimestamp": now_ms - 10 * 60 * 1000,  # 10 min earlier
            "realizedPnl": 50.0,
        },
    ]

    events = {
        "orphan-close": {"id": "orphan-close", "pnl": 999.0, "symbol": "ETH/USDT:USDT"},
    }

    fetcher._match_pnls(closes, positions, events)

    # Unmatched close should have PnL set to 0
    assert events["orphan-close"]["pnl"] == 0.0


def test_kucoin_match_pnls_single_fill_gets_full_pnl():
    """Test that a single fill matching a position gets the full PnL."""
    fetcher = KucoinFetcher(api=None)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    closes = [
        {
            "id": "single-close",
            "symbol": "SOL/USDT:USDT",
            "timestamp": now_ms,
            "qty": 50.0,
        },
    ]

    positions = [
        {
            "symbol": "SOL/USDT:USDT",
            "lastUpdateTimestamp": now_ms + 10,
            "realizedPnl": 25.5,
        },
    ]

    events = {
        "single-close": {"id": "single-close", "pnl": 0.0, "symbol": "SOL/USDT:USDT"},
    }

    fetcher._match_pnls(closes, positions, events)

    assert events["single-close"]["pnl"] == pytest.approx(25.5)


def test_kucoin_match_pnls_multiple_positions_multiple_fills():
    """Test matching multiple position closes to their respective fills."""
    fetcher = KucoinFetcher(api=None)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    ten_min_ms = 10 * 60 * 1000  # 10 minutes to ensure clear separation

    closes = [
        # First position close fills (at time 0)
        {"id": "btc-close-1", "symbol": "BTC/USDT:USDT", "timestamp": now_ms, "qty": 10.0},
        {"id": "btc-close-2", "symbol": "BTC/USDT:USDT", "timestamp": now_ms + 50, "qty": 10.0},
        # Second position close fills (10 min later - well outside 5-min window)
        {
            "id": "btc-close-3",
            "symbol": "BTC/USDT:USDT",
            "timestamp": now_ms + ten_min_ms,
            "qty": 5.0,
        },
    ]

    positions = [
        # First position close (near time 0)
        {"symbol": "BTC/USDT:USDT", "lastUpdateTimestamp": now_ms + 25, "realizedPnl": 100.0},
        # Second position close (10 min later)
        {
            "symbol": "BTC/USDT:USDT",
            "lastUpdateTimestamp": now_ms + ten_min_ms + 10,
            "realizedPnl": 50.0,
        },
    ]

    events = {
        "btc-close-1": {"id": "btc-close-1", "pnl": 0.0, "symbol": "BTC/USDT:USDT"},
        "btc-close-2": {"id": "btc-close-2", "pnl": 0.0, "symbol": "BTC/USDT:USDT"},
        "btc-close-3": {"id": "btc-close-3", "pnl": 0.0, "symbol": "BTC/USDT:USDT"},
    }

    fetcher._match_pnls(closes, positions, events)

    # First position: 100 PnL distributed 50/50 (10+10 qty)
    assert events["btc-close-1"]["pnl"] == pytest.approx(50.0)
    assert events["btc-close-2"]["pnl"] == pytest.approx(50.0)
    # Second position: 50 PnL to single fill
    assert events["btc-close-3"]["pnl"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# GateioFetcher tests
# ---------------------------------------------------------------------------


class _FakeGateioAPI:
    """Fake GateIO API for testing GateioFetcher."""

    def __init__(
        self,
        trades_batches: List[List[Dict[str, Any]]],
        orders_batches: List[List[Dict[str, Any]]],
    ) -> None:
        self._trades_batches = list(trades_batches)
        self._orders_batches = list(orders_batches)
        self.trade_calls: List[Dict[str, Any]] = []
        self.order_calls: List[Dict[str, Any]] = []

    async def fetch_my_trades(self, params: Dict[str, Any] = None):
        self.trade_calls.append(dict(params or {}))
        if not self._trades_batches:
            return []
        return self._trades_batches.pop(0)

    async def private_futures_get_settle_my_trades_timerange(self, params: Dict[str, Any] = None):
        """Return trades in raw Gate.io format for the timerange endpoint."""
        self.trade_calls.append(dict(params or {}))
        if not self._trades_batches:
            return []
        ccxt_trades = self._trades_batches.pop(0)
        # Convert CCXT-like format to raw Gate.io format
        raw_trades = []
        for t in ccxt_trades:
            info = t.get("info", {})
            # Convert ms timestamp to seconds (float)
            ts_s = t.get("timestamp", 0) / 1000.0
            # Get size with sign from side
            size = float(info.get("size") or t.get("amount") or 0)
            if t.get("side") == "sell":
                size = -abs(size)
            # Convert symbol to contract format (BTC/USDT:USDT -> BTC_USDT)
            symbol = t.get("symbol", "")
            contract = symbol.replace("/", "_").replace(":USDT", "") if symbol else ""
            raw_trades.append(
                {
                    "trade_id": t.get("id") or info.get("trade_id"),
                    "order_id": t.get("order") or info.get("order_id"),
                    "create_time": str(ts_s),
                    "contract": contract,
                    "size": size,
                    "price": float(t.get("price") or info.get("price") or 0),
                    "fee": float(t.get("fee", {}).get("cost", 0)) if t.get("fee") else 0,
                    "text": info.get("text", ""),
                    "close_size": float(info.get("close_size", 0)),
                    "role": "maker",
                }
            )
        return raw_trades

    async def fetch_closed_orders(self, params: Dict[str, Any] = None):
        self.order_calls.append(dict(params or {}))
        if not self._orders_batches:
            return []
        return self._orders_batches.pop(0)


def _make_gateio_trade(
    trade_id: str,
    order_id: str,
    ts: int,
    symbol: str = "BTC/USDT:USDT",
    side: str = "buy",
    amount: float = 1.0,
    price: float = 50000.0,
    fee_cost: float = 0.01,
    text: str = "",
    close_size: float = 0.0,
) -> Dict[str, Any]:
    """Helper to create a GateIO trade dict."""
    return {
        "id": trade_id,
        "order": order_id,
        "symbol": symbol,
        "side": side,
        "amount": amount,
        "price": price,
        "timestamp": ts,
        "fee": {"currency": "USDT", "cost": fee_cost},
        "info": {
            "trade_id": trade_id,
            "order_id": order_id,
            "text": text,
            "close_size": str(close_size),
            "size": str(amount),
            "price": str(price),
        },
    }


def _make_gateio_order(
    order_id: str,
    ts: int,
    symbol: str = "BTC/USDT:USDT",
    side: str = "buy",
    amount: float = 1.0,
    pnl: float = 0.0,
    client_order_id: str = "",
    reduce_only: bool = False,
) -> Dict[str, Any]:
    """Helper to create a GateIO order dict."""
    return {
        "id": order_id,
        "symbol": symbol,
        "side": side,
        "amount": amount,
        "timestamp": ts,
        "clientOrderId": client_order_id,
        "reduceOnly": reduce_only,
        "info": {
            "id": order_id,
            "pnl": str(pnl),
            "text": client_order_id,
            "is_reduce_only": reduce_only,
        },
    }


@pytest.mark.asyncio
async def test_gateio_fetcher_merges_trades_with_orders(monkeypatch):
    """Test that trades are fetched and merged with order PnL."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    trades = [
        _make_gateio_trade(
            trade_id="trade-1",
            order_id="order-1",
            ts=now_ms - 1000,
            amount=1.0,
            text="entry_initial_normal_long",
        ),
        _make_gateio_trade(
            trade_id="trade-2",
            order_id="order-2",
            ts=now_ms - 500,
            side="sell",
            amount=1.0,
            text="close_grid_long",
            close_size=1.0,
        ),
    ]

    orders = [
        _make_gateio_order(
            order_id="order-1",
            ts=now_ms - 1000,
            pnl=0.0,
            client_order_id="entry_initial_normal_long",
        ),
        _make_gateio_order(
            order_id="order-2",
            ts=now_ms - 500,
            side="sell",
            pnl=10.5,
            client_order_id="close_grid_long",
            reduce_only=True,
        ),
    ]

    api = _FakeGateioAPI([trades], [orders])
    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda v: v or "unknown")

    fetcher = GateioFetcher(api, trade_limit=100)
    detail_cache: Dict[str, Tuple[str, str]] = {}

    events = await fetcher.fetch(
        since_ms=now_ms - 2000,
        until_ms=now_ms,
        detail_cache=detail_cache,
    )

    assert len(events) == 2

    # Check entry trade
    entry = events[0]
    assert entry["id"] == "trade-1"
    assert entry["pnl"] == pytest.approx(0.0)
    assert entry["fees"]["cost"] == pytest.approx(0.01)
    assert entry["client_order_id"] == "entry_initial_normal_long"
    assert entry["position_side"] == "long"

    # Check close trade
    close = events[1]
    assert close["id"] == "trade-2"
    assert close["pnl"] == pytest.approx(10.5)
    assert close["fees"]["cost"] == pytest.approx(0.01)
    assert close["position_side"] == "long"  # Closing a long position


@pytest.mark.asyncio
async def test_gateio_fetcher_distributes_pnl_proportionally(monkeypatch):
    """Test that order PnL is distributed across multiple fills."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # Two trades for one order
    trades = [
        _make_gateio_trade(
            trade_id="trade-1",
            order_id="order-1",
            ts=now_ms - 100,
            side="sell",
            amount=30.0,  # 30% of order
            close_size=30.0,
        ),
        _make_gateio_trade(
            trade_id="trade-2",
            order_id="order-1",
            ts=now_ms - 50,
            side="sell",
            amount=70.0,  # 70% of order
            close_size=70.0,
        ),
    ]

    orders = [
        _make_gateio_order(
            order_id="order-1",
            ts=now_ms - 100,
            side="sell",
            amount=100.0,
            pnl=100.0,  # Total PnL to distribute
            reduce_only=True,
        ),
    ]

    api = _FakeGateioAPI([trades], [orders])
    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda v: v or "unknown")

    fetcher = GateioFetcher(api, trade_limit=100)

    events = await fetcher.fetch(
        since_ms=now_ms - 1000,
        until_ms=now_ms,
        detail_cache={},
    )

    assert len(events) == 2

    # PnL should be distributed 30/70
    pnls = {ev["id"]: ev["pnl"] for ev in events}
    assert pnls["trade-1"] == pytest.approx(30.0)  # 30%
    assert pnls["trade-2"] == pytest.approx(70.0)  # 70%


@pytest.mark.asyncio
async def test_gateio_fetcher_uses_detail_cache(monkeypatch):
    """Test that detail cache is used and populated."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    trades = [
        _make_gateio_trade(
            trade_id="trade-cached",
            order_id="order-1",
            ts=now_ms - 100,
            text="",  # No text in trade
        ),
    ]

    orders = [
        _make_gateio_order(
            order_id="order-1",
            ts=now_ms - 100,
            client_order_id="",  # No client order ID
        ),
    ]

    api = _FakeGateioAPI([trades], [orders])

    # Pre-populated cache
    detail_cache: Dict[str, Tuple[str, str]] = {
        "trade-cached": ("cached-client-id", "cached-pb-type"),
    }

    fetcher = GateioFetcher(api, trade_limit=100)

    events = await fetcher.fetch(
        since_ms=now_ms - 1000,
        until_ms=now_ms,
        detail_cache=detail_cache,
    )

    assert len(events) == 1
    event = events[0]
    assert event["client_order_id"] == "cached-client-id"
    assert event["pb_order_type"] == "cached-pb-type"


@pytest.mark.asyncio
async def test_gateio_fetcher_captures_fees(monkeypatch):
    """Test that fees are captured from trades (not orders)."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    trades = [
        _make_gateio_trade(
            trade_id="trade-1",
            order_id="order-1",
            ts=now_ms - 100,
            fee_cost=0.05,
        ),
    ]

    orders = [
        _make_gateio_order(
            order_id="order-1",
            ts=now_ms - 100,
        ),
    ]

    api = _FakeGateioAPI([trades], [orders])
    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda v: v or "unknown")

    fetcher = GateioFetcher(api, trade_limit=100)

    events = await fetcher.fetch(
        since_ms=now_ms - 1000,
        until_ms=now_ms,
        detail_cache={},
    )

    assert len(events) == 1
    assert events[0]["fees"] is not None
    assert events[0]["fees"]["cost"] == pytest.approx(0.05)
    assert events[0]["fees"]["currency"] == "USDT"


@pytest.mark.asyncio
async def test_gateio_fetcher_position_side_inference(monkeypatch):
    """Test position side is inferred correctly from side and close status."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    trades = [
        # Entry long (buy, not close)
        _make_gateio_trade(
            trade_id="entry-long",
            order_id="order-1",
            ts=now_ms - 400,
            side="buy",
            close_size=0,
        ),
        # Close long (sell, is close)
        _make_gateio_trade(
            trade_id="close-long",
            order_id="order-2",
            ts=now_ms - 300,
            side="sell",
            close_size=1.0,
        ),
        # Entry short (sell, not close)
        _make_gateio_trade(
            trade_id="entry-short",
            order_id="order-3",
            ts=now_ms - 200,
            side="sell",
            close_size=0,
        ),
        # Close short (buy, is close)
        _make_gateio_trade(
            trade_id="close-short",
            order_id="order-4",
            ts=now_ms - 100,
            side="buy",
            close_size=1.0,
        ),
    ]

    orders = [
        _make_gateio_order(order_id="order-1", ts=now_ms - 400, side="buy", pnl=0),
        _make_gateio_order(
            order_id="order-2", ts=now_ms - 300, side="sell", pnl=5.0, reduce_only=True
        ),
        _make_gateio_order(order_id="order-3", ts=now_ms - 200, side="sell", pnl=0),
        _make_gateio_order(
            order_id="order-4", ts=now_ms - 100, side="buy", pnl=3.0, reduce_only=True
        ),
    ]

    api = _FakeGateioAPI([trades], [orders])
    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda v: v or "unknown")

    fetcher = GateioFetcher(api, trade_limit=100)

    events = await fetcher.fetch(
        since_ms=now_ms - 1000,
        until_ms=now_ms,
        detail_cache={},
    )

    events_by_id = {ev["id"]: ev for ev in events}

    assert events_by_id["entry-long"]["position_side"] == "long"
    assert events_by_id["close-long"]["position_side"] == "long"
    assert events_by_id["entry-short"]["position_side"] == "short"
    assert events_by_id["close-short"]["position_side"] == "short"


# ---------------------------------------------------------------------------
# compute_psize_pprice tests
# ---------------------------------------------------------------------------


def test_compute_psize_pprice_long_entries():
    """Test psize/pprice for consecutive long entries."""
    events = [
        {
            "id": "1",
            "timestamp": 1000,
            "symbol": "BTC",
            "side": "buy",
            "qty": 1.0,
            "price": 100.0,
            "position_side": "long",
        },
        {
            "id": "2",
            "timestamp": 2000,
            "symbol": "BTC",
            "side": "buy",
            "qty": 1.0,
            "price": 110.0,
            "position_side": "long",
        },
    ]
    ensure_qty_signage(events)
    compute_psize_pprice(events)

    assert events[0]["psize"] == 1.0
    assert events[0]["pprice"] == 100.0
    assert events[1]["psize"] == 2.0
    assert events[1]["pprice"] == 105.0  # VWAP: (1*100 + 1*110) / 2


def test_compute_psize_pprice_partial_close():
    """Test psize/pprice when partially closing a long position."""
    events = [
        {
            "id": "1",
            "timestamp": 1000,
            "symbol": "BTC",
            "side": "buy",
            "qty": 2.0,
            "price": 100.0,
            "position_side": "long",
        },
        {
            "id": "2",
            "timestamp": 2000,
            "symbol": "BTC",
            "side": "sell",
            "qty": 1.0,  # ensure_qty_signage will make this -1.0
            "price": 120.0,
            "position_side": "long",
        },
    ]
    ensure_qty_signage(events)
    compute_psize_pprice(events)

    assert events[0]["psize"] == 2.0
    assert events[0]["pprice"] == 100.0
    assert events[1]["psize"] == 1.0
    assert events[1]["pprice"] == 100.0  # VWAP unchanged on reduce


def test_compute_psize_pprice_full_close():
    """Test psize/pprice resets when position fully closes."""
    events = [
        {
            "id": "1",
            "timestamp": 1000,
            "symbol": "BTC",
            "side": "buy",
            "qty": 1.0,
            "price": 100.0,
            "position_side": "long",
        },
        {
            "id": "2",
            "timestamp": 2000,
            "symbol": "BTC",
            "side": "sell",
            "qty": 1.0,
            "price": 120.0,
            "position_side": "long",
        },
    ]
    ensure_qty_signage(events)
    compute_psize_pprice(events)

    assert events[0]["psize"] == 1.0
    assert events[0]["pprice"] == 100.0
    assert events[1]["psize"] == 0.0
    assert events[1]["pprice"] == 0.0


def test_compute_psize_pprice_short_entries():
    """Test psize/pprice for consecutive short entries."""
    events = [
        {
            "id": "1",
            "timestamp": 1000,
            "symbol": "ETH",
            "side": "sell",
            "qty": 2.0,
            "price": 200.0,
            "position_side": "short",
        },
        {
            "id": "2",
            "timestamp": 2000,
            "symbol": "ETH",
            "side": "sell",
            "qty": 2.0,
            "price": 220.0,
            "position_side": "short",
        },
    ]
    ensure_qty_signage(events)
    compute_psize_pprice(events)

    assert events[0]["psize"] == 2.0
    assert events[0]["pprice"] == 200.0
    assert events[1]["psize"] == 4.0
    assert events[1]["pprice"] == 210.0  # VWAP: (2*200 + 2*220) / 4


def test_compute_psize_pprice_short_partial_close():
    """Test psize/pprice when partially closing a short position."""
    events = [
        {
            "id": "1",
            "timestamp": 1000,
            "symbol": "ETH",
            "side": "sell",
            "qty": 4.0,
            "price": 200.0,
            "position_side": "short",
        },
        {
            "id": "2",
            "timestamp": 2000,
            "symbol": "ETH",
            "side": "buy",
            "qty": 2.0,  # buy reduces short
            "price": 180.0,
            "position_side": "short",
        },
    ]
    ensure_qty_signage(events)
    compute_psize_pprice(events)

    assert events[0]["psize"] == 4.0
    assert events[0]["pprice"] == 200.0
    assert events[1]["psize"] == 2.0
    assert events[1]["pprice"] == 200.0  # VWAP unchanged on reduce


def test_compute_psize_pprice_multiple_symbols():
    """Test psize/pprice handles multiple symbols independently."""
    events = [
        {
            "id": "1",
            "timestamp": 1000,
            "symbol": "BTC",
            "side": "buy",
            "qty": 1.0,
            "price": 100.0,
            "position_side": "long",
        },
        {
            "id": "2",
            "timestamp": 2000,
            "symbol": "ETH",
            "side": "buy",
            "qty": 10.0,
            "price": 50.0,
            "position_side": "long",
        },
        {
            "id": "3",
            "timestamp": 3000,
            "symbol": "BTC",
            "side": "buy",
            "qty": 1.0,
            "price": 120.0,
            "position_side": "long",
        },
    ]
    ensure_qty_signage(events)
    compute_psize_pprice(events)

    assert events[0]["psize"] == 1.0  # BTC
    assert events[0]["pprice"] == 100.0
    assert events[1]["psize"] == 10.0  # ETH
    assert events[1]["pprice"] == 50.0
    assert events[2]["psize"] == 2.0  # BTC again
    assert events[2]["pprice"] == 110.0  # VWAP: (1*100 + 1*120) / 2


def test_compute_psize_pprice_reopen_after_close():
    """Test psize/pprice when closing and reopening a position."""
    events = [
        {
            "id": "1",
            "timestamp": 1000,
            "symbol": "BTC",
            "side": "buy",
            "qty": 1.0,
            "price": 100.0,
            "position_side": "long",
        },
        {
            "id": "2",
            "timestamp": 2000,
            "symbol": "BTC",
            "side": "sell",
            "qty": 1.0,
            "price": 120.0,
            "position_side": "long",
        },
        {
            "id": "3",
            "timestamp": 3000,
            "symbol": "BTC",
            "side": "buy",
            "qty": 2.0,
            "price": 90.0,
            "position_side": "long",
        },
    ]
    ensure_qty_signage(events)
    compute_psize_pprice(events)

    assert events[0]["psize"] == 1.0
    assert events[0]["pprice"] == 100.0
    assert events[1]["psize"] == 0.0
    assert events[1]["pprice"] == 0.0
    assert events[2]["psize"] == 2.0
    assert events[2]["pprice"] == 90.0  # Fresh position, new pprice


def test_compute_psize_pprice_empty_events():
    """Test compute_psize_pprice handles empty list."""
    events: List[Dict[str, object]] = []
    result = compute_psize_pprice(events)
    assert result == {}


def test_compute_psize_pprice_initial_state():
    """Test compute_psize_pprice with initial position state."""
    events = [
        {
            "id": "1",
            "timestamp": 1000,
            "symbol": "BTC",
            "side": "buy",
            "qty": 1.0,
            "price": 120.0,
            "position_side": "long",
        },
    ]
    ensure_qty_signage(events)
    initial = {("BTC", "long"): (2.0, 100.0)}  # Starting with 2 BTC at $100
    result = compute_psize_pprice(events, initial)

    assert events[0]["psize"] == 3.0  # 2 + 1
    # VWAP: (2*100 + 1*120) / 3 = 320/3 ≈ 106.67
    assert abs(events[0]["pprice"] - 106.66666666666667) < 0.01
    assert result[("BTC", "long")] == (3.0, events[0]["pprice"])
