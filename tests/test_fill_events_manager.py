import asyncio
import errno
import json
import logging
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

import src.fill_events_manager as fem
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
    FillEventCacheContractError,
    FillEventCacheDiskFullError,
    FillEventsManager,
    PnlObservation,
    GAP_REASON_FETCH_FAILED,
    KUCOIN_POSITION_HISTORY_LOOKAHEAD_MS,
    apply_hyperliquid_raw_psize_overrides,
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
    def __init__(
        self,
        events: List[Dict[str, object]],
        observations: Optional[List[PnlObservation]] = None,
    ):
        self.events = list(events)
        self._observations = list(observations or [])
        self.pnl_observations: List[PnlObservation] = []
        self.calls: List[Tuple[Optional[int], Optional[int]]] = []

    async def fetch(self, since_ms, until_ms, detail_cache, on_batch=None):
        self.calls.append((since_ms, until_ms))
        self.pnl_observations = list(self._observations)
        payload = [dict(ev) for ev in self.events]
        if on_batch:
            on_batch(payload)
        return payload


class _SequencedFetcher(BaseFetcher):
    def __init__(
        self,
        batches: List[List[Dict[str, object]]],
        observations: Optional[List[List[PnlObservation]]] = None,
    ):
        self.batches = list(batches)
        self.observations = list(observations or [])
        self.pnl_observations: List[PnlObservation] = []
        self.calls: List[Tuple[Optional[int], Optional[int]]] = []

    async def fetch(self, since_ms, until_ms, detail_cache, on_batch=None):
        self.calls.append((since_ms, until_ms))
        batch = [dict(ev) for ev in (self.batches.pop(0) if self.batches else [])]
        self.pnl_observations = self.observations.pop(0) if self.observations else []
        if on_batch and batch:
            on_batch(batch)
        return batch


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

    async def fetch_my_trades(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.calls.append(
            {
                "symbol": symbol,
                "since": since,
                "limit": limit,
                "params": dict(params or {}),
            }
        )
        if not self._batches:
            return []
        return self._batches.pop(0)


class _FakeBinanceTradeAPI:
    def __init__(self, batches: List[List[Dict[str, Any]]]) -> None:
        self._batches = list(batches)
        self.calls: List[Dict[str, Any]] = []

    async def fetch_my_trades(
        self,
        symbol: str,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.calls.append(
            {
                "symbol": symbol,
                "limit": limit,
                "params": dict(params or {}),
            }
        )
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
async def test_binance_fetcher_merges_commission_income_into_realized_pnl(monkeypatch):
    ts = 1_700_000_000_000
    fetcher = BinanceFetcher(
        api=object(),
        symbol_resolver=lambda sym: "ADA/USDT:USDT" if sym == "ADAUSDT" else sym or "",
        positions_provider=lambda: [],
        open_orders_provider=lambda: [],
    )
    realized = fetcher._normalize_income(
        {
            "tradeId": "trade-commission",
            "time": ts,
            "symbol": "ADAUSDT",
            "income": "1.5",
            "asset": "USDT",
            "positionSide": "LONG",
        },
        income_type="REALIZED_PNL",
    )
    commission = fetcher._normalize_income(
        {
            "tradeId": "trade-commission",
            "time": ts,
            "symbol": "ADAUSDT",
            "income": "-0.003",
            "asset": "USDT",
            "positionSide": "LONG",
        },
        income_type="COMMISSION",
    )

    async def fake_fetch_income(self, *_args, **_kwargs):
        return [realized, commission]

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

    events = await fetcher.fetch(ts, ts + 1_000, detail_cache={})

    assert len(events) == 1
    assert events[0]["pnl"] == pytest.approx(1.5)
    assert events[0]["fees"] == [{"currency": "USDT", "fee_paid": -0.003}]
    fee_paid, meta = fem._normalize_fee_paid_from_payload(events[0])
    assert fee_paid == pytest.approx(-0.003)
    assert meta["fee_source"] == fem.FEE_SOURCE_REPORTED_QUOTE


@pytest.mark.asyncio
async def test_binance_fetch_symbol_trades_uses_now_bound_and_one_percent_margin():
    now_ms = 1_700_000_000_000
    seven_days_ms = 7 * 24 * 60 * 60 * 1000
    expected_span = int(seven_days_ms * 0.99)
    api = _FakeBinanceTradeAPI(batches=[[], []])
    fetcher = BinanceFetcher(
        api=api,
        symbol_resolver=lambda sym: sym or "",
        now_func=lambda: now_ms,
        trade_limit=1000,
    )

    trades = await fetcher._fetch_symbol_trades(
        "BTC/USDT:USDT",
        since_ms=now_ms - seven_days_ms,
        until_ms=None,
    )

    assert trades == []
    assert len(api.calls) == 2
    first_call = api.calls[0]["params"]
    second_call = api.calls[1]["params"]
    assert first_call["startTime"] == now_ms - seven_days_ms
    assert first_call["endTime"] - first_call["startTime"] == expected_span
    assert second_call["startTime"] == first_call["endTime"] + 1
    assert second_call["endTime"] == now_ms
    assert all(call["params"]["endTime"] <= now_ms for call in api.calls)


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
async def test_binance_fetcher_does_not_use_config_fallback_symbols_by_default(monkeypatch):
    fetched_symbols: List[str] = []

    fetcher = BinanceFetcher(
        api=object(),
        symbol_resolver=lambda sym: sym or "",
        positions_provider=lambda: [],
        open_orders_provider=lambda: [],
        fallback_symbols=["ADA/USDT:USDT", "BTC/USDT:USDT"],
    )

    async def fake_fetch_income(self, *_args, **_kwargs):
        return []

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

    events = await fetcher.fetch(
        since_ms=1_700_000_000_000,
        until_ms=1_700_000_060_000,
        detail_cache={},
    )

    assert events == []
    assert fetched_symbols == []


@pytest.mark.asyncio
async def test_binance_fetcher_uses_config_fallback_symbols_when_enabled(monkeypatch):
    fetched_symbols: List[str] = []

    fetcher = BinanceFetcher(
        api=object(),
        symbol_resolver=lambda sym: sym or "",
        positions_provider=lambda: [],
        open_orders_provider=lambda: [],
        fallback_symbols=["ADA/USDT:USDT", "BTC/USDT:USDT"],
        allow_fallback_symbols=True,
    )

    async def fake_fetch_income(self, *_args, **_kwargs):
        return []

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

    events = await fetcher.fetch(
        since_ms=1_700_000_000_000,
        until_ms=1_700_000_060_000,
        detail_cache={},
    )

    assert events == []
    assert fetched_symbols == ["ADA/USDT:USDT", "BTC/USDT:USDT"]


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
    assert event.pnl_status == "complete"


def test_fill_event_from_dict_preserves_pending_pnl_status():
    data = {
        "id": "t1",
        "timestamp": 1_000,
        "symbol": "BTC/USDT",
        "side": "sell",
        "qty": -1.0,
        "price": 10.0,
        "pnl": 0.0,
        "pnl_status": "pending",
        "pb_order_type": "close_grid_long",
        "position_side": "long",
        "client_order_id": "cid",
    }
    event = FillEvent.from_dict(data)
    assert event.pnl_status == "pending"
    assert event.pnl_pending is True


def test_fee_policy_uses_reported_quote_fee():
    fee_paid, meta = fem._normalize_fee_paid_from_payload(
        {
            "symbol": "BTC/USDT:USDT",
            "qty": 1.0,
            "price": 1000.0,
            "fees": {"currency": "USDT", "cost": 0.2},
        }
    )

    assert fee_paid == pytest.approx(-0.2)
    assert meta["fee_source"] == fem.FEE_SOURCE_REPORTED_QUOTE
    assert meta["fee_quality"] == fem.FEE_QUALITY_EXACT


def test_fee_policy_uses_scalar_quote_fee():
    for raw_fee, expected in [(0.25, -0.25), ("0.25", -0.25), (-0.05, 0.05)]:
        payload = {
            "symbol": "BTC/USDT:USDT",
            "qty": 1.0,
            "price": 1000.0,
            "fee": raw_fee,
        }

        assert fem.signed_fee_paid_from_payload(payload) == pytest.approx(expected)
        fee_paid, meta = fem._normalize_fee_paid_from_payload(payload)
        assert fee_paid == pytest.approx(expected)
        assert meta["fee_source"] == fem.FEE_SOURCE_REPORTED_QUOTE
        assert meta["fee_quality"] == fem.FEE_QUALITY_EXACT


def test_fee_policy_converts_reported_non_quote_fee_to_quote():
    fee_paid, meta = fem._normalize_fee_paid_from_payload(
        {
            "symbol": "BTC/USDT:USDT",
            "qty": 1.0,
            "price": 1000.0,
            "fees": {"currency": "BNB", "cost": 0.0001},
        },
        conversion_rates={"BNB": 300.0},
    )

    assert fee_paid == pytest.approx(-0.03)
    assert meta["fee_source"] == fem.FEE_SOURCE_REPORTED_CONVERTED
    assert meta["fee_quality"] == fem.FEE_QUALITY_CONVERTED
    assert meta["fee_conversion_source"] == "ticker"


def test_fee_policy_preserves_existing_conversion_metadata():
    fee_paid, meta = fem._normalize_fee_paid_from_payload(
        {
            "symbol": "BTC/USDT:USDT",
            "qty": 1.0,
            "price": 1000.0,
            "fee_paid": -0.03,
            "fee_source": fem.FEE_SOURCE_REPORTED_CONVERTED,
            "fee_quality": fem.FEE_QUALITY_CONVERTED,
            "fee_currency": "BNB",
            "fee_conversion_source": "ticker",
            "pnl_contract": fem.PNL_CONTRACT_CURRENT,
        }
    )

    assert fee_paid == pytest.approx(-0.03)
    assert meta["fee_source"] == fem.FEE_SOURCE_REPORTED_CONVERTED
    assert meta["fee_quality"] == fem.FEE_QUALITY_CONVERTED
    assert meta["fee_conversion_source"] == "ticker"


def test_fee_policy_falls_back_when_mixed_fee_has_unresolved_non_quote():
    fee_paid, meta = fem._normalize_fee_paid_from_payload(
        {
            "symbol": "BTC/USDT:USDT",
            "qty": 1.0,
            "price": 1000.0,
            "fees": [
                {"currency": "USDT", "cost": 0.10},
                {"currency": "BNB", "cost": 0.0001},
            ],
        },
        fee_pct_fallback=0.0002,
    )

    assert fee_paid == pytest.approx(-0.2)
    assert meta["fee_source"] == fem.FEE_SOURCE_FALLBACK_PCT
    assert meta["fee_quality"] == fem.FEE_QUALITY_FALLBACK
    assert meta["fee_unresolved_non_quote"] is True


def test_fee_policy_falls_back_to_configured_pct_with_contract_multiplier():
    fee_paid, meta = fem._normalize_fee_paid_from_payload(
        {
            "symbol": "BTC/USDT:USDT",
            "qty": 2.0,
            "price": 100.0,
            "c_mult": 10.0,
            "fees": {"currency": "BNB", "cost": 0.0001},
        },
        fee_pct_fallback=0.0002,
    )

    assert fee_paid == pytest.approx(-0.4)
    assert meta["fee_source"] == fem.FEE_SOURCE_FALLBACK_PCT
    assert meta["fee_notional"] == pytest.approx(2000.0)


def test_fee_policy_zero_fallback_means_net_equals_gross_when_fee_unresolved():
    fee_paid, meta = fem._normalize_fee_paid_from_payload(
        {
            "symbol": "BTC/USDT:USDT",
            "qty": 1.0,
            "price": 1000.0,
            "fees": {"currency": "BNB", "cost": 0.0001},
        },
        fee_pct_fallback=0.0,
    )

    assert fee_paid == pytest.approx(0.0)
    assert meta["fee_source"] == fem.FEE_SOURCE_FALLBACK_PCT


def test_fee_policy_sanity_replaces_absurd_reported_fee_with_fallback():
    fee_paid, meta = fem._normalize_fee_paid_from_payload(
        {
            "symbol": "BTC/USDT:USDT",
            "qty": 1.0,
            "price": 1000.0,
            "fees": {"currency": "USDT", "cost": 10.0},
        },
        fee_pct_fallback=0.0002,
        fee_pct_sanity_abs_max=0.001,
    )

    assert fee_paid == pytest.approx(-0.2)
    assert meta["fee_quality"] == fem.FEE_QUALITY_SANITY_REPLACED
    assert meta["fee_ratio"] == pytest.approx(-0.0002)
    assert meta["fee_original_paid"] == pytest.approx(-10.0)
    assert meta["fee_original_ratio"] == pytest.approx(-0.01)
    assert meta["fee_original_source"] == fem.FEE_SOURCE_REPORTED_QUOTE
    assert meta["fee_original_quality"] == fem.FEE_QUALITY_EXACT


@pytest.mark.asyncio
async def test_fee_policy_warning_dedupes_and_logs_original_sanity_ratio(
    tmp_path: Path, caplog
):
    manager = FillEventsManager(
        exchange="bybit",
        user="fee_user",
        fetcher=_StaticFetcher([]),
        cache_path=tmp_path / "fills",
        fee_pct_fallback=0.0002,
        fee_pct_sanity_abs_max=0.001,
    )
    raw = {
        "id": "duplicate-fee-event",
        "timestamp": 1_700_000_000_000,
        "symbol": "BTC/USDT:USDT",
        "qty": 1.0,
        "price": 1000.0,
        "fees": {"currency": "USDT", "cost": 10.0},
    }

    caplog.set_level(logging.WARNING, logger=fem.logger.name)
    await manager._apply_fee_policy_to_batch([dict(raw)])
    await manager._apply_fee_policy_to_batch([dict(raw)])

    warnings = [
        record.getMessage()
        for record in caplog.records
        if "fee policy used non-exact fee accounting" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "new_degraded=1" in warnings[0]
    assert "symbols=BTC/USDT:USDT:1" in warnings[0]
    assert "reasons=sanity_replaced:1" in warnings[0]
    assert "id=duplicate-fee-event" in warnings[0]
    assert "ratio=-0.00020000" in warnings[0]
    assert "orig_source=reported_quote" in warnings[0]
    assert "orig_ratio=-0.01000000" in warnings[0]
    assert "orig_paid=-10.00000000" in warnings[0]


def test_fee_policy_does_not_trust_fee_source_none_over_fee_entries():
    fee_paid, meta = fem._normalize_fee_paid_from_payload(
        {
            "symbol": "BTC/USDT:USDT",
            "qty": 1.0,
            "price": 1000.0,
            "fee_paid": 0.0,
            "fee_source": fem.FEE_SOURCE_NONE,
            "fees": {"currency": "USDT", "cost": 0.25},
        },
        fee_pct_fallback=0.0002,
    )

    assert fee_paid == pytest.approx(-0.25)
    assert meta["fee_source"] == fem.FEE_SOURCE_REPORTED_QUOTE
    assert meta["fee_quality"] == fem.FEE_QUALITY_EXACT


@pytest.mark.asyncio
async def test_fee_conversion_cache_does_not_reuse_stale_fill_age(tmp_path: Path):
    class _TickerApi:
        markets = {"BNB/USDT": {}}

        def __init__(self, ts: int) -> None:
            self.ts = ts
            self.calls = 0

        async def fetch_ticker(self, symbol: str):
            self.calls += 1
            return {"symbol": symbol, "timestamp": self.ts, "last": 300.0}

    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    fetcher = _StaticFetcher([])
    api = _TickerApi(now_ms)
    fetcher.api = api
    manager = FillEventsManager(
        exchange="binance",
        user="user",
        fetcher=fetcher,
        cache_path=tmp_path,
        fee_conversion_max_age_ms=24 * 60 * 60 * 1000,
    )

    stale_fill_ts = now_ms - 2 * 24 * 60 * 60 * 1000
    assert await manager._fee_conversion_rate("BNB", "USDT", stale_fill_ts) is None
    assert api.calls == 0

    assert await manager._fee_conversion_rate("BNB", "USDT", now_ms) == pytest.approx(300.0)
    assert api.calls == 1


@pytest.mark.asyncio
async def test_refresh_does_not_persist_out_of_range_fetcher_rows(tmp_path: Path):
    in_range_ts = 1_500
    manager = FillEventsManager(
        exchange="hyperliquid",
        user="user",
        fetcher=_StaticFetcher(
            [
                {
                    "id": "old",
                    "timestamp": 500,
                    "datetime": "",
                    "symbol": "BTC/USDC:USDC",
                    "side": "buy",
                    "qty": 1.0,
                    "price": 100.0,
                    "pnl": 0.0,
                    "fees": {"currency": "USDC", "cost": 0.01},
                    "pb_order_type": "unknown",
                    "position_side": "long",
                    "client_order_id": "",
                },
                {
                    "id": "current",
                    "timestamp": in_range_ts,
                    "datetime": "",
                    "symbol": "BTC/USDC:USDC",
                    "side": "buy",
                    "qty": 1.0,
                    "price": 100.0,
                    "pnl": 0.0,
                    "fees": {"currency": "USDC", "cost": 0.01},
                    "pb_order_type": "unknown",
                    "position_side": "long",
                    "client_order_id": "",
                },
            ]
        ),
        cache_path=tmp_path,
    )

    await manager.refresh(start_ms=1_000, end_ms=2_000)

    loaded = FillEventCache(tmp_path).load()
    assert [event.id for event in loaded] == ["current"]
    assert loaded[0].timestamp == in_range_ts


@pytest.mark.asyncio
async def test_refresh_range_empty_bounded_window_does_not_fetch_unbounded_latest(
    tmp_path: Path,
):
    fetcher = _StaticFetcher(
        [
            {
                "id": "old",
                "timestamp": 500,
                "datetime": "",
                "symbol": "BTC/USDC:USDC",
                "side": "buy",
                "qty": 1.0,
                "price": 100.0,
                "pnl": 0.0,
                "fees": {"currency": "USDC", "cost": 0.01},
                "pb_order_type": "unknown",
                "position_side": "long",
                "client_order_id": "",
            }
        ]
    )
    manager = FillEventsManager(
        exchange="hyperliquid",
        user="user",
        fetcher=fetcher,
        cache_path=tmp_path,
    )

    await manager.refresh_range(start_ms=1_000, end_ms=2_000)

    assert fetcher.calls == [(1_000, 2_000)]
    assert FillEventCache(tmp_path).load() == []


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
                c_mult=10.0,
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
    daily_files = sorted(p.name for p in cache_dir.glob("*.json") if p.name != "metadata.json")
    assert daily_files == ["2023-11-14.json", "2023-11-15.json"]

    loaded = cache.load()
    assert [ev.id for ev in loaded] == ["t1", "t0"]
    assert loaded[0].c_mult == pytest.approx(10.0)


def test_fill_event_cache_rejects_malformed_current_contract_record(tmp_path: Path):
    cache_dir = tmp_path / "fills"
    cache_dir.mkdir()
    (cache_dir / "metadata.json").write_text(
        json.dumps({"pnl_contract": fem.PNL_CONTRACT_CURRENT}),
        encoding="utf-8",
    )
    (cache_dir / "2026-02-02.json").write_text(
        json.dumps([{"id": "broken", "pnl_contract": fem.PNL_CONTRACT_CURRENT}]),
        encoding="utf-8",
    )

    with pytest.raises(FillEventCacheContractError, match="malformed"):
        FillEventCache(cache_dir).load()


@pytest.mark.asyncio
async def test_fill_event_cache_rejects_legacy_missing_pnl_contract(tmp_path: Path):
    cache_dir = tmp_path / "legacy_contract"
    cache_dir.mkdir()
    legacy_payload = [
        {
            "id": "legacy-entry",
            "timestamp": 1_700_000_000_000,
            "datetime": "",
            "symbol": "TON/USDT:USDT",
            "side": "buy",
            "qty": 1.0,
            "price": 10.0,
            "pnl": -0.01,
            "fees": {"currency": "USDT", "cost": 0.01},
            "pb_order_type": "entry_grid_long",
            "position_side": "long",
            "client_order_id": "cid-legacy",
            "raw": [],
        }
    ]
    (cache_dir / "2023-11-14.json").write_text(json.dumps(legacy_payload), encoding="utf-8")
    manager = FillEventsManager(
        exchange="kucoin",
        user="default",
        fetcher=_StaticFetcher([]),
        cache_path=cache_dir,
        fee_pct_sanity_abs_max=1.0,
    )

    with pytest.raises(FillEventCacheContractError, match="legacy or missing pnl_contract"):
        await manager.ensure_loaded()


@pytest.mark.asyncio
async def test_doctor_reports_unsupported_legacy_cache_without_raising(tmp_path: Path):
    cache_dir = tmp_path / "unsupported_legacy_contract"
    cache_dir.mkdir()
    legacy_payload = [
        {
            "id": "legacy-bitget-entry",
            "timestamp": 1_700_000_000_000,
            "datetime": "",
            "symbol": "TON/USDT:USDT",
            "side": "buy",
            "qty": 1.0,
            "price": 10.0,
            "pnl": 0.0,
            "fees": {"currency": "USDT", "cost": 0.01},
            "pb_order_type": "entry_grid_long",
            "position_side": "long",
            "client_order_id": "cid-legacy",
            "raw": [],
        }
    ]
    (cache_dir / "2023-11-14.json").write_text(json.dumps(legacy_payload), encoding="utf-8")
    manager = FillEventsManager(
        exchange="bitget",
        user="default",
        fetcher=_StaticFetcher([]),
        cache_path=cache_dir,
        fee_pct_sanity_abs_max=1.0,
    )

    report = await manager.run_doctor(auto_repair=True)

    assert report["legacy_contract"] is True
    assert report["unsupported_legacy_contract"] is True
    assert report["repaired"] is False
    assert report["action"] == "rebuild_cache"
    assert "startup auto-migration" in report["message"]
    assert report["anomaly_events"] == 1
    assert report["anomaly_examples"] == ["legacy-bitget-entry"]
    assert report["anomaly_events_after"] == 1


@pytest.mark.asyncio
async def test_doctor_repairs_metadata_only_legacy_cache_for_any_exchange(tmp_path: Path):
    cache_dir = tmp_path / "metadata_only_legacy"
    cache_dir.mkdir()
    event = _kucoin_manager_fill(
        "current-row",
        1_700_000_000_000,
        side="buy",
        qty=1.0,
        price=10.0,
        fees={"currency": "USDT", "cost": 0.1},
    )
    event["pnl_contract"] = fem.PNL_CONTRACT_CURRENT
    event["fee_paid"] = -0.1
    (cache_dir / "2023-11-14.json").write_text(json.dumps([event]), encoding="utf-8")
    manager = FillEventsManager(
        exchange="hyperliquid",
        user="default",
        fetcher=_StaticFetcher([]),
        cache_path=cache_dir,
    )

    report = await manager.run_doctor(auto_repair=True)

    assert report["legacy_contract"] is True
    assert report["repaired"] is True
    assert FillEventCache(cache_dir).load_metadata()["pnl_contract"] == fem.PNL_CONTRACT_CURRENT
    assert [ev.id for ev in FillEventCache(cache_dir).load()] == ["current-row"]


def test_fill_event_cache_quarantine_for_rebuild_moves_legacy_payload(tmp_path: Path):
    cache_dir = tmp_path / "fills"
    cache_dir.mkdir()
    (cache_dir / "2023-11-14.json").write_text("[]", encoding="utf-8")
    cache = FillEventCache(cache_dir)

    quarantined = cache.quarantine_for_rebuild(reason="legacy pnl contract")

    assert quarantined is not None
    quarantine_path = Path(str(quarantined))
    assert quarantine_path.exists()
    assert (quarantine_path / "2023-11-14.json").exists()
    assert cache_dir.exists()
    assert list(cache_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_kucoin_doctor_repairs_legacy_contract_with_backup(tmp_path: Path):
    cache_dir = tmp_path / "legacy_repair"
    cache_dir.mkdir()
    ts = 1_700_000_000_000
    legacy_payload = [
        _kucoin_manager_fill(
            "legacy-entry",
            ts,
            side="buy",
            qty=1.0,
            price=10.0,
            fees={"currency": "USDT", "cost": 0.1},
        ),
        _kucoin_manager_fill(
            "legacy-close",
            ts + 60_000,
            side="sell",
            qty=-1.0,
            price=12.0,
            fees={"currency": "USDT", "cost": 0.2},
        ),
    ]
    legacy_payload[0]["pnl"] = -0.1
    legacy_payload[1]["pnl"] = 1.7
    (cache_dir / "2023-11-14.json").write_text(json.dumps(legacy_payload), encoding="utf-8")
    manager = FillEventsManager(
        exchange="kucoin",
        user="default",
        fetcher=_StaticFetcher([]),
        cache_path=cache_dir,
        fee_pct_sanity_abs_max=1.0,
    )

    report = await manager.run_doctor(auto_repair=True)

    assert report["legacy_contract"] is True
    assert report["repaired"] is True
    assert report["backup_path"]
    assert Path(str(report["backup_path"])).exists()
    repaired = FillEventCache(cache_dir).load()
    by_id = {event.id: event for event in repaired}
    assert by_id["legacy-entry"].pnl == pytest.approx(0.0)
    assert by_id["legacy-entry"].fee_paid == pytest.approx(-0.1)
    assert by_id["legacy-close"].pnl == pytest.approx(2.0)
    assert by_id["legacy-close"].fee_paid == pytest.approx(-0.2)


@pytest.mark.asyncio
async def test_kucoin_doctor_clean_cache_does_not_report_repaired(tmp_path: Path):
    cache_dir = tmp_path / "clean_kucoin"
    cache_dir.mkdir()
    event = _kucoin_manager_fill(
        "clean-entry",
        1_700_000_000_000,
        side="buy",
        qty=1.0,
        price=10.0,
        fees={"currency": "USDT", "cost": 0.1},
    )
    event["pnl_contract"] = fem.PNL_CONTRACT_CURRENT
    event["fee_paid"] = -0.1
    (cache_dir / "2023-11-14.json").write_text(json.dumps([event]), encoding="utf-8")
    (cache_dir / "metadata.json").write_text(
        json.dumps({"pnl_contract": fem.PNL_CONTRACT_CURRENT}),
        encoding="utf-8",
    )
    manager = FillEventsManager(
        exchange="kucoin",
        user="default",
        fetcher=_StaticFetcher([]),
        cache_path=cache_dir,
    )

    report = await manager.run_doctor(auto_repair=False)

    assert report["anomaly_events"] == 0
    assert report["repaired"] is False


@pytest.mark.asyncio
async def test_kucoin_doctor_repair_applies_contract_multiplier(tmp_path: Path):
    cache_dir = tmp_path / "legacy_repair_c_mult"
    cache_dir.mkdir()
    ts = 1_700_000_000_000
    legacy_payload = [
        _kucoin_manager_fill(
            "legacy-entry",
            ts,
            side="buy",
            qty=1.0,
            price=10.0,
            fees={"currency": "USDT", "cost": 0.1},
        ),
        _kucoin_manager_fill(
            "legacy-close",
            ts + 60_000,
            side="sell",
            qty=-1.0,
            price=12.0,
            fees={"currency": "USDT", "cost": 0.2},
        ),
    ]
    for row in legacy_payload:
        row["c_mult"] = 5.0
    (cache_dir / "2023-11-14.json").write_text(json.dumps(legacy_payload), encoding="utf-8")
    manager = FillEventsManager(
        exchange="kucoin",
        user="default",
        fetcher=_StaticFetcher([]),
        cache_path=cache_dir,
        fee_pct_sanity_abs_max=1.0,
    )

    report = await manager.run_doctor(auto_repair=True)

    assert report["repaired"] is True
    repaired = FillEventCache(cache_dir).load()
    by_id = {event.id: event for event in repaired}
    assert by_id["legacy-entry"].c_mult == pytest.approx(5.0)
    assert by_id["legacy-close"].c_mult == pytest.approx(5.0)
    assert by_id["legacy-close"].pnl == pytest.approx(10.0)


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
async def test_bitget_fetcher_preserves_signed_fee_detail(monkeypatch):
    fills = [
        [
            {
                "tradeId": "tid-fee",
                "orderId": "oid-fee",
                "cTime": "1000",
                "symbol": "XLMUSDT",
                "side": "buy",
                "baseVolume": "174",
                "price": "0.22375",
                "profit": "0",
                "feeDetail": [
                    {
                        "feeCoin": "USDT",
                        "totalFee": "-0.0077865",
                        "totalDeductionFee": "0",
                    }
                ],
            }
        ]
    ]
    api = _FakeBitgetAPI(fills)
    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda value: value)
    resolver = lambda s: f"{s[:-4]}/USDT:USDT" if s and s.endswith("USDT") else s
    fetcher = BitgetFetcher(api, symbol_resolver=resolver)

    events = await fetcher.fetch(
        since_ms=500,
        until_ms=2000,
        detail_cache={"tid-fee": ("client", "entry_grid_long")},
    )

    assert len(events) == 1
    fee_paid, meta = fem._normalize_fee_paid_from_payload(events[0])
    assert fee_paid == pytest.approx(-0.0077865)
    assert meta["fee_source"] == fem.FEE_SOURCE_REPORTED_QUOTE
    assert meta["fee_quality"] == fem.FEE_QUALITY_EXACT


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
    # Canonical PnL is gross; fees are carried separately as signed fee_paid.
    assert event["pnl"] == pytest.approx(0.5, rel=1e-3)
    assert FillEvent.from_dict(event).fee_paid == pytest.approx(-0.0001)
    assert event["pnl_status"] == "complete"
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
    assert event["pnl_status"] == "pending"


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
    assert [event["pnl_status"] for event in events] == ["complete", "complete"]

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
async def test_bybit_fetcher_deduplicates_duplicate_exec_ids_before_coalescing():
    """Duplicate pages must not inflate canonical qty for the same execId."""
    base_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    duplicated_trade = {
        "id": "exec-dup",
        "timestamp": base_ts,
        "amount": 0.06,
        "price": 346.47,
        "side": "sell",
        "symbol": "XMR/USDT:USDT",
        "info": {
            "orderId": "order-dup",
            "orderLinkId": "0x0018004f2efba1b246dfb30ed39e7060eb",
            "closedSize": "0.06",
            "execQty": "0.06",
            "orderQty": "13.26",
        },
    }

    # Simulate Bybit pagination returning the same fill in two pages.
    api = _FakeBybitAPI(
        trades_batches=[[dict(duplicated_trade)], [dict(duplicated_trade)], []],
        positions_batches=[[]],
    )
    fetcher = BybitFetcher(api, trade_limit=1)

    events = await fetcher.fetch(
        since_ms=base_ts - 1_000,
        until_ms=base_ts + 1_000,
        detail_cache={},
    )

    assert len(events) == 1
    event = events[0]
    assert event["qty"] == pytest.approx(0.06)
    assert event["side"] == "sell"


@pytest.mark.asyncio
async def test_fill_events_manager_bybit_doctor_detects_cross_event_duplicates(tmp_path: Path):
    cache_path = tmp_path / "fills_doctor_detect"
    ts = 1_770_000_000_000

    trade_a = {
        "id": "exec-a",
        "timestamp": ts,
        "amount": 0.4,
        "price": 100.0,
        "side": "sell",
        "symbol": "BTC/USDT:USDT",
        "info": {"execId": "exec-a", "orderId": "order-1", "execQty": "0.4", "closedSize": "0.4"},
    }
    trade_b = {
        "id": "exec-b",
        "timestamp": ts,
        "amount": 0.6,
        "price": 101.0,
        "side": "sell",
        "symbol": "BTC/USDT:USDT",
        "info": {"execId": "exec-b", "orderId": "order-1", "execQty": "0.6", "closedSize": "0.6"},
    }

    events = [
        FillEvent.from_dict(
            {
                "id": "exec-a+exec-b",
                "source_ids": ["exec-a", "exec-b"],
                "timestamp": ts,
                "datetime": "2026-02-28T00:00:00",
                "symbol": "BTC/USDT:USDT",
                "side": "sell",
                "qty": -1.0,
                "price": 100.6,
                "pnl": -1.0,
                "fees": {"currency": "USDT", "cost": 0.05},
                "pb_order_type": "close_auto_reduce_wel_long",
                "position_side": "long",
                "client_order_id": "0xabc",
                "raw": [
                    {"source": "fetch_my_trades", "data": dict(trade_a)},
                    {"source": "fetch_my_trades", "data": dict(trade_b)},
                ],
            }
        ),
        FillEvent.from_dict(
            {
                "id": "exec-a",
                "source_ids": ["exec-a"],
                "timestamp": ts,
                "datetime": "2026-02-28T00:00:00",
                "symbol": "BTC/USDT:USDT",
                "side": "sell",
                "qty": -0.4,
                "price": 100.0,
                "pnl": -0.4,
                "fees": {"currency": "USDT", "cost": 0.02},
                "pb_order_type": "close_auto_reduce_wel_long",
                "position_side": "long",
                "client_order_id": "0xabc",
                "raw": [{"source": "fetch_my_trades", "data": dict(trade_a)}],
            }
        ),
    ]
    FillEventCache(cache_path).save(events)

    manager = FillEventsManager(
        exchange="bybit",
        user="u",
        fetcher=_StaticFetcher([]),
        cache_path=cache_path,
    )
    await manager.ensure_loaded()

    report = await manager.run_doctor(auto_repair=False)
    assert report["anomaly_events"] > 0
    assert report["repaired"] is False


@pytest.mark.asyncio
async def test_fill_events_manager_bybit_doctor_auto_repairs_cross_event_duplicates(tmp_path: Path):
    cache_path = tmp_path / "fills_doctor_repair"
    ts = 1_770_000_000_000

    trade_a = {
        "id": "exec-a",
        "timestamp": ts,
        "amount": 0.4,
        "price": 100.0,
        "side": "sell",
        "symbol": "BTC/USDT:USDT",
        "fee": {"currency": "USDT", "cost": 0.02},
        "info": {"execId": "exec-a", "orderId": "order-1", "execQty": "0.4", "closedSize": "0.4"},
    }
    trade_b = {
        "id": "exec-b",
        "timestamp": ts,
        "amount": 0.6,
        "price": 101.0,
        "side": "sell",
        "symbol": "BTC/USDT:USDT",
        "fee": {"currency": "USDT", "cost": 0.03},
        "info": {"execId": "exec-b", "orderId": "order-1", "execQty": "0.6", "closedSize": "0.6"},
    }
    pos_hist = {
        "source": "positions_history",
        "data": {
            "info": {
                "orderId": "order-1",
                "avgEntryPrice": "110.0",
                "closedSize": "1.0",
                "closeFee": "0.05",
                "openFee": "0.0",
            }
        },
    }

    malformed_events = [
        FillEvent.from_dict(
            {
                "id": "exec-a+exec-b+exec-a",
                "source_ids": ["exec-a", "exec-b"],
                "timestamp": ts,
                "datetime": "2026-02-28T00:00:00",
                "symbol": "BTC/USDT:USDT",
                "side": "sell",
                "qty": -1.4,
                "price": 100.4,
                "pnl": -14.0,
                "fees": {"currency": "USDT", "cost": 0.07},
                "pb_order_type": "close_auto_reduce_wel_long",
                "position_side": "long",
                "client_order_id": "0xabc",
                "raw": [
                    {"source": "fetch_my_trades", "data": dict(trade_a)},
                    {"source": "fetch_my_trades", "data": dict(trade_b)},
                    {"source": "fetch_my_trades", "data": dict(trade_a)},
                    dict(pos_hist),
                ],
            }
        ),
        FillEvent.from_dict(
            {
                "id": "exec-b",
                "source_ids": ["exec-b"],
                "timestamp": ts,
                "datetime": "2026-02-28T00:00:00",
                "symbol": "BTC/USDT:USDT",
                "side": "sell",
                "qty": -0.6,
                "price": 101.0,
                "pnl": -5.4,
                "fees": {"currency": "USDT", "cost": 0.03},
                "pb_order_type": "close_auto_reduce_wel_long",
                "position_side": "long",
                "client_order_id": "0xabc",
                "raw": [{"source": "fetch_my_trades", "data": dict(trade_b)}, dict(pos_hist)],
            }
        ),
    ]
    FillEventCache(cache_path).save(malformed_events)

    manager = FillEventsManager(
        exchange="bybit",
        user="u",
        fetcher=_StaticFetcher([]),
        cache_path=cache_path,
    )
    await manager.ensure_loaded()

    report = await manager.run_doctor(auto_repair=True)
    assert report["anomaly_events"] > 0
    assert report["anomaly_events_after"] == 0
    assert report["repaired"] is True

    repaired_events = manager.get_events()
    assert len(repaired_events) == 1
    assert repaired_events[0].qty == pytest.approx(-1.0)
    assert repaired_events[0].id == "exec-a+exec-b"


@pytest.mark.asyncio
async def test_fill_events_manager_bybit_doctor_repairs_legacy_duplicate_cache(tmp_path: Path):
    cache_path = tmp_path / "fills_doctor_legacy_bybit"
    cache_path.mkdir()
    ts = 1_770_000_000_000
    trade_a = {
        "id": "exec-a",
        "timestamp": ts,
        "amount": 0.4,
        "price": 100.0,
        "side": "sell",
        "symbol": "BTC/USDT:USDT",
        "fee": {"currency": "USDT", "cost": 0.02},
        "info": {"execId": "exec-a", "orderId": "order-1", "execQty": "0.4", "closedSize": "0.4"},
    }
    pos_hist = {
        "source": "positions_history",
        "data": {
            "info": {
                "orderId": "order-1",
                "avgEntryPrice": "110.0",
                "closedSize": "0.4",
                "closeFee": "0.02",
                "openFee": "0.0",
            }
        },
    }
    legacy_payload = [
        {
            "id": "exec-a+exec-a",
            "source_ids": ["exec-a"],
            "timestamp": ts,
            "datetime": "2026-02-02T00:00:00",
            "symbol": "BTC/USDT:USDT",
            "side": "sell",
            "qty": -0.8,
            "price": 100.0,
            "pnl": -8.0,
            "c_mult": 10.0,
            "fees": {"currency": "USDT", "cost": 0.04},
            "pb_order_type": "close_auto_reduce_wel_long",
            "position_side": "long",
            "client_order_id": "0xabc",
            "raw": [
                {"source": "fetch_my_trades", "data": dict(trade_a)},
                {"source": "fetch_my_trades", "data": dict(trade_a)},
                dict(pos_hist),
            ],
        }
    ]
    (cache_path / "2026-02-02.json").write_text(json.dumps(legacy_payload), encoding="utf-8")

    manager = FillEventsManager(
        exchange="bybit",
        user="u",
        fetcher=_StaticFetcher([]),
        cache_path=cache_path,
    )

    with pytest.raises(FillEventCacheContractError):
        await manager.ensure_loaded()

    report = await manager.run_doctor(auto_repair=True)
    assert report["legacy_contract"] is True
    assert report["anomaly_events"] > 0
    assert report["repaired"] is True

    repaired = FillEventCache(cache_path).load()
    assert len(repaired) == 1
    assert repaired[0].pnl_contract == fem.PNL_CONTRACT_CURRENT
    assert repaired[0].qty == pytest.approx(-0.4)
    assert repaired[0].c_mult == pytest.approx(10.0)
    assert repaired[0].pnl == pytest.approx(-40.0)


@pytest.mark.asyncio
async def test_bybit_doctor_does_not_stamp_unproven_legacy_rows_current(tmp_path: Path):
    cache_path = tmp_path / "fills_doctor_legacy_bybit_mixed"
    cache_path.mkdir()
    ts = 1_770_000_000_000
    trade_a = {
        "id": "exec-a",
        "timestamp": ts,
        "amount": 0.4,
        "price": 100.0,
        "side": "sell",
        "symbol": "BTC/USDT:USDT",
        "fee": {"currency": "USDT", "cost": 0.02},
        "info": {"execId": "exec-a", "orderId": "order-1", "execQty": "0.4", "closedSize": "0.4"},
    }
    pos_hist = {
        "source": "positions_history",
        "data": {
            "info": {
                "orderId": "order-1",
                "avgEntryPrice": "110.0",
                "closedSize": "0.4",
                "closeFee": "0.02",
                "openFee": "0.0",
            }
        },
    }
    legacy_payload = [
        {
            "id": "exec-a+exec-a",
            "source_ids": ["exec-a"],
            "timestamp": ts,
            "datetime": "2026-02-02T00:00:00",
            "symbol": "BTC/USDT:USDT",
            "side": "sell",
            "qty": -0.8,
            "price": 100.0,
            "pnl": -8.0,
            "fees": {"currency": "USDT", "cost": 0.04},
            "pb_order_type": "close_auto_reduce_wel_long",
            "position_side": "long",
            "client_order_id": "0xabc",
            "raw": [
                {"source": "fetch_my_trades", "data": dict(trade_a)},
                {"source": "fetch_my_trades", "data": dict(trade_a)},
                dict(pos_hist),
            ],
        },
        {
            "id": "legacy-unproven",
            "source_ids": ["legacy-unproven"],
            "timestamp": ts + 1,
            "datetime": "2026-02-02T00:00:01",
            "symbol": "ETH/USDT:USDT",
            "side": "sell",
            "qty": -1.0,
            "price": 100.0,
            "pnl": 9.95,
            "fee_paid": -0.05,
            "fees": {"currency": "USDT", "cost": 0.05},
            "pb_order_type": "close_grid_long",
            "position_side": "long",
            "client_order_id": "0xdef",
            "raw": [{"source": "fetch_my_trades", "data": {"id": "legacy-unproven"}}],
        },
    ]
    (cache_path / "2026-02-02.json").write_text(json.dumps(legacy_payload), encoding="utf-8")

    manager = FillEventsManager(
        exchange="bybit",
        user="u",
        fetcher=_StaticFetcher([]),
        cache_path=cache_path,
    )

    report = await manager.run_doctor(auto_repair=True)
    assert report["legacy_contract"] is True
    assert report["repaired"] is False
    assert report["legacy_unrepairable_events"] == 1
    assert report["legacy_unrepairable_examples"] == ["legacy-unproven"]

    with pytest.raises(FillEventCacheContractError):
        FillEventCache(cache_path).load()


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
    assert api.calls[0]["since"] == base_ts - 1
    assert api.calls[0]["limit"] == fetcher.trade_limit
    assert api.calls[0]["params"] == {}


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
        fee_pct_fallback=0.0,
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


def test_manager_pnl_helpers_report_net_pnl(tmp_path: Path):
    ts = 1_700_000_000_000
    manager = FillEventsManager(
        exchange="bitget",
        user="default",
        fetcher=_StaticFetcher([]),
        cache_path=tmp_path / "fills_net_pnl",
    )
    manager._events = [
        FillEvent.from_dict(
            {
                "id": "gross-profit",
                "timestamp": ts,
                "datetime": "",
                "symbol": "BTC/USDT",
                "side": "sell",
                "qty": -1.0,
                "price": 11.0,
                "pnl": 1.0,
                "fees": {"currency": "USDT", "cost": 0.1},
                "pb_order_type": "close",
                "position_side": "long",
                "client_order_id": "cid-1",
            }
        ),
        FillEvent.from_dict(
            {
                "id": "rebated-loss",
                "timestamp": ts + 1_000,
                "datetime": "",
                "symbol": "BTC/USDT",
                "side": "buy",
                "qty": 1.0,
                "price": 10.5,
                "pnl": -0.5,
                "fees": {"currency": "USDT", "cost": -0.05},
                "pb_order_type": "close",
                "position_side": "short",
                "client_order_id": "cid-2",
            }
        ),
    ]
    manager._loaded = True

    assert manager.get_pnl_sum() == pytest.approx(0.45)
    assert manager.get_pnl_cumsum()[-1][1] == pytest.approx(0.45)
    assert manager.reconstruct_equity_curve(starting_equity=10.0)[-1][1] == pytest.approx(10.45)


@pytest.mark.asyncio
async def test_manager_pnl_helpers_fail_on_pending_close_pnl(tmp_path: Path, sample_events):
    manager = FillEventsManager(
        exchange="bybit",
        user="default",
        fetcher=_StaticFetcher([]),
        cache_path=tmp_path / "fills_pending",
    )
    events = [FillEvent.from_dict(dict(ev)) for ev in sample_events]
    pending_payload = dict(sample_events[-1])
    pending_payload["pnl_status"] = "pending"
    events[-1] = FillEvent.from_dict(pending_payload)
    manager._events = events
    manager._loaded = True

    with pytest.raises(RuntimeError, match="realized PnL pending"):
        manager.get_pnl_sum()
    with pytest.raises(RuntimeError, match="realized PnL pending"):
        manager.get_pnl_cumsum()
    with pytest.raises(RuntimeError, match="realized PnL pending"):
        manager.reconstruct_equity_curve()


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
async def test_manager_refresh_latest_can_bound_start_from_last_successful_refresh(
    tmp_path: Path, sample_events
):
    cache_dir = tmp_path / "fills_latest_bounded"
    last_refresh_ms = sample_events[-1]["timestamp"] + 24 * 60 * 60 * 1000
    overlap_ms = 60 * 60 * 1000

    class _RecordingFetcher(BaseFetcher):
        def __init__(self, batch):
            self.batch = batch
            self.calls = []

        async def fetch(self, since_ms, until_ms, detail_cache, on_batch=None):
            self.calls.append((since_ms, until_ms))
            if on_batch and self.batch:
                on_batch(self.batch)
            batch = self.batch
            self.batch = []
            return batch

    fetcher = _RecordingFetcher([dict(event) for event in sample_events])
    manager = FillEventsManager(
        exchange="bitget",
        user="default",
        fetcher=fetcher,
        cache_path=cache_dir,
    )

    await manager.refresh()
    metadata = manager.cache.load_metadata()
    metadata["last_refresh_ms"] = last_refresh_ms
    manager.cache.save_metadata(metadata)

    await manager.refresh_latest(overlap=20, last_refresh_overlap_ms=overlap_ms)

    assert fetcher.calls[1] == (last_refresh_ms - overlap_ms, None)


@pytest.mark.asyncio
async def test_manager_replaces_synthetic_pnl_when_authoritative_arrives(
    tmp_path: Path,
):
    cache_dir = tmp_path / "fills_synthetic_replaced"
    entry_ts = 1_700_000_000_000
    pending_ts = entry_ts + 60_000
    last_refresh_ms = pending_ts + 4 * 60 * 60 * 1000
    overlap_ms = 10 * 60 * 1000

    entry_event = dict(
        id="entry",
        timestamp=entry_ts,
        datetime="",
        symbol="TON/USDT:USDT",
        side="buy",
        qty=10.0,
        price=2.0,
        pnl=0.0,
        pnl_status="complete",
        pb_order_type="entry_grid_long",
        position_side="long",
        client_order_id="cid-entry",
    )
    pending_event = dict(
        id="pending-close",
        timestamp=pending_ts,
        datetime="",
        symbol="TON/USDT:USDT",
        side="sell",
        qty=-4.0,
        price=2.5,
        pnl=0.0,
        pnl_status="pending",
        pb_order_type="close_unstuck_long",
        position_side="long",
        client_order_id="cid-pending",
    )
    enriched_event = dict(pending_event)
    enriched_event["pnl"] = 1.23
    enriched_event["pnl_status"] = "complete"

    fetcher = _SequencedFetcher([[entry_event, pending_event], [enriched_event]])
    manager = FillEventsManager(
        exchange="kucoin",
        user="default",
        fetcher=fetcher,
        cache_path=cache_dir,
        fee_pct_fallback=0.0,
    )

    await manager.refresh()
    close = manager.get_events(symbol="TON/USDT:USDT")[-1]
    assert close.pnl_status == "complete"
    assert close.pnl_source == fem.PNL_SOURCE_SYNTHETIC_EXACT
    assert close.pnl == pytest.approx(2.0)
    assert not FillEventsManager.pending_pnl_events(manager.get_events())

    metadata = manager.cache.load_metadata()
    metadata["last_refresh_ms"] = last_refresh_ms
    manager.cache.save_metadata(metadata)

    await manager.refresh_latest(overlap=20, last_refresh_overlap_ms=overlap_ms)

    assert fetcher.calls[1] == (
        max(0, pending_ts - fem.PENDING_PNL_REFRESH_MARGIN_MS),
        None,
    )
    close = manager.get_events(symbol="TON/USDT:USDT")[-1]
    assert close.pnl == pytest.approx(1.23)
    assert close.pnl_source == fem.PNL_SOURCE_AUTHORITATIVE
    assert not FillEventsManager.pending_pnl_events(manager.get_events())


@pytest.mark.asyncio
async def test_manager_reconciles_cycle_pnl_without_leaving_synthetic_anchor(
    tmp_path: Path,
):
    cache_dir = tmp_path / "fills_cycle_reconciled"
    entry_ts = 1_700_000_000_000

    entry_event = dict(
        id="entry",
        timestamp=entry_ts,
        datetime="",
        symbol="TON/USDT:USDT",
        side="buy",
        qty=10.0,
        price=10.0,
        pnl=0.0,
        pnl_status="complete",
        pb_order_type="entry_grid_long",
        position_side="long",
        client_order_id="cid-entry",
    )
    partial_close = dict(
        id="partial-close",
        timestamp=entry_ts + 60_000,
        datetime="",
        symbol="TON/USDT:USDT",
        side="sell",
        qty=-4.0,
        price=8.0,
        pnl=0.0,
        pnl_status="pending",
        pb_order_type="close_grid_long",
        position_side="long",
        client_order_id="cid-partial",
    )
    final_close = dict(
        id="final-close",
        timestamp=entry_ts + 120_000,
        datetime="",
        symbol="TON/USDT:USDT",
        side="sell",
        qty=-6.0,
        price=12.0,
        pnl=0.0,
        pnl_status="pending",
        pb_order_type="close_grid_long",
        position_side="long",
        client_order_id="cid-final",
    )

    fetcher = _StaticFetcher(
        [entry_event, partial_close, final_close],
        observations=[
            PnlObservation(
                scope="position_cycle",
                source="kucoin_positions_history",
                symbol="TON/USDT:USDT",
                position_side="long",
                realized_pnl=5.0,
                source_id="kucoin-cycle-1",
                close_time=entry_ts + 120_000,
            )
        ],
    )
    manager = FillEventsManager(
        exchange="kucoin",
        user="default",
        fetcher=fetcher,
        cache_path=cache_dir,
        fee_pct_fallback=0.0,
    )

    await manager.refresh()

    events = manager.get_events(symbol="TON/USDT:USDT")
    by_id = {ev.id: ev for ev in events}
    assert by_id["partial-close"].pnl == pytest.approx(-7.6)
    assert by_id["final-close"].pnl == pytest.approx(12.6)
    assert sum(ev.pnl for ev in events if "close" in ev.pb_order_type) == pytest.approx(5.0)
    assert by_id["partial-close"].pnl_source == fem.PNL_SOURCE_AUTHORITATIVE_CYCLE_RECONCILED
    assert by_id["final-close"].pnl_source == fem.PNL_SOURCE_AUTHORITATIVE_CYCLE_RECONCILED
    assert not FillEventsManager.pending_pnl_events(events)
    assert not FillEventsManager.synthetic_pnl_events(events)

    await manager.refresh()
    refreshed_events = manager.get_events(symbol="TON/USDT:USDT")
    assert sum(ev.pnl for ev in refreshed_events if "close" in ev.pb_order_type) == pytest.approx(
        5.0
    )

    reloaded = FillEventsManager(
        exchange="kucoin",
        user="default",
        fetcher=_StaticFetcher([]),
        cache_path=cache_dir,
    )
    await reloaded.ensure_loaded()
    reloaded_events = reloaded.get_events(symbol="TON/USDT:USDT")
    assert sum(ev.pnl for ev in reloaded_events if "close" in ev.pb_order_type) == pytest.approx(5.0)
    assert not FillEventsManager.synthetic_pnl_events(reloaded_events)


@pytest.mark.asyncio
async def test_manager_reconciles_multiple_closed_cycles_from_observations(
    tmp_path: Path,
):
    cache_dir = tmp_path / "fills_multiple_cycles_reconciled"
    entry_ts = 1_700_000_000_000
    events = [
        dict(
            id="entry-1",
            timestamp=entry_ts,
            datetime="",
            symbol="TON/USDT:USDT",
            side="buy",
            qty=1.0,
            price=10.0,
            pnl=0.0,
            pnl_status="complete",
            pb_order_type="entry_grid_long",
            position_side="long",
            client_order_id="cid-entry-1",
        ),
        dict(
            id="close-1",
            timestamp=entry_ts + 60_000,
            datetime="",
            symbol="TON/USDT:USDT",
            side="sell",
            qty=-1.0,
            price=11.0,
            pnl=0.0,
            pnl_status="pending",
            pb_order_type="close_grid_long",
            position_side="long",
            client_order_id="cid-close-1",
        ),
        dict(
            id="entry-2",
            timestamp=entry_ts + 120_000,
            datetime="",
            symbol="TON/USDT:USDT",
            side="buy",
            qty=1.0,
            price=20.0,
            pnl=0.0,
            pnl_status="complete",
            pb_order_type="entry_grid_long",
            position_side="long",
            client_order_id="cid-entry-2",
        ),
        dict(
            id="close-2",
            timestamp=entry_ts + 180_000,
            datetime="",
            symbol="TON/USDT:USDT",
            side="sell",
            qty=-1.0,
            price=22.0,
            pnl=0.0,
            pnl_status="pending",
            pb_order_type="close_grid_long",
            position_side="long",
            client_order_id="cid-close-2",
        ),
    ]
    observations = [
        PnlObservation(
            scope="position_cycle",
            source="kucoin_positions_history",
            symbol="TON/USDT:USDT",
            position_side="long",
            realized_pnl=1.0,
            source_id="kucoin-cycle-1",
            close_time=entry_ts + 60_000,
        ),
        PnlObservation(
            scope="position_cycle",
            source="kucoin_positions_history",
            symbol="TON/USDT:USDT",
            position_side="long",
            realized_pnl=2.0,
            source_id="kucoin-cycle-2",
            close_time=entry_ts + 180_000,
        ),
    ]
    manager = FillEventsManager(
        exchange="kucoin",
        user="default",
        fetcher=_StaticFetcher(events, observations=observations),
        cache_path=cache_dir,
        fee_pct_fallback=0.0,
    )

    await manager.refresh()

    stored = manager.get_events(symbol="TON/USDT:USDT")
    by_id = {ev.id: ev for ev in stored}
    assert by_id["close-1"].pnl == pytest.approx(1.0)
    assert by_id["close-2"].pnl == pytest.approx(2.0)
    assert sum(ev.pnl for ev in stored if "close" in ev.pb_order_type) == pytest.approx(3.0)
    assert by_id["close-1"].pnl_source == fem.PNL_SOURCE_AUTHORITATIVE_CYCLE_RECONCILED
    assert by_id["close-2"].pnl_source == fem.PNL_SOURCE_AUTHORITATIVE_CYCLE_RECONCILED
    assert not FillEventsManager.synthetic_pnl_events(stored)


@pytest.mark.asyncio
async def test_manager_refresh_latest_keeps_synthetic_pnl_in_enrichment_window(
    tmp_path: Path,
):
    cache_dir = tmp_path / "fills_synthetic_window"
    entry_ts = 1_700_000_000_000
    close_ts = entry_ts + 1_000
    last_refresh_ms = close_ts + 40 * 60_000
    overlap_ms = 10 * 60_000

    entry = dict(
        id="entry",
        timestamp=entry_ts,
        datetime="",
        symbol="SOL/USDT:USDT",
        side="buy",
        qty=5.0,
        price=10.0,
        pnl=0.0,
        pnl_status="complete",
        pb_order_type="entry_grid_long",
        position_side="long",
        client_order_id="cid-entry",
    )
    pending_close = dict(
        id="close",
        timestamp=close_ts,
        datetime="",
        symbol="SOL/USDT:USDT",
        side="sell",
        qty=-2.0,
        price=12.0,
        pnl=0.0,
        pnl_status="pending",
        pb_order_type="close_grid_long",
        position_side="long",
        client_order_id="cid-close",
    )
    authoritative_close = dict(pending_close)
    authoritative_close["pnl"] = 9.99
    authoritative_close["pnl_status"] = "complete"
    later_events = [
        dict(
            id=f"later-{idx}",
            timestamp=close_ts + (idx + 1) * 60_000,
            datetime="",
            symbol="SOL/USDT:USDT",
            side="buy",
            qty=0.1,
            price=12.0 + idx,
            pnl=0.0,
            pnl_status="complete",
            pb_order_type="entry_grid_long",
            position_side="long",
            client_order_id=f"cid-later-{idx}",
        )
        for idx in range(30)
    ]

    class _SinceFilteringFetcher(BaseFetcher):
        def __init__(self, batches):
            self.batches = list(batches)
            self.calls: List[Tuple[Optional[int], Optional[int]]] = []

        async def fetch(self, since_ms, until_ms, detail_cache, on_batch=None):
            self.calls.append((since_ms, until_ms))
            batch = [dict(ev) for ev in (self.batches.pop(0) if self.batches else [])]
            if since_ms is not None:
                batch = [ev for ev in batch if int(ev["timestamp"]) >= since_ms]
            if until_ms is not None:
                batch = [ev for ev in batch if int(ev["timestamp"]) <= until_ms]
            if on_batch and batch:
                on_batch(batch)
            return batch

    fetcher = _SinceFilteringFetcher(
        [
            [entry, pending_close, *later_events],
            [authoritative_close, *later_events],
        ]
    )
    manager = FillEventsManager(
        exchange="example",
        user="default",
        fetcher=fetcher,
        cache_path=cache_dir,
    )

    await manager.refresh()
    close = [ev for ev in manager.get_events(symbol="SOL/USDT:USDT") if ev.id == "close"][0]
    assert close.pnl == pytest.approx(4.0)
    assert close.pnl_source == fem.PNL_SOURCE_SYNTHETIC_EXACT

    metadata = manager.cache.load_metadata()
    metadata["last_refresh_ms"] = last_refresh_ms
    manager.cache.save_metadata(metadata)

    await manager.refresh_latest(overlap=20, last_refresh_overlap_ms=overlap_ms)

    assert fetcher.calls[1] == (
        max(0, close_ts - fem.PENDING_PNL_REFRESH_MARGIN_MS),
        None,
    )
    close = [ev for ev in manager.get_events(symbol="SOL/USDT:USDT") if ev.id == "close"][0]
    assert close.pnl == pytest.approx(9.99)
    assert close.pnl_source == fem.PNL_SOURCE_AUTHORITATIVE


@pytest.mark.asyncio
async def test_manager_synthesizes_pending_close_pnl_with_contract_value(
    tmp_path: Path, caplog
):
    entry_ts = 1_700_000_000_000
    close_ts = entry_ts + 1_000
    events = [
        dict(
            id="entry-contract",
            timestamp=entry_ts,
            datetime="",
            symbol="BTC/USDT:USDT",
            side="buy",
            qty=1.0,
            price=10.0,
            pnl=0.0,
            pnl_status="complete",
            fees=None,
            raw=[{"source": "fetch_my_trades", "data": {"cost": "1.0"}}],
            pb_order_type="entry_grid_long",
            position_side="long",
            client_order_id="cid-entry",
        ),
        dict(
            id="close-contract",
            timestamp=close_ts,
            datetime="",
            symbol="BTC/USDT:USDT",
            side="sell",
            qty=-1.0,
            price=20.0,
            pnl=0.0,
            pnl_status="pending",
            fees={"currency": "USDT", "cost": 0.1},
            raw=[{"source": "fetch_my_trades", "data": {"cost": "2.0"}}],
            pb_order_type="close_grid_long",
            position_side="long",
            client_order_id="cid-close",
        ),
    ]
    manager = FillEventsManager(
        exchange="example",
        user="default",
        fetcher=_StaticFetcher(events),
        cache_path=tmp_path / "fills_synthetic_exact",
        fee_pct_sanity_abs_max=1.0,
    )

    with caplog.at_level(logging.WARNING, logger=fem.logger.name):
        await manager.refresh()

    close = manager.get_events(symbol="BTC/USDT:USDT")[-1]
    assert close.pnl_status == "complete"
    assert close.pnl_source == fem.PNL_SOURCE_SYNTHETIC_EXACT
    assert close.pnl_synthetic_reason == ""
    assert close.pnl == pytest.approx(1.0)
    assert close.fee_paid == pytest.approx(-0.1)
    assert not FillEventsManager.pending_pnl_events(manager.get_events())
    assert any(
        "synthesized missing realized PnL from fill events" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_manager_synthesizes_degraded_pnl_when_cache_starts_mid_position(
    tmp_path: Path,
):
    event = dict(
        id="orphan-close",
        timestamp=1_700_000_000_000,
        datetime="",
        symbol="ETH/USDT:USDT",
        side="sell",
        qty=-2.0,
        price=1500.0,
        pnl=0.42,
        pnl_status="pending",
        pb_order_type="close_grid_long",
        position_side="long",
        client_order_id="cid-close",
    )
    manager = FillEventsManager(
        exchange="example",
        user="default",
        fetcher=_StaticFetcher([event]),
        cache_path=tmp_path / "fills_synthetic_degraded",
    )

    await manager.refresh()

    close = manager.get_events(symbol="ETH/USDT:USDT")[0]
    assert close.pnl_status == "complete"
    assert close.pnl_source == fem.PNL_SOURCE_SYNTHETIC_DEGRADED
    assert close.pnl_synthetic_reason == "incomplete_position_basis"
    assert close.pnl == pytest.approx(0.42)
    assert not FillEventsManager.pending_pnl_events(manager.get_events())


@pytest.mark.asyncio
async def test_manager_preserves_synthetic_pnl_when_later_fetch_is_still_pending(
    tmp_path: Path,
):
    entry_ts = 1_700_000_000_000
    close_ts = entry_ts + 1_000
    entry = dict(
        id="entry",
        timestamp=entry_ts,
        datetime="",
        symbol="SOL/USDT:USDT",
        side="buy",
        qty=5.0,
        price=10.0,
        pnl=0.0,
        pnl_status="complete",
        pb_order_type="entry_grid_long",
        position_side="long",
        client_order_id="cid-entry",
    )
    close = dict(
        id="close",
        timestamp=close_ts,
        datetime="",
        symbol="SOL/USDT:USDT",
        side="sell",
        qty=-2.0,
        price=12.0,
        pnl=0.0,
        pnl_status="pending",
        pb_order_type="close_grid_long",
        position_side="long",
        client_order_id="cid-close",
    )
    later_pending = dict(close)
    later_pending["pnl"] = 0.0
    fetcher = _SequencedFetcher([[entry, close], [later_pending]])
    manager = FillEventsManager(
        exchange="example",
        user="default",
        fetcher=fetcher,
        cache_path=tmp_path / "fills_synthetic_preserved",
    )

    await manager.refresh()
    await manager.refresh_latest()

    close_event = manager.get_events(symbol="SOL/USDT:USDT")[-1]
    assert close_event.pnl_status == "complete"
    assert close_event.pnl_source == fem.PNL_SOURCE_SYNTHETIC_EXACT
    assert close_event.pnl == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_manager_refresh_records_successful_empty_refresh(tmp_path: Path):
    cache_dir = tmp_path / "fills_empty_refresh"

    class _EmptyFetcher(BaseFetcher):
        def __init__(self):
            self.calls = []

        async def fetch(self, since_ms, until_ms, detail_cache, on_batch=None):
            self.calls.append((since_ms, until_ms))
            return []

    fetcher = _EmptyFetcher()
    manager = FillEventsManager(
        exchange="bitget",
        user="default",
        fetcher=fetcher,
        cache_path=cache_dir,
    )

    await manager.refresh()
    metadata = manager.cache.load_metadata()

    assert fetcher.calls == [(None, None)]
    assert metadata["last_refresh_ms"] > 0
    assert manager.get_events() == []


@pytest.mark.asyncio
async def test_manager_refresh_logs_fetcher_request_timing(tmp_path: Path, caplog):
    cache_dir = tmp_path / "fills_request_timing"

    class _Api:
        async def fetch_a(self):
            return {"ok": True}

        async def fetch_b(self):
            return {"ok": True}

    class _ApiFetcher(BaseFetcher):
        def __init__(self):
            self.api = _Api()

        async def fetch(self, since_ms, until_ms, detail_cache, on_batch=None):
            await self.api.fetch_a()
            await self.api.fetch_b()
            await self.api.fetch_b()
            return []

    manager = FillEventsManager(
        exchange="bitget",
        user="default",
        fetcher=_ApiFetcher(),
        cache_path=cache_dir,
    )

    with caplog.at_level(logging.DEBUG, logger=fem.logger.name):
        await manager.refresh(start_ms=123, end_ms=456)

    messages = [record.message for record in caplog.records]
    assert any("[fills] fetcher request timing" in msg for msg in messages)
    assert any("requests=3" in msg for msg in messages)
    assert any("fetch_b:n=2" in msg for msg in messages)
    assert any("range=1970-01-01 00:00:00..1970-01-01 00:00:00" in msg for msg in messages)


@pytest.mark.asyncio
async def test_manager_refresh_logs_fetcher_error_detail(tmp_path: Path, caplog):
    cache_dir = tmp_path / "fills_request_error_timing"

    class _Api:
        async def fetch_a(self):
            raise RuntimeError("remote timeout detail")

    class _ApiFetcher(BaseFetcher):
        def __init__(self):
            self.api = _Api()

        async def fetch(self, since_ms, until_ms, detail_cache, on_batch=None):
            await self.api.fetch_a()
            return []

    manager = FillEventsManager(
        exchange="kucoin",
        user="default",
        fetcher=_ApiFetcher(),
        cache_path=cache_dir,
    )

    with caplog.at_level(logging.INFO, logger=fem.logger.name):
        with pytest.raises(RuntimeError, match="remote timeout detail"):
            await manager.refresh(start_ms=123, end_ms=456)

    messages = [record.message for record in caplog.records]
    assert any("[fills] fetcher request timing" in msg for msg in messages)
    assert any("fetch_a:n=1" in msg for msg in messages)
    assert any("err_type=RuntimeError" in msg for msg in messages)
    assert any("err_msg=remote timeout detail" in msg for msg in messages)


def test_fill_event_cache_disk_full_raises_clear_error(tmp_path: Path, monkeypatch, sample_events):
    cache = FillEventCache(tmp_path / "fills_disk_full")
    event = FillEvent.from_dict(dict(sample_events[0]))

    def fail_replace(src, dst):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(fem.os, "replace", fail_replace)

    with pytest.raises(FillEventCacheDiskFullError, match="disk full"):
        cache.save([event])

    with pytest.raises(FillEventCacheDiskFullError, match="disk full"):
        cache.save_metadata({"last_refresh_ms": event.timestamp})


def test_fill_event_cache_load_ignores_metadata_file(tmp_path: Path, sample_events, caplog):
    cache = FillEventCache(tmp_path)
    cache.save([FillEvent.from_dict(dict(sample_events[0]))])
    cache.save_metadata(
        {
            "last_refresh_ms": sample_events[0]["timestamp"],
            "oldest_event_ts": sample_events[0]["timestamp"],
            "newest_event_ts": sample_events[0]["timestamp"],
            "covered_start_ms": sample_events[0]["timestamp"],
            "known_gaps": [],
            "history_scope": "window",
        }
    )

    with caplog.at_level(logging.DEBUG):
        loaded = cache.load()

    assert [event.id for event in loaded] == [sample_events[0]["id"]]
    assert not any("metadata.json" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_manager_refresh_for_lookback_persists_coverage_across_restart(tmp_path: Path):
    cache_dir = tmp_path / "fills_lookback_coverage"
    start_ms = 1_700_000_000_000
    event_ts = start_ms + 2 * 24 * 60 * 60 * 1000
    event = dict(
        id="late-fill-1",
        timestamp=event_ts,
        datetime="",
        symbol="BTC/USDT",
        side="buy",
        qty=0.1,
        price=10.0,
        pnl=0.0,
        pb_order_type="entry",
        position_side="long",
        client_order_id="cid-late-fill-1",
    )

    class _RecordingFetcher(BaseFetcher):
        def __init__(self, batches):
            self.batches = list(batches)
            self.calls: List[Tuple[Optional[int], Optional[int]]] = []

        async def fetch(self, since_ms, until_ms, detail_cache, on_batch=None):
            self.calls.append((since_ms, until_ms))
            batch = self.batches.pop(0) if self.batches else []
            if on_batch and batch:
                on_batch(batch)
            return batch

    fetcher1 = _RecordingFetcher([[dict(event)]])
    manager1 = FillEventsManager(
        exchange="bybit",
        user="default",
        fetcher=fetcher1,
        cache_path=cache_dir,
    )

    await manager1.refresh_for_lookback(start_ms=start_ms)

    assert fetcher1.calls == [(start_ms, None)]
    assert manager1.cache.load_metadata()["covered_start_ms"] == start_ms
    assert manager1.get_events()[0].timestamp == event_ts

    fetcher2 = _RecordingFetcher([[]])
    manager2 = FillEventsManager(
        exchange="bybit",
        user="default",
        fetcher=fetcher2,
        cache_path=cache_dir,
    )

    await manager2.refresh_for_lookback(start_ms=start_ms)

    assert fetcher2.calls == [(event_ts, None)]


@pytest.mark.asyncio
async def test_manager_refresh_for_lookback_rebuilds_when_metadata_claims_history_but_cache_is_empty(
    tmp_path: Path,
):
    cache_dir = tmp_path / "fills_lookback_rebuild"
    start_ms = 1_700_000_000_000

    class _RecordingFetcher(BaseFetcher):
        def __init__(self):
            self.calls: List[Tuple[Optional[int], Optional[int]]] = []

        async def fetch(self, since_ms, until_ms, detail_cache, on_batch=None):
            self.calls.append((since_ms, until_ms))
            return []

    manager = FillEventsManager(
        exchange="bybit",
        user="default",
        fetcher=_RecordingFetcher(),
        cache_path=cache_dir,
    )
    manager.cache.save_metadata(
        {
            "last_refresh_ms": 1,
            "oldest_event_ts": start_ms + 86_400_000,
            "newest_event_ts": start_ms + 172_800_000,
            "covered_start_ms": start_ms,
            "known_gaps": [],
        }
    )

    await manager.refresh_for_lookback(start_ms=start_ms)

    assert manager.fetcher.calls == [(start_ms, None)]


@pytest.mark.asyncio
async def test_manager_refresh_for_lookback_preserves_metadata_only_no_fill_coverage(tmp_path: Path):
    cache_dir = tmp_path / "fills_lookback_no_fill_coverage"
    start_ms = 1_700_000_000_000

    class _RecordingFetcher(BaseFetcher):
        def __init__(self):
            self.calls: List[Tuple[Optional[int], Optional[int]]] = []

        async def fetch(self, since_ms, until_ms, detail_cache, on_batch=None):
            self.calls.append((since_ms, until_ms))
            return []

    manager = FillEventsManager(
        exchange="bybit",
        user="default",
        fetcher=_RecordingFetcher(),
        cache_path=cache_dir,
    )
    manager.cache.save_metadata(
        {
            "last_refresh_ms": 1,
            "oldest_event_ts": 0,
            "newest_event_ts": 0,
            "covered_start_ms": start_ms,
            "known_gaps": [],
        }
    )

    await manager.refresh_for_lookback(start_ms=start_ms)

    assert manager.fetcher.calls == [(None, None)]


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

        async def fetch_my_trades(
            self,
            symbol: Optional[str] = None,
            since: Optional[int] = None,
            limit: Optional[int] = None,
            params: Optional[Dict[str, Any]] = None,
        ):
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
async def test_okx_fetcher_uses_market_contract_size_for_fee_notional(monkeypatch):
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    fill = _make_okx_fill(
        trade_id="tid-doge",
        ts=now_ms - 100,
        inst_id="DOGE-USDT-SWAP",
        qty="1",
        price="0.20",
    )
    fill["fee"] = "-0.004"
    api = _FakeOkxAPI([[fill]])
    api.markets = {"DOGE/USDT:USDT": {"id": "DOGE-USDT-SWAP", "contractSize": 100.0}}
    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda v: v or "unknown")

    fetcher = OkxFetcher(api, trade_limit=100)
    events = await fetcher.fetch(
        since_ms=now_ms - 1_000,
        until_ms=now_ms,
        detail_cache={},
    )

    assert events[0]["c_mult"] == pytest.approx(100.0)
    fee_paid, meta = fem._normalize_fee_paid_from_payload(
        events[0],
        fee_pct_fallback=0.0002,
        fee_pct_sanity_abs_max=0.001,
    )
    assert fee_paid == pytest.approx(-0.004)
    assert meta["fee_notional"] == pytest.approx(20.0)
    assert meta["fee_ratio"] == pytest.approx(-0.0002)
    assert meta["fee_quality"] == fem.FEE_QUALITY_EXACT


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


def _kucoin_manager_fill(
    fill_id: str,
    ts: int,
    *,
    side: str,
    qty: float,
    price: float,
    symbol: str = "TON/USDT:USDT",
    position_side: str = "long",
    pb_order_type: Optional[str] = None,
    fees: Optional[object] = None,
) -> Dict[str, object]:
    is_close = (position_side == "long" and side == "sell") or (
        position_side == "short" and side == "buy"
    )
    return {
        "id": fill_id,
        "timestamp": ts,
        "datetime": "",
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": price,
        "pnl": 0.0,
        "fees": fees,
        "pnl_status": "pending" if is_close else "complete",
        "pb_order_type": pb_order_type
        or ("close_grid_long" if is_close else "entry_grid_long"),
        "position_side": position_side,
        "client_order_id": f"cid-{fill_id}",
    }


def _kucoin_cycle_observation(
    source_id: str,
    *,
    realized_pnl: float,
    close_time: Optional[int] = None,
    update_time: Optional[int] = None,
    open_time: Optional[int] = None,
    close_size: Optional[float] = None,
    symbol: str = "TON/USDT:USDT",
    position_side: str = "long",
) -> PnlObservation:
    return PnlObservation(
        scope="position_cycle",
        source="kucoin_positions_history",
        symbol=symbol,
        position_side=position_side,
        realized_pnl=realized_pnl,
        source_id=source_id,
        close_time=close_time,
        update_time=update_time,
        open_time=open_time,
        close_size=close_size,
    )


def test_kucoin_position_history_observation_preserves_cycle_scope():
    obs = KucoinFetcher._position_history_observation(
        {
            "symbol": "TON/USDT:USDT",
            "side": "long",
            "contracts": "3.5",
            "lastUpdateTimestamp": 1_700_000_240_000,
            "realizedPnl": "12.34",
            "info": {
                "closeId": "123",
                "openTime": "1700000000000",
                "closeTime": "1700000060000",
            },
        }
    )

    assert obs is not None
    assert obs.scope == "position_cycle"
    assert obs.source == "kucoin_positions_history"
    assert obs.symbol == "TON/USDT:USDT"
    assert obs.position_side == "long"
    assert obs.realized_pnl == pytest.approx(12.34)
    assert obs.source_id == "123"
    assert obs.open_time == 1_700_000_000_000
    assert obs.close_time == 1_700_000_060_000
    assert obs.update_time == 1_700_000_240_000
    assert obs.close_size == pytest.approx(3.5)


def test_kucoin_trade_reconstruction_uses_raw_contract_multiplier():
    ts = 1_700_000_000_000
    trades = [
        {
            "id": "entry",
            "timestamp": ts,
            "symbol": "ZEC/USDT:USDT",
            "side": "buy",
            "qty": 100.0,
            "price": 600.0,
            "position_side": "long",
            "raw": [
                {
                    "source": "fetch_my_trades",
                    "data": {"info": {"value": "600.0"}},
                }
            ],
        },
        {
            "id": "close",
            "timestamp": ts + 60_000,
            "symbol": "ZEC/USDT:USDT",
            "side": "sell",
            "qty": 100.0,
            "price": 610.0,
            "position_side": "long",
            "raw": [
                {
                    "source": "fetch_my_trades",
                    "data": {"info": {"value": "610.0"}},
                }
            ],
        },
    ]

    pnls, final_positions = fem.compute_realized_pnls_from_trades(trades)

    assert pnls["entry"] == pytest.approx(0.0)
    assert pnls["close"] == pytest.approx(10.0)
    assert final_positions[("ZEC/USDT:USDT", "long")] == pytest.approx((0.0, 0.0))


@pytest.mark.asyncio
async def test_kucoin_fetcher_emits_observations_without_mutating_fill_pnl(monkeypatch):
    fetcher = KucoinFetcher(api=object())
    ts = 1_700_000_000_000
    trades = [
        _kucoin_manager_fill("entry", ts, side="buy", qty=1.0, price=10.0),
        _kucoin_manager_fill(
            "close",
            ts + 60_000,
            side="sell",
            qty=1.0,
            price=12.0,
        ),
    ]
    positions = [
        {
            "symbol": "TON/USDT:USDT",
            "lastUpdateTimestamp": ts + 180_000,
            "realizedPnl": "9.99",
            "info": {"closeId": "123", "closeTime": ts + 60_000},
        }
    ]

    async def _fetch_trades(since_ms, until_ms):
        return [dict(trade) for trade in trades]

    async def _fetch_positions_history(start_ms, end_ms):
        return positions

    async def _enrich_with_order_details_bulk(events, detail_cache):
        return None

    monkeypatch.setattr(fetcher, "_fetch_trades", _fetch_trades)
    monkeypatch.setattr(fetcher, "_fetch_positions_history", _fetch_positions_history)
    monkeypatch.setattr(fetcher, "_enrich_with_order_details_bulk", _enrich_with_order_details_bulk)

    out = await fetcher.fetch(ts, ts + 240_000, detail_cache={})
    by_id = {event["id"]: event for event in out}

    assert by_id["close"]["pnl_status"] == "pending"
    assert by_id["close"]["pnl_source"] == fem.PNL_SOURCE_PENDING
    assert "pnl_cycle_realized_pnl" not in by_id["close"]
    assert len(fetcher.pnl_observations) == 1
    assert fetcher.pnl_observations[0].realized_pnl == pytest.approx(9.99)
    assert fetcher.pnl_observations[0].close_time == ts + 60_000


@pytest.mark.asyncio
async def test_kucoin_fetcher_entry_fee_is_signed_fee_paid(monkeypatch):
    fetcher = KucoinFetcher(api=object())
    ts = 1_700_000_000_000
    trades = [
        _kucoin_manager_fill(
            "entry-fee",
            ts,
            side="buy",
            qty=1.0,
            price=10.0,
            fees={"currency": "USDT", "cost": 0.25},
        )
    ]

    async def _fetch_trades(since_ms, until_ms):
        return [dict(trade) for trade in trades]

    async def _enrich_with_order_details_bulk(events, detail_cache):
        return None

    monkeypatch.setattr(fetcher, "_fetch_trades", _fetch_trades)
    monkeypatch.setattr(fetcher, "_enrich_with_order_details_bulk", _enrich_with_order_details_bulk)

    out = await fetcher.fetch(ts, ts + 1_000, detail_cache={})

    assert out[0]["pnl"] == pytest.approx(0.0)
    assert out[0]["fee_paid"] == pytest.approx(-0.25)


def test_kucoin_normalize_trade_reads_plural_fees():
    ts = 1_700_000_000_000
    event = KucoinFetcher._normalize_trade(
        {
            "id": "plural-fee",
            "timestamp": ts,
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "amount": 1.0,
            "price": 10.0,
            "fees": [{"currency": "USDT", "cost": 0.25}],
            "info": {"tradeId": "plural-fee", "orderId": "order-1"},
        }
    )

    assert event["fees"] == [{"currency": "USDT", "cost": 0.25}]
    assert event["fee_paid"] == pytest.approx(-0.25)


def test_signed_fee_paid_sums_raw_kucoin_fee_fields():
    fee_paid = fem.signed_fee_paid_from_payload(
        {
            "raw": [
                {
                    "source": "fetch_my_trades",
                    "data": {
                        "info": {
                            "openFeePay": "0.1",
                            "closeFeePay": "0.2",
                            "fee": "-0.03",
                        }
                    },
                }
            ]
        }
    )

    assert fee_paid == pytest.approx(-0.27)


@pytest.mark.asyncio
async def test_kucoin_fetcher_maker_rebate_is_positive_fee_paid(monkeypatch):
    fetcher = KucoinFetcher(api=object())
    ts = 1_700_000_000_000
    trades = [
        _kucoin_manager_fill(
            "entry-rebate",
            ts,
            side="buy",
            qty=1.0,
            price=10.0,
            fees={"currency": "USDT", "cost": -0.03, "rate": -0.0001},
        )
    ]

    async def _fetch_trades(since_ms, until_ms):
        return [dict(trade) for trade in trades]

    async def _enrich_with_order_details_bulk(events, detail_cache):
        return None

    monkeypatch.setattr(fetcher, "_fetch_trades", _fetch_trades)
    monkeypatch.setattr(fetcher, "_enrich_with_order_details_bulk", _enrich_with_order_details_bulk)

    out = await fetcher.fetch(ts, ts + 1_000, detail_cache={})

    assert out[0]["pnl"] == pytest.approx(0.0)
    assert out[0]["fee_paid"] == pytest.approx(0.03)


def test_kucoin_positions_history_window_extends_for_delayed_pnl_records():
    fetcher = KucoinFetcher(api=None)
    close_ts = 1_700_000_000_000
    closes = [
        {
            "id": "delayed-close",
            "symbol": "TON/USDT:USDT",
            "timestamp": close_ts,
            "qty": 7.0,
        }
    ]

    start_ms, end_ms = fetcher._positions_history_window(
        closes, close_ts + 4 * 60 * 1000
    )

    assert start_ms == close_ts - 60_000
    assert end_ms == close_ts + 4 * 60 * 1000

    _start_ms, capped_end_ms = fetcher._positions_history_window(
        closes, close_ts + 60 * 60 * 1000
    )
    assert capped_end_ms == close_ts + KUCOIN_POSITION_HISTORY_LOOKAHEAD_MS


@pytest.mark.asyncio
async def test_manager_reconciles_rapid_lifecycles_by_observation_close_time(
    tmp_path: Path,
):
    ts = 1_700_000_000_000
    events = [
        _kucoin_manager_fill("entry-1", ts, side="buy", qty=1.0, price=10.0),
        _kucoin_manager_fill("close-1", ts + 60_000, side="sell", qty=-1.0, price=11.0),
        _kucoin_manager_fill("entry-2", ts + 120_000, side="buy", qty=1.0, price=20.0),
        _kucoin_manager_fill("close-2", ts + 180_000, side="sell", qty=-1.0, price=22.0),
    ]
    observations = [
        _kucoin_cycle_observation("100", realized_pnl=1.25, close_time=ts + 60_000),
        _kucoin_cycle_observation("101", realized_pnl=2.5, close_time=ts + 180_000),
    ]
    manager = FillEventsManager(
        exchange="kucoin",
        user="default",
        fetcher=_StaticFetcher(events, observations),
        cache_path=tmp_path / "rapid_cycles",
        fee_pct_fallback=0.0,
    )

    await manager.refresh()

    by_id = {event.id: event for event in manager.get_events(symbol="TON/USDT:USDT")}
    assert by_id["close-1"].pnl == pytest.approx(1.25)
    assert by_id["close-2"].pnl == pytest.approx(2.5)
    assert by_id["close-1"].pnl_source == fem.PNL_SOURCE_AUTHORITATIVE_CYCLE_RECONCILED
    assert by_id["close-2"].pnl_source == fem.PNL_SOURCE_AUTHORITATIVE_CYCLE_RECONCILED


@pytest.mark.asyncio
async def test_manager_reconciles_kucoin_net_cycle_pnl_to_gross_close_pnl(tmp_path: Path):
    ts = 1_700_000_000_000
    events = [
        _kucoin_manager_fill(
            "entry-fee",
            ts,
            side="buy",
            qty=1.0,
            price=10.0,
            fees={"currency": "USDT", "cost": 0.10},
        ),
        _kucoin_manager_fill(
            "close-fee",
            ts + 60_000,
            side="sell",
            qty=-1.0,
            price=12.0,
            fees={"currency": "USDT", "cost": 0.20},
        ),
    ]
    observations = [
        _kucoin_cycle_observation("100", realized_pnl=1.70, close_time=ts + 60_000),
    ]
    manager = FillEventsManager(
        exchange="kucoin",
        user="default",
        fetcher=_StaticFetcher(events, observations),
        cache_path=tmp_path / "net_to_gross_cycle",
        fee_pct_sanity_abs_max=1.0,
    )

    await manager.refresh()

    by_id = {event.id: event for event in manager.get_events(symbol="TON/USDT:USDT")}
    assert by_id["entry-fee"].pnl == pytest.approx(0.0)
    assert by_id["entry-fee"].fee_paid == pytest.approx(-0.10)
    assert by_id["close-fee"].pnl == pytest.approx(2.0)
    assert by_id["close-fee"].fee_paid == pytest.approx(-0.20)
    net_pnl = (
        by_id["close-fee"].pnl
        + by_id["entry-fee"].fee_paid
        + by_id["close-fee"].fee_paid
    )
    assert net_pnl == pytest.approx(1.70)


@pytest.mark.asyncio
async def test_manager_distributes_kucoin_gross_cycle_delta_across_multiple_closes(
    tmp_path: Path,
):
    ts = 1_700_000_000_000
    events = [
        _kucoin_manager_fill(
            "entry",
            ts,
            side="buy",
            qty=10.0,
            price=10.0,
            fees={"currency": "USDT", "cost": 1.0},
        ),
        _kucoin_manager_fill(
            "close-a",
            ts + 60_000,
            side="sell",
            qty=-4.0,
            price=11.0,
            fees={"currency": "USDT", "cost": 0.4},
        ),
        _kucoin_manager_fill(
            "close-b",
            ts + 120_000,
            side="sell",
            qty=-6.0,
            price=12.0,
            fees={"currency": "USDT", "cost": 0.6},
        ),
    ]
    observations = [
        _kucoin_cycle_observation("100", realized_pnl=8.0, close_time=ts + 120_000),
    ]
    manager = FillEventsManager(
        exchange="kucoin",
        user="default",
        fetcher=_StaticFetcher(events, observations),
        cache_path=tmp_path / "multi_close_net_to_gross",
        fee_pct_sanity_abs_max=1.0,
    )

    await manager.refresh()

    by_id = {event.id: event for event in manager.get_events(symbol="TON/USDT:USDT")}
    assert by_id["close-a"].pnl == pytest.approx(1.6)
    assert by_id["close-b"].pnl == pytest.approx(8.4)
    gross_sum = by_id["close-a"].pnl + by_id["close-b"].pnl
    fee_sum = sum(event.fee_paid for event in by_id.values())
    assert gross_sum == pytest.approx(10.0)
    assert gross_sum + fee_sum == pytest.approx(8.0)


@pytest.mark.asyncio
async def test_manager_reconciles_out_of_order_delayed_rows_by_close_time(
    tmp_path: Path,
):
    ts = 1_700_000_000_000
    events = [
        _kucoin_manager_fill("entry-1", ts, side="buy", qty=1.0, price=10.0),
        _kucoin_manager_fill("close-1", ts + 60_000, side="sell", qty=-1.0, price=11.0),
        _kucoin_manager_fill("entry-2", ts + 120_000, side="buy", qty=1.0, price=20.0),
        _kucoin_manager_fill("close-2", ts + 180_000, side="sell", qty=-1.0, price=22.0),
    ]
    observations = [
        _kucoin_cycle_observation(
            "101",
            realized_pnl=2.5,
            close_time=ts + 180_000,
            update_time=ts + 180_000,
        ),
        _kucoin_cycle_observation(
            "100",
            realized_pnl=1.25,
            close_time=ts + 60_000,
            update_time=ts + 240_000,
        ),
    ]
    manager = FillEventsManager(
        exchange="kucoin",
        user="default",
        fetcher=_StaticFetcher(events, observations),
        cache_path=tmp_path / "out_of_order_delayed",
        fee_pct_fallback=0.0,
    )

    await manager.refresh()

    by_id = {event.id: event for event in manager.get_events(symbol="TON/USDT:USDT")}
    assert by_id["close-1"].pnl == pytest.approx(1.25)
    assert by_id["close-2"].pnl == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_manager_defers_source_id_only_observations_when_close_time_missing(
    caplog,
    tmp_path: Path,
):
    ts = 1_700_000_000_000
    events = [
        _kucoin_manager_fill("entry-1", ts, side="buy", qty=1.0, price=10.0),
        _kucoin_manager_fill("close-1", ts + 60_000, side="sell", qty=-1.0, price=11.0),
        _kucoin_manager_fill("entry-2", ts + 120_000, side="buy", qty=1.0, price=20.0),
        _kucoin_manager_fill("close-2", ts + 180_000, side="sell", qty=-1.0, price=22.0),
    ]
    observations = [
        _kucoin_cycle_observation("101", realized_pnl=2.5, update_time=ts + 180_000),
        _kucoin_cycle_observation("100", realized_pnl=1.25, update_time=ts + 240_000),
    ]
    manager = FillEventsManager(
        exchange="kucoin",
        user="default",
        fetcher=_StaticFetcher(events, observations),
        cache_path=tmp_path / "source_id_order",
    )

    caplog.set_level(logging.WARNING, logger=fem.logger.name)
    await manager.refresh()

    by_id = {event.id: event for event in manager.get_events(symbol="TON/USDT:USDT")}
    assert by_id["close-1"].pnl == pytest.approx(1.0)
    assert by_id["close-2"].pnl == pytest.approx(2.0)
    assert by_id["close-1"].pnl_source == fem.PNL_SOURCE_SYNTHETIC_EXACT
    assert by_id["close-2"].pnl_source == fem.PNL_SOURCE_SYNTHETIC_EXACT
    assert any(
        "deferred aggregate realized PnL observation matching" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_manager_defers_ambiguous_cycle_observations(caplog, tmp_path: Path):
    ts = 1_700_000_000_000
    events = [
        _kucoin_manager_fill("entry-1", ts, side="buy", qty=1.0, price=10.0),
        _kucoin_manager_fill("close-1", ts + 60_000, side="sell", qty=-1.0, price=11.0),
        _kucoin_manager_fill("entry-2", ts + 120_000, side="buy", qty=1.0, price=20.0),
        _kucoin_manager_fill("close-2", ts + 180_000, side="sell", qty=-1.0, price=22.0),
    ]
    observations = [
        _kucoin_cycle_observation("", realized_pnl=1.25, update_time=ts + 240_000),
        _kucoin_cycle_observation("", realized_pnl=2.5, update_time=ts + 180_000),
    ]
    manager = FillEventsManager(
        exchange="kucoin",
        user="default",
        fetcher=_StaticFetcher(events, observations),
        cache_path=tmp_path / "ambiguous_cycles",
    )

    caplog.set_level(logging.WARNING, logger=fem.logger.name)
    await manager.refresh()

    by_id = {event.id: event for event in manager.get_events(symbol="TON/USDT:USDT")}
    assert by_id["close-1"].pnl == pytest.approx(1.0)
    assert by_id["close-2"].pnl == pytest.approx(2.0)
    assert by_id["close-1"].pnl_source == fem.PNL_SOURCE_SYNTHETIC_EXACT
    assert by_id["close-2"].pnl_source == fem.PNL_SOURCE_SYNTHETIC_EXACT
    assert any(
        "deferred aggregate realized PnL observation matching" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_manager_mid_position_cycle_observation_stays_degraded(
    caplog,
    tmp_path: Path,
):
    ts = 1_700_000_000_000
    events = [
        _kucoin_manager_fill(
            "close-1a",
            ts + 60_000,
            side="sell",
            qty=-0.4,
            price=11.0,
        ),
        _kucoin_manager_fill(
            "close-1b",
            ts + 61_000,
            side="sell",
            qty=-0.6,
            price=12.0,
        ),
    ]
    observations = [
        _kucoin_cycle_observation("100", realized_pnl=1.25, close_time=ts + 61_000),
    ]
    manager = FillEventsManager(
        exchange="kucoin",
        user="default",
        fetcher=_StaticFetcher(events, observations),
        cache_path=tmp_path / "mid_position",
    )

    caplog.set_level(logging.WARNING, logger=fem.logger.name)
    await manager.refresh()

    by_id = {event.id: event for event in manager.get_events(symbol="TON/USDT:USDT")}
    assert by_id["close-1a"].pnl_source == fem.PNL_SOURCE_SYNTHETIC_DEGRADED
    assert by_id["close-1b"].pnl_source == fem.PNL_SOURCE_SYNTHETIC_DEGRADED
    assert by_id["close-1a"].pnl_synthetic_reason == "incomplete_position_basis"
    assert by_id["close-1b"].pnl_synthetic_reason == "incomplete_position_basis"
    assert any(
        "deferred aggregate realized PnL observation matching" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_kucoin_fetcher_empty_fetch_logs_debug(monkeypatch, caplog):
    fetcher = KucoinFetcher(api=object())

    async def _fetch_trades(since_ms, until_ms):
        return []

    monkeypatch.setattr(fetcher, "_fetch_trades", _fetch_trades)

    with caplog.at_level(logging.DEBUG, logger=fem.logger.name):
        out = await fetcher.fetch(
            since_ms=1_700_000_000_000,
            until_ms=None,
            detail_cache={},
        )

    assert out == []
    records = [
        record
        for record in caplog.records
        if "KucoinFetcher: fetched 0 trade events" in record.message
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    assert not any(
        "KucoinFetcher: fetching fill history" in record.message
        and record.levelno >= logging.INFO
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_refresh_latest_does_not_extend_to_old_synthetic_pnl(tmp_path: Path):
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    old_ts = now_ms - 3 * 24 * 60 * 60 * 1000
    events = [
        _kucoin_manager_fill(
            "old-synthetic",
            old_ts,
            side="sell",
            qty=-1.0,
            price=10.0,
        )
    ]
    events[0]["pnl_status"] = "complete"
    events[0]["pnl_source"] = fem.PNL_SOURCE_SYNTHETIC_DEGRADED
    events[0]["pnl_synthetic_reason"] = "incomplete_position_basis"
    for idx in range(30):
        ts = now_ms - (30 - idx) * 60_000
        events.append(
            _kucoin_manager_fill(
                f"recent-{idx}",
                ts,
                side="buy",
                qty=1.0,
                price=10.0 + idx,
            )
        )
    fetcher = _StaticFetcher([])
    manager = FillEventsManager(
        exchange="kucoin",
        user="default",
        fetcher=fetcher,
        cache_path=tmp_path / "refresh_latest_synthetic_bound",
        fee_pct_fallback=0.0,
    )
    manager._events = [FillEvent.from_dict(ev) for ev in events]
    manager._loaded = True

    await manager.refresh_latest(overlap=20)

    assert fetcher.calls
    assert fetcher.calls[-1][0] is not None
    assert fetcher.calls[-1][0] > old_ts


def test_kucoin_pnl_discrepancy_logs_symbol_diagnostics(caplog):
    fetcher = KucoinFetcher(api=object())
    fem._pnl_discrepancy_last_log.clear()
    fem._pnl_discrepancy_last_delta.clear()
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    trades = [
        {
            "id": "close-1",
            "symbol": "TAO/USDT:USDT",
            "side": "sell",
            "position_side": "long",
        },
        {
            "id": "open-1",
            "symbol": "SUI/USDT:USDT",
            "side": "buy",
            "position_side": "long",
        },
    ]
    positions = [
        {
            "symbol": "TAO/USDT:USDT",
            "lastUpdateTimestamp": now_ms,
            "realizedPnl": 1.25,
        },
        {
            "symbol": "SUI/USDT:USDT",
            "lastUpdateTimestamp": now_ms + 1000,
            "realizedPnl": -0.25,
        },
    ]

    caplog.set_level(logging.DEBUG, logger=fem.logger.name)
    fetcher._log_discrepancies(
        {"close-1": 25.0, "open-1": 0.0},
        positions,
        trades,
    )

    messages = [rec.getMessage() for rec in caplog.records]
    assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)
    assert any(
        "diagnostic trade-derived local sum 25.00 differs from authoritative positions_history 1.00"
        in msg
        for msg in messages
    )
    assert any("top=TAO/USDT:USDT:local=25.00,history=1.25" in msg for msg in messages)
    assert any(
        "KucoinFetcher reconciliation detail symbol=TAO/USDT:USDT" in msg
        and "close_fills=1" in msg
        and "KuCoin contract multiplier/PnL model mismatch" in msg
        for msg in messages
    )


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
async def test_gateio_fetcher_uses_market_contract_size_for_fee_notional(monkeypatch):
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    trades = [
        _make_gateio_trade(
            trade_id="trade-doge",
            order_id="order-doge",
            ts=now_ms - 100,
            symbol="DOGE/USDT:USDT",
            amount=1.0,
            price=0.20,
            fee_cost=0.004,
        ),
    ]
    orders = [_make_gateio_order(order_id="order-doge", ts=now_ms - 100)]
    api = _FakeGateioAPI([trades], [orders])
    api.markets = {"DOGE/USDT:USDT": {"id": "DOGE_USDT", "contractSize": 100.0}}
    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda v: v or "unknown")

    fetcher = GateioFetcher(api, trade_limit=100)
    events = await fetcher.fetch(
        since_ms=now_ms - 1_000,
        until_ms=now_ms,
        detail_cache={},
    )

    assert events[0]["c_mult"] == pytest.approx(100.0)
    fee_paid, meta = fem._normalize_fee_paid_from_payload(
        events[0],
        fee_pct_fallback=0.0002,
        fee_pct_sanity_abs_max=0.001,
    )
    assert fee_paid == pytest.approx(-0.004)
    assert meta["fee_notional"] == pytest.approx(20.0)
    assert meta["fee_ratio"] == pytest.approx(-0.0002)
    assert meta["fee_quality"] == fem.FEE_QUALITY_EXACT


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


def test_hyperliquid_raw_start_position_overrides_wrong_reconstructed_psize():
    events = [
        {
            "id": "hl-close",
            "timestamp": 1779396722362,
            "symbol": "HYPE/USDC:USDC",
            "side": "sell",
            "qty": 496.4,
            "price": 56.663393493150686,
            "pnl": -548.82307,
            "position_side": "long",
            "raw": [
                {
                    "source": "fetch_my_trades",
                    "data": {
                        "id": "5",
                        "side": "sell",
                        "amount": 12.45,
                        "price": 56.64,
                        "pnl": -13.8,
                        "info": {
                            "tid": "5",
                            "side": "sell",
                            "sz": "12.45",
                            "px": "56.64",
                            "closedPnl": "-13.8",
                            "startPosition": "12.45",
                            "dir": "Close Long",
                        },
                    },
                },
                {
                    "source": "fetch_my_trades",
                    "data": {
                        "id": "1",
                        "side": "sell",
                        "amount": 119.0,
                        "price": 56.66,
                        "pnl": -131.5,
                        "info": {
                            "tid": "1",
                            "side": "sell",
                            "sz": "119",
                            "px": "56.66",
                            "closedPnl": "-131.5",
                            "startPosition": "496.4",
                            "dir": "Close Long",
                        },
                    },
                },
            ],
        }
    ]
    ensure_qty_signage(events)
    compute_psize_pprice(
        events,
        {("HYPE/USDC:USDC", "long"): (739.53, 56.79734)},
    )

    assert events[0]["psize"] == pytest.approx(243.13)

    apply_hyperliquid_raw_psize_overrides(events)

    assert events[0]["psize"] == 0.0
    assert events[0]["pprice"] == 0.0
