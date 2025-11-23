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

from src.fill_events_manager import (
    BaseFetcher,
    BinanceFetcher,
    BitgetFetcher,
    BybitFetcher,
    HyperliquidFetcher,
    FillEvent,
    FillEventCache,
    FillEventsManager,
    custom_id_to_snake,
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
    base_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    trades_batches = [
        [
            {
                "id": "trade-1",
                "timestamp": base_ts + 1_000,
                "amount": 0.1,
                "price": 100.0,
                "side": "buy",
                "symbol": "BTC/USDT",
                "fee": {"cost": 0.0001, "currency": "USDT"},
                "info": {
                    "orderId": "order-1",
                    "orderLinkId": "0xabc",
                    "closedSize": "0",
                },
            }
        ]
    ]
    positions_batches = [
        [
            {
                "timestamp": base_ts + 1_500,
                "symbol": "BTC/USDT",
                "realizedPnl": "5.0",
                "info": {
                    "orderId": "order-1",
                    "side": "Buy",
                },
            }
        ]
    ]

    api = _FakeBybitAPI(trades_batches, positions_batches)

    monkeypatch.setattr("src.fill_events_manager.custom_id_to_snake", lambda value: "entry")

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
    assert event["pb_order_type"] == "entry"
    assert event["pnl"] == pytest.approx(5.0)
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
    base_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    trades_batches = [
        [
            {
                "id": "trade-ord-1",
                "timestamp": base_ts,
                "amount": 2.0,
                "price": 10.0,
                "side": "sell",
                "symbol": "LTC/USDT:USDT",
                "info": {
                    "orderId": "order-dist",
                    "closedSize": "0",
                },
            },
            {
                "id": "trade-ord-2",
                "timestamp": base_ts + 1,
                "amount": 3.0,
                "price": 10.5,
                "side": "sell",
                "symbol": "LTC/USDT:USDT",
                "info": {
                    "orderId": "order-dist",
                    "closedSize": "0",
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
                    "closedSize": "5",
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
    # Expect proportional distribution: qty 2 vs 3, total pnl -5 -> [-2, -3]
    assert pnls[0] == pytest.approx(-2.0, rel=1e-9, abs=1e-9)
    assert pnls[1] == pytest.approx(-3.0, rel=1e-9, abs=1e-9)


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
