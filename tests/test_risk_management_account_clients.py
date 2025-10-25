import asyncio
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from risk_management.account_clients import CCXTAccountClient, BaseError


class StubExchange:
    def __init__(self, *, bid: float | None = None, ask: float | None = None, last: float | None = None):
        self._bid = bid
        self._ask = ask
        self._last = last
        self._cancel_calls = []
        self._orders = []
        self.markets = True

    async def cancel_all_orders(self, symbol=None, params=None):
        self._cancel_calls.append({"symbol": symbol, "params": params})

    async def fetch_positions(self, params=None):
        return [
            {
                "symbol": "BTC/USDT",
                "contracts": 1,
                "info": {},
            }
        ]

    async def fetch_order_book(self, symbol):
        raise BaseError("order book unavailable")

    async def fetch_ticker(self, symbol):
        payload = {"info": {}}
        if self._bid is not None:
            payload["bid"] = self._bid
        if self._ask is not None:
            payload["ask"] = self._ask
        if self._last is not None:
            payload["last"] = self._last
        return payload

    async def create_order(self, symbol, order_type, side, amount, price, params=None):
        self._orders.append(
            {
                "symbol": symbol,
                "type": order_type,
                "side": side,
                "amount": amount,
                "price": price,
                "params": params,
            }
        )


def test_kill_switch_falls_back_to_ticker_price(caplog):
    exchange = StubExchange(bid=101.2)
    client = CCXTAccountClient.__new__(CCXTAccountClient)
    client.config = SimpleNamespace(name="Demo", symbols=None)
    client.client = exchange
    client._balance_params = {}
    client._positions_params = {}
    client._orders_params = {}
    client._close_params = {}
    client._markets_loaded = None
    client._debug_api_payloads = False

    caplog.set_level(logging.INFO, "risk_management")

    summary = asyncio.run(client.kill_switch("BTC/USDT"))

    assert summary["closed_positions"], "Position should be closed when ticker price is available"
    order = exchange._orders[0]
    assert order["price"] == pytest.approx(101.2)
    assert order["params"]["reduceOnly"] is True
    assert any("Executing kill switch" in record.message for record in caplog.records)
    assert any("Kill switch completed" in record.message for record in caplog.records)


def test_kill_switch_logs_failures_when_price_missing(caplog):
    exchange = StubExchange()
    client = CCXTAccountClient.__new__(CCXTAccountClient)
    client.config = SimpleNamespace(name="Demo", symbols=None)
    client.client = exchange
    client._balance_params = {}
    client._positions_params = {}
    client._orders_params = {}
    client._close_params = {}
    client._markets_loaded = None
    client._debug_api_payloads = False

    caplog.set_level(logging.DEBUG, "risk_management")

    summary = asyncio.run(client.kill_switch("BTC/USDT"))

    assert not exchange._orders, "Order should not be placed when price cannot be determined"
    assert summary["failed_position_closures"], "Failure should be recorded when price is missing"
    assert any("Kill switch completed" in record.message for record in caplog.records)
    # Debug details are only emitted when failures occur
    assert any("Kill switch details" in record.message for record in caplog.records)
