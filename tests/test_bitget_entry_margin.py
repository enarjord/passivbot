"""Bitget UTA margin rejections are terminal and local to the rejected entry."""

import asyncio
import json
from types import SimpleNamespace

import ccxt
import pytest

from config.schema import get_expanded_config_prototype
from exchanges import bitget
from live import executor
from live.event_bus import ReasonCodes


@pytest.fixture
def connector(monkeypatch):
    bot = bitget.BitgetBot.__new__(bitget.BitgetBot)
    bot.config = get_expanded_config_prototype()
    bot.is_uta = True
    bot._entry_margin_retry_after = {}
    bot._hsl_live = SimpleNamespace(_write_lock=asyncio.Lock(), admit=lambda order: True)
    calls, confirmations, budget = [], [], []
    clock = [1.0]
    code = ["25203"]
    client = ccxt.bitget()

    async def create_order(**kwargs):
        calls.append(kwargs)
        if code[0] is not None:
            response = {"code": code[0], "msg": "Insufficient margin", "data": None}
            client.handle_errors(
                200, "OK", "/api/v3/trade/place-order", "POST", {},
                json.dumps(response), response, None, None,
            )
        return {"id": "accepted", "status": "open"}

    async def charge_budget(failures):
        budget.extend(failures)

    bot.cca = SimpleNamespace(create_order=create_order)
    bot._request_authoritative_confirmation = confirmations.append
    bot._handle_order_write_failures = charge_budget
    bot._emit_execution_connector_call_started_event = lambda **kwargs: None
    monkeypatch.setattr(executor, "record_create_connector_admission", lambda *args: None)
    monkeypatch.setattr(bitget, "monotonic", lambda: clock[0])
    order = {
        "symbol": "TEST/USDT:USDT", "type": "limit", "side": "buy",
        "position_side": "long", "qty": 1.0, "price": 100.0, "reduce_only": False,
        "custom_id": "test",
    }
    return SimpleNamespace(
        bot=bot, order=order, calls=calls, confirmations=confirmations,
        budget=budget, clock=clock, code=code,
    )


@pytest.mark.asyncio
async def test_uta_entry_margin_rejection_uses_scope_local_pre_transport_retry(connector):
    c = connector
    reject, = await c.bot.execute_orders([c.order])
    assert executor._create_rejection_reason(reject) == "terminal_rejection"
    assert reject["info"]["code"] == "25203"
    assert not c.bot.did_create_order(reject)
    assert c.calls[0]["params"]["posSide"] == "long"
    assert c.confirmations == [{"balance", "positions", "open_orders"}]
    assert c.budget == []

    c.code[0] = None
    c.clock[0] += 59.0
    deferred, = await c.bot.execute_orders([c.order])
    assert deferred == executor.DeferredOrderCreation(ReasonCodes.EXCHANGE_MARGIN_RETRY)
    assert len(c.calls) == len(c.confirmations) == 1
    assert c.budget == []

    close = dict(c.order, side="sell", reduce_only=True)
    other_side = dict(c.order, side="sell", position_side="short")
    other_symbol = dict(c.order, symbol="OTHER/USDT:USDT")
    for order in (close, other_side, other_symbol):
        assert (await c.bot.execute_orders([order]))[0]["id"] == "accepted"

    c.clock[0] += 1.0
    assert (await c.bot.execute_orders([c.order]))[0]["id"] == "accepted"
    assert c.bot._entry_margin_retry_after == {}
    assert len(c.calls) == len(c.confirmations) == 5


@pytest.mark.parametrize(
    ("uta", "order_changes", "code"),
    [
        (False, {}, "25203"),
        (True, {"reduce_only": True, "side": "sell"}, "25203"),
        (True, {"side": "sell"}, "25203"),
        (True, {}, "25212"),
    ],
)
@pytest.mark.asyncio
async def test_other_failures_retain_existing_order_write_error_budget(
    connector, uta, order_changes, code
):
    c = connector
    c.bot.is_uta = uta
    c.code[0] = code
    order = dict(c.order, **order_changes)
    out, = await c.bot.execute_orders([order])
    assert isinstance(out, ccxt.ExchangeError)
    assert c.budget == [("create", order, out)]
    assert c.bot._entry_margin_retry_after == {}
    assert c.confirmations == [{"balance", "positions", "open_orders"}]
