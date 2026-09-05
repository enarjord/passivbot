from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from exchanges.okx import OKXBot


def make_order(i):
    return {"id": str(i), "timestamp": i, "amount": 1.0,
            "symbol": "BTC/USDT:USDT", "side": "buy", "info": {"posSide": "long"}}


def make_bot(pages):
    bot = OKXBot.__new__(OKXBot)
    bot.cca = SimpleNamespace(fetch_open_orders=AsyncMock(side_effect=pages))
    bot._record_live_margin_mode_from_payload = lambda _: None
    return bot


@pytest.mark.asyncio
async def test_all_pages_are_normalized_using_oldest_id_cursor():
    # Parsed CCXT orders are ascending, despite the endpoint's descending order.
    bot = make_bot([[make_order(i) for i in range(106, 206)],
                    [make_order(i) for i in range(6, 106)],
                    [make_order(i) for i in range(1, 6)]])
    orders = await bot.fetch_open_orders()
    assert [o["id"] for o in orders] == [str(i) for i in range(1, 206)]
    assert all(o["position_side"] == "long" and o["qty"] == 1 for o in orders)
    calls = bot.cca.fetch_open_orders.await_args_list
    assert [c.kwargs["params"] for c in calls] == [
        {"paginate": False}, {"paginate": False, "after": "106"},
        {"paginate": False, "after": "6"}]
    assert all(c.kwargs["limit"] == 100 for c in calls)


@pytest.mark.asyncio
async def test_exact_full_page_requires_terminal_empty_and_preserves_symbol_filter():
    bot = make_bot([[make_order(i) for i in range(1, 101)], []])
    raw, normalized = await bot.capture_open_orders_snapshot("BTC/USDT:USDT")
    assert len(raw) == len(normalized) == 100
    assert all(c.kwargs["symbol"] == "BTC/USDT:USDT"
               for c in bot.cca.fetch_open_orders.await_args_list)


@pytest.mark.asyncio
async def test_later_page_failure_does_not_return_partial_snapshot():
    bot = make_bot([[make_order(i) for i in range(1, 101)], RuntimeError("offline error")])
    with pytest.raises(RuntimeError, match="offline error"):
        await bot.fetch_open_orders()


@pytest.mark.asyncio
async def test_overlapping_page_deduplicates_ids_while_cursor_advances():
    bot = make_bot([[make_order(i) for i in range(6, 106)],
                    [make_order(i) for i in range(1, 7)]])
    assert len(await bot.fetch_open_orders()) == 105


@pytest.mark.asyncio
@pytest.mark.parametrize("page", [[make_order(5)], [{"id": None}], None,
                                 [make_order(i) for i in range(101)]])
async def test_stalled_or_malformed_page_rejects_snapshot(page):
    bot = make_bot([[make_order(i) for i in range(5, 105)], page])
    with pytest.raises((ValueError, RuntimeError)):
        await bot.fetch_open_orders()


@pytest.mark.asyncio
async def test_page_budget_exhaustion_rejects_partial_snapshot():
    bot = make_bot([[make_order(i) for i in range(n, n + 100)]
                    for n in range(100000, 0, -100)])
    with pytest.raises(RuntimeError, match="exhausted"):
        await bot.fetch_open_orders()
