"""Bitget attribution metadata must not exclude executions from accounting."""

from copy import deepcopy

import pytest

from fill_events_manager import BitgetFetcher, FillEventsManager
from test_unstucking_safeguards import (
    _dummy_config,
    _make_dummy_bot,
    _set_basic_state,
    _make_candles,
)

SYMBOL = "TEST/USDT:USDT"
T0 = 1_700_000_000_000


class BitgetFixtureAPI:
    def __init__(self, mode, pside, include_external=True):
        self.options = {"uta": mode == "uta"}
        entry_side = "buy" if pside == "long" else "sell"
        close_side = "sell" if pside == "long" else "buy"
        # A complete old round trip, followed by two partial opening fills.
        specifications = [
            ("old-entry", 0, entry_side, "open", 1.0, 100.0, "bot-old"),
            ("external-close", 60_000, close_side, "close", 1.0, 101.0, ""),
            ("new-a", 120_000, entry_side, "open", 0.4, 110.0, "bot-new"),
            ("new-b", 120_000, entry_side, "open", 0.6, 110.0, "bot-new"),
        ]
        if not include_external:
            specifications[1] = (*specifications[1][:-1], "bot-close")
        self.cids = {s[0]: s[-1] for s in specifications}
        self.rows = []
        for event_id, delta, side, action, qty, price, cid in specifications:
            pnl = (
                (price - 100.0) * (1 if pside == "long" else -1)
                if action == "close"
                else 0.0
            )
            if mode == "classic":
                row = dict(
                    tradeId=event_id,
                    orderId=event_id,
                    cTime=str(T0 + delta),
                    symbol="TESTUSDT",
                    side=side,
                    tradeSide=action,
                    posMode="hedge_mode",
                    baseVolume=str(qty),
                    price=str(price),
                    profit=str(pnl),
                    feeDetail=[{"feeCoin": "USDT", "totalFee": "0"}],
                )
            else:
                row = dict(
                    execId=event_id,
                    orderId=event_id,
                    createdTime=str(T0 + delta),
                    symbol="TESTUSDT",
                    side=side,
                    tradeSide=action,
                    posSide=pside,
                    execQty=str(qty),
                    execPrice=str(price),
                    execValue=str(qty * price),
                    execPnl=str(pnl),
                    clientOid=cid,
                    feeDetail=[{"feeCoin": "USDT", "fee": "0"}],
                )
            self.rows.append(row)

    async def private_mix_get_v2_mix_order_fill_history(self, params):
        rows = [r for r in reversed(self.rows) if int(r["cTime"]) <= params["endTime"]]
        return {"data": {"fillList": deepcopy(rows)}}

    async def private_mix_get_v2_mix_order_detail(self, params):
        return {"data": {"clientOid": self.cids[params["orderId"]]}}

    async def private_uta_get_v3_trade_fills(self, params):
        rows = [
            r
            for r in reversed(self.rows)
            if params.get("startTime", 0) <= int(r["createdTime"]) <= params["endTime"]
        ]
        return {"data": {"list": deepcopy(rows)}}


def make_manager(tmp_path, mode, pside, include_external=True):
    api = BitgetFixtureAPI(mode, pside, include_external)
    fetcher = BitgetFetcher(api, symbol_resolver=lambda value: SYMBOL)
    manager = FillEventsManager(
        exchange="bitget",
        user="offline_fixture",
        fetcher=fetcher,
        cache_path=tmp_path / "cache",
    )
    return api, fetcher, manager


def make_bot(manager, pside, *, initial_size=1.0):
    bot = _make_dummy_bot(_dummy_config())
    _set_basic_state(bot, SYMBOL)
    bot.exchange = "bitget"
    bot._pnls_manager = manager
    bot.is_trailing = lambda symbol, side=None: side == pside
    bot.get_exchange_time = lambda: T0 + 300_000
    bot._apply_positions_snapshot(
        [
            dict(
                symbol=SYMBOL,
                position_side=pside,
                size=initial_size,
                price=110.0 if initial_size else 0.0,
                lastUpdateTimestamp=T0 + 120_016,
            )
        ]
    )
    bot._trailing_fill_fetch_generation = 1
    bot._trailing_fill_refresh_started_generation = 1

    async def candles(*args, **kwargs):
        start = ((T0 + 120_000) // 60_000 + 1) * 60_000
        return _make_candles(
            [
                (ts, 110.0, 112.0, 109.0, 111.0, 1.0)
                for ts in range(start, T0 + 300_000, 60_000)
            ]
        )

    bot.cm.get_candles = candles
    return bot


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["classic", "uta"])
@pytest.mark.parametrize("pside", ["long", "short"])
async def test_complete_fetch_retains_external_close_and_confirms_position(
    tmp_path, mode, pside
):
    api, fetcher, manager = make_manager(tmp_path, mode, pside)
    returned = await fetcher.fetch(T0 - 1, T0 + 300_000, {})
    assert {e["id"] for e in returned} == {
        "old-entry",
        "external-close",
        "new-a",
        "new-b",
    }
    await manager.refresh(start_ms=T0 - 1, end_ms=T0 + 300_000)
    bot = make_bot(manager, pside)
    await bot.update_trailing_data()
    detail = bot._trailing_fill_confirmation_diagnostics
    observed = {e.id: (e.psize, e.pprice) for e in manager.get_events()}
    assert "external-close" in observed
    assert observed["external-close"] == (0.0, 0.0)
    assert manager.get_events()[-1].psize == pytest.approx(1.0)
    assert manager.get_events()[-1].pprice == pytest.approx(110.0)
    assert sum(e.pnl for e in manager.get_events()) == pytest.approx(
        1.0 if pside == "long" else -1.0
    )
    assert not detail
    assert bot._trailing_pending_fill_confirmations == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["classic", "uta"])
async def test_all_attributed_fills_confirm_normally(tmp_path, mode):
    api, fetcher, manager = make_manager(
        tmp_path, mode, "short", include_external=False
    )
    await manager.refresh(start_ms=T0 - 1, end_ms=T0 + 300_000)
    bot = make_bot(manager, "short")
    await bot.update_trailing_data()
    assert bot._trailing_pending_fill_confirmations == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["classic", "uta"])
async def test_callback_and_return_have_same_source_id_set(tmp_path, mode):
    api, fetcher, manager = make_manager(tmp_path, mode, "short")
    batches = []
    returned = await fetcher.fetch(T0 - 1, T0 + 300_000, {}, on_batch=batches.extend)
    assert {e["id"] for e in batches} == {e["id"] for e in returned}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["classic", "uta"])
async def test_unattributed_only_batch_is_not_lost(tmp_path, mode):
    api, fetcher, manager = make_manager(tmp_path, mode, "short")
    api.rows = [api.rows[1]]
    await manager.refresh(start_ms=T0 - 1, end_ms=T0 + 300_000)
    assert [e.id for e in manager.get_events()] == ["external-close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["classic", "uta"])
async def test_full_backfill_repairs_missing_close_and_survives_restart(tmp_path, mode):
    api, fetcher, manager = make_manager(tmp_path, mode, "short")
    complete_rows = deepcopy(api.rows)
    # Model a cache previously populated while an external close was omitted.
    api.rows = [
        r for r in api.rows if r.get("tradeId", r.get("execId")) != "external-close"
    ]
    await manager.refresh(start_ms=T0 - 1, end_ms=T0 + 300_000)
    bot = make_bot(manager, "short")
    await bot.update_trailing_data()
    assert bot._trailing_fill_confirmation_diagnostics[(SYMBOL, "short")][
        "failed_predicates"
    ] == ["fill_after_state_mismatch"]
    api.rows = complete_rows
    # Re-fetching only the new entry cannot repair an older missing close.
    for generation in (2, 3, 4):
        await manager.refresh(start_ms=T0 + 120_000, end_ms=T0 + 300_000)
        bot._trailing_fill_fetch_generation = generation
        await bot.update_trailing_data()
        assert bot._trailing_fill_confirmation_diagnostics[(SYMBOL, "short")][
            "failed_predicates"
        ] == ["fill_after_state_mismatch"]
    # Re-fetch the complete affected history, retaining source identities.
    await manager.refresh(start_ms=T0 - 1, end_ms=T0 + 300_000)
    bot._trailing_fill_fetch_generation = 5
    await bot.update_trailing_data()
    assert bot._trailing_pending_fill_confirmations == {}
    assert len(manager.get_events()) == 4
    await manager.refresh(start_ms=T0 - 1, end_ms=T0 + 300_000)
    assert len(manager.get_events()) == 4
    restarted = FillEventsManager(
        exchange="bitget",
        user="offline_fixture",
        fetcher=fetcher,
        cache_path=tmp_path / "cache",
    )
    await restarted.ensure_loaded()
    assert {e.id for e in restarted.get_events()} == {
        "old-entry",
        "external-close",
        "new-a",
        "new-b",
    }
    restarted_bot = make_bot(restarted, "short")
    await restarted.refresh(start_ms=T0 - 1, end_ms=T0 + 300_000)
    await restarted_bot.update_trailing_data()
    assert restarted_bot._trailing_pending_fill_confirmations == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["classic", "uta"])
async def test_flat_start_then_staggered_partial_fill_delivery(tmp_path, mode):
    api, fetcher, manager = make_manager(tmp_path, mode, "short")
    complete_rows = deepcopy(api.rows)
    api.rows = complete_rows[:2]
    await manager.refresh(start_ms=T0 - 1, end_ms=T0 + 300_000)
    bot = make_bot(manager, "short", initial_size=0.0)
    # REST position sees the complete order, while fill history initially
    # exposes only its first execution. A later refresh delivers the second.
    api.rows = complete_rows[:3]
    await manager.refresh(start_ms=T0 - 1, end_ms=T0 + 300_000)
    bot._apply_positions_snapshot(
        [
            dict(
                symbol=SYMBOL,
                position_side="short",
                size=1.0,
                price=110.0,
                lastUpdateTimestamp=T0 + 120_016,
            )
        ]
    )
    api.rows = complete_rows
    await manager.refresh(start_ms=T0 - 1, end_ms=T0 + 300_000)
    bot._trailing_fill_fetch_generation = 2
    await bot.update_trailing_data()
    assert bot._trailing_pending_fill_confirmations == {}
