import asyncio
from dataclasses import replace

import pytest

from live import hsl_runtime
from live.hsl_candles import Sources
from test_hsl_ordinary_freshness import planning_bot
from test_hsl_runtime import NOW, SYMBOL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "raw_balance",
        "position",
        "fills",
        "fill_receipt",
        "candle_receipt",
        "no_publication",
        "slow_quote",
    ],
)
async def test_hsl_disabled_ordinary_waits_for_existing_account_refresh(monkeypatch, change):
    fixture = planning_bot(monkeypatch, stage="ticker")
    bot, owner = fixture.bot, fixture.owner
    # Retain an earlier receipt so identity changes can be exercised even
    # though disabled HSL will no longer acquire one itself.
    bot._hsl_fill_observation = hsl_runtime.observe_fills(bot, (NOW, NOW))
    bot.config["bot"]["long"]["hsl"]["enabled"] = False
    assert not hsl_runtime.enabled(bot)
    owner.sources[SYMBOL] = Sources((), (), 0)
    ledger = bot._ensure_freshness_ledger()
    quotes = bot._get_orchestrator_market_snapshots
    quote_epochs = []
    quoted = asyncio.Event()

    async def track_quotes(symbols):
        quote_epochs.append(ledger.epoch)
        if change == "slow_quote" and len(quote_epochs) == 2:
            fixture.clock[0] += 11_000
        result = await quotes(symbols)
        quoted.set()
        return result

    bot._get_orchestrator_market_snapshots = track_quotes
    task = asyncio.create_task(owner._ordinary_plan())
    try:
        await asyncio.wait_for(fixture.entered.wait(), 1)
        async with owner._refresh_lock:
            # A read starts after quote acquisition begins, with the old raw
            # balance still published when that quote is ready for planning.
            fixture.clock[0] += 15_000
            ledger.begin_epoch(now_ms=fixture.clock[0])
            fixture.release.set()
            await asyncio.wait_for(quoted.wait(), 1)
            assert not fixture.captures and not task.done()
            if change != "no_publication":
                for surface in ("balance", "positions", "open_orders"):
                    ledger.stamp(surface, now_ms=fixture.clock[0])
            if change == "raw_balance":
                bot.get_raw_balance = lambda: 1000.001
            elif change == "position":
                bot.positions[SYMBOL]["long"]["size"] += 1.0
            elif change == "fills":
                ledger.stamp("fills", ("new_fill",), now_ms=fixture.clock[0])
            elif change == "fill_receipt":
                bot._hsl_fill_observation = replace(bot._hsl_fill_observation)
            elif change == "candle_receipt":
                owner.sources[SYMBOL] = replace(owner.sources[SYMBOL])
        plan = await asyncio.wait_for(task, 1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert not fixture.calls, "the ordinary task must not start another account read"
    if change == "raw_balance":
        assert plan is not None and plan[3].balance == 1000.001
        assert owner._account_matches(plan[3], fixture.clock[0])
        assert quote_epochs == [0, 1]
        assert not fixture.defers
        bot.get_raw_balance = lambda: 1000.002
        assert owner._account_mismatch_reason(plan[3], fixture.clock[0]) == "raw_balance_changed"
    else:
        assert plan is None
        expected = {
            "position": "account_facts_changed_during_preparation",
            "fills": "account_facts_changed_during_preparation",
            "fill_receipt": "hsl_fill_receipt_replaced",
            "candle_receipt": "hsl_candle_receipt_replaced",
            "no_publication": "current_account_inputs_unavailable",
            "slow_quote": "current_account_inputs_unavailable",
        }
        assert fixture.defers[-1]["defer_reason"] == expected[change]


@pytest.mark.asyncio
async def test_hsl_disabled_ordinary_without_refresh_preserves_existing_preparation(monkeypatch):
    fixture = planning_bot(monkeypatch)
    fixture.bot.config["bot"]["long"]["hsl"]["enabled"] = False
    fixture.release.set()
    plan = await fixture.owner._ordinary_plan()
    assert plan is not None and fixture.owner._account_matches(plan[3], fixture.clock[0])
    assert not fixture.calls and not fixture.defers
