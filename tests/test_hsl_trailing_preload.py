"""Trailing acquisition owns its cohort but never changes readiness."""

import asyncio
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

from passivbot import Passivbot


class CandleReader:
    def __init__(self, read):
        self.read = read
        self.budgets = []
        self.active_budget = None

    @contextmanager
    def remote_fetch_budget(self, seconds):
        self.budgets.append(seconds)
        self.active_budget = seconds
        try:
            yield
        finally:
            self.active_budget = None

    async def get_candles_with_resolution_ladder(self, symbol, **kwargs):
        assert self.active_budget == 10.0
        assert kwargs == dict(start_ts=180_000, end_ts=None, strict=False)
        return await self.read(symbol)


def preload_bot(reader):
    return SimpleNamespace(
        cm=reader,
        _get_last_position_change_anchors=lambda: {
            symbol: {"long": {"timestamp": 120_000}}
            for symbol in ("A/USDT:USDT", "B/USDT:USDT")
        },
        has_position=lambda side, symbol: True,
        is_trailing=lambda symbol, side: True,
        trailing_prices={"keep": {"max_since_open": 7.0}},
        _trailing_pending_fill_confirmations={"keep": "pending"},
        _trailing_pending_fill_min_generations={"keep": 3},
        _trailing_pending_position_states={"keep": (2.0, 10.0)},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [None, OSError("unavailable"), ValueError("malformed"), RuntimeError("unexpected")])
async def test_preload_runs_complete_cohort_under_one_budget_without_derivation(error):
    entered, release = set(), asyncio.Event()

    async def read(symbol):
        entered.add(symbol)
        if len(entered) == 2:
            release.set()
        await release.wait()
        if symbol.startswith("A") and error is not None:
            raise error
        return object()

    reader = CandleReader(read)
    bot = preload_bot(reader)
    before = deepcopy({key: value for key, value in vars(bot).items()
                       if key.startswith("_trailing") or key == "trailing_prices"})
    if isinstance(error, (ValueError, RuntimeError)):
        with pytest.raises(type(error)):
            await Passivbot._preload_trailing_candles(bot)
    else:
        await Passivbot._preload_trailing_candles(bot)
    assert len(entered) == 2
    assert reader.budgets == [10.0]
    assert {key: value for key, value in vars(bot).items() if key in before} == before


@pytest.mark.asyncio
@pytest.mark.parametrize("resists_once", [False, True])
async def test_preload_cancellation_owns_and_observes_children(resists_once):
    entered, cancelled, finished = set(), set(), set()
    ready = asyncio.Event()
    children = []

    async def read(symbol):
        children.append(asyncio.current_task())
        entered.add(symbol)
        if len(entered) == 2:
            ready.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.add(symbol)
            if resists_once:
                await asyncio.sleep(0.001)
            raise
        finally:
            finished.add(symbol)

    reader = CandleReader(read)
    bot = preload_bot(reader)
    before = deepcopy(bot._trailing_pending_fill_confirmations)
    task = asyncio.create_task(Passivbot._preload_trailing_candles(bot))
    await ready.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1.0)
    assert len(entered) == len(cancelled) == len(finished) == 2
    assert all(child.done() for child in children)
    assert bot._trailing_pending_fill_confirmations == before
    assert reader.active_budget is None
