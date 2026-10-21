"""Live inception discovery follows the sides which consume the fact."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import passivbot as live
from config.schema import get_template_config


FIRST = 1_704_067_200_000
SIDES = ("long", "short")


def _bot(*, ages=(2.5, 0.0), enabled=(True, True), exchange="binance"):
    bot = object.__new__(live.Passivbot)
    bot.config = get_template_config()
    for side, days, active in zip(SIDES, ages, enabled):
        bot.config["bot"][side]["universe"]["minimum_coin_age_days"] = days
        bot.config["bot"][side]["risk"]["total_wallet_exposure_limit"] = 1.0 if active else 0.0
        bot.config["bot"][side]["risk"]["max_positions"] = 1
    bot.exchange = exchange
    bot.first_timestamps = {}
    bot.approved_coins_minus_ignored_coins = {
        "long": {"LONG/USDT:USDT"},
        "short": {f"SHORT{index:02d}/USDT:USDT" for index in range(24)},
    }
    bot.markets_dict = {
        symbol: {}
        for symbols in bot.approved_coins_minus_ignored_coins.values()
        for symbol in symbols
    }
    bot.coin_to_symbol = lambda value, **_kwargs: value
    return bot


@pytest.mark.asyncio
@pytest.mark.parametrize("planning", [False, True], ids=["direct", "planning"])
@pytest.mark.parametrize(
    "ages,enabled,required_sides",
    [
        ((2.5, 0.0), (True, True), ("long",)),
        ((0.0, 2.5), (True, True), ("short",)),
        ((2.5, 3.0), (True, False), ("long",)),
        ((2.5, 3.0), (False, True), ("short",)),
        ((2.5, 0.0), (False, True), ()),
        ((0.0, 2.5), (True, False), ()),
        ((0.0, 0.0), (True, True), ()),
        ((2.5, 3.0), (False, False), ()),
        ((2.5, 3.0), (True, True), SIDES),
    ],
)
async def test_default_discovery_only_requests_enabled_positive_age_sides(
    monkeypatch, ages, enabled, required_sides, planning,
):
    bot = _bot(ages=ages, enabled=enabled)
    lookup = AsyncMock(side_effect=lambda symbols, **_kwargs: {symbol: FIRST for symbol in symbols})
    monkeypatch.setattr(live, "get_first_timestamps_unified", lookup)
    if planning:
        bot.update_effective_min_cost = AsyncMock()
        bot.refresh_approved_ignored_coins_lists = lambda: None
        bot.set_wallet_exposure_limits = lambda: None
        bot._assert_supported_live_state = lambda: None
        bot._build_live_symbol_universe = lambda: []
        await bot.prepare_planning_universe()
    else:
        await bot.update_first_timestamps()

    expected = sorted({
        symbol
        for side in required_sides
        for symbol in bot.approved_coins_minus_ignored_coins[side]
    })
    if expected:
        lookup.assert_awaited_once_with(expected, exchange="binance")
    else:
        lookup.assert_not_awaited()
    assert bot.first_timestamps == {symbol: FIRST for symbol in expected}


@pytest.mark.asyncio
@pytest.mark.parametrize("disabled_by", ["exposure", "positions"])
async def test_positive_age_disabled_side_does_not_acquire_facts(monkeypatch, disabled_by):
    bot = _bot(ages=(2.5, 0.0))
    if disabled_by == "exposure":
        bot.config["bot"]["long"]["risk"]["total_wallet_exposure_limit"] = 0.0
    else:
        bot.config["bot"]["long"]["risk"]["max_positions"] = 0
    lookup = AsyncMock(side_effect=AssertionError("disabled sides do not require inception"))
    monkeypatch.setattr(live, "get_first_timestamps_unified", lookup)
    await bot.update_first_timestamps()
    lookup.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("requested", [[], ["MANUAL/USDT:USDT"], ["SHORT00/USDT:USDT"]])
async def test_explicit_symbol_request_is_not_widened_to_any_side_universe(monkeypatch, requested):
    bot = _bot()
    lookup = AsyncMock(side_effect=lambda symbols, **_kwargs: {symbol: FIRST for symbol in symbols})
    monkeypatch.setattr(live, "get_first_timestamps_unified", lookup)
    await bot.update_first_timestamps(requested)
    if requested:
        lookup.assert_awaited_once_with(requested, exchange="binance")
    else:
        lookup.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_known_alias_repairs_symbol_without_fetch_or_losing_positive_fact(
    monkeypatch, explicit,
):
    bot = _bot(ages=(0.0, 0.0))
    symbol = "LONG/USDT:USDT"
    bot.first_timestamps = {"LONG": FIRST, symbol: 0.0}
    bot.coin_to_symbol = lambda value, **_kwargs: symbol if value == "LONG" else value
    lookup = AsyncMock(side_effect=AssertionError("known factual aliases need no lookup"))
    monkeypatch.setattr(live, "get_first_timestamps_unified", lookup)
    await bot.update_first_timestamps([symbol] if explicit else None)
    assert bot.first_timestamps == {"LONG": FIRST, symbol: FIRST}
    lookup.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", [None, 0.0])
async def test_only_required_unknown_fact_retries_while_known_and_unused_facts_survive(
    monkeypatch, unknown,
):
    bot = _bot()
    required = "LONG/USDT:USDT"
    known = "KNOWN/USDT:USDT"
    unused = "SHORT00/USDT:USDT"
    bot.approved_coins_minus_ignored_coins["long"].add(known)
    bot.first_timestamps = {required: unknown, known: FIRST, unused: 0.0}
    lookup = AsyncMock(side_effect=[{required: 0.0}, {required: FIRST}])
    monkeypatch.setattr(live, "get_first_timestamps_unified", lookup)

    await bot.update_first_timestamps()
    await bot.update_first_timestamps()
    await bot.update_first_timestamps()

    assert lookup.await_count == 2
    assert all(call.args[0] == [required] for call in lookup.await_args_list)
    assert bot.first_timestamps == {required: FIRST, known: FIRST, unused: 0.0}


@pytest.mark.asyncio
@pytest.mark.parametrize("positive_side", SIDES)
async def test_fake_harness_only_reads_required_side_candles(monkeypatch, positive_side):
    ages = tuple(2.5 if side == positive_side else 0.0 for side in SIDES)
    bot = _bot(ages=ages, exchange="fake")
    lookup = AsyncMock(side_effect=AssertionError("fake discovery must remain offline"))
    monkeypatch.setattr(live, "get_first_timestamps_unified", lookup)
    bot.cca = SimpleNamespace(fetch_ohlcv=AsyncMock(return_value=[[FIRST, 1, 1, 1, 1, 1]]))

    await bot.update_first_timestamps()

    expected = sorted(bot.approved_coins_minus_ignored_coins[positive_side])
    assert [call.args[0] for call in bot.cca.fetch_ohlcv.await_args_list] == expected
    assert all(
        call.kwargs == {"timeframe": "1m", "since": 1}
        for call in bot.cca.fetch_ohlcv.await_args_list
    )
    assert bot.first_timestamps == {symbol: FIRST for symbol in expected}
    lookup.assert_not_awaited()
