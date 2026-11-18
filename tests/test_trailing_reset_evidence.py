"""Offline coverage of Binance trailing reset proof and its v9 consumers."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from ccxt.base.errors import (
    AuthenticationError,
    ExchangeError,
    InvalidNonce,
    RateLimitExceeded,
    RequestTimeout,
)

from fill_events_manager import BinanceFetcher, FillEvent, FillEventsManager
from live import hsl_live
from test_unstucking_safeguards import (
    _dummy_config,
    _make_candles,
    _make_dummy_bot,
    _set_basic_state,
    mock_pbr,
)


TIMESTAMP = 180_000
SYMBOL = "TEST/USDT:USDT"


def trade(pside="long", **patch):
    return {
        "id": 42,
        "time": TIMESTAMP,
        "symbol": "TESTUSDT",
        "positionSide": pside.upper(),
        "side": "BUY" if pside == "long" else "SELL",
        "qty": "1.0",
        "price": "100.0",
        **patch,
    }


def manager_for(tmp_path, pside="long", *, portfolio_margin=False):
    api = SimpleNamespace(
        market=lambda _: {"id": "TESTUSDT", "linear": True, "swap": True},
        fapiPrivateGetUserTrades=AsyncMock(return_value=[trade(pside)]),
    )
    fetcher = BinanceFetcher(
        api, symbol_resolver=str, portfolio_margin=portfolio_margin
    )
    manager = FillEventsManager(
        exchange="binance", user="test", fetcher=fetcher, cache_path=tmp_path
    )
    manager._events = [
        FillEvent.from_dict(
            {
                "id": "42",
                "symbol": SYMBOL,
                "timestamp": TIMESTAMP,
                "side": "buy" if pside == "long" else "sell",
                "position_side": pside,
                "qty": 1.0 if pside == "long" else -1.0,
                "price": 100.0,
                "pnl": 0.0,
                "pb_order_type": "",
                "client_order_id": "",
                "psize": 2.0,
                "pprice": 100.0,
                "source_ids": ["42"],
            }
        )
    ]
    return manager, api.fapiPrivateGetUserTrades


async def pending_bot(tmp_path, pside="long", **position_patch):
    bot = _make_dummy_bot(_dummy_config())
    _set_basic_state(bot, SYMBOL)
    manager, query = manager_for(tmp_path, pside)
    bot.exchange = "binance"
    bot._pnls_manager = manager
    bot.is_trailing = lambda symbol, side: side == pside
    bot.get_exchange_time = lambda: 361_000
    bot._apply_positions_snapshot(
        [
            {
                "symbol": SYMBOL,
                "position_side": pside,
                "size": 1.0 if pside == "long" else -1.0,
                "price": 100.0,
                "lastUpdateTimestamp": TIMESTAMP,
                **position_patch,
            }
        ]
    )

    async def candles(*args, **kwargs):
        return _make_candles(
            [
                (240_000, 100.0, 102.0, 99.0, 100.0, 1.0),
                (300_000, 100.0, 103.0, 99.0, 100.0, 1.0),
            ]
        )

    bot.cm.get_candles = candles
    await bot.update_trailing_data()
    await bot._refresh_trailing_reset_evidence(1)
    query.assert_not_awaited()
    bot._trailing_fill_fetch_generation = 1
    bot._authoritative_pending_confirmations = {}
    await bot.update_trailing_data()
    assert (SYMBOL, pside) in bot._trailing_pending_fill_confirmations
    return bot, manager, query


@pytest.mark.asyncio
@pytest.mark.parametrize("pside", ["long", "short"])
async def test_exact_fresh_probe_recovers_after_restart_and_survives_revalidation(
    tmp_path, pside
):
    bot, manager, query = await pending_bot(tmp_path, pside)
    key = SYMBOL, pside
    events = manager.get_events()
    cache_files = list(tmp_path.iterdir())
    source_calls = []
    bot._monitor_record_event = (
        lambda kind, tags, payload, **kwargs: source_calls.append((kind, payload))
    )
    # The optional proof cannot alter the canonical HSL fill observation.
    bot._hsl_fill_observation = object()
    owner = hsl_live.owner(bot)
    prepared = owner.ordinary_receipts()
    await bot._refresh_trailing_reset_evidence(2)
    assert (
        prepared.mismatch_reason(owner.ordinary_receipts())
        == "hsl_trailing_reset_evidence_replaced"
    )
    assert prepared.fills is bot._hsl_fill_observation
    assert bot._trailing_fill_history_recovery_start_ms(None) is None
    for _ in range(2):
        current = await owner.prepare_ordinary_consumption(None)
        assert current.matches(owner.ordinary_receipts())
        assert key not in bot._trailing_pending_fill_confirmations
        assert bot.trailing_prices[SYMBOL][pside]["max_since_open"] == 103.0
        bot._trailing_fill_fetch_generation += 1
        bot._prune_trailing_reset_evidence(check_fills=True)
    query.assert_awaited_once_with(
        {
            "symbol": "TESTUSDT",
            "startTime": TIMESTAMP,
            "endTime": TIMESTAMP,
            "limit": 1000,
        }
    )
    assert manager.get_events() == events
    assert list(tmp_path.iterdir()) == cache_files
    assert [
        payload["recovery"]
        for kind, payload in source_calls
        if kind == "fill.position_open_boundary_recovery_used"
    ] == ["position_update_fresh_single_fill"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        [],
        [trade(), trade(id=43)],
        [trade(time=TIMESTAMP + 1)],
        [trade(symbol="OTHERUSDT")],
        [trade(positionSide="BOTH")],
        [trade(id=None)],
        [trade(side="SELL")],
        [trade(price="NaN")],
        [trade(price="bad")],
        [trade(price="100.01")],
    ],
)
async def test_incomplete_or_contradictory_probe_keeps_trailing_pending(
    tmp_path, response
):
    bot, _, query = await pending_bot(tmp_path)
    owner = hsl_live.owner(bot)
    initial_receipts = owner.ordinary_receipts()
    query.return_value = response
    await bot._refresh_trailing_reset_evidence(2)
    await bot.update_trailing_data()
    assert (SYMBOL, "long") in bot._trailing_pending_fill_confirmations
    assert not getattr(bot, "_trailing_reset_evidence", {})
    assert initial_receipts.matches(owner.ordinary_receipts())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "position_patch",
    [
        {"lastUpdateTimestamp": TIMESTAMP + 1},
        {"openTime": TIMESTAMP - 60_000},
    ],
)
async def test_a_conflicting_position_timestamp_is_not_probed(tmp_path, position_patch):
    bot, _, query = await pending_bot(tmp_path, **position_patch)
    await bot._refresh_trailing_reset_evidence(2)
    await bot.update_trailing_data()
    query.assert_not_awaited()
    assert (SYMBOL, "long") in bot._trailing_pending_fill_confirmations


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    ["manager", "price", "multi_source", "position_update", "position_round_trip"],
)
async def test_confirmed_proof_cannot_bypass_later_position_or_fill_changes(
    tmp_path, change
):
    bot, manager, _ = await pending_bot(tmp_path)
    await bot._refresh_trailing_reset_evidence(2)
    await bot.update_trailing_data()
    assert not bot._trailing_pending_fill_confirmations
    if change == "manager":
        bot._pnls_manager, _ = manager_for(tmp_path / "replacement")
    elif change == "price":
        manager._events[0] = replace(manager._events[0], price=100.01)
    elif change == "multi_source":
        manager._events[0] = replace(manager._events[0], source_ids=["42", "43"])
    else:
        bot._apply_positions_snapshot(
            [
                {
                    "symbol": SYMBOL,
                    "position_side": "long",
                    "size": 1.0,
                    "price": 100.0,
                    "lastUpdateTimestamp": TIMESTAMP + 1,
                }
            ]
        )
        if change == "position_round_trip":
            bot._apply_positions_snapshot(
                [
                    {
                        "symbol": SYMBOL,
                        "position_side": "long",
                        "size": 1.0,
                        "price": 100.0,
                        "lastUpdateTimestamp": TIMESTAMP,
                    }
                ]
            )
    await bot.update_trailing_data()
    assert (SYMBOL, "long") in bot._trailing_pending_fill_confirmations


@pytest.mark.asyncio
async def test_rejected_and_saturated_probe_are_scoped(tmp_path, caplog):
    bot, manager, query = await pending_bot(tmp_path)
    manager.fetcher.trade_limit = 1
    assert await manager.fetch_trailing_reset_fill(SYMBOL, "long", TIMESTAMP) is None
    query.side_effect = RequestTimeout("unavailable")
    await bot._refresh_trailing_reset_evidence(2)
    assert "action=keep_trailing_pending" in caplog.text
    assert not getattr(bot, "_trailing_reset_evidence", {})
    await bot._refresh_trailing_reset_evidence(3)
    assert (
        query.await_count == 2
    )  # Saturated probe plus one timed-out recovery attempt.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [AuthenticationError, ExchangeError, InvalidNonce, RateLimitExceeded]
)
async def test_probe_preserves_fill_owner_exchange_error_policy(tmp_path, error):
    bot, _, query = await pending_bot(tmp_path)
    query.side_effect = error("retry through owner")
    with pytest.raises(error):
        await bot._refresh_trailing_reset_evidence(2)


@pytest.mark.asyncio
async def test_portfolio_margin_does_not_use_the_usdm_proof_endpoint(tmp_path):
    manager, query = manager_for(tmp_path, portfolio_margin=True)
    assert await manager.fetch_trailing_reset_fill(SYMBOL, "long", TIMESTAMP) is None
    query.assert_not_awaited()


@pytest.mark.asyncio
async def test_hsl_history_owner_probes_under_fill_lock_without_widening(tmp_path):
    bot, manager, query = await pending_bot(tmp_path)
    bot.config["bot"]["portfolio"]["hsl"]["scope"] = "coin"
    bot.config["bot"]["long"]["hsl"]["enabled"] = True
    bot.init_pnls = AsyncMock()
    manager.set_history_scope("all")
    manager.refresh_latest = AsyncMock()
    manager.refresh = AsyncMock()

    async def assert_locked(_request):
        assert bot._pnls_refresh_lock.locked()
        assert bot._hsl_fill_observation is not None
        assert bot._trailing_fill_refresh_generation == 2
        return [trade()]

    query.side_effect = assert_locked
    await hsl_live.owner(bot)._refresh_history({})
    query.assert_awaited_once()
    manager.refresh.assert_not_awaited()
    manager.refresh_latest.assert_awaited_once()
    await bot.update_trailing_data()
    assert not bot._trailing_pending_fill_confirmations


@pytest.mark.asyncio
async def test_hsl_probe_timeout_keeps_published_fill_observation_valid(
    tmp_path, monkeypatch
):
    import passivbot

    bot, manager, query = await pending_bot(tmp_path)
    bot.config["bot"]["portfolio"]["hsl"]["scope"] = "coin"
    bot.config["bot"]["long"]["hsl"]["enabled"] = True
    monkeypatch.setattr(passivbot, "TRAILING_RESET_PROBE_TIMEOUT_SECONDS", 0.05)
    bot.init_pnls = AsyncMock()
    manager.set_history_scope("all")
    manager.refresh_latest = AsyncMock()
    observed = []

    async def slow_proof(_request):
        observed.append(bot._hsl_fill_observation)
        await asyncio.Future()

    query.side_effect = slow_proof
    ready = await asyncio.wait_for(
        hsl_live.owner(bot)._refresh_history({}), timeout=1.0
    )
    assert ready
    assert observed and observed[0] is bot._hsl_fill_observation is not None
    assert not bot._pnls_refresh_lock.locked()
    await bot.update_trailing_data()
    assert (SYMBOL, "long") in bot._trailing_pending_fill_confirmations


@pytest.mark.asyncio
async def test_probe_rotates_past_four_unresolved_coin_sides(tmp_path):
    bot, manager, _ = await pending_bot(tmp_path)
    symbols = [f"TEST{i}/USDT:USDT" for i in range(6)]
    manager._events = [replace(manager._events[0], symbol=symbol) for symbol in symbols]
    initial = bot.positions[SYMBOL]["long"]
    bot.positions = {symbol: {"long": initial.copy()} for symbol in symbols}
    bot._trailing_pending_fill_min_generations = {
        (symbol, "long"): 1 for symbol in symbols
    }
    bot._trailing_fill_confirmation_diagnostics = {
        (symbol, "long"): {"failed_predicates": ["fill_after_state_mismatch"]}
        for symbol in symbols
    }
    manager.fetch_trailing_reset_fill = AsyncMock(return_value=None)
    await bot._refresh_trailing_reset_evidence(2)
    assert manager.fetch_trailing_reset_fill.await_count == 4
    await bot._refresh_trailing_reset_evidence(3)
    assert [
        call.args[0] for call in manager.fetch_trailing_reset_fill.await_args_list
    ] == symbols


@pytest.mark.asyncio
async def test_probe_cannot_attach_to_position_that_changed_in_flight(tmp_path):
    bot, _, query = await pending_bot(tmp_path)

    async def changed_position(_request):
        bot._apply_positions_snapshot(
            [
                {
                    "symbol": SYMBOL,
                    "position_side": "long",
                    "size": 2.0,
                    "price": 100.0,
                    "lastUpdateTimestamp": TIMESTAMP + 1,
                }
            ]
        )
        return [trade()]

    query.side_effect = changed_position
    await bot._refresh_trailing_reset_evidence(2)
    query.assert_awaited_once()
    assert not getattr(bot, "_trailing_reset_evidence", {})
    assert (SYMBOL, "long") in bot._trailing_pending_fill_confirmations


def test_same_millisecond_reduction_cannot_be_discarded_by_explicit_open_replay(
    tmp_path,
):
    bot = _make_dummy_bot(_dummy_config())
    _set_basic_state(bot, SYMBOL)
    manager, _ = manager_for(tmp_path)
    opening = manager._events[0]
    reduction = replace(
        opening, id="41", source_ids=["41"], side="sell", qty=-0.5, psize=1.0
    )
    manager._events = [reduction, opening]
    bot._pnls_manager = manager
    bot.positions[SYMBOL]["long"].update(
        openTime=TIMESTAMP, lastUpdateTimestamp=TIMESTAMP
    )
    key = SYMBOL, "long"
    anchor = bot._latest_fill_position_change_anchors()[key]
    assert (
        bot._fill_anchor_position_state_match_kind(*key, (1.0, 100.0), anchor) is None
    )
