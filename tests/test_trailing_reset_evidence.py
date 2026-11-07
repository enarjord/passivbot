"""Offline proofs of trailing reset boundaries with incomplete historical replay."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from ccxt.base.errors import RequestTimeout

from fill_events_manager import (
    BaseFetcher,
    BinanceFetcher,
    FillEvent,
    FillEventsManager,
    ensure_qty_signage,
)
from test_unstucking_safeguards import (
    mock_pbr,
    _dummy_config,
    _make_dummy_bot,
    _set_basic_state,
    _make_candles,
    _supply_fresh_reset_evidence,
)


def _trade(pside="long", **patch):
    return {
        "id": "opening-fill",
        "timestamp": 180_000,
        "symbol": "TEST/USDT",
        "side": "buy" if pside == "long" else "sell",
        "amount": 1.0,
        "price": 100.0,
        "info": {"positionSide": pside.upper()},
        **patch,
    }


def _manager(tmp_path, trades):
    api = type("API", (), {})()
    api.fetch_my_trades = AsyncMock(return_value=trades)
    fetcher = BinanceFetcher(api, symbol_resolver=lambda value: value)
    manager = FillEventsManager(
        exchange="binance", user="offline-test", fetcher=fetcher, cache_path=tmp_path
    )
    return manager, api


async def _pending_bot(tmp_path, pside="long", **position_patch):
    bot = _make_dummy_bot(_dummy_config())
    symbol = _set_basic_state(bot)
    manager, api = _manager(tmp_path, [_trade(pside)])
    row = manager.fetcher._normalize_trade(_trade(pside))
    ensure_qty_signage([row])
    # A previous close is absent; these derived annotations retain old residue.
    row.update(psize=2.0, pprice=100.0)
    manager._events = [FillEvent.from_dict(row)]
    bot._pnls_manager = manager
    bot.is_trailing = lambda sym, side=None: side == pside
    bot.get_exchange_time = lambda: 361_000
    bot._apply_positions_snapshot(
        [
            {
                "symbol": symbol,
                "position_side": pside,
                "size": 1.0 if pside == "long" else -1.0,
                "price": 100.0,
                "updateTime": 180_000,
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
    api.fetch_my_trades.assert_not_awaited()  # Mandatory snapshot refresh comes first.
    bot._trailing_fill_fetch_generation = 1
    await bot.update_trailing_data()
    assert bot._trailing_pending_fill_confirmations
    return bot, symbol, manager, api


@pytest.mark.asyncio
@pytest.mark.parametrize("pside", ["long", "short"])
async def test_fresh_bounded_query_recovers_only_trailing_and_preserves_history(
    tmp_path, pside
):
    bot, symbol, manager, api = await _pending_bot(tmp_path, pside)
    before_events = deepcopy(manager.get_events())
    before_metadata = list(tmp_path.iterdir())
    await bot._refresh_trailing_reset_evidence(2)
    # Proving this reset must avoid widening into irretrievable older history.
    assert bot._trailing_fill_history_recovery_start_ms(None) is None
    await bot.update_trailing_data()
    assert bot._trailing_pending_fill_confirmations == {}
    assert bot._orchestrator_trailing_unavailable_symbols == set()
    assert bot.trailing_prices[symbol][pside]["max_since_open"] == 103.0
    api.fetch_my_trades.assert_awaited_once_with(
        symbol, limit=1000, params={"startTime": 180_000, "endTime": 180_000}
    )
    assert manager.get_events() == before_events
    assert list(tmp_path.iterdir()) == before_metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("open_time", [None, 360_000])
async def test_later_update_never_confirms_old_fill_even_with_matching_size_price(
    tmp_path, open_time
):
    patch = {"updateTime": 360_000}
    if open_time is not None:
        patch["openTime"] = open_time
    bot, symbol, manager, api = await _pending_bot(tmp_path, **patch)
    _supply_fresh_reset_evidence(bot, symbol, "long")
    await bot.update_trailing_data()
    assert bot._trailing_pending_fill_confirmations
    api.fetch_my_trades.assert_not_awaited()


@pytest.mark.asyncio
async def test_conflicting_open_time_does_not_trigger_update_fallback(tmp_path):
    bot, symbol, manager, api = await _pending_bot(tmp_path, openTime=120_000)
    await bot._refresh_trailing_reset_evidence(2)
    await bot.update_trailing_data()
    assert bot._trailing_pending_fill_confirmations
    api.fetch_my_trades.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["old_generation", "new_manager", "coalesced_cache", "corrected_fill"]
)
async def test_receipt_cannot_bypass_snapshot_identity_or_cached_cohort(
    tmp_path, failure
):
    bot, symbol, manager, api = await _pending_bot(tmp_path)
    await bot._refresh_trailing_reset_evidence(0 if failure == "old_generation" else 2)
    if failure == "new_manager":
        replacement, _ = _manager(tmp_path / "replacement", [_trade()])
        replacement._events = list(manager._events)
        bot._pnls_manager = replacement
    elif failure == "coalesced_cache":
        manager._events[0] = replace(
            manager._events[0], source_ids=["opening-fill", "other-fill"]
        )
    elif failure == "corrected_fill":
        manager._events[0] = replace(manager._events[0], price=100.01)
    await bot.update_trailing_data()
    assert bot._trailing_pending_fill_confirmations


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        [],
        [_trade(), _trade(id="second-fill")],
        [_trade(timestamp=180_001)],
        [_trade(symbol="OTHER/USDT")],
        [_trade(id=None)],
        [_trade(side="sell")],
        [_trade(price=float("nan"))],
    ],
)
async def test_incomplete_or_contradictory_fresh_evidence_keeps_trailing_pending(
    tmp_path, response
):
    bot, symbol, manager, api = await _pending_bot(tmp_path)
    api.fetch_my_trades.return_value = response
    await bot._refresh_trailing_reset_evidence(2)
    await bot.update_trailing_data()
    assert bot._trailing_pending_fill_confirmations


@pytest.mark.asyncio
async def test_saturated_query_is_not_coverage_proof(tmp_path):
    manager, api = _manager(tmp_path, [_trade()])
    manager.fetcher.trade_limit = 1
    assert (
        await manager.fetch_trailing_reset_evidence("TEST/USDT", "long", 180_000)
        is None
    )


@pytest.mark.asyncio
async def test_unsupported_connector_does_not_infer_coverage(tmp_path):
    manager, api = _manager(tmp_path, [_trade()])
    manager.fetcher = BaseFetcher()
    assert (
        await manager.fetch_trailing_reset_evidence("TEST/USDT", "long", 180_000)
        is None
    )
    api.fetch_my_trades.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_failure_remains_scoped_to_optional_trailing_proof(
    tmp_path, caplog
):
    bot, symbol, manager, api = await _pending_bot(tmp_path)
    api.fetch_my_trades.side_effect = RequestTimeout("unavailable")
    await bot._refresh_trailing_reset_evidence(2)
    assert bot._trailing_reset_evidence == {}
    assert "action=keep_trailing_pending" in caplog.text
    await bot.update_trailing_data()
    assert bot._trailing_pending_fill_confirmations


@pytest.mark.asyncio
async def test_query_pass_rotates_past_unresolved_cohorts(tmp_path):
    bot, symbol, manager, api = await _pending_bot(tmp_path)
    event = manager._events[0]
    symbols = [f"TEST{i}/USDT" for i in range(6)]
    manager._events = [replace(event, symbol=sym) for sym in symbols]
    bot.positions = {
        sym: {"long": {"size": 1.0, "price": 100.0, "updateTime": 180_000}}
        for sym in symbols
    }
    bot._trailing_fill_confirmation_diagnostics = {
        (sym, "long"): {"failed_predicates": ["fill_after_state_mismatch"]}
        for sym in symbols
    }
    manager.fetch_trailing_reset_evidence = AsyncMock(return_value=None)
    await bot._refresh_trailing_reset_evidence(2)
    assert manager.fetch_trailing_reset_evidence.await_count == 4
    await bot._refresh_trailing_reset_evidence(3)
    calls = manager.fetch_trailing_reset_evidence.await_args_list
    assert [call.args[0] for call in calls[:6]] == symbols


def test_known_same_millisecond_reduction_cannot_be_discarded_by_open_replay(tmp_path):
    bot = _make_dummy_bot(_dummy_config())
    symbol = _set_basic_state(bot)
    manager, _ = _manager(tmp_path, [])
    opening = manager.fetcher._normalize_trade(_trade())
    opening.update(psize=2.0, pprice=100.0)
    reduction = {
        **opening,
        "id": "reduction",
        "side": "sell",
        "qty": -0.5,
        "psize": 1.0,
        "raw": [],
        "source_ids": ["reduction"],
    }
    manager._events = [FillEvent.from_dict(reduction), FillEvent.from_dict(opening)]
    bot._pnls_manager = manager
    bot.positions[symbol]["long"].update(openTime=180_000, updateTime=180_000)
    anchor = bot._latest_fill_position_change_anchors()[(symbol, "long")]
    assert (
        bot._fill_anchor_position_state_match_kind(symbol, "long", (1.0, 100.0), anchor)
        is None
    )
