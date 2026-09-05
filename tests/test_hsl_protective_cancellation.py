from types import MethodType
from unittest.mock import AsyncMock

import pytest

import passivbot_hsl as hsl
from passivbot import Passivbot
from test_hsl_cooldown_boundary_replay import _manual_ownership_bot, _manual_stop_events


def _protective_bot(signal_mode, events):
    bot, state, symbol = _manual_ownership_bot(signal_mode, events, refreshed_ms=999)
    bot._equity_hard_stop_coin_initialized = True
    bot.config["live"]["execution_delay_seconds"] = 0.25
    bot._canonical_open_order_reduce_only = MethodType(
        Passivbot._canonical_open_order_reduce_only, bot
    )
    bot._equity_hard_stop_handle_position_during_cooldown = AsyncMock()
    bot._equity_hard_stop_handle_coin_position_during_cooldown = AsyncMock()
    bot.open_orders = {
        "A": [
            {
                "id": "initial",
                "symbol": "A",
                "position_side": "long",
                "reduce_only": False,
            },
            {
                "id": "close",
                "symbol": "A",
                "position_side": "long",
                "reduce_only": True,
            },
        ]
    }
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(
        return_value=([], [])
    )
    bot.execute_order_plan_to_exchange = AsyncMock()
    bot.update_pnls = AsyncMock(return_value=False)
    bot._sleep_unless_shutdown = AsyncMock()
    bot.refresh_protective_authoritative_state = AsyncMock(return_value=True)
    return bot, state, symbol


@pytest.mark.asyncio
@pytest.mark.parametrize("signal_mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize(
    "policy", ["panic", "manual", "tp_only", "graceful_stop", "normal"]
)
@pytest.mark.parametrize("deadline", [None, 200_000])
@pytest.mark.parametrize("held", [False, True])
async def test_terminal_scope_cancels_entries_without_cooldown_or_creations(
    signal_mode, policy, deadline, held
):
    bot, state, _ = _protective_bot(signal_mode, _manual_stop_events())
    bot.config["live"]["hsl_position_during_cooldown_policy"] = policy
    state.update(
        no_restart_latched=True,
        cooldown_until_ms=deadline,
        cooldown_intervention_active=True,
    )
    bot.positions = {"A": {"long": {"size": float(held)}}}
    assert await Passivbot._run_halted_hsl_protection_if_active(bot)
    cancels, creates = bot.execute_order_plan_to_exchange.await_args.args
    assert [order["id"] for order in cancels] == ["initial"]
    assert creates == []
    assert state["halted"] and state["no_restart_latched"]
    assert state["cooldown_until_ms"] == deadline
    assert bot.positions["A"]["long"]["size"] == float(held)
    bot.calc_protective_panic_orders_to_cancel_and_create.assert_not_awaited()
    bot._equity_hard_stop_handle_position_during_cooldown.assert_not_awaited()
    bot._equity_hard_stop_handle_coin_position_during_cooldown.assert_not_awaited()
    bot.update_pnls.assert_not_awaited()
    bot._sleep_unless_shutdown.assert_awaited_once_with(
        0.25, stage="hsl_cooldown_protection"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("signal_mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize(
    "outcome", ["no_entry", "entry", "failure", "invalidation", "confirmation"]
)
async def test_manual_proof_refreshes_fill_tail_after_account_observation(
    signal_mode, outcome
):
    events = _manual_stop_events(intervention=False)
    bot, state, symbol = _protective_bot(signal_mode, events)
    clock = {"account": 1_000, "fills": 999}
    calls = []
    metadata = bot._pnls_manager.cache.load_metadata()
    bot._pnls_manager.cache.load_metadata = lambda: {
        **metadata,
        "last_refresh_ms": clock["fills"],
    }

    async def account():
        calls.append("account")
        clock["account"] += 10
        bot.freshness_ledger.begin_epoch()
        for surface in ("positions", "open_orders"):
            bot.freshness_ledger.stamp(surface, (), now_ms=clock["account"])
        bot._authoritative_pending_confirmations = {}
        return True

    async def fills(*, source):
        calls.append("fills")
        assert source == "hsl_cooldown_protection"
        assert (
            hsl._equity_hard_stop_manual_cooldown_intervention(bot, "long", symbol)
            is None
        )
        if outcome == "failure":
            return False
        clock["fills"] = clock["account"] + 1
        if outcome == "entry":
            events.extend(_manual_stop_events()[1:])
        elif outcome == "invalidation":
            bot._account_invalidation_generation = 1
        elif outcome == "confirmation":
            bot._authoritative_pending_confirmations = {
                "positions": bot.freshness_ledger.epoch + 1
            }
        return True

    bot.refresh_protective_authoritative_state = AsyncMock(side_effect=account)
    bot.update_pnls = AsyncMock(side_effect=fills)
    did_work = await Passivbot._run_halted_hsl_protection_if_active(bot)
    assert did_work is (outcome == "no_entry")
    assert calls == (
        ["account", "fills", "account"]
        if outcome in {"invalidation", "confirmation"}
        else ["account", "fills"]
    )
    if outcome == "no_entry":
        cancels, creates = bot.execute_order_plan_to_exchange.await_args.args
        assert [order["id"] for order in cancels] == ["initial"]
        assert creates == []
        # An unchanged resting initial on the next deferred cycle must again
        # acquire a newer fill tail; it cannot remain perpetually behind.
        assert await Passivbot._run_halted_hsl_protection_if_active(bot)
        assert calls == ["account", "fills", "account", "fills"]
    else:
        bot.execute_order_plan_to_exchange.assert_not_awaited()
    bot.calc_protective_panic_orders_to_cancel_and_create.assert_not_awaited()
    assert state["halted"]


@pytest.mark.asyncio
@pytest.mark.parametrize("known_entry", [False, True])
async def test_manual_unknown_fill_failure_does_not_suppress_terminal_scope(
    known_entry,
):
    bot, _, _ = _protective_bot("coin", _manual_stop_events(intervention=known_entry))
    terminal = bot._hsl_coin_state("long", "B")
    terminal.update(halted=True, no_restart_latched=True)
    bot.open_orders["B"] = [
        {"id": "terminal", "symbol": "B", "position_side": "long", "reduce_only": False}
    ]
    assert await Passivbot._run_halted_hsl_protection_if_active(bot)
    cancels, creates = bot.execute_order_plan_to_exchange.await_args.args
    assert [order["id"] for order in cancels] == ["terminal"]
    assert not creates
    assert bot.update_pnls.await_count == (0 if known_entry else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("signal_mode", ["coin", "pside", "unified"])
async def test_manual_without_resting_entry_does_not_refresh_fills(signal_mode):
    bot, _, _ = _protective_bot(signal_mode, _manual_stop_events(intervention=False))
    bot.open_orders["A"] = [bot.open_orders["A"][1]]
    assert not await Passivbot._run_halted_hsl_protection_if_active(bot)
    bot.update_pnls.assert_not_awaited()
    bot.execute_order_plan_to_exchange.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("active_red", [False, True])
async def test_terminal_cancel_wave_includes_newly_replayed_current_red(active_red):
    bot, state, _ = _protective_bot("coin", _manual_stop_events())
    bot.config["live"]["hsl_position_during_cooldown_policy"] = "normal"
    bot.positions = {"A": {"long": {"size": 1.0}}}
    terminal = bot._hsl_coin_state("long", "B")
    terminal.update(halted=True, no_restart_latched=True)
    bot.open_orders["B"] = [
        {"id": "terminal", "symbol": "B", "position_side": "long", "reduce_only": False}
    ]
    protective_close = {"symbol": "A", "position_side": "long", "reduce_only": True}
    bot.calc_protective_panic_orders_to_cancel_and_create.return_value = (
        [],
        [protective_close],
    )

    async def replay(*args):
        state["halted"] = False
        for ts, equity in [(60_000, 100.0), (120_000, 70.0)]:
            state["runtime"].apply_sample(
                timestamp_ms=ts,
                equity=equity,
                peak_strategy_equity=100.0,
                red_threshold=0.2,
                ema_span_minutes=1.0,
                tier_ratio_yellow=0.5,
                tier_ratio_orange=0.75,
                latch_red=True,
            )
        state["last_metrics"] = {"red_active_now": active_red}
        return True

    bot._equity_hard_stop_handle_coin_position_during_cooldown.side_effect = replay
    assert await Passivbot._run_halted_hsl_protection_if_active(bot)
    cancels, creates = bot.execute_order_plan_to_exchange.await_args.args
    assert [order["id"] for order in cancels] == ["terminal"]
    assert creates == ([protective_close] if active_red else [])
    assert bot.calc_protective_panic_orders_to_cancel_and_create.await_count == int(
        active_red
    )
    bot.execute_order_plan_to_exchange.assert_awaited_once()
