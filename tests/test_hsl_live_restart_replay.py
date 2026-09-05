import asyncio
from types import MethodType

import pytest

import passivbot_hsl as hsl
from live.freshness import FreshnessLedger
from live.state_refresh import AuthoritativeSurfaceUnavailable
from test_hsl_coin_mode import _make_aggregate_episode_bot, make_coin_bot
from test_hsl_cooldown_boundary_replay import _normal_override_history


def _live_restart_bot(mode, kind, *, fee=0.0, upnl=300.0):
    if kind == "normal":
        bot = _normal_override_history("pside" if mode == "coin" else mode, same_minute=True)
        events = bot._pnls_manager.get_events()
        del events[3:]
        now, stop, deadline = 180_900, 60_500, 360_500
    else:
        bot = _make_aggregate_episode_bot("pside" if mode == "coin" else mode, closing_loss=300.0)
        bot.hsl["long"].update(cooldown_minutes_after_red=1.0, restart_after_red_policy="always")
        events = bot._pnls_manager.get_events()
        events[-1]["timestamp"] = 240_600
        now, stop, deadline = 300_900, 180_500, 240_500
    events[-1]["fee_paid"] = -fee
    bot.config["live"]["hsl_signal_mode"] = mode
    for name in (
        "_equity_hard_stop_handle_position_during_cooldown",
        "_equity_hard_stop_position_symbols",
        "_equity_hard_stop_log_cooldown_status",
    ):
        setattr(bot, name, MethodType(getattr(hsl, name), bot))
    bot.get_exchange_time = lambda: now

    async def current_upnl(pside=None, symbol=None):
        return -upnl if pside in (None, "long") else 0.0

    bot._calc_upnl_sum_strict = current_upnl
    original_history = bot.get_balance_equity_history

    async def history(**kwargs):
        result = await original_history(**kwargs)
        for row in result["timeline"]:
            row["unrealized_pnl_long"] = (
                -upnl if not row["is_flat_long"] and row["timestamp"] > stop else 0.0
            )
            row["realized_pnl_by_coin_pside"] = {
                "A": {"long": row["realized_pnl_long"], "short": 0.0}
            }
            row["unrealized_pnl_by_coin_pside"] = {
                "A": {"long": row["unrealized_pnl_long"], "short": 0.0}
            }
        return result

    bot.get_balance_equity_history = history
    if mode == "coin":
        bot.bot_value = lambda pside, key: 1.0
        bot._equity_hard_stop_coin_initialized = True
        bot._equity_hard_stop_apply_coin_metrics_sample(
            "long", "A", stop - 1, 1000.0, 0.0, 0.0, 0.0
        )
        state = bot._hsl_coin_state("long", "A")
    else:
        bot._equity_hard_stop_apply_sample(
            "long", stop - 1, 1000.0, 0.0, 0.0, 0.0, unrealized_pnl_total=0.0
        )
        state = bot._hsl_state("long")
    state.update(halted=True, cooldown_until_ms=deadline, pnl_reset_timestamp_ms=stop + 1)
    return bot, state


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("kind", ["normal", "expiry"])
@pytest.mark.parametrize("fee", [0.0, 2.0])
async def test_live_restart_preserves_loss_and_matches_canonical_replay(mode, kind, fee):
    bot, old = _live_restart_bot(mode, kind, fee=fee)
    await hsl._equity_hard_stop_check(bot)
    state = bot._hsl_coin_state("long", "A") if mode == "coin" else bot._hsl_state("long")
    assert state is old
    live = dict(state["last_metrics"])
    assert live["drawdown_raw"] > 0.4
    assert live["tier"] == "red"
    if mode == "coin":
        await bot._equity_hard_stop_initialize_coin_from_history()
        state = bot._hsl_coin_state("long", "A")
    else:
        await bot._equity_hard_stop_initialize_from_history()
        state = bot._hsl_state("long")
    for key in ("drawdown_raw", "drawdown_ema", "peak_strategy_equity"):
        assert live[key] == pytest.approx(state["last_metrics"][key])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize(
    "outcome",
    [
        "ready",
        "unavailable",
        "malformed",
        "epoch",
        "generation",
        "not_ready",
        "false_return",
        "pending_confirmation",
    ],
)
async def test_staged_restart_keeps_live_protection_until_ready_publish(mode, outcome):
    bot = make_coin_bot()
    bot.config["live"]["hsl_signal_mode"] = mode
    bot.freshness_ledger = FreshnessLedger()
    bot.freshness_ledger.begin_epoch()
    bot._account_invalidation_generation = 1
    bot._runtime_forced_modes = {"long": {"A": "panic", "B": "manual"}, "short": {"B": "panic"}}
    symbol = "A" if mode == "coin" else None
    state = bot._hsl_coin_state("long", symbol) if symbol else bot._hsl_state("long")
    state.update(
        halted=True, cooldown_until_ms=400_000,
        last_stop_event={"stop_event_timestamp_ms": 100_000},
    )
    old_stop = state["last_stop_event"]
    old_runtime = state["runtime"]
    old_maps = bot._runtime_forced_modes
    old_coin = bot._equity_hard_stop_coin
    other = bot._hsl_coin_state("long", "B") if symbol else bot._hsl_state("short")
    if mode != "unified":
        other["no_restart_latched"] = True
    writes = []
    events = []
    bot._emit_live_event = lambda *args, **kwargs: events.append(args)
    bot._equity_hard_stop_write_latch = lambda *args, **kwargs: writes.append(args)

    async def initializer(staged):
        assert staged is not bot
        staged._runtime_forced_modes["long"]["A"] = "graceful_stop"
        target = staged._hsl_coin_state("long", symbol) if symbol else staged._hsl_state("long")
        target.update(halted=True, cooldown_until_ms=500_000)
        staged._equity_hard_stop_write_latch(
            "long", {"stop_event_timestamp_ms": 200_000}, symbol=symbol
        )
        staged._emit_live_event("staged", {})
        await asyncio.sleep(0)
        assert state["halted"] and state["cooldown_until_ms"] == 400_000
        assert state["runtime"] is old_runtime
        assert bot._equity_hard_stop_coin is old_coin
        assert bot._runtime_forced_modes is old_maps
        assert bot._runtime_forced_modes["long"]["A"] == "panic"
        assert not writes and not events
        if outcome == "unavailable":
            raise AuthoritativeSurfaceUnavailable("hsl_episode_boundaries", "missing candles")
        if outcome == "malformed":
            raise ValueError("malformed canonical replay")
        if outcome == "epoch":
            bot.freshness_ledger.begin_epoch()
        if outcome == "generation":
            bot._account_invalidation_generation += 1
        if outcome == "pending_confirmation":
            bot._authoritative_pending_confirmations = {"positions": bot.freshness_ledger.epoch + 1}
        if outcome == "not_ready":
            target["halted"] = False
        staged._equity_hard_stop_coin_initialized = True
        staged._equity_hard_stop_coin_replay_ready_pairs = (
            {("long", "A")} if outcome != "not_ready" else set()
        )
        if outcome == "false_return":
            return False

    name = (
        "_equity_hard_stop_initialize_coin_from_history"
        if symbol
        else "_equity_hard_stop_initialize_from_history"
    )
    setattr(bot, name, MethodType(initializer, bot))
    if outcome == "malformed":
        with pytest.raises(ValueError, match="malformed canonical replay"):
            await hsl._equity_hard_stop_replay_live_restart(bot, "long", symbol)
    else:
        assert await hsl._equity_hard_stop_replay_live_restart(bot, "long", symbol) is (
            outcome == "ready"
        )
    assert not bot._hsl_live_restart_replay_active
    if outcome == "ready":
        assert state["cooldown_until_ms"] == 500_000
        assert state["last_stop_event"] is old_stop
        assert len(writes) == 1
        if mode == "coin":
            assert bot._runtime_forced_modes["long"]["B"] == "manual"
            assert ("long", "A") in bot._equity_hard_stop_coin_replay_ready_pairs
    else:
        assert state["runtime"] is old_runtime
        assert state["cooldown_until_ms"] == 400_000
        assert not writes
    if mode != "unified":
        assert other["no_restart_latched"]
    assert not events


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("kind", ["normal", "expiry"])
async def test_unavailable_real_restart_retains_existing_runtime_and_modes(mode, kind):
    bot, state = _live_restart_bot(mode, kind)
    before = state["runtime"]
    expected_mode = "graceful_stop" if kind == "normal" else "panic"
    bot._runtime_forced_modes["long"]["A"] = expected_mode

    async def unavailable(**kwargs):
        assert state["halted"]
        assert state["runtime"] is before
        await asyncio.sleep(0)
        raise AuthoritativeSurfaceUnavailable("hsl_episode_boundaries", "missing candle history")

    bot.get_balance_equity_history = unavailable
    await hsl._equity_hard_stop_check(bot)
    assert state["runtime"] is before
    assert state["halted"]
    assert bot._runtime_forced_modes["long"]["A"] == expected_mode


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
async def test_restart_defers_while_background_replay_owns_state(mode):
    bot, state = _live_restart_bot(mode, "normal")
    task = asyncio.create_task(asyncio.Event().wait())
    bot._equity_hard_stop_coin_replay_task = task
    before = state["runtime"]
    try:
        assert not await hsl._equity_hard_stop_replay_live_restart(
            bot, "long", "A" if mode == "coin" else None
        )
        assert state["runtime"] is before and state["halted"]
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
async def test_live_restart_does_not_resample_pre_await_observation(mode):
    bot, state = _live_restart_bot(mode, "normal")
    original_history = bot.get_balance_equity_history
    now = [180_900]
    bot.get_exchange_time = lambda: now[0]

    async def history(**kwargs):
        result = await original_history(**kwargs)
        now[0] = 240_900
        return result

    bot.get_balance_equity_history = history
    await hsl._equity_hard_stop_check(bot)
    assert state["last_metrics"]["timestamp_ms"] >= 180_900
    # Canonical replay owns its final sample; the outer check cannot write an
    # earlier observation after that sample or compound its EMA.
    assert state["last_metrics"]["drawdown_raw"] == pytest.approx(3.0 / 7.0)
