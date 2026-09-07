import pytest

from test_hsl_cooldown_boundary_replay import _normal_override_history


def _coin_normal_override_bot(*, upnl=0.0, fee=0.0, close_loss=None):
    bot = _normal_override_history("pside", same_minute=True)
    bot.config["live"]["hsl_signal_mode"] = "coin"
    events = bot._pnls_manager.get_events()
    del events[3:]
    events[-1]["fee_paid"] = -fee
    if close_loss is not None:
        events.append(
            dict(
                timestamp=60_800,
                symbol="A",
                pside="long",
                action="decrease",
                qty=1.0,
                pnl=-close_loss,
            )
        )
        bot.positions["A"]["long"]["size"] = 0.0
    bot.get_exchange_time = lambda: 180_900
    bot.bot_value = lambda pside, key: 1.0

    async def current_upnl(pside=None, symbol=None):
        return -upnl

    bot._calc_upnl_sum_strict = current_upnl
    original_history = bot.get_balance_equity_history

    async def history(**kwargs):
        result = await original_history(**kwargs)
        for row in result["timeline"]:
            row["unrealized_pnl_long"] = -upnl if not row["is_flat_long"] else 0.0
            row["realized_pnl_by_coin_pside"] = {
                "A": {"long": row["realized_pnl_long"], "short": 0.0}
            }
            row["unrealized_pnl_by_coin_pside"] = {
                "A": {"long": row["unrealized_pnl_long"], "short": 0.0}
            }
        return result

    bot.get_balance_equity_history = history
    return bot


@pytest.mark.asyncio
@pytest.mark.parametrize("upnl", [0.0, 50.0])
@pytest.mark.parametrize("fee", [0.0, 2.0])
@pytest.mark.parametrize("span", [1.0, 5.0])
async def test_coin_normal_override_discards_prior_stop_loss_but_keeps_new_loss(
    upnl, fee, span
):
    bot = _coin_normal_override_bot(upnl=upnl, fee=fee)
    bot.hsl["long"].update(ema_span_minutes=span, red_threshold=0.1)
    for _ in range(2):
        await bot._equity_hard_stop_initialize_coin_from_history()
        state = bot._hsl_coin_state("long", "A")
        assert state["pnl_reset_timestamp_ms"] == 60_600
        assert not state["halted"]
        metrics = state["last_metrics"]
        assert metrics["realized_pnl"] == pytest.approx(-fee)
        assert metrics["drawdown_raw"] == pytest.approx((fee + upnl) / (700.0 - fee))
        # Current live samples must retain the same reset after reconstruction.
        current = bot._equity_hard_stop_apply_coin_sample(
            "long", "A", 240_900, 700.0 - fee, -upnl
        )
        assert current["realized_pnl"] == pytest.approx(-fee)
        assert current["drawdown_raw"] == pytest.approx(metrics["drawdown_raw"])


@pytest.mark.asyncio
async def test_coin_normal_override_retains_later_flatten_in_intervention_minute():
    bot = _coin_normal_override_bot(fee=2.0, close_loss=300.0)
    await bot._equity_hard_stop_initialize_coin_from_history()
    state = bot._hsl_coin_state("long", "A")
    assert state["halted"]
    assert state["last_stop_event"]["stop_event_timestamp_ms"] == 60_800
    assert state["cooldown_until_ms"] == 360_800


@pytest.mark.asyncio
@pytest.mark.parametrize("restart_policy", ["always", "threshold"])
@pytest.mark.parametrize("upnl", [0.0, 50.0])
@pytest.mark.parametrize("fee", [0.0, 2.0])
async def test_coin_expired_episode_replay_and_current_sample_share_baseline(
    restart_policy, upnl, fee
):
    from test_hsl_coin_mode import _make_aggregate_episode_bot

    bot = _make_aggregate_episode_bot("coin", closing_loss=300.0)
    bot.hsl["long"].update(
        cooldown_minutes_after_red=1.0, restart_after_red_policy=restart_policy
    )
    events = bot._pnls_manager.get_events()
    events[-1].update(timestamp=240_600, fee_paid=-fee)
    bot.get_exchange_time = lambda: 300_900
    bot.bot_value = lambda pside, key: 1.0

    async def current_upnl(pside=None, symbol=None):
        return -upnl

    bot._calc_upnl_sum_strict = current_upnl
    original_history = bot.get_balance_equity_history

    async def history(**kwargs):
        result = await original_history(**kwargs)
        for row in result["timeline"]:
            row["unrealized_pnl_long"] = -upnl if row["timestamp"] >= 240_000 else 0.0
            row["realized_pnl_by_coin_pside"] = {
                "A": {"long": row["realized_pnl_long"], "short": 0.0}
            }
            row["unrealized_pnl_by_coin_pside"] = {
                "A": {"long": row["unrealized_pnl_long"], "short": 0.0}
            }
        return result

    bot.get_balance_equity_history = history
    for _ in range(2):
        await bot._equity_hard_stop_initialize_coin_from_history()
        state = bot._hsl_coin_state("long", "A")
        assert not state["halted"]
        assert state["pnl_reset_timestamp_ms"] >= 180_501
        metrics = state["last_metrics"]
        assert metrics["realized_pnl"] == pytest.approx(-fee)
        assert metrics["drawdown_raw"] == pytest.approx(
            (fee + upnl) / metrics["slot_budget"]
        )
        current = bot._equity_hard_stop_apply_coin_sample(
            "long", "A", 360_900, 700.0 - fee, -upnl
        )
        assert current["realized_pnl"] == pytest.approx(-fee)
        assert current["drawdown_raw"] == pytest.approx(
            (fee + upnl) / current["slot_budget"]
        )
