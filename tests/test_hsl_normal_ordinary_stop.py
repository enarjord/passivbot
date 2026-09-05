import pytest

from test_hsl_coin_mode import _make_aggregate_episode_bot


def _normal_after_red_bot(
    mode, *, panic_marker=False, close_again=False, terminal=False
):
    bot = _make_aggregate_episode_bot(mode, closing_loss=300.0)
    bot._equity_hard_stop_cooldown_position_policy = lambda: "normal"
    bot.config["live"]["hsl_position_during_cooldown_policy"] = "normal"
    bot.get_exchange_time = lambda: 300_900
    if terminal:
        bot.hsl["long"]["no_restart_drawdown_threshold"] = 0.2
    events = bot._pnls_manager.get_events()
    if panic_marker:
        events[1]["pb_order_type"] = "close_panic_long"
    if close_again:
        events.append(
            dict(
                timestamp=180_700,
                symbol="A",
                pside="long",
                action="decrease",
                qty=1.0,
                pnl=-300.0,
            )
        )
        bot.positions["A"]["long"]["size"] = 0.0

    async def upnl(pside=None, symbol=None):
        return 0.0

    bot._calc_upnl_sum_strict = upnl
    original = bot.get_balance_equity_history

    async def history(**kwargs):
        result = await original(**kwargs)
        for row in result["timeline"]:
            row["unrealized_pnl_long"] = 0.0
            row["realized_pnl_by_coin_pside"] = {
                "A": {"long": row["realized_pnl_long"], "short": 0.0}
            }
            row["unrealized_pnl_by_coin_pside"] = {"A": {"long": 0.0, "short": 0.0}}
        if panic_marker:
            result["panic_flatten_events"] = [
                dict(
                    timestamp=180_500,
                    minute_timestamp=180_000,
                    pside="long",
                    symbol="A",
                )
            ]
        return result

    bot.get_balance_equity_history = history
    bot.bot_value = lambda pside, key: 1.0
    return bot


async def _replay(bot, mode):
    if mode == "coin":
        await bot._equity_hard_stop_initialize_coin_from_history()
        return bot._hsl_coin_state("long", "A")
    await bot._equity_hard_stop_initialize_from_history()
    return bot._hsl_state("long")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("close_again", [False, True])
async def test_normal_override_after_red_is_independent_of_close_order_type(
    mode, close_again
):
    results = []
    for panic_marker in (False, True):
        bot = _normal_after_red_bot(
            mode, panic_marker=panic_marker, close_again=close_again
        )
        for _ in range(2):
            state = await _replay(bot, mode)
            if close_again:
                assert state["halted"]
                assert state["last_stop_event"]["stop_event_timestamp_ms"] == 180_700
                assert state["cooldown_until_ms"] == 480_700
            else:
                assert not state["halted"]
                assert state["last_metrics"]["tier"] == "green"
                assert 0 < state["last_metrics"]["drawdown_raw"] < 0.002
        results.append(state["last_metrics"]["drawdown_raw"])
    assert results[0] == pytest.approx(results[1])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("panic_marker", [False, True])
async def test_normal_entry_cannot_release_terminal_ordinary_red_stop(
    mode, panic_marker
):
    bot = _normal_after_red_bot(mode, terminal=True, panic_marker=panic_marker)
    state = await _replay(bot, mode)
    assert state["halted"] and state["no_restart_latched"]
    assert state["last_stop_event"]["stop_event_timestamp_ms"] == 180_500
