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


def _tie_normal_stop_entry(bot, reverse=False, evidence=True):
    events = bot._pnls_manager.get_events()
    for event, before in zip(events[1:3], [1.0, 0.0]):
        event["timestamp"] = 180_500
        if evidence:
            event["raw"] = [
                {
                    "data": {
                        "side": "sell" if event["action"] == "decrease" else "buy",
                        "amount": event["qty"],
                        "price": 1.0,
                        "info": {"startPosition": str(before)},
                    }
                }
            ]
    if reverse:
        events[1:3] = reversed(events[1:3])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("panic_marker", [False, True])
async def test_normal_tied_stop_entry_preserves_fee_after_restart_and_current_poll(
    mode, reverse, panic_marker
):
    bot = _normal_after_red_bot(mode, panic_marker=panic_marker)
    _tie_normal_stop_entry(bot, reverse)
    for _ in range(2):
        state = await _replay(bot, mode)
        assert not state["halted"]
        assert state["last_metrics"]["tier"] == "green"
        raw = state["last_metrics"]["drawdown_raw"]
        assert 0 < raw < 0.002
        if mode == "coin":
            current = bot._equity_hard_stop_apply_coin_sample(
                "long", "A", 360_900, 699.0, 0.0
            )
            assert current["realized_pnl"] == pytest.approx(-1.0)
        else:
            current = bot._equity_hard_stop_apply_sample(
                "long", 360_900, 699.0, -301.0, -301.0, 0.0, unrealized_pnl_total=0.0
            )
        assert current["drawdown_raw"] == pytest.approx(raw)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("reverse", [False, True])
async def test_normal_tied_stop_entry_cannot_release_terminal_scope(mode, reverse):
    bot = _normal_after_red_bot(mode, terminal=True)
    _tie_normal_stop_entry(bot, reverse)
    state = await _replay(bot, mode)
    assert state["halted"] and state["no_restart_latched"]
    assert state["last_stop_event"]["stop_event_timestamp_ms"] == 180_500


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
async def test_normal_tied_stop_entry_requires_exchange_ordering_evidence(mode):
    from live.state_refresh import AuthoritativeSurfaceUnavailable

    bot = _normal_after_red_bot(mode)
    _tie_normal_stop_entry(bot, evidence=False)
    if mode == "coin":
        state = await _replay(bot, mode)
        assert state["runtime"].red_latched()
        assert state["pnl_reset_timestamp_ms"] is None
        assert bot._runtime_forced_modes["long"]["A"] == "panic"
    else:
        with pytest.raises(AuthoritativeSurfaceUnavailable):
            await _replay(bot, mode)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("reverse", [False, True])
async def test_normal_branching_tied_stops_do_not_guess_intervention_prefix(
    mode, reverse
):
    bot = _normal_after_red_bot(mode)
    events = bot._pnls_manager.get_events()
    events[2]["qty"] = 2.0
    events.extend(
        [
            dict(
                timestamp=180_500,
                symbol="A",
                pside="long",
                action="decrease",
                qty=2.0,
                pnl=-150.0,
            ),
            dict(
                timestamp=180_500,
                symbol="A",
                pside="long",
                action="increase",
                qty=3.0,
                pnl=0.0,
                fee_paid=-2.0,
            ),
        ]
    )
    for event, before in zip(events[1:], [1.0, 0.0, 2.0, 0.0]):
        event["timestamp"] = 180_500
        event["raw"] = [
            {
                "data": {
                    "side": "sell" if event["action"] == "decrease" else "buy",
                    "amount": event["qty"],
                    "price": 1.0,
                    "info": {"startPosition": str(before)},
                }
            }
        ]
    if reverse:
        events[1:] = reversed(events[1:])
    bot.positions["A"]["long"]["size"] = 3.0
    from live.state_refresh import AuthoritativeSurfaceUnavailable

    # Repeated exits from zero do not have a unique successor in the existing
    # exchange-chain proof. Never pick an entry or PnL prefix from list order.
    for _ in range(2):
        if mode == "coin":
            state = await _replay(bot, mode)
            assert state["runtime"].red_latched()
            assert state["pnl_reset_timestamp_ms"] is None
            assert bot._runtime_forced_modes["long"]["A"] == "panic"
        else:
            with pytest.raises(AuthoritativeSurfaceUnavailable):
                await _replay(bot, mode)


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("residue", [0.001, 0.01])
@pytest.mark.parametrize("pside", ["long", "short"])
async def test_coin_tied_intervention_uses_symbol_flat_tolerance(
    reverse, residue, pside
):
    from test_hsl_tied_intervention_evidence import _fill

    bot = _normal_after_red_bot("coin")
    bot.qty_steps = {"A": 0.01}
    events = bot._pnls_manager.get_events()
    events[:] = [
        _fill("open", 60_000, 0.0, 1.0, pside=pside),
        _fill("stop", 180_500, 1.0, residue, pnl=-300.0, pside=pside),
        _fill("entry", 180_500, residue, residue + 1.0, fee=-1.0, pside=pside),
    ]
    if reverse:
        events[1:] = reversed(events[1:])
    if pside == "short":
        bot.hsl["short"] = dict(bot.hsl["long"])
        bot.hsl["long"]["enabled"] = False
    bot.positions = {
        "A": {
            side: {
                "size": (
                    (1.0 + residue) * (1 if side == "long" else -1)
                    if side == pside
                    else 0.0
                )
            }
            for side in ("long", "short")
        }
    }

    async def history(**kwargs):
        rows = []
        for ts in (60_000, 120_000, 180_000, 240_000):
            realized = sum(
                e["pnl"] + e["fee_paid"] for e in events if e["timestamp"] < ts + 60_000
            )
            rows.append(
                dict(
                    timestamp=ts,
                    balance=1000.0 + realized,
                    realized_pnl=realized,
                    realized_pnl_long=realized if pside == "long" else 0.0,
                    realized_pnl_short=realized if pside == "short" else 0.0,
                    unrealized_pnl_long=0.0,
                    unrealized_pnl_short=0.0,
                    realized_pnl_by_coin_pside={
                        "A": {
                            side: realized if side == pside else 0.0
                            for side in ("long", "short")
                        }
                    },
                    unrealized_pnl_by_coin_pside={"A": {"long": 0.0, "short": 0.0}},
                    is_flat=False,
                    is_flat_long=pside != "long",
                    is_flat_short=pside != "short",
                )
            )
        return dict(timeline=rows, fill_events=events, panic_flatten_events=[])

    bot.get_balance_equity_history = history
    stop = next(event for event in events if event["id"] == "stop")
    stop["pb_order_type"] = f"close_panic_{pside}"
    inferred = bot._equity_hard_stop_infer_coin_replay_contract(
        pside, "A", events, 300_900
    )
    assert inferred["intervention_entry_ts"] == (180_500 if residue < 0.005 else None)
    stop.pop("pb_order_type")
    for _ in range(2):
        await bot._equity_hard_stop_initialize_coin_from_history()
        state = bot._hsl_coin_state(pside, "A")
        if residue < 0.005:
            assert not state["halted"]
            assert state["last_metrics"]["tier"] == "green"
            assert state["last_metrics"]["realized_pnl"] == pytest.approx(-1.0)
            current = bot._equity_hard_stop_apply_coin_sample(
                pside, "A", 360_900, 699.0, 0.0
            )
            assert current["realized_pnl"] == pytest.approx(-1.0)
            assert current["drawdown_raw"] == pytest.approx(1.0 / 699.0)
        else:
            assert state["runtime"].red_latched()
            assert state["pnl_reset_timestamp_ms"] is None
            assert bot._runtime_forced_modes[pside]["A"] == "panic"
