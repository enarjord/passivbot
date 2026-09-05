import pytest

from test_hsl_coin_mode import _make_aggregate_episode_bot


def _normal_override_history(signal_mode, *, closing_loss=100.0, same_minute=False):
    bot = _make_aggregate_episode_bot(signal_mode, closing_loss=closing_loss)
    bot._equity_hard_stop_cooldown_position_policy = lambda: "normal"
    events = bot._pnls_manager.get_events()
    panic_ts = 60_500 if same_minute else 30_500
    events[0]["timestamp"] = 60_600 if same_minute else 60_000
    events[:0] = [
        dict(timestamp=10_000, symbol="A", pside="long", action="increase", qty=1.0, pnl=0.0),
        dict(
            timestamp=panic_ts,
            symbol="A",
            pside="long",
            action="decrease",
            qty=1.0,
            pnl=-300.0,
            pb_order_type="close_panic_long",
        ),
    ]
    original_history = bot.get_balance_equity_history

    async def history(**kwargs):
        result = await original_history(**kwargs)
        result["panic_flatten_events"] = [
            dict(
                timestamp=panic_ts,
                minute_timestamp=panic_ts // 60_000 * 60_000,
                pside="long",
                symbol="A",
            )
        ]
        return result

    bot.get_balance_equity_history = history
    return bot


@pytest.mark.asyncio
@pytest.mark.parametrize("signal_mode", ["pside", "unified"])
@pytest.mark.parametrize("same_minute", [False, True])
@pytest.mark.parametrize("ema_span", [1.0, 5.0])
async def test_normal_cooldown_override_replays_later_ordinary_episode_reset(
    signal_mode, same_minute, ema_span
):
    bot = _normal_override_history(signal_mode, same_minute=same_minute)
    bot.hsl["long"]["ema_span_minutes"] = ema_span
    for _ in range(2):  # Restart reconstruction must reproduce the same reset.
        await bot._equity_hard_stop_initialize_from_history()
        state = bot._hsl_state("long")
        assert not state["halted"]
        assert state["last_stop_event"] is None
        assert state["pnl_reset_timestamp_ms"] == 180_501
        metrics = state["last_metrics"]
        assert metrics["peak_strategy_equity"] == pytest.approx(600.0)
        assert metrics["drawdown_raw"] == pytest.approx(51.0 / 600.0)
        assert metrics["tier"] != "red"
        expected_ema = 51.0 / 600.0 * (1.0 - (1.0 - 2.0 / (ema_span + 1.0)) ** 2)
        assert metrics["drawdown_ema"] == pytest.approx(expected_ema)


@pytest.mark.asyncio
@pytest.mark.parametrize("signal_mode", ["pside", "unified"])
@pytest.mark.parametrize("same_minute", [False, True])
async def test_normal_cooldown_override_still_finalizes_new_red_at_exact_fill(
    signal_mode, same_minute
):
    bot = _normal_override_history(signal_mode, closing_loss=300.0, same_minute=same_minute)
    await bot._equity_hard_stop_initialize_from_history()
    state = bot._hsl_state("long")
    assert state["halted"]
    assert state["last_stop_event"]["stop_event_timestamp_ms"] == 180_500
    assert state["cooldown_until_ms"] == 480_500
    assert state["last_stop_event"]["drawdown_raw"] == pytest.approx(300.0 / 700.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("signal_mode", ["pside", "unified"])
async def test_normal_override_keeps_each_same_minute_reset_and_reentry_fee(signal_mode):
    bot = _normal_override_history(signal_mode)
    events = bot._pnls_manager.get_events()
    events.extend(
        [
            dict(
                timestamp=180_700, symbol="A", pside="long", action="decrease", qty=1.0, pnl=-20.0
            ),
            dict(
                timestamp=180_800,
                symbol="A",
                pside="long",
                action="increase",
                qty=1.0,
                pnl=0.0,
                fee_paid=-2.0,
            ),
        ]
    )
    await bot._equity_hard_stop_initialize_from_history()
    state = bot._hsl_state("long")
    assert not state["halted"]
    assert state["pnl_reset_timestamp_ms"] == 180_701
    assert state["last_metrics"]["drawdown_raw"] == pytest.approx(52.0 / 579.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("signal_mode", ["pside", "unified"])
async def test_normal_override_without_later_flatten_keeps_its_new_episode(signal_mode):
    bot = _normal_override_history(signal_mode, same_minute=True)
    events = bot._pnls_manager.get_events()
    del events[3:]  # Retain the old panic and its in-minute intervention entry.
    await bot._equity_hard_stop_initialize_from_history()
    state = bot._hsl_state("long")
    assert not state["halted"]
    assert state["last_stop_event"] is None
    assert state["pnl_reset_timestamp_ms"] is None
    assert state["last_metrics"]["drawdown_raw"] == pytest.approx(50.0 / 700.0)


from live.freshness import FreshnessLedger
import passivbot_hsl as hsl
from test_hsl_coin_mode import make_coin_bot, make_fake_pnls_manager


def _manual_ownership_bot(signal_mode, events, *, refreshed_ms=1_000):
    bot = make_coin_bot(policy="manual")
    bot.config["live"]["hsl_signal_mode"] = signal_mode
    bot.get_exchange_time = lambda: 300_000
    bot._pnls_manager = make_fake_pnls_manager(events)
    original_metadata = bot._pnls_manager.cache.load_metadata
    bot._pnls_manager.cache.load_metadata = lambda: {
        **original_metadata(),
        "last_refresh_ms": refreshed_ms,
    }
    bot.freshness_ledger = FreshnessLedger()
    bot.freshness_ledger.begin_epoch()
    for surface in ("positions", "open_orders"):
        bot.freshness_ledger.stamp(surface, (), now_ms=1_000)
    symbol = "A" if signal_mode == "coin" else None
    state = bot._hsl_coin_state("long", symbol) if symbol else bot._hsl_state("long")
    state.update(
        halted=True,
        cooldown_until_ms=480_500,
        last_stop_event={"stop_event_timestamp_ms": 180_500},
        cooldown_intervention_active=False,
    )
    return bot, state, symbol


def _manual_stop_events(*, intervention=True, pside="long", symbol="A"):
    events = [dict(timestamp=180_500, symbol="A", pside="long", action="decrease", qty=1.0)]
    if intervention:
        events.extend(
            [
                dict(timestamp=180_600, symbol=symbol, pside=pside, action="increase", qty=1.0),
                dict(timestamp=210_000, symbol=symbol, pside=pside, action="decrease", qty=1.0),
            ]
        )
    return events


@pytest.mark.parametrize("signal_mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("refreshed_ms", [0, 999, 1_000])
def test_manual_entry_then_flat_ownership_survives_missing_recent_tail(signal_mode, refreshed_ms):
    bot, state, symbol = _manual_ownership_bot(
        signal_mode,
        _manual_stop_events(),
        refreshed_ms=refreshed_ms,
    )
    assert not bot.positions
    # This lifecycle field is deliberately false after flatten/restart.
    assert not state["cooldown_intervention_active"]
    assert hsl._equity_hard_stop_manual_cooldown_intervention(bot, "long", symbol) is True


@pytest.mark.parametrize("signal_mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize(
    "refreshed_ms, expected", [(0, None), (999, None), (1_000, False), (1_001, False)]
)
def test_manual_no_entry_requires_fill_tail_through_protective_observation(
    signal_mode, refreshed_ms, expected
):
    bot, state, symbol = _manual_ownership_bot(
        signal_mode,
        _manual_stop_events(intervention=False),
        refreshed_ms=refreshed_ms,
    )
    # A stale RAM flag cannot acquire manual ownership either.
    state["cooldown_intervention_active"] = True
    assert hsl._equity_hard_stop_manual_cooldown_intervention(bot, "long", symbol) is expected


@pytest.mark.parametrize("signal_mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize(
    "missing", ["stop", "stop_fill", "coverage", "snapshot", "quantity", "tied_entry"]
)
def test_manual_unknown_history_preserves_ownership(signal_mode, missing):
    events = _manual_stop_events(intervention=False)
    bot, state, symbol = _manual_ownership_bot(signal_mode, events)
    if missing == "stop":
        state["last_stop_event"] = None
    elif missing == "stop_fill":
        events.clear()
    elif missing == "coverage":
        bot._pnls_manager.cache.get_known_gaps = lambda: [{"start_ts": 180_500, "end_ts": 300_000}]
        bot._pnls_manager.get_coverage_status = lambda **kwargs: {"ready": False}
    elif missing == "snapshot":
        bot.freshness_ledger.begin_epoch()
    elif missing == "quantity":
        events.append(dict(timestamp=180_700, symbol="A", pside="long", action="increase"))
    else:
        events.append(dict(timestamp=180_500, symbol="A", pside="long", action="increase", qty=1.0))
    assert hsl._equity_hard_stop_manual_cooldown_intervention(bot, "long", symbol) is None


@pytest.mark.parametrize("signal_mode", ["coin", "pside", "unified"])
def test_prior_cooldown_entry_does_not_own_new_stop(signal_mode):
    events = _manual_stop_events()
    events.append(dict(timestamp=250_000, symbol="A", pside="long", action="decrease", qty=1.0))
    bot, state, symbol = _manual_ownership_bot(signal_mode, events)
    state["last_stop_event"] = {"stop_event_timestamp_ms": 250_000}
    state["cooldown_until_ms"] = 550_000
    assert hsl._equity_hard_stop_manual_cooldown_intervention(bot, "long", symbol) is False


@pytest.mark.parametrize(
    "signal_mode,entry_side,entry_symbol,expected",
    [
        ("coin", "long", "B", False),
        ("coin", "short", "A", False),
        ("pside", "long", "B", True),
        ("pside", "short", "A", False),
        ("unified", "short", "B", True),
    ],
)
def test_manual_ownership_matches_configured_scope(signal_mode, entry_side, entry_symbol, expected):
    bot, state, symbol = _manual_ownership_bot(
        signal_mode,
        _manual_stop_events(pside=entry_side, symbol=entry_symbol),
    )
    assert hsl._equity_hard_stop_manual_cooldown_intervention(bot, "long", symbol) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("signal_mode", ["pside", "unified"])
async def test_restart_reconstructs_manual_entry_then_flat_ownership_from_ordinary_stop(
    signal_mode,
):
    bot = _make_aggregate_episode_bot(signal_mode, closing_loss=300.0)
    bot._equity_hard_stop_cooldown_position_policy = lambda: "manual"
    events = bot._pnls_manager.get_events()
    events.append(
        dict(timestamp=210_000, symbol="A", pside="long", action="decrease", qty=1.0, pnl=0.0)
    )
    bot.positions["A"]["long"]["size"] = 0.0
    for _ in range(2):
        await bot._equity_hard_stop_initialize_from_history()
        state = bot._hsl_state("long")
        assert state["halted"]
        assert state["last_stop_event"]["stop_event_timestamp_ms"] == 180_500
        assert not state["cooldown_intervention_active"]
        assert hsl._equity_hard_stop_manual_cooldown_intervention(bot, "long") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("signal_mode", ["coin", "pside", "unified"])
async def test_live_manual_flatten_clears_transient_flag_but_preserves_fill_ownership(signal_mode):
    bot, state, symbol = _manual_ownership_bot(signal_mode, _manual_stop_events())
    bot.positions = {"A": {"long": {"size": 1.0}, "short": {"size": 0.0}}}
    if symbol:
        await hsl._equity_hard_stop_handle_coin_position_during_cooldown(
            bot, "long", symbol, 200_000
        )
    else:
        await hsl._equity_hard_stop_handle_position_during_cooldown(bot, "long", 200_000)
    assert state["cooldown_intervention_active"]
    bot.positions["A"]["long"]["size"] = 0.0
    if symbol:
        await hsl._equity_hard_stop_handle_coin_position_during_cooldown(
            bot, "long", symbol, 300_000
        )
    else:
        await hsl._equity_hard_stop_handle_position_during_cooldown(bot, "long", 300_000)
    assert not state["cooldown_intervention_active"]
    assert hsl._equity_hard_stop_manual_cooldown_intervention(bot, "long", symbol) is True


def test_manual_ownership_reads_canonical_fill_objects():
    from fill_events_manager import FillEvent

    events = [
        FillEvent(
            id=str(index),
            timestamp=row["timestamp"],
            datetime="",
            symbol=row["symbol"],
            side="buy" if row["action"] == "increase" else "sell",
            qty=row["qty"],
            price=100.0,
            pnl=0.0,
            fee_paid=0.0,
            pnl_status="confirmed",
            fees=[],
            pb_order_type="",
            position_side="long",
            client_order_id="",
        )
        for index, row in enumerate(_manual_stop_events())
    ]
    bot, state, symbol = _manual_ownership_bot("coin", events)
    assert hsl._equity_hard_stop_manual_cooldown_intervention(bot, "long", symbol) is True
