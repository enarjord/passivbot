import pytest

import passivbot_hsl as hsl
from test_hsl_coin_mode import make_coin_bot, make_fake_pnls_manager


def _delayed_boundary_bot(*, policy="always", cooldown=0.0):
    bot = make_coin_bot()
    bot.bot_value = lambda pside, key: 1.0
    bot.hsl["long"].update(
        restart_after_red_policy=policy,
        cooldown_minutes_after_red=cooldown,
        red_threshold=0.15,
    )
    bot._equity_hard_stop_coin_initialized = True
    bot._equity_hard_stop_apply_coin_metrics_sample(
        "long", "A", 240_000, 1000.0, 0.0, 0.0, 0.0
    )
    events = [
        dict(timestamp=60_000, symbol="A", pside="long", action="increase", qty=1.0, pnl=0.0),
        dict(timestamp=180_500, symbol="A", pside="long", action="decrease", qty=1.0, pnl=-300.0),
    ]
    bot._pnls_manager = make_fake_pnls_manager(events)
    bot.get_exchange_time = lambda: 300_000
    return bot, events


@pytest.mark.asyncio
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize(
    "entry_ts,reverse_cohort",
    [(180_500, False), (180_500, True), (180_600, False), (240_500, False)],
)
async def test_delayed_boundary_seeds_bounded_reentry_without_discarding_fees(
    entry_ts, reverse_cohort, compact
):
    bot, events = _delayed_boundary_bot()
    entry = dict(
        timestamp=entry_ts,
        symbol="A",
        pside="long",
        action="increase",
        qty=1.0,
        pnl=0.0,
        fee_paid=-2.0,
    )
    events.append(entry)
    if entry_ts == 180_500:
        for event, before in [(events[1], 1.0), (entry, 0.0)]:
            event["raw"] = [{"data": {
                "side": "sell" if event["action"] == "decrease" else "buy",
                "amount": 1.0,
                "price": 1.0,
                "info": {"startPosition": str(before)},
            }}]
        if reverse_cohort:
            events[1:] = reversed(events[1:])
    retained = [event for event in events if event["timestamp"] >= entry_ts]
    retained_pnl = sum(event["pnl"] + event.get("fee_paid", 0.0) for event in retained)
    bot.positions = {"A": {"long": {"size": 1.0}, "short": {"size": 0.0}}}
    bot.get_raw_balance = lambda: 698.0
    calls = []

    async def history(**kwargs):
        calls.append(kwargs)
        assert kwargs["hsl_replay_start_ms"] == entry_ts
        if compact:
            return {
                "hsl_coin_compact_replay": {
                    "timestamps": [entry_ts // 60_000 * 60_000],
                    "balances": [698.0],
                    "realized_pnl": [retained_pnl],
                    "pair_values": {
                        ("long", "A"): {
                            "realized_pnl": [retained_pnl],
                            "unrealized_pnl": [-50.0],
                        }
                    },
                },
                "fill_events": retained,
                "panic_flatten_events": [],
            }
        return {
            "timeline": [
                {
                    "timestamp": entry_ts // 60_000 * 60_000,
                    "balance": 698.0,
                    "realized_pnl": retained_pnl,
                    "realized_pnl_by_coin_pside": {"A": {"long": retained_pnl}},
                    "unrealized_pnl_by_coin_pside": {"A": {"long": -50.0}},
                }
            ],
            "fill_events": retained,
            "panic_flatten_events": [],
        }

    async def upnl(*_args):
        return -50.0

    bot.get_balance_equity_history = history
    bot._calc_upnl_sum_strict = upnl
    assert await hsl._equity_hard_stop_refresh_live_coin_episode_boundaries(bot, 300_000, 698.0)
    state = bot._hsl_coin_state("long", "A")
    assert state["last_metrics"]["realized_pnl"] == -2.0
    assert state["last_metrics"]["drawdown_raw"] == pytest.approx(52.0 / 698.0)
    assert state["last_metrics"]["drawdown_ema"] == pytest.approx(52.0 / 698.0)
    assert not state["last_metrics"]["red_seen_in_episode"]
    assert not state["runtime"].red_latched()
    assert "A" not in bot._runtime_forced_modes["long"]
    assert not await hsl._equity_hard_stop_refresh_live_coin_episode_boundaries(
        bot, 360_000, 698.0
    )
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["always", "threshold", "never"])
async def test_delayed_boundary_retains_required_red_stop_history(policy):
    bot, events = _delayed_boundary_bot(policy=policy, cooldown=5.0)
    bot.positions = {"A": {"long": {"size": 0.0}, "short": {"size": 0.0}}}
    bot.get_raw_balance = lambda: 700.0

    async def history(**kwargs):
        # The recent flat scope still needs its full RED episode, even for always.
        assert kwargs["hsl_replay_start_ms"] < 60_000
        return {
            "timeline": [
                {
                    "timestamp": ts,
                    "balance": 1000.0 + pnl,
                    "realized_pnl": pnl,
                    "realized_pnl_by_coin_pside": {"A": {"long": pnl}},
                    "unrealized_pnl_by_coin_pside": {"A": {"long": 0.0}},
                }
                for ts, pnl in [(60_000, 0.0), (120_000, 0.0), (180_000, -300.0)]
            ],
            "fill_events": events,
            "panic_flatten_events": [],
        }

    bot.get_balance_equity_history = history
    assert await hsl._equity_hard_stop_refresh_live_coin_episode_boundaries(bot, 300_000, 700.0)
    state = bot._hsl_coin_state("long", "A")
    assert state["halted"]
    assert state["last_stop_event"]["stop_event_timestamp_ms"] == 180_500
    assert state["no_restart_latched"] is (policy == "never")
    if policy != "never":
        assert state["cooldown_until_ms"] == 480_500


@pytest.mark.asyncio
async def test_new_red_boundary_remains_actionable_when_staged_replay_defers(monkeypatch):
    bot = make_coin_bot()
    bot.bot_value = lambda pside, key: 1.0
    bot.hsl["long"]["red_threshold"] = 0.15
    bot._equity_hard_stop_coin_initialized = True
    bot._equity_hard_stop_apply_coin_metrics_sample(
        "long", "A", 60_000, 1_000.0, 0.0, 0.0, 0.0
    )
    bot._pnls_manager = make_fake_pnls_manager(
        [
            dict(
                timestamp=60_000,
                symbol="A",
                pside="long",
                action="increase",
                qty=1.0,
                pnl=0.0,
            ),
            dict(
                timestamp=180_500,
                symbol="A",
                pside="long",
                action="decrease",
                qty=1.0,
                pnl=-300.0,
            ),
        ]
    )
    bot.positions = {"A": {"long": {"size": 0.0}, "short": {"size": 0.0}}}

    async def unavailable(*_args, **_kwargs):
        return False

    monkeypatch.setattr(hsl, "_equity_hard_stop_replay_live_restart", unavailable)

    with pytest.raises(
        hsl.AuthoritativeSurfaceUnavailable, match="canonical replay unavailable"
    ):
        await hsl._equity_hard_stop_refresh_live_coin_episode_boundaries(
            bot, 300_000, 700.0
        )

    state = bot._hsl_coin_state("long", "A")
    assert state["runtime"].red_latched()
    assert state["pending_red_since_ms"] == 180_500
    assert state["pending_stop_event"] is None
    assert bot._runtime_forced_modes["long"]["A"] == "panic"
