import pytest

from test_hsl_coin_mode import make_coin_bot, make_fake_pnls_manager


@pytest.mark.asyncio
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("missing_opening", [False, True])
async def test_expired_flat_scope_does_not_replay_another_sides_longer_horizon(
    compact, missing_opening, monkeypatch
):
    bot = make_coin_bot()
    bot.bot_value = lambda pside, key: 1.0
    for side, cooldown in [("long", 20.0), ("short", 5.0)]:
        bot.hsl[side].update(enabled=True, restart_after_red_policy="always",
                             cooldown_minutes_after_red=cooldown, red_threshold=0.1,
                             ema_span_minutes=10.0)
    now = 1_800_001
    bot.get_exchange_time = lambda: now
    bot.positions = {"A": {side: {"size": 0.0} for side in ("long", "short")}}
    events = [
        dict(timestamp=60_000, symbol="A", pside="short", action="increase", qty=2.0, pnl=0.0),
        dict(timestamp=1_200_000, symbol="A", pside="short", action="increase", qty=1.0, pnl=0.0),
        dict(timestamp=1_500_000, symbol="A", pside="short", action="decrease", qty=3.0, pnl=-30.0,
             pb_order_type="close_panic_short"),
    ]
    if missing_opening:
        events.pop(0)
    bot._pnls_manager = make_fake_pnls_manager(events)

    async def history(**kwargs):
        # The other side needs a longer coverage horizon, but these fills do
        # not belong to any currently required short episode.
        assert kwargs["hsl_replay_start_ms"] == now - 1_200_000
        retained = [e for e in events if e["timestamp"] >= kwargs["hsl_replay_start_ms"]]
        rows = [{"timestamp": ts, "balance": 100.0, "realized_pnl": -30.0 if ts >= 1_500_000 else 0.0,
                 "realized_pnl_by_coin_pside": {"A": {"short": -30.0 if ts >= 1_500_000 else 0.0}},
                 "unrealized_pnl_by_coin_pside": {"A": {"short": 0.0}}}
                for ts in range(660_000, 1_800_001, 60_000)]
        result = {"fill_events": retained, "panic_flatten_events": [
            dict(timestamp=1_500_000, minute_timestamp=1_500_000, pside="short", symbol="A")]}
        if compact:
            result["hsl_coin_compact_replay"] = {
                "timestamps": [r["timestamp"] for r in rows], "balances": [100.0] * len(rows),
                "realized_pnl": [r["realized_pnl"] for r in rows],
                "pair_values": {("short", "A"): {
                    "realized_pnl": [r["realized_pnl"] for r in rows],
                    "unrealized_pnl": [0.0] * len(rows)}}}
        else:
            result["timeline"] = rows
        return result

    bot.get_balance_equity_history = history
    await bot._equity_hard_stop_initialize_coin_from_history()
    for _ in range(3):
        await bot._equity_hard_stop_check_coin()
        state = bot._hsl_coin_state("short", "A")
        assert state["last_metrics"]["realized_pnl"] == 0.0
        assert not bot._equity_hard_stop_coin_red_active()
        assert state["pending_red_since_ms"] is None
        now += 60_000

    # New activity belongs to the next episode, including its entry fee.
    events.append(dict(timestamp=now-1, symbol="A", pside="short",
                       action="increase", qty=1.0, pnl=0.0, fee_paid=-0.25))
    bot.positions["A"]["short"]["size"] = -1.0
    await bot._equity_hard_stop_check_coin()
    assert bot._hsl_coin_state("short", "A")["last_metrics"]["realized_pnl"] == -0.25
    assert not bot._equity_hard_stop_coin_red_active()

    # Late activity inside the original cooldown horizon must still invalidate
    # the consumed observation, even after time has advanced beyond that horizon.
    import passivbot_hsl as hsl

    calls = []

    async def replay(self, pside, symbol, **kwargs):
        calls.append((pside, symbol))
        return True

    monkeypatch.setattr(hsl, "_equity_hard_stop_replay_live_restart", replay)
    events.extend([
        dict(timestamp=1_501_000, symbol="A", pside="short", action="increase", qty=1.0, pnl=0.0),
        dict(timestamp=1_502_000, symbol="A", pside="short", action="decrease", qty=1.0, pnl=-5.0),
    ])
    assert await hsl._equity_hard_stop_refresh_live_coin_episode_boundaries(bot, now, 100.0)
    assert calls == [("short", "A")]


@pytest.mark.parametrize("policy", ["always", "threshold", "never"])
@pytest.mark.parametrize("age_ms", [299_999, 300_000])
def test_flat_pair_within_its_own_horizon_keeps_unproven_history(policy, age_ms):
    import passivbot_hsl as hsl

    bot = make_coin_bot()
    bot.hsl["long"].update(restart_after_red_policy=policy, cooldown_minutes_after_red=5.0)
    now = 1_800_000
    bot.get_exchange_time = lambda: now
    bot.positions = {"A": {"long": {"size": 0.0}, "short": {"size": 0.0}}}
    bot._pnls_manager = make_fake_pnls_manager([
        dict(timestamp=now-age_ms, symbol="A", pside="long", action="decrease", qty=1.0, pnl=-30.0)
    ])
    assert hsl._equity_hard_stop_required_fill_history_scope(
        bot, now, pnl_start_ms=0
    ) == (True, 0, None)
