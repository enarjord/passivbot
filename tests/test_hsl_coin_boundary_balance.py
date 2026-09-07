"""Coin risk samples use the account balance at the exact closing fill."""

from types import MethodType

import pytest

from passivbot import Passivbot
from test_hsl_coin_mode import make_coin_bot, make_fake_pnls_manager


def _two_coin_boundary_bot(
    *, other_pnl, other_pside, threshold, other_ts=180_700, other_fee=0.0
):
    bot = make_coin_bot()
    for name in (
        "_equity_hard_stop_runtime_initialized",
        "_equity_hard_stop_log_status",
    ):
        setattr(bot, name, MethodType(getattr(Passivbot, name), bot))
    bot._equity_hard_stop_status_log_interval_ms = 60_000
    bot.hsl["long"].update(red_threshold=threshold, ema_span_minutes=1.0)
    # A single configured budget slot makes the closing risk denominator explicit.
    bot.bot_value = lambda pside, key: 1.0
    events = [
        dict(
            timestamp=60_000,
            symbol="A",
            pside="long",
            action="increase",
            qty=1.0,
            pnl=0.0,
        ),
        dict(
            timestamp=90_000,
            symbol="B",
            pside=other_pside,
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
        dict(
            timestamp=other_ts,
            symbol="B",
            pside=other_pside,
            action="decrease",
            qty=1.0,
            pnl=other_pnl,
            fee_paid=other_fee,
        ),
    ]
    bot.positions = {
        symbol: {side: {"size": 0.0} for side in ("long", "short")}
        for symbol in ("A", "B")
    }
    bot._pnls_manager = make_fake_pnls_manager(events)
    bot.get_raw_balance = lambda: 700.0 + other_pnl + other_fee
    bot.get_exchange_time = lambda: 240_900
    bot._equity_hard_stop_realized_pnl_now = lambda pside=None: sum(
        (e["pnl"] + e.get("fee_paid", 0.0))
        for e in events
        if pside is None or e["pside"] == pside
    )

    async def history(**kwargs):
        timeline = []
        for ts in (60_000, 120_000, 180_000, 240_000):
            prior = [e for e in events if e["timestamp"] < ts + 60_000]
            by_pair = {
                symbol: {
                    side: sum(
                        e["pnl"] + e.get("fee_paid", 0.0)
                        for e in prior
                        if e["symbol"] == symbol and e["pside"] == side
                    )
                    for side in ("long", "short")
                }
                for symbol in ("A", "B")
            }
            realized = sum(e["pnl"] + e.get("fee_paid", 0.0) for e in prior)
            timeline.append(
                {
                    "timestamp": ts,
                    "balance": 1000.0 + realized,
                    "realized_pnl": realized,
                    "realized_pnl_by_coin_pside": by_pair,
                    "unrealized_pnl_by_coin_pside": {
                        symbol: {"long": 0.0, "short": 0.0} for symbol in ("A", "B")
                    },
                }
            )
        return {"timeline": timeline, "fill_events": events, "panic_flatten_events": []}

    bot.get_balance_equity_history = history
    samples = []
    original = bot._equity_hard_stop_apply_coin_metrics_sample

    def record_sample(self, pside, symbol, timestamp_ms, balance, *args, **kwargs):
        result = original(pside, symbol, timestamp_ms, balance, *args, **kwargs)
        if symbol == "A" and timestamp_ms == 180_500 and kwargs.get("at_fill_boundary"):
            samples.append((balance, dict(result)))
        return result

    bot._equity_hard_stop_apply_coin_metrics_sample = MethodType(record_sample, bot)
    return bot, samples


@pytest.mark.asyncio
@pytest.mark.parametrize("other_pside", ["long", "short"])
@pytest.mark.parametrize(
    "other_pnl,threshold,expected_red", [(1000.0, 0.4, True), (-100.0, 0.45, False)]
)
async def test_coin_closing_risk_uses_account_balance_before_other_later_fill(
    other_pside, other_pnl, threshold, expected_red
):
    bot, samples = _two_coin_boundary_bot(
        other_pnl=other_pnl, other_pside=other_pside, threshold=threshold
    )
    # Repeating startup must reconstruct the same historical decision from scratch.
    for _ in range(2):
        samples.clear()
        await bot._equity_hard_stop_initialize_coin_from_history()
        assert samples
        balance, metrics = samples[0]
        assert balance == pytest.approx(700.0)
        assert metrics["realized_pnl"] == pytest.approx(-300.0)
        assert metrics["drawdown_raw"] == pytest.approx(300.0 / 700.0)
        assert (metrics["tier"] == "red") is expected_red
        state = bot._hsl_coin_state("long", "A")
        assert bool(state["halted"]) is expected_red
        if expected_red:
            assert state["last_stop_event"]["stop_event_timestamp_ms"] == 180_500
            assert state["cooldown_until_ms"] == 480_500
        else:
            assert state["last_stop_event"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("other_pside", ["long", "short"])
@pytest.mark.parametrize("other_pnl", [1000.0, -100.0])
async def test_coin_boundary_includes_other_pair_same_timestamp_cohort(
    other_pside, other_pnl
):
    bot, samples = _two_coin_boundary_bot(
        other_pnl=other_pnl, other_pside=other_pside, threshold=0.4, other_ts=180_500
    )
    await bot._equity_hard_stop_initialize_coin_from_history()
    balance, metrics = samples[0]
    # Separate pairs at the same timestamp have no global execution order. The
    # existing incremental convention includes the whole timestamp cohort.
    assert balance == pytest.approx(700.0 + other_pnl)
    assert metrics["realized_pnl"] == pytest.approx(-300.0)
    assert metrics["drawdown_raw"] == pytest.approx(300.0 / (700.0 + other_pnl))


@pytest.mark.asyncio
@pytest.mark.parametrize("other_pside", ["long", "short"])
async def test_coin_boundary_excludes_other_pair_later_fee(other_pside):
    bot, samples = _two_coin_boundary_bot(
        other_pnl=0.0, other_pside=other_pside, threshold=0.4293, other_fee=-2.0
    )
    await bot._equity_hard_stop_initialize_coin_from_history()
    balance, metrics = samples[0]
    assert balance == pytest.approx(700.0)
    assert metrics["drawdown_raw"] == pytest.approx(300.0 / 700.0)
    assert metrics["tier"] != "red"
    assert not bot._hsl_coin_state("long", "A")["halted"]
