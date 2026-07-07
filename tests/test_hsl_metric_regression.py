"""Metric-level regression pins for HSL signal modes (action plan A1.6).

These tests freeze the numeric drawdown/tier semantics of the three HSL signal
modes with hand-computed fixtures so future refactors (canonical equity-history
redesign, replay-cache reuse) cannot silently change metric meaning.

All fixtures use ema_span_minutes=1.0 so drawdown_ema == drawdown_raw ==
drawdown_score and expected values stay hand-checkable.
"""

from types import MethodType

import pytest

pbr = pytest.importorskip("passivbot_rust", reason="passivbot_rust extension not available")
hsl = pytest.importorskip("passivbot_hsl", reason="live HSL dependencies not available")


_BOUND_METHODS = (
    "_hsl_psides",
    "_hsl_state",
    "_hsl_coin_state",
    "_equity_hard_stop_make_state",
    "_equity_hard_stop_signal_mode",
    "_equity_hard_stop_signal_values",
    "_equity_hard_stop_lookback_ms",
    "_equity_hard_stop_apply_sample",
    "_equity_hard_stop_apply_coin_metrics_sample",
    "_equity_hard_stop_runtime_tier",
    "_equity_hard_stop_runtime_red_latched",
)


class _MetricBot:
    pass


def _make_bot(signal_mode, *, n_positions=4.0, red_threshold=0.2):
    bot = _MetricBot()
    for name in _BOUND_METHODS:
        setattr(bot, name, MethodType(getattr(hsl, name), bot))
    bot.config = {
        "live": {
            "hsl_signal_mode": signal_mode,
            "pnls_max_lookback_days": 30.0,
        }
    }
    bot.live_value = lambda key: bot.config["live"][key]
    bot.bot_value = lambda pside, key: {"n_positions": n_positions}[key]
    side_cfg = {
        "enabled": True,
        "red_threshold": red_threshold,
        "tier_ratios": {"yellow": 0.5, "orange": 0.75},
        "ema_span_minutes": 1.0,
        "cooldown_minutes_after_red": 5.0,
        "no_restart_drawdown_threshold": 0.9,
        "restart_after_red_policy": "threshold",
        "orange_tier_mode": "tp_only_with_active_entry_cancellation",
        "panic_close_order_type": "market",
    }
    bot.hsl = {"long": dict(side_cfg), "short": dict(side_cfg)}
    bot._equity_hard_stop = {
        "long": bot._equity_hard_stop_make_state(),
        "short": bot._equity_hard_stop_make_state(),
    }
    bot._equity_hard_stop_coin = {"long": {}, "short": {}}
    return bot


def test_unified_metrics_pin_drawdown_and_tier_ladder():
    # red_threshold 0.2, ratios 0.5/0.75 -> yellow at 0.10, orange at 0.15.
    bot = _make_bot("unified")

    def sample(ts, balance, realized_total, upnl_total):
        return bot._equity_hard_stop_apply_sample(
            "long",
            ts,
            balance,
            realized_total,
            0.0,
            0.0,
            unrealized_pnl_total=upnl_total,
            latch_red=False,
        )

    m = sample(60_000, 100.0, 0.0, 0.0)
    assert m["signal_mode"] == "unified"
    assert m["baseline_balance"] == pytest.approx(100.0)
    assert m["strategy_equity"] == pytest.approx(100.0)
    assert m["drawdown_raw"] == pytest.approx(0.0)
    assert m["tier"] == "green"

    m = sample(120_000, 100.0, 0.0, 10.0)
    assert m["strategy_equity"] == pytest.approx(110.0)
    assert m["drawdown_raw"] == pytest.approx(0.0)
    assert m["tier"] == "green"

    m = sample(180_000, 100.0, 0.0, -6.0)
    assert m["strategy_equity"] == pytest.approx(94.0)
    # Peak equity 110 -> (110 - 94) / 110.
    assert m["drawdown_raw"] == pytest.approx(16.0 / 110.0)
    assert m["drawdown_ema"] == pytest.approx(m["drawdown_raw"])
    assert m["drawdown_score"] == pytest.approx(m["drawdown_raw"])
    assert m["tier"] == "yellow"

    # Realized losses move balance but keep baseline anchored.
    m = sample(240_000, 92.0, -8.0, 0.0)
    assert m["baseline_balance"] == pytest.approx(100.0)
    assert m["strategy_equity"] == pytest.approx(92.0)
    assert m["drawdown_raw"] == pytest.approx(18.0 / 110.0)
    assert m["tier"] == "orange"

    m = sample(300_000, 92.0, -8.0, -14.0)
    assert m["strategy_equity"] == pytest.approx(78.0)
    assert m["drawdown_raw"] == pytest.approx(32.0 / 110.0)
    assert m["tier"] == "red"


def test_pside_metrics_use_scoped_signal_and_ignore_unified_upnl():
    bot = _make_bot("pside")

    def sample(ts, balance, realized_total, realized_pside, upnl_pside):
        return bot._equity_hard_stop_apply_sample(
            "long",
            ts,
            balance,
            realized_total,
            realized_pside,
            upnl_pside,
            # Pside mode must not require the unified total.
            unrealized_pnl_total=None,
            latch_red=False,
        )

    m = sample(60_000, 100.0, 0.0, 0.0, 0.0)
    assert m["signal_mode"] == "pside"
    assert m["strategy_equity"] == pytest.approx(100.0)
    assert m["tier"] == "green"

    m = sample(120_000, 100.0, 0.0, 0.0, 10.0)
    assert m["strategy_equity"] == pytest.approx(110.0)
    assert m["tier"] == "green"

    m = sample(180_000, 100.0, 0.0, 0.0, -6.0)
    assert m["strategy_equity"] == pytest.approx(94.0)
    assert m["drawdown_raw"] == pytest.approx(16.0 / 110.0)
    assert m["tier"] == "yellow"


def test_pside_baseline_subtracts_total_realized_before_adding_scoped_signal():
    # Contract pin: baseline_balance = balance - realized_pnl_total even in
    # pside mode, so scoped equity = balance - r_total + r_pside + u_pside.
    bot = _make_bot("pside")
    m = bot._equity_hard_stop_apply_sample(
        "long",
        60_000,
        100.0,
        20.0,
        -5.0,
        0.0,
        latch_red=False,
    )
    assert m["baseline_balance"] == pytest.approx(80.0)
    assert m["strategy_equity"] == pytest.approx(75.0)


def test_unified_mode_requires_unrealized_pnl_total():
    bot = _make_bot("unified")
    with pytest.raises(ValueError, match="unrealized_pnl_total"):
        bot._equity_hard_stop_apply_sample(
            "long", 60_000, 100.0, 0.0, 0.0, 0.0, latch_red=False
        )


def test_coin_metrics_pin_slot_budget_drawdown_ladder():
    # balance 100 / n_positions 4 -> slot budget 25.
    bot = _make_bot("coin")
    symbol = "A"

    def sample(ts, peak_realized, last_realized, upnl):
        return bot._equity_hard_stop_apply_coin_metrics_sample(
            "long",
            symbol,
            ts,
            100.0,
            peak_realized,
            last_realized,
            upnl,
            latch_red=False,
        )

    m = sample(60_000, 0.0, 0.0, 0.0)
    assert m["signal_mode"] == "coin"
    assert m["slot_budget"] == pytest.approx(25.0)
    assert m["drawdown_usd"] == pytest.approx(0.0)
    assert m["drawdown_raw"] == pytest.approx(0.0)
    assert m["tier"] == "green"

    m = sample(120_000, 0.0, 0.0, -3.0)
    assert m["drawdown_usd"] == pytest.approx(3.0)
    assert m["drawdown_raw"] == pytest.approx(3.0 / 25.0)
    assert m["tier"] == "yellow"

    # Realized recovery shrinks the drawdown again while unlatched.
    m = sample(180_000, 2.0, 2.0, -2.0)
    assert m["drawdown_usd"] == pytest.approx(2.0)
    assert m["drawdown_raw"] == pytest.approx(2.0 / 25.0)
    assert m["tier"] == "green"

    m = sample(240_000, 2.0, -1.0, -2.5)
    assert m["drawdown_usd"] == pytest.approx(5.5)
    assert m["drawdown_raw"] == pytest.approx(5.5 / 25.0)
    assert m["tier"] == "red"


def test_coin_drawdown_scales_with_n_positions_not_wallet_exposure():
    # A3.4 contract: coin HSL sensitivity is anchored to the configured slot
    # count. Raising exposure limits must not change the drawdown percentage,
    # while changing n_positions must.
    def coin_drawdown(bot):
        # Warm up with a zero-drawdown sample: the runtime anchors its peak on
        # the first observed sample, mirroring replay priming.
        bot._equity_hard_stop_apply_coin_metrics_sample(
            "long", "A", 60_000, 100.0, 0.0, 0.0, 0.0, latch_red=False
        )
        return bot._equity_hard_stop_apply_coin_metrics_sample(
            "long", "A", 120_000, 100.0, 0.0, 0.0, -3.0, latch_red=False
        )["drawdown_raw"]

    base_bot = _make_bot("coin", n_positions=4.0)
    baseline = coin_drawdown(base_bot)
    assert baseline == pytest.approx(3.0 / 25.0)

    boosted_bot = _make_bot("coin", n_positions=4.0)
    boosted_bot.hsl["long"]["red_threshold"] = 0.2
    # Simulate a config with much larger exposure allowances; the metric layer
    # never consults them, so any exposure knob is invisible here by design.
    boosted_bot.config["bot"] = {
        "long": {"total_wallet_exposure_limit": 10.0, "we_excess_allowance_pct": 5.0}
    }
    assert coin_drawdown(boosted_bot) == pytest.approx(baseline)

    fewer_slots_bot = _make_bot("coin", n_positions=2.0)
    assert coin_drawdown(fewer_slots_bot) == pytest.approx(3.0 / 50.0)


def test_coin_metrics_reject_nonpositive_slot_inputs():
    bot = _make_bot("coin", n_positions=0.0)
    with pytest.raises(ValueError, match="n_positions"):
        bot._equity_hard_stop_apply_coin_metrics_sample(
            "long", "A", 60_000, 100.0, 0.0, 0.0, 0.0, latch_red=False
        )
    bot = _make_bot("coin")
    with pytest.raises(ValueError, match="balance"):
        bot._equity_hard_stop_apply_coin_metrics_sample(
            "long", "A", 60_000, 0.0, 0.0, 0.0, 0.0, latch_red=False
        )


def test_no_restart_trigger_uses_max_of_raw_and_ema():
    # Contract (plan B2.1, clarified 2026-07-06): the permanent no-restart
    # halt trips on max(drawdown_raw, drawdown_ema) — conservative, catching
    # catastrophic instantaneous damage OR sustained smoothed damage — while
    # the RED/panic-now tier score stays min(raw, ema).
    cfg = {
        "restart_after_red_policy": "threshold",
        "no_restart_drawdown_threshold": 0.2,
    }
    assert hsl._equity_hard_stop_no_restart_latched(cfg, 0.25, 0.05) is True
    # Sustained smoothed damage trips the halt even after raw recovered.
    assert hsl._equity_hard_stop_no_restart_latched(cfg, 0.05, 0.25) is True
    assert hsl._equity_hard_stop_no_restart_latched(cfg, 0.15, 0.19) is False
    assert (
        hsl._equity_hard_stop_no_restart_latched(
            dict(cfg, restart_after_red_policy="always"), 1.0, 1.0
        )
        is False
    )
    assert (
        hsl._equity_hard_stop_no_restart_latched(
            dict(cfg, restart_after_red_policy="never"), 0.0, 0.0
        )
        is True
    )
    with pytest.raises(ValueError):
        hsl._equity_hard_stop_no_restart_latched(
            dict(cfg, restart_after_red_policy="sometimes"), 1.0, 1.0
        )


def test_red_tier_score_is_min_of_raw_and_ema():
    # Pin the existing Rust runtime semantics as the #1122 RED contract:
    # min(raw, ema) must cross red_threshold, so a raw flash-crash spike with
    # a calm EMA does not trigger RED, and a stale high EMA after recovery
    # does not either.
    bot = _make_bot("unified")

    def sample(ts, upnl_total):
        return bot._equity_hard_stop_apply_sample(
            "long",
            ts,
            100.0,
            0.0,
            0.0,
            0.0,
            unrealized_pnl_total=upnl_total,
            latch_red=False,
        )

    slow_bot = _make_bot("unified")
    slow_bot.hsl["long"]["ema_span_minutes"] = 60.0

    def slow_sample(ts, upnl_total):
        return slow_bot._equity_hard_stop_apply_sample(
            "long",
            ts,
            100.0,
            0.0,
            0.0,
            0.0,
            unrealized_pnl_total=upnl_total,
            latch_red=False,
        )

    slow_sample(60_000, 10.0)
    # Raw crashes 30/110 > red 0.2 but the 60m EMA stays calm: no RED.
    m = slow_sample(120_000, -20.0)
    assert m["drawdown_raw"] > m["red_threshold"]
    assert m["drawdown_ema"] < m["red_threshold"]
    assert m["drawdown_score"] == pytest.approx(
        min(m["drawdown_raw"], m["drawdown_ema"])
    )
    assert m["tier"] != "red"


def test_red_latching_holds_tier_after_recovery():
    bot = _make_bot("unified")

    def sample(ts, upnl_total, *, latch_red):
        return bot._equity_hard_stop_apply_sample(
            "long",
            ts,
            100.0,
            0.0,
            0.0,
            0.0,
            unrealized_pnl_total=upnl_total,
            latch_red=latch_red,
        )

    sample(60_000, 10.0, latch_red=True)
    m = sample(120_000, -15.0, latch_red=True)
    # Peak 110, equity 85 -> drawdown 25/110 > red threshold 0.2.
    assert m["tier"] == "red"
    assert bot._equity_hard_stop_runtime_red_latched("long") is True

    # Full recovery does not clear a latched red tier.
    m = sample(180_000, 12.0, latch_red=True)
    assert m["drawdown_raw"] == pytest.approx(0.0)
    assert m["tier"] == "red"
