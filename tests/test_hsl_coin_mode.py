import asyncio
import logging
from types import MethodType, SimpleNamespace

import pytest

from passivbot import Passivbot

pbr = pytest.importorskip("passivbot_rust", reason="passivbot_rust extension not available")
hsl = pytest.importorskip("passivbot_hsl", reason="live HSL dependencies not available")

if bool(getattr(pbr, "__is_stub__", False)):
    pytest.skip("passivbot_rust extension not available", allow_module_level=True)


class FakeHslBot(SimpleNamespace):
    pass


class FakeRiskCache:
    def __init__(self, covered_start_ms=1, history_scope="all"):
        self.covered_start_ms = covered_start_ms
        self.history_scope = history_scope

    def get_known_gaps(self):
        return []

    def get_covered_start_ms(self):
        return self.covered_start_ms

    def get_history_scope(self):
        return self.history_scope

    def load_metadata(self):
        return {
            "known_gaps": [],
            "covered_start_ms": self.covered_start_ms,
            "history_scope": self.history_scope,
            "oldest_event_ts": self.covered_start_ms,
            "newest_event_ts": 0,
        }


def make_fake_pnls_manager(events, *, covered_start_ms=1, history_scope="all"):
    cache = FakeRiskCache(covered_start_ms=covered_start_ms, history_scope=history_scope)
    return SimpleNamespace(
        get_events=lambda: events,
        cache=cache,
        get_history_scope=cache.get_history_scope,
    )


def bind_hsl_methods(bot):
    for name in (
        "_hsl_psides",
        "_hsl_state",
        "_equity_hard_stop_enabled",
        "_equity_hard_stop_runtime_red_latched",
        "_equity_hard_stop_runtime_tier",
        "_equity_hard_stop_signal_mode",
        "_equity_hard_stop_cooldown_position_policy",
        "_calc_upnl_sum_strict",
        "_equity_hard_stop_apply_coin_sample",
        "_equity_hard_stop_apply_coin_metrics_sample",
        "_equity_hard_stop_activate_coin_red_from_metrics",
        "_equity_hard_stop_coin_active_pside",
        "_equity_hard_stop_coin_realized_pnl_peak_last",
        "_equity_hard_stop_coin_needs_panic_supervision",
        "_equity_hard_stop_coin_red_active",
        "_equity_hard_stop_coin_symbols",
        "_equity_hard_stop_handle_coin_position_during_cooldown",
        "_equity_hard_stop_has_open_position_symbol",
        "_equity_hard_stop_history_coin_value",
        "_equity_hard_stop_initialize_coin_from_history",
        "_equity_hard_stop_infer_coin_replay_contract",
        "_equity_hard_stop_lookback_ms",
        "_equity_hard_stop_log_transition",
        "_equity_hard_stop_build_latch_payload",
        "_equity_hard_stop_check_coin",
        "_equity_hard_stop_clear_coin_runtime_forced_mode",
        "_equity_hard_stop_compute_coin_stop_event",
        "_equity_hard_stop_finalize_coin_red_stop",
        "_equity_hard_stop_latest_panic_fill_timestamp_ms",
        "_equity_hard_stop_log_coin_cooldown_status",
        "_equity_hard_stop_make_state",
        "_equity_hard_stop_prime_coin_runtime_for_replay",
        "_equity_hard_stop_reset_coin_after_restart",
        "_equity_hard_stop_set_coin_runtime_forced_mode",
        "_equity_hard_stop_symbol_supported_for_coin_replay",
        "_hsl_coin_state",
    ):
        setattr(bot, name, MethodType(getattr(hsl, name), bot))
    for name in (
        "_assert_no_pending_pnl_events",
        "_pnl_history_coverage_status",
        "_pnl_blocking_known_gaps",
        "_pnl_gap_is_confirmed_legitimate",
        "_pnl_gap_overlaps",
        "_pnl_event_preview",
        "_assert_pnl_history_safe_for_risk",
    ):
        setattr(bot, name, MethodType(getattr(Passivbot, name), bot))


def make_coin_bot(policy="panic"):
    bot = FakeHslBot()
    bind_hsl_methods(bot)
    bot.user = "test_user"
    bot.exchange = "test_exchange"
    bot._equity_hard_stop = {
        "long": bot._equity_hard_stop_make_state(),
        "short": bot._equity_hard_stop_make_state(),
    }
    bot._equity_hard_stop_coin = {"long": {}, "short": {}}
    bot._runtime_forced_modes = {"long": {}, "short": {}}
    bot._pnls_manager = None
    bot.positions = {}
    bot.open_orders = {}
    bot.active_symbols = []
    bot.fetched_positions = []
    bot.c_mults = {}
    bot.config = {
        "live": {
            "hsl_signal_mode": "coin",
            "hsl_position_during_cooldown_policy": policy,
            "pnls_max_lookback_days": 30.0,
        }
    }
    bot.hsl = {
        "long": {
            "enabled": True,
            "red_threshold": 0.5,
            "tier_ratios": {"yellow": 0.5, "orange": 0.75},
            "ema_span_minutes": 1.0,
            "cooldown_minutes_after_red": 5.0,
            "no_restart_drawdown_threshold": 0.9,
            "orange_tier_mode": "tp_only_with_active_entry_cancellation",
            "panic_close_order_type": "market",
        },
        "short": {
            "enabled": False,
            "red_threshold": 0.5,
            "tier_ratios": {"yellow": 0.5, "orange": 0.75},
            "ema_span_minutes": 1.0,
            "cooldown_minutes_after_red": 5.0,
            "no_restart_drawdown_threshold": 0.9,
            "orange_tier_mode": "tp_only_with_active_entry_cancellation",
            "panic_close_order_type": "market",
        },
    }
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._equity_hard_stop_write_latch = lambda pside, payload, symbol=None: "/tmp/hsl_coin.json"
    bot._equity_hard_stop_remove_latch_file = lambda pside, symbol=None: None
    bot.get_raw_balance = lambda: 100.0
    bot.get_exchange_time = lambda: 180_000
    bot.live_value = lambda key: bot.config["live"][key]
    bot._equity_hard_stop_realized_pnl_now = lambda pside=None: 0.0

    def bot_value(pside, key):
        values = {
            "n_positions": 2,
            "total_wallet_exposure_limit": 2.0,
        }
        return values[key]

    bot.bot_value = bot_value

    async def calc_upnl(pside=None, symbol=None):
        return 0.0

    bot._calc_upnl_sum_strict = calc_upnl
    return bot


def test_passivbot_binds_coin_hsl_replay_support_helper():
    assert hasattr(Passivbot, "_equity_hard_stop_symbol_supported_for_coin_replay")


def test_calc_upnl_sum_strict_preserves_symbol_filter():
    bot = FakeHslBot()
    bind_hsl_methods(bot)
    bot.fetched_positions = [
        {"symbol": "A", "position_side": "long", "price": 100.0, "size": 1.0},
        {"symbol": "B", "position_side": "long", "price": 100.0, "size": 2.0},
    ]
    bot.c_mults = {"A": 1.0, "B": 1.0}

    async def get_live_last_prices(symbols, max_age_ms, context):
        return {"A": 90.0, "B": 80.0}

    bot._get_live_last_prices = get_live_last_prices

    assert asyncio.run(bot._calc_upnl_sum_strict("long")) == pytest.approx(-50.0)
    assert asyncio.run(bot._calc_upnl_sum_strict("long", "A")) == pytest.approx(-10.0)


def test_coin_hsl_slot_budget_rejects_zero_n_positions():
    bot = FakeHslBot()
    bind_hsl_methods(bot)
    bot._equity_hard_stop_coin = {"long": {}, "short": {}}
    bot._pnls_manager = None
    bot.config = {"live": {"pnls_max_lookback_days": 30.0}}
    bot.hsl = {
        "long": {
            "red_threshold": 0.5,
            "tier_ratios": {"yellow": 0.5, "orange": 0.75},
            "ema_span_minutes": 1.0,
        }
    }

    def bot_value(pside, key):
        values = {
            "n_positions": 0,
            "total_wallet_exposure_limit": 1.0,
        }
        return values[key]

    bot.bot_value = bot_value

    with pytest.raises(ValueError, match="n_positions"):
        bot._equity_hard_stop_apply_coin_sample("long", "A", 60_000, 100.0, -1.0)


@pytest.mark.asyncio
async def test_coin_hsl_check_skips_enabled_side_with_zero_budget():
    bot = make_coin_bot()
    symbol = "A"
    bot.hsl["short"]["enabled"] = True
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    def bot_value(pside, key):
        values = {
            "long": {"n_positions": 2, "total_wallet_exposure_limit": 2.0},
            "short": {"n_positions": 3, "total_wallet_exposure_limit": 0.0},
        }
        return values[pside][key]

    bot.bot_value = bot_value

    out = await bot._equity_hard_stop_check_coin()

    assert set(out) == {f"long:{symbol}"}
    assert symbol in bot._equity_hard_stop_coin["long"]
    assert bot._equity_hard_stop_coin["short"] == {}


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_skips_enabled_side_with_zero_budget():
    bot = make_coin_bot()
    symbol = "A"
    bot.hsl["short"]["enabled"] = True
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    def bot_value(pside, key):
        values = {
            "long": {"n_positions": 2, "total_wallet_exposure_limit": 2.0},
            "short": {"n_positions": 3, "total_wallet_exposure_limit": 0.0},
        }
        return values[pside][key]

    bot.bot_value = bot_value

    async def fake_history(current_balance=None):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    assert symbol in bot._equity_hard_stop_coin["long"]
    assert bot._equity_hard_stop_coin["short"] == {}


@pytest.mark.parametrize(
    "n_positions,total_wallet_exposure_limit,match",
    [
        (-1, 1.0, "n_positions"),
        (float("nan"), 1.0, "n_positions"),
        (0.4, 1.0, "round to > 0"),
        (1, -1.0, "total_wallet_exposure_limit"),
        (1, float("inf"), "total_wallet_exposure_limit"),
    ],
)
def test_coin_hsl_active_side_rejects_invalid_budget_config(
    n_positions, total_wallet_exposure_limit, match
):
    bot = make_coin_bot()

    def bot_value(pside, key):
        values = {
            "n_positions": n_positions,
            "total_wallet_exposure_limit": total_wallet_exposure_limit,
        }
        return values[key]

    bot.bot_value = bot_value

    with pytest.raises(ValueError, match=match):
        bot._equity_hard_stop_coin_active_pside("long")


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_latches_red_without_panic_marker():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    async def fake_history(current_balance=None):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": -80.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 60_000,
                    "symbol": symbol,
                    "pside": "long",
                    "pnl": 0.0,
                }
            ],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["runtime"].red_latched() is True
    assert state["runtime"].tier() == "red"
    assert state["pending_red_since_ms"] == 120_000
    assert state["pending_stop_event"] is None
    assert bot._runtime_forced_modes["long"][symbol] == "panic"
    assert bot._equity_hard_stop_coin_red_active() is True


@pytest.mark.asyncio
async def test_coin_hsl_check_defers_stop_event_until_flat_confirmation():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    bot.get_exchange_time = lambda: 180_000

    async def calc_upnl(pside=None, symbol=None):
        return -80.0

    async def fail_compute(*_args, **_kwargs):
        raise AssertionError("coin HSL must not snapshot stop event at RED trigger time")

    bot._calc_upnl_sum_strict = calc_upnl
    bot._equity_hard_stop_compute_coin_stop_event = fail_compute
    bot._equity_hard_stop_apply_coin_metrics_sample(
        "long",
        symbol,
        60_000,
        100.0,
        0.0,
        0.0,
        0.0,
    )

    out = await bot._equity_hard_stop_check_coin()

    state = bot._hsl_coin_state("long", symbol)
    assert out[f"long:{symbol}"]["tier"] == "red"
    assert state["pending_red_since_ms"] == 180_000
    assert state["pending_stop_event"] is None
    assert bot._runtime_forced_modes["long"][symbol] == "panic"


@pytest.mark.asyncio
async def test_coin_hsl_finalize_uses_latest_panic_fill_for_reset_boundary():
    bot = make_coin_bot()
    symbol = "A"
    bot.get_exchange_time = lambda: 180_000
    state = bot._hsl_coin_state("long", symbol)
    state["pending_red_since_ms"] = 120_000
    bot._pnls_manager = make_fake_pnls_manager(
        [
            {
                "timestamp": 170_000,
                "symbol": symbol,
                "pside": "long",
                "pb_order_type": "close_panic_long",
                "pnl": -12.0,
                "fee_paid": -0.1,
            }
        ]
    )

    await bot._equity_hard_stop_finalize_coin_red_stop("long", symbol)

    assert state["last_stop_event"]["stop_event_timestamp_ms"] == 170_000
    assert state["pnl_reset_timestamp_ms"] == 170_001
    assert state["cooldown_until_ms"] == 470_000


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_rebases_lookback_window_realized_points():
    bot = make_coin_bot()
    symbol = "A"
    bot.config["live"]["pnls_max_lookback_days"] = 1.0 / 1440.0
    bot.get_exchange_time = lambda: 240_000
    fill_events = [
        {"timestamp": 120_000, "symbol": symbol, "pside": "long", "pnl": -20.0},
        {"timestamp": 240_000, "symbol": symbol, "pside": "long", "pnl": -35.0},
    ]
    bot._pnls_manager = make_fake_pnls_manager(fill_events)

    async def fake_history(current_balance=None):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": -20.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -20.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 240_000,
                    "balance": 100.0,
                    "realized_pnl": -55.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -55.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": fill_events,
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["runtime"].red_latched() is False
    assert state["last_metrics"]["realized_pnl"] == pytest.approx(-35.0)
    assert state["last_metrics"]["drawdown_raw"] == pytest.approx(0.35)


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_uses_stop_drawdown_for_no_restart():
    bot = make_coin_bot()
    symbol = "A"
    bot.hsl["long"]["no_restart_drawdown_threshold"] = 0.7

    async def fake_history(current_balance=None):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": -80.0, "short": 0.0}},
                },
                {
                    "timestamp": 180_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [
                {
                    "timestamp": 120_500,
                    "minute_timestamp": 120_000,
                    "pside": "long",
                    "symbol": symbol,
                }
            ],
            "fill_events": [],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is True
    assert state["no_restart_latched"] is True
    assert state["cooldown_until_ms"] is None
    assert state["last_stop_event"]["drawdown_raw"] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_requires_coin_timeline_fields():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    async def fake_history(current_balance=None):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 60_000,
                    "symbol": symbol,
                    "pside": "long",
                    "pnl": 0.0,
                }
            ],
        }

    bot.get_balance_equity_history = fake_history

    with pytest.raises(ValueError, match="realized_pnl_by_coin_pside"):
        await bot._equity_hard_stop_initialize_coin_from_history()


@pytest.mark.asyncio
async def test_coin_hsl_open_position_missing_history_uses_current_sample():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    async def fake_history(current_balance=None):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["last_metrics"]["timestamp_ms"] == 180_000
    assert state["last_metrics"]["tier"] == "green"


@pytest.mark.asyncio
async def test_coin_hsl_open_position_empty_coin_history_uses_current_sample():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    async def fake_history(current_balance=None):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {},
                    "unrealized_pnl_by_coin_pside": {},
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["last_metrics"]["timestamp_ms"] == 180_000
    assert state["last_metrics"]["tier"] == "green"


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_allows_leading_rows_before_first_fill():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    async def fake_history(current_balance=None):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {},
                    "unrealized_pnl_by_coin_pside": {},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 120_000,
                    "symbol": symbol,
                    "pside": "long",
                    "pnl": 0.0,
                }
            ],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["last_metrics"]["timestamp_ms"] == 180_000
    assert state["last_metrics"]["tier"] == "green"


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_requires_relevant_symbol_fields():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    async def fake_history(current_balance=None):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {},
                    "unrealized_pnl_by_coin_pside": {},
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 60_000,
                    "symbol": symbol,
                    "pside": "long",
                    "pnl": 0.0,
                }
            ],
        }

    bot.get_balance_equity_history = fake_history

    with pytest.raises(ValueError, match="missing required coin HSL symbol"):
        await bot._equity_hard_stop_initialize_coin_from_history()


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_allows_flat_realized_only_rows():
    bot = make_coin_bot()
    symbol = "A"

    async def fake_history(current_balance=None):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 95.0,
                    "realized_pnl": -5.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -5.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 60_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "increase",
                    "qty": 1.0,
                    "price": 100.0,
                    "pnl": 0.0,
                },
                {
                    "timestamp": 120_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "qty": 1.0,
                    "price": 95.0,
                    "pnl": -5.0,
                },
            ],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["last_metrics"]["timestamp_ms"] == 180_000
    assert state["last_metrics"]["tier"] == "green"


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_requires_upnl_for_carry_in_decrease():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {
        symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}
    }

    async def fake_history(current_balance=None):
        return {
            "timeline": [
                {
                    "timestamp": 120_000,
                    "balance": 95.0,
                    "realized_pnl": -5.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -5.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 120_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "qty": 1.0,
                    "price": 95.0,
                    "pnl": -5.0,
                },
            ],
        }

    bot.get_balance_equity_history = fake_history

    with pytest.raises(ValueError, match="unrealized_pnl_by_coin_pside"):
        await bot._equity_hard_stop_initialize_coin_from_history()


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_skips_flat_missing_upnl_for_historical_pair(caplog):
    bot = make_coin_bot()
    symbol = "A"

    async def fake_history(current_balance=None):
        return {
            "timeline": [
                {
                    "timestamp": 120_000,
                    "balance": 95.0,
                    "realized_pnl": -5.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -5.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 120_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "qty": 1.0,
                    "price": 95.0,
                    "pnl": -5.0,
                },
            ],
        }

    bot.get_balance_equity_history = fake_history

    with caplog.at_level(logging.WARNING):
        await bot._equity_hard_stop_initialize_coin_from_history()

    assert bot._equity_hard_stop_coin_initialized is True
    assert any(
        "skipped flat historical pairs with missing unrealized_pnl_by_coin_pside"
        in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_coin_hsl_reconstructs_unresolved_panic_residue_on_restart():
    bot = make_coin_bot(policy="normal")
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    bot.get_exchange_time = lambda: 200_000

    async def fake_history(current_balance=None):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": -80.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 121_500,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "pb_order_type": "close_panic_long",
                }
            ],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is True
    assert state["cooldown_until_ms"] == 421_500
    assert state["cooldown_unresolved_residue"] is True
    assert state["cooldown_intervention_active"] is False
    assert bot._runtime_forced_modes["long"][symbol] == "panic"

    changed = await bot._equity_hard_stop_handle_coin_position_during_cooldown(
        "long", symbol, 200_000
    )
    assert changed is False
    assert state["halted"] is True


@pytest.mark.asyncio
async def test_coin_hsl_reconstructs_manual_cooldown_intervention_on_restart():
    bot = make_coin_bot(policy="manual")
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    bot.get_exchange_time = lambda: 180_000

    async def fake_history(current_balance=None):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": -80.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 100_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "pb_order_type": "close_panic_long",
                },
                {
                    "timestamp": 130_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "increase",
                    "pb_order_type": "entry_initial_normal_long",
                },
            ],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is True
    assert state["cooldown_until_ms"] == 400_000
    assert state["cooldown_unresolved_residue"] is False
    assert state["cooldown_intervention_active"] is True
    assert bot._runtime_forced_modes["long"][symbol] == "manual"


@pytest.mark.asyncio
async def test_coin_hsl_reconstructs_normal_cooldown_intervention_as_override():
    bot = make_coin_bot(policy="normal")
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    bot.get_exchange_time = lambda: 180_000

    async def fake_history(current_balance=None):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 180_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 100_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "pb_order_type": "close_panic_long",
                },
                {
                    "timestamp": 130_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "increase",
                    "pb_order_type": "entry_initial_normal_long",
                },
            ],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is False
    assert state["cooldown_until_ms"] is None
    assert symbol not in bot._runtime_forced_modes["long"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy,expected_mode",
    [
        ("manual", "manual"),
        ("tp_only", "tp_only_with_active_entry_cancellation"),
    ],
)
async def test_coin_hsl_check_preserves_cooldown_policy_forced_mode(policy, expected_mode):
    bot = make_coin_bot(policy=policy)
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    state = bot._hsl_coin_state("long", symbol)
    state["halted"] = True
    state["cooldown_until_ms"] = 300_000

    await bot._equity_hard_stop_check_coin()

    assert bot._runtime_forced_modes["long"][symbol] == expected_mode


@pytest.mark.asyncio
async def test_coin_hsl_check_tp_only_orange_skips_flat_symbols():
    bot = make_coin_bot()
    open_symbol = "A"
    flat_symbol = "B"
    bot.positions = {
        open_symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}},
        flat_symbol: {"long": {"size": 0.0, "price": 0.0}, "short": {"size": 0.0}},
    }

    async def calc_upnl(pside=None, symbol=None):
        return -40.0

    bot._calc_upnl_sum_strict = calc_upnl
    bot._equity_hard_stop_prime_coin_runtime_for_replay("long", open_symbol, 180_000)
    bot._equity_hard_stop_prime_coin_runtime_for_replay("long", flat_symbol, 180_000)

    out = await bot._equity_hard_stop_check_coin()

    assert out[f"long:{open_symbol}"]["tier"] == "orange"
    assert out[f"long:{flat_symbol}"]["tier"] == "orange"
    assert (
        bot._runtime_forced_modes["long"][open_symbol]
        == "tp_only_with_active_entry_cancellation"
    )
    assert flat_symbol not in bot._runtime_forced_modes["long"]
