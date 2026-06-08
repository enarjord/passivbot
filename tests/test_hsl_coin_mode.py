import asyncio
from types import MethodType, SimpleNamespace

import pytest

try:
    import passivbot_rust as pbr
except Exception:  # pragma: no cover
    pbr = None

import passivbot_hsl as hsl

pbr_is_stub = bool(getattr(pbr, "__is_stub__", False)) if pbr is not None else False


class FakeHslBot(SimpleNamespace):
    pass


def bind_hsl_methods(bot):
    for name in (
        "_calc_upnl_sum_strict",
        "_equity_hard_stop_apply_coin_sample",
        "_equity_hard_stop_coin_realized_pnl_peak_last",
        "_equity_hard_stop_lookback_ms",
        "_equity_hard_stop_make_state",
        "_hsl_coin_state",
    ):
        setattr(bot, name, MethodType(getattr(hsl, name), bot))


@pytest.mark.skipif(
    pbr is None or pbr_is_stub,
    reason="passivbot_rust extension not available",
)
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


@pytest.mark.skipif(
    pbr is None or pbr_is_stub,
    reason="passivbot_rust extension not available",
)
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
