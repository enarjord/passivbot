import pytest

from passivbot import Passivbot
from passivbot_exceptions import FatalBotException


def _bot_with_market(symbol: str, market: dict) -> Passivbot:
    bot = Passivbot.__new__(Passivbot)
    bot.markets_dict = {symbol: market}
    bot.active_symbols = []
    bot.coin_overrides = {}
    bot.approved_coins_minus_ignored_coins = {"long": set(), "short": set()}
    bot.positions = {}
    bot.open_orders = {}
    return bot


def test_supported_live_state_allows_perpetual_swaps():
    symbol = "BTC/USDT:USDT"
    bot = _bot_with_market(
        symbol,
        {
            "swap": True,
            "future": False,
            "expiry": None,
        },
    )
    bot.approved_coins_minus_ignored_coins["long"].add(symbol)

    bot._assert_supported_live_state()


def test_supported_live_state_blocks_approved_dated_futures():
    symbol = "BTC_260327/USDT:USDT"
    bot = _bot_with_market(
        symbol,
        {
            "swap": False,
            "future": True,
            "expiry": 1774579200000,
        },
    )
    bot.approved_coins_minus_ignored_coins["long"].add(symbol)

    with pytest.raises(FatalBotException, match="Unsupported dated futures contracts"):
        bot._assert_supported_live_state()


def test_supported_live_state_blocks_dated_futures_positions_and_orders():
    symbol = "ETH_260327/USDT:USDT"
    bot = _bot_with_market(
        symbol,
        {
            "swap": False,
            "type": "future",
            "info": {"deliveryDate": "2026-03-27"},
        },
    )
    bot.positions = {
        symbol: {
            "long": {"size": 0.1, "price": 3000.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    bot.open_orders = {symbol: [{"id": "1"}]}

    with pytest.raises(FatalBotException, match=symbol):
        bot._assert_supported_live_state()
