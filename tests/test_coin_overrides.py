from copy import deepcopy

from config_utils import parse_overrides
from passivbot import Passivbot


def test_coin_override_forced_mode_manual(monkeypatch):
    base_config = {
        "bot": {"long": {}, "short": {}},
        "live": {
            "user": "dummy",
            "forced_mode_long": "",
            "forced_mode_short": "",
        },
        "coin_overrides": {
            "DOGEUSDT": {
                "live": {
                    "forced_mode_long": "manual",
                }
            }
        },
    }

    config = parse_overrides(deepcopy(base_config), verbose=False)
    assert "DOGE" in config["coin_overrides"]

    bot = Passivbot.__new__(Passivbot)
    bot.config = config
    bot.exchange = "binance"
    bot.markets_dict = {"DOGE/USDT:USDT": {"active": True}}

    def fake_coin_to_symbol(self, coin, verbose=True):
        if coin in {"DOGE", "DOGEUSDT"}:
            return "DOGE/USDT:USDT"
        return ""

    bot.coin_to_symbol = fake_coin_to_symbol.__get__(bot, Passivbot)
    bot.init_coin_overrides()

    assert "DOGE/USDT:USDT" in bot.coin_overrides
    assert bot.get_forced_PB_mode("long", "DOGE/USDT:USDT") == "manual"
