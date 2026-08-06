from copy import deepcopy

import passivbot
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
    assert "DOGEUSDT" in config["coin_overrides"]

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


def test_forced_mode_shorthand_expansion(monkeypatch):
    """Verify shorthand modes ('p', 'gs', 'm') are expanded to full names."""
    base_config = {
        "bot": {"long": {}, "short": {}},
        "live": {
            "user": "dummy",
            "forced_mode_long": "p",  # shorthand for "panic"
            "forced_mode_short": "gs",  # shorthand for "graceful_stop"
        },
        "coin_overrides": {},
    }

    config = parse_overrides(deepcopy(base_config), verbose=False)

    bot = Passivbot.__new__(Passivbot)
    bot.config = config
    bot.exchange = "binance"
    bot.markets_dict = {"ETH/USDT:USDT": {"active": True}}
    bot.coin_overrides = {}

    # Shorthand "p" should expand to "panic"
    assert bot.get_forced_PB_mode("long", "ETH/USDT:USDT") == "panic"
    # Shorthand "gs" should expand to "graceful_stop"
    assert bot.get_forced_PB_mode("short", "ETH/USDT:USDT") == "graceful_stop"


def test_bot_symbol_cache_does_not_reuse_lossy_canonical_alias(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "testexchange"
    bot.quote = "USDT"
    bot.coin_to_symbol_map = {"ABC": "ABC/USDT:USDT"}
    calls = []

    def exact_resolver(identifier, exchange, quote=None, verbose=True):
        calls.append((identifier, exchange, quote))
        return "1000ABC/USDT:USDT"

    monkeypatch.setattr(passivbot, "coin_to_symbol", exact_resolver)
    monkeypatch.setattr(passivbot, "symbol_to_coin", lambda *_args, **_kwargs: "ABC")

    assert bot.coin_to_symbol("1000ABC/USDT:USDT") == "1000ABC/USDT:USDT"
    assert calls == [("1000ABC/USDT:USDT", "testexchange", "USDT")]


def test_bot_symbol_cache_revalidates_previously_cached_alias(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "testexchange"
    bot.quote = "USDT"
    bot.coin_to_symbol_map = {"ABC": "STALE/USDT:USDT"}

    monkeypatch.setattr(
        passivbot,
        "coin_to_symbol",
        lambda identifier, exchange, quote=None, verbose=True: "CURRENT/USDT:USDT",
    )

    assert bot.coin_to_symbol("ABC") == "CURRENT/USDT:USDT"
