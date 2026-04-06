import logging

from passivbot import Passivbot


def test_add_to_coins_lists_skips_symbols_not_in_eligible_markets(caplog):
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "bitget"
    bot.markets_dict = {"AAA/USDT:USDT": {"swap": True}}
    bot.eligible_symbols = {"AAA/USDT:USDT"}
    bot.approved_coins = {"long": set(), "short": set()}
    bot.ignored_coins = {"long": set(), "short": set()}

    def fake_coin_to_symbol(self, coin, verbose=True):
        mapping = {"AAA": "AAA/USDT:USDT", "BBB": "BBB/USDT:USDT"}
        return mapping.get(coin, f"{coin}/USDT:USDT")

    bot.coin_to_symbol = fake_coin_to_symbol.__get__(bot, Passivbot)

    with caplog.at_level(logging.INFO):
        bot.add_to_coins_lists(
            {"long": ["AAA", "BBB"], "short": []},
            "approved_coins",
            log_psides={"long"},
        )
        bot.add_to_coins_lists(
            {"long": ["AAA", "BBB"], "short": []},
            "approved_coins",
            log_psides={"long"},
        )

    assert bot.approved_coins["long"] == {"AAA/USDT:USDT"}
    assert bot.approved_coins["short"] == set()
    warnings = [
        rec.message for rec in caplog.records if "skipping unsupported markets" in rec.message.lower()
    ]
    assert len(warnings) == 1


def test_refresh_approved_ignored_coin_lists_supports_explicit_all_per_side():
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "bitget"
    bot.eligible_symbols = {"AAA/USDT:USDT", "BBB/USDT:USDT"}
    bot.approved_coins = {"long": set(), "short": set()}
    bot.ignored_coins = {"long": set(), "short": set()}
    bot.approved_coins_minus_ignored_coins = {"long": set(), "short": set()}
    bot._disabled_psides_logged = set()
    bot._unsupported_coin_warnings = set()
    bot.config = {
        "_coins_sources": {
            "approved_coins": {"long": ["AAA"], "short": "all"},
            "ignored_coins": {"long": [], "short": []},
        },
        "live": {},
    }

    def fake_coin_to_symbol(self, coin, verbose=True):
        mapping = {"AAA": "AAA/USDT:USDT", "BBB": "BBB/USDT:USDT"}
        return mapping.get(coin, f"{coin}/USDT:USDT")

    bot.coin_to_symbol = fake_coin_to_symbol.__get__(bot, Passivbot)
    bot.is_pside_enabled = lambda pside: True
    bot.live_value = lambda key: bot.config["live"][key]
    bot._filter_approved_symbols = lambda pside, symbols: symbols

    bot.refresh_approved_ignored_coins_lists()

    assert bot.approved_coins["long"] == {"AAA/USDT:USDT"}
    assert bot.approved_coins["short"] == {"AAA/USDT:USDT", "BBB/USDT:USDT"}
    assert bot.approved_coins_minus_ignored_coins["short"] == {"AAA/USDT:USDT", "BBB/USDT:USDT"}


def test_refresh_approved_ignored_coin_lists_supports_migrated_global_all():
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "bitget"
    bot.eligible_symbols = {"AAA/USDT:USDT", "BBB/USDT:USDT"}
    bot.approved_coins = {"long": set(), "short": set()}
    bot.ignored_coins = {"long": set(), "short": set()}
    bot.approved_coins_minus_ignored_coins = {"long": set(), "short": set()}
    bot._disabled_psides_logged = set()
    bot._unsupported_coin_warnings = set()
    bot.config = {
        "_coins_sources": {
            "approved_coins": "all",
            "ignored_coins": {"long": [], "short": []},
        },
        "live": {},
    }

    def fake_coin_to_symbol(self, coin, verbose=True):
        mapping = {"AAA": "AAA/USDT:USDT", "BBB": "BBB/USDT:USDT"}
        return mapping.get(coin, f"{coin}/USDT:USDT")

    bot.coin_to_symbol = fake_coin_to_symbol.__get__(bot, Passivbot)
    bot.is_pside_enabled = lambda pside: True
    bot.live_value = lambda key: bot.config["live"][key]
    bot._filter_approved_symbols = lambda pside, symbols: symbols

    bot.refresh_approved_ignored_coins_lists()

    assert bot.approved_coins["long"] == {"AAA/USDT:USDT", "BBB/USDT:USDT"}
    assert bot.approved_coins["short"] == {"AAA/USDT:USDT", "BBB/USDT:USDT"}
