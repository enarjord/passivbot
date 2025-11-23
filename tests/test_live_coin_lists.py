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

    with caplog.at_level(logging.WARNING):
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
        rec.message
        for rec in caplog.records
        if rec.levelno >= logging.WARNING and "Skipping unsupported markets" in rec.message
    ]
    assert len(warnings) == 1
