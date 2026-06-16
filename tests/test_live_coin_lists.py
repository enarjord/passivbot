import logging

import pytest

from passivbot import Passivbot


def _make_mode_override_bot():
    ignored_symbol = "DOGE/USDT:USDT"
    approved_symbol = "BTC/USDT:USDT"
    bot = Passivbot.__new__(Passivbot)
    bot.positions = {
        ignored_symbol: {
            "long": {"size": 0.0, "price": 0.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    bot.open_orders = {}
    bot.coin_overrides = {}
    bot.approved_coins_minus_ignored_coins = {
        "long": {approved_symbol},
        "short": set(),
    }
    bot.ignored_coins = {"long": {ignored_symbol}, "short": set()}
    bot.markets_dict = {
        ignored_symbol: {"active": True},
        approved_symbol: {"active": True},
    }
    bot.ineligible_symbols = {}
    bot._runtime_forced_modes = {"long": {}, "short": {}}
    bot._equity_hard_stop_enabled = lambda pside: False
    bot.config_get = lambda path, symbol=None: ""
    bot.is_old_enough = lambda pside, symbol: True
    return bot, ignored_symbol, approved_symbol


def test_ignored_coin_retained_flat_position_gets_graceful_stop_override():
    bot, ignored_symbol, approved_symbol = _make_mode_override_bot()

    universe = bot._build_live_symbol_universe()
    assert ignored_symbol in universe
    assert ignored_symbol not in bot.approved_coins_minus_ignored_coins["long"]

    overrides = bot._build_orchestrator_mode_overrides(universe)

    assert overrides["long"][ignored_symbol] == "graceful_stop"
    assert overrides["short"][ignored_symbol] is None
    assert overrides["long"][approved_symbol] is None


@pytest.mark.parametrize("forced_mode", ["manual", "panic", "tp_only"])
def test_ignored_coin_preserves_stricter_forced_modes(forced_mode):
    bot, ignored_symbol, _approved_symbol = _make_mode_override_bot()
    bot.config_get = (
        lambda path, symbol=None: forced_mode
        if path == ["live", "forced_mode_long"] and symbol == ignored_symbol
        else ""
    )

    assert bot._orchestrator_mode_override("long", ignored_symbol) == forced_mode


def test_ignored_coin_overrides_forced_normal_to_graceful_stop():
    bot, ignored_symbol, _approved_symbol = _make_mode_override_bot()
    bot.config_get = (
        lambda path, symbol=None: "normal"
        if path == ["live", "forced_mode_long"] and symbol == ignored_symbol
        else ""
    )

    assert bot._orchestrator_mode_override("long", ignored_symbol) == "graceful_stop"


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
