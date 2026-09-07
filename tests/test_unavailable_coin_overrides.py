"""Offline live-universe regressions; no exchange sessions or account requests."""

from copy import deepcopy
import logging

import pytest

from passivbot import Passivbot
from utils import AmbiguousMarketIdentifier, UnknownMarketIdentifier


def make_bot(overrides):
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "bitget"
    bot.config = {
        "coin_overrides": overrides,
        "live": {"forced_mode_long": "", "forced_mode_short": ""},
    }
    bot.markets_dict = {"AAA/USDT:USDT": {"active": True}}
    bot.positions = {}
    bot.open_orders = {}
    bot.approved_coins_minus_ignored_coins = {"long": {"AAA/USDT:USDT"}, "short": set()}
    bot._equity_hard_stop_enabled = lambda _pside: False
    bot.coin_to_symbol = lambda coin, verbose=True: f"{coin}/USDT:USDT"
    return bot


def test_unavailable_overrides_do_not_reach_startup_mode_lookup(caplog):
    bot = make_bot({"AAA": {}, "UNLISTED": {"live": {"forced_mode_long": "normal"}}})
    original = deepcopy(bot.config)
    with caplog.at_level(logging.INFO):
        bot.init_coin_overrides()
        bot.init_coin_overrides()
    assert bot.coin_overrides == {"AAA/USDT:USDT": {}}
    assert bot.get_symbols_approved_or_has_pos() == {"AAA/USDT:USDT"}
    assert bot.config == original
    assert caplog.text.count("skipping unavailable coin_overrides") == 1
    assert "UNLISTED" in caplog.text
    assert "exchange=bitget" in caplog.text


def test_unknown_exact_identifier_is_skipped_but_ambiguity_propagates():
    bot = make_bot({"AAA": {}, "UNLISTED/USDT:USDT": {}})

    def resolve(coin, verbose=True):
        if coin == "AAA":
            return "AAA/USDT:USDT"
        raise UnknownMarketIdentifier("unavailable")

    bot.coin_to_symbol = resolve
    bot.init_coin_overrides()
    assert bot.coin_overrides == {"AAA/USDT:USDT": {}}
    prior = deepcopy(bot.coin_overrides)

    def ambiguous(coin, verbose=True):
        raise AmbiguousMarketIdentifier("ambiguous")

    bot.coin_to_symbol = ambiguous
    with pytest.raises(AmbiguousMarketIdentifier):
        bot.init_coin_overrides()
    assert bot.coin_overrides == prior


def test_market_refresh_reconsiders_original_overrides(caplog):
    bot = make_bot({"AAA": {}, "UNLISTED": {"live": {"forced_mode_long": "manual"}}})
    with caplog.at_level(logging.INFO):
        bot.init_coin_overrides()
        bot.markets_dict["UNLISTED/USDT:USDT"] = {"active": True}
        bot.init_coin_overrides()
    assert bot.get_forced_PB_mode("long", "UNLISTED/USDT:USDT") == "manual"
    assert "all coin_overrides available" in caplog.text


def test_inactive_held_market_retains_override_and_protection():
    override = {"bot": {"long": {"strategy": {"custom": 0.123}}}}
    bot = make_bot({"INACTIVE": override})
    symbol = "INACTIVE/USDT:USDT"
    bot.markets_dict[symbol] = {"active": False}
    bot.positions[symbol] = {"long": {"size": 1}, "short": {"size": 0}}
    bot.init_coin_overrides()
    assert bot.coin_overrides[symbol] == override
    assert bot.get_forced_PB_mode("long", symbol) == "tp_only"
    assert symbol in bot.get_symbols_approved_or_has_pos()
    bot._assert_supported_live_state()


@pytest.mark.parametrize("surface", ["positions", "open_orders"])
def test_unavailable_override_never_hides_existing_exchange_state(surface):
    bot = make_bot({"UNLISTED": {}})
    symbol = "UNLISTED/USDT:USDT"
    getattr(bot, surface)[symbol] = (
        {"long": {"size": 1}, "short": {"size": 0}}
        if surface == "positions"
        else [{"id": "test"}]
    )
    original = deepcopy(getattr(bot, surface))
    bot.init_coin_overrides()
    with pytest.raises(KeyError, match="UNLISTED"):
        bot._assert_supported_live_state()
    assert getattr(bot, surface) == original


def test_all_overrides_unavailable_is_valid_empty_runtime_map():
    bot = make_bot({"UNLISTED": {}})
    bot.approved_coins_minus_ignored_coins = {"long": set(), "short": set()}
    bot.init_coin_overrides()
    assert bot.coin_overrides == {}
    assert bot.get_symbols_approved_or_has_pos() == set()


def test_skip_notice_is_bounded_and_omits_invalid_identifier_text(caplog):
    bot = make_bot(
        {"0api_key=SECRET\nINJECTED": {}, **{f"ZZ{i:03d}": {} for i in range(50)}}
    )
    with caplog.at_level(logging.INFO):
        bot.init_coin_overrides()
    notice = next(
        r.message for r in caplog.records if "skipping unavailable" in r.message
    )
    assert len(notice) < 240
    assert "SECRET" not in notice
    assert "INJECTED" not in notice
