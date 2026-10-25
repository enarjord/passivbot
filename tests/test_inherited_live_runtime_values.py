"""Owned live allocation values never become authored bot policy."""
from copy import deepcopy

import pytest

from config import prepare_config
from config.bot_inheritance import bot_policy_source_view, get_bot_policy
from config.runtime_compile import compile_runtime_config
from config.universe import get_side_universe_sources
from passivbot import Passivbot


def bot_fixture(*, compiled=False):
    config = prepare_config({"config_version": "v9.0.0", "bot": {
        "long": {"enabled": True}, "global": { "risk": {
            "total_wallet_exposure_limit_ratio": 2.0, "max_positions": 4},
            "universe": {"approved_coins": ["AAA"], "minimum_coin_age_days": 0.0}},
        "short": {"enabled": True, "risk": {"total_wallet_exposure_limit_ratio": 1.0, "max_positions": 2}},
    }}, verbose=False)
    bot = Passivbot.__new__(Passivbot)
    bot.config = compile_runtime_config(config, runtime="live") if compiled else config
    bot.coin_overrides = {"AAA": {"bot": {
        "long": {"wallet_exposure_limit_ratio": 0.2}, "short": {"wallet_exposure_limit_ratio": 0.0}}}}
    return bot


@pytest.mark.parametrize("compiled", [False, True])
def test_actual_allocation_writer_retains_source_and_owned_aliases(compiled):
    bot = bot_fixture(compiled=compiled)
    source = bot_policy_source_view(bot.config)
    policy = get_bot_policy(bot.config)
    aliases = bot.config["bot"]._runtime_aliases
    overrides = deepcopy(bot.coin_overrides)
    for _ in range(2):
        bot.set_wallet_exposure_limit_ratios()
        assert bot.config["bot"]["long"]["wallet_exposure_limit_ratio"] == 0.5
        assert bot.config["bot"]["short"]["wallet_exposure_limit_ratio"] == 0.5
        assert bot.get_wallet_exposure_limit_ratio("long", "AAA") == 0.2
        assert bot.get_wallet_exposure_limit_ratio("short", "AAA") == 0.0
        assert bot.coin_overrides == overrides
        assert bot_policy_source_view(bot.config) == source
        assert get_bot_policy(bot.config) == policy
        assert aliases <= bot.config["bot"]._runtime_aliases
        assert get_side_universe_sources(bot.config, "approved_coins") == {
            "long": ["AAA"], "short": ["AAA"]}
    bot.config["bot"]["long"]["wallet_exposure_limit_ratio"] = 9.0
    with pytest.raises(ValueError, match="aliases cannot grant authored policy"):
        bot_policy_source_view(bot.config)


def test_actual_allocation_writer_adopts_prior_direct_policy_edits():
    bot = bot_fixture()
    bot.config["bot"]["long"]["risk"]["total_wallet_exposure_limit_ratio"] = 3.0
    bot.set_wallet_exposure_limit_ratios()
    assert bot.config["bot"]["long"]["wallet_exposure_limit_ratio"] == 0.75
    source = bot_policy_source_view(bot.config)
    assert source["bot"]["long"]["risk"]["total_wallet_exposure_limit_ratio"] == 3.0
    assert "wallet_exposure_limit_ratio" not in source["bot"]["long"]


def test_runtime_allocation_still_cannot_be_authored_at_root():
    with pytest.raises(ValueError, match="wallet_exposure_limit_ratio.*unknown"):
        prepare_config({"config_version": "v9.0.0", "bot": {
            "long": {"wallet_exposure_limit_ratio": 0.5}}}, verbose=False)
