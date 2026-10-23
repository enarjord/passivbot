"""Command ownership and explicit directional nulls survive lean discovery."""
import argparse
from copy import deepcopy

import pytest

from config import prepare_config
from config.bot_inheritance import bot_policy_source_view
from config.schema import get_template_config
from config_utils import add_config_arguments, update_config_with_args


def parser_for(command):
    parser = argparse.ArgumentParser()
    keys = add_config_arguments(parser, get_template_config(), command=command)
    return parser, keys


def test_optimizer_discovery_retains_global_bounds_without_numeric_runtime_flags():
    parser, keys = parser_for("optimize")
    runtime = "bot.global.entry_cooldown.base_duration_minutes"
    domain = "optimize.bounds.global.entry_cooldown.base_duration_minutes"
    assert runtime not in keys
    assert domain in keys
    assert "bot.global.enabled" not in keys
    assert "bot.long.enabled" in keys
    assert "bot.short.enabled" in keys
    with pytest.raises(SystemExit):
        parser.parse_args(["--bot.global.enabled", "true"])
    with pytest.raises(SystemExit):
        parser.parse_args([f"--{runtime}", "5.0"])
    assert vars(parser.parse_args([f"--{domain}", "0,5"]))[domain] == [0, 5]


@pytest.mark.parametrize("command", ["live", "backtest", "optimize"])
def test_directional_approved_null_inherits_and_ignored_null_clears(command):
    parser, keys = parser_for(command)
    raw = get_template_config()
    raw["bot"]["global"]["universe"].update(approved_coins=["BTC", "ETH"], ignored_coins=["BTC"])
    prepared = prepare_config(raw, verbose=False)
    args = parser.parse_args(["--symbols", '{long:null,short:["ETH"]}',
                              "--ignored-coins", '{long:null,short:[]}'])
    update_config_with_args(prepared, args, allowed_keys=keys)
    source = bot_policy_source_view(prepared)
    assert source["bot"]["long"]["universe"]["approved_coins"] is None
    assert source["bot"]["short"]["universe"]["approved_coins"] == ["ETH"]
    loaded = prepare_config(prepared, verbose=False)
    assert loaded["bot"]["long"]["universe"]["approved_coins"] == ["BTC", "ETH"]
    assert loaded["bot"]["long"]["universe"]["ignored_coins"] == []
    assert loaded["bot"]["short"]["universe"]["ignored_coins"] == []
    before = deepcopy(source)
    with pytest.raises(ValueError, match="explicit side"):
        update_config_with_args(prepared, parser.parse_args(["--symbols", "null"]), allowed_keys=keys)
    assert bot_policy_source_view(prepared) == before
