"""Native live parameter lookups read canonical policy without inventing aliases."""
from copy import deepcopy

import pytest

from config import prepare_config
from config.runtime_compile import compile_runtime_config
from passivbot import Passivbot


@pytest.fixture(params=["canonical", "compiled"])
def bot(request):
    config = prepare_config({"config_version": "v9.0.0", "bot": {"global": {
        "entry_cooldown": {"base_duration_minutes": 37.0, "max_duration_minutes": 60.0,
            "weights_minutes": {"exposure_ratio": 2.0, "adverse_directionality": 4.0}},
    }}}, verbose=False)
    if request.param == "compiled":
        config = compile_runtime_config(config, runtime="live", record_step=False)
    value = object.__new__(Passivbot)
    value.config = config
    value.coin_overrides = {}
    return value


@pytest.mark.parametrize("side", ["long", "short"])
def test_native_shared_lookup_uses_canonical_policy_without_mutation(bot, side):
    original = deepcopy(bot.config)
    assert bot.config_get(["bot", side, "risk_entry_cooldown_minutes"]) == 37.0
    assert bot.bp(side, "entry_cooldown_weights_minutes") == {
        "exposure_ratio": 2.0, "adverse_directionality": 4.0,
    }
    assert bot.config == original


@pytest.mark.parametrize("side", ["long", "short"])
def test_coin_patch_precedence_and_partial_weight_merge_remain_intact(bot, side):
    original = deepcopy(bot.config)
    bot.coin_overrides["AAA"] = {"bot": {side: {"entry_cooldown": {
        "base_duration_minutes": 0.0, "weights_minutes": {"exposure_ratio": 8.0},
    }}}}
    assert bot.bp(side, "risk_entry_cooldown_minutes", "AAA") == 0.0
    assert bot.bp(side, "entry_cooldown_weights_minutes", "AAA") == {
        "exposure_ratio": 8.0, "adverse_directionality": 4.0,
    }
    assert bot.bp(side, "risk_entry_cooldown_minutes", "BBB") == 37.0
    assert bot.config == original


@pytest.mark.parametrize("side", ["long", "short"])
def test_unknown_or_missing_required_lookup_remains_fatal(bot, side):
    with pytest.raises(KeyError):
        bot.bp(side, "not_a_policy_field", "AAA")
    del bot.config["bot"][side]["entry_cooldown"]["base_duration_minutes"]
    bot.config["bot"][side].pop("risk_entry_cooldown_minutes", None)
    with pytest.raises(KeyError):
        bot.bp(side, "risk_entry_cooldown_minutes")
