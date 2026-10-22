"""Scenario source declarations remain separate from active runtime shapes."""
from copy import deepcopy

import pytest

from config import prepare_config
from config.bot_inheritance import bot_policy_source_view
from config.schema import get_expanded_config_prototype
from suite_runner import apply_scenario_overrides, build_scenarios


@pytest.mark.parametrize("kind,inactive", [("trailing_martingale", "ema_anchor"),
                                          ("ema_anchor", "trailing_martingale")])
def test_scenario_registered_inactive_source_survives_without_runtime_hydration(kind, inactive):
    source = {"config_version": "v9.0.0", "bot": {"portfolio": {"strategy_kind": kind}}}
    config = prepare_config(source, verbose=False)
    replacement = deepcopy(get_expanded_config_prototype()["bot"])
    replacement["portfolio"]["strategy_kind"] = kind
    replacement["short"]["universe"]["minimum_coin_age_days"] = 7.25
    scenarios, _ = build_scenarios({"scenarios": [{"label": "whole", "overrides": {"bot": replacement}}]}, ["binance"])
    apply_scenario_overrides(config, scenarios[0].overrides)
    assert config["bot"]["short"]["universe"]["minimum_coin_age_days"] == 7.25
    assert set(config["bot"]["long"]["strategy"]) == {kind}
    assert bot_policy_source_view(config)["bot"]["long"]["strategy"][inactive] == replacement["long"]["strategy"][inactive]
    with pytest.raises(ValueError, match="unknown"):
        apply_scenario_overrides(config, {f"bot.long.strategy.{inactive}.invented": 1.0})
    with pytest.raises(ValueError, match="missing required keys"):
        apply_scenario_overrides(config, {"bot.long.forager.score_weights": {"volume": 1.0}})


def test_untyped_and_nonbot_scenario_paths_still_require_existing_fields():
    for config, path in [({"bot": {"long": {"strategy": {}}}}, "bot.long.strategy.ema_anchor.base_qty_fraction"),
                         (prepare_config({"config_version": "v9.0.0"}, verbose=False), "backtest.invented")]:
        with pytest.raises(KeyError, match="Unknown override path"):
            apply_scenario_overrides(config, {path: 1.0})


@pytest.mark.parametrize("owner", ["scenario", "facade"])
@pytest.mark.parametrize("path,value", [
    ("bot.portfolio.invented", 1.0),
    ("bot.portfolio.execution.invented", 1.0),
    ("bot.portfolio.execution", {"invented": 1.0}),
    ("bot.portfolio", {"execution": {"invented": 1.0}}),
    ("bot", {"portfolio": {"execution": {"invented": 1.0}}}),
])
def test_typed_scenario_and_facade_reject_unknown_portfolio_leaves_and_objects(owner, path, value):
    from config.bot_inheritance import apply_bot_config_patch
    config = prepare_config({"config_version": "v9.0.0"}, verbose=False)
    before = bot_policy_source_view(config)
    apply = apply_scenario_overrides if owner == "scenario" else apply_bot_config_patch
    with pytest.raises((ValueError, KeyError), match="unknown|Unknown"):
        apply(config, {path: value})
    assert bot_policy_source_view(config) == before
