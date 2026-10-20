"""Exposure controls share one object without changing native parameter values."""

from copy import deepcopy

import pytest

from config import prepare_config, get_template_config
from config.optimize_bounds import flatten_optimize_bounds
from config.param_paths import resolve_optimizer_key_path
from config.shared_bot import flatten_shared_bot_side
from released_v8_fixtures import released_v8_policy
from tools.migrate_config_v9 import migrate


@pytest.mark.parametrize("side", ["long", "short"])
def test_exposure_enforcer_bounds_mirror_the_bot_feature_objects(side):
    config = prepare_config(get_template_config(), verbose=False)
    flat = flatten_optimize_bounds(config["optimize"]["bounds"], strategy_kind="trailing_martingale")
    for owner, native in (("position", "wel"), ("total", "twel")):
        key = f"{side}_risk_{native}_enforcer_threshold"
        assert resolve_optimizer_key_path(config, key) == (
            "bot", side, "risk", f"{owner}_exposure_enforcer", "threshold",
        )
        assert flat[key] == config["optimize"]["bounds"][side]["risk"][f"{owner}_exposure_enforcer"]["threshold"]


@pytest.mark.parametrize("field", [
    "position_exposure_enforcer_enabled", "position_exposure_enforcer_threshold",
    "total_exposure_enforcer_enabled", "total_exposure_enforcer_threshold",
    "total_exposure_enforcer_policy",
])
def test_old_development_enforcer_leaves_are_not_loading_aliases(field):
    config = get_template_config()
    config["bot"]["long"]["risk"][field] = False
    with pytest.raises(ValueError, match="unknown configuration field"):
        prepare_config(config, verbose=False)


@pytest.mark.parametrize("structured", [False, True])
def test_released_v8_enforcers_and_override_objects_migrate_explicitly(structured):
    source = released_v8_policy()
    old_risk = deepcopy(source["bot"]["long"]["risk"])
    old_risk["position_exposure_enforcer_threshold"] = 0.73
    overrides = (
        {"bot.long.risk": old_risk} if structured else
        {"bot.long.risk.position_exposure_enforcer_threshold": 0.73}
    )
    source["backtest"]["scenarios"] = [{"label": "risk", "overrides": overrides}]
    original = deepcopy(source)
    migration = migrate(source, memory_span_minutes=123.5)
    assert source == original
    for side in ("long", "short"):
        native = flatten_shared_bot_side(migration.config["bot"][side])
        released = original["bot"][side]["risk"]
        for owner, prefix in (("position", "wel"), ("total", "twel")):
            for leaf in ("enabled", "threshold"):
                assert native[f"risk_{prefix}_enforcer_{leaf}"] == released[f"{owner}_exposure_enforcer_{leaf}"]
        assert native["risk_twel_enforcer_policy"] == released["total_exposure_enforcer_policy"]
    assert any("exposure-enforcer object conversion" in item["reason"] for item in migration.changes)
    assert migration.config["backtest"]["scenarios"][0]["overrides"]


@pytest.mark.parametrize("owner,native", [("position", "wel"), ("total", "twel")])
@pytest.mark.parametrize("prefix", ["", "risk.", "long.risk.", "bot.long.risk.", "short.risk."])
def test_released_fixed_selector_suffixes_keep_their_gene_authority(owner, native, prefix):
    from config.param_paths import resolve_bound_selectors

    source = released_v8_policy()
    selector = f"{prefix}{owner}_exposure_enforcer_threshold"
    source["optimize"]["fixed_params"] = [selector]
    migrated = migrate(source, memory_span_minutes=123.5)
    config = migrated.config
    flat = flatten_optimize_bounds(config["optimize"]["bounds"], strategy_kind="trailing_martingale")
    selected = resolve_bound_selectors(config, config["optimize"]["fixed_params"], flat)
    sides = {"long"} if "long" in prefix else {"short"} if "short" in prefix else {"long", "short"}
    assert set(selected) == {f"{side}_risk_{native}_enforcer_threshold" for side in sides}
    assert config["optimize"]["fixed_params"] == [f"{prefix}{owner}_exposure_enforcer.threshold"]
    assert any(item["path"] == "optimize.fixed_params" and item["old"] == [selector]
               and item["new"] == config["optimize"]["fixed_params"] for item in migrated.changes)
