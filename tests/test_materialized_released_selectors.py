"""Released selectors retain leaf/object authority after coin-file conversion."""

from copy import deepcopy
import json

import pytest

from config import prepare_config
from config.migrations.final_v9 import released_runtime_selector
from config.overrides import parse_overrides
from config.shared_bot import flatten_shared_bot_side
from optimization.warmup import _apply_config_overrides
from released_v8_fixtures import released_v8_policy
from suite_runner import apply_scenario_overrides
from tools.migrate_config_v9 import migrate


@pytest.mark.parametrize("owner,native", [
    ("position_exposure_enforcer", "risk_wel_enforcer_threshold"),
    ("total_exposure_enforcer", "risk_twel_enforcer_threshold"),
])
@pytest.mark.parametrize("shape", ["flat", "object"])
@pytest.mark.parametrize("layer", ["fixed", "scenario"])
def test_materialized_enforcer_override_keeps_policy_value_and_scope(
    tmp_path, owner, native, shape, layer,
):
    source = released_v8_policy()
    file = tmp_path / "coin.json"
    file.write_text(json.dumps({"bot": {"long": {"risk": {
        f"{owner}_threshold": 0.91,
    }}}}))
    source["coin_overrides"] = {
        "BTC": {"override_config_path": file.name},
        "ETH": {"live": {"forced_mode_long": "manual"}},
    }
    selector = f"coin_overrides.BTC.bot.long.risk.{owner}"
    if shape == "flat":
        selector += "_threshold"
        value = 0.73
    else:
        value = {"threshold": 0.73}
    if layer == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = {selector: value}
    else:
        source["optimize"]["fixed_runtime_overrides"] = {
            "coin_overrides.ETH.live.forced_mode_long": "panic",
        }
        source["backtest"]["scenarios"] = [{
            "label": "one", "overrides": {selector: value},
        }]
    original, file_bytes = deepcopy(source), file.read_bytes()
    # Portfolio TWEL controls remain forbidden in coin patches, and fixed
    # overrides still target leaves only. Conversion must reach these owning
    # policy checks rather than fail to locate an already converted file leaf.
    if owner == "total_exposure_enforcer" or (layer == "fixed" and shape == "object"):
        error, reason = ((ValueError, "not overridable")
                         if owner == "total_exposure_enforcer"
                         else (TypeError, "must target leaf settings"))
        with pytest.raises(error, match=reason):
            migrate(source, memory_span_minutes=123.5,
                    base_config_path=str(tmp_path / "input.json"))
        assert source == original
        assert file.read_bytes() == file_bytes
        return
    migration = migrate(
        source, memory_span_minutes=123.5,
        base_config_path=str(tmp_path / "input.json"),
    )
    config = deepcopy(migration.config)
    fixed = config["optimize"]["fixed_runtime_overrides"]
    _apply_config_overrides(config, fixed)
    if layer == "fixed":
        patch = fixed
    else:
        patch = config["backtest"]["scenarios"][0]["overrides"]
        apply_scenario_overrides(config, patch)
    # A scalar must remain a scalar; an object replaces precisely that object.
    # Neither may replace the complete coin map and erase unrelated authority.
    assert len(patch) == 1
    target, converted_value = next(iter(patch.items()))
    assert target.startswith(f"coin_overrides.BTC.bot.long.risk.{owner}")
    assert isinstance(converted_value, dict) == (shape == "object")
    config = parse_overrides(prepare_config(config, verbose=False), verbose=False)
    assert flatten_shared_bot_side(config["coin_overrides"]["BTC"]["bot"]["long"])[native] == 0.73
    assert config["coin_overrides"]["ETH"]["live"]["modes"]["long"] == (
        "panic" if layer == "scenario" else "manual"
    )
    assert source == original
    assert file.read_bytes() == file_bytes


@pytest.mark.parametrize("old,new", [
    ("bot.long.risk.n_positions", "bot.long.risk.max_positions"),
    ("bot.long.hsl.red_threshold", "bot.long.hsl.drawdown_ratio_threshold"),
    ("bot.short.unstuck.threshold", "bot.short.unstuck.exposure_trigger_ratio"),
    ("bot.short.unstuck.ema_dist", "bot.short.unstuck.ema_distance_fraction"),
])
def test_released_runtime_path_conversion_preserves_coin_prefix(old, new):
    assert released_runtime_selector(old) == new
    assert released_runtime_selector(f"coin_overrides.BTC.{old}") == f"coin_overrides.BTC.{new}"
