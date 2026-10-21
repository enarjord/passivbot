"""Released v8 time values move directly to final public unit names."""

from copy import deepcopy
import json

import pytest

from config import prepare_config
from config.migrations.released_v8 import get_released_v8_defaults
from config.optimize_bounds import flatten_optimize_bounds
from config.param_paths import resolve_bound_selectors, resolve_optimizer_key_path
from config.shared_bot import get_grouped_bot_value
from tools.migrate_config_v9 import migrate


def released_source(kind):
    source = get_released_v8_defaults(strategy_kind=kind)
    source["backtest"]["visible_metrics"] = [name for name in source["backtest"]["visible_metrics"]
                                             if not name.endswith("_w")]
    source["optimize"]["scoring"] = [item for item in source["optimize"]["scoring"]
                                      if not item["metric"].endswith("_w")]
    return source


def read_path(config, path):
    for part in path:
        config = config[part]
    return config


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_released_root_bounds_fixed_scenario_and_millisecond_values_keep_their_units(kind):
    source = released_source(kind)
    strategy = source["bot"]["long"]["strategy"][kind]
    (strategy["entry"] if kind == "trailing_martingale" else strategy)["ema_span_0"] = 123.75
    source["bot"]["long"]["forager"]["volume_ema_span_1m"] = 456.25
    source["bot"]["long"]["hsl"]["cooldown_minutes_after_red"] = 0.125
    source["live"].update(recv_window_ms=2345, fee_conversion_max_age_ms=987654,
                          startup_phase_budgets={"account": {"elapsed_ms": 3456, "since_previous_ms": 789}})
    source["monitor"]["price_tick_min_interval_ms"] = 654
    source["optimize"]["bounds"]["long"]["forager"]["volume_ema_span_1m"] = [400.5, 500.75]
    old_selector = f"bot.long.strategy.{kind}." + ("entry." if kind == "trailing_martingale" else "") + "ema_span_0"
    source["optimize"]["fixed_runtime_overrides"] = {old_selector: 234.5}
    source["backtest"]["scenarios"] = [{"label": "span", "overrides": {old_selector: 345.25}}]
    original = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5)
    config = prepare_config(result.config, verbose=False)
    target = resolve_optimizer_key_path(config, "long_ema_span_0")
    selector = ".".join(target)
    assert read_path(config, target) == 123.75
    assert config["optimize"]["fixed_runtime_overrides"] == {selector: 234.5}
    assert config["backtest"]["scenarios"][0]["overrides"][selector] == 345.25
    assert config["optimize"]["bounds"]["long"]["forager"]["volume_ema_span_minutes"] == [400.5, 500.75]
    assert get_grouped_bot_value(config["bot"]["long"], "forager_volume_ema_span_1m") == 456.25
    assert get_grouped_bot_value(config["bot"]["long"], "hsl_cooldown_minutes_after_red") == 0.125
    assert config["live"]["api"]["recv_window_milliseconds"] == 2345
    assert config["live"]["fills"]["fee_conversion_max_age_milliseconds"] == 987654
    assert config["live"]["startup"]["startup_phase_budgets"] == {
        "account": {"elapsed_milliseconds": 3456, "since_previous_milliseconds": 789},
    }
    assert config["monitor"]["price_tick_min_interval_milliseconds"] == 654
    for old in ("ema_span_0", "volume_ema_span_1m", "cooldown_minutes_after_red",
                "recv_window_ms", "fee_conversion_max_age_ms", "elapsed_ms", "since_previous_ms",
                "price_tick_min_interval_ms"):
        assert any(item["path"].endswith(old) for item in result.changes), old
    assert source == original


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("location", ["root", "scenario"])
def test_released_coin_file_and_inline_span_names_convert_before_their_merge(tmp_path, kind, location):
    source = released_source(kind)
    file_patch = {"bot": {"long": {"strategy": {kind: {"ema_span_0": 111.5}}}}}
    coin_file = tmp_path / "coin.json"
    coin_file.write_text(json.dumps(file_patch), encoding="utf-8")
    original_file = coin_file.read_bytes()
    patch = {"override_config_path": "coin.json",
             "bot": {"long": {"strategy": {kind: {"ema_span_0": 222.5}}}}}
    if location == "root":
        source["coin_overrides"] = {"TEST": patch}
    else:
        source["backtest"]["scenarios"] = [{"label": "span", "overrides": {"coin_overrides": {"TEST": patch}}}]
    original = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5, base_config_path=str(tmp_path / "source.json"))
    config = result.config
    if location == "scenario":
        from tools.migrate_config_v9 import _set_patch

        config = deepcopy(config)
        _set_patch(config, config["backtest"]["scenarios"][0]["overrides"], scenario=True)
    config = prepare_config(config, base_config_path=str(tmp_path / "relocated.json"), verbose=False)
    target = resolve_optimizer_key_path(config, "long_ema_span_0")
    assert read_path(config["coin_overrides"]["TEST"], target) == 222.5
    assert "override_config_path" not in config["coin_overrides"]["TEST"]
    assert any("override_config_path::" in item["path"] and item["path"].endswith("ema_span_0")
               for item in result.changes)
    assert source == original
    assert coin_file.read_bytes() == original_file


@pytest.mark.parametrize("prefix", ["", "forager.", "long.forager.", "bot.long.forager."])
def test_released_span_selector_suffixes_preserve_native_gene_membership(prefix):
    source = released_source("trailing_martingale")
    source["optimize"]["fixed_params"] = [f"{prefix}volume_ema_span_1m"]
    result = migrate(source, memory_span_minutes=123.5)
    config = result.config
    flat = flatten_optimize_bounds(config["optimize"]["bounds"], strategy_kind="trailing_martingale")
    selected = resolve_bound_selectors(config, config["optimize"]["fixed_params"], flat)
    sides = ["long"] if "long" in prefix else ["long", "short"]
    assert set(selected) == {f"{side}_forager_volume_ema_span_1m" for side in sides}
    assert config["optimize"]["fixed_params"] == [f"{prefix}volume_ema_span_minutes"]
    assert any(item["path"].startswith("optimize.fixed_params") for item in result.changes)
