"""Named control policies retain their existing offline consumer behavior."""

from argparse import ArgumentParser
from copy import deepcopy
import json
import re
import sys

import pytest

from config import prepare_config
from config.authoring import explain_config
from config.json_schema import generate_schema
from config.overrides import parse_overrides
from config.schema import get_template_config
from config_utils import add_arguments_recursively, project_template_config_for_cli, update_config_with_args
from passivbot import Passivbot
from released_v8_fixtures import released_v8_policy
from tools.migrate_config_v9 import migrate
from warmup_utils import compute_backtest_warmup_minutes, live_warmup_cap_minutes


def set_path(config, path, value):
    node = config
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def get_path(config, path):
    node = config
    for part in path.split("."):
        node = node[part]
    return node


def test_defaults_are_explicit_and_roundtrip():
    source = get_template_config()
    original = deepcopy(source)
    result = prepare_config(source, verbose=False)
    assert result["live"]["modes"]["long"] is None
    assert result["live"]["modes"]["short"] is None
    assert result["bot"]["global"]["warmup"]["max_minutes"] is None
    assert result["live"]["startup"]["warmup_concurrency"] == "auto"
    assert prepare_config(result, verbose=False)["live"] == result["live"]
    assert source == original


@pytest.mark.parametrize("path,value", [
    ("live.modes.long", ""), ("live.modes.short", " "),
    ("live.modes.long", False), ("live.modes.long", 0),
    ("live.modes.long", "null"), ("live.modes.long", "unknown"),
    *[("bot.global.warmup.max_minutes", value)
      for value in (0, 0.0, -1, True, False, float("nan"), float("inf"), "0", [])],
    *[("live.startup.warmup_concurrency", value)
      for value in (0, -1, 1.5, 1.0, True, False, None, float("nan"), "0", "3", "AUTO")],
])
@pytest.mark.parametrize("layer", ["root", "fixed", "scenario"])
def test_old_or_malformed_authored_controls_fail_before_normalization(path, value, layer):
    source = get_template_config()
    if layer == "root":
        set_path(source, path, value)
    elif layer == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = {path: value}
    else:
        source["backtest"]["scenarios"] = [{"label": "one", "overrides": {path: value}}]
    with pytest.raises((ValueError, TypeError), match=path):
        prepare_config(source, verbose=False)


def test_per_coin_null_clears_only_the_root_forced_mode():
    source = get_template_config()
    source["live"]["modes"]["long"] = "manual"
    source["coin_overrides"] = {"BTC": {"live": {"modes": {"long": None}}}}
    config = parse_overrides(prepare_config(source, verbose=False), verbose=False)
    bot = object.__new__(Passivbot)
    bot.config, bot.coin_overrides = config, config["coin_overrides"]
    bot.markets_dict = {"BTC": {"active": True}, "ETH": {"active": True}}
    assert bot.get_forced_PB_mode("long", "BTC") is None
    assert bot.get_forced_PB_mode("long", "ETH") == "manual"
    bot._runtime_forced_modes = {"long": {"BTC": "panic"}}
    assert bot.get_forced_PB_mode("long", "BTC") == "panic"


@pytest.mark.parametrize("layer", ["inline", "file"])
def test_coin_modes_accept_null_and_reject_old_empty_strings(tmp_path, layer):
    source = get_template_config()
    patch = {"live": {"modes": {"long": None}}}
    path = tmp_path / "coin.json"
    path.write_text(json.dumps(patch))
    source["coin_overrides"] = {"BTC": patch if layer == "inline" else {"override_config_path": path.name}}
    config = prepare_config(source, base_config_path=str(tmp_path / "input.json"), verbose=False)
    assert parse_overrides(config, verbose=False)["coin_overrides"]["BTC"] == patch
    patch["live"]["modes"]["long"] = ""
    path.write_text(json.dumps(patch))
    with pytest.raises((TypeError, ValueError), match="mode"):
        parse_overrides(prepare_config(source, base_config_path=str(tmp_path / "input.json"), verbose=False), verbose=False)


def test_fixed_and_scenario_null_clear_without_changing_authored_roots():
    source = get_template_config()
    source["live"]["modes"]["long"] = "manual"
    source["optimize"]["fixed_runtime_overrides"] = {"live.modes.long": None}
    source["backtest"]["scenarios"] = [{"label": "one", "overrides": {"live.modes.long": "panic"}}]
    assert explain_config(source, "live.modes.long", optimize=True)["value"] is None
    assert explain_config(source, "live.modes.long", optimize=True, scenario="one")["value"] == "panic"
    assert source["live"]["modes"]["long"] == "manual"


def test_optimizer_and_suite_apply_explicit_controls_before_actual_consumers():
    from optimization.warmup import _apply_config_overrides
    from suite_runner import apply_scenario_overrides

    source = get_template_config()
    source["live"]["modes"]["long"] = "manual"
    source["live"]["startup"]["warmup_concurrency"] = 3
    source["bot"]["global"]["warmup"]["max_minutes"] = 4.0
    source["optimize"]["fixed_runtime_overrides"] = {
        "live.modes.long": None, "live.startup.warmup_concurrency": "auto",
        "bot.global.warmup.max_minutes": None,
    }
    scenario = {"live.modes.long": "panic", "live.startup.warmup_concurrency": 2,
                "bot.global.warmup.max_minutes": 1.0}
    source["backtest"]["scenarios"] = [{"label": "one", "overrides": scenario}]
    original = deepcopy(source)
    config = parse_overrides(prepare_config(source, verbose=False), verbose=False)
    _apply_config_overrides(config, config["optimize"]["fixed_runtime_overrides"])
    bot = object.__new__(Passivbot)
    bot.config, bot.coin_overrides = config, config["coin_overrides"]
    bot.exchange, bot.max_n_concurrent_ohlcvs_1m_updates = "binance", 9
    bot.markets_dict = {"BTC": {"active": True}}
    assert bot.get_forced_PB_mode("long", "BTC") is None
    assert bot._candle_fetch_concurrency() == 9
    assert compute_backtest_warmup_minutes(config, include_rms=False) > 4
    apply_scenario_overrides(config, scenario)
    assert bot.get_forced_PB_mode("long", "BTC") == "panic"
    assert bot._candle_fetch_concurrency() == 2
    assert compute_backtest_warmup_minutes(config, include_rms=False) == 1
    assert source == original


def test_uncapped_and_positive_warmup_round_up_at_acquisition():
    source = get_template_config()
    uncapped = compute_backtest_warmup_minutes(source, include_rms=False)
    assert uncapped > 1
    assert live_warmup_cap_minutes(source) == 0
    source["bot"]["global"]["warmup"]["max_minutes"] = 1.0
    assert compute_backtest_warmup_minutes(source, include_rms=False) == 1
    source["bot"]["global"]["warmup"]["max_minutes"] = 5.5
    assert live_warmup_cap_minutes(source) == 6
    source["bot"]["global"]["warmup"]["max_minutes"] = 0.5
    assert live_warmup_cap_minutes(source) == 1
    assert compute_backtest_warmup_minutes(source, include_rms=False) == 1
    source["bot"]["global"]["warmup"]["max_minutes"] = 0
    with pytest.raises(ValueError, match="positive finite"):
        compute_backtest_warmup_minutes(source)


@pytest.mark.parametrize("exchange,context,expected", [
    ("binance", "runtime", 9), ("binance", "history_replay", 2),
    ("binance", "background warmup", 1), ("hyperliquid", "runtime", 1),
    ("kucoin", "history_replay", 1), ("gateio", "history_replay", 1),
])
def test_auto_concurrency_retains_context_and_exchange_policy(exchange, context, expected):
    bot = object.__new__(Passivbot)
    bot.config, bot.exchange = get_template_config(), exchange
    bot.max_n_concurrent_ohlcvs_1m_updates = 9
    assert bot._candle_fetch_concurrency(context=context) == expected
    bot.config["live"]["startup"]["warmup_concurrency"] = 7
    assert bot._candle_fetch_concurrency(context=context) == 7
    bot.config["live"]["startup"]["warmup_concurrency"] = 0
    with pytest.raises(ValueError, match="positive integer"):
        bot._candle_fetch_concurrency(context=context)


def test_cli_distinguishes_omission_from_null_and_keeps_nullable_projection():
    source = get_template_config()
    source["live"]["modes"]["long"] = "manual"
    source["bot"]["global"]["warmup"]["max_minutes"] = 500.0
    parser = ArgumentParser()
    add_arguments_recursively(parser, source)
    update_config_with_args(source, parser.parse_args([]))
    assert source["live"]["modes"]["long"] == "manual"
    args = parser.parse_args(["--live.modes.long", "null", "--bot.global.warmup.max_minutes", "null",
                              "--live.startup.warmup_concurrency", "3"])
    update_config_with_args(source, args)
    assert source["live"]["modes"]["long"] is None
    assert source["bot"]["global"]["warmup"]["max_minutes"] is None
    assert source["live"]["startup"]["warmup_concurrency"] == 3
    assert "max_minutes" in project_template_config_for_cli(source, "optimize")["bot"]["global"]["warmup"]


@pytest.mark.parametrize("flag,value", [("--live.modes.long", ""),
    ("--bot.global.warmup.max_minutes", "0"), ("--bot.global.warmup.max_minutes", "nan"),
    ("--live.startup.warmup_concurrency", "0"), ("--live.startup.warmup_concurrency", "1.5")])
def test_cli_rejects_retired_or_malformed_sentinels(flag, value):
    parser = ArgumentParser()
    add_arguments_recursively(parser, get_template_config())
    with pytest.raises(SystemExit) as exc:
        parser.parse_args([flag, value])
    assert exc.value.code == 2


def test_generated_schema_models_null_and_numeric_auto_domains():
    import jsonschema

    source = get_template_config()
    validator = jsonschema.Draft202012Validator(generate_schema())
    assert not list(validator.iter_errors(source))
    for path, valid, invalid in (
        ("live.modes.long", [None, "manual"], ["", False]),
        ("bot.global.warmup.max_minutes", [None, 0.5], [0, -1, False]),
        ("live.startup.warmup_concurrency", ["auto", 1, 7], [0, None, "unknown", 1.5, False]),
    ):
        for value in valid:
            set_path(source, path, value)
            assert not list(validator.iter_errors(source)), (path, value)
        for value in invalid:
            set_path(source, path, value)
            assert list(validator.iter_errors(source)), (path, value)
        set_path(source, path, get_path(get_template_config(), path))


def _mode_policy(value, layer):
    source = get_template_config()
    source["live"]["modes"]["long"] = "manual"
    if layer == "root":
        source["live"]["modes"]["long"] = value
    elif layer == "coin":
        source["coin_overrides"] = {"BTC": {"live": {"modes": {"long": value}}}}
    elif layer == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = {"live.modes.long": value}
    else:
        source["backtest"]["scenarios"] = [
            {"label": "mode", "overrides": {"live.modes.long": value}}
        ]
    return source


@pytest.mark.parametrize("value,expected", [
    (None, None), ("n", "normal"), ("normal", "normal"), ("m", "manual"),
    ("manual", "manual"), ("gs", "graceful_stop"),
    ("graceful_stop", "graceful_stop"), ("graceful-stop", "graceful_stop"),
    ("p", "panic"), ("panic", "panic"), ("t", "tp_only"), ("tp", "tp_only"),
    ("tp_only", "tp_only"), ("tp-only", "tp_only"),
    (" \tGs\n", "graceful_stop"), ("\u001cTP\u0085", "tp_only"),
    ("\u2003NoRmAl\u3000", "normal"),
])
@pytest.mark.parametrize("layer", ["root", "coin", "fixed", "scenario"])
def test_mode_schema_loader_and_cli_share_supported_inputs(value, expected, layer):
    import jsonschema

    source = _mode_policy(value, layer)
    original = deepcopy(source)
    jsonschema.Draft202012Validator(generate_schema()).validate(source)
    prepared = parse_overrides(prepare_config(source, verbose=False), verbose=False)
    if layer == "coin":
        assert prepared["coin_overrides"]["BTC"]["live"]["modes"]["long"] == expected
    elif layer == "root":
        assert prepared["live"]["modes"]["long"] == expected
        parser = ArgumentParser()
        add_arguments_recursively(parser, get_template_config())
        args = parser.parse_args(["--live.modes.long", "null" if value is None else value])
        updated = get_template_config()
        update_config_with_args(updated, args)
        assert updated["live"]["modes"]["long"] == expected
    else:
        report = explain_config(source, "live.modes.long", optimize=layer == "fixed",
                                scenario="mode" if layer == "scenario" else None)
        assert report["value"] == expected
    assert source == original


@pytest.mark.parametrize("value", ["", " \t", "take_profit_only", "graceful__stop", False, 0,
                                   "\ufeffm", "ſ"])
@pytest.mark.parametrize("layer", ["root", "coin", "fixed", "scenario"])
def test_mode_schema_and_loader_reject_the_same_invalid_inputs(value, layer):
    import jsonschema

    source = _mode_policy(value, layer)
    assert list(jsonschema.Draft202012Validator(generate_schema()).iter_errors(source))
    with pytest.raises((ValueError, TypeError), match="mode"):
        prepare_config(source, verbose=False)


def test_mode_schema_pattern_preserves_python_strip_whitespace_domain():
    from config.control_sentinels import forced_mode_input_pattern, optional_forced_mode

    pattern = forced_mode_input_pattern()
    whitespace = (chr(code) for code in range(sys.maxunicode + 1) if chr(code).isspace())
    for char in whitespace:
        value = f"{char}m{char}"
        assert optional_forced_mode(value, path="live.modes.long") == "manual"
        assert re.fullmatch(pattern, value)
    for char in ("\u180e", "\u200b", "\ufeff"):
        assert not re.fullmatch(pattern, f"{char}m{char}")


def test_released_v8_sentinels_convert_at_every_layer_and_report(tmp_path):
    source = released_v8_policy()
    source["live"].update(forced_mode_long="manual", max_warmup_minutes=0, warmup_concurrency=0)
    file = tmp_path / "coin.json"
    file.write_text(json.dumps({"live": {"forced_mode_long": ""}}))
    source["coin_overrides"] = {"BTC": {"override_config_path": file.name}}
    source["optimize"]["fixed_runtime_overrides"] = {
        "live.forced_mode_long": "", "live.max_warmup_minutes": 0, "live.warmup_concurrency": 0,
    }
    source["backtest"]["scenarios"] = [{"label": "one", "overrides": {"live": {
        "forced_mode_short": "", "max_warmup_minutes": 0, "warmup_concurrency": 0,
    }}}]
    original, file_bytes = deepcopy(source), file.read_bytes()
    result = migrate(source, best_effort=True, base_config_path=str(tmp_path / "input.json"))
    assert result.config["coin_overrides"]["BTC"]["live"]["modes"]["long"] is None
    for patch in (result.config["optimize"]["fixed_runtime_overrides"],
                  result.config["backtest"]["scenarios"][0]["overrides"]):
        assert patch["bot.global.warmup.max_minutes"] is None
        assert patch["live.startup.warmup_concurrency"] == "auto"
    assert result.config["optimize"]["fixed_runtime_overrides"]["live.modes.long"] is None
    assert result.config["backtest"]["scenarios"][0]["overrides"]["live.modes.short"] is None
    reported = [event for event in result.changes if event["reason"] == "explicit released-v8 control sentinel conversion"]
    assert len(reported) >= 8
    assert source == original
    assert file.read_bytes() == file_bytes


def test_legitimate_zeros_and_backtest_null_inheritance_remain_authored():
    source = get_template_config()
    source["bot"]["long"]["risk"]["total_wallet_exposure_limit"] = 0.0
    source["bot"]["long"]["forager"]["score_weights"] = dict.fromkeys(
        source["bot"]["long"]["forager"]["score_weights"], 0.0,
    )
    source["live"]["execution"]["execution_delay_seconds"] = 0.0
    source["backtest"]["filter_by_min_effective_cost"] = None
    result = prepare_config(source, verbose=False)
    assert result["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 0.0
    assert not any(result["bot"]["long"]["forager"]["score_weights"].values())
    assert result["live"]["execution"]["execution_delay_seconds"] == 0.0
    assert result["backtest"]["filter_by_min_effective_cost"] is None
