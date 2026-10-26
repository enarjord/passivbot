"""Public risk units are strict spellings over the unchanged native contract."""

from copy import deepcopy

import pytest

from config import get_template_config, prepare_config
from config.authoring import compact_config, resolved_config
from config.field_registry import get_field_registry
from config.hsl import generated_template, native_hsl_policy
from config.param_paths import resolve_optimizer_key_path
from config.shared_bot import flatten_shared_bot_side
from config.strategy import merge_runtime_bot_side
from tools.clean_config import cleanup_config


RENAMES = [
    *((f"bot.{scope}.hsl.{old}", f"bot.{scope}.hsl.{new}", value)
      for scope in ("global", "long", "short", "portfolio")
      for old, new, value in (
          ("drawdown_ratio_threshold", "drawdown_threshold_fraction", 0.125),
          ("drawdown_ratio_ema_span_minutes", "drawdown_fraction_ema_span_minutes", 12.5),
      )),
    ("bot.portfolio.loss_gate.max_drawdown_ratio",
     "bot.portfolio.loss_gate.max_drawdown_fraction", 0.125),
    *((f"bot.{scope}.risk.total_wallet_exposure_limit",
       f"bot.{scope}.risk.total_wallet_exposure_limit_ratio", 2.375)
      for scope in ("global", "long", "short")),
]


def put(config, path, value):
    parts = path.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def source_for(path):
    config = generated_template(get_template_config(),
                                "portfolio" if ".portfolio.hsl." in path else "coin")
    config["optimize"]["bounds"] = {}
    return config


@pytest.mark.parametrize("old,new,value", RENAMES)
@pytest.mark.parametrize("channel", ["root", "fixed", "scenario"])
def test_old_development_risk_spellings_are_rejected_without_conversion(old, new, value, channel):
    config = source_for(new)
    if channel == "root":
        put(config, old, value)
    elif channel == "fixed":
        config["optimize"]["fixed_runtime_overrides"] = {old: value}
    else:
        config["backtest"]["scenarios"] = [{"label": "invalid", "overrides": {old: value}}]
    original = deepcopy(config)
    with pytest.raises((KeyError, TypeError, ValueError), match=old.split(".")[-1]):
        prepare_config(config, live_only=True, verbose=False)
    assert config == original


@pytest.mark.parametrize("old,new,value", RENAMES)
def test_new_risk_spellings_roundtrip_values_through_every_export(old, new, value):
    config = source_for(new)
    put(config, new, value)
    prepared = prepare_config(config, live_only=True, verbose=False)
    for export in (cleanup_config, compact_config, resolved_config):
        dumped = export(prepared)
        loaded = prepare_config(dumped, live_only=True, verbose=False)
        node = loaded
        for part in new.split("."):
            node = node[part]
        assert node == value
        field = get_field_registry().find(tuple(new.split(".")))
        assert field is not None
        assert get_field_registry().find(tuple(old.split("."))) is None


def test_new_public_hsl_spans_and_threshold_lower_once_without_rescaling():
    config = source_for("bot.long.hsl")
    config["bot"]["long"].setdefault("hsl", {}).update(
        drawdown_threshold_fraction=0.125, drawdown_fraction_ema_span_minutes=12.5,
    )
    prepared = prepare_config(config, live_only=True, verbose=False)
    policy = prepared["bot"]["long"]["hsl"]
    original = deepcopy(policy)
    native = native_hsl_policy(policy)
    assert native["drawdown_threshold_fraction"] == 0.125
    assert native["drawdown_fraction_ema_span_minutes"] == 12.5
    assert "red_threshold" not in native
    assert "drawdown_ratio_ema_span_minutes" not in native
    assert policy == original


@pytest.mark.parametrize("side", ["long", "short"])
def test_per_coin_exposure_ratio_remains_public_and_lowers_to_native_wallet_limit(side):
    config = source_for("bot.long.hsl")
    config["coin_overrides"] = {"BTC": {"bot": {side: {"wallet_exposure_limit_ratio": 0.625}}}}
    prepared = prepare_config(config, live_only=True, verbose=False)
    patch = prepared["coin_overrides"]["BTC"]["bot"][side]
    assert patch["wallet_exposure_limit_ratio"] == 0.625
    native = merge_runtime_bot_side(prepared["bot"][side], override_side=patch)
    assert native["wallet_exposure_limit"] == 0.625
    assert "wallet_exposure_limit_ratio" not in native
    dumped = cleanup_config(prepared)
    assert dumped["coin_overrides"]["BTC"]["bot"][side] == {"wallet_exposure_limit_ratio": 0.625}
    config["coin_overrides"]["BTC"]["bot"][side] = {"wallet_exposure_limit": 0.625}
    with pytest.raises((KeyError, TypeError, ValueError), match="wallet_exposure_limit"):
        prepare_config(config, live_only=True, verbose=False)


def test_public_exposure_and_hsl_quantities_resolve_current_native_optimizer_ids():
    config = prepare_config(source_for("bot.long.hsl"), live_only=True, verbose=False)
    assert resolve_optimizer_key_path(config, "long_total_wallet_exposure_limit") == (
        "bot", "long", "risk", "total_wallet_exposure_limit_ratio")
    assert resolve_optimizer_key_path(config, "long_hsl_drawdown_threshold_fraction") == (
        "bot", "long", "hsl", "drawdown_threshold_fraction")
    assert resolve_optimizer_key_path(config, "long_hsl_drawdown_fraction_ema_span_minutes") == (
        "bot", "long", "hsl", "drawdown_fraction_ema_span_minutes")
    assert flatten_shared_bot_side(config["bot"]["long"])["total_wallet_exposure_limit"] == 1.5


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("allocation_ratio", [0.0, 0.625])
def test_live_allocation_and_backtest_payload_honor_the_same_public_coin_ratio(side, allocation_ratio):
    from backtest import prep_backtest_args
    from passivbot import Passivbot

    config = source_for("bot.long.hsl")
    for pside in ("long", "short"):
        config["bot"].setdefault(pside, {}).update(enabled=True)
        config["bot"][pside].setdefault("universe", {})["approved_coins"] = ["BTC"]
    config["coin_overrides"] = {
        "BTC": {"bot": {side: {"wallet_exposure_limit_ratio": allocation_ratio}}}
    }
    config = prepare_config(config, verbose=False, target="canonical", runtime=None)
    config["backtest"]["coins"] = {"binance": ["BTC"]}
    markets = {"BTC": dict(qty_step=0.001, price_step=0.01, min_qty=0.001,
                           min_cost=1.0, c_mult=1.0, maker=0.0002,
                           taker=0.0005, exchange="binance")}
    pair = prep_backtest_args(config, markets, "binance")[0][0]
    assert pair[side]["wallet_exposure_limit"] == allocation_ratio
    assert pair[side]["entry_eligible"] is (allocation_ratio != 0.0)
    assert "wallet_exposure_limit_ratio" not in pair[side]

    # Exercise allocation accessors without constructing any exchange client.
    bot = Passivbot.__new__(Passivbot)
    bot.config = config
    bot.coin_overrides = deepcopy(config["coin_overrides"])
    assert bot.get_wallet_exposure_limit(side, "BTC") == allocation_ratio
    assert bot.coin_overrides["BTC"]["bot"][side] == {
        "wallet_exposure_limit_ratio": allocation_ratio
    }


@pytest.mark.parametrize("retired_flag", [
    "--bot.long.risk.total_wallet_exposure_limit",
    "--bot_long_risk_total_wallet_exposure_limit",
    "--loss-gate-max-drawdown-ratio",
])
def test_command_parser_rejects_retired_names_that_are_prefixes_of_new_flags(retired_flag):
    from cli_utils import build_command_parser
    from config_utils import add_config_arguments, project_template_config_for_cli

    parser = build_command_parser(prog="backtest", description="", usage="", epilog="")
    template = project_template_config_for_cli(get_template_config(), "backtest")
    add_config_arguments(parser, template, command="backtest", help_all=True)
    for flag in ("--bot.long.risk.total_wallet_exposure_limit_ratio", "-ltwel"):
        args = parser.parse_args([flag, "1.75"])
        assert getattr(args, "bot.long.risk.total_wallet_exposure_limit_ratio") == 1.75
    args = parser.parse_args(["--loss-gate-max-drawdown-fraction", "0.125"])
    assert getattr(args, "bot.portfolio.loss_gate.max_drawdown_fraction") == 0.125
    with pytest.raises(SystemExit):
        parser.parse_args([retired_flag, "0.125"])
