"""Canonical fixed policy flags and reloadable directional coin sources."""

import argparse
from copy import deepcopy

import pytest

from config.schema import get_template_config
from config_utils import add_config_arguments, project_template_config_for_cli, update_config_with_args


def parser_for(command):
    parser = argparse.ArgumentParser()
    keys = add_config_arguments(parser, project_template_config_for_cli(get_template_config(), command), command=command)
    return parser, keys


@pytest.mark.parametrize("command", ["live", "backtest", "optimize"])
def test_shared_policy_flags_and_directional_sources_apply_to_every_consumer(command):
    parser, keys = parser_for(command)
    # Directional authority wins regardless of argv order.
    args = parser.parse_args(["--bot.short.universe.approved_coins", "SOL", "-s", "BTC,ETH",
                              "--bot.global.warmup.span_multiplier", "0.8", "--hedge-mode", "y"])
    config = get_template_config()
    update_config_with_args(config, args, allowed_keys=keys)
    assert config["bot"]["long"]["universe"]["approved_coins"] == ["BTC", "ETH"]
    assert config["bot"]["short"]["universe"]["approved_coins"] == ["SOL"]
    assert config["bot"]["global"]["warmup"]["span_multiplier"] == 0.8
    assert config["bot"]["global"]["hedge_mode"] is True
    assert config["_coins_sources"]["approved_coins"] == {"long": ["BTC", "ETH"], "short": ["SOL"]}
    assert "approved_coins" not in config["live"]


def test_one_side_cli_source_keeps_the_other_side_file_refreshable(tmp_path):
    config = get_template_config()
    config["bot"]["short"]["universe"]["approved_coins"] = "short.json"
    previous_short = deepcopy(config["bot"]["short"]["universe"])
    source = tmp_path / "coins.json"
    source.write_text('["BTC", "XRP"]')
    parser, keys = parser_for("live")
    args = parser.parse_args(["--bot.long.universe.approved_coins", str(source)])
    update_config_with_args(config, args, allowed_keys=keys)
    assert config["bot"]["long"]["universe"]["approved_coins"] == [str(source)]
    assert config["bot"]["short"]["universe"] == previous_short
    assert config["_coins_sources"]["approved_coins"] == {"long": [str(source)], "short": "short.json"}


@pytest.mark.parametrize("flag", ["--live.approved_coins", "--live_ignored_coins", "--live.hedge_mode",
                                  "--live.market_orders_allowed", "--live.max_warmup_minutes"])
def test_retired_live_policy_flags_are_rejected(flag):
    parser, _keys = parser_for("live")
    with pytest.raises(SystemExit) as error:
        parser.parse_args([flag, "1"])
    assert error.value.code == 2


def test_optimizer_policy_projection_keeps_numeric_genes_in_bounds():
    parser, _keys = parser_for("optimize")
    with pytest.raises(SystemExit):
        parser.parse_args(["--bot.long.strategy.trailing_martingale.entry.ema_span_0", "100"])


@pytest.mark.asyncio
async def test_download_cli_applies_side_sources_before_offline_preparation(monkeypatch):
    import ohlcv_download

    captured = []
    source = get_template_config()
    monkeypatch.setattr(ohlcv_download, "load_input_config", lambda _path: (source, "", deepcopy(source)))
    async def prepared_only(config):
        captured.append(config)
    monkeypatch.setattr(ohlcv_download, "warm_ohlcv_caches", prepared_only)
    monkeypatch.setattr("sys.argv", ["download", "-s", "BTC", "--bot.short.universe.approved_coins", "ETH",
                                    "--bot.long.universe.minimum_coin_age_days", "5"])
    await ohlcv_download.main()
    assert len(captured) == 1
    config = captured[0]
    assert config["bot"]["long"]["universe"]["approved_coins"] == ["BTC"]
    assert config["bot"]["short"]["universe"]["approved_coins"] == ["ETH"]
    assert config["bot"]["long"]["universe"]["minimum_coin_age_days"] == 5.0
