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
                              "--bot.portfolio.warmup.span_multiplier", "0.8", "--hedge-mode", "y"])
    config = get_template_config()
    update_config_with_args(config, args, allowed_keys=keys)
    assert config["bot"]["long"]["universe"]["approved_coins"] == ["BTC", "ETH"]
    assert config["bot"]["short"]["universe"]["approved_coins"] == ["SOL"]
    assert config["bot"]["portfolio"]["warmup"]["span_multiplier"] == 0.8
    assert config["bot"]["portfolio"]["hedge_mode"] is True
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


@pytest.mark.asyncio
@pytest.mark.parametrize('flag,field', [('--symbols','approved_coins'), ('--ignored-coins','ignored_coins')])
async def test_download_cli_decodes_directional_convenience_sources(monkeypatch, flag, field):
    import ohlcv_download
    source = get_template_config()
    captured = []
    monkeypatch.setattr(ohlcv_download, 'load_input_config', lambda _path: (source, '', deepcopy(source)))
    async def prepared_only(config):
        captured.append(config)
    monkeypatch.setattr(ohlcv_download, 'warm_ohlcv_caches', prepared_only)
    monkeypatch.setattr('sys.argv', ['download', flag, '{long: ["BTC"], short: ["ETH"]}'])
    await ohlcv_download.main()
    assert len(captured) == 1
    assert {side:captured[0]['bot'][side]['universe'][field] for side in ('long','short')} == {'long':['BTC'],'short':['ETH']}


@pytest.mark.parametrize("command", ["live", "backtest", "optimize"])
@pytest.mark.parametrize(
    "flag,value",
    [
        ("--symbols", "null"),
        ("--ignored-coins", "null"),
        ("--bot.global.universe.approved_coins", "null"),
        ("--symbols", '{long: null, short: ["ETH"]}'),
        ("--ignored-coins", "{long: [], short: null}"),
    ],
)
def test_nonnullable_coin_source_flags_reject_null_before_mutation(
    command, flag, value
):
    parser, keys = parser_for(command)
    config = get_template_config()
    before = deepcopy(config)
    with pytest.raises(ValueError, match="explicit side.*approved_coins"):
        update_config_with_args(
            config, parser.parse_args([flag, value]), allowed_keys=keys
        )
    assert config == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flag,value",
    [
        ("--symbols", "null"),
        ("--ignored-coins", "null"),
        ("--bot.global.universe.approved_coins", "null"),
        ("--symbols", '{long: null, short: ["ETH"]}'),
    ],
)
async def test_download_rejects_null_sources_before_mutation(monkeypatch, flag, value):
    import ohlcv_download

    source = get_template_config()
    before = deepcopy(source)
    monkeypatch.setattr(
        ohlcv_download,
        "load_input_config",
        lambda _path: (source, "", deepcopy(source)),
    )

    async def unexpected(*args, **kwargs):
        pytest.fail("invalid source must not reach download preparation")

    monkeypatch.setattr(ohlcv_download, "warm_ohlcv_caches", unexpected)
    monkeypatch.setattr("sys.argv", ["download", flag, value])
    with pytest.raises(ValueError, match="explicit side.*approved_coins"):
        await ohlcv_download.main()
    assert source == before


@pytest.mark.parametrize("command", ["live", "backtest", "optimize"])
def test_explicit_side_null_and_empty_convenience_have_distinct_authority(command):
    from config.universe import resolve_universe_sources

    parser, keys = parser_for(command)
    config = get_template_config()
    config["bot"]["global"]["universe"]["approved_coins"] = ["BTC"]
    for side in ("long", "short"):
        config["bot"][side]["universe"]["approved_coins"] = ["ETH"]
    update_config_with_args(
        config,
        parser.parse_args(["--bot.long.universe.approved_coins", "null"]),
        allowed_keys=keys,
    )
    assert config["bot"]["long"]["universe"]["approved_coins"] is None
    assert resolve_universe_sources(config, "approved_coins")[0] == {
        "long": ["BTC"],
        "short": ["ETH"],
    }
    update_config_with_args(config, parser.parse_args(["-s", ""]), allowed_keys=keys)
    assert resolve_universe_sources(config, "approved_coins")[0] == {
        "long": [],
        "short": [],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("side_null", [False, True])
async def test_download_empty_source_or_explicit_side_null(monkeypatch, side_null):
    from config.universe import resolve_universe_sources
    import ohlcv_download

    source = get_template_config()
    source["bot"]["global"]["universe"]["approved_coins"] = ["BTC"]
    for side in ("long", "short"):
        source["bot"][side]["universe"]["approved_coins"] = ["ETH"]
    monkeypatch.setattr(
        ohlcv_download,
        "load_input_config",
        lambda _path: (source, "", deepcopy(source)),
    )
    observed = []

    async def inspect(config):
        observed.append(resolve_universe_sources(config, "approved_coins")[0])

    monkeypatch.setattr(ohlcv_download, "warm_ohlcv_caches", inspect)
    monkeypatch.setattr(
        "sys.argv",
        (
            ["download", "--bot.long.universe.approved_coins", "null"]
            if side_null
            else ["download", "-s", ""]
        ),
    )
    await ohlcv_download.main()
    assert observed == (
        [{"long": ["BTC"], "short": ["ETH"]}]
        if side_null
        else [{"long": [], "short": []}]
    )
