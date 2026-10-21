"""Shared policy ownership, released defaults and persistent selection contracts."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from config import prepare_config
from config.migrations.shared_policy import LIVE_POLICY_PATHS
from config.migrations.v9 import migrate_v8_to_v9
from config.policy import effective_minimum_cost_filter
from config.schema import get_template_config
from config.universe import always_include_matches
from released_v8_fixtures import released_v8_policy


def test_new_policy_defaults_are_shared_and_backtest_override_retains_inheritance():
    config = prepare_config(get_template_config(), verbose=False)
    assert not (
        set(config["live"]) & {*LIVE_POLICY_PATHS, "approved_coins", "ignored_coins"}
    )
    assert config["bot"]["global"]["universe"]["filter_by_min_effective_cost"] is True
    assert config["backtest"]["filter_by_min_effective_cost"] is None
    assert effective_minimum_cost_filter(config) is True
    config["bot"]["global"]["universe"]["filter_by_min_effective_cost"] = False
    assert effective_minimum_cost_filter(config) is False
    config["backtest"]["filter_by_min_effective_cost"] = True
    assert effective_minimum_cost_filter(config) is True


@pytest.mark.parametrize(
    "field", [*LIVE_POLICY_PATHS, "approved_coins", "ignored_coins"]
)
@pytest.mark.parametrize("location", ["root", "scenario", "fixed", "selector"])
def test_old_shared_policy_paths_reject_before_hydration(field, location):
    config = get_template_config()
    if location == "root":
        config["live"][field] = False
    elif location == "scenario":
        config["backtest"]["scenarios"] = [
            {"label": "old", "overrides": {f"live.{field}": False}}
        ]
    elif location == "fixed":
        config["optimize"]["fixed_runtime_overrides"] = {f"live.{field}": False}
    else:
        config["optimize"]["fixed_params"] = [f"live.{field}"]
    with pytest.raises(ValueError, match="requires bot ownership"):
        prepare_config(config, verbose=False)


@pytest.mark.parametrize(
    "shared,override,effective",
    [
        (True, None, True),
        (False, None, False),
        (True, False, False),
        (False, True, True),
    ],
)
def test_backtest_payload_resolves_shared_policy_and_persistent_selection(
    shared, override, effective
):
    from backtest import prep_backtest_args

    config = get_template_config()
    config["bot"]["global"]["universe"]["filter_by_min_effective_cost"] = shared
    config["backtest"]["filter_by_min_effective_cost"] = override
    config["backtest"]["coins"] = {"binance": ["BTC", "ETH"]}
    config["bot"]["long"]["universe"]["approved_coins"] = ["BTC", "ETH"]
    config["bot"]["long"]["forager"]["always_include"] = ["BTC"]
    mss = {
        coin: {
            "qty_step": 0.001,
            "price_step": 0.1,
            "min_qty": 0.001,
            "min_cost": 5.0,
            "c_mult": 1.0,
            "maker": 0.0002,
            "taker": 0.0005,
            "exchange": "binance",
            "symbol": f"{coin}/USDT:USDT",
        }
        for coin in ("BTC", "ETH")
    }
    bots, _, _, backtest = prep_backtest_args(config, mss, "binance")
    assert backtest["filter_by_min_effective_cost"] is effective
    assert [pair["long"]["is_forced_active"] for pair in bots] == [True, False]
    assert [pair["short"]["is_forced_active"] for pair in bots] == [False, False]


def test_always_include_is_ordered_deduplicated_and_scoped_to_market(
    tmp_path, monkeypatch
):
    import utils

    monkeypatch.chdir(tmp_path)
    markets = {
        "BTC/USDT:USDT": {
            "id": "BTCUSDT",
            "base": "BTC",
            "swap": True,
            "linear": True,
            "active": True,
        },
        "1000BTC/USDT:USDT": {
            "id": "1000BTCUSDT",
            "base": "1000BTC",
            "swap": True,
            "linear": True,
            "active": True,
        },
    }
    utils.create_coin_symbol_map_cache("binance", markets, verbose=False)
    config = get_template_config()
    config["bot"]["long"]["forager"]["always_include"] = [
        " BTC ",
        "ETH",
        "BTC",
        "binance::BTCUSDT",
    ]
    prepared = prepare_config(config, verbose=False)
    assert prepared["bot"]["long"]["forager"]["always_include"] == [
        "BTC",
        "ETH",
        "binance::BTCUSDT",
    ]
    assert always_include_matches(
        prepared,
        "long",
        "BTC/USDT:USDT",
        exchange="binance",
        market_symbol="BTC/USDT:USDT",
    )
    assert not always_include_matches(prepared, "short", "BTC", exchange="binance")
    assert not always_include_matches(
        prepared,
        "long",
        "1000BTC/USDT:USDT",
        exchange="binance",
        market_symbol="1000BTC/USDT:USDT",
    )
    prepared["bot"]["long"]["forager"]["always_include"] = ["binance::BTCUSDT"]
    assert always_include_matches(
        prepared, "long", "BTC", exchange="binance", market_symbol="BTC/USDT:USDT"
    )
    assert not always_include_matches(
        prepared, "long", "BTCUSDT", exchange="bybit", market_symbol="BTC/USDT:USDT"
    )
    assert not always_include_matches(prepared, "long", "BTCUSDT")


@pytest.mark.parametrize(
    "operational,expected",
    [(None, "normal"), ("manual", "manual"), ("tp", "tp_only"), ("n", "normal")],
)
def test_persistent_selection_preserves_live_operational_precedence(
    operational, expected
):
    from passivbot import Passivbot

    config = get_template_config()
    config["bot"]["long"]["forager"]["always_include"] = ["BTC"]
    stub = SimpleNamespace(
        config=config, markets_dict={"BTC": {"active": True}}, exchange="binance"
    )
    stub.config_get = lambda path, symbol: operational
    stub._apply_entry_eligibility_mode = lambda side, symbol, mode=None: mode
    assert Passivbot._orchestrator_mode_override(stub, "long", "BTC") == expected
    stub.ineligible_symbols = {"BTC": "too young"}
    if operational is None:
        assert Passivbot._orchestrator_mode_override(stub, "long", "BTC") == "manual"


def test_root_operational_mode_still_disables_forager():
    from passivbot import Passivbot

    stub = SimpleNamespace(approved_coins_minus_ignored_coins={"long": {"BTC", "ETH"}})
    stub.bot_value = lambda *args: 1.0
    stub.get_max_n_positions = lambda side: 1
    stub.live_value = lambda key: "normal"
    assert Passivbot.is_forager_mode(stub, "long") is False
    stub.live_value = lambda key: ""
    assert Passivbot.is_forager_mode(stub, "long") is True


@pytest.mark.parametrize("override", ["missing", None, False, True])
def test_v8_minimum_cost_defaults_preserve_missing_versus_inherited(override):
    source = released_v8_policy()
    source["live"].pop("filter_by_min_effective_cost", None)
    source["backtest"].pop("filter_by_min_effective_cost", None)
    if override != "missing":
        source["backtest"]["filter_by_min_effective_cost"] = override
    result = migrate_v8_to_v9(source, memory_span_minutes=100.0).config
    assert result["bot"]["global"]["universe"]["filter_by_min_effective_cost"] is True
    assert result["backtest"]["filter_by_min_effective_cost"] is (
        False if override == "missing" else override
    )
    assert effective_minimum_cost_filter(result) is (
        False if override == "missing" else True if override is None else override
    )


def test_v8_root_scenario_fixed_sources_and_coin_normal_selection_move_together():
    source = released_v8_policy()
    source["live"]["approved_coins"] = {"long": "coins.json", "short": ["ETH"]}
    source["coin_overrides"] = {
        "BTC": {"live": {"forced_mode_long": "normal"}},
        "ETH": {"live": {"forced_mode_short": "manual"}},
    }
    source["optimize"]["fixed_params"] = ["live.approved_coins", "live.hedge_mode"]
    source["optimize"]["fixed_runtime_overrides"] = {"live.market_orders_allowed": True}
    source["backtest"]["scenarios"] = [
        {
            "label": "selection",
            "overrides": {
                "live.approved_coins": {"long": ["ETH"], "short": []},
                "live.warmup_ratio": 0.5,
                "coin_overrides": {"ETH": {"live": {"forced_mode_long": "normal"}}},
            },
        }
    ]
    original = deepcopy(source)
    result = migrate_v8_to_v9(source, memory_span_minutes=100.0).config
    assert source == original
    assert result["bot"]["long"]["universe"]["approved_coins"] == "coins.json"
    assert result["bot"]["long"]["forager"]["always_include"] == ["BTC"]
    assert result["coin_overrides"]["BTC"]["live"]["modes"]["long"] == "normal"
    assert result["optimize"]["fixed_params"] == [
        "bot.*.universe.approved_coins",
        "bot.global.hedge_mode",
    ]
    assert result["optimize"]["fixed_runtime_overrides"] == {
        "bot.global.execution.market_orders_allowed": True
    }
    patch = result["backtest"]["scenarios"][0]["overrides"]
    assert patch["bot.long.universe.approved_coins"] == ["ETH"]
    assert patch["bot.short.universe.approved_coins"] == []
    assert patch["bot.global.warmup.span_multiplier"] == 0.5
    assert patch["bot.long.forager.always_include"] == ["ETH"]
    assert patch["bot.short.forager.always_include"] == []


@pytest.mark.parametrize("value", [0, 1, "false", [], float("nan")])
def test_minimum_cost_consumer_rejects_nonboolean_policy(value):
    config = get_template_config()
    config["backtest"]["filter_by_min_effective_cost"] = value
    with pytest.raises(TypeError, match="boolean or null"):
        effective_minimum_cost_filter(config)
    config["backtest"]["filter_by_min_effective_cost"] = None
    config["bot"]["global"]["universe"]["filter_by_min_effective_cost"] = value
    with pytest.raises(TypeError, match="must be a boolean"):
        effective_minimum_cost_filter(config)


def _forced_selection_payload(config, *, optimized=False, scenario=False):
    """Read the complete native payload after applying the actual runtime patches."""
    from backtest import prep_backtest_args
    from tools.migrate_config_v9 import _set_patch

    effective = prepare_config(deepcopy(config), verbose=False)
    if optimized:
        _set_patch(effective, effective["optimize"]["fixed_runtime_overrides"])
        from optimizer_overrides import optimizer_overrides

        overrides = effective["optimize"]["enable_overrides"]
        effective = optimizer_overrides(overrides, effective, None)
        for side in ("long", "short"):
            effective = optimizer_overrides(overrides, effective, side)
    if scenario:
        _set_patch(
            effective, effective["backtest"]["scenarios"][0]["overrides"], scenario=True
        )
    effective = prepare_config(effective, verbose=False)
    coins = ["BTC", "ETH", "XRP"]
    effective["backtest"]["coins"] = {"binance": coins}
    for side in ("long", "short"):
        effective["bot"][side]["universe"]["approved_coins"] = coins
    mss = {
        coin: {
            "qty_step": 0.001,
            "price_step": 0.1,
            "min_qty": 0.001,
            "min_cost": 5.0,
            "c_mult": 1.0,
            "maker": 0.0002,
            "taker": 0.0005,
            "exchange": "binance",
            "symbol": f"{coin}/USDT:USDT",
        }
        for coin in coins
    }
    bots, _, _, _ = prep_backtest_args(effective, mss, "binance")
    return {
        side: {
            coin for coin, pair in zip(coins, bots) if pair[side]["is_forced_active"]
        }
        for side in ("long", "short")
    }


@pytest.mark.parametrize("converter", ["core", "tool"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("mode", ["manual", "", "tp_only", "graceful_stop"])
@pytest.mark.parametrize("shape", ["leaf", "live", "coin", "map"])
def test_v8_scenario_mode_replacements_clear_inherited_selection(
    converter, side, mode, shape
):
    from tools.migrate_config_v9 import migrate

    source = released_v8_policy()
    source["coin_overrides"] = {
        "BTC": {"live": {"forced_mode_long": "normal", "forced_mode_short": "normal"}}
    }
    source["optimize"]["fixed_runtime_overrides"] = {}
    live = {"forced_mode_long": "normal", "forced_mode_short": "normal"}
    live[f"forced_mode_{side}"] = mode
    patch = {
        "leaf": {f"coin_overrides.BTC.live.forced_mode_{side}": mode},
        "live": {"coin_overrides.BTC.live": live},
        "coin": {"coin_overrides.BTC": {"live": live}},
        "map": {"coin_overrides": {"BTC": {"live": live}}},
    }[shape]
    source["backtest"]["scenarios"] = [{"label": "change-mode", "overrides": patch}]
    original = deepcopy(source)
    result = (migrate_v8_to_v9 if converter == "core" else migrate)(
        source, memory_span_minutes=100.0
    )
    assert source == original
    assert _forced_selection_payload(result.config) == {
        "long": {"BTC"},
        "short": {"BTC"},
    }
    expected = {"long": {"BTC"}, "short": {"BTC"}}
    expected[side] = set()
    for optimized in (False, True):
        assert (
            _forced_selection_payload(result.config, optimized=optimized, scenario=True)
            == expected
        )
    converted_patch = result.config["backtest"]["scenarios"][0]["overrides"]
    assert converted_patch[f"bot.{side}.forager.always_include"] == []
    records = [
        change
        for change in result.changes
        if "deterministic context intersection" in change["reason"]
    ]
    assert records
    assert all(not change.get("best_effort") for change in records)


@pytest.mark.parametrize("converter", ["core", "tool"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize(
    "scenario_coin,mode,expected_lists",
    [
        ("BTC", "", []),
        ("BTC", "manual", []),
        ("ETH", "normal", ["ETH"]),
    ],
)
def test_v8_scenario_selection_intersects_root_and_fixed_contexts(
    converter, side, scenario_coin, mode, expected_lists
):
    from tools.migrate_config_v9 import migrate

    source = released_v8_policy()
    source["coin_overrides"] = {
        "BTC": {"live": {f"forced_mode_{side}": "normal"}},
        "ETH": {"live": {f"forced_mode_{side}": "manual"}},
    }
    source["optimize"]["fixed_runtime_overrides"] = {
        f"coin_overrides.BTC.live.forced_mode_{side}": "manual",
        f"coin_overrides.ETH.live.forced_mode_{side}": "normal",
    }
    source["backtest"]["scenarios"] = [
        {
            "label": "change-mode",
            "overrides": {
                f"coin_overrides.{scenario_coin}.live.forced_mode_{side}": mode
            },
        }
    ]
    result = (migrate_v8_to_v9 if converter == "core" else migrate)(
        source, memory_span_minutes=100.0
    )
    other = "short" if side == "long" else "long"
    assert _forced_selection_payload(result.config) == {side: {"BTC"}, other: set()}
    assert _forced_selection_payload(result.config, optimized=True) == {
        side: {"ETH"},
        other: set(),
    }
    for optimized in (False, True):
        # These are the released effective mode maps: the scenario replaces
        # only its authored leaf, retaining unrelated optimizer-fixed modes.
        expected = {"ETH"} if optimized else {"BTC"}
        if scenario_coin == "BTC":
            expected.discard("BTC")
        else:
            expected.add("ETH")
        assert _forced_selection_payload(
            result.config, optimized=optimized, scenario=True
        ) == {side: expected, other: set()}
    patch = result.config["backtest"]["scenarios"][0]["overrides"]
    assert patch[f"bot.{side}.forager.always_include"] == expected_lists


@pytest.mark.parametrize("shape", ["leaf", "map"])
def test_v8_file_normal_selection_obeys_scenario_and_atomic_replacement(
    tmp_path, shape
):
    import json
    from tools.migrate_config_v9 import migrate

    (tmp_path / "coin.json").write_text(
        json.dumps(
            {"live": {"forced_mode_long": "normal", "forced_mode_short": "normal"}}
        )
    )
    source = released_v8_policy()
    source["coin_overrides"] = {"BTC": {"override_config_path": "coin.json"}}
    source["optimize"]["fixed_runtime_overrides"] = {}
    source["backtest"]["scenarios"] = [
        {
            "label": "replace-file-mode",
            "overrides": (
                {"coin_overrides.BTC.live.forced_mode_long": "manual"}
                if shape == "leaf"
                else {
                    "coin_overrides": {"ETH": {"live": {"forced_mode_short": "normal"}}}
                }
            ),
        }
    ]
    result = migrate(
        source,
        memory_span_minutes=100.0,
        base_config_path=str(tmp_path / "config.json"),
    )
    assert result.config["coin_overrides"]["BTC"]["live"] == {
        "modes": {"long": "normal", "short": "normal"},
    }
    assert _forced_selection_payload(result.config) == {
        "long": {"BTC"},
        "short": {"BTC"},
    }
    expected = {"long": set(), "short": {"BTC"} if shape == "leaf" else {"ETH"}}
    for optimized in (False, True):
        assert (
            _forced_selection_payload(result.config, optimized=optimized, scenario=True)
            == expected
        )


def test_optimizer_mirror_preserves_directional_selection_and_mirrors_numeric_forager():
    from optimizer_overrides import optimizer_overrides

    config = get_template_config()
    config["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    for side, coin in (("long", "BTC"), ("short", "ETH")):
        config["bot"][side]["forager"]["always_include"] = [coin]
    weights = config["bot"]["long"]["forager"]["score_weights"]
    weights.update(volume=0.7, volatility=0.3, unilateralness=0.0, ema_readiness=0.0)
    mirrored = optimizer_overrides(
        config["optimize"]["enable_overrides"], deepcopy(config)
    )
    assert mirrored["bot"]["short"]["forager"]["score_weights"] == weights
    assert mirrored["bot"]["long"]["forager"]["always_include"] == ["BTC"]
    assert mirrored["bot"]["short"]["forager"]["always_include"] == ["ETH"]
    assert _forced_selection_payload(config, optimized=True) == {
        "long": {"BTC"},
        "short": {"ETH"},
    }


@pytest.mark.parametrize("converter", ["core", "tool"])
def test_v8_mirrored_strategy_keeps_original_directional_normal_selection(converter):
    from tools.migrate_config_v9 import migrate

    source = released_v8_policy()
    source["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    source["optimize"]["fixed_runtime_overrides"] = {}
    source["coin_overrides"] = {
        "BTC": {"live": {"forced_mode_long": "normal"}},
        "ETH": {"live": {"forced_mode_short": "normal"}},
    }
    result = (migrate_v8_to_v9 if converter == "core" else migrate)(
        source, memory_span_minutes=100.0
    )
    for optimized in (False, True):
        assert _forced_selection_payload(result.config, optimized=optimized) == {
            "long": {"BTC"},
            "short": {"ETH"},
        }
