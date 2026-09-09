"""Regression coverage for independent unstuck horizons and legacy composition."""

from copy import deepcopy
import json
import logging

import pytest

from config import (
    get_template_config,
    prepare_config,
    load_prepared_config,
    compile_runtime_config,
)
from config.overrides import parse_overrides
from config.optimize_bounds import flatten_optimize_bounds
from warmup_utils import compute_per_coin_warmup_minutes


def legacy_config(kind="trailing_martingale"):
    c = get_template_config()
    c["config_version"] = "v8.2.0"
    c["live"]["strategy_kind"] = kind
    for side in ("long", "short"):
        for key in ("ema_span_0", "ema_span_1"):
            c["bot"][side]["unstuck"].pop(key)
            c["optimize"]["bounds"][side]["unstuck"].pop(key)
    return c


@pytest.mark.parametrize(
    "kind", ["trailing_martingale", "ema_anchor", "trailing_grid_v7"]
)
def test_legacy_composed_spans_and_roundtrip(kind, caplog):
    c = legacy_config(kind)
    c["bot"]["long"]["strategy"][kind].update(ema_span_0=101.25, ema_span_1=902.75)
    c["coin_overrides"] = {
        "BTC": {"bot": {"long": {"strategy": {kind: {"ema_span_0": 17.5}}}}},
        "ETH": {"bot": {"short": {"strategy": {kind: {"ema_span_1": 211.25}}}}},
    }
    with caplog.at_level(logging.INFO):
        prepared = prepare_config(c, verbose=False)
    assert prepared["bot"]["long"]["unstuck"]["ema_span_0"] == 101.25
    assert prepared["bot"]["long"]["unstuck"]["ema_span_1"] == 902.75
    assert prepared["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"] == {
        "ema_span_0": 17.5
    }
    assert prepared["coin_overrides"]["ETH"]["bot"]["short"]["unstuck"] == {
        "ema_span_1": 211.25
    }
    resolved = parse_overrides(prepared, verbose=False)
    reloaded = parse_overrides(prepare_config(resolved, verbose=False), verbose=False)
    assert reloaded["bot"] == resolved["bot"]
    assert reloaded["coin_overrides"] == resolved["coin_overrides"]
    assert "1-to-1 optimizer-search migration is impossible" in caplog.text
    assert "Preserved 2 coin-specific" in caplog.text
    assert prepared["optimize"]["bounds"]["long"]["unstuck"]["ema_span_0"] == [
        101.25,
        101.25,
    ]


def test_explicit_global_spans_are_inherited_by_strategy_overrides():
    c = get_template_config()
    c["bot"]["long"]["unstuck"].update(ema_span_0=300.5, ema_span_1=700.25)
    c["coin_overrides"] = {
        "BTC": {
            "bot": {"long": {"strategy": {"trailing_martingale": {"ema_span_0": 17.5}}}}
        }
    }
    resolved = parse_overrides(prepare_config(c, verbose=False), verbose=False)
    assert "unstuck" not in resolved["coin_overrides"]["BTC"]["bot"]["long"]
    assert resolved["bot"]["long"]["unstuck"]["ema_span_0"] == 300.5


def test_file_inline_precedence_migration(tmp_path):
    c = legacy_config()
    external = {
        "bot": {
            "long": {
                "strategy": {
                    "trailing_martingale": {"ema_span_0": 19.5, "ema_span_1": 301.5}
                }
            }
        }
    }
    (tmp_path / "coin.json").write_text(json.dumps(external))
    c["coin_overrides"] = {
        "BTC": {
            "override_config_path": "coin.json",
            "bot": {
                "long": {"strategy": {"trailing_martingale": {"ema_span_1": 601.5}}}
            },
        }
    }
    path = tmp_path / "master.json"
    path.write_text(json.dumps(c))
    resolved = parse_overrides(
        load_prepared_config(str(path), verbose=False), verbose=False
    )
    assert resolved["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"] == {
        "ema_span_0": 19.5,
        "ema_span_1": 601.5,
    }


def test_partial_explicit_spans_and_bounds_win():
    c = legacy_config()
    c["bot"]["long"]["unstuck"]["ema_span_0"] = 81.5
    c["optimize"]["bounds"]["long"]["unstuck"]["ema_span_1"] = [10, 1000, 0.25]
    c["coin_overrides"] = {
        "BTC": {
            "bot": {
                "long": {
                    "unstuck": {"ema_span_1": 77.5},
                    "strategy": {
                        "trailing_martingale": {"ema_span_0": 50, "ema_span_1": 100}
                    },
                }
            }
        }
    }
    prepared = parse_overrides(prepare_config(c, verbose=False), verbose=False)
    assert prepared["bot"]["long"]["unstuck"]["ema_span_0"] == 81.5
    assert prepared["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"] == {
        "ema_span_1": 77.5
    }
    assert prepared["optimize"]["bounds"]["long"]["unstuck"]["ema_span_1"] == [
        10,
        1000,
        0.25,
    ]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
@pytest.mark.parametrize("coin_override", [False, True])
def test_invalid_spans_rejected(value, coin_override):
    c = get_template_config()
    target = c["bot"]["long"]["unstuck"]
    if coin_override:
        c["coin_overrides"] = {"BTC": {"bot": {"long": {"unstuck": {}}}}}
        target = c["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]
    target["ema_span_0"] = value
    with pytest.raises((ValueError, TypeError)):
        parse_overrides(prepare_config(c, verbose=False), verbose=False)


def test_defaults_and_runtime_mapping():
    c = get_template_config()
    example = load_prepared_config(
        "configs/examples/default_trailing_martingale_long.json", verbose=False
    )
    for side in ("long", "short"):
        assert c["bot"][side]["unstuck"] == example["bot"][side]["unstuck"]
        assert (
            c["optimize"]["bounds"][side]["unstuck"]
            == example["optimize"]["bounds"][side]["unstuck"]
        )
    runtime = compile_runtime_config(
        prepare_config(c, verbose=False), runtime="backtest"
    )
    assert runtime["bot"]["long"]["unstuck_ema_span_0"] == 790.0
    bounds = flatten_optimize_bounds(
        c["optimize"]["bounds"], strategy_kind="trailing_martingale"
    )
    assert bounds["long_unstuck_ema_span_1"] == [60, 2880, 10]


def test_gpu_guard_rejects_independent_search_but_allows_matching_fixed_spans():
    from optimization.gpu.unstuck_scope import validate_independent_unstuck_scope

    c = get_template_config()
    c["optimize"]["bounds"] = {}
    validate_independent_unstuck_scope(c)
    c["bot"]["long"]["unstuck"]["ema_span_0"] = 999.5
    with pytest.raises(ValueError, match="does not yet model independent"):
        validate_independent_unstuck_scope(c)
    c["bot"]["long"]["unstuck"]["ema_gating_enabled"] = False
    validate_independent_unstuck_scope(c)


def test_unstuck_spans_extend_per_coin_and_optimizer_warmup():
    from warmup_utils import compute_backtest_warmup_minutes

    c = get_template_config()
    c["live"].update(warmup_ratio=2.0, max_warmup_minutes=1_000_000)
    c["optimize"]["bounds"] = {}
    for side in ("long", "short"):
        strategy = c["bot"][side]["strategy"]["trailing_martingale"]
        for key in (
            "ema_span_0",
            "ema_span_1",
            "volatility_ema_span_1m",
            "volatility_ema_span_1h",
        ):
            strategy[key] = 1.0
    c["coin_overrides"] = {
        "BTC": {"bot": {"long": {"unstuck": {"ema_span_0": 40_000.25}}}}
    }
    prepared = parse_overrides(prepare_config(c, verbose=False), verbose=False)
    assert compute_per_coin_warmup_minutes(prepared)["BTC"] == 80_001
    assert compute_backtest_warmup_minutes(prepared) >= 80_001


def test_legacy_scenario_span_override_migrates():
    c = legacy_config()
    c["backtest"]["scenarios"] = [
        {
            "label": "horizon",
            "overrides": {"bot.long.strategy.trailing_martingale.ema_span_0": 401.5},
        }
    ]
    prepared = prepare_config(c, verbose=False)
    assert (
        prepared["backtest"]["scenarios"][0]["overrides"]["bot.long.unstuck.ema_span_0"]
        == 401.5
    )


def test_gpu_guard_checks_fixed_bound_values_and_coin_overrides():
    from optimization.gpu.unstuck_scope import validate_independent_unstuck_scope

    c = get_template_config()
    c["optimize"]["bounds"] = {
        "long": {"strategy": {"trailing_martingale": {"ema_span_0": [10, 10]}}}
    }
    with pytest.raises(ValueError, match="independent unstuck"):
        validate_independent_unstuck_scope(c)
    c["optimize"]["bounds"] = {}
    c["coin_overrides"] = {"BTC": {"bot": {"long": {"unstuck": {"ema_span_0": 10.5}}}}}
    with pytest.raises(ValueError, match="BTC long.ema_span_0"):
        validate_independent_unstuck_scope(c)


def test_optimizer_paths_keep_strategy_and_unstuck_spans_separate():
    from optimization.config_adapter import get_optimization_key_paths

    paths = dict(get_optimization_key_paths(get_template_config()))
    assert paths["long_unstuck_ema_span_0"] == ("bot", "long", "unstuck", "ema_span_0")
    assert paths["long_ema_span_0"] == (
        "bot",
        "long",
        "strategy",
        "trailing_martingale",
        "ema_span_0",
    )


@pytest.mark.parametrize("bounds", [[0, 10], [-1, 10], [1, float("inf")]])
def test_optimizer_rejects_nonpositive_or_nonfinite_span_bounds(bounds):
    from optimization.config_adapter import get_optimization_key_paths

    c = get_template_config()
    c["optimize"]["bounds"]["long"]["unstuck"]["ema_span_0"] = bounds
    with pytest.raises(ValueError):
        get_optimization_key_paths(c)


def test_legacy_coin_flags_file_spans_are_preserved(tmp_path):
    c = legacy_config()
    external = {
        "bot": {
            "long": {
                "strategy": {
                    "trailing_martingale": {"ema_span_0": 41.25, "ema_span_1": 111.5}
                }
            }
        }
    }
    (tmp_path / "coin.json").write_text(json.dumps(external))
    c["live"]["coin_flags"] = {"BTC": "-lc coin.json"}
    path = tmp_path / "master.json"
    path.write_text(json.dumps(c))
    resolved = parse_overrides(
        load_prepared_config(str(path), verbose=False), verbose=False
    )
    assert resolved["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"] == {
        "ema_span_0": 41.25,
        "ema_span_1": 111.5,
    }


def test_zero_span_on_disabled_legacy_side_warns_and_gets_positive_default(caplog):
    c = legacy_config()
    c["bot"]["short"]["strategy"]["trailing_martingale"]["ema_span_0"] = 0
    prepared = prepare_config(c, verbose=False)
    assert prepared["bot"]["short"]["unstuck"]["ema_span_0"] > 0
    assert "Cannot copy zero strategy span" in caplog.text
    assert "review spans before enabling" in caplog.text


def test_missing_legacy_bounds_warn_about_implicitly_coupled_search(caplog):
    c = legacy_config()
    c["optimize"]["bounds"] = {}
    prepared = prepare_config(c, verbose=False)
    assert "1-to-1 optimizer-search migration is impossible" in caplog.text
    value = prepared["bot"]["long"]["unstuck"]["ema_span_0"]
    assert prepared["optimize"]["bounds"]["long"]["unstuck"]["ema_span_0"] == [
        value,
        value,
    ]


def test_zero_span_on_disabled_legacy_coin_override_warns_and_migrates(caplog):
    c = legacy_config()
    c["coin_overrides"] = {
        "BTC": {
            "bot": {"short": {"strategy": {"trailing_martingale": {"ema_span_0": 0}}}}
        }
    }
    prepared = parse_overrides(prepare_config(c, verbose=False), verbose=False)
    assert (
        prepared["coin_overrides"]["BTC"]["bot"]["short"]["unstuck"]["ema_span_0"] > 0
    )
    assert (
        "Cannot copy zero strategy span to coin_overrides.BTC.bot.short.unstuck.ema_span_0"
        in caplog.text
    )
