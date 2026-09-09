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


@pytest.mark.parametrize("omitted", ["bounds", "optimize"])
def test_omitted_legacy_bounds_are_fixed_after_hydration(omitted, caplog):
    c = legacy_config()
    if omitted == "bounds":
        del c["optimize"]["bounds"]
    else:
        del c["optimize"]
    c["bot"]["long"]["strategy"]["trailing_martingale"]["ema_span_0"] = 403.5
    prepared = prepare_config(c, verbose=False)
    assert prepared["optimize"]["bounds"]["long"]["unstuck"]["ema_span_0"] == [
        403.5,
        403.5,
    ]
    assert "1-to-1 optimizer-search migration is impossible" in caplog.text


def test_inactive_coin_wel_zero_span_migrates(caplog):
    c = legacy_config()
    c["coin_overrides"] = {
        "BTC": {
            "bot": {
                "long": {
                    "wallet_exposure_limit": 0,
                    "strategy": {"trailing_martingale": {"ema_span_0": 0}},
                }
            }
        }
    }
    prepared = parse_overrides(prepare_config(c, verbose=False), verbose=False)
    assert prepared["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]["ema_span_0"] > 0
    assert "Cannot copy zero strategy span" in caplog.text


@pytest.mark.parametrize("explicit", [False, True])
def test_nested_scenario_spans_migrate_and_explicit_unstuck_wins(explicit):
    from suite_runner import _normalize_scenario_overrides

    c = legacy_config()
    side = {"strategy": {"trailing_martingale": {"ema_span_0": 400.5}}}
    if explicit:
        side["unstuck"] = {"ema_span_0": 91.5}
    c["backtest"]["scenarios"] = [
        {"label": "nested", "overrides": {"bot": {"long": side}}}
    ]
    prepared = prepare_config(c, verbose=False)
    overrides = _normalize_scenario_overrides(
        prepared["backtest"]["scenarios"][0]["overrides"]
    )
    assert overrides["bot.long.unstuck.ema_span_0"] == (91.5 if explicit else 400.5)


@pytest.mark.parametrize("toggle", ["enabled", "ema_gating_enabled"])
def test_disabled_unstuck_warmup_ignores_values_and_bounds_but_honors_coin_activation(
    toggle,
):
    from warmup_utils import compute_backtest_warmup_minutes

    c = get_template_config()
    c["live"].update(warmup_ratio=1.0, max_warmup_minutes=1_000_000)
    for side in ("long", "short"):
        c["bot"][side]["unstuck"][toggle] = False
    baseline = compute_backtest_warmup_minutes(c)
    baseline_coin = compute_per_coin_warmup_minutes(c)["__default__"]
    for side in ("long", "short"):
        c["bot"][side]["unstuck"].update(ema_span_0=400_000.0, ema_span_1=500_000.0)
        c["optimize"]["bounds"][side]["unstuck"].update(
            ema_span_0=[10, 600_000], ema_span_1=[10, 700_000]
        )
    assert compute_backtest_warmup_minutes(c) == baseline
    assert compute_per_coin_warmup_minutes(c)["__default__"] == baseline_coin
    c["coin_overrides"] = {"BTC": {"bot": {"long": {"unstuck": {toggle: True}}}}}
    assert compute_backtest_warmup_minutes(c) == 700_000
    assert compute_per_coin_warmup_minutes(c)["BTC"] == 500_000


@pytest.mark.parametrize("toggle", ["unstuck_enabled", "unstuck_ema_gating_enabled"])
def test_live_warmup_ignores_disabled_unstuck(toggle):
    from passivbot import compute_live_warmup_windows

    # Use the same callable surface as live configuration lookup.
    values = {
        "ema_span_0": 10,
        "ema_span_1": 20,
        "unstuck_ema_span_0": 400_000,
        "unstuck_ema_span_1": 500_000,
        "unstuck_enabled": True,
        "unstuck_ema_gating_enabled": True,
    }
    values[toggle] = False

    def lookup(side, key, symbol):
        return values.get(key, 0.0)

    kwargs = dict(
        symbols_by_side={"long": {"BTC"}, "short": set()},
        bp_lookup=lookup,
        warmup_ratio=1.0,
        max_warmup_minutes=1_000_000,
        forager_enabled={"long": False, "short": False},
        span_buffer=1.0,
    )
    windows, _, _ = compute_live_warmup_windows(**kwargs)
    assert windows["BTC"] == 20


@pytest.mark.parametrize("key", ["ema_span_0", "ema_span_1"])
def test_legacy_fixed_strategy_bound_preserves_optimizer_unstuck_span(key):
    c = legacy_config()
    c["bot"]["long"]["strategy"]["trailing_martingale"][key] = 401.5
    c["optimize"]["bounds"]["long"]["strategy"]["trailing_martingale"][key] = [
        199.25,
        199.25,
    ]
    prepared = prepare_config(c, verbose=False)
    assert prepared["bot"]["long"]["unstuck"][key] == 401.5
    assert prepared["optimize"]["bounds"]["long"]["unstuck"][key] == [199.25, 199.25]


def test_legacy_inactive_zero_fixed_bound_gets_positive_fallback(caplog):
    c = legacy_config()
    c["optimize"]["bounds"]["short"]["strategy"]["trailing_martingale"][
        "ema_span_0"
    ] = [0, 0]
    prepared = prepare_config(c, verbose=False)
    assert prepared["optimize"]["bounds"]["short"]["unstuck"]["ema_span_0"][0] > 0
    assert "optimize.bounds.short.unstuck.ema_span_0" in caplog.text


@pytest.mark.parametrize("key", ["loss_allowance_pct", "close_pct", "threshold"])
@pytest.mark.parametrize("coin_override", [False, True])
def test_gpu_independent_spans_allow_fixed_inactive_reducer_only(key, coin_override):
    from optimization.gpu.unstuck_scope import validate_independent_unstuck_scope

    c = get_template_config()
    # Keep the global pair supported when testing an independent coin-specific pair.
    c["optimize"]["bounds"] = {}
    target = c["bot"]["long"]["unstuck"]
    if coin_override:
        c["coin_overrides"] = {"BTC": {"bot": {"long": {"unstuck": {}}}}}
        target = c["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]
    target.update(ema_span_0=400_000.5)
    target[key] = 0.0
    c["optimize"]["bounds"]["long"] = {"unstuck": {key: [0, 0]}}
    validate_independent_unstuck_scope(c)
    c["optimize"]["bounds"]["long"]["unstuck"][key] = [0, 1]
    if coin_override:
        # A coin pin wins even when the global search can enable the reducer.
        validate_independent_unstuck_scope(c)
        del target[key]
    with pytest.raises(ValueError, match="does not yet model independent"):
        validate_independent_unstuck_scope(c)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("inactive", ["no_slots", "no_exposure", "no_coins"])
def test_gpu_independent_guard_uses_effective_side_eligibility(side, inactive):
    from optimization.gpu.model import gpu_side_enabled
    from optimization.gpu.unstuck_scope import validate_independent_unstuck_scope

    c = get_template_config()
    c["optimize"]["bounds"] = {}
    c["live"]["approved_coins"] = {"long": ["BTC"], "short": ["ETH"]}
    for pside in ("long", "short"):
        c["bot"][pside]["risk"].update(n_positions=1, total_wallet_exposure_limit=1.0)
    c["bot"][side]["unstuck"]["ema_span_0"] = 400_000.5
    if inactive == "no_slots":
        c["bot"][side]["risk"]["n_positions"] = 0
    elif inactive == "no_exposure":
        c["bot"][side]["risk"]["total_wallet_exposure_limit"] = 0.0
    else:
        c["live"]["approved_coins"][side] = []
    assert not gpu_side_enabled(c, side)
    assert gpu_side_enabled(c, "short" if side == "long" else "long")
    validate_independent_unstuck_scope(c)
    c["bot"][side]["risk"].update(n_positions=1, total_wallet_exposure_limit=1.0)
    c["live"]["approved_coins"][side] = ["BTC"]
    with pytest.raises(ValueError, match="does not yet model independent"):
        validate_independent_unstuck_scope(c)


@pytest.mark.parametrize("control", ["loss_allowance_pct", "close_pct", "threshold"])
def test_inactive_unstuck_warmup_respects_optimizer_reactivation_and_coin_pins(control):
    from warmup_utils import compute_backtest_warmup_minutes

    c = get_template_config()
    c["live"].update(warmup_ratio=1.0, max_warmup_minutes=1_000_000)
    c["bot"]["short"]["unstuck"]["enabled"] = False
    c["bot"]["long"]["unstuck"][control] = 0.0
    c["optimize"]["bounds"]["long"]["unstuck"][control] = [0, 0, 1]
    baseline = compute_backtest_warmup_minutes(c)
    baseline_coin = compute_per_coin_warmup_minutes(c)["__default__"]
    c["bot"]["long"]["unstuck"].update(ema_span_0=400_000, ema_span_1=500_000)
    c["optimize"]["bounds"]["long"]["unstuck"].update(
        ema_span_0=[10, 600_000], ema_span_1=[10, 700_000]
    )
    assert compute_backtest_warmup_minutes(c) == baseline
    assert compute_per_coin_warmup_minutes(c)["__default__"] == baseline_coin
    c["optimize"]["bounds"]["long"]["unstuck"][control] = [0, 0.1]
    c["coin_overrides"] = {"BTC": {"bot": {"long": {"unstuck": {control: 0}}}}}
    assert compute_backtest_warmup_minutes(c) == 700_000
    assert compute_per_coin_warmup_minutes(c)["__default__"] == 500_000
    assert compute_per_coin_warmup_minutes(c)["BTC"] == baseline_coin
    c["optimize"]["bounds"]["long"]["unstuck"][control] = [0, 0]
    c["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"][control] = 0.1
    assert compute_backtest_warmup_minutes(c) == 700_000
    assert compute_per_coin_warmup_minutes(c)["BTC"] == 500_000


def test_config_version_help_matches_canonical_schema():
    from config.schema import CONFIG_SCHEMA_VERSION
    from config_utils import CLI_HELP_OVERRIDES

    assert CONFIG_SCHEMA_VERSION in CLI_HELP_OVERRIDES["config_version"]
