from config.universe import require_side_universe_sources
from config.universe import set_side_universe_sources
from copy import deepcopy
import json
import logging
from pathlib import Path

import passivbot_rust as pbr
import pytest

from config import (
    CONFIG_SCHEMA_VERSION,
    compile_runtime_config,
    get_template_config,
    load_prepared_config,
    prepare_config,
    project_config,
)
from config.optimize_bounds import flatten_optimize_bounds, get_optimize_bounds_defaults
from config.overrides import parse_overrides
from config.strategy import build_runtime_strategy_side
from config.strategy_spec import get_supported_strategy_kinds
from config.strategy_paths import project_native_strategy_side, project_strategy_spec


def _set_path(mapping, path, value):
    current = mapping
    for part in path[:-1]:
        current = current.setdefault(part, {})
    current[path[-1]] = value


def _nested_strategy_values_from_spec(spec, field):
    result = {"long": {}, "short": {}}
    for param in [*spec["parameters"], *spec.get("fixed_parameters", [])]:
        config_path = param["config_path"]
        pside = config_path[1]
        _set_path(result[pside], config_path[2:], param[field])
    return result


def _strategy_side(config, pside, kind=None):
    if kind is None:
        kind = config['bot']['global']['strategy_kind']
    return config["bot"][pside]["strategy"][kind]


def test_shipped_ema_anchor_smoke_example_loads_with_preserved_inputs():
    path = Path(__file__).resolve().parents[1] / "configs/examples/ema_anchor_smoke_test.json"
    authored = json.loads(path.read_text())
    original = deepcopy(authored)
    config = load_prepared_config(str(path), verbose=False, log_info=False)
    compiled = compile_runtime_config(config, runtime="backtest")

    assert authored["config_version"] == config["config_version"] == CONFIG_SCHEMA_VERSION
    assert config['bot']['global']['strategy_kind'] == "ema_anchor"
    assert config["live"]["account"]["user"] == "example"
    # This explicit demonstration span is not a converted rolling-window policy.
    assert config["bot"]["global"]["unstuck"]["memory_ema_span_minutes"] == 43200.0
    assert config["live"]["startup"]["risk_bootstrap_max_lookback_minutes"] == 129600.0
    assert config["bot"]["global"]["loss_gate"]["max_drawdown_ratio"] == 1.0
    assert config["bot"]["global"]["hsl"]["scope"] == "coin"
    assert require_side_universe_sources(config, 'approved_coins') == {
        "long": ["BTC", "SOL"], "short": ["BTC", "SOL"],
    }
    for key, expected in {
        "start_date": "2023-01-01", "end_date": "2023-01-15",
        "exchanges": ["binance", "bybit"], "candle_interval_minutes": 5,
        "starting_balance": 100000.0, "maker_fee_override": 0.0004,
        "btc_collateral_cap": 0.0, "suite_enabled": False,
    }.items():
        assert config["backtest"][key] == expected
    expected_native_strategy = {
        "base_qty_pct": 0.04, "ema_span_0": 48.0, "ema_span_1": 288.0,
        "offset": 0.003, "offset_volatility_ema_span_1m": 60.0,
        "offset_volatility_1m_weight": 0.0, "offset_volatility_ema_span_1h": 24.0,
        "offset_volatility_1h_weight": 0.0, "offset_psize_weight": 0.2,
    }
    expected_strategy = project_native_strategy_side("ema_anchor", expected_native_strategy)
    for side in ("long", "short"):
        assert config["bot"][side]["hsl"]["enabled"] is False
        assert config["bot"][side]["risk"]["max_positions"] == 2.0
        assert config["bot"][side]["risk"]["total_wallet_exposure_limit"] == 1.0
        for key, expected in expected_strategy.items():
            assert _strategy_side(config, side)[key] == expected
            assert _strategy_side(compiled, side)[key] == expected
        native_strategy = build_runtime_strategy_side(
            _strategy_side(compiled, side), strategy_kind="ema_anchor", pside=side,
        )
        for key, expected in expected_native_strategy.items():
            assert native_strategy[key] == expected
        assert config["bot"][side]["unstuck"]["ema_span_0_minutes"] == 48.0
        assert config["bot"][side]["unstuck"]["ema_span_1_minutes"] == 288.0
    for retired in ("pnls_max_lookback_days", "realized_loss_gate_lookback_minutes",
                    "no_restart_drawdown_threshold", "orange_tier_mode", "tier_ratios"):
        assert retired not in path.read_text()
    assert json.loads(path.read_text()) == original


def test_startup_phase_budgets_validate_and_roundtrip():
    source = get_template_config()
    assert source["live"]["startup"]["startup_phase_budgets"] == {}
    source["live"]["startup"]["startup_phase_budgets"] = {
        "account": {"elapsed_milliseconds": 120_000, "since_previous_milliseconds": 60_000},
        "full-warmup": {"elapsed_milliseconds": 900_000},
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert (
        prepared["live"]["startup"]["startup_phase_budgets"]
        == source["live"]["startup"]["startup_phase_budgets"]
    )


def test_prepare_config_preserves_fixed_runtime_override_mapping():
    source = get_template_config()
    source["optimize"]["fixed_runtime_overrides"] = {
        "bot.long.strategy.trailing_martingale.entry.threshold_base_fraction": 0.123,
        "bot.short.hsl.drawdown_ratio_threshold": 0.1,
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert (
        prepared["optimize"]["fixed_runtime_overrides"]
        == source["optimize"]["fixed_runtime_overrides"]
    )


@pytest.mark.parametrize("location", ["fixed", "scenario"])
@pytest.mark.parametrize("path", [
    "bot.long.unstuck_ema_span_0",
    "bot.long.strategy.trailing_martingale.ema_span_0",
    "long.unstuck.ema_span_0",
    "bot.long.strategy.entry.ema_span_0",
    "long.strategy.entry.ema_span_0",
])
def test_authored_patches_reject_retired_paths_and_selector_shorthand(location, path):
    source = get_template_config()
    if location == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = {path: 12.5}
    else:
        source["backtest"]["scenarios"] = [{"label": "invalid", "overrides": {path: 12.5}}]
    with pytest.raises(ValueError, match="unknown configuration field"):
        prepare_config(source, verbose=False)


@pytest.mark.parametrize("location", ["fixed", "scenario"])
def test_authored_patches_accept_literal_canonical_paths(location):
    source = get_template_config()
    patch = {"bot.long.unstuck.ema_span_0_minutes": 12.5,
             "bot.long.strategy.trailing_martingale.entry.ema_span_0_minutes": 25.0}
    if location == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = patch
    else:
        source["backtest"]["scenarios"] = [{"label": "canonical", "overrides": patch}]
    prepared = prepare_config(source, verbose=False)
    if location == "fixed":
        assert prepared["optimize"]["fixed_runtime_overrides"] == patch
    else:
        assert prepared["backtest"]["scenarios"][0]["overrides"] == patch


@pytest.mark.parametrize("source", [None, "", "all", "coins.json", [], ["BTC", "ETH"],
    {}, {"long": "all", "short": ["BTC"]}, {"long": "coins.json", "short": None}])
@pytest.mark.parametrize("field", ["approved_coins", "ignored_coins"])
def test_authored_coin_sources_preserve_supported_shapes_without_resolving_files(source, field):
    from config.authored import validate_authored_config

    config = get_template_config()
    set_side_universe_sources(config, field, source)
    before = deepcopy(config)
    validate_authored_config(config, get_template_config())
    assert config == before


@pytest.mark.parametrize("source", [{"typo": ["BTC"]}, {"long": {"file": "coins.json"}}, ["BTC", 2], 2])
def test_authored_coin_sources_reject_unknown_shapes(source):
    from config.authored import validate_authored_config

    config = get_template_config()
    with pytest.raises((TypeError, ValueError), match="approved_coins"):
        set_side_universe_sources(config, 'approved_coins', source)
        validate_authored_config(config, get_template_config())


@pytest.mark.parametrize("location", ["fixed", "scenario"])
@pytest.mark.parametrize(
    "path,value", [("bot.long.unstuck.ema_span_0_minutes", 12.5), ("live.account.leverage", 3)]
)
def test_authored_patches_accept_sparse_local_coin_paths(location, path, value):
    source = get_template_config()
    source["coin_overrides"] = {
        "BTC": {
            "bot": {"long": {"unstuck": {"ema_span_0_minutes": 60.25}}},
            "live": {"account": {"leverage": 3}},
        }
    }
    patch = {f"coin_overrides.BTC.{path}": value}
    if location == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = patch
    else:
        source["backtest"]["scenarios"] = [{"label": "local", "overrides": patch}]
    assert prepare_config(source, verbose=False)


@pytest.mark.parametrize("path", ["bot.long.unstuck_ema_span_0", "bot.global.loss_gate.memory_ema_span_minutes",
                                  "bot.long.risk.max_positions", "bot.long.hsl.ema_span_minutes"])
@pytest.mark.parametrize("location", ["fixed", "scenario"])
def test_authored_coin_path_pins_reject_aliases_and_global_policy(path, location):
    source = get_template_config()
    source["coin_overrides"] = {"BTC": {}}
    patch = {f"coin_overrides.BTC.{path}": 12.5}
    if location == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = patch
    else:
        source["backtest"]["scenarios"] = [{"label": "invalid", "overrides": patch}]
    with pytest.raises(ValueError, match="(?:not overridable|retired)"):
        prepare_config(source, verbose=False)


@pytest.mark.parametrize("location", ["config", "runtime"])
def test_canonical_portfolio_hsl_rejects_live_balance_override(location):
    from config.hsl import require_live_balance_support

    config = get_template_config()
    config["bot"]["global"]["hsl"] = {**config["bot"]["long"]["hsl"], "scope": "portfolio", "enabled": True}
    override = 100.0 if location == "runtime" else None
    if location == "config":
        config["live"]["account"]["balance_override"] = 100.0
    with pytest.raises(
        ValueError, match="does not support live.account.balance_override"
    ):
        require_live_balance_support(config, override)


@pytest.mark.parametrize("enabled", [False, True])
def test_portfolio_metric_validation_reads_canonical_global_controller(enabled):
    from config.hsl import validate_optimizer_metrics

    config = get_template_config()
    config["bot"]["global"]["hsl"] = {**config["bot"]["long"]["hsl"], "scope": "portfolio", "enabled": enabled}
    if enabled:
        validate_optimizer_metrics(config, ["hard_stop_restarts"])
    else:
        with pytest.raises(ValueError, match="no enabled controller"):
            validate_optimizer_metrics(config, ["hard_stop_restarts"])


@pytest.mark.parametrize(
    "overrides,error_type,error_match",
    [
        ([], TypeError, "fixed_runtime_overrides must be a dict"),
        ({1: 2}, TypeError, "keys must be strings"),
        ({"bot.long.not_a_setting": 1}, ValueError, "unknown configuration field"),
        (
            {
                "bot.long.risk.total_wallet_exposure_limit": 1.0,
                "bot.long.total_wallet_exposure_limit": 2.0,
            },
            ValueError,
            "unknown configuration field",
        ),
        (
            {"bot.long.strategy.trailing_martingale": {}},
            TypeError,
            "must target leaf settings",
        ),
    ],
)
def test_prepare_config_rejects_invalid_fixed_runtime_overrides(
    overrides, error_type, error_match
):
    source = get_template_config()
    source["optimize"]["fixed_runtime_overrides"] = overrides

    with pytest.raises(error_type, match=error_match):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


@pytest.mark.parametrize(
    "budgets,error_type,error_match",
    [
        ([], TypeError, "startup_phase_budgets must be a dict"),
        ({"unknown": {"elapsed_milliseconds": 1}}, ValueError, "phase must be one of"),
        ({"account": []}, TypeError, "account must be a dict"),
        ({"account": {}}, ValueError, "must define at least one budget"),
        ({"account": {"other_ms": 1}}, ValueError, "unknown keys"),
        ({"account": {"elapsed_milliseconds": True}}, TypeError, "must be an integer"),
        ({"account": {"elapsed_milliseconds": 1.5}}, TypeError, "must be an integer"),
        ({"account": {"elapsed_milliseconds": -1}}, ValueError, "must be >= 0"),
    ],
)
def test_startup_phase_budgets_reject_invalid_values(budgets, error_type, error_match):
    source = get_template_config()
    source["live"]["startup"]["startup_phase_budgets"] = budgets

    with pytest.raises(error_type, match=error_match):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


@pytest.mark.parametrize("strategy_kind", list(get_supported_strategy_kinds()))
def test_rust_strategy_spec_matches_python_template_defaults(strategy_kind):
    source = get_template_config()
    source['bot']['global']['strategy_kind'] = strategy_kind
    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)
    expected = _nested_strategy_values_from_spec(
        project_strategy_spec(strategy_kind, pbr.get_strategy_spec(strategy_kind)), "default"
    )

    assert _strategy_side(prepared, "long", strategy_kind) == expected["long"]
    assert _strategy_side(prepared, "short", strategy_kind) == expected["short"]


@pytest.mark.parametrize("strategy_kind", list(get_supported_strategy_kinds()))
def test_rust_strategy_spec_matches_generated_strategy_optimize_bounds(strategy_kind):
    spec = pbr.get_strategy_spec(strategy_kind)
    expected = spec["optimize_bounds"]
    generated = get_optimize_bounds_defaults()

    flat_generated = flatten_optimize_bounds(
        {side: {"strategy": {strategy_kind: generated[side]["strategy"][strategy_kind]}}
         for side in ("long", "short")},
        strategy_kind=strategy_kind,
    )

    assert flat_generated == expected


def test_trailing_martingale_ema_gate_mode_is_fixed_config_not_optimizer_bound():
    source = get_template_config()
    source["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "ema_gate_mode"
    ] = "REENTRY"
    source["bot"]["short"]["unstuck"]["ema_gating_enabled"] = False

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert (
        prepared["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
            "ema_gate_mode"
        ]
        == "reentry"
    )
    assert prepared["bot"]["short"]["unstuck"]["ema_gating_enabled"] is False
    assert "ema_gate_mode" not in json.dumps(prepared["optimize"]["bounds"])
    spec = pbr.get_strategy_spec("trailing_martingale")
    assert any(
        param["config_path"] == ["strategy", "long", "entry", "ema_gate_mode"]
        and param["default"] == "all"
        and param["allowed_values"] == ["disabled", "all", "initial", "reentry"]
        for param in spec["fixed_parameters"]
    )


def test_prepare_config_rejects_invalid_trailing_martingale_ema_gate_mode():
    source = get_template_config()
    source["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "ema_gate_mode"
    ] = "re_entry"

    with pytest.raises(ValueError, match="ema_gate_mode must be one of"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


@pytest.mark.parametrize(
    ("target", "expected_sections"),
    [
        (
            "canonical",
            {
                "backtest",
                "bot",
                "coin_overrides",
                "live",
                "logging",
                "monitor",
                "optimize",
            },
        ),
        ("live", {"bot", "coin_overrides", "live", "logging", "monitor"}),
        ("backtest", {"backtest", "bot", "coin_overrides", "live", "logging"}),
        (
            "optimize",
            {"backtest", "bot", "coin_overrides", "live", "logging", "optimize"},
        ),
        ("monitor", {"live", "logging", "monitor"}),
    ],
)
def test_project_config_keeps_only_target_sections_and_metadata(
    target, expected_sections
):
    cfg = get_template_config()
    cfg["_raw"] = {"live": {"account": {"user": "raw_user"}}}
    cfg["_raw_effective"] = {"live": {"account": {"user": "effective_user"}}}
    cfg["_transform_log"] = [{"step": "seed"}]
    cfg["_coins_sources"] = {"approved_coins": "configs/approved_coins.json"}

    projected = project_config(cfg, target)

    metadata_keys = {
        "_raw",
        "_raw_effective",
        "_transform_log",
        "_coins_sources",
        "config_version",
    }
    assert set(projected) == expected_sections | metadata_keys
    for section in (
        "backtest",
        "bot",
        "coin_overrides",
        "live",
        "logging",
        "monitor",
        "optimize",
    ):
        if section in expected_sections:
            assert section in projected
        else:
            assert section not in projected


def test_prepare_config_canonical_omits_runtime_aliases():
    prepared = prepare_config(
        get_template_config(),
        verbose=False,
        target="canonical",
        runtime=None,
    )

    assert "n_positions" not in prepared["bot"]["long"]
    assert "total_wallet_exposure_limit" not in prepared["bot"]["long"]
    assert "risk_wel_enforcer_threshold" not in prepared["bot"]["long"]
    assert "unstuck_threshold" not in prepared["bot"]["long"]
    assert "hsl_red_threshold" not in prepared["bot"]["long"]
    assert "forager_volume_ema_span_1m" not in prepared["bot"]["long"]
    assert "filter_volume_ema_span_1m" not in prepared["bot"]["long"]
    assert "filter_volatility_ema_span_1m" not in prepared["bot"]["long"]
    assert "long_filter_volume_ema_span_1m" not in prepared["optimize"]["bounds"]
    assert "long_filter_volatility_ema_span_1m" not in prepared["optimize"]["bounds"]
    assert prepared['bot']['global']['strategy_kind'] == "trailing_martingale"
    assert "ema_span_0" not in get_template_config()["bot"]["long"]
    assert "entry_grid_spacing_pct" not in get_template_config()["bot"]["short"]
    assert (
        _strategy_side(prepared, "long")["entry"]["ema_span_0_minutes"]
        == _strategy_side(get_template_config(), "long", "trailing_martingale")[
            "entry"
        ]["ema_span_0_minutes"]
    )


def test_prepare_config_rejects_v7_flat_trailing_grid_fields_with_trailing_martingale():
    source = get_template_config()
    source['bot']['global']['strategy_kind'] = "trailing_martingale"
    source["bot"]["long"]["entry_grid_spacing_pct"] = 0.01
    source["bot"]["long"]["entry_trailing_grid_ratio"] = -0.5
    source["bot"]["long"]["close_grid_markup_start"] = 0.01

    with pytest.raises(ValueError, match="unknown configuration field"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


@pytest.mark.parametrize("strategy_kind", ["ema_anchor"])
@pytest.mark.parametrize(
    "flat_key", ["entry_grid_spacing_pct", "entry_volatility_ema_span_1h"]
)
def test_prepare_config_rejects_v7_flat_trailing_grid_fields_with_any_strategy_kind(
    strategy_kind,
    flat_key,
):
    source = get_template_config()
    source['bot']['global']['strategy_kind'] = strategy_kind
    source["bot"]["long"][flat_key] = 0.5

    with pytest.raises(
        ValueError,
        match=rf"bot\.long\.{flat_key}.*unknown configuration field",
    ):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_rejects_old_v7_flat_aliases_with_trailing_martingale():
    source = get_template_config()
    source['bot']['global']['strategy_kind'] = "trailing_martingale"
    source["bot"]["long"]["entry_grid_spacing_weight"] = 0.5
    source["bot"]["long"]["entry_grid_spacing_log_weight"] = 0.6
    source["bot"]["long"]["entry_grid_spacing_log_span_hours"] = 12.0

    with pytest.raises(ValueError, match="unknown configuration field"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


@pytest.mark.parametrize(
    "flat_key",
    [
        "entry_grid_spacing_weight",
        "entry_grid_spacing_pct",
        "entry_volatility_ema_span_1h",
    ],
)
def test_parse_overrides_rejects_v7_flat_strategy_override_keys(flat_key):
    source = get_template_config()
    source['bot']['global']['strategy_kind'] = "trailing_martingale"
    source["coin_overrides"] = {
        "BTC": {
            "bot": {
                "long": {
                    flat_key: 0.5,
                }
            }
        }
    }
    with pytest.raises(ValueError, match=rf"coin_overrides\.BTC\.bot\.long.*unsupported flat strategy.*{flat_key}"):
        prepare_config(source, verbose=False)
    with pytest.raises(ValueError, match=rf"unsupported flat strategy.*{flat_key}"):
        parse_overrides(source, verbose=False, symbol_normalizer=lambda coin: coin)


@pytest.mark.parametrize(
    "bound_key",
    [
        "long_hsl_enabled",
        "short_hsl_enabled",
        "long_hsl_panic_close_order_type",
        "short_hsl_panic_close_order_type",
    ],
)
def test_prepare_config_rejects_nontunable_bot_bounds(bound_key):
    cfg = get_template_config()
    cfg["optimize"]["bounds"][bound_key] = [0.0, 1.0]

    with pytest.raises(
        ValueError, match=rf"optimize.bounds.{bound_key}.*unknown configuration field"
    ):
        prepare_config(cfg, verbose=False, target="canonical", runtime=None)


def test_compile_runtime_config_adds_runtime_aliases_without_removing_canonical_keys():
    canonical = prepare_config(
        get_template_config(),
        verbose=False,
        target="optimize",
        runtime=None,
    )

    compiled = compile_runtime_config(canonical, runtime="optimize")

    assert (
        compiled["bot"]["long"]["n_positions"]
        == canonical["bot"]["long"]["risk"]["max_positions"]
    )
    assert (
        compiled["bot"]["long"]["total_wallet_exposure_limit"]
        == canonical["bot"]["long"]["risk"]["total_wallet_exposure_limit"]
    )
    assert (
        compiled["bot"]["long"]["risk_wel_enforcer_threshold"]
        == canonical['bot']['long']['risk']['position_exposure_enforcer']['threshold_ratio']
    )
    assert (
        compiled["bot"]["long"]["unstuck_threshold"]
        == canonical["bot"]["long"]["unstuck"]["exposure_trigger_ratio"]
    )
    assert (
        compiled["bot"]["long"]["hsl_red_threshold"]
        == canonical["bot"]["long"]["hsl"]["drawdown_ratio_threshold"]
    )
    assert (
        compiled["bot"]["long"]["forager_volume_ema_span_1m"]
        == canonical["bot"]["long"]["forager"]["volume_ema_span_minutes"]
    )
    assert (
        compiled["bot"]["long"]["filter_volume_ema_span_1m"]
        == canonical["bot"]["long"]["forager"]["volume_ema_span_minutes"]
    )
    assert (
        compiled["bot"]["long"]["filter_volatility_ema_span_1m"]
        == canonical["bot"]["long"]["forager"]["volatility_ema_span_minutes"]
    )
    assert compiled["bot"]["long"]["filter_volatility_drop_pct"] == pytest.approx(0.0)
    assert compiled["optimize"]["bounds"] == canonical["optimize"]["bounds"]
    assert (
        _strategy_side(compiled, "long")["entry"]["ema_span_0_minutes"]
        == _strategy_side(canonical, "long")["entry"]["ema_span_0_minutes"]
    )


def test_prepare_config_preserves_nested_strategy_namespace():
    source = get_template_config()
    source["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "ema_span_0_minutes"
    ] = 321.0
    source["bot"]["short"]["strategy"]["trailing_martingale"]["entry"][
        "threshold_base_fraction"
    ] = 0.0123
    source["live"].pop("strategy_kind", None)

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared['bot']['global']['strategy_kind'] == "trailing_martingale"
    assert _strategy_side(prepared, "long")["entry"]["ema_span_0_minutes"] == pytest.approx(
        321.0
    )
    assert _strategy_side(prepared, "short")["entry"][
        "threshold_base_fraction"
    ] == pytest.approx(0.0123)


def test_prepare_config_supports_ema_anchor_canonical_strategy_section():
    source = get_template_config()
    source['bot']['global']['strategy_kind'] = "ema_anchor"
    source["bot"]["long"]["entry_cooldown"]["base_duration_minutes"] = 2.5
    source["bot"]["long"]["strategy"]["ema_anchor"] = {
        "base_qty_fraction": 0.02,
        "ema_span_0_minutes": 55.0,
        "ema_span_1_minutes": 144.0,
        "entry_double_down_factor": 0.8,
        "offset_fraction": 0.003,
        "offset_psize_weight": 0.2,
    }
    source["bot"]["short"]["strategy"]["ema_anchor"] = {
        "base_qty_fraction": 0.03,
        "ema_span_0_minutes": 34.0,
        "ema_span_1_minutes": 89.0,
        "entry_double_down_factor": 0.4,
        "offset_fraction": 0.004,
        "offset_psize_weight": 0.1,
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)
    compiled = compile_runtime_config(prepared, runtime="backtest")

    assert prepared['bot']['global']['strategy_kind'] == "ema_anchor"
    assert prepared["bot"]["long"]["entry_cooldown"]["base_duration_minutes"] == pytest.approx(2.5)
    assert _strategy_side(prepared, "long")["base_qty_fraction"] == pytest.approx(0.02)
    assert _strategy_side(prepared, "long")[
        "entry_double_down_factor"
    ] == pytest.approx(0.8)
    assert _strategy_side(prepared, "short")["offset_fraction"] == pytest.approx(0.004)
    assert _strategy_side(prepared, "short")[
        "entry_double_down_factor"
    ] == pytest.approx(0.4)
    assert "base_qty_pct" not in compiled["bot"]["long"]
    assert "offset" not in compiled["bot"]["short"]
    assert compiled["bot"]["long"]["risk_entry_cooldown_minutes"] == pytest.approx(2.5)


def test_prepare_config_hydrates_ema_anchor_defaults_when_strategy_section_missing():
    source = get_template_config()
    source['bot']['global']['strategy_kind'] = "ema_anchor"
    source["bot"]["long"]["strategy"] = {}
    source["bot"]["short"]["strategy"] = {}

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared['bot']['global']['strategy_kind'] == "ema_anchor"
    assert _strategy_side(prepared, "long")["base_qty_fraction"] == pytest.approx(0.01)
    assert _strategy_side(prepared, "long")["ema_span_0_minutes"] == pytest.approx(200.0)
    assert _strategy_side(prepared, "long")[
        "entry_double_down_factor"
    ] == pytest.approx(0.0)
    assert _strategy_side(prepared, "short")["offset_fraction"] == pytest.approx(0.002)
    assert "base_qty_pct" not in prepared["bot"]["long"]


def test_prepare_config_rejects_negative_entry_cooldown_minutes():
    source = get_template_config()
    source["bot"]["long"]["entry_cooldown"]["base_duration_minutes"] = -0.1

    with pytest.raises(ValueError, match="bot.long.entry_cooldown.base_duration_minutes"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_rejects_positive_twel_with_zero_positions():
    source = get_template_config()
    risk = source["bot"]["long"]["risk"]
    risk["total_wallet_exposure_limit"] = 1.0
    risk["max_positions"] = 0
    risk['total_exposure_enforcer']['enabled'] = False

    with pytest.raises(
        ValueError,
        match=(
            r"bot\.long\.risk\.max_positions must be > 0 when "
            r"bot\.long\.risk\.total_wallet_exposure_limit is > 0"
        ),
    ):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


@pytest.mark.parametrize("n_positions", [-1, -0.1])
def test_prepare_config_rejects_negative_positions_even_when_side_disabled(n_positions):
    source = get_template_config()
    risk = source["bot"]["short"]["risk"]
    risk["total_wallet_exposure_limit"] = 0.0
    risk["max_positions"] = n_positions

    with pytest.raises(ValueError, match=r"bot\.short\.risk\.max_positions"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_rejects_retired_bounded_selector():
    source = get_template_config()
    source["bot"]["long"]["risk"]["we_excess_allowance_mode"] = "BOUNDED"
    with pytest.raises(ValueError, match="we_excess_allowance_mode.*unknown configuration field"):
        prepare_config(source, verbose=False)


def test_prepare_config_rejects_retired_raw_coin_override():
    source = get_template_config()
    source["coin_overrides"] = {"BTC": {"bot": {"long": {"risk": {"we_excess_allowance_mode": "LEGACY_RAW"}}}}}
    with pytest.raises(ValueError, match="coin_overrides.BTC.*not overridable"):
        prepare_config(source, verbose=False)


def test_parse_overrides_rejects_retired_raw_coin_override():
    source = get_template_config()
    source["coin_overrides"] = {"BTC": {"bot": {"long": {"risk": {"we_excess_allowance_mode": "LEGACY_RAW"}}}}}
    with pytest.raises(ValueError, match="coin_overrides.BTC.*not overridable"):
        parse_overrides(source, verbose=False)


def test_prepare_config_rejects_invalid_we_excess_allowance_mode():
    source = get_template_config()
    source["bot"]["long"]["risk"]["we_excess_allowance_mode"] = "raw"

    with pytest.raises(ValueError, match="bot.long.risk.we_excess_allowance_mode"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_rejects_invalid_coin_override_we_excess_allowance_mode():
    source = get_template_config()
    source["coin_overrides"] = {
        "BTC": {
            "bot": {
                "long": {
                    "risk": {"we_excess_allowance_mode": "raw"},
                }
            }
        }
    }

    with pytest.raises(
        ValueError,
        match="coin_overrides.BTC.bot.long.risk.we_excess_allowance_mode",
    ):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_target_and_runtime_preserve_raw_metadata_and_record_steps():
    source = get_template_config()
    source["live"]["account"]["user"] = "override_user"
    raw_snapshot = deepcopy(get_template_config())

    prepared = prepare_config(
        source,
        verbose=False,
        target="live",
        runtime="live",
        raw_snapshot=raw_snapshot,
        effective_snapshot=deepcopy(source),
    )

    assert (
        prepared["_raw"]["live"]["account"]["user"]
        == raw_snapshot["live"]["account"]["user"]
    )
    assert prepared["_raw_effective"]["live"]["account"]["user"] == "override_user"
    steps = [entry["step"] for entry in prepared["_transform_log"]]
    assert "normalize_config" in steps
    assert "project_config" in steps
    assert "compile_runtime_config" in steps
    assert "backtest" not in prepared
    assert "optimize" not in prepared
    assert "monitor" in prepared
    assert "strategy" in prepared["bot"]["long"]


def test_load_prepared_config_without_path_uses_schema_defaults_pipeline():
    prepared = load_prepared_config(
        None,
        verbose=False,
        target="backtest",
        runtime="backtest",
        log_info=False,
    )

    template = get_template_config()
    assert (
        prepared["backtest"]["market_order_slippage_fraction"]
        == template["backtest"]["market_order_slippage_fraction"]
    )
    assert (
        prepared["bot"]["long"]["filter_volume_ema_span_1m"]
        == template["bot"]["long"]["forager"]["volume_ema_span_minutes"]
    )
    assert (
        _strategy_side(prepared, "long")["entry"]["ema_span_0_minutes"]
        == _strategy_side(template, "long", "trailing_martingale")["entry"][
            "ema_span_0_minutes"
        ]
    )
    assert (
        prepared["backtest"]["visible_metrics"]
        == template["backtest"]["visible_metrics"]
    )
    assert prepared["_raw"] == template
    assert prepared["_raw_effective"] == template


def test_load_prepared_config_accepts_rounded_forager_weights_from_saved_artifact(
    tmp_path,
):
    source = get_template_config()
    source['bot']['global']['strategy_kind'] = "ema_anchor"
    rounded = {
        "volume": 0.323,
        "ema_readiness": 0.434,
        "volatility": 0.242,
    }
    source["bot"]["long"]["forager"]["score_weights"] = rounded
    source["bot"]["short"]["forager"]["score_weights"] = rounded
    path = tmp_path / "rounded_artifact_like.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    prepared = load_prepared_config(str(path), verbose=False, log_info=False)

    assert prepared["bot"]["long"]["forager"]["score_weights"][
        "volume"
    ] == pytest.approx(0.3233233233233233)
    assert prepared["bot"]["short"]["forager"]["score_weights"][
        "ema_readiness"
    ] == pytest.approx(0.4344344344344344)


def test_prepare_config_preserves_backtest_visible_metrics():
    source = {
        "config_version": CONFIG_SCHEMA_VERSION,
        "backtest": {
            "visible_metrics": [
                "gain",
                "drawdown_worst_hsl",
                "hard_stop_restarts_short",
            ]
        },
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["backtest"]["visible_metrics"] == [
        "gain",
        "drawdown_worst_hsl",
        "hard_stop_restarts_short",
    ]


def test_prepare_config_rejects_unknown_backtest_visible_metrics():
    source = {
        "config_version": CONFIG_SCHEMA_VERSION,
        "backtest": {"visible_metrics": ["not_a_metric"]},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    with pytest.raises(ValueError, match="unknown backtest.visible_metrics entries"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_rejects_invalid_backtest_visible_metrics_type():
    source = {
        "config_version": CONFIG_SCHEMA_VERSION,
        "backtest": {"visible_metrics": "adg"},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    with pytest.raises(
        ValueError, match="backtest.visible_metrics must be null, \\[\\], or"
    ):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_rejects_unversioned_configs_without_explicit_migration():
    source = {
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    with pytest.raises(ValueError, match="explicit migration"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


@pytest.mark.parametrize(
    "previous_version", ["v8.0.0", "v8.1.0", "v8.2.0", "v8.3.0", "v8.4.0"]
)
def test_prepare_config_requires_explicit_previous_schema_migration(
    previous_version,
):
    source = {
        "config_version": previous_version,
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    with pytest.raises(ValueError, match="explicit migration"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


@pytest.mark.parametrize("version,message", [
    ("v9.1.999", "newer than supported schema"),
    ("v9.0.999", "newer than supported schema"),
])
def test_prepare_config_rejects_unreleased_same_major_schema(version, message):
    source = {
        "config_version": version,
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    with pytest.raises(ValueError, match=message):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_rejects_future_config_version():
    source = {
        "config_version": "v10.0.0",
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    with pytest.raises(ValueError, match="newer than supported schema"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_rejects_malformed_config_version():
    source = {
        "config_version": "banana",
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    with pytest.raises(ValueError, match="must be a semantic version"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_preserves_independent_memory_spans():
    source = get_template_config()
    source["bot"]["global"]["unstuck"]["memory_ema_span_minutes"] = 30.5
    source["bot"]["global"]["loss_gate"]["memory_ema_span_minutes"] = 60.5
    source["bot"]["long"]["hsl"]["memory_ema_span_minutes"] = 90.5
    prepared = prepare_config(source, verbose=False)
    assert prepared["bot"]["global"]["unstuck"]["memory_ema_span_minutes"] == 30.5
    assert prepared["bot"]["global"]["loss_gate"]["memory_ema_span_minutes"] == 60.5
    assert prepared["bot"]["long"]["hsl"]["memory_ema_span_minutes"] == 90.5
    assert "lookback_ema_span_minutes" not in prepared["live"]


def test_prepare_config_rejects_retired_all_history_gate_window():
    source = {
        "config_version": CONFIG_SCHEMA_VERSION,
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {"realized_loss_gate_lookback_minutes": -1.0},
        "optimize": {"bounds": {}},
    }

    with pytest.raises(ValueError, match="realized_loss_gate_lookback_minutes.*removed"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_rejects_invalid_memory_ema_span_minutes_string():
    source = get_template_config()
    source["bot"]["global"]["unstuck"]["memory_ema_span_minutes"] = "everything"
    with pytest.raises(ValueError, match="bot.global.unstuck.memory_ema_span_minutes must be a finite number"):
        prepare_config(source, verbose=False)


@pytest.mark.parametrize(
    ("pside", "value", "expected"),
    [
        ("long", -1.0, r"bot\.long\.unstuck\.ema_distance_fraction must be > -1\.0"),
        ("short", 1.0, r"bot\.short\.unstuck\.ema_distance_fraction must be < 1\.0"),
    ],
)
def test_prepare_config_rejects_invalid_unstuck_ema_dist(pside, value, expected):
    source = {
        "config_version": CONFIG_SCHEMA_VERSION,
        "backtest": {},
        "bot": {
            "long": {},
            "short": {},
        },
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }
    source["bot"][pside]["unstuck"] = {"ema_distance_fraction": value}

    with pytest.raises(ValueError, match=expected):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_keeps_live_market_orders_allowed_without_backtest_override():
    source = {
        "config_version": CONFIG_SCHEMA_VERSION,
        "backtest": {},
        "bot": {
            "long": {},
            "short": {},
            "global": {
                "execution": {
                    "market_orders_allowed": True
                }
            }
        },
        "coin_overrides": {},
        "live": {},
        "optimize": {
            "bounds": {}
        }
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared['bot']['global']['execution']['market_orders_allowed'] is True
    assert "market_orders_allowed" not in prepared["backtest"]


def test_template_config_defaults_market_orders_disabled():
    template = get_template_config()
    assert template['bot']['global']['execution']['market_orders_allowed'] is False


def test_prepare_config_rejects_cancellations_not_greater_than_creations():
    source = get_template_config()
    source["live"]["execution"]["max_n_creations_per_batch"] = 3
    source["live"]["execution"]["max_n_cancellations_per_batch"] = 3

    with pytest.raises(
        ValueError,
        match=(
            "config\\.live\\.execution\\.max_n_cancellations_per_batch must be greater than config\\.live\\.execution\\.max_n_creations_per_batch"
        ),
    ):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "market_snapshot_ticker_strategy",
            "fast",
            "config\\.live\\.api\\.market_snapshot_ticker_strategy must be one of",
        ),
        (
            "forager_score_hysteresis_pct",
            -0.1,
            "config\\.live\\.forager_score_hysteresis_pct must be finite and >= 0\\.0",
        ),
        (
            "max_active_candle_tail_gap_minutes",
            0,
            "config\\.live\\.candles\\.max_active_candle_tail_gap_minutes must be finite and > 0\\.0",
        ),
        (
            "max_forager_candle_refresh_seconds",
            -1,
            "config\\.live\\.candles\\.max_forager_candle_refresh_seconds must be finite and > 0\\.0",
        ),
        (
            "max_forager_candle_refresh_seconds",
            0,
            "config\\.live\\.candles\\.max_forager_candle_refresh_seconds must be finite and > 0\\.0",
        ),
        (
            "max_forager_candle_refresh_seconds",
            float("inf"),
            "config\\.live\\.candles\\.max_forager_candle_refresh_seconds must be finite and > 0\\.0",
        ),
        (
            "max_forager_candle_refresh_seconds",
            float("nan"),
            "config\\.live\\.candles\\.max_forager_candle_refresh_seconds must be finite and > 0\\.0",
        ),
        (
            "max_forager_candle_refresh_seconds",
            "not-a-number",
            "config\\.live\\.candles\\.max_forager_candle_refresh_seconds must be numeric",
        ),
        (
            "forager_ws_candle_rest_audit_minutes",
            0,
            "config\\.live\\.candles\\.forager_ws_candle_rest_audit_minutes must be finite",
        ),
        (
            "forager_ws_candle_rest_audit_minutes",
            61,
            "config\\.live\\.candles\\.forager_ws_candle_rest_audit_minutes must be finite",
        ),
        (
            "forager_ws_candle_rest_audit_minutes",
            "not-a-number",
            "config\\.live\\.candles\\.forager_ws_candle_rest_audit_minutes must be numeric",
        ),
        (
            "enable_forager_ws_candles",
            "false",
            "config\\.live\\.candles\\.enable_forager_ws_candles must be a boolean",
        ),
    ],
)
def test_prepare_config_rejects_invalid_staged_live_controls(field, value, match):
    source = get_template_config()
    if field == "forager_score_hysteresis_pct":
        source["bot"]["global"]["forager"]["score_hysteresis_margin"] = value
        match = r"bot\.global\.forager\.score_hysteresis_margin.*finite"
    else:
        from config.field_registry import LIVE_OPERATIONAL_PATHS

        _, group, leaf = LIVE_OPERATIONAL_PATHS[field]
        source["live"][group][leaf] = value

    with pytest.raises((TypeError, ValueError), match=match):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_preserves_fractional_hsl_ema_span():
    source = get_template_config()
    source["bot"]["long"]["hsl"]["drawdown_ratio_ema_span_minutes"] = 1.25

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["bot"]["long"]["hsl"]["drawdown_ratio_ema_span_minutes"] == pytest.approx(1.25)


def test_prepare_config_rejects_retired_hsl_terminal_threshold():
    source = get_template_config()
    source["bot"]["long"]["hsl"]["drawdown_ratio_threshold"] = 0.20
    source["bot"]["long"]["hsl"]["no_restart_drawdown_threshold"] = 0.10

    with pytest.raises(ValueError, match="removed in v9"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


@pytest.mark.parametrize(
    ("group", "field", "value", "match"),
    [
        ("hsl", "drawdown_ratio_threshold", 0.0, r"bot\.long\.hsl\.drawdown_ratio_threshold"),
        (
            "hsl",
            "cooldown_after_red_minutes",
            -1.0,
            r"bot\.long\.hsl\.cooldown_after_red_minutes",
        ),
        (
            "risk",
            "we_excess_allowance_fraction",
            -0.01,
            r"bot\.long\.risk\.we_excess_allowance_fraction must be finite and >= 0\.0",
        ),
        (
            "unstuck",
            "close_fraction",
            1.01,
            r"bot\.long\.unstuck\.close_fraction must be finite and <= 1\.0",
        ),
        (
            "unstuck",
            "loss_allowance_fraction",
            float("nan"),
            r"bot\.long\.unstuck\.loss_allowance_fraction must be finite",
        ),
    ],
)
def test_prepare_config_rejects_invalid_hsl_risk_unstuck_numerics(
    group, field, value, match
):
    source = get_template_config()
    source["bot"]["long"][group][field] = value

    with pytest.raises((TypeError, ValueError), match=match):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_preserves_live_candle_budget_controls():
    source = get_template_config()
    source["live"]["startup"]["defer_broad_candle_warmup"] = False
    source["live"]["startup"]["force_cold_startup"] = True
    source["live"]["candles"]["max_forager_candle_staleness_minutes"] = 7.5

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["live"]["startup"]["defer_broad_candle_warmup"] is False
    assert prepared["live"]["startup"]["force_cold_startup"] is True
    assert prepared["live"]["candles"]["max_forager_candle_staleness_minutes"] == 7.5


def test_prepare_config_preserves_live_fill_refresh_economy_controls():
    source = get_template_config()
    source["live"]["fills"]["fills_recent_overlap_minutes"] = 3.5
    source["live"]["fills"]["fills_confirmation_overlap_minutes"] = 45.0

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["live"]["fills"]["fills_recent_overlap_minutes"] == pytest.approx(
        3.5
    )
    assert prepared["live"]["fills"][
        "fills_confirmation_overlap_minutes"
    ] == pytest.approx(45.0)


def test_prepare_config_preserves_live_custom_endpoints_path():
    source = get_template_config()
    source["live"]["api"]["custom_endpoints_path"] = "configs/custom_endpoints.json"

    prepared = prepare_config(source, verbose=False, target="live", runtime="live")

    assert (
        prepared["live"]["api"]["custom_endpoints_path"]
        == "configs/custom_endpoints.json"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("market_orders_allowed", False),
        ("market_order_near_touch_threshold", 0.0),
        ("lookback_ema_span_minutes", 0.0),
        ("realized_loss_gate_lookback_minutes", 0.0),
    ],
)
def test_prepare_config_rejects_backtest_inherited_live_fields(field, value):
    source = {
        "config_version": CONFIG_SCHEMA_VERSION,
        "backtest": {field: value},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    with pytest.raises(ValueError, match=f"backtest\\.{field}"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_rejects_legacy_backtest_market_slippage_key():
    source = get_template_config()
    source["backtest"]["panic_market_slippage_pct"] = 0.0015
    with pytest.raises(ValueError, match="panic_market_slippage_pct.*unknown configuration field"):
        prepare_config(source, verbose=False)


def test_prepare_config_rejects_legacy_empty_means_all_approved():
    source = get_template_config()
    source["live"]["empty_means_all_approved"] = True
    with pytest.raises(ValueError, match="empty_means_all_approved.*unknown configuration field"):
        prepare_config(source, verbose=False)


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("location", ["root", "coin"])
def test_prepare_config_rejects_retired_entry_grid_inflation(enabled, location):
    source = get_template_config()
    if location == "root":
        source["bot"]["long"]["entry_grid_inflation_enabled"] = enabled
    else:
        source["coin_overrides"] = {"BTC": {"bot": {"long": {"entry_grid_inflation_enabled": enabled}}}}
    with pytest.raises(ValueError, match="entry_grid_inflation_enabled.*(?:unknown configuration field|not overridable)"):
        prepare_config(source, verbose=False)


def test_prepare_config_sparse_canonical_bot_omissions_hydrate_defaults(caplog):
    source = get_template_config()
    trailing_martingale = source["bot"]["long"]["strategy"]["trailing_martingale"]
    risk = source["bot"]["long"]["risk"]
    trailing_martingale.pop("volatility_ema_span_1h_candles")
    trailing_martingale.pop("volatility_ema_span_minutes")
    for key in (
        "threshold_volatility_1h_weight",
        "threshold_volatility_1m_weight",
        "threshold_we_weight",
    ):
        trailing_martingale["entry"].pop(key)
    risk["total_exposure_enforcer"].pop("policy")
    risk["total_exposure_enforcer"].pop("threshold_ratio")
    risk["position_exposure_enforcer"].pop("threshold_ratio")
    risk.pop("we_excess_allowance_fraction")

    with caplog.at_level(logging.INFO):
        prepared = prepare_config(
            source, verbose=True, target="canonical", runtime=None
        )

    long_strategy = _strategy_side(prepared, "long")
    long_risk = prepared["bot"]["long"]["risk"]
    template_strategy = _strategy_side(get_template_config(), "long")
    template_risk = get_template_config()["bot"]["long"]["risk"]
    assert long_strategy["volatility_ema_span_1h_candles"] == pytest.approx(
        template_strategy["volatility_ema_span_1h_candles"]
    )
    assert long_strategy["volatility_ema_span_minutes"] == pytest.approx(
        template_strategy["volatility_ema_span_minutes"]
    )
    assert long_strategy["entry"]["threshold_volatility_1h_weight"] == pytest.approx(
        template_strategy["entry"]["threshold_volatility_1h_weight"]
    )
    assert long_strategy["entry"]["threshold_volatility_1m_weight"] == pytest.approx(
        template_strategy["entry"]["threshold_volatility_1m_weight"]
    )
    assert long_strategy["entry"]["threshold_we_weight"] == pytest.approx(
        template_strategy["entry"]["threshold_we_weight"]
    )
    assert long_risk['total_exposure_enforcer']['threshold_ratio'] == pytest.approx(
        template_risk['total_exposure_enforcer']['threshold_ratio']
    )
    assert "total_exposure_entry_gate_enabled" not in long_risk
    assert long_risk['total_exposure_enforcer']['policy'] == "reduce_overweight"
    assert long_risk["we_excess_allowance_fraction"] == pytest.approx(
        template_risk["we_excess_allowance_fraction"]
    )
    assert long_risk['position_exposure_enforcer']['threshold_ratio'] == pytest.approx(
        template_risk['position_exposure_enforcer']['threshold_ratio']
    )


def test_prepare_config_normalizes_twel_policy_and_runtime_flattening():
    source = get_template_config()
    source['bot']['long']['risk']['total_exposure_enforcer']['policy'] = "REDUCE_PORTFOLIO"

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)
    compiled = compile_runtime_config(prepared)

    assert (
        prepared['bot']['long']['risk']['total_exposure_enforcer']['policy']
        == "reduce_portfolio"
    )
    assert compiled["bot"]["long"]["risk_twel_enforcer_policy"] == "reduce_portfolio"
    assert "risk_twel_entry_gate_enabled" not in compiled["bot"]["long"]


def test_prepare_config_rejects_invalid_twel_policy():
    source = get_template_config()
    source['bot']['long']['risk']['total_exposure_enforcer']['policy'] = "least_stuck"

    with pytest.raises(
        ValueError, match="bot.long.risk.total_exposure_enforcer.policy"
    ):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_twel_policy_and_entry_gate_are_not_optimizer_bounds():
    bounds = get_optimize_bounds_defaults()

    for pside in ("long", "short"):
        risk_bounds = bounds[pside]["risk"]
        assert "total_exposure_entry_gate_enabled" not in risk_bounds
        assert "total_exposure_enforcer_policy" not in risk_bounds


def test_load_fake_live_hsl_config_keeps_disabled_sparse_side_loadable():
    from hsl_fixture import load_fake_hsl_config

    prepared = prepare_config(load_fake_hsl_config(), verbose=False, target="live")

    assert prepared["bot"]["short"]["risk"]["total_wallet_exposure_limit"] == 0.0
    assert _strategy_side(prepared, "short")["entry"][
        "double_down_factor"
    ] == pytest.approx(0.5)








def test_prepare_config_normalizes_all_zero_long_forager_weights_to_ema_readiness_only():
    source = get_template_config()
    source["bot"]["long"]["forager"]["score_weights"] = {
        "volume": 0.0,
        "ema_readiness": 0.0,
        "volatility": 0.0,
    }
    source["bot"]["long"]["forager"]["volume_ema_span_minutes"] = 0.0
    source["bot"]["long"]["forager"]["volatility_ema_span_minutes"] = 0.0

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["bot"]["long"]["forager"]["score_weights"] == {
        "unilateralness": 0.0,
        "volume": 0.0,
        "ema_readiness": 1.0,
        "volatility": 0.0,
    }
    assert prepared["bot"]["long"]["forager"]["volume_ema_span_minutes"] == 0.0
    assert prepared["bot"]["long"]["forager"]["volatility_ema_span_minutes"] == 0.0


def test_prepare_config_preserves_disabled_all_zero_short_forager_weights():
    source = get_template_config()
    source["bot"]["short"]["risk"]["total_wallet_exposure_limit"] = 0.0
    source["bot"]["short"]["risk"]["max_positions"] = 1
    source["bot"]["short"]["forager"]["score_weights"] = {
        "volume": 0.0,
        "ema_readiness": 0.0,
        "volatility": 0.0,
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["bot"]["short"]["forager"]["score_weights"] == {
        "unilateralness": 0.0,
        "volume": 0.0,
        "ema_readiness": 0.0,
        "volatility": 0.0,
    }
