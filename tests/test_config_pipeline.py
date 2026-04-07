from copy import deepcopy

import pytest

from config import (
    CONFIG_SCHEMA_VERSION,
    compile_runtime_config,
    get_template_config,
    load_prepared_config,
    prepare_config,
    project_config,
)


def _strategy_side(config, pside, kind=None):
    if kind is None:
        kind = config["live"]["strategy_kind"]
    return config["bot"][pside]["strategy"][kind]


@pytest.mark.parametrize(
    ("target", "expected_sections"),
    [
        ("canonical", {"backtest", "bot", "coin_overrides", "live", "logging", "monitor", "optimize"}),
        ("live", {"bot", "coin_overrides", "live", "logging", "monitor"}),
        ("backtest", {"backtest", "bot", "coin_overrides", "live", "logging"}),
        ("optimize", {"backtest", "bot", "coin_overrides", "live", "logging", "optimize"}),
        ("monitor", {"live", "logging", "monitor"}),
    ],
)
def test_project_config_keeps_only_target_sections_and_metadata(target, expected_sections):
    cfg = get_template_config()
    cfg["_raw"] = {"live": {"user": "raw_user"}}
    cfg["_raw_effective"] = {"live": {"user": "effective_user"}}
    cfg["_transform_log"] = [{"step": "seed"}]
    cfg["_coins_sources"] = {"approved_coins": "configs/approved_coins.json"}

    projected = project_config(cfg, target)

    metadata_keys = {"_raw", "_raw_effective", "_transform_log", "_coins_sources", "config_version"}
    assert set(projected) == expected_sections | metadata_keys
    for section in ("backtest", "bot", "coin_overrides", "live", "logging", "monitor", "optimize"):
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

    assert "filter_volume_ema_span" not in prepared["bot"]["long"]
    assert "filter_volatility_ema_span" not in prepared["bot"]["long"]
    assert "long_filter_volume_ema_span" not in prepared["optimize"]["bounds"]
    assert "long_filter_volatility_ema_span" not in prepared["optimize"]["bounds"]
    assert prepared["live"]["strategy_kind"] == "trailing_grid"
    assert "ema_span_0" not in get_template_config()["bot"]["long"]
    assert "entry_grid_spacing_pct" not in get_template_config()["bot"]["short"]
    assert _strategy_side(prepared, "long")["ema_span_0"] == _strategy_side(
        get_template_config(), "long", "trailing_grid"
    )["ema_span_0"]


def test_compile_runtime_config_adds_runtime_aliases_without_removing_canonical_keys():
    canonical = prepare_config(
        get_template_config(),
        verbose=False,
        target="optimize",
        runtime=None,
    )

    compiled = compile_runtime_config(canonical, runtime="optimize")

    assert compiled["bot"]["long"]["forager_volume_ema_span"] == canonical["bot"]["long"]["forager"][
        "volume_ema_span"
    ]
    assert compiled["bot"]["long"]["filter_volume_ema_span"] == canonical["bot"]["long"]["forager"][
        "volume_ema_span"
    ]
    assert compiled["bot"]["long"]["filter_volatility_ema_span"] == canonical["bot"]["long"][
        "forager"
    ]["volatility_ema_span"]
    assert compiled["bot"]["long"]["filter_volatility_drop_pct"] == pytest.approx(0.0)
    assert compiled["optimize"]["bounds"] == canonical["optimize"]["bounds"]
    assert _strategy_side(compiled, "long")["ema_span_0"] == _strategy_side(canonical, "long")[
        "ema_span_0"
    ]


def test_prepare_config_preserves_nested_strategy_namespace():
    source = get_template_config()
    source["bot"]["long"]["strategy"]["trailing_grid"]["ema_span_0"] = 321.0
    source["bot"]["short"]["strategy"]["trailing_grid"]["entry_grid_spacing_pct"] = 0.0123
    source["live"].pop("strategy_kind", None)

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["live"]["strategy_kind"] == "trailing_grid"
    assert _strategy_side(prepared, "long")["ema_span_0"] == pytest.approx(321.0)
    assert _strategy_side(prepared, "short")["entry_grid_spacing_pct"] == pytest.approx(0.0123)


def test_prepare_config_supports_ema_anchor_canonical_strategy_section():
    source = get_template_config()
    source["live"]["strategy_kind"] = "ema_anchor"
    source["bot"]["long"]["strategy"]["ema_anchor"] = {
            "base_qty_pct": 0.02,
            "ema_span_0": 55.0,
            "ema_span_1": 144.0,
            "offset": 0.003,
            "offset_psize_weight": 0.2,
    }
    source["bot"]["short"]["strategy"]["ema_anchor"] = {
            "base_qty_pct": 0.03,
            "ema_span_0": 34.0,
            "ema_span_1": 89.0,
            "offset": 0.004,
            "offset_psize_weight": 0.1,
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)
    compiled = compile_runtime_config(prepared, runtime="backtest")

    assert prepared["live"]["strategy_kind"] == "ema_anchor"
    assert _strategy_side(prepared, "long")["base_qty_pct"] == pytest.approx(0.02)
    assert _strategy_side(prepared, "short")["offset"] == pytest.approx(0.004)
    assert "base_qty_pct" not in compiled["bot"]["long"]
    assert "offset" not in compiled["bot"]["short"]


def test_prepare_config_hydrates_ema_anchor_defaults_when_strategy_section_missing():
    source = get_template_config()
    source["live"]["strategy_kind"] = "ema_anchor"
    source["bot"]["long"]["strategy"] = {}
    source["bot"]["short"]["strategy"] = {}

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["live"]["strategy_kind"] == "ema_anchor"
    assert _strategy_side(prepared, "long")["base_qty_pct"] == pytest.approx(0.01)
    assert _strategy_side(prepared, "long")["ema_span_0"] == pytest.approx(200.0)
    assert _strategy_side(prepared, "short")["offset"] == pytest.approx(0.002)
    assert "base_qty_pct" not in prepared["bot"]["long"]


def test_prepare_config_target_and_runtime_preserve_raw_metadata_and_record_steps():
    source = get_template_config()
    source["live"]["user"] = "override_user"
    raw_snapshot = deepcopy(get_template_config())

    prepared = prepare_config(
        source,
        verbose=False,
        target="live",
        runtime="live",
        raw_snapshot=raw_snapshot,
        effective_snapshot=deepcopy(source),
    )

    assert prepared["_raw"]["live"]["user"] == raw_snapshot["live"]["user"]
    assert prepared["_raw_effective"]["live"]["user"] == "override_user"
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
    assert prepared["backtest"]["market_order_slippage_pct"] == template["backtest"]["market_order_slippage_pct"]
    assert prepared["bot"]["long"]["filter_volume_ema_span"] == template["bot"]["long"]["forager"][
        "volume_ema_span"
    ]
    assert _strategy_side(prepared, "long")["ema_span_0"] == _strategy_side(
        template, "long", "trailing_grid"
    )["ema_span_0"]
    assert prepared["backtest"]["visible_metrics"] is None
    assert prepared["_raw"] == template
    assert prepared["_raw_effective"] == template


def test_prepare_config_preserves_backtest_visible_metrics():
    source = {
        "backtest": {"visible_metrics": ["gain", "drawdown_worst_hsl", "hard_stop_restarts_short"]},
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
        "backtest": {"visible_metrics": "adg"},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    with pytest.raises(ValueError, match="backtest.visible_metrics must be null, \\[\\], or"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_assigns_current_schema_version_to_legacy_configs():
    source = {
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["config_version"] == CONFIG_SCHEMA_VERSION


def test_prepare_config_rejects_future_config_version():
    source = {
        "config_version": "v8.0.0",
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


def test_prepare_config_migrates_pre_v79_backtest_pnls_lookback_override():
    source = {
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {"pnls_max_lookback_days": 30.0},
        "optimize": {"bounds": {}},
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["backtest"]["pnls_max_lookback_days"] == pytest.approx(0.0)
    assert prepared["live"]["pnls_max_lookback_days"] == pytest.approx(30.0)


def test_prepare_config_migrates_pre_v79_backtest_market_orders_allowed_override():
    source = {
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {"market_orders_allowed": True},
        "optimize": {"bounds": {}},
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["backtest"]["market_orders_allowed"] is False
    assert prepared["live"]["market_orders_allowed"] is True


def test_prepare_config_migrates_legacy_backtest_market_slippage_key():
    source = {
        "backtest": {"panic_market_slippage_pct": 0.0015},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["backtest"]["market_order_slippage_pct"] == pytest.approx(0.0015)
    assert "panic_market_slippage_pct" not in prepared["backtest"]


def test_prepare_config_removes_empty_means_all_approved_from_canonical_shape():
    source = {
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {
            "approved_coins": [],
            "ignored_coins": {"long": [], "short": []},
            "empty_means_all_approved": True,
        },
        "optimize": {"bounds": {}},
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert "empty_means_all_approved" not in prepared["live"]
    assert prepared["live"]["approved_coins"] == {"long": ["all"], "short": ["all"]}
    assert prepared["_coins_sources"]["approved_coins"] == "all"
    assert prepared["_raw"]["live"]["empty_means_all_approved"] is True
    assert prepared["_raw_effective"]["live"]["empty_means_all_approved"] is True
