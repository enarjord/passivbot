from copy import deepcopy
import json
import logging

import pytest

from config import (
    CONFIG_SCHEMA_VERSION,
    compile_runtime_config,
    get_template_config,
    load_prepared_config,
    prepare_config,
    project_config,
)


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


@pytest.mark.parametrize(
    "bound_key",
    [
        "long_hsl_enabled",
        "short_hsl_enabled",
        "long_hsl_orange_tier_mode",
        "short_hsl_orange_tier_mode",
        "long_hsl_panic_close_order_type",
        "short_hsl_panic_close_order_type",
    ],
)
def test_prepare_config_rejects_nontunable_bot_bounds(bound_key):
    cfg = get_template_config()
    cfg["optimize"]["bounds"][bound_key] = [0.0, 1.0]

    with pytest.raises(KeyError, match=rf"optimize bound {bound_key} must map to a numeric bot\."):
        prepare_config(cfg, verbose=False, target="canonical", runtime=None)


def test_compile_runtime_config_adds_runtime_aliases_without_removing_canonical_keys():
    canonical = prepare_config(
        get_template_config(),
        verbose=False,
        target="optimize",
        runtime=None,
    )

    compiled = compile_runtime_config(canonical, runtime="optimize")

    assert compiled["bot"]["long"]["forager_volume_ema_span"] == canonical["bot"]["long"]["forager_volume_ema_span"]
    assert compiled["bot"]["long"]["filter_volume_ema_span"] == canonical["bot"]["long"]["forager_volume_ema_span"]
    assert compiled["bot"]["long"]["filter_volatility_ema_span"] == canonical["bot"]["long"]["forager_volatility_ema_span"]
    assert compiled["bot"]["long"]["filter_volatility_drop_pct"] == pytest.approx(0.0)
    assert (
        compiled["optimize"]["bounds"]["long_filter_volume_ema_span"]
        == canonical["optimize"]["bounds"]["long_forager_volume_ema_span"]
    )
    assert (
        compiled["optimize"]["bounds"]["long_filter_volatility_ema_span"]
        == canonical["optimize"]["bounds"]["long_forager_volatility_ema_span"]
    )


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
    assert prepared["backtest"]["visible_metrics"] is None
    assert prepared["bot"]["long"]["filter_volume_ema_span"] == template["bot"]["long"]["forager_volume_ema_span"]
    assert prepared["_raw"] == template
    assert prepared["_raw_effective"] == template


def test_load_prepared_config_accepts_rounded_forager_weights_from_saved_artifact(tmp_path):
    source = get_template_config()
    rounded = {
        "volume": 0.323,
        "ema_readiness": 0.434,
        "volatility": 0.242,
    }
    source["bot"]["long"]["forager_score_weights"] = rounded
    source["bot"]["short"]["forager_score_weights"] = rounded
    path = tmp_path / "rounded_artifact_like.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    prepared = load_prepared_config(str(path), verbose=False, log_info=False)

    assert prepared["bot"]["long"]["forager_score_weights"]["volume"] == pytest.approx(
        0.3233233233233233
    )
    assert prepared["bot"]["short"]["forager_score_weights"]["ema_readiness"] == pytest.approx(
        0.4344344344344344
    )


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


def test_prepare_config_keeps_live_pnls_max_lookback_days_without_backtest_override():
    source = {
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {"pnls_max_lookback_days": 30.0},
        "optimize": {"bounds": {}},
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["live"]["pnls_max_lookback_days"] == pytest.approx(30.0)
    assert "pnls_max_lookback_days" not in prepared["backtest"]


def test_prepare_config_normalizes_all_pnls_max_lookback_days():
    source = {
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {"pnls_max_lookback_days": "ALL"},
        "optimize": {"bounds": {}},
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["live"]["pnls_max_lookback_days"] == "all"


def test_prepare_config_rejects_invalid_pnls_max_lookback_days_string():
    source = {
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {"pnls_max_lookback_days": "everything"},
        "optimize": {"bounds": {}},
    }

    with pytest.raises(ValueError, match="live\\.pnls_max_lookback_days must be >= 0 or 'all'"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


@pytest.mark.parametrize(
    ("pside", "value", "expected"),
    [
        ("long", -1.0, r"bot\.long\.unstuck_ema_dist must be > -1\.0"),
        ("short", 1.0, r"bot\.short\.unstuck_ema_dist must be < 1\.0"),
    ],
)
def test_prepare_config_rejects_invalid_unstuck_ema_dist(pside, value, expected):
    source = {
        "backtest": {},
        "bot": {
            "long": {},
            "short": {},
        },
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }
    source["bot"][pside]["unstuck_ema_dist"] = value

    with pytest.raises(ValueError, match=expected):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_keeps_live_market_orders_allowed_without_backtest_override():
    source = {
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {"market_orders_allowed": True},
        "optimize": {"bounds": {}},
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["live"]["market_orders_allowed"] is True
    assert "market_orders_allowed" not in prepared["backtest"]


def test_prepare_config_rejects_cancellations_not_greater_than_creations():
    source = get_template_config()
    source["live"]["max_n_creations_per_batch"] = 3
    source["live"]["max_n_cancellations_per_batch"] = 3

    with pytest.raises(
        ValueError,
        match=(
            "config\\.live\\.max_n_cancellations_per_batch must be greater than "
            "config\\.live\\.max_n_creations_per_batch"
        ),
    ):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("market_orders_allowed", False),
        ("market_order_near_touch_threshold", 0.0),
        ("pnls_max_lookback_days", 0.0),
    ],
)
def test_prepare_config_rejects_backtest_inherited_live_fields(field, value):
    source = {
        "backtest": {field: value},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    with pytest.raises(ValueError, match=f"backtest\\.{field}"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


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


def test_prepare_config_warns_and_removes_entry_grid_inflation_enabled(caplog):
    source = get_template_config()
    source["bot"]["long"]["entry_grid_inflation_enabled"] = True
    source["bot"]["short"]["entry_grid_inflation_enabled"] = True

    with caplog.at_level(logging.WARNING):
        prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert "entry_grid_inflation_enabled" not in prepared["bot"]["long"]
    assert "entry_grid_inflation_enabled" not in prepared["bot"]["short"]
    assert any(
        "entry_grid_inflation_enabled" in rec.message and "has no effect; removing it" in rec.message
        for rec in caplog.records
    )


def test_prepare_config_legacy_bot_omissions_do_not_backfill_schema_defaults(caplog):
    source = get_template_config()
    for key in [
        "entry_trailing_retracement_volatility_weight",
        "entry_trailing_retracement_we_weight",
        "entry_trailing_threshold_volatility_weight",
        "entry_trailing_threshold_we_weight",
        "entry_volatility_ema_span_hours",
        "risk_twel_enforcer_threshold",
        "risk_we_excess_allowance_pct",
        "risk_wel_enforcer_threshold",
    ]:
        source["bot"]["long"].pop(key)

    with caplog.at_level(logging.INFO):
        prepared = prepare_config(source, verbose=True, target="canonical", runtime=None)

    long_cfg = prepared["bot"]["long"]
    assert long_cfg["entry_trailing_retracement_volatility_weight"] == 0.0
    assert long_cfg["entry_trailing_retracement_we_weight"] == 0.0
    assert long_cfg["entry_trailing_threshold_volatility_weight"] == 0.0
    assert long_cfg["entry_trailing_threshold_we_weight"] == 0.0
    assert long_cfg["entry_volatility_ema_span_hours"] == 0.0
    assert long_cfg["risk_twel_enforcer_threshold"] == 0.0
    assert long_cfg["risk_we_excess_allowance_pct"] == 0.0
    assert long_cfg["risk_wel_enforcer_threshold"] == 0.0
    assert any(
        "hydrating omitted bot.long.risk_wel_enforcer_threshold" in rec.message
        for rec in caplog.records
    )


def test_load_fake_live_hsl_config_keeps_disabled_sparse_side_loadable():
    prepared = load_prepared_config("configs/fake_live_hsl_btc.hjson", verbose=False, target="live")

    assert prepared["bot"]["short"]["total_wallet_exposure_limit"] == 0.0
    assert prepared["bot"]["short"]["entry_trailing_double_down_factor"] == pytest.approx(1.0)


def test_prepare_config_silently_removes_disabled_entry_grid_inflation_flag(caplog):
    source = get_template_config()
    source["bot"]["long"]["entry_grid_inflation_enabled"] = False
    source["bot"]["short"]["entry_grid_inflation_enabled"] = False

    with caplog.at_level(logging.WARNING):
        prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert "entry_grid_inflation_enabled" not in prepared["bot"]["long"]
    assert "entry_grid_inflation_enabled" not in prepared["bot"]["short"]
    assert not any("entry_grid_inflation_enabled" in rec.message for rec in caplog.records)


def test_prepare_config_removes_entry_grid_inflation_flag_in_coin_overrides():
    source = get_template_config()
    source["coin_overrides"] = {
        "BTC": {"bot": {"long": {"entry_grid_inflation_enabled": "false"}}}
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert "entry_grid_inflation_enabled" not in prepared["coin_overrides"]["BTC"]["bot"]["long"]


def test_prepare_config_warns_and_removes_coin_override_entry_grid_inflation(caplog):
    source = get_template_config()
    source["coin_overrides"] = {
        "BTC": {"bot": {"long": {"entry_grid_inflation_enabled": True}}}
    }

    with caplog.at_level(logging.WARNING):
        prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert any(
        "coin_overrides.BTC.bot.long.entry_grid_inflation_enabled" in rec.message
        and "has no effect; removing it" in rec.message
        for rec in caplog.records
    )


def test_prepare_config_normalizes_all_zero_long_forager_weights_to_ema_readiness_only():
    source = get_template_config()
    source["bot"]["long"]["forager_score_weights"] = {
        "volume": 0.0,
        "ema_readiness": 0.0,
        "volatility": 0.0,
    }
    source["bot"]["long"]["forager_volume_ema_span"] = 0.0
    source["bot"]["long"]["forager_volatility_ema_span"] = 0.0
    source["bot"]["long"]["forager_volume_drop_pct"] = 0.0

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["bot"]["long"]["forager_score_weights"] == {
        "volume": 0.0,
        "ema_readiness": 1.0,
        "volatility": 0.0,
    }
    assert prepared["bot"]["long"]["forager_volume_ema_span"] == 0.0
    assert prepared["bot"]["long"]["forager_volatility_ema_span"] == 0.0
