import copy
import logging
import pytest

from config_utils import (
    format_config,
    get_template_config,
)

# Internal helpers were intentionally made top-level; import directly for unit tests
from config_utils import detect_flavor, build_base_config_from_flavor


def _template():
    return get_template_config()


def test_detect_flavor_variants():
    tmpl = _template()

    pb_multi = {
        "user": "u",
        "pnls_max_lookback_days": 30,
        "loss_allowance_pct": 0.1,
        "stuck_threshold": 0.2,
        "unstuck_close_pct": 0.3,
        "TWE_long": 1.0,
        "TWE_short": 1.0,
        "universal_live_config": {"long": {"n_close_orders": 5}, "short": {"n_close_orders": 5}},
        "approved_symbols": ["BTC", "ETH"],
        "ignored_symbols": ["X"],
    }
    assert detect_flavor(pb_multi, tmpl) == "pb_multi"

    v7_legacy = {
        "common": {
            "approved_symbols": ["BTC"],
            "symbol_flags": {"BTC": "--long_mode n"},
        },
        "bot": tmpl["bot"],
        "live": tmpl["live"],
        "optimize": tmpl["optimize"],
        "backtest": tmpl["backtest"],
    }
    assert detect_flavor(v7_legacy, tmpl) == "v7_legacy"

    current = copy.deepcopy(tmpl)
    assert detect_flavor(current, tmpl) == "current"

    nested = {"config": copy.deepcopy(tmpl)}
    assert detect_flavor(nested, tmpl) == "nested_current"

    live_only = {"bot": tmpl["bot"], "live": tmpl["live"]}
    assert detect_flavor(live_only, tmpl) == "live_only"


def test_build_base_config_pb_multi():
    tmpl = _template()
    cfg = {
        "user": "u",
        "pnls_max_lookback_days": 30,
        "loss_allowance_pct": 0.1,
        "stuck_threshold": 0.2,
        "unstuck_close_pct": 0.3,
        "TWE_long": 1.2,
        "TWE_short": 1.5,
        "universal_live_config": {
            "long": {"n_close_orders": 4},
            "short": {"n_close_orders": 5},
        },
        "approved_symbols": ["BTC", "ETH"],
        "ignored_symbols": ["X"],
    }
    base = build_base_config_from_flavor(cfg, tmpl, "pb_multi", verbose=True)
    assert sorted(base["live"]["approved_coins"]) == ["BTC", "ETH"]
    assert sorted(base["live"]["ignored_coins"]) == ["X"]
    # close_grid_qty_pct derived from n_close_orders
    assert pytest.approx(base["bot"]["long"]["close_grid_qty_pct"]) == 1.0 / 4.0
    assert pytest.approx(base["bot"]["short"]["close_grid_qty_pct"]) == 1.0 / 5.0
    # total_wallet_exposure_limit derived from TWE_x when enabled by default
    assert base["bot"]["long"]["total_wallet_exposure_limit"] == 1.2
    assert base["bot"]["short"]["total_wallet_exposure_limit"] == 1.5


def test_build_base_config_v7_legacy():
    tmpl = _template()
    cfg = {
        "common": {
            "approved_symbols": ["BTC"],
            "symbol_flags": {"BTC": "--long_mode n"},
        },
        "bot": tmpl["bot"],
        "live": tmpl["live"],
        "optimize": tmpl["optimize"],
        "backtest": tmpl["backtest"],
    }
    base = build_base_config_from_flavor(cfg, tmpl, "v7_legacy", verbose=True)
    assert base["live"]["approved_coins"] == ["BTC"]
    assert base["live"]["coin_flags"] == {"BTC": "--long_mode n"}


def test_format_config_live_only_adds_sections():
    tmpl = _template()
    live_only = {"bot": tmpl["bot"], "live": tmpl["live"]}
    out = format_config(live_only, verbose=False)
    out_live_only = format_config(live_only, verbose=False, live_only=True)
    # ensure missing sections were added
    assert "optimize" in out and "backtest" in out
    assert "optimize" in out_live_only and "backtest" in out_live_only


def test_format_config_current_roundtrip_basic():
    # Provide a minimally valid current config and ensure format_config returns a dict with
    # expected top-level sections while normalising new exposure controls
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    current["bot"]["long"]["risk_wel_enforcer_threshold"] = 0.99
    current["bot"]["long"]["risk_we_excess_allowance_pct"] = 0.25
    current["bot"]["long"]["risk_twel_enforcer_threshold"] = 0.95
    current["bot"]["short"]["risk_wel_enforcer_threshold"] = 1.02
    current["bot"]["short"]["risk_twel_enforcer_threshold"] = 1.08
    out = format_config(current, verbose=False)
    for k in ["bot", "live", "optimize", "backtest"]:
        assert k in out
    assert isinstance(out["bot"]["long"]["risk_wel_enforcer_threshold"], (int, float))
    assert isinstance(out["bot"]["long"]["risk_we_excess_allowance_pct"], float)
    assert isinstance(out["bot"]["long"]["risk_twel_enforcer_threshold"], (int, float))
    assert isinstance(out["bot"]["short"]["risk_wel_enforcer_threshold"], (int, float))
    assert isinstance(out["bot"]["short"]["risk_twel_enforcer_threshold"], (int, float))
    assert "risk_twel_enforcer_threshold" not in out


def test_format_config_preserves_approved_coins_dict():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    current["live"]["approved_coins"] = {"long": ["BTC"], "short": ["BTC"]}
    out = format_config(current, verbose=False)
    assert out["live"]["approved_coins"] == {"long": ["BTC"], "short": ["BTC"]}


def test_format_config_prunes_unknown_keys_recursively():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    current["live"]["unexpected"] = "value"
    current["bot"]["long"]["foo"] = 1
    current["bot"]["short"]["bar"] = 2
    current["optimize"]["bounds"]["extra"] = [0, 1]
    current["extra_section"] = {"nested": 1}
    out = format_config(current, verbose=False)
    assert "unexpected" not in out["live"]
    assert "foo" not in out["bot"]["long"]
    assert "bar" not in out["bot"]["short"]
    assert "extra" not in out["optimize"]["bounds"]
    assert "extra_section" not in out

def test_format_config_preserves_live_optimize_bounds():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    current["optimize"]["bounds"]["common_equity_hard_stop_loss_red_threshold"] = [0.1, 0.3, 0.01]
    current["optimize"]["bounds"]["common_equity_hard_stop_loss_ema_span_minutes"] = [
        10.0,
        120.0,
        5.0,
    ]

    out = format_config(current, verbose=False)

    assert out["optimize"]["bounds"]["common_equity_hard_stop_loss_red_threshold"] == [
        0.1,
        0.3,
        0.01,
    ]
    assert out["optimize"]["bounds"]["common_equity_hard_stop_loss_ema_span_minutes"] == [
        10.0,
        120.0,
        5.0,
    ]


def test_format_config_normalizes_hsl_position_during_cooldown_policy():
    current = copy.deepcopy(_template())
    current["live"]["hsl_position_during_cooldown_policy"] = "manual_quarantine"
    out = format_config(current, verbose=False, live_only=True)
    assert out["live"]["hsl_position_during_cooldown_policy"] == "manual_quarantine"


def test_format_config_rejects_invalid_hsl_position_during_cooldown_policy():
    current = copy.deepcopy(_template())
    current["live"]["hsl_position_during_cooldown_policy"] = "bad_policy"
    with pytest.raises(ValueError, match="live.hsl_position_during_cooldown_policy"):
        format_config(current, verbose=False, live_only=True)


def test_format_config_current_with_empty_optimize_adds_bounds():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    current["optimize"] = {}

    out = format_config(current, verbose=False, live_only=True)

    assert "bounds" in out["optimize"]
    assert out["optimize"]["bounds"]["long_close_grid_markup_start"] == tmpl["optimize"]["bounds"][
        "long_close_grid_markup_start"
    ]
    assert out["optimize"]["bounds"]["short_close_grid_markup_end"] == tmpl["optimize"]["bounds"][
        "short_close_grid_markup_end"
    ]


def test_format_config_current_with_optimize_missing_bounds_adds_defaults():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    current["optimize"] = {"scoring": ["adg"]}

    out = format_config(current, verbose=False, live_only=True)

    assert out["optimize"]["scoring"] == ["adg"]
    assert "long_close_grid_qty_pct" in out["optimize"]["bounds"]
    assert "short_close_grid_qty_pct" in out["optimize"]["bounds"]


def test_format_config_current_with_missing_bot_side_adds_defaults():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    del current["bot"]["short"]

    out = format_config(current, verbose=False, live_only=True)

    assert out["bot"]["short"]["close_grid_markup_end"] == tmpl["bot"]["short"]["close_grid_markup_end"]
    assert out["bot"]["short"]["n_positions"] == tmpl["bot"]["short"]["n_positions"]


def test_format_config_raises_on_non_dict_optimize_bounds():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    current["optimize"]["bounds"] = []

    with pytest.raises(TypeError, match="config.optimize.bounds must be a dict"):
        format_config(current, verbose=False, live_only=True)


def test_format_config_preserves_legacy_derivations_before_hydration():
    current = copy.deepcopy(_template())
    legacy_per_side = {
        "long": {
            "min_markup": 0.006,
            "markup_range": 0.018,
            "ddown_factor": 1.23,
            "bounds_min_markup": [0.005, 0.02],
            "bounds_close_grid_min_markup": [0.003, 0.015],
        },
        "short": {
            "min_markup": 0.007,
            "markup_range": 0.02,
            "ddown_factor": 1.11,
            "bounds_min_markup": [0.006, 0.021],
            "bounds_close_grid_min_markup": [0.004, 0.016],
        },
    }

    for pside, values in legacy_per_side.items():
        current["bot"][pside]["close_grid_min_markup"] = values["min_markup"]
        current["bot"][pside]["close_grid_markup_range"] = values["markup_range"]
        current["bot"][pside]["entry_grid_double_down_factor"] = values["ddown_factor"]
        current["bot"][pside].pop("close_grid_markup_start", None)
        current["bot"][pside].pop("close_grid_markup_end", None)
        current["bot"][pside].pop("entry_trailing_double_down_factor", None)

        current["optimize"]["bounds"][f"{pside}_min_markup"] = values["bounds_min_markup"]
        current["optimize"]["bounds"][f"{pside}_close_grid_min_markup"] = values[
            "bounds_close_grid_min_markup"
        ]
        current["optimize"]["bounds"].pop(f"{pside}_close_grid_markup_start", None)
        current["optimize"]["bounds"].pop(f"{pside}_close_grid_markup_end", None)

    out = format_config(current, verbose=False, live_only=True)

    for pside, values in legacy_per_side.items():
        assert out["bot"][pside]["close_grid_markup_start"] == pytest.approx(
            values["min_markup"] + values["markup_range"]
        )
        assert out["bot"][pside]["close_grid_markup_end"] == pytest.approx(values["min_markup"])
        assert out["bot"][pside]["entry_trailing_double_down_factor"] == pytest.approx(
            values["ddown_factor"]
        )
        assert out["optimize"]["bounds"][f"{pside}_close_grid_markup_start"] == values[
            "bounds_min_markup"
        ]
        assert out["optimize"]["bounds"][f"{pside}_close_grid_markup_end"] == values[
            "bounds_close_grid_min_markup"
        ]
        assert f"{pside}_min_markup" not in out["optimize"]["bounds"]
        assert f"{pside}_close_grid_min_markup" not in out["optimize"]["bounds"]


def test_format_config_migrates_legacy_forager_volume_key():
    current = copy.deepcopy(_template())
    current["bot"]["long"]["filter_volume_drop_pct"] = 0.42
    current["optimize"]["bounds"]["long_filter_volume_drop_pct"] = [0.1, 0.9]
    current["bot"]["long"].pop("forager_volume_drop_pct", None)
    current["optimize"]["bounds"].pop("long_forager_volume_drop_pct", None)

    out = format_config(current, verbose=False, live_only=True)

    assert out["bot"]["long"]["forager_volume_drop_pct"] == pytest.approx(0.42)
    assert "filter_volume_drop_pct" not in out["bot"]["long"]
    assert out["optimize"]["bounds"]["long_forager_volume_drop_pct"] == [0.1, 0.9]
    assert "long_filter_volume_drop_pct" not in out["optimize"]["bounds"]


def test_format_config_adds_missing_forager_score_weights():
    current = copy.deepcopy(_template())
    current["bot"]["long"].pop("forager_score_weights", None)

    out = format_config(current, verbose=False, live_only=True)

    assert out["bot"]["long"]["forager_score_weights"] == {
        "volume": 0.0,
        "ema_readiness": 0.0,
        "volatility": 1.0,
    }


def test_format_config_maps_all_zero_forager_weights_to_volume_only():
    current = copy.deepcopy(_template())
    current["bot"]["long"]["forager_score_weights"] = {
        "volume": 0.0,
        "ema_readiness": 0.0,
        "volatility": 0.0,
    }

    out = format_config(current, verbose=False, live_only=True)

    assert out["bot"]["long"]["forager_score_weights"] == {
        "volume": 1.0,
        "ema_readiness": 0.0,
        "volatility": 0.0,
    }


def test_format_config_normalizes_positive_forager_weights_to_unit_sum():
    current = copy.deepcopy(_template())
    current["bot"]["long"]["forager_score_weights"] = {
        "volume": 2.0,
        "ema_readiness": 1.0,
        "volatility": 1.0,
    }

    out = format_config(current, verbose=False, live_only=True)

    assert out["bot"]["long"]["forager_score_weights"] == {
        "volume": pytest.approx(0.5),
        "ema_readiness": pytest.approx(0.25),
        "volatility": pytest.approx(0.25),
    }


def test_format_config_requires_positive_forager_volume_span_when_volume_weight_enabled():
    current = copy.deepcopy(_template())
    current["bot"]["long"]["forager_score_weights"] = {
        "volume": 1.0,
        "ema_readiness": 0.0,
        "volatility": 0.0,
    }
    current["bot"]["long"]["forager_volume_ema_span"] = 0.0

    with pytest.raises(
        ValueError,
        match="bot.long.forager_volume_ema_span must be > 0 when forager volume ranking or volume pruning is enabled",
    ):
        format_config(current, verbose=False, live_only=True)


def test_get_template_config_uses_canonical_forager_span_keys_and_weight_bounds():
    template = _template()

    assert "forager_volatility_ema_span" in template["bot"]["long"]
    assert "forager_volume_ema_span" in template["bot"]["long"]
    assert "filter_volatility_ema_span" not in template["bot"]["long"]
    assert "filter_volume_ema_span" not in template["bot"]["long"]

    bounds = template["optimize"]["bounds"]
    assert "long_forager_volatility_ema_span" in bounds
    assert "long_forager_volume_ema_span" in bounds
    assert "short_forager_volatility_ema_span" in bounds
    assert "short_forager_volume_ema_span" in bounds
    assert "long_forager_score_weights_volume" in bounds
    assert "long_forager_score_weights_ema_readiness" in bounds
    assert "long_forager_score_weights_volatility" in bounds
    assert "short_forager_score_weights_volume" in bounds
    assert "short_forager_score_weights_ema_readiness" in bounds
    assert "short_forager_score_weights_volatility" in bounds


def test_format_config_adds_internal_forager_aliases_for_runtime_compatibility():
    current = copy.deepcopy(_template())
    current["bot"]["long"].pop("filter_volatility_ema_span", None)
    current["bot"]["long"].pop("filter_volume_ema_span", None)
    current["optimize"]["bounds"].pop("long_filter_volatility_ema_span", None)
    current["optimize"]["bounds"].pop("long_filter_volume_ema_span", None)

    out = format_config(current, verbose=False, live_only=True)

    assert out["bot"]["long"]["filter_volatility_ema_span"] == out["bot"]["long"][
        "forager_volatility_ema_span"
    ]
    assert out["bot"]["long"]["filter_volume_ema_span"] == out["bot"]["long"][
        "forager_volume_ema_span"
    ]
    assert out["optimize"]["bounds"]["long_filter_volatility_ema_span"] == out["optimize"][
        "bounds"
    ]["long_forager_volatility_ema_span"]
    assert out["optimize"]["bounds"]["long_filter_volume_ema_span"] == out["optimize"]["bounds"][
        "long_forager_volume_ema_span"
    ]


def test_format_config_is_idempotent_for_lean_live_config():
    tmpl = _template()
    lean_live = {"bot": copy.deepcopy(tmpl["bot"]), "live": copy.deepcopy(tmpl["live"])}

    first = format_config(lean_live, verbose=False, live_only=True)
    second = format_config(copy.deepcopy(first), verbose=False, live_only=True)

    for key in ("backtest", "bot", "coin_overrides", "live", "logging", "optimize"):
        assert first[key] == second[key]


def test_format_config_drops_obsolete_forager_keys_without_misleading_unused_logs(caplog):
    current = copy.deepcopy(_template())
    current["bot"]["long"]["filter_volatility_drop_pct"] = 0.0
    current["bot"]["short"]["filter_volatility_drop_pct"] = 0.0
    current["optimize"]["bounds"]["long_filter_volatility_drop_pct"] = [0.0, 1.0]
    current["optimize"]["bounds"]["short_filter_volatility_drop_pct"] = [0.0, 1.0]

    with caplog.at_level(logging.INFO):
        out = format_config(current, verbose=True, live_only=True)

    assert "common" not in out["bot"]
    assert "filter_volatility_drop_pct" not in out["bot"]["long"]
    assert "filter_volatility_drop_pct" not in out["bot"]["short"]
    assert "long_filter_volatility_drop_pct" not in out["optimize"]["bounds"]
    assert "short_filter_volatility_drop_pct" not in out["optimize"]["bounds"]

    messages = [record.message for record in caplog.records]
    assert any("dropping obsolete parameter bot.long.filter_volatility_drop_pct" in msg for msg in messages)
    assert any(
        "dropping obsolete parameter optimize.bounds.long_filter_volatility_drop_pct" in msg
        for msg in messages
    )
    assert not any("Removed unused key from config: bot.common" in msg for msg in messages)
    assert not any(
        "Removed unused key from config: bot.long.filter_volatility_drop_pct" in msg
        for msg in messages
    )
    assert not any(
        "Removed unused key from config: optimize.bounds.long_filter_volatility_drop_pct" in msg
        for msg in messages
    )


def test_format_config_adds_monitor_defaults():
    current = copy.deepcopy(_template())
    current.pop("monitor", None)

    out = format_config(current, verbose=False, live_only=True)

    assert out["monitor"] == {
        "enabled": False,
        "root_dir": "monitor",
        "snapshot_interval_seconds": 1.0,
        "checkpoint_interval_minutes": 10.0,
        "event_rotation_mb": 128.0,
        "event_rotation_minutes": 60.0,
        "retain_days": 7.0,
        "max_total_bytes": 1073741824,
        "retain_price_ticks": True,
        "retain_candles": True,
        "retain_fills": True,
        "compress_rotated_segments": True,
        "price_tick_min_interval_ms": 500,
        "emit_completed_candles": True,
        "include_raw_fill_payloads": False,
    }


def test_format_config_adds_logging_silence_watchdog_default():
    current = copy.deepcopy(_template())
    current["logging"].pop("silence_watchdog_seconds", None)

    out = format_config(current, verbose=False, live_only=True)

    assert out["logging"]["silence_watchdog_seconds"] == 60.0


def test_format_config_rejects_invalid_monitor_snapshot_interval():
    current = copy.deepcopy(_template())
    current["monitor"]["snapshot_interval_seconds"] = 0.0

    with pytest.raises(ValueError, match="config.monitor.snapshot_interval_seconds"):
        format_config(current, verbose=False, live_only=True)
