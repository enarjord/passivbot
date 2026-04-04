import copy
import pytest

from config_utils import (
    format_config,
    get_template_config,
)

# Internal helpers were intentionally made top-level; import directly for unit tests
from config_utils import detect_flavor, build_base_config_from_flavor


def _template():
    return get_template_config()


def _bound(config, pside, *path):
    cur = config["optimize"]["bounds"][pside]
    for part in path:
        cur = cur[part]
    return cur


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
    assert isinstance(out["bot"]["long"]["risk"]["wel_enforcer_threshold"], (int, float))
    assert isinstance(out["bot"]["long"]["risk"]["we_excess_allowance_pct"], float)
    assert isinstance(out["bot"]["long"]["risk"]["twel_enforcer_threshold"], (int, float))
    assert isinstance(out["bot"]["short"]["risk"]["wel_enforcer_threshold"], (int, float))
    assert isinstance(out["bot"]["short"]["risk"]["twel_enforcer_threshold"], (int, float))
    assert "risk_twel_enforcer_threshold" not in out["bot"]["long"]


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


def test_format_config_normalizes_hsl_position_during_cooldown_policy():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    current["live"]["hsl_position_during_cooldown_policy"] = "manual"

    out = format_config(current, verbose=False, live_only=True)

    assert out["live"]["hsl_position_during_cooldown_policy"] == "manual"


def test_format_config_rejects_invalid_hsl_position_during_cooldown_policy():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    current["live"]["hsl_position_during_cooldown_policy"] = "bad_policy"

    with pytest.raises(ValueError, match="live.hsl_position_during_cooldown_policy"):
        format_config(current, verbose=False, live_only=True)


def test_format_config_adds_monitor_defaults():
    current = copy.deepcopy(_template())
    current.pop("monitor", None)

    out = format_config(current, verbose=False, live_only=True)

    assert out["monitor"] == {
        "enabled": True,
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


def test_format_config_rejects_invalid_monitor_snapshot_interval():
    current = copy.deepcopy(_template())
    current["monitor"]["snapshot_interval_seconds"] = 0.0

    with pytest.raises(ValueError, match="config.monitor.snapshot_interval_seconds"):
        format_config(current, verbose=False, live_only=True)

def test_format_config_current_with_empty_optimize_adds_bounds():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    current["optimize"] = {}

    out = format_config(current, verbose=False, live_only=True)

    assert "bounds" in out["optimize"]
    assert _bound(out, "long", "strategy", "trailing_grid", "close_grid_markup_start") == _bound(
        tmpl, "long", "strategy", "trailing_grid", "close_grid_markup_start"
    )
    assert _bound(out, "short", "strategy", "trailing_grid", "close_grid_markup_end") == _bound(
        tmpl, "short", "strategy", "trailing_grid", "close_grid_markup_end"
    )


def test_format_config_current_with_optimize_missing_bounds_adds_defaults():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    current["optimize"] = {"scoring": ["adg"]}

    out = format_config(current, verbose=False, live_only=True)

    assert out["optimize"]["scoring"] == [{"metric": "adg_usd", "goal": "max"}]
    assert "close_grid_qty_pct" in out["optimize"]["bounds"]["long"]["strategy"]["trailing_grid"]
    assert "close_grid_qty_pct" in out["optimize"]["bounds"]["short"]["strategy"]["trailing_grid"]


def test_format_config_current_with_missing_bot_side_adds_defaults():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    del current["bot"]["short"]

    out = format_config(current, verbose=False, live_only=True)

    assert out["bot"]["short"]["strategy"]["trailing_grid"]["close_grid_markup_end"] == tmpl["bot"]["short"]["strategy"]["trailing_grid"]["close_grid_markup_end"]
    assert out["bot"]["short"]["risk"]["n_positions"] == tmpl["bot"]["short"]["risk"]["n_positions"]


def test_format_config_raises_on_non_dict_optimize_bounds():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    current["optimize"]["bounds"] = []

    with pytest.raises(TypeError, match="config.optimize.bounds must be a dict"):
        format_config(current, verbose=False, live_only=True)


def test_format_config_preserves_nested_strategy_bounds_before_hydration():
    current = copy.deepcopy(_template())
    current["optimize"]["bounds"]["long"]["strategy"]["trailing_grid"]["close_grid_markup_start"] = [
        0.005,
        0.02,
    ]
    current["optimize"]["bounds"]["short"]["strategy"]["trailing_grid"]["close_grid_markup_end"] = [
        0.004,
        0.016,
    ]

    out = format_config(current, verbose=False, live_only=True)

    assert _bound(out, "long", "strategy", "trailing_grid", "close_grid_markup_start") == [
        0.005,
        0.02,
    ]
    assert _bound(out, "short", "strategy", "trailing_grid", "close_grid_markup_end") == [
        0.004,
        0.016,
    ]


def test_format_config_is_idempotent_for_lean_live_config():
    tmpl = _template()
    lean_live = {"bot": copy.deepcopy(tmpl["bot"]), "live": copy.deepcopy(tmpl["live"])}

    first = format_config(lean_live, verbose=False, live_only=True)
    second = format_config(copy.deepcopy(first), verbose=False, live_only=True)

    for key in ("backtest", "bot", "coin_overrides", "live", "logging", "optimize"):
        assert first[key] == second[key]


def test_format_config_preserves_live_optimize_bounds():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    current["optimize"]["bounds"]["long"]["hsl"]["red_threshold"] = [0.1, 0.3, 0.01]
    current["optimize"]["bounds"]["long"]["hsl"]["ema_span_minutes"] = [10.0, 120.0, 5.0]

    out = format_config(current, verbose=False)

    assert out["optimize"]["bounds"]["long"]["hsl"]["red_threshold"] == [
        0.1,
        0.3,
        0.01,
    ]
    assert out["optimize"]["bounds"]["long"]["hsl"]["ema_span_minutes"] == [
        10.0,
        120.0,
        5.0,
    ]
