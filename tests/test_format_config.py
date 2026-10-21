from config.universe import require_side_universe_sources
from config.universe import set_side_universe_sources
import copy
import logging
import pytest

from config_utils import (
    format_config,
    get_template_config,
)



def _template():
    return get_template_config()


def _bound(config, pside, *path):
    cur = config["optimize"]["bounds"][pside]
    for part in path:
        cur = cur[part]
    return cur


@pytest.mark.parametrize("wrapped", [False, True])
def test_unsupported_config_wrappers_are_rejected(wrapped):
    config = {"config": _template()} if wrapped else {
        "TWE_long": 1.0, "TWE_short": 1.0, "universal_live_config": {},
    }
    with pytest.raises(ValueError, match="explicit migration"):
        format_config(config, verbose=False)


def test_legacy_shape_cannot_claim_final_schema_version():
    config = {"config_version": "v9.0.0", "TWE_long": 1.2, "universal_live_config": {}}
    with pytest.raises(ValueError, match="unknown configuration field"):
        format_config(config, verbose=False)


def test_format_config_live_only_adds_sections():
    tmpl = _template()
    live_only = {
        "config_version": tmpl["config_version"],
        "bot": tmpl["bot"],
        "live": tmpl["live"],
    }
    out = format_config(live_only, verbose=False)
    out_live_only = format_config(live_only, verbose=False, live_only=True)
    # ensure missing sections were added
    assert "optimize" in out and "backtest" in out
    assert "optimize" in out_live_only and "backtest" in out_live_only


def test_format_config_missing_short_side_stays_disabled():
    current = {
        "config_version": _template()["config_version"],
        "bot": {
            "long": {
                "risk": {"total_wallet_exposure_limit": 0.7},
            }
        },
        "live": {},
        "backtest": {},
        "optimize": {},
        "logging": {},
        "monitor": {},
    }

    out = format_config(current, verbose=False)

    assert out["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 0.7
    assert out["bot"]["short"]["risk"]["total_wallet_exposure_limit"] == 0.0


def test_format_config_missing_long_side_stays_disabled():
    current = {
        "config_version": _template()["config_version"],
        "bot": {
            "short": {
                "risk": {"total_wallet_exposure_limit": 0.7},
            }
        },
        "live": {},
        "backtest": {},
        "optimize": {},
        "logging": {},
        "monitor": {},
    }

    out = format_config(current, verbose=False)

    assert out["bot"]["short"]["risk"]["total_wallet_exposure_limit"] == 0.7
    assert out["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 0.0


def test_format_config_current_roundtrip_basic():
    # Provide a minimally valid current config and ensure format_config returns a dict with
    # expected top-level sections while normalising new exposure controls
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    current['bot']['long']['risk']['position_exposure_enforcer']['threshold'] = 0.99
    current["bot"]["long"]["risk"]["we_excess_allowance_pct"] = 0.25
    current['bot']['long']['risk']['total_exposure_enforcer']['threshold'] = 0.95
    current['bot']['short']['risk']['position_exposure_enforcer']['threshold'] = 1.02
    current['bot']['short']['risk']['total_exposure_enforcer']['threshold'] = 1.08
    out = format_config(current, verbose=False)
    for k in ["bot", "live", "optimize", "backtest"]:
        assert k in out
    assert isinstance(
        out['bot']['long']['risk']['position_exposure_enforcer']['threshold'], (int, float)
    )
    assert isinstance(out["bot"]["long"]["risk"]["we_excess_allowance_pct"], float)
    assert "we_excess_allowance_mode" not in out["bot"]["long"]["risk"]
    assert isinstance(
        out['bot']['long']['risk']['total_exposure_enforcer']['threshold'], (int, float)
    )
    assert isinstance(
        out['bot']['short']['risk']['position_exposure_enforcer']['threshold'],
        (int, float),
    )
    assert isinstance(
        out['bot']['short']['risk']['total_exposure_enforcer']['threshold'], (int, float)
    )
    assert "risk_twel_enforcer_threshold" not in out["bot"]["long"]


def test_format_config_restores_missing_trailing_martingale_weight_defaults():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    strategy = current["bot"]["long"]["strategy"]["trailing_martingale"]
    strategy["entry"].pop("threshold_volatility_1h_weight")
    strategy["close"].pop("threshold_volatility_1h_weight")

    out = format_config(current, verbose=False)
    trailing_martingale = out["bot"]["long"]["strategy"]["trailing_martingale"]
    default_strategy = tmpl["bot"]["long"]["strategy"]["trailing_martingale"]

    assert trailing_martingale["entry"][
        "threshold_volatility_1h_weight"
    ] == pytest.approx(default_strategy["entry"]["threshold_volatility_1h_weight"])
    assert trailing_martingale["close"][
        "threshold_volatility_1h_weight"
    ] == pytest.approx(default_strategy["close"]["threshold_volatility_1h_weight"])


def test_format_config_preserves_approved_coins_dict():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    set_side_universe_sources(current, 'approved_coins', {"long": ["BTC"], "short": ["BTC"]})
    out = format_config(current, verbose=False)
    assert require_side_universe_sources(out, 'approved_coins') == {"long": ["BTC"], "short": ["BTC"]}


@pytest.mark.parametrize("path", [("live", "unexpected"), ("bot", "long", "foo"),
                                  ("optimize", "bounds", "extra"), ("extra_section",)])
def test_format_config_rejects_unknown_keys_recursively(path):
    current = copy.deepcopy(_template())
    target = current
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = 1
    with pytest.raises(ValueError, match="unknown configuration field"):
        format_config(current, verbose=False)


@pytest.mark.parametrize("removed_value", ["manual", "bad_policy"])
def test_format_config_rejects_removed_cooldown_position_policy(removed_value):
    current = copy.deepcopy(_template())
    current["live"]["hsl_position_during_cooldown_policy"] = removed_value
    with pytest.raises(ValueError, match="migrate-config-v9"):
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


def test_format_config_adds_logging_defaults():
    current = copy.deepcopy(_template())
    current.pop("logging", None)

    out = format_config(current, verbose=False, live_only=True)

    assert out["logging"] == {
        "backup_count": 5,
        "dir": "logs",
        "level": 1,
        "live_event_debug_profiles": [],
        "max_bytes_mb": 10.0,
        "memory_snapshot_interval_minutes": 30,
        "persist_to_file": True,
        "rotation": True,
        "volume_refresh_info_threshold_seconds": 30,
    }


def test_format_config_rejects_invalid_logging_dir():
    current = copy.deepcopy(_template())
    current["logging"]["dir"] = "   "

    with pytest.raises(ValueError, match="config.logging.dir"):
        format_config(current, verbose=False, live_only=True)


def test_format_config_rejects_invalid_monitor_snapshot_interval():
    current = copy.deepcopy(_template())
    current["monitor"]["snapshot_interval_seconds"] = 0.0

    with pytest.raises(ValueError, match="config.monitor.snapshot_interval_seconds"):
        format_config(current, verbose=False, live_only=True)


def test_format_config_empty_optimize_freezes_configured_bounds():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    current["optimize"] = {}
    current["bot"]["long"]["strategy"]["trailing_martingale"]["close"]["threshold_base_pct"] = 0.006
    current["bot"]["short"]["strategy"]["trailing_martingale"]["close"]["threshold_base_pct"] = -0.003

    out = format_config(current, verbose=False, live_only=True)

    assert "bounds" in out["optimize"]
    assert _bound(
        out, "long", "strategy", "trailing_martingale", "close", "threshold_base_pct"
    ) == [0.006, 0.006]
    assert _bound(
        out, "short", "strategy", "trailing_martingale", "close", "threshold_base_pct"
    ) == [-0.003, -0.003]


def test_format_config_current_with_optimize_missing_bounds_adds_defaults():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    current["optimize"] = {"scoring": [{"metric": "adg_usd", "goal": "max"}]}

    out = format_config(current, verbose=False, live_only=True)

    assert out["optimize"]["scoring"] == [{"metric": "adg_usd", "goal": "max"}]
    assert (
        "qty_pct"
        in out["optimize"]["bounds"]["long"]["strategy"]["trailing_martingale"]["close"]
    )
    assert (
        "qty_pct"
        in out["optimize"]["bounds"]["short"]["strategy"]["trailing_martingale"][
            "close"
        ]
    )


def test_format_config_current_with_missing_bot_side_adds_defaults():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    del current["bot"]["short"]

    out = format_config(current, verbose=False, live_only=True)

    assert (
        out["bot"]["short"]["strategy"]["trailing_martingale"]["close"][
            "threshold_base_pct"
        ]
        == tmpl["bot"]["short"]["strategy"]["trailing_martingale"]["close"][
            "threshold_base_pct"
        ]
    )
    assert (
        out["bot"]["short"]["risk"]["max_positions"]
        == tmpl["bot"]["short"]["risk"]["max_positions"]
    )


def test_format_config_raises_on_non_dict_optimize_bounds():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    current["optimize"]["bounds"] = []

    with pytest.raises(TypeError, match="config.optimize.bounds must be a dict"):
        format_config(current, verbose=False, live_only=True)


def test_format_config_preserves_nested_strategy_bounds_before_hydration():
    current = copy.deepcopy(_template())
    current["optimize"]["bounds"]["long"]["strategy"]["trailing_martingale"]["close"][
        "threshold_base_pct"
    ] = [
        0.005,
        0.02,
    ]
    current["optimize"]["bounds"]["short"]["strategy"]["trailing_martingale"]["close"][
        "threshold_base_pct"
    ] = [
        0.004,
        0.016,
    ]

    out = format_config(current, verbose=False, live_only=True)

    assert _bound(
        out, "long", "strategy", "trailing_martingale", "close", "threshold_base_pct"
    ) == [
        0.005,
        0.02,
    ]
    assert _bound(
        out, "short", "strategy", "trailing_martingale", "close", "threshold_base_pct"
    ) == [
        0.004,
        0.016,
    ]


def test_format_config_canonical_omissions_restore_documented_defaults():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    trailing_martingale = current["bot"]["long"]["strategy"]["trailing_martingale"]
    trailing_martingale.pop("volatility_ema_span_1h")
    trailing_martingale.pop("volatility_ema_span_1m")
    for key in (
        "threshold_volatility_1h_weight",
        "threshold_volatility_1m_weight",
        "threshold_we_weight",
        "retracement_base_pct",
    ):
        trailing_martingale["entry"].pop(key)
    for key in ("threshold_volatility_1h_weight", "retracement_base_pct"):
        trailing_martingale["close"].pop(key)
    current["bot"]["long"]["forager"].pop("volatility_ema_span_minutes")
    current["bot"]["long"]["forager"].pop("volume_ema_span_minutes")
    current["bot"]["long"]["hsl"].pop("panic_close_order_type")
    current["bot"]["long"]["risk"]["total_exposure_enforcer"].pop("threshold")
    current["bot"]["long"]["risk"].pop("we_excess_allowance_pct")
    current["bot"]["long"]["risk"]["position_exposure_enforcer"].pop("threshold")
    current["bot"]["long"]["unstuck"].pop("close_pct")
    current["bot"]["long"]["unstuck"].pop("ema_distance_fraction")
    current["bot"]["long"]["unstuck"].pop("loss_allowance_pct")
    current["bot"]["long"]["unstuck"].pop("exposure_trigger_ratio")

    out = format_config(current, verbose=False, live_only=True)

    long_cfg = out["bot"]["long"]
    long_strategy = long_cfg["strategy"]["trailing_martingale"]
    default_long = tmpl["bot"]["long"]
    default_strategy = default_long["strategy"]["trailing_martingale"]
    assert long_strategy["volatility_ema_span_1h"] == pytest.approx(
        default_strategy["volatility_ema_span_1h"]
    )
    assert long_strategy["volatility_ema_span_1m"] == pytest.approx(
        default_strategy["volatility_ema_span_1m"]
    )
    assert long_strategy["entry"]["threshold_volatility_1h_weight"] == pytest.approx(
        default_strategy["entry"]["threshold_volatility_1h_weight"]
    )
    assert long_strategy["entry"]["threshold_volatility_1m_weight"] == pytest.approx(
        default_strategy["entry"]["threshold_volatility_1m_weight"]
    )
    assert long_strategy["entry"]["threshold_we_weight"] == pytest.approx(
        default_strategy["entry"]["threshold_we_weight"]
    )
    assert long_strategy["entry"]["retracement_base_pct"] == pytest.approx(
        default_strategy["entry"]["retracement_base_pct"]
    )
    assert long_strategy["close"]["threshold_volatility_1h_weight"] == pytest.approx(
        default_strategy["close"]["threshold_volatility_1h_weight"]
    )
    assert long_strategy["close"]["retracement_base_pct"] == pytest.approx(
        default_strategy["close"]["retracement_base_pct"]
    )
    assert long_cfg["forager"]["volatility_ema_span_1m"] == pytest.approx(
        default_long["forager"]["volatility_ema_span_1m"]
    )
    assert long_cfg["forager"]["volume_ema_span_1m"] == pytest.approx(
        default_long["forager"]["volume_ema_span_1m"]
    )
    assert (
        long_cfg["hsl"]["panic_close_order_type"]
        == default_long["hsl"]["panic_close_order_type"]
    )
    assert long_cfg['risk']['total_exposure_enforcer']['threshold'] == pytest.approx(
        default_long['risk']['total_exposure_enforcer']['threshold']
    )
    assert long_cfg["risk"]["we_excess_allowance_pct"] == pytest.approx(
        default_long["risk"]["we_excess_allowance_pct"]
    )
    assert "we_excess_allowance_mode" not in long_cfg["risk"]
    assert long_cfg['risk']['position_exposure_enforcer']['threshold'] == pytest.approx(
        default_long['risk']['position_exposure_enforcer']['threshold']
    )
    assert long_cfg["unstuck"]["close_pct"] == pytest.approx(
        default_long["unstuck"]["close_pct"]
    )
    assert long_cfg["unstuck"]["ema_distance_fraction"] == pytest.approx(
        default_long["unstuck"]["ema_distance_fraction"]
    )
    assert long_cfg["unstuck"]["loss_allowance_pct"] == pytest.approx(
        default_long["unstuck"]["loss_allowance_pct"]
    )
    assert long_cfg["unstuck"]["exposure_trigger_ratio"] == pytest.approx(
        default_long["unstuck"]["exposure_trigger_ratio"]
    )


def test_format_config_restores_missing_current_close_qty_pct_from_schema_default():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    current["bot"]["long"]["strategy"]["trailing_martingale"]["close"].pop("qty_pct")

    out = format_config(current, verbose=False, live_only=True)

    assert out["bot"]["long"]["strategy"]["trailing_martingale"]["close"][
        "qty_pct"
    ] == pytest.approx(
        tmpl["bot"]["long"]["strategy"]["trailing_martingale"]["close"]["qty_pct"]
    )
    assert "n_closes" not in out["bot"]["long"]


def test_format_config_restores_missing_current_enabled_side_core_params():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    current["bot"]["long"]["strategy"]["trailing_martingale"]["entry"].pop(
        "threshold_base_pct"
    )

    out = format_config(current, verbose=False, live_only=True)

    assert out["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "threshold_base_pct"
    ] == pytest.approx(
        tmpl["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
            "threshold_base_pct"
        ]
    )


def test_format_config_warns_and_snaps_cliff_edge_thresholds(caplog):
    current = copy.deepcopy(_template())
    current['bot']['long']['risk']['position_exposure_enforcer']['enabled'] = False
    current['bot']['long']['risk']['position_exposure_enforcer']['threshold'] = 5e-10
    current['bot']['long']['risk']['total_exposure_enforcer']['threshold'] = 0.05
    current["bot"]["long"]["unstuck"]["exposure_trigger_ratio"] = 5e-10

    with caplog.at_level(logging.WARNING):
        out = format_config(current, verbose=True, live_only=True)

    assert out['bot']['long']['risk']['position_exposure_enforcer']['threshold'] == 0.0
    assert out['bot']['long']['risk']['total_exposure_enforcer']['threshold'] == pytest.approx(0.05)
    assert out["bot"]["long"]["unstuck"]["exposure_trigger_ratio"] == 0.0
    assert any(
        "bot.long.risk.position_exposure_enforcer.threshold" in rec.message
        and "snapping to 0.0" in rec.message
        for rec in caplog.records
    )
    assert any(
        "bot.long.risk.total_exposure_enforcer.threshold=0.05" in rec.message
        for rec in caplog.records
    )
    assert any(
        "bot.long.unstuck.exposure_trigger_ratio" in rec.message and "snapping to 0.0" in rec.message
        for rec in caplog.records
    )


def test_format_config_rejects_enabled_enforcer_threshold_snapped_to_zero():
    current = copy.deepcopy(_template())
    current['bot']['long']['risk']['position_exposure_enforcer']['enabled'] = True
    current['bot']['long']['risk']['position_exposure_enforcer']['threshold'] = 5e-10

    with pytest.raises(
        ValueError,
        match="position_exposure_enforcer.threshold must be finite and > 0.0",
    ):
        format_config(current, verbose=False, live_only=True)


@pytest.mark.parametrize("key", ["close_grid_min_markup", "close_grid_markup_range", "n_closes"])
def test_format_config_rejects_legacy_close_grid_aliases(key):
    current = copy.deepcopy(_template())
    current["bot"]["long"][key] = 0.007
    with pytest.raises(ValueError, match="unknown configuration field"):
        format_config(current, verbose=False)


def test_format_config_is_idempotent_for_lean_live_config():
    tmpl = _template()
    lean_live = {
        "config_version": tmpl["config_version"],
        "bot": copy.deepcopy(tmpl["bot"]),
        "live": copy.deepcopy(tmpl["live"]),
    }

    first = format_config(lean_live, verbose=False, live_only=True)
    second = format_config(copy.deepcopy(first), verbose=False, live_only=True)

    for key in ("backtest", "bot", "coin_overrides", "live", "logging", "optimize"):
        assert first[key] == second[key]


def test_format_config_preserves_live_optimize_bounds():
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    current["optimize"]["bounds"]["long"]["hsl"]["drawdown_ratio_threshold"] = [0.1, 0.3, 0.01]
    current["optimize"]["bounds"]["long"]["hsl"]["drawdown_ratio_ema_span_minutes"] = [
        10.0,
        120.0,
        5.0,
    ]

    out = format_config(current, verbose=False)

    assert out["optimize"]["bounds"]["long"]["hsl"]["drawdown_ratio_threshold"] == [
        0.1,
        0.3,
        0.01,
    ]
    assert out["optimize"]["bounds"]["long"]["hsl"]["drawdown_ratio_ema_span_minutes"] == [
        10.0,
        120.0,
        5.0,
    ]


def test_format_config_rejects_obsolete_grid_close_price_anchor():
    current = copy.deepcopy(_template())
    current["bot"]["long"]["strategy"]["trailing_martingale"]["grid_close_price_anchor"] = "ema_band"
    with pytest.raises(ValueError, match="unknown configuration field"):
        format_config(current, verbose=False)
