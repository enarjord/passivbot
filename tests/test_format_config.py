import copy
import pytest

from config_utils import (
    format_config,
    get_template_config,
)

# Internal helpers were intentionally made top-level; import directly for unit tests
from config_utils import detect_flavor, build_base_config_from_flavor


def _template():
    return get_template_config("v7")


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
    # ensure missing sections were added
    assert "optimize" in out and "backtest" in out


def test_format_config_current_roundtrip_basic():
    # Provide a minimally valid current config and ensure format_config returns a dict with
    # expected top-level sections and enforces boolean casting for enforce_exposure_limit
    tmpl = _template()
    current = copy.deepcopy(tmpl)
    # set a non-bool value for enforce_exposure_limit to verify casting to bool
    current["bot"]["long"]["enforce_exposure_limit"] = 1
    current["bot"]["short"]["enforce_exposure_limit"] = 0
    out = format_config(current, verbose=False)
    for k in ["bot", "live", "optimize", "backtest"]:
        assert k in out
    assert isinstance(out["bot"]["long"]["enforce_exposure_limit"], bool)
    assert isinstance(out["bot"]["short"]["enforce_exposure_limit"], bool)


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
