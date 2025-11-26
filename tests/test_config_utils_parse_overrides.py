import pytest
from copy import deepcopy
import config_utils


def _base_config_with_override(coin_key, override_value):
    return {
        "bot": {"long": {}, "short": {}},
        "live": {"user": "tester", "approved_coins": [], "ignored_coins": {"long": [], "short": []}},
        "coin_overrides": {coin_key: override_value},
    }


def test_parse_overrides_keeps_allowed_mods(monkeypatch):
    cfg = _base_config_with_override("btcusdt", {"bot": {"long": {"entry_grid_spacing_pct": 0.03}}})

    # Make symbol_to_coin identity to avoid renaming
    monkeypatch.setattr(config_utils, "symbol_to_coin", lambda x: x)
    # Ensure we don't try to load any external override files
    monkeypatch.setattr(config_utils, "load_override_config", lambda c, coin: {})

    res = config_utils.parse_overrides(deepcopy(cfg), verbose=False)

    assert "btcusdt" in res["coin_overrides"]
    assert res["coin_overrides"]["btcusdt"] == {"bot": {"long": {"entry_grid_spacing_pct": 0.03}}}


def test_parse_overrides_renames_coins(monkeypatch):
    cfg = _base_config_with_override("btcusdt", {"bot": {"long": {"entry_grid_spacing_pct": 0.05}}})

    # symbol_to_coin will canonicalize the coin name
    monkeypatch.setattr(config_utils, "symbol_to_coin", lambda x: "BTC/USDT")
    monkeypatch.setattr(config_utils, "load_override_config", lambda c, coin: {})

    res = config_utils.parse_overrides(deepcopy(cfg), verbose=False)

    # original key should be removed, new canonical key should exist
    assert "btcusdt" not in res["coin_overrides"]
    assert "BTC/USDT" in res["coin_overrides"]
    assert res["coin_overrides"]["BTC/USDT"] == {"bot": {"long": {"entry_grid_spacing_pct": 0.05}}}


def test_parse_overrides_retains_coin_flags_key(monkeypatch):
    cfg = _base_config_with_override("BTC", {"live": {"forced_mode_long": "manual"}})
    cfg["live"]["coin_flags"] = None
    monkeypatch.setattr(config_utils, "load_override_config", lambda c, coin: {})
    res = config_utils.parse_overrides(deepcopy(cfg), verbose=False)
    assert "coin_flags" in res["live"]
    assert isinstance(res["live"]["coin_flags"], dict)
