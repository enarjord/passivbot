import pytest
from copy import deepcopy
import config_utils


def _base_config_with_override(coin_key, override_value):
    return {
        "bot": {"long": {}, "short": {}},
        "live": {"user": "tester", "approved_coins": [], "ignored_coins": {"long": [], "short": []}},
        "coin_overrides": {coin_key: override_value},
    }


def test_parse_overrides_rejects_flat_strategy_mods(monkeypatch):
    cfg = _base_config_with_override("btcusdt", {"bot": {"long": {"entry_grid_spacing_pct": 0.03}}})

    monkeypatch.setattr(config_utils, "symbol_to_coin", lambda x: x)
    monkeypatch.setattr(config_utils, "load_override_config", lambda c, coin: {})

    with pytest.raises(ValueError, match="unsupported flat strategy override"):
        config_utils.parse_overrides(deepcopy(cfg), verbose=False)


def test_parse_overrides_rejects_flat_strategy_mods_after_renaming(monkeypatch):
    cfg = _base_config_with_override("btcusdt", {"bot": {"long": {"entry_grid_spacing_pct": 0.05}}})

    monkeypatch.setattr(config_utils, "symbol_to_coin", lambda x: "BTC/USDT")
    monkeypatch.setattr(config_utils, "load_override_config", lambda c, coin: {})

    with pytest.raises(ValueError, match="coin_overrides.BTC/USDT.bot.long"):
        config_utils.parse_overrides(deepcopy(cfg), verbose=False)


def test_parse_overrides_keeps_nested_strategy_mods(monkeypatch):
    cfg = _base_config_with_override(
        "btcusdt",
        {
            "bot": {
                "long": {
                    "strategy": {
                        "trailing_martingale": {
                            "entry": {"threshold_base_pct": 0.05}
                        }
                    }
                }
            }
        },
    )

    monkeypatch.setattr(config_utils, "symbol_to_coin", lambda x: "BTC/USDT")
    monkeypatch.setattr(config_utils, "load_override_config", lambda c, coin: {})

    res = config_utils.parse_overrides(deepcopy(cfg), verbose=False)

    assert "btcusdt" not in res["coin_overrides"]
    assert "BTC/USDT" in res["coin_overrides"]
    assert res["coin_overrides"]["BTC/USDT"] == {
        "bot": {
            "long": {
                "strategy": {
                    "trailing_martingale": {
                        "entry": {"threshold_base_pct": 0.05}
                    }
                }
            }
        }
    }


def test_parse_overrides_retains_coin_flags_key(monkeypatch):
    cfg = _base_config_with_override("BTC", {"live": {"forced_mode_long": "manual"}})
    cfg["live"]["coin_flags"] = None
    monkeypatch.setattr(config_utils, "load_override_config", lambda c, coin: {})
    res = config_utils.parse_overrides(deepcopy(cfg), verbose=False)
    assert "coin_flags" in res["live"]
    assert isinstance(res["live"]["coin_flags"], dict)


def test_parse_overrides_keeps_entry_cooldown_minutes(monkeypatch):
    cfg = _base_config_with_override(
        "HYPE",
        {
            "bot": {
                "long": {
                    "risk": {"entry_cooldown_minutes": 50.0},
                }
            }
        },
    )
    cfg["bot"]["long"] = {"risk": {"entry_cooldown_minutes": 7.0}}
    cfg["bot"]["short"] = {"risk": {"entry_cooldown_minutes": 7.0}}

    monkeypatch.setattr(config_utils, "symbol_to_coin", lambda x: x)
    monkeypatch.setattr(config_utils, "load_override_config", lambda c, coin: {})

    res = config_utils.parse_overrides(deepcopy(cfg), verbose=False)

    assert res["coin_overrides"]["HYPE"]["bot"]["long"]["risk"][
        "entry_cooldown_minutes"
    ] == pytest.approx(50.0)


def test_parse_overrides_keeps_flat_entry_cooldown_minutes(monkeypatch):
    cfg = _base_config_with_override(
        "HYPE",
        {
            "bot": {
                "long": {
                    "risk_entry_cooldown_minutes": 42.5,
                }
            }
        },
    )
    cfg["bot"]["long"] = {"risk_entry_cooldown_minutes": 7.0}
    cfg["bot"]["short"] = {"risk_entry_cooldown_minutes": 7.0}

    monkeypatch.setattr(config_utils, "symbol_to_coin", lambda x: x)
    monkeypatch.setattr(config_utils, "load_override_config", lambda c, coin: {})

    res = config_utils.parse_overrides(deepcopy(cfg), verbose=False)

    assert res["coin_overrides"]["HYPE"]["bot"]["long"][
        "risk_entry_cooldown_minutes"
    ] == pytest.approx(42.5)
