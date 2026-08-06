from copy import deepcopy

import json
import os

import pytest

import config_utils


def _write_config(path, cfg):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def test_override_path_resolves_relative_to_base_config(tmp_path):
    base_cfg = config_utils.get_template_config()
    base_cfg["live"]["user"] = "tester"
    base_path = tmp_path / "base.json"
    base_cfg["live"]["base_config_path"] = str(base_path)
    base_cfg["coin_overrides"] = {
        "XRP": {
            "override_config_path": "overrides/xrp.json",
            # inline override should merge with file-loaded overrides
            "bot": {
                "short": {
                    "strategy": {
                        "trailing_martingale": {
                            "entry": {
                                "threshold_base_pct": 0.77,
                            },
                        },
                    },
                },
            },
        }
    }

    override_cfg = config_utils.get_template_config()
    override_cfg["live"]["user"] = "tester"
    override_cfg["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "threshold_base_pct"
    ] = 0.99
    override_cfg["disallowed_root"] = "drop_me"
    overrides_dir = tmp_path / "overrides"
    overrides_dir.mkdir()
    override_path = overrides_dir / "xrp.json"
    _write_config(override_path, override_cfg)

    _write_config(base_path, base_cfg)

    loaded = config_utils.load_config(str(base_path), verbose=False)
    parsed = config_utils.parse_overrides(deepcopy(loaded), verbose=False)

    assert "XRP" in parsed["coin_overrides"]
    xrp_ov = parsed["coin_overrides"]["XRP"]
    # allowed field from file
    assert xrp_ov["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "threshold_base_pct"
    ] == pytest.approx(0.99)
    # inline override merged on top
    assert xrp_ov["bot"]["short"]["strategy"]["trailing_martingale"]["entry"][
        "threshold_base_pct"
    ] == pytest.approx(0.77)
    # disallowed root key should be stripped
    assert "disallowed_root" not in xrp_ov


def test_override_file_not_found_yields_empty_override(tmp_path, monkeypatch):
    base_cfg = config_utils.get_template_config()
    base_cfg["live"]["user"] = "tester"
    base_cfg["live"]["base_config_path"] = str(tmp_path / "base.json")
    base_cfg["coin_overrides"] = {"DOGE": {"override_config_path": "overrides/missing.json"}}
    base_path = tmp_path / "base.json"
    _write_config(base_path, base_cfg)

    loaded = config_utils.load_config(str(base_path), verbose=False)
    parsed = config_utils.parse_overrides(deepcopy(loaded), verbose=False)

    # override exists but has no allowed diff because the file was missing
    assert "DOGE" in parsed["coin_overrides"]
    assert parsed["coin_overrides"]["DOGE"] == {}


def test_file_override_entry_cooldown_is_resolved_for_live_bp(tmp_path):
    """File-loaded grouped cooldown must resolve via live bp() after parse (no recompile)."""
    from config import prepare_config
    from passivbot import Passivbot

    base_cfg = config_utils.get_template_config()
    base_cfg["live"]["user"] = "tester"
    base_cfg["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 7.0
    base_path = tmp_path / "base.json"
    base_cfg["live"]["base_config_path"] = str(base_path)
    override_path = tmp_path / "hype.json"
    base_cfg["coin_overrides"] = {
        "HYPE": {"override_config_path": str(override_path)},
    }

    override_cfg = {
        "bot": {
            "long": {
                "risk": {"entry_cooldown_minutes": 50.0},
            }
        },
        "live": {"user": "tester"},
    }
    _write_config(override_path, override_cfg)
    _write_config(base_path, base_cfg)

    # Mirror live entry: compile first, then parse_overrides (no second compile).
    prepared = prepare_config(
        config_utils.load_config(str(base_path), verbose=False),
        base_config_path=str(base_path),
        live_only=True,
        verbose=False,
        target="live",
        runtime="live",
    )
    parsed = config_utils.parse_overrides(deepcopy(prepared), verbose=False)

    hype = parsed["coin_overrides"]["HYPE"]["bot"]["long"]
    assert hype["risk"]["entry_cooldown_minutes"] == pytest.approx(50.0)
    # Durable parse output stays grouped-only (no early flat mirror).
    assert "risk_entry_cooldown_minutes" not in hype

    bot = Passivbot.__new__(Passivbot)
    bot.config = parsed
    bot.exchange = "binance"
    bot.markets_dict = {"HYPE/USDT:USDT": {"active": True}}
    bot.coin_to_symbol = lambda coin, verbose=True: (
        "HYPE/USDT:USDT" if coin in {"HYPE", "HYPEUSDT"} else ""
    )
    bot.init_coin_overrides()
    assert bot.bp("long", "risk_entry_cooldown_minutes", "HYPE/USDT:USDT") == pytest.approx(
        50.0
    )
    assert bot.bp("long", "risk_entry_cooldown_minutes") == pytest.approx(7.0)


def test_nested_current_file_override_preserves_entry_cooldown(tmp_path):
    """Wrapped {\"config\": {...}} override files must not be pruned to empty."""
    base_cfg = config_utils.get_template_config()
    base_cfg["live"]["user"] = "tester"
    base_cfg["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 7.0
    base_path = tmp_path / "base.json"
    base_cfg["live"]["base_config_path"] = str(base_path)
    override_path = tmp_path / "wrapped.json"
    base_cfg["coin_overrides"] = {
        "HYPE": {"override_config_path": str(override_path)},
    }

    inner = config_utils.get_template_config()
    inner["live"]["user"] = "tester"
    inner["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 50.0
    inner["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "threshold_base_pct"
    ] = 0.05
    _write_config(override_path, {"config": inner})
    _write_config(base_path, base_cfg)

    loaded = config_utils.load_config(str(base_path), verbose=False)
    parsed = config_utils.parse_overrides(deepcopy(loaded), verbose=False)

    hype = parsed["coin_overrides"]["HYPE"]["bot"]["long"]
    assert hype["risk"]["entry_cooldown_minutes"] == pytest.approx(50.0)
    assert hype["strategy"]["trailing_martingale"]["entry"][
        "threshold_base_pct"
    ] == pytest.approx(0.05)


def test_compile_prefers_grouped_cooldown_over_stale_flat_alias():
    """Later group-only updates must win over inject-created flat aliases."""
    from config import compile_runtime_config, get_template_config
    from config.shared_bot import inject_flattened_shared_bot_side

    cfg = get_template_config()
    cfg["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 7.0
    cfg["coin_overrides"] = {
        "HYPE": {
            "bot": {
                "long": {
                    "risk": {"entry_cooldown_minutes": 50.0},
                }
            }
        }
    }
    # Simulate prior inject + later group-only transform (suite/CLI).
    inject_flattened_shared_bot_side(cfg["coin_overrides"]["HYPE"]["bot"]["long"])
    assert cfg["coin_overrides"]["HYPE"]["bot"]["long"][
        "risk_entry_cooldown_minutes"
    ] == pytest.approx(50.0)
    cfg["coin_overrides"]["HYPE"]["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 12.0

    compiled = compile_runtime_config(cfg, runtime="backtest", record_step=False)
    side = compiled["coin_overrides"]["HYPE"]["bot"]["long"]
    assert side["risk"]["entry_cooldown_minutes"] == pytest.approx(12.0)
    assert side["risk_entry_cooldown_minutes"] == pytest.approx(12.0)


def test_partial_file_override_does_not_synthesize_template_entry_cooldown(tmp_path):
    """Hydrated template defaults must not become per-coin cooldown overrides."""
    base_cfg = config_utils.get_template_config()
    base_cfg["live"]["user"] = "tester"
    base_cfg["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 7.0
    base_cfg["bot"]["short"]["risk"]["entry_cooldown_minutes"] = 7.0
    base_path = tmp_path / "base.json"
    base_cfg["live"]["base_config_path"] = str(base_path)
    override_path = tmp_path / "partial.json"
    base_cfg["coin_overrides"] = {
        "HYPE": {"override_config_path": str(override_path)},
    }

    # Partial override: strategy only; no entry_cooldown_minutes in the file.
    override_cfg = {
        "bot": {
            "long": {
                "strategy": {
                    "trailing_martingale": {
                        "entry": {"threshold_base_pct": 0.11},
                    }
                }
            }
        },
        "live": {"user": "tester"},
    }
    _write_config(override_path, override_cfg)
    _write_config(base_path, base_cfg)

    loaded = config_utils.load_config(str(base_path), verbose=False)
    parsed = config_utils.parse_overrides(deepcopy(loaded), verbose=False)

    hype = parsed["coin_overrides"]["HYPE"]
    long_ov = hype.get("bot", {}).get("long", {})
    assert long_ov["strategy"]["trailing_martingale"]["entry"][
        "threshold_base_pct"
    ] == pytest.approx(0.11)
    assert "entry_cooldown_minutes" not in long_ov.get("risk", {})
    assert "risk_entry_cooldown_minutes" not in long_ov
    short_ov = hype.get("bot", {}).get("short", {})
    assert "entry_cooldown_minutes" not in short_ov.get("risk", {})
    assert "risk_entry_cooldown_minutes" not in short_ov
