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


def test_suite_flat_coin_override_cooldown_path_updates_grouped_form():
    """Flat coin_overrides dotted selectors must remap to the grouped durable path."""
    from config import compile_runtime_config, get_template_config
    from config.param_paths import require_existing_config_path
    from config.shared_bot import inject_flattened_shared_bot_side

    cfg = get_template_config()
    cfg["coin_overrides"] = {
        "HYPE": {
            "bot": {
                "long": {
                    "risk": {"entry_cooldown_minutes": 50.0},
                }
            }
        }
    }
    # Prior compile inject leaves a flat mirror alongside the group.
    inject_flattened_shared_bot_side(cfg["coin_overrides"]["HYPE"]["bot"]["long"])
    cfg["coin_overrides"]["HYPE"]["bot"]["long"]["risk_entry_cooldown_minutes"] = 50.0

    resolved = require_existing_config_path(
        cfg, "coin_overrides.HYPE.bot.long.risk_entry_cooldown_minutes"
    )
    assert resolved == (
        "coin_overrides",
        "HYPE",
        "bot",
        "long",
        "risk",
        "entry_cooldown_minutes",
    )
    # Suite-style write through the resolved path.
    target = cfg
    for part in resolved[:-1]:
        target = target[part]
    target[resolved[-1]] = 12.0

    compiled = compile_runtime_config(cfg, runtime="backtest", record_step=False)
    side = compiled["coin_overrides"]["HYPE"]["bot"]["long"]
    assert side["risk"]["entry_cooldown_minutes"] == pytest.approx(12.0)
    assert side["risk_entry_cooldown_minutes"] == pytest.approx(12.0)


def test_cli_flat_coin_override_cooldown_survives_compile_discard():
    """Iterative/CLI flat selectors must remap before compile discards flat aliases."""
    from config import compile_runtime_config, get_template_config
    from config.param_paths import resolve_dotted_config_path
    from config.shared_bot import inject_flattened_shared_bot_side
    from config_utils import set_nested_value_safe

    cfg = get_template_config()
    cfg["coin_overrides"] = {
        "HYPE": {
            "bot": {
                "long": {
                    "risk": {"entry_cooldown_minutes": 50.0},
                }
            }
        }
    }
    inject_flattened_shared_bot_side(cfg["coin_overrides"]["HYPE"]["bot"]["long"])

    dotted = "coin_overrides.HYPE.bot.long.risk_entry_cooldown_minutes"
    resolved = resolve_dotted_config_path(cfg, dotted)
    assert resolved == (
        "coin_overrides",
        "HYPE",
        "bot",
        "long",
        "risk",
        "entry_cooldown_minutes",
    )
    assert set_nested_value_safe(cfg, list(resolved), 12.0, create_missing=True)

    compiled = compile_runtime_config(cfg, runtime="backtest", record_step=False)
    side = compiled["coin_overrides"]["HYPE"]["bot"]["long"]
    assert side["risk"]["entry_cooldown_minutes"] == pytest.approx(12.0)
    assert side["risk_entry_cooldown_minutes"] == pytest.approx(12.0)


def test_file_override_non_finite_entry_cooldown_fails_closed(tmp_path):
    """Invalid cooldown in override_config_path must not fall back to global."""
    base_cfg = config_utils.get_template_config()
    base_cfg["live"]["user"] = "tester"
    base_cfg["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 7.0
    base_path = tmp_path / "base.json"
    base_cfg["live"]["base_config_path"] = str(base_path)
    override_path = tmp_path / "bad.json"
    base_cfg["coin_overrides"] = {
        "HYPE": {"override_config_path": str(override_path)},
    }
    # 1e309 becomes +inf when coerced to float; JSON cannot encode NaN portably.
    override_cfg = {
        "bot": {
            "long": {
                "risk": {"entry_cooldown_minutes": 1e309},
            }
        },
        "live": {"user": "tester"},
    }
    _write_config(override_path, override_cfg)
    _write_config(base_path, base_cfg)

    loaded = config_utils.load_config(str(base_path), verbose=False)
    with pytest.raises((ValueError, KeyError), match="entry_cooldown_minutes"):
        config_utils.parse_overrides(deepcopy(loaded), verbose=False)


def test_load_override_config_propagates_value_errors(tmp_path):
    """Loader validation failures must not be swallowed into an empty override."""
    from config.overrides import load_override_config

    override_path = tmp_path / "missing_but_declared.json"
    # Path is checked for existence before loader call; create a placeholder file.
    override_path.write_text("{}", encoding="utf-8")
    config = {
        "live": {},
        "coin_overrides": {
            "HYPE": {"override_config_path": str(override_path)},
        },
    }

    def boom(_path):
        raise ValueError(
            "bot.long.risk.entry_cooldown_minutes must be a finite number >= 0.0"
        )

    with pytest.raises(ValueError, match="entry_cooldown_minutes must be a finite number"):
        load_override_config(config, "HYPE", config_loader=boom)


def test_file_override_preserves_flat_strategy_param_moved_during_prepare(tmp_path):
    """Raw flat strategy keys must survive prune after preparation nests them."""
    base_cfg = config_utils.get_template_config()
    base_cfg["live"]["user"] = "tester"
    base_cfg["bot"]["long"]["strategy"]["trailing_martingale"]["ema_span_0"] = 100.0
    base_cfg["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 7.0
    base_path = tmp_path / "base.json"
    base_cfg["live"]["base_config_path"] = str(base_path)
    override_path = tmp_path / "flat_strategy.json"
    base_cfg["coin_overrides"] = {
        "HYPE": {"override_config_path": str(override_path)},
    }
    # Supported load path: flat side strategy field + grouped cooldown.
    override_cfg = {
        "bot": {
            "long": {
                "ema_span_0": 321.0,
                "risk": {"entry_cooldown_minutes": 50.0},
            }
        },
        "live": {"user": "tester"},
    }
    _write_config(override_path, override_cfg)
    _write_config(base_path, base_cfg)

    loaded = config_utils.load_config(str(base_path), verbose=False)
    parsed = config_utils.parse_overrides(deepcopy(loaded), verbose=False)

    hype = parsed["coin_overrides"]["HYPE"]["bot"]["long"]
    assert hype["risk"]["entry_cooldown_minutes"] == pytest.approx(50.0)
    assert hype["strategy"]["trailing_martingale"]["ema_span_0"] == pytest.approx(321.0)


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
