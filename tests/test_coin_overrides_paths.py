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
            "bot": {"short": {"entry_grid_spacing_pct": 0.77}},
        }
    }

    override_cfg = config_utils.get_template_config()
    override_cfg["live"]["user"] = "tester"
    override_cfg["bot"]["long"]["entry_grid_spacing_pct"] = 0.99
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
    assert xrp_ov["bot"]["long"]["entry_grid_spacing_pct"] == pytest.approx(0.99)
    # inline override merged on top
    assert xrp_ov["bot"]["short"]["entry_grid_spacing_pct"] == pytest.approx(0.77)
    # disallowed root key should be stripped
    assert "disallowed_root" not in xrp_ov


def test_override_file_not_found_yields_empty_override(tmp_path, monkeypatch):
    base_cfg = config_utils.get_template_config()
    base_cfg["live"]["user"] = "tester"
    base_cfg["live"]["base_config_path"] = str(tmp_path / "base.json")
    base_cfg["coin_overrides"] = {
        "DOGE": {"override_config_path": "overrides/missing.json"}
    }
    base_path = tmp_path / "base.json"
    _write_config(base_path, base_cfg)

    loaded = config_utils.load_config(str(base_path), verbose=False)
    parsed = config_utils.parse_overrides(deepcopy(loaded), verbose=False)

    # override exists but has no allowed diff because the file was missing
    assert "DOGE" in parsed["coin_overrides"]
    assert parsed["coin_overrides"]["DOGE"] == {}
