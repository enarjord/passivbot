from copy import deepcopy

import json
import os

import pytest

import config_utils
from config.load import prepare_config
from config.overrides import load_override_config


def _write_config(path, cfg):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def test_override_path_resolves_relative_to_base_config(tmp_path):
    base_cfg = config_utils.get_template_config()
    base_cfg["live"]["account"]["user"] = "tester"
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
                                "threshold_base_fraction": 0.77,
                            },
                        },
                    },
                },
            },
        }
    }

    override_cfg = config_utils.get_template_config()
    override_cfg["live"]["account"]["user"] = "tester"
    override_cfg["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "threshold_base_fraction"
    ] = 0.99
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
        "threshold_base_fraction"
    ] == pytest.approx(0.99)
    # inline override merged on top
    assert xrp_ov["bot"]["short"]["strategy"]["trailing_martingale"]["entry"][
        "threshold_base_fraction"
    ] == pytest.approx(0.77)
    # Ordinary full-config fields are validated then excluded from the patch.
    assert set(xrp_ov) == {"bot", "live"}
    assert "risk" not in xrp_ov["bot"]
    assert "max_positions" not in xrp_ov["bot"]["long"].get("risk", {})
    assert "total_wallet_exposure_limit" not in xrp_ov["bot"]["long"].get("risk", {})
    assert "forager" not in xrp_ov["bot"]["long"]


def test_override_file_not_found_fails_closed(tmp_path):
    base_cfg = config_utils.get_template_config()
    base_cfg["live"]["account"]["user"] = "tester"
    base_cfg["live"]["base_config_path"] = str(tmp_path / "base.json")
    base_cfg["coin_overrides"] = {
        "DOGE": {"override_config_path": "overrides/missing.json"}
    }
    base_path = tmp_path / "base.json"
    _write_config(base_path, base_cfg)

    with pytest.raises(
        FileNotFoundError,
        match=r"coin_overrides\.DOGE\.override_config_path not found",
    ):
        loaded = config_utils.load_config(str(base_path), verbose=False)
        config_utils.parse_overrides(deepcopy(loaded), verbose=False)


def test_malformed_override_file_fails_closed(tmp_path):
    base_cfg = config_utils.get_template_config()
    base_cfg["live"]["account"]["user"] = "tester"
    base_path = tmp_path / "base.json"
    override_path = tmp_path / "broken.json"
    override_path.write_text('{"bot": ', encoding="utf-8")
    base_cfg["live"]["base_config_path"] = str(base_path)
    base_cfg["coin_overrides"] = {"DOGE": {"override_config_path": override_path.name}}
    _write_config(base_path, base_cfg)

    with pytest.raises(
        ValueError,
        match=r"coin_overrides\.DOGE\.override_config_path .* is invalid",
    ):
        loaded = config_utils.load_config(str(base_path), verbose=False)
        config_utils.parse_overrides(deepcopy(loaded), verbose=False)


def test_structurally_invalid_override_file_fails_closed(tmp_path):
    base_cfg = config_utils.get_template_config()
    base_cfg["live"]["account"]["user"] = "tester"
    base_path = tmp_path / "base.json"
    override_path = tmp_path / "invalid.json"
    override_path.write_text("[]", encoding="utf-8")
    base_cfg["live"]["base_config_path"] = str(base_path)
    base_cfg["coin_overrides"] = {"DOGE": {"override_config_path": override_path.name}}
    _write_config(base_path, base_cfg)

    with pytest.raises(
        ValueError,
        match=r"coin_overrides\.DOGE\.override_config_path .* is invalid",
    ):
        loaded = config_utils.load_config(str(base_path), verbose=False)
        config_utils.parse_overrides(deepcopy(loaded), verbose=False)


def test_unreadable_override_file_error_propagates(tmp_path):
    override_path = tmp_path / "override.json"
    override_path.write_text("{}", encoding="utf-8")
    config = {
        "live": {"base_config_path": str(tmp_path / "base.json")},
        "coin_overrides": {"DOGE": {"override_config_path": override_path.name}},
    }

    def raise_permission_error(path):
        raise PermissionError(f"cannot read {path}")

    with pytest.raises(PermissionError, match="cannot read"):
        load_override_config(config, "DOGE", config_loader=raise_permission_error)


def test_referenced_retired_policy_error_identifies_coin_and_file(tmp_path):
    config = config_utils.get_template_config()
    override = tmp_path / "doge-policy.json"
    _write_config(override, {"bot": {"long": {"hsl": {"restart_after_red_policy": "never"}}}})
    config["coin_overrides"] = {"DOGE": {"override_config_path": str(override)}}
    base = tmp_path / "base.json"
    _write_config(base, config)

    with pytest.raises(ValueError, match=r"coin_overrides\.DOGE.*doge-policy\.json.*migrate-config-v9") as error:
        config_utils.load_config(str(base), verbose=False)
    assert isinstance(error.value.__cause__, ValueError)


def _config_with_file(tmp_path, document, *, scope="coin"):
    config = config_utils.get_template_config()
    config["live"]["account"]["user"] = "tester"
    config["bot"]["global"]["hsl"]["scope"] = scope
    if scope == "portfolio":
        config["bot"]["global"]["hsl"].update(deepcopy(config["bot"]["long"]["hsl"]))
        for side in ("long", "short"):
            config["optimize"]["bounds"][side].pop("hsl", None)
    config["live"]["base_config_path"] = str(tmp_path / "main.json")
    path = tmp_path / "override.json"
    _write_config(path, document)
    config["coin_overrides"] = {"BTC": {"override_config_path": str(path)}}
    return config, path


def _prepare_file_config(config):
    prepared = prepare_config(config, verbose=False)
    return config_utils.parse_overrides(prepared, verbose=False)


@pytest.mark.parametrize("pside", ["long", "short"])
def test_stored_scenario_file_is_not_a_cycle_in_effective_coin_validation(tmp_path, pside):
    from config.optimize_bounds import set_flat_optimize_bound

    document = {"bot": {pside: {"unstuck": {"close_fraction": 0.075}}}}
    config, path = _config_with_file(tmp_path, document)
    scenario = {"label": "file-policy", "overrides": {
        "coin_overrides": {"BTC": {"override_config_path": str(path)}},
    }}
    config["backtest"].update(suite_enabled=True, scenarios=[scenario])
    config["bot"][pside]["entry_cooldown"].update(
        max_duration_minutes=30.0,
        weights_minutes={"exposure_ratio": 1.0, "adverse_directionality": 0.0},
    )
    config["optimize"]["bounds"] = {}
    for suffix, value in (
        ("entry_cooldown_min_duration_minutes", [0.0, 20.0]),
        ("entry_cooldown_max_duration_minutes", [10.0, 40.0]),
        ("entry_cooldown_weights_minutes_exposure_ratio", [0.0, 1.0]),
    ):
        set_flat_optimize_bound(config["optimize"]["bounds"], "trailing_martingale",
                                f"{pside}_{suffix}", value)
    fixed = {f"bot.{pside}.entry_cooldown.max_duration_minutes": 30.0}
    config["optimize"]["fixed_runtime_overrides"] = fixed
    before, source = deepcopy(config), path.read_bytes()

    parsed = _prepare_file_config(config)

    assert parsed["coin_overrides"]["BTC"] == document
    assert parsed["backtest"]["scenarios"] == [scenario]
    assert parsed["backtest"]["suite_enabled"] is True
    assert parsed["optimize"]["fixed_runtime_overrides"] == fixed
    assert config == before
    assert path.read_bytes() == source
    # The numeric pin is required: without it the independent search corners
    # can choose a floor of 20 and a ceiling of 10.
    invalid = deepcopy(config)
    invalid["backtest"]["scenarios"] = []
    invalid["coin_overrides"] = {}
    invalid["optimize"]["fixed_runtime_overrides"] = {}
    with pytest.raises(ValueError, match="min_duration_minutes.*max_duration_minutes"):
        _prepare_file_config(invalid)


def test_effective_coin_validation_preserves_outer_scenario_semantic_errors(tmp_path):
    config, path = _config_with_file(tmp_path, {"live": {"account": {"leverage": 2.0}}})
    config["backtest"]["scenarios"] = [{"label": "invalid-ew", "overrides": {
        "coin_overrides": {"BTC": {"override_config_path": str(path)}},
        "backtest.ew_half_life_days": 10.0,
    }}]
    before, source = deepcopy(config), path.read_bytes()
    with pytest.raises(ValueError, match=r"Scenario overrides cannot set backtest\.ew_half_life_days"):
        _prepare_file_config(config)
    assert config == before
    assert path.read_bytes() == source


def test_override_file_own_scenario_reference_remains_a_real_cycle(tmp_path):
    document = config_utils.get_template_config()
    config, path = _config_with_file(tmp_path, document)
    document["backtest"]["scenarios"] = [{"label": "self-reference", "overrides": {
        "coin_overrides": {"ETH": {"override_config_path": str(path)}},
    }}]
    _write_config(path, document)
    before, source = deepcopy(config), path.read_bytes()
    with pytest.raises(ValueError, match="cyclic override_config_path"):
        _prepare_file_config(config)
    assert config == before
    assert path.read_bytes() == source


def test_sparse_file_keeps_only_authored_pins_and_explicit_wallet_extension(tmp_path):
    document = {"bot": {"long": {
        "wallet_exposure_limit": 0.12,
        "unstuck": {"enabled": False, "ema_distance_fraction": 0.0},
    }}}
    config, path = _config_with_file(tmp_path, document)
    config["coin_overrides"]["BTC"]["bot"] = {"long": {"unstuck": {"ema_distance_fraction": -0.1}}}
    parsed = _prepare_file_config(config)
    assert parsed["coin_overrides"]["BTC"] == {"bot": {"long": {
        "wallet_exposure_limit": 0.12,
        "unstuck": {"enabled": False, "ema_distance_fraction": -0.1},
    }}}
    assert json.loads(path.read_text()) == document


@pytest.mark.parametrize("scope", ["position_side", "portfolio"])
def test_full_file_hsl_cannot_authorize_its_own_scope(tmp_path, caplog, scope):
    document = config_utils.get_template_config()
    document["bot"]["global"]["hsl"]["scope"] = "coin"
    document["bot"]["long"]["hsl"]["enabled"] = True
    document["bot"]["long"]["unstuck"]["enabled"] = False
    config, _ = _config_with_file(tmp_path, document, scope=scope)
    parsed = _prepare_file_config(config)
    assert parsed["bot"]["global"]["hsl"]["scope"] == scope
    assert "hsl" not in parsed["coin_overrides"]["BTC"]["bot"]["long"]
    assert parsed["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]["enabled"] is False
    assert "file HSL values are ignored" in caplog.text


@pytest.mark.parametrize("invalid", ["unknown_root", "unknown_inactive_leaf", "invalid_nonoverridable"])
def test_full_file_rejects_invalid_source_before_filtering(tmp_path, invalid):
    document = config_utils.get_template_config()
    if invalid == "unknown_root":
        document["unknown_setting"] = True
        match = "unknown_setting.*unknown configuration field"
    elif invalid == "unknown_inactive_leaf":
        document["bot"]["long"]["strategy"]["ema_anchor"]["unknown_gene"] = 1.0
        match = "ema_anchor.unknown_gene.*unknown configuration field"
    else:
        document["live"]["modes"]["disapproved_position"] = "invalid-mode"
        match = "modes.disapproved_position"
    config, _ = _config_with_file(tmp_path, document)
    with pytest.raises((ValueError, TypeError), match=match):
        _prepare_file_config(config)


@pytest.mark.parametrize("indirect", [False, True])
def test_file_cycles_fail_and_validation_context_unwinds(tmp_path, indirect):
    first = config_utils.get_template_config()
    config, path = _config_with_file(tmp_path, first)
    second_path = tmp_path / "second.json"
    first["coin_overrides"] = {"ETH": {"override_config_path": str(second_path if indirect else path)}}
    _write_config(path, first)
    if indirect:
        second = config_utils.get_template_config()
        second["coin_overrides"] = {"SOL": {"override_config_path": str(path)}}
        _write_config(second_path, second)
    with pytest.raises(ValueError, match="cyclic override_config_path") as error:
        _prepare_file_config(config)
    assert "override.json" in str(error.value)
    if indirect:
        assert "second.json" in str(error.value)
    _write_config(path, {"live": {"account": {"leverage": 2.0}}})
    assert _prepare_file_config(config)["coin_overrides"]["BTC"] == {
        "live": {"account": {"leverage": 2.0}}
    }


def test_nested_file_configuration_is_validated_without_becoming_parent_pins(tmp_path):
    document = config_utils.get_template_config()
    config, path = _config_with_file(tmp_path, document)
    nested_path = tmp_path / "nested.json"
    document["coin_overrides"] = {"ETH": {"override_config_path": str(nested_path)}}
    _write_config(path, document)
    _write_config(
        nested_path, {"live": {"modes": {"disapproved_position": "invalid-mode"}}}
    )
    with pytest.raises(ValueError, match="nested.json.*modes.disapproved_position"):
        _prepare_file_config(config)
    _write_config(nested_path, {"live": {"account": {"leverage": 2.0}}})
    parsed = _prepare_file_config(config)
    assert "coin_overrides" not in parsed["coin_overrides"]["BTC"]
    assert (
        parsed["coin_overrides"]["BTC"]["live"]["account"]["leverage"]
        == document["live"]["account"]["leverage"]
    )


def test_nested_relative_references_use_the_resolved_override_file_directory(tmp_path):
    base = config_utils.get_template_config()
    main_path = tmp_path / "main.json"
    base["coin_overrides"] = {"BTC": {"override_config_path": "sub/override.json"}}
    document = config_utils.get_template_config()
    document["coin_overrides"] = {"ETH": {"override_config_path": "nested.json"}}
    directory = tmp_path / "sub"
    directory.mkdir()
    _write_config(directory / "override.json", document)
    _write_config(directory / "nested.json", {"live": {"account": {"leverage": 2.0}}})
    _write_config(main_path, base)
    parsed = config_utils.parse_overrides(config_utils.load_config(str(main_path), verbose=False), verbose=False)
    assert "coin_overrides" not in parsed["coin_overrides"]["BTC"]


@pytest.mark.parametrize("version", ["v8.6.0", "v9.1.0"])
def test_full_override_file_requires_supported_version(tmp_path, version):
    document = config_utils.get_template_config()
    document["config_version"] = version
    config, _ = _config_with_file(tmp_path, document)
    with pytest.raises(ValueError, match="explicit migration|newer than supported"):
        _prepare_file_config(config)
