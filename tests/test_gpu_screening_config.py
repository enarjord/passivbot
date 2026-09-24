"""Public config loading, migration, CLI and default parity for GPU screening."""

import copy
import json

import pytest

from config import get_template_config, load_prepared_config, prepare_config
from config_utils import clean_config, format_config
from optimization.backends.gpu_backend import GPU_DEFAULTS, _resolve_options
from passivbot_exceptions import GPUScreeningMigrationError


@pytest.mark.parametrize("legacy", [None, {}, {"enabled": False}, {"enabled": "false"},
    {"enabled": False, "history_fractions": "ignored", "screening_scenarios": ["gone"]}])
@pytest.mark.parametrize("wrapper", [False, True])
def test_disabled_legacy_migrates_without_enabling_screening(tmp_path, caplog, legacy, wrapper):
    source = get_template_config()
    source["optimize"]["gpu"]["successive_halving"] = legacy
    original = copy.deepcopy(source)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"config": source} if wrapper else source))
    prepared = load_prepared_config(str(path), verbose=False)
    assert "successive_halving" not in prepared["optimize"]["gpu"]
    assert prepared["optimize"]["gpu"]["screening"]["scenarios"] == []
    assert "Removed disabled legacy" in caplog.text
    cleaned = clean_config(prepared)
    caplog.clear()
    assert clean_config(format_config(cleaned, verbose=False)) == cleaned
    assert "legacy" not in caplog.text
    assert source == original


@pytest.mark.parametrize("prepare", [prepare_config, format_config, clean_config, _resolve_options])
def test_enabled_legacy_fails_actionably_even_with_new_screening(prepare):
    source = get_template_config()
    source["optimize"]["gpu"]["successive_halving"] = {
        "enabled": True, "history_fractions": [0.1, 1.0],
    }
    source["optimize"]["gpu"]["screening"]["scenarios"] = ["recent"]
    with pytest.raises(GPUScreeningMigrationError) as exc:
        prepare(source)
    message = str(exc.value)
    for token in ("backtest.scenarios", "screening.scenarios", "survival_fraction",
                  "min_survivors", "remove", "new optimization"):
        assert token in message
    assert source["optimize"]["gpu"]["successive_halving"]["enabled"] is True


def test_disabled_legacy_preserves_explicit_new_policy(caplog):
    source = get_template_config()
    policy = {"scenarios": ["recent"], "survival_fraction": .2, "min_survivors": 32}
    source["optimize"]["gpu"].update(screening=policy, successive_halving={"enabled": False})
    result = prepare_config(source, verbose=False)
    assert result["optimize"]["gpu"]["screening"] == policy
    assert "successive_halving" not in clean_config(result)["optimize"]["gpu"]


def test_screening_defaults_agree_across_schema_normalizer_cleaner_and_runtime():
    config = get_template_config()
    expected = GPU_DEFAULTS["screening"]
    for payload in (config, format_config(config, verbose=False), clean_config(config)):
        assert payload["optimize"]["gpu"]["screening"] == expected
        assert _resolve_options(payload)["screening"] == expected
        assert "successive_halving" not in payload["optimize"]["gpu"]


@pytest.mark.parametrize("command", ["backtest", "optimize"])
def test_cli_migration_error_is_concise_and_nonzero(monkeypatch, capsys, command):
    from passivbot_cli import main as cli
    def run_module(*args, **kwargs):
        source = get_template_config()
        source["optimize"]["gpu"]["successive_halving"] = {"enabled": True}
        prepare_config(source, verbose=False)
    monkeypatch.setattr(cli, "_invoke_module_main", run_module)
    assert cli._run_module(command, "passivbot " + command, []) == 2
    error = capsys.readouterr().err
    assert "Configuration migration required:" in error
    assert "screening.scenarios" in error
    assert "Traceback" not in error


@pytest.mark.parametrize("policy", [
    {"history_fractions": [.1, 1.]}, {"scenarios": None},
    {"scenarios": ["a", "a"]}, {"survival_fraction": 0},
    {"survival_fraction": float("nan")}, {"min_survivors": 0},
])
def test_invalid_screening_fails_before_template_pruning(policy):
    config = get_template_config()
    config["optimize"]["gpu"]["screening"] = policy
    with pytest.raises(ValueError, match="screening"):
        prepare_config(config, verbose=False)


@pytest.mark.parametrize("value,expected", [
    ("recent", ["recent"]), ("recent,older", ["recent", "older"]),
    ('["recent", "older"]', ["recent", "older"]), ("[]", []),
])
def test_screening_cli_override_survives_normalization(value, expected):
    import argparse
    from config_utils import add_config_arguments, update_config_with_args
    config = get_template_config()
    config["optimize"]["gpu"]["screening"]["scenarios"] = ["previous"]
    parser = argparse.ArgumentParser()
    allowed = add_config_arguments(parser, config, command="optimize", help_all=True)
    args = parser.parse_args(["--optimize.gpu.screening.scenarios", value])
    update_config_with_args(config, args, allowed_keys=allowed)
    prepared = prepare_config(config, verbose=False)
    assert prepared["optimize"]["gpu"]["screening"]["scenarios"] == expected
