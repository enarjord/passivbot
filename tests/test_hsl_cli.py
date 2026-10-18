"""V9 commands reject retired controls and require explicit policy migration."""

import argparse
from copy import deepcopy
import json

import pytest

from config import prepare_config
from config.schema import get_template_config
from config_utils import add_config_arguments, project_template_config_for_cli
from passivbot_cli.main import main as cli_main
from tools.migrate_config_v9 import migrate as migrate_config_v9


def _v8_config():
    config = get_template_config()
    config["config_version"] = "v8.5.0"
    for key in ("lookback_ema_span_minutes", "risk_bootstrap_max_lookback_minutes",
                "realized_loss_gate_lookback_minutes"):
        config["live"].pop(key, None)
    config["live"]["pnls_max_lookback_days"] = 30.0
    config["optimize"]["bounds"].pop("live", None)
    for side in ("long", "short"):
        config["bot"][side]["hsl"]["restart_after_red_policy"] = "always"
    return config


@pytest.mark.parametrize("command", ["live", "backtest", "optimize"])
@pytest.mark.parametrize(
    "flag", ["--hsl-engine", "--live.hsl_engine", "--live_hsl_engine",
             "--bot.long.hsl.restart_after_red_policy",
             "--bot_long_hsl_restart_after_red_policy"]
)
def test_removed_engine_selector_is_rejected(command, flag):
    parser = argparse.ArgumentParser()
    add_config_arguments(
        parser,
        project_template_config_for_cli(get_template_config(), command),
        command=command,
    )
    with pytest.raises(SystemExit):
        parser.parse_args([flag, "always" if "restart_after" in flag else "revised"])
    assert "hsl_engine" not in parser.format_help()
    assert "restart_after_red_policy" not in parser.format_help()


def test_absent_engine_selector_uses_the_sole_engine():
    normalized = prepare_config(
        get_template_config(), verbose=False, target="canonical", runtime=None
    )
    assert "hsl_engine" not in normalized["live"]


@pytest.mark.parametrize(
    "selector", [None, "revised", "hsl", "legacy", "unknown", True, 0]
)
def test_explicit_engine_selectors_require_migration_when_enabled(selector):
    config = get_template_config()
    config["live"]["hsl_engine"] = selector
    config["bot"]["long"]["hsl"]["enabled"] = True
    with pytest.raises(ValueError, match="migrate-config-v9"):
        prepare_config(config, verbose=False, target="canonical", runtime=None)


def test_removed_engine_requires_explicit_migration():
    config = get_template_config()
    config["live"]["hsl_engine"] = "legacy"
    config["bot"]["long"]["hsl"]["enabled"] = True
    with pytest.raises(ValueError, match="migrate-config-v9"):
        prepare_config(config, verbose=False, target="canonical", runtime=None)


@pytest.mark.parametrize("command", ["live", "backtest", "optimize"])
@pytest.mark.parametrize("version", ["v8.5.0", "v8.6.0", "v9.0.0"])
def test_runtime_schema_override_is_rejected_before_any_config_mutation(command, version):
    from config.hsl import generated_template
    from config_utils import update_config_with_args

    config = generated_template(get_template_config())
    config["config_version"] = version
    config["bot"]["long"]["hsl"]["enabled"] = True
    original = deepcopy(config)
    parser = argparse.ArgumentParser()
    keys = add_config_arguments(
        parser,
        project_template_config_for_cli(get_template_config(), command),
        command=command,
        help_all=True,
    )
    args = parser.parse_args(
        ["--fee-pct-fallback", "0.123", "--config_version", "v8.6.0"]
    )
    with pytest.raises(
        ValueError, match="config_version.*cannot be overridden.*migrate-config-v9"
    ):
        update_config_with_args(config, args, allowed_keys=keys)
    assert config == original


def test_fake_hsl_example_preserves_automatic_gpu_sizing():
    import json
    from pathlib import Path

    example = json.loads(
        (Path(__file__).parents[1] / "configs/examples/fake_live_hsl.json").read_text()
    )
    policy = example["optimize"]["gpu"]
    effective = prepare_config(
        example, verbose=False, target="canonical", runtime=None
    )["optimize"]["gpu"]
    canonical = get_template_config()["optimize"]["gpu"]
    assert policy["batch_size"] == effective["batch_size"] == canonical["batch_size"] is None
    assert set(policy) == set(canonical)


@pytest.mark.parametrize("command", ["live", "backtest", "optimize"])
@pytest.mark.parametrize("enable", [False, True])
def test_legacy_schema_requires_explicit_migration_after_cli_enablement(
    command, enable, tmp_path
):
    from config_utils import update_config_with_args
    config = _v8_config()
    config["live"]["hsl_engine"] = "legacy"
    config["bot"]["long"]["hsl"]["enabled"] = not enable
    parser = argparse.ArgumentParser()
    keys = add_config_arguments(parser, project_template_config_for_cli(get_template_config(), command), command=command)
    args = parser.parse_args(["--bot.long.hsl.enabled", str(enable).lower()])
    update_config_with_args(config, args, allowed_keys=keys)
    assert config["bot"]["long"]["hsl"]["enabled"] is enable
    with pytest.raises(ValueError, match="migrate-config-v9"):
        prepare_config(config, verbose=False)
    source = tmp_path / "input.v8.json"
    destination = tmp_path / "output.v9.json"
    source.write_text(json.dumps(config))
    original = source.read_bytes()
    assert cli_main([
        "tool", "migrate-config-v9", str(source), str(destination),
        "--lookback-ema-span-minutes", "43200.0"
    ]) == 0
    assert source.read_bytes() == original
    normalized = prepare_config(json.loads(destination.read_text()), verbose=False)
    assert normalized["bot"]["long"]["hsl"]["enabled"] is enable
    assert "hsl_engine" not in normalized["live"]
    assert "restart_after_red_policy" not in normalized["bot"]["long"]["hsl"]


@pytest.mark.parametrize("enable", [False, True])
def test_cli_hsl_enablement_overrides_legacy_flat_alias(enable):
    from config_utils import update_config_with_args
    config = _v8_config()
    config["bot"]["long"]["hsl_enabled"] = not enable
    update_config_with_args(config, argparse.Namespace(**{"bot.long.hsl.enabled": enable}))
    assert config["bot"]["long"]["hsl_enabled"] is enable
    with pytest.raises(ValueError, match="migrate-config-v9"):
        prepare_config(config, verbose=False)
    migrated = migrate_config_v9(config, memory_span_minutes=43200.0).config
    assert prepare_config(migrated, verbose=False)["bot"]["long"]["hsl"]["enabled"] is enable


def test_retired_hsl_migration_cli_rejects_without_writing(tmp_path, capsys):
    source = tmp_path / "input.v8.json"
    destination = tmp_path / "output.json"
    source.write_text(json.dumps(_v8_config()))
    original = source.read_bytes()
    assert cli_main(["tool", "migrate-hsl", str(source), str(destination)]) == 2
    message = capsys.readouterr().err
    assert "migrate-config-v9" in message
    assert "lookback-ema-span-minutes" in message
    assert source.read_bytes() == original
    assert not destination.exists()


@pytest.mark.parametrize("command", ["live", "backtest", "optimize"])
def test_cli_supports_global_excess_budget_scaling_and_removes_allowance_selector(
    command,
):
    from config_utils import update_config_with_args

    config = get_template_config()
    parser = argparse.ArgumentParser()
    keys = add_config_arguments(
        parser, project_template_config_for_cli(config, command), command=command
    )
    args = parser.parse_args(
        ["--bot.long.hsl.scale_budget_with_excess_allowance", "true"]
    )
    update_config_with_args(config, args, allowed_keys=keys)
    assert (
        prepare_config(config, verbose=False)["bot"]["long"]["hsl"][
            "scale_budget_with_excess_allowance"
        ]
        is True
    )
    assert "we_excess_allowance_mode" not in parser.format_help()
