"""HSL engine selection survives command projection and canonical loading."""
import argparse

import pytest

from config import prepare_config
from config.schema import get_template_config
from config_utils import (
    add_config_arguments,
    project_template_config_for_cli,
    update_config_with_args,
)


@pytest.mark.parametrize("command", ["live", "backtest", "optimize"])
@pytest.mark.parametrize("flag", ["--hsl-engine", "--live.hsl_engine", "--live_hsl_engine"])
@pytest.mark.parametrize("selected", ["legacy", "revised"])
def test_engine_startup_override_reaches_canonical_config(command, flag, selected):
    parser = argparse.ArgumentParser()
    allowed = add_config_arguments(
        parser,
        project_template_config_for_cli(get_template_config(), command),
        command=command,
    )
    args = parser.parse_args([flag, selected])
    config = get_template_config()
    config["live"]["hsl_engine"] = "legacy" if selected == "revised" else "revised"
    for side in ("long", "short"):
        config["bot"][side]["hsl"]["restart_after_red_policy"] = "always"
    update_config_with_args(config, args, allowed_keys=allowed)
    normalized = prepare_config(config, verbose=False, target="canonical", runtime=None)
    assert normalized["live"]["hsl_engine"] == selected


@pytest.mark.parametrize("command", ["backtest", "optimize"])
def test_engine_is_visible_in_command_help_without_changing_default(command):
    parser = argparse.ArgumentParser()
    allowed = add_config_arguments(
        parser,
        project_template_config_for_cli(get_template_config(), command),
        command=command,
    )
    assert "--hsl-engine" in parser.format_help()
    config = get_template_config()
    update_config_with_args(config, parser.parse_args([]), allowed_keys=allowed)
    assert config["live"]["hsl_engine"] == "legacy"
