"""Commands expose one HSL implementation and no engine selector."""

import argparse

import pytest

from config import prepare_config
from config.schema import get_template_config
from config_utils import add_config_arguments, project_template_config_for_cli


@pytest.mark.parametrize("command", ["live", "backtest", "optimize"])
@pytest.mark.parametrize(
    "flag", ["--hsl-engine", "--live.hsl_engine", "--live_hsl_engine"]
)
def test_removed_engine_selector_is_rejected(command, flag):
    parser = argparse.ArgumentParser()
    add_config_arguments(
        parser,
        project_template_config_for_cli(get_template_config(), command),
        command=command,
    )
    with pytest.raises(SystemExit):
        parser.parse_args([flag, "revised"])
    assert "hsl_engine" not in parser.format_help()


@pytest.mark.parametrize("selector", [None, "revised", "hsl"])
def test_previous_current_engine_selector_is_removed_during_migration(selector):
    config = get_template_config()
    if selector is not None:
        config["live"]["hsl_engine"] = selector
    normalized = prepare_config(config, verbose=False, target="canonical", runtime=None)
    assert "hsl_engine" not in normalized["live"]


def test_removed_engine_requires_explicit_migration():
    config = get_template_config()
    config["live"]["hsl_engine"] = "legacy"
    with pytest.raises(ValueError, match="migrate-hsl"):
        prepare_config(config, verbose=False, target="canonical", runtime=None)
