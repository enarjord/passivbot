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


def test_absent_engine_selector_uses_the_sole_engine():
    normalized = prepare_config(
        get_template_config(), verbose=False, target="canonical", runtime=None
    )
    assert "hsl_engine" not in normalized["live"]


@pytest.mark.parametrize(
    "selector", [None, "revised", "hsl", "legacy", "unknown", True, 0]
)
def test_explicit_engine_selectors_require_migration(selector):
    config = get_template_config()
    config["live"]["hsl_engine"] = selector
    with pytest.raises(ValueError, match="migrate-hsl"):
        prepare_config(config, verbose=False, target="canonical", runtime=None)


def test_removed_engine_requires_explicit_migration():
    config = get_template_config()
    config["live"]["hsl_engine"] = "legacy"
    with pytest.raises(ValueError, match="migrate-hsl"):
        prepare_config(config, verbose=False, target="canonical", runtime=None)
