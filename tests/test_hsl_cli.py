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


@pytest.mark.parametrize("command", ["live", "backtest", "optimize"])
@pytest.mark.parametrize("version", ["v8.5.0", "v8.6.0"])
def test_runtime_schema_override_is_rejected_before_any_config_mutation(command, version):
    from copy import deepcopy
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
        ValueError, match="config_version.*cannot be overridden.*migrate-hsl"
    ):
        update_config_with_args(config, args, allowed_keys=keys)
    assert config == original


def test_fake_hsl_example_preserves_automatic_exact_sizing():
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
    for field in ("exact_workers", "max_pending_exact"):
        assert policy[field] == effective[field] == canonical[field] is None
