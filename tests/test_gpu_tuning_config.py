"""Authored GPU tuning modes share defaults across config entry points."""

import argparse
import json

import pytest
from jsonschema import Draft202012Validator

from config import get_template_config, load_prepared_config, prepare_config
from config.json_schema import generate_schema
from config.schema import get_starter_config
from config_utils import add_config_arguments, update_config_with_args
from tools.clean_config import cleanup_config


def test_default_gpu_tuning_policy():
    for config in (get_template_config(), get_starter_config(),
                   prepare_config({"config_version": "v9.0.0"}, verbose=False)):
        assert config["optimize"]["gpu"]["tuning_mode"] == "conservative"


@pytest.mark.parametrize("authored,expected", [
    ("conservative", "conservative"), (" AuTo ", "conservative"),
    ("max", "max"), ("refresh", "refresh"), ("off", "off"),
])
def test_saved_gpu_tuning_config_round_trip(tmp_path, authored, expected):
    config = get_starter_config()
    config["optimize"]["gpu"]["tuning_mode"] = authored
    Draft202012Validator(generate_schema()).validate(config)
    source = {"config": config, "metrics": {}}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(source))

    prepared = load_prepared_config(str(path), allow_result_records=True, verbose=False)
    assert prepared["optimize"]["gpu"]["tuning_mode"] == expected
    assert cleanup_config(prepared)["optimize"]["gpu"]["tuning_mode"] == expected
    assert json.loads(path.read_text()) == source


@pytest.mark.parametrize("mode", [None, True, "", "extreme"])
def test_invalid_gpu_tuning_mode_is_rejected_by_schema_and_loader(mode):
    config = {"config_version": "v9.0.0", "optimize": {"gpu": {"tuning_mode": mode}}}
    assert not Draft202012Validator(generate_schema()).is_valid(config)
    with pytest.raises(ValueError, match=r"optimize\.gpu\.tuning_mode"):
        prepare_config(config, verbose=False)


def test_gpu_tuning_cli_override_survives_normalization():
    config = get_template_config()
    parser = argparse.ArgumentParser()
    allowed = add_config_arguments(parser, config, command="optimize", help_all=True)
    args = parser.parse_args(["--optimize.gpu.tuning_mode", "max"])
    update_config_with_args(config, args, allowed_keys=allowed)

    prepared = prepare_config(config, verbose=False)
    assert prepared["optimize"]["gpu"]["tuning_mode"] == "max"
