"""Runtime bookkeeping cannot become authored per-coin policy."""

from copy import deepcopy
import re

import pytest

from config.load import prepare_config
from config.overrides import parse_overrides
from config.schema import get_expanded_config_prototype


RUNTIME_KEYS = (
    "_optimizer_prepared_dataset_identity",
    "_optimizer_evaluation_contract",
    "_fine_tune_anchor_plan",
    "_optimizer_anchor",
    "_hlcvs_dataset_override_meta",
    "results_dir",
    "results_filename",
)


def runtime_metadata(key):
    if key == "results_dir":
        return "optimizer-output"
    if key == "results_filename":
        return "optimizer-output/all_results.bin"
    return {"sentinel": [1, 2, 3]}


def source_config():
    config = get_expanded_config_prototype()
    config["coin_overrides"] = {
        "BTC": {"bot": {"long": {"unstuck": {"exposure_trigger_ratio": 0.7}}}}
    }
    return config


def prepared_config():
    return prepare_config(source_config(), verbose=False, log_config_transforms=False)


@pytest.mark.parametrize("key", RUNTIME_KEYS)
def test_runtime_metadata_survives_coin_policy_validation_without_mutation(key):
    config = prepared_config()
    config[key] = runtime_metadata(key)
    before = deepcopy(config)

    parsed = parse_overrides(config, verbose=False, symbol_normalizer=lambda coin: coin)

    assert config == before
    assert parsed[key] == before[key]
    if isinstance(config[key], dict):
        assert parsed[key] is not config[key]
    assert parsed["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"][
        "exposure_trigger_ratio"
    ] == 0.7


@pytest.mark.parametrize("key", (*RUNTIME_KEYS, "_optimizer_typo"))
def test_authored_runtime_metadata_remains_rejected(key):
    source = source_config()
    source[key] = runtime_metadata(key)
    before = deepcopy(source)

    with pytest.raises(ValueError, match=re.escape(key) + " is an unknown configuration field"):
        prepare_config(source, verbose=False, log_config_transforms=False)

    assert source == before


@pytest.mark.parametrize("key", [
    "_optimizer_typo", "_optimizer_prepared_dataset_identitiy", "results_dri",
])
def test_internal_unknown_fields_are_not_hidden_by_runtime_projection(key):
    config = prepared_config()
    for runtime_key in RUNTIME_KEYS:
        config[runtime_key] = runtime_metadata(runtime_key)
    config[key] = True
    before = deepcopy(config)

    with pytest.raises(ValueError, match=re.escape(key) + " is an unknown configuration field"):
        parse_overrides(config, verbose=False, symbol_normalizer=lambda coin: coin)

    assert config == before


def test_runtime_metadata_does_not_authorize_unknown_coin_patch_fields():
    config = prepared_config()
    config["_optimizer_prepared_dataset_identity"] = {"sentinel": True}
    config["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]["_optimizer_typo"] = True
    before = deepcopy(config)

    with pytest.raises(ValueError, match="_optimizer_typo.*not overridable"):
        parse_overrides(config, verbose=False, symbol_normalizer=lambda coin: coin)

    assert config == before


def test_evaluation_contract_retains_prepared_identity_and_normalized_coin_policy(monkeypatch):
    from optimization import evaluation_contract

    # Native-code identity is owned separately; this regression covers config
    # projection and the prepared-data identity retained by the contract owner.
    monkeypatch.setattr(evaluation_contract, "evaluation_implementation_identity",
                        lambda: {"version": 1, "sentinel": "implementation"})
    config = prepared_config()
    for key in RUNTIME_KEYS:
        config[key] = runtime_metadata(key)
    config["_optimizer_evaluation_contract"] = {"prepared_data": {"digest": "stale"}}
    fixed_path = ["bot", "long", "entry_cooldown", "base_duration_minutes"]
    fixed_value = config["bot"]["long"]["entry_cooldown"]["base_duration_minutes"]
    config["_fine_tune_anchor_plan"] = {
        "key_paths": [fixed_path],
        "anchors": [{"fixed_values": [{"path": fixed_path, "value": fixed_value}]}],
    }
    identity = {"version": 3, "digest": "a" * 64}
    config["_optimizer_prepared_dataset_identity"] = deepcopy(identity)
    before = deepcopy(config)

    first = evaluation_contract.build_evaluation_contract(config)

    assert config == before
    assert first["prepared_data"] == identity
    assert first["prepared_data"] is not config["_optimizer_prepared_dataset_identity"]
    assert first["anchors"] == {
        "key_paths": [fixed_path],
        "anchors": [[{"path": fixed_path, "value": fixed_value}]],
    }
    assert first["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"][
        "exposure_trigger_ratio"
    ] == 0.7
    changed = deepcopy(config)
    changed["_optimizer_prepared_dataset_identity"]["digest"] = "b" * 64
    changed["results_dir"] = "other-optimizer-output"
    changed["results_filename"] = "other-optimizer-output/all_results.bin"
    second = evaluation_contract.build_evaluation_contract(changed)
    expected = deepcopy(first)
    expected["prepared_data"] = changed["_optimizer_prepared_dataset_identity"]
    assert second == expected and second != first
