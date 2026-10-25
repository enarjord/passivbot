"""Owned optimizer quantities retain pymoo's external probability gates."""

from copy import deepcopy

import numpy as np
import pytest

from cli_utils import build_command_parser
from config import get_template_config, prepare_config
from config.authoring import compact_config, resolved_config
from config.field_registry import get_field_registry
from released_v8_fixtures import released_v8_policy
from config_utils import add_config_arguments
from optimization.backends import pymoo_backend
from optimization.bounds import Bound
from tools.clean_config import cleanup_config
from tools.migrate_config_v9 import migrate


RENAMES = [
    ("crossover_eta", "crossover_distribution_index", 31.5),
    ("crossover_prob_var", "crossover_probability_per_variable", 0.375),
    ("mutation_eta", "mutation_distribution_index", 12.75),
    ("mutation_prob", "mutation_probability", 0.125),
    ("mutation_prob_per_variable", "mutation_probability_per_variable", 0.625),
]


def source():
    config = get_template_config()
    config["optimize"]["bounds"] = {}
    return config


@pytest.mark.parametrize("old,new,value", RENAMES)
@pytest.mark.parametrize("channel", ["root", "fixed", "scenario"])
def test_old_optimizer_probability_spellings_are_rejected_without_conversion(old, new, value, channel):
    config = source()
    path = f"optimize.pymoo.shared.{old}"
    if channel == "root":
        config["optimize"]["pymoo"]["shared"][old] = value
    elif channel == "fixed":
        config["optimize"]["fixed_runtime_overrides"] = {path: value}
    else:
        config["backtest"]["scenarios"] = [{"label": "invalid", "overrides": {path: value}}]
    before = deepcopy(config)
    with pytest.raises((KeyError, TypeError, ValueError), match=old):
        prepare_config(config, verbose=False)
    assert config == before
    if channel == "root":
        with pytest.raises((KeyError, TypeError, ValueError), match=old):
            migrate(config)
        assert config == before


@pytest.mark.parametrize("old,new,value", RENAMES)
def test_optimizer_probability_values_roundtrip_and_registry_exposes_only_final_name(old, new, value):
    config = source()
    config["optimize"]["pymoo"]["shared"][new] = value
    prepared = prepare_config(config, verbose=False)
    for export in (cleanup_config, compact_config, resolved_config):
        assert prepare_config(export(prepared), verbose=False)["optimize"]["pymoo"]["shared"][new] == value
    path = ("optimize", "pymoo", "shared", new)
    field = get_field_registry().find(path)
    assert field is not None
    assert field.unit == ("distribution index" if new.endswith("_index") else "probability")
    assert get_field_registry().find(("optimize", "pymoo", "shared", old)) is None


@pytest.fixture(scope="module")
def parser():
    result = build_command_parser(prog="optimize", description="", usage="", epilog="")
    add_config_arguments(result, source(), command="optimize", help_all=True)
    return result


@pytest.mark.parametrize("old,new,value", RENAMES)
def test_cli_final_probability_paths_work_and_retired_paths_fail(parser, old, new, value):
    name = f"optimize.pymoo.shared.{new}"
    assert float(vars(parser.parse_args([f"--{name}", str(value)]))[name]) == value
    with pytest.raises(SystemExit) as error:
        parser.parse_args([f"--optimize.pymoo.shared.{old}", str(value)])
    assert error.value.code == 2


@pytest.mark.parametrize("flag", ["--optimize.pymoo.shared.mutation_prob_var", "--optimize_pymoo_shared_mutation_prob_var", "--optimize_pymoo_shared_mutation_prob"])
def test_retired_full_mutation_flag_aliases_are_not_preserved(parser, flag):
    with pytest.raises(SystemExit) as error:
        parser.parse_args([flag, "0.125"])
    assert error.value.code == 2


def test_explicit_mutation_short_alias_keeps_its_numeric_meaning(parser):
    assert float(vars(parser.parse_args(["-psmpv", "0.125"]))["optimize.pymoo.shared.mutation_probability"]) == 0.125


@pytest.mark.parametrize("n_params", [1, 2, 5])
@pytest.mark.parametrize("automatic", [False, True])
def test_owned_names_reach_real_pymoo_operators_without_changing_kwargs_or_auto_formulas(monkeypatch, n_params, automatic):
    config = source()
    shared = config["optimize"]["pymoo"]["shared"]
    shared.update({new: value for old, new, value in RENAMES})
    if automatic:
        shared.update(mutation_probability="auto", mutation_probability_per_variable="auto")
    captured = {}
    for name in ("SBX", "PM"):
        original = getattr(pymoo_backend, name)
        def create(_original=original, _name=name, **kwargs):
            captured[_name] = kwargs
            return _original(**kwargs)
        monkeypatch.setattr(pymoo_backend, name, create)
    algorithm = pymoo_backend._build_algorithm(
        config=config, sampling=np.zeros((4, n_params)),
        bounds=[Bound(0.0, 1.0)] * n_params, sig_digits=3,
        population_plan={"algorithm_name": "nsga2", "requested_population_size": 4, "actual_population_size": 4},
    )
    assert algorithm is not None
    assert captured["SBX"] == {"prob_var": 0.375, "eta": 31.5}
    assert captured["PM"] == {
        "prob": 1.0 / n_params if automatic else 0.125,
        "prob_var": min(0.5, 1.0 / n_params) if automatic else 0.625,
        "eta": 12.75,
    }


@pytest.mark.parametrize("retired_mutation", [False, True])
def test_released_v8_optimizer_values_and_report_targets_remain_exact(retired_mutation):
    config = released_v8_policy()
    shared = config["optimize"]["pymoo"]["shared"]
    shared.update({old: value for old, new, value in RENAMES})
    if retired_mutation:
        shared["mutation_prob_var"] = shared.pop("mutation_prob")
    before = deepcopy(config)
    result = migrate(config, memory_span_minutes=123.5)
    assert config == before
    output = result.config["optimize"]["pymoo"]["shared"]
    for old, new, value in RENAMES:
        assert output[new] == value
        target = f"optimize.pymoo.shared.{new}"
        assert any(isinstance(row["new"], dict) and (target in row["new"] or new in row["new"]) for row in result.changes)
    assert not ({old for old, new, value in RENAMES} & output.keys())


@pytest.mark.parametrize("name", ["crossover_probability_per_variable", "mutation_probability", "mutation_probability_per_variable"])
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_probability_endpoint_domains_are_unchanged(name, value):
    config = source()
    config["optimize"]["pymoo"]["shared"][name] = value
    assert prepare_config(config, verbose=False)["optimize"]["pymoo"]["shared"][name] == value


@pytest.mark.parametrize("name", ["mutation_probability", "mutation_probability_per_variable"])
@pytest.mark.parametrize("value", ["auto", " AUTO "])
def test_mutation_auto_sentinel_normalization_is_unchanged(name, value):
    config = source()
    config["optimize"]["pymoo"]["shared"][name] = value
    assert prepare_config(config, verbose=False)["optimize"]["pymoo"]["shared"][name] == "auto"
