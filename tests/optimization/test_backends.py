import argparse
import copy

import pytest

from config_utils import (
    add_config_arguments,
    format_config,
    get_template_config,
    update_config_with_args,
)
from optimization.backends import get_backend_runner
from optimization.backends.deap_backend import run_backend as run_deap_backend
from optimization.backends.pymoo_backend import run_backend as run_pymoo_backend


def _optimize_parser():
    parser = argparse.ArgumentParser()
    template = get_template_config()
    del template["bot"]
    keep_live_keys = {
        "approved_coins",
        "minimum_coin_age_days",
    }
    for key in sorted(template["live"]):
        if key not in keep_live_keys:
            del template["live"][key]
    add_config_arguments(parser, template)
    return parser


def test_format_config_defaults_backend_to_deap():
    current = copy.deepcopy(get_template_config())
    current["optimize"].pop("backend", None)

    out = format_config(current, verbose=False)

    assert out["optimize"]["backend"] == "deap"
    assert "deap" in out["optimize"]
    assert "pymoo" in out["optimize"]
    assert "shared" in out["optimize"]["deap"]
    assert "shared" in out["optimize"]["pymoo"]
    assert "algorithms" in out["optimize"]["pymoo"]


def test_format_config_normalizes_backend_name():
    current = copy.deepcopy(get_template_config())
    current["optimize"]["backend"] = "PyMoO"

    out = format_config(current, verbose=False)

    assert out["optimize"]["backend"] == "pymoo"


def test_format_config_rejects_unknown_backend():
    current = copy.deepcopy(get_template_config())
    current["optimize"]["backend"] = "unknown"

    with pytest.raises(ValueError, match="optimize.backend must be one of"):
        format_config(current, verbose=False)


def test_format_config_migrates_flat_optimizer_hyperparams_into_backend_sections():
    current = copy.deepcopy(get_template_config())
    current["optimize"]["backend"] = "pymoo"
    current["optimize"].pop("deap", None)
    current["optimize"].pop("pymoo", None)
    current["optimize"]["crossover_eta"] = 17.0
    current["optimize"]["crossover_probability"] = 0.33
    current["optimize"]["mutation_eta"] = 11.0
    current["optimize"]["mutation_indpb"] = 0.22
    current["optimize"]["mutation_probability"] = 0.44
    current["optimize"]["offspring_multiplier"] = 1.7

    out = format_config(current, verbose=False)

    assert "crossover_eta" not in out["optimize"]
    assert out["optimize"]["deap"]["shared"]["crossover_eta"] == 17.0
    assert out["optimize"]["deap"]["shared"]["crossover_probability"] == 0.33
    assert out["optimize"]["deap"]["shared"]["mutation_eta"] == 11.0
    assert out["optimize"]["deap"]["shared"]["mutation_indpb"] == 0.22
    assert out["optimize"]["deap"]["shared"]["mutation_probability"] == 0.44
    assert out["optimize"]["deap"]["shared"]["offspring_multiplier"] == 1.7
    assert out["optimize"]["pymoo"]["shared"]["crossover_eta"] == 17.0
    assert out["optimize"]["pymoo"]["shared"]["crossover_prob_var"] == 0.33
    assert out["optimize"]["pymoo"]["shared"]["mutation_eta"] == 11.0
    assert out["optimize"]["pymoo"]["shared"]["mutation_prob_var"] == 0.22


def test_format_config_preserves_nested_backend_specific_settings():
    current = copy.deepcopy(get_template_config())
    current["optimize"]["deap"]["shared"]["crossover_eta"] = 31.0
    current["optimize"]["pymoo"]["algorithm"] = "nsga2"
    current["optimize"]["pymoo"]["shared"]["crossover_prob_var"] = 0.41
    current["optimize"]["pymoo"]["algorithms"]["nsga3"]["ref_dirs"]["n_partitions"] = 9

    out = format_config(current, verbose=False)

    assert out["optimize"]["deap"]["shared"]["crossover_eta"] == 31.0
    assert out["optimize"]["pymoo"]["algorithm"] == "nsga2"
    assert out["optimize"]["pymoo"]["shared"]["crossover_prob_var"] == 0.41
    assert out["optimize"]["pymoo"]["algorithms"]["nsga3"]["ref_dirs"]["n_partitions"] == 9


def test_format_config_rejects_unknown_pymoo_algorithm():
    current = copy.deepcopy(get_template_config())
    current["optimize"]["pymoo"]["algorithm"] = "moead"

    with pytest.raises(ValueError, match="optimize.pymoo.algorithm must be one of"):
        format_config(current, verbose=False)


def test_format_config_accepts_auto_pymoo_algorithm():
    current = copy.deepcopy(get_template_config())
    current["optimize"]["pymoo"]["algorithm"] = "auto"

    out = format_config(current, verbose=False)

    assert out["optimize"]["pymoo"]["algorithm"] == "auto"


def test_format_config_normalizes_zero_mutation_prob_sentinels_centrally():
    current = copy.deepcopy(get_template_config())
    current["optimize"]["deap"]["shared"]["mutation_indpb"] = 0.0
    current["optimize"]["pymoo"]["shared"]["mutation_prob_var"] = 0.0

    out = format_config(current, verbose=False)

    assert out["optimize"]["deap"]["shared"]["mutation_indpb"] > 0.0
    assert out["optimize"]["pymoo"]["shared"]["mutation_prob_var"] > 0.0


def test_format_config_rejects_invalid_offspring_multiplier():
    current = copy.deepcopy(get_template_config())
    current["optimize"]["deap"]["shared"]["offspring_multiplier"] = 0.0

    with pytest.raises(ValueError, match="offspring_multiplier"):
        format_config(current, verbose=False)


def test_format_config_rejects_invalid_pymoo_mutation_prob_var():
    current = copy.deepcopy(get_template_config())
    current["optimize"]["pymoo"]["shared"]["mutation_prob_var"] = 1.5

    with pytest.raises(ValueError, match="mutation_prob_var"):
        format_config(current, verbose=False)


def test_format_config_rejects_invalid_nsga3_ref_dirs_method():
    current = copy.deepcopy(get_template_config())
    current["optimize"]["pymoo"]["algorithms"]["nsga3"]["ref_dirs"]["method"] = "bad"

    with pytest.raises(ValueError, match="ref_dirs.method"):
        format_config(current, verbose=False)


def test_optimizer_backend_cli_alias_updates_config():
    parser = _optimize_parser()
    args = parser.parse_args(["--optimizer-backend", "pymoo"])
    config = copy.deepcopy(get_template_config())

    update_config_with_args(config, args, verbose=False)
    out = format_config(config, verbose=False)

    assert out["optimize"]["backend"] == "pymoo"


def test_optimizer_backend_cli_explicit_deap_matches_default():
    parser = _optimize_parser()
    args = parser.parse_args(["--optimizer-backend", "deap"])
    config = copy.deepcopy(get_template_config())

    update_config_with_args(config, args, verbose=False)
    out = format_config(config, verbose=False)

    assert out["optimize"]["backend"] == "deap"


def test_get_backend_runner_resolves_supported_backends():
    assert get_backend_runner("deap") is run_deap_backend
    assert get_backend_runner("pymoo") is run_pymoo_backend


def test_get_backend_runner_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unsupported optimizer backend"):
        get_backend_runner("unknown")
