import argparse
import copy
import functools
from multiprocessing.reduction import ForkingPickler

import pytest

from config.scoring import ObjectiveSpec, engine_space_fitness_weights, to_engine_value
from config_utils import (
    add_config_arguments,
    format_config,
    get_template_config,
    update_config_with_args,
)
from optimization.backends import get_backend_runner
from optimization.backends.deap_backend import (
    DEFAULT_DEAP_POPULATION_SIZE,
    _clone_evaluated_individual,
    _evaluate_deap_worker,
    _initialize_deap_worker,
    _resolve_deap_population_size,
)
from optimization.backends.deap_backend import run_backend as run_deap_backend
from optimization.backends.pymoo_backend import run_backend as run_pymoo_backend
from optimization.evaluation_payload import apply_evaluation_payload


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


def test_format_config_defaults_backend_to_pymoo():
    current = copy.deepcopy(get_template_config())
    current["optimize"].pop("backend", None)

    out = format_config(current, verbose=False)

    assert out["optimize"]["backend"] == "pymoo"


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


def test_format_config_normalizes_pymoo_nested_defaults_and_legacy_fallbacks():
    current = copy.deepcopy(get_template_config())
    current["optimize"]["backend"] = "pymoo"
    current["optimize"]["crossover_eta"] = 17.0
    current["optimize"]["crossover_probability"] = 0.33
    current["optimize"]["mutation_eta"] = 11.0
    current["optimize"]["mutation_indpb"] = 0.07
    current["optimize"].pop("pymoo", None)

    out = format_config(current, verbose=False)

    assert out["optimize"]["pymoo"]["algorithm"] == "auto"
    assert out["optimize"]["pymoo"]["shared"] == {
        "crossover_eta": 17.0,
        "crossover_prob_var": 0.33,
        "mutation_eta": 11.0,
        "mutation_prob_var": 0.07,
        "eliminate_duplicates": True,
    }
    assert out["optimize"]["pymoo"]["algorithms"]["nsga3"]["ref_dirs"] == {
        "method": "das_dennis",
        "n_partitions": "auto",
    }


def test_template_defaults_use_null_population_and_pareto_1000():
    current = get_template_config()

    assert current["optimize"]["backend"] == "pymoo"
    assert current["optimize"]["population_size"] is None
    assert current["optimize"]["pareto_max_size"] == 1000
    assert current["optimize"]["seed"] is None


def test_format_config_preserves_null_population_size_for_pymoo():
    current = copy.deepcopy(get_template_config())
    current["optimize"]["backend"] = "pymoo"
    current["optimize"]["population_size"] = None

    out = format_config(current, verbose=False)

    assert out["optimize"]["population_size"] is None


def test_format_config_normalizes_optional_optimizer_seed():
    current = copy.deepcopy(get_template_config())
    current["optimize"]["seed"] = "123"

    out = format_config(current, verbose=False)

    assert out["optimize"]["seed"] == 123

    current["optimize"]["seed"] = "random"
    out = format_config(current, verbose=False)
    assert out["optimize"]["seed"] is None


def test_format_config_rejects_negative_optimizer_seed():
    current = copy.deepcopy(get_template_config())
    current["optimize"]["seed"] = -1

    with pytest.raises(ValueError, match="optimize.seed must be null"):
        format_config(current, verbose=False)


def test_resolve_deap_population_size_uses_fallback_for_null():
    assert _resolve_deap_population_size({"optimize": {"population_size": None}}) == DEFAULT_DEAP_POPULATION_SIZE


def test_deap_backend_apply_payload_uses_worker_evaluation_vector():
    class Fitness:
        values = ()
        constraint_violation = None

    class Candidate(list):
        def __init__(self, values):
            super().__init__(values)
            self.fitness = Fitness()

    individual = Candidate([1.0, 2.0])
    payload = {
        "fitness": (0.3,),
        "constraint_violation": 4.0,
        "metrics": {"objective": 1.0},
        "evaluation_vector": [5.0, 6.0],
    }

    metrics = apply_evaluation_payload(individual, payload)

    assert individual == [5.0, 6.0]
    assert individual.fitness.values == (0.3,)
    assert individual.fitness.constraint_violation == 4.0
    assert individual.constraint_violation == 4.0
    assert metrics == {"objective": 1.0}


def test_clone_evaluated_individual_preserves_fitness_and_metrics():
    class Fitness:
        def __init__(self):
            self.values = ()
            self.valid = False
            self.constraint_violation = None

    class Candidate(list):
        def __init__(self, values):
            super().__init__(values)
            self.fitness = Fitness()

    individual = Candidate([1.0, 2.0])
    individual.fitness.values = (0.7,)
    individual.fitness.valid = True
    individual.fitness.constraint_violation = 1.5
    individual.constraint_violation = 1.5
    individual.evaluation_metrics = {"nested": {"value": 1}}

    clone = _clone_evaluated_individual(individual)

    assert clone is not individual
    assert clone == [1.0, 2.0]
    assert clone.fitness.values == (0.7,)
    assert clone.fitness.constraint_violation == 1.5
    assert clone.constraint_violation == 1.5
    assert clone.evaluation_metrics == {"nested": {"value": 1}}
    individual.evaluation_metrics["nested"]["value"] = 2
    assert clone.evaluation_metrics == {"nested": {"value": 1}}


def test_deap_worker_seed_initializer_is_pickleable():
    ForkingPickler.dumps(functools.partial(_initialize_deap_worker, None, None))


def test_deap_worker_evaluate_uses_initializer_globals():
    class Evaluator:
        def evaluate(self, individual, overrides_list):
            return tuple(individual), tuple(overrides_list)

    _initialize_deap_worker(None, None, Evaluator(), ["override.a"])

    assert _evaluate_deap_worker([1.0, 2.0]) == ((1.0, 2.0), ("override.a",))


def test_deap_worker_task_evaluator_is_pickleable_without_bound_evaluator():
    ForkingPickler.dumps(_evaluate_deap_worker)


def test_get_backend_runner_resolves_supported_backends():
    assert get_backend_runner("deap") is run_deap_backend
    assert get_backend_runner("pymoo") is run_pymoo_backend


def test_get_backend_runner_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unsupported optimizer backend"):
        get_backend_runner("unknown")


def test_engine_space_fitness_weights_keep_deap_in_minimization_mode():
    specs = [
        ObjectiveSpec(metric="adg_strategy_pnl_rebased", goal="max"),
        ObjectiveSpec(metric="drawdown_worst", goal="min"),
    ]

    assert engine_space_fitness_weights(specs) == (-1.0, -1.0)


def test_deap_engine_space_still_prefers_higher_raw_value_for_goal_max():
    pytest.importorskip("deap")
    from deap import creator
    from optimize import ConstraintAwareFitness

    if hasattr(creator, "TestEngineSpaceFitness"):
        del creator.TestEngineSpaceFitness

    creator.create(
        "TestEngineSpaceFitness",
        ConstraintAwareFitness,
        weights=engine_space_fitness_weights(
            [
                ObjectiveSpec(metric="adg_strategy_pnl_rebased", goal="max"),
                ObjectiveSpec(metric="drawdown_worst", goal="min"),
            ]
        ),
    )

    better = creator.TestEngineSpaceFitness(
        (
            to_engine_value(ObjectiveSpec(metric="adg_strategy_pnl_rebased", goal="max"), 10.0),
            to_engine_value(ObjectiveSpec(metric="drawdown_worst", goal="min"), 2.0),
        )
    )
    worse = creator.TestEngineSpaceFitness(
        (
            to_engine_value(ObjectiveSpec(metric="adg_strategy_pnl_rebased", goal="max"), 5.0),
            to_engine_value(ObjectiveSpec(metric="drawdown_worst", goal="min"), 3.0),
        )
    )
    better.constraint_violation = 0.0
    worse.constraint_violation = 0.0

    assert better.dominates(worse)
    assert not worse.dominates(better)

    del creator.TestEngineSpaceFitness
