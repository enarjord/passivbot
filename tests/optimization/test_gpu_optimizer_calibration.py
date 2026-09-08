"""Synthetic regressions for proposal controls and independent drift calibration."""

import hashlib
import json

import numpy as np
import pytest
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.operators.mutation.pm import PM

from config.schema import get_template_config
from config.scoring import ObjectiveSpec
from config_utils import format_config
from optimization.backends import pymoo_backend
from optimization.backends.gpu_backend import (
    _build_gpu_nsga2,
    _checkpoint_signature,
    _exact_drift_objectives,
    _gpu_nsga2_checkpoint_contract,
    _ObjectiveScale,
    _proxy_drift_objectives,
    _resolve_max_pending_exact,
    _GPU_SUITE_OBJECTIVES_KEY,
    _GPU_SUITE_UNPENALIZED_OBJECTIVES_KEY,
)
from optimization.bounds import Bound


def _mutation(config, dimensions, backend):
    sampling = np.full((8, dimensions), 0.5)
    if backend == "gpu":
        algorithm = _build_gpu_nsga2(
            config, sampling=sampling, population_size=8, n_params=dimensions
        )
    else:
        algorithm = pymoo_backend._build_algorithm(
            config=config,
            sampling=sampling,
            bounds=[Bound(0.0, 1.0)] * dimensions,
            sig_digits=None,
            population_plan=pymoo_backend._resolve_pymoo_population_plan(config, n_obj=2),
        )
    return algorithm.mating.mutation


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("dimensions", [1, 18, 84])
@pytest.mark.parametrize("legacy_probability", ["auto", 0.2])
def test_legacy_mutation_configs_preserve_actual_offspring(
    backend, dimensions, legacy_probability
):
    config = get_template_config()
    config["optimize"]["pymoo"]["shared"] = {"mutation_prob_var": legacy_probability}
    normalized = format_config(config, verbose=False)
    shared = normalized["optimize"]["pymoo"]["shared"]
    assert "mutation_prob_var" not in shared
    assert shared["mutation_prob"] == legacy_probability
    assert (
        format_config(normalized, verbose=False)["optimize"]["pymoo"]
        == normalized["optimize"]["pymoo"]
    )
    problem = Problem(n_var=dimensions, xl=0.0, xu=1.0)
    population = Population.new(X=np.full((2048, dimensions), 0.5))
    old = PM(
        prob=1.0 / dimensions if legacy_probability == "auto" else legacy_probability, eta=20.0
    )
    expected = old.do(
        problem, population, inplace=False, random_state=np.random.default_rng(42)
    ).get("X")
    actual = (
        _mutation(normalized, dimensions, backend)
        .do(problem, population, inplace=False, random_state=np.random.default_rng(42))
        .get("X")
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize(
    "individual,coordinate,changed", [(0.0, 1.0, False), (1.0, 0.0, False), (1.0, 1.0, True)]
)
def test_mutation_gates_can_be_controlled_independently(backend, individual, coordinate, changed):
    config = get_template_config()
    config["optimize"]["pymoo"]["shared"].update(
        mutation_prob=individual, mutation_prob_per_variable=coordinate
    )
    problem = Problem(n_var=5, xl=0.0, xu=1.0)
    population = Population.new(X=np.full((128, 5), 0.5))
    actual = (
        _mutation(config, 5, backend)
        .do(problem, population, inplace=False, random_state=np.random.default_rng(7))
        .get("X")
    )
    assert bool(np.all(actual != 0.5) if changed else np.all(actual == 0.5))


@pytest.mark.parametrize(
    "workers,quota,explicit,expected",
    [(4, 8, 0, 16), (1, 8, 0, 16), (12, 8, 0, 24), (4, 8, 8, 8), (4, 8, 24, 24)],
)
def test_default_queue_overlaps_full_validation_batches(workers, quota, explicit, expected):
    assert (
        _resolve_max_pending_exact(
            {"max_pending_exact": explicit, "validate_per_generation": quota}, workers
        )
        == expected
    )


def test_drift_keeps_both_objective_scales_despite_large_constraint_penalties():
    specs = [
        ObjectiveSpec(metric="adg_strategy_eq", goal="max"),
        ObjectiveSpec(metric="position_held_time_weighted_mean_hours", goal="min"),
    ]
    rows = [
        {"adg_strategy_eq": x * 0.001, "position_held_time_weighted_mean_hours": x * 1000.0}
        for x in range(1, 5)
    ]
    raw = _proxy_drift_objectives(rows, specs)
    scale = _ObjectiveScale()
    scale.fit(raw)
    np.testing.assert_allclose(scale.spread, [0.0015, 1500.0])
    # Huge penalties remain evolutionary/constraint data; they cannot flatten
    # either raw objective in proxy/exact drift comparisons.
    payload = {"F": [1e12, 1e12], "metrics": {"unpenalized_objectives": raw[0]}}
    np.testing.assert_allclose(scale.score(_exact_drift_objectives(payload)), scale.score(raw[:1]))
    base = np.array([[-0.0025, 2500.0]])
    assert scale.score(base + [[0.0015, 0.0]])[0] == pytest.approx(0.5)
    assert scale.score(base + [[0.0, 1500.0]])[0] == pytest.approx(0.5)


def test_suite_drift_retains_ordered_scenario_objectives_for_repeated_metrics():
    specs = [ObjectiveSpec(metric="adg_strategy_eq", goal="max")] * 2
    row = {
        _GPU_SUITE_OBJECTIVES_KEY: [1e9, 1e9],
        _GPU_SUITE_UNPENALIZED_OBJECTIVES_KEY: [-0.01, -0.03],
    }
    np.testing.assert_array_equal(_proxy_drift_objectives([row], specs), [[-0.01, -0.03]])
    with pytest.raises(KeyError):
        _exact_drift_objectives({"F": [1e9]})


def test_changed_calibration_rejects_previous_checkpoint_signature():
    old = hashlib.sha256(
        json.dumps({"active": [], "scoring": [], "version": 3}, sort_keys=True).encode()
    ).hexdigest()
    assert _checkpoint_signature([], []) != old
    config = get_template_config()
    original = _gpu_nsga2_checkpoint_contract(config, population_size=8, n_params=18)
    config["optimize"]["pymoo"]["shared"]["mutation_prob_per_variable"] = 0.3
    assert _gpu_nsga2_checkpoint_contract(config, population_size=8, n_params=18) != original


def test_canonical_suite_scoring_exposes_raw_ordered_values_under_penalties():
    from types import SimpleNamespace
    from metrics_schema import build_scenario_metrics
    from optimize import Evaluator, SuiteEvaluator
    from suite_runner import ScenarioResult, SuiteScenario

    config = get_template_config()
    config["optimize"]["scoring"] = [
        {"metric": "adg_strategy_eq", "goal": "max", "scenario": "base"},
        {"metric": "adg_strategy_eq", "goal": "max", "scenario": "stress"},
        {"metric": "adg_strategy_eq", "goal": "min", "scenario": None, "aggregate": "max"},
    ]
    config["optimize"]["limits"] = [
        {"metric": "adg_strategy_eq", "penalize_if": "less_than", "value": 0.5},
    ]
    base = Evaluator(hlcvs_specs={}, btc_usd_specs={}, msss={}, config=config)
    suite = SuiteEvaluator(
        base, [SimpleNamespace(label=label) for label in ("base", "stress")], {"default": "mean"}
    )
    rows = []
    for label, value in (("base", 0.01), ("stress", 0.03)):
        per_exchange = {"exchange": {"adg_strategy_eq": value}}
        rows.append(
            ScenarioResult(
                scenario=SuiteScenario(label, None, None, None, None),
                per_exchange=per_exchange,
                metrics=build_scenario_metrics(per_exchange),
                elapsed_seconds=0.0,
                output_path=None,
            )
        )
    scored = suite.score_scenario_results(rows)
    assert scored["constraint_violation"] > 0.0
    assert all(value > 0.03 for value in scored["objectives"])
    assert scored["unpenalized_objectives"] == pytest.approx((-0.01, -0.03, 0.03))


@pytest.mark.parametrize(
    "individual_flag",
    [
        "--optimize.pymoo.shared.mutation_prob",
        "--optimize.pymoo.shared.mutation_prob_var",
        "--optimize_pymoo_shared_mutation_prob_var",
        "-psmpv",
    ],
)
def test_mutation_controls_survive_cli_overrides_and_normalization(individual_flag):
    import argparse
    from config_utils import add_config_arguments, update_config_with_args

    config = get_template_config()
    parser = argparse.ArgumentParser()
    add_config_arguments(parser, config)
    args = parser.parse_args(
        [
            individual_flag,
            "1.0",
            "--optimize.pymoo.shared.mutation_prob_per_variable",
            "0.2",
        ]
    )
    update_config_with_args(config, args, verbose=False)
    normalized = format_config(config, verbose=False)
    assert normalized["optimize"]["pymoo"]["shared"]["mutation_prob"] == 1.0
    assert normalized["optimize"]["pymoo"]["shared"]["mutation_prob_per_variable"] == 0.2
