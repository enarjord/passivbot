from __future__ import annotations

import logging
import multiprocessing
from typing import Any

import numpy as np

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.parallelization.starmap import StarmapParallelization
    from pymoo.termination import get_termination
except ImportError:  # pragma: no cover
    NSGA2 = None

from optimization.problem import PassivbotProblem, PymooEvaluatorAdapter
from optimization.callback import PymooRecorderCallback
from optimization.repair import BoundsRepair


def _build_initial_population(
    *,
    bounds,
    population_size: int,
    get_starting_configs,
    configs_to_individuals,
    starting_configs_path: str | None,
    sig_digits: int | None,
) -> np.ndarray:
    starting_configs = get_starting_configs(starting_configs_path)
    if starting_configs:
        logging.info(
            "Loaded %d starting configs before quantization (population size=%d)",
            len(starting_configs),
            population_size,
        )
    else:
        logging.info("No starting configs provided; population will be random-initialized")
    starting_individuals = configs_to_individuals(
        starting_configs,
        bounds,
        sig_digits,
    )
    population = [[bound.random_on_grid() for bound in bounds] for _ in range(population_size)]
    if starting_individuals:
        trimmed = starting_individuals[:population_size]
        if len(starting_individuals) > population_size:
            logging.info(
                "Trimmed starting configs to population size for pymoo backend (kept %d)",
                len(trimmed),
            )
        for idx, individual in enumerate(trimmed):
            population[idx] = list(individual)
    return np.asarray(population, dtype=np.float64)


def run_backend(
    *,
    config: dict[str, Any],
    evaluator,
    evaluator_for_pool,
    recorder,
    overrides_list,
    duplicate_counter,
    starting_configs_path: str | None,
    constraint_fitness_cls,
    ignore_sigint_in_worker,
    get_starting_configs,
    configs_to_individuals,
    record_individual_result,
    run_evolution,
    build_config_fn,
    overrides_fn,
) -> dict[str, Any]:
    del evaluator
    del duplicate_counter
    del constraint_fitness_cls
    del record_individual_result
    del run_evolution
    if NSGA2 is None:  # pragma: no cover
        raise ModuleNotFoundError("pymoo is required for the pymoo optimizer backend")

    base_evaluator = getattr(evaluator_for_pool, "base", evaluator_for_pool)
    if hasattr(base_evaluator, "use_duplicate_guard"):
        base_evaluator.use_duplicate_guard = False
    pool = None
    try:
        bounds = base_evaluator.bounds
        sig_digits = config["optimize"]["round_to_n_significant_digits"]
        sampling = _build_initial_population(
            bounds=bounds,
            population_size=config["optimize"]["population_size"],
            get_starting_configs=get_starting_configs,
            configs_to_individuals=configs_to_individuals,
            starting_configs_path=starting_configs_path,
            sig_digits=sig_digits,
        )
        pool = multiprocessing.Pool(
            processes=config["optimize"]["n_cpus"],
            initializer=ignore_sigint_in_worker,
        )
        runner = StarmapParallelization(pool.starmap)
        problem = PassivbotProblem(
            bounds=bounds,
            scoring_keys=config["optimize"]["scoring"],
            evaluator_adapter=PymooEvaluatorAdapter(
                evaluator_for_pool,
                overrides_list=overrides_list,
            ),
            elementwise_runner=runner,
        )
        callback = PymooRecorderCallback(
            recorder=recorder,
            template=base_evaluator.config,
            build_config_fn=build_config_fn,
            overrides_fn=overrides_fn,
            overrides_list=overrides_list,
        )
        algorithm = NSGA2(
            pop_size=config["optimize"]["population_size"],
            sampling=sampling,
            crossover=SBX(
                prob_var=float(config["optimize"].get("crossover_probability", 0.7)),
                eta=float(config["optimize"].get("crossover_eta", 20.0)),
            ),
            mutation=PM(
                prob=_resolve_mutation_prob(config, len(bounds)),
                eta=float(config["optimize"].get("mutation_eta", 20.0)),
            ),
            repair=BoundsRepair(bounds, sig_digits),
            eliminate_duplicates=True,
        )
        population_size = max(1, int(config["optimize"]["population_size"]))
        ngen = max(1, int(config["optimize"]["iters"] / population_size))
        logging.info("Starting optimize...")
        pymoo_minimize(
            problem,
            algorithm,
            get_termination("n_gen", ngen),
            seed=1,
            verbose=False,
            callback=callback,
        )
        logging.info("Optimization complete.")
        return {
            "pool": pool,
            "pool_terminated": False,
        }
    except Exception:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise


def _resolve_mutation_prob(config: dict[str, Any], n_params: int) -> float:
    raw = config["optimize"].get("mutation_indpb", 0.0)
    if isinstance(raw, (int, float)) and raw > 0.0:
        return max(0.0, min(1.0, float(raw)))
    return 1.0 / max(1, int(n_params))
