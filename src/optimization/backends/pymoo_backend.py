from __future__ import annotations

import logging
import math
import multiprocessing
from typing import Any

import numpy as np

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.parallelization.starmap import StarmapParallelization
    from pymoo.termination import get_termination
    from pymoo.util.ref_dirs import get_reference_directions
except ImportError:  # pragma: no cover
    NSGA2 = None

from optimization.callback import PymooRecorderCallback
from optimization.problem import PassivbotProblem, PymooEvaluatorAdapter
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


def _resolve_algorithm_name(pymoo_cfg: dict[str, Any], n_objectives: int) -> str:
    raw = str(pymoo_cfg["algorithm"]).strip().lower()
    if raw == "auto":
        return "nsga2" if n_objectives <= 3 else "nsga3"
    return raw


def _compute_n_partitions(n_obj: int, pop_size: int) -> int:
    if n_obj < 2 or pop_size < 1:
        raise ValueError(
            f"n_obj must be >= 2 and pop_size >= 1, got n_obj={n_obj}, pop_size={pop_size}"
        )
    p = 1
    while math.comb(p + n_obj - 1, n_obj - 1) < pop_size:
        p += 1
    if p > 1:
        below = math.comb(p - 1 + n_obj - 1, n_obj - 1)
        above = math.comb(p + n_obj - 1, n_obj - 1)
        return p - 1 if (pop_size - below) <= (above - pop_size) else p
    return p


def _build_nsga3_ref_dirs(pymoo_cfg: dict[str, Any], n_objectives: int, pop_size: int):
    if n_objectives < 2:
        raise ValueError("pymoo nsga3 requires at least 2 objectives")
    ref_cfg = pymoo_cfg["algorithms"]["nsga3"]["ref_dirs"]
    method = str(ref_cfg["method"]).strip().lower().replace("_", "-")
    n_partitions = ref_cfg["n_partitions"]
    if n_partitions == "auto":
        n_partitions = _compute_n_partitions(n_objectives, pop_size)
    return get_reference_directions(method, n_objectives, n_partitions=int(n_partitions))


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
        n_objectives = len(config["optimize"]["scoring"])
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
        pymoo_cfg = config["optimize"]["pymoo"]
        pymoo_shared = pymoo_cfg["shared"]
        algorithm_name = _resolve_algorithm_name(pymoo_cfg, n_objectives)
        common_kwargs = {
            "sampling": sampling,
            "crossover": SBX(
                prob_var=float(pymoo_shared["crossover_prob_var"]),
                eta=float(pymoo_shared["crossover_eta"]),
            ),
            "mutation": PM(
                prob=_resolve_mutation_prob(pymoo_shared, len(bounds)),
                eta=float(pymoo_shared["mutation_eta"]),
            ),
            "repair": BoundsRepair(bounds, sig_digits),
            "eliminate_duplicates": bool(pymoo_shared["eliminate_duplicates"]),
        }
        population_size = max(1, int(config["optimize"]["population_size"]))
        if algorithm_name == "nsga2":
            algorithm = NSGA2(
                pop_size=population_size,
                **common_kwargs,
            )
        elif algorithm_name == "nsga3":
            algorithm = NSGA3(
                ref_dirs=_build_nsga3_ref_dirs(
                    pymoo_cfg,
                    n_objectives,
                    population_size,
                ),
                pop_size=population_size,
                **common_kwargs,
            )
        else:
            raise NotImplementedError(
                f"unsupported pymoo algorithm {algorithm_name!r}; expected nsga2 or nsga3"
            )
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


def _resolve_mutation_prob(pymoo_shared: dict[str, Any], n_params: int) -> float:
    del n_params
    return float(pymoo_shared["mutation_prob_var"])
