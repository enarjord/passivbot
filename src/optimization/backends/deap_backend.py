from __future__ import annotations

import logging
import multiprocessing
import time
from copy import deepcopy
from typing import Any, Callable

import numpy as np

try:
    from deap import base, creator, tools
except ImportError:  # pragma: no cover - allow import in minimal test envs
    base = creator = tools = None

from optimization.bounds import enforce_bounds
from optimization.deap_adapters import (
    cxSimulatedBinaryBoundedWrapper,
    mutPolynomialBoundedWrapper,
)

DEFAULT_DEAP_POPULATION_SIZE = 500


def _resolve_deap_population_size(config: dict[str, Any]) -> int:
    raw = config["optimize"].get("population_size")
    if raw is None:
        logging.info(
            "optimize.population_size=null is not supported by deap; using %d",
            DEFAULT_DEAP_POPULATION_SIZE,
        )
        return DEFAULT_DEAP_POPULATION_SIZE
    return max(1, int(raw))


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
    ignore_sigint_in_worker: Callable[[], None],
    get_starting_configs: Callable[[str | None], list],
    configs_to_individuals: Callable[[list, Any, int], list],
    record_individual_result: Callable[[Any, dict, list, Any], None],
    run_evolution: Callable[..., tuple[Any, Any]],
    build_config_fn,
    overrides_fn,
) -> dict[str, Any]:
    del build_config_fn
    del overrides_fn
    if base is None or creator is None or tools is None:  # pragma: no cover
        raise ModuleNotFoundError("deap is required for the deap optimizer backend")

    pool = None
    pool_state = {"terminated": False}
    try:
        n_objectives = len(config["optimize"]["scoring"])
        if not hasattr(creator, "FitnessMulti"):
            creator.create(
                "FitnessMulti",
                constraint_fitness_cls,
                weights=(-1.0,) * n_objectives,
            )
        else:
            creator.FitnessMulti.weights = (-1.0,) * n_objectives
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        bounds = evaluator.bounds
        sig_digits = config["optimize"]["round_to_n_significant_digits"]
        crossover_eta = config["optimize"].get("crossover_eta", 20.0)
        mutation_eta = config["optimize"].get("mutation_eta", 20.0)
        mutation_indpb_raw = config["optimize"].get("mutation_indpb", 0.0)
        if isinstance(mutation_indpb_raw, (int, float)) and mutation_indpb_raw > 0.0:
            mutation_indpb = max(0.0, min(1.0, float(mutation_indpb_raw)))
        else:
            mutation_indpb = 1.0 / len(bounds) if bounds else 1.0
        offspring_multiplier = config["optimize"].get("offspring_multiplier", 1.0)
        if not isinstance(offspring_multiplier, (int, float)) or offspring_multiplier <= 0.0:
            offspring_multiplier = 1.0

        def _make_random_attr(bound):
            return bound.random_on_grid()

        for i, bound in enumerate(bounds):
            toolbox.register(f"attr_{i}", _make_random_attr, bound)

        toolbox.register(
            "mate",
            cxSimulatedBinaryBoundedWrapper,
            eta=crossover_eta,
            bounds=bounds,
        )
        toolbox.register(
            "mutate",
            mutPolynomialBoundedWrapper,
            eta=mutation_eta,
            indpb=mutation_indpb,
            bounds=bounds,
        )
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("evaluate", evaluator_for_pool.evaluate, overrides_list=overrides_list)

        logging.info("Initializing multiprocessing pool. N cpus: %s", config["optimize"]["n_cpus"])
        pool = multiprocessing.Pool(
            processes=config["optimize"]["n_cpus"],
            initializer=ignore_sigint_in_worker,
        )
        toolbox.register("map", pool.map)
        logging.info("Finished initializing multiprocessing pool.")
        logging.info("Creating initial population...")

        def _evaluate_initial(individuals):
            if not individuals:
                return 0
            total = len(individuals)
            pending = {}
            for ind in individuals:
                pending[pool.apply_async(toolbox.evaluate, (ind,))] = ind
            completed = 0
            try:
                while pending:
                    ready = [res for res in pending if res.ready()]
                    if not ready:
                        time.sleep(0.05)
                        continue
                    for res in ready:
                        ind = pending.pop(res)
                        fit_values, penalty, metrics = res.get()
                        ind.fitness.values = fit_values
                        ind.fitness.constraint_violation = penalty
                        ind.constraint_violation = penalty
                        if metrics is not None:
                            ind.evaluation_metrics = metrics
                            record_individual_result(
                                ind,
                                evaluator.config,
                                overrides_list,
                                recorder,
                            )
                        elif hasattr(ind, "evaluation_metrics"):
                            delattr(ind, "evaluation_metrics")
                        completed += 1
                        logging.info("Evaluated %d/%d starting configs", completed, total)
            except KeyboardInterrupt:
                logging.info("Evaluation interrupted; terminating pending starting configs...")
                for res in pending:
                    try:
                        res.cancel()
                    except Exception:
                        pass
                if not pool_state["terminated"]:
                    logging.info("Terminating worker pool immediately due to interrupt...")
                    pool.terminate()
                    pool_state["terminated"] = True
                raise
            return completed

        population_size = _resolve_deap_population_size(config)
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

        def _make_random_individual():
            values = [bound.random_on_grid() for bound in bounds]
            return creator.Individual(values)

        population = [_make_random_individual() for _ in range(population_size)]
        if starting_individuals:
            evaluated_seeds = [creator.Individual(ind) for ind in starting_individuals]
            eval_count = _evaluate_initial(evaluated_seeds)
            logging.info("Evaluated %d starting configs", eval_count)
            if len(evaluated_seeds) > population_size:
                evaluated_seeds = tools.selNSGA2(evaluated_seeds, population_size)
                logging.info(
                    "Trimmed starting configs to population size via NSGA-II crowding (kept %d)",
                    len(evaluated_seeds),
                )
            for i, ind in enumerate(evaluated_seeds):
                population[i] = creator.Individual(ind)

            remaining = population_size - len(evaluated_seeds)
            seed_pool = evaluated_seeds if evaluated_seeds else []
            if seed_pool and remaining > 0:
                for i in range(len(evaluated_seeds), len(evaluated_seeds) + remaining // 2):
                    population[i] = deepcopy(seed_pool[np.random.choice(range(len(seed_pool)))])
        for i in range(len(population)):
            population[i][:] = enforce_bounds(population[i], bounds, sig_digits)

        logging.info("Initial population size: %d", len(population))

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "min", "max"
        hof = tools.ParetoFront()

        logging.info("Starting optimize...")
        lambda_size = max(1, int(round(population_size * offspring_multiplier)))
        population, logbook = run_evolution(
            population,
            toolbox,
            mu=population_size,
            lambda_=lambda_size,
            cxpb=config["optimize"]["crossover_probability"],
            mutpb=config["optimize"]["mutation_probability"],
            ngen=max(1, int(config["optimize"]["iters"] / len(population))),
            stats=stats,
            halloffame=hof,
            verbose=False,
            recorder=recorder,
            evaluator_config=evaluator.config,
            overrides_list=overrides_list,
            pool=pool,
            duplicate_counter=duplicate_counter,
            pool_state=pool_state,
        )
        logging.info("Optimization complete.")
        return {
            "population": population,
            "logbook": logbook,
            "halloffame": hof,
            "pool": pool,
            "pool_terminated": pool_state["terminated"],
        }
    except Exception:
        if pool is not None and not pool_state["terminated"]:
            pool.terminate()
            pool_state["terminated"] = True
            pool.join()
        raise
