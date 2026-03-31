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
    from pymoo.termination import get_termination
    from pymoo.util.ref_dirs import get_reference_directions
except ImportError:  # pragma: no cover
    NSGA2 = None
    NSGA3 = None
    get_reference_directions = None

from optimization.problem import (
    PassivbotProblem,
    PymooAsyncRecordingRunner,
    PymooEvaluatorAdapter,
)
from optimization.repair import BoundsRepair

DEFAULT_PYMOO_ALGORITHM = "auto"
DEFAULT_PYMOO_SHARED = {
    "crossover_eta": 20.0,
    "crossover_prob_var": 0.5,
    "mutation_eta": 20.0,
    "mutation_prob_var": "auto",
    "eliminate_duplicates": True,
}
DEFAULT_PYMOO_REF_DIRS = {
    "method": "das_dennis",
    "n_partitions": "auto",
}
DEFAULT_AUTO_REF_DIR_TARGET = 330
SUPPORTED_PYMOO_ALGORITHMS = {"auto", "nsga2", "nsga3"}
SUPPORTED_REF_DIR_METHODS = {"das_dennis"}


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


def _extend_sampling_to_size(sampling: np.ndarray, bounds, target_size: int) -> np.ndarray:
    current_size = int(len(sampling))
    if current_size >= target_size:
        return sampling
    extra = [
        [bound.random_on_grid() for bound in bounds]
        for _ in range(int(target_size) - current_size)
    ]
    if not extra:
        return sampling
    return np.vstack([sampling, np.asarray(extra, dtype=np.float64)])


def _resolve_requested_population_size(config: dict[str, Any]) -> int | None:
    raw = config["optimize"].get("population_size")
    if raw is None:
        return None
    return max(1, int(raw))


def _resolve_pymoo_algorithm_name(config: dict[str, Any], *, n_obj: int | None = None) -> str:
    pymoo_cfg = config["optimize"].get("pymoo", {})
    raw = pymoo_cfg.get("algorithm", DEFAULT_PYMOO_ALGORITHM)
    algorithm = str(raw).strip().lower()
    if algorithm not in SUPPORTED_PYMOO_ALGORITHMS:
        allowed = ", ".join(sorted(SUPPORTED_PYMOO_ALGORITHMS))
        raise ValueError(f"unsupported optimize.pymoo.algorithm {raw!r}; expected one of {{{allowed}}}")
    if algorithm == "auto":
        if n_obj is None:
            return "auto"
        return "nsga2" if int(n_obj) <= 3 else "nsga3"
    return algorithm


def _resolve_pymoo_shared(config: dict[str, Any]) -> dict[str, Any]:
    optimize_cfg = config["optimize"]
    pymoo_cfg = optimize_cfg.get("pymoo", {})
    shared = pymoo_cfg.get("shared", {}) if isinstance(pymoo_cfg, dict) else {}
    if not isinstance(shared, dict):
        shared = {}

    def _fallback(name: str, legacy_name: str | None = None):
        legacy_name = legacy_name or name
        if name in shared and shared[name] is not None:
            return shared[name]
        if legacy_name in optimize_cfg and optimize_cfg[legacy_name] is not None:
            return optimize_cfg[legacy_name]
        return DEFAULT_PYMOO_SHARED[name]

    mutation_prob = shared.get("mutation_prob_var")
    if mutation_prob is None:
        legacy_mutation = optimize_cfg.get("mutation_indpb")
        if isinstance(legacy_mutation, (int, float)) and float(legacy_mutation) > 0.0:
            mutation_prob = float(legacy_mutation)
        else:
            mutation_prob = DEFAULT_PYMOO_SHARED["mutation_prob_var"]

    return {
        "crossover_eta": float(_fallback("crossover_eta")),
        "crossover_prob_var": float(_fallback("crossover_prob_var", "crossover_probability")),
        "mutation_eta": float(_fallback("mutation_eta")),
        "mutation_prob_var": mutation_prob,
        "eliminate_duplicates": bool(_fallback("eliminate_duplicates")),
    }


def _resolve_mutation_prob(shared: dict[str, Any], n_params: int) -> float:
    raw = shared.get("mutation_prob_var", DEFAULT_PYMOO_SHARED["mutation_prob_var"])
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        return 1.0 / max(1, int(n_params))
    return max(0.0, min(1.0, float(raw)))


def _reference_direction_count(n_obj: int, n_partitions: int) -> int:
    return math.comb(int(n_obj) + int(n_partitions) - 1, int(n_partitions))


def _resolve_auto_n_partitions(
    *,
    n_obj: int,
    population_size: int | None,
    target_ref_dirs: int = DEFAULT_AUTO_REF_DIR_TARGET,
    max_partitions: int = 32,
) -> int:
    if n_obj <= 1:
        return 1
    best = 1
    upper = max(1, int(max_partitions))
    for n_partitions in range(1, upper + 1):
        count = _reference_direction_count(n_obj, n_partitions)
        if population_size is None:
            if count > int(target_ref_dirs):
                break
        elif count > population_size:
            break
        best = n_partitions
    return best


def _resolve_nsga3_ref_dirs(
    config: dict[str, Any],
    *,
    n_obj: int,
    population_size: int | None,
) -> tuple[np.ndarray, int, str]:
    if get_reference_directions is None:  # pragma: no cover
        raise ModuleNotFoundError("pymoo is required for the pymoo optimizer backend")
    pymoo_cfg = config["optimize"].get("pymoo", {})
    algorithms = pymoo_cfg.get("algorithms", {}) if isinstance(pymoo_cfg, dict) else {}
    nsga3_cfg = algorithms.get("nsga3", {}) if isinstance(algorithms, dict) else {}
    ref_dirs_cfg = nsga3_cfg.get("ref_dirs", {}) if isinstance(nsga3_cfg, dict) else {}
    if not isinstance(ref_dirs_cfg, dict):
        ref_dirs_cfg = {}

    method = str(ref_dirs_cfg.get("method", DEFAULT_PYMOO_REF_DIRS["method"])).strip().lower()
    method = method.replace("-", "_")
    if method not in SUPPORTED_REF_DIR_METHODS:
        allowed = ", ".join(sorted(SUPPORTED_REF_DIR_METHODS))
        raise ValueError(
            f"unsupported optimize.pymoo.algorithms.nsga3.ref_dirs.method {method!r}; "
            f"expected one of {{{allowed}}}"
        )

    raw_n_partitions = ref_dirs_cfg.get("n_partitions", DEFAULT_PYMOO_REF_DIRS["n_partitions"])
    if isinstance(raw_n_partitions, str) and raw_n_partitions.strip().lower() == "auto":
        n_partitions = _resolve_auto_n_partitions(
            n_obj=n_obj,
            population_size=population_size,
        )
        resolution = "auto"
    else:
        n_partitions = max(1, int(raw_n_partitions))
        resolution = "explicit"

    ref_dirs = get_reference_directions(method.replace("_", "-"), n_obj, n_partitions=n_partitions)
    return ref_dirs, n_partitions, resolution


def _resolve_pymoo_population_plan(
    config: dict[str, Any],
    *,
    n_obj: int,
) -> dict[str, Any]:
    algorithm_name = _resolve_pymoo_algorithm_name(config, n_obj=n_obj)
    requested_population_size = _resolve_requested_population_size(config)

    if algorithm_name == "nsga3" and n_obj < 2:
        logging.warning(
            "optimize.pymoo.algorithm=nsga3 requested with %d objective; falling back to nsga2",
            n_obj,
        )
        algorithm_name = "nsga2"

    if algorithm_name == "nsga3":
        ref_dirs, n_partitions, resolution = _resolve_nsga3_ref_dirs(
            config,
            n_obj=n_obj,
            population_size=requested_population_size,
        )
        actual_population_size = (
            int(len(ref_dirs))
            if requested_population_size is None
            else max(requested_population_size, int(len(ref_dirs)))
        )
        return {
            "algorithm_name": algorithm_name,
            "requested_population_size": requested_population_size,
            "actual_population_size": actual_population_size,
            "ref_dirs": ref_dirs,
            "n_partitions": n_partitions,
            "resolution": resolution,
        }

    if requested_population_size is None:
        raise ValueError("optimize.population_size must be set when optimize.pymoo.algorithm=nsga2")

    return {
        "algorithm_name": algorithm_name,
        "requested_population_size": requested_population_size,
        "actual_population_size": requested_population_size,
        "ref_dirs": None,
        "n_partitions": None,
        "resolution": None,
    }


def _build_algorithm(
    *,
    config: dict[str, Any],
    sampling: np.ndarray,
    bounds,
    sig_digits: int | None,
    population_plan: dict[str, Any],
):
    if NSGA2 is None:  # pragma: no cover
        raise ModuleNotFoundError("pymoo is required for the pymoo optimizer backend")

    shared = _resolve_pymoo_shared(config)
    repair = BoundsRepair(bounds, sig_digits)
    crossover = SBX(
        prob_var=float(shared["crossover_prob_var"]),
        eta=float(shared["crossover_eta"]),
    )
    mutation = PM(
        prob=_resolve_mutation_prob(shared, len(bounds)),
        eta=float(shared["mutation_eta"]),
    )
    eliminate_duplicates = bool(shared["eliminate_duplicates"])

    algorithm_name = population_plan["algorithm_name"]
    requested_population_size = population_plan["requested_population_size"]

    if algorithm_name == "nsga3":
        if NSGA3 is None:  # pragma: no cover
            raise ModuleNotFoundError("pymoo NSGA3 is required for the pymoo optimizer backend")
        ref_dirs = population_plan["ref_dirs"]
        n_partitions = population_plan["n_partitions"]
        resolution = population_plan["resolution"]
        actual_population_size = population_plan["actual_population_size"]
        if requested_population_size is None:
            logging.info(
                "Using pymoo nsga3 auto population size=%d from %d reference directions",
                actual_population_size,
                len(ref_dirs),
            )
        elif actual_population_size != requested_population_size:
            logging.info(
                "Adjusted pymoo nsga3 population size from %d to %d to cover %d reference directions",
                requested_population_size,
                actual_population_size,
                len(ref_dirs),
            )
        if len(sampling) < actual_population_size:
            sampling = _extend_sampling_to_size(sampling, bounds, actual_population_size)
        logging.info(
            "Using pymoo nsga3 | n_obj=%d | ref_dirs=%d | n_partitions=%d (%s)",
            ref_dirs.shape[1],
            len(ref_dirs),
            n_partitions,
            resolution,
        )
        algorithm = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=None if requested_population_size is None else actual_population_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            repair=repair,
            eliminate_duplicates=eliminate_duplicates,
        )
        return algorithm

    logging.info("Using pymoo nsga2")
    return NSGA2(
        pop_size=population_plan["actual_population_size"],
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        repair=repair,
        eliminate_duplicates=eliminate_duplicates,
    )


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
        population_plan = _resolve_pymoo_population_plan(
            config,
            n_obj=len(config["optimize"]["scoring"]),
        )
        sampling = _build_initial_population(
            bounds=bounds,
            population_size=population_plan["actual_population_size"],
            get_starting_configs=get_starting_configs,
            configs_to_individuals=configs_to_individuals,
            starting_configs_path=starting_configs_path,
            sig_digits=sig_digits,
        )
        pool = multiprocessing.Pool(
            processes=config["optimize"]["n_cpus"],
            initializer=ignore_sigint_in_worker,
        )
        runner = PymooAsyncRecordingRunner(
            pool=pool,
            recorder=recorder,
            template=base_evaluator.config,
            build_config_fn=build_config_fn,
            overrides_fn=overrides_fn,
            overrides_list=overrides_list,
        )
        problem = PassivbotProblem(
            bounds=bounds,
            scoring_keys=config["optimize"]["scoring"],
            evaluator_adapter=PymooEvaluatorAdapter(
                evaluator_for_pool,
                overrides_list=overrides_list,
            ),
            elementwise_runner=runner,
        )
        algorithm = _build_algorithm(
            config=config,
            sampling=sampling,
            bounds=bounds,
            sig_digits=sig_digits,
            population_plan=population_plan,
        )
        population_size = population_plan["actual_population_size"]
        ngen = max(1, int(config["optimize"]["iters"] / population_size))
        logging.info("Starting optimize...")
        pymoo_minimize(
            problem,
            algorithm,
            get_termination("n_gen", ngen),
            seed=1,
            verbose=False,
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
