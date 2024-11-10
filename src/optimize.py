import os
import shutil
import sys
import passivbot_rust as pbr
import asyncio
import argparse
import multiprocessing
import subprocess
import mmap
from multiprocessing import Queue, Process
from backtest import (
    prepare_hlcvs_mss,
    prep_backtest_args,
)
from pure_funcs import (
    get_template_live_config,
    symbol_to_coin,
    ts_to_date_utc,
    denumpyize,
    sort_dict_keys,
    calc_hash,
)
from procedures import (
    make_get_filepath,
    utc_ms,
    load_hjson_config,
    load_config,
    format_config,
    add_arguments_recursively,
    update_config_with_args,
)
from copy import deepcopy
from main import manage_rust_compilation
import numpy as np
from uuid import uuid4
import logging
import traceback
import json
import pprint
from deap import base, creator, tools, algorithms
from contextlib import contextmanager
import tempfile
import time
import fcntl
from tqdm import tqdm
import dictdiffer  # Added import for dictdiffer


def make_json_serializable(obj):
    """
    Recursively convert tuples in the object to lists to make it JSON serializable.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return [make_json_serializable(e) for e in obj]
    elif isinstance(obj, list):
        return [make_json_serializable(e) for e in obj]
    else:
        return obj


def results_writer_process(queue: Queue, results_filename: str):
    """
    Manager process that handles writing results to file.
    Runs in a separate process and receives results through a queue.
    Applies diffing to the entire data dictionary.
    """
    prev_data = None  # Initialize previous data as None
    try:
        while True:
            data = queue.get()
            if data == "DONE":  # Sentinel value to signal shutdown
                break
            try:
                if prev_data is None:
                    # First data entry, write full data
                    output_data = data
                else:
                    # Compute diff of the entire data dictionary
                    diff = list(dictdiffer.diff(prev_data, data))
                    for i in range(len(diff)):
                        if diff[i][0] == "change":
                            diff[i] = [diff[i][1], diff[i][2][1]]
                    output_data = {"diff": make_json_serializable(diff)}

                prev_data = data

                # Write to disk
                with open(results_filename, "a") as f:
                    json.dump(denumpyize(output_data), f)
                    f.write("\n")
            except Exception as e:
                logging.error(f"Error writing results: {e}")
    except Exception as e:
        logging.error(f"Results writer process error: {e}")


def create_shared_memory_file(hlcvs):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    logging.info(f"Creating shared memory file: {temp_file.name}...")
    shared_memory_file = temp_file.name

    try:
        total_size = hlcvs.nbytes
        chunk_size = 1024 * 1024  # 1 MB chunks
        hlcvs_bytes = hlcvs.tobytes()

        with open(shared_memory_file, "wb") as f:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Writing to shared memory"
            ) as pbar:
                for i in range(0, len(hlcvs_bytes), chunk_size):
                    chunk = hlcvs_bytes[i : i + chunk_size]
                    f.write(chunk)
                    pbar.update(len(chunk))

    except IOError as e:
        logging.error(f"Error writing to shared memory file: {e}")
        raise
    logging.info(f"Done creating shared memory file")
    return shared_memory_file


def check_disk_space(path, required_space):
    total, used, free = shutil.disk_usage(path)
    logging.info(
        f"Disk space - Total: {total/(1024**3):.2f} GB, Used: {used/(1024**3):.2f} GB, Free: {free/(1024**3):.2f} GB"
    )
    if free < required_space:
        raise IOError(
            f"Not enough disk space. Required: {required_space/(1024**3):.2f} GB, Available: {free/(1024**3):.2f} GB"
        )


def mutPolynomialBoundedWrapper(individual, eta, low, up, indpb):
    """
    A wrapper around DEAP's mutPolynomialBounded function to pre-process
    bounds and handle the case where lower and upper bounds may be equal.

    Args:
        individual: Sequence individual to be mutated.
        eta: Crowding degree of the mutation.
        low: A value or sequence of values that is the lower bound of the search space.
        up: A value or sequence of values that is the upper bound of the search space.
        indpb: Independent probability for each attribute to be mutated.

    Returns:
        A tuple of one individual, mutated with consideration for equal lower and upper bounds.
    """
    # Convert low and up to numpy arrays for easier manipulation
    low_array = np.array(low)
    up_array = np.array(up)

    # Identify dimensions where lower and upper bounds are equal
    equal_bounds_mask = low_array == up_array

    # Temporarily adjust bounds for those dimensions
    # This adjustment is arbitrary and won't affect the outcome since the mutation
    # won't be effective in these dimensions
    temp_low = np.where(equal_bounds_mask, low_array - 1e-6, low_array)
    temp_up = np.where(equal_bounds_mask, up_array + 1e-6, up_array)

    # Call the original mutPolynomialBounded function with the temporarily adjusted bounds
    tools.mutPolynomialBounded(individual, eta, list(temp_low), list(temp_up), indpb)

    # Reset values in dimensions with originally equal bounds to ensure they remain unchanged
    for i, equal in enumerate(equal_bounds_mask):
        if equal:
            individual[i] = low[i]

    return (individual,)


def cxSimulatedBinaryBoundedWrapper(ind1, ind2, eta, low, up):
    """
    A wrapper around DEAP's cxSimulatedBinaryBounded function to pre-process
    bounds and handle the case where lower and upper bounds are equal.

    Args:
        ind1: The first individual participating in the crossover.
        ind2: The second individual participating in the crossover.
        eta: Crowding degree of the crossover.
        low: A value or sequence of values that is the lower bound of the search space.
        up: A value or sequence of values that is the upper bound of the search space.

    Returns:
        A tuple of two individuals after crossover operation.
    """
    # Convert low and up to numpy arrays for easier manipulation
    low_array = np.array(low)
    up_array = np.array(up)

    # Identify dimensions where lower and upper bounds are equal
    equal_bounds_mask = low_array == up_array

    # Temporarily adjust bounds for those dimensions to prevent division by zero
    # This adjustment is arbitrary and won't affect the outcome since the crossover
    # won't modify these dimensions
    low_array[equal_bounds_mask] -= 1e-6
    up_array[equal_bounds_mask] += 1e-6

    # Call the original cxSimulatedBinaryBounded function with adjusted bounds
    tools.cxSimulatedBinaryBounded(ind1, ind2, eta, list(low_array), list(up_array))

    # Ensure that values in dimensions with originally equal bounds are reset
    # to the bound value (since they should not be modified)
    for i, equal in enumerate(equal_bounds_mask):
        if equal:
            ind1[i] = low[i]
            ind2[i] = low[i]

    return ind1, ind2


def individual_to_config(individual, template=None):
    if template is None:
        template = get_template_live_config("v7")
    config = deepcopy(template)
    keys = sorted(config["bot"]["long"])
    i = 0
    for pside in ["long", "short"]:
        for key in keys:
            config["bot"][pside][key] = individual[i]
            i += 1
    return config


def config_to_individual(config):
    individual = []
    for pside in ["long", "short"]:
        individual += [v for k, v in sorted(config["bot"][pside].items())]
    return individual


@contextmanager
def managed_mmap(filename, dtype, shape):
    mmap = None
    try:
        mmap = np.memmap(filename, dtype=dtype, mode="r", shape=shape)
        yield mmap
    except FileNotFoundError:
        if shutdown_event.is_set():
            yield None
        else:
            raise
    finally:
        if mmap is not None:
            del mmap


class Evaluator:
    def __init__(self, shared_memory_file, hlcvs_shape, hlcvs_dtype, config, mss, results_queue):
        logging.info("Initializing Evaluator...")
        self.shared_memory_file = shared_memory_file
        self.hlcvs_shape = hlcvs_shape
        self.hlcvs_dtype = hlcvs_dtype

        logging.info("Setting up managed_mmap...")
        self.mmap_context = managed_mmap(self.shared_memory_file, self.hlcvs_dtype, self.hlcvs_shape)
        logging.info("Entering mmap_context...")
        self.shared_hlcvs_np = self.mmap_context.__enter__()
        logging.info("mmap_context entered successfully.")

        self.config = config
        _, self.exchange_params, self.backtest_params = prep_backtest_args(config, mss)
        logging.info("Evaluator initialization complete.")
        self.results_queue = results_queue

    def evaluate(self, individual):
        config = individual_to_config(individual, template=self.config)
        bot_params, _, _ = prep_backtest_args(
            config, [], exchange_params=self.exchange_params, backtest_params=self.backtest_params
        )
        fills, equities, analysis = pbr.run_backtest(
            self.shared_memory_file,
            self.shared_hlcvs_np.shape,
            self.shared_hlcvs_np.dtype.str,
            bot_params,
            self.exchange_params,
            self.backtest_params,
        )
        w_0, w_1 = self.calc_fitness(analysis)
        analysis.update({"w_0": w_0, "w_1": w_1})
        self.results_queue.put({"analysis": analysis, "config": config})
        return w_0, w_1

    def calc_fitness(self, analysis):
        modifier = 0.0
        for i, key in [
            (4, "drawdown_worst"),
            (3, "equity_balance_diff_mean"),
            (2, "loss_profit_ratio"),
        ]:
            modifier += (
                max(self.config["optimize"]["limits"][f"lower_bound_{key}"], analysis[key])
                - self.config["optimize"]["limits"][f"lower_bound_{key}"]
            ) * 10**i
        if analysis["drawdown_worst"] >= 1.0 or analysis["equity_balance_diff_max"] < 0.1:
            w_0 = w_1 = modifier
        else:
            w_0 = modifier - analysis[self.config["optimize"]["scoring"][0]]
            w_1 = modifier - analysis[self.config["optimize"]["scoring"][1]]
        return w_0, w_1

    def __del__(self):
        if hasattr(self, "mmap_context"):
            self.mmap_context.__exit__(None, None, None)

    def __getstate__(self):
        # This method is called when pickling. We exclude mmap_context and shared_hlcvs_np
        state = self.__dict__.copy()
        del state["mmap_context"]
        del state["shared_hlcvs_np"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.mmap_context = managed_mmap(self.shared_memory_file, self.hlcvs_dtype, self.hlcvs_shape)
        self.shared_hlcvs_np = self.mmap_context.__enter__()
        if self.shared_hlcvs_np is None:
            print("Warning: Unable to recreate shared memory mapping during unpickling.")


def add_extra_options(parser):
    parser.add_argument(
        "-t",
        "--start",
        type=str,
        required=False,
        dest="starting_configs",
        default=None,
        help="Start with given live configs. Single json file or dir with multiple json files",
    )


def get_starting_configs(starting_configs: str):
    if starting_configs is None:
        return []
    cfgs = []
    if os.path.isdir(starting_configs):
        filenames = [
            os.path.join(starting_configs, f)
            for f in os.listdir(starting_configs)
            if f.endswith("json") or f.endswith("hjson")
        ]
    else:
        filenames = [starting_configs]
    for path in filenames:
        try:
            cfgs.append(load_config(path, verbose=False))
        except Exception as e:
            logging.error(f"failed to load live config {path} {e}")
    return cfgs


def configs_to_individuals(cfgs):
    inds = {}
    for cfg in cfgs:
        try:
            fcfg = format_config(cfg, verbose=False)
            individual = config_to_individual(fcfg)
            inds[calc_hash(individual)] = individual
            # add duplicate of config, but with lowered total wallet exposure limit
            fcfg2 = deepcopy(fcfg)
            for pside in ["long", "short"]:
                fcfg2["bot"][pside]["total_wallet_exposure_limit"] *= 0.75
            individual2 = config_to_individual(fcfg2)
            inds[calc_hash(individual2)] = individual2
        except Exception as e:
            logging.error(f"error loading starting config: {e}")
    return list(inds.values())


async def main():
    manage_rust_compilation()
    parser = argparse.ArgumentParser(prog="optimize", description="run optimizer")
    parser.add_argument(
        "config_path", type=str, default=None, nargs="?", help="path to json passivbot config"
    )
    template_config = get_template_live_config("v7")
    del template_config["bot"]
    keep_live_keys = {
        "approved_coins",
        "minimum_coin_age_days",
        "ohlcv_rolling_window",
        "relative_volume_filter_clip_pct",
    }
    for key in sorted(template_config["live"]):
        if key not in keep_live_keys:
            del template_config["live"][key]
    add_arguments_recursively(parser, template_config)
    add_extra_options(parser)
    args = parser.parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    if args.config_path is None:
        logging.info(f"loading default template config configs/template.json")
        config = load_config("configs/template.json")
    else:
        logging.info(f"loading config {args.config_path}")
        config = load_config(args.config_path)
    old_config = deepcopy(config)
    update_config_with_args(config, args)
    config = format_config(config)
    symbols, hlcvs, mss, results_path, cache_dir = await prepare_hlcvs_mss(config)
    config["backtest"]["symbols"] = symbols
    config["backtest"]["cache_dir"] = str(cache_dir)
    date_fname = ts_to_date_utc(utc_ms())[:19].replace(":", "_")
    coins = [symbol_to_coin(s) for s in config["backtest"]["symbols"]]
    coins_fname = "_".join(coins) if len(coins) <= 6 else f"{len(coins)}_coins"
    hash_snippet = uuid4().hex[:8]
    config["results_filename"] = make_get_filepath(
        f"optimize_results/{date_fname}_{coins_fname}_{hash_snippet}_all_results.txt"
    )

    try:
        required_space = hlcvs.nbytes * 1.1  # Add 10% buffer
        check_disk_space(tempfile.gettempdir(), required_space)
        logging.info(f"Starting to create shared memory file...")
        shared_memory_file = create_shared_memory_file(hlcvs)
        logging.info(f"Finished creating shared memory file: {shared_memory_file}")

        # Create results queue and start manager process
        manager = multiprocessing.Manager()
        results_queue = manager.Queue()
        writer_process = Process(
            target=results_writer_process, args=(results_queue, config["results_filename"])
        )
        writer_process.start()

        # Initialize evaluator with results queue
        evaluator = Evaluator(
            shared_memory_file, hlcvs.shape, hlcvs.dtype, config, mss, results_queue
        )

        logging.info(f"Finished initializing evaluator...")
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))  # Minimize both objectives
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()

        # Define parameter bounds
        param_bounds = sort_dict_keys(config["optimize"]["bounds"])

        # Register attribute generators
        for i, (param_name, (low, high)) in enumerate(param_bounds.items()):
            toolbox.register(f"attr_{i}", np.random.uniform, low, high)

        def create_individual():
            return creator.Individual(
                [getattr(toolbox, f"attr_{i}")() for i in range(len(param_bounds))]
            )

        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Register the evaluation function
        toolbox.register("evaluate", evaluator.evaluate)

        # Register genetic operators
        toolbox.register(
            "mate",
            cxSimulatedBinaryBoundedWrapper,
            eta=20.0,
            low=[low for low, high in param_bounds.values()],
            up=[high for low, high in param_bounds.values()],
        )
        toolbox.register(
            "mutate",
            mutPolynomialBoundedWrapper,
            eta=20.0,
            low=[low for low, high in param_bounds.values()],
            up=[high for low, high in param_bounds.values()],
            indpb=1.0 / len(param_bounds),
        )
        toolbox.register("select", tools.selNSGA2)

        # Parallelization setup
        logging.info(f"Initializing multiprocessing pool. N cpus: {config['optimize']['n_cpus']}")
        pool = multiprocessing.Pool(processes=config["optimize"]["n_cpus"])
        toolbox.register("map", pool.map)
        logging.info(f"Finished initializing multiprocessing pool.")

        # Create initial population
        logging.info(f"Creating initial population...")

        starting_individuals = configs_to_individuals(get_starting_configs(args.starting_configs))
        if len(starting_individuals) > config["optimize"]["population_size"]:
            logging.info(
                f"increasing population size: {config['optimize']['population_size']} -> {len(starting_individuals)}"
            )
            config["optimize"]["population_size"] = len(starting_individuals)

        population = toolbox.population(n=config["optimize"]["population_size"])
        if starting_individuals:
            bounds = [(low, high) for low, high in param_bounds.values()]
            for i in range(len(starting_individuals)):
                adjusted = [
                    max(min(x, bounds[z][1]), bounds[z][0])
                    for z, x in enumerate(starting_individuals[i])
                ]
                population[i] = creator.Individual(adjusted)

        logging.info(f"Initial population size: {len(population)}")

        # Set up statistics and hall of fame
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        hof = tools.ParetoFront()

        # Run the optimization
        logging.info(f"Starting optimize...")
        population, logbook = algorithms.eaMuPlusLambda(
            population,
            toolbox,
            mu=config["optimize"]["population_size"],
            lambda_=config["optimize"]["population_size"],
            cxpb=config["optimize"]["crossover_probability"],
            mutpb=config["optimize"]["mutation_probability"],
            ngen=max(1, int(config["optimize"]["iters"] / len(population))),
            stats=stats,
            halloffame=hof,
            verbose=True,
        )

        # Print statistics
        print(logbook)

        logging.info(f"Optimization complete.")

        try:
            logging.info(f"Extracting best config...")
            result = subprocess.run(
                ["python3", "src/tools/extract_best_config.py", config["results_filename"], "-v"],
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
        except Exception as e:
            logging.error(f"failed to extract best config {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        # Signal the writer process to shut down and wait for it
        if "results_queue" in locals():
            results_queue.put("DONE")
            writer_process.join()
        if "pool" in locals():
            logging.info("Closing and terminating the process pool...")
            pool.close()
            pool.terminate()
            pool.join()

        if shared_memory_file and os.path.exists(shared_memory_file):
            logging.info(f"Removing shared memory file: {shared_memory_file}")
            try:
                os.unlink(shared_memory_file)
            except Exception as e:
                logging.error(f"Error removing shared memory file: {e}")

        logging.info("Cleanup complete. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
