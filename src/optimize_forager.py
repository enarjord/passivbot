import os
import sys
import passivbot_rust as pbr
import asyncio
import argparse
import multiprocessing
from multiprocessing import shared_memory
from backtest_forager import (
    prepare_hlcs_mss,
    prep_backtest_args,
    convert_to_v7,
    add_argparse_args_to_config,
    calc_preferred_coins,
)
from pure_funcs import (
    get_template_live_config,
    symbol_to_coin,
    ts_to_date_utc,
    denumpyize,
    sort_dict_keys,
    calc_hash,
)
from procedures import make_get_filepath, utc_ms, load_hjson_config, load_config, format_config
from copy import deepcopy
import numpy as np
from uuid import uuid4
import signal
import logging
import traceback
import json
import pprint
from deap import base, creator, tools, algorithms


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


def signal_handler(signal, frame):
    print("\nOptimization interrupted by user. Exiting gracefully...")
    sys.exit(0)


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


class Evaluator:
    def __init__(self, hlcs, preferred_coins, config, mss):
        self.hlcs = hlcs
        self.shared_hlcs = shared_memory.SharedMemory(create=True, size=self.hlcs.nbytes)
        self.shared_hlcs_np = np.ndarray(
            self.hlcs.shape, dtype=self.hlcs.dtype, buffer=self.shared_hlcs.buf
        )
        np.copyto(self.shared_hlcs_np, self.hlcs)
        del self.hlcs

        self.preferred_coins = preferred_coins
        self.shared_preferred_coins = shared_memory.SharedMemory(
            create=True, size=self.preferred_coins.nbytes
        )
        self.shared_preferred_coins_np = np.ndarray(
            self.preferred_coins.shape,
            dtype=self.preferred_coins.dtype,
            buffer=self.shared_preferred_coins.buf,
        )
        np.copyto(self.shared_preferred_coins_np, self.preferred_coins)
        del self.preferred_coins
        self.config = config

        _, self.exchange_params, self.backtest_params = prep_backtest_args(config, mss)

    def evaluate(self, individual):
        config = individual_to_config(individual, template=self.config)
        bot_params, _, _ = prep_backtest_args(
            config, [], exchange_params=self.exchange_params, backtest_params=self.backtest_params
        )
        fills, equities, analysis = pbr.run_backtest(
            self.shared_hlcs_np,
            self.shared_preferred_coins_np,
            bot_params,
            self.exchange_params,
            self.backtest_params,
        )
        w_0, w_1 = self.calc_fitness(analysis)
        analysis.update({"w_0": w_0, "w_1": w_1})
        with open(self.config["results_filename"], "a") as f:
            f.write(json.dumps(denumpyize({"analysis": analysis, "config": config})) + "\n")
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

    def cleanup(self):
        # Close and unlink the shared memory
        self.shared_hlcs.close()
        self.shared_hlcs.unlink()
        self.shared_preferred_coins.close()
        self.shared_preferred_coins.unlink()


def add_argparse_args_optimize_forager(parser):
    parser_items = [
        ("s", "symbols", "symbols", str, ", comma separated (SYM1USDT,SYM2USDT,...)"),
        ("e", "exchange", "exchange", str, ""),
        ("sd", "start_date", "start_date", str, ""),
        (
            "ed",
            "end_date",
            "end_date",
            str,
            ", if end date is 'now', will use current date as end date",
        ),
        ("sb", "starting_balance", "starting_balance", float, ""),
        ("bd", "base_dir", "base_dir", str, ""),
        ("c", "n_cpus", "n_cpus", int, ""),
        ("i", "iters", "iters", int, ""),
        ("p", "population_size", "population_size", int, ""),
    ]
    template = get_template_live_config("v7")
    shortened_already_added = set([x[0] for x in parser_items])
    for key in list(template["optimize"]["bounds"]) + list(template["optimize"]["limits"]):
        shortened = "".join([x[0] for x in key.split("_")])
        if shortened in shortened_already_added:
            for i in range(100):
                shortened = "".join([x[0] for x in key.split("_")]) + str(i)
                if shortened not in shortened_already_added:
                    break
            else:
                raise Exception(f"too many duplicates of shortened key {key}")
        parser_items.append((shortened, key, key, float, ", fixing optimizing bounds"))
        shortened_already_added.add(shortened)
    for k0, k1, d, t, h in parser_items:
        parser.add_argument(
            *[f"-{k0}", f"--{k1}"] + ([f"--{k1.replace('_', '-')}"] if "_" in k1 else []),
            type=t,
            required=False,
            dest=d,
            default=None,
            help=f"specify {k1}{h}, overriding value from config.",
        )
    parser.add_argument(
        "-t",
        "--start",
        type=str,
        required=False,
        dest="starting_configs",
        default=None,
        help="start with given live configs.  single json file or dir with multiple json files",
    )
    args = parser.parse_args()
    return args


def get_starting_configs(starting_configs: str):
    if starting_configs is None:
        return []
    cfgs = []
    if os.path.isdir(starting_configs):
        filenames = [os.path.join(starting_configs, f) for f in os.listdir(starting_configs)]
    else:
        filenames = [starting_configs]
    for path in filenames:
        try:
            cfgs.append(load_hjson_config(path))
        except Exception as e:
            logging.error(f"failed to load live config {path} {e}")
    return cfgs


def configs_to_individuals(cfgs):
    inds = {}
    for cfg in cfgs:
        try:
            individual = config_to_individual(format_config(cfg))
            inds[calc_hash(individual)] = individual
        except Exception as e:
            logging.error(f"error with config_to_individual {e}")
    return list(inds.values())


async def main():
    parser = argparse.ArgumentParser(prog="optimize_forager", description="run forager optimizer")
    parser.add_argument(
        "config_path", type=str, default=None, nargs="?", help="path to json passivbot config"
    )
    args = add_argparse_args_optimize_forager(parser)
    signal.signal(signal.SIGINT, signal_handler)
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
    config = add_argparse_args_to_config(config, args)
    hlcs, mss, results_path = await prepare_hlcs_mss(config)
    preferred_coins = calc_preferred_coins(hlcs, config)
    date_fname = ts_to_date_utc(utc_ms())[:19].replace(":", "_")
    coins = [symbol_to_coin(s) for s in config["backtest"]["symbols"]]
    coins_fname = "_".join(coins) if len(coins) <= 6 else f"{len(coins)}_coins"
    hash_snippet = uuid4().hex[:8]
    config["results_filename"] = make_get_filepath(
        f"opt_results_forager/{date_fname}_{coins_fname}_{hash_snippet}_all_results.txt"
    )
    try:
        evaluator = Evaluator(hlcs, preferred_coins, config, mss)
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))  # Minimize both objectives
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()

        # Define parameter bounds
        param_bounds = sort_dict_keys(config["optimize"]["bounds"])
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
        pool = multiprocessing.Pool(processes=config["optimize"]["n_cpus"])
        toolbox.register("map", pool.map)

        # Create initial population

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
        ########
    except Exception as e:
        traceback.print_exc()
    finally:
        # Close the pool
        logging.info(f"attempting clean shutdown...")
        evaluator.cleanup()
        sys.exit(0)
        # pool.close()
        # pool.join()


if __name__ == "__main__":
    asyncio.run(main())
