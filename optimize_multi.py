import asyncio
import random
import multiprocessing
import pprint
import numpy as np
import pandas as pd
import json
import logging
import argparse
from deap import base, creator, tools, algorithms
from collections import OrderedDict
from procedures import utc_ms, make_get_filepath

from pure_funcs import (
    live_config_dict_to_list_recursive_grid,
    numpyize,
    calc_drawdowns,
    ts_to_date_utc,
    denumpyize,
    tuplify,
)
from backtest_multi import backtest_multi, prep_config_multi, prep_hlcs_mss_config
from njit_multisymbol import backtest_multisymbol_recursive_grid


class Evaluator:
    def __init__(self, hlcs, config):
        self.hlcs = hlcs
        self.results_cache_fname = config["results_cache_fname"]
        self.config = {
            key: config[key]
            for key in [
                "start_date",
                "end_date",
                "long_enabled",
                "short_enabled",
                "starting_balance",
                "maker_fee",
                "do_longs",
                "do_shorts",
                "c_mults",
                "symbols",
                "qty_steps",
                "price_steps",
                "min_costs",
                "min_qtys",
            ]
        }

    def evaluate(self, individual):
        # individual is a list of floats
        config_ = self.config.copy()
        live_configs = individual_to_live_configs(individual, config_["symbols"])
        for key in [
            "loss_allowance_pct",
            "stuck_threshold",
            "unstuck_close_pct",
        ]:
            config_[key] = live_configs[key]

        config_["live_configs"] = numpyize(
            [
                live_config_dict_to_list_recursive_grid(live_configs[symbol])
                for symbol in config_["symbols"]
            ]
        )
        res = backtest_multi(self.hlcs, config_)
        fills, stats = res
        stats_eqs = [(x[0], x[5]) for x in stats]
        fills_eqs = [(x[0], x[5]) for x in fills]
        all_eqs = pd.DataFrame(stats_eqs + fills_eqs).set_index(0).sort_index()[1]
        drawdowns = calc_drawdowns(all_eqs)
        worst_drawdown = abs(drawdowns.min())

        thr = config_["starting_balance"] * 1e-6
        stats_eqs_df = pd.DataFrame(stats_eqs).set_index(0)
        daily_eqs = stats_eqs_df.groupby(stats_eqs_df.index // 1440).last()[1]
        drawdowns_daily = calc_drawdowns(daily_eqs)
        drawdowns_daily_mean = abs(daily_drawdowns.mean())
        daily_eqs_pct_change = daily_eqs.pct_change()
        if daily_eqs.iloc[-1] <= thr:
            # ensure adg is negative if final equity is low
            adg = (max(thr, daily_eqs.iloc[-1]) / daily_eqs.iloc[0]) ** (1 / len(daily_eqs)) - 1
        else:
            adg = daily_eqs_pct_change.mean()
        sharpe_ratio = adg / daily_eqs_pct_change.std()
        # daily_min_drawdowns = drawdowns.groupby(drawdowns.index // 1440).min()
        # mean_of_10_worst_drawdowns_daily = abs(daily_min_drawdowns.sort_values().iloc[:10].mean())

        to_dump = {
            key: self.config[key]
            for key in [
                "symbols",
                "start_date",
                "end_date",
                "starting_balance",
                "long_enabled",
                "short_enabled",
            ]
        }
        to_dump.update(
            {
                "live_config": decode_individual(individual),
                "adg": adg,
                "worst_drawdown": worst_drawdown,
                "drawdowns_daily_mean": drawdowns_daily_mean,
                "sharpe_ratio": sharpe_ratio,
            }
        )
        with open(self.results_cache_fname, "a") as f:
            f.write(json.dumps(denumpyize(to_dump)) + "\n")
        return adg, sharpe_ratio, worst_drawdown


def get_individual_keys():
    return [
        "global_TWE_long",
        "global_TWE_short",
        "global_loss_allowance_pct",
        "global_stuck_threshold",
        "global_unstuck_close_pct",
        "long_ddown_factor",
        "long_ema_span_0",
        "long_ema_span_1",
        "long_initial_eprice_ema_dist",
        "long_initial_qty_pct",
        "long_markup_range",
        "long_min_markup",
        "long_n_close_orders",
        "long_rentry_pprice_dist",
        "long_rentry_pprice_dist_wallet_exposure_weighting",
        "short_ddown_factor",
        "short_ema_span_0",
        "short_ema_span_1",
        "short_initial_eprice_ema_dist",
        "short_initial_qty_pct",
        "short_markup_range",
        "short_min_markup",
        "short_n_close_orders",
        "short_rentry_pprice_dist",
        "short_rentry_pprice_dist_wallet_exposure_weighting",
    ]


def config_to_individual(config):
    keys = get_individual_keys()
    individual = [0.0 for _ in range(len(keys))]
    for i, key in enumerate(keys):
        key_ = key[key.find("_") + 1 :]
        if key.startswith("global"):
            if key_ in config:
                individual[i] = config[key_]
            elif "global" in config and key_ in config["global"]:
                individual[i] = config["global"][key_]
            else:
                print(f"warning: '{key_}' missing from config. Setting {key_} = 0.0")
        else:
            pside = key[: key.find("_")]
            individual[i] = config[pside][key_]
    return individual


def decode_individual(individual):
    decoded = {"global": {}, "long": {}, "short": {}}
    for i, key in enumerate(get_individual_keys()):
        for k0 in decoded:
            if key.startswith(k0):
                decoded[k0][key.replace(k0 + "_", "")] = individual[i]
                break
    return decoded


def individual_to_live_configs(individual, symbols):
    keys = get_individual_keys()
    assert len(keys) == len(individual)
    live_configs = {symbol: {"long": {}, "short": {}} for symbol in symbols}
    for i, key in enumerate(keys):
        if key.startswith("global"):
            if "TWE" in key:
                pside = key[key.find("TWE") + 4 :]
                for symbol in live_configs:
                    live_configs[symbol][pside]["wallet_exposure_limit"] = individual[i] / len(
                        symbols
                    )
            else:
                live_configs[key.replace("global_", "")] = individual[i]
        else:
            for symbol in symbols:
                if key.startswith("long"):
                    live_configs[symbol]["long"][key.replace("long_", "")] = individual[i]
                elif key.startswith("short"):
                    live_configs[symbol]["short"][key.replace("short_", "")] = individual[i]
    for symbol in symbols:
        for key, val in [
            ("auto_unstuck_delay_minutes", 0.0),
            ("auto_unstuck_ema_dist", 0.0),
            ("auto_unstuck_qty_pct", 0.0),
            ("auto_unstuck_wallet_exposure_threshold", 0.0),
            ("backwards_tp", 1.0),
            ("enabled", 1.0),
        ]:
            live_configs[symbol]["long"][key] = val
            live_configs[symbol]["short"][key] = val
    return live_configs


def backtest_multi(hlcs, config):
    res = backtest_multisymbol_recursive_grid(
        hlcs,
        config["starting_balance"],
        config["maker_fee"],
        config["do_longs"],
        config["do_shorts"],
        config["c_mults"],
        config["symbols"],
        config["qty_steps"],
        config["price_steps"],
        config["min_costs"],
        config["min_qtys"],
        config["live_configs"],
        config["loss_allowance_pct"],
        config["stuck_threshold"],
        config["unstuck_close_pct"],
    )
    return res


def add_starting_configs(pop, config):
    for cfg in config['starting_configs']:
        pass


async def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    parser = argparse.ArgumentParser(prog="optimize_multi", description="run multisym optimize")
    parser.add_argument(
        "-oc",
        "--optimize_config",
        type=str,
        required=False,
        dest="optimize_config_path",
        default="configs/optimize/multi.hjson",
        help="optimize config hjson file",
    )
    config = prep_config_multi(parser)
    '''
    parser.add_argument(
        "-t",
        "--start",
        type=str,
        required=False,
        dest="starting_configs",
        default=None,
        help="start with given live configs.  single json file or dir with multiple json files",
    )
    '''
    config["symbols"] = OrderedDict({k: v for k, v in sorted(config["symbols"].items())})
    config["results_cache_fname"] = make_get_filepath(
        f"results_multi/{ts_to_date_utc(utc_ms())[:19].replace(':', '_')}_all_results.txt"
    )

    hlcs, mss, config = await prep_hlcs_mss_config(config)
    config["qty_steps"] = tuplify([mss[symbol]["qty_step"] for symbol in config["symbols"]])
    config["price_steps"] = tuplify([mss[symbol]["price_step"] for symbol in config["symbols"]])
    config["min_costs"] = tuplify([mss[symbol]["min_cost"] for symbol in config["symbols"]])
    config["min_qtys"] = tuplify([mss[symbol]["min_qty"] for symbol in config["symbols"]])
    config["c_mults"] = tuplify([mss[symbol]["c_mult"] for symbol in config["symbols"]])
    config["do_longs"] = tuplify([config["long_enabled"] for _ in config["symbols"]])
    config["do_shorts"] = tuplify([config["short_enabled"] for _ in config["symbols"]])
    config["maker_fee"] = next(iter(mss.values()))["maker"]
    config["symbols"] = tuple(sorted(config["symbols"]))

    evaluator = Evaluator(hlcs, config)

    NUMBER_OF_VARIABLES = len(config["bounds"])
    BOUNDS = [(x[0], x[1]) for x in config["bounds"].values()]
    n_cpus = max(1, config["n_cpus"])  # Specify the number of CPUs to use

    # Define the problem as a multi-objective optimization
    # Maximize adg; minimize max_drawdown
    weights = (1.0, 1.0, -1.0)
    creator.create("FitnessMulti", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    # Toolbox initialization
    toolbox = base.Toolbox()

    # Attribute generator - generates one float for each parameter with unique bounds
    def create_individual():
        return [random.uniform(BOUND_LOW, BOUND_UP) for BOUND_LOW, BOUND_UP in BOUNDS]

    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluator.evaluate)
    toolbox.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        low=[bound[0] for bound in BOUNDS],
        up=[bound[1] for bound in BOUNDS],
        eta=20.0,
    )
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=[bound[0] for bound in BOUNDS],
        up=[bound[1] for bound in BOUNDS],
        eta=20.0,
        indpb=1.0 / NUMBER_OF_VARIABLES,
    )
    toolbox.register("select", tools.selNSGA2)

    # Parallelization setup
    pool = multiprocessing.Pool(processes=n_cpus)
    toolbox.register("map", pool.map)

    # Population setup
    pop = toolbox.population(n=100)
    #pop = add_starting_configs(pop, config)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    for i, w in enumerate(weights):
        stats.register(f"avg{i}", lambda pop: sum(f[i] for f in pop) / len(pop))
        if w < 0.0:
            stats.register(f"min{i}", lambda pop: min(f[i] for f in pop))
        else:
            stats.register(f"max{i}", lambda pop: max(f[i] for f in pop))

    logging.info(f"starting optimize")
    try:
        # Run the algorithm
        algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            mu=100,
            lambda_=200,
            cxpb=0.7,
            mutpb=0.3,
            ngen=40,
            stats=stats,
            halloffame=hof,
            verbose=True,
        )
    finally:
        # Close the pool
        pool.close()
        pool.join()

    return pop, stats, hof


if __name__ == "__main__":
    asyncio.run(main())
