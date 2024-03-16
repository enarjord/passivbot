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
from multiprocessing import shared_memory

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


def calc_pa_dist_mean(stats):
    elms = []
    for x in stats:
        for lp, sp, p in zip(x[1], x[2], x[3]):
            if lp[1]:
                elms.append(abs(lp[1] - p) / p)
            if sp[1]:
                elms.append(abs(sp[1] - p) / p)
    return (sum(elms) / len(elms)) if elms else 1.0


def analyze_fills_opti(fills, stats, config):
    starting_balance = config["starting_balance"]
    stats_eqs = [(x[0], x[5]) for x in stats]
    fills_eqs = [(x[0], x[5]) for x in fills]

    all_eqs = pd.DataFrame(stats_eqs + fills_eqs).set_index(0).sort_index()[1]
    drawdowns_all = calc_drawdowns(all_eqs)
    worst_drawdown = abs(drawdowns_all.min())

    eq_threshold = starting_balance * 1e-4
    stats_eqs_df = pd.DataFrame(stats_eqs).set_index(0)
    eqs_daily = stats_eqs_df.groupby(stats_eqs_df.index // 1440).last()[1]
    n_days = len(eqs_daily)
    drawdowns_daily = calc_drawdowns(eqs_daily)
    drawdowns_daily_mean = abs(drawdowns_daily.mean())
    eqs_daily_pct_change = eqs_daily.pct_change()
    if eqs_daily.iloc[-1] <= eq_threshold:
        # ensure adg is negative if final equity is low
        adg = (max(eq_threshold, eqs_daily.iloc[-1]) / starting_balance) ** (1.0 / n_days) - 1.0
        adg_weighted = adg
    else:
        # weigh adg to prefer higher adg closer to present
        adgs = [
            eqs_daily_pct_change.iloc[int(len(eqs_daily_pct_change) * (1 - 1 / i)) :].mean()
            for i in range(1, 11)
        ]
        adg = adgs[0]
        adg_weighted = np.mean(adgs)
    eqs_daily_pct_change_std = eqs_daily_pct_change.std()
    sharpe_ratio = adg / eqs_daily_pct_change_std if eqs_daily_pct_change_std else 0.0

    price_action_distance_mean = calc_pa_dist_mean(stats)

    loss_sum_long, profit_sum_long = 0.0, 0.0
    loss_sum_short, profit_sum_short = 0.0, 0.0
    for x in fills:
        if "long" in x[10]:
            if x[2] > 0.0:
                profit_sum_long += x[2]
            elif x[2] < 0.0:
                loss_sum_long += x[2]
        elif "short" in x[10]:
            if x[2] > 0.0:
                profit_sum_short += x[2]
            elif x[2] < 0.0:
                loss_sum_short += x[2]
    loss_profit_ratio_long = abs(loss_sum_long) / profit_sum_long if profit_sum_long > 0.0 else 1.0
    loss_profit_ratio_short = (
        abs(loss_sum_short) / profit_sum_short if profit_sum_short > 0.0 else 1.0
    )
    loss_profit_ratio = (
        abs(loss_sum_long + loss_sum_short) / (profit_sum_long + profit_sum_short)
        if (profit_sum_long + profit_sum_short) > 0.0
        else 1.0
    )
    pnl_long = profit_sum_long + loss_sum_long
    pnl_short = profit_sum_short + loss_sum_short
    pnl_sum = pnl_long + pnl_short
    pnl_ratio_long_short = pnl_long / pnl_sum if pnl_sum else 0.0

    worst_drawdown_mod = (
        max(config["worst_drawdown_lower_bound"], worst_drawdown)
        - config["worst_drawdown_lower_bound"]
    ) * 10**1
    return {
        "w_adg_weighted": worst_drawdown_mod - adg_weighted,
        "w_price_action_distance_mean": worst_drawdown_mod + price_action_distance_mean,
        "w_loss_profit_ratio": worst_drawdown_mod + loss_profit_ratio,
        "w_sharpe_ratio": worst_drawdown_mod - sharpe_ratio,
        "w_drawdowns_daily_mean": worst_drawdown_mod + drawdowns_daily_mean,
        "worst_drawdown": worst_drawdown,
        "n_days": n_days,
        "drawdowns_daily_mean": drawdowns_daily_mean,
        "price_action_distance_mean": price_action_distance_mean,
        "adg_weighted": adg_weighted,
        "adg": adg,
        "sharpe_ratio": sharpe_ratio,
        "loss_profit_ratio": loss_profit_ratio,
        "loss_profit_ratio_long": loss_profit_ratio_long,
        "loss_profit_ratio_short": loss_profit_ratio_short,
        "pnl_ratio_long_short": pnl_ratio_long_short,
    }


class Evaluator:
    def __init__(self, hlcs, config):
        self.hlcs = hlcs
        self.shared_hlcs = shared_memory.SharedMemory(create=True, size=self.hlcs.nbytes)
        self.shared_hlcs_np = np.ndarray(
            self.hlcs.shape, dtype=self.hlcs.dtype, buffer=self.shared_hlcs.buf
        )
        np.copyto(self.shared_hlcs_np, self.hlcs)
        del self.hlcs
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
                "exchange",
                "qty_steps",
                "price_steps",
                "min_costs",
                "min_qtys",
                "worst_drawdown_lower_bound",
                "selected_metrics",
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
        res = backtest_multi(self.shared_hlcs_np, config_)
        fills, stats = res
        analysis = analyze_fills_opti(fills, stats, config_)

        to_dump = {
            "analysis": analysis,
            "live_config": decode_individual(individual),
            "args": {
                "symbols": self.config["symbols"],
                "start_date": self.config["start_date"],
                "end_date": self.config["end_date"],
                "starting_balance": self.config["starting_balance"],
                "exchange": self.config["exchange"],
                "long_enabled": self.config["long_enabled"],
                "short_enabled": self.config["short_enabled"],
                "worst_drawdown_lower_bound": self.config["worst_drawdown_lower_bound"],
            },
        }
        with open(self.results_cache_fname, "a") as f:
            f.write(json.dumps(denumpyize(to_dump)) + "\n")
        return tuple([analysis[k] for k in self.config["selected_metrics"]])

    def cleanup(self):
        # Close and unlink the shared memory
        self.shared_hlcs.close()
        self.shared_hlcs.unlink()


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
    for cfg in config["starting_configs"]:
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
    parser_items = [
        ("c", "n_cpus", "n_cpus", int, ""),
        ("i", "iters", "iters", int, ""),
        ("wd", "worst_drawdown_lower_bound", "worst_drawdown_lower_bound", float, ""),
    ]
    for k0, k1, d, t, h in parser_items:
        parser.add_argument(
            *[f"-{k0}", f"--{k1}"] + ([f"--{k1.replace('_', '-')}"] if "_" in k1 else []),
            type=t,
            required=False,
            dest=d,
            default=None,
            help=f"specify {k1}{h}, overriding value from hjson config.",
        )
    config = prep_config_multi(parser)
    """
    parser.add_argument(
        "-t",
        "--start",
        type=str,
        required=False,
        dest="starting_configs",
        default=None,
        help="start with given live configs.  single json file or dir with multiple json files",
    )
    """
    config["symbols"] = OrderedDict({k: v for k, v in sorted(config["symbols"].items())})
    config["results_cache_fname"] = make_get_filepath(
        f"results_multi/{ts_to_date_utc(utc_ms())[:19].replace(':', '_')}_all_results.txt"
    )
    for key, default_val in [("worst_drawdown_lower_bound", 0.5)]:
        if key not in config:
            config[key] = default_val

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

    config["selected_metrics"] = ("w_adg_weighted", "w_sharpe_ratio")

    try:
        evaluator = Evaluator(hlcs, config)

        NUMBER_OF_VARIABLES = len(config["bounds"])
        BOUNDS = [
            (x[0] * (1 - 1e-12), x[1] * (1 + 1e-12)) if x[0] == x[1] else (x[0], x[1])
            for x in config["bounds"].values()
        ]
        n_cpus = max(1, config["n_cpus"])  # Specify the number of CPUs to use

        # Define the problem as a multi-objective optimization
        weights = (-1.0, -1.0)  # minimize
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
        # pop = add_starting_configs(pop, config)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        for i, w in enumerate(weights):
            stats.register(f"avg{i}", lambda pop: sum(f[i] for f in pop) / len(pop))
            if w < 0.0:
                stats.register(f"min{i}", lambda pop: min(f[i] for f in pop))
            else:
                stats.register(f"max{i}", lambda pop: max(f[i] for f in pop))

        logging.info(f"starting optimize")
        # Run the algorithm
        algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            mu=100,
            lambda_=200,
            cxpb=0.7,
            mutpb=0.3,
            ngen=max(1, int(config["iters"] / 200)),
            stats=stats,
            halloffame=hof,
            verbose=True,
        )
    finally:
        # Close the pool
        logging.info(f"attempting clean shutdown...")
        evaluator.cleanup()
        pool.close()
        pool.join()

    return pop, stats, hof


if __name__ == "__main__":
    asyncio.run(main())
