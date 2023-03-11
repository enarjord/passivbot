import os

os.environ["NOJIT"] = "false"

from downloader import Downloader, load_hlc_cache
import argparse
import asyncio
import json
import numpy as np
import traceback
from copy import deepcopy
from backtest import backtest
from multiprocessing import Pool, shared_memory
from njit_funcs import round_dynamic
from pure_funcs import (
    analyze_fills,
    denumpyize,
    get_template_live_config,
    ts_to_date,
    ts_to_date_utc,
    date_to_ts,
    tuplify,
    sort_dict_keys,
    determine_passivbot_mode,
    get_empty_analysis,
    calc_scores,
)
from procedures import (
    add_argparse_args,
    prepare_optimize_config,
    load_live_config,
    make_get_filepath,
    prepare_backtest_config,
    dump_live_config,
    utc_ms,
)
from time import sleep, time
import logging
import logging.config

logging.config.dictConfig({"version": 1, "disable_existing_loggers": True})


class ParticleSwarmOptimization:
    def __init__(self, config: dict, backtest_wrap):
        self.backtest_wrap = backtest_wrap
        self.config = config
        self.do_long = config["long"]["enabled"]
        self.do_short = config["short"]["enabled"]
        self.n_particles = max(config["n_particles"], len(config["starting_configs"]))
        self.w = config["w"]
        self.c0 = config["c0"]
        self.c1 = config["c1"]
        self.starting_configs = config["starting_configs"]
        self.iters = config["iters"]
        self.n_cpus = config["n_cpus"]
        self.pool = Pool(processes=config["n_cpus"])
        self.long_bounds = sort_dict_keys(config[f"bounds_{self.config['passivbot_mode']}"]["long"])
        self.short_bounds = sort_dict_keys(config[f"bounds_{self.config['passivbot_mode']}"]["short"])
        self.symbols = config["symbols"]
        self.identifying_name = (
            f"{len(self.symbols)}_symbols" if len(self.symbols) > 1 else self.symbols[0]
        )
        self.now_date = ts_to_date(time())[:19].replace(":", "-")
        self.results_fpath = make_get_filepath(
            f"results_particle_swarm_optimization_{self.config['passivbot_mode']}/{self.now_date}_{self.identifying_name}/"
        )
        self.exchange_name = config["exchange"] + ("_spot" if config["market_type"] == "spot" else "")
        self.market_specific_settings = {
            s: json.load(
                open(
                    os.path.join(
                        self.config["base_dir"],
                        self.exchange_name,
                        s,
                        "caches",
                        "market_specific_settings.json",
                    )
                )
            )
            for s in self.symbols
        }
        self.date_range = f"{self.config['start_date']}_{self.config['end_date']}"
        self.bt_dir = os.path.join(self.config["base_dir"], self.exchange_name)
        self.ticks_cache_fname = (
            f"caches/{self.date_range}{'_ohlcv_cache.npy' if config['ohlcv'] else '_ticks_cache.npy'}"
        )
        self.ticks_caches = config["ticks_caches"]
        self.current_best_config = None

        # [{'config': dict, 'task': process, 'id_key': tuple}]
        self.workers = [None for _ in range(self.n_cpus)]

        # swarm = {swarm_key: str: {'long': {'score': float, 'config': dict}, 'short': {...}}}
        self.swarm = {}

        # velocities_long/short = {swarm_key: {k: for k in bounds}}
        self.velocities_long = {}
        self.velocities_short = {}
        # lbests_long/short = {swarm_key: {'config': dict, 'score': float}}
        self.lbests_long = {}
        self.lbests_short = {}
        self.gbest_long = None
        self.gbest_short = None

        # {identifier: {'config': dict,
        #               'single_results': {symbol_finished: single_backtest_result},
        #               'in_progress': set({symbol_in_progress}))}
        self.unfinished_evals = {}

        self.iter_counter = 0

    def post_process(self, wi: int):
        # a worker has finished a job; process it
        cfg = deepcopy(self.workers[wi]["config"])
        id_key = self.workers[wi]["id_key"]
        swarm_key = cfg["swarm_key"]
        symbol = cfg["symbol"]
        self.unfinished_evals[id_key]["single_results"][symbol] = self.workers[wi]["task"].get()
        self.unfinished_evals[id_key]["in_progress"].remove(symbol)
        results = deepcopy(self.unfinished_evals[id_key]["single_results"])
        for s in results:
            results[s]["timestamp_finished"] = utc_ms()
        with open(self.results_fpath + "positions.txt", "a") as f:
            f.write(
                json.dumps({"long": cfg["long"], "short": cfg["short"], "swarm_key": swarm_key})
                + "\n"
            )
        if set(results) == set(self.symbols):
            # completed multisymbol iter
            scores_res = calc_scores(self.config, results)
            scores, means, raws, keys = (
                scores_res["scores"],
                scores_res["means"],
                scores_res["raws"],
                scores_res["keys"],
            )

            self.swarm[swarm_key]["long"]["score"] = scores["long"]
            self.swarm[swarm_key]["short"]["score"] = scores["short"]
            # check if better than lbest long
            if (
                type(self.lbests_long[swarm_key]["score"]) == str
                or scores["long"] < self.lbests_long[swarm_key]["score"]
            ):
                self.lbests_long[swarm_key] = deepcopy(
                    {"config": cfg["long"], "score": scores["long"]}
                )
            # check if better than lbest short
            if (
                type(self.lbests_short[swarm_key]["score"]) == str
                or scores["short"] < self.lbests_short[swarm_key]["score"]
            ):
                self.lbests_short[swarm_key] = deepcopy(
                    {"config": cfg["short"], "score": scores["short"]}
                )
            tmp_fname = f"{self.results_fpath}{cfg['config_no']:06}_best_config"
            is_better = False
            # check if better than gbest long
            if self.gbest_long is None or scores["long"] < self.gbest_long["score"]:
                self.gbest_long = deepcopy({"config": cfg["long"], "score": scores["long"]})
                is_better = True
                line = f"i{cfg['config_no']} - new best config long, score {round_dynamic(scores['long'], 12)} "
                for key, _ in keys:
                    line += f"{key} {round_dynamic(raws['long'][key], 4)} "
                logging.info(line)
                tmp_fname += "_long"
                json.dump(
                    results,
                    open(f"{self.results_fpath}{cfg['config_no']:06}_result_long.json", "w"),
                    indent=4,
                    sort_keys=True,
                )
            # check if better than gbest short
            if self.gbest_short is None or scores["short"] < self.gbest_short["score"]:
                self.gbest_short = deepcopy({"config": cfg["short"], "score": scores["short"]})
                is_better = True
                line = f"i{cfg['config_no']} - new best config short, score {round_dynamic(scores['short'], 12)} "
                for key, _ in keys:
                    line += f"{key} {round_dynamic(raws['short'][key], 4)} "
                logging.info(line)
                tmp_fname += "_short"
                json.dump(
                    results,
                    open(f"{self.results_fpath}{cfg['config_no']:06}_result_short.json", "w"),
                    indent=4,
                    sort_keys=True,
                )
            if is_better:
                best_config = {
                    "long": deepcopy(self.gbest_long["config"]),
                    "short": deepcopy(self.gbest_short["config"]),
                }
                best_config["result"] = {
                    "symbol": f"{len(self.symbols)}_symbols",
                    "exchange": self.config["exchange"],
                    "start_date": self.config["start_date"],
                    "end_date": self.config["end_date"],
                }
                dump_live_config(best_config, tmp_fname + ".json")
            elif cfg["config_no"] % 25 == 0:
                logging.info(f"i{cfg['config_no']}")
            results["config_no"] = cfg["config_no"]
            with open(self.results_fpath + "all_results.txt", "a") as f:
                f.write(
                    json.dumps(
                        {"config": {"long": cfg["long"], "short": cfg["short"]}, "results": results}
                    )
                    + "\n"
                )
            del self.unfinished_evals[id_key]
        self.workers[wi] = None

    def start_new_particle_position(self, wi: int):
        self.iter_counter += 1  # up iter counter on each new config started
        swarm_key = self.swarm_keys[self.iter_counter % self.n_particles]
        template = get_template_live_config(self.config["passivbot_mode"])
        new_position = {
            **{
                "long": deepcopy(template["long"]),
                "short": deepcopy(template["short"]),
            },
            **{
                k: self.config[k]
                for k in [
                    "starting_balance",
                    "latency_simulation_ms",
                    "market_type",
                    "adg_n_subdivisions",
                ]
            },
            **{"symbol": self.symbols[0], "config_no": self.iter_counter},
        }
        for side in ["long", "short"]:
            new_position[side]["enabled"] = getattr(self, f"do_{side}")
            new_position[side]["backwards_tp"] = self.config[f"backwards_tp_{side}"]
        for key in self.long_bounds:
            # get new velocities from gbest and lbest
            self.velocities_long[swarm_key][key] = (
                self.w * self.velocities_long[swarm_key][key]
                + self.c0
                * np.random.random()
                * (
                    self.lbests_long[swarm_key]["config"][key]
                    - self.swarm[swarm_key]["long"]["config"][key]
                )
                + self.c1
                * np.random.random()
                * (self.gbest_long["config"][key] - self.swarm[swarm_key]["long"]["config"][key])
            )
            new_position["long"][key] = max(
                min(
                    self.swarm[swarm_key]["long"]["config"][key]
                    + self.velocities_long[swarm_key][key],
                    self.long_bounds[key][1],
                ),
                self.long_bounds[key][0],
            )
            self.velocities_short[swarm_key][key] = (
                self.w * self.velocities_short[swarm_key][key]
                + self.c0
                * np.random.random()
                * (
                    self.lbests_short[swarm_key]["config"][key]
                    - self.swarm[swarm_key]["short"]["config"][key]
                )
                + self.c1
                * np.random.random()
                * (self.gbest_short["config"][key] - self.swarm[swarm_key]["short"]["config"][key])
            )
            new_position["short"][key] = max(
                min(
                    self.swarm[swarm_key]["short"]["config"][key]
                    + self.velocities_short[swarm_key][key],
                    self.short_bounds[key][1],
                ),
                self.short_bounds[key][0],
            )
        self.swarm[swarm_key]["long"] = {
            "config": deepcopy(new_position["long"]),
            "score": "in_progress",
        }
        self.swarm[swarm_key]["short"] = {
            "config": deepcopy(new_position["short"]),
            "score": "in_progress",
        }
        logging.debug(
            f"starting new position {new_position['config_no']} - long "
            + " ".join([str(round_dynamic(e[1], 3)) for e in sorted(new_position["long"].items())])
            + " - short: "
            + " ".join([str(round_dynamic(e[1], 3)) for e in sorted(new_position["short"].items())])
        )

        new_position["market_specific_settings"] = self.market_specific_settings[
            new_position["symbol"]
        ]
        new_position[
            "ticks_cache_fname"
        ] = f"{self.bt_dir}/{new_position['symbol']}/{self.ticks_cache_fname}"
        new_position["passivbot_mode"] = self.config["passivbot_mode"]
        new_position["swarm_key"] = swarm_key
        self.workers[wi] = {
            "config": deepcopy(new_position),
            "task": self.pool.apply_async(
                self.backtest_wrap, args=(deepcopy(new_position), self.ticks_caches)
            ),
            "id_key": new_position["config_no"],
        }
        self.unfinished_evals[new_position["config_no"]] = {
            "config": deepcopy(new_position),
            "single_results": {},
            "in_progress": set([self.symbols[0]]),
        }

    def start_new_initial_eval(self, wi: int, swarm_key: str):
        self.iter_counter += 1  # up iter counter on each new config started
        config = {
            **{
                "long": deepcopy(self.swarm[swarm_key]["long"]["config"]),
                "short": deepcopy(self.swarm[swarm_key]["short"]["config"]),
            },
            **{
                k: self.config[k]
                for k in [
                    "starting_balance",
                    "latency_simulation_ms",
                    "market_type",
                    "adg_n_subdivisions",
                ]
            },
            **{
                "symbol": self.symbols[0],
                "initial_eval_key": swarm_key,
                "config_no": self.iter_counter,
                "swarm_key": swarm_key,
            },
        }
        line = f"starting new initial eval {config['config_no']} of {self.n_particles} "
        logging.info(line)

        config["market_specific_settings"] = self.market_specific_settings[config["symbol"]]
        config["ticks_cache_fname"] = f"{self.bt_dir}/{config['symbol']}/{self.ticks_cache_fname}"
        config["passivbot_mode"] = self.config["passivbot_mode"]

        self.workers[wi] = {
            "config": deepcopy(config),
            "task": self.pool.apply_async(
                self.backtest_wrap, args=(deepcopy(config), self.ticks_caches)
            ),
            "id_key": config["config_no"],
        }
        self.unfinished_evals[config["config_no"]] = {
            "config": deepcopy(config),
            "single_results": {},
            "in_progress": set([self.symbols[0]]),
        }
        self.swarm[swarm_key]["long"]["score"] = "in_progress"
        self.swarm[swarm_key]["short"]["score"] = "in_progress"

    def run(self):
        try:
            self.run_()
        finally:
            pass

    def run_(self):

        # initialize ticks cache
        """
        if self.n_cpus >= len(self.symbols) or (
            "cache_ticks" in self.config and self.config["cache_ticks"]
        ):
        """
        # initialize swarm
        for _ in range(self.n_particles):
            cfg_long = deepcopy(self.config["long"])
            cfg_short = deepcopy(self.config["short"])
            swarm_key = str(time()) + str(np.random.random())
            self.velocities_long[swarm_key] = {}
            self.velocities_short[swarm_key] = {}
            for k in self.long_bounds:
                cfg_long[k] = np.random.uniform(self.long_bounds[k][0], self.long_bounds[k][1])
                cfg_short[k] = np.random.uniform(self.short_bounds[k][0], self.short_bounds[k][1])
                self.velocities_long[swarm_key][k] = np.random.uniform(
                    -abs(self.long_bounds[k][0] - self.long_bounds[k][1]),
                    abs(self.long_bounds[k][0] - self.long_bounds[k][1]),
                )
                self.velocities_short[swarm_key][k] = np.random.uniform(
                    -abs(self.short_bounds[k][0] - self.short_bounds[k][1]),
                    abs(self.short_bounds[k][0] - self.short_bounds[k][1]),
                )
            self.swarm[swarm_key] = {
                "long": {"score": "not_started", "config": cfg_long},
                "short": {"score": "not_started", "config": cfg_short},
            }
            self.lbests_long[swarm_key] = deepcopy(self.swarm[swarm_key]["long"])
            self.lbests_short[swarm_key] = deepcopy(self.swarm[swarm_key]["short"])
        self.gbest_long = deepcopy({"config": cfg_long, "score": np.inf})
        self.gbest_short = deepcopy({"config": cfg_short, "score": np.inf})
        self.swarm_keys = sorted(self.swarm)

        # add starting configs
        for side in ["long", "short"]:
            swarm_keys = sorted(self.swarm)
            bounds = getattr(self, f"{side}_bounds")
            for cfg in self.starting_configs:
                cfg = {k: max(bounds[k][0], min(bounds[k][1], cfg[side][k])) for k in bounds}
                cfg["enabled"] = getattr(self, f"do_{side}")
                cfg["backwards_tp"] = self.config[f"backwards_tp_{side}"]
                if cfg not in [self.swarm[k][side]["config"] for k in self.swarm]:
                    self.swarm[swarm_keys.pop()][side]["config"] = deepcopy(cfg)

        # start main loop
        while True:
            # first check for finished jobs
            for wi in range(len(self.workers)):
                if self.workers[wi] is not None and self.workers[wi]["task"].ready():
                    self.post_process(wi)
            if self.iter_counter >= self.iters + self.n_particles:
                if all(worker is None for worker in self.workers):
                    # break when all work is finished
                    break
            else:
                # check for idle workers
                for wi in range(len(self.workers)):
                    if self.workers[wi] is not None:
                        continue
                    # a worker is idle; give it a job
                    for id_key in self.unfinished_evals:
                        # check if unfinished evals
                        missing_symbols = set(self.symbols) - (
                            set(self.unfinished_evals[id_key]["single_results"])
                            | self.unfinished_evals[id_key]["in_progress"]
                        )
                        if missing_symbols:
                            # start eval for missing symbol
                            symbol = sorted(missing_symbols)[0]
                            config = deepcopy(self.unfinished_evals[id_key]["config"])
                            config["symbol"] = symbol
                            config["market_specific_settings"] = self.market_specific_settings[
                                config["symbol"]
                            ]
                            config[
                                "ticks_cache_fname"
                            ] = f"{self.bt_dir}/{config['symbol']}/{self.ticks_cache_fname}"
                            config["passivbot_mode"] = self.config["passivbot_mode"]
                            self.workers[wi] = {
                                "config": config,
                                "task": self.pool.apply_async(
                                    self.backtest_wrap, args=(config, self.ticks_caches)
                                ),
                                "id_key": id_key,
                            }
                            self.unfinished_evals[id_key]["in_progress"].add(symbol)
                            break
                    else:
                        # means all symbols are accounted for in all unfinished evals; start new eval
                        for swarm_key in self.swarm:
                            if self.swarm[swarm_key]["long"]["score"] == "not_started":
                                # means initial evals not yet done
                                self.start_new_initial_eval(wi, swarm_key)
                                break
                        else:
                            # means initial evals are done; start new position
                            self.start_new_particle_position(wi)
                        sleep(0.0001)


if __name__ == "__main__":
    from optimize import main as main_

    asyncio.run(main_(algorithm="particle_swarm_optimization"))
