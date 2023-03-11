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


class HarmonySearch:
    def __init__(self, config: dict, backtest_wrap):
        self.backtest_wrap = backtest_wrap
        self.config = config
        self.do_long = config["long"]["enabled"]
        self.do_short = config["short"]["enabled"]
        self.n_harmonies = max(config["n_harmonies"], len(config["starting_configs"]))
        self.starting_configs = config["starting_configs"]
        self.hm_considering_rate = config["hm_considering_rate"]
        self.bandwidth = config["bandwidth"]
        self.pitch_adjusting_rate = config["pitch_adjusting_rate"]
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
            f"results_harmony_search_{self.config['passivbot_mode']}/{self.now_date}_{self.identifying_name}/"
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
        """
        self.ticks_caches = (
            {s: np.load(f"{self.bt_dir}/{s}/{self.ticks_cache_fname}") for s in self.symbols}
            if self.n_harmonies > len(self.symbols)
            else {}
        )
        """
        self.ticks_caches = config["ticks_caches"]
        self.current_best_config = None

        # [{'config': dict, 'task': process, 'id_key': tuple}]
        self.workers = [None for _ in range(self.n_cpus)]

        # hm = {hm_key: str: {'long': {'score': float, 'config': dict}, 'short': {...}}}
        self.hm = {}

        # {identifier: {'config': dict,
        #               'single_results': {symbol_finished: single_backtest_result},
        #               'in_progress': set({symbol_in_progress}))}
        self.unfinished_evals = {}

        self.iter_counter = 0

    def post_process(self, wi: int):
        # a worker has finished a job; process it
        cfg = deepcopy(self.workers[wi]["config"])
        id_key = self.workers[wi]["id_key"]
        symbol = cfg["symbol"]
        self.unfinished_evals[id_key]["single_results"][symbol] = self.workers[wi]["task"].get()
        self.unfinished_evals[id_key]["in_progress"].remove(symbol)
        results = deepcopy(self.unfinished_evals[id_key]["single_results"])
        for s in results:
            results[s]["timestamp_finished"] = utc_ms()
        if set(results) == set(self.symbols):
            # completed multisymbol iter
            scores_res = calc_scores(self.config, results)
            scores, means, raws, keys = (
                scores_res["scores"],
                scores_res["means"],
                scores_res["raws"],
                scores_res["keys"],
            )
            # check whether initial eval or new harmony
            if "initial_eval_key" in cfg:
                self.hm[cfg["initial_eval_key"]]["long"]["score"] = scores["long"]
                self.hm[cfg["initial_eval_key"]]["short"]["score"] = scores["short"]
            else:
                # check if better than worst in harmony memory
                worst_key_long = sorted(
                    self.hm,
                    key=lambda x: self.hm[x]["long"]["score"]
                    if type(self.hm[x]["long"]["score"]) != str
                    else -np.inf,
                )[-1]
                if (
                    self.do_long
                    and not isinstance(self.hm[worst_key_long]["long"]["score"], str)
                    and scores["long"] < self.hm[worst_key_long]["long"]["score"]
                ):
                    self.hm[worst_key_long]["long"] = {
                        "config": deepcopy(cfg["long"]),
                        "score": scores["long"],
                    }
                    json.dump(
                        self.hm,
                        open(f"{self.results_fpath}hm_{cfg['config_no']:06}.json", "w"),
                        indent=4,
                        sort_keys=True,
                    )
                worst_key_short = sorted(
                    self.hm,
                    key=lambda x: self.hm[x]["short"]["score"]
                    if type(self.hm[x]["short"]["score"]) != str
                    else -np.inf,
                )[-1]
                if (
                    self.do_short
                    and not isinstance(self.hm[worst_key_short]["short"]["score"], str)
                    and scores["short"] < self.hm[worst_key_short]["short"]["score"]
                ):
                    self.hm[worst_key_short]["short"] = {
                        "config": deepcopy(cfg["short"]),
                        "score": scores["short"],
                    }
                    json.dump(
                        self.hm,
                        open(f"{self.results_fpath}hm_{cfg['config_no']:06}.json", "w"),
                        indent=4,
                        sort_keys=True,
                    )
            best_key_long = sorted(
                self.hm,
                key=lambda x: self.hm[x]["long"]["score"]
                if type(self.hm[x]["long"]["score"]) != str
                else np.inf,
            )[0]
            best_key_short = sorted(
                self.hm,
                key=lambda x: self.hm[x]["short"]["score"]
                if type(self.hm[x]["short"]["score"]) != str
                else np.inf,
            )[0]
            best_config = {
                "long": deepcopy(self.hm[best_key_long]["long"]["config"]),
                "short": deepcopy(self.hm[best_key_short]["short"]["config"]),
            }
            best_config["result"] = {
                "symbol": f"{len(self.symbols)}_symbols",
                "exchange": self.config["exchange"],
                "start_date": self.config["start_date"],
                "end_date": self.config["end_date"],
            }
            tmp_fname = f"{self.results_fpath}{cfg['config_no']:06}_best_config"
            is_better = False
            if (
                self.do_long
                and not isinstance(self.hm[best_key_long]["long"]["score"], str)
                and scores["long"] <= self.hm[best_key_long]["long"]["score"]
            ):
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
            if (
                self.do_short
                and not isinstance(self.hm[best_key_short]["short"]["score"], str)
                and scores["short"] <= self.hm[best_key_short]["short"]["score"]
            ):
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

    def start_new_harmony(self, wi: int):
        self.iter_counter += 1  # up iter counter on each new config started
        template = get_template_live_config(self.config["passivbot_mode"])
        new_harmony = {
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
            new_harmony[side]["enabled"] = getattr(self, f"do_{side}")
            new_harmony[side]["backwards_tp"] = self.config[f"backwards_tp_{side}"]
        for key in self.long_bounds:
            if np.random.random() < self.hm_considering_rate:
                # take note randomly from harmony memory
                new_note_long = self.hm[np.random.choice(list(self.hm))]["long"]["config"][key]
                new_note_short = self.hm[np.random.choice(list(self.hm))]["short"]["config"][key]
                if np.random.random() < self.pitch_adjusting_rate:
                    # tweak note
                    new_note_long = new_note_long + self.bandwidth * (np.random.random() - 0.5) * abs(
                        self.long_bounds[key][0] - self.long_bounds[key][1]
                    )
                    new_note_short = new_note_short + self.bandwidth * (
                        np.random.random() - 0.5
                    ) * abs(self.short_bounds[key][0] - self.short_bounds[key][1])
                # ensure note is within bounds
                new_note_long = max(
                    self.long_bounds[key][0], min(self.long_bounds[key][1], new_note_long)
                )
                new_note_short = max(
                    self.short_bounds[key][0], min(self.short_bounds[key][1], new_note_short)
                )
            else:
                # new random note
                new_note_long = np.random.uniform(self.long_bounds[key][0], self.long_bounds[key][1])
                new_note_short = np.random.uniform(
                    self.short_bounds[key][0], self.short_bounds[key][1]
                )
            new_harmony["long"][key] = new_note_long
            new_harmony["short"][key] = new_note_short
        logging.debug(
            f"starting new harmony {new_harmony['config_no']} - long "
            + " ".join([str(round_dynamic(e[1], 3)) for e in sorted(new_harmony["long"].items())])
            + " - short: "
            + " ".join([str(round_dynamic(e[1], 3)) for e in sorted(new_harmony["short"].items())])
        )

        new_harmony["market_specific_settings"] = self.market_specific_settings[new_harmony["symbol"]]
        new_harmony[
            "ticks_cache_fname"
        ] = f"{self.bt_dir}/{new_harmony['symbol']}/{self.ticks_cache_fname}"
        new_harmony["passivbot_mode"] = self.config["passivbot_mode"]
        self.workers[wi] = {
            "config": deepcopy(new_harmony),
            "task": self.pool.apply_async(
                self.backtest_wrap, args=(deepcopy(new_harmony), self.ticks_caches)
            ),
            "id_key": new_harmony["config_no"],
        }
        self.unfinished_evals[new_harmony["config_no"]] = {
            "config": deepcopy(new_harmony),
            "single_results": {},
            "in_progress": set([self.symbols[0]]),
        }

    def start_new_initial_eval(self, wi: int, hm_key: str):
        self.iter_counter += 1  # up iter counter on each new config started
        config = {
            **{
                "long": deepcopy(self.hm[hm_key]["long"]["config"]),
                "short": deepcopy(self.hm[hm_key]["short"]["config"]),
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
            **{"symbol": self.symbols[0], "initial_eval_key": hm_key, "config_no": self.iter_counter},
        }
        line = f"starting new initial eval {config['config_no']} of {self.n_harmonies} "
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
        self.hm[hm_key]["long"]["score"] = "in_progress"
        self.hm[hm_key]["short"]["score"] = "in_progress"

    def run(self):
        # initialize harmony memory
        for _ in range(self.n_harmonies):
            cfg_long = deepcopy(self.config["long"])
            cfg_short = deepcopy(self.config["short"])
            for k in self.long_bounds:
                cfg_long[k] = np.random.uniform(self.long_bounds[k][0], self.long_bounds[k][1])
                cfg_short[k] = np.random.uniform(self.short_bounds[k][0], self.short_bounds[k][1])
            hm_key = str(time()) + str(np.random.random())
            self.hm[hm_key] = {
                "long": {"score": "not_started", "config": cfg_long},
                "short": {"score": "not_started", "config": cfg_short},
            }

        # add starting configs
        for side in ["long", "short"]:
            hm_keys = list(self.hm)
            bounds = getattr(self, f"{side}_bounds")
            for cfg in self.starting_configs:
                cfg = {k: max(bounds[k][0], min(bounds[k][1], cfg[side][k])) for k in bounds}
                cfg["enabled"] = getattr(self, f"do_{side}")
                cfg["backwards_tp"] = self.config[f"backwards_tp_{side}"]
                if cfg not in [self.hm[k][side]["config"] for k in self.hm]:
                    self.hm[hm_keys.pop()][side]["config"] = deepcopy(cfg)

        # start main loop
        while True:
            # first check for finished jobs
            for wi in range(len(self.workers)):
                if self.workers[wi] is not None and self.workers[wi]["task"].ready():
                    self.post_process(wi)
            if self.iter_counter >= self.iters + self.n_harmonies:
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
                        for hm_key in self.hm:
                            if self.hm[hm_key]["long"]["score"] == "not_started":
                                # means initial evals not yet done
                                self.start_new_initial_eval(wi, hm_key)
                                break
                        else:
                            # means initial evals are done; start new harmony
                            self.start_new_harmony(wi)
                        sleep(0.0001)


if __name__ == "__main__":
    from optimize import main as main_

    asyncio.run(main_(algorithm="harmony_search"))
