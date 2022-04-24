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
)
from procedures import (
    add_argparse_args,
    prepare_optimize_config,
    load_live_config,
    make_get_filepath,
    load_exchange_key_secret,
    prepare_backtest_config,
    dump_live_config,
)
from time import sleep, time
import logging
import logging.config

logging.config.dictConfig({"version": 1, "disable_existing_loggers": True})


def backtest_wrap(config_: dict, ticks_caches: dict):
    """
    loads historical data from disk, runs backtest and returns relevant metrics
    """
    config = {
        **{"long": deepcopy(config_["long"]), "short": deepcopy(config_["short"])},
        **{
            k: config_[k]
            for k in [
                "starting_balance",
                "latency_simulation_ms",
                "symbol",
                "market_type",
                "config_no",
            ]
        },
        **{k: v for k, v in config_["market_specific_settings"].items()},
    }
    if config["symbol"] in ticks_caches:
        ticks = ticks_caches[config["symbol"]]
    else:
        ticks = np.load(config_["ticks_cache_fname"])
    try:
        fills_long, fills_short, stats = backtest(config, ticks)
        longs, shorts, sdf, analysis = analyze_fills(fills_long, fills_short, stats, config)
        """
        with open("logs/debug_harmonysearch.txt", "a") as f:
            f.write(json.dumps({"config": denumpyize(config), "analysis": analysis}) + "\n")
        """
        logging.debug(
            f"backtested {config['symbol']: <12} pa distance long {analysis['pa_distance_mean_long']:.6f} "
            + f"pa distance short {analysis['pa_distance_mean_short']:.6f} adg long {analysis['adg_long']:.6f} "
            + f"adg short {analysis['adg_short']:.6f} std long {analysis['pa_distance_std_long']:.5f} "
            + f"std short {analysis['pa_distance_std_short']:.5f}"
        )
    except Exception as e:
        analysis = get_empty_analysis()
        logging.error(f'error with {config["symbol"]} {e}')
        logging.error("config")
        traceback.print_exc()
        with open(make_get_filepath("tmp/harmony_search_errors.txt"), "a") as f:
            f.write(json.dumps([time(), "error", str(e), denumpyize(config)]) + "\n")
    return analysis


class HarmonySearch:
    def __init__(self, config: dict):
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
        self.ticks_caches = {}
        self.shms = {}  # shared memories
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
        if set(results) == set(self.symbols):
            # completed multisymbol iter
            adgs_long = [v["adg_long"] for v in results.values()]
            adg_mean_long = np.mean(adgs_long)
            pa_distance_std_long_raw = np.mean([v["pa_distance_std_long"] for v in results.values()])
            pa_distance_std_long = np.mean(
                [
                    max(self.config["maximum_pa_distance_std_long"], v["pa_distance_std_long"])
                    for v in results.values()
                ]
            )
            PAD_mean_long_raw = np.mean([v["pa_distance_mean_long"] for v in results.values()])
            PAD_mean_long = np.mean(
                [
                    max(self.config["maximum_pa_distance_mean_long"], v["pa_distance_mean_long"])
                    for v in results.values()
                ]
            )
            adg_DGstd_ratios_long = [v["adg_DGstd_ratio_long"] for v in results.values()]
            adg_DGstd_ratios_long_mean = np.mean(adg_DGstd_ratios_long)
            adgs_short = [v["adg_short"] for v in results.values()]
            adg_mean_short = np.mean(adgs_short)
            pa_distance_std_short_raw = np.mean(
                [v["pa_distance_std_short"] for v in results.values()]
            )

            pa_distance_std_short = np.mean(
                [
                    max(self.config["maximum_pa_distance_std_short"], v["pa_distance_std_short"])
                    for v in results.values()
                ]
            )
            PAD_mean_short_raw = np.mean([v["pa_distance_mean_short"] for v in results.values()])

            PAD_mean_short = np.mean(
                [
                    max(self.config["maximum_pa_distance_mean_short"], v["pa_distance_mean_short"])
                    for v in results.values()
                ]
            )
            adg_DGstd_ratios_short = [v["adg_DGstd_ratio_short"] for v in results.values()]
            adg_DGstd_ratios_short_mean = np.mean(adg_DGstd_ratios_short)

            if self.config["score_formula"] == "adg_PAD_mean":
                score_long = -adg_mean_long * min(
                    1.0, self.config["maximum_pa_distance_mean_long"] / PAD_mean_long
                )
                score_short = -adg_mean_short * min(
                    1.0, self.config["maximum_pa_distance_mean_short"] / PAD_mean_short
                )
            elif self.config["score_formula"] == "adg_realized_PAD_mean":
                adgs_realized_long = [v["adg_realized_per_exposure_long"] for v in results.values()]
                adgs_realized_short = [v["adg_realized_per_exposure_short"] for v in results.values()]

                score_long = -np.mean(adgs_realized_long) / max(
                    self.config["maximum_pa_distance_mean_long"], PAD_mean_long
                )
                score_short = -np.mean(adgs_realized_short) / max(
                    self.config["maximum_pa_distance_mean_short"], PAD_mean_short
                )
            elif self.config["score_formula"] == "adg_PAD_std":
                score_long = -adg_mean_long / max(
                    self.config["maximum_pa_distance_std_long"], pa_distance_std_long
                )
                score_short = -adg_mean_short / max(
                    self.config["maximum_pa_distance_std_short"], pa_distance_std_short
                )
            elif self.config["score_formula"] == "adg_DGstd_ratio":
                score_long = -adg_DGstd_ratios_long_mean
                score_short = -adg_DGstd_ratios_short_mean
            elif self.config["score_formula"] == "adg_mean":
                score_long = -adg_mean_long
                score_short = -adg_mean_short
            elif self.config["score_formula"] == "adg_min":
                score_long = -min(adgs_long)
                score_short = -min(adgs_short)
            elif self.config["score_formula"] == "adg_PAD_std_min":
                # best worst score
                scores_long = [
                    v["adg_long"]
                    / max(v["pa_distance_std_long"], self.config["maximum_pa_distance_std_long"])
                    for v in results.values()
                ]
                score_long = -min(scores_long)
                scores_short = [
                    v["adg_short"]
                    / max(v["pa_distance_std_short"], self.config["maximum_pa_distance_std_short"])
                    for v in results.values()
                ]
                score_short = -min(scores_short)
            else:
                raise Exception(f"unknown score formula {self.config['score_formula']}")

            line = f"completed multisymbol iter {cfg['config_no']} "
            if self.do_long:
                line += f"- adg long {adg_mean_long:.6f} PAD long {PAD_mean_long:.6f} std long "
                line += f"{pa_distance_std_long:.5f} score long {score_long:.7f} "
            if self.do_short:
                line += f"- adg short {adg_mean_short:.6f} PAD short {PAD_mean_short:.6f} std short "
                line += f"{pa_distance_std_short:.5f} score short {score_short:.7f}"
            logging.debug(line)
            # check whether initial eval or new harmony
            if "initial_eval_key" in cfg:
                self.hm[cfg["initial_eval_key"]]["long"]["score"] = score_long
                self.hm[cfg["initial_eval_key"]]["short"]["score"] = score_short
            else:
                # check if better than worst in harmony memory
                worst_key_long = sorted(
                    self.hm,
                    key=lambda x: self.hm[x]["long"]["score"]
                    if type(self.hm[x]["long"]["score"]) != str
                    else -np.inf,
                )[-1]
                if self.do_long and score_long < self.hm[worst_key_long]["long"]["score"]:
                    logging.debug(
                        f"improved long harmony, prev score "
                        + f"{self.hm[worst_key_long]['long']['score']:.7f} new score {score_long:.7f} - "
                        + " ".join([str(round_dynamic(e[1], 3)) for e in sorted(cfg["long"].items())])
                    )
                    self.hm[worst_key_long]["long"] = {
                        "config": deepcopy(cfg["long"]),
                        "score": score_long,
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
                if self.do_short and score_short < self.hm[worst_key_short]["short"]["score"]:
                    logging.debug(
                        f"improved short harmony, prev score "
                        + f"{self.hm[worst_key_short]['short']['score']:.7f} new score {score_short:.7f} - "
                        + " ".join(
                            [str(round_dynamic(e[1], 3)) for e in sorted(cfg["short"].items())]
                        ),
                    )
                    self.hm[worst_key_short]["short"] = {
                        "config": deepcopy(cfg["short"]),
                        "score": score_short,
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
            if self.do_long and score_long <= self.hm[best_key_long]["long"]["score"]:
                is_better = True
                logging.info(
                    f"i{cfg['config_no']} - new best config long, score {score_long:.7f} "
                    + f"adg {adg_mean_long / cfg['long']['wallet_exposure_limit']:.7f} "
                    + f"PAD mean {PAD_mean_long_raw:.7f} "
                    + f"PAD std {pa_distance_std_long_raw:.5f} adg/DGstd {adg_DGstd_ratios_long_mean:.7f}"
                )
                tmp_fname += "_long"
                json.dump(
                    results,
                    open(f"{self.results_fpath}{cfg['config_no']:06}_result_long.json", "w"),
                    indent=4,
                    sort_keys=True,
                )
            if self.do_short and score_short <= self.hm[best_key_short]["short"]["score"]:
                is_better = True
                logging.info(
                    f"i{cfg['config_no']} - new best config short, score {score_short:.7f} "
                    + f"adg {adg_mean_short / cfg['short']['wallet_exposure_limit']:.7f} "
                    + f"PAD mean {PAD_mean_short_raw:.7f} "
                    + f"PAD std {pa_distance_std_short_raw:.5f} adg/DGstd {adg_DGstd_ratios_short_mean:.7f}"
                )
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
                for k in ["starting_balance", "latency_simulation_ms", "market_type"]
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
                backtest_wrap, args=(deepcopy(new_harmony), self.ticks_caches)
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
                for k in ["starting_balance", "latency_simulation_ms", "market_type"]
            },
            **{"symbol": self.symbols[0], "initial_eval_key": hm_key, "config_no": self.iter_counter},
        }
        line = f"starting new initial eval {config['config_no']} of {self.n_harmonies} "
        if self.do_long:
            line += " - long: " + " ".join(
                [
                    f"{e[0][:2]}{e[0][-2:]}" + str(round_dynamic(e[1], 3))
                    for e in sorted(self.hm[hm_key]["long"]["config"].items())
                ]
            )
        if self.do_short:
            line += " - short: " + " ".join(
                [
                    f"{e[0][:2]}{e[0][-2:]}" + str(round_dynamic(e[1], 3))
                    for e in sorted(self.hm[hm_key]["short"]["config"].items())
                ]
            )
        logging.info(line)

        config["market_specific_settings"] = self.market_specific_settings[config["symbol"]]
        config["ticks_cache_fname"] = f"{self.bt_dir}/{config['symbol']}/{self.ticks_cache_fname}"
        config["passivbot_mode"] = self.config["passivbot_mode"]

        self.workers[wi] = {
            "config": deepcopy(config),
            "task": self.pool.apply_async(backtest_wrap, args=(deepcopy(config), self.ticks_caches)),
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
        try:
            self.run_()
        finally:
            for s in self.shms:
                self.shms[s].close()
                self.shms[s].unlink()

    def run_(self):

        # initialize ticks cache
        """
        if self.n_cpus >= len(self.symbols) or (
            "cache_ticks" in self.config and self.config["cache_ticks"]
        ):
        """
        if False:
            for s in self.symbols:
                ticks = np.load(f"{self.bt_dir}/{s}/{self.ticks_cache_fname}")
                self.shms[s] = shared_memory.SharedMemory(create=True, size=ticks.nbytes)
                self.ticks_caches[s] = np.ndarray(
                    ticks.shape, dtype=ticks.dtype, buffer=self.shms[s].buf
                )
                self.ticks_caches[s][:] = ticks[:]
                del ticks
                logging.info(f"loaded {s} ticks into shared memory")

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
                                    backtest_wrap, args=(config, self.ticks_caches)
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
                        sleep(0.25)


async def main():
    logging.basicConfig(format="", level=os.environ.get("LOGLEVEL", "INFO"))

    parser = argparse.ArgumentParser(
        prog="Optimize multi symbol", description="Optimize passivbot config multi symbol"
    )
    parser.add_argument(
        "-o",
        "--optimize_config",
        type=str,
        required=False,
        dest="optimize_config_path",
        default="configs/optimize/harmony_search.hjson",
        help="optimize config hjson file",
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
    parser.add_argument(
        "-i", "--iters", type=int, required=False, dest="iters", default=None, help="n optimize iters"
    )
    parser.add_argument(
        "-c", "--n_cpus", type=int, required=False, dest="n_cpus", default=None, help="n cpus"
    )
    parser.add_argument(
        "-le",
        "--long",
        type=str,
        required=False,
        dest="long_enabled",
        default=None,
        help="long enabled: [y/n]",
    )
    parser.add_argument(
        "-se",
        "--short",
        type=str,
        required=False,
        dest="short_enabled",
        default=None,
        help="short enabled: [y/n]",
    )
    parser.add_argument(
        "-pm",
        "--passivbot_mode",
        "--passivbot-mode",
        type=str,
        required=False,
        dest="passivbot_mode",
        default=None,
        help="passivbot mode options: [s/static_grid, r/recursive_grid]",
    )
    parser.add_argument(
        "-sf",
        "--score_formula",
        "--score-formula",
        type=str,
        required=False,
        dest="score_formula",
        default=None,
        help="passivbot score formula options: [adg_PAD_mean, adg_PAD_std, adg_DGstd_ratio, adg_mean, adg_min, adg_PAD_std_min]",
    )
    parser.add_argument(
        "-oh",
        "--ohlcv",
        help="use 1m ohlcv instead of 1s ticks",
        action="store_true",
    )
    parser = add_argparse_args(parser)
    args = parser.parse_args()
    args.symbol = "BTCUSDT"  # dummy symbol
    config = await prepare_optimize_config(args)
    if args.score_formula is not None:
        if args.score_formula not in [
            "adg_PAD_mean",
            "adg_PAD_std",
            "adg_DGstd_ratio",
            "adg_mean",
            "adg_min",
            "adg_PAD_std_min",
        ]:
            logging.error(f"unknown score formula {args.score_formula}")
            logging.error(f"using score formula {config['score_formula']}")
        else:
            config["score_formula"] = args.score_formula
    if args.passivbot_mode is not None:
        if args.passivbot_mode in ["s", "static_grid", "static"]:
            config["passivbot_mode"] = "static_grid"
        elif args.passivbot_mode in ["r", "recursive_grid", "recursive"]:
            config["passivbot_mode"] = "recursive_grid"
        else:
            raise Exception(f"unknown passivbot mode {args.passivbot_mode}")
    passivbot_mode = config["passivbot_mode"]
    assert passivbot_mode in [
        "recursive_grid",
        "static_grid",
    ], f"unknown passivbot mode {passivbot_mode}"
    config.update(get_template_live_config(passivbot_mode))
    config["long"]["backwards_tp"] = config["backwards_tp_long"]
    config["short"]["backwards_tp"] = config["backwards_tp_short"]
    config["exchange"], _, _ = load_exchange_key_secret(config["user"])
    args = parser.parse_args()
    if args.long_enabled is None:
        config["long"]["enabled"] = config["do_long"]
    else:
        if "y" in args.long_enabled.lower():
            config["long"]["enabled"] = config["do_long"] = True
        elif "n" in args.long_enabled.lower():
            config["long"]["enabled"] = config["do_long"] = False
        else:
            raise Exception("please specify y/n with kwarg -le/--long")
    if args.short_enabled is None:
        config["short"]["enabled"] = config["do_short"]
    else:
        if "y" in args.short_enabled.lower():
            config["short"]["enabled"] = config["do_short"] = True
        elif "n" in args.short_enabled.lower():
            config["short"]["enabled"] = config["do_short"] = False
        else:
            raise Exception("please specify y/n with kwarg -le/--short")
    if args.symbol is not None:
        config["symbols"] = args.symbol.split(",")
    if args.n_cpus is not None:
        config["n_cpus"] = args.n_cpus
    config["base_dir"] = args.base_dir
    config["ohlcv"] = args.ohlcv
    print()
    lines = [(k, getattr(args, k)) for k in args.__dict__ if args.__dict__[k] is not None]
    for line in lines:
        logging.info(f"{line[0]: <{max([len(x[0]) for x in lines]) + 2}} {line[1]}")
    print()

    # download ticks .npy file if missing
    if config["ohlcv"]:
        cache_fname = f"{config['start_date']}_{config['end_date']}_ohlcv_cache.npy"
    else:
        cache_fname = f"{config['start_date']}_{config['end_date']}_ticks_cache.npy"
    exchange_name = config["exchange"] + ("_spot" if config["market_type"] == "spot" else "")
    config["symbols"] = sorted(config["symbols"])
    for symbol in config["symbols"]:
        cache_dirpath = os.path.join(args.base_dir, exchange_name, symbol, "caches", "")
        if not os.path.exists(cache_dirpath + cache_fname) or not os.path.exists(
            cache_dirpath + "market_specific_settings.json"
        ):
            logging.info(f"fetching data {symbol}")
            args.symbol = symbol
            tmp_cfg = await prepare_backtest_config(args)
            if config["ohlcv"]:
                data = load_hlc_cache(
                    symbol,
                    config["start_date"],
                    config["end_date"],
                    base_dir=config["base_dir"],
                    spot=config["spot"],
                    exchange=config["exchange"],
                )
            else:
                downloader = Downloader({**config, **tmp_cfg})
                await downloader.get_sampled_ticks()

    # prepare starting configs
    cfgs = []
    if args.starting_configs is not None:
        logging.info("preparing starting configs...")
        if os.path.isdir(args.starting_configs):
            for fname in os.listdir(args.starting_configs):
                try:
                    cfg = load_live_config(os.path.join(args.starting_configs, fname))
                    assert determine_passivbot_mode(cfg) == passivbot_mode, "wrong passivbot mode"
                    cfgs.append(cfg)
                except Exception as e:
                    logging.error(f"error loading config {fname}: {e}")
        elif os.path.exists(args.starting_configs):
            hm_load_failed = True
            if "hm_" in args.starting_configs:
                try:
                    hm = json.load(open(args.starting_configs))
                    for k in hm:
                        cfg = {"long": hm[k]["long"]["config"], "short": hm[k]["short"]["config"]}
                        assert (
                            determine_passivbot_mode(cfg) == passivbot_mode
                        ), "wrong passivbot mode in harmony memory"
                        cfgs.append(cfg)
                    logging.info(f"loaded harmony memory {args.starting_configs}")
                    hm_load_failed = False
                except Exception as e:
                    logging.error(f"error loading harmony memory {args.starting_configs}: {e}")
            if hm_load_failed:
                try:
                    cfg = load_live_config(args.starting_configs)
                    assert determine_passivbot_mode(cfg) == passivbot_mode, "wrong passivbot mode"
                    cfgs.append(cfg)
                except Exception as e:
                    logging.error(f"error loading config {args.starting_configs}: {e}")
    config["starting_configs"] = cfgs
    harmony_search = HarmonySearch(config)
    harmony_search.run()


if __name__ == "__main__":
    asyncio.run(main())
