import os

os.environ["NOJIT"] = "false"

from downloader import Downloader
import argparse
import asyncio
import json
import numpy as np
import traceback
from backtest import backtest
from multiprocessing import Pool
from njit_funcs import round_dynamic
from pure_funcs import (
    analyze_fills,
    pack_config,
    numpyize,
    denumpyize,
    get_template_live_config,
    ts_to_date,
    tuplify,
    sort_dict_keys,
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


def backtest_single_wrap(config_: dict):
    """
    loads historical data from disk, runs backtest and returns relevant metrics
    """
    config = config_.copy()
    exchange_name = config["exchange"] + (
        "_spot" if config["market_type"] == "spot" else ""
    )
    cache_filepath = f"backtests/{exchange_name}/{config['symbol']}/caches/"
    ticks_filepath = (
        cache_filepath + f"{config['start_date']}_{config['end_date']}_ticks_cache.npy"
    )
    mss = json.load(open(cache_filepath + "market_specific_settings.json"))
    ticks = np.load(ticks_filepath)
    config.update(mss)
    try:
        fills, stats = backtest(config, ticks)
        fdf, sdf, analysis = analyze_fills(fills, stats, config)
        pa_distance_long = analysis["pa_distance_mean_long"]
        pa_distance_short = analysis["pa_distance_mean_short"]
        adg_long = analysis["adg_long"]
        adg_short = analysis["adg_short"]
        print(
            f"backtested {config['symbol']: <12} pa distance long {pa_distance_long:.6f} "
            f"pa distance short {pa_distance_short:.6f} adg long {adg_long:.6f} adg short {adg_short:.6f}"
        )
    except Exception as e:
        print(f'error with {config["symbol"]} {e}')
        print("config")
        traceback.print_exc()
        adg_long, adg_short = 0.0, 0.0
        pa_distance_long = pa_distance_short = 100.0
        with open(make_get_filepath("tmp/harmony_search_errors.txt"), "a") as f:
            f.write(json.dumps([time(), "error", str(e), denumpyize(config)]) + "\n")
    return {
        "pa_distance_long": pa_distance_long,
        "pa_distance_short": pa_distance_short,
        "adg_long": adg_long,
        "adg_short": adg_short,
    }


class HarmonySearch:
    def __init__(self, config: dict):
        self.config = config
        self.n_harmonies = max(config["n_harmonies"], len(config["starting_configs"]))
        self.starting_configs = config["starting_configs"]
        self.hm_considering_rate = config["hm_considering_rate"]
        self.bandwidth = config["bandwidth"]
        self.pitch_adjusting_rate = config["pitch_adjusting_rate"]
        self.iters = config["iters"]
        self.n_cpus = config["n_cpus"]
        self.pool = Pool(processes=config["n_cpus"])
        self.long_bounds = sort_dict_keys(config["bounds"]["long"])
        self.short_bounds = sort_dict_keys(config["bounds"]["short"])
        self.symbols = config["symbols"]
        self.identifying_name = "".join([e[0] for e in config["symbols"]])
        self.now_date = ts_to_date(time())[:19].replace(":", "-")
        self.results_fname = make_get_filepath(
            f"tmp/harmony_search_results_{self.identifying_name}_{self.now_date}.txt"
        )
        self.best_conf_fname = f"tmp/harmony_search_best_config_{self.identifying_name}_{self.now_date}.json"

        # [{'config': dict, 'task': process, 'id_key': tuple}]
        self.workers = [None for _ in range(self.n_cpus)]

        # hm_long/short = [{long/short_conf0}, {long/short_conf1}, ...]
        self.hm_long = []
        self.hm_short = []

        # [score: float]
        self.hm_evals_long = []
        self.hm_evals_short = []

        # {identifier: {'config': dict,
        #               'single_results': {symbol_finished: single_backtest_result},
        #               'in_progress': set({symbol_in_progress}))}
        self.unfinished_evals = {}

        self.iter_counter = 0

    def post_process(self, wi: int):
        # a worker has finished a job; process it
        cfg = self.workers[wi]["config"]
        id_key = self.workers[wi]["id_key"]
        symbol = cfg["symbol"]
        self.unfinished_evals[id_key]["single_results"][symbol] = self.workers[wi][
            "task"
        ].get()
        self.unfinished_evals[id_key]["in_progress"].remove(symbol)
        results = self.unfinished_evals[id_key]["single_results"]
        if set(results) == set(self.symbols):
            # completed multisymbol iter
            adg_mean_long = np.mean([v["adg_long"] for v in results.values()])
            pad_mean_long = np.mean([v["pa_distance_long"] for v in results.values()])
            long_score = -adg_mean_long * min(
                1.0,
                self.config["maximum_pa_distance_mean_long"] / pad_mean_long,
            )
            adg_mean_short = np.mean([v["adg_short"] for v in results.values()])
            pad_mean_short = np.mean([v["pa_distance_short"] for v in results.values()])
            short_score = -adg_mean_short * min(
                1.0,
                self.config["maximum_pa_distance_mean_short"] / pad_mean_short,
            )
            print(
                f"completed multisymbol iter {self.iter_counter}",
                f"adg long {adg_mean_long:.6f} pad long {pad_mean_long:.6f} score long {long_score:.6f}",
                f"adg short {adg_mean_short:.6f} pad short {pad_mean_short:.6f} score short {short_score:.6f}",
            )
            # check whether initial eval or new harmony
            if "initial_eval_i" in cfg:
                self.hm_evals_long[cfg["initial_eval_i"]] = long_score
                self.hm_evals_short[cfg["initial_eval_i"]] = short_score
            else:
                self.iter_counter += 1
                # check if better than worst in harmony memory
                worst_long_i = np.argmax(
                    [-np.inf if type(e) == str else e for e in self.hm_evals_long]
                )
                if long_score < self.hm_evals_long[worst_long_i]:
                    print(
                        f"improved long harmony, prev score ",
                        f"{self.hm_evals_long[worst_long_i]:.5f} new score {long_score:.5f}",
                        " ".join(
                            [str(round_dynamic(e[1], 3)) for e in sorted(cfg["long"].items())]
                        ),
                    )
                    self.hm_long[worst_long_i] = cfg["long"]
                    self.hm_evals_long[worst_long_i] = long_score
                worst_short_i = np.argmax(
                    [-np.inf if type(e) == str else e for e in self.hm_evals_short]
                )
                if short_score < self.hm_evals_short[worst_short_i]:
                    print(
                        f"improved short harmony, prev score ",
                        f"{self.hm_evals_short[worst_short_i]:.5f} new score {short_score:.5f}",
                        " ".join(
                            [str(round_dynamic(e[1], 3)) for e in sorted(cfg["short"].items())]
                        ),
                    )
                    self.hm_short[worst_short_i] = cfg["short"]
                    self.hm_evals_short[worst_short_i] = short_score
            best_config = {
                "long": self.hm_long[
                    np.argmin(
                        [np.inf if type(e) == str else e for e in self.hm_evals_long]
                    )
                ],
                "short": self.hm_short[
                    np.argmin(
                        [np.inf if type(e) == str else e for e in self.hm_evals_short]
                    )
                ],
            }
            best_config["result"] = {
                "symbol": f"{len(self.symbols)}_symbols",
                "exchange": self.config["exchange"],
                "start_date": self.config["start_date"],
                "end_date": self.config["end_date"],
            }
            dump_live_config(best_config, self.best_conf_fname)
            with open(self.results_fname, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "config": {
                                "long": cfg["long"],
                                "short": cfg["short"],
                            },
                            "results": results,
                        }
                    )
                    + "\n"
                )
            del self.unfinished_evals[id_key]
        self.workers[wi] = None

    def start_new_harmony(self, wi: int):
        new_harmony = self.config.copy()
        new_harmony["symbol"] = self.symbols[0]
        for key in self.long_bounds:
            if np.random.random() < self.hm_considering_rate:
                # take note randomly from harmony memory
                new_note_long = np.random.choice(self.hm_long)[key]
                new_note_short = np.random.choice(self.hm_short)[key]
                if np.random.random() < self.pitch_adjusting_rate:
                    # tweak note
                    new_note_long = new_note_long + self.bandwidth * (
                        np.random.random() - 0.5
                    ) * abs(self.long_bounds[key][0] - self.long_bounds[key][1])
                    new_note_short = new_note_short + self.bandwidth * (
                        np.random.random() - 0.5
                    ) * abs(self.short_bounds[key][0] - self.short_bounds[key][1])
                # ensure note is within bounds
                new_note_long = max(
                    self.long_bounds[key][0],
                    min(self.long_bounds[key][1], new_note_long),
                )
                new_note_short = max(
                    self.short_bounds[key][0],
                    min(self.short_bounds[key][1], new_note_short),
                )
            else:
                # new random note
                new_note_long = np.random.uniform(
                    self.long_bounds[key][0],
                    self.long_bounds[key][1],
                )
                new_note_short = np.random.uniform(
                    self.short_bounds[key][0],
                    self.short_bounds[key][1],
                )
            new_harmony["long"][key] = new_note_long
            new_harmony["short"][key] = new_note_short
        print(
            f"starting new harmony {self.iter_counter}, long",
            " ".join([str(round_dynamic(e[1], 3)) for e in sorted(new_harmony["long"].items())]),
            "short:",
            " ".join([str(round_dynamic(e[1], 3)) for e in sorted(new_harmony["short"].items())]),
            self.symbols[0],
        )
        # arbitrary unique identifier
        id_key = str(time()) + str(np.random.random())
        self.workers[wi] = {
            "config": new_harmony,
            "task": self.pool.apply_async(backtest_single_wrap, args=(new_harmony,)),
            "id_key": id_key,
        }
        self.unfinished_evals[id_key] = {
            "config": new_harmony,
            "single_results": {},
            "in_progress": set([self.symbols[0]]),
        }

    def start_new_initial_eval(self, wi: int, ei: int):
        config = self.config.copy()
        config["long"] = self.hm_long[ei]
        config["short"] = self.hm_short[ei]
        config["symbol"] = self.symbols[0]
        config["initial_eval_i"] = ei
        print(
            f"starting new initial eval {ei}, long:",
            " ".join([str(round_dynamic(e[1], 3)) for e in sorted(self.hm_long[ei].items())]),
            "short:",
            " ".join([str(round_dynamic(e[1], 3)) for e in sorted(self.hm_short[ei].items())]),
            self.symbols[0],
        )
        # arbitrary unique identifier
        id_key = str(time()) + str(np.random.random())
        self.workers[wi] = {
            "config": config,
            "task": self.pool.apply_async(backtest_single_wrap, args=(config,)),
            "id_key": id_key,
        }
        self.unfinished_evals[id_key] = {
            "config": config,
            "single_results": {},
            "in_progress": set([self.symbols[0]]),
        }
        self.hm_evals_long[ei] = "in_progress"
        self.hm_evals_short[ei] = "in_progress"

    def run(self):

        # initialize harmony memory
        for _ in range(self.n_harmonies):
            new_cfg = {
                "long": self.config["long"].copy(),
                "short": self.config["short"].copy(),
            }
            for k in self.long_bounds:
                new_cfg["long"][k] = np.random.uniform(
                    self.long_bounds[k][0], self.long_bounds[k][1]
                )
                new_cfg["short"][k] = np.random.uniform(
                    self.short_bounds[k][0], self.short_bounds[k][1]
                )
            self.hm_long.append(new_cfg["long"])
            self.hm_short.append(new_cfg["short"])

        # add starting configs
        seen_long, seen_short = set(), set()
        i_long, i_short = 0, 0
        for cfg in self.starting_configs:
            # ensure starting configs are within bounds and override enabled: true/false
            cfg["long"] = {
                k: max(
                    self.long_bounds[k][0], min(self.long_bounds[k][1], cfg["long"][k])
                )
                for k in self.long_bounds
            }
            cfg["long"]["enabled"] = self.config["long"]["enabled"]
            key_long = tuplify(cfg["long"], sort=True)
            if key_long not in seen_long:
                # prevent duplicates
                self.hm_long[i_long] = cfg["long"]
                i_long += 1
                seen_long.add(key_long)
            cfg["short"] = {
                k: max(
                    self.short_bounds[k][0],
                    min(self.short_bounds[k][1], cfg["short"][k]),
                )
                for k in self.short_bounds
            }
            cfg["short"]["enabled"] = self.config["short"]["enabled"]
            key_short = tuplify(cfg["short"], sort=True)
            if key_short not in seen_short:
                self.hm_short[i_short] = cfg["short"]
                i_short += 1
                seen_short.add(key_short)

        self.hm_evals_long = ["not_started" for _ in range(len(self.hm_long))]
        self.hm_evals_short = ["not_started" for _ in range(len(self.hm_short))]

        # start main loop
        while True:
            # first check for finished jobs
            for wi in range(len(self.workers)):
                if self.workers[wi] is not None and self.workers[wi]["task"].ready():
                    self.post_process(wi)
            if self.iter_counter >= self.iters:
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
                        # check of unfinished evals
                        missing_symbols = set(self.symbols) - (
                            set(self.unfinished_evals[id_key]["single_results"])
                            | self.unfinished_evals[id_key]["in_progress"]
                        )
                        if missing_symbols:
                            # start eval for missing symbol
                            symbol = sorted(missing_symbols).pop(0)
                            config = self.unfinished_evals[id_key]["config"].copy()
                            config["symbol"] = symbol
                            self.workers[wi] = {
                                "config": config,
                                "task": self.pool.apply_async(
                                    backtest_single_wrap, args=(config,)
                                ),
                                "id_key": id_key,
                            }
                            self.unfinished_evals[id_key]["in_progress"].add(symbol)
                            break
                    else:
                        # means all symbols are accounted for in all unfinished evals; start new eval
                        for ei in range(len(self.hm_evals_long)):
                            if self.hm_evals_long[ei] == "not_started":
                                # means initial evals not yet done
                                self.start_new_initial_eval(wi, ei)
                                break
                        else:
                            # means initial evals are done; start new harmony
                            self.start_new_harmony(wi)
                sleep(0.5)


async def main():
    parser = argparse.ArgumentParser(
        prog="Optimize multi symbol",
        description="Optimize passivbot config multi symbol",
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
        "-i",
        "--iters",
        type=int,
        required=False,
        dest="iters",
        default=None,
        help="n optimize iters",
    )
    parser = add_argparse_args(parser)
    args = parser.parse_args()
    args.symbol = "BTCUSDT"  # dummy symbol
    config = await prepare_optimize_config(args)
    config.update(get_template_live_config())
    config["exchange"], _, _ = load_exchange_key_secret(config["user"])
    config["long"]["enabled"] = config["do_long"]
    config["short"]["enabled"] = config["do_short"]

    # download ticks .npy file if missing
    cache_fname = f"{config['start_date']}_{config['end_date']}_ticks_cache.npy"
    exchange_name = config["exchange"] + (
        "_spot" if config["market_type"] == "spot" else ""
    )
    config["symbols"] = sorted(config["symbols"])
    for symbol in config["symbols"]:
        cache_dirpath = f"backtests/{exchange_name}/{symbol}/caches/"
        if not os.path.exists(cache_dirpath + cache_fname) or not os.path.exists(
            cache_dirpath + "market_specific_settings.json"
        ):
            print(f"fetching data {symbol}")
            args.symbol = symbol
            tmp_cfg = await prepare_backtest_config(args)
            downloader = Downloader({**config, **tmp_cfg})
            await downloader.get_sampled_ticks()

    # prepare starting configs
    cfgs = []
    if args.starting_configs is not None:
        if os.path.isdir(args.starting_configs):
            cfgs = []
            for fname in os.listdir(args.starting_configs):
                try:
                    cfgs.append(
                        load_live_config(os.path.join(args.starting_configs, fname))
                    )
                except Exception as e:
                    print("error loading config:", e)
        elif os.path.exists(args.starting_configs):
            try:
                cfgs = [load_live_config(args.starting_configs)]
            except Exception as e:
                print("error loading config:", e)

    config["starting_configs"] = cfgs
    harmony_search = HarmonySearch(config)
    harmony_search.run()


if __name__ == "__main__":
    asyncio.run(main())
