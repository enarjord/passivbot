from __future__ import annotations

import argparse
import asyncio
import json
import logging
import multiprocessing
import pathlib
import time
from pprint import pprint
from typing import Any

import numpy as np

from passivbot.backtest import backtest
from passivbot.downloader import Downloader
from passivbot.types.config import BaseConfig
from passivbot.utils.funcs.pure import analyze_fills
from passivbot.utils.funcs.pure import candidate_to_live_config
from passivbot.utils.funcs.pure import denumpyize
from passivbot.utils.funcs.pure import get_template_live_config
from passivbot.utils.funcs.pure import numpyize
from passivbot.utils.funcs.pure import pack_config
from passivbot.utils.funcs.pure import round_values
from passivbot.utils.funcs.pure import ts_to_date
from passivbot.utils.funcs.pure import unpack_config
from passivbot.utils.procedures import add_backtesting_argparse_args
from passivbot.utils.procedures import dump_live_config
from passivbot.utils.procedures import load_exchange_key_secret
from passivbot.utils.procedures import load_live_config
from passivbot.utils.procedures import post_process_backtesting_argparse_parsed_args
from passivbot.utils.procedures import prepare_backtest_config
from passivbot.utils.procedures import prepare_optimize_config

log = logging.getLogger(__name__)


def backtest_single_wrap(config_: dict[str, Any]) -> tuple[float, float, float]:
    config = config_.copy()
    exchange_name = config["exchange"] + ("_spot" if config["market_type"] == "spot" else "")

    cache_filepath = config["basedir"] / "backtests" / exchange_name / {config["symbol"]} / "caches"
    cache_fname = f"{config['start_date']}_{config['end_date']}_ticks_cache.npy"
    ticks_filepath = cache_filepath / cache_fname

    mss = json.loads(cache_filepath.joinpath("market_specific_settings.json").read_text())
    ticks = np.load(ticks_filepath)
    config.update(mss)
    try:
        fills, stats = backtest(config, ticks)
        fdf, sdf, analysis = analyze_fills(fills, stats, config)
        pa_distance = analysis["pa_distance_mean_long"]
        adg = analysis["average_daily_gain"]
        score = adg * (min(1.0, config["maximum_pa_distance_mean_long"] / pa_distance) ** 2)
        log.info(
            f"backtested {config['symbol']: <12} pa distance {analysis['pa_distance_mean_long']:.6f} "
            f"adg {adg:.6f} score {score:.8f}"
        )
    except Exception as e:
        log.error("error with %s: %s\nConfig: %s", config["symbol"], e, pprint.pformat(config))
        score = -9999999999999.9
        adg = 0.0
        pa_distance = 100.0
        errors_path = config["basedir"] / "tmp" / "harmony_search_errors.txt"
        errors_path.parent.makdir(parents=True, exist_ok=True)
        with errors_path.open("a") as f:
            f.write(json.dumps([time.time(), "error", str(e), denumpyize(config)]) + "\n")
    return (pa_distance, adg, score)


def backtest_multi_wrap(config: dict[str, Any], pool):
    tasks = {}
    for s in sorted(config["symbols"]):
        tasks[s] = pool.apply_async(backtest_single_wrap, args=({**config, **{"symbol": s}},))
    while True:
        if all([task.ready() for task in tasks.values()]):
            break
        time.sleep(0.1)
    results = {k: v.get() for k, v in tasks.items()}
    mean_pa_distance = np.mean([v[0] for v in results.values()])
    mean_adg = np.mean([v[1] for v in results.values()])
    mean_score = np.mean([v[2] for v in results.values()])
    new_score = mean_adg * min(1.0, config["maximum_pa_distance_mean_long"] / mean_pa_distance)
    log.info(
        f"pa distance {mean_pa_distance:.6f} adg {mean_adg:.6f} score {mean_score:8f} new score {new_score:.8f}"
    )
    return -new_score, results


def harmony_search(
    func,
    bounds: np.ndarray,
    n_harmonies: int,
    hm_considering_rate: float,
    bandwidth: float,
    pitch_adjusting_rate: float,
    iters: int,
    starting_xs: list[np.ndarray] = [],
    post_processing_func=None,
):
    # hm == harmony memory
    n_harmonies = max(n_harmonies, len(starting_xs))
    seen = set()
    hm = numpyize(
        [
            [np.random.uniform(bounds[0][i], bounds[1][i]) for i in range(len(bounds[0]))]
            for _ in range(n_harmonies)
        ]
    )
    for i in range(len(starting_xs)):
        assert len(starting_xs[i]) == len(bounds[0])
        harmony = np.array(starting_xs[i])
        for z in range(len(bounds[0])):
            harmony[z] = max(bounds[0][z], min(bounds[1][z], harmony[z]))
        tpl = tuple(harmony)
        if tpl not in seen:
            hm[i] = harmony
        seen.add(tpl)
    log.info("evaluating initial harmonies...")
    hm_evals = numpyize([func(h) for h in hm])

    log.info("best harmony")
    log.info(round_values(denumpyize(hm[hm_evals.argmin()]), 5), f"{hm_evals.min():.8f}")
    if post_processing_func is not None:
        post_processing_func(hm[hm_evals.argmin()])
    log.info("starting search...")
    worst_eval_i = hm_evals.argmax()
    for itr in range(iters):
        new_harmony = np.zeros(len(bounds[0]))
        for note_i in range(len(bounds[0])):
            if np.random.random() < hm_considering_rate:
                new_note = hm[np.random.randint(0, len(hm))][note_i]
                if np.random.random() < pitch_adjusting_rate:
                    new_note = new_note + bandwidth * (np.random.random() - 0.5) * abs(
                        bounds[0][note_i] - bounds[1][note_i]
                    )
                    new_note = max(bounds[0][note_i], min(bounds[1][note_i], new_note))
            else:
                new_note = np.random.uniform(bounds[0][note_i], bounds[1][note_i])
            new_harmony[note_i] = new_note
        h_eval = func(new_harmony)
        if h_eval < hm_evals[worst_eval_i]:
            hm[worst_eval_i] = new_harmony
            hm_evals[worst_eval_i] = h_eval
            worst_eval_i = hm_evals.argmax()
            log.info(
                "improved harmony: %s %s", round_values(denumpyize(new_harmony), 5), f"{h_eval:.8f}"
            )
        log.info(
            "best harmony: %s %s",
            round_values(denumpyize(hm[hm_evals.argmin()]), 5),
            f"{hm_evals.min():.8f}",
        )
        log.info("iteration %s of %s", itr, iters)
        if post_processing_func is not None:
            post_processing_func(hm[hm_evals.argmin()])
    return hm[hm_evals.argmin()]


class FuncWrap:
    def __init__(self, pool, base_config):
        self.pool = pool
        self.base_config = base_config
        self.xs_conf_map = [k for k in sorted(base_config["ranges"])]
        self.bounds = numpyize(
            [
                [self.base_config["ranges"][k][0] for k in self.xs_conf_map],
                [self.base_config["ranges"][k][1] for k in self.xs_conf_map],
            ]
        )
        self.now_date = ts_to_date(time.time())[:19].replace(":", "-")
        self.results_fname = (
            base_config["basedir"] / "tmp" / f"harmony_search_results_{self.now_date}.json"
        )
        self.best_conf_fname = (
            base_config["basedir"] / "tmp" / f"harmony_search_best_config_{self.now_date}.json"
        )

    def xs_to_config(self, xs):
        config = unpack_config(self.base_config.copy())
        for i, x in enumerate(xs):
            config[self.xs_conf_map[i]] = x
        return pack_config(config)

    def config_to_xs(self, config):
        unpacked = unpack_config(config)
        return [unpacked[k] for k in self.xs_conf_map]

    def func(self, xs):
        config = self.xs_to_config(xs)
        score, results = backtest_multi_wrap(config, self.pool)
        with self.results_fname.open("a") as f:
            f.write(
                json.dumps({"config": candidate_to_live_config(config), "results": results}) + "\n"
            )
        return score

    def post_processing_func(self, xs):
        dump_live_config(self.xs_to_config(xs), self.best_conf_fname)


async def _main(args: argparse.Namespace) -> None:
    args.symbol = "BTCUSDT"  # dummy symbol
    config = await prepare_optimize_config(args)
    config.update(get_template_live_config())
    config["exchange"], _, _ = load_exchange_key_secret(config["user"])

    # download ticks .npy file if missing
    cache_fname = f"{config['start_date']}_{config['end_date']}_ticks_cache.npy"
    exchange_name = config["exchange"] + ("_spot" if config["market_type"] == "spot" else "")
    for symbol in sorted(config["symbols"]):
        cache_dirpath = args.backtests_dir / exchange_name / symbol / "caches"
        cache_config = cache_dirpath / cache_fname
        if (
            not cache_config.is_file()
            or not cache_dirpath.joinpath("market_specific_settings.json").isfile()
        ):
            log.info(f"fetching data {symbol}")
            args.symbol = symbol
            tmp_cfg = await prepare_backtest_config(args)
            downloader = Downloader({**config, **tmp_cfg})
            await downloader.get_sampled_ticks()

    pool = multiprocessing.Pool(processes=config["n_cpus"])

    func_wrap = FuncWrap(pool, config)
    cfgs = []
    for path in args.starting_configs:
        if path.isdir():
            for fpath in path.iterdir():
                try:
                    cfgs.append(load_live_config(fpath))
                except Exception as e:
                    log.error("error loading config: %s", e)
        elif path.isfile():
            try:
                cfgs.append(load_live_config(path))
            except Exception as e:
                log.error("error loading config: %s", e)
        # TODO: Else show an Error?

    starting_xs = [func_wrap.config_to_xs(cfg) for cfg in cfgs]

    n_harmonies = config["n_harmonies"]
    hm_considering_rate = config["hm_considering_rate"]
    bandwidth = config["bandwidth"]
    pitch_adjusting_rate = config["pitch_adjusting_rate"]
    iters = config["iters"]
    best_harmony = harmony_search(
        func_wrap.func,
        func_wrap.bounds,
        n_harmonies,
        hm_considering_rate,
        bandwidth,
        pitch_adjusting_rate,
        iters,
        starting_xs=starting_xs,
        post_processing_func=func_wrap.post_processing_func,
    )
    best_conf = func_wrap.xs_to_config(best_harmony)
    log.info("best conf:\n%s", pprint.pformat(best_conf))
    return


def main(args: argparse.Namespace) -> None:
    asyncio.run(_main(args))


def setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o",
        "--optimize_config",
        "--optimize-config",
        type=pathlib.Path,
        required=False,
        dest="optimize_config_path",
        default="configs/optimize/multi_symbol.hjson",
        help="optimize config hjson file",
    )
    parser.add_argument(
        "-t",
        "--start",
        type=pathlib.Path,
        action="append",
        required=False,
        dest="starting_configs",
        default=[],
        help="Start with given live configs. Single JSON file or directory with multiple JSON files",
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
    add_backtesting_argparse_args(parser)
    parser.set_defaults(func=main)


def process_argparse_parsed_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    pass


def post_process_argparse_parsed_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace, config: BaseConfig
) -> None:
    post_process_backtesting_argparse_parsed_args(parser, args)
    if args.optimize_config_path:
        args.optimize_config_path = args.optimize_config_path.resolve()
    else:
        args.optimize_config_path = args.basedir / "optimize" / "multi_symbol.hjson"
