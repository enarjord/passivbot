import pyswarms as ps
import asyncio
import aiomultiprocess
from multiprocessing import shared_memory, Lock
from collections import OrderedDict
from backtest import backtest
from plotting import plot_fills
from downloader import Downloader, prep_config
from pure_funcs import (
    denumpyize,
    numpyize,
    get_template_live_config,
    candidate_to_live_config,
    calc_spans,
    get_template_live_config,
    unpack_config,
    pack_config,
    analyze_fills,
    ts_to_date,
    denanify,
)
from procedures import (
    dump_live_config,
    load_live_config,
    make_get_filepath,
    add_argparse_args,
)
from time import time
from optimize import (
    iter_slices,
    iter_slices_full_first,
    objective_function,
    get_expanded_ranges,
    single_sliding_window_run,
)
import os
import sys
import argparse
import pprint
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import asyncio

lock = Lock()
BEST_OBJECTIVE = 0.0


def get_bounds(ranges: dict) -> tuple:
    return (
        np.array([float(v[0]) for k, v in ranges.items()]),
        np.array([float(v[1]) for k, v in ranges.items()]),
    )


class BacktestPSO:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.expanded_ranges = get_expanded_ranges(config)
        for k in list(self.expanded_ranges):
            if self.expanded_ranges[k][0] == self.expanded_ranges[k][1]:
                del self.expanded_ranges[k]
        self.bounds = get_bounds(self.expanded_ranges)

    def config_to_xs(self, config):
        xs = np.zeros(len(self.bounds[0]))
        unpacked = unpack_config(config)
        for i, k in enumerate(self.expanded_ranges):
            xs[i] = unpacked[k]
        return xs

    def xs_to_config(self, xs):
        config = self.config.copy()
        for i, k in enumerate(self.expanded_ranges):
            config[k] = xs[i]
        return numpyize(denanify(pack_config(config)))

    def rf(self, xss):
        return np.array([self.single_rf(xs) for xs in xss])

    def single_rf(self, xs):
        config = self.xs_to_config(xs)
        objective, analyses = single_sliding_window_run(config, self.data)
        global lock, BEST_OBJECTIVE
        if analyses:
            try:
                lock.acquire()
                to_dump = {}
                for k in ["average_daily_gain", "score"]:
                    to_dump[k] = np.mean([e[k] for e in analyses])
                for k in ["lowest_eqbal_ratio", "closest_bkr"]:
                    to_dump[k] = np.min([e[k] for e in analyses])
                for k in ["max_hrs_no_fills", "max_hrs_no_fills_same_side"]:
                    to_dump[k] = np.max([e[k] for e in analyses])
                to_dump["objective"] = objective
                to_dump.update(candidate_to_live_config(config))
                with open(
                    self.config["optimize_dirpath"] + "intermediate_results.txt", "a"
                ) as f:
                    f.write(json.dumps(to_dump) + "\n")
                if objective > BEST_OBJECTIVE:
                    if analyses:
                        config["average_daily_gain"] = np.mean(
                            [e["average_daily_gain"] for e in analyses]
                        )
                    dump_live_config(
                        {**config, **{"objective": objective}},
                        self.config["optimize_dirpath"]
                        + "intermediate_best_results.json",
                    )
                    BEST_OBJECTIVE = objective
            finally:
                lock.release()
        return -objective


async def main():
    parser = argparse.ArgumentParser(
        prog="Optimize", description="Optimize passivbot config."
    )
    parser = add_argparse_args(parser)
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
    for config in await prep_config(args):
        try:

            template_live_config = get_template_live_config(config["n_spans"])
            config = {**template_live_config, **config}
            dl = Downloader(config)
            data = await dl.get_data()
            shms = [
                shared_memory.SharedMemory(create=True, size=d.nbytes) for d in data
            ]
            shdata = [
                np.ndarray(d.shape, dtype=d.dtype, buffer=shms[i].buf)
                for i, d in enumerate(data)
            ]
            for i in range(len(data)):
                shdata[i][:] = data[i][:]
            del data
            config["n_days"] = (shdata[2][-1] - shdata[2][0]) / (1000 * 60 * 60 * 24)
            config["optimize_dirpath"] = make_get_filepath(
                os.path.join(
                    config["optimize_dirpath"],
                    ts_to_date(time())[:19].replace(":", ""),
                    "",
                )
            )

            print()
            for k in (
                keys := [
                    "exchange",
                    "symbol",
                    "starting_balance",
                    "start_date",
                    "end_date",
                    "latency_simulation_ms",
                    "do_long",
                    "do_short",
                    "minimum_bankruptcy_distance",
                    "maximum_hrs_no_fills",
                    "maximum_hrs_no_fills_same_side",
                    "iters",
                    "n_particles",
                    "sliding_window_size",
                    "n_spans",
                ]
            ):
                if k in config:
                    print(f"{k: <{max(map(len, keys)) + 2}} {config[k]}")
            print()

            bpso = BacktestPSO(tuple(shdata), config)

            optimizer = ps.single.GlobalBestPSO(
                n_particles=24,
                dimensions=len(bpso.bounds[0]),
                options=config["options"],
                bounds=bpso.bounds,
                init_pos=None,
            )
            # todo: implement starting configs
            cost, pos = optimizer.optimize(
                bpso.rf, iters=config["iters"], n_processes=config["num_cpus"]
            )
            print(cost, pos)
            best_candidate = bpso.xs_to_config(pos)
            print("best candidate", best_candidate)
            """
            conf = bpso.xs_to_config(xs)
            print('starting...')
            objective = bpso.rf(xs)
            print(objective)
            """
        finally:
            del shdata
            for shm in shms:
                shm.close()
                shm.unlink()


if __name__ == "__main__":
    asyncio.run(main())
