from multiprocessing import shared_memory, Pool
from collections import OrderedDict
from backtest import backtest
from plotting import plot_fills
from downloader import Downloader, prep_config
from pure_funcs import denumpyize, numpyize, get_template_live_config, candidate_to_live_config, calc_spans, \
    get_template_live_config, unpack_config, pack_config, analyze_fills, ts_to_date, denanify, round_dynamic
from procedures import dump_live_config, load_live_config, make_get_filepath, add_argparse_args, get_starting_configs
from time import time, sleep
from optimize import get_expanded_ranges, single_sliding_window_run
from bisect import insort
from typing import Callable
import os
import sys
import argparse
import pprint
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import asyncio
import glob


def pso_multiprocess(reward_func: Callable,
                     n_particles: int,
                     bounds: np.ndarray,
                     c1: float,
                     c2: float,
                     w: float,
                     lr: float = 1.0,
                     initial_positions: [np.ndarray] = [],
                     n_cpus: int = 3,
                     iters: int = 10000,
                     post_processing_func: Callable = lambda x: x):
    '''
    if len(initial_positions) <= n_particles: use initial positions as particles, let remainder be random
    else: let n_particles = len(initial_positions)
    '''
    if len(initial_positions) > n_particles:
        positions = numpyize(initial_positions)
    else:
        positions = numpyize([[np.random.uniform(bounds[0][i], bounds[1][i])
                               for i in range(len(bounds[0]))]
                              for _ in range(n_particles)])
        if len(initial_positions) > 0:
            positions[:len(initial_positions)] = initial_positions[:len(positions)]
    positions = np.where(positions > bounds[0], positions, bounds[0])
    positions = np.where(positions < bounds[1], positions, bounds[1])
    velocities = np.zeros_like(positions)
    lbests = np.zeros_like(positions)
    lbest_scores = np.zeros(len(positions))
    lbest_scores[:] = np.inf
    gbest = np.zeros_like(positions[0])
    gbest_score = np.inf

    itr_counter = 0
    worker_cycler = 0
    pos_cycler = 0

    workers = [None for _ in range(n_cpus)]
    working = set()
    pool = Pool(processes=n_cpus)

    while True:
        if itr_counter >= iters:
            if all(worker is None for worker in workers):
                break
        else:
            if workers[worker_cycler] is None:
                if pos_cycler not in working:
                    workers[worker_cycler] = (pos_cycler, pool.apply_async(reward_func, args=(positions[pos_cycler],)))
                    working = set([e[0] for e in workers if e is not None])
                pos_cycler = (pos_cycler + 1) % len(positions)
        if workers[worker_cycler] is not None and workers[worker_cycler][1].ready():
            score = post_processing_func(workers[worker_cycler][1].get())
            pos_idx = workers[worker_cycler][0]
            workers[worker_cycler] = None
            working = set([e[0] for e in workers if e is not None])
            itr_counter += 1
            if score < lbest_scores[pos_idx]:
                lbests[pos_idx], lbest_scores[pos_idx] = positions[pos_idx], score
                if score < gbest_score:
                    gbest, gbest_score = positions[pos_idx], score
            velocities[pos_cycler] = (
                w * velocities[pos_cycler]
                + c1 * np.random.random(velocities[pos_cycler].shape) * (lbests[pos_cycler] - positions[pos_cycler])
                + c2 * np.random.random(velocities[pos_cycler].shape) * (gbest - positions[pos_cycler])
            )
            positions[pos_cycler] = positions[pos_cycler] + lr * velocities[pos_cycler]
            positions[pos_cycler] = np.where(positions[pos_cycler] > bounds[0], positions[pos_cycler], bounds[0])
            positions[pos_cycler] = np.where(positions[pos_cycler] < bounds[1], positions[pos_cycler], bounds[1])

        worker_cycler = (worker_cycler + 1) % len(workers)
        sleep(0.001)
    return gbest, gbest_score


class PostProcessing:
    def __init__(self):
        self.all_backtest_analyses = []

    def process(self, result):
        #score, analysis = single_sliding_window_run(config, self.data)
        score, analyses, config = result
        score = -score
        best_score = self.all_backtest_analyses[0][0] if self.all_backtest_analyses else 9e9
        insort(self.all_backtest_analyses, (score, analyses))
        to_dump = denumpyize({**analyses[0], **config})
        line = f"best_score {best_score:.6f} current score {score:.6f}"
        with open('tmp/btresults.txt', 'a') as f:
            f.write(json.dumps(to_dump) + '\n')
        if score < best_score:
            line += ' new best'
            dump_live_config(to_dump, 'tmp/current_best.json')
        print(line)
        return score


def get_bounds(ranges: dict) -> tuple:     
    return np.array([np.array([float(v[0]) for k, v in ranges.items()]),
                     np.array([float(v[1]) for k, v in ranges.items()])])


def simple_backtest(config, data):
    sample_size_ms = data[1][0] - data[0][0]
    max_span_ito_n_samples = int(config['max_span'] * 60 / (sample_size_ms / 1000))
    fills, info = backtest(pack_config(config), data)
    _, analysis = analyze_fills(fills, {**config, **{'lowest_eqbal_ratio': info[1], 'closest_bkr': info[2]}},
                                data[max_span_ito_n_samples][0],
                                data[-1][0])
    score = analysis['average_daily_gain']
    return score, [analysis]


class BacktestWrap:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.expanded_ranges = get_expanded_ranges(config)
        for k in list(self.expanded_ranges):
            if self.expanded_ranges[k][0] == self.expanded_ranges[k][1]:
                del self.expanded_ranges[k]
        self.bounds = get_bounds(self.expanded_ranges)
        self.starting_configs = get_starting_configs(config)
    
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

    def rf(self, xs):
        config = self.xs_to_config(xs)
        score, analyses = simple_backtest(config, self.data)
        return score, analyses, config

    def post_processing(self, xs, score, analyses):
        if analyses:
            config = self.xs_to_config(xs)
            to_dump = {}
            for k in ['average_daily_gain', 'score']:
                to_dump[k] = np.mean([e[k] for e in analyses])
            for k in ['lowest_eqbal_ratio', 'closest_bkr']:
                to_dump[k] = np.min([e[k] for e in analyses])
            for k in ['max_hrs_no_fills', 'max_hrs_no_fills_same_side']:
                to_dump[k] = np.max([e[k] for e in analyses])
            line = ''
            for k, v in to_dump.items():
                line += f'{k} {round_dynamic(v, 4)} '
            print(line)
            to_dump['score'] = score
            to_dump.update(candidate_to_live_config(config))
            with open(self.config['optimize_dirpath'] + 'results.txt', 'a') as f:
                f.write(json.dumps(to_dump) + '\n')
            return
            # check if new global best
            if new_gbest:
                if analyses:
                    config['average_daily_gain'] = np.mean([e['average_daily_gain'] for e in analyses])
                dump_live_config({**config, **{'score': score, 'n_days': analyses[-1]['n_days']}},
                                 self.config['optimize_dirpath'] + 'best_config.json')
    

async def main():
    parser = argparse.ArgumentParser(prog='Optimize', description='Optimize passivbot config.')
    parser = add_argparse_args(parser)
    parser.add_argument('-t', '--start', type=str, required=False, dest='starting_configs',
                        default=None,
                        help='start with given live configs.  single json file or dir with multiple json files')
    args = parser.parse_args()
    for config in await prep_config(args):
        try:
            template_live_config = get_template_live_config(config['n_spans'])
            config = {**template_live_config, **config}
            dl = Downloader(config)
            data = await dl.get_sampled_ticks()
            shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
            shdata = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
            shdata[:] = data
            del data
            config['n_days'] = (shdata[-1][0] - shdata[0][0]) / (1000 * 60 * 60 * 24)
            config['optimize_dirpath'] = make_get_filepath(os.path.join(config['optimize_dirpath'],
                                                                        ts_to_date(time())[:19].replace(':', ''), ''))

            print()
            for k in (keys := ['exchange', 'symbol', 'starting_balance', 'start_date', 'end_date', 'latency_simulation_ms',
                               'do_long', 'do_shrt', 'minimum_bankruptcy_distance', 'maximum_hrs_no_fills',
                               'maximum_hrs_no_fills_same_side', 'iters', 'n_particles', 'sliding_window_size',
                               'n_spans']):
                if k in config:
                    print(f"{k: <{max(map(len, keys)) + 2}} {config[k]}")
            print()

            backtest_wrap = BacktestWrap(shdata, config)
            post_processing = PostProcessing()
            initial_positions = []#get_initial_positions(args, config, backtest_wrap)
            pso_multiprocess(backtest_wrap.rf,
                             config['n_particles'],
                             backtest_wrap.bounds,
                             config['options']['c1'],
                             config['options']['c2'],
                             config['options']['w'],
                             post_processing_func=post_processing.process)
        finally:
            del shdata
            shm.close()
            shm.unlink()


if __name__ == '__main__':
    asyncio.run(main())
