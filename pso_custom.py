from multiprocessing import shared_memory, Pool
from collections import OrderedDict
from backtest import backtest
from plotting import plot_fills
from downloader import Downloader, prep_config
from pure_funcs import denumpyize, numpyize, get_template_live_config, candidate_to_live_config, calc_spans, \
    get_template_live_config, unpack_config, pack_config, analyze_fills, ts_to_date, denanify, round_dynamic
from procedures import dump_live_config, load_live_config, make_get_filepath, add_argparse_args
from time import time, sleep
from optimize import get_expanded_ranges, single_sliding_window_run
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


def pso_multiprocess(bt, n_particles, bounds, c1, c2, w, lr=1.0, initial_positions: [np.ndarray] = []):
    positions = np.array([[np.random.uniform(bounds[0][i], bounds[1][i])
                           for i in range(len(bounds[0]))]
                          for _ in range(n_particles)])
    if len(initial_positions) > n_particles:
        print(f'warning: {len(initial_positions)} given starting positions with {n_particles} particles')
        print('will choose random subset of starting positions')
        print('to use all starting positions, increase n particles >= n starting positions')
    for i, pos in enumerate(np.random.permutation(initial_positions)[:len(positions)]):
        print('starting_pos', pos)
        positions[i] = np.where(pos > bounds[0], pos, bounds[0])
        positions[i] = np.where(positions[i] < bounds[1], positions[i], bounds[1])

    velocities = np.zeros_like(positions)
    lbests = np.zeros_like(positions)
    lbest_scores = np.zeros(len(positions))
    lbest_scores[:] = np.inf
    gbest = np.zeros_like(positions[0])
    gbest_score = np.inf

    iters = 10000
    k = 0
    z = 0
    i = 0

    num_cpus = 3
    workers = [None for _ in range(num_cpus)]
    working = set()
    pool = Pool(processes=num_cpus)

    while True:
        if k >= iters:
            if all(worker is None for worker in workers):
                break
        else:
            if workers[z] is None:
                if i not in working:
                    workers[z] = (pool.apply_async(bt.rf, args=(positions[i],)), i)
                    working.add(i)
                i = (i + 1) % len(positions)
        if workers[z] is not None and workers[z][0].ready():
            score, analyses = workers[z][0].get()
            q = workers[z][1]
            working.remove(q)
            workers[z] = None
            k += 1
            new_gbest = False
            if score < lbest_scores[q]:
                lbests[q], lbest_scores[q] = positions[q], score
                if score < gbest_score:
                    gbest, gbest_score = positions[q], score
                    new_gbest = True
            bt.post_processing(positions[q], score, analyses, new_gbest)
            velocities[i] = w * velocities[i] + (c1 * np.random.random(velocities[i].shape) * (lbests[i] - positions[i]) +
                                                 c2 * np.random.random(velocities[i].shape) * (gbest - positions[i]))
            positions[i] = positions[i] + lr * velocities[i]
            positions[i] = np.where(positions[i] > bounds[0], positions[i], bounds[0])
            positions[i] = np.where(positions[i] < bounds[1], positions[i], bounds[1])

        z = (z + 1) % len(workers)
        sleep(0.001)
    return gbest, gbest_score


def get_bounds(ranges: dict) -> tuple:     
    return np.array([np.array([float(v[0]) for k, v in ranges.items()]),
                     np.array([float(v[1]) for k, v in ranges.items()])])


class BacktestWrap:
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

    def rf(self, xs):
        config = self.xs_to_config(xs)
        score, analyses = single_sliding_window_run(config, self.data)
        return -score, analyses

    def post_processing(self, xs, score, analyses, new_gbest: bool):
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
    config = await prep_config(args)
    try:

        template_live_config = get_template_live_config(config['n_spans'])
        config = {**template_live_config, **config}
        dl = Downloader(config)
        data = await dl.get_data()
        shms = [shared_memory.SharedMemory(create=True, size=d.nbytes) for d in data]
        shdata = [np.ndarray(d.shape, dtype=d.dtype, buffer=shms[i].buf) for i, d in enumerate(data)]
        for i in range(len(data)):
            shdata[i][:] = data[i][:]
        del data
        config['n_days'] = (shdata[2][-1] - shdata[2][0]) / (1000 * 60 * 60 * 24)
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


        backtest_wrap = BacktestWrap(tuple(shdata), config)
        initial_positions = []
        if args.starting_configs is not None:
            try:
                if os.path.isdir(args.starting_configs):
                    candidates = [load_live_config(f) for f in glob.glob(os.path.join(args.starting_configs, '*.json'))]
                    print('Starting with all configurations in directory.')
                else:
                    candidates = [load_live_config(args.starting_configs)]
                    print('Starting with specified configuration.')
                initial_positions.extend([backtest_wrap.config_to_xs(c) for c in candidates])
            except Exception as e:
                print('Could not find specified configuration.', e)


        pso_multiprocess(backtest_wrap, config['n_particles'], backtest_wrap.bounds,
                         config['options']['c1'], config['options']['c2'], config['options']['w'],
                         lr=1.0, initial_positions=initial_positions)
    finally:
        del shdata
        for shm in shms:
            shm.close()
            shm.unlink()


if __name__ == '__main__':
    asyncio.run(main())
