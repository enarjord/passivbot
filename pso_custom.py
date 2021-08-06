from multiprocessing import shared_memory, Pool
from collections import OrderedDict
from backtest import backtest
from plotting import plot_fills
from downloader import Downloader, prep_config
from pure_funcs import denumpyize, numpyize, get_template_live_config, candidate_to_live_config, calc_spans, \
    get_template_live_config, unpack_config, pack_config, analyze_fills, ts_to_date, denanify, round_dynamic, \
    tuplify
from procedures import dump_live_config, load_live_config, make_get_filepath, add_argparse_args, get_starting_configs
from time import time, sleep
from optimize import get_expanded_ranges, single_sliding_window_run, objective_function
from bisect import insort
from typing import Callable
from prettytable import PrettyTable
from hashlib import sha256
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

    def get_new_velocity_and_position(velocity, position, lbest_, gbest_) -> (np.ndarray, np.ndarray):

        new_velocity = (
            w * velocity
            + c1 * np.random.random(velocity.shape) * (lbest_ - position)
            + c2 * np.random.random(velocity.shape) * (gbest_ - position)
        )
        new_position = position + lr * new_velocity
        new_position = np.where(new_position > bounds[0], new_position, bounds[0])
        new_position = np.where(new_position < bounds[1], new_position, bounds[1])
        return new_velocity, new_position

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

    tested = set()

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
                    pos_hash = sha256(str(positions[pos_cycler]).encode('utf-8')).hexdigest()
                    for _ in range(100):
                        if pos_hash not in tested:
                            break
                        print('debug duplicate candidate')
                        velocities[pos_cycler], positions[pos_cycler] = \
                            get_new_velocity_and_position(velocities[pos_cycler],
                                                          positions[pos_cycler],
                                                          lbests[pos_cycler],
                                                          gbest)
                        pos_hash = sha256(str(positions[pos_cycler]).encode('utf-8')).hexdigest()
                    else:
                        raise Exception('too many duplicate candidates')
                    tested.add(pos_hash)
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
            velocities[pos_cycler], positions[pos_cycler] = \
                get_new_velocity_and_position(velocities[pos_cycler],
                                              positions[pos_cycler],
                                              lbests[pos_cycler],
                                              gbest)
        worker_cycler = (worker_cycler + 1) % len(workers)
        sleep(0.001)
    return gbest, gbest_score


class PostProcessing:
    def __init__(self):
        self.all_backtest_analyses = []

    def process(self, result):
        score, analysis, config = result
        score = -score
        best_score = self.all_backtest_analyses[0][0] if self.all_backtest_analyses else 9e9
        analysis['score'] = score
        try:
            insort(self.all_backtest_analyses, (score, analysis))
        except Exception as e:
            print(e)
            print('score', score)
            print('analysis', analysis)
            print('config', config)
            raise Exception('debug')
        to_dump = denumpyize({**analysis, **pack_config(config)})
        f"{len(self.all_backtest_analyses): <5}"
        table = PrettyTable()
        table.field_names = ['adg', 'bkr_dist', 'eqbal_ratio', 'shrp', 'hrs_no_fills',
                             'hrs_no_fills_ss', 'mean_hrs_btwn_fills', 'n_slices', 'score']
        for elm in self.all_backtest_analyses[:20] + [(score, analysis)]:
            row = [round_dynamic(e, 6)
                   for e in [elm[1]['average_daily_gain'],
                             elm[1]['closest_bkr'],
                             elm[1]['lowest_eqbal_ratio'],
                             elm[1]['sharpe_ratio'],
                             elm[1]['max_hrs_no_fills'],
                             elm[1]['max_hrs_no_fills_same_side'],
                             elm[1]['mean_hrs_between_fills'],
                             elm[1]['completed_slices'],
                             elm[1]['score']]]
            table.add_row(row)
        output = table.get_string(border=True, padding_width=1)
        print(f'\n\n{len(self.all_backtest_analyses)}')
        print(output)
        with open(config['optimize_dirpath'] + 'results.txt', 'a') as f:
            f.write(json.dumps(to_dump) + '\n')
        if score < best_score:
            dump_live_config(to_dump, config['optimize_dirpath'] + 'current_best.json')
        return score


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
        score, analyses = single_sliding_window_run(config, self.data, do_print=True)
        analysis = {}
        for key in ['exchange', 'symbol', 'n_days', 'starting_balance']:
            analysis[key] = analyses[-1][key]
        for key in ['average_periodic_gain', 'average_daily_gain', 'adjusted_daily_gain', 'sharpe_ratio']:
            analysis[key] = np.mean([a[key] for a in analyses])
        for key in ['final_balance', 'final_equity', 'net_pnl_plus_fees', 'gain', 'profit_sum',
                    'n_fills', 'n_entries', 'n_closes', 'n_reentries', 'n_initial_entries',
                    'n_normal_closes', 'n_stop_loss_closes', 'biggest_psize', 'mean_hrs_between_fills',
                    'mean_hrs_between_fills_long', 'mean_hrs_between_fills_shrt', 'max_hrs_no_fills_long',
                    'max_hrs_no_fills_shrt', 'max_hrs_no_fills_same_side', 'max_hrs_no_fills']:
            analysis[key] = np.max([a[key] for a in analyses])
        for key in ['loss_sum', 'fee_sum', 'lowest_eqbal_ratio', 'closest_bkr']:
            analysis[key] = np.min([a[key] for a in analyses])
        analysis['completed_slices'] = len(analyses)
        return score, analysis, config


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
            if config['starting_configs']:
                starting_configs = get_starting_configs(config)
                initial_positions = [backtest_wrap.config_to_xs(cfg) for cfg in starting_configs]
            else:
                initial_positions = []
            pso_multiprocess(backtest_wrap.rf,
                             config['n_particles'],
                             backtest_wrap.bounds,
                             config['options']['c1'],
                             config['options']['c2'],
                             config['options']['w'],
                             n_cpus=config['num_cpus'],
                             iters=config['iters'],
                             initial_positions=initial_positions,
                             post_processing_func=post_processing.process)
        finally:
            del shdata
            shm.close()
            shm.unlink()


if __name__ == '__main__':
    asyncio.run(main())
