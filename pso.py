import pyswarms as ps
from multiprocessing import shared_memory, Lock
from collections import OrderedDict
from backtest import backtest
from plotting import plot_fills
from downloader import Downloader, prep_config
from pure_funcs import denumpyize, numpyize, get_template_live_config, candidate_to_live_config, calc_spans, \
    get_template_live_config, unpack_config, pack_config, analyze_fills, ts_to_date
from procedures import dump_live_config, load_live_config, make_get_filepath, add_argparse_args
from time import time
from optimize import iter_slices
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


def get_expand_ranges(config: dict) -> dict:
    expanded_ranges = OrderedDict()
    unpacked = unpack_config(get_template_live_config(config['n_spans']))
    
    for k0 in unpacked:
        if '£' in k0 or k0 in unpacked:
            for k1 in config['ranges']:
                if k1 in k0 and config['ranges'][k1][0] != config['ranges'][k1][1]:
                    expanded_ranges[k0] = config['ranges'][k1]
                    if 'leverage' in k0:
                        expanded_ranges[k0] = [expanded_ranges[k0][0],
                                              min(expanded_ranges[k0][1], config['max_leverage'])]
    return expanded_ranges
                    
def get_bounds(ranges: dict) -> tuple:     
    return (np.array([float(v[0]) for k, v in ranges.items()]),
            np.array([float(v[1]) for k, v in ranges.items()]))


class BacktestPSO:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.expanded_ranges = get_expand_ranges(config)
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
        return numpyize(pack_config(config))
    
    
    def objective_function(self, analysis: dict, config: dict) -> float:
        if analysis['n_fills'] == 0:
            return -1.0
        return (analysis['adjusted_daily_gain']
                * min(1.0, config["maximum_hrs_no_fills"] / analysis["max_hrs_no_fills"])
                * min(1.0, config["maximum_hrs_no_fills_same_side"] / analysis["max_hrs_no_fills_same_side"])
                * min(1.0, analysis["closest_bkr"] / config["minimum_bankruptcy_distance"]))
    
    def rf(self, xss):
        return np.array([self.single_rf(xs) for xs in xss])

    def single_rf(self, xs):
        config = self.xs_to_config(xs)
        #pprint.pprint(config)
        sliding_window_days = max([config['maximum_hrs_no_fills'] / 24,
                                   config['maximum_hrs_no_fills_same_side'] / 24,
                                   config['sliding_window_days']]) * 1.05
        analyses = []
        objective = 0.0
        for z, data_slice in enumerate(iter_slices(self.data, sliding_window_days,
                                                   ticks_to_prepend=int(config['max_span']),
                                                   minimum_days=sliding_window_days * 0.95)):
            if len(data_slice[0]) == 0:
                print('debug b no data')
                continue
            try:
                fills, info = backtest(config, data_slice)
            except Exception as e:
                print(e)
                break
            result = {**config, **{'lowest_eqbal_ratio': info[1], 'closest_bkr': info[2]}}
            _, analysis = analyze_fills(fills, {**config, **{'lowest_eqbal_ratio': info[1], 'closest_bkr': info[2]}},
                                        data_slice[2][int(config['max_span'])],
                                        data_slice[2][-1])
            analysis['score'] = self.objective_function(analysis, config) * (analysis['n_days'] / config['n_days'])
            analyses.append(analysis)
            objective = np.mean([e['score'] for e in analyses]) * 2 ** (z + 1)
            analyses[-1]['objective'] = objective
            print(f'{str(z).rjust(3, " ")} adg {analysis["average_daily_gain"]:.4f}, bkr {analysis["closest_bkr"]:.4f}, '
                  f'eqbal {analysis["lowest_eqbal_ratio"]:.4f} n_days {analysis["n_days"]:.1f}, '
                  f'score {analysis["score"]:.4f}, objective {objective:.4f}, '
                  f'hrs stuck ss {str(round(analysis["max_hrs_no_fills_same_side"], 1)).zfill(4)}, '
                  f'scores {[round(e["score"], 2) for e in analyses]}, ')
            bef = config['break_early_factor']
            if bef > 0.0:
                if analysis['closest_bkr'] < config['minimum_bankruptcy_distance'] * (1 - bef):
                    break
                if analysis['max_hrs_no_fills'] > config['maximum_hrs_no_fills'] * (1 + bef):
                    break
                if analysis['max_hrs_no_fills_same_side'] > config['maximum_hrs_no_fills_same_side'] * (1 + bef):
                    break
        global lock, BEST_OBJECTIVE
        try:
            lock.acquire()
            with open(self.config['optimize_dirpath'] + 'intermediate_results.txt', 'a') as f:
                f.write(json.dumps(analyses) + '\n')
            if objective > BEST_OBJECTIVE:
                if analyses:
                    config['average_daily_gain'] = np.mean([e['average_daily_gain'] for e in analyses])
                dump_live_config(config, self.config['optimize_dirpath'] + 'intermediate_best_results.json')
                BEST_OBJECTIVE = objective
        finally:
            lock.release()
        return -objective
        



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

        bpso = BacktestPSO(tuple(shdata), config)
        lc = load_live_config('configs/live/binance_btsusdt.json')
        xs = bpso.config_to_xs(lc)
        optimizer = ps.single.GlobalBestPSO(n_particles=24, dimensions=len(xs), options=config['options'],
                                            bounds=bpso.bounds, init_pos=None)
        # todo: implement starting configs
        cost, pos = optimizer.optimize(bpso.rf, iters=config['iters'], n_processes=config['num_cpus'])
        print(cost, pos)
        best_candidate = bpso.xs_to_config(pos)
        print('best candidate', best_candidate)
        '''
        conf = bpso.xs_to_config(xs)
        print('starting...')
        objective = bpso.rf(xs)
        print(objective)
        '''
    finally:
        del shdata
        for shm in shms:
            shm.close()
            shm.unlink()


if __name__ == '__main__':
    asyncio.run(main())



