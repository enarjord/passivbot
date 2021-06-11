import argparse
import asyncio
import glob
import json
import os
import pprint
from time import time
from typing import Union

import nevergrad as ng
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.nevergrad import NevergradSearch

from analyze import analyze_fills, get_empty_analysis
from backtest import plot_wrap, backtest
from downloader import Downloader, prep_config
from njit_funcs import round_
from passivbot import add_argparse_args
from procedures import make_get_ticks_cache
from reporter import LogReporter
from pure_funcs import pack_config, unpack_config, get_template_live_config, ts_to_date

os.environ['TUNE_GLOBAL_CHECKPOINT_S'] = '240'


def create_config(config: dict) -> dict:
    updated_ranges = {}
    unpacked = unpack_config(get_template_live_config(config['min_span'], config['max_span'], config['n_spans']))

    for k0 in unpacked:
        if '§' in k0:
            for k1 in config['ranges']:
                if k1 in k0:
                    updated_ranges[k0] = config['ranges'][k1]
                    if 'MA_idx' in k0:
                        updated_ranges[k0] = [updated_ranges[k0][0],
                                              min(updated_ranges[k0][1], config['n_spans'])]
                    elif 'leverage' in k0:
                        updated_ranges[k0] = [updated_ranges[k0][0],
                                              min(updated_ranges[k0][1], config['max_leverage'])]
    for k in updated_ranges:
        if updated_ranges[k][0] == updated_ranges[k][1]:
            unpacked[k] = updated_ranges[k][0]
        elif any(q in k for q in ['MA_idx']):
            unpacked[k] = tune.randint(updated_ranges[k][0], updated_ranges[k][1])
        else:
            unpacked[k] = tune.uniform(updated_ranges[k][0], updated_ranges[k][1])
    return {**config, **unpacked, **{'ranges': updated_ranges}}


def clean_start_config(start_config: dict, config: dict) -> dict:
    clean_start = {}
    for k, v in unpack_config(start_config).items():
        if k in config:
            if type(config[k]) == ray.tune.sample.Float or type(config[k]) == ray.tune.sample.Integer:
                clean_start[k] = min(max(v, config['ranges'][k][0]), config['ranges'][k][1])
    return clean_start


def clean_result_config(config: dict) -> dict:
    for k, v in config.items():
        if type(v) == np.float64:
            config[k] = float(v)
        if type(v) == np.int64 or type(v) == np.int32 or type(v) == np.int16 or type(v) == np.int8:
            config[k] = int(v)
    return config


def iter_slices(data, sliding_window_days: float):
    ms_span = data[2][-1] - data[2][0]
    sliding_window_ms = sliding_window_days * 24 * 60 * 60 * 1000
    n_windows = int(round(ms_span / sliding_window_ms) + 1)
    if sliding_window_ms > ms_span:
        yield data
        return
    ms_thresholds = np.linspace(data[2][0], data[2][-1] - sliding_window_ms, n_windows)
    for ms_threshold in ms_thresholds:
        start_i = np.searchsorted(data[2], ms_threshold)
        end_i = np.searchsorted(data[2], ms_threshold + sliding_window_ms)
        yield tuple(d[start_i:end_i] for d in data)
    for ds in iter_slices(data, sliding_window_days * 2):
        yield ds


def objective_function(analysis: dict, config: dict) -> float:
    if analysis['n_fills'] == 0:
        return -1.0
    return (analysis['adjusted_daily_gain']
            * min(1.0, config["maximum_hrs_no_fills"] / analysis["max_hrs_no_fills"])
            * min(1.0, config["maximum_hrs_no_fills_same_side"] / analysis["max_hrs_no_fills_same_side"])
            * min(1.0, analysis["closest_bkr"] / config["minimum_bankruptcy_distance"]))


def simple_sliding_window_wrap(config, data, do_print=False):
    analyses = []
    objective = 0.0
    n_days = config['n_days']
    sliding_window_days = max(3.0, config['n_days'] * config['sliding_window_size']) # at least 3 days per slice
    config['sliding_window_days'] = sliding_window_days
    data_slices = list(iter_slices(data, sliding_window_days)) if config['sliding_window_size'] < 1.0 else [data]
    n_slices = len(data_slices)
    print('n_days', n_days, 'sliding_window_days', config['sliding_window_days'], 'n_slices', n_slices)
    for z, data_slice in enumerate(data_slices):
        fills, info = backtest(pack_config(config), data_slice, do_print=do_print)
        _, analysis = analyze_fills(fills, {**config, **{'lowest_eqbal_ratio': info[1], 'closest_bkr': info[2]}},
                                    data_slice[2][0], data_slice[2][-1])
        analysis['score'] = objective_function(analysis, config) * (analysis['n_days'] / n_days)
        analyses.append(analysis)
        objective = np.mean([r['score'] for r in analyses]) * ((z + 1) / n_slices)
        print(f'z {z}, n {n_slices}, adg {analysis["average_daily_gain"]:.4f}, bkr {analysis["closest_bkr"]:.4f}, '
              f'eqbal {analysis["lowest_eqbal_ratio"]:.4f} n_days {analysis["n_days"]:.1f}, '
              f'score {analysis["score"]:.4f}, objective {objective:.4f}, '
              f'hrs stuck ss {str(round(analysis["max_hrs_no_fills_same_side"], 1)).zfill(4)}, '
              f'scores {[round(e["score"], 2) for e in analyses]}, ')
        if analysis['closest_bkr'] < config['bankruptcy_distance_break_thr']:
            break
    tune.report(objective=objective,
                daily_gain=np.mean([r['average_daily_gain'] for r in analyses]),
                closest_bankruptcy=np.min([r['closest_bkr'] for r in analyses]),
                max_hrs_no_fills=np.max([r['max_hrs_no_fills'] for r in analyses]),
                max_hrs_no_fills_same_side=np.max([r['max_hrs_no_fills_same_side'] for r in analyses]))


def tune_report(result):
    tune.report(
        objective=result["objective_gmean"],
        daily_gain=result["daily_gains_gmean"],
        closest_bankruptcy=result["closest_bkr"],
        max_hrs_no_fills=result["max_hrs_no_fills"],
        max_hrs_no_fills_same_side=result["max_hrs_no_fills_same_side"],
    )


def backtest_tune(data: np.ndarray, config: dict, current_best: Union[dict, list] = None):
    config = create_config(config)
    print('tuning:')
    for k, v in config.items():
        if type(v) in [ray.tune.sample.Float, ray.tune.sample.Integer]:
            print(k, v)
    config['optimize_dirpath'] = os.path.join(config['optimize_dirpath'],
                                                     ts_to_date(time())[:19].replace(':', ''), '')
    if 'iters' in config:
        iters = config['iters']
    else:
        print('Parameter iters should be defined in the configuration. Defaulting to 10.')
        iters = 10
    if 'num_cpus' in config:
        num_cpus = config['num_cpus']
    else:
        print('Parameter num_cpus should be defined in the configuration. Defaulting to 2.')
        num_cpus = 2
    n_particles = config['n_particles'] if 'n_particles' in config else 10
    phi1 = 1.4962
    phi2 = 1.4962
    omega = 0.7298
    if 'options' in config:
        phi1 = config['options']['c1']
        phi2 = config['options']['c2']
        omega = config['options']['w']
    current_best_params = []
    if current_best:
        if type(current_best) == list:
            for c in current_best:
                c = clean_start_config(c, config)
                if c not in current_best_params:
                    current_best_params.append(c)
        else:
            current_best = clean_start_config(current_best, config)
            current_best_params.append(current_best)

    ray.init(num_cpus=num_cpus)#, logging_level=logging.FATAL, log_to_driver=False)
    pso = ng.optimizers.ConfiguredPSO(transform='identity', popsize=n_particles, omega=omega, phip=phi1, phig=phi2)
    algo = NevergradSearch(optimizer=pso, points_to_evaluate=current_best_params)
    algo = ConcurrencyLimiter(algo, max_concurrent=num_cpus)
    scheduler = AsyncHyperBandScheduler()

    print('\n\nsimple sliding window optimization\n\n')

    backtest_wrap = tune.with_parameters(simple_sliding_window_wrap, data=data)
    analysis = tune.run(
        backtest_wrap, metric='objective', mode='max', name='search',
        search_alg=algo, scheduler=scheduler, num_samples=iters, config=config, verbose=1,
        reuse_actors=True, local_dir=config['optimize_dirpath'],
        progress_reporter=LogReporter(
            metric_columns=['daily_gain',
                            'closest_bankruptcy',
                            'max_hrs_no_fills',
                            'max_hrs_no_fills_same_side',
                            'objective'],
            parameter_columns=[k for k in config['ranges']
                               if any(k0 in k for k0 in ['const', 'leverage', 'stop_psize_pct']) and '§' in k]),
                               #if type(config[k]) == ray.tune.sample.Float
                               #or type(config[k]) == ray.tune.sample.Integer]),
        raise_on_failed_trial=False
    )
    ray.shutdown()
    return analysis


def save_results(analysis, config):
    df = analysis.results_df
    df.reset_index(inplace=True)
    df.rename(columns={column: column.replace('config.', '') for column in df.columns}, inplace=True)
    df = df.sort_values('objective', ascending=False)
    df.to_csv(os.path.join(config['optimize_dirpath'], 'results.csv'), index=False)
    print('Best candidate found:')
    pprint.pprint(analysis.best_config)


async def main():
    parser = argparse.ArgumentParser(prog='Optimize', description='Optimize passivbot config.')
    parser = add_argparse_args(parser)
    parser.add_argument('-t', '--start', type=str, required=False, dest='starting_configs',
                        default='none',
                        help='start with given live configs.  single json file or dir with multiple json files')
    args = parser.parse_args()

    config = await prep_config(args)
    if config['exchange'] == 'bybit' and not config['inverse']:
        print('bybit usdt linear backtesting not supported')
        return
    downloader = Downloader(config)
    print()
    for k in (keys := ['exchange', 'symbol', 'starting_balance', 'start_date', 'end_date', 'latency_simulation_ms',
                       'do_long', 'do_shrt', 'minimum_bankruptcy_distance', 'maximum_hrs_no_fills',
                       'maximum_hrs_no_fills_same_side', 'iters', 'n_particles', 'sliding_window_size', 'min_span', 'max_span', 'n_spans']):
        if k in config:
            print(f"{k: <{max(map(len, keys)) + 2}} {config[k]}")
    print()
    ticks = await downloader.get_ticks(True)
    data = make_get_ticks_cache(config, ticks)
    config['n_days'] = (data[2][-1] - data[2][0]) / (1000 * 60 * 60 * 24)

    start_candidate = None
    if args.starting_configs != 'none':
        try:
            if os.path.isdir(args.starting_configs):
                start_candidate = [json.load(open(f)) for f in glob.glob(os.path.join(args.starting_configs, '*.json'))]
                print('Starting with all configurations in directory.')
            else:
                start_candidate = json.load(open(args.starting_configs))
                print('Starting with specified configuration.')
        except Exception as e:
            print('Could not find specified configuration.', e)
    if start_candidate:
        analysis = backtest_tune(data, config, start_candidate)
    else:
        analysis = backtest_tune(data, config)
    save_results(analysis, config)
    config.update(clean_result_config(analysis.best_config))
    plot_wrap(pack_config(config), data)


if __name__ == '__main__':
    asyncio.run(main())
