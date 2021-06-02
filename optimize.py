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

from analyze import analyze_fills, get_empty_analysis, objective_function
from backtest import backtest, plot_wrap
from downloader import Downloader, prep_config
from njit_funcs import round_
from passivbot import add_argparse_args
from procedures import make_get_ticks_cache
from reporter import LogReporter
from walk_forward_optimization import WFO
from pure_funcs import pack_config, unpack_config, fill_template_config, get_template_live_config, ts_to_date

os.environ['TUNE_GLOBAL_CHECKPOINT_S'] = '240'


def create_config(config: dict) -> dict:
    updated_ranges = {}
    unpacked = unpack_config(fill_template_config(get_template_live_config(config['n_spans'])))
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
        elif any(q in k for q in ['leverage', 'MA_idx']):
            unpacked[k] = tune.randint(updated_ranges[k][0], updated_ranges[k][1])
        else:
            unpacked[k] = tune.uniform(updated_ranges[k][0], updated_ranges[k][1])
    return {**config, **unpacked, **{'ranges': updated_ranges}}


def clean_start_config(start_config: dict, config: dict) -> dict:
    clean_start = unpack_config(start_config)
    for k, v in clean_start.items():
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


def iter_slices(data, sliding_window_size: float, n_windows: int, yield_full: bool = False):
    for ix in np.linspace(1 - sliding_window_size, 0.0, n_windows):
        yield tuple([d[int(round(len(data[0]) * ix)):int(round(len(data[0]) * (ix + sliding_window_size)))]
                     for d in data])
    if yield_full:
        yield data


def simple_sliding_window_wrap(config, data, do_print=False):
    results = []
    n = config['n_sliding_windows']
    for z, data_slice in enumerate(iter_slices(data, config['sliding_window_size'], n)):
        try:
            fills, _, did_finish = backtest(pack_config(config), data_slice, do_print=do_print)
        except Exception as e:
            print('debug a', e, config)
            fills = []
        try:
            _, result = analyze_fills(fills, config, data_slice[2][-1])
        except Exception as e:
            print('debug b', e)
            result = get_empty_analysis(config)
        result['score'] = result['average_daily_gain']
        #result['score'] = objective_function(result, 'average_daily_gain', config)
        results.append(result)
        result['objective'] = np.mean([r['score'] for r in results])
        print(f'\nz {z}, n {n}, adg {result["average_daily_gain"]:.4f}, bkr {result["closest_bkr"]:.4f}, '
              f'score {result["score"]:.4f}, objective {result["objective"]:.4f}')
        if result['objective'] < 1 - config['break_early_factor']:
            break
    tune.report(objective=result['objective'],
                daily_gain=np.mean([r['average_daily_gain'] for r in results]),
                closest_bankruptcy=np.min([r['closest_bkr'] for r in results]),
                max_hrs_no_fills=np.max([r['max_hrs_no_fills'] for r in results]),
                max_hrs_no_fills_same_side=np.max([r['max_hrs_no_fills_same_side'] for r in results]))


def tanh(x):
    return np.tanh(10 * (x - 1))


def simple_sliding_window_wrap_old(config, ticks):
    results = []
    finished_windows = 0.0
    for ticks_slice in iter_slices(ticks, config['sliding_window_size'], config['n_windows'],
                                   yield_full=config['test_full']):
        try:
            fills, _, did_finish = backtest(pack_config(config), ticks_slice, do_print=False)
        except Exception as e:
            print('debug a', e, config)
            fills = []
            did_finish = False
        try:
            _, result_ = analyze_fills(fills, config, ticks_slice[-1][2])
        except Exception as e:
            print('b', e)
            result_ = get_empty_analysis(config)
        results.append(result_)
        finished_windows += 1.0
        if config['break_early_factor'] > 0.0 and \
                (not did_finish or
                 result_['closest_bkr'] < config['minimum_bankruptcy_distance'] * (1 - config['break_early_factor']) or
                 result_['max_hrs_no_fills'] > config['max_hrs_no_fills'] * (1 + config['break_early_factor']) or
                 result_['max_hrs_no_fills_same_side'] > config['max_hrs_no_fills_same_side'] * (
                         1 + config['break_early_factor'])):
            break
    if results:
        result = {}
        for k in results[0]:
            try:
                if k == 'closest_bkr':
                    result[k] = np.min([r[k] for r in results])
                elif k == 'average_daily_gain':
                    if (denominator := np.sum([r['n_days'] for r in results])) == 0.0:
                        result[k] = 1.0
                    else:
                        result[k] = np.sum([r[k] * r['n_days'] for r in results]) / denominator
                    result['adjusted_daily_gain'] = np.mean([tanh(r[k]) for r in results])
                elif 'max_hrs_no_fills' in k:
                    result[k] = np.max([r[k] for r in results])
                else:
                    result[k] = np.mean([r[k] for r in results])
            except:
                result[k] = results[0][k]
    else:
        result = get_empty_analysis(config)

    try:
        objective = objective_function(result, 'average_daily_gain', config) * finished_windows / config['n_windows']
    except Exception as e:
        print('c', e)
        objective = -1
    tune.report(objective=objective, daily_gain=result['average_daily_gain'], closest_bankruptcy=result['closest_bkr'],
                max_hrs_no_fills=result['max_hrs_no_fills'],
                max_hrs_no_fills_same_side=result['max_hrs_no_fills_same_side'])


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

    if 'wfo' in config and config['wfo']:
        print('\n\nwalk forward optimization\n\n')
        wfo = WFO(data, config, P_train=0.5).set_train_N(4)
        backtest_wrap = lambda config: tune_report(wfo.backtest(config))
    else:
        print('\n\nsimple sliding window optimization\n\n')
        backtest_wrap = tune.with_parameters(simple_sliding_window_wrap, data=data)
    analysis = tune.run(
        backtest_wrap, metric='objective', mode='max', name='search',
        search_alg=algo, scheduler=scheduler, num_samples=iters, config=config, verbose=1,
        reuse_actors=True, local_dir=config['optimize_dirpath'],
        progress_reporter=LogReporter(metric_columns=['daily_gain',
                                                      'closest_bankruptcy',
                                                      'max_hrs_no_fills',
                                                      'max_hrs_no_fills_same_side',
                                                      'objective'],
                                      parameter_columns=[k for k in config['ranges']
                                                         if 'const' in k and '§' in k]),
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
    df = df[list(config['ranges'].keys()) + ['daily_gain', 'closest_bankruptcy', 'max_hrs_no_fills',
                                                      'max_hrs_no_fills_same_side', 'objective']].sort_values(
        'objective', ascending=False)
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
    for k in (keys := ['exchange', 'symbol', 'starting_balance', 'start_date', 'end_date',
                       'latency_simulation_ms', 'do_long', 'do_shrt', 'minimum_bankruptcy_distance',
                       'max_hrs_no_fills', 'max_hrs_no_fills_same_side', 'iters', 'n_particles', 'sliding_window_size',
                       'n_sliding_windows' 'test_full']):
        if k in config:
            print(f"{k: <{max(map(len, keys)) + 2}} {config[k]}")
    print()
    ticks = await downloader.get_ticks(True)
    data = make_get_ticks_cache(config, ticks)

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
    plot_wrap(config, data, clean_result_config(analysis.best_config))


if __name__ == '__main__':
    asyncio.run(main())
