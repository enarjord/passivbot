import argparse
import asyncio
import glob
import json
import os
import pprint
import sys
from time import time
from typing import Union

import nevergrad as ng
import numpy as np
import psutil
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.nevergrad import NevergradSearch

from collections import OrderedDict
from backtest import backtest
from backtest import plot_wrap
from downloader import Downloader
from procedures import prep_config, add_argparse_args
from pure_funcs import pack_config, unpack_config, get_template_live_config, ts_to_date, analyze_fills
from reporter import LogReporter

os.environ['TUNE_GLOBAL_CHECKPOINT_S'] = '240'


def get_expanded_ranges(config: dict) -> dict:
    updated_ranges = OrderedDict()
    unpacked = unpack_config(get_template_live_config(config['n_spans']))

    for k0 in unpacked:
        if '£' in k0 or k0 in config['ranges']:
            for k1 in config['ranges']:
                if k1 in k0:
                    updated_ranges[k0] = config['ranges'][k1]
                    if 'leverage' in k0:
                        updated_ranges[k0] = [updated_ranges[k0][0],
                                              min(updated_ranges[k0][1], config['max_leverage'])]
    return updated_ranges


def create_config(config: dict) -> dict:
    updated_ranges = get_expanded_ranges(config)
    template = get_template_live_config(config['n_spans'])
    template['long']['enabled'] = config['do_long']
    template['shrt']['enabled'] = config['do_shrt']
    unpacked = unpack_config(template)
    for k in updated_ranges:
        if updated_ranges[k][0] == updated_ranges[k][1]:
            unpacked[k] = updated_ranges[k][0]
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


def iter_slices_full_first(data, sliding_window_days, ticks_to_prepend, minimum_days):
    yield data
    for d in iter_slices(data, sliding_window_days, ticks_to_prepend, minimum_days):
        yield d


def iter_slices(data, sliding_window_days: float, ticks_to_prepend: int = 0):
    sliding_window_ms = sliding_window_days * 24 * 60 * 60 * 1000
    span_ms = data[2][-1] - data[2][0]
    if sliding_window_ms > span_ms * 0.999:
        yield data
        return
    n_windows = int(np.ceil(span_ms / sliding_window_ms)) + 1
    thresholds_ms = np.linspace(data[2][ticks_to_prepend], data[2][-1] - sliding_window_ms, n_windows)

    for threshold_ms in thresholds_ms[::-1]:
        start_i = max(0, int(np.argmax(data[2] >= threshold_ms) - ticks_to_prepend))
        end_i = min(len(data[2]) - 1, int(np.argmax(data[2] >= threshold_ms + sliding_window_ms)))
        yield tuple(d[start_i:end_i] for d in data)
    for ds in iter_slices(data, sliding_window_days * 2, ticks_to_prepend):
        yield ds


def objective_function(analysis: dict, config: dict, metric='adjusted_daily_gain') -> float:
    if analysis['n_fills'] == 0:
        return -1.0
    return (analysis[metric]
            * min(1.0, config["maximum_hrs_no_fills"] / analysis["max_hrs_no_fills"])
            * min(1.0, config["maximum_hrs_no_fills_same_side"] / analysis["max_hrs_no_fills_same_side"])
            * min(1.0, analysis["closest_bkr"] / config["minimum_bankruptcy_distance"]))


def single_sliding_window_run(config, data, do_print=False) -> (float, [dict]):
    analyses = []
    objective = 0.0
    n_days = config['n_days']
    metric = config['metric'] if 'metric' in config else 'adjusted_daily_gain'
    if config['sliding_window_days'] == 0.0:
        sliding_window_days = n_days
    else:
        # sliding window n days should be greater than max hrs no fills
        sliding_window_days = min(n_days, max([config['maximum_hrs_no_fills'] * 2.1 / 24,
                                               config['maximum_hrs_no_fills_same_side'] * 2.1 / 24,
                                               config['sliding_window_days']]))
    analyses = []
    for z, data_slice in enumerate(iter_slices(data, sliding_window_days,
                                               ticks_to_prepend=int(config['max_span']))):
        if len(data_slice[0]) == 0:
            print('debug b no data')
            continue
        try:
            fills, info = backtest(pack_config(config), data_slice)
        except Exception as e:
            print(e)
            break
        result = {**config, **{'lowest_eqbal_ratio': info[1], 'closest_bkr': info[2]}}
        _, analysis = analyze_fills(fills, {**config, **{'lowest_eqbal_ratio': info[1], 'closest_bkr': info[2]}},
                                    data_slice[2][int(config['max_span'])],
                                    data_slice[2][-1])
        analysis['score'] = objective_function(analysis, config, metric=metric) * (analysis['n_days'] / config['n_days'])
        analyses.append(analysis)
        objective = np.mean([e['score'] for e in analyses]) * max(1.01, config['reward_multiplier_base']) ** (z + 1)
        analyses[-1]['objective'] = objective
        line = (f'{str(z).rjust(3, " ")} adg {analysis["average_daily_gain"]:.4f}, '
                f'bkr {analysis["closest_bkr"]:.4f}, '
                f'eqbal {analysis["lowest_eqbal_ratio"]:.4f} n_days {analysis["n_days"]:.1f}, '
                f'score {analysis["score"]:.4f}, objective {objective:.4f}, '
                f'hrs stuck ss {str(round(analysis["max_hrs_no_fills_same_side"], 1)).zfill(4)}, ')
        bef = config['break_early_factor']
        if bef > 0.0:
            if analysis['closest_bkr'] < config['minimum_bankruptcy_distance'] * (1 - bef):
                line += f"broke on min_bkr_dist {analysis['closest_bkr']:.4f}, {config['minimum_bankruptcy_distance']}, {bef}"
                print(line)
                break
            if analysis['max_hrs_no_fills'] > config['maximum_hrs_no_fills'] * (1 + bef):
                line += f"broke on max_hrs_no_fills {analysis['max_hrs_no_fills']:.4f}, {config['maximum_hrs_no_fills']}, {bef}"
                print(line)
                break
            if analysis['max_hrs_no_fills_same_side'] > config['maximum_hrs_no_fills_same_side'] * (1 + bef):
                line += f"broke on max_hrs_no_fills_ss {analysis['max_hrs_no_fills_same_side']:.4f}, {config['maximum_hrs_no_fills_same_side']}, {bef}"
                print(line)
                break
            if analysis['average_daily_gain'] < 0.996:
                line += f"broke on low adg {analysis['average_daily_gain']:.4f} "
                print(line)
                break
            mean_adg = np.mean([e['average_daily_gain'] for e in analyses])
            if z > 2 and mean_adg < 1.0:
                line += f"broke on low mean adg {mean_adg:.4f} "
                print(line)
                break
        print(line)
    return objective, analyses

def simple_sliding_window_wrap(config, data, do_print=False):
    objective, analyses = single_sliding_window_run(config, data)
    tune.report(objective=objective,
                daily_gain=np.mean([r['average_daily_gain'] for r in analyses]),
                closest_bankruptcy=np.min([r['closest_bkr'] for r in analyses]),
                max_hrs_no_fills=np.max([r['max_hrs_no_fills'] for r in analyses]),
                max_hrs_no_fills_same_side=np.max([r['max_hrs_no_fills_same_side'] for r in analyses]))


def backtest_tune(data: np.ndarray, config: dict, current_best: Union[dict, list] = None):
    memory = int(np.sum([sys.getsizeof(d) for d in data]) * 1.2)
    virtual_memory = psutil.virtual_memory()
    if (virtual_memory.available - memory) / virtual_memory.total < 0.1:
        print("Available memory would drop below 10%. Please reduce the time span.")
        return None
    config = create_config(config)
    print('tuning:')
    for k, v in config.items():
        if type(v) in [ray.tune.sample.Float, ray.tune.sample.Integer]:
            print(k, (v.lower, v.upper))
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
    if current_best is not None:
        if type(current_best) == list:
            for c in current_best:
                c = clean_start_config(c, config)
                if c not in current_best_params:
                    current_best_params.append(c)
        else:
            current_best = clean_start_config(current_best, config)
            current_best_params.append(current_best)

    ray.init(num_cpus=num_cpus,
             object_store_memory=memory if memory > 4000000000 else None)  # , logging_level=logging.FATAL, log_to_driver=False)
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
                               if (any(k0 in k for k0 in ['const', 'leverage', 'stop_psize_pct', 'PBr']) and '£' in k)
                               or '_span' in k]),
        # if type(config[k]) == ray.tune.sample.Float
        # or type(config[k]) == ray.tune.sample.Integer]),
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
                        default=None,
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
                       'maximum_hrs_no_fills_same_side', 'iters', 'n_particles', 'sliding_window_days', 'metric',
                       'min_span', 'max_span', 'n_spans']):
        if k in config:
            print(f"{k: <{max(map(len, keys)) + 2}} {config[k]}")
    print()
    data = await downloader.get_data()
    config['n_days'] = (data[2][-1] - data[2][0]) / (1000 * 60 * 60 * 24)
    config['optimize_dirpath'] = os.path.join(config['optimize_dirpath'],
                                              ts_to_date(time())[:19].replace(':', ''), '')

    start_candidate = None
    if args.starting_configs is not None:
        try:
            if os.path.isdir(args.starting_configs):
                start_candidate = [json.load(open(f)) for f in glob.glob(os.path.join(args.starting_configs, '*.json'))]
                print('Starting with all configurations in directory.')
            else:
                start_candidate = json.load(open(args.starting_configs))
                print('Starting with specified configuration.')
        except Exception as e:
            print('Could not find specified configuration.', e)
    analysis = backtest_tune(data, config, start_candidate)
    if analysis:
        save_results(analysis, config)
        config.update(clean_result_config(analysis.best_config))
        plot_wrap(pack_config(config), data)


if __name__ == '__main__':
    asyncio.run(main())
