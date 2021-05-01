import asyncio
import glob
import json
import logging
import os
import pprint
import sys
from time import time
from typing import Union

import nevergrad as ng
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.nevergrad import NevergradSearch

from backtest import backtest, plot_wrap, prepare_result, prep_backtest_config
from downloader import Downloader
from passivbot import make_get_filepath, ts_to_date
from jitted import round_
from reporter import LogReporter

os.environ['TUNE_GLOBAL_CHECKPOINT_S'] = '120'


def objective_function(result: dict,
                       liq_cap: float,
                       max_hours_between_fills_cap: int) -> float:
    try:
        return (result['average_daily_gain'] /
                max(1.0, result['max_n_hours_between_fills'] / max_hours_between_fills_cap) *
                min(1.0, result['closest_liq'] / liq_cap))
    except Exception as e:
        print('error with objective function', e, result)
        return 0.0


def create_config(backtest_config: dict) -> dict:
    config = {k: backtest_config[k] for k in backtest_config
              if k not in {'session_name', 'user', 'symbol', 'start_date', 'end_date', 'ranges'}}
    for k in backtest_config['ranges']:
        if backtest_config['ranges'][k][0] == backtest_config['ranges'][k][1]:
            config[k] = backtest_config['ranges'][k][0]
        elif k in ['n_close_orders', 'leverage']:
            config[k] = tune.randint(backtest_config['ranges'][k][0], backtest_config['ranges'][k][1] + 1)
        else:
            config[k] = tune.uniform(backtest_config['ranges'][k][0], backtest_config['ranges'][k][1])
    return config


def clean_start_config(start_config: dict, config: dict, ranges: dict) -> dict:
    clean_start = {}
    for k, v in start_config.items():
        if k in config and k not in ['do_long', 'do_shrt']:
            if type(config[k]) == ray.tune.sample.Float or type(config[k]) == ray.tune.sample.Integer:
                clean_start[k] = min(max(v, ranges[k][0]), ranges[k][1])
    return clean_start


def clean_result_config(config: dict) -> dict:
    for k, v in config.items():
        if type(v) == np.float64:
            config[k] = float(v)
        if type(v) == np.int64 or type(v) == np.int32 or type(v) == np.int16 or type(v) == np.int8:
            config[k] = int(v)
    return config


def wrap_backtest(config, ticks):
    fills, did_finish = backtest(config, ticks)
    result = prepare_result(fills, ticks, config['do_long'], config['do_shrt'])
    objective = objective_function(result,
                                   config['minimum_liquidation_distance'],
                                   config['max_n_hours_between_fills'])
    tune.report(objective=objective, daily_gain=result['average_daily_gain'], closest_liquidation=result['closest_liq'])


def backtest_tune(ticks: np.ndarray, backtest_config: dict, current_best: Union[dict, list] = None):
    config = create_config(backtest_config)
    n_days = round_((ticks[-1][2] - ticks[0][2]) / (1000 * 60 * 60 * 24), 0.1)
    session_dirpath = make_get_filepath(os.path.join('reports', backtest_config['exchange'], backtest_config['symbol'],
                                                     f"{n_days}_days_{ts_to_date(time())[:19].replace(':', '')}", ''))
    iters = 10
    if 'iters' in backtest_config:
        iters = backtest_config['iters']
    else:
        print('Parameter iters should be defined in the configuration. Defaulting to 10.')
    num_cpus = 2
    if 'num_cpus' in backtest_config:
        num_cpus = backtest_config['num_cpus']
    else:
        print('Parameter num_cpus should be defined in the configuration. Defaulting to 2.')
    n_particles = 10
    if 'n_particles' in backtest_config:
        n_particles = backtest_config['n_particles']
    phi1 = 1.4962
    phi2 = 1.4962
    omega = 0.7298
    if 'options' in backtest_config:
        phi1 = backtest_config['options']['c1']
        phi2 = backtest_config['options']['c2']
        omega = backtest_config['options']['w']
    current_best_params = []
    if current_best:
        if type(current_best) == list:
            for c in current_best:
                c = clean_start_config(c, config, backtest_config['ranges'])
                current_best_params.append(c)
        else:
            current_best = clean_start_config(current_best, config, backtest_config['ranges'])
            current_best_params.append(current_best)

    ray.init(num_cpus=num_cpus, logging_level=logging.FATAL, log_to_driver=False)
    pso = ng.optimizers.ConfiguredPSO(transform='identity', popsize=n_particles, omega=omega, phip=phi1, phig=phi2)
    algo = NevergradSearch(optimizer=pso, points_to_evaluate=current_best_params)
    algo = ConcurrencyLimiter(algo, max_concurrent=num_cpus)
    scheduler = AsyncHyperBandScheduler()

    analysis = tune.run(tune.with_parameters(wrap_backtest, ticks=ticks), metric='objective', mode='max', name='search',
                        search_alg=algo, scheduler=scheduler, num_samples=iters, config=config, verbose=1,
                        reuse_actors=True, local_dir=session_dirpath,
                        progress_reporter=LogReporter(metric_columns=['daily_gain',
                                                                      'closest_liquidation',
                                                                      'max_n_hours_between_fills',
                                                                      'objective'],
                                                      parameter_columns=[k for k in backtest_config['ranges'] if type(
                                                          config[k]) == ray.tune.sample.Float or type(
                                                          config[k]) == ray.tune.sample.Integer]))

    ray.shutdown()
    return analysis


def save_results(analysis, backtest_config):
    df = analysis.results_df
    df.reset_index(inplace=True)
    df.drop(columns=['trial_id', 'time_this_iter_s', 'done', 'timesteps_total', 'episodes_total', 'training_iteration',
                     'experiment_id', 'date', 'timestamp', 'time_total_s', 'pid', 'hostname', 'node_ip',
                     'time_since_restore', 'timesteps_since_restore', 'iterations_since_restore', 'experiment_tag'],
            inplace=True)
    df.to_csv(os.path.join(backtest_config['session_dirpath'], 'results.csv'), index=False)
    print('Best candidate found:')
    pprint.pprint(analysis.best_config)


async def main(args: list):
    config_name = args[1]
    backtest_config = await prep_backtest_config(config_name)
    if backtest_config['exchange'] == 'bybit' and not backtest_config['inverse']:
        print('bybit usdt linear backtesting not supported')
        return
    downloader = Downloader(backtest_config)
    ticks = await downloader.get_ticks(True)
    backtest_config['n_days'] = round_((ticks[-1][2] - ticks[0][2]) / (1000 * 60 * 60 * 24), 0.1)

    start_candidate = None
    if (s := '--start') in args:
        try:
            if os.path.isdir(args[args.index(s) + 1]):
                start_candidate = [json.load(open(f)) for f in
                                   glob.glob(os.path.join(args[args.index(s) + 1], '*.json'))]
                print('Starting with all configurations in directory.')
            else:
                start_candidate = json.load(open(args[args.index(s) + 1]))
                print('Starting with specified configuration.')
        except:
            print('Could not find specified configuration.')
    if start_candidate:
        analysis = backtest_tune(ticks, backtest_config, start_candidate)
    else:
        analysis = backtest_tune(ticks, backtest_config)
    save_results(analysis, backtest_config)
    plot_wrap(backtest_config, ticks, clean_result_config(analysis.best_config))


if __name__ == '__main__':
    asyncio.run(main(sys.argv))
