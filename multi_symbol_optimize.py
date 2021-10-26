import os
os.environ['NOJIT'] = 'false'

from downloader import Downloader
import argparse
import asyncio
import json
import hjson
import numpy as np
from backtest import backtest
from multiprocessing import Pool
from pure_funcs import analyze_fills, pack_config, unpack_config, numpyize, denumpyize, config_pretty_str, \
    get_template_live_config, candidate_to_live_config, ts_to_date, round_values
from procedures import add_argparse_args, prepare_optimize_config, load_live_config, make_get_filepath, \
    load_exchange_key_secret, prepare_backtest_config, dump_live_config
from time import sleep, time


def backtest_single_wrap(config_: dict):
    config = config_.copy()
    exchange_name = config['exchange'] + ('_spot' if config['market_type'] == 'spot' else '')
    cache_filepath = f"backtests/{exchange_name}/{config['symbol']}/caches/"
    ticks_filepath = cache_filepath + f"{config['start_date']}_{config['end_date']}_ticks_cache.npy"
    mss = json.load(open(cache_filepath + 'market_specific_settings.json'))
    ticks = np.load(ticks_filepath)
    config.update(mss)
    try:
        fills, stats = backtest(config, ticks)
        fdf, sdf, analysis = analyze_fills(fills, stats, config)
        pa_closeness = analysis['pa_closeness_mean_long']
        adg = analysis['average_daily_gain']
        score = adg * (min(1.0, config['maximum_pa_closeness_mean_long'] / pa_closeness)**2)
        print(f"backtested {config['symbol']: <12} pa closeness {analysis['pa_closeness_mean_long']:.6f} "
              f"adg {adg:.6f} score {score:.8f}")
    except Exception as e:
        print(f'error with {config["symbol"]} {e}')
        print('config')
        print(config)
        score = -9999999999999.9
        adg = 0.0
        pa_closeness = 100.0
        with open(make_get_filepath('tmp/harmony_search_errors.txt'), 'a') as f:
            f.write(json.dumps([time(), 'error', str(e), denumpyize(config)]) + '\n')
    return (pa_closeness, adg, score)


def backtest_multi_wrap(config: dict, pool):
    tasks = {}
    for s in sorted(config['symbols']):
        tasks[s] = pool.apply_async(backtest_single_wrap, args=({**config, **{'symbol': s}},))
    while True:
        if all([task.ready() for task in tasks.values()]):
            break
        sleep(0.1)
    results = {k: v.get() for k, v in tasks.items()}
    mean_pa_closeness = np.mean([v[0] for v in results.values()])
    mean_adg = np.mean([v[1] for v in results.values()])
    mean_score = np.mean([v[2] for v in results.values()])
    new_score = mean_adg * min(1.0, config['maximum_pa_closeness_mean_long'] / mean_pa_closeness)
    print(f'pa closeness {mean_pa_closeness:.6f} adg {mean_adg:.6f} score {mean_score:8f} new score {new_score:.8f}')
    return -new_score, results


def harmony_search(
        func, 
        bounds: np.ndarray, 
        n_harmonies: int, 
        hm_considering_rate: float, 
        bandwidth: float, 
        pitch_adjusting_rate: float, 
        iters: int,
        starting_xs: [np.ndarray] = [],
        post_processing_func = None):
    # hm == harmony memory
    n_harmonies = max(n_harmonies, len(starting_xs))
    seen = set()
    hm = numpyize([[np.random.uniform(bounds[0][i], bounds[1][i]) for i in range(len(bounds[0]))] for _ in range(n_harmonies)])
    for i in range(len(starting_xs)):
        assert len(starting_xs[i]) == len(bounds[0])
        harmony = np.array(starting_xs[i])
        for z in range(len(bounds[0])):
            harmony[z] = max(bounds[0][z], min(bounds[1][z], harmony[z]))
        tpl = tuple(harmony)
        if tpl not in seen:
            hm[i] = harmony
        seen.add(tpl)
    print('evaluating initial harmonies...')
    hm_evals = numpyize([func(h) for h in hm])

    print('best harmony')
    print(round_values(denumpyize(hm[hm_evals.argmin()]), 5), f'{hm_evals.min():.8f}')
    if post_processing_func is not None:
        post_processing_func(hm[hm_evals.argmin()])
    print('starting search...')
    worst_eval_i = hm_evals.argmax()
    for itr in range(iters):
        new_harmony = np.zeros(len(bounds[0]))
        for note_i in range(len(bounds[0])):
            if np.random.random() < hm_considering_rate:
                new_note = hm[np.random.randint(0, len(hm))][note_i]
                if np.random.random() < pitch_adjusting_rate:
                    new_note = new_note + bandwidth * (np.random.random() - 0.5) * abs(bounds[0][note_i] - bounds[1][note_i])
                    new_note = max(bounds[0][note_i], min(bounds[1][note_i], new_note))
            else:
                new_note = np.random.uniform(bounds[0][note_i], bounds[1][note_i])
            new_harmony[note_i] = new_note
        h_eval = func(new_harmony)
        if h_eval < hm_evals[worst_eval_i]:
            hm[worst_eval_i] = new_harmony
            hm_evals[worst_eval_i] = h_eval
            worst_eval_i = hm_evals.argmax()
            print('improved harmony')
            print(round_values(denumpyize(new_harmony), 5), f'{h_eval:.8f}')
        print('best harmony')
        print(round_values(denumpyize(hm[hm_evals.argmin()]), 5), f'{hm_evals.min():.8f}')
        print('iteration', itr, 'of', iters)
        if post_processing_func is not None:
            post_processing_func(hm[hm_evals.argmin()])
    return hm[hm_evals.argmin()]


class FuncWrap:
    def __init__(self, pool, base_config):
        self.pool = pool
        self.base_config = base_config
        self.xs_conf_map = [k for k in sorted(base_config['ranges'])]
        self.bounds = numpyize([[self.base_config['ranges'][k][0] for k in self.xs_conf_map],
                                [self.base_config['ranges'][k][1] for k in self.xs_conf_map]])
        self.now_date = ts_to_date(time())[:19].replace(':', '-')
        self.results_fname = make_get_filepath(f'tmp/harmony_search_results_{self.now_date}.txt')
        self.best_conf_fname = f'tmp/harmony_search_best_config_{self.now_date}.json'

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
        with open(self.results_fname, 'a') as f:
            f.write(json.dumps({'config': candidate_to_live_config(config), 'results': results}) + '\n')
        return score

    def post_processing_func(self, xs):
        dump_live_config(self.xs_to_config(xs), self.best_conf_fname)


async def main():
    parser = argparse.ArgumentParser(prog='Optimize multi symbol', description='Optimize passivbot config multi symbol')
    parser.add_argument('-o', '--optimize_config', type=str, required=False, dest='optimize_config_path',
                        default='configs/optimize/multi_symbol.hjson', help='optimize config hjson file')
    parser.add_argument('-t', '--start', type=str, required=False, dest='starting_configs',
                        default=None,
                        help='start with given live configs.  single json file or dir with multiple json files')
    parser.add_argument('-i', '--iters', type=int, required=False, dest='iters', default=None, help='n optimize iters')
    parser = add_argparse_args(parser)
    args = parser.parse_args()
    args.symbol = 'BTCUSDT' # dummy symbol
    config = await prepare_optimize_config(args)
    config.update(get_template_live_config())
    config['exchange'], _, _ = load_exchange_key_secret(config['user'])

    # download ticks .npy file if missing
    cache_fname = f"{config['start_date']}_{config['end_date']}_ticks_cache.npy"
    exchange_name = config['exchange'] + ('_spot' if config['market_type'] == 'spot' else '')
    for symbol in sorted(config['symbols']):
        cache_dirpath = f"backtests/{exchange_name}/{symbol}/caches/"
        if not os.path.exists(cache_dirpath + cache_fname) or not os.path.exists(cache_dirpath + 'market_specific_settings.json'):
            print(f'fetching data {symbol}')
            args.symbol = symbol
            tmp_cfg = await prepare_backtest_config(args)
            downloader = Downloader({**config, **tmp_cfg})
            await downloader.get_sampled_ticks()

    pool = Pool(processes=config['n_cpus'])

    func_wrap = FuncWrap(pool, config)
    cfgs = []
    if args.starting_configs is not None:
        if os.path.isdir(args.starting_configs):
            cfgs = []
            for fname in os.listdir(args.starting_configs):
                try:
                    cfgs.append(load_live_config(os.path.join(args.starting_configs, fname)))
                except Exception as e:
                    print('error loading config:', e)
        elif os.path.exists(args.starting_configs):
            try:
                cfgs = [load_live_config(args.starting_configs)]
            except Exception as e:
                print('error loading config:', e)
    starting_xs = [func_wrap.config_to_xs(cfg) for cfg in cfgs]

    n_harmonies = config['n_harmonies']
    hm_considering_rate = config['hm_considering_rate']
    bandwidth = config['bandwidth']
    pitch_adjusting_rate = config['pitch_adjusting_rate']
    iters = config['iters']
    best_harmony = harmony_search(func_wrap.func, func_wrap.bounds, n_harmonies,
                                  hm_considering_rate, bandwidth, pitch_adjusting_rate, iters,
                                  starting_xs=starting_xs,
                                  post_processing_func=func_wrap.post_processing_func)
    best_conf = func_wrap.xs_to_config(best_harmony)
    print('best conf')
    print(best_conf)
    return


if __name__ == '__main__':
    asyncio.run(main())
