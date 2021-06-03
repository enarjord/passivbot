import sys
import os
from time import time

import numpy as np
import hjson
import json

from pure_funcs import numpyize, denumpyize, candidate_to_live_config, ts_to_date, ticks_to_ticks_cache, \
    get_dummy_settings, calc_spans


def load_live_config(live_config_path: str) -> dict:
    try:
        live_config = json.load(open(live_config_path))
        return numpyize(live_config)
    except Exception as e:
        raise Exception(f'failed to load live config {live_config_path} {e}')


def dump_live_config(config: dict, path: str):
    json.dump(denumpyize(candidate_to_live_config(config)), open(path, 'w'), indent=4)


async def prep_config(args) -> dict:
    try:
        bc = hjson.load(open(args.backtest_config_path))
    except Exception as e:
        raise Exception('failed to load backtest config', args.backtest_config_path, e)
    try:
        oc = hjson.load(open(args.optimize_config_path))
    except Exception as e:
        raise Exception('failed to load optimize config', args.optimize_config_path, e)
    config = {**oc, **bc}
    for key in ['symbol', 'user', 'start_date', 'end_date']:
        if getattr(args, key) != 'none':
            config[key] = getattr(args, key)
    end_date = config['end_date'] if config['end_date'] and config['end_date'] != -1 else ts_to_date(time())[:16]
    config['session_name'] = f"{config['start_date'].replace(' ', '').replace(':', '').replace('.', '')}_" \
                             f"{end_date.replace(' ', '').replace(':', '').replace('.', '')}"

    base_dirpath = os.path.join('backtests', config['exchange'], config['symbol'])
    config['caches_dirpath'] = make_get_filepath(os.path.join(base_dirpath, 'caches', ''))
    config['optimize_dirpath'] = make_get_filepath(os.path.join(base_dirpath, 'optimize', ''))
    config['plots_dirpath'] = make_get_filepath(os.path.join(base_dirpath, 'plots', ''))

    if os.path.exists((mss := config['caches_dirpath'] + 'market_specific_settings.json')):
        market_specific_settings = json.load(open(mss))
    else:
        market_specific_settings = await fetch_market_specific_settings(config['user'], config['exchange'],
                                                                        config['symbol'])
        json.dump(market_specific_settings, open(mss, 'w'), indent=4)
    config.update(market_specific_settings)

    # setting absolute min/max ranges
    for key in ['qty_pct', 'ddown_factor', 'ema_span', 'grid_spacing']:
        if key in config['ranges']:
            config['ranges'][key][0] = max(0.0, config['ranges'][key][0])
    for key in ['qty_pct']:
        if key in config['ranges']:
            config['ranges'][key][1] = min(1.0, config['ranges'][key][1])

    if 'leverage' in config['ranges']:
        config['ranges']['leverage'][1] = min(config['ranges']['leverage'][1], config['max_leverage'])
        config['ranges']['leverage'][0] = min(config['ranges']['leverage'][0], config['ranges']['leverage'][1])
    config['spans'] = calc_spans(config['min_span'], config['max_span'], config['n_spans'])

    return config


def make_get_filepath(filepath: str) -> str:
    '''
    if not is path, creates dir and subdirs for path, returns path
    '''
    dirpath = os.path.dirname(filepath) if filepath[-1] != '/' else filepath
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    return filepath


def load_key_secret(exchange: str, user: str) -> (str, str):
    try:
        keyfile = json.load(open('api-keys.json'))
        # Checks that the user exists, and it is for the correct exchange
        if user in keyfile and keyfile[user]["exchange"] == exchange:

            # If we need to get the `market` key:
            # market = keyfile[user]["market"]
            # print("The Market Type is " + str(market))

            keyList = [str(keyfile[user]["key"]), str(keyfile[user]["secret"])]

            return keyList
        elif user not in keyfile or keyfile[user]["exchange"] != exchange:
            print("Looks like the keys aren't configured yet, or you entered the wrong username!")
        raise Exception('API KeyFile Missing!')
    except FileNotFoundError:
        print("File Not Found!")
        raise Exception('API KeyFile Missing!')


def print_(args, r=False, n=False):
    line = ts_to_date(time())[:19] + '  '
    str_args = '{} ' * len(args)
    line += str_args.format(*args)
    if n:
        print('\n' + line, end=' ')
    elif r:
        print('\r' + line, end=' ')
    else:
        print(line)
    return line


def make_get_ticks_cache(config: dict, ticks: np.ndarray) -> (np.ndarray,):
    cache_dirpath = os.path.join(
        config['caches_dirpath'],
        f"{config['session_name']}_spans_{'_'.join(map(str, config['spans']))}_idx_{config['MA_idx']}", '')
    if os.path.exists(cache_dirpath):
        print('loading cached tick data')
        arrs = []
        for fname in ['prices', 'is_buyer_maker', 'timestamps', 'emas', 'ratios']:
            arrs.append(np.load(f'{cache_dirpath}{fname}.npy'))
        return tuple(arrs)
    else:
        print('dumping cache...')
        fpath = make_get_filepath(cache_dirpath)
        data = ticks_to_ticks_cache(ticks, config['spans'], config['MA_idx'])
        for fname, arr in zip(['prices', 'is_buyer_maker', 'timestamps', 'emas', 'ratios'],
                              data):
            np.save(f'{fpath}{fname}.npy', arr)
        size_mb = np.sum([sys.getsizeof(d) for d in data]) / (1000 * 1000)
        print(f'dumped {size_mb:.2f} mb of data')
    return data


async def fetch_market_specific_settings(user: str, exchange: str, symbol: str):
    tmp_live_settings = get_dummy_settings(user, exchange, symbol)
    settings_from_exchange = {}
    if exchange == 'binance':
        bot = await create_binance_bot(tmp_live_settings)
        settings_from_exchange['maker_fee'] = 0.00018
        settings_from_exchange['taker_fee'] = 0.00036
        settings_from_exchange['exchange'] = 'binance'
    elif exchange == 'bybit':
        bot = await create_bybit_bot(tmp_live_settings)
        settings_from_exchange['maker_fee'] = -0.00025
        settings_from_exchange['taker_fee'] = 0.00075
        settings_from_exchange['exchange'] = 'bybit'
    else:
        raise Exception(f'unknown exchange {exchange}')
    await bot.session.close()
    if 'inverse' in bot.market_type:
        settings_from_exchange['inverse'] = True
    elif 'linear' in bot.market_type:
        settings_from_exchange['inverse'] = False
    else:
        raise Exception('unknown market type')
    for key in ['max_leverage', 'min_qty', 'min_cost', 'qty_step', 'price_step', 'max_leverage',
                'c_mult']:
        settings_from_exchange[key] = getattr(bot, key)
    return settings_from_exchange


async def create_binance_bot(config: dict):
    from binance import BinanceBot
    bot = BinanceBot(config)
    await bot._init()
    return bot


async def create_bybit_bot(config: dict):
    from bybit import Bybit
    bot = Bybit(config)
    await bot._init()
    return bot

