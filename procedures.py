import json
import pprint
import os
import hjson
import pandas as pd
import numpy as np
from time import time
from pure_funcs import numpyize, denumpyize, candidate_to_live_config, ts_to_date, get_dummy_settings, calc_spans, \
    config_pretty_str, date_to_ts
from njit_funcs import calc_samples


def load_live_config(live_config_path: str) -> dict:
    try:
        live_config = json.load(open(live_config_path))
        return numpyize(live_config)
    except Exception as e:
        raise Exception(f'failed to load live config {live_config_path} {e}')


def dump_live_config(config: dict, path: str):
    pretty_str = config_pretty_str(candidate_to_live_config(config))
    with open(path, 'w') as f:
        f.write(pretty_str)


async def prep_config(args) -> dict:
    try:
        bc = hjson.load(open(args.backtest_config_path, encoding='utf-8'))
    except Exception as e:
        raise Exception('failed to load backtest config', args.backtest_config_path, e)
    try:
        oc = hjson.load(open(args.optimize_config_path, encoding='utf-8'))
    except Exception as e:
        raise Exception('failed to load optimize config', args.optimize_config_path, e)
    config = {**oc, **bc}
    for key in ['symbol', 'user', 'start_date', 'end_date', 'starting_balance']:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for key in ['spot']:
        if getattr(args, key) == False:
            config[key] = False
        elif getattr(args, key):
            config[key] = True
    config['exchange'], _, _ = load_exchange_key_secret(config['user'])

    if config['exchange'] == 'bybit' and config['symbol'].endswith('USDT'):
        raise Exception('error: bybit linear usdt markets backtesting and optimizing not supported at this time')

    end_date = config['end_date'] if config['end_date'] and config['end_date'] != -1 else ts_to_date(time())[:16]
    config['session_name'] = f"{config['start_date'].replace(' ', '').replace(':', '').replace('.', '')}_" \
                             f"{end_date.replace(' ', '').replace(':', '').replace('.', '')}"

    base_dirpath = os.path.join('backtests',
                                f"{config['exchange']}{'_spot' if 'spot' in config and config['spot'] else ''}",
                                config['symbol'])
    config['caches_dirpath'] = make_get_filepath(os.path.join(base_dirpath, 'caches', ''))
    config['optimize_dirpath'] = make_get_filepath(os.path.join(base_dirpath, 'optimize', ''))
    config['plots_dirpath'] = make_get_filepath(os.path.join(base_dirpath, 'plots', ''))

    config['avg_periodic_gain_key'] = f"avg_{int(round(config['periodic_gain_n_days']))}days_gain"

    mss = config['caches_dirpath'] + 'market_specific_settings.json'
    try:
        print('fetching market_specific_settings...')
        market_specific_settings = await fetch_market_specific_settings(config)
        json.dump(market_specific_settings, open(mss, 'w'), indent=4)
    except Exception as e:
        print('failed to fetch market_specific_settings', e)
        try:
            if os.path.exists(mss):
                market_specific_settings = json.load(open(mss))
            print('using cached market_specific_settings')
        except Exception as e1:
            raise Exception('failed to load cached market_specific_settings')
    config.update(market_specific_settings)

    if 'pbr_limit' in config['ranges']:
        config['ranges']['pbr_limit'][1] = min(config['ranges']['pbr_limit'][1], config['max_leverage'])
        config['ranges']['pbr_limit'][0] = min(config['ranges']['pbr_limit'][0], config['ranges']['pbr_limit'][1])
    if 'spot' in config and config['spot']:
        config['do_long'] = True
        config['do_shrt'] = False

    return config


def make_get_filepath(filepath: str) -> str:
    '''
    if not is path, creates dir and subdirs for path, returns path
    '''
    dirpath = os.path.dirname(filepath) if filepath[-1] != '/' else filepath
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    return filepath


def load_exchange_key_secret(user: str) -> (str, str, str):
    try:
        keyfile = json.load(open('api-keys.json'))
        if user in keyfile:
            return keyfile[user]['exchange'], keyfile[user]['key'], keyfile[user]['secret']
        else:
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


async def fetch_market_specific_settings(config: dict):
    user = config['user']
    exchange = config['exchange']
    symbol = config['symbol']
    tmp_live_settings = get_dummy_settings(user, exchange, symbol)
    settings_from_exchange = {}
    if exchange == 'binance':
        if 'spot' in config and config['spot']:
            bot = await create_binance_bot_spot(tmp_live_settings)
            settings_from_exchange['maker_fee'] = 0.001
            settings_from_exchange['taker_fee'] = 0.001
            settings_from_exchange['spot'] = True
            settings_from_exchange['hedge_mode'] = False
        else:
            bot = await create_binance_bot(tmp_live_settings)
            settings_from_exchange['maker_fee'] = 0.0002
            settings_from_exchange['taker_fee'] = 0.0004
            settings_from_exchange['spot'] = False
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
    elif any(x in bot.market_type for x in ['linear', 'spot']):
        settings_from_exchange['inverse'] = False
    else:
        raise Exception('unknown market type')
    for key in ['max_leverage', 'min_qty', 'min_cost', 'qty_step', 'price_step', 'max_leverage',
                'c_mult', 'hedge_mode']:
        settings_from_exchange[key] = getattr(bot, key)
    return settings_from_exchange


async def create_binance_bot(config: dict):
    from binance import BinanceBot
    bot = BinanceBot(config)
    await bot._init()
    return bot


async def create_binance_bot_spot(config: dict):
    from binance_spot import BinanceBotSpot
    bot = BinanceBotSpot(config)
    await bot._init()
    return bot


async def create_bybit_bot(config: dict):
    from bybit import Bybit
    bot = Bybit(config)
    await bot._init()
    return bot


def add_argparse_args(parser):
    parser.add_argument('--nojit', help='disable numba', action='store_true')
    parser.add_argument('-b', '--backtest_config', type=str, required=False, dest='backtest_config_path',
                        default='configs/backtest/default.hjson', help='backtest config hjson file')
    parser.add_argument('-o', '--optimize_config', type=str, required=False, dest='optimize_config_path',
                        default='configs/optimize/default.hjson', help='optimize config hjson file')
    parser.add_argument('-d', '--download-only', help='download only, do not dump ticks caches', action='store_true')
    parser.add_argument('-s', '--symbol', type=str, required=False, dest='symbol',
                        default=None, help='specify symbol, overriding symbol from backtest config')
    parser.add_argument('-u', '--user', type=str, required=False, dest='user', default=None,
                        help='specify user, a.k.a. account_name, overriding user from backtest config')
    parser.add_argument('--start_date', type=str, required=False, dest='start_date',
                        default=None,
                        help='specify start date, overriding value from backtest config')
    parser.add_argument('--end_date', type=str, required=False, dest='end_date',
                        default=None,
                        help='specify end date, overriding value from backtest config')
    parser.add_argument('--starting_balance', type=float, required=False, dest='starting_balance',
                        default=None,
                        help='specify starting_balance, overriding value from backtest config')
    parser.add_argument('--spot', action='store_true',
                        help='specify whether spot, overriding value from backtest config')

    return parser


def make_tick_samples(config: dict, sec_span: int = 1):

    '''
    makes tick samples from agg_trades
    tick samples are [(qty, price, timestamp)]
    config must include parameters
    - exchange: str
    - symbol: str
    - spot: bool
    - start_date: str
    - end_date: str
    '''
    for key in ['exchange', 'symbol', 'spot', 'start_date', 'end_date']:
        assert key in config
    start_ts = date_to_ts(config['start_date'])
    end_ts = date_to_ts(config['end_date'])
    ticks_filepath = os.path.join('historical_data', config['exchange'], f"agg_trades_{'spot' if config['spot'] else 'futures'}", config['symbol'], '')
    if not os.path.exists(ticks_filepath):
        return
    ticks_filenames = sorted([f for f in os.listdir(ticks_filepath) if f.endswith('.csv')])
    ticks = np.empty((0, 3))
    sts = time()
    for f in ticks_filenames:
        _, _, first_ts, last_ts = map(int, f.replace('.csv', '').split('_'))
        if first_ts > end_ts or last_ts < start_ts:
            continue
        print(f'\rloading chunk {ts_to_date(first_ts / 1000)}', end='  ')
        tdf = pd.read_csv(ticks_filepath + f)
        tdf = tdf[(tdf.timestamp >= start_ts) & (tdf.timestamp <= end_ts)]
        ticks = np.concatenate((ticks, tdf[['qty', 'price', 'timestamp']].values))
        del tdf
    samples = calc_samples(ticks[ticks[:, 2].argsort()], sec_span * 1000)
    print(f'took {time() - sts:.2f} seconds to load {len(ticks)} ticks, creating {len(samples)} samples')
    del ticks
    return samples




