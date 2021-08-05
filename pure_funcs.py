import datetime

import numpy as np
import pandas as pd
import pprint
from dateutil import parser

from njit_funcs import round_dynamic, calc_emas


def format_float(num):
    return np.format_float_positional(num, trim='-')


def compress_float(n: float, d: int) -> str:
    if n / 10 ** d >= 1:
        n = round(n)
    else:
        n = round_dynamic(n, d)
    nstr = format_float(n)
    if nstr.startswith('0.'):
        nstr = nstr[1:]
    elif nstr.startswith('-0.'):
        nstr = '-' + nstr[2:]
    elif nstr.endswith('.0'):
        nstr = nstr[:-2]
    return nstr


def calc_spans(min_span: int, max_span: int, n_spans: int) -> np.ndarray:
    return np.array([min_span * ((max_span / min_span) ** (1 / (n_spans - 1))) ** i for i in range(0, n_spans)])


def get_xk_keys():
    return ['spot', 'hedge_mode', 'inverse', 'do_long', 'do_shrt', 'qty_step', 'price_step', 'min_qty', 'min_cost', 'c_mult',
            'max_leverage', 'spans', 'pbr_stop_loss', 'pbr_limit', 'iqty_const', 'iprc_const', 'rqty_const',
            'rprc_const', 'markup_const', 'iqty_MAr_coeffs', 'iprc_MAr_coeffs', 'rprc_PBr_coeffs',
            'rqty_MAr_coeffs', 'rprc_MAr_coeffs', 'markup_MAr_coeffs']


def create_xk(config: dict) -> dict:
    xk = {}
    config_ = config.copy()
    if 'spot' in config_['market_type']:
        config_ = spotify_config(config_)
    else:
        config_['spot'] = False
        config_['do_long'] = config['long']['enabled']
        config_['do_shrt'] = config['shrt']['enabled']
    config_['spans'] = calc_spans(config['min_span'], config['max_span'], config['n_spans'])
    for k in get_xk_keys():
        if k in config_['long']:
            xk[k] = (config_['long'][k], config_['shrt'][k])
        elif k in config_:
            xk[k] = config_[k]
        else:
            raise Exception('failed to create xk', k)
    return xk


def numpyize(x):
    if type(x) in [list, tuple]:
        return np.array([numpyize(e) for e in x])
    elif type(x) == dict:
        numpyd = {}
        for k, v in x.items():
            numpyd[k] = numpyize(v)
        return numpyd
    else:
        return x


def denumpyize(x):
    if type(x) in [np.float64, np.float32, np.float16]:
        return float(x)
    elif type(x) in [np.int64, np.int32, np.int16, np.int8]:
        return int(x)
    elif type(x) == np.ndarray:
        return [denumpyize(e) for e in x]
    elif type(x) == np.bool_:
        return bool(x)
    elif type(x) == dict:
        denumpyd = {}
        for k, v in x.items():
            denumpyd[k] = denumpyize(v)
        return denumpyd
    elif type(x) == list:
        return [denumpyize(z) for z in x]
    elif type(x) == tuple:
        return tuple([denumpyize(z) for z in x])
    else:
        return x


def denanify(x, nan=0.0, posinf=0.0, neginf=0.0):
    try:
        assert type(x) != str
        _ = float(x)
        return np.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    except:
        if type(x) == list:
            return [denanify(e) for e in x]
        elif type(x) == tuple:
            return tuple(denanify(e) for e in x)
        elif type(x) == np.ndarray:
            return np.array([denanify(e) for e in x], dtype=x.dtype)
        elif type(x) == dict:
            denanified = {}
            for k, v in x.items():
                denanified[k] = denanify(v)
            return denanified
        else:
            return x


def ts_to_date(timestamp: float) -> str:
    return str(datetime.datetime.fromtimestamp(timestamp)).replace(' ', 'T')


def date_to_ts(d):
    return int(parser.parse(d).replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)


def config_pretty_str(config: dict):
    pretty_str = pprint.pformat(config)
    for r in [("'", '"'), ('True', 'true'), ('False', 'false')]:
        pretty_str = pretty_str.replace(*r)
    return pretty_str


def candidate_to_live_config(candidate: dict) -> dict:
    packed = pack_config(candidate)
    live_config = get_template_live_config(n_spans=candidate['n_spans'])
    sides = ['long', 'shrt']
    for side in sides:
        for k in live_config[side]:
            if k in packed[side]:
                live_config[side][k] = packed[side][k]
    for k in live_config:
        if k not in sides and k in packed:
            live_config[k] = packed[k]

    result_dict = candidate['result'] if 'result' in candidate else candidate
    name = f"{result_dict['exchange'].lower()}_" if 'exchange' in result_dict else 'unknown_'
    name += f"{result_dict['symbol'].lower()}" if 'symbol' in result_dict else 'unknown'
    if 'n_days' in result_dict:
        n_days = result_dict['n_days']
    elif 'start_date' in result_dict:
        n_days = (date_to_ts(result_dict['end_date']) -
                  date_to_ts(result_dict['start_date'])) / (1000 * 60 * 60 * 24)
    else:
        n_days = 0
    name += f"_{n_days:.0f}days"
    if 'average_daily_gain' in result_dict:
        name += f"_adg{(result_dict['average_daily_gain'] - 1) * 100:.2f}%"
    elif 'daily_gain' in result_dict:
        name += f"_adg{(result_dict['daily_gain'] - 1) * 100:.2f}%"
    if 'closest_bkr' in result_dict:
        name += f"_bkr{(result_dict['closest_bkr']) * 100:.2f}%"
    live_config['config_name'] = name
    return denumpyize(live_config)


def unpack_config(d):
    new = {}
    for k, v in flatten_dict(d, sep='£').items():
        try:
            assert type(v) != str
            for _ in v:
                break
            for i in range(len(v)):
                new[f'{k}${str(i).zfill(2)}'] = v[i]
        except:
            new[k] = v
    if new == d:
        return new
    return unpack_config(new)


def pack_config(d):
    result = {}
    while any('$' in k for k in d):
        new = {}
        for k, v in denumpyize(d).items():
            if '$' in k:
                ks = k.split('$')
                k0 = '$'.join(ks[:-1])
                if k0 in new:
                    new[k0].append(v)
                else:
                    new[k0] = [v]
            else:
                new[k] = v
        d = new
    new = {}
    for k, v in d.items():
        if type(v) == list:
            new[k] = np.array(v)
        else:
            new[k] = v
    d = new

    new = {}
    for k, v in d.items():
        if '£' in k:
            k0, k1 = k.split('£')
            if k0 in new:
                new[k0][k1] = v
            else:
                new[k0] = {k1: v}
        else:
            new[k] = v
    return new


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if type(v) == dict:
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def sort_dict_keys(d):
    if type(d) == list:
        return [sort_dict_keys(e) for e in d]
    if type(d) != dict:
        return d
    return {key: sort_dict_keys(d[key]) for key in sorted(d)}


def filter_orders(actual_orders: [dict],
                  ideal_orders: [dict],
                  keys: [str] = ('symbol', 'side', 'qty', 'price')) -> ([dict], [dict]):
    # returns (orders_to_delete, orders_to_create)

    if not actual_orders:
        return [], ideal_orders
    if not ideal_orders:
        return actual_orders, []
    actual_orders = actual_orders.copy()
    orders_to_create = []
    ideal_orders_cropped = [{k: o[k] for k in keys} for o in ideal_orders]
    actual_orders_cropped = [{k: o[k] for k in keys} for o in actual_orders]
    for ioc, io in zip(ideal_orders_cropped, ideal_orders):
        matches = [(aoc, ao) for aoc, ao in zip(actual_orders_cropped, actual_orders) if aoc == ioc]
        if matches:
            actual_orders.remove(matches[0][1])
            actual_orders_cropped.remove(matches[0][0])
        else:
            orders_to_create.append(io)
    return actual_orders, orders_to_create


def get_dummy_settings(config: dict):
    dummy_settings = get_template_live_config(n_spans=3)
    dummy_settings.update({k: 1.0 for k in get_xk_keys() + ['stop_loss_liq_diff', 'ema_span']})
    dummy_settings.update({'user': config['user'], 'exchange': config['exchange'], 'symbol': config['symbol'],
                           'config_name': '', 'logging_level': 0, 'spans': np.array([6000, 90000])})
    return {**config, **dummy_settings}


def flatten(lst: list) -> list:
    return [y for x in lst for y in x]


def get_template_live_config(n_spans: int, randomize_coeffs=False):
    config = {
        "config_name": "name",
        "logging_level": 0,
        "min_span": 10.0,  # minutes
        "max_span": 420.0,  # minutes
        "n_spans": n_spans,
        "long": {
            "enabled": True,
            "pbr_stop_loss": 0.05,  # % of psize for stop loss order
            "pbr_limit": 3.0,  # max pcost = balance * pbr_limit
            "iqty_const": 0.01,  # initial entry qty pct
            "iprc_const": 0.991,  # initial entry price ema_spread
            "rqty_const": 1.0,  # reentry qty ddown factor
            "rprc_const": 0.98,  # reentry price grid spacing
            "markup_const": 1.003,  # markup

            # coeffs: [[quadratic_coeff, linear_coeff]] * n_spans
            # e.g. n_spans = 3,
            # coeffs = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            # all coeff ranges [min, max] = [-10.0, 10.0]
            "iqty_MAr_coeffs": [],  # initial qty pct Moving Average ratio coeffs    formerly qty_pct
            "iprc_MAr_coeffs": [],  # initial price pct Moving Average ratio coeffs  formerly ema_spread
            "rqty_MAr_coeffs": [],  # reentry qty pct Moving Average ratio coeffs    formerly ddown_factor
            "rprc_MAr_coeffs": [],  # reentry price pct Moving Average ratio coeffs  formerly grid_spacing
            "rprc_PBr_coeffs": [],  # reentry Position cost to Balance ratio coeffs (PBr**2, PBr)
            # formerly pos_margin_grid_coeff
            "markup_MAr_coeffs": [],  # markup price pct Moving Average ratio coeffs
        },
        "shrt": {
            "enabled": True,
            "pbr_stop_loss": 0.05,  # % of psize for stop loss order
            "pbr_limit": 3.0,  # max pcost = balance * pbr_limit
            "iqty_const": 0.01,  # initial entry qty pct
            "iprc_const": 1.009,  # initial entry price ema_spread
            "rqty_const": 1.0,  # reentry qty ddown factor
            "rprc_const": 1.02,  # reentry price grid spacing
            "markup_const": 0.997,  # markup
            # coeffs: [[quadratic_coeff, linear_coeff]] * n_spans
            # e.g. n_spans = 3,
            # coeffs = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            # all coeff ranges [min, max] = [-10.0, 10.0]
            "iqty_MAr_coeffs": [],  # initial qty pct Moving Average ratio coeffs    formerly qty_pct
            "iprc_MAr_coeffs": [],  # initial price pct Moving Average ratio coeffs  formerly ema_spread
            "rqty_MAr_coeffs": [],  # reentry qty pct Moving Average ratio coeffs    formerly ddown_factor
            "rprc_MAr_coeffs": [],  # reentry price pct Moving Average ratio coeffs  formerly grid_spacing
            "rprc_PBr_coeffs": [],  # reentry Position cost to Balance ratio coeffs (PBr**2, PBr)
            # formerly pos_margin_grid_coeff
            "markup_MAr_coeffs": [],  # markup price pct Moving Average ratio coeffs
        }
    }
    for side in ['long', 'shrt']:
        for k in config[side]:
            if 'MAr' in k:
                config[side][k] = np.random.random((config['n_spans'], 2)) * 0.1 - 0.05 \
                    if randomize_coeffs else np.zeros((config['n_spans'], 2))
            elif 'PBr_coeff' in k:
                config[side][k] = np.random.random((1, 2)) * 0.1 - 0.05 \
                    if randomize_coeffs else np.zeros((1, 2))
    return config


def get_ids_to_fetch(spans: [int], last_id: int, max_n_samples: int = 60, ticks_per_fetch: int = 1000):
    max_span = max(spans)
    n_samples = int(round((max_span - ticks_per_fetch * 2) / ticks_per_fetch))
    first_fetch_id = last_id - ticks_per_fetch * 2
    if n_samples < max_n_samples:
        return np.arange(first_fetch_id, last_id - max_span - ticks_per_fetch, -ticks_per_fetch)
    if len(spans) == 1:
        return np.linspace(first_fetch_id, last_id - spans[0], max_n_samples).round().astype(int)
    samples_per_span = max_n_samples // len(spans)
    all_idxs = []
    prev_last_id = last_id
    for i in range(len(spans)):
        idxs = get_ids_to_fetch(spans[i:i + 1], prev_last_id, samples_per_span)
        all_idxs.append(idxs)
        samples_leftover = max_n_samples - sum(map(len, all_idxs))
        samples_per_span = samples_leftover // max(1, len(spans) - i - 1)
        if len(idxs) > 0:
            prev_last_id = idxs[-1] + 1000
    idxs = np.array(flatten(all_idxs))[::-1]
    return np.unique(idxs[idxs > 0])


def calc_indicators_from_ticks_with_gaps(spans, ticks_with_gaps):
    df = pd.DataFrame(ticks_with_gaps).set_index('trade_id').sort_index()
    df = df.reindex(np.arange(df.index[0], df.index[-1])).interpolate(method='linear')
    df = df.groupby(
        (~((df.price == df.price.shift(1)) & (df.is_buyer_maker == df.is_buyer_maker.shift(1)))).cumsum()).agg(
        {'price': 'first', 'is_buyer_maker': 'first'})
    emas = calc_emas(df.price.values, np.array(spans))[-1]
    return emas


def drop_consecutive_same_prices(ticks: [dict]) -> [dict]:
    compressed = [ticks[0]]
    for i in range(1, len(ticks)):
        if ticks[i]['price'] != compressed[-1]['price'] or \
                ticks[i]['is_buyer_maker'] != compressed[-1]['is_buyer_maker']:
            compressed.append(ticks[i])
    return compressed


import numpy as np
import pandas as pd


def get_empty_analysis(bc: dict) -> dict:
    return {
        'exchange': bc['exchange'] if 'exchange' in bc else 'unknown',
        'symbol': bc['symbol'] if 'symbol' in bc else 'unknown',
        'net_pnl_plus_fees': 0.0,
        'profit_sum': 0.0,
        'loss_sum': 0.0,
        'fee_sum': 0.0,
        'final_equity': bc['starting_balance'],
        'gain': 1.0,
        'max_drawdown': 0.0,
        'n_days': 0.0,
        'average_periodic_gain': 0.0,
        'average_daily_gain': 0.0,
        'adjusted_daily_gain': 0.0,
        'sharpe_ratio': 0.0,
        'lowest_eqbal_ratio': 0.0,
        'closest_bkr': 1.0,
        'n_fills': 0.0,
        'n_entries': 0.0,
        'n_closes': 0.0,
        'n_reentries': 0.0,
        'n_initial_entries': 0.0,
        'n_normal_closes': 0.0,
        'n_stop_loss_closes': 0.0,
        'n_stop_loss_entries': 0.0,
        'biggest_psize': 0.0,
        'max_hrs_no_fills_same_side': 1000.0,
        'max_hrs_no_fills': 1000.0,
        'mean_hrs_between_fills': 1000.0,
    }


def analyze_fills(fills: list, bc: dict, first_ts: float, last_ts: float) -> (pd.DataFrame, dict):
    fdf = pd.DataFrame(fills)

    if fdf.empty:
        return fdf, get_empty_analysis(bc)
    fdf.columns = ['trade_id', 'timestamp', 'pnl', 'fee_paid', 'balance', 'equity', 'pbr', 'qty', 'price', 'psize', 'pprice', 'type']
    adgs = (fdf.equity / bc['starting_balance']) ** (1 / ((fdf.timestamp - first_ts) / (1000 * 60 * 60 * 24)))
    fdf = fdf.join(adgs.rename('adg')).set_index('trade_id')

    longs = fdf[fdf.type.str.contains('long')]
    shrts = fdf[fdf.type.str.contains('shrt')]

    if bc['do_long']:
        if len(longs) > 0:
            long_fill_ts_diffs = np.diff([first_ts] + list(longs.timestamp) + [last_ts]) / (1000 * 60 * 60)
            long_stuck_mean = np.mean(long_fill_ts_diffs)
            long_stuck = np.max(long_fill_ts_diffs)
        else:
            long_stuck_mean = 1000.0
            long_stuck = 1000.0
    else:
        long_stuck_mean = 0.0
        long_stuck = 0.0
    if bc['do_shrt']:
        if len(shrts) > 0:
            shrt_fill_ts_diffs = np.diff([first_ts] + list(shrts.timestamp) + [last_ts]) / (1000 * 60 * 60)
            shrt_stuck_mean = np.mean(shrt_fill_ts_diffs)
            shrt_stuck = np.max(shrt_fill_ts_diffs)
        else:
            shrt_stuck_mean = 1000.0
            shrt_stuck = 1000.0
    else:
        shrt_stuck_mean = 0.0
        shrt_stuck = 0.0

    ms_span = 1000 * 60 * 60 * 24 * bc['periodic_gain_n_days']
    buckets = fdf.timestamp // ms_span * ms_span
    buckets = buckets + (fdf.timestamp.iloc[0] - buckets.iloc[0])
    groups = fdf.groupby(buckets)
    periodic_gains = groups.balance.last() / groups.balance.first() - 1  # realized profits
    periodic_gains = periodic_gains.reindex(np.arange(periodic_gains.index[0], periodic_gains.index[-1], ms_span)).fillna(0.0)
    periodic_gains_mean = np.nan_to_num(periodic_gains.mean())
    periodic_gains_std = periodic_gains.std()
    sharpe_ratio = periodic_gains_mean / periodic_gains_std if periodic_gains_std != 0.0 else -20.0
    sharpe_ratio = np.nan_to_num(sharpe_ratio)
    result = {
        'exchange': bc['exchange'] if 'exchange' in bc else 'unknown',
        'symbol': bc['symbol'] if 'symbol' in bc else 'unknown',
        'starting_balance': bc['starting_balance'],
        'final_balance': fdf.iloc[-1].balance,
        'final_equity': fdf.iloc[-1].equity,
        'net_pnl_plus_fees': fdf.pnl.sum() + fdf.fee_paid.sum(),
        'gain': (gain := fdf.iloc[-1].equity / bc['starting_balance']),
        'n_days': (n_days := (last_ts - first_ts) / (1000 * 60 * 60 * 24)),
        'average_daily_gain': (adg := gain ** (1 / n_days) if gain > 0.0 and n_days > 0.0 else 0.0),
        'average_periodic_gain': periodic_gains_mean,
        'adjusted_daily_gain': np.tanh(10 * (adg - 1)) + 1,
        'sharpe_ratio': sharpe_ratio,
        'profit_sum': fdf[fdf.pnl > 0.0].pnl.sum(),
        'loss_sum': fdf[fdf.pnl < 0.0].pnl.sum(),
        'fee_sum': fdf.fee_paid.sum(),
        'lowest_eqbal_ratio': bc['lowest_eqbal_ratio'],
        'closest_bkr': bc['closest_bkr'],
        'n_fills': len(fdf),
        'n_entries': len(fdf[fdf.type.str.contains('entry')]),
        'n_closes': len(fdf[fdf.type.str.contains('close')]),
        'n_reentries': len(fdf[fdf.type.str.contains('rentry')]),
        'n_initial_entries': len(fdf[fdf.type.str.contains('ientry')]),
        'n_normal_closes': len(fdf[fdf.type.str.contains('nclose')]),
        'n_stop_loss_closes': len(fdf[fdf.type.str.contains('sclose')]),
        'biggest_psize': fdf.psize.abs().max(),
        'mean_hrs_between_fills': np.mean(np.diff([first_ts] + list(fdf.timestamp) + [last_ts])) / (1000 * 60 * 60),
        'mean_hrs_between_fills_long': long_stuck_mean,
        'mean_hrs_between_fills_shrt': shrt_stuck_mean,
        'max_hrs_no_fills_long': long_stuck,
        'max_hrs_no_fills_shrt': shrt_stuck,
        'max_hrs_no_fills_same_side': max(long_stuck, shrt_stuck),
        'max_hrs_no_fills': np.max(np.diff([first_ts] + list(fdf.timestamp) + [last_ts])) / (1000 * 60 * 60),
    }
    return fdf, result


def calc_pprice_from_fills(coin_balance, fills, n_fills_limit=100):
    # assumes fills are sorted old to new
    if coin_balance == 0.0 or len(fills) == 0:
        return 0.0
    relevant_fills = []
    qty_sum = 0.0
    for fill in fills[::-1][:n_fills_limit]:
        abs_qty = fill['qty']
        if fill['side'] == 'buy':
            adjusted_qty = min(abs_qty, coin_balance - qty_sum)
            qty_sum += adjusted_qty
            relevant_fills.append({**fill, **{'qty': adjusted_qty}})
            if qty_sum >= coin_balance * 0.999:
                break
        else:
            qty_sum -= abs_qty
            relevant_fills.append(fill)
    psize, pprice = 0.0, 0.0
    for fill in relevant_fills[::-1]:
        abs_qty = abs(fill['qty'])
        if fill['side'] == 'buy':
            new_psize = psize + abs_qty
            pprice = pprice * (psize / new_psize) + fill['price'] * (abs_qty / new_psize)
            psize = new_psize
        else:
            psize -= abs_qty
    return pprice


def get_position_fills(long_psize: float, shrt_psize: float, fills: [dict]) -> ([dict], [dict]):
    '''
    assumes fills are sorted old to new
    returns fills since and including initial entry
    '''
    long_psize *= 0.999
    shrt_psize *= 0.999
    long_qty_sum = 0.0
    shrt_qty_sum = 0.0
    long_done, shrt_done = long_psize == 0.0, shrt_psize == 0.0
    if long_done and shrt_done:
        return [], []
    long_pfills, shrt_pfills = [], []
    for x in fills[::-1]:
        if x['position_side'] == 'long':
            if not long_done:
                long_qty_sum += x['qty'] * (1.0 if x['side'] == 'buy' else -1.0)
                long_pfills.append(x)
                long_done = long_qty_sum >= long_psize
        elif x['position_side'] == 'shrt':
            if not shrt_done:
                shrt_qty_sum += x['qty'] * (1.0 if x['side'] == 'sell' else -1.0)
                shrt_pfills.append(x)
                shrt_done = shrt_qty_sum >= shrt_psize
    return long_pfills[::-1], shrt_pfills[::-1]


def calc_long_pprice(long_psize, long_pfills):
    '''
    assumes long pfills are sorted old to new
    '''
    psize, pprice = 0.0, 0.0
    for fill in long_pfills:
        abs_qty = abs(fill['qty'])
        if fill['side'] == 'buy':
            new_psize = psize + abs_qty
            pprice = pprice * (psize / new_psize) + fill['price'] * (abs_qty / new_psize)
            psize = new_psize
        else:
            psize -= abs_qty
    return pprice


def nullify(x):
    if type(x) in [list, tuple]:
        return [nullify(x1) for x1 in x]
    elif type(x) == np.ndarray:
        return numpyize([nullify(x1) for x1 in x])
    elif type(x) == dict:
        return {k: nullify(x[k]) for k in x}
    elif type(x) in [bool, np.bool_]:
        return x
    else:
        return 0.0


def spotify_config(config: dict, nullify_shrt=True) -> dict:
    spotified = config.copy()

    spotified['spot'] = True
    if 'market_type' not in spotified:
        spotified['market_type'] = 'spot'
    elif 'spot' not in spotified['market_type']:
        spotified['market_type'] += '_spot'
    spotified['do_long'] = spotified['long']['enabled'] = True
    spotified['do_shrt'] = spotified['shrt']['enabled'] = False
    spotified['long']['pbr_stop_loss'] = min(1.0, spotified['long']['pbr_stop_loss'])
    if spotified['long']['pbr_stop_loss'] <= 0.0:
        spotified['long']['pbr_limit'] = min(1.0, spotified['long']['pbr_limit'])
    else:
        spotified['long']['pbr_limit'] = max(0.0, min(spotified['long']['pbr_limit'],
                                                      1.0 - spotified['long']['pbr_stop_loss']))
    if nullify_shrt:
        spotified['shrt'] = nullify(spotified['shrt'])
    return spotified






