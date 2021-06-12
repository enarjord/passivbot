import numpy as np
import pandas as pd
import datetime
from dateutil import parser
from njit_funcs import round_dynamic, calc_emas


def format_float(num):
    return np.format_float_positional(num, trim='-')


def compress_float(n: float, d: int) -> str:
    if n / 10**d >= 1:
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
    return np.array([int(round(min_span * ((max_span / min_span)**(1 / (n_spans - 1))) ** i))
                     for i in range(0, n_spans)])


def get_xk_keys():
    return ['inverse', 'do_long', 'do_shrt', 'qty_step', 'price_step', 'min_qty', 'min_cost', 'c_mult',
            'max_leverage', 'stop_psize_pct', 'leverage', 'iqty_const', 'iprc_const', 'rqty_const',
            'rprc_const', 'markup_const', 'iqty_MAr_coeffs', 'iprc_MAr_coeffs', 'rprc_PBr_coeffs',
            'rqty_MAr_coeffs', 'rprc_MAr_coeffs', 'markup_MAr_coeffs']


def create_xk(config: dict) -> dict:
    xk = {}
    for k in get_xk_keys():
        if k in config['long']:
            xk[k] = (config['long'][k], config['shrt'][k])
        elif k in config:
            xk[k] = config[k]
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


def ts_to_date(timestamp: float) -> str:
    return str(datetime.datetime.fromtimestamp(timestamp)).replace(' ', 'T')


def date_to_ts(d):
    return int(parser.parse(d).replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)


def candidate_to_live_config(candidate: dict) -> dict:
    packed = pack_config(candidate)
    live_config = get_template_live_config(min_span=candidate['min_span'],
                                           max_span=candidate['max_span'],
                                           n_spans=candidate['n_spans'])
    sides = ['long', 'shrt']
    for side in sides:
        for k in live_config[side]:
            if k in packed[side]:
                live_config[side][k] = packed[side][k]
    for k in live_config:
        if k not in sides and k in packed:
            live_config[k] = packed[k]
    live_config['spans'] = calc_spans(live_config['min_span'], live_config['max_span'], live_config['n_spans'])
    name = f"{packed['symbol'].lower()}"
    if 'start_date' in candidate:
        n_days = round((date_to_ts(candidate['end_date']) -
                        date_to_ts(candidate['start_date'])) / (1000 * 60 * 60 * 24), 1)
        name += f"_{n_days}_days"
    if 'average_daily_gain' in candidate:
        name += f"_adg{(candidate['average_daily_gain'] - 1) * 100:.2f}"
    elif 'daily_gain' in candidate:
        name += f"_adg{(candidate['daily_gain'] - 1) * 100:.2f}%"
    live_config['config_name'] = name
    return live_config


def unpack_config(d):
    new = {}
    for k, v in flatten_dict(d, sep='§').items():
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
        if '§' in k:
            k0, k1 = k.split('§')
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


def get_dummy_settings(user: str, exchange: str, symbol: str):
    dummy_settings = get_template_live_config()
    dummy_settings.update({k: 0.01 for k in get_xk_keys() + ['stop_loss_liq_diff', 'ema_span']})
    dummy_settings.update({'user': user, 'exchange': exchange, 'symbol': symbol,
                           'config_name': '', 'logging_level': 0})
    return dummy_settings


def ticks_to_ticks_cache(ticks: np.ndarray, spans: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    emas = calc_emas(ticks[:,0], spans)
    prices = ticks[:,0]#.astype(np.float32)
    is_buyer_maker = ticks[:,1].astype(np.int8)
    timestamps = ticks[:,2].astype(np.int64)

    return (prices[max(spans):], is_buyer_maker[max(spans):], timestamps[max(spans):],
            emas[max(spans):])#.astype(np.float32))


def flatten(lst: list) -> list:
    return [y for x in lst for y in x]



def get_template_live_config(min_span=6000, max_span=300000, n_spans=3, randomize_coeffs=False):
    config = {
        "config_name": "name",
        "logging_level": 0,
        "min_span": min_span,
        "max_span": max_span,
        "n_spans": n_spans,
        "long": {
            "enabled":            True,
            "stop_psize_pct":     0.05,   # % of psize for stop loss order
            "leverage":           3.0,    # max pcost = balance * leverage
            "iqty_const":         0.01,   # initial entry qty pct
            "iprc_const":         0.991,  # initial entry price ema_spread
            "rqty_const":         1.0,    # reentry qty ddown factor
            "rprc_const":         0.98,   # reentry price grid spacing
            "markup_const":       1.003,  # markup

                                          # coeffs: [[quadratic_coeff, linear_coeff]] * n_spans
                                          # e.g. n_spans = 3,
                                          # coeffs = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
                                          # all coeff ranges [min, max] = [-10.0, 10.0]
            "iqty_MAr_coeffs":    [],     # initial qty pct Moving Average ratio coeffs    formerly qty_pct
            "iprc_MAr_coeffs":    [],     # initial price pct Moving Average ratio coeffs  formerly ema_spread
            "rqty_MAr_coeffs":    [],     # reentry qty pct Moving Average ratio coeffs    formerly ddown_factor
            "rprc_MAr_coeffs":    [],     # reentry price pct Moving Average ratio coeffs  formerly grid_spacing
            "rprc_PBr_coeffs":    [],     # reentry Position cost to Balance ratio coeffs (PBr**2, PBr)
                                          # formerly pos_margin_grid_coeff
            "markup_MAr_coeffs":  [],     # markup price pct Moving Average ratio coeffs
        },
        "shrt": {
            "enabled":            True,
            "stop_psize_pct":     0.05,   # % of psize for stop loss order
            "leverage":           3.0,    # max pcost = balance * leverage
            "iqty_const":         0.01,   # initial entry qty pct
            "iprc_const":         1.009,  # initial entry price ema_spread
            "rqty_const":         1.0,    # reentry qty ddown factor
            "rprc_const":         1.02,   # reentry price grid spacing
            "markup_const":       0.997,  # markup
                                          # coeffs: [[quadratic_coeff, linear_coeff]] * n_spans
                                          # e.g. n_spans = 3,
                                          # coeffs = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
                                          # all coeff ranges [min, max] = [-10.0, 10.0]
            "iqty_MAr_coeffs":    [],     # initial qty pct Moving Average ratio coeffs    formerly qty_pct
            "iprc_MAr_coeffs":    [],     # initial price pct Moving Average ratio coeffs  formerly ema_spread
            "rqty_MAr_coeffs":    [],     # reentry qty pct Moving Average ratio coeffs    formerly ddown_factor
            "rprc_MAr_coeffs":    [],     # reentry price pct Moving Average ratio coeffs  formerly grid_spacing
            "rprc_PBr_coeffs":    [],     # reentry Position cost to Balance ratio coeffs (PBr**2, PBr)
                                          # formerly pos_margin_grid_coeff
            "markup_MAr_coeffs":  [],     # markup price pct Moving Average ratio coeffs
        }
    }
    for side in ['long', 'shrt']:
        for k in config[side]:
            if 'MAr' in k:
                config[side][k] = np.random.random((config['n_spans'], 2)) * 0.1 - 0.05 \
                    if randomize_coeffs else np.zeros((config['n_spans'], 2))
            elif 'PBr_coeff' in k:
                config[side][k] = np.random.random((1, 2)) * 0.1 - 0.05 \
                    if randomize_coeffs else  np.zeros((1, 2))
    config['spans'] = calc_spans(config['min_span'], config['max_span'], config['n_spans'])
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
        idxs = get_ids_to_fetch(spans[i:i+1], prev_last_id, samples_per_span)
        all_idxs.append(idxs)
        samples_leftover = max_n_samples - sum(map(len, all_idxs))
        samples_per_span = samples_leftover // max(1, len(spans) - i - 1)
        prev_last_id = idxs[-1] + 1000
    return np.array(flatten(all_idxs))[::-1]


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
