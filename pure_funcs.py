import numpy as np
import datetime
from dateutil import parser
from njit_funcs import round_dynamic, calc_emas, calc_ratios


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
    return np.array([1] + [int(round(min_span * ((max_span / min_span)**(1 / (n_spans - 1))) ** i))
                           for i in range(0, n_spans)])


def fill_template_config(c, r=False):
    for side in ['long', 'shrt']:
        for k in c[side]:
            if 'MAr' in k:
                c[side][k] = np.random.random((c['n_spans'], 2)) * 0.1 - 0.05 if r else np.zeros((c['n_spans'], 2))
            elif 'PBr_coeff' in k:
                c[side][k] = np.random.random((1, 2)) * 0.1 - 0.05 if r else  np.zeros((1, 2))
    c['spans'] = calc_spans(c['min_span'], c['max_span'], c['n_spans'])
    return c


def get_keys():
    return ['inverse', 'do_long', 'do_shrt', 'qty_step', 'price_step', 'min_qty', 'min_cost',
            'c_mult', 'leverage', 'hedge_bkr_diff_thr', 'hedge_psize_pct', 'stop_bkr_diff_thr',
            'stop_psize_pct', 'stop_eqbal_ratio_thr', 'entry_bkr_diff_thr', 'iqty_const', 'iprc_const', 'rqty_const',
            'rprc_const', 'markup_const', 'iqty_MAr_coeffs', 'rprc_PBr_coeffs', 'iprc_MAr_coeffs',
            'rqty_MAr_coeffs', 'rprc_MAr_coeffs', 'markup_MAr_coeffs', 'stop_PBr_thr']

def create_xk(config: dict) -> dict:
    xk = {}
    for k in get_keys():
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
    else:
        return x


def ts_to_date(timestamp: float) -> str:
    return str(datetime.datetime.fromtimestamp(timestamp)).replace(' ', 'T')


def date_to_ts(d):
    return int(parser.parse(d).replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)


def candidate_to_live_config(candidate: dict) -> dict:
    packed = pack_config(candidate)
    live_config = get_template_live_config()
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
    dummy_settings.update({k: 0.01 for k in get_keys() + ['stop_loss_liq_diff', 'ema_span']})
    dummy_settings.update({'user': user, 'exchange': exchange, 'symbol': symbol,
                           'config_name': '', 'logging_level': 0})
    return dummy_settings


def ticks_to_ticks_cache(ticks: np.ndarray, spans: np.ndarray, MA_idx: int) -> (np.ndarray,):
    emas = calc_emas(ticks[:,0], spans)
    ratios = calc_ratios(emas)
    prices = ticks[:,0].astype(np.float64)
    is_buyer_maker = ticks[:,1].astype(np.int8)
    timestamps = ticks[:,2].astype(np.float64)
    return (prices[max(spans):], is_buyer_maker[max(spans):], timestamps[max(spans):],
            emas[max(spans):][:, MA_idx].astype(np.float64), ratios[max(spans):].astype(np.float64))


def flatten(lst: list) -> list:
    return [y for x in lst for y in x]


def get_template_live_config(n_spans=3):
    return {
        "config_name": "name",
        "logging_level": 0,
        "min_span": 6000,
        "max_span": 300000,
        "n_spans": n_spans,
        "MA_idx":             1,      # index of ema span from which to calc initial entry prices
        "stop_psize_pct":     0.05,   # % of psize for stop loss order
        "long": {
            "enabled":            True,
            "leverage":           10,     # borrow cap
            "stop_PBr_thr":       1.0,    # partially close pos at a loss if long PBr > thr
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
            "leverage":           10,     # borrow cap
            "stop_PBr_thr":       1.0,    # partially close pos at a loss if shrt PBr > thr
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


def get_bid_ask_thresholds(data, config):
    pass








