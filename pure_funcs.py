import sys
import numpy as np


if '--nojit' in sys.argv:
    print('not using numba')
    def njit(pyfunc=None, **kwargs):
        def wrap(func):
            return func

        if pyfunc is not None:
            return wrap(pyfunc)
        else:
            return wrap
else:
    print('using numba')
    from numba import njit


@njit
def round_dynamic(n: float, d: int):
    if n == 0.0:
        return n
    return round(n, d - int(np.floor(np.log10(abs(n)))) - 1)


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


@njit
def round_up(n, step, safety_rounding=10) -> float:
    return np.round(np.ceil(np.round(n / step, safety_rounding)) * step, safety_rounding)


@njit
def round_dn(n, step, safety_rounding=10) -> float:
    return np.round(np.floor(np.round(n / step, safety_rounding)) * step, safety_rounding)


@njit
def round_(n, step, safety_rounding=10) -> float:
    return np.round(np.round(n / step) * step, safety_rounding)


@njit
def calc_diff(x, y):
    return abs(x - y) / abs(y)


@njit
def nan_to_0(x) -> float:
    return x if x == x else 0.0


@njit
def calc_min_entry_qty(price, inverse, qty_step, min_qty, min_cost, contract_multiplier) -> float:
    return min_qty if inverse else max(min_qty, round_up(min_cost / price if price > 0.0 else 0.0, qty_step))


@njit
def cost_to_qty(cost, price, inverse, contract_multiplier):
    return cost * price / contract_multiplier if inverse else (cost / price if price > 0.0 else 0.0)


@njit
def qty_to_cost(qty, price, inverse, contract_multiplier) -> float:
    return (abs(qty / price) if price > 0.0 else 0.0) * contract_multiplier if inverse else abs(qty * price)


@njit
def calc_ema(alpha, alpha_, prev_ema, new_val) -> float:
    return prev_ema * alpha_ + new_val * alpha


@njit
def calc_emas(xs: [float], span: int) -> np.ndarray:
    alpha = 2 / (span + 1)
    alpha_ = 1 - alpha
    emas = np.empty_like(xs)
    emas[0] = xs[0]
    for i in range(1, len(xs)):
        emas[i] = emas[i - 1] * alpha_ + xs[i] * alpha
    return emas


@njit
def iter_MA_ratios_chunks(xs: [float], spans: [int], chunk_size: int = 65536):

    def to_ratios(emass_):
        ratios = np.empty((emass_.shape[0], emass_.shape[1] - 1))
        for i in range(1, emass_.shape[1]):
            ratios[:,i - 1] = emass_[:,i - 1] / emass_[:,i]
        return ratios

    max_spans = max(spans)
    if len(xs) < max_spans:
        return

    chunk_size = max(chunk_size, max_spans)

    n_chunks = int(round_up(len(xs) / chunk_size, 1.0))

    alphas = 2 / (spans + 1)
    alphas_ = 1 - alphas

    emass = np.empty((chunk_size, len(spans)), dtype=np.float64)
    emass[0] = xs[0]
    for i in range(1, chunk_size):
        emass[i] = emass[i - 1] * alphas_ + xs[i] * alphas
    yield to_ratios(emass), 0

    for k in range(1, n_chunks):
        kc = chunk_size * k
        new_emass = np.empty((chunk_size, len(spans)), dtype=np.float64)
        new_emass[0] = emass[-1] * alphas_ + xs[kc] * alphas
        for i in range(1, chunk_size):
            new_emass[i] = new_emass[i - 1] * alphas_ + xs[kc + i] * alphas
        yield to_ratios(new_emass), k
        emass = new_emass
    return emass


@njit
def calc_long_pnl(entry_price, close_price, qty, inverse, contract_multiplier) -> float:
    if inverse:
        if entry_price == 0.0 or close_price == 0.0:
            return 0.0
        return abs(qty) * contract_multiplier * (1 / entry_price - 1 / close_price)
    else:
        return abs(qty) * (close_price - entry_price)


@njit
def calc_shrt_pnl(entry_price, close_price, qty, inverse, contract_multiplier) -> float:
    if inverse:
        if entry_price == 0.0 or close_price == 0.0:
            return 0.0
        return abs(qty) * contract_multiplier * (1 / close_price - 1 / entry_price)
    else:
        return abs(qty) * (entry_price - close_price)


@njit
def calc_available_margin(balance,
                          long_psize,
                          long_pprice,
                          shrt_psize,
                          shrt_pprice,
                          last_price,
                          inverse, contract_multiplier, leverage) -> float:
    used_margin = 0.0
    equity = balance
    if long_pprice and long_psize:
        long_psize_real = long_psize * contract_multiplier
        equity += calc_long_pnl(long_pprice, last_price, long_psize_real, inverse,
                                contract_multiplier)
        used_margin += qty_to_cost(long_psize_real, long_pprice,
                                   inverse, contract_multiplier) / leverage
    if shrt_pprice and shrt_psize:
        shrt_psize_real = shrt_psize * contract_multiplier
        equity += calc_shrt_pnl(shrt_pprice, last_price, shrt_psize_real, inverse,
                                contract_multiplier)
        used_margin += qty_to_cost(shrt_psize_real, shrt_pprice,
                                   inverse, contract_multiplier) / leverage
    return max(0.0, equity - used_margin)


@njit
def calc_new_psize_pprice(psize, pprice, qty, price, qty_step) -> (float, float):
    if qty == 0.0:
        return psize, pprice
    new_psize = round_(psize + qty, qty_step)
    if new_psize == 0.0:
        return 0.0, 0.0
    return new_psize, nan_to_0(pprice) * (psize / new_psize) + price * (qty / new_psize)


@njit
def eqf(MA_ratios: [float], coeffs: [float]):
    if len(MA_ratios) == len(coeffs):
        return np.sum(MA_ratios * coeffs)
    return np.sum(MA_ratios * coeffs[:-1]) + coeffs[-1]


@njit
def calc_entry_price(balance, psize, pprice, MA, MA_ratios, grid_spacing_coeffs, pcost_bal_coeffs, MA_pct_coeffs,
                     inverse, contract_multiplier):
    return MA * eqf(MA_ratios, MA_pct_coeffs) if psize == 0.0 else \
           pprice * (eqf(MA_ratios, grid_spacing_coeffs) +
                     eqf(np.repeat(qty_to_cost(psize, pprice, inverse, contract_multiplier) / balance, len(pcost_bal_coeffs)),
                         pcost_bal_coeffs))


@njit
def calc_entry_qty(balance, psize, entry_price, MA_ratios, qty_pct_coeffs, ddown_factor_coeffs, available_margin,
                   inverse, contract_multiplier, qty_step, min_qty, min_cost):
    min_entry_qty = calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost, contract_multiplier)
    max_entry_qty = round_dn(cost_to_qty(available_margin, entry_price, inverse, contract_multiplier), qty_step)
    qty = round_dn(min(max_entry_qty, max(
        min_entry_qty,
        (cost_to_qty(balance, entry_price, inverse, contract_multiplier) * eqf(MA_ratios, qty_pct_coeffs)
         if psize == 0.0 else psize * eqf(MA_ratios, ddown_factor_coeffs))
    )), qty_step)
    return qty if qty >= min_entry_qty else 0.0


@njit
def iter_orders():
    available_margin = calc_available_margin(balance, long_psize, long_pprice,
                                             shrt_psize, shrt_pprice, last_price,
                                             inverse, contract_multiplier, max_leverage)
    while True:
        long_entry, long_close = (0.0, 0.0, 0.0, 0.0, ''), (0.0, 0.0, 0.0, 0.0, '')
        shrt_entry, shrt_close = (0.0, 0.0, 0.0, 0.0, ''), (0.0, 0.0, 0.0, 0.0, '')
        if do_long:
            long_entry_price = min(highest_bid,
                                   round_dn(calc_entry_price(balance, long_psize, long_pprice, long_MA, MA_ratios, long_grid_spacing_coeffs,
                                                             long_pcost_bal_coeffs, long_MA_pct_coeffs,
                                                             inverse, contract_multiplier), price_step))
            if long_entry_price > 0.0:
                long_entry_qty = calc_entry_qty(balance, long_psize, long_entry_price, MA_ratios, long_qty_pct_coeffs,
                                                long_ddown_factor_coeffs, available_margin, inverse, contract_multiplier,
                                                qty_step, min_qty, min_cost)
                if long_entry_qty > 0.0:
                    new_long_psize, new_long_pprice = calc_new_psize_pprice(long_psize, long_pprice, long_entry_qty,
                                                                            long_entry_price, qty_step)
                    long_entry = (long_entry_qty, long_entry_price, new_long_psize, new_long_pprice)
            if long_psize > 0.0:
                long_close = (-long_psize, max(lowest_ask, round_up(long_pprice * eqf(MA_ratios, long_markup_coeffs))),
                              0.0, 0.0, 'long_close')
                long_entry = long_entry[:4] + ('long_reentry',)
            else:
                long_entry = long_entry[:4] + ('long_initial_entry',)
        if do_shrt:
            shrt_entry_price = max(lowest_ask,
                                   round_up(calc_entry_price(balance, shrt_psize, shrt_pprice, shrt_MA, MA_ratios, shrt_grid_spacing_coeffs,
                                                             shrt_pcost_bal_coeffs, shrt_MA_pct_coeffs,
                                                             inverse, contract_multiplier), price_step))
            if shrt_entry_price > 0.0:
                shrt_entry_qty = -calc_entry_qty(balance, shrt_psize, shrt_entry_price, MA_ratios, shrt_qty_pct_coeffs,
                                                 shrt_ddown_factor_coeffs, available_margin, inverse, contract_multiplier,
                                                 qty_step, min_qty, min_cost)
                if shrt_entry_qty < 0.0:
                    new_shrt_psize, new_shrt_pprice = calc_new_psize_pprice(shrt_psize, shrt_pprice, shrt_entry_qty,
                                                                            shrt_entry_price, qty_step)
                    shrt_entry = (shrt_entry_qty, shrt_entry_price, new_shrt_psize, new_shrt_pprice)
            if shrt_psize < 0.0:
                shrt_close = (-shrt_psize, min(highest_bid, round_dn(shrt_pprice * eqf(MA_ratios, shrt_markup_coeffs))),
                              0.0, 0.0, 'shrt_close')
                shrt_close = shrt_close[:4] + ('shrt_reentry',)
            else:
                shrt_close = shrt_close[:4] + ('shrt_initial_entry',)


        orders = sorted([long_entry, shrt_entry, long_close, shrt_close], key=lambda x: calc_diff(x[1], last_price))
        if orders[0][0] == 0.0:
            break
        yield orders[0]
        if 'entry' in orders[0][4]:
            if 'long' in orders[0][4]:
                long_psize, long_pprice = orders[0][2:4]
                available_margin = max(0.0, available_margin - calc_margin_cost(long_entry[0], long_entry[1],
                                                                                inverse, contract_multiplier,
                                                                                max_leverage))
            else:
                shrt_psize, shrt_pprice = orders[0][2:4]
                available_margin = max(0.0, available_margin - calc_margin_cost(shrt_entry[0], shrt_entry[1],
                                                                                inverse, contract_multiplier,
                                                                                max_leverage))

@njit
def calc_liq_price_universal(balance,
                             long_psize,
                             long_pprice,
                             shrt_psize,
                             shrt_pprice,
                             inverse, contract_multiplier, leverage):
    long_pprice = nan_to_0(long_pprice)
    shrt_pprice = nan_to_0(shrt_pprice)
    long_psize *= contract_multiplier
    abs_shrt_psize = abs(shrt_psize) * contract_multiplier
    if inverse:
        shrt_cost = abs_shrt_psize / shrt_pprice if shrt_pprice > 0.0 else 0.0
        long_cost = long_psize / long_pprice if long_pprice > 0.0 else 0.0
        denominator = (shrt_cost - long_cost - balance)
        if denominator == 0.0:
            return 0.0
        liq_price = (abs_shrt_psize - long_psize) / denominator
    else:
        denominator = long_psize - abs_shrt_psize
        if denominator == 0.0:
            return 0.0
        liq_price = (-balance + long_psize * long_pprice - abs_shrt_psize * shrt_pprice) / denominator
    return max(0.0, liq_price)


def get_template_live_config():
    return {
        "config_name": "name",
        "logging_level": 0,
        "ma_spans": [1, 6000, 10800, 19440, 34992, 62986, 113374, 204073],
        "long": {
            "enabled":             True,
            "grid_spacing_coeffs":   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "pcost_bal_coeffs":    [0.0, 0.0],
            "qty_pct_coeffs":      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "ddown_factor_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "MA_pct_coeffs":       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "markup_coeffs":       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "MA_idx":              2
        },
        "shrt": {
            "enabled": True,
            "grid_spacing_coeffs":   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "pcost_bal_coeffs":    [0.0, 0.0],
            "qty_pct_coeffs":      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "ddown_factor_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "MA_pct_coeffs":       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "markup_coeffs":       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "MA_idx":              2
        }
    }















