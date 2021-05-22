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
def calc_min_entry_qty(price, inverse, qty_step, min_qty, min_cost, c_mult) -> float:
    return min_qty if inverse else max(min_qty, round_up(min_cost / price if price > 0.0 else 0.0, qty_step))


@njit
def calc_max_entry_qty(inverse, qty_step, c_mult, entry_price, available_margin):
    return round_dn(cost_to_qty(available_margin, entry_price, inverse, c_mult), qty_step)


@njit
def cost_to_qty(cost, price, inverse, c_mult):
    return cost * price / c_mult if inverse else (cost / price if price > 0.0 else 0.0)


@njit
def qty_to_cost(qty, price, inverse, c_mult) -> float:
    return (abs(qty / price) if price > 0.0 else 0.0) * c_mult if inverse else abs(qty * price)


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
def calc_ema_ratios(xs: [float], spans: [int]):
    alphas = 2 / (spans + 1)
    alphas_ = 1 - alphas
    emas = np.copy(xs[0])
    ema_ratios = np.empty((len(xs), len(spans) - 1), dtype=np.float32)
    ema_ratios[0] = 0.0
    for i in range(1, len(xs)):
        for k in range(1, len(spans)):
            ema_ratios[i][k] = ema



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
def calc_long_pnl(entry_price, close_price, qty, inverse, c_mult) -> float:
    if inverse:
        if entry_price == 0.0 or close_price == 0.0:
            return 0.0
        return abs(qty) * c_mult * (1 / entry_price - 1 / close_price)
    else:
        return abs(qty) * (close_price - entry_price)


@njit
def calc_shrt_pnl(entry_price, close_price, qty, inverse, c_mult) -> float:
    if inverse:
        if entry_price == 0.0 or close_price == 0.0:
            return 0.0
        return abs(qty) * c_mult * (1 / close_price - 1 / entry_price)
    else:
        return abs(qty) * (entry_price - close_price)


@njit
def calc_available_margin(balance,
                          long_psize,
                          long_pprice,
                          shrt_psize,
                          shrt_pprice,
                          last_price,
                          inverse, c_mult, leverage) -> float:
    used_margin = 0.0
    equity = balance
    if long_pprice and long_psize:
        long_psize_real = long_psize * c_mult
        equity += calc_long_pnl(long_pprice, last_price, long_psize_real, inverse, c_mult)
        used_margin += qty_to_cost(long_psize_real, long_pprice, inverse, c_mult) / leverage
    if shrt_pprice and shrt_psize:
        shrt_psize_real = shrt_psize * c_mult
        equity += calc_shrt_pnl(shrt_pprice, last_price, shrt_psize_real, inverse, c_mult)
        used_margin += qty_to_cost(shrt_psize_real, shrt_pprice, inverse, c_mult) / leverage
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
def eqf(vals: [float], coeffs: [float]) -> float:
    r = coeffs[-1] if len(coeffs) > len(vals) else 0.0
    for v, c in zip(vals, coeffs):
        r += v * c
    return r


@njit
def calc_entry_price(balance, psize, pprice, MA, MA_ratios, iprc_PBr_coeffs, iprc_MAr_coeffs, rprc_MAr_coeffs,
                     inverse, c_mult):
    if psize == 0.0:
        return MA * eqf(MA_ratios, iprc_MAr_coeffs)
    else:
        pcost_bal_ratio = qty_to_cost(psize, pprice, inverse, c_mult) / balance
        return pprice * (eqf(MA_ratios, rprc_MAr_coeffs) + eqf([pcost_bal_ratio**2, pcost_bal_ratio], iprc_PBr_coeffs))


@njit
def calc_entry_qty(balance, psize, entry_price, MA_ratios, iqty_MAr_coeffs, rqty_MAr_coeffs, available_margin,
                   inverse, c_mult, qty_step, min_qty, min_cost):
    min_entry_qty = calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost, c_mult)
    qty = round_dn(min(calc_max_entry_qty(inverse, qty_step, c_mult, entry_price, available_margin), max(
        min_entry_qty,
        (cost_to_qty(balance, entry_price, inverse, c_mult) * eqf(MA_ratios, iqty_MAr_coeffs)
         if psize == 0.0 else psize * eqf(MA_ratios, rqty_MAr_coeffs))
    )), qty_step)
    return qty if qty >= min_entry_qty else 0.0


@njit
def iter_orders(
        balance,
        long_psize,
        long_pprice,
        shrt_psize,
        shrt_pprice,
        liq_price,
        highest_bid,
        lowest_ask,
        long_MA,
        shrt_MA,
        last_price,
        MA_ratios,

        inverse,
        do_long,
        do_shrt,
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
        leverage,
        hedge_liq_diff_thr,
        hedge_psize_pct,
        stop_liq_diff_thr,
        stop_psize_pct,
        entry_liq_diff_thr,
        iqty_MAr_coeffs,
        iprc_PBr_coeffs,
        iprc_MAr_coeffs,
        rqty_MAr_coeffs,
        rprc_MAr_coeffs,
        markup_MAr_coeffs):
    available_margin = calc_available_margin(balance, long_psize, long_pprice,
                                             shrt_psize, shrt_pprice, last_price,
                                             inverse, c_mult, leverage)
    while True:
        orders = []

        ### long entry ###
        if do_long:
            long_entry_price = min(highest_bid,
                                   round_dn(calc_entry_price(balance, long_psize, long_pprice, long_MA, MA_ratios,
                                                             iprc_PBr_coeffs[0], iprc_MAr_coeffs[0],
                                                             rprc_MAr_coeffs[0], inverse, c_mult), price_step))
            if long_entry_price > 0.0:
                long_entry_qty = calc_entry_qty(balance, long_psize, long_entry_price, MA_ratios, iqty_MAr_coeffs[0],
                                                rqty_MAr_coeffs[0], available_margin, inverse, c_mult,
                                                qty_step, min_qty, min_cost)
                if long_entry_qty > 0.0:
                    new_long_psize, new_long_pprice = calc_new_psize_pprice(long_psize, long_pprice, long_entry_qty,
                                                                            long_entry_price, qty_step)
                    bankruptcy_price = calc_bankruptcy_price(balance, new_long_psize, new_long_pprice, shrt_psize,
                                                             shrt_pprice, inverse, c_mult)
                    if calc_diff(bankruptcy_price, last_price) > entry_liq_diff_thr:
                        orders.append((long_entry_qty, long_entry_price, new_long_psize, new_long_pprice,
                                       'long_ientry' if long_psize == 0.0 else 'long_rentry'))

            ### long normal close ###
            if long_psize > 0.0:
                orders.append((-long_psize, max(lowest_ask, round_up(long_pprice * eqf(MA_ratios, markup_MAr_coeffs[0]))),
                               0.0, 0.0, 'long_nclose'))

        ### shrt entry ###
        if do_shrt:
            shrt_entry_price = max(lowest_ask,
                                   round_up(calc_entry_price(balance, shrt_psize, shrt_pprice, shrt_MA, MA_ratios,
                                                             iprc_PBr_coeffs[1], iprc_MAr_coeffs[1],
                                                             rprc_MAr_coeffs[1], inverse, c_mult), price_step))
            shrt_entry_qty = -calc_entry_qty(balance, shrt_psize, shrt_entry_price, MA_ratios, iqty_MAr_coeffs[1],
                                             rqty_MAr_coeffs[1], available_margin, inverse, c_mult,
                                             qty_step, min_qty, min_cost)
            if shrt_entry_qty < 0.0:
                new_shrt_psize, new_shrt_pprice = calc_new_psize_pprice(shrt_psize, shrt_pprice, shrt_entry_qty,
                                                                        shrt_entry_price, qty_step)
                bankruptcy_price = calc_bankruptcy_price(balance, long_psize, long_pprice, new_shrt_psize, new_shrt_pprice,
                                                         inverse, c_mult)
                if calc_diff(bankruptcy_price, last_price) > entry_liq_diff_thr:
                    orders.append((shrt_entry_qty, shrt_entry_price, new_shrt_psize, new_shrt_pprice,
                                   'shrt_ientry' if shrt_psize == 0.0 else 'shrt_rentry'))

            ### shrt normal close ###
            if shrt_psize < 0.0:
                orders.append((-shrt_psize, min(highest_bid, round_dn(shrt_pprice * eqf(MA_ratios, markup_MAr_coeffs[1]))),
                               0.0, 0.0, 'shrt_nclose'))

        ### hedge order ###
        if calc_diff(liq_price, last_price) < hedge_liq_diff_thr:
            if long_psize > abs(shrt_psize):
                if do_shrt:
                    min_entry_qty = calc_min_entry_qty(lowest_ask, inverse, qty_step, min_qty, min_cost, c_mult)
                    max_entry_qty = calc_max_entry_qty(inverse, qty_step, c_mult, lowest_ask, available_margin)
                    hedge_qty = max(min_entry_qty, min(max_entry_qty, round_dn(long_psize * hedge_psize_pct, qty_step)))
                    if hedge_qty >= min_entry_qty:
                        hedge_qty = -hedge_qty
                        new_shrt_psize, new_shrt_pprice = calc_new_psize_pprice(shrt_psize, shrt_pprice, hedge_qty,
                                                                                lowest_ask, qty_step)
                        orders.append((hedge_qty, lowest_ask, new_shrt_psize, new_shrt_pprice, 'shrt_hentry'))
            else:
                if do_long:
                    min_entry_qty = calc_min_entry_qty(highest_bid, inverse, qty_step, min_qty, min_cost, c_mult)
                    max_entry_qty = calc_max_entry_qty(inverse, qty_step, c_mult, highest_bid, available_margin)
                    hedge_qty = max(min_entry_qty, min(max_entry_qty, round_dn(long_psize * hedge_psize_pct, qty_step)))
                    if hedge_qty >= min_entry_qty:
                        new_long_psize, new_long_pprice = calc_new_psize_pprice(long_psize, long_pprice,
                                                                                hedge_qty, highest_bid, qty_step)
                        orders.append((hedge_qty, highest_bid, new_long_psize, new_long_pprice, 'long_hentry'))

        ### stop order ###
        if calc_diff(liq_price, last_price) < stop_liq_diff_thr:
            abs_shrt_psize = abs(shrt_psize)
            if long_psize > abs_shrt_psize:
                stop_qty = min(long_psize, max(min_qty, round_dn(long_psize * stop_psize_pct, qty_step)))
                if stop_qty > min_qty:
                    orders.append((-stop_qty, lowest_ask, round_(long_psize - stop_qty, qty_step), long_pprice, 'long_sclose'))
            else:
                stop_qty = min(abs_shrt_psize, max(min_qty, round_dn(abs_shrt_psize * stop_psize_pct, qty_step)))
                if stop_qty > min_qty:
                    orders.append((stop_qty, highest_bid, round_(shrt_psize + stop_qty, qty_step), long_pprice, 'shrt_sclose'))

        orders = sorted(orders, key=lambda x: calc_diff(x[1], last_price))
        if orders[0][0] == 0.0:
            break
        yield orders[0]
        if 'entry' in orders[0][4]:
            if 'long' in orders[0][4]:
                long_psize, long_pprice = orders[0][2:4]
            else:
                shrt_psize, shrt_pprice = orders[0][2:4]
            available_margin = max(0.0, available_margin - qty_to_cost(orders[0], orders[1], inverse, c_mult) / leverage)


@njit
def calc_bankruptcy_price(balance,
                          long_psize,
                          long_pprice,
                          shrt_psize,
                          shrt_pprice,
                          inverse, c_mult):
    long_pprice = nan_to_0(long_pprice)
    shrt_pprice = nan_to_0(shrt_pprice)
    long_psize *= c_mult
    abs_shrt_psize = abs(shrt_psize) * c_mult
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
            "enabled":            True,
            "leverage":           10,
            "hedge_liq_diff_thr": 0.5,    # make counter order if diff(liq, last) < thr
            "hedge_psize_pct":    0.05,   # % of psize for hedge order
            "stop_liq_diff_thr":  0.21,   # partially close pos at a loss if diff(liq, last) < thr
            "stop_psize_pct":     0.05,   # % of psize for stop loss order
            "entry_liq_diff_thr": 0.21,   # prevent entries whose filling would result in diff(new_liq, last) < thr
            "iqty_MAr_coeffs":    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "iprc_PBr_coeffs":    [0.0, 0.0],
            "iprc_MAr_coeffs":    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "rqty_MAr_coeffs":    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "rprc_MAr_coeffs":    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "markup_MAr_coeffs":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "MA_idx":             2
        },
        "shrt": {
            "enabled":            True,
            "leverage":           10,
            "hedge_liq_diff_thr": 0.5,    # make counter order if diff(liq, last) < thr
            "hedge_psize_pct":    0.05,   # % of psize for hedge order
            "stop_liq_diff_thr":  0.21,   # partially close pos at a loss if diff(liq, last) < thr
            "stop_psize_pct":     0.05,   # % of psize for stop loss order
            "entry_liq_diff_thr": 0.21,   # prevent entries whose filling would result in diff(new_liq, last) < thr
            "iqty_MAr_coeffs":    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "iprc_PBr_coeffs":    [0.0, 0.0],
            "iprc_MAr_coeffs":    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "rqty_MAr_coeffs":    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "rprc_MAr_coeffs":    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "markup_MAr_coeffs":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "MA_idx":             2
        }
    }















