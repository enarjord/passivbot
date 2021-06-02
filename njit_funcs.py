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
def calc_min_entry_qty(price, inverse, qty_step, min_qty, min_cost) -> float:
    return min_qty if inverse else max(min_qty, round_up(min_cost / price if price > 0.0 else 0.0, qty_step))


@njit
def calc_max_entry_qty(entry_price, available_margin, inverse, qty_step, c_mult):
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
def calc_bid_ask_thresholds(prices: np.ndarray, emas: np.ndarray, ratios: np.ndarray,
                            iprc_const, iprc_MAr_coeffs):
    bids = np.zeros(len(prices))
    asks = np.zeros(len(prices))
    for i in range(len(prices)):
        bids[i] = emas[i] * (iprc_const[0] + eqf(ratios[i], iprc_MAr_coeffs[0]))
        asks[i] = emas[i] * (iprc_const[1] + eqf(ratios[i], iprc_MAr_coeffs[1]))
    return bids, asks


@njit
def calc_emas(xs, spans):
    emas = np.zeros((len(xs), len(spans)))
    alphas = 2 / (spans + 1)
    alphas_ = 1 - alphas
    emas[0] = xs[0]
    for i in range(1, len(xs)):
        emas[i] = emas[i - 1] * alphas_ + xs[i] * alphas
    return emas


@njit
def calc_ratios(emas):
    return emas[:,:-1] / emas[:,1:]


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
def calc_equity(balance, long_psize, long_pprice, shrt_psize, shrt_pprice, last_price, inverse, c_mult):
    equity = balance
    if long_pprice and long_psize:
        long_psize_real = long_psize * c_mult
        equity += calc_long_pnl(long_pprice, last_price, long_psize_real, inverse, c_mult)
    if shrt_pprice and shrt_psize:
        shrt_psize_real = shrt_psize * c_mult
        equity += calc_shrt_pnl(shrt_pprice, last_price, shrt_psize_real, inverse, c_mult)
    return equity


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
        used_margin += qty_to_cost(long_psize_real, long_pprice, inverse, c_mult) / leverage[0]
    if shrt_pprice and shrt_psize:
        shrt_psize_real = shrt_psize * c_mult
        equity += calc_shrt_pnl(shrt_pprice, last_price, shrt_psize_real, inverse, c_mult)
        used_margin += qty_to_cost(shrt_psize_real, shrt_pprice, inverse, c_mult) / leverage[1]
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
def eqf(vals: np.ndarray, coeffs: np.ndarray, minus: float = 1.0) -> float:
    return np.sum((vals ** 2 - minus) * coeffs[:, 0] + (vals - minus) * coeffs[:, 1])


@njit
def calc_ientry_qty(balance, entry_price, MA_ratios, iqty_const, iqty_MAr_coeffs, qty_step, min_entry_qty,
                    max_entry_qty, inverse, c_mult):
    qty = round_dn(min(max_entry_qty, max(min_entry_qty, (cost_to_qty(balance, entry_price, inverse, c_mult) *
                                                          (iqty_const + eqf(MA_ratios, iqty_MAr_coeffs))))), qty_step)
    return qty if qty >= min_entry_qty else 0.0


@njit
def calc_rentry_qty(psize, entry_price, MA_ratios, rqty_const, rqty_MAr_coeffs, qty_step, min_entry_qty, max_entry_qty):
    qty = round_dn(min(max_entry_qty, max(min_entry_qty, abs(psize) * (rqty_const + eqf(MA_ratios, rqty_MAr_coeffs)))),
                   qty_step)
    return qty if qty >= min_entry_qty else 0.0


@njit
def calc_ientry_price(MA, MA_ratios, iprc_const, iprc_MAr_coeffs):
    # returns unrounded price
    return MA * (iprc_const + eqf(MA_ratios, iprc_MAr_coeffs))


@njit
def calc_rentry_price(balance, psize, pprice, MA_ratios, rprc_const, rprc_PBr_coeffs, rprc_MAr_coeffs, inverse, c_mult):
    # returns unrounded price
    pcost_bal_ratio = qty_to_cost(psize, pprice, inverse, c_mult) / balance
    return pprice * (rprc_const + eqf(MA_ratios, rprc_MAr_coeffs) + eqf(np.array([pcost_bal_ratio]), rprc_PBr_coeffs, minus=0.0))


@njit
def calc_stop_order(balance,
                    long_psize,
                    long_pprice,
                    shrt_psize,
                    shrt_pprice,
                    highest_bid,
                    lowest_ask,
                    equity,
                    bkr_diff,
                    qty_step,
                    min_qty,
                    stop_bkr_diff_thr,
                    stop_psize_pct,
                    stop_eqbal_ratio_thr):
    abs_shrt_psize = abs(shrt_psize)
    if long_psize > abs_shrt_psize:
        if bkr_diff < stop_bkr_diff_thr[0] or equity / balance < stop_eqbal_ratio_thr:
            stop_qty = min(long_psize, max(min_qty, round_dn(long_psize * stop_psize_pct, qty_step)))
            if stop_qty > min_qty:
                long_psize = max(0.0, round_(long_psize - stop_qty, qty_step))
                return -stop_qty, lowest_ask, long_psize, long_pprice, 'long_sclose'
    else:
        if bkr_diff < stop_bkr_diff_thr[1] or equity / balance < stop_eqbal_ratio_thr:
            stop_qty = min(abs_shrt_psize, max(min_qty, round_dn(abs_shrt_psize * stop_psize_pct, qty_step)))
            if stop_qty > min_qty:
                shrt_psize = min(0.0, round_(shrt_psize + stop_qty, qty_step))
                return stop_qty, highest_bid, shrt_psize, shrt_pprice, 'shrt_sclose'
    return 0.0, 0.0, 0.0, 0.0, ''


@njit
def calc_stop_order_new():
    pcost_bal_ratio = qty_to_cost(psize, pprice, inverse, c_mult)
    stop_qty = round_dn(psize * stop_psize_pct, qty_step)
    stop_price = MA * (sprc_const[0] +
                          eqf(MA_ratios, sprc_MAr_coeffs[0]) +
                          eqf(np.array([pcost_bal_ratio]), sprc_PBr_coeffs[0]))
    return stop_qty, stop_price


def calc_long_stop_order():
    qty, price = calc_stop_order_new()
    price = max(MA, round_up(price, price_step))
    return -qty, price


def calc_shrt_stop_order():
    qty, price = calc_stop_order_new()
    price = min(MA, round_dn(price, price_step))
    return qty, price

                


@njit
def calc_long_close(long_psize, long_pprice, lowest_ask, MA_ratios, price_step, markup_const, markup_MAr_coeffs):
    if long_psize > 0.0:
        return (-long_psize,
                max(lowest_ask, round_up(long_pprice * (markup_const[0] + eqf(MA_ratios, markup_MAr_coeffs[0])), price_step)),
                0.0, 0.0, 'long_nclose')
    return 0.0, 0.0, 0.0, 0.0, ''



@njit
def calc_shrt_close(shrt_psize, shrt_pprice, highest_bid, MA_ratios, price_step, markup_const, markup_MAr_coeffs):
    if shrt_psize < 0.0:
        return (-shrt_psize,
                min(highest_bid, round_dn(shrt_pprice * (markup_const[1] + eqf(MA_ratios, markup_MAr_coeffs[1])), price_step)),
                0.0, 0.0, 'shrt_nclose')
    return 0.0, 0.0, 0.0, 0.0, ''


'''
@njit
def calc_long_order():

            if long_psize == 0.0:
                ### initial long entry ###
                long_entry_price = min(highest_bid, round_dn(calc_ientry_price(MA, MA_ratios, iprc_const[0],
                                                                               iprc_MAr_coeffs[0]), price_step))
                if long_entry_price > 0.0:
                    min_entry_qty = calc_min_entry_qty(long_entry_price, inverse, qty_step, min_qty, min_cost)
                    max_entry_qty = calc_max_entry_qty(long_entry_price, available_margin, inverse, qty_step, c_mult)
                    long_entry_qty = calc_ientry_qty(balance, long_entry_price, MA_ratios, iqty_const[0],
                                                     iqty_MAr_coeffs[0], qty_step, min_entry_qty, max_entry_qty,
                                                     inverse, c_mult)
                    if long_entry_qty > 0.0:
                        new_bankruptcy_price = calc_bankruptcy_price(balance, long_entry_qty, long_entry_price,
                                                                     shrt_psize, shrt_pprice, inverse, c_mult)
                        if calc_diff(new_bankruptcy_price, last_price) > entry_bkr_diff_thr[0]:
                            orders.append((long_entry_qty, long_entry_price, long_entry_qty, long_entry_price, 'long_ientry'))
            else:
                ### long reentry ###
                long_entry_price = min(highest_bid,
                                       round_dn(calc_rentry_price(balance, long_psize, long_pprice, MA_ratios,
                                                                  rprc_const[0], rprc_PBr_coeffs[0], rprc_MAr_coeffs[0],
                                                                  inverse, c_mult), price_step))
                if long_entry_price > 0.0:
                    min_entry_qty = calc_min_entry_qty(long_entry_price, inverse, qty_step, min_qty, min_cost)
                    max_entry_qty = calc_max_entry_qty(long_entry_price, available_margin, inverse, qty_step, c_mult)
                    long_entry_qty = 1.0
                    long_entry_qty = calc_rentry_qty(long_psize, long_entry_price, MA_ratios, rqty_const[0],
                                                     rqty_MAr_coeffs[0], qty_step, min_entry_qty, max_entry_qty)
                    if long_entry_qty > 0.0:
                        new_long_psize, new_long_pprice = calc_new_psize_pprice(long_psize, long_pprice, long_entry_qty,
                                                                                long_entry_price, qty_step)
                        new_bankruptcy_price = calc_bankruptcy_price(balance, new_long_psize, new_long_pprice,
                                                                     shrt_psize, shrt_pprice, inverse, c_mult)
                        if calc_diff(new_bankruptcy_price, last_price) > entry_bkr_diff_thr[0]:
                            orders.append((long_entry_qty, long_entry_price,
                                           new_long_psize, new_long_pprice, 'long_rentry'))
'''



def iter_orders(balance,
                long_psize,
                long_pprice,
                shrt_psize,
                shrt_pprice,
                highest_bid,
                lowest_ask,
                MA,
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
                hedge_bkr_diff_thr,
                hedge_psize_pct,
                stop_bkr_diff_thr,
                stop_psize_pct,
                stop_eqbal_ratio_thr,
                entry_bkr_diff_thr,
                iqty_const,
                iprc_const,
                rqty_const,
                rprc_const,
                markup_const,
                iqty_MAr_coeffs,
                rprc_PBr_coeffs,
                iprc_MAr_coeffs,
                rqty_MAr_coeffs,
                rprc_MAr_coeffs,
                markup_MAr_coeffs):
    '''

    :param balance: float
    :param long_psize: float
    :param long_pprice: float
    :param shrt_psize: float
    :param shrt_pprice: float
    :param highest_bid: float
    :param lowest_ask: float
    :param MA: float
    :param last_price: float
    :param MA_ratios: float
    :param inverse: float
    :param do_long: float
    :param do_shrt: float
    :param qty_step: float
    :param price_step: float
    :param min_qty: float
    :param min_cost: float
    :param c_mult: float
    :param leverage: tuple(int, int)
    :param hedge_bkr_diff_thr: tuple(float, float)
    :param hedge_psize_pct: tuple(float, float)
    :param stop_bkr_diff_thr: tuple(float, float)
    :param stop_psize_pct: tuple(float, float)
    :param stop_eqbal_ratio_thr: float
    :param entry_bkr_diff_thr: tuple(float, float)
    :param iqty_const: tuple(float, float)
    :param iprc_const: tuple(float, float)
    :param rqty_const: tuple(float, float)
    :param rprc_const: tuple(float, float)
    :param markup_const: tuple(float, float)
    :param iqty_MAr_coeffs: tuple(np.ndarray(2,n_spans), np.ndarray(2,n_spans))
    :param rprc_PBr_coeffs: tuple(np.ndarray(2,n_spans), np.ndarray(2,n_spans))
    :param iprc_MAr_coeffs: tuple(np.ndarray(2,n_spans), np.ndarray(2,n_spans))
    :param rqty_MAr_coeffs: tuple(np.ndarray(2,n_spans), np.ndarray(2,n_spans))
    :param rprc_MAr_coeffs: tuple(np.ndarray(2,n_spans), np.ndarray(2,n_spans))
    :param markup_MAr_coeffs: tuple(np.ndarray(2,n_spans), np.ndarray(2,n_spans))
    :return: generator object which yields tuple (float, float, float, float, str)

    '''
    bankruptcy_price = calc_bankruptcy_price(balance, long_psize, long_pprice, shrt_psize, shrt_pprice, inverse, c_mult)
    bkr_diff = calc_diff(bankruptcy_price, last_price)
    equity = calc_equity(balance, long_psize, long_pprice, shrt_psize, shrt_pprice, last_price, inverse, c_mult)

    ### stop order ###
    stop_order = calc_stop_order(balance, long_psize, long_pprice, shrt_psize, shrt_pprice, highest_bid, lowest_ask,
                                 equity, bkr_diff, qty_step, min_qty, stop_bkr_diff_thr, stop_psize_pct,
                                 stop_eqbal_ratio_thr)
    if stop_order[0] != 0.0:
        yield stop_order
    long_close = calc_long_close(long_psize, long_pprice, lowest_ask, MA_ratios, price_step, markup_const, markup_MAr_coeffs)
    if long_close[0] != 0.0:
        yield long_close
    shrt_close = calc_shrt_close(shrt_psize, shrt_pprice, highest_bid, MA_ratios, price_step, markup_const, markup_MAr_coeffs)
    if shrt_close[0] != 0.0:
        yield shrt_close
    while True:
        available_margin = calc_available_margin(balance, long_psize, long_pprice, shrt_psize, shrt_pprice, last_price,
                                                 inverse, c_mult, leverage)
        orders = []
        if do_long:
            if long_psize == 0.0:
                ### initial long entry ###
                long_entry_price = min(highest_bid, round_dn(calc_ientry_price(MA, MA_ratios, iprc_const[0],
                                                                               iprc_MAr_coeffs[0]), price_step))
                if long_entry_price > 0.0:
                    min_entry_qty = calc_min_entry_qty(long_entry_price, inverse, qty_step, min_qty, min_cost)
                    max_entry_qty = calc_max_entry_qty(long_entry_price, available_margin, inverse, qty_step, c_mult)
                    long_entry_qty = calc_ientry_qty(balance, long_entry_price, MA_ratios, iqty_const[0],
                                                     iqty_MAr_coeffs[0], qty_step, min_entry_qty, max_entry_qty,
                                                     inverse, c_mult)
                    if long_entry_qty > 0.0:
                        new_bankruptcy_price = calc_bankruptcy_price(balance, long_entry_qty, long_entry_price,
                                                                     shrt_psize, shrt_pprice, inverse, c_mult)
                        if calc_diff(new_bankruptcy_price, last_price) > entry_bkr_diff_thr[0]:
                            orders.append((long_entry_qty, long_entry_price, long_entry_qty, long_entry_price, 'long_ientry'))
            else:
                ### long reentry ###
                long_entry_price = min(highest_bid,
                                       round_dn(calc_rentry_price(balance, long_psize, long_pprice, MA_ratios,
                                                                  rprc_const[0], rprc_PBr_coeffs[0], rprc_MAr_coeffs[0],
                                                                  inverse, c_mult), price_step))
                if long_entry_price > 0.0:
                    min_entry_qty = calc_min_entry_qty(long_entry_price, inverse, qty_step, min_qty, min_cost)
                    max_entry_qty = calc_max_entry_qty(long_entry_price, available_margin, inverse, qty_step, c_mult)
                    long_entry_qty = 1.0
                    long_entry_qty = calc_rentry_qty(long_psize, long_entry_price, MA_ratios, rqty_const[0],
                                                     rqty_MAr_coeffs[0], qty_step, min_entry_qty, max_entry_qty)
                    if long_entry_qty > 0.0:
                        new_long_psize, new_long_pprice = calc_new_psize_pprice(long_psize, long_pprice, long_entry_qty,
                                                                                long_entry_price, qty_step)
                        new_bankruptcy_price = calc_bankruptcy_price(balance, new_long_psize, new_long_pprice,
                                                                     shrt_psize, shrt_pprice, inverse, c_mult)
                        if calc_diff(new_bankruptcy_price, last_price) > entry_bkr_diff_thr[0]:
                            orders.append((long_entry_qty, long_entry_price,
                                           new_long_psize, new_long_pprice, 'long_rentry'))
        if do_shrt:
            if shrt_psize == 0.0:
                ### initial shrt entry ###
                shrt_entry_price = max(lowest_ask, round_up(calc_ientry_price(MA, MA_ratios, iprc_const[1],
                                                                              iprc_MAr_coeffs[1]), price_step))
                if shrt_entry_price > 0.0:
                    min_entry_qty = calc_min_entry_qty(shrt_entry_price, inverse, qty_step, min_qty, min_cost)
                    max_entry_qty = calc_max_entry_qty(shrt_entry_price, available_margin, inverse, qty_step, c_mult)
                    shrt_entry_qty = -calc_ientry_qty(balance, shrt_entry_price, MA_ratios, iqty_const[1],
                                                      iqty_MAr_coeffs[1], qty_step, min_entry_qty, max_entry_qty,
                                                      inverse, c_mult)
                    if shrt_entry_qty < 0.0:
                        new_bankruptcy_price = calc_bankruptcy_price(balance, shrt_entry_qty, shrt_entry_price,
                                                                     shrt_psize, shrt_pprice, inverse, c_mult)
                        if calc_diff(new_bankruptcy_price, last_price) > entry_bkr_diff_thr[1]:
                            orders.append((shrt_entry_qty, shrt_entry_price, shrt_entry_qty, shrt_entry_price, 'shrt_ientry'))
            else:
                ### shrt reentry ###
                shrt_entry_price = max(lowest_ask,
                                       round_up(calc_rentry_price(balance, shrt_psize, shrt_pprice, MA_ratios,
                                                                  rprc_const[1], rprc_PBr_coeffs[1], rprc_MAr_coeffs[1],
                                                                  inverse, c_mult), price_step))
                if shrt_entry_price > 0.0:
                    min_entry_qty = calc_min_entry_qty(shrt_entry_price, inverse, qty_step, min_qty, min_cost)
                    max_entry_qty = calc_max_entry_qty(shrt_entry_price, available_margin, inverse, qty_step, c_mult)
                    shrt_entry_qty = calc_rentry_qty(shrt_psize, shrt_entry_price, MA_ratios, rqty_const[1],
                                                     rqty_MAr_coeffs[1], qty_step, min_entry_qty, max_entry_qty)
                    if shrt_entry_qty > 0.0:
                        new_shrt_psize, new_shrt_pprice = calc_new_psize_pprice(shrt_psize, shrt_pprice, -shrt_entry_qty,
                                                                                shrt_entry_price, qty_step)
                        new_bankruptcy_price = calc_bankruptcy_price(balance, new_shrt_psize, new_shrt_pprice,
                                                                     shrt_psize, shrt_pprice, inverse, c_mult)
                        if calc_diff(new_bankruptcy_price, last_price) > entry_bkr_diff_thr[1]:
                            orders.append((-shrt_entry_qty, shrt_entry_price, new_shrt_psize,
                                           new_shrt_pprice, 'shrt_rentry'))

        ### hedge order ###

        if long_psize > abs(shrt_psize):
            if do_shrt and bkr_diff < hedge_bkr_diff_thr[0]:
                min_entry_qty = calc_min_entry_qty(lowest_ask, inverse, qty_step, min_qty, min_cost)
                max_entry_qty = calc_max_entry_qty(lowest_ask, available_margin, inverse, qty_step, c_mult)
                hedge_qty = max(min_entry_qty, min(max_entry_qty, round_dn(long_psize * hedge_psize_pct, qty_step)))
                if hedge_qty >= min_entry_qty:
                    hedge_qty = -hedge_qty
                    new_shrt_psize, new_shrt_pprice = calc_new_psize_pprice(shrt_psize, shrt_pprice, hedge_qty,
                                                                            lowest_ask, qty_step)
                    orders.append((hedge_qty, lowest_ask, new_shrt_psize, new_shrt_pprice, 'shrt_hentry'))
        else:
            if do_long and bkr_diff < hedge_bkr_diff_thr[1]:
                min_entry_qty = calc_min_entry_qty(highest_bid, inverse, qty_step, min_qty, min_cost)
                max_entry_qty = calc_max_entry_qty(highest_bid, available_margin, inverse, qty_step, c_mult)
                hedge_qty = max(min_entry_qty, min(max_entry_qty, round_dn(long_psize * hedge_psize_pct, qty_step)))
                if hedge_qty >= min_entry_qty:
                    new_long_psize, new_long_pprice = calc_new_psize_pprice(long_psize, long_pprice,
                                                                            hedge_qty, highest_bid, qty_step)
                    orders.append((hedge_qty, highest_bid, new_long_psize, new_long_pprice, 'long_hentry'))
        if not orders:
            break
        orders = [o + (calc_diff(o[1] ,last_price),) for o in orders]
        orders = sorted(orders, key=lambda x: x[-1])
        #orders = sorted(orders, key=lambda x: calc_diff(x[1], last_price))
        if orders[0][0] == 0.0:
            break
        yield orders[0][:5]
        if 'entry' in orders[0][4]:
            if 'long' in orders[0][4]:
                long_psize, long_pprice = orders[0][2:4]
            else:
                shrt_psize, shrt_pprice = orders[0][2:4]
        bankruptcy_price = calc_bankruptcy_price(balance, long_psize, long_pprice,
                                                 shrt_psize, shrt_pprice, inverse, c_mult)
        bkr_diff = calc_diff(bankruptcy_price, last_price)


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
        bankruptcy_price = (abs_shrt_psize - long_psize) / denominator
    else:
        denominator = long_psize - abs_shrt_psize
        if denominator == 0.0:
            return 0.0
        bankruptcy_price = (-balance + long_psize * long_pprice - abs_shrt_psize * shrt_pprice) / denominator
    return max(0.0, bankruptcy_price)

