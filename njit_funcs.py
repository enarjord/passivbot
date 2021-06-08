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
    return emas[:, :-1] / emas[:, 1:]


@njit
def calc_long_pnl(entry_price, close_price, qty, inverse, c_mult) -> float:
    if inverse:
        if entry_price == 0.0 or close_price == 0.0:
            return 0.0
        return abs(qty) * c_mult * (1.0 / entry_price - 1.0 / close_price)
    else:
        return abs(qty) * (close_price - entry_price)


@njit
def calc_shrt_pnl(entry_price, close_price, qty, inverse, c_mult) -> float:
    if inverse:
        if entry_price == 0.0 or close_price == 0.0:
            return 0.0
        return abs(qty) * c_mult * (1.0 / close_price - 1.0 / entry_price)
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
    return np.sum((vals ** 2 - minus) * coeffs[:, 0] + np.abs(vals - minus) * coeffs[:, 1])


@njit
def calc_long_orders(balance,
                     long_psize,
                     long_pprice,
                     highest_bid,
                     lowest_ask,
                     MA,
                     MA_ratios,
                     available_margin,
 
                     inverse,
                     qty_step,
                     price_step,
                     min_qty,
                     min_cost,
                     c_mult,
                     stop_psize_pct,
                     stop_PBr_thr,
                     iqty_const,
                     iprc_const,
                     rqty_const,
                     rprc_const,
                     markup_const,
                     iqty_MAr_coeffs,
                     iprc_MAr_coeffs,
                     rprc_PBr_coeffs,
                     rqty_MAr_coeffs,
                     rprc_MAr_coeffs,
                     markup_MAr_coeffs) -> ((float, float, float, float, str), [(float, float, float, float, str)]):
    if long_psize == 0.0:
        entry_price = min(highest_bid, round_dn(MA * (iprc_const + eqf(MA_ratios, iprc_MAr_coeffs)), price_step))
        entry_qty = max(calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost),
                        round_dn(cost_to_qty(balance, entry_price, inverse, c_mult) *
                                 (iqty_const + eqf(MA_ratios, iqty_MAr_coeffs)), qty_step))
        entry_type = 'long_ientry'
        long_closes = [(0.0, 0.0, 0.0, 0.0, 'long_nclose')]
    elif long_psize > 0.0:
        pbr = qty_to_cost(long_psize, long_pprice, inverse, c_mult) / balance
        nclose_price = round_up(long_pprice * (markup_const + eqf(MA_ratios, markup_MAr_coeffs)), price_step)
        if pbr > stop_PBr_thr:

            entry_price = round_dn(min([highest_bid, MA * iprc_const,
                                        long_pprice * (rprc_const + eqf(MA_ratios, rprc_MAr_coeffs) +
                                                       eqf(np.array([pbr]), rprc_PBr_coeffs, minus=0.0))]), price_step)
            stop_qty = -max(min_qty, round_dn(long_psize * stop_psize_pct, qty_step))
            stop_price = max(lowest_ask, round_up(MA * (2.0 - iprc_const), price_step))
            long_closes = [(stop_qty, stop_price, round_(long_psize + stop_qty, qty_step), long_pprice, 'long_sclose'),
                           (-max(0.0, round_(long_psize + stop_qty, qty_step)), nclose_price, 0.0, 0.0, 'long_nclose')]
        else:
            entry_price = round_dn(min(highest_bid,
                                       long_pprice * (rprc_const + eqf(MA_ratios, rprc_MAr_coeffs) +
                                                      eqf(np.array([pbr]), rprc_PBr_coeffs, minus=0.0))), price_step)
            long_closes = [(-long_psize, nclose_price, 0.0, 0.0, 'long_nclose')]
        min_entry_qty = calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost)
        max_entry_qty = cost_to_qty(min(balance * (stop_PBr_thr + stop_psize_pct - pbr), available_margin),
                                    entry_price, inverse, c_mult)
        entry_qty = round_dn(min(max_entry_qty,
                                 max(min_entry_qty, long_psize * (rqty_const + eqf(MA_ratios, rqty_MAr_coeffs)))),
                             qty_step)
        if entry_qty < min_entry_qty:
            if pbr > stop_PBr_thr:
                entry_qty = 0.0
            else:
                entry_qty = min_entry_qty
        entry_type = 'long_rentry'
    else:
        raise Exception('long psize is less than 0.0')

    new_psize, new_pprice = calc_new_psize_pprice(long_psize, long_pprice, entry_qty, entry_price, qty_step)
    return ((entry_qty, entry_price, new_psize, new_pprice, entry_type),
            sorted(long_closes, key=lambda x: x[1], reverse=False))


@njit
def calc_shrt_orders(balance,
                     shrt_psize,
                     shrt_pprice,
                     highest_bid,
                     lowest_ask,
                     MA,
                     MA_ratios,
                     available_margin,
 
                     inverse,
                     qty_step,
                     price_step,
                     min_qty,
                     min_cost,
                     c_mult,
                     stop_psize_pct,
                     stop_PBr_thr,
                     iqty_const,
                     iprc_const,
                     rqty_const,
                     rprc_const,
                     markup_const,
                     iqty_MAr_coeffs,
                     iprc_MAr_coeffs,
                     rprc_PBr_coeffs,
                     rqty_MAr_coeffs,
                     rprc_MAr_coeffs,
                     markup_MAr_coeffs) -> ((float, float, float, float, str), [(float, float, float, float, str)]):
    if shrt_psize == 0.0:
        entry_price = max(lowest_ask, round_up(MA * (iprc_const + eqf(MA_ratios, iprc_MAr_coeffs)), price_step))
        entry_qty = max(calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost),
                       round_dn(cost_to_qty(balance, entry_price, inverse, c_mult) *
                                (iqty_const + eqf(MA_ratios, iqty_MAr_coeffs)), qty_step))
        entry_type = 'shrt_ientry'
        shrt_closes = [(0.0, 0.0, 0.0, 0.0, 'shrt_nclose')]
    elif shrt_psize < 0.0:
        pbr = qty_to_cost(shrt_psize, shrt_pprice, inverse, c_mult) / balance
        nclose_price = round_dn(shrt_pprice * (markup_const + eqf(MA_ratios, markup_MAr_coeffs)), price_step)
        if pbr > stop_PBr_thr:
            entry_price = round_up(max([lowest_ask, MA * iprc_const,
                                        shrt_pprice * (rprc_const + eqf(MA_ratios, rprc_MAr_coeffs) +
                                                       eqf(np.array([pbr]), rprc_PBr_coeffs, minus=0.0))]), price_step)
            stop_qty = max(min_qty, round_dn(-shrt_psize * stop_psize_pct, qty_step))
            stop_price = min(highest_bid, round_dn(MA * (2.0 - iprc_const), price_step))
            shrt_closes = [(stop_qty, stop_price, round_(shrt_psize + stop_qty, qty_step), shrt_pprice, 'shrt_sclose'),
                           (max(0.0, round_(-shrt_psize - stop_qty, qty_step)), nclose_price, 0.0, 0.0, 'shrt_nclose')]
        else:
            entry_price = round_up(max(lowest_ask,
                                       shrt_pprice * (rprc_const + eqf(MA_ratios, rprc_MAr_coeffs) +
                                                      eqf(np.array([pbr]), rprc_PBr_coeffs, minus=0.0))), price_step)
            shrt_closes = [(-shrt_psize, nclose_price, 0.0, 0.0, 'shrt_nclose')]
        min_entry_qty = calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost)
        max_entry_qty = cost_to_qty(min(balance * (stop_PBr_thr + stop_psize_pct - pbr), available_margin),
                                    entry_price, inverse, c_mult)
        entry_qty = round_dn(min(max_entry_qty,
                                 max(min_entry_qty, -shrt_psize * (rqty_const + eqf(MA_ratios, rqty_MAr_coeffs)))),
                             qty_step)
        if entry_qty < min_entry_qty:
            if pbr > stop_PBr_thr:
                entry_qty = 0.0
            else:
                entry_qty = min_entry_qty
        entry_type = 'shrt_rentry'
    else:
        raise Exception('shrt psize is greater than 0.0 ')
    entry_qty = -entry_qty
    new_psize, new_pprice = calc_new_psize_pprice(shrt_psize, shrt_pprice, entry_qty, entry_price, qty_step)
    return ((entry_qty, entry_price, new_psize, new_pprice, entry_type),
            sorted(shrt_closes, key=lambda x: x[1], reverse=True))


@njit
def calc_upnl(long_psize,
              long_pprice,
              shrt_psize,
              shrt_pprice,
              last_price,
              inverse, c_mult):
    return calc_long_pnl(long_pprice, last_price, long_psize, inverse, c_mult) + \
           calc_shrt_pnl(shrt_pprice, last_price, shrt_psize, inverse, c_mult)


@njit
def calc_orders(balance,
                long_psize,
                long_pprice,
                shrt_psize,
                shrt_pprice,
                highest_bid,
                lowest_ask,
                last_price,
                MA,
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
                stop_psize_pct,
                stop_PBr_thr,
                iqty_const,
                iprc_const,
                rqty_const,
                rprc_const,
                markup_const,
                iqty_MAr_coeffs,
                iprc_MAr_coeffs,
                rprc_PBr_coeffs,
                rqty_MAr_coeffs,
                rprc_MAr_coeffs,
                markup_MAr_coeffs):
    available_margin = calc_available_margin(balance, long_psize, long_pprice, shrt_psize, shrt_pprice,
                                             last_price, inverse, c_mult, leverage)
    long_entry, long_closes = calc_long_orders(balance,
                     long_psize,
                     long_pprice,
                     highest_bid,
                     lowest_ask,
                     MA,
                     MA_ratios,
                     available_margin,

                     inverse,
                     qty_step,
                     price_step,
                     min_qty,
                     min_cost,
                     c_mult,
                     stop_psize_pct,
                     stop_PBr_thr[0],
                     iqty_const[0],
                     iprc_const[0],
                     rqty_const[0],
                     rprc_const[0],
                     markup_const[0],
                     iqty_MAr_coeffs[0],
                     iprc_MAr_coeffs[0],
                     rprc_PBr_coeffs[0],
                     rqty_MAr_coeffs[0],
                     rprc_MAr_coeffs[0],
                     markup_MAr_coeffs[0]) if do_long else ((0.0, 0.0, 0.0, 0.0, ''), [(0.0, 0.0, 0.0, 0.0, '')])
    shrt_entry, shrt_closes = calc_shrt_orders(balance,
                     shrt_psize,
                     shrt_pprice,
                     highest_bid,
                     lowest_ask,
                     MA,
                     MA_ratios,
                     available_margin,

                     inverse,
                     qty_step,
                     price_step,
                     min_qty,
                     min_cost,
                     c_mult,
                     stop_psize_pct,
                     stop_PBr_thr[1],
                     iqty_const[1],
                     iprc_const[1],
                     rqty_const[1],
                     rprc_const[1],
                     markup_const[1],
                     iqty_MAr_coeffs[1],
                     iprc_MAr_coeffs[1],
                     rprc_PBr_coeffs[1],
                     rqty_MAr_coeffs[1],
                     rprc_MAr_coeffs[1],
                     markup_MAr_coeffs[1]) if do_shrt else ((0.0, 0.0, 0.0, 0.0, ''), [(0.0, 0.0, 0.0, 0.0, '')])
    bkr_price = calc_bankruptcy_price(balance, long_psize, long_pprice, shrt_psize, shrt_pprice, inverse, c_mult)
    return long_entry, shrt_entry, long_closes, shrt_closes, bkr_price, available_margin


@njit
def backtest(data: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray),
             starting_balance,
             latency_simulation_ms,
             maker_fee,
             inverse,
             do_long,
             do_shrt,
             qty_step,
             price_step,
             min_qty,
             min_cost,
             c_mult,
             leverage,
             stop_psize_pct,
             stop_PBr_thr,
             iqty_const,
             iprc_const,
             rqty_const,
             rprc_const,
             markup_const,
             iqty_MAr_coeffs,
             iprc_MAr_coeffs,
             rprc_PBr_coeffs,
             rqty_MAr_coeffs,
             rprc_MAr_coeffs,
             markup_MAr_coeffs):
    prices, buyer_maker, timestamps, emas, ratios = data
    static_params = (inverse, do_long, do_shrt, qty_step, price_step, min_qty, min_cost, c_mult, leverage,
                     stop_psize_pct, stop_PBr_thr, iqty_const, iprc_const, rqty_const, rprc_const, markup_const,
                     iqty_MAr_coeffs, iprc_MAr_coeffs, rprc_PBr_coeffs, rqty_MAr_coeffs, rprc_MAr_coeffs,
                     markup_MAr_coeffs)

    balance = starting_balance

    long_psize = 0.0
    long_pprice = 0.0
    shrt_psize = 0.0
    shrt_pprice = 0.0

    next_update_ts = 0

    ob = [prices[0], prices[0]]

    fills = []

    long_entry, shrt_entry = (0.0, 0.0, 0.0, 0.0, ''), (0.0, 0.0, 0.0, 0.0, '')
    long_closes, shrt_closes = [(0.0, 0.0, 0.0, 0.0, '')], [(0.0, 0.0, 0.0, 0.0, '')]
    bkr_price, available_margin = 0.0, 0.0

    prev_k = 0
    prev_ob = [0.0, 0.0]

    for k in range(len(prices)):

        if timestamps[k] > next_update_ts:
            long_entry, shrt_entry, long_closes, shrt_closes, bkr_price, available_margin = calc_orders(
                balance,
                long_psize,
                long_pprice,
                shrt_psize,
                shrt_pprice,
                ob[0],
                ob[1],
                prices[k],
                emas[k],
                ratios[k],

                *static_params)
            next_update_ts = timestamps[k] + 5000
            prev_k = k
            prev_ob = ob

        if calc_diff(bkr_price, prices[k]) < 0.05:
            if long_psize != 0.0:
                fee_paid = -qty_to_cost(long_psize, long_pprice, inverse, c_mult) * maker_fee
                pnl = calc_long_pnl(long_pprice, prices[k], -long_psize, inverse, c_mult)
                balance = balance + fee_paid + pnl
                long_psize, long_pprice = long_entry[2:4]
                equity = balance + calc_upnl(long_psize, long_pprice, shrt_psize, shrt_pprice,
                                             prices[k], inverse, c_mult)
                fills.append((k, pnl, fee_paid, balance, equity, 0.0, -long_psize, prices[k], 0.0, 0.0, 'long_bankruptcy'))
            if shrt_psize != 0.0:

                fee_paid = -qty_to_cost(shrt_psize, shrt_pprice, inverse, c_mult) * maker_fee
                pnl = calc_shrt_pnl(shrt_pprice, prices[k], -shrt_psize, inverse, c_mult)
                balance = balance + fee_paid + pnl
                shrt_psize, shrt_pprice = shrt_entry[2:4]
                equity = balance + calc_upnl(long_psize, long_pprice, shrt_psize, shrt_pprice,
                                             prices[k], inverse, c_mult)
                fills.append((k, pnl, fee_paid, balance, equity, 0.0, -shrt_psize, prices[k], 0.0, 0.0, 'shrt_bankruptcy'))

            return fills, False

        if buyer_maker[k]:
            while long_entry[0] != 0.0 and prices[k] < long_entry[1]:
                fee_paid = -qty_to_cost(long_entry[0], long_entry[1], inverse, c_mult) * maker_fee
                balance += fee_paid
                long_psize, long_pprice = long_entry[2:4]
                equity = balance + calc_upnl(long_psize, long_pprice, shrt_psize, shrt_pprice,
                                             prices[k], inverse, c_mult)
                pbr = qty_to_cost(long_psize, long_pprice, inverse, c_mult) / balance
                fills.append((k, 0.0, fee_paid, balance, equity, pbr) + long_entry)
                next_update_ts = min(next_update_ts, timestamps[k] + latency_simulation_ms)
                long_entry, _ = calc_long_orders(balance,
                                                 long_psize,
                                                 long_pprice,
                                                 prev_ob[0],
                                                 prev_ob[1],
                                                 emas[prev_k],
                                                 ratios[prev_k],
                                                 available_margin,

                                                 inverse,
                                                 qty_step,
                                                 price_step,
                                                 min_qty,
                                                 min_cost,
                                                 c_mult,
                                                 stop_psize_pct,
                                                 stop_PBr_thr[0],
                                                 iqty_const[0],
                                                 iprc_const[0],
                                                 rqty_const[0],
                                                 rprc_const[0],
                                                 markup_const[0],
                                                 iqty_MAr_coeffs[0],
                                                 iprc_MAr_coeffs[0],
                                                 rprc_PBr_coeffs[0],
                                                 rqty_MAr_coeffs[0],
                                                 rprc_MAr_coeffs[0],
                                                 markup_MAr_coeffs[0])
            while shrt_closes and shrt_psize != 0.0 and shrt_closes[0][0] != 0.0 and prices[k] < shrt_closes[0][1]:
                fee_paid = -qty_to_cost(shrt_closes[0][0], shrt_closes[0][1], inverse, c_mult) * maker_fee
                pnl = calc_shrt_pnl(shrt_pprice, shrt_closes[0][1], shrt_closes[0][0], inverse, c_mult)
                balance = balance + fee_paid + pnl
                shrt_psize, shrt_pprice = shrt_closes[0][2:4]
                equity = balance + calc_upnl(long_psize, long_pprice, shrt_psize, shrt_pprice,
                                             prices[k], inverse, c_mult)
                pbr = qty_to_cost(shrt_psize, shrt_pprice, inverse, c_mult) / balance
                fills.append((k, pnl, fee_paid, balance, equity, pbr) + shrt_closes[0])
                shrt_closes = shrt_closes[1:]
                next_update_ts = min(next_update_ts, timestamps[k] + latency_simulation_ms)
            ob[0] = prices[k]
        else:
            while shrt_entry[0] != 0.0 and prices[k] > shrt_entry[1]:
                fee_paid = -qty_to_cost(shrt_entry[0], shrt_entry[1], inverse, c_mult) * maker_fee
                balance += fee_paid
                shrt_psize, shrt_pprice = shrt_entry[2:4]
                equity = balance + calc_upnl(long_psize, long_pprice, shrt_psize, shrt_pprice,
                                             prices[k], inverse, c_mult)
                pbr = qty_to_cost(shrt_psize, shrt_pprice, inverse, c_mult) / balance
                fills.append((k, 0.0, fee_paid, balance, equity, pbr) + shrt_entry)
                next_update_ts = min(next_update_ts, timestamps[k] + latency_simulation_ms)
                shrt_entry, _ = calc_shrt_orders(balance,
                                                 shrt_psize,
                                                 shrt_pprice,
                                                 prev_ob[0],
                                                 prev_ob[1],
                                                 emas[prev_k],
                                                 ratios[prev_k],
                                                 available_margin,

                                                 inverse,
                                                 qty_step,
                                                 price_step,
                                                 min_qty,
                                                 min_cost,
                                                 c_mult,
                                                 stop_psize_pct,
                                                 stop_PBr_thr[1],
                                                 iqty_const[1],
                                                 iprc_const[1],
                                                 rqty_const[1],
                                                 rprc_const[1],
                                                 markup_const[1],
                                                 iqty_MAr_coeffs[1],
                                                 iprc_MAr_coeffs[1],
                                                 rprc_PBr_coeffs[1],
                                                 rqty_MAr_coeffs[1],
                                                 rprc_MAr_coeffs[1],
                                                 markup_MAr_coeffs[1])
            while long_closes and long_psize != 0.0 and long_closes[0][0] != 0.0 and prices[k] > long_closes[0][1]:
                fee_paid = -qty_to_cost(long_closes[0][0], long_closes[0][1], inverse, c_mult) * maker_fee
                pnl = calc_long_pnl(long_pprice, long_closes[0][1], long_closes[0][0], inverse, c_mult)
                balance = balance + fee_paid + pnl
                long_psize, long_pprice = long_closes[0][2:4]
                equity = balance + calc_upnl(long_psize, long_pprice, shrt_psize, shrt_pprice,
                                             prices[k], inverse, c_mult)
                pbr = qty_to_cost(shrt_psize, shrt_pprice, inverse, c_mult) / balance
                fills.append((k, pnl, fee_paid, balance, equity, pbr) + long_closes[0])
                long_closes = long_closes[1:]
                next_update_ts = min(next_update_ts, timestamps[k] + latency_simulation_ms)
            ob[1] = prices[k]
    return fills, True


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

