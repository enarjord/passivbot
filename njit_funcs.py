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
def calc_bid_ask_thresholds(prices: np.ndarray, MAs: np.ndarray, iprc_const, iprc_MAr_coeffs):
    bids = np.zeros(len(prices))
    asks = np.zeros(len(prices))
    for i in range(len(prices)):
        ratios = np.append(prices[i], MAs[i][:-1]) / MAs[i]
        bids[i] = MAs[i].min() * (iprc_const[0] + eqf(ratios, iprc_MAr_coeffs[0]))
        asks[i] = MAs[i].max() * (iprc_const[1] + eqf(ratios, iprc_MAr_coeffs[1]))
    return bids, asks


@njit
def calc_samples(ticks: np.ndarray, ms_sample_size: int = 1000) -> np.ndarray:
    # ticks [[qty, price, timestamp]]
    sampled_timestamps = np.arange(ticks[:, 2][0] // ms_sample_size * ms_sample_size,
                                   ticks[:, 2][-1] // ms_sample_size * ms_sample_size,
                                   ms_sample_size)
    samples = np.zeros((len(sampled_timestamps), 3))
    samples[:, 2] = sampled_timestamps
    i = 0
    ts = sampled_timestamps[0]
    k = 0
    while i < len(ticks):
        if ts == samples[:, 2][k]:
            samples[:,0][k] += ticks[:, 0][i]
            samples[:,1][k] = ticks[:, 1][i]
            i += 1
            ts = ticks[:, 2][i] // ms_sample_size * ms_sample_size
        else:
            k += 1
            if k >= len(samples):
                break
            samples[:, 1][k] = samples[:, 1][k - 1]
    return samples


@njit
def calc_emas(xs, spans):
    emas = np.zeros((len(xs), len(spans)))#, dtype=np.float32)
    alphas = 2 / (spans + 1)
    alphas_ = 1 - alphas
    emas[0] = xs[0]
    for i in range(1, len(xs)):
        emas[i] = emas[i - 1] * alphas_ + xs[i] * alphas
    return emas


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
        equity += calc_long_pnl(long_pprice, last_price, long_psize, inverse, c_mult)
    if shrt_pprice and shrt_psize:
        equity += calc_shrt_pnl(shrt_pprice, last_price, shrt_psize, inverse, c_mult)
    return equity


@njit
def calc_available_margin(balance,
                          long_psize,
                          long_pprice,
                          shrt_psize,
                          shrt_pprice,
                          last_price,
                          inverse, c_mult, max_leverage) -> float:
    used_margin = 0.0
    equity = balance
    if long_pprice and long_psize:
        equity += calc_long_pnl(long_pprice, last_price, long_psize, inverse, c_mult)
        used_margin += qty_to_cost(long_psize, long_pprice, inverse, c_mult)
    if shrt_pprice and shrt_psize:
        equity += calc_shrt_pnl(shrt_pprice, last_price, shrt_psize, inverse, c_mult)
        used_margin += qty_to_cost(shrt_psize, shrt_pprice, inverse, c_mult)
    return max(0.0, equity * max_leverage - used_margin)


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
                     MA_band_lower,
                     MA_band_upper,
                     MA_ratios,
                     available_margin,
 
                     spot,
                     inverse,
                     qty_step,
                     price_step,
                     min_qty,
                     min_cost,
                     c_mult,
                     pbr_stop_loss,
                     pbr_limit,
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
                     markup_MAr_coeffs) -> ((float, float, float, float, str), (float, float, float, float, str)):
    entry_price = min(highest_bid, round_dn(MA_band_lower * (iprc_const + eqf(MA_ratios, iprc_MAr_coeffs)), price_step))
    if long_psize == 0.0 or (spot and (long_psize < calc_min_entry_qty(long_pprice, inverse, qty_step, min_qty, min_cost))):
        min_entry_qty = calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost)
        max_entry_qty = cost_to_qty(min(balance * (pbr_limit + max(0.0, pbr_stop_loss)), available_margin),
                                    entry_price, inverse, c_mult)
        base_entry_qty = cost_to_qty(balance, entry_price, inverse, c_mult) * (iqty_const + eqf(MA_ratios, iqty_MAr_coeffs))
        entry_qty = max(min_entry_qty, round_dn(min(max_entry_qty, base_entry_qty), qty_step))
        entry_type = 'long_ientry'
        long_close = (0.0, 0.0, 0.0, 0.0, 'long_nclose')
    elif long_psize > 0.0:
        pbr = qty_to_cost(long_psize, long_pprice, inverse, c_mult) / balance
        entry_price = min(entry_price,
                          round_dn(long_pprice * (rprc_const + eqf(MA_ratios, rprc_MAr_coeffs) +
                                                  eqf(np.array([pbr]), rprc_PBr_coeffs, minus=0.0)), price_step))
        min_entry_qty = calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost)
        max_entry_qty = cost_to_qty(min(balance * (pbr_limit + max(0.0, pbr_stop_loss) - pbr), available_margin),
                                    entry_price, inverse, c_mult)
        base_entry_qty = cost_to_qty(balance, entry_price, inverse, c_mult) * (iqty_const + eqf(MA_ratios, iqty_MAr_coeffs))
        entry_qty = round_dn(min(max_entry_qty,
                                 base_entry_qty + (long_psize * (rqty_const + eqf(MA_ratios, rqty_MAr_coeffs)))), qty_step)
        nclose_price = max(lowest_ask, round_up(long_pprice * (markup_const + eqf(MA_ratios, markup_MAr_coeffs)), price_step))
        if entry_qty < min_entry_qty:
            entry_qty = 0.0
        if pbr_stop_loss < 0.0:
            # v3.6.2 behavior
            close_price = max(lowest_ask, min(nclose_price, round_up(MA_band_upper, price_step)))
            close_type = 'long_nclose' if close_price > long_pprice else 'long_sclose'
            long_close = (-long_psize, close_price, 0.0, 0.0, close_type)
        elif pbr_stop_loss == 0.0:
            long_close = (-long_psize, nclose_price, 0.0, 0.0, 'long_nclose')
        else:
            # v3.6.1 behavior
            if pbr > pbr_limit:
                sclose_price = max(lowest_ask, round_up(MA_band_upper, price_step))
                sclose_qty = -min(long_psize, max(min_qty, round_dn(cost_to_qty(balance * min(1.0, pbr - pbr_limit),
                                                                                sclose_price, inverse, c_mult), qty_step)))
                if sclose_price >= nclose_price:
                    long_close = (-long_psize, nclose_price, 0.0, 0.0, 'long_nclose')
                else:
                    long_close = (sclose_qty, sclose_price, round_(long_psize + sclose_qty, qty_step), long_pprice, 'long_sclose')
            else:
                entry_qty = max(entry_qty, min_entry_qty)
                long_close = (-long_psize, nclose_price, 0.0, 0.0, 'long_nclose')
        entry_type = 'long_rentry'
    else:
        raise Exception('long psize is less than 0.0')

    if spot:
        if entry_qty != 0.0:
            equity = calc_equity(balance, long_psize, long_pprice, 0.0, 0.0, highest_bid, inverse, c_mult)
            excess_cost = max(0.0, qty_to_cost(long_psize + entry_qty, highest_bid, inverse, c_mult) - equity)
            if excess_cost:
                entry_qty = round_dn((qty_to_cost(entry_qty, entry_price, inverse, c_mult) - excess_cost) / entry_price, qty_step)
                if entry_qty < min_entry_qty:
                    entry_qty = 0.0
        if long_close[0] != 0.0:
            min_close_qty = calc_min_entry_qty(long_close[1], inverse, qty_step, min_qty, min_cost)
            close_qty = round_dn(min(long_psize, max(min_close_qty, abs(long_close[0]))), qty_step)
            if close_qty < min_close_qty:
                long_close = (0.0, 0.0, 0.0, 0.0, 'long_nclose')
            else:
                long_close = (-close_qty,) + long_close[1:]

    new_psize, new_pprice = calc_new_psize_pprice(long_psize, long_pprice, entry_qty, entry_price, qty_step)
    return (entry_qty, entry_price, new_psize, new_pprice, entry_type), long_close


@njit
def calc_shrt_orders(balance,
                     shrt_psize,
                     shrt_pprice,
                     highest_bid,
                     lowest_ask,
                     MA_band_lower,
                     MA_band_upper,
                     MA_ratios,
                     available_margin,
 
                     spot,
                     inverse,
                     qty_step,
                     price_step,
                     min_qty,
                     min_cost,
                     c_mult,
                     pbr_stop_loss,
                     pbr_limit,
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
    entry_price = max(lowest_ask, round_up(MA_band_upper * (iprc_const + eqf(MA_ratios, iprc_MAr_coeffs)), price_step))
    if shrt_psize == 0.0:
        min_entry_qty = calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost)
        max_entry_qty = cost_to_qty(min(balance * (pbr_limit + max(0.0, pbr_stop_loss)), available_margin),
                                    entry_price, inverse, c_mult)
        base_entry_qty = cost_to_qty(balance, entry_price, inverse, c_mult) * (iqty_const + eqf(MA_ratios, iqty_MAr_coeffs))
        entry_qty = max(min_entry_qty, round_dn(min(max_entry_qty, base_entry_qty), qty_step))
        entry_type = 'shrt_ientry'
        shrt_close = (0.0, 0.0, 0.0, 0.0, 'shrt_nclose')
    elif shrt_psize < 0.0:
        pbr = qty_to_cost(shrt_psize, shrt_pprice, inverse, c_mult) / balance
        entry_price = max(entry_price,
                          round_up(shrt_pprice * (rprc_const + eqf(MA_ratios, rprc_MAr_coeffs) +
                                                  eqf(np.array([pbr]), rprc_PBr_coeffs, minus=0.0)), price_step))
        min_entry_qty = calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost)
        max_entry_qty = cost_to_qty(min(balance * (pbr_limit + max(0.0, pbr_stop_loss) - pbr), available_margin),
                                    entry_price, inverse, c_mult)

        base_entry_qty = cost_to_qty(balance, entry_price, inverse, c_mult) * (iqty_const + eqf(MA_ratios, iqty_MAr_coeffs))
        entry_qty = round_dn(min(max_entry_qty,
                                 base_entry_qty + (-shrt_psize * (rqty_const + eqf(MA_ratios, rqty_MAr_coeffs)))), qty_step)
        nclose_price = round_dn(shrt_pprice * (markup_const + eqf(MA_ratios, markup_MAr_coeffs)), price_step)
        if entry_qty < min_entry_qty:
            entry_qty = 0.0
        if pbr_stop_loss < 0.0:
            # v3.6.2 behavior
            close_price = min(highest_bid, max(nclose_price, round_dn(MA_band_lower, price_step)))
            close_type = 'shrt_nclose' if close_price < shrt_pprice else 'shrt_sclose'
            shrt_close = (-shrt_psize, close_price, 0.0, 0.0, close_type)
        elif pbr_stop_loss == 0.0:
            shrt_close = (-shrt_psize, nclose_price, 0.0, 0.0, 'shrt_nclose')
        else:
            # v3.6.1 beahvior
            if pbr > pbr_limit:
                sclose_price = min(highest_bid, round_dn(MA_band_lower, price_step))
                sclose_qty = min(-shrt_psize, max(min_qty, round_dn(cost_to_qty(balance * min(1.0, pbr - pbr_limit),
                                                                                sclose_price, inverse, c_mult), qty_step)))
                if sclose_price <= nclose_price:
                    shrt_close = (-shrt_psize, nclose_price, 0.0, 0.0, 'shrt_nclose')
                else:
                    shrt_close = (sclose_qty, sclose_price, round_(shrt_psize + sclose_qty, qty_step), shrt_pprice, 'shrt_sclose')
            else:
                entry_qty = max(entry_qty, min_entry_qty)
                shrt_close = (-shrt_psize, nclose_price, 0.0, 0.0, 'shrt_nclose')

        entry_type = 'shrt_rentry'
    else:
        raise Exception('shrt psize is greater than 0.0 ')
    entry_qty = -entry_qty
    new_psize, new_pprice = calc_new_psize_pprice(shrt_psize, shrt_pprice, entry_qty, entry_price, qty_step)
    return (entry_qty, entry_price, new_psize, new_pprice, entry_type), shrt_close


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
                MAs,
 
                spot,
                hedge_mode,
                inverse,
                do_long,
                do_shrt,
                qty_step,
                price_step,
                min_qty,
                min_cost,
                c_mult,
                max_leverage,
                spans,
                pbr_stop_loss,
                pbr_limit,
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
    MA_ratios = np.append(last_price, MAs[:-1]) / MAs
    MA_band_lower = MAs.min()
    MA_band_upper = MAs.max()
    available_margin = calc_available_margin(balance, long_psize, long_pprice, shrt_psize, shrt_pprice,
                                             last_price, inverse, c_mult, max_leverage)
    if hedge_mode:
        do_long_ = do_long
        do_shrt_ = do_shrt
    else:
        no_pos = long_psize == 0.0 and shrt_psize == 0.0
        do_long_ = (no_pos and do_long) or long_psize != 0.0
        do_shrt_ = (no_pos and do_shrt) or shrt_psize != 0.0
    long_entry, long_close = calc_long_orders(balance,
                     long_psize,
                     long_pprice,
                     highest_bid,
                     lowest_ask,
                     MA_band_lower,
                     MA_band_upper,
                     MA_ratios,
                     available_margin,

                     spot,
                     inverse,
                     qty_step,
                     price_step,
                     min_qty,
                     min_cost,
                     c_mult,
                     pbr_stop_loss[0],
                     pbr_limit[0],
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
                     markup_MAr_coeffs[0]) if (spot or do_long_) else ((0.0, 0.0, 0.0, 0.0, ''), (0.0, 0.0, 0.0, 0.0, ''))
    shrt_entry, shrt_close = calc_shrt_orders(balance,
                     shrt_psize,
                     shrt_pprice,
                     highest_bid,
                     lowest_ask,
                     MA_band_lower,
                     MA_band_upper,
                     MA_ratios,
                     available_margin,

                     spot,
                     inverse,
                     qty_step,
                     price_step,
                     min_qty,
                     min_cost,
                     c_mult,
                     pbr_stop_loss[1],
                     pbr_limit[1],
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
                     markup_MAr_coeffs[1]) if (do_shrt_ and not spot) else ((0.0, 0.0, 0.0, 0.0, ''), (0.0, 0.0, 0.0, 0.0, ''))
    bkr_price = calc_bankruptcy_price(balance, long_psize, long_pprice, shrt_psize, shrt_pprice, inverse, c_mult)
    return long_entry, shrt_entry, long_close, shrt_close, bkr_price, available_margin



@njit
def calc_emas_last(xs, spans):
    alphas = 2.0 / (spans + 1.0)
    alphas_ = 1.0 - alphas
    emas = np.repeat(xs[0], len(spans))
    for i in range(1, len(xs)):
        emas = emas * alphas_ + xs[i] * alphas
    return emas


@njit
def njit_backtest(data: (np.ndarray, np.ndarray, np.ndarray),
                  starting_balance,
                  latency_simulation_ms,
                  maker_fee,
                  spot,
                  hedge_mode,
                  inverse,
                  do_long,
                  do_shrt,
                  qty_step,
                  price_step,
                  min_qty,
                  min_cost,
                  c_mult,
                  max_leverage,
                  spans,
                  pbr_stop_loss,
                  pbr_limit,
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

    prices, buyer_maker, timestamps = data
    static_params = (spot, hedge_mode, inverse, do_long, do_shrt, qty_step, price_step, min_qty, min_cost,
                     c_mult, max_leverage, spans, pbr_stop_loss, pbr_limit, iqty_const, iprc_const,
                     rqty_const, rprc_const, markup_const, iqty_MAr_coeffs, iprc_MAr_coeffs, rprc_PBr_coeffs,
                     rqty_MAr_coeffs, rprc_MAr_coeffs, markup_MAr_coeffs)

    balance = equity = starting_balance
    long_psize, long_pprice, shrt_psize, shrt_pprice = 0.0, 0.0, 0.0, 0.0
    next_update_ts = 0
    ob = [prices[0], prices[0]]
    fills = []

    long_entry = shrt_entry = long_close = shrt_close = (0.0, 0.0, 0.0, 0.0, '')
    bkr_price, available_margin = 0.0, 0.0

    prev_k = 0
    prev_ob = [0.0, 0.0]
    closest_bkr = 1.0
    lowest_eqbal_ratio = 1.0
    alphas = 2.0 / (spans + 1.0)
    alphas_ = 1.0 - alphas
    MAs = calc_emas_last(prices[:spans.max()], spans)
    for k in range(spans.max(), len(prices)):

        closest_bkr = min(closest_bkr, calc_diff(bkr_price, prices[k]))
        if timestamps[k] > next_update_ts:
            long_entry, shrt_entry, long_close, shrt_close, bkr_price, available_margin = calc_orders(
                balance,
                long_psize,
                long_pprice,
                shrt_psize,
                shrt_pprice,
                ob[0],
                ob[1],
                prices[k],
                MAs,

                *static_params)
            equity = balance + calc_upnl(long_psize, long_pprice, shrt_psize, shrt_pprice,
                                         prices[k], inverse, c_mult)
            lowest_eqbal_ratio = min(lowest_eqbal_ratio, equity / balance)
            next_update_ts = timestamps[k] + 5000
            prev_k = k
            prev_MAs = MAs
            prev_ob = ob

            if equity / starting_balance < 0.1:
                return fills, (False, lowest_eqbal_ratio, closest_bkr)

            if closest_bkr < 0.06:
                if long_psize != 0.0:
                    fee_paid = -qty_to_cost(long_psize, long_pprice, inverse, c_mult) * maker_fee
                    pnl = calc_long_pnl(long_pprice, prices[k], -long_psize, inverse, c_mult)
                    balance = 0.0
                    equity = 0.0
                    long_psize, long_pprice = 0.0, 0.0
                    fills.append((k, timestamps[k], pnl, fee_paid, balance, equity, 0.0, -long_psize, prices[k], 0.0, 0.0, 'long_bankruptcy'))
                if shrt_psize != 0.0:
    
                    fee_paid = -qty_to_cost(shrt_psize, shrt_pprice, inverse, c_mult) * maker_fee
                    pnl = calc_shrt_pnl(shrt_pprice, prices[k], -shrt_psize, inverse, c_mult)
                    balance, equity = 0.0, 0.0
                    shrt_psize, shrt_pprice = 0.0, 0.0
                    fills.append((k, timestamps[k], pnl, fee_paid, balance, equity, 0.0, -shrt_psize, prices[k], 0.0, 0.0, 'shrt_bankruptcy'))
    
                return fills, (False, lowest_eqbal_ratio, closest_bkr)

        if buyer_maker[k]:
            while long_entry[0] != 0.0 and prices[k] < long_entry[1]:
                fee_paid = -qty_to_cost(long_entry[0], long_entry[1], inverse, c_mult) * maker_fee
                balance += fee_paid
                long_psize, long_pprice = calc_new_psize_pprice(long_psize, long_pprice, long_entry[0],
                                                                long_entry[1], qty_step)
                equity = balance + calc_upnl(long_psize, long_pprice, shrt_psize, shrt_pprice,
                                             prices[k], inverse, c_mult)
                pbr = qty_to_cost(long_psize, long_pprice, inverse, c_mult) / balance
                fills.append((k, timestamps[k], 0.0, fee_paid, balance, equity, pbr) + long_entry)
                next_update_ts = min(next_update_ts, timestamps[k] + latency_simulation_ms)
                long_entry, _ = calc_long_orders(balance,
                                                 long_psize,
                                                 long_pprice,
                                                 prev_ob[0],
                                                 prev_ob[1],
                                                 prev_MAs.min(),
                                                 prev_MAs.max(),
                                                 np.append(prices[prev_k], prev_MAs[:-1]) / prev_MAs,
                                                 available_margin,

                                                 spot,
                                                 inverse,
                                                 qty_step,
                                                 price_step,
                                                 min_qty,
                                                 min_cost,
                                                 c_mult,
                                                 pbr_stop_loss[0],
                                                 pbr_limit[0],
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
            if shrt_psize != 0.0 and shrt_close[0] != 0.0 and prices[k] < shrt_close[1]:
                if shrt_close[0] > -shrt_psize:
                    print('warning: shrt close qty greater than shrt psize')
                    print('shrt_psize', shrt_psize)
                    print('shrt_pprice', shrt_pprice)
                    print('shrt_close', shrt_close)
                    shrt_close = (-shrt_psize,) + shrt_close[1:]
                fee_paid = -qty_to_cost(shrt_close[0], shrt_close[1], inverse, c_mult) * maker_fee
                pnl = calc_shrt_pnl(shrt_pprice, shrt_close[1], shrt_close[0], inverse, c_mult)
                balance = balance + fee_paid + pnl
                shrt_psize = round_(shrt_psize + shrt_close[0], qty_step)
                equity = balance + calc_upnl(long_psize, long_pprice, shrt_psize, shrt_pprice,
                                             prices[k], inverse, c_mult)
                pbr = qty_to_cost(shrt_psize, shrt_pprice, inverse, c_mult) / balance
                fills.append((k, timestamps[k], pnl, fee_paid, balance, equity, pbr) + shrt_close)
                shrt_close = (0.0, 0.0, 0.0, 0.0, '')
                next_update_ts = min(next_update_ts, timestamps[k] + latency_simulation_ms)
            ob[0] = prices[k]
        else:
            while shrt_entry[0] != 0.0 and prices[k] > shrt_entry[1]:
                fee_paid = -qty_to_cost(shrt_entry[0], shrt_entry[1], inverse, c_mult) * maker_fee
                balance += fee_paid
                shrt_psize, shrt_pprice = calc_new_psize_pprice(shrt_psize, shrt_pprice, shrt_entry[0],
                                                                shrt_entry[1], qty_step)
                equity = balance + calc_upnl(long_psize, long_pprice, shrt_psize, shrt_pprice,
                                             prices[k], inverse, c_mult)
                pbr = qty_to_cost(shrt_psize, shrt_pprice, inverse, c_mult) / balance
                fills.append((k, timestamps[k], 0.0, fee_paid, balance, equity, pbr) + shrt_entry)
                next_update_ts = min(next_update_ts, timestamps[k] + latency_simulation_ms)
                shrt_entry, _ = calc_shrt_orders(balance,
                                                 shrt_psize,
                                                 shrt_pprice,
                                                 prev_ob[0],
                                                 prev_ob[1],
                                                 prev_MAs.min(),
                                                 prev_MAs.max(),
                                                 np.append(prices[prev_k], prev_MAs[:-1]) / prev_MAs,
                                                 available_margin,

                                                 spot,
                                                 inverse,
                                                 qty_step,
                                                 price_step,
                                                 min_qty,
                                                 min_cost,
                                                 c_mult,
                                                 pbr_stop_loss[1],
                                                 pbr_limit[1],
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
            if long_psize != 0.0 and long_close[0] != 0.0 and prices[k] > long_close[1]:
                if -long_close[0] > long_psize:
                    print('warning: long close qty greater than long psize')
                    print('long_psize', long_psize)
                    print('long_pprice', long_pprice)
                    print('long_close', long_close)
                    long_close = (-long_psize,) + long_close[1:]
                fee_paid = -qty_to_cost(long_close[0], long_close[1], inverse, c_mult) * maker_fee
                pnl = calc_long_pnl(long_pprice, long_close[1], long_close[0], inverse, c_mult)
                balance = balance + fee_paid + pnl
                long_psize = round_(long_psize + long_close[0], qty_step)
                equity = balance + calc_upnl(long_psize, long_pprice, shrt_psize, shrt_pprice,
                                             prices[k], inverse, c_mult)
                pbr = qty_to_cost(long_psize, long_pprice, inverse, c_mult) / balance
                fills.append((k, timestamps[k], pnl, fee_paid, balance, equity, pbr) + long_close)

                long_close = (0.0, 0.0, 0.0, 0.0, '')
                next_update_ts = min(next_update_ts, timestamps[k] + latency_simulation_ms)
            ob[1] = prices[k]
        MAs = MAs * alphas_ + prices[k] * alphas
    return fills, (True, lowest_eqbal_ratio, closest_bkr)


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

