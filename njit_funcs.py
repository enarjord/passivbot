import os
import numpy as np


if 'NOJIT' in os.environ and os.environ['NOJIT'] == 'true':
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
def cost_to_qty(cost, price, inverse, c_mult):
    return cost * price / c_mult if inverse else (cost / price if price > 0.0 else 0.0)


@njit
def qty_to_cost(qty, price, inverse, c_mult) -> float:
    return (abs(qty / price) if price > 0.0 else 0.0) * c_mult if inverse else abs(qty * price)


@njit
def calc_ema(alpha, alpha_, prev_ema, new_val) -> float:
    return prev_ema * alpha_ + new_val * alpha


@njit
def calc_samples(ticks: np.ndarray, sample_size_ms: int = 1000) -> np.ndarray:
    # ticks [[timestamp, qty, price]]
    sampled_timestamps = np.arange(ticks[0][0] // sample_size_ms * sample_size_ms,
                                   ticks[-1][0] // sample_size_ms * sample_size_ms + sample_size_ms,
                                   sample_size_ms)
    samples = np.zeros((len(sampled_timestamps), 3))
    samples[:, 0] = sampled_timestamps
    ts = sampled_timestamps[0]
    i = 0
    k = 0
    while True:
        if ts == samples[k][0]:
            samples[k][1] += ticks[i][1]
            samples[k][2] = ticks[i][2]
            i += 1
            if i >= len(ticks):
                break
            ts = ticks[i][0] // sample_size_ms * sample_size_ms
        else:
            k += 1
            if k >= len(samples):
                break
            samples[k][2] = samples[k - 1][2]
    return samples


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
def calc_new_psize_pprice(psize, pprice, qty, price, qty_step) -> (float, float):
    if qty == 0.0:
        return psize, pprice
    new_psize = round_(psize + qty, qty_step)
    if new_psize == 0.0:
        return 0.0, 0.0
    return new_psize, nan_to_0(pprice) * (psize / new_psize) + price * (qty / new_psize)


@njit
def calc_pbr_if_filled(balance, psize, pprice, qty, price, inverse, c_mult, qty_step):
    psize, qty = abs(psize), abs(qty)
    new_psize, new_pprice = calc_new_psize_pprice(psize, pprice, qty, price, qty_step)
    return qty_to_cost(new_psize, new_pprice, inverse, c_mult) / balance


@njit
def calc_long_close_grid(balance,
                         long_psize,
                         long_pprice,
                         lowest_ask,

                         spot,
                         inverse,
                         qty_step,
                         price_step,
                         min_qty,
                         min_cost,
                         c_mult,

                         initial_qty_pct,
                         min_markup,
                         markup_range,
                         n_close_orders) -> [(float, float, str)]:
    if long_psize == 0.0:
        return [(0.0, 0.0, '')]
    minm = long_pprice * (1 + min_markup)
    if spot and round_dn(long_psize, qty_step) < calc_min_entry_qty(minm, inverse, qty_step, min_qty, min_cost):
        return [(0.0, 0.0, '')]
    close_prices = []
    for p in np.linspace(minm, long_pprice * (1 + min_markup + markup_range), n_close_orders):
        price_ = round_up(p, price_step)
        if price_ >= lowest_ask:
            close_prices.append(price_)
    if len(close_prices) == 0:
        return [(-long_psize, lowest_ask, 'long_nclose')]
    elif len(close_prices) == 1:
        return [(-long_psize, close_prices[0], 'long_nclose')]
    else:
        min_close_qty = calc_min_entry_qty(close_prices[0], inverse, qty_step, min_qty, min_cost)
        default_qty = round_dn(long_psize / len(close_prices), qty_step)
        if default_qty == 0.0:
            return [(-long_psize, close_prices[0], 'long_nclose')]
        default_qty = max(min_close_qty, default_qty)
        long_closes = []
        remaining = long_psize
        for close_price in close_prices:
            if remaining < max([min_close_qty,
                                cost_to_qty(balance, close_price, inverse, c_mult) * initial_qty_pct * 0.5,
                                default_qty * 0.5]):
                break
            close_qty = min(remaining, max(default_qty, min_close_qty))
            long_closes.append((-close_qty, close_price, 'long_nclose'))
            remaining = round_(remaining - close_qty, qty_step)
        if remaining:
            if long_closes:
                long_closes[-1] = (round_(long_closes[-1][0] - remaining, qty_step), long_closes[-1][1], long_closes[-1][2])
            else:
                long_closes = [(-long_psize, close_prices[0], 'long_nclose')]
        return long_closes


@njit
def calc_shrt_close_grid(balance,
                         shrt_psize,
                         shrt_pprice,
                         highest_bid,

                         spot,
                         inverse,
                         qty_step,
                         price_step,
                         min_qty,
                         min_cost,
                         c_mult,
                         max_leverage,

                         initial_qty_pct,
                         min_markup,
                         markup_range,
                         n_close_orders) -> [(float, float, str)]:
    if shrt_psize == 0.0:
        return [(0.0, 0.0, '')]
    minm = shrt_pprice * (1 - min_markup)
    close_prices = []
    for p in np.linspace(minm, shrt_pprice * (1 - min_markup - markup_range), n_close_orders):
        price_ = round_dn(p, price_step)
        if price_ <= highest_bid:
            close_prices.append(price_)
    if len(close_prices) == 0:
        return [(-shrt_psize, highest_bid, 'shrt_nclose')]
    elif len(close_prices) == 1:
        return [(-shrt_psize, close_prices[0], 'shrt_nclose')]
    else:
        min_close_qty = calc_min_entry_qty(close_prices[-1], inverse, qty_step, min_qty, min_cost)
        default_qty = round_dn(-shrt_psize / len(close_prices), qty_step)
        if default_qty == 0.0:
            return [(-shrt_psize, close_prices[0], 'shrt_nclose')]
        default_qty = max(min_close_qty, default_qty)
        shrt_closes = []
        remaining = -shrt_psize
        for close_price in close_prices:
            if remaining < max([min_close_qty,
                                cost_to_qty(balance, close_price, inverse, c_mult) * initial_qty_pct * 0.5,
                                default_qty * 0.5]):
                break
            close_qty = min(remaining, default_qty)
            shrt_closes.append((close_qty, close_price, 'shrt_nclose'))
            remaining = round_(remaining - close_qty, qty_step)
        if remaining:
            if shrt_closes:
                shrt_closes[-1] = (round_(shrt_closes[-1][0] + remaining, qty_step), shrt_closes[-1][1], shrt_closes[-1][2])
            else:
                shrt_closes = [(-shrt_psize, close_prices[0], 'shrt_nclose')]
        return shrt_closes


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
def calc_emas_last(xs, spans):
    alphas = 2.0 / (spans + 1.0)
    alphas_ = 1.0 - alphas
    emas = np.repeat(xs[0], len(spans))
    for i in range(1, len(xs)):
        emas = emas * alphas_ + xs[i] * alphas
    return emas


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


@njit
def basespace(start, end, base, n):
    if base == 1.0:
        return np.linspace(start, end, n)
    a = np.array([base**i for i in range(n)])
    a = ((a - a.min()) / (a.max() - a.min()))
    return a * (end - start) + start


@njit
def calc_m_b(x0, x1, y0, y1):
    denom = x1 - x0
    if denom == 0.0:
        # zero div, return high number
        m = 9.0e32
    else:
        m = (y1 - y0) / (x1 - x0)
    return m, y0 - m * x0


@njit
def calc_long_entry_qty(psize, pprice, entry_price, eprice_pprice_diff):
    return -(psize * (entry_price * eprice_pprice_diff + entry_price - pprice) / (entry_price * eprice_pprice_diff))


@njit
def find_qty_bringing_pbr_to_target(
        balance,
        psize,
        pprice,
        pbr_limit,
        entry_price,
        inverse,
        qty_step,
        c_mult,
        error_tolerance=0.01,
        max_n_iters=10) -> float:
    guesses = [cost_to_qty(balance * x, entry_price, inverse, c_mult) for x in [0.1, 0.11]]
    vals = [calc_pbr_if_filled(balance, psize, pprice, guess, entry_price, inverse, c_mult, qty_step) for guess in guesses]
    diffs = sorted([(abs(val - pbr_limit) / pbr_limit, guess, val) for guess, val in zip(guesses, vals)])[:2]
    i = 0
    while True:
        i += 1
        if diffs[0][0] < error_tolerance:
            return round_(diffs[0][1], qty_step)
        if i >= max_n_iters:
            if abs(diffs[0][2] - pbr_limit) / pbr_limit > 0.1:
                print('warning, in find_qty_bringing_pbr_to_target diff between pbr limit and val greater than 10%\n',
                      diffs, pbr_limit)
            return round_(diffs[0][1], qty_step)
        m, b = calc_m_b(diffs[0][2], diffs[1][2], diffs[0][1], diffs[1][1])
        guess = max(0.0, m * pbr_limit + b)
        val = calc_pbr_if_filled(balance, psize, pprice, guess, entry_price, inverse, c_mult, qty_step)
        if guess in guesses:
            if val > pbr_limit:
                guess *= 2.0
            else:
                guess *= 0.5
            val = calc_pbr_if_filled(balance, psize, pprice, guess, entry_price, inverse, c_mult, qty_step)
        guesses.append(guess)
        diffs = sorted([(abs(val - pbr_limit) / pbr_limit, guess, val)] + diffs)[:2]



@njit
def find_eprice_pprice_diff_pbr_weighting(
        balance,
        initial_entry_price,
    
        inverse,
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
    
        grid_span,
        pbr_limit,
        max_n_entry_orders,
        initial_qty_pct,
        eprice_pprice_diff,
        eprice_exp_base=1.618034,
        max_n_iters=20,
        error_tolerance=0.01,
        eprices=None,
        prev_pprice=None):
    guesses = [1.0, 1.01]
    vals = [eval_long_entry_grid(balance, initial_entry_price, inverse, qty_step, price_step, min_qty,
                                 min_cost, c_mult, grid_span, max_n_entry_orders, initial_qty_pct,
                                 eprice_pprice_diff, guess, eprice_exp_base=eprice_exp_base,
                                 eprices=eprices, prev_pprice=prev_pprice)[-1][4]
            for guess in guesses]
    diffs = sorted([(abs(val - pbr_limit) / pbr_limit, guess, val) for guess, val in zip(guesses, vals)])[:2]
    i = 0
    while True:
        i += 1
        if diffs[0][0] < error_tolerance:
            return diffs[0][1]
        if i >= max_n_iters:
            if abs(diffs[0][2] - pbr_limit) / pbr_limit > 0.1:
                print('warning, in find_eprice_pprice_diff_pbr_weighting diff between pbr limit and val greater than 10%\n',
                      diffs, pbr_limit)
            return diffs[0][1]
        if diffs[0][2] == diffs[1][2]:
            return diffs[0][1]
        m, b = calc_m_b(diffs[0][2], diffs[1][2], diffs[0][1], diffs[1][1])
        guess = max(0.0, m * pbr_limit + b)
        
        val = eval_long_entry_grid(balance, initial_entry_price, inverse, qty_step, price_step, min_qty,
                                   min_cost, c_mult, grid_span, max_n_entry_orders, initial_qty_pct,
                                   eprice_pprice_diff, guess, eprice_exp_base=eprice_exp_base,
                                   eprices=eprices, prev_pprice=prev_pprice)[-1][4]
        if guess in guesses:
            if val > pbr_limit:
                new_guess = guess * 2.0
            else:
                new_guess = guess * 0.5
            guess = new_guess
            val = eval_long_entry_grid(balance, initial_entry_price, inverse, qty_step, price_step, min_qty,
                                       min_cost, c_mult, grid_span, max_n_entry_orders, initial_qty_pct,
                                       eprice_pprice_diff, guess, eprice_exp_base=eprice_exp_base,
                                       eprices=eprices, prev_pprice=prev_pprice)[-1][4]
        guesses.append(guess)
        diffs = sorted([(abs(val - pbr_limit) / pbr_limit, guess, val)] + diffs)[:2]


@njit
def eval_long_entry_grid(
        balance,
        initial_entry_price,
    
        inverse,
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,

        grid_span,
        max_n_entry_orders,
        initial_qty_pct,
        eprice_pprice_diff,
        eprice_pprice_diff_pbr_weighting,
        eprice_exp_base=1.618034,
        eprices=None,
        prev_pprice=None):
    
    # returns [qty, price, psize, pprice, pbr]
    if eprices is None:
        grid = np.zeros((max_n_entry_orders, 5))
        grid[:,1] = [round_dn(p, price_step)
                     for p in basespace(initial_entry_price, initial_entry_price * (1 - grid_span),
                                        eprice_exp_base, max_n_entry_orders)]
    else:
        max_n_entry_orders = len(eprices)
        grid = np.zeros((max_n_entry_orders, 5))
        grid[:,1] = eprices

    grid[0][0] = max(calc_min_entry_qty(grid[0][1], inverse, qty_step, min_qty, min_cost),
                     round_(balance * initial_qty_pct / initial_entry_price, qty_step))
    grid[0][2] = psize = grid[0][0]
    grid[0][3] = pprice = grid[0][1] if prev_pprice is None else prev_pprice
    grid[0][4] = qty_to_cost(psize, pprice, inverse, c_mult) / balance
    for i in range(1, max_n_entry_orders):
        adjusted_eprice_pprice_diff = eprice_pprice_diff * (1 + grid[i - 1][4] * eprice_pprice_diff_pbr_weighting)
        qty = round_(calc_long_entry_qty(psize, pprice, grid[i][1], adjusted_eprice_pprice_diff), qty_step)
        if qty < calc_min_entry_qty(grid[i][1], inverse, qty_step, min_qty, min_cost):
            qty = 0.0
        psize, pprice = calc_new_psize_pprice(psize, pprice, qty, grid[i][1], qty_step)
        grid[i][0] = qty
        grid[i][2:] = [psize, pprice, qty_to_cost(psize, pprice, inverse, c_mult) / balance]
    return grid


@njit
def calc_whole_long_entry_grid(
        balance,
        initial_entry_price,
    
        inverse,
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,

        grid_span,
        pbr_limit,
        max_n_entry_orders,
        initial_qty_pct,
        eprice_pprice_diff,
        eprice_exp_base=1.618034,
        eprices=None,
        prev_pprice=None):

    # [qty, price, psize, pprice, pbr]
    eprice_pprice_diff_pbr_weighting = find_eprice_pprice_diff_pbr_weighting(
        balance, initial_entry_price, inverse, qty_step, price_step, min_qty, min_cost,
        c_mult, grid_span, pbr_limit, max_n_entry_orders, initial_qty_pct, eprice_pprice_diff, eprice_exp_base,
        eprices=eprices, prev_pprice=prev_pprice
    )
    grid = eval_long_entry_grid(balance, initial_entry_price, inverse, qty_step, price_step,
                                min_qty, min_cost, c_mult, grid_span, max_n_entry_orders, initial_qty_pct,
                                eprice_pprice_diff, eprice_pprice_diff_pbr_weighting, eprice_exp_base,
                                eprices=eprices, prev_pprice=prev_pprice)
    return grid[grid[:,0] > 0.0]


@njit
def calc_long_entry_grid(
        balance,
        psize,
        pprice,
        highest_bid,
        
        inverse,
        do_long,
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,

        grid_span,
        pbr_limit,
        max_n_entry_orders,
        initial_qty_pct,
        eprice_pprice_diff,
        secondary_pbr_allocation,
        secondary_grid_spacing,
        eprice_exp_base=1.618034) -> [(float, float, str)]:
    entry_price = highest_bid
    min_entry_qty = calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost)
    if do_long or psize > min_entry_qty:
        entry_qty = round_(cost_to_qty(balance * initial_qty_pct, entry_price, inverse, c_mult), qty_step)
        if round_dn(psize, qty_step) < max(min_entry_qty, entry_qty * 0.45):
            max_entry_qty = round_dn(cost_to_qty(balance * pbr_limit, entry_price, inverse, c_mult), qty_step)
            entry_qty = min(max_entry_qty, max(min_entry_qty, entry_qty))
            if entry_qty < min_entry_qty:
                return [(0.0, 0.0, '')]
            return [(entry_qty, entry_price, 'long_ientry')]
        else:
            pbr = qty_to_cost(psize, pprice, inverse, c_mult) / balance
            if pbr > pbr_limit * 0.97:
                return [(0.0, 0.0, '')]
            primary_pbr_limit = pbr_limit * (1 - secondary_pbr_allocation)
            if secondary_pbr_allocation != 0.0 and pbr > primary_pbr_limit * 0.97:
                entry_price = round_dn(pprice * (1 - secondary_grid_spacing), price_step)
                qty = find_qty_bringing_pbr_to_target(balance, psize, pprice, pbr_limit, entry_price,
                                                      inverse, qty_step, c_mult)
                return [(qty, entry_price, 'secondary_long_rentry')]
            template_grid = calc_whole_long_entry_grid(
                balance, pprice, inverse, qty_step, price_step, min_qty, min_cost, c_mult, grid_span, primary_pbr_limit,
                max_n_entry_orders, initial_qty_pct, eprice_pprice_diff, eprice_exp_base=eprice_exp_base)
            closest_node = sorted([(abs(psize - g[2]) / psize, i) for i, g in enumerate(template_grid)])[0][1]
            ratio = pprice / template_grid[closest_node][3]
            eprices = np.flip(np.unique(np.array([min(highest_bid, round_(p * ratio, price_step))
                                                  for p in template_grid[closest_node:, 1]])))
            if len(eprices) <= 1:
                return [(0.0, 0.0, '')]
            elif len(eprices) == 2:
                qty = find_qty_bringing_pbr_to_target(balance, psize, pprice, primary_pbr_limit, eprices[-1], inverse, qty_step, c_mult)
                return [(qty, eprices[-1], 'primary_long_rentry')]
            fitted_grid = calc_whole_long_entry_grid(
                balance, pprice, inverse, qty_step, price_step, min_qty, min_cost, c_mult, grid_span,
                primary_pbr_limit, max_n_entry_orders, pbr, eprice_pprice_diff, eprice_exp_base=eprice_exp_base,
                eprices=eprices, prev_pprice=pprice)
            long_entries = [(g[0], g[1], 'primary_long_rentry') for g in fitted_grid[1:]]
            if secondary_pbr_allocation != 0.0:
                entry_price = round_dn(fitted_grid[-1][3] * (1 - secondary_grid_spacing), price_step)
                qty = find_qty_bringing_pbr_to_target(balance, fitted_grid[-1][2], fitted_grid[-1][3], pbr_limit, entry_price,
                                                      inverse, qty_step, c_mult)
                long_entries.append((qty, entry_price, 'secondary_long_rentry'))
            return long_entries if long_entries else [(0.0, 0.0, '')]


@njit
def njit_backtest(
        ticks,
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

        grid_span,
        pbr_limit,
        max_n_entry_orders,
        initial_qty_pct,
        eprice_pprice_diff,
        secondary_pbr_allocation,
        secondary_grid_spacing,
        eprice_exp_base,
        min_markup,
        markup_range,
        n_close_orders):

    timestamps = ticks[:, 0]
    qtys = ticks[:, 1]
    prices = ticks[:, 2]

    balance = equity = starting_balance
    long_psize, long_pprice, shrt_psize, shrt_pprice = 0.0, 0.0, 0.0, 0.0
    next_update_ts = 0
    fills = []

    long_entries = long_closes = [(0.0, 0.0, '')]
    bkr_price = 0.0

    stats = []
    next_stats_update = 0

    prev_k = 0
    closest_bkr = 1.0

    for k in range(1, len(prices)):
        if qtys[k] == 0.0:
            continue

        bkr_diff = calc_diff(bkr_price, prices[k])
        closest_bkr = min(closest_bkr, bkr_diff)
        if timestamps[k] >= next_stats_update:
            equity = balance + calc_upnl(long_psize, long_pprice, shrt_psize, shrt_pprice,
                                         prices[k], inverse, c_mult)
            stats.append((timestamps[k], balance, equity, bkr_price, long_psize, long_pprice,
                          shrt_psize, shrt_pprice, prices[k], closest_bkr))
            next_stats_update = timestamps[k] + 60 * 1000
        if timestamps[k] >= next_update_ts:
            # simulate small delay between bot and exchange
            long_entries = calc_long_entry_grid(
                balance, long_psize, long_pprice, prices[k - 1], inverse, do_long, qty_step, price_step,
                min_qty, min_cost, c_mult, grid_span[0], pbr_limit[0], max_n_entry_orders[0], initial_qty_pct[0],
                eprice_pprice_diff[0], secondary_pbr_allocation[0], secondary_grid_spacing[0], eprice_exp_base[0])
            long_closes = calc_long_close_grid(
                balance, long_psize, long_pprice, prices[k - 1], spot, inverse, qty_step, price_step, min_qty,
                min_cost, c_mult, initial_qty_pct[0], min_markup[0], markup_range[0], n_close_orders[0])

            bkr_price = calc_bankruptcy_price(balance, long_psize, long_pprice, shrt_psize, shrt_pprice, inverse, c_mult)

            equity = balance + calc_upnl(long_psize, long_pprice, shrt_psize, shrt_pprice,
                                         prices[k], inverse, c_mult)

            # if not in pos, wait 5 secs between updates, else wait 10 minutes
            next_update_ts = timestamps[k] + 1000 * (5 if long_psize == 0.0 else 60 * 10)

            if equity / starting_balance < 0.1:
                # break if 90% of starting balance is lost
                return fills, stats

            if closest_bkr < 0.06:
                # consider bankruptcy within 6% as liquidation
                if long_psize != 0.0:
                    fee_paid = -qty_to_cost(long_psize, long_pprice, inverse, c_mult) * maker_fee
                    pnl = calc_long_pnl(long_pprice, prices[k], -long_psize, inverse, c_mult)
                    balance = 0.0
                    equity = 0.0
                    long_psize, long_pprice = 0.0, 0.0
                    fills.append((k, timestamps[k], pnl, fee_paid, balance, equity,
                                  -long_psize, prices[k], 0.0, 0.0, 'long_bankruptcy'))
                if shrt_psize != 0.0:

                    fee_paid = -qty_to_cost(shrt_psize, shrt_pprice, inverse, c_mult) * maker_fee
                    pnl = calc_shrt_pnl(shrt_pprice, prices[k], -shrt_psize, inverse, c_mult)
                    balance, equity = 0.0, 0.0
                    shrt_psize, shrt_pprice = 0.0, 0.0
                    fills.append((k, timestamps[k], pnl, fee_paid, balance, equity,
                                  -shrt_psize, prices[k], 0.0, 0.0, 'shrt_bankruptcy'))

                return fills, stats

        while long_entries and long_entries[0][0] > 0.0 and prices[k] < long_entries[0][1]:
            if long_psize == 0.0:
                next_update_ts = timestamps[k] + latency_simulation_ms
            long_psize, long_pprice = calc_new_psize_pprice(long_psize, long_pprice, long_entries[0][0],
                                                            long_entries[0][1], qty_step)
            fee_paid = -qty_to_cost(long_entries[0][0], long_entries[0][1], inverse, c_mult) * maker_fee
            balance += fee_paid
            equity = calc_equity(balance, long_psize, long_pprice, shrt_psize, shrt_pprice, prices[k], inverse, c_mult)
            fills.append((k, timestamps[k], 0.0, fee_paid, balance, equity, long_entries[0][0], long_entries[0][1],
                          long_psize, long_pprice, long_entries[0][2]))
            long_entries = long_entries[1:]
            bkr_price = calc_bankruptcy_price(balance, long_psize, long_pprice, shrt_psize, shrt_pprice, inverse, c_mult)
        while long_psize > 0.0 and long_closes and long_closes[0][0] < 0.0 and prices[k] > long_closes[0][1]:
            next_update_ts = timestamps[k] + latency_simulation_ms
            long_close_qty = long_closes[0][0]
            new_long_psize = round_(long_psize + long_close_qty, qty_step)
            if new_long_psize < 0.0:
                print('warning: long close qty greater than long psize')
                print('long_psize', long_psize)
                print('long_pprice', long_pprice)
                print('long_closes[0]', long_closes[0])
                long_close_qty = -long_psize
                new_long_psize, long_pprice = 0.0, 0.0
            long_psize = new_long_psize
            fee_paid = -qty_to_cost(long_close_qty, long_closes[0][1], inverse, c_mult) * maker_fee
            pnl = calc_long_pnl(long_pprice, long_closes[0][1], long_close_qty, inverse, c_mult)
            balance += fee_paid + pnl
            equity = calc_equity(balance, long_psize, long_pprice, shrt_psize, shrt_pprice, prices[k], inverse, c_mult)
            fills.append((k, timestamps[k], pnl, fee_paid, balance, equity, long_close_qty, long_closes[0][1],
                          long_psize, long_pprice, long_closes[0][2]))
            long_closes = long_closes[1:]
            bkr_price = calc_bankruptcy_price(balance, long_psize, long_pprice, shrt_psize, shrt_pprice, inverse, c_mult)
    return fills, stats
