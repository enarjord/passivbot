from __future__ import annotations

import os

import numpy as np

if "NOJIT" in os.environ and os.environ["NOJIT"] == "true":
    print("not using numba")

    def njit(pyfunc=None, **kwargs):
        def wrap(func):
            return func

        if pyfunc is not None:
            return wrap(pyfunc)
        else:
            return wrap

else:
    print("using numba")
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
    return (
        min_qty
        if inverse
        else max(min_qty, round_up(min_cost / price if price > 0.0 else 0.0, qty_step))
    )


@njit
def cost_to_qty(cost, price, inverse, c_mult):
    return (cost * price if inverse else (cost / price if price > 0.0 else 0.0)) / c_mult


@njit
def qty_to_cost(qty, price, inverse, c_mult) -> float:
    return ((abs(qty / price) if price > 0.0 else 0.0) if inverse else abs(qty * price)) * c_mult


@njit
def calc_ema(alpha, alpha_, prev_ema, new_val) -> float:
    return prev_ema * alpha_ + new_val * alpha


@njit
def calc_samples(ticks: np.ndarray, sample_size_ms: int = 1000) -> np.ndarray:
    # ticks [[timestamp, qty, price]]
    sampled_timestamps = np.arange(
        ticks[0][0] // sample_size_ms * sample_size_ms,
        ticks[-1][0] // sample_size_ms * sample_size_ms + sample_size_ms,
        sample_size_ms,
    )
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
def calc_pnl_long(entry_price, close_price, qty, inverse, c_mult) -> float:
    if inverse:
        if entry_price == 0.0 or close_price == 0.0:
            return 0.0
        return abs(qty) * c_mult * (1.0 / entry_price - 1.0 / close_price)
    else:
        return abs(qty) * c_mult * (close_price - entry_price)


@njit
def calc_pnl_short(entry_price, close_price, qty, inverse, c_mult) -> float:
    if inverse:
        if entry_price == 0.0 or close_price == 0.0:
            return 0.0
        return abs(qty) * c_mult * (1.0 / close_price - 1.0 / entry_price)
    else:
        return abs(qty) * c_mult * (entry_price - close_price)


@njit
def calc_equity(
    balance,
    psize_long,
    pprice_long,
    psize_short,
    pprice_short,
    last_price,
    inverse,
    c_mult,
):
    equity = balance
    if pprice_long and psize_long:
        equity += calc_pnl_long(pprice_long, last_price, psize_long, inverse, c_mult)
    if pprice_short and psize_short:
        equity += calc_pnl_short(pprice_short, last_price, psize_short, inverse, c_mult)
    return equity


@njit
def calc_new_psize_pprice(psize, pprice, qty, price, qty_step) -> (float, float):
    if qty == 0.0:
        return psize, pprice
    new_psize = round_(psize + qty, qty_step)
    if new_psize == 0.0:
        return 0.0, 0.0
    return (
        new_psize,
        nan_to_0(pprice) * (psize / new_psize) + price * (qty / new_psize),
    )


@njit
def calc_wallet_exposure_if_filled(balance, psize, pprice, qty, price, inverse, c_mult, qty_step):
    psize, qty = round_(abs(psize), qty_step), round_(abs(qty), qty_step)
    new_psize, new_pprice = calc_new_psize_pprice(psize, pprice, qty, price, qty_step)
    return qty_to_cost(new_psize, new_pprice, inverse, c_mult) / balance


@njit
def calc_close_grid_long(
    backwards_tp,
    balance,
    psize,
    pprice,
    lowest_ask,
    ema_band_upper,
    inverse,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    wallet_exposure_limit,
    min_markup,
    markup_range,
    n_close_orders,
    auto_unstuck_wallet_exposure_threshold,
    auto_unstuck_ema_dist,
):
    if backwards_tp:
        return calc_close_grid_backwards_long(
            balance,
            psize,
            pprice,
            lowest_ask,
            ema_band_upper,
            inverse,
            qty_step,
            price_step,
            min_qty,
            min_cost,
            c_mult,
            wallet_exposure_limit,
            min_markup,
            markup_range,
            n_close_orders,
            auto_unstuck_wallet_exposure_threshold,
            auto_unstuck_ema_dist,
        )
    else:
        return calc_close_grid_frontwards_long(
            balance,
            psize,
            pprice,
            lowest_ask,
            ema_band_upper,
            inverse,
            qty_step,
            price_step,
            min_qty,
            min_cost,
            c_mult,
            wallet_exposure_limit,
            min_markup,
            markup_range,
            n_close_orders,
            auto_unstuck_wallet_exposure_threshold,
            auto_unstuck_ema_dist,
        )


@njit
def calc_close_grid_short(
    backwards_tp,
    balance,
    psize,
    pprice,
    highest_bid,
    ema_band_lower,
    inverse,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    wallet_exposure_limit,
    min_markup,
    markup_range,
    n_close_orders,
    auto_unstuck_wallet_exposure_threshold,
    auto_unstuck_ema_dist,
):
    if backwards_tp:
        return calc_close_grid_backwards_short(
            balance,
            psize,
            pprice,
            highest_bid,
            ema_band_lower,
            inverse,
            qty_step,
            price_step,
            min_qty,
            min_cost,
            c_mult,
            wallet_exposure_limit,
            min_markup,
            markup_range,
            n_close_orders,
            auto_unstuck_wallet_exposure_threshold,
            auto_unstuck_ema_dist,
        )
    else:
        return calc_close_grid_frontwards_short(
            balance,
            psize,
            pprice,
            highest_bid,
            ema_band_lower,
            inverse,
            qty_step,
            price_step,
            min_qty,
            min_cost,
            c_mult,
            wallet_exposure_limit,
            min_markup,
            markup_range,
            n_close_orders,
            auto_unstuck_wallet_exposure_threshold,
            auto_unstuck_ema_dist,
        )


@njit
def calc_close_grid_backwards_long(
    balance,
    psize,
    pprice,
    lowest_ask,
    ema_band_upper,
    inverse,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    wallet_exposure_limit,
    min_markup,
    markup_range,
    n_close_orders,
    auto_unstuck_wallet_exposure_threshold,
    auto_unstuck_ema_dist,
):
    psize = psize_ = round_dn(psize, qty_step)  # round down for spot
    if psize == 0.0:
        return [(0.0, 0.0, "")]
    minm = pprice * (1 + min_markup)
    full_psize = cost_to_qty(balance * wallet_exposure_limit, pprice, inverse, c_mult)
    n_close_orders = min(
        n_close_orders,
        full_psize / calc_min_entry_qty(pprice, inverse, qty_step, min_qty, min_cost),
    )
    n_close_orders = max(1, int(round(n_close_orders)))
    raw_close_prices = np.linspace(minm, pprice * (1 + min_markup + markup_range), n_close_orders)
    close_prices = []
    close_prices_all = []
    for p_ in raw_close_prices:
        price = round_up(p_, price_step)
        if price not in close_prices_all:
            close_prices_all.append(price)
            if price >= lowest_ask:
                close_prices.append(price)
    if len(close_prices) == 0:
        return [(-psize, lowest_ask, "long_nclose")]
    wallet_exposure = qty_to_cost(psize, pprice, inverse, c_mult) / balance
    closes = []
    threshold = wallet_exposure_limit * (1 - auto_unstuck_wallet_exposure_threshold)
    if auto_unstuck_wallet_exposure_threshold != 0.0 and wallet_exposure > threshold:
        unstuck_close_price = max(
            lowest_ask, round_up(ema_band_upper * (1 + auto_unstuck_ema_dist), price_step)
        )
        if unstuck_close_price < close_prices[0]:
            unstuck_close_qty = find_close_qty_long_bringing_wallet_exposure_to_target(
                balance,
                psize_,
                pprice,
                threshold * 1.01,
                unstuck_close_price,
                inverse,
                qty_step,
                c_mult,
            )
            min_entry_qty = calc_min_entry_qty(
                unstuck_close_price, inverse, qty_step, min_qty, min_cost
            )
            unstuck_close_qty = max(min_entry_qty, unstuck_close_qty)
            psize_ = round_(psize_ - unstuck_close_qty, qty_step)
            if psize_ < min_entry_qty:
                # close whole pos; include leftovers
                return [(-psize, unstuck_close_price, "long_unstuck_close")]
            closes.append((-unstuck_close_qty, unstuck_close_price, "long_unstuck_close"))
    if len(close_prices) == 1:
        if psize_ >= calc_min_entry_qty(close_prices[0], inverse, qty_step, min_qty, min_cost):
            closes.append((-psize_, close_prices[0], "long_nclose"))
        return closes
    qty_per_close = max(min_qty, round_up(full_psize / len(close_prices_all), qty_step))
    for price in close_prices[::-1]:
        min_entry_qty = calc_min_entry_qty(price, inverse, qty_step, min_qty, min_cost)
        qty = min(psize_, max(qty_per_close, min_entry_qty))
        if qty < min_entry_qty:
            if closes:
                closes[-1] = (round_(closes[-1][0] - psize_, qty_step), closes[-1][1], closes[-1][2])
            else:
                closes.append((-psize_, price, "long_nclose"))
            psize_ = 0.0
            break
        closes.append((-qty, price, "long_nclose"))
        psize_ = round_(psize_ - qty, qty_step)
        if psize_ <= 0.0:
            break
    if psize_ > 0.0 and closes:
        closes[-1] = (round_(closes[-1][0] - psize_, qty_step), closes[-1][1], closes[-1][2])
    return sorted(closes, key=lambda x: x[1])


@njit
def calc_close_grid_frontwards_long(
    balance,
    psize,
    pprice,
    lowest_ask,
    ema_band_upper,
    inverse,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    wallet_exposure_limit,
    min_markup,
    markup_range,
    n_close_orders,
    auto_unstuck_wallet_exposure_threshold,
    auto_unstuck_ema_dist,
):
    psize = psize_ = round_dn(psize, qty_step)  # round down for spot
    if psize == 0.0:
        return [(0.0, 0.0, "")]
    minm = pprice * (1 + min_markup)
    raw_close_prices = np.linspace(
        minm, pprice * (1 + min_markup + markup_range), int(round(n_close_orders))
    )
    close_prices = []
    for p_ in raw_close_prices:
        price = round_up(p_, price_step)
        if price >= lowest_ask:
            close_prices.append(price)
    closes = []
    if len(close_prices) == 0:
        return [(-psize, lowest_ask, "long_nclose")]
    wallet_exposure = qty_to_cost(psize, pprice, inverse, c_mult) / balance
    threshold = wallet_exposure_limit * (1 - auto_unstuck_wallet_exposure_threshold)
    if auto_unstuck_wallet_exposure_threshold != 0.0 and wallet_exposure > threshold:
        unstuck_close_price = max(
            lowest_ask, round_up(ema_band_upper * (1 + auto_unstuck_ema_dist), price_step)
        )
        if unstuck_close_price < close_prices[0]:
            unstuck_close_qty = find_close_qty_long_bringing_wallet_exposure_to_target(
                balance,
                psize_,
                pprice,
                threshold * 1.01,
                unstuck_close_price,
                inverse,
                qty_step,
                c_mult,
            )
            min_entry_qty = calc_min_entry_qty(
                unstuck_close_price, inverse, qty_step, min_qty, min_cost
            )
            unstuck_close_qty = max(min_entry_qty, unstuck_close_qty)
            psize_ = round_(psize_ - unstuck_close_qty, qty_step)
            if psize_ < min_entry_qty:
                # close whole pos; include leftovers
                return [(-psize, unstuck_close_price, "long_unstuck_close")]
            closes.append((-unstuck_close_qty, unstuck_close_price, "long_unstuck_close"))
    if len(close_prices) == 1:
        if psize_ >= calc_min_entry_qty(close_prices[0], inverse, qty_step, min_qty, min_cost):
            closes.append((-psize_, close_prices[0], "long_nclose"))
        return closes
    default_close_qty = round_dn(psize_ / len(close_prices), qty_step)
    for price in close_prices[:-1]:
        min_close_qty = calc_min_entry_qty(price, inverse, qty_step, min_qty, min_cost)
        if psize_ < min_close_qty:
            break
        close_qty = min(psize_, max(min_close_qty, default_close_qty))
        closes.append((-close_qty, price, "long_nclose"))
        psize_ = round_(psize_ - close_qty, qty_step)
    min_close_qty = calc_min_entry_qty(close_prices[-1], inverse, qty_step, min_qty, min_cost)
    if psize_ >= min_close_qty:
        closes.append((-psize_, close_prices[-1], "long_nclose"))
    elif len(closes) > 0:
        closes[-1] = (-round_(abs(closes[-1][0]) + psize_, qty_step), closes[-1][1], closes[-1][2])
    return closes


@njit
def calc_close_grid_backwards_short(
    balance,
    psize,
    pprice,
    highest_bid,
    ema_band_lower,
    inverse,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    wallet_exposure_limit,
    min_markup,
    markup_range,
    n_close_orders,
    auto_unstuck_wallet_exposure_threshold,
    auto_unstuck_ema_dist,
):
    psize = psize_ = round_dn(abs(psize), qty_step)  # round down for spot
    if psize == 0.0:
        return [(0.0, 0.0, "")]
    minm = pprice * (1 - min_markup)
    full_psize = cost_to_qty(balance * wallet_exposure_limit, pprice, inverse, c_mult)
    n_close_orders = min(
        n_close_orders,
        full_psize / calc_min_entry_qty(pprice, inverse, qty_step, min_qty, min_cost),
    )
    n_close_orders = max(1, int(round(n_close_orders)))
    raw_close_prices = np.linspace(minm, pprice * (1 - min_markup - markup_range), n_close_orders)
    close_prices = []
    close_prices_all = []
    for p_ in raw_close_prices:
        price = round_dn(p_, price_step)
        if price not in close_prices_all:
            close_prices_all.append(price)
            if price <= highest_bid:
                close_prices.append(price)
    if len(close_prices) == 0:
        return [(psize, highest_bid, "short_nclose")]
    wallet_exposure = qty_to_cost(psize, pprice, inverse, c_mult) / balance
    closes = []
    threshold = wallet_exposure_limit * (1 - auto_unstuck_wallet_exposure_threshold)
    if auto_unstuck_wallet_exposure_threshold != 0.0 and wallet_exposure > threshold:
        unstuck_close_price = min(
            highest_bid, round_dn(ema_band_lower * (1 - auto_unstuck_ema_dist), price_step)
        )
        if unstuck_close_price > close_prices[0]:
            unstuck_close_qty = find_close_qty_short_bringing_wallet_exposure_to_target(
                balance,
                psize,
                pprice,
                threshold * 1.01,
                unstuck_close_price,
                inverse,
                qty_step,
                c_mult,
            )
            min_entry_qty = calc_min_entry_qty(
                unstuck_close_price, inverse, qty_step, min_qty, min_cost
            )
            unstuck_close_qty = max(min_entry_qty, unstuck_close_qty)
            psize_ = round_(psize_ - unstuck_close_qty, qty_step)
            if psize_ < min_entry_qty:
                # close whole pos; include leftovers
                return [(psize, unstuck_close_price, "short_unstuck_close")]
            closes.append((unstuck_close_qty, unstuck_close_price, "short_unstuck_close"))
    if len(close_prices) == 1:
        if psize_ >= calc_min_entry_qty(close_prices[0], inverse, qty_step, min_qty, min_cost):
            closes.append((psize_, close_prices[0], "short_nclose"))
        return closes
    qty_per_close = max(min_qty, round_up(full_psize / len(close_prices_all), qty_step))
    for price in close_prices[::-1]:
        min_entry_qty = calc_min_entry_qty(price, inverse, qty_step, min_qty, min_cost)
        qty = min(psize_, max(qty_per_close, min_entry_qty))
        if qty < min_entry_qty:
            if closes:
                closes[-1] = (round_(closes[-1][0] + psize_, qty_step), closes[-1][1], closes[-1][2])
            else:
                closes.append((psize_, price, "short_nclose"))
            psize_ = 0.0
            break
        closes.append((qty, price, "short_nclose"))
        psize_ = round_(psize_ - qty, qty_step)
        if psize_ <= 0.0:
            break
    if psize_ > 0.0 and closes:
        closes[-1] = (round_(closes[-1][0] + psize_, qty_step), closes[-1][1], closes[-1][2])
    return sorted(closes, key=lambda x: x[1], reverse=True)


@njit
def calc_close_grid_frontwards_short(
    balance,
    psize,
    pprice,
    highest_bid,
    ema_band_lower,
    inverse,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    wallet_exposure_limit,
    min_markup,
    markup_range,
    n_close_orders,
    auto_unstuck_wallet_exposure_threshold,
    auto_unstuck_ema_dist,
):
    abs_psize = abs_psize_ = round_dn(abs(psize), qty_step)  # round down for spot
    if abs_psize == 0.0:
        return [(0.0, 0.0, "")]
    minm = pprice * (1 - min_markup)
    raw_close_prices = np.linspace(
        minm, pprice * (1 - min_markup - markup_range), int(round(n_close_orders))
    )
    close_prices = []
    for p_ in raw_close_prices:
        price = round_dn(p_, price_step)
        if price <= highest_bid:
            close_prices.append(price)
    closes = []
    if len(close_prices) == 0:
        return [(abs_psize, highest_bid, "short_nclose")]
    wallet_exposure = qty_to_cost(psize, pprice, inverse, c_mult) / balance
    threshold = wallet_exposure_limit * (1 - auto_unstuck_wallet_exposure_threshold)
    if auto_unstuck_wallet_exposure_threshold != 0.0 and wallet_exposure > threshold:
        unstuck_close_price = min(
            highest_bid, round_dn(ema_band_lower * (1 - auto_unstuck_ema_dist), price_step)
        )
        if unstuck_close_price > close_prices[0]:
            unstuck_close_qty = find_close_qty_short_bringing_wallet_exposure_to_target(
                balance,
                psize,
                pprice,
                threshold * 1.01,
                unstuck_close_price,
                inverse,
                qty_step,
                c_mult,
            )
            min_entry_qty = calc_min_entry_qty(
                unstuck_close_price, inverse, qty_step, min_qty, min_cost
            )
            unstuck_close_qty = max(min_entry_qty, unstuck_close_qty)
            abs_psize_ = round_(abs_psize_ - unstuck_close_qty, qty_step)
            if abs_psize_ < min_entry_qty:
                # close whole pos; include leftovers
                return [(abs_psize, unstuck_close_price, "short_unstuck_close")]
            closes.append((unstuck_close_qty, unstuck_close_price, "short_unstuck_close"))
    if len(close_prices) == 1:
        if abs_psize_ >= calc_min_entry_qty(close_prices[0], inverse, qty_step, min_qty, min_cost):
            closes.append((abs_psize_, close_prices[0], "short_nclose"))
        return closes
    default_close_qty = round_dn(abs_psize_ / len(close_prices), qty_step)
    for price in close_prices[:-1]:
        min_close_qty = calc_min_entry_qty(price, inverse, qty_step, min_qty, min_cost)
        if abs_psize_ < min_close_qty:
            break
        close_qty = min(abs_psize_, max(min_close_qty, default_close_qty))
        closes.append((close_qty, price, "short_nclose"))
        abs_psize_ = round_(abs_psize_ - close_qty, qty_step)
    min_close_qty = calc_min_entry_qty(close_prices[-1], inverse, qty_step, min_qty, min_cost)
    if abs_psize_ >= min_close_qty:
        closes.append((abs_psize_, close_prices[-1], "short_nclose"))
    elif len(closes) > 0:
        closes[-1] = (round_(closes[-1][0] + abs_psize_, qty_step), closes[-1][1], closes[-1][2])
    return closes


@njit
def calc_upnl(psize_long, pprice_long, psize_short, pprice_short, last_price, inverse, c_mult):
    return calc_pnl_long(pprice_long, last_price, psize_long, inverse, c_mult) + calc_pnl_short(
        pprice_short, last_price, psize_short, inverse, c_mult
    )


@njit
def calc_emas_last(xs, spans):
    alphas = 2.0 / (spans + 1.0)
    alphas_ = 1.0 - alphas
    emas = np.repeat(xs[0], len(spans))
    for i in range(1, len(xs)):
        emas = emas * alphas_ + xs[i] * alphas
    return emas


@njit
def calc_bankruptcy_price(
    balance, psize_long, pprice_long, psize_short, pprice_short, inverse, c_mult
):
    pprice_long = nan_to_0(pprice_long)
    pprice_short = nan_to_0(pprice_short)
    psize_long *= c_mult
    abs_psize_short = abs(psize_short) * c_mult
    if inverse:
        short_cost = abs_psize_short / pprice_short if pprice_short > 0.0 else 0.0
        long_cost = psize_long / pprice_long if pprice_long > 0.0 else 0.0
        denominator = short_cost - long_cost - balance
        if denominator == 0.0:
            return 0.0
        bankruptcy_price = (abs_psize_short - psize_long) / denominator
    else:
        denominator = psize_long - abs_psize_short
        if denominator == 0.0:
            return 0.0
        bankruptcy_price = (
            -balance + psize_long * pprice_long - abs_psize_short * pprice_short
        ) / denominator
    return max(0.0, bankruptcy_price)


@njit
def basespace(start, end, base, n):
    if base == 1.0:
        return np.linspace(start, end, n)
    elif base <= 0.0:
        raise Exception("not defined for base <= 0.0")
    elif base < 1.0:
        a = -np.array([base ** i for i in range(n)])
    else:
        a = np.array([base ** i for i in range(n)])
    a = (a - a.min()) / (a.max() - a.min())
    return a * (end - start) + start


@njit
def powspace(start, stop, power, num):
    start = np.power(start, 1 / float(power))
    stop = np.power(stop, 1 / float(power))
    return np.power(np.linspace(start, stop, num=num), power)


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
def calc_entry_qty_long(psize, pprice, entry_price, eprice_pprice_diff):
    if entry_price == 0.0:
        print("error entry_price", entry_price)
    if eprice_pprice_diff == 0.0:
        print("error eprice_pprice_diff", eprice_pprice_diff)
    return -(
        psize
        * (entry_price * eprice_pprice_diff + entry_price - pprice)
        / (entry_price * eprice_pprice_diff)
    )


@njit
def calc_initial_entry_qty(
    balance,
    initial_entry_price,
    inverse,
    qty_step,
    min_qty,
    min_cost,
    c_mult,
    wallet_exposure_limit,
    initial_qty_pct,
):
    return max(
        calc_min_entry_qty(initial_entry_price, inverse, qty_step, min_qty, min_cost),
        round_(
            cost_to_qty(
                balance * wallet_exposure_limit * initial_qty_pct,
                initial_entry_price,
                inverse,
                c_mult,
            ),
            qty_step,
        ),
    )


@njit
def calc_entry_qty_short(psize, pprice, entry_price, eprice_pprice_diff):
    return -(
        (psize * (entry_price * (eprice_pprice_diff - 1) + pprice))
        / (entry_price * eprice_pprice_diff)
    )


@njit
def calc_entry_price_long(psize, pprice, entry_qty, eprice_pprice_diff):
    return (psize * pprice) / (psize * eprice_pprice_diff + psize + entry_qty * eprice_pprice_diff)


@njit
def interpolate(x, xs, ys):
    return np.sum(
        np.array(
            [
                np.prod(np.array([(x - xs[m]) / (xs[j] - xs[m]) for m in range(len(xs)) if m != j]))
                * ys[j]
                for j in range(len(xs))
            ]
        )
    )


@njit
def find_close_qty_long_bringing_wallet_exposure_to_target(
    balance,
    psize,
    pprice,
    wallet_exposure_target,
    close_price,
    inverse,
    qty_step,
    c_mult,
) -> float:
    def eval_(guess_):
        return qty_to_cost(psize - guess_, pprice, inverse, c_mult) / (
            balance + calc_pnl_long(pprice, close_price, guess_, inverse, c_mult)
        )

    if wallet_exposure_target == 0.0:
        return psize
    wallet_exposure = qty_to_cost(psize, pprice, inverse, c_mult) / balance
    if wallet_exposure <= wallet_exposure_target * 1.001:
        # wallet_exposure within 0.1% of target: return zero
        return 0.0
    guesses = []
    vals = []
    evals = []
    guesses.append(
        min(
            psize,
            max(0.0, round_(psize * (1 - wallet_exposure_target / wallet_exposure), qty_step)),
        )
    )
    vals.append(eval_(guesses[-1]))
    evals.append(abs(vals[-1] - wallet_exposure_target) / wallet_exposure_target)
    guesses.append(
        min(psize, max(0.0, round_(max(guesses[-1] * 1.2, guesses[-1] + qty_step), qty_step)))
    )
    if guesses[-1] == guesses[-2]:
        guesses[-1] = min(
            psize, max(0.0, round_(min(guesses[-1] * 0.8, guesses[-1] - qty_step), qty_step))
        )
    vals.append(eval_(guesses[-1]))
    evals.append(abs(vals[-1] - wallet_exposure_target) / wallet_exposure_target)
    for _ in range(15):
        egv = sorted([(e, g, v) for e, g, v in zip(evals, guesses, vals)])
        try:
            new_guess = interpolate(
                wallet_exposure_target,
                np.array([egv[0][2], egv[1][2]]),
                np.array([egv[0][1], egv[1][1]]),
            )
        except:
            """
            print("debug zero div error find_close_qty_long_bringing_wallet_exposure_to_target")
            print(
                "balance, psize, pprice, wallet_exposure_target, close_price, inverse, qty_step, c_mult,"
            )
            print(
                balance,
                psize,
                pprice,
                wallet_exposure_target,
                close_price,
                inverse,
                qty_step,
                c_mult,
            )
            print("guesses, vals", guesses, vals)
            """
            new_guess = (egv[0][1] + egv[1][1]) / 2
        new_guess = min(psize, max(0.0, round_(new_guess, qty_step)))
        if new_guess in guesses:
            new_guess = min(psize, max(0.0, round_(new_guess - qty_step, qty_step)))
            if new_guess in guesses:
                new_guess = min(psize, max(0.0, round_(new_guess + 2 * qty_step, qty_step)))
                if new_guess in guesses:
                    break
        guesses.append(new_guess)
        vals.append(eval_(guesses[-1]))
        evals.append(abs(vals[-1] - wallet_exposure_target) / wallet_exposure_target)
        if evals[-1] < 0.01:
            # close enough
            break
    evals_guesses = sorted([(e, g) for e, g in zip(evals, guesses)])
    if False:  # evals_guesses[0][0] > 0.15:
        print("debug find_close_qty_long_bringing_wallet_exposure_to_target")
        print(
            "balance, psize, pprice, wallet_exposure_target, close_price, inverse, qty_step, c_mult,"
        )
        print(
            balance,
            psize,
            pprice,
            wallet_exposure_target,
            close_price,
            inverse,
            qty_step,
            c_mult,
        )
        print("wallet_exposure", wallet_exposure)
        print("wallet_exposure_target", wallet_exposure_target)
        print(
            "guess, val, target diff",
            [(g, round_dynamic(v, 4), round_dynamic(e, 4)) for g, v, e in zip(guesses, vals, evals)],
        )
        print("n tries", len(guesses))
        print()
    return evals_guesses[0][1]


@njit
def find_close_qty_short_bringing_wallet_exposure_to_target(
    balance,
    psize,
    pprice,
    wallet_exposure_target,
    close_price,
    inverse,
    qty_step,
    c_mult,
) -> float:
    def eval_(guess_):
        return qty_to_cost(abs(psize) - guess_, pprice, inverse, c_mult) / (
            balance + calc_pnl_short(pprice, close_price, guess_, inverse, c_mult)
        )

    if wallet_exposure_target == 0.0:
        return abs(psize)
    wallet_exposure = qty_to_cost(psize, pprice, inverse, c_mult) / balance
    if wallet_exposure <= wallet_exposure_target * 1.001:
        # wallet_exposure within 0.1% of target: return zero
        return 0.0
    guesses = []
    vals = []
    evals = []
    abs_psize = abs(psize)
    guesses.append(
        min(
            abs_psize,
            max(0.0, round_(abs_psize * (1 - wallet_exposure_target / wallet_exposure), qty_step)),
        )
    )
    vals.append(eval_(guesses[-1]))
    evals.append(abs(vals[-1] - wallet_exposure_target) / wallet_exposure_target)
    guesses.append(
        min(abs_psize, max(0.0, round_(max(guesses[-1] * 1.2, guesses[-1] + qty_step), qty_step)))
    )
    if guesses[-1] == guesses[-2]:
        guesses[-1] = min(
            abs_psize, max(0.0, round_(min(guesses[-1] * 0.8, guesses[-1] - qty_step), qty_step))
        )
    vals.append(eval_(guesses[-1]))
    evals.append(abs(vals[-1] - wallet_exposure_target) / wallet_exposure_target)
    for _ in range(15):
        egv = sorted([(e, g, v) for e, g, v in zip(evals, guesses, vals)])
        try:
            new_guess = interpolate(
                wallet_exposure_target,
                np.array([egv[0][2], egv[1][2]]),
                np.array([egv[0][1], egv[1][1]]),
            )
        except:
            """
            print("debug zero div error find_close_qty_short_bringing_wallet_exposure_to_target")
            print(
                "balance, psize, pprice, wallet_exposure_target, close_price, inverse, qty_step, c_mult,"
            )
            print(
                balance,
                psize,
                pprice,
                wallet_exposure_target,
                close_price,
                inverse,
                qty_step,
                c_mult,
            )
            print("guesses, vals", guesses, vals)
            """
            new_guess = (egv[0][1] + egv[1][1]) / 2
        new_guess = min(abs_psize, max(0.0, round_(new_guess, qty_step)))
        if new_guess in guesses:
            new_guess = min(abs_psize, max(0.0, round_(new_guess - qty_step, qty_step)))
            if new_guess in guesses:
                new_guess = min(abs_psize, max(0.0, round_(new_guess + 2 * qty_step, qty_step)))
                if new_guess in guesses:
                    break
        guesses.append(new_guess)
        vals.append(eval_(guesses[-1]))
        evals.append(abs(vals[-1] - wallet_exposure_target) / wallet_exposure_target)
        if evals[-1] < 0.01:
            # close enough
            break
    evals_guesses = sorted([(e, g) for e, g in zip(evals, guesses)])
    if False:  # evals_guesses[0][0] > 0.15:
        print("debug find_close_qty_short_bringing_wallet_exposure_to_target")
        print(
            "balance, psize, pprice, wallet_exposure_target, close_price, inverse, qty_step, c_mult,"
        )
        print(
            balance,
            psize,
            pprice,
            wallet_exposure_target,
            close_price,
            inverse,
            qty_step,
            c_mult,
        )
        print("wallet_exposure", wallet_exposure)
        print("wallet_exposure_target", wallet_exposure_target)
        print(
            "guess, val, target diff",
            [(g, round_dynamic(v, 4), round_dynamic(e, 4)) for g, v, e in zip(guesses, vals, evals)],
        )
        print("n tries", len(guesses))
        print()
    return evals_guesses[0][1]


@njit
def find_entry_qty_bringing_wallet_exposure_to_target(
    balance,
    psize,
    pprice,
    wallet_exposure_target,
    entry_price,
    inverse,
    qty_step,
    c_mult,
) -> float:
    if wallet_exposure_target == 0.0:
        return 0.0
    wallet_exposure = qty_to_cost(psize, pprice, inverse, c_mult) / balance
    if wallet_exposure >= wallet_exposure_target * 0.99:
        # return zero if wallet_exposure already is within 1% of target
        return 0.0
    guesses = []
    vals = []
    evals = []
    guesses.append(round_(abs(psize) * wallet_exposure_target / max(0.01, wallet_exposure), qty_step))
    vals.append(
        calc_wallet_exposure_if_filled(
            balance, psize, pprice, guesses[-1], entry_price, inverse, c_mult, qty_step
        )
    )
    evals.append(abs(vals[-1] - wallet_exposure_target) / wallet_exposure_target)
    guesses.append(max(0.0, round_(max(guesses[-1] * 1.2, guesses[-1] + qty_step), qty_step)))
    vals.append(
        calc_wallet_exposure_if_filled(
            balance, psize, pprice, guesses[-1], entry_price, inverse, c_mult, qty_step
        )
    )
    evals.append(abs(vals[-1] - wallet_exposure_target) / wallet_exposure_target)
    for _ in range(15):
        if guesses[-1] == guesses[-2]:
            guesses[-1] = abs(round_(max(guesses[-2] * 1.1, guesses[-2] + qty_step), qty_step))
            vals[-1] = calc_wallet_exposure_if_filled(
                balance, psize, pprice, guesses[-1], entry_price, inverse, c_mult, qty_step
            )
        guesses.append(
            max(
                0.0,
                round_(
                    interpolate(wallet_exposure_target, np.array(vals[-2:]), np.array(guesses[-2:])),
                    qty_step,
                ),
            )
        )
        vals.append(
            calc_wallet_exposure_if_filled(
                balance, psize, pprice, guesses[-1], entry_price, inverse, c_mult, qty_step
            )
        )
        evals.append(abs(vals[-1] - wallet_exposure_target) / wallet_exposure_target)
        if evals[-1] < 0.01:
            # close enough
            break
    evals_guesses = sorted([(e, g) for e, g in zip(evals, guesses)])
    if False:  # evals_guesses[0][0] > 0.15:
        print("debug find_entry_qty_bringing_wallet_exposure_to_target")
        print(
            "balance, psize, pprice, wallet_exposure_target, entry_price, inverse, qty_step, c_mult,"
        )
        print(
            balance,
            psize,
            pprice,
            wallet_exposure_target,
            entry_price,
            inverse,
            qty_step,
            c_mult,
        )
        print("wallet_exposure", wallet_exposure)
        print("wallet_exposure_target", wallet_exposure_target)
        print(
            "guess, val, target diff",
            [(g, round_dynamic(v, 4), round_dynamic(e, 4)) for g, v, e in zip(guesses, vals, evals)],
        )
        print()
    return evals_guesses[0][1]


@njit
def find_eprice_pprice_diff_wallet_exposure_weighting(
    is_long: bool,
    balance,
    initial_entry_price,
    inverse,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    grid_span,
    wallet_exposure_limit,
    max_n_entry_orders,
    initial_qty_pct,
    eprice_pprice_diff,
    eprice_exp_base=1.618034,
    max_n_iters=20,
    error_tolerance=0.01,
    eprices=None,
    prev_pprice=None,
):
    def eval_(guess_):
        if is_long:
            return eval_entry_grid_long(
                balance,
                initial_entry_price,
                inverse,
                qty_step,
                price_step,
                min_qty,
                min_cost,
                c_mult,
                grid_span,
                wallet_exposure_limit,
                max_n_entry_orders,
                initial_qty_pct,
                eprice_pprice_diff,
                guess_,
                eprice_exp_base=eprice_exp_base,
                eprices=eprices,
                prev_pprice=prev_pprice,
            )[-1][4]
        else:
            return eval_entry_grid_short(
                balance,
                initial_entry_price,
                inverse,
                qty_step,
                price_step,
                min_qty,
                min_cost,
                c_mult,
                grid_span,
                wallet_exposure_limit,
                max_n_entry_orders,
                initial_qty_pct,
                eprice_pprice_diff,
                guess_,
                eprice_exp_base=eprice_exp_base,
                eprices=eprices,
                prev_pprice=prev_pprice,
            )[-1][4]

    guess = 0.0
    val = eval_(guess)
    if val < wallet_exposure_limit:
        return guess
    too_low = (guess, val)
    guess = 1000.0
    val = eval_(guess)
    if val > wallet_exposure_limit:
        guess = 10000.0
        val = eval_(guess)
        if val > wallet_exposure_limit:
            guess = 100000.0
            val = eval_(guess)
            if val > wallet_exposure_limit:
                return guess
    too_high = (guess, val)
    guesses = [too_low[1], too_high[1]]
    vals = [too_low[0], too_high[0]]
    guess = interpolate(wallet_exposure_limit, np.array(vals), np.array(guesses))
    val = eval_(guess)
    if val < wallet_exposure_limit:
        too_high = (guess, val)
    else:
        too_low = (guess, val)
    i = 0
    old_guess = 0.0
    best_guess = (abs(val - wallet_exposure_limit) / wallet_exposure_limit, guess, val)
    while True:
        i += 1
        diff = abs(val - wallet_exposure_limit) / wallet_exposure_limit
        if diff < best_guess[0]:
            best_guess = (diff, guess, val)
        if diff < error_tolerance:
            return best_guess[1]
        if i >= max_n_iters or abs(old_guess - guess) / guess < error_tolerance * 0.1:
            """
            if best_guess[0] > 0.15:
                log.info('debug find_eprice_pprice_diff_wallet_exposure_weighting')
                log.info('is_long, balance, initial_entry_price, inverse, qty_step, price_step, min_qty, min_cost, c_mult, grid_span, wallet_exposure_limit, max_n_entry_orders, initial_qty_pct, eprice_pprice_diff, eprice_exp_base, max_n_iters, error_tolerance, eprices, prev_pprice')
                log.info(is_long, ',', balance, ',', initial_entry_price, ',', inverse, ',', qty_step, ',', price_step, ',', min_qty, ',', min_cost, ',', c_mult, ',', grid_span, ',', wallet_exposure_limit, ',', max_n_entry_orders, ',', initial_qty_pct, ',', eprice_pprice_diff, ',', eprice_exp_base, ',', max_n_iters, ',', error_tolerance, ',', eprices, ',', prev_pprice)
            """
            return best_guess[1]
        old_guess = guess
        guess = (too_high[0] + too_low[0]) / 2
        val = eval_(guess)
        if val < wallet_exposure_limit:
            too_high = (guess, val)
        else:
            too_low = (guess, val)


@njit
def eval_entry_grid_long(
    balance,
    initial_entry_price,
    inverse,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    grid_span,
    wallet_exposure_limit,
    max_n_entry_orders,
    initial_qty_pct,
    eprice_pprice_diff,
    eprice_pprice_diff_wallet_exposure_weighting,
    eprice_exp_base=1.618034,
    eprices=None,
    prev_pprice=None,
):

    # returns [qty, price, psize, pprice, wallet_exposure]
    if eprices is None:
        grid = np.zeros((max_n_entry_orders, 5))
        grid[:, 1] = [
            max(price_step, round_dn(p, price_step))
            for p in basespace(
                initial_entry_price,
                initial_entry_price * (1 - grid_span),
                eprice_exp_base,
                max_n_entry_orders,
            )
        ]
    else:
        max_n_entry_orders = len(eprices)
        grid = np.zeros((max_n_entry_orders, 5))
        grid[:, 1] = eprices

    grid[0][0] = max(
        calc_min_entry_qty(grid[0][1], inverse, qty_step, min_qty, min_cost),
        round_(
            cost_to_qty(
                balance * wallet_exposure_limit * initial_qty_pct,
                initial_entry_price,
                inverse,
                c_mult,
            ),
            qty_step,
        ),
    )
    grid[0][2] = psize = grid[0][0]
    grid[0][3] = pprice = grid[0][1] if prev_pprice is None else prev_pprice
    grid[0][4] = qty_to_cost(psize, pprice, inverse, c_mult) / balance
    for i in range(1, max_n_entry_orders):
        adjusted_eprice_pprice_diff = eprice_pprice_diff * (
            1 + grid[i - 1][4] * eprice_pprice_diff_wallet_exposure_weighting
        )
        qty = round_(
            calc_entry_qty_long(psize, pprice, grid[i][1], adjusted_eprice_pprice_diff),
            qty_step,
        )
        if qty < calc_min_entry_qty(grid[i][1], inverse, qty_step, min_qty, min_cost):
            qty = 0.0
        psize, pprice = calc_new_psize_pprice(psize, pprice, qty, grid[i][1], qty_step)
        grid[i][0] = qty
        grid[i][2:] = [
            psize,
            pprice,
            qty_to_cost(psize, pprice, inverse, c_mult) / balance,
        ]
    return grid


@njit
def eval_entry_grid_short(
    balance,
    initial_entry_price,
    inverse,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    grid_span,
    wallet_exposure_limit,
    max_n_entry_orders,
    initial_qty_pct,
    eprice_pprice_diff,
    eprice_pprice_diff_wallet_exposure_weighting,
    eprice_exp_base=1.618034,
    eprices=None,
    prev_pprice=None,
):

    # returns [qty, price, psize, pprice, wallet_exposure]
    if eprices is None:
        grid = np.zeros((max_n_entry_orders, 5))
        grid[:, 1] = [
            round_up(p, price_step)
            for p in basespace(
                initial_entry_price,
                initial_entry_price * (1 + grid_span),
                eprice_exp_base,
                max_n_entry_orders,
            )
        ]
    else:
        max_n_entry_orders = len(eprices)
        grid = np.zeros((max_n_entry_orders, 5))
        grid[:, 1] = eprices

    grid[0][0] = -max(
        calc_min_entry_qty(grid[0][1], inverse, qty_step, min_qty, min_cost),
        round_(
            cost_to_qty(
                balance * wallet_exposure_limit * initial_qty_pct,
                initial_entry_price,
                inverse,
                c_mult,
            ),
            qty_step,
        ),
    )
    grid[0][2] = psize = grid[0][0]
    grid[0][3] = pprice = grid[0][1] if prev_pprice is None else prev_pprice
    grid[0][4] = qty_to_cost(psize, pprice, inverse, c_mult) / balance
    for i in range(1, max_n_entry_orders):
        adjusted_eprice_pprice_diff = eprice_pprice_diff * (
            1 + grid[i - 1][4] * eprice_pprice_diff_wallet_exposure_weighting
        )
        qty = round_(
            calc_entry_qty_short(psize, pprice, grid[i][1], adjusted_eprice_pprice_diff),
            qty_step,
        )
        if -qty < calc_min_entry_qty(grid[i][1], inverse, qty_step, min_qty, min_cost):
            qty = 0.0
        psize, pprice = calc_new_psize_pprice(psize, pprice, qty, grid[i][1], qty_step)
        grid[i][0] = qty
        grid[i][2:] = [
            psize,
            pprice,
            qty_to_cost(psize, pprice, inverse, c_mult) / balance,
        ]
    return grid


@njit
def calc_whole_entry_grid_long(
    balance,
    initial_entry_price,
    inverse,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    grid_span,
    wallet_exposure_limit,
    max_n_entry_orders,
    initial_qty_pct,
    eprice_pprice_diff,
    secondary_allocation,
    secondary_pprice_diff,
    eprice_exp_base=1.618034,
    eprices=None,
    prev_pprice=None,
):

    # [qty, price, psize, pprice, wallet_exposure]
    if secondary_allocation <= 0.05:
        # set to zero if secondary allocation less than 5%
        secondary_allocation = 0.0
    elif secondary_allocation >= 1.0:
        raise Exception("secondary_allocation cannot be >= 1.0")
    primary_wallet_exposure_allocation = 1.0 - secondary_allocation
    primary_wallet_exposure_limit = wallet_exposure_limit * primary_wallet_exposure_allocation
    eprice_pprice_diff_wallet_exposure_weighting = find_eprice_pprice_diff_wallet_exposure_weighting(
        True,
        balance,
        initial_entry_price,
        inverse,
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
        grid_span,
        primary_wallet_exposure_limit,
        max_n_entry_orders,
        initial_qty_pct / primary_wallet_exposure_allocation,
        eprice_pprice_diff,
        eprice_exp_base,
        eprices=eprices,
        prev_pprice=prev_pprice,
    )
    grid = eval_entry_grid_long(
        balance,
        initial_entry_price,
        inverse,
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
        grid_span,
        primary_wallet_exposure_limit,
        max_n_entry_orders,
        initial_qty_pct / primary_wallet_exposure_allocation,
        eprice_pprice_diff,
        eprice_pprice_diff_wallet_exposure_weighting,
        eprice_exp_base,
        eprices=eprices,
        prev_pprice=prev_pprice,
    )
    if secondary_allocation > 0.0:
        entry_price = min(
            round_dn(grid[-1][3] * (1 - secondary_pprice_diff), price_step), grid[-1][1]
        )
        qty = find_entry_qty_bringing_wallet_exposure_to_target(
            balance,
            grid[-1][2],
            grid[-1][3],
            wallet_exposure_limit,
            entry_price,
            inverse,
            qty_step,
            c_mult,
        )
        new_psize, new_pprice = calc_new_psize_pprice(
            grid[-1][2], grid[-1][3], qty, entry_price, qty_step
        )
        new_wallet_exposure = qty_to_cost(new_psize, new_pprice, inverse, c_mult) / balance
        grid = np.append(
            grid,
            np.array([[qty, entry_price, new_psize, new_pprice, new_wallet_exposure]]),
            axis=0,
        )
    return grid[grid[:, 0] > 0.0]


@njit
def calc_whole_entry_grid_short(
    balance,
    initial_entry_price,
    inverse,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    grid_span,
    wallet_exposure_limit,
    max_n_entry_orders,
    initial_qty_pct,
    eprice_pprice_diff,
    secondary_allocation,
    secondary_pprice_diff,
    eprice_exp_base=1.618034,
    eprices=None,
    prev_pprice=None,
):

    # [qty, price, psize, pprice, wallet_exposure]
    if secondary_allocation <= 0.05:
        # set to zero if secondary allocation less than 5%
        secondary_allocation = 0.0
    elif secondary_allocation >= 1.0:
        raise Exception("secondary_allocation cannot be >= 1.0")
    primary_wallet_exposure_allocation = 1.0 - secondary_allocation
    primary_wallet_exposure_limit = wallet_exposure_limit * primary_wallet_exposure_allocation
    eprice_pprice_diff_wallet_exposure_weighting = find_eprice_pprice_diff_wallet_exposure_weighting(
        False,
        balance,
        initial_entry_price,
        inverse,
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
        grid_span,
        primary_wallet_exposure_limit,
        max_n_entry_orders,
        initial_qty_pct / primary_wallet_exposure_allocation,
        eprice_pprice_diff,
        eprice_exp_base,
        eprices=eprices,
        prev_pprice=prev_pprice,
    )
    grid = eval_entry_grid_short(
        balance,
        initial_entry_price,
        inverse,
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
        grid_span,
        primary_wallet_exposure_limit,
        max_n_entry_orders,
        initial_qty_pct / primary_wallet_exposure_allocation,
        eprice_pprice_diff,
        eprice_pprice_diff_wallet_exposure_weighting,
        eprice_exp_base,
        eprices=eprices,
        prev_pprice=prev_pprice,
    )
    if secondary_allocation > 0.0:
        entry_price = max(
            round_up(grid[-1][3] * (1 + secondary_pprice_diff), price_step), grid[-1][1]
        )
        qty = -find_entry_qty_bringing_wallet_exposure_to_target(
            balance,
            grid[-1][2],
            grid[-1][3],
            wallet_exposure_limit,
            entry_price,
            inverse,
            qty_step,
            c_mult,
        )
        new_psize, new_pprice = calc_new_psize_pprice(
            grid[-1][2], grid[-1][3], qty, entry_price, qty_step
        )
        new_wallet_exposure = qty_to_cost(new_psize, new_pprice, inverse, c_mult) / balance
        grid = np.append(
            grid,
            np.array([[qty, entry_price, new_psize, new_pprice, new_wallet_exposure]]),
            axis=0,
        )
    return grid[grid[:, 0] < 0.0]


@njit
def calc_entry_grid_long(
    balance,
    psize,
    pprice,
    highest_bid,
    ema_band_lower,
    inverse,
    do_long,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    grid_span,
    wallet_exposure_limit,
    max_n_entry_orders,
    initial_qty_pct,
    initial_eprice_ema_dist,
    eprice_pprice_diff,
    secondary_allocation,
    secondary_pprice_diff,
    eprice_exp_base,
    auto_unstuck_wallet_exposure_threshold,
    auto_unstuck_ema_dist,
) -> [(float, float, str)]:
    if wallet_exposure_limit == 0.0:
        return [(0.0, 0.0, "")]
    min_entry_qty = calc_min_entry_qty(highest_bid, inverse, qty_step, min_qty, min_cost)
    if do_long or psize > min_entry_qty:
        if psize == 0.0:
            entry_price = min(
                highest_bid,
                round_dn(ema_band_lower * (1 - initial_eprice_ema_dist), price_step),
            )
            entry_qty = calc_initial_entry_qty(
                balance,
                entry_price,
                inverse,
                qty_step,
                min_qty,
                min_cost,
                c_mult,
                wallet_exposure_limit,
                initial_qty_pct,
            )
            return [(entry_qty, entry_price, "long_ientry")]
        else:
            wallet_exposure = qty_to_cost(psize, pprice, inverse, c_mult) / balance
            if wallet_exposure >= wallet_exposure_limit:
                return [(0.0, 0.0, "")]
            if auto_unstuck_wallet_exposure_threshold != 0.0:
                threshold = wallet_exposure_limit * (1 - auto_unstuck_wallet_exposure_threshold)
                if wallet_exposure > threshold * 0.99:
                    auto_unstuck_entry_price = min(
                        highest_bid,
                        round_dn(ema_band_lower * (1 - auto_unstuck_ema_dist), price_step),
                    )
                    auto_unstuck_qty = find_entry_qty_bringing_wallet_exposure_to_target(
                        balance,
                        psize,
                        pprice,
                        wallet_exposure_limit,
                        auto_unstuck_entry_price,
                        inverse,
                        qty_step,
                        c_mult,
                    )
                    min_entry_qty = calc_min_entry_qty(
                        auto_unstuck_entry_price, inverse, qty_step, min_qty, min_cost
                    )
                    return [
                        (
                            max(auto_unstuck_qty, min_entry_qty),
                            auto_unstuck_entry_price,
                            "long_unstuck_entry",
                        )
                    ]
            grid = approximate_long_grid(
                balance,
                psize,
                pprice,
                inverse,
                qty_step,
                price_step,
                min_qty,
                min_cost,
                c_mult,
                grid_span,
                wallet_exposure_limit,
                max_n_entry_orders,
                initial_qty_pct,
                eprice_pprice_diff,
                secondary_allocation,
                secondary_pprice_diff,
                eprice_exp_base=eprice_exp_base,
            )
            if len(grid) == 0:
                return [(0.0, 0.0, "")]
            if calc_diff(grid[0][3], grid[0][1]) < 0.00001:
                # means initial entry was partially filled
                entry_price = min(
                    highest_bid,
                    round_dn(ema_band_lower * (1 - initial_eprice_ema_dist), price_step),
                )
                min_entry_qty = calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost)
                max_entry_qty = round_(
                    cost_to_qty(
                        balance * wallet_exposure_limit * initial_qty_pct,
                        entry_price,
                        inverse,
                        c_mult,
                    ),
                    qty_step,
                )
                entry_qty = max(min_entry_qty, min(max_entry_qty, grid[0][0]))
                if (
                    qty_to_cost(entry_qty, entry_price, inverse, c_mult) / balance
                    > wallet_exposure_limit * 1.1
                ):
                    print("\n\nwarning: abnormally large partial ientry")
                    print("grid:")
                    for e in grid:
                        print(list(e))
                    print("args:")
                    print(
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
                        wallet_exposure_limit,
                        max_n_entry_orders,
                        initial_qty_pct,
                        eprice_pprice_diff,
                        secondary_allocation,
                        secondary_pprice_diff,
                        eprice_exp_base,
                    )
                    print("\n\n")
                return [(entry_qty, entry_price, "long_ientry")]
        if len(grid) == 0:
            return [(0.0, 0.0, "")]
        entries = []
        for i in range(len(grid)):
            if grid[i][2] < psize * 1.05 or grid[i][1] > pprice * 0.9995:
                continue
            if grid[i][4] > wallet_exposure_limit * 1.01:
                break
            entry_price = min(highest_bid, grid[i][1])
            min_entry_qty = calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost)
            grid[i][1] = entry_price
            grid[i][0] = max(min_entry_qty, grid[i][0])
            comment = (
                "long_secondary_rentry"
                if i == len(grid) - 1 and secondary_allocation > 0.05
                else "long_primary_rentry"
            )
            if not entries or (entries[-1][1] != entry_price):
                entries.append((grid[i][0], grid[i][1], comment))
        return entries if entries else [(0.0, 0.0, "")]
    return [(0.0, 0.0, "")]


@njit
def calc_entry_grid_short(
    balance,
    psize,
    pprice,
    lowest_ask,
    ema_band_upper,
    inverse,
    do_short,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    grid_span,
    wallet_exposure_limit,
    max_n_entry_orders,
    initial_qty_pct,
    initial_eprice_ema_dist,
    eprice_pprice_diff,
    secondary_allocation,
    secondary_pprice_diff,
    eprice_exp_base,
    auto_unstuck_wallet_exposure_threshold,
    auto_unstuck_ema_dist,
) -> [(float, float, str)]:
    if wallet_exposure_limit == 0.0:
        return [(0.0, 0.0, "")]
    min_entry_qty = calc_min_entry_qty(lowest_ask, inverse, qty_step, min_qty, min_cost)
    abs_psize = abs(psize)
    if do_short or abs_psize > min_entry_qty:
        if psize == 0.0:

            entry_price = max(
                lowest_ask,
                round_up(ema_band_upper * (1 + initial_eprice_ema_dist), price_step),
            )
            entry_qty = calc_initial_entry_qty(
                balance,
                entry_price,
                inverse,
                qty_step,
                min_qty,
                min_cost,
                c_mult,
                wallet_exposure_limit,
                initial_qty_pct,
            )
            return [(-entry_qty, entry_price, "short_ientry")]
        else:
            wallet_exposure = qty_to_cost(psize, pprice, inverse, c_mult) / balance
            if wallet_exposure >= wallet_exposure_limit:
                return [(0.0, 0.0, "")]
            if auto_unstuck_wallet_exposure_threshold != 0.0:
                threshold = wallet_exposure_limit * (1 - auto_unstuck_wallet_exposure_threshold)
                if wallet_exposure > threshold * 0.99:
                    auto_unstuck_entry_price = max(
                        lowest_ask,
                        round_up(ema_band_upper * (1 + auto_unstuck_ema_dist), price_step),
                    )
                    auto_unstuck_qty = find_entry_qty_bringing_wallet_exposure_to_target(
                        balance,
                        psize,
                        pprice,
                        wallet_exposure_limit,
                        auto_unstuck_entry_price,
                        inverse,
                        qty_step,
                        c_mult,
                    )
                    min_entry_qty = calc_min_entry_qty(
                        auto_unstuck_entry_price, inverse, qty_step, min_qty, min_cost
                    )
                    return [
                        (
                            -max(auto_unstuck_qty, min_entry_qty),
                            auto_unstuck_entry_price,
                            "short_unstuck_entry",
                        )
                    ]
            grid = approximate_short_grid(
                balance,
                psize,
                pprice,
                inverse,
                qty_step,
                price_step,
                min_qty,
                min_cost,
                c_mult,
                grid_span,
                wallet_exposure_limit,
                max_n_entry_orders,
                initial_qty_pct,
                eprice_pprice_diff,
                secondary_allocation,
                secondary_pprice_diff,
                eprice_exp_base=eprice_exp_base,
            )
            if len(grid) == 0:
                return [(0.0, 0.0, "")]
            if calc_diff(grid[0][3], grid[0][1]) < 0.00001:
                # means initial entry was partially filled
                entry_price = max(
                    lowest_ask,
                    round_up(ema_band_upper * (1 + initial_eprice_ema_dist), price_step),
                )
                min_entry_qty = calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost)
                max_entry_qty = round_(
                    cost_to_qty(
                        balance * wallet_exposure_limit * initial_qty_pct,
                        entry_price,
                        inverse,
                        c_mult,
                    ),
                    qty_step,
                )
                entry_qty = -max(min_entry_qty, min(max_entry_qty, abs(grid[0][0])))
                if (
                    qty_to_cost(entry_qty, entry_price, inverse, c_mult) / balance
                    > wallet_exposure_limit * 1.1
                ):
                    print("\n\nwarning: abnormally large partial ientry")
                    print("grid:")
                    for e in grid:
                        print(list(e))
                    print("args:")
                    print(
                        balance,
                        psize,
                        pprice,
                        lowest_ask,
                        inverse,
                        do_short,
                        qty_step,
                        price_step,
                        min_qty,
                        min_cost,
                        c_mult,
                        grid_span,
                        wallet_exposure_limit,
                        max_n_entry_orders,
                        initial_qty_pct,
                        eprice_pprice_diff,
                        secondary_allocation,
                        secondary_pprice_diff,
                        eprice_exp_base,
                    )
                    print("\n\n")
                return [(entry_qty, entry_price, "short_ientry")]
        if len(grid) == 0:
            return [(0.0, 0.0, "")]
        entries = []
        for i in range(len(grid)):
            if grid[i][2] > psize * 1.05 or grid[i][1] < pprice * 0.9995:
                continue
            entry_price = max(lowest_ask, grid[i][1])
            min_entry_qty = calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost)
            grid[i][1] = entry_price
            grid[i][0] = -max(min_entry_qty, abs(grid[i][0]))
            comment = (
                "short_secondary_rentry"
                if i == len(grid) - 1 and secondary_allocation > 0.05
                else "short_primary_rentry"
            )
            if not entries or (entries[-1][1] != entry_price):
                entries.append((grid[i][0], grid[i][1], comment))
        return entries if entries else [(0.0, 0.0, "")]
    return [(0.0, 0.0, "")]


@njit
def approximate_long_grid(
    balance,
    psize,
    pprice,
    inverse,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    grid_span,
    wallet_exposure_limit,
    max_n_entry_orders,
    initial_qty_pct,
    eprice_pprice_diff,
    secondary_allocation,
    secondary_pprice_diff,
    eprice_exp_base=1.618034,
    crop: bool = True,
):
    def eval_(ientry_price_guess, psize_):
        ientry_price_guess = round_(ientry_price_guess, price_step)
        grid = calc_whole_entry_grid_long(
            balance,
            ientry_price_guess,
            inverse,
            qty_step,
            price_step,
            min_qty,
            min_cost,
            c_mult,
            grid_span,
            wallet_exposure_limit,
            max_n_entry_orders,
            initial_qty_pct,
            eprice_pprice_diff,
            secondary_allocation,
            secondary_pprice_diff,
            eprice_exp_base=eprice_exp_base,
        )
        # find node whose psize is closest to psize
        diff, i = sorted([(abs(grid[i][2] - psize_) / psize_, i) for i in range(len(grid))])[0]
        return grid, diff, i

    if pprice == 0.0:
        raise Exception("cannot make grid without pprice")
    if psize == 0.0:
        return calc_whole_entry_grid_long(
            balance,
            pprice,
            inverse,
            qty_step,
            price_step,
            min_qty,
            min_cost,
            c_mult,
            grid_span,
            wallet_exposure_limit,
            max_n_entry_orders,
            initial_qty_pct,
            eprice_pprice_diff,
            secondary_allocation,
            secondary_pprice_diff,
            eprice_exp_base=eprice_exp_base,
        )

    grid, diff, i = eval_(pprice, psize)
    grid, diff, i = eval_(pprice * (pprice / grid[i][3]), psize)
    if diff < 0.01:
        # good guess
        grid, diff, i = eval_(grid[0][1] * (pprice / grid[i][3]), psize)
        return grid[i + 1 :] if crop else grid
    # no close matches
    # assume partial fill
    k = 0
    while k < len(grid) - 1 and grid[k][2] <= psize * 0.99999:
        # find first node whose psize > psize
        k += 1
    if k == 0:
        # means psize is less than iqty
        # return grid with adjusted iqty
        min_ientry_qty = calc_min_entry_qty(grid[0][1], inverse, qty_step, min_qty, min_cost)
        grid[0][0] = max(min_ientry_qty, round_(grid[0][0] - psize, qty_step))
        grid[0][2] = round_(psize + grid[0][0], qty_step)
        grid[0][4] = qty_to_cost(grid[0][2], grid[0][3], inverse, c_mult) / balance
        return grid
    if k == len(grid):
        # means wallet_exposure limit is exceeded
        return np.empty((0, 5)) if crop else grid
    for _ in range(5):
        # find grid as if partial fill were full fill
        remaining_qty = round_(grid[k][2] - psize, qty_step)
        npsize, npprice = calc_new_psize_pprice(psize, pprice, remaining_qty, grid[k][1], qty_step)
        grid, diff, i = eval_(npprice, npsize)
        if k >= len(grid):
            k = len(grid) - 1
            continue
        grid, diff, i = eval_(npprice * (npprice / grid[k][3]), npsize)
        k = 0
        while k < len(grid) - 1 and grid[k][2] <= psize * 0.99999:
            # find first node whose psize > psize
            k += 1
    min_entry_qty = calc_min_entry_qty(grid[k][1], inverse, qty_step, min_qty, min_cost)
    grid[k][0] = max(min_entry_qty, round_(grid[k][2] - psize, qty_step))
    return grid[k:] if crop else grid


@njit
def approximate_short_grid(
    balance,
    psize,
    pprice,
    inverse,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    grid_span,
    wallet_exposure_limit,
    max_n_entry_orders,
    initial_qty_pct,
    eprice_pprice_diff,
    secondary_allocation,
    secondary_pprice_diff,
    eprice_exp_base=1.618034,
    crop: bool = True,
):
    def eval_(ientry_price_guess, psize_):
        ientry_price_guess = round_(ientry_price_guess, price_step)
        grid = calc_whole_entry_grid_short(
            balance,
            ientry_price_guess,
            inverse,
            qty_step,
            price_step,
            min_qty,
            min_cost,
            c_mult,
            grid_span,
            wallet_exposure_limit,
            max_n_entry_orders,
            initial_qty_pct,
            eprice_pprice_diff,
            secondary_allocation,
            secondary_pprice_diff,
            eprice_exp_base=eprice_exp_base,
        )
        # find node whose psize is closest to psize
        abs_psize_ = abs(psize_)
        diff, i = sorted(
            [(abs(abs(grid[i][2]) - abs_psize_) / abs_psize_, i) for i in range(len(grid))]
        )[0]
        return grid, diff, i

    abs_psize = abs(psize)

    if pprice == 0.0:
        raise Exception("cannot make grid without pprice")
    if psize == 0.0:
        return calc_whole_entry_grid_short(
            balance,
            pprice,
            inverse,
            qty_step,
            price_step,
            min_qty,
            min_cost,
            c_mult,
            grid_span,
            wallet_exposure_limit,
            max_n_entry_orders,
            initial_qty_pct,
            eprice_pprice_diff,
            secondary_allocation,
            secondary_pprice_diff,
            eprice_exp_base=eprice_exp_base,
        )

    grid, diff, i = eval_(pprice, psize)
    grid, diff, i = eval_(pprice * (pprice / grid[i][3]), psize)
    if diff < 0.01:
        # good guess
        grid, diff, i = eval_(grid[0][1] * (pprice / grid[i][3]), psize)
        return grid[i + 1 :] if crop else grid
    # no close matches
    # assume partial fill
    k = 0
    while k < len(grid) - 1 and abs(grid[k][2]) <= abs_psize * 0.99999:
        # find first node whose psize > psize
        k += 1
    if k == 0:
        # means psize is less than iqty
        # return grid with adjusted iqty
        min_ientry_qty = calc_min_entry_qty(grid[0][1], inverse, qty_step, min_qty, min_cost)
        grid[0][0] = -max(min_ientry_qty, round_(abs(grid[0][0]) - abs_psize, qty_step))
        grid[0][2] = round_(psize + grid[0][0], qty_step)
        grid[0][4] = qty_to_cost(grid[0][2], grid[0][3], inverse, c_mult) / balance
        return grid
    if k == len(grid):
        # means wallet_exposure limit is exceeded
        return np.empty((0, 5)) if crop else grid
    for _ in range(5):
        # find grid as if partial fill were full fill
        remaining_qty = round_(grid[k][2] - psize, qty_step)
        npsize, npprice = calc_new_psize_pprice(psize, pprice, remaining_qty, grid[k][1], qty_step)
        grid, diff, i = eval_(npprice, npsize)
        if k >= len(grid):
            k = len(grid) - 1
            continue
        grid, diff, i = eval_(npprice * (npprice / grid[k][3]), npsize)
        k = 0
        while k < len(grid) - 1 and abs(grid[k][2]) <= abs_psize * 0.99999:
            # find first node whose psize > psize
            k += 1
    min_entry_qty = calc_min_entry_qty(grid[k][1], inverse, qty_step, min_qty, min_cost)
    grid[k][0] = -max(min_entry_qty, round_(abs(grid[k][2]) - abs_psize, qty_step))
    return grid[k:] if crop else grid


@njit
def backtest_static_grid(
    ticks,
    starting_balance,
    latency_simulation_ms,
    maker_fee,
    inverse,
    do_long,
    do_short,
    backwards_tp,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    ema_span_0,
    ema_span_1,
    eprice_exp_base,
    eprice_pprice_diff,
    grid_span,
    initial_eprice_ema_dist,
    initial_qty_pct,
    markup_range,
    max_n_entry_orders,
    min_markup,
    n_close_orders,
    wallet_exposure_limit,
    secondary_allocation,
    secondary_pprice_diff,
    auto_unstuck_ema_dist,
    auto_unstuck_wallet_exposure_threshold,
):
    if len(ticks[0]) == 3:
        timestamps = ticks[:, 0]
        closes = ticks[:, 2]
        lows = closes
        highs = closes
    else:
        timestamps = ticks[:, 0]
        highs = ticks[:, 1]
        lows = ticks[:, 2]
        closes = ticks[:, 3]

    balance_long = balance_short = equity_long = equity_short = starting_balance
    psize_long, pprice_long, psize_short, pprice_short = 0.0, 0.0, 0.0, 0.0

    fills_long, fills_short, stats = [], [], []

    entries_long = closes_long = [(0.0, 0.0, "")]
    entries_short = closes_short = [(0.0, 0.0, "")]
    bkr_price_long = bkr_price_short = 0.0

    next_entry_grid_update_ts_long = 0
    next_entry_grid_update_ts_short = 0
    next_close_grid_update_ts_long = 0
    next_close_grid_update_ts_short = 0
    next_stats_update = 0

    closest_bkr_long = closest_bkr_short = 1.0

    spans_multiplier = 60 / ((timestamps[1] - timestamps[0]) / 1000)

    spans_long = [ema_span_0[0], (ema_span_0[0] * ema_span_1[0]) ** 0.5, ema_span_1[0]]
    spans_long = np.array(sorted(spans_long)) * spans_multiplier if do_long else np.ones(3)
    spans_short = [ema_span_0[1], (ema_span_0[1] * ema_span_1[1]) ** 0.5, ema_span_1[1]]
    spans_short = np.array(sorted(spans_short)) * spans_multiplier if do_short else np.ones(3)
    assert max(spans_long) < len(closes), "max ema_span long larger than len(closes)"
    assert max(spans_short) < len(closes), "max ema_span short larger than len(closes)"
    spans_long = np.where(spans_long < 1.0, 1.0, spans_long)
    spans_short = np.where(spans_short < 1.0, 1.0, spans_short)
    max_span_long = int(round(max(spans_long)))
    max_span_short = int(round(max(spans_short)))
    emas_long, emas_short = np.repeat(closes[0], 3), np.repeat(closes[0], 3)
    alphas_long = 2.0 / (spans_long + 1.0)
    alphas__long = 1.0 - alphas_long
    alphas_short = 2.0 / (spans_short + 1.0)
    alphas__short = 1.0 - alphas_short

    long_wallet_exposure = 0.0
    short_wallet_exposure = 0.0
    long_wallet_exposure_auto_unstuck_threshold = (
        (wallet_exposure_limit[0] * (1 - auto_unstuck_wallet_exposure_threshold[0]))
        if auto_unstuck_wallet_exposure_threshold[0] != 0.0
        else wallet_exposure_limit[0] * 10
    )
    short_wallet_exposure_auto_unstuck_threshold = (
        (wallet_exposure_limit[1] * (1 - auto_unstuck_wallet_exposure_threshold[1]))
        if auto_unstuck_wallet_exposure_threshold[1] != 0.0
        else wallet_exposure_limit[1] * 10
    )

    for k in range(1, len(closes)):
        if do_long:
            emas_long = calc_ema(alphas_long, alphas__long, emas_long, closes[k - 1])
            if k >= max_span_long:
                # check bankruptcy
                bkr_diff_long = calc_diff(bkr_price_long, closes[k])
                closest_bkr_long = min(closest_bkr_long, bkr_diff_long)
                if closest_bkr_long < 0.06:
                    # consider bankruptcy within 6% as liquidation
                    if psize_long != 0.0:
                        fee_paid = -qty_to_cost(psize_long, pprice_long, inverse, c_mult) * maker_fee
                        pnl = calc_pnl_long(pprice_long, closes[k], -psize_long, inverse, c_mult)
                        balance_long = starting_balance * 1e-6
                        equity_long = 0.0
                        psize_long, pprice_long = 0.0, 0.0
                        fills_long.append(
                            (
                                k,
                                timestamps[k],
                                pnl,
                                fee_paid,
                                balance_long,
                                equity_long,
                                -psize_long,
                                closes[k],
                                0.0,
                                0.0,
                                "long_bankruptcy",
                            )
                        )
                    do_long = False
                    if not do_short:
                        stats.append(
                            (
                                next_stats_update,
                                bkr_price_long,
                                bkr_price_short,
                                psize_long,
                                pprice_long,
                                psize_short,
                                pprice_short,
                                closes[k],
                                closest_bkr_long,
                                closest_bkr_short,
                                balance_long,
                                balance_short,
                                equity_long,
                                equity_short,
                            )
                        )
                        return fills_long, fills_short, stats

                # check if long entry grid should be updated
                if timestamps[k] >= next_entry_grid_update_ts_long:
                    entries_long = calc_entry_grid_long(
                        balance_long,
                        psize_long,
                        pprice_long,
                        closes[k - 1],
                        min(emas_long),
                        inverse,
                        do_long,
                        qty_step,
                        price_step,
                        min_qty,
                        min_cost,
                        c_mult,
                        grid_span[0],
                        wallet_exposure_limit[0],
                        max_n_entry_orders[0],
                        initial_qty_pct[0],
                        initial_eprice_ema_dist[0],
                        eprice_pprice_diff[0],
                        secondary_allocation[0],
                        secondary_pprice_diff[0],
                        eprice_exp_base[0],
                        auto_unstuck_wallet_exposure_threshold[0],
                        auto_unstuck_ema_dist[0],
                    )
                    next_entry_grid_update_ts_long = timestamps[k] + 1000 * 60 * 5
                # check if close grid should be updated
                if timestamps[k] >= next_close_grid_update_ts_long:
                    closes_long = calc_close_grid_long(
                        backwards_tp[0],
                        balance_long,
                        psize_long,
                        pprice_long,
                        closes[k - 1],
                        max(emas_long),
                        inverse,
                        qty_step,
                        price_step,
                        min_qty,
                        min_cost,
                        c_mult,
                        wallet_exposure_limit[0],
                        min_markup[0],
                        markup_range[0],
                        n_close_orders[0],
                        auto_unstuck_wallet_exposure_threshold[0],
                        auto_unstuck_ema_dist[0],
                    )
                    next_close_grid_update_ts_long = timestamps[k] + 1000 * 60 * 5

                # check for long entry fills
                while entries_long and entries_long[0][0] > 0.0 and lows[k] < entries_long[0][1]:
                    next_entry_grid_update_ts_long = min(
                        next_entry_grid_update_ts_long, timestamps[k] + latency_simulation_ms
                    )
                    next_close_grid_update_ts_long = min(
                        next_close_grid_update_ts_long, timestamps[k] + latency_simulation_ms
                    )
                    psize_long, pprice_long = calc_new_psize_pprice(
                        psize_long,
                        pprice_long,
                        entries_long[0][0],
                        entries_long[0][1],
                        qty_step,
                    )
                    fee_paid = (
                        -qty_to_cost(entries_long[0][0], entries_long[0][1], inverse, c_mult)
                        * maker_fee
                    )
                    balance_long = max(starting_balance * 1e-6, balance_long + fee_paid)

                    equity_long = balance_long + calc_pnl_long(
                        pprice_long, closes[k], psize_long, inverse, c_mult
                    )
                    fills_long.append(
                        (
                            k,
                            timestamps[k],
                            0.0,
                            fee_paid,
                            balance_long,
                            equity_long,
                            entries_long[0][0],
                            entries_long[0][1],
                            psize_long,
                            pprice_long,
                            entries_long[0][2],
                        )
                    )
                    entries_long = entries_long[1:]
                    bkr_price_long = calc_bankruptcy_price(
                        balance_long,
                        psize_long,
                        pprice_long,
                        0.0,
                        0.0,
                        inverse,
                        c_mult,
                    )
                    long_wallet_exposure = (
                        qty_to_cost(psize_long, pprice_long, inverse, c_mult) / balance_long
                    )

                # check if long closes filled
                while (
                    psize_long > 0.0
                    and closes_long
                    and closes_long[0][0] < 0.0
                    and highs[k] > closes_long[0][1]
                ):
                    next_entry_grid_update_ts_long = min(
                        next_entry_grid_update_ts_long, timestamps[k] + latency_simulation_ms
                    )
                    next_close_grid_update_ts_long = min(
                        next_close_grid_update_ts_long, timestamps[k] + latency_simulation_ms
                    )
                    close_qty_long = closes_long[0][0]
                    new_psize_long = round_(psize_long + close_qty_long, qty_step)
                    if new_psize_long < 0.0:
                        print("warning: long close qty greater than long psize")
                        print("psize_long", psize_long)
                        print("pprice_long", pprice_long)
                        print("closes_long[0]", closes_long[0])
                        close_qty_long = -psize_long
                        new_psize_long, pprice_long = 0.0, 0.0
                    psize_long = new_psize_long
                    fee_paid = (
                        -qty_to_cost(close_qty_long, closes_long[0][1], inverse, c_mult) * maker_fee
                    )
                    pnl = calc_pnl_long(
                        pprice_long, closes_long[0][1], close_qty_long, inverse, c_mult
                    )
                    balance_long = max(starting_balance * 1e-6, balance_long + fee_paid + pnl)

                    equity_long = balance_long + calc_pnl_long(
                        pprice_long, closes[k], psize_long, inverse, c_mult
                    )
                    fills_long.append(
                        (
                            k,
                            timestamps[k],
                            pnl,
                            fee_paid,
                            balance_long,
                            equity_long,
                            close_qty_long,
                            closes_long[0][1],
                            psize_long,
                            pprice_long,
                            closes_long[0][2],
                        )
                    )
                    closes_long = closes_long[1:]
                    bkr_price_long = calc_bankruptcy_price(
                        balance_long,
                        psize_long,
                        pprice_long,
                        0.0,
                        0.0,
                        inverse,
                        c_mult,
                    )
                    long_wallet_exposure = (
                        qty_to_cost(psize_long, pprice_long, inverse, c_mult) / balance_long
                    )
                if psize_long == 0.0:
                    # update entry order
                    next_entry_grid_update_ts_long = min(
                        next_entry_grid_update_ts_long,
                        timestamps[k] + latency_simulation_ms,
                    )
                else:
                    if closes[k] > pprice_long:
                        # update closes after 2.5 sec
                        next_close_grid_update_ts_long = min(
                            next_close_grid_update_ts_long,
                            timestamps[k] + latency_simulation_ms + 2500,
                        )
                    elif long_wallet_exposure >= long_wallet_exposure_auto_unstuck_threshold:
                        # update both entry grid and closes after 15 secs
                        next_close_grid_update_ts_long = min(
                            next_close_grid_update_ts_long,
                            timestamps[k] + latency_simulation_ms + 15000,
                        )
                        next_entry_grid_update_ts_long = min(
                            next_entry_grid_update_ts_long,
                            timestamps[k] + latency_simulation_ms + 15000,
                        )

        if do_short:
            emas_short = calc_ema(alphas_short, alphas__short, emas_short, closes[k - 1])
            if k >= max_span_short:
                # check bankruptcy
                bkr_diff_short = calc_diff(bkr_price_short, closes[k])
                closest_bkr_short = min(closest_bkr_short, bkr_diff_short)

                if closest_bkr_short < 0.06:
                    # consider bankruptcy within 6% as liquidation
                    if psize_short != 0.0:
                        fee_paid = (
                            -qty_to_cost(psize_short, pprice_short, inverse, c_mult) * maker_fee
                        )
                        pnl = calc_pnl_short(pprice_short, closes[k], -psize_short, inverse, c_mult)
                        balance_short = starting_balance * 1e-6
                        equity_short = 0.0
                        psize_short, pprice_short = 0.0, 0.0
                        fills_short.append(
                            (
                                k,
                                timestamps[k],
                                pnl,
                                fee_paid,
                                balance_short,
                                equity_short,
                                -psize_short,
                                closes[k],
                                0.0,
                                0.0,
                                "short_bankruptcy",
                            )
                        )
                    do_short = False
                    if not do_long:
                        stats.append(
                            (
                                next_stats_update,
                                bkr_price_long,
                                bkr_price_short,
                                psize_long,
                                pprice_long,
                                psize_short,
                                pprice_short,
                                closes[k],
                                closest_bkr_long,
                                closest_bkr_short,
                                balance_long,
                                balance_short,
                                equity_long,
                                equity_short,
                            )
                        )
                        return fills_long, fills_short, stats

                # check if entry grid should be updated
                if timestamps[k] >= next_entry_grid_update_ts_short:
                    entries_short = calc_entry_grid_short(
                        balance_short,
                        psize_short,
                        pprice_short,
                        closes[k - 1],
                        max(emas_short),
                        inverse,
                        do_short,
                        qty_step,
                        price_step,
                        min_qty,
                        min_cost,
                        c_mult,
                        grid_span[1],
                        wallet_exposure_limit[1],
                        max_n_entry_orders[1],
                        initial_qty_pct[1],
                        initial_eprice_ema_dist[1],
                        eprice_pprice_diff[1],
                        secondary_allocation[1],
                        secondary_pprice_diff[1],
                        eprice_exp_base[1],
                        auto_unstuck_wallet_exposure_threshold[1],
                        auto_unstuck_ema_dist[1],
                    )
                    next_entry_grid_update_ts_short = timestamps[k] + 1000 * 60 * 5

                # check if close grid should be updated
                if timestamps[k] >= next_close_grid_update_ts_short:
                    closes_short = calc_close_grid_short(
                        backwards_tp[1],
                        balance_short,
                        psize_short,
                        pprice_short,
                        closes[k - 1],
                        min(emas_short),
                        inverse,
                        qty_step,
                        price_step,
                        min_qty,
                        min_cost,
                        c_mult,
                        wallet_exposure_limit[1],
                        min_markup[1],
                        markup_range[1],
                        n_close_orders[1],
                        auto_unstuck_wallet_exposure_threshold[1],
                        auto_unstuck_ema_dist[1],
                    )
                    next_close_grid_update_ts_short = timestamps[k] + 1000 * 60 * 5  # five mins delay

                while entries_short and entries_short[0][0] < 0.0 and highs[k] > entries_short[0][1]:
                    next_entry_grid_update_ts_short = min(
                        next_entry_grid_update_ts_short, timestamps[k] + latency_simulation_ms
                    )
                    next_close_grid_update_ts_short = min(
                        next_close_grid_update_ts_short, timestamps[k] + latency_simulation_ms
                    )
                    psize_short, pprice_short = calc_new_psize_pprice(
                        psize_short,
                        pprice_short,
                        entries_short[0][0],
                        entries_short[0][1],
                        qty_step,
                    )
                    fee_paid = (
                        -qty_to_cost(entries_short[0][0], entries_short[0][1], inverse, c_mult)
                        * maker_fee
                    )
                    balance_short = max(starting_balance * 1e-6, balance_short + fee_paid)
                    equity_short = balance_short + calc_pnl_short(
                        pprice_short, closes[k], psize_short, inverse, c_mult
                    )
                    fills_short.append(
                        (
                            k,
                            timestamps[k],
                            0.0,
                            fee_paid,
                            balance_short,
                            equity_short,
                            entries_short[0][0],
                            entries_short[0][1],
                            psize_short,
                            pprice_short,
                            entries_short[0][2],
                        )
                    )
                    entries_short = entries_short[1:]
                    bkr_price_short = calc_bankruptcy_price(
                        balance_short,
                        0.0,
                        0.0,
                        psize_short,
                        pprice_short,
                        inverse,
                        c_mult,
                    )
                    short_wallet_exposure = (
                        qty_to_cost(psize_short, pprice_short, inverse, c_mult) / balance_short
                    )

                # check if short closes filled
                while (
                    psize_short < 0.0
                    and closes_short
                    and closes_short[0][0] > 0.0
                    and lows[k] < closes_short[0][1]
                ):
                    next_entry_grid_update_ts_short = min(
                        next_entry_grid_update_ts_short, timestamps[k] + latency_simulation_ms
                    )
                    next_close_grid_update_ts_short = min(
                        next_close_grid_update_ts_short, timestamps[k] + latency_simulation_ms
                    )
                    close_qty_short = closes_short[0][0]
                    new_psize_short = round_(psize_short + close_qty_short, qty_step)
                    if new_psize_short > 0.0:
                        print("warning: short close qty greater than short psize")
                        print("psize_short", psize_short)
                        print("pprice_short", pprice_short)
                        print("closes_short[0]", closes_short[0])
                        close_qty_short = abs(psize_short)
                        new_psize_short, pprice_short = 0.0, 0.0
                    psize_short = new_psize_short
                    fee_paid = (
                        -qty_to_cost(close_qty_short, closes_short[0][1], inverse, c_mult) * maker_fee
                    )
                    pnl = calc_pnl_short(
                        pprice_short, closes_short[0][1], close_qty_short, inverse, c_mult
                    )
                    balance_short = max(starting_balance * 1e-6, balance_short + fee_paid + pnl)
                    equity_short = balance_short + calc_pnl_short(
                        pprice_short, closes[k], psize_short, inverse, c_mult
                    )
                    fills_short.append(
                        (
                            k,
                            timestamps[k],
                            pnl,
                            fee_paid,
                            balance_short,
                            equity_short,
                            close_qty_short,
                            closes_short[0][1],
                            psize_short,
                            pprice_short,
                            closes_short[0][2],
                        )
                    )
                    closes_short = closes_short[1:]
                    bkr_price_short = calc_bankruptcy_price(
                        balance_short,
                        0.0,
                        0.0,
                        psize_short,
                        pprice_short,
                        inverse,
                        c_mult,
                    )
                    short_wallet_exposure = (
                        qty_to_cost(psize_short, pprice_short, inverse, c_mult) / balance_short
                    )
                if psize_short == 0.0:
                    next_entry_grid_update_ts_short = min(
                        next_entry_grid_update_ts_short,
                        timestamps[k] + latency_simulation_ms,
                    )
                else:
                    if closes[k] < pprice_short:
                        next_close_grid_update_ts_short = min(
                            next_close_grid_update_ts_short,
                            timestamps[k] + latency_simulation_ms + 2500,
                        )
                    elif short_wallet_exposure >= short_wallet_exposure_auto_unstuck_threshold:
                        next_close_grid_update_ts_short = min(
                            next_close_grid_update_ts_short,
                            timestamps[k] + latency_simulation_ms + 15000,
                        )
                        next_entry_grid_update_ts_short = min(
                            next_entry_grid_update_ts_short,
                            timestamps[k] + latency_simulation_ms + 15000,
                        )

        # process stats
        if timestamps[k] >= next_stats_update:
            equity_long = balance_long + calc_pnl_long(
                pprice_long, closes[k], psize_long, inverse, c_mult
            )
            equity_short = balance_short + calc_pnl_short(
                pprice_short, closes[k], psize_short, inverse, c_mult
            )
            stats.append(
                (
                    timestamps[k],
                    bkr_price_long,
                    bkr_price_short,
                    psize_long,
                    pprice_long,
                    psize_short,
                    pprice_short,
                    closes[k],
                    closest_bkr_long,
                    closest_bkr_short,
                    balance_long,
                    balance_short,
                    equity_long,
                    equity_short,
                )
            )
            next_stats_update = round(timestamps[k] + 60 * 60 * 1000)

    stats.append(
        (
            next_stats_update,
            bkr_price_long,
            bkr_price_short,
            psize_long,
            pprice_long,
            psize_short,
            pprice_short,
            closes[k],
            closest_bkr_long,
            closest_bkr_short,
            balance_long,
            balance_short,
            equity_long,
            equity_short,
        )
    )
    return fills_long, fills_short, stats
