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
def round_dynamic_up(n: float, d: int) -> float:
    if n == 0.0:
        return n
    # Calculate the scaling factor
    shift = d - int(np.floor(np.log10(abs(n)))) - 1
    scaled_n = n * (10**shift)
    # Apply np.ceil to the scaled number and then scale back
    rounded_n = np.ceil(scaled_n) / (10**shift)
    return rounded_n


@njit
def round_dynamic_dn(n: float, d: int) -> float:
    if n == 0.0:
        return n
    # Calculate the scaling factor
    shift = d - int(np.floor(np.log10(abs(n)))) - 1
    scaled_n = n * (10**shift)
    # Apply np.floor to the scaled number and then scale back
    rounded_n = np.floor(scaled_n) / (10**shift)
    return rounded_n


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
def calc_min_entry_qty(price, inverse, c_mult, qty_step, min_qty, min_cost) -> float:
    return (
        min_qty
        if inverse
        else max(min_qty, round_up(cost_to_qty(min_cost, price, inverse, c_mult), qty_step))
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
def calc_pnl(pside, entry_price, close_price, qty, inverse, c_mult):
    if pside == "long":
        return calc_pnl_long(entry_price, close_price, qty, inverse, c_mult)
    if pside == "short":
        return calc_pnl_short(entry_price, close_price, qty, inverse, c_mult)
    raise Exception("unknown position side " + pside)


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
    if psize == 0.0:
        return qty, price
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
def calc_delay_between_fills_ms_bid(pprice, price, delay_between_fills_ms, delay_weight):
    # lowest delay is 1 minute
    # reduce delay between bids in some proportion to diff between pos price and market price
    pprice_diff = (pprice / price - 1) if price > 0.0 else 0.0
    return max(60000.0, delay_between_fills_ms * min(1.0, (1 - pprice_diff * delay_weight)))


@njit
def calc_delay_between_fills_ms_ask(pprice, price, delay_between_fills_ms, delay_weight):
    # lowest delay is 1 minute
    # reduce delay between asks in some proportion to diff between pos price and market price
    pprice_diff = (price / pprice - 1) if pprice > 0.0 else 0.0
    return max(60000.0, delay_between_fills_ms * min(1.0, (1 - pprice_diff * delay_weight)))


@njit
def calc_pprice_diff(pside: str, pprice: float, price: float):
    if pside == "long":
        return (1.0 - price / pprice) if pprice > 0.0 else 0.0
    elif pside == "short":
        return (price / pprice - 1.0) if pprice > 0.0 else 0.0
    else:
        raise Exception("unknown pside " + pside)


@njit
def calc_clock_qty(
    balance,
    wallet_exposure,
    entry_price,
    inverse,
    qty_step,
    min_qty,
    min_cost,
    c_mult,
    qty_pct,
    we_multiplier,
    wallet_exposure_limit,
):
    ratio = wallet_exposure / wallet_exposure_limit
    cost = balance * wallet_exposure_limit * qty_pct * (1 + ratio * we_multiplier)
    return max(
        calc_min_entry_qty(entry_price, inverse, c_mult, qty_step, min_qty, min_cost),
        round_(cost_to_qty(cost, entry_price, inverse, c_mult), qty_step),
    )


@njit
def calc_auto_unstuck_entry_long(
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
    auto_unstuck_ema_dist,
):
    # legacy AU mode
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
        auto_unstuck_entry_price, inverse, c_mult, qty_step, min_qty, min_cost
    )
    return (
        max(auto_unstuck_qty, min_entry_qty),
        auto_unstuck_entry_price,
        "long_unstuck_entry",
    )


@njit
def calc_auto_unstuck_entry_short(
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
    auto_unstuck_ema_dist,
):
    # legacy AU mode
    auto_unstuck_entry_price = max(
        lowest_ask,
        round_up(ema_band_upper * (1 + auto_unstuck_ema_dist), price_step),
    )
    auto_unstuck_qty = find_entry_qty_bringing_wallet_exposure_to_target(
        balance,
        abs(psize),
        pprice,
        wallet_exposure_limit,
        auto_unstuck_entry_price,
        inverse,
        qty_step,
        c_mult,
    )
    min_entry_qty = calc_min_entry_qty(
        auto_unstuck_entry_price, inverse, c_mult, qty_step, min_qty, min_cost
    )
    return (
        -max(auto_unstuck_qty, min_entry_qty),
        auto_unstuck_entry_price,
        "short_unstuck_entry",
    )


@njit
def calc_close_grid_long(
    backwards_tp,
    balance,
    psize,
    pprice,
    lowest_ask,
    ema_band_upper,
    utc_now_ms,
    prev_AU_fill_ts_close,
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
    auto_unstuck_delay_minutes,
    auto_unstuck_qty_pct,
):
    if backwards_tp:
        return calc_close_grid_backwards_long(
            balance,
            psize,
            pprice,
            lowest_ask,
            ema_band_upper,
            utc_now_ms,
            prev_AU_fill_ts_close,
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
            auto_unstuck_delay_minutes,
            auto_unstuck_qty_pct,
        )
    else:
        return calc_close_grid_frontwards_long(
            balance,
            psize,
            pprice,
            lowest_ask,
            ema_band_upper,
            utc_now_ms,
            prev_AU_fill_ts_close,
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
            auto_unstuck_delay_minutes,
            auto_unstuck_qty_pct,
        )


@njit
def calc_close_grid_short(
    backwards_tp,
    balance,
    psize,
    pprice,
    highest_bid,
    ema_band_lower,
    utc_now_ms,
    prev_AU_fill_ts_close,
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
    auto_unstuck_delay_minutes,
    auto_unstuck_qty_pct,
):
    if backwards_tp:
        return calc_close_grid_backwards_short(
            balance,
            psize,
            pprice,
            highest_bid,
            ema_band_lower,
            utc_now_ms,
            prev_AU_fill_ts_close,
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
            auto_unstuck_delay_minutes,
            auto_unstuck_qty_pct,
        )
    else:
        return calc_close_grid_frontwards_short(
            balance,
            psize,
            pprice,
            highest_bid,
            ema_band_lower,
            utc_now_ms,
            prev_AU_fill_ts_close,
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
            auto_unstuck_delay_minutes,
            auto_unstuck_qty_pct,
        )


@njit
def calc_auto_unstuck_close_long(
    balance,
    psize,
    pprice,
    lowest_ask,
    ema_band_upper,
    utc_now_ms,
    prev_AU_fill_ts_close,
    inverse,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    wallet_exposure_limit,
    auto_unstuck_wallet_exposure_threshold,
    auto_unstuck_ema_dist,
    auto_unstuck_delay_minutes,
    auto_unstuck_qty_pct,
    lowest_normal_close_price,
):
    threshold = wallet_exposure_limit * (1 - auto_unstuck_wallet_exposure_threshold)
    wallet_exposure = qty_to_cost(psize, pprice, inverse, c_mult) / balance
    if wallet_exposure > threshold:
        # auto unstuck active
        unstuck_close_qty = 0.0
        unstuck_close_price = max(
            lowest_ask, round_up(ema_band_upper * (1 + auto_unstuck_ema_dist), price_step)
        )
        if unstuck_close_price < lowest_normal_close_price:
            # auto unstuck price lower than lowest normal close price
            if auto_unstuck_delay_minutes != 0.0 and auto_unstuck_qty_pct != 0.0:
                # timed AU mode
                delay = calc_delay_between_fills_ms_ask(
                    pprice, lowest_ask, auto_unstuck_delay_minutes * 60 * 1000, 0.0
                )
                if utc_now_ms - prev_AU_fill_ts_close > delay:
                    # timer is up
                    unstuck_close_qty = min(
                        psize,
                        calc_clock_qty(
                            balance,
                            wallet_exposure,
                            unstuck_close_price,
                            inverse,
                            qty_step,
                            min_qty,
                            min_cost,
                            c_mult,
                            auto_unstuck_qty_pct,
                            0.0,
                            wallet_exposure_limit,
                        ),
                    )
            else:
                # legacy AU mode
                unstuck_close_qty = find_close_qty_long_bringing_wallet_exposure_to_target(
                    balance,
                    psize,
                    pprice,
                    threshold * 1.01,
                    unstuck_close_price,
                    inverse,
                    qty_step,
                    c_mult,
                )
        if unstuck_close_qty != 0.0:
            min_entry_qty = calc_min_entry_qty(
                unstuck_close_price, inverse, c_mult, qty_step, min_qty, min_cost
            )
            unstuck_close_qty = max(min_entry_qty, unstuck_close_qty)
            return (-unstuck_close_qty, unstuck_close_price, "unstuck_close_long")
    return (0.0, 0.0, "unstuck_close_long")


@njit
def calc_auto_unstuck_close_short(
    balance,
    psize,
    pprice,
    highest_bid,
    ema_band_lower,
    utc_now_ms,
    prev_AU_fill_ts_close,
    inverse,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    wallet_exposure_limit,
    auto_unstuck_wallet_exposure_threshold,
    auto_unstuck_ema_dist,
    auto_unstuck_delay_minutes,
    auto_unstuck_qty_pct,
    highest_normal_close_price,
):
    threshold = wallet_exposure_limit * (1 - auto_unstuck_wallet_exposure_threshold)
    wallet_exposure = qty_to_cost(psize, pprice, inverse, c_mult) / balance
    if wallet_exposure > threshold:
        # auto unstuck active
        unstuck_close_qty = 0.0
        unstuck_close_price = min(
            highest_bid, round_dn(ema_band_lower * (1 - auto_unstuck_ema_dist), price_step)
        )
        if unstuck_close_price > highest_normal_close_price:
            # auto unstuck price higher than highest normal close price
            if auto_unstuck_delay_minutes != 0.0 and auto_unstuck_qty_pct != 0.0:
                # timed AU mode
                delay = calc_delay_between_fills_ms_bid(
                    pprice, highest_bid, auto_unstuck_delay_minutes * 60 * 1000, 0.0
                )
                if utc_now_ms - prev_AU_fill_ts_close > delay:
                    # timer is up
                    unstuck_close_qty = min(
                        abs(psize),
                        calc_clock_qty(
                            balance,
                            wallet_exposure,
                            unstuck_close_price,
                            inverse,
                            qty_step,
                            min_qty,
                            min_cost,
                            c_mult,
                            auto_unstuck_qty_pct,
                            0.0,
                            wallet_exposure_limit,
                        ),
                    )
            else:
                # legacy AU mode
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
        if unstuck_close_qty != 0.0:
            min_entry_qty = calc_min_entry_qty(
                unstuck_close_price, inverse, c_mult, qty_step, min_qty, min_cost
            )
            unstuck_close_qty = max(min_entry_qty, unstuck_close_qty)
            return (unstuck_close_qty, unstuck_close_price, "unstuck_close_short")
    return (0.0, 0.0, "unstuck_close_short")


@njit
def calc_close_grid_backwards_long(
    balance,
    psize,
    pprice,
    lowest_ask,
    ema_band_upper,
    utc_now_ms,
    prev_AU_fill_ts_close,
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
    auto_unstuck_delay_minutes,
    auto_unstuck_qty_pct,
):
    psize = psize_ = round_dn(psize, qty_step)  # round down for spot
    if psize == 0.0:
        return [(0.0, 0.0, "")]
    minm = pprice * (1 + min_markup)
    full_psize = cost_to_qty(balance * wallet_exposure_limit, pprice, inverse, c_mult)
    n_close_orders = min(
        n_close_orders,
        full_psize / calc_min_entry_qty(pprice, inverse, c_mult, qty_step, min_qty, min_cost),
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
    closes = []
    if auto_unstuck_wallet_exposure_threshold != 0.0:
        # auto unstuck enabled
        auto_unstuck_close = calc_auto_unstuck_close_long(
            balance,
            psize,
            pprice,
            lowest_ask,
            ema_band_upper,
            utc_now_ms,
            prev_AU_fill_ts_close,
            inverse,
            qty_step,
            price_step,
            min_qty,
            min_cost,
            c_mult,
            wallet_exposure_limit,
            auto_unstuck_wallet_exposure_threshold,
            auto_unstuck_ema_dist,
            auto_unstuck_delay_minutes,
            auto_unstuck_qty_pct,
            close_prices[0],
        )
        if auto_unstuck_close[0] != 0.0:
            psize_ = round_(psize_ - abs(auto_unstuck_close[0]), qty_step)
            if psize_ < calc_min_entry_qty(
                auto_unstuck_close[1], inverse, c_mult, qty_step, min_qty, min_cost
            ):
                # close whole pos; include leftovers
                return [(-psize, auto_unstuck_close[1], "unstuck_close_long")]
            closes.append(auto_unstuck_close)
    if len(close_prices) == 1:
        if psize_ >= calc_min_entry_qty(
            close_prices[0], inverse, c_mult, qty_step, min_qty, min_cost
        ):
            closes.append((-psize_, close_prices[0], "long_nclose"))
        else:
            return [(0.0, 0.0, "")]
        return closes
    qty_per_close = max(min_qty, round_up(full_psize / len(close_prices_all), qty_step))
    for price in close_prices[::-1]:
        min_entry_qty = calc_min_entry_qty(price, inverse, c_mult, qty_step, min_qty, min_cost)
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
    utc_now_ms,
    prev_AU_fill_ts_close,
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
    auto_unstuck_delay_minutes,
    auto_unstuck_qty_pct,
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
    if auto_unstuck_wallet_exposure_threshold != 0.0:
        # auto unstuck enabled
        auto_unstuck_close = calc_auto_unstuck_close_long(
            balance,
            psize,
            pprice,
            lowest_ask,
            ema_band_upper,
            utc_now_ms,
            prev_AU_fill_ts_close,
            inverse,
            qty_step,
            price_step,
            min_qty,
            min_cost,
            c_mult,
            wallet_exposure_limit,
            auto_unstuck_wallet_exposure_threshold,
            auto_unstuck_ema_dist,
            auto_unstuck_delay_minutes,
            auto_unstuck_qty_pct,
            close_prices[0],
        )
        if auto_unstuck_close[0] != 0.0:
            psize_ = round_(psize_ - abs(auto_unstuck_close[0]), qty_step)
            if psize_ < calc_min_entry_qty(
                auto_unstuck_close[1], inverse, c_mult, qty_step, min_qty, min_cost
            ):
                # close whole pos; include leftovers
                return [(-psize, auto_unstuck_close[1], "unstuck_close_long")]
            closes.append(auto_unstuck_close)
    if len(close_prices) == 1:
        if psize_ >= calc_min_entry_qty(
            close_prices[0], inverse, c_mult, qty_step, min_qty, min_cost
        ):
            closes.append((-psize_, close_prices[0], "long_nclose"))
        else:
            return [(0.0, 0.0, "")]
        return closes if closes else [(0.0, 0.0, "")]
    default_close_qty = round_dn(psize_ / len(close_prices), qty_step)
    for price in close_prices[:-1]:
        min_close_qty = calc_min_entry_qty(price, inverse, c_mult, qty_step, min_qty, min_cost)
        if psize_ < min_close_qty:
            break
        close_qty = min(psize_, max(min_close_qty, default_close_qty))
        closes.append((-close_qty, price, "long_nclose"))
        psize_ = round_(psize_ - close_qty, qty_step)
    min_close_qty = calc_min_entry_qty(close_prices[-1], inverse, c_mult, qty_step, min_qty, min_cost)
    if psize_ >= min_close_qty:
        closes.append((-psize_, close_prices[-1], "long_nclose"))
    elif len(closes) > 0:
        closes[-1] = (-round_(abs(closes[-1][0]) + psize_, qty_step), closes[-1][1], closes[-1][2])
    return closes if closes else [(0.0, 0.0, "")]


@njit
def calc_close_grid_backwards_short(
    balance,
    psize,
    pprice,
    highest_bid,
    ema_band_lower,
    utc_now_ms,
    prev_AU_fill_ts_close,
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
    auto_unstuck_delay_minutes,
    auto_unstuck_qty_pct,
):
    psize = psize_ = round_dn(abs(psize), qty_step)  # round down for spot
    if psize == 0.0:
        return [(0.0, 0.0, "")]
    minm = pprice * (1 - min_markup)
    full_psize = cost_to_qty(balance * wallet_exposure_limit, pprice, inverse, c_mult)
    n_close_orders = min(
        n_close_orders,
        full_psize / calc_min_entry_qty(pprice, inverse, c_mult, qty_step, min_qty, min_cost),
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
    closes = []
    if auto_unstuck_wallet_exposure_threshold != 0.0:
        # auto unstuck enabled
        auto_unstuck_close = calc_auto_unstuck_close_short(
            balance,
            psize,
            pprice,
            highest_bid,
            ema_band_lower,
            utc_now_ms,
            prev_AU_fill_ts_close,
            inverse,
            qty_step,
            price_step,
            min_qty,
            min_cost,
            c_mult,
            wallet_exposure_limit,
            auto_unstuck_wallet_exposure_threshold,
            auto_unstuck_ema_dist,
            auto_unstuck_delay_minutes,
            auto_unstuck_qty_pct,
            close_prices[0],
        )
        if auto_unstuck_close[0] != 0.0:
            psize_ = round_(psize_ - abs(auto_unstuck_close[0]), qty_step)
            if psize_ < calc_min_entry_qty(
                auto_unstuck_close[1], inverse, c_mult, qty_step, min_qty, min_cost
            ):
                # close whole pos; include leftovers
                return [(psize, auto_unstuck_close[1], "unstuck_close_short")]
            closes.append(auto_unstuck_close)
    if len(close_prices) == 1:
        if psize_ >= calc_min_entry_qty(
            close_prices[0], inverse, c_mult, qty_step, min_qty, min_cost
        ):
            closes.append((psize_, close_prices[0], "short_nclose"))
        else:
            return [(0.0, 0.0, "")]
        return closes
    qty_per_close = max(min_qty, round_up(full_psize / len(close_prices_all), qty_step))
    for price in close_prices[::-1]:
        min_entry_qty = calc_min_entry_qty(price, inverse, c_mult, qty_step, min_qty, min_cost)
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
    utc_now_ms,
    prev_AU_fill_ts_close,
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
    auto_unstuck_delay_minutes,
    auto_unstuck_qty_pct,
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
    wallet_exposure = qty_to_cost(abs_psize, pprice, inverse, c_mult) / balance
    if auto_unstuck_wallet_exposure_threshold != 0.0:
        # auto unstuck enabled
        auto_unstuck_close = calc_auto_unstuck_close_short(
            balance,
            abs_psize,
            pprice,
            highest_bid,
            ema_band_lower,
            utc_now_ms,
            prev_AU_fill_ts_close,
            inverse,
            qty_step,
            price_step,
            min_qty,
            min_cost,
            c_mult,
            wallet_exposure_limit,
            auto_unstuck_wallet_exposure_threshold,
            auto_unstuck_ema_dist,
            auto_unstuck_delay_minutes,
            auto_unstuck_qty_pct,
            close_prices[0],
        )
        if auto_unstuck_close[0] != 0.0:
            abs_psize_ = round_(abs_psize_ - abs(auto_unstuck_close[0]), qty_step)
            if abs_psize_ < calc_min_entry_qty(
                auto_unstuck_close[1], inverse, c_mult, qty_step, min_qty, min_cost
            ):
                # close whole pos; include leftovers
                return [(abs_psize, auto_unstuck_close[1], "unstuck_close_short")]
            closes.append(auto_unstuck_close)
    if len(close_prices) == 1:
        if abs_psize_ >= calc_min_entry_qty(
            close_prices[0], inverse, c_mult, qty_step, min_qty, min_cost
        ):
            closes.append((abs_psize_, close_prices[0], "short_nclose"))
        else:
            return [(0.0, 0.0, "")]
        return closes if closes else [(0.0, 0.0, "")]
    default_close_qty = round_dn(abs_psize_ / len(close_prices), qty_step)
    for price in close_prices[:-1]:
        min_close_qty = calc_min_entry_qty(price, inverse, c_mult, qty_step, min_qty, min_cost)
        if abs_psize_ < min_close_qty:
            break
        close_qty = min(abs_psize_, max(min_close_qty, default_close_qty))
        closes.append((close_qty, price, "short_nclose"))
        abs_psize_ = round_(abs_psize_ - close_qty, qty_step)
    min_close_qty = calc_min_entry_qty(close_prices[-1], inverse, c_mult, qty_step, min_qty, min_cost)
    if abs_psize_ >= min_close_qty:
        closes.append((abs_psize_, close_prices[-1], "short_nclose"))
    elif len(closes) > 0:
        closes[-1] = (round_(closes[-1][0] + abs_psize_, qty_step), closes[-1][1], closes[-1][2])
    return closes if closes else [(0.0, 0.0, "")]


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
        a = -np.array([base**i for i in range(n)])
    else:
        a = np.array([base**i for i in range(n)])
    a = (a - a.min()) / (a.max() - a.min())
    return a * (end - start) + start


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
        calc_min_entry_qty(initial_entry_price, inverse, c_mult, qty_step, min_qty, min_cost),
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
                "balance, psize, pprice, wallet_exposure_target, close_price, inverse, c_mult, qty_step, c_mult,"
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
            "balance, psize, pprice, wallet_exposure_target, close_price, inverse, c_mult, qty_step, c_mult,"
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
                "balance, psize, pprice, wallet_exposure_target, close_price, inverse, c_mult, qty_step, c_mult,"
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
            "balance, psize, pprice, wallet_exposure_target, close_price, inverse, c_mult, qty_step, c_mult,"
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
            "balance, psize, pprice, wallet_exposure_target, entry_price, inverse, c_mult, qty_step,"
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
