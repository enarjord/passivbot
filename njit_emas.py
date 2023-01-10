from __future__ import annotations

import os

import numpy as np
import pandas as pd
from njit_funcs import (
    calc_pnl_long,
    calc_pnl_short,
    calc_new_psize_pprice,
    qty_to_cost,
    cost_to_qty,
    calc_ema,
    round_dn,
    round_up,
    round_,
    calc_bankruptcy_price,
    calc_close_grid_backwards_long,
    calc_close_grid_backwards_short,
    calc_min_entry_qty,
    find_entry_qty_bringing_wallet_exposure_to_target,
)

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
def calc_ema_price_bid(lower_ema_band, highest_bid, ema_dist_lower, price_step):
    return min(highest_bid, round_dn(lower_ema_band * (1 - ema_dist_lower), price_step))


@njit
def calc_ema_price_ask(upper_ema_band, lowest_ask, ema_dist_upper, price_step):
    return max(lowest_ask, round_up(upper_ema_band * (1 + ema_dist_upper), price_step))


@njit
def calc_ema_qty(
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
        calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost),
        round_(cost_to_qty(cost, entry_price, inverse, c_mult), qty_step),
    )


@njit
def calc_delay_between_fills_ms(delay_between_fills_ms, pprice_diff, delay_weight):
    # lowest delay is 1 minute
    # reduce trade delay in some proportion to pprice diff
    return max(60000.0, delay_between_fills_ms * (1 - pprice_diff * delay_weight))



def calc_orders_emas(
    inverse,
    long_enabled,
    short_enabled,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    highest_bid,
    lowest_ask,
    emas,
    ema_dist_lower,
    ema_dist_upper,
    qty_pct_entry_long,
    qty_pct_entry_short,
    qty_pct_close_long,
    qty_pct_close_short,
    we_multiplier_entry_long,
    we_multiplier_entry_short,
    we_multiplier_close_long,
    we_multiplier_close_short,
    delay_weight_entry_long,
    delay_weight_close_long,
    delay_weight_entry_short,
    delay_weight_close_short,
    delay_between_fills_minutes_entry_long,
    delay_between_fills_minutes_close_long,
    delay_between_fills_minutes_entry_short,
    delay_between_fills_minutes_close_short,
    min_markup_long,
    min_markup_short,
    markup_range_long,
    markup_range_short,
    n_close_orders,
    wallet_exposure_limit_long,
    wallet_exposure_limit_short,
):
    bid_price = calc_ema_price_bid(emas.min(), highest_bid, ema_dist_lower, price_step)
    ask_price = calc_ema_price_ask(emas.max(), lowest_ask, ema_dist_upper, price_step)
    




@njit
def backtest_emas(
    hlc,
    starting_balance,
    maker_fee,
    inverse,
    long_enabled,
    short_enabled,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    ema_span_0,
    ema_span_1,
    ema_dist_lower,
    ema_dist_upper,
    qty_pct_entry_long,
    qty_pct_entry_short,
    qty_pct_close_long,
    qty_pct_close_short,
    we_multiplier_entry_long,
    we_multiplier_entry_short,
    we_multiplier_close_long,
    we_multiplier_close_short,
    delay_weight_entry_long,
    delay_weight_close_long,
    delay_weight_entry_short,
    delay_weight_close_short,
    delay_between_fills_minutes_entry_long,
    delay_between_fills_minutes_close_long,
    delay_between_fills_minutes_entry_short,
    delay_between_fills_minutes_close_short,
    min_markup_long,
    min_markup_short,
    markup_range_long,
    markup_range_short,
    n_close_orders,
    wallet_exposure_limit_long,
    wallet_exposure_limit_short,
):
    # hlc [[ts, high, low, close]] 1m
    timestamps = hlc[:, 0]
    highs = hlc[:, 1]
    lows = hlc[:, 2]
    closes = hlc[:, 3]
    if wallet_exposure_limit_long == 0.0:
        long_enabled = False
    if wallet_exposure_limit_short == 0.0:
        short_enabled = False
    delay_between_fills_ms_entry_long = delay_between_fills_minutes_entry_long * 60 * 1000
    delay_between_fills_ms_close_long = delay_between_fills_minutes_close_long * 60 * 1000
    delay_between_fills_ms_entry_short = delay_between_fills_minutes_entry_short * 60 * 1000
    delay_between_fills_ms_close_short = delay_between_fills_minutes_close_short * 60 * 1000
    spans = np.array(sorted([ema_span_0, (ema_span_0 * ema_span_1) ** 0.5, ema_span_1]))
    psize_long, pprice_long, psize_short, pprice_short = 0.0, 0.0, 0.0, 0.0
    close_grid_long = [(0.0, np.inf, "")]
    close_grid_short = [(0.0, 0.0, "")]
    balance = starting_balance
    alphas = 2.0 / (spans + 1.0)
    alphas_ = 1.0 - alphas
    emas = np.repeat(closes[0], 3)
    prev_ema_fill_ts_entry_long, prev_ema_fill_ts_close_long = 0, 0
    prev_ema_fill_ts_entry_short, prev_ema_fill_ts_close_short = 0, 0
    fills, stats = [], []
    next_stats_update = 0
    closest_bkr = 1.0
    for k in range(1, len(hlc)):
        emas = calc_ema(alphas, alphas_, emas, closes[k - 1])
        # process stats
        if timestamps[k] >= next_stats_update:
            bkr_price = calc_bankruptcy_price(
                balance,
                psize_long,
                pprice_long,
                psize_short,
                psize_short,
                inverse,
                c_mult,
            )
            closest_bkr = min(closest_bkr, abs(bkr_price - closes[k]) / closes[k])
            upnl = calc_pnl_long(
                pprice_long, closes[k], psize_long, inverse, c_mult
            ) + calc_pnl_short(pprice_short, closes[k], psize_short, inverse, c_mult)
            equity = balance + upnl
            stats.append(
                (
                    timestamps[k],
                    bkr_price,
                    psize_long,
                    pprice_long,
                    psize_short,
                    pprice_short,
                    closes[k],
                    closest_bkr,
                    closest_bkr,
                    balance,
                    equity,
                )
            )
            if equity <= 0.0:
                # bankruptcy
                return fills, stats
            next_stats_update = min(timestamps[-1], timestamps[k] + 1000 * 60 * 60)  # hourly
        # check if markup close
        if psize_long > 0.0:
            if highs[k] > pprice_long:
                close_grid_long = calc_close_grid_backwards_long(
                    balance,
                    psize_long,
                    pprice_long,
                    closes[k - 1],
                    emas.max(),
                    inverse,
                    qty_step,
                    price_step,
                    min_qty,
                    min_cost,
                    c_mult,
                    wallet_exposure_limit_long,
                    min_markup_long,
                    markup_range_long,
                    n_close_orders,
                    0.0,
                    0.0,
                )
                if not close_grid_long:
                    # close remainder
                    close_grid_long = [(-psize_long, closes[k - 1], "close_markup_long")]
                if highs[k] > close_grid_long[0][1]:
                    while close_grid_long and highs[k] > close_grid_long[0][1]:
                        # close long pos
                        close_qty = abs(close_grid_long[0][0])
                        pnl = calc_pnl_long(
                            pprice_long, close_grid_long[0][1], close_qty, inverse, c_mult
                        )
                        fee_paid = (
                            -qty_to_cost(close_qty, close_grid_long[0][1], inverse, c_mult)
                            * maker_fee
                        )
                        balance += pnl + fee_paid
                        psize_long = max(0.0, round_(psize_long - close_qty, qty_step))
                        if psize_long == 0.0:
                            pprice_long = 0.0
                        upnl = calc_pnl_long(
                            pprice_long, closes[k], psize_long, inverse, c_mult
                        ) + calc_pnl_short(pprice_short, closes[k], psize_short, inverse, c_mult)
                        equity = balance + upnl
                        fills.append(
                            (
                                k,
                                timestamps[k],
                                pnl,
                                fee_paid,
                                balance,
                                equity,
                                -close_qty,
                                close_grid_long[0][1],
                                psize_long,
                                pprice_long,
                                "close_markup_long",
                            )
                        )
                        close_grid_long = close_grid_long[1:]
                    continue
        if psize_short > 0.0:
            if lows[k] < pprice_short:
                close_grid_short = calc_close_grid_backwards_short(
                    balance,
                    psize_short,
                    pprice_short,
                    closes[k - 1],
                    emas.min(),
                    inverse,
                    qty_step,
                    price_step,
                    min_qty,
                    min_cost,
                    c_mult,
                    wallet_exposure_limit_short,
                    min_markup_short,
                    markup_range_short,
                    n_close_orders,
                    0.0,
                    0.0,
                )
                if not close_grid_short:
                    # close remainder
                    close_grid_short = [(psize_short, closes[k - 1], "close_markup_short")]
                if lows[k] < close_grid_short[0][1]:
                    while close_grid_short and lows[k] < close_grid_short[0][1]:
                        # close short pos
                        close_qty = abs(close_grid_short[0][0])
                        pnl = calc_pnl_short(
                            pprice_short, close_grid_short[0][1], close_qty, inverse, c_mult
                        )
                        fee_paid = (
                            -qty_to_cost(close_qty, close_grid_short[0][1], inverse, c_mult)
                            * maker_fee
                        )
                        balance += pnl + fee_paid
                        psize_short = max(0.0, round_(psize_short - close_qty, qty_step))
                        if psize_short == 0.0:
                            pprice_short = 0.0
                        upnl = calc_pnl_short(
                            pprice_short, closes[k], psize_short, inverse, c_mult
                        ) + calc_pnl_short(pprice_short, closes[k], psize_short, inverse, c_mult)
                        equity = balance + upnl
                        fills.append(
                            (
                                k,
                                timestamps[k],
                                pnl,
                                fee_paid,
                                balance,
                                equity,
                                close_qty,
                                close_grid_short[0][1],
                                psize_short,
                                pprice_short,
                                "close_markup_short",
                            )
                        )
                        close_grid_short = close_grid_short[1:]
                    continue
        do_long_entry, do_long_close, do_short_entry, do_short_close = False, False, False, False
        both_zero = psize_long == 0.0 and psize_short == 0.0
        if long_enabled and (psize_long > 0.0 or both_zero):
            pprice_diff_long = (pprice_long / closes[k - 1] - 1) if pprice_long > 0.0 else 0.0
            do_long_entry = timestamps[k] - prev_ema_fill_ts_entry_long > calc_delay_between_fills_ms(
                delay_between_fills_ms_entry_long, pprice_diff_long, delay_weight_entry_long
            )
            do_long_close = timestamps[k] - prev_ema_fill_ts_close_long > calc_delay_between_fills_ms(
                delay_between_fills_ms_close_long, pprice_diff_long, delay_weight_close_long
            )
        if short_enabled and (psize_short > 0.0 or both_zero):
            pprice_diff_short = (closes[k - 1] / pprice_short - 1) if pprice_short > 0.0 else 0.0
            do_short_entry = timestamps[
                k
            ] - prev_ema_fill_ts_entry_short > calc_delay_between_fills_ms(
                delay_between_fills_ms_entry_short, pprice_diff_short, delay_weight_entry_short
            )
            do_short_close = timestamps[
                k
            ] - prev_ema_fill_ts_close_short > calc_delay_between_fills_ms(
                delay_between_fills_ms_close_short, pprice_diff_short, delay_weight_close_short
            )
        if do_long_entry or do_short_close:
            bid_price = calc_ema_price_bid(emas.min(), closes[k - 1], ema_dist_lower, price_step)
            if lows[k] < bid_price:
                # bid filled
                if psize_short > 0.0 and do_short_close:
                    # close short
                    prev_ema_fill_ts_close_short = timestamps[k]
                    wallet_exposure_short = (
                        qty_to_cost(psize_short, pprice_short, inverse, c_mult) / balance
                    )
                    qty_short = min(
                        psize_short,
                        calc_ema_qty(
                            balance,
                            wallet_exposure_short,
                            bid_price,
                            inverse,
                            qty_step,
                            min_qty,
                            min_cost,
                            c_mult,
                            qty_pct_close_short,
                            we_multiplier_close_short,
                            wallet_exposure_limit_short,
                        ),
                    )
                    psize_short = round_(psize_short - qty_short, qty_step)
                    pnl = calc_pnl_short(pprice_short, bid_price, qty_short, inverse, c_mult)
                    fee_paid = -qty_to_cost(qty_short, bid_price, inverse, c_mult) * maker_fee
                    balance += pnl + fee_paid
                    upnl = calc_pnl_long(
                        pprice_long, closes[k], psize_long, inverse, c_mult
                    ) + calc_pnl_short(pprice_short, closes[k], psize_short, inverse, c_mult)
                    equity = balance + upnl
                    if psize_short == 0.0:
                        pprice_short = 0.0
                    fills.append(
                        (
                            k,
                            timestamps[k],
                            pnl,
                            fee_paid,
                            balance,
                            equity,
                            qty_short,
                            bid_price,
                            -psize_short,
                            pprice_short,
                            "close_ema_short",
                        )
                    )
                    continue
                if long_enabled and do_long_entry:
                    wallet_exposure_long = (
                        qty_to_cost(psize_long, pprice_long, inverse, c_mult) / balance
                    )
                    if wallet_exposure_long < wallet_exposure_limit_long * 0.99:
                        # entry long
                        prev_ema_fill_ts_entry_long = timestamps[k]
                        qty_long = calc_ema_qty(
                            balance,
                            wallet_exposure_long,
                            bid_price,
                            inverse,
                            qty_step,
                            min_qty,
                            min_cost,
                            c_mult,
                            qty_pct_entry_long,
                            we_multiplier_entry_long,
                            wallet_exposure_limit_long,
                        )
                        new_psize_long, new_pprice_long = calc_new_psize_pprice(
                            psize_long, pprice_long, qty_long, bid_price, qty_step
                        )
                        wallet_exposure_after_fill = (
                            qty_to_cost(new_psize_long, new_pprice_long, inverse, c_mult) / balance
                        )
                        if wallet_exposure_after_fill > wallet_exposure_limit_long * 1.01:
                            qty_long = find_entry_qty_bringing_wallet_exposure_to_target(
                                balance,
                                psize_long,
                                pprice_long,
                                wallet_exposure_limit_long,
                                bid_price,
                                inverse,
                                qty_step,
                                c_mult,
                            )
                            new_psize_long, new_pprice_long = calc_new_psize_pprice(
                                psize_long, pprice_long, qty_long, bid_price, qty_step
                            )
                        if qty_long > 0.0:
                            psize_long, pprice_long = new_psize_long, new_pprice_long
                            upnl = calc_pnl_long(
                                pprice_long, closes[k], psize_long, inverse, c_mult
                            ) + calc_pnl_short(pprice_short, closes[k], psize_short, inverse, c_mult)
                            equity = balance + upnl
                            pnl = 0.0
                            fee_paid = -qty_to_cost(qty_long, bid_price, inverse, c_mult) * maker_fee
                            balance += fee_paid
                            fills.append(
                                (
                                    k,
                                    timestamps[k],
                                    pnl,
                                    fee_paid,
                                    balance,
                                    equity,
                                    qty_long,
                                    bid_price,
                                    psize_long,
                                    pprice_long,
                                    "entry_ema_long",
                                )
                            )
                            continue
        if do_short_entry or do_long_close:
            ask_price = calc_ema_price_ask(emas.max(), closes[k - 1], ema_dist_upper, price_step)
            if highs[k] > ask_price:
                # ask filled
                if psize_long > 0.0 and do_long_close:
                    # close long
                    prev_ema_fill_ts_close_long = timestamps[k]
                    wallet_exposure_long = (
                        qty_to_cost(psize_long, pprice_long, inverse, c_mult) / balance
                    )
                    qty_long = min(
                        psize_long,
                        calc_ema_qty(
                            balance,
                            wallet_exposure_long,
                            ask_price,
                            inverse,
                            qty_step,
                            min_qty,
                            min_cost,
                            c_mult,
                            qty_pct_close_long,
                            we_multiplier_close_long,
                            wallet_exposure_limit_long,
                        ),
                    )
                    psize_long = round_(psize_long - qty_long, qty_step)
                    pnl = calc_pnl_long(pprice_long, ask_price, qty_long, inverse, c_mult)
                    fee_paid = -qty_to_cost(qty_long, ask_price, inverse, c_mult) * maker_fee
                    balance += pnl + fee_paid
                    upnl = calc_pnl_long(
                        pprice_long, closes[k], psize_long, inverse, c_mult
                    ) + calc_pnl_short(pprice_short, closes[k], psize_short, inverse, c_mult)
                    equity = balance + upnl
                    if psize_long == 0.0:
                        pprice_long = 0.0
                    fills.append(
                        (
                            k,
                            timestamps[k],
                            pnl,
                            fee_paid,
                            balance,
                            equity,
                            -qty_long,
                            ask_price,
                            psize_long,
                            pprice_long,
                            "close_ema_long",
                        )
                    )
                    continue
                if short_enabled and do_short_entry:
                    wallet_exposure_short = (
                        qty_to_cost(psize_short, pprice_short, inverse, c_mult) / balance
                    )
                    if wallet_exposure_short < wallet_exposure_limit_short * 0.99:
                        # entry short
                        prev_ema_fill_ts_entry_short = timestamps[k]
                        qty_short = calc_ema_qty(
                            balance,
                            wallet_exposure_short,
                            ask_price,
                            inverse,
                            qty_step,
                            min_qty,
                            min_cost,
                            c_mult,
                            qty_pct_entry_short,
                            we_multiplier_entry_short,
                            wallet_exposure_limit_short,
                        )
                        new_psize_short, new_pprice_short = calc_new_psize_pprice(
                            psize_short, pprice_short, qty_short, ask_price, qty_step
                        )
                        wallet_exposure_after_fill = (
                            qty_to_cost(new_psize_short, new_pprice_short, inverse, c_mult) / balance
                        )
                        if wallet_exposure_after_fill > wallet_exposure_limit_short * 1.01:
                            qty_short = find_entry_qty_bringing_wallet_exposure_to_target(
                                balance,
                                psize_short,
                                pprice_short,
                                wallet_exposure_limit_short,
                                ask_price,
                                inverse,
                                qty_step,
                                c_mult,
                            )
                            new_psize_short, new_pprice_short = calc_new_psize_pprice(
                                psize_short, pprice_short, qty_short, ask_price, qty_step
                            )
                        if qty_short > 0.0:
                            psize_short, pprice_short = new_psize_short, new_pprice_short
                            upnl = calc_pnl_long(
                                pprice_long, closes[k], psize_long, inverse, c_mult
                            ) + calc_pnl_short(pprice_short, closes[k], psize_short, inverse, c_mult)
                            equity = balance + upnl
                            pnl = 0.0
                            fee_paid = -qty_to_cost(qty_short, ask_price, inverse, c_mult) * maker_fee
                            balance += fee_paid
                            fills.append(
                                (
                                    k,
                                    timestamps[k],
                                    pnl,
                                    fee_paid,
                                    balance,
                                    equity,
                                    -qty_short,
                                    ask_price,
                                    -psize_short,
                                    pprice_short,
                                    "entry_ema_short",
                                )
                            )
    return fills, stats
