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
    do_long,
    do_short,
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
    do_long,
    do_short,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    ema_span_0,
    ema_span_1,
    ema_dist_lower,
    ema_dist_upper,
    qty_pct_entry,
    qty_pct_close,
    we_multiplier_entry,
    we_multiplier_close,
    delay_weight_entry,
    delay_weight_close,
    delay_between_fills_minutes_entry,
    delay_between_fills_minutes_close,
    min_markup,
    markup_range,
    n_close_orders,
    wallet_exposure_limit,
):
    # hlc [[ts, high, low, close]] 1m
    timestamps = hlc[:, 0]
    highs = hlc[:, 1]
    lows = hlc[:, 2]
    closes = hlc[:, 3]
    if wallet_exposure_limit[0] == 0.0:
        do_long = False
    if wallet_exposure_limit[1] == 0.0:
        do_short = False
    # (long, short)
    delay_between_fills_ms_entry = (
        delay_between_fills_minutes_entry[0] * 60 * 1000,
        delay_between_fills_minutes_entry[1] * 60 * 1000,
    )
    delay_between_fills_ms_close = (
        delay_between_fills_minutes_close[0] * 60 * 1000,
        delay_between_fills_minutes_close[1] * 60 * 1000,
    )
    psize_long, pprice_long, psize_short, pprice_short = 0.0, 0.0, 0.0, 0.0
    close_grid_long = [(0.0, np.inf, "")]
    close_grid_short = [(0.0, 0.0, "")]
    balance_long, balance_short = starting_balance, starting_balance

    spans_long = [ema_span_0[0], (ema_span_0[0] * ema_span_1[0]) ** 0.5, ema_span_1[0]]
    spans_long = np.array(sorted(spans_long)) if do_long else np.ones(3)
    spans_short = [ema_span_0[1], (ema_span_0[1] * ema_span_1[1]) ** 0.5, ema_span_1[1]]
    spans_short = np.array(sorted(spans_short)) if do_short else np.ones(3)
    assert max(spans_long) < len(hlc), "ema span long larger than len(prices)"
    assert max(spans_short) < len(hlc), "ema span short larger than len(prices)"
    spans_long = np.where(spans_long < 1.0, 1.0, spans_long)
    spans_short = np.where(spans_short < 1.0, 1.0, spans_short)
    max_span_long = int(round(max(spans_long)))
    max_span_short = int(round(max(spans_short)))
    emas_long, emas_short = np.repeat(closes[0], 3), np.repeat(closes[0], 3)
    alphas_long = 2.0 / (spans_long + 1.0)
    alphas__long = 1.0 - alphas_long
    alphas_short = 2.0 / (spans_short + 1.0)
    alphas__short = 1.0 - alphas_short

    prev_ema_fill_ts_entry_long, prev_ema_fill_ts_close_long = 0, 0
    prev_ema_fill_ts_entry_short, prev_ema_fill_ts_close_short = 0, 0
    fills_long, fills_short, stats = [], [], []
    next_stats_update = 0
    closest_bkr_long, closest_bkr_short = 1.0, 1.0
    for k in range(1, len(hlc)):
        # process stats
        if timestamps[k] >= next_stats_update:
            bkr_price_long = calc_bankruptcy_price(
                balance_long,
                psize_long,
                pprice_long,
                0.0,
                0.0,
                inverse,
                c_mult,
            )
            bkr_price_short = calc_bankruptcy_price(
                balance_short,
                0.0,
                0.0,
                psize_short,
                pprice_short,
                inverse,
                c_mult,
            )
            closest_bkr_long = min(closest_bkr_long, abs(bkr_price_long - closes[k]) / closes[k])
            closest_bkr_short = min(closest_bkr_short, abs(bkr_price_short - closes[k]) / closes[k])
            upnl_long = calc_pnl_long(pprice_long, closes[k], psize_long, inverse, c_mult)
            upnl_short = calc_pnl_short(pprice_short, closes[k], psize_short, inverse, c_mult)
            equity_long = balance_long + upnl_long
            equity_short = balance_short + upnl_short

            stats.append(
                (
                    timestamps[k],
                    bkr_price_long,
                    bkr_price_short,
                    psize_long,
                    pprice_long,
                    -psize_short,
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
            if equity_long <= 0.05:
                do_long = False
            if equity_short <= 0.05:
                do_short = False
            if not (do_long and do_short):
                # bankruptcy
                return fills_long, fills_short, stats
            next_stats_update = min(timestamps[-1], timestamps[k] + 1000 * 60 * 60)  # hourly
        if do_long:
            emas_long = calc_ema(alphas_long, alphas__long, emas_long, closes[k - 1])
            # check if markup close
            if psize_long > 0.0 and highs[k] > pprice_long:
                close_grid_long = calc_close_grid_backwards_long(
                    balance_long,
                    psize_long,
                    pprice_long,
                    closes[k - 1],
                    emas_long.max(),
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
                    0.0,
                    0.0,
                )
                if not close_grid_long:
                    # close remainder
                    close_grid_long = [
                        (
                            -psize_long,
                            round_up(pprice_long * (1 + min_markup[0]), price_step),
                            "close_markup_long",
                        )
                    ]
                while close_grid_long and highs[k] > close_grid_long[0][1]:
                    # close long pos
                    close_qty = abs(close_grid_long[0][0])
                    pnl = calc_pnl_long(
                        pprice_long, close_grid_long[0][1], close_qty, inverse, c_mult
                    )
                    fee_paid = (
                        -qty_to_cost(close_qty, close_grid_long[0][1], inverse, c_mult) * maker_fee
                    )
                    balance_long += pnl + fee_paid
                    psize_long = max(0.0, round_(psize_long - close_qty, qty_step))
                    if psize_long == 0.0:
                        pprice_long = 0.0
                        prev_ema_fill_ts_entry_long, prev_ema_fill_ts_close_long = 0, 0
                    upnl = calc_pnl_long(pprice_long, closes[k], psize_long, inverse, c_mult)
                    equity_long = balance_long + upnl
                    fills_long.append(
                        (
                            k,
                            timestamps[k],
                            pnl,
                            fee_paid,
                            balance_long,
                            equity_long,
                            -close_qty,
                            close_grid_long[0][1],
                            psize_long,
                            pprice_long,
                            "close_markup_long",
                        )
                    )
                    close_grid_long = close_grid_long[1:]
            # check if ema entry long
            pprice_diff_long = (pprice_long / closes[k - 1] - 1) if pprice_long > 0.0 else 0.0
            if timestamps[k - 1] - prev_ema_fill_ts_entry_long > calc_delay_between_fills_ms(
                delay_between_fills_ms_entry[0], pprice_diff_long, delay_weight_entry[0]
            ):
                bid_price_long = calc_ema_price_bid(
                    emas_long.min(), closes[k - 1], ema_dist_lower[0], price_step
                )
                if lows[k] < bid_price_long:
                    # long ema entry
                    wallet_exposure_long = (
                        qty_to_cost(psize_long, pprice_long, inverse, c_mult) / balance_long
                    )
                    if wallet_exposure_long < wallet_exposure_limit[0] * 0.99:
                        # entry long
                        prev_ema_fill_ts_entry_long = timestamps[k]
                        qty_long = calc_ema_qty(
                            balance_long,
                            wallet_exposure_long,
                            bid_price_long,
                            inverse,
                            qty_step,
                            min_qty,
                            min_cost,
                            c_mult,
                            qty_pct_entry[0],
                            we_multiplier_entry[0],
                            wallet_exposure_limit[0],
                        )
                        new_psize_long, new_pprice_long = calc_new_psize_pprice(
                            psize_long, pprice_long, qty_long, bid_price_long, qty_step
                        )
                        wallet_exposure_after_fill = (
                            qty_to_cost(new_psize_long, new_pprice_long, inverse, c_mult)
                            / balance_long
                        )
                        if wallet_exposure_after_fill > wallet_exposure_limit[0] * 1.01:
                            qty_long = find_entry_qty_bringing_wallet_exposure_to_target(
                                balance_long,
                                psize_long,
                                pprice_long,
                                wallet_exposure_limit[0],
                                bid_price_long,
                                inverse,
                                qty_step,
                                c_mult,
                            )
                            new_psize_long, new_pprice_long = calc_new_psize_pprice(
                                psize_long, pprice_long, qty_long, bid_price_long, qty_step
                            )
                        if qty_long > 0.0:
                            psize_long, pprice_long = new_psize_long, new_pprice_long
                            upnl = calc_pnl_long(pprice_long, closes[k], psize_long, inverse, c_mult)
                            equity_long = balance_long + upnl
                            pnl = 0.0
                            fee_paid = (
                                -qty_to_cost(qty_long, bid_price_long, inverse, c_mult) * maker_fee
                            )
                            balance_long += fee_paid
                            fills_long.append(
                                (
                                    k,
                                    timestamps[k],
                                    pnl,
                                    fee_paid,
                                    balance_long,
                                    equity_long,
                                    qty_long,
                                    bid_price_long,
                                    psize_long,
                                    pprice_long,
                                    "entry_ema_long",
                                )
                            )
            # check if ema close long
            delay = calc_delay_between_fills_ms(
                delay_between_fills_ms_close[0], pprice_diff_long, delay_weight_close[0]
            )
            if psize_long > 0.0 and timestamps[k - 1] - prev_ema_fill_ts_close_long > delay:
                ask_price_long = calc_ema_price_ask(
                    emas_long.max(), closes[k - 1], ema_dist_upper[0], price_step
                )
                if highs[k] > ask_price_long:
                    # close long
                    prev_ema_fill_ts_close_long = timestamps[k]
                    wallet_exposure_long = (
                        qty_to_cost(psize_long, pprice_long, inverse, c_mult) / balance_long
                    )
                    qty_long = min(
                        psize_long,
                        calc_ema_qty(
                            balance_long,
                            wallet_exposure_long,
                            ask_price_long,
                            inverse,
                            qty_step,
                            min_qty,
                            min_cost,
                            c_mult,
                            qty_pct_close[0],
                            we_multiplier_close[0],
                            wallet_exposure_limit[0],
                        ),
                    )
                    psize_long = round_(psize_long - qty_long, qty_step)
                    pnl = calc_pnl_long(pprice_long, ask_price_long, qty_long, inverse, c_mult)
                    fee_paid = -qty_to_cost(qty_long, ask_price_long, inverse, c_mult) * maker_fee
                    balance_long += pnl + fee_paid
                    upnl = calc_pnl_long(pprice_long, closes[k], psize_long, inverse, c_mult)
                    equity_long = balance_long + upnl
                    if psize_long == 0.0:
                        pprice_long = 0.0
                        prev_ema_fill_ts_entry_long, prev_ema_fill_ts_close_long = 0, 0
                    fills_long.append(
                        (
                            k,
                            timestamps[k],
                            pnl,
                            fee_paid,
                            balance_long,
                            equity_long,
                            -qty_long,
                            ask_price_long,
                            psize_long,
                            pprice_long,
                            "close_ema_long",
                        )
                    )
        if do_short:
            emas_short = calc_ema(alphas_short, alphas__short, emas_short, closes[k - 1])
            # check if markup close
            if psize_short > 0.0 and lows[k] < pprice_short:
                close_grid_short = calc_close_grid_backwards_short(
                    balance_short,
                    psize_short,
                    pprice_short,
                    closes[k - 1],
                    emas_short.min(),
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
                    0.0,
                    0.0,
                )
                if not close_grid_short:
                    # close remainder
                    close_grid_short = [
                        (
                            psize_short,
                            round_dn(pprice_short * (1 - min_markup[1]), price_step),
                            "close_markup_short",
                        )
                    ]
                while close_grid_short and lows[k] < close_grid_short[0][1]:
                    # close short pos
                    close_qty = abs(close_grid_short[0][0])
                    pnl = calc_pnl_short(
                        pprice_short, close_grid_short[0][1], close_qty, inverse, c_mult
                    )
                    fee_paid = (
                        -qty_to_cost(close_qty, close_grid_short[0][1], inverse, c_mult) * maker_fee
                    )
                    balance_short += pnl + fee_paid
                    psize_short = max(0.0, round_(psize_short - close_qty, qty_step))
                    if psize_short == 0.0:
                        pprice_short = 0.0
                        prev_ema_fill_ts_entry_short, prev_ema_fill_ts_close_short = 0, 0
                    upnl = calc_pnl_short(pprice_short, closes[k], psize_short, inverse, c_mult)
                    equity_short = balance_short + upnl
                    fills_short.append(
                        (
                            k,
                            timestamps[k],
                            pnl,
                            fee_paid,
                            balance_short,
                            equity_short,
                            close_qty,
                            close_grid_short[0][1],
                            -psize_short,
                            pprice_short,
                            "close_markup_short",
                        )
                    )
                    close_grid_short = close_grid_short[1:]

            pprice_diff_short = (closes[k - 1] / pprice_short - 1) if pprice_short > 0.0 else 0.0
            if timestamps[k] - prev_ema_fill_ts_entry_short > calc_delay_between_fills_ms(
                delay_between_fills_ms_entry[1], pprice_diff_short, delay_weight_entry[1]
            ):
                ask_price_short = calc_ema_price_ask(
                    emas_short.max(), closes[k - 1], ema_dist_upper[1], price_step
                )
                if highs[k] > ask_price_short:
                    wallet_exposure_short = (
                        qty_to_cost(psize_short, pprice_short, inverse, c_mult) / balance_short
                    )
                    if wallet_exposure_short < wallet_exposure_limit[1] * 0.99:
                        # entry ema short
                        prev_ema_fill_ts_entry_short = timestamps[k]
                        qty_short = calc_ema_qty(
                            balance_short,
                            wallet_exposure_short,
                            ask_price_short,
                            inverse,
                            qty_step,
                            min_qty,
                            min_cost,
                            c_mult,
                            qty_pct_entry[1],
                            we_multiplier_entry[1],
                            wallet_exposure_limit[1],
                        )
                        new_psize_short, new_pprice_short = calc_new_psize_pprice(
                            psize_short, pprice_short, qty_short, ask_price_short, qty_step
                        )
                        wallet_exposure_after_fill = (
                            qty_to_cost(new_psize_short, new_pprice_short, inverse, c_mult)
                            / balance_short
                        )
                        if wallet_exposure_after_fill > wallet_exposure_limit[1] * 1.01:
                            qty_short = find_entry_qty_bringing_wallet_exposure_to_target(
                                balance_short,
                                psize_short,
                                pprice_short,
                                wallet_exposure_limit[1],
                                ask_price_short,
                                inverse,
                                qty_step,
                                c_mult,
                            )
                            new_psize_short, new_pprice_short = calc_new_psize_pprice(
                                psize_short, pprice_short, qty_short, ask_price_short, qty_step
                            )
                        if qty_short > 0.0:
                            psize_short, pprice_short = new_psize_short, new_pprice_short
                            upnl = calc_pnl_short(
                                pprice_short, closes[k], psize_short, inverse, c_mult
                            )
                            equity_short = balance_short + upnl
                            pnl = 0.0
                            fee_paid = (
                                -qty_to_cost(qty_short, ask_price_short, inverse, c_mult) * maker_fee
                            )
                            balance_short += fee_paid
                            fills_short.append(
                                (
                                    k,
                                    timestamps[k],
                                    pnl,
                                    fee_paid,
                                    balance_short,
                                    equity_short,
                                    -qty_short,
                                    ask_price_short,
                                    -psize_short,
                                    pprice_short,
                                    "entry_ema_short",
                                )
                            )
            delay = calc_delay_between_fills_ms(
                delay_between_fills_ms_close[1], pprice_diff_short, delay_weight_close[1]
            )
            if psize_short > 0.0 and timestamps[k] - prev_ema_fill_ts_close_short > delay:
                bid_price_short = calc_ema_price_bid(
                    emas_short.min(), closes[k - 1], ema_dist_lower[1], price_step
                )
                if lows[k] < bid_price_short:
                    # close short
                    prev_ema_fill_ts_close_short = timestamps[k]
                    wallet_exposure_short = (
                        qty_to_cost(psize_short, pprice_short, inverse, c_mult) / balance_short
                    )
                    qty_short = min(
                        psize_short,
                        calc_ema_qty(
                            balance_short,
                            wallet_exposure_short,
                            bid_price_short,
                            inverse,
                            qty_step,
                            min_qty,
                            min_cost,
                            c_mult,
                            qty_pct_close[1],
                            we_multiplier_close[1],
                            wallet_exposure_limit[1],
                        ),
                    )
                    psize_short = round_(psize_short - qty_short, qty_step)
                    pnl = calc_pnl_short(pprice_short, bid_price_short, qty_short, inverse, c_mult)
                    fee_paid = -qty_to_cost(qty_short, bid_price_short, inverse, c_mult) * maker_fee
                    balance_short += pnl + fee_paid
                    upnl = calc_pnl_short(pprice_short, closes[k], psize_short, inverse, c_mult)
                    equity_short = balance_short + upnl
                    if psize_short == 0.0:
                        pprice_short = 0.0
                        prev_ema_fill_ts_entry_short, prev_ema_fill_ts_close_short = 0, 0
                    fills_short.append(
                        (
                            k,
                            timestamps[k],
                            pnl,
                            fee_paid,
                            balance_short,
                            equity_short,
                            qty_short,
                            bid_price_short,
                            -psize_short,
                            pprice_short,
                            "close_ema_short",
                        )
                    )

    return fills_long, fills_short, stats
