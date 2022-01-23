import os

# os.environ["NOJIT"] = "true"

import numpy as np
from numba import njit
from njit_funcs import (
    round_dn,
    round_,
    calc_min_entry_qty,
    cost_to_qty,
    qty_to_cost,
    calc_new_psize_pprice,
    calc_bankruptcy_price,
    calc_ema,
    calc_diff,
    calc_long_pnl,
    calc_short_pnl,
    calc_upnl,
    calc_equity,
    calc_emas_last,
    calc_wallet_exposure_if_filled,
    find_entry_qty_bringing_wallet_exposure_to_target,
    calc_long_close_grid,
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
def calc_long_entry(
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
    iqty_pct,
    iprice_ema_dist,
    ddown_factor,
    rentry_pprice_dist,
    rentry_pprice_dist_wallet_exposure_weighting,
    wallet_exposure_limit,
    auto_unstuck_ema_dist,
    auto_unstuck_wallet_exposure_threshold,
):
    ientry_price = min(highest_bid, round_dn(ema_band_lower * (1 - iprice_ema_dist), price_step))
    min_entry_qty = calc_min_entry_qty(ientry_price, inverse, qty_step, min_qty, min_cost)
    ientry_qty = max(
        min_entry_qty,
        round_(
            cost_to_qty(balance, ientry_price, inverse, c_mult) * wallet_exposure_limit * iqty_pct,
            qty_step,
        ),
    )
    if psize == 0.0:
        # normal ientry
        return ientry_qty, ientry_price, "long_ientry_normal"
    elif psize < ientry_qty * 0.8:
        # partial ientry
        entry_qty = max(min_entry_qty, round_(psize - ientry_qty, qty_step))
        return entry_qty, ientry_price, "long_ientry_partial"
    else:
        wallet_exposure = qty_to_cost(psize, pprice, inverse, c_mult) / balance
        if wallet_exposure >= wallet_exposure_limit * 0.995:
            # no entry if wallet_exposure within 0.5% of limit
            return 0.0, 0.0, ""
        threshold = wallet_exposure_limit * (1 - auto_unstuck_wallet_exposure_threshold)
        if auto_unstuck_wallet_exposure_threshold != 0.0 and wallet_exposure > threshold:
            # auto unstuck mode
            entry_price = round_dn(
                min([highest_bid, pprice, ema_band_lower * (1 - auto_unstuck_ema_dist)]), price_step
            )
            entry_qty = find_entry_qty_bringing_wallet_exposure_to_target(
                balance, psize, pprice, wallet_exposure_limit, entry_price, inverse, qty_step, c_mult
            )
            return (
                (entry_qty, entry_price, "long_unstuck_rentry")
                if entry_qty > calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost)
                else (0.0, 0.0, "")
            )
        else:
            # normal reentry
            ratio = wallet_exposure / wallet_exposure_limit
            entry_price = round_dn(
                pprice
                * (
                    1
                    - rentry_pprice_dist * (1 + ratio * rentry_pprice_dist_wallet_exposure_weighting)
                ),
                price_step,
            )
            min_entry_qty = calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost)
            entry_qty = max(min_entry_qty, round_(psize * ddown_factor, qty_step))
            wallet_exposure_if_filled = calc_wallet_exposure_if_filled(
                balance, psize, pprice, entry_qty, entry_price, inverse, c_mult, qty_step
            )
            if wallet_exposure_if_filled > wallet_exposure_limit * 1.01:
                entry_qty = find_entry_qty_bringing_wallet_exposure_to_target(
                    balance,
                    psize,
                    pprice,
                    wallet_exposure_limit,
                    entry_price,
                    inverse,
                    qty_step,
                    c_mult,
                )
                if entry_qty < min_entry_qty:
                    return 0.0, 0.0, ""
            return entry_qty, entry_price, "long_rentry"


@njit
def calc_long_entries(
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
    iqty_pct,
    iprice_ema_dist,
    ddown_factor,
    rentry_pprice_dist,
    rentry_pprice_dist_wallet_exposure_weighting,
    wallet_exposure_limit,
    auto_unstuck_ema_dist,
    auto_unstuck_wallet_exposure_threshold,
):
    entries = []
    psize_ = psize
    pprice_ = pprice
    highest_bid_ = highest_bid
    while True:
        entry_qty, entry_price, entry_type = calc_long_entry(
            balance,
            psize_,
            pprice_,
            highest_bid_,
            ema_band_lower,
            inverse,
            qty_step,
            price_step,
            min_qty,
            min_cost,
            c_mult,
            iqty_pct,
            iprice_ema_dist,
            ddown_factor,
            rentry_pprice_dist,
            rentry_pprice_dist_wallet_exposure_weighting,
            wallet_exposure_limit,
            auto_unstuck_ema_dist,
            auto_unstuck_wallet_exposure_threshold,
        )
        if entry_qty == 0.0:
            break
        psize_, pprice_ = calc_new_psize_pprice(psize_, pprice_, entry_qty, entry_price, qty_step)
        highest_bid_ = min(highest_bid, entry_price)
        wallet_exposure = qty_to_cost(psize_, pprice_, inverse, c_mult) / balance
        if "unstuck" in entry_type:
            if len(entries) == 0:
                # return unstucking entry only if it's the only one
                return [(entry_qty, entry_price, entry_type, psize_, pprice_, wallet_exposure)]
        else:
            entries.append((entry_qty, entry_price, entry_type, psize_, pprice_, wallet_exposure))
    return entries


@njit
def backtest_recursive_grid(
    ticks,
    starting_balance,
    latency_simulation_ms,
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
    iqty_pct,
    iprice_ema_dist,
    wallet_exposure_limit,
    ddown_factor,
    rentry_pprice_dist,
    rentry_pprice_dist_wallet_exposure_weighting,
    min_markup,
    markup_range,
    n_close_orders,
    auto_unstuck_wallet_exposure_threshold,
    auto_unstuck_ema_dist,
):
    balance = starting_balance
    timestamps = ticks[:, 0]
    qtys = ticks[:, 1]
    prices = ticks[:, 2]

    balance = equity = balance_long = balance_short = equity_long = equity_short = starting_balance
    long_psize, long_pprice, short_psize, short_pprice = 0.0, 0.0, 0.0, 0.0

    fills = []
    stats = []

    long_entry, short_entry = (0.0, 0.0, ""), (0.0, 0.0, "")

    long_closes = [(0.0, 0.0, "")]
    short_closes = [(0.0, 0.0, "")]
    bkr_price = 0.0

    next_entry_update_ts_long = 0
    next_entry_update_ts_short = 0
    next_closes_update_ts_long = 0
    next_closes_update_ts_short = 0
    next_stats_update = 0

    prev_k = 0
    closest_bkr = 1.0

    spans_long = [ema_span_0[0], (ema_span_0[0] * ema_span_1[0]) ** 0.5, ema_span_1[0]]
    spans_long = np.array(sorted(spans_long)) * 60.0
    spans_short = [ema_span_0[1], (ema_span_0[1] * ema_span_1[1]) ** 0.5, ema_span_1[1]]
    spans_short = np.array(sorted(spans_short)) * 60.0
    assert max(spans_long) < len(prices), "ema_span_max long larger than len(prices)"
    assert max(spans_short) < len(prices), "ema_span_max short larger than len(prices)"
    spans_long = np.where(spans_long < 1.0, 1.0, spans_long)
    spans_short = np.where(spans_short < 1.0, 1.0, spans_short)
    max_span = int(round(max(max(spans_long), max(spans_short))))
    # print("prepping emas")
    emas_long = (
        calc_emas_last(prices[:max_span], spans_long) if do_long else np.zeros(len(spans_long))
    )
    emas_short = (
        calc_emas_last(prices[:max_span], spans_short) if do_short else np.zeros(len(spans_short))
    )
    alphas_long = 2.0 / (spans_long + 1.0)
    alphas__long = 1.0 - alphas_long
    alphas_short = 2.0 / (spans_short + 1.0)
    alphas__short = 1.0 - alphas_short

    long_wallet_exposure = 0.0
    short_wallet_exposure = 0.0
    long_wallet_exposure_auto_unstuck_threshold = (
        (wallet_exposure_limit[0] * (1 - auto_unstuck_wallet_exposure_threshold[0]) * 0.99)
        if auto_unstuck_wallet_exposure_threshold[0] != 0.0
        else wallet_exposure_limit[0] * 10
    )
    short_wallet_exposure_auto_unstuck_threshold = (
        (wallet_exposure_limit[1] * (1 - auto_unstuck_wallet_exposure_threshold[1]) * 0.99)
        if auto_unstuck_wallet_exposure_threshold[1] != 0.0
        else wallet_exposure_limit[1] * 10
    )
    # print("starting iter")
    for k in range(max_span, len(prices)):
        if do_long:
            emas_long = calc_ema(alphas_long, alphas__long, emas_long, prices[k])
        if do_short:
            emas_short = calc_ema(alphas_short, alphas__short, emas_short, prices[k])

        bkr_diff = calc_diff(bkr_price, prices[k])
        closest_bkr = min(closest_bkr, bkr_diff)
        if timestamps[k] >= next_stats_update:
            equity = balance + calc_upnl(
                long_psize,
                long_pprice,
                short_psize,
                short_pprice,
                prices[k],
                inverse,
                c_mult,
            )
            equity_long = balance_long + calc_long_pnl(
                long_pprice, prices[k], long_psize, inverse, c_mult
            )
            equity_short = balance_short + calc_short_pnl(
                short_pprice, prices[k], short_psize, inverse, c_mult
            )
            stats.append(
                (
                    timestamps[k],
                    balance,
                    equity,
                    bkr_price,
                    long_psize,
                    long_pprice,
                    short_psize,
                    short_pprice,
                    prices[k],
                    closest_bkr,
                    balance_long,
                    balance_short,
                    equity_long,
                    equity_short,
                )
            )
            if equity / starting_balance < 0.2:
                # break early when equity is less than 20% of starting balance
                return fills, stats
            next_stats_update = timestamps[k] + 60 * 1000

        if timestamps[k] >= next_entry_update_ts_long:
            long_entry = (
                calc_long_entry(
                    balance,
                    long_psize,
                    long_pprice,
                    prices[k - 1],
                    min(emas_long),
                    inverse,
                    qty_step,
                    price_step,
                    min_qty,
                    min_cost,
                    c_mult,
                    iqty_pct[0],
                    iprice_ema_dist[0],
                    ddown_factor[0],
                    rentry_pprice_dist[0],
                    rentry_pprice_dist_wallet_exposure_weighting[0],
                    wallet_exposure_limit[0],
                    auto_unstuck_ema_dist[0],
                    auto_unstuck_wallet_exposure_threshold[0],
                )
                if do_long
                else (0.0, 0.0, "")
            )
            next_entry_update_ts_long = timestamps[k] + 1000 * 60 * 5  # five mins delay

        if timestamps[k] >= next_closes_update_ts_long:
            long_closes = (
                calc_long_close_grid(
                    balance,
                    long_psize,
                    long_pprice,
                    prices[k - 1],
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
                if do_long
                else [(0.0, 0.0, "")]
            )
            next_closes_update_ts_long = timestamps[k] + 1000 * 60 * 5  # five mins delay

        if closest_bkr < 0.06:
            # consider bankruptcy within 6% as liquidation
            if long_psize != 0.0:
                fee_paid = -qty_to_cost(long_psize, long_pprice, inverse, c_mult) * maker_fee
                pnl = calc_long_pnl(long_pprice, prices[k], -long_psize, inverse, c_mult)
                balance = 0.0
                equity = 0.0
                long_psize, long_pprice = 0.0, 0.0
                fills.append(
                    (
                        k,
                        timestamps[k],
                        pnl,
                        fee_paid,
                        balance,
                        equity,
                        -long_psize,
                        prices[k],
                        0.0,
                        0.0,
                        "long_bankruptcy",
                    )
                )
            return fills, stats

        if long_entry[0] != 0.0 and prices[k] < long_entry[1]:
            next_entry_update_ts_long = min(
                next_entry_update_ts_long, timestamps[k] + latency_simulation_ms
            )
            next_closes_update_ts_long = min(
                next_closes_update_ts_long, timestamps[k] + latency_simulation_ms
            )
            long_psize, long_pprice = calc_new_psize_pprice(
                long_psize,
                long_pprice,
                long_entry[0],
                long_entry[1],
                qty_step,
            )
            fee_paid = -qty_to_cost(long_entry[0], long_entry[1], inverse, c_mult) * maker_fee
            balance += fee_paid
            balance_long += fee_paid
            equity = calc_equity(
                balance,
                long_psize,
                long_pprice,
                short_psize,
                short_pprice,
                prices[k],
                inverse,
                c_mult,
            )
            fills.append(
                (
                    k,
                    timestamps[k],
                    0.0,
                    fee_paid,
                    balance,
                    equity,
                    long_entry[0],
                    long_entry[1],
                    long_psize,
                    long_pprice,
                    long_entry[2],
                )
            )
            bkr_price = calc_bankruptcy_price(
                balance,
                long_psize,
                long_pprice,
                short_psize,
                short_pprice,
                inverse,
                c_mult,
            )
            long_wallet_exposure = qty_to_cost(long_psize, long_pprice, inverse, c_mult) / balance
            # print("long entry fill", long_entry, long_psize, long_pprice, balance)

        while (
            long_psize > 0.0
            and long_closes
            and long_closes[0][0] < 0.0
            and prices[k] > long_closes[0][1]
        ):
            next_entry_update_ts_long = min(
                next_entry_update_ts_long, timestamps[k] + latency_simulation_ms
            )
            next_closes_update_ts_long = min(
                next_closes_update_ts_long, timestamps[k] + latency_simulation_ms
            )
            long_close_qty = long_closes[0][0]
            new_long_psize = round_(long_psize + long_close_qty, qty_step)
            if new_long_psize < 0.0:
                print("warning: long close qty greater than long psize")
                print("long_psize", long_psize)
                print("long_pprice", long_pprice)
                print("long_closes[0]", long_closes[0])
                long_close_qty = -long_psize
                new_long_psize, long_pprice = 0.0, 0.0
            long_psize = new_long_psize
            fee_paid = -qty_to_cost(long_close_qty, long_closes[0][1], inverse, c_mult) * maker_fee
            pnl = calc_long_pnl(long_pprice, long_closes[0][1], long_close_qty, inverse, c_mult)
            balance += fee_paid + pnl
            balance_long += fee_paid + pnl
            equity = calc_equity(
                balance,
                long_psize,
                long_pprice,
                short_psize,
                short_pprice,
                prices[k],
                inverse,
                c_mult,
            )
            fills.append(
                (
                    k,
                    timestamps[k],
                    pnl,
                    fee_paid,
                    balance,
                    equity,
                    long_close_qty,
                    long_closes[0][1],
                    long_psize,
                    long_pprice,
                    long_closes[0][2],
                )
            )
            # print("long close fill", long_closes[0], long_psize, long_pprice, balance)
            long_closes = long_closes[1:]
            bkr_price = calc_bankruptcy_price(
                balance,
                long_psize,
                long_pprice,
                short_psize,
                short_pprice,
                inverse,
                c_mult,
            )
            long_wallet_exposure = qty_to_cost(long_psize, long_pprice, inverse, c_mult) / balance
        if do_long:
            if long_psize == 0.0:
                next_entry_update_ts_long = min(
                    next_entry_update_ts_long,
                    timestamps[k] + latency_simulation_ms,
                )
            else:
                if prices[k] > long_pprice:
                    next_closes_update_ts_long = min(
                        next_closes_update_ts_long,
                        timestamps[k] + latency_simulation_ms + 2500,
                    )
                elif long_wallet_exposure >= long_wallet_exposure_auto_unstuck_threshold:
                    next_closes_update_ts_long = min(
                        next_closes_update_ts_long,
                        timestamps[k] + latency_simulation_ms + 15000,
                    )
                    next_entry_update_ts_long = min(
                        next_entry_update_ts_long,
                        timestamps[k] + latency_simulation_ms + 15000,
                    )
    return fills, stats
