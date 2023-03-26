import os
import numpy as np

from njit_funcs import (
    round_dn,
    round_up,
    round_,
    calc_min_entry_qty,
    cost_to_qty,
    qty_to_cost,
    calc_new_psize_pprice,
    calc_bankruptcy_price,
    calc_ema,
    calc_diff,
    calc_pnl_long,
    calc_pnl_short,
    calc_upnl,
    calc_equity,
    calc_emas_last,
    calc_wallet_exposure_if_filled,
    find_entry_qty_bringing_wallet_exposure_to_target,
    calc_close_grid_long,
    calc_close_grid_short,
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
def calc_recursive_entry_long(
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
    initial_qty_pct,
    initial_eprice_ema_dist,
    ddown_factor,
    rentry_pprice_dist,
    rentry_pprice_dist_wallet_exposure_weighting,
    wallet_exposure_limit,
    auto_unstuck_ema_dist,
    auto_unstuck_wallet_exposure_threshold,
):
    if wallet_exposure_limit == 0.0:
        return 0.0, 0.0, ""
    ientry_price = max(
        price_step,
        min(highest_bid, round_dn(ema_band_lower * (1 - initial_eprice_ema_dist), price_step)),
    )
    if ientry_price == price_step:
        return 0.0, ientry_price, ""
    min_entry_qty = calc_min_entry_qty(ientry_price, inverse, qty_step, min_qty, min_cost)
    ientry_qty = max(
        min_entry_qty,
        round_(
            cost_to_qty(balance, ientry_price, inverse, c_mult)
            * wallet_exposure_limit
            * initial_qty_pct,
            qty_step,
        ),
    )
    if psize == 0.0:
        # normal ientry
        return ientry_qty, ientry_price, "long_ientry_normal"
    elif psize < ientry_qty * 0.8:
        # partial ientry
        entry_qty = max(min_entry_qty, round_(ientry_qty - psize, qty_step))
        return entry_qty, ientry_price, "long_ientry_partial"
    else:
        wallet_exposure = qty_to_cost(psize, pprice, inverse, c_mult) / balance
        if wallet_exposure >= wallet_exposure_limit * 1.001:
            # no entry if wallet_exposure within 0.1% of limit
            return 0.0, 0.0, ""
        threshold = wallet_exposure_limit * (1 - auto_unstuck_wallet_exposure_threshold)
        if auto_unstuck_wallet_exposure_threshold != 0.0 and wallet_exposure > threshold * 0.99:
            # auto unstuck mode
            entry_price = round_dn(
                min([highest_bid, pprice, ema_band_lower * (1 - auto_unstuck_ema_dist)]), price_step
            )
            entry_qty = find_entry_qty_bringing_wallet_exposure_to_target(
                balance, psize, pprice, wallet_exposure_limit, entry_price, inverse, qty_step, c_mult
            )
            min_entry_qty = calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost)
            return (max(entry_qty, min_entry_qty), entry_price, "long_unstuck_entry")
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
            if entry_price <= price_step:
                return 0.0, price_step, ""
            entry_price = min(highest_bid, entry_price)
            min_entry_qty = calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost)
            entry_qty = max(min_entry_qty, round_(psize * ddown_factor, qty_step))
            wallet_exposure_if_filled = calc_wallet_exposure_if_filled(
                balance, psize, pprice, entry_qty, entry_price, inverse, c_mult, qty_step
            )
            adjust = False
            if wallet_exposure_if_filled > wallet_exposure_limit * 1.01:
                adjust = True
            else:
                # preview next reentry
                new_psize, new_pprice = calc_new_psize_pprice(
                    psize, pprice, entry_qty, entry_price, qty_step
                )
                new_wallet_exposure = qty_to_cost(new_psize, new_pprice, inverse, c_mult) / balance
                new_ratio = new_wallet_exposure / wallet_exposure_limit
                new_entry_price = round_dn(
                    new_pprice
                    * (
                        1
                        - rentry_pprice_dist
                        * (1 + new_ratio * rentry_pprice_dist_wallet_exposure_weighting)
                    ),
                    price_step,
                )
                new_entry_qty = max(min_entry_qty, round_(new_psize * ddown_factor, qty_step))
                wallet_exposure_if_next_filled = calc_wallet_exposure_if_filled(
                    balance,
                    new_psize,
                    new_pprice,
                    new_entry_qty,
                    new_entry_price,
                    inverse,
                    c_mult,
                    qty_step,
                )
                if wallet_exposure_if_next_filled > wallet_exposure_limit * 1.2:
                    adjust = True
            if adjust:
                # increase qty if next reentry is too small
                # decrease qty if current reentry is too big
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
                entry_qty = max(entry_qty, min_entry_qty)
            return entry_qty, entry_price, "long_rentry"


@njit
def calc_recursive_entry_short(
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
    initial_qty_pct,
    initial_eprice_ema_dist,
    ddown_factor,
    rentry_pprice_dist,
    rentry_pprice_dist_wallet_exposure_weighting,
    wallet_exposure_limit,
    auto_unstuck_ema_dist,
    auto_unstuck_wallet_exposure_threshold,
):
    if wallet_exposure_limit == 0.0:
        return 0.0, 0.0, ""
    abs_psize = abs(psize)
    ientry_price = max(
        lowest_ask, round_up(ema_band_upper * (1 + initial_eprice_ema_dist), price_step)
    )
    min_entry_qty = calc_min_entry_qty(ientry_price, inverse, qty_step, min_qty, min_cost)
    ientry_qty = max(
        min_entry_qty,
        round_(
            cost_to_qty(balance, ientry_price, inverse, c_mult)
            * wallet_exposure_limit
            * initial_qty_pct,
            qty_step,
        ),
    )
    if abs_psize == 0.0:
        # normal ientry
        return -ientry_qty, ientry_price, "short_ientry_normal"
    elif abs_psize < ientry_qty * 0.8:
        # partial ientry
        entry_qty = max(min_entry_qty, round_(ientry_qty - abs_psize, qty_step))
        return -entry_qty, ientry_price, "short_ientry_partial"
    else:
        wallet_exposure = qty_to_cost(abs_psize, pprice, inverse, c_mult) / balance
        if wallet_exposure >= wallet_exposure_limit * 1.001:
            # no entry if wallet_exposure within 0.1% of limit
            return 0.0, 0.0, ""
        threshold = wallet_exposure_limit * (1 - auto_unstuck_wallet_exposure_threshold)
        if auto_unstuck_wallet_exposure_threshold != 0.0 and wallet_exposure > threshold * 0.99:
            # auto unstuck mode
            entry_price = round_up(
                max([lowest_ask, pprice, ema_band_upper * (1 + auto_unstuck_ema_dist)]), price_step
            )
            entry_qty = find_entry_qty_bringing_wallet_exposure_to_target(
                balance,
                abs_psize,
                pprice,
                wallet_exposure_limit,
                entry_price,
                inverse,
                qty_step,
                c_mult,
            )
            min_entry_qty = calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost)
            return (-max(entry_qty, min_entry_qty), entry_price, "short_unstuck_entry")
        else:
            # normal reentry
            ratio = wallet_exposure / wallet_exposure_limit
            entry_price = round_up(
                pprice
                * (
                    1
                    + rentry_pprice_dist * (1 + ratio * rentry_pprice_dist_wallet_exposure_weighting)
                ),
                price_step,
            )
            entry_price = max(entry_price, lowest_ask)
            min_entry_qty = calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost)
            entry_qty = max(min_entry_qty, round_(abs_psize * ddown_factor, qty_step))
            wallet_exposure_if_filled = calc_wallet_exposure_if_filled(
                balance, abs_psize, pprice, entry_qty, entry_price, inverse, c_mult, qty_step
            )
            adjust = False
            if wallet_exposure_if_filled > wallet_exposure_limit * 1.01:
                adjust = True
            else:
                # preview next reentry
                new_psize, new_pprice = calc_new_psize_pprice(
                    abs_psize, pprice, entry_qty, entry_price, qty_step
                )
                new_wallet_exposure = qty_to_cost(new_psize, new_pprice, inverse, c_mult) / balance
                new_ratio = new_wallet_exposure / wallet_exposure_limit
                new_entry_price = round_up(
                    new_pprice
                    * (
                        1
                        + rentry_pprice_dist
                        * (1 + new_ratio * rentry_pprice_dist_wallet_exposure_weighting)
                    ),
                    price_step,
                )
                new_entry_qty = max(min_entry_qty, round_(new_psize * ddown_factor, qty_step))
                wallet_exposure_if_next_filled = calc_wallet_exposure_if_filled(
                    balance,
                    new_psize,
                    new_pprice,
                    new_entry_qty,
                    new_entry_price,
                    inverse,
                    c_mult,
                    qty_step,
                )
                if wallet_exposure_if_next_filled > wallet_exposure_limit * 1.2:
                    adjust = True
            if adjust:
                # increase qty if next reentry is too small
                # or decrease qty if current reentry is too big
                entry_qty = find_entry_qty_bringing_wallet_exposure_to_target(
                    balance,
                    abs_psize,
                    pprice,
                    wallet_exposure_limit,
                    entry_price,
                    inverse,
                    qty_step,
                    c_mult,
                )
                entry_qty = max(entry_qty, min_entry_qty)
            return -entry_qty, entry_price, "short_rentry"


@njit
def calc_recursive_entries_long(
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
    initial_qty_pct,
    initial_eprice_ema_dist,
    ddown_factor,
    rentry_pprice_dist,
    rentry_pprice_dist_wallet_exposure_weighting,
    wallet_exposure_limit,
    auto_unstuck_ema_dist,
    auto_unstuck_wallet_exposure_threshold,
    whole_grid=False,
):
    entries = []
    psize_ = psize
    pprice_ = pprice
    highest_bid_ = highest_bid
    i = 0
    infinite_loop_break = 30
    while True:
        i += 1
        if i > infinite_loop_break:
            break
        entry_qty, entry_price, entry_type = calc_recursive_entry_long(
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
            initial_qty_pct,
            initial_eprice_ema_dist,
            ddown_factor,
            rentry_pprice_dist,
            rentry_pprice_dist_wallet_exposure_weighting,
            wallet_exposure_limit,
            auto_unstuck_ema_dist,
            auto_unstuck_wallet_exposure_threshold,
        )
        if entry_qty == 0.0:
            break
        if entries and entry_price == entries[-1][1]:
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
        if not whole_grid and psize == 0.0:
            break
    return entries


@njit
def calc_recursive_entries_short(
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
    initial_qty_pct,
    initial_eprice_ema_dist,
    ddown_factor,
    rentry_pprice_dist,
    rentry_pprice_dist_wallet_exposure_weighting,
    wallet_exposure_limit,
    auto_unstuck_ema_dist,
    auto_unstuck_wallet_exposure_threshold,
    whole_grid=False,
):
    entries = []
    psize_ = psize
    pprice_ = pprice
    lowest_ask_ = lowest_ask
    i = 0
    infinite_loop_break = 30
    while True:
        i += 1
        if i > infinite_loop_break:
            break
        entry_qty, entry_price, entry_type = calc_recursive_entry_short(
            balance,
            psize_,
            pprice_,
            lowest_ask_,
            ema_band_upper,
            inverse,
            qty_step,
            price_step,
            min_qty,
            min_cost,
            c_mult,
            initial_qty_pct,
            initial_eprice_ema_dist,
            ddown_factor,
            rentry_pprice_dist,
            rentry_pprice_dist_wallet_exposure_weighting,
            wallet_exposure_limit,
            auto_unstuck_ema_dist,
            auto_unstuck_wallet_exposure_threshold,
        )
        if entry_qty == 0.0:
            break
        if entries and entry_price == entries[-1][1]:
            break
        psize_, pprice_ = calc_new_psize_pprice(psize_, pprice_, entry_qty, entry_price, qty_step)
        lowest_ask_ = max(lowest_ask, entry_price)
        wallet_exposure = qty_to_cost(psize_, pprice_, inverse, c_mult) / balance
        if "unstuck" in entry_type:
            if len(entries) == 0:
                # return unstucking entry only if it's the only one
                return [(entry_qty, entry_price, entry_type, psize_, pprice_, wallet_exposure)]
        else:
            entries.append((entry_qty, entry_price, entry_type, psize_, pprice_, wallet_exposure))
        if not whole_grid and psize == 0.0:
            break
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
    backwards_tp,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    ema_span_0,
    ema_span_1,
    initial_qty_pct,
    initial_eprice_ema_dist,
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

    entry_long, entry_short = (0.0, 0.0, ""), (0.0, 0.0, "")
    closes_long, closes_short = [(0.0, 0.0, "")], [(0.0, 0.0, "")]
    bkr_price_long = bkr_price_short = 0.0

    next_entry_update_ts_long = 0
    next_entry_update_ts_short = 0
    next_close_grid_update_ts_long = 0
    next_close_grid_update_ts_short = 0
    next_stats_update = 0

    closest_bkr_long = closest_bkr_short = 1.0

    spans_multiplier = 60 / ((timestamps[1] - timestamps[0]) / 1000)

    spans_long = [ema_span_0[0], (ema_span_0[0] * ema_span_1[0]) ** 0.5, ema_span_1[0]]
    spans_long = np.array(sorted(spans_long)) * spans_multiplier if do_long else np.ones(3)
    spans_short = [ema_span_0[1], (ema_span_0[1] * ema_span_1[1]) ** 0.5, ema_span_1[1]]
    spans_short = np.array(sorted(spans_short)) * spans_multiplier if do_short else np.ones(3)
    assert max(spans_long) < len(ticks), "ema_span_1 long larger than len(prices)"
    assert max(spans_short) < len(ticks), "ema_span_1 short larger than len(prices)"
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
    for k in range(1, len(ticks)):
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
                        return fills_long, fills_short, stats

                # check if long entry order should be updated
                if timestamps[k] >= next_entry_update_ts_long:
                    entry_long = calc_recursive_entry_long(
                        balance_long,
                        psize_long,
                        pprice_long,
                        closes[k - 1],
                        min(emas_long),
                        inverse,
                        qty_step,
                        price_step,
                        min_qty,
                        min_cost,
                        c_mult,
                        initial_qty_pct[0],
                        initial_eprice_ema_dist[0],
                        ddown_factor[0],
                        rentry_pprice_dist[0],
                        rentry_pprice_dist_wallet_exposure_weighting[0],
                        wallet_exposure_limit[0],
                        auto_unstuck_ema_dist[0],
                        auto_unstuck_wallet_exposure_threshold[0],
                    )
                    next_entry_update_ts_long = timestamps[k] + 1000 * 60 * 5  # five mins delay

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
                    next_close_grid_update_ts_long = timestamps[k] + 1000 * 60 * 5  # five mins delay

                # check if long entry filled
                while entry_long[0] != 0.0 and lows[k] < entry_long[1]:
                    next_entry_update_ts_long = min(
                        next_entry_update_ts_long, timestamps[k] + latency_simulation_ms
                    )
                    next_close_grid_update_ts_long = min(
                        next_close_grid_update_ts_long, timestamps[k] + latency_simulation_ms
                    )
                    psize_long, pprice_long = calc_new_psize_pprice(
                        psize_long,
                        pprice_long,
                        entry_long[0],
                        entry_long[1],
                        qty_step,
                    )
                    fee_paid = -qty_to_cost(entry_long[0], entry_long[1], inverse, c_mult) * maker_fee
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
                            entry_long[0],
                            entry_long[1],
                            psize_long,
                            pprice_long,
                            entry_long[2],
                        )
                    )
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
                    entry_long = calc_recursive_entry_long(
                        balance_long,
                        psize_long,
                        pprice_long,
                        closes[k - 1],
                        min(emas_long),
                        inverse,
                        qty_step,
                        price_step,
                        min_qty,
                        min_cost,
                        c_mult,
                        initial_qty_pct[0],
                        initial_eprice_ema_dist[0],
                        ddown_factor[0],
                        rentry_pprice_dist[0],
                        rentry_pprice_dist_wallet_exposure_weighting[0],
                        wallet_exposure_limit[0],
                        auto_unstuck_ema_dist[0],
                        auto_unstuck_wallet_exposure_threshold[0],
                    )
                    if entry_long[2] == "long_unstuck_entry":
                        break

                # check if long closes filled
                while (
                    psize_long > 0.0
                    and closes_long
                    and closes_long[0][0] < 0.0
                    and highs[k] > closes_long[0][1]
                ):
                    next_entry_update_ts_long = min(
                        next_entry_update_ts_long, timestamps[k] + latency_simulation_ms
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
                    next_entry_update_ts_long = min(
                        next_entry_update_ts_long,
                        timestamps[k] + latency_simulation_ms,
                    )
                else:
                    if closes[k] > pprice_long:
                        # update closes after 2.5 secs
                        next_close_grid_update_ts_long = min(
                            next_close_grid_update_ts_long,
                            timestamps[k] + latency_simulation_ms + 2500,
                        )
                    elif long_wallet_exposure >= long_wallet_exposure_auto_unstuck_threshold:
                        # update both entry and closes after 15 secs
                        next_close_grid_update_ts_long = min(
                            next_close_grid_update_ts_long,
                            timestamps[k] + latency_simulation_ms + 15000,
                        )
                        next_entry_update_ts_long = min(
                            next_entry_update_ts_long,
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
                        return fills_long, fills_short, stats

                # check if entry order should be updated
                if timestamps[k] >= next_entry_update_ts_short:
                    entry_short = calc_recursive_entry_short(
                        balance_short,
                        psize_short,
                        pprice_short,
                        closes[k - 1],
                        max(emas_short),
                        inverse,
                        qty_step,
                        price_step,
                        min_qty,
                        min_cost,
                        c_mult,
                        initial_qty_pct[1],
                        initial_eprice_ema_dist[1],
                        ddown_factor[1],
                        rentry_pprice_dist[1],
                        rentry_pprice_dist_wallet_exposure_weighting[1],
                        wallet_exposure_limit[1],
                        auto_unstuck_ema_dist[1],
                        auto_unstuck_wallet_exposure_threshold[1],
                    )
                    next_entry_update_ts_short = timestamps[k] + 1000 * 60 * 5  # five mins delay
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

                # check if short entry filled
                while entry_short[0] != 0.0 and highs[k] > entry_short[1]:
                    next_entry_update_ts_short = min(
                        next_entry_update_ts_short, timestamps[k] + latency_simulation_ms
                    )
                    next_close_grid_update_ts_short = min(
                        next_close_grid_update_ts_short, timestamps[k] + latency_simulation_ms
                    )
                    psize_short, pprice_short = calc_new_psize_pprice(
                        psize_short,
                        pprice_short,
                        entry_short[0],
                        entry_short[1],
                        qty_step,
                    )
                    fee_paid = (
                        -qty_to_cost(entry_short[0], entry_short[1], inverse, c_mult) * maker_fee
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
                            entry_short[0],
                            entry_short[1],
                            psize_short,
                            pprice_short,
                            entry_short[2],
                        )
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
                    short_wallet_exposure = (
                        qty_to_cost(psize_short, pprice_short, inverse, c_mult) / balance_short
                    )
                    entry_short = calc_recursive_entry_short(
                        balance_short,
                        psize_short,
                        pprice_short,
                        closes[k - 1],
                        max(emas_short),
                        inverse,
                        qty_step,
                        price_step,
                        min_qty,
                        min_cost,
                        c_mult,
                        initial_qty_pct[1],
                        initial_eprice_ema_dist[1],
                        ddown_factor[1],
                        rentry_pprice_dist[1],
                        rentry_pprice_dist_wallet_exposure_weighting[1],
                        wallet_exposure_limit[1],
                        auto_unstuck_ema_dist[1],
                        auto_unstuck_wallet_exposure_threshold[1],
                    )
                    if entry_short[2] == "short_unstuck_entry":
                        break
                # check if short closes filled
                while (
                    psize_short < 0.0
                    and closes_short
                    and closes_short[0][0] > 0.0
                    and lows[k] < closes_short[0][1]
                ):
                    next_entry_update_ts_short = min(
                        next_entry_update_ts_short, timestamps[k] + latency_simulation_ms
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
                    # update entry order now
                    next_entry_update_ts_short = min(
                        next_entry_update_ts_short,
                        timestamps[k] + latency_simulation_ms,
                    )
                else:
                    if closes[k] > pprice_short:
                        # update closes after 2.5 secs
                        next_close_grid_update_ts_short = min(
                            next_close_grid_update_ts_short,
                            timestamps[k] + latency_simulation_ms + 2500,
                        )
                    elif short_wallet_exposure >= short_wallet_exposure_auto_unstuck_threshold:
                        # update both entry and closes after 15 secs
                        next_close_grid_update_ts_short = min(
                            next_close_grid_update_ts_short,
                            timestamps[k] + latency_simulation_ms + 15000,
                        )
                        next_entry_update_ts_short = min(
                            next_entry_update_ts_short,
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
            next_stats_update = timestamps[k] + 60 * 60 * 1000

    return fills_long, fills_short, stats
