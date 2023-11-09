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

from njit_funcs import (
    calc_ema,
    calc_new_psize_pprice,
    qty_to_cost,
    cost_to_qty,
    calc_close_grid_long,
    calc_close_grid_short,
    calc_pnl_long,
    calc_pnl_short,
    round_,
    calc_min_entry_qty,
)
from njit_funcs_recursive_grid import calc_recursive_entry_long


def live_config_dict_to_list_recursive_grid(live_config: dict) -> list:
    keys = [
        "auto_unstuck_delay_minutes",
        "auto_unstuck_ema_dist",
        "auto_unstuck_qty_pct",
        "auto_unstuck_wallet_exposure_threshold",
        "backwards_tp",
        "ddown_factor",
        "ema_span_0",
        "ema_span_1",
        "enabled",
        "initial_eprice_ema_dist",
        "initial_qty_pct",
        "markup_range",
        "min_markup",
        "n_close_orders",
        "rentry_pprice_dist",
        "rentry_pprice_dist_wallet_exposure_weighting",
        "wallet_exposure_limit",
    ]
    return [(float(live_config["long"][key]), float(live_config["short"][key])) for key in keys]


@njit
def calc_pnl_sum(poss_long, poss_short, market_prices, c_mults):
    pnl_sum = 0.0
    for i in range(len(poss_long)):
        pnl_sum += calc_pnl_long(
            poss_long[i][1], market_prices[i], poss_long[i][0], False, c_mults[i]
        )
    for i in range(len(poss_short)):
        pnl_sum += calc_pnl_short(
            poss_short[i][1], market_prices[i], poss_short[i][0], False, c_mults[i]
        )
    return pnl_sum


@njit
def get_open_orders_long(
    close_price,
    balance,
    pos_long,
    emas,
    inverse,
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    cfgl,
):
    # returns entries and closes
    entries = calc_recursive_entry_long(
        balance,
        pos_long[0],
        pos_long[1],
        close_price,
        min(emas),
        inverse,
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
        cfgl[10],
        cfgl[9],
        cfgl[5],
        cfgl[14],
        cfgl[15],
        cfgl[16],
        cfgl[1],
        cfgl[3],
        cfgl[0] or cfgl[2],
    )
    closes = calc_close_grid_long(
        cfgl[4],  # backwards_tp
        balance,
        pos_long[0],
        pos_long[1],
        close_price,  # close price
        max(emas),
        0,  # utc_now_ms: timed AU is disabled
        0,  # prev_AU_fill_ts_close: timed AU is disabled
        inverse,
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
        cfgl[16],  # wallet_exposure_limit
        cfgl[12],  # min_markup,
        cfgl[11],  # markup_range,
        cfgl[13],  # n_close_orders,
        cfgl[3],  # auto_unstuck_wallet_exposure_threshold,
        cfgl[1],  # auto_unstuck_ema_dist,
        cfgl[0],  # auto_unstuck_delay_minutes,
        cfgl[2],  # auto_unstuck_qty_pct,
    )
    return entries, closes


@njit
def get_fills_long(
    k,
    poss_long,
    poss_short,
    idx,
    symbol,
    balance,
    entry,
    closes,
    hlc,
    inverse,
    qty_step,
    c_mults: np.ndarray,
    maker_fee,
):
    """
    returns fills: [tuple], new_pos: (float, float), new_balance: float
    """
    fills = []
    new_pos = (poss_long[idx][0], poss_long[idx][1])
    new_balance = balance
    if entry[0] > 0.0 and hlc[idx][1] < entry[1]:
        new_pos = calc_new_psize_pprice(
            new_pos[0],
            new_pos[1],
            entry[0],
            entry[1],
            qty_step,
        )
        fee_paid = -qty_to_cost(entry[0], entry[1], inverse, c_mults[idx]) * maker_fee
        new_balance = max(new_balance * 1e-6, new_balance + fee_paid)
        new_equity = new_balance + calc_pnl_sum(
            poss_long, poss_short, hlc[:, 2], c_mults
        )  # compute total equity
        fills.append(
            (
                k,  # index
                symbol,
                0.0,  # realized pnl
                fee_paid,
                new_balance,
                new_equity,  # equity
                entry[0],  # fill qty
                entry[1],  # fill price
                new_pos[0],  # psize after fill
                new_pos[1],  # pprice after fill
                entry[2],  # fill type
            )
        )
    for close in closes:
        if new_pos[0] <= 0.0 or close[0] == 0.0 or close[1] >= hlc[idx][0]:
            break
        # long close fill
        new_pos_ = (round_(new_pos[0] + close[0], qty_step), new_pos[1])
        if new_pos_[0] < 0.0:
            print("warning: long close qty greater than long psize")
            print("symbol", symbol)
            print("pos long", new_pos)
            print("closes order", close)
            close = (-new_pos[0], close[1], close[2])
            new_pos_ = (0.0, 0.0)
        elif new_pos_[0] == 0.0:
            new_pos_ = (0.0, 0.0)
        fee_paid = -qty_to_cost(close[0], close[1], inverse, c_mults[idx]) * maker_fee
        pnl = calc_pnl_long(new_pos[1], close[1], close[0], inverse, c_mults[idx])
        new_pos = new_pos_
        new_balance = max(new_balance * 1e-6, new_balance + fee_paid + pnl)
        new_equity = new_balance + calc_pnl_sum(
            poss_long, poss_short, hlc[:, 2], c_mults
        )  # compute total equity
        fills.append(
            (
                k,  # index
                symbol,
                pnl,  # realized pnl
                fee_paid,
                new_balance,  # post fill
                new_equity,  # post fill
                close[0],  # fill qty
                close[1],  # fill price
                new_pos[0],  # psize after fill
                new_pos[1],  # pprice after fill
                close[2],  # fill type
            )
        )

    return fills, new_pos, new_balance, new_equity


@njit
def calc_AU_allowance(pnls, balance, loss_allowance_pct=0.01, drop_since_peak_abs=-1.0):
    """
    allow up to 1% drop from balance peak for auto unstuck
    """
    if drop_since_peak_abs == -1.0:
        pnl_cumsum = pnls.cumsum()
        drop_since_peak_abs = pnl_cumsum.max() - pnl_cumsum[-1]
    balance_peak = balance + drop_since_peak_abs
    drop_since_peak_pct = balance / balance_peak - 1
    AU_allowance = max(0.0, balance_peak * (loss_allowance_pct + drop_since_peak_pct))
    return AU_allowance


@njit
def backtest_multisymbol_recursive_grid(
    hlcs,
    starting_balance,
    maker_fee,
    do_longs,
    do_shorts,
    c_mults,
    symbols,
    qty_steps,
    price_steps,
    min_costs,
    min_qtys,
    live_configs,
):
    """
    multi symbol backtest
    all symbols share same wallet

    interval is 1m
    hlcs: [[[sym1_high_0, sym1_low_0, sym1_close_0],
            [sym1_high_1, sym1_low_1, sym1_close_1],
            ...],
           [[sym1_high_0, sym1_low_0, sym1_close_0],
            [sym1_high_1, sym1_low_1, sym1_close_1],
            ...],
           ...
           ]
    # static values
    do_longs: (True, True, ...)
    do_shorts: (True, True, ...)
    c_mults: (float, float, ...)
    symbols: (str, str, ...)
    qty_steps: (float, float, ...)
    price_steps: (float, float, ...)
    min_costs: (float, float, ...)
    min_qtys: (float, float, ...)

    # each symbol has its own config
    # no dicts, structs, classes or named tubles allowed with numba...
    # a config is a list of float values
    # indices:

    0  auto_unstuck_delay_minutes
    1  auto_unstuck_ema_dist
    2  auto_unstuck_qty_pct
    3  auto_unstuck_wallet_exposure_threshold
    4  backwards_tp
    5  ddown_factor
    6  ema_span_0
    7  ema_span_1
    8  enabled
    9  initial_eprice_ema_dist
    10 initial_qty_pct
    11 markup_range
    12 min_markup
    13 n_close_orders
    14 rentry_pprice_dist
    15 rentry_pprice_dist_wallet_exposure_weighting
    16 wallet_exposure_limit

    live_configs: [((float, float, ...), (float, float, ...)), ((float, float, ...), (float, float, ...))]
    [(long, short), (long, short), ...]
    """

    inverse = False

    idxs = np.arange(len(symbols))

    ll = [[z[0] for z in x] for x in live_configs]  # live configs long
    ls = [[z[1] for z in x] for x in live_configs]  # live configs short
    # disable auto unstuck
    ll = [[0.0] * 4 + x[4:] for x in ll]
    ls = [[0.0] * 4 + x[4:] for x in ls]

    balance = starting_balance
    poss_long = [
        (0.0, 0.0) for _ in range(len(symbols))
    ]  # [psize: float, pprice: float, wallet_exposure: float]
    poss_short = [
        (0.0, 0.0) for _ in range(len(symbols))
    ]  # [psize: float, pprice: float, wallet_exposure: float]
    fills = []
    stats = [
        (
            0,
            poss_long.copy(),
            poss_short.copy(),
            hlcs[:, 0, 2],
            balance,
            balance,
        )
    ]
    entries_long = [(0.0, 0.0, "") for _ in idxs]  # (qty: float, price: float, type: str)
    entries_short = [(0.0, 0.0, "") for _ in idxs]
    closes_long = [[(0.0, 0.0, "")] for _ in idxs]  # [(qty: float, price: float, type: str), (), ...]
    closes_short = [
        [(0.0, 0.0, "")] for _ in idxs
    ]  # [(qty: float, price: float, type: str), (), ...]

    ema_spans_long = [np.array(sorted((x[6], (x[6] * x[7]) ** 0.5, x[7]))) for x in ll]
    ema_spans_short = [np.array(sorted((x[6], (x[6] * x[7]) ** 0.5, x[7]))) for x in ll]
    ema_spans_long = [np.where(x < 1.0, 1.0, x) for x in ema_spans_long]
    ema_spans_short = [np.where(x < 1.0, 1.0, x) for x in ema_spans_short]

    # find first non zero hlcs
    first_non_zero_idxs = [0 for _ in idxs]
    for i in idxs:
        for k in range(len(hlcs[i])):
            if hlcs[i][k][2] != 0.0:
                first_non_zero_idxs[i] = k
                break
    emas_long = [np.repeat(hlcs[i][k][2], 3) for i, k in enumerate(first_non_zero_idxs)]
    emas_short = [np.repeat(hlcs[i][k][2], 3) for i, k in enumerate(first_non_zero_idxs)]

    alphas_long = [2.0 / (x + 1.0) for x in ema_spans_long]
    alphas__long = [1.0 - x for x in alphas_long]
    alphas_short = [2.0 / (x + 1.0) for x in ema_spans_short]
    alphas__short = [1.0 - x for x in alphas_short]
    any_do_long = False
    for x in do_longs:
        if x:
            any_do_long = True
    any_do_short = False
    for x in do_shorts:
        if x:
            any_do_short = True

    stuck_positions_long = np.zeros(len(symbols))  # [timestamp first stuck]
    stuck_positions_long.fill(np.nan)
    stuck_positions_short = np.zeros(len(symbols))  # [timestamp first stuck]
    stuck_positions_short.fill(np.nan)

    bankrupt = False
    pnl_cumsum_running = 0.0
    pnl_cumsum_max = 0.0

    for k in range(1, len(hlcs[0])):
        any_fill = False
        if any_do_long:
            # check for fills
            for i in idxs:
                if hlcs[i][k][0] == 0.0:
                    continue
                emas_long[i] = calc_ema(alphas_long[i], alphas__long[i], emas_long[i], hlcs[i][k][2])
                try:
                    if (entries_long[i][0] > 0.0 and hlcs[i][k][1] < entries_long[i][1]) or (
                        poss_long[i][0] > 0.0
                        and closes_long[i][0][0] != 0.0
                        and hlcs[i][k][0] > closes_long[i][0][1]
                    ):
                        # there were fills
                        new_fills, new_pos_long, new_balance, new_equity = get_fills_long(
                            k,
                            poss_long,
                            poss_short,
                            idxs[i],
                            symbols[i],
                            balance,
                            entries_long[i],
                            closes_long[i],
                            hlcs[:, k],
                            inverse,
                            qty_steps[i],
                            c_mults,
                            maker_fee,
                        )
                        if len(new_fills) > 0:
                            any_fill = True
                        if new_equity <= 0.0:
                            bankrupt = True
                        for fill in new_fills:
                            pnl_cumsum_running += fill[2]
                            pnl_cumsum_max = max(pnl_cumsum_max, pnl_cumsum_running)
                        fills.extend(new_fills)
                        poss_long[i] = new_pos_long
                        balance = new_balance
                except:
                    print("debug")
                    print("closes_long", closes_long)
                    print("poss_long", poss_long)
                    print("k", k)
                    print("balance", balance)
                    return fills, stats
            any_stuck = False
            # check if open orders long need to be updated
            for i in idxs:
                if hlcs[i][k][0] == 0.0:
                    continue
                if any_fill or poss_long[i][0] == 0.0:
                    # calc entries if any fill or if psize is zero
                    entries_long[i], closes_long[i] = get_open_orders_long(
                        hlcs[i][k][2],
                        balance,
                        poss_long[i],
                        emas_long[i],
                        inverse,
                        qty_steps[i],
                        price_steps[i],
                        min_qtys[i],
                        min_costs[i],
                        c_mults[i],
                        ll[i],
                    )
                    wallet_exposure = (
                        qty_to_cost(poss_long[i][0], poss_long[i][1], inverse, c_mults[i]) / balance
                    )
                    if (
                        wallet_exposure / ll[i][16] > 0.9
                    ):  # ratio of wallet exposure to exposure limit
                        # is stuck
                        any_stuck = True
                        if np.isnan(stuck_positions_long[i]):
                            # record when pos first got stuck
                            stuck_positions_long[i] = k
                    else:
                        # is unstuck
                        stuck_positions_long[i] = np.nan

        # do shorts...
        # TODO

        if any_stuck:
            # find which position to unstuck
            # lowest pprice diff is chosen
            uside = 0  # 0==long, 1==short
            ui = 0  # index
            lowest_pprice_diff = 100.0
            for i in idxs:
                if not np.isnan(stuck_positions_long[i]):
                    # long is stuck
                    pprice_diff = 1.0 - hlcs[i][k][2] / poss_long[i][1]
                    if pprice_diff < lowest_pprice_diff:
                        lowest_pprice_diff = pprice_diff
                        ui = i
                        uside = 0
                if not np.isnan(stuck_positions_short[i]):
                    # short is stuck
                    pprice_diff = hlcs[i][k][2] / poss_short[i][1] - 1.0
                    if pprice_diff < lowest_pprice_diff:
                        lowest_pprice_diff = pprice_diff
                        ui = i
                        uside = 1
            AU_allowance = calc_AU_allowance(
                np.array([0.0]), balance, drop_since_peak_abs=(pnl_cumsum_max - pnl_cumsum_running)
            )
            if AU_allowance > 0.0:
                # create unstuck close
                if uside == 0:  # long
                    close_price = emas_long[ui].max()  # upper ema band
                    close_qty_min = calc_min_entry_qty(
                        close_price, inverse, qty_steps[ui], min_qtys[ui], min_costs[ui]
                    )
                    close_cost_max = (
                        balance * (ll[ui][16] if uside == 0 else ls[ui][16]) * 0.05
                    )  # close max 5% of max psize at a time
                    close_qty_max = min(
                        poss_long[ui][0],
                        round_(
                            cost_to_qty(close_cost_max, close_price, inverse, c_mults[ui]),
                            qty_steps[ui],
                        ),
                    )
                    close_qty = round_(
                        cost_to_qty(AU_allowance, close_price, inverse, c_mults[ui]),
                        qty_steps[ui],
                    )
                    close_qty = max(close_qty_min, min(close_qty_max, close_qty))
                    closes_long[ui] = [(-abs(close_qty), close_price, "unstuck_close_long")]
                else:  # short
                    pass
            """
            for idx in idxs:
                if np.isnan(stuck_positions_long)
            """

            # TODO: also check if in auto unstuck mode. For now, single coin AU is disabled, in favor of multi sym AU
        if k % 60 == 0:
            # update stats hourly
            equity = balance + calc_pnl_sum(poss_long, poss_short, hlcs[:, k, 2], c_mults)
            stats.append(
                (
                    k,
                    poss_long.copy(),
                    poss_short.copy(),
                    hlcs[:, k, 2],
                    balance,
                    equity,
                )
            )
            if equity <= 0.0 or bankrupt:
                # bankrupt
                break
    if stats[-1][0] != k:
        stats.append(
            (
                stats[-1][0] + 60,
                poss_long.copy(),
                poss_short.copy(),
                hlcs[:, k, 2],
                balance,
                balance + calc_pnl_sum(poss_long, poss_short, hlcs[:, k, 2], c_mults),
            )
        )
    return fills, stats
