import numpy as np
import os

from njit_funcs import (
    calc_min_entry_qty,
    calc_initial_entry_qty,
    cost_to_qty,
    qty_to_cost,
    basespace,
    round_,
    round_dn,
    round_up,
    calc_new_psize_pprice,
    interpolate,
    calc_close_grid_long,
    calc_close_grid_short,
    calc_ema,
    calc_pnl_long,
    calc_pnl_short,
    calc_diff,
    calc_bankruptcy_price,
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
def calc_neat_grid_long(
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
    eqty_exp_base,
    eprice_exp_base,
    auto_unstuck_wallet_exposure_threshold,
    auto_unstuck_ema_dist,
) -> [(float, float, str)]:
    if wallet_exposure_limit == 0.0:
        return [(0.0, 0.0, "")]
    if not do_long and psize == 0.0:
        return [(0.0, 0.0, "")]
    ientry_price = min(
        highest_bid,
        round_dn(ema_band_lower * (1 - initial_eprice_ema_dist), price_step),
    )
    min_ientry_qty = calc_min_entry_qty(ientry_price, inverse, qty_step, min_qty, min_cost)
    if psize < min_ientry_qty * 0.9:  # initial entry
        entry_qty = calc_initial_entry_qty(
            balance,
            ientry_price,
            inverse,
            qty_step,
            min_qty,
            min_cost,
            c_mult,
            wallet_exposure_limit,
            initial_qty_pct,
        )
        if psize > 0.0:  # partial initial entry
            entry_qty = max(min_ientry_qty, round_(entry_qty - psize, qty_step))
        return [(entry_qty, ientry_price, "long_ientry")]
    wallet_exposure = qty_to_cost(psize, pprice, inverse, c_mult) / balance
    if wallet_exposure >= wallet_exposure_limit * 0.99:
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
    grid = approximate_neat_grid_long(
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
        eqty_exp_base,
        eprice_exp_base,
    )
    if len(grid) == 0:
        return [(0.0, 0.0, "")]
    entries = []
    for i in range(len(grid)):
        if grid[i][1] > pprice * 0.9995:
            continue
        if grid[i][4] > wallet_exposure_limit * 1.1:
            break
        entry_price = min(highest_bid, grid[i][1])
        min_entry_qty = calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost)
        grid[i][1] = entry_price
        grid[i][0] = max(min_entry_qty, grid[i][0])
        comment = "long_primary_rentry"
        if not entries or (entries[-1][1] != entry_price):
            entries.append((grid[i][0], grid[i][1], comment))
    return entries if entries else [(0.0, 0.0, "")]


@njit
def calc_neat_grid_short(
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
    eqty_exp_base,
    eprice_exp_base,
    auto_unstuck_wallet_exposure_threshold,
    auto_unstuck_ema_dist,
) -> [(float, float, str)]:
    if wallet_exposure_limit == 0.0:
        return [(0.0, 0.0, "")]
    if not do_short and psize == 0.0:
        return [(0.0, 0.0, "")]
    ientry_price = max(
        lowest_ask,
        round_up(ema_band_upper * (1 + initial_eprice_ema_dist), price_step),
    )
    abs_psize = abs(psize)
    min_ientry_qty = calc_min_entry_qty(ientry_price, inverse, qty_step, min_qty, min_cost)
    if abs_psize < min_ientry_qty * 0.9:  # initial entry
        entry_qty = calc_initial_entry_qty(
            balance,
            ientry_price,
            inverse,
            qty_step,
            min_qty,
            min_cost,
            c_mult,
            wallet_exposure_limit,
            initial_qty_pct,
        )
        if abs_psize > 0.0:  # partial initial entry
            entry_qty = max(min_ientry_qty, round_(entry_qty - abs_psize, qty_step))
        return [(-entry_qty, ientry_price, "short_ientry")]
    wallet_exposure = qty_to_cost(abs_psize, pprice, inverse, c_mult) / balance
    if wallet_exposure >= wallet_exposure_limit * 0.99:
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
                abs_psize,
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
    grid = approximate_neat_grid_short(
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
        eqty_exp_base,
        eprice_exp_base,
    )
    if len(grid) == 0:
        return [(0.0, 0.0, "")]
    entries = []
    for i in range(len(grid)):
        if grid[i][1] < pprice * 0.9995:
            continue
        if grid[i][4] > wallet_exposure_limit * 1.1:
            break
        entry_price = max(lowest_ask, grid[i][1])
        min_entry_qty = calc_min_entry_qty(entry_price, inverse, qty_step, min_qty, min_cost)
        grid[i][1] = entry_price
        grid[i][0] = -max(min_entry_qty, abs(grid[i][0]))
        comment = "short_primary_rentry"
        if not entries or (entries[-1][1] != entry_price):
            entries.append((grid[i][0], grid[i][1], comment))
    return entries if entries else [(0.0, 0.0, "")]


@njit
def approximate_neat_grid_long(
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
    eqty_exp_base,
    eprice_exp_base,
    crop: bool = True,
):
    def eval_(ientry_price_guess, psize_):
        ientry_price_guess = round_(ientry_price_guess, price_step)
        grid = calc_whole_neat_entry_grid_long(
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
            eqty_exp_base,
            eprice_exp_base,
        )
        # find node whose psize is closest to psize
        diff, i = sorted([(abs(grid[i][2] - psize_) / psize_, i) for i in range(len(grid))])[0]
        return grid, diff, i

    if pprice == 0.0 or psize == 0.0:
        raise Exception("cannot appriximate grid without pprice and psize")
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
def approximate_neat_grid_short(
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
    eqty_exp_base,
    eprice_exp_base,
    crop: bool = True,
):
    def eval_(ientry_price_guess, psize_):
        ientry_price_guess = round_(ientry_price_guess, price_step)
        grid = calc_whole_neat_entry_grid_short(
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
            eqty_exp_base,
            eprice_exp_base,
        )
        # find node whose psize is closest to psize
        abs_psize_ = abs(psize_)
        diff, i = sorted(
            [(abs(abs(grid[i][2]) - abs_psize_) / abs_psize_, i) for i in range(len(grid))]
        )[0]
        return grid, diff, i

    abs_psize = abs(psize)
    if pprice == 0.0 or psize == 0.0:
        raise Exception("cannot appriximate grid without pprice and psize")
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
def eval_neat_entry_grid_long(
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
    eqty_exp_base,
    eprice_exp_base,
    last_entry_qty,
):

    # returns [[qty, price, psize, pprice, wallet_exposure]]
    grid = np.zeros((max_n_entry_orders, 5))
    grid[:, 1] = [
        round_dn(p, price_step)
        for p in basespace(
            initial_entry_price,
            initial_entry_price * (1 - grid_span),
            eprice_exp_base,
            max_n_entry_orders,
        )
    ]

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
    grid[0][3] = pprice = grid[0][1]
    grid[0][4] = qty_to_cost(psize, pprice, inverse, c_mult) / balance
    qtys = basespace(grid[0][0], last_entry_qty, eqty_exp_base, max_n_entry_orders)
    for i in range(1, max_n_entry_orders):
        qty = max(
            calc_min_entry_qty(grid[i][1], inverse, qty_step, min_qty, min_cost),
            round_(qtys[i], qty_step),
        )
        psize, pprice = calc_new_psize_pprice(psize, pprice, qty, grid[i][1], qty_step)
        grid[i][0] = qty
        grid[i][2:] = [
            psize,
            pprice,
            qty_to_cost(psize, pprice, inverse, c_mult) / balance,
        ]
    return grid


@njit
def eval_neat_entry_grid_short(
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
    eqty_exp_base,
    eprice_exp_base,
    last_entry_qty,
):

    # returns [[qty, price, psize, pprice, wallet_exposure]]
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
    grid[0][3] = pprice = grid[0][1]
    grid[0][4] = qty_to_cost(psize, pprice, inverse, c_mult) / balance
    qtys = basespace(abs(grid[0][0]), last_entry_qty, eqty_exp_base, max_n_entry_orders)
    for i in range(1, max_n_entry_orders):
        qty = -max(
            calc_min_entry_qty(grid[i][1], inverse, qty_step, min_qty, min_cost),
            round_(qtys[i], qty_step),
        )
        psize, pprice = calc_new_psize_pprice(psize, pprice, qty, grid[i][1], qty_step)
        grid[i][0] = qty
        grid[i][2:] = [
            psize,
            pprice,
            qty_to_cost(psize, pprice, inverse, c_mult) / balance,
        ]
    return grid


@njit
def find_last_entry_qty_long(
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
    eqty_exp_base,
    eprice_exp_base,
):
    guess0 = max(
        calc_min_entry_qty(initial_entry_price, inverse, qty_step, min_qty, min_cost),
        cost_to_qty(
            balance * wallet_exposure_limit * 0.5,
            initial_entry_price,
            inverse,
            c_mult,
        ),
    )
    val0 = eval_neat_entry_grid_long(
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
        eqty_exp_base,
        eprice_exp_base,
        guess0,
    )[-1][-1]
    eval0 = abs(val0 - wallet_exposure_limit) / wallet_exposure_limit
    guess1 = guess0 * (1.2 if val0 < wallet_exposure_limit else 0.8)
    val1 = eval_neat_entry_grid_long(
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
        eqty_exp_base,
        eprice_exp_base,
        guess1,
    )[-1][-1]
    if val0 == val1:
        guess1 = guess0 * 10
        val1 = eval_neat_entry_grid_long(
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
            eqty_exp_base,
            eprice_exp_base,
            guess1,
        )[-1][-1]
    eval1 = abs(val0 - wallet_exposure_limit) / wallet_exposure_limit
    return round_(
        interpolate(wallet_exposure_limit, np.array([val0, val1]), np.array([guess0, guess1])),
        qty_step,
    )


@njit
def find_last_entry_qty_short(
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
    eqty_exp_base,
    eprice_exp_base,
):
    guess0 = max(
        calc_min_entry_qty(initial_entry_price, inverse, qty_step, min_qty, min_cost),
        cost_to_qty(
            balance * wallet_exposure_limit * 0.5,
            initial_entry_price,
            inverse,
            c_mult,
        ),
    )
    val0 = eval_neat_entry_grid_short(
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
        eqty_exp_base,
        eprice_exp_base,
        guess0,
    )[-1][-1]
    eval0 = abs(val0 - wallet_exposure_limit) / wallet_exposure_limit
    guess1 = guess0 * (1.2 if val0 < wallet_exposure_limit else 0.8)
    val1 = eval_neat_entry_grid_short(
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
        eqty_exp_base,
        eprice_exp_base,
        guess1,
    )[-1][-1]
    if val0 == val1:
        guess1 = guess0 * 10
        val1 = eval_neat_entry_grid_short(
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
            eqty_exp_base,
            eprice_exp_base,
            guess1,
        )[-1][-1]
    eval1 = abs(val0 - wallet_exposure_limit) / wallet_exposure_limit
    return round_(
        interpolate(wallet_exposure_limit, np.array([val0, val1]), np.array([guess0, guess1])),
        qty_step,
    )


@njit
def calc_whole_neat_entry_grid_long(
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
    eqty_exp_base,
    eprice_exp_base,
):

    # [qty, price, psize, pprice, wallet_exposure]
    last_entry_qty = find_last_entry_qty_long(
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
        eqty_exp_base,
        eprice_exp_base,
    )
    return eval_neat_entry_grid_long(
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
        eqty_exp_base,
        eprice_exp_base,
        last_entry_qty,
    )


@njit
def calc_whole_neat_entry_grid_short(
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
    eqty_exp_base,
    eprice_exp_base,
):

    # [qty, price, psize, pprice, wallet_exposure]
    last_entry_qty = find_last_entry_qty_short(
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
        eqty_exp_base,
        eprice_exp_base,
    )
    return eval_neat_entry_grid_short(
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
        eqty_exp_base,
        eprice_exp_base,
        last_entry_qty,
    )


@njit
def backtest_neat_grid(
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
    eqty_exp_base,
    eprice_exp_base,
    grid_span,
    initial_eprice_ema_dist,
    initial_qty_pct,
    markup_range,
    max_n_entry_orders,
    min_markup,
    n_close_orders,
    wallet_exposure_limit,
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
                    entries_long = calc_neat_grid_long(
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
                        eqty_exp_base[0],
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

                # check if short entry grid should be updated
                if timestamps[k] >= next_entry_grid_update_ts_short:
                    entries_short = calc_neat_grid_short(
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
                        eqty_exp_base[1],
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
                    next_close_grid_update_ts_short = timestamps[k] + 1000 * 60 * 5

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
