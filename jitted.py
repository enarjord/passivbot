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
def round_up(n, step, safety_rounding=10) -> float:
    return np.round(np.ceil(n / step) * step, safety_rounding)


@njit
def round_dn(n, step, safety_rounding=10) -> float:
    return np.round(np.floor(n / step) * step, safety_rounding)


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
def calc_ema(alpha, alpha_, prev_ema, new_val) -> float:
    return prev_ema * alpha_ + new_val * alpha


@njit
def calc_emas(xs: [float], span: int) -> np.ndarray:
    alpha = 2 / (span + 1)
    alpha_ = 1 - alpha
    emas = np.empty_like(xs)
    emas[0] = xs[0]
    for i in range(1, len(xs)):
        emas[i] = emas[i - 1] * alpha_ + xs[i] * alpha
    return emas


@njit
def calc_stds(xs: [float], span: int) -> np.ndarray:
    stds = np.zeros_like(xs)
    if len(stds) <= span:
        return stds
    xsum = xs[:span].sum()
    xsum_sq = (xs[:span] ** 2).sum()
    stds[span] = np.sqrt((xsum_sq / span) - (xsum / span) ** 2)
    for i in range(span, len(xs)):
        xsum += xs[i] - xs[i - span]
        xsum_sq += xs[i] ** 2 - xs[i - span] ** 2
        stds[i] = np.sqrt((xsum_sq / span) - (xsum / span) ** 2)
    return stds


@njit
def iter_indicator_chunks(xs: [float], span: int, chunk_size: int = 65536):


    if len(xs) < span:
        return

    chunk_size = max(chunk_size, span)

    n_chunks = int(round_up(len(xs) / chunk_size, 1.0))

    alpha = 2 / (span + 1)
    alpha_ = 1 - alpha

    emas = np.zeros(chunk_size)
    emas[0] = xs[0]
    for i in range(1, chunk_size):
        emas[i] = emas[i - 1] * alpha_ + xs[i] * alpha

    stds = np.zeros(chunk_size)

    xsum = xs[:span].sum()
    xsum_sq = (xs[:span] ** 2).sum()
    for i in range(span, chunk_size):
        xsum += xs[i] - xs[i - span]
        xsum_sq += xs[i] ** 2 - xs[i - span] ** 2
        stds[i] = np.sqrt((xsum_sq / span) - (xsum / span) ** 2)
    yield emas, stds, 0

    for k in range(1, n_chunks):
        kc = chunk_size * k
        new_emas = np.zeros(chunk_size)
        new_stds = np.zeros(chunk_size)
        new_emas[0] = emas[-1] * alpha_ + xs[kc] * alpha
        xsum += xs[kc] - xs[kc - span]
        xsum_sq += xs[kc] ** 2 - xs[kc - span] ** 2
        new_stds[0] = np.sqrt((xsum_sq / span) - (xsum / span) ** 2)
        for i in range(1, chunk_size):
            new_emas[i] = new_emas[i - 1] * alpha_ + xs[kc + i] * alpha
            xsum += xs[kc + i] - xs[kc + i - span]
            xsum_sq += xs[kc + i] ** 2 - xs[kc + i - span] ** 2
            new_stds[i] = np.sqrt((xsum_sq / span) - (xsum / span) ** 2)
        yield new_emas, new_stds, k
        emas, stds = new_emas, new_stds


@njit
def calc_min_entry_qty(price, inverse, qty_step, min_qty, min_cost, contract_multiplier) -> float:
    return max(min_qty, round_up(min_cost * (price / contract_multiplier if inverse else 1 / price),
                                 qty_step))


@njit
def calc_initial_entry_qty(balance,
                           price,
                           available_margin,
                           volatility,
                           inverse, qty_step, min_qty, min_cost, contract_multiplier, leverage,
                           qty_pct, volatility_qty_coeff) -> float:
    min_entry_qty = calc_min_entry_qty(price, inverse, qty_step, min_qty, min_cost,
                                       contract_multiplier)
    if inverse:
        qty = round_dn(min(available_margin * leverage * price / contract_multiplier,
                       max(min_entry_qty, ((balance / contract_multiplier) * price * leverage *
                                           qty_pct * (1 + volatility * volatility_qty_coeff)))),
                       qty_step)
    else:
        qty = round_dn(min(available_margin * leverage / price,
                       max(min_entry_qty, ((balance / price) * leverage * qty_pct *
                                           (1 + volatility * volatility_qty_coeff)))),
                       qty_step)
    return qty if qty >= min_entry_qty else 0.0


@njit
def calc_reentry_qty(psize, price, available_margin, inverse, qty_step, min_qty,
                     min_cost, contract_multiplier, ddown_factor, leverage) -> float:
    min_entry_qty = calc_min_entry_qty(price, inverse, qty_step, min_qty, min_cost,
                                       contract_multiplier)
    qty = min(round_dn(available_margin * leverage * (price / contract_multiplier
                                                      if inverse else 1 / price), qty_step),
              max(min_entry_qty, round_dn(abs(psize) * ddown_factor, qty_step)))
    return qty if qty >= min_entry_qty else 0.0


@njit
def calc_new_psize_pprice(psize, pprice, qty, price, qty_step) -> (float, float):
    if qty == 0.0:
        return psize, pprice
    new_psize = round_(psize + qty, qty_step)
    if new_psize == 0.0:
        return 0.0, 0.0
    return new_psize, nan_to_0(pprice) * (psize / new_psize) + price * (qty / new_psize)


@njit
def calc_long_pnl(entry_price, close_price, qty, inverse, contract_multiplier) -> float:
    if inverse:
        if entry_price == 0.0 or close_price == 0.0:
            return 0.0
        return abs(qty) * contract_multiplier * (1 / entry_price - 1 / close_price)
    else:
        return abs(qty) * (close_price - entry_price)


@njit
def calc_shrt_pnl(entry_price, close_price, qty, inverse, contract_multiplier) -> float:
    if inverse:
        if entry_price == 0.0 or close_price == 0.0:
            return 0.0
        return abs(qty) * contract_multiplier * (1 / close_price - 1 / entry_price)
    else:
        return abs(qty) * (entry_price - close_price)


@njit
def calc_cost(qty, price, inverse, contract_multiplier) -> float:
    return abs(qty / price) * contract_multiplier if inverse else abs(qty * price)


@njit
def calc_margin_cost(qty, price, inverse, contract_multiplier, leverage) -> float:
    return calc_cost(qty, price, inverse, contract_multiplier) / leverage


@njit
def calc_available_margin(balance,
                          long_psize,
                          long_pprice,
                          shrt_psize,
                          shrt_pprice,
                          last_price,
                          inverse, contract_multiplier, leverage) -> float:
    used_margin = 0.0
    equity = balance
    if long_pprice and long_psize:
        long_psize_real = long_psize * contract_multiplier
        equity += calc_long_pnl(long_pprice, last_price, long_psize_real, inverse,
                                contract_multiplier)
        used_margin += calc_cost(long_psize_real, long_pprice,
                                 inverse, contract_multiplier) / leverage
    if shrt_pprice and shrt_psize:
        shrt_psize_real = shrt_psize * contract_multiplier
        equity += calc_shrt_pnl(shrt_pprice, last_price, shrt_psize_real, inverse,
                                contract_multiplier)
        used_margin += calc_cost(shrt_psize_real, shrt_pprice,
                                 inverse, contract_multiplier) / leverage
    return equity - used_margin


@njit
def iter_entries(
        balance,
        long_psize,
        long_pprice,
        shrt_psize,
        shrt_pprice,
        liq_price,
        highest_bid,
        lowest_ask,
        ema,
        last_price,
        volatility,

        inverse, do_long, do_shrt, qty_step, price_step, min_qty, min_cost, contract_multiplier,
        ddown_factor, qty_pct, leverage, n_close_orders, grid_spacing, pos_margin_grid_coeff,
        volatility_grid_coeff, volatility_qty_coeff, min_markup, markup_range, ema_span, ema_spread,
        stop_loss_liq_diff, stop_loss_pos_pct):

    available_margin = calc_available_margin(balance, long_psize, long_pprice,
                                             shrt_psize, shrt_pprice, last_price,
                                             inverse, contract_multiplier, leverage)
    stop_loss_order = calc_stop_loss(balance, long_psize, long_pprice, shrt_psize, shrt_pprice,
                                     liq_price, highest_bid, lowest_ask, last_price,
                                     available_margin, inverse, do_long, do_shrt, qty_step, min_qty,
                                     min_cost, contract_multiplier, leverage, stop_loss_liq_diff,
                                     stop_loss_pos_pct)

    if stop_loss_order[0] != 0.0:
        yield stop_loss_order
        if 'long' in stop_loss_order[4]:
            long_psize, long_pprice = stop_loss_order[2:4]
        elif 'shrt' in stop_loss_order[4]:
            shrt_psize, shrt_pprice = stop_loss_order[2:4]
        if 'entry' in stop_loss_order[4]:
            available_margin -= calc_margin_cost(stop_loss_order[0], stop_loss_order[1], inverse,
                                                 contract_multiplier, leverage)
        elif 'close' in stop_loss_order[4]:
            available_margin += calc_margin_cost(stop_loss_order[0], stop_loss_order[1], inverse,
                                                 contract_multiplier, leverage)
    while True:
        if do_long:
            if long_psize == 0.0:
                price = min(highest_bid, round_dn(ema * (1 - ema_spread), price_step))
                qty = calc_initial_entry_qty(balance, price, available_margin, volatility, inverse,
                                             qty_step, min_qty, min_cost, contract_multiplier,
                                             leverage, qty_pct, volatility_qty_coeff)
                long_entry = (qty, price, qty, price, 'initial_long_entry')
            else:
                modifier = (1 + (calc_margin_cost(long_psize, long_pprice, inverse,
                                                  contract_multiplier, leverage) / balance) *
                            pos_margin_grid_coeff) * \
                           (1 + volatility * volatility_grid_coeff)
                price = min(round_(highest_bid, price_step),
                            round_dn(long_pprice * (1 - grid_spacing * modifier), price_step))
                if price <= 0.0:
                    long_entry = (0.0, 0.0, long_psize, long_pprice, 'long_reentry')
                else:
                    qty = calc_reentry_qty(long_psize, price, available_margin, inverse, qty_step,
                                           min_qty, min_cost, contract_multiplier, ddown_factor,
                                           leverage)
                    long_psize, long_pprice = calc_new_psize_pprice(long_psize, long_pprice,
                                                                              qty, price, qty_step)
                    long_entry = (qty, price, long_psize, long_pprice, 'long_reentry')
        else:
            long_entry = (0.0, 0.0, long_psize, long_pprice, '')

        if do_shrt:
            if shrt_psize == 0.0:
                price = max(lowest_ask, round_up(ema * (1 + ema_spread), price_step))
                qty = -calc_initial_entry_qty(balance, price, available_margin, volatility, inverse,
                                              qty_step, min_qty, min_cost, contract_multiplier,
                                              leverage, qty_pct, volatility_qty_coeff)
                shrt_entry = (qty, price, qty, price, 'initial_shrt_entry')
            else:
                modifier = (1 + (calc_margin_cost(shrt_psize, shrt_pprice, inverse,
                                                  contract_multiplier, leverage) /
                                 balance) * pos_margin_grid_coeff) * \
                           (1 + volatility * volatility_grid_coeff)
                price = max(round_(lowest_ask, price_step),
                            round_dn(shrt_pprice * (1 + grid_spacing * modifier), price_step))
                qty = -calc_reentry_qty(shrt_psize, price, available_margin, inverse, qty_step,
                                        min_qty, min_cost, contract_multiplier, ddown_factor,
                                        leverage)
                shrt_psize, shrt_pprice = calc_new_psize_pprice(shrt_psize, shrt_pprice, qty, price,
                                                                qty_step)
                shrt_entry = (qty, price, shrt_psize, shrt_pprice, 'shrt_reentry')
        else:
            shrt_entry = (0.0, 0.0, shrt_psize, shrt_pprice, '')

        if long_entry[0] > 0.0:
            if shrt_entry[0] == 0.0:
                long_first = True
            else:
                long_first = (calc_diff(long_entry[1], last_price) <
                              calc_diff(shrt_entry[1], last_price))
        elif shrt_entry[0] < 0.0:
            long_first = False
        else:
            break
        if long_first:
            yield long_entry
            long_psize, long_pprice = long_entry[2:4]
            if long_entry[1]:
                available_margin -= calc_margin_cost(long_entry[0], long_entry[1], inverse,
                                                     contract_multiplier, leverage)
        else:
            yield shrt_entry
            shrt_psize, shrt_pprice = shrt_entry[2:4]
            if shrt_entry[1]:
                available_margin -= calc_margin_cost(shrt_entry[0], shrt_entry[1], inverse,
                                                     contract_multiplier, leverage)


@njit
def calc_stop_loss(balance,
                   long_psize,
                   long_pprice,
                   shrt_psize,
                   shrt_pprice,
                   liq_price,
                   highest_bid,
                   lowest_ask,
                   last_price,
                   available_margin,
                   inverse, do_long, do_shrt, qty_step, min_qty, min_cost, contract_multiplier,
                   leverage, stop_loss_liq_diff, stop_loss_pos_pct):
    # returns (qty, price, psize if taken, pprice if taken, comment)
    abs_shrt_psize = abs(shrt_psize)
    if calc_diff(liq_price, last_price) < stop_loss_liq_diff:
        if long_psize > abs_shrt_psize:
            stop_loss_qty = min(long_psize,
                                max(calc_min_entry_qty(lowest_ask, inverse, qty_step, min_qty,
                                                       min_cost, contract_multiplier),
                                    round_dn(long_psize * stop_loss_pos_pct, qty_step)))
            # if sufficient margin available, increase short pos, otherwise reduce long pos
            margin_cost = calc_margin_cost(stop_loss_qty, lowest_ask, inverse, contract_multiplier,
                                           leverage)
            if margin_cost < available_margin and do_shrt:
                # add to shrt pos
                shrt_psize, shrt_pprice = calc_new_psize_pprice(shrt_psize, shrt_pprice,
                                                                -stop_loss_qty, lowest_ask,
                                                                qty_step)
                return -stop_loss_qty, lowest_ask, shrt_psize, shrt_pprice, 'stop_loss_shrt_entry'
            else:
                # reduce long pos
                long_psize = round_(long_psize - stop_loss_qty, qty_step)
                return -stop_loss_qty, lowest_ask, long_psize, long_pprice, 'stop_loss_long_close'
        else:
            stop_loss_qty = min(abs_shrt_psize,
                                max(calc_min_entry_qty(highest_bid, inverse, qty_step, min_qty,
                                                       min_cost, contract_multiplier),
                                    round_dn(abs_shrt_psize * stop_loss_pos_pct, qty_step)))
            # if sufficient margin available, increase long pos, otherwise, reduce shrt pos
            margin_cost = calc_margin_cost(stop_loss_qty, highest_bid, inverse, contract_multiplier,
                                           leverage)
            if margin_cost < available_margin and do_long:
                # add to long pos
                long_psize, long_pprice = calc_new_psize_pprice(long_psize, long_pprice,
                                                                stop_loss_qty, highest_bid,
                                                                qty_step)
                return stop_loss_qty, highest_bid, long_psize, long_pprice, 'stop_loss_long_entry'
            else:
                # reduce shrt pos
                shrt_psize = round_(shrt_psize + stop_loss_qty, qty_step)
                return stop_loss_qty, highest_bid, shrt_psize, shrt_pprice, 'stop_loss_shrt_close'
    return 0.0, 0.0, 0.0, 0.0, ''


@njit
def iter_long_closes(
        balance, 
        psize, 
        pprice, 
        lowest_ask,

        inverse, do_long, do_shrt, qty_step, price_step, min_qty, min_cost, contract_multiplier,
        ddown_factor, qty_pct, leverage, n_close_orders, grid_spacing, pos_margin_grid_coeff,
        volatility_grid_coeff, volatility_qty_coeff, min_markup, markup_range, ema_span, ema_spread,
        stop_loss_liq_diff, stop_loss_pos_pct):
    # yields (qty, price, psize_if_taken)
    if psize == 0.0 or pprice == 0.0:
        return
    minm = pprice * (1 + min_markup)
    prices = np.linspace(minm, pprice * (1 + min_markup + markup_range),
                         int(n_close_orders))
    prices = [p for p in sorted(set([round_up(p_, price_step)
                                     for p_ in prices])) if p >= lowest_ask]
    if len(prices) == 0:
        yield -psize, max(lowest_ask, round_up(minm, price_step)), 0.0
    else:
        n_orders = int(min([n_close_orders, len(prices), int(psize / min_qty)]))
        for price in prices:
            if n_orders == 0:
                break
            else:
                qty = min(psize, max(calc_initial_entry_qty(balance, lowest_ask, balance, 0.0,
                                                            inverse, qty_step, min_qty, min_cost,
                                                            contract_multiplier, leverage, qty_pct,
                                                            volatility_qty_coeff),
                                     round_up(psize / n_orders, qty_step)))
                if psize != 0.0 and qty / psize > 0.75:
                    qty = psize
            if qty == 0.0:
                break
            psize = round_(psize - qty, qty_step)
            yield -qty, price, psize
            lowest_ask = price
            n_orders -= 1
        if psize > 0.0:
            yield -psize, max(lowest_ask, round_up(minm, price_step)), 0.0


@njit
def iter_shrt_closes(
        balance, 
        psize, 
        pprice, 
        highest_bid,

        inverse, do_long, do_shrt, qty_step, price_step, min_qty, min_cost, contract_multiplier,
        ddown_factor, qty_pct, leverage, n_close_orders, grid_spacing, pos_margin_grid_coeff,
        volatility_grid_coeff, volatility_qty_coeff, min_markup, markup_range, ema_span, ema_spread,
        stop_loss_liq_diff, stop_loss_pos_pct):
    # yields (qty, price, psize_if_taken)
    abs_psize = abs(psize)
    if psize == 0.0:
        return
    minm = pprice * (1 - min_markup)
    prices = np.linspace(minm, pprice * (1 - (min_markup + markup_range)),
                         int(n_close_orders))
    prices = [p for p in sorted(set([round_dn(p_, price_step)
                                     for p_ in prices]), reverse=True) if p <= highest_bid]
    if len(prices) == 0:
        yield abs_psize, min(highest_bid, round_dn(minm, price_step)), 0.0
    else:
        n_orders = int(min([n_close_orders, len(prices),
                            int(abs_psize / min_qty)]))
        for price in prices:
            if n_orders == 0:
                break
            else:
                qty = min(abs_psize, max(calc_initial_entry_qty(balance, highest_bid, balance, 0.0,
                                                                inverse, qty_step, min_qty,
                                                                min_cost, contract_multiplier,
                                                                leverage, qty_pct,
                                                                volatility_qty_coeff),
                                         round_up(abs_psize / n_orders, qty_step)))
                if abs_psize != 0.0 and qty / abs_psize > 0.75:
                    qty = abs_psize
            if qty == 0.0:
                break
            abs_psize = round_(abs_psize - qty, qty_step)
            yield qty, price, abs_psize
            highest_bid = price
            n_orders -= 1
        if abs_psize > 0.0:
            yield abs_psize, min(highest_bid, round_dn(minm, price_step)), 0.0


@njit
def calc_liq_price_binance(balance,
                           long_psize,
                           long_pprice,
                           shrt_psize,
                           shrt_pprice,
                           inverse, contract_multiplier, leverage):
    abs_long_psize = abs(long_psize)
    abs_shrt_psize = abs(shrt_psize)
    long_pprice = nan_to_0(long_pprice)
    shrt_pprice = nan_to_0(shrt_pprice)
    if inverse:
        mml = 0.02
        mms = 0.02
        numerator = abs_long_psize * mml + abs_shrt_psize * mms + abs_long_psize - abs_shrt_psize
        long_pcost = abs_long_psize / long_pprice if long_pprice > 0.0 else 0.0
        shrt_pcost = abs_shrt_psize / shrt_pprice if shrt_pprice > 0.0 else 0.0
        denom = balance / contract_multiplier + long_pcost - shrt_pcost
        if denom == 0.0:
            return 0.0
        return max(0.0, numerator / denom)
    else:
        mml = 0.006
        mms = 0.006
        # tmm = max(long_pos_margin, shrt_pos_margin)
        numerator = (balance - abs_long_psize * long_pprice + abs_shrt_psize * shrt_pprice)
        denom = (abs_long_psize * mml + abs_shrt_psize * mms - abs_long_psize + abs_shrt_psize)
        if denom == 0.0:
            return 0.0
        return max(0.0, numerator / denom)


@njit
def calc_liq_price_bybit(balance,
                         long_psize,
                         long_pprice,
                         shrt_psize,
                         shrt_pprice,
                         inverse, contract_multiplier, leverage):
    mm = 0.005
    abs_shrt_psize = abs(shrt_psize)
    if inverse:
        if long_psize > abs_shrt_psize:
            long_pprice = nan_to_0(long_pprice)
            if long_pprice == 0.0:
                return 0.0
            order_cost = long_psize / long_pprice if long_pprice > 0.0 else 0.0
            order_margin = order_cost / leverage
            bpdenom = order_cost + (balance - order_margin)
            bankruptcy_price = (1.00075 * long_psize) / bpdenom if bpdenom else 0.0
            if bankruptcy_price == 0.0:
                return 0.0
            rhs = -(balance - order_margin - (long_psize / long_pprice) * mm -
                    (long_psize * 0.00075) / bankruptcy_price)
            rdenom = long_psize - long_pprice * rhs
            return max(0.0, (long_pprice * long_psize) / rdenom if rdenom else 0.0)
        else:
            shrt_pprice = nan_to_0(shrt_pprice)
            if shrt_pprice == 0.0:
                return 0.0
            order_cost = abs_shrt_psize / shrt_pprice if shrt_pprice > 0.0 else 0.0
            order_margin = order_cost / leverage
            bpdenom = order_cost - (balance - order_margin)
            bankruptcy_price = (0.99925 * abs_shrt_psize) / bpdenom if bpdenom else 0.0
            if bankruptcy_price == 0.0:
                return 0.0
            rhs = -(balance - order_margin - (abs_shrt_psize / shrt_pprice) * mm -
                    (abs_shrt_psize * 0.00075) / bankruptcy_price)
            return max(0.0, (shrt_pprice * abs_shrt_psize) / (shrt_pprice * rhs + abs_shrt_psize))
    else:
        raise Exception('bybit linear liq price not implemented')



