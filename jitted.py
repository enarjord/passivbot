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


def set_XK(XK):
    globals()['XK'] = XK


@njit
def round_up(n: float, step: float, safety_rounding=10) -> float:
    return np.round(np.ceil(n / step) * step, safety_rounding)


@njit
def round_dn(n: float, step: float, safety_rounding=10) -> float:
    return np.round(np.floor(n / step) * step, safety_rounding)


@njit
def round_(n: float, step: float, safety_rounding=10) -> float:
    return np.round(np.round(n / step) * step, safety_rounding)


@njit
def calc_diff(x, y):
    return abs(x - y) / abs(y)


@njit
def nan_to_0(x: float) -> float:
    return x if x == x else 0.0


@njit
def calc_ema(alpha: float, alpha_: float, prev_ema: float, new_val: float) -> float:
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
    stds = np.empty_like(xs)
    stds.fill(0.0)
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
def calc_initial_long_entry_price(ema: float, highest_bid: float) -> float:
    return min(highest_bid, round_dn(ema * (1 - XK.ema_spread.value), XK.price_step.value))


@njit
def calc_initial_long_entry_price_test(ema: float, highest_bid: float) -> float:
    return min(highest_bid, round_dn(ema * (1 - XK.ema_spread.value), XK.price_step.value))


@njit
def calc_initial_shrt_entry_price(ema: float, lowest_ask: float) -> float:
    return max(lowest_ask, round_up(ema * (1 + XK.ema_spread.value), XK.price_step.value))


@njit
def calc_min_entry_qty(price: float) -> float:
    return max(XK.min_qty.value,
               round_up(XK.min_cost.value * (price / XK.contract_multiplier.value
                                             if XK.inverse.value else 1 / price),
                        XK.qty_step.value))


@njit
def calc_initial_entry_qty(balance: float,
                           price: float,
                           available_margin: float,
                           volatility: float) -> float:
    min_entry_qty = calc_min_entry_qty(price)
    if XK.inverse.value:
        qty = round_dn(
            min(available_margin * XK.leverage.value * price / XK.contract_multiplier.value,
                max(min_entry_qty,
                    ((balance / XK.contract_multiplier.value) * price * XK.leverage.value *
                      XK.qty_pct.value * (1 + volatility * XK.volatility_qty_coeff.value)))),
            XK.qty_step.value
        )
    else:
        qty = round_dn(
            min(available_margin * XK.leverage.value / price,
                max(min_entry_qty,
                    ((balance / price) * XK.leverage.value * XK.qty_pct.value *
                      (1 + volatility * XK.volatility_qty_coeff.value)))),
            XK.qty_step.value
        )
    return qty if qty >= min_entry_qty else 0.0


@njit
def calc_reentry_qty(psize: float, price: float, available_margin: float) -> float:
    min_entry_qty = calc_min_entry_qty(price)
    qty = min(round_dn(available_margin * (price / XK.contract_multiplier.value if XK.inverse.value
                                           else 1 / price), XK.qty_step.value),
              max(min_entry_qty, round_dn(abs(psize) * XK.ddown_factor.value, XK.qty_step.value)))
    return qty if qty >= min_entry_qty else 0.0


@njit
def calc_long_reentry_price(balance: float,
                            psize: float,
                            pprice: float,
                            volatility: float) -> float:
    modifier = (1 + (calc_margin_cost(psize, pprice) / balance) * XK.pos_margin_grid_coeff.value) * \
               (1 + volatility * XK.volatility_grid_coeff.value)
    return round_dn(pprice * (1 - XK.grid_spacing.value * modifier), XK.price_step.value)


@njit
def calc_shrt_reentry_price(balance: float,
                            psize: float,
                            pprice: float,
                            volatility: float) -> float:
    modifier = (1 + (calc_margin_cost(psize, pprice) / balance) * XK.pos_margin_grid_coeff.value) * \
               (1 + volatility * XK.volatility_grid_coeff.value)
    return round_dn(pprice * (1 + XK.grid_spacing.value * modifier), XK.price_step.value)


@njit
def calc_new_psize_pprice(psize: float,
                          pprice: float,
                          qty: float,
                          price: float) -> (float, float):
    if qty == 0.0:
        return psize, pprice
    new_psize = round_(psize + qty, XK.qty_step.value)
    if new_psize == 0.0:
        return 0.0, 0.0
    return new_psize, nan_to_0(pprice) * (psize / new_psize) + price * (qty / new_psize)


@njit
def calc_long_pnl(entry_price: float, close_price: float, qty: float) -> float:
    if XK.inverse.value:
        return abs(qty) * XK.contract_multiplier.value * (1 / entry_price - 1 / close_price)
    else:
        return abs(qty) * (close_price - entry_price)


@njit
def calc_shrt_pnl(entry_price: float, close_price: float, qty: float) -> float:
    if XK.inverse.value:
        return abs(qty) * XK.contract_multiplier.value * (1 / close_price - 1 / entry_price)
    else:
        return abs(qty) * (entry_price - close_price)


@njit
def calc_cost(qty: float, price: float) -> float:
    return abs(qty / price) * XK.contract_multiplier.value if XK.inverse.value else abs(qty * price)


@njit
def calc_margin_cost(qty: float, price: float) -> float:
    return calc_cost(qty, price) / XK.leverage.value


@njit
def calc_available_margin(balance: float,
                          long_psize: float,
                          long_pprice: float,
                          shrt_psize: float,
                          shrt_pprice: float,
                          last_price: float) -> float:
    used_margin = 0.0
    equity = balance
    if long_pprice and long_psize:
        long_psize_real = long_psize * XK.contract_multiplier.value
        equity += calc_long_pnl(long_pprice, last_price, long_psize_real)
        used_margin += calc_cost(long_psize_real, long_pprice) / XK.leverage.value
    if shrt_pprice and shrt_psize:
        shrt_psize_real = shrt_psize * XK.contract_multiplier.value
        equity += calc_shrt_pnl(shrt_pprice, last_price, shrt_psize_real)
        used_margin += calc_cost(shrt_psize_real, shrt_pprice) / XK.leverage.value
    return equity - used_margin


@njit
def iter_entries(balance: float,
                 long_psize: float,
                 long_pprice: float,
                 shrt_psize: float,
                 shrt_pprice: float,
                 liq_price: float,
                 highest_bid: float,
                 lowest_ask: float,
                 ema: float,
                 last_price: float,
                 volatility: float,
                 do_long: bool = None,
                 do_shrt: bool = None,):

    do_long_ = XK.do_long.value if do_long is None else do_long
    do_shrt_ = XK.do_shrt.value if do_shrt is None else do_shrt

    available_margin = calc_available_margin(balance, long_psize, long_pprice,
                                             shrt_psize, shrt_pprice, last_price)
    stop_loss_order = calc_stop_loss(balance, long_psize, long_pprice, shrt_psize, shrt_pprice,
                                     liq_price, highest_bid, lowest_ask, last_price,
                                     available_margin, do_long_, do_shrt_)

    if stop_loss_order[0] != 0.0:
        yield stop_loss_order
        if 'long' in stop_loss_order[4]:
            long_psize, long_pprice = stop_loss_order[2:4]
        elif 'shrt' in stop_loss_order[4]:
            shrt_psize, shrt_pprice = stop_loss_order[2:4]
        available_margin -= calc_margin_cost(stop_loss_order[0], stop_loss_order[1])
    while True:
        if do_long_:
            long_entry = calc_next_long_entry(balance, long_psize, long_pprice,
                                              highest_bid, ema, available_margin, volatility)
        else:
            long_entry = (0.0, 0.0, long_psize, long_pprice, '')

        if do_shrt_:
            shrt_entry = calc_next_shrt_entry(balance, shrt_psize, shrt_pprice,
                                              lowest_ask, ema, available_margin, volatility)
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
                available_margin -= calc_margin_cost(long_entry[0], long_entry[1])
        else:
            yield shrt_entry
            shrt_psize, shrt_pprice = shrt_entry[2:4]
            if shrt_entry[1]:
                available_margin -= calc_margin_cost(shrt_entry[0], shrt_entry[1])


@njit
def calc_stop_loss(balance: float,
                   long_psize: float,
                   long_pprice: float,
                   shrt_psize: float,
                   shrt_pprice: float,
                   liq_price: float,
                   highest_bid: float,
                   lowest_ask: float,
                   last_price: float,
                   available_margin: float,
                   do_long: bool,
                   do_shrt: bool,):
    # returns (qty, price, psize if taken, pprice if taken, comment)
    abs_shrt_psize = abs(shrt_psize)
    if calc_diff(liq_price, last_price) < XK.stop_loss_liq_diff.value:
        if long_psize > abs_shrt_psize:
            stop_loss_qty = min(long_psize, max(calc_min_entry_qty(lowest_ask),
                                                round_dn(long_psize * XK.stop_loss_pos_pct.value,
                                                         XK.qty_step.value)))
            # if sufficient margin available, increase short pos, otherwise reduce long pos
            margin_cost = calc_margin_cost(stop_loss_qty, lowest_ask)
            if margin_cost < available_margin and do_shrt:
                # add to shrt pos
                shrt_psize, shrt_pprice = calc_new_psize_pprice(shrt_psize, shrt_pprice,
                                                                -stop_loss_qty, lowest_ask)
                return -stop_loss_qty, lowest_ask, shrt_psize, shrt_pprice, 'stop_loss_shrt_entry'
            else:
                # reduce long pos
                long_psize = round_(long_psize - stop_loss_qty, XK.qty_step.value)
                return -stop_loss_qty, lowest_ask, long_psize, long_pprice, 'stop_loss_long_close'
        else:
            stop_loss_qty = min(abs_shrt_psize,
                                max(calc_min_entry_qty(highest_bid),
                                    round_dn(abs_shrt_psize * XK.stop_loss_pos_pct.value,
                                             XK.qty_step.value)))
            # if sufficient margin available, increase long pos, otherwise, reduce shrt pos
            margin_cost = calc_margin_cost(stop_loss_qty, highest_bid)
            if margin_cost < available_margin and do_long:
                # add to long pos
                long_psize, long_pprice = calc_new_psize_pprice(long_psize, long_pprice,
                                                                stop_loss_qty, highest_bid)
                return stop_loss_qty, highest_bid, long_psize, long_pprice, 'stop_loss_long_entry'
            else:
                # reduce shrt pos
                shrt_psize = round_(shrt_psize + stop_loss_qty, XK.qty_step.value)
                return stop_loss_qty, highest_bid, shrt_psize, shrt_pprice, 'stop_loss_shrt_close'
    return 0.0, 0.0, 0.0, 0.0, ''


@njit
def calc_next_long_entry(balance: float,
                         psize: float,
                         pprice: float,
                         highest_bid: float,
                         ema: float,
                         available_margin: float,
                         volatility: float) -> (float, float, float, float, str):
    if psize == 0.0:
        price = calc_initial_long_entry_price(ema, highest_bid)
        qty = calc_initial_entry_qty(balance, price, available_margin, volatility)
        return qty, price, qty, price, 'initial_long_entry'
    else:
        price = min(round_(highest_bid, XK.price_step.value),
                    calc_long_reentry_price(balance, psize, pprice, volatility))
        if price <= 0.0:
            return 0.0, 0.0, psize, pprice, 'long_reentry'
        qty = calc_reentry_qty(psize, price, available_margin)
        psize, pprice = calc_new_psize_pprice(psize, pprice, qty, price)
        return qty, price, psize, pprice, 'long_reentry'


@njit
def calc_next_shrt_entry(balance: float,
                         psize: float,
                         pprice: float,
                         lowest_ask: float,
                         ema: float,
                         available_margin: float,
                         volatility: float) -> (float, float, float, float, str):
    if psize == 0.0:
        price = calc_initial_shrt_entry_price(ema, lowest_ask)
        qty = -calc_initial_entry_qty(balance, price, available_margin, volatility)
        return qty, price, qty, price, 'initial_shrt_entry'
    else:
        price = max(round_(lowest_ask, XK.price_step.value),
                    calc_shrt_reentry_price(balance, psize, pprice, volatility))
        qty = -calc_reentry_qty(psize, price, available_margin)
        psize, pprice = calc_new_psize_pprice(psize, pprice, qty, price)
        return qty, price, psize, pprice, 'shrt_reentry'


@njit
def iter_long_closes(balance: float, psize: float, pprice: float, lowest_ask: float):
    # yields (qty, price, psize_if_taken)
    if psize == 0.0 or pprice == 0.0:
        return
    minm = pprice * (1 + XK.min_markup.value)
    prices = np.linspace(minm, pprice * (1 + XK.min_markup.value + XK.markup_range.value),
                         int(XK.n_close_orders.value))
    prices = [p for p in sorted(set([round_up(p_, XK.price_step.value)
                                     for p_ in prices])) if p >= lowest_ask]
    if len(prices) == 0:
        yield -psize, max(lowest_ask, round_up(minm, XK.price_step.value)), 0.0
    else:
        n_orders = int(min([XK.n_close_orders.value, len(prices), int(psize / XK.min_qty.value)]))
        for price in prices:
            if n_orders == 0:
                break
            else:
                qty = min(psize, max(calc_initial_entry_qty(balance, lowest_ask, balance, 0.0),
                                     round_up(psize / n_orders, XK.qty_step.value)))
                if psize != 0.0 and qty / psize > 0.75:
                    qty = psize
            if qty == 0.0:
                break
            psize = round_(psize - qty, XK.qty_step.value)
            yield -qty, price, psize
            lowest_ask = price
            n_orders -= 1
        if psize > 0.0:
            yield -psize, max(lowest_ask, round_up(minm, XK.price_step.value)), 0.0


@njit
def iter_shrt_closes(balance: float, psize: float, pprice: float, highest_bid: float):
    # yields (qty, price, psize_if_taken)
    abs_psize = abs(psize)
    if psize == 0.0:
        return
    minm = pprice * (1 - XK.min_markup.value)
    prices = np.linspace(minm, pprice * (1 - (XK.min_markup.value + XK.markup_range.value)),
                         int(XK.n_close_orders.value))
    prices = [p for p in sorted(set([round_dn(p_, XK.price_step.value)
                                     for p_ in prices]), reverse=True) if p <= highest_bid]
    if len(prices) == 0:
        yield abs_psize, min(highest_bid, round_dn(minm, XK.price_step.value)), 0.0
    else:
        n_orders = int(min([XK.n_close_orders.value, len(prices),
                            int(abs_psize / XK.min_qty.value)]))
        for price in prices:
            if n_orders == 0:
                break
            else:
                qty = min(abs_psize, max(calc_initial_entry_qty(balance, highest_bid, balance, 0.0),
                                         round_up(abs_psize / n_orders, XK.qty_step.value)))
                if abs_psize != 0.0 and qty / abs_psize > 0.75:
                    qty = abs_psize
            if qty == 0.0:
                break
            abs_psize = round_(abs_psize - qty, XK.qty_step.value)
            yield qty, price, abs_psize
            highest_bid = price
            n_orders -= 1
        if abs_psize > 0.0:
            yield abs_psize, min(highest_bid, round_dn(minm, XK.price_step.value)), 0.0


@njit
def calc_liq_price_binance(balance: float,
                           long_psize: float,
                           long_pprice: float,
                           shrt_psize: float,
                           shrt_pprice: float):
    abs_long_psize = abs(long_psize)
    abs_shrt_psize = abs(shrt_psize)
    long_pprice = nan_to_0(long_pprice)
    shrt_pprice = nan_to_0(shrt_pprice)
    if XK.inverse.value:
        mml = 0.02
        mms = 0.02
        numerator = abs_long_psize * mml + abs_shrt_psize * mms + abs_long_psize - abs_shrt_psize
        long_pcost = abs_long_psize / long_pprice if long_pprice > 0.0 else 0.0
        shrt_pcost = abs_shrt_psize / shrt_pprice if shrt_pprice > 0.0 else 0.0
        denom = balance / XK.contract_multiplier.value + long_pcost - shrt_pcost
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
def calc_liq_price_bybit(balance: float,
                         long_psize: float,
                         long_pprice: float,
                         shrt_psize: float,
                         shrt_pprice: float):
    mm = 0.005
    abs_shrt_psize = abs(shrt_psize)
    if XK.inverse.value:
        if long_psize > abs_shrt_psize:
            long_pprice = nan_to_0(long_pprice)
            order_cost = long_psize / long_pprice if long_pprice > 0.0 else 0.0
            order_margin = order_cost / XK.leverage.value
            bankruptcy_price = (1.00075 * long_psize) / (order_cost + (balance - order_margin))
            if bankruptcy_price == 0.0:
                return 0.0
            rhs = -(balance - order_margin - (long_psize / long_pprice) * mm -
                    (long_psize * 0.00075) / bankruptcy_price)
            return max(0.0, (long_pprice * long_psize) / (long_psize - long_pprice * rhs))
        else:
            shrt_pprice = nan_to_0(shrt_pprice)
            order_cost = abs_shrt_psize / shrt_pprice if shrt_pprice > 0.0 else 0.0
            order_margin = order_cost / XK.leverage.value
            bankruptcy_price = (0.99925 * abs_shrt_psize) / (order_cost - (balance - order_margin))
            if bankruptcy_price == 0.0:
                return 0.0
            rhs = -(balance - order_margin - (abs_shrt_psize / shrt_pprice) * mm -
                    (abs_shrt_psize * 0.00075) / bankruptcy_price)
            return max(0.0, (shrt_pprice * abs_shrt_psize) / (shrt_pprice * rhs + abs_shrt_psize))
    else:
        raise Exception('bybit linear liq price not implemented')
