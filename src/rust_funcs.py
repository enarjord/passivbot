import passivbot_rust as pbr


def calc_entries_long(
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    entry_grid_double_down_factor,
    entry_grid_spacing_weight,
    entry_grid_spacing_pct,
    entry_initial_ema_dist,
    entry_initial_qty_pct,
    entry_trailing_retracement_pct,
    entry_trailing_grid_ratio,
    entry_trailing_threshold_pct,
    wallet_exposure_limit,
    balance,
    position_size,
    position_price,
    min_price_since_open,
    max_price_since_min,
    ema_bands_lower,
    order_book_bid,
    whole_grid=False,
):
    entries = []
    psize = position_size
    pprice = position_price
    bid = order_book_bid
    for _ in range(500):
        entry = pbr.calc_next_entry_long_py(
            qty_step,
            price_step,
            min_qty,
            min_cost,
            c_mult,
            entry_grid_double_down_factor,
            entry_grid_spacing_weight,
            entry_grid_spacing_pct,
            entry_initial_ema_dist,
            entry_initial_qty_pct,
            entry_trailing_retracement_pct,
            entry_trailing_grid_ratio,
            entry_trailing_threshold_pct,
            wallet_exposure_limit,
            balance,
            psize,
            pprice,
            min_price_since_open,
            max_price_since_min,
            ema_bands_lower,
            bid,
        )
        if entry[0] == 0.0:
            break
        if entries:
            if "trailing" in entry[2]:
                break
            if entries[-1][1] == entry[1]:
                break
        psize, pprice = pbr.calc_new_psize_pprice(psize, pprice, entry[0], entry[1], qty_step)
        bid = min(bid, entry[1])
        entries.append(entry + (psize, pprice))
        if "initial" in entry[2] and not whole_grid:
            break
    return entries


def calc_closes_long(
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    close_grid_markup_range,
    close_grid_min_markup,
    close_grid_qty_pct,
    close_trailing_retracement_pct,
    close_trailing_grid_ratio,
    close_trailing_threshold_pct,
    wallet_exposure_limit,
    balance,
    position_size,
    position_price,
    max_price_since_open,
    min_price_since_max,
    order_book_ask,
):
    closes = []
    psize = position_size
    ask = order_book_ask
    for _ in range(500):
        close = pbr.calc_next_close_long_py(
            qty_step,
            price_step,
            min_qty,
            min_cost,
            c_mult,
            close_grid_markup_range,
            close_grid_min_markup,
            close_grid_qty_pct,
            close_trailing_retracement_pct,
            close_trailing_grid_ratio,
            close_trailing_threshold_pct,
            wallet_exposure_limit,
            balance,
            psize,
            position_price,
            max_price_since_open,
            min_price_since_max,
            ask,
        )
        if close[0] == 0.0:
            break
        if closes and "trailing" in close[2]:
            break
        psize = pbr.round_(psize + close[0], qty_step)
        ask = max(ask, close[1])
        if closes:
            if "trailing" in close[2]:
                break
            if closes[-1][1] == close[1]:
                closes = closes[:-1] + [(closes[-1][0] + close[0], close[1], close[2], psize, position_price)]
                continue
        closes.append(close + (psize, position_price))
    return closes


def calc_entries_short(
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    entry_grid_double_down_factor,
    entry_grid_spacing_weight,
    entry_grid_spacing_pct,
    entry_initial_ema_dist,
    entry_initial_qty_pct,
    entry_trailing_retracement_pct,
    entry_trailing_grid_ratio,
    entry_trailing_threshold_pct,
    wallet_exposure_limit,
    balance,
    position_size,
    position_price,
    max_price_since_open,
    min_price_since_max,
    ema_bands_upper,
    order_book_ask,
    whole_grid=False,
):
    entries = []
    psize = position_size
    pprice = position_price
    ask = order_book_ask
    for _ in range(500):
        entry = pbr.calc_next_entry_short_py(
            qty_step,
            price_step,
            min_qty,
            min_cost,
            c_mult,
            entry_grid_double_down_factor,
            entry_grid_spacing_weight,
            entry_grid_spacing_pct,
            entry_initial_ema_dist,
            entry_initial_qty_pct,
            entry_trailing_retracement_pct,
            entry_trailing_grid_ratio,
            entry_trailing_threshold_pct,
            wallet_exposure_limit,
            balance,
            psize,
            pprice,
            max_price_since_open,
            min_price_since_max,
            ema_bands_upper,
            ask,
        )
        if entry[0] == 0.0:
            break
        if entries:
            if "trailing" in entry[2]:
                break
            if entries[-1][1] == entry[1]:
                break
        psize, pprice = pbr.calc_new_psize_pprice(psize, pprice, entry[0], entry[1], qty_step)
        ask = max(ask, entry[1])
        entries.append(entry + (psize, pprice))
        if "initial" in entry[2] and not whole_grid:
            break
    return entries


def calc_closes_short(
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    close_grid_markup_range,
    close_grid_min_markup,
    close_grid_qty_pct,
    close_trailing_retracement_pct,
    close_trailing_grid_ratio,
    close_trailing_threshold_pct,
    wallet_exposure_limit,
    balance,
    position_size,
    position_price,
    min_price_since_open,
    max_price_since_min,
    order_book_bid,
):
    closes = []
    psize = position_size
    bid = order_book_bid
    for _ in range(500):
        close = pbr.calc_next_close_short_py(
            qty_step,
            price_step,
            min_qty,
            min_cost,
            c_mult,
            close_grid_markup_range,
            close_grid_min_markup,
            close_grid_qty_pct,
            close_trailing_retracement_pct,
            close_trailing_grid_ratio,
            close_trailing_threshold_pct,
            wallet_exposure_limit,
            balance,
            psize,
            position_price,
            min_price_since_open,
            max_price_since_min,
            order_book_bid,
        )
        if close[0] == 0.0:
            break
        psize = pbr.round_(psize + close[0], qty_step)
        bid = min(bid, close[1])
        if closes:
            if "trailing" in close[2]:
                break
            if closes[-1][1] == close[1]:
                closes = closes[:-1] + [(closes[-1][0] + close[0], close[1], close[2], psize, position_price)]
                continue
        closes.append(close + (psize, position_price))
    return closes
