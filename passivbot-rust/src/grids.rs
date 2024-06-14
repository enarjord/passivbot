use crate::trailing::{calc_trailing_close_long, calc_trailing_entry_long};
use crate::types::{
    BotParams, EMABands, ExchangeParams, Order, OrderBook, OrderType, Position, StateParams,
};
use crate::utils::{
    calc_min_entry_qty, calc_new_psize_pprice, calc_wallet_exposure,
    calc_wallet_exposure_if_filled, cost_to_qty, interpolate, qty_to_cost, round_, round_dn,
    round_up,
};

fn calc_initial_entry(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    position: &Position,
) -> Order {
    let initial_entry_price = f64::min(
        state_params.order_book.bid,
        round_dn(
            state_params.ema_bands.lower * (1.0 - bot_params.entry_initial_ema_dist),
            exchange_params.price_step,
        ),
    );
    if initial_entry_price <= exchange_params.price_step {
        return Order::default();
    }
    let min_entry_qty = calc_min_entry_qty(initial_entry_price, &exchange_params);
    let initial_entry_qty = f64::max(
        min_entry_qty,
        round_(
            cost_to_qty(
                state_params.balance,
                initial_entry_price,
                exchange_params.c_mult,
            ) * bot_params.wallet_exposure_limit
                * bot_params.entry_initial_qty_pct,
            exchange_params.qty_step,
        ),
    );

    if position.size == 0.0 {
        // normal initial entry
        Order {
            qty: initial_entry_qty,
            price: initial_entry_price,
            order_type: OrderType::EntryInitialNormalLong,
        }
    } else if position.size < initial_entry_qty * 0.8 {
        // partial initial entry
        let entry_qty = f64::max(
            min_entry_qty,
            round_dn(initial_entry_qty - position.size, exchange_params.qty_step),
        );
        Order {
            qty: entry_qty,
            price: initial_entry_price,
            order_type: OrderType::EntryInitialPartialLong,
        }
    } else {
        Order::default()
    }
}

pub fn calc_next_grid_close_long(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    position: &Position,
) -> Order {
    if position.size <= 0.0 {
        return Order::default();
    }
    if bot_params.close_grid_markup_range <= 0.0 || bot_params.close_grid_qty_pct <= 0.0 {
        return Order {
            qty: -position.size,
            price: f64::max(
                state_params.order_book.ask,
                round_up(
                    position.price * (1.0 + bot_params.close_grid_min_markup),
                    exchange_params.price_step,
                ),
            ),
            order_type: OrderType::CloseGridLong,
        };
    }

    let wallet_exposure = calc_wallet_exposure(
        exchange_params.c_mult,
        state_params.balance,
        position.size,
        position.price,
    );
    let wallet_exposure_ratio = f64::min(1.0, wallet_exposure / bot_params.wallet_exposure_limit);
    let close_price = position.price
        * (1.0
            + bot_params.close_grid_min_markup
            + bot_params.close_grid_markup_range * (1.0 - wallet_exposure_ratio));
    let close_price = round_up(close_price, exchange_params.price_step);
    let full_psize = cost_to_qty(
        state_params.balance * bot_params.wallet_exposure_limit,
        position.price,
        exchange_params.c_mult,
    );
    let leftover = f64::max(0.0, position.size - full_psize);
    let close_qty = f64::max(
        calc_min_entry_qty(close_price, &exchange_params),
        round_up(
            full_psize * bot_params.close_grid_qty_pct + leftover,
            exchange_params.qty_step,
        ),
    );
    let close_qty = -f64::min(position.size, close_qty);
    Order {
        qty: close_qty,
        price: close_price,
        order_type: OrderType::CloseGridLong,
    }
}

pub fn calc_next_grid_entry_long(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    position: &Position,
) -> Order {
    if bot_params.wallet_exposure_limit == 0.0 || state_params.balance <= 0.0 {
        return Order::default();
    }

    let initial_entry = calc_initial_entry(exchange_params, state_params, bot_params, position);
    if initial_entry.qty > 0.0 {
        return initial_entry;
    }

    let wallet_exposure = calc_wallet_exposure(
        exchange_params.c_mult,
        state_params.balance,
        position.size,
        position.price,
    );
    if wallet_exposure >= bot_params.wallet_exposure_limit * 0.999 {
        return Order::default();
    }

    // normal re-entry
    let (reentry_order, wallet_exposure_if_filled, is_cropped) = calc_reentry_order(
        &exchange_params,
        &bot_params,
        &position,
        &state_params,
        &initial_entry,
    );
    if is_cropped {
        return reentry_order;
    }
    // preview next reentry
    let (next_psize, next_pprice) = calc_new_psize_pprice(
        position.size,
        position.price,
        reentry_order.qty,
        reentry_order.price,
        exchange_params.qty_step,
    );
    let (next_reentry_order, next_wallet_exposure_if_filled, next_is_cropped) = calc_reentry_order(
        &exchange_params,
        &bot_params,
        &Position {
            size: next_psize,
            price: next_pprice,
        },
        &state_params,
        &initial_entry,
    );
    if !next_is_cropped {
        return reentry_order;
    }
    let effective_double_down_factor = next_reentry_order.qty / next_psize;
    if effective_double_down_factor < bot_params.entry_grid_double_down_factor * 0.25 {
        // next reentry too small. Inflate current reentry.
        let new_entry_qty = interpolate(
            bot_params.wallet_exposure_limit,
            &[wallet_exposure, wallet_exposure_if_filled],
            &[position.size, position.size + reentry_order.qty],
        ) - position.size;
        Order {
            qty: round_(new_entry_qty, exchange_params.qty_step),
            price: reentry_order.price,
            order_type: OrderType::EntryGridInflatedLong,
        }
    } else {
        Order {
            qty: reentry_order.qty,
            price: reentry_order.price,
            order_type: OrderType::EntryGridNormalLong,
        }
    }
}

pub fn calc_reentry_qty(
    entry_price: f64,
    balance: f64,
    position_size: f64,
    exchange_params: &ExchangeParams,
    bot_params: &BotParams,
) -> f64 {
    f64::max(
        calc_min_entry_qty(entry_price, &exchange_params),
        round_(
            f64::max(
                position_size * bot_params.entry_grid_double_down_factor,
                cost_to_qty(balance, entry_price, exchange_params.c_mult)
                    * bot_params.wallet_exposure_limit
                    * bot_params.entry_initial_qty_pct,
            ),
            exchange_params.qty_step,
        ),
    )
}

fn calc_reentry_price(
    balance: f64,
    position_price: f64,
    wallet_exposure: f64,
    order_book_bid: f64,
    exchange_params: &ExchangeParams,
    bot_params: &BotParams,
) -> f64 {
    let multiplier =
        (wallet_exposure / bot_params.wallet_exposure_limit) * bot_params.entry_grid_spacing_weight;
    let entry_price = round_dn(
        position_price * (1.0 - bot_params.entry_grid_spacing_pct * (1.0 + multiplier)),
        exchange_params.price_step,
    );
    let entry_price = f64::min(order_book_bid, entry_price);
    if entry_price <= exchange_params.price_step {
        0.0
    } else {
        entry_price
    }
}

fn calc_reentry_order(
    exchange_params: &ExchangeParams,
    bot_params: &BotParams,
    position: &Position,
    state_params: &StateParams,
    initial_entry: &Order,
) -> (Order, f64, bool) {
    // returns (Order, wallet_exposure_if_filled, is_cropped)
    let wallet_exposure = calc_wallet_exposure(
        exchange_params.c_mult,
        state_params.balance,
        position.size,
        position.price,
    );
    if wallet_exposure >= bot_params.wallet_exposure_limit * 0.999 {
        return (Order::default(), 0.0, false);
    }

    let entry_price = calc_reentry_price(
        state_params.balance,
        position.price,
        wallet_exposure,
        state_params.order_book.bid,
        &exchange_params,
        &bot_params,
    );
    if entry_price == 0.0 {
        return (
            Order {
                qty: 0.0,
                price: 0.0,
                order_type: OrderType::Empty,
            },
            0.0,
            false,
        );
    }
    let entry_qty = f64::max(
        calc_reentry_qty(
            entry_price,
            state_params.balance,
            position.size,
            &exchange_params,
            &bot_params,
        ),
        initial_entry.qty,
    );
    let wallet_exposure_if_filled = calc_wallet_exposure_if_filled(
        state_params.balance,
        position.size,
        position.price,
        entry_qty,
        entry_price,
        &exchange_params,
    );
    if wallet_exposure_if_filled > bot_params.wallet_exposure_limit * 1.01 {
        // reentry too big. Crop current reentry qty.
        let entry_qty = interpolate(
            bot_params.wallet_exposure_limit,
            &[wallet_exposure, wallet_exposure_if_filled],
            &[position.size, position.size + entry_qty],
        ) - position.size;
        (
            Order {
                qty: round_(entry_qty, exchange_params.qty_step),
                price: entry_price,
                order_type: OrderType::EntryGridCroppedLong,
            },
            wallet_exposure_if_filled,
            true,
        )
    } else {
        (
            Order {
                qty: entry_qty,
                price: entry_price,
                order_type: OrderType::EntryGridNormalLong,
            },
            wallet_exposure_if_filled,
            false,
        )
    }
}

pub fn calc_next_close_long(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    position: &Position,
    max_price_since_open: f64,
    min_price_since_max: f64,
) -> Order {
    if position.size == 0.0 {
        // no position
        return Order::default();
    }
    let wallet_exposure = calc_wallet_exposure(
        exchange_params.c_mult,
        state_params.balance,
        position.size,
        position.price,
    );
    if bot_params.close_trailing_grid_ratio >= 1.0 || bot_params.close_trailing_grid_ratio <= -1.0 {
        // return trailing only
        return calc_trailing_close_long(
            &exchange_params,
            &state_params,
            &bot_params,
            &position,
            max_price_since_open,
            min_price_since_max,
        );
    }
    let wallet_exposure_ratio = wallet_exposure / bot_params.wallet_exposure_limit;
    if bot_params.close_trailing_grid_ratio > 0.0 {
        // trailing first
        if wallet_exposure_ratio < bot_params.close_trailing_grid_ratio {
            // return trailing order, but crop to max bot_params.wallet_exposure_limit * bot_params.close_trailing_grid_ratio + 1%
            let mut bot_params_modified = bot_params.clone();
            bot_params_modified.wallet_exposure_limit =
                bot_params.wallet_exposure_limit * bot_params.close_trailing_grid_ratio * 1.01;
            return calc_trailing_close_long(
                &exchange_params,
                &state_params,
                &bot_params_modified,
                &position,
                max_price_since_open,
                min_price_since_max,
            );
        } else {
            // return grid order
            return calc_next_grid_close_long(
                &exchange_params,
                &state_params,
                &bot_params,
                &position,
            );
        }
    }
    if bot_params.close_trailing_grid_ratio < 0.0 {
        // grid first
        if wallet_exposure_ratio < 1.0 + bot_params.close_trailing_grid_ratio {
            // return grid order, but crop to max bot_params.wallet_exposure_limit * (1.0 + bot_params.close_trailing_grid_ratio) + 1%
            let mut bot_params_modified = bot_params.clone();
            bot_params_modified.wallet_exposure_limit = bot_params.wallet_exposure_limit
                * (1.0 + bot_params.close_trailing_grid_ratio)
                * 1.01;
            return calc_next_grid_close_long(
                &exchange_params,
                &state_params,
                &bot_params_modified,
                &position,
            );
        } else {
            return calc_trailing_close_long(
                &exchange_params,
                &state_params,
                &bot_params,
                &position,
                max_price_since_open,
                min_price_since_max,
            );
        }
    }
    // return grid only
    calc_next_grid_close_long(&exchange_params, &state_params, &bot_params, &position)
}

pub fn calc_next_entry_long(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    position: &Position,
    min_price_since_open: f64,
    max_price_since_min: f64,
) -> Order {
    // determines whether trailing or grid order, returns Order
    if bot_params.wallet_exposure_limit == 0.0 || state_params.balance <= 0.0 {
        // no orders
        return Order::default();
    }
    let wallet_exposure = calc_wallet_exposure(
        exchange_params.c_mult,
        state_params.balance,
        position.size,
        position.price,
    );
    if wallet_exposure >= bot_params.wallet_exposure_limit * 0.999 {
        // wallet exposure exceeded; return empty order
        return Order::default();
    }
    let initial_entry = calc_initial_entry(exchange_params, state_params, bot_params, position);
    if initial_entry.qty > 0.0 {
        // initial entry
        return initial_entry;
    }
    if bot_params.entry_trailing_grid_ratio >= 1.0 || bot_params.entry_trailing_grid_ratio <= -1.0 {
        // return trailing only
        return calc_trailing_entry_long(
            &exchange_params,
            &state_params,
            &bot_params,
            &position,
            min_price_since_open,
            max_price_since_min,
        );
    }
    let wallet_exposure_ratio = wallet_exposure / bot_params.wallet_exposure_limit;
    if bot_params.entry_trailing_grid_ratio > 0.0 {
        // trailing first
        if wallet_exposure_ratio < bot_params.entry_trailing_grid_ratio {
            // return trailing order, but crop to max bot_params.wallet_exposure_limit * bot_params.entry_trailing_grid_ratio + 1%
            let mut bot_params_modified = bot_params.clone();
            bot_params_modified.wallet_exposure_limit =
                bot_params.wallet_exposure_limit * bot_params.entry_trailing_grid_ratio * 1.01;
            return calc_trailing_entry_long(
                &exchange_params,
                &state_params,
                &bot_params_modified,
                &position,
                min_price_since_open,
                max_price_since_min,
            );
        } else {
            // return grid order
            return calc_next_grid_entry_long(
                &exchange_params,
                &state_params,
                &bot_params,
                &position,
            );
        }
    }
    if bot_params.entry_trailing_grid_ratio < 0.0 {
        // grid first
        if wallet_exposure_ratio < 1.0 + bot_params.entry_trailing_grid_ratio {
            // return grid order, but crop to max bot_params.wallet_exposure_limit * (1.0 + bot_params.entry_trailing_grid_ratio) + 1%
            let mut bot_params_modified = bot_params.clone();
            bot_params_modified.wallet_exposure_limit = bot_params.wallet_exposure_limit
                * (1.0 + bot_params.entry_trailing_grid_ratio)
                * 1.01;
            return calc_next_grid_entry_long(
                &exchange_params,
                &state_params,
                &bot_params_modified,
                &position,
            );
        } else {
            return calc_trailing_entry_long(
                &exchange_params,
                &state_params,
                &bot_params,
                &position,
                min_price_since_open,
                max_price_since_min,
            );
        }
    }
    // return grid only
    calc_next_grid_entry_long(&exchange_params, &state_params, &bot_params, &position)
}
