use crate::entries::{calc_min_entry_qty, wallet_exposure_limit_with_allowance};
use crate::types::{
    BotParams, ExchangeParams, Order, OrderType, Position, StateParams, TrailingPriceBundle,
};
use crate::utils::{
    calc_wallet_exposure, cost_to_qty, quantize_price, quantize_qty, round_, round_dn, round_up,
    RoundingMode,
};

pub fn calc_close_qty(
    exchange_params: &ExchangeParams,
    bot_params: &BotParams,
    position: &Position,
    close_qty_pct: f64,
    balance: f64,
    close_price: f64,
) -> f64 {
    let full_psize = cost_to_qty(
        balance * wallet_exposure_limit_with_allowance(bot_params),
        position.price,
        exchange_params.c_mult,
    );
    let position_size_abs = position.size.abs();
    let leftover = f64::max(0.0, position_size_abs - full_psize);
    let min_entry_qty = calc_min_entry_qty(close_price, &exchange_params);
    let close_qty = f64::min(
        round_(position_size_abs, exchange_params.qty_step),
        f64::max(
            min_entry_qty,
            round_up(
                full_psize * close_qty_pct + leftover,
                exchange_params.qty_step,
            ),
        ),
    );
    if close_qty > 0.0
        && close_qty < position_size_abs
        && position_size_abs - close_qty < min_entry_qty
    {
        position_size_abs
    } else {
        close_qty
    }
}

fn calc_wel_auto_reduce_long(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    position: &Position,
    wallet_exposure: f64,
) -> Option<Order> {
    if bot_params.risk_wel_enforcer_threshold <= 0.0 {
        return None;
    }
    if state_params.balance <= 0.0 || position.price <= 0.0 {
        return None;
    }
    let allowed_limit = wallet_exposure_limit_with_allowance(bot_params);
    if allowed_limit <= 0.0 {
        return None;
    }
    // Strict target: must enforce WE < target_exposure
    let target_exposure = allowed_limit * bot_params.risk_wel_enforcer_threshold;
    if wallet_exposure <= target_exposure {
        return None;
    }
    let position_size_abs = position.size.abs();
    if position_size_abs <= f64::EPSILON {
        return None;
    }
    // Compute strict reduction: ensure WE(new) < target_exposure after rounding/min constraints
    let target_psize_strict =
        (target_exposure * state_params.balance) / (position.price * exchange_params.c_mult);
    let market_price = if state_params.order_book.ask > 0.0 {
        state_params.order_book.ask
    } else {
        position.price
    };
    if market_price <= 0.0 {
        return None;
    }
    let min_qty = calc_min_entry_qty(market_price, exchange_params);
    // Iteratively increase reduction until resulting WE is strictly below target or pos is fully closed
    let mut close_qty;
    let mut steps = 0usize;
    let max_steps = 10_000usize;
    let mut reduce_qty = (position_size_abs - target_psize_strict).max(0.0);
    if reduce_qty <= f64::EPSILON {
        // Already at or below target size: emit dust if exposure still over due to rounding
        reduce_qty = exchange_params.qty_step;
    }
    loop {
        let rq = round_up(reduce_qty, exchange_params.qty_step);
        close_qty = f64::min(
            position_size_abs,
            f64::max(min_qty, rq.min(position_size_abs)),
        );
        if close_qty <= f64::EPSILON {
            return None;
        }
        let new_abs_psize = (position_size_abs - close_qty).max(0.0);
        let new_exposure = calc_wallet_exposure(
            exchange_params.c_mult,
            state_params.balance,
            new_abs_psize,
            position.price,
        );
        if new_exposure < target_exposure - 1e-12 {
            break;
        }
        if new_abs_psize <= f64::EPSILON {
            break;
        }
        reduce_qty += exchange_params.qty_step;
        steps += 1;
        if steps > max_steps {
            // safeguard: break to avoid infinite loop
            break;
        }
    }
    Some(Order {
        qty: -close_qty,
        price: market_price,
        order_type: OrderType::CloseAutoReduceWelLong,
    })
}

fn calc_wel_auto_reduce_short(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    position: &Position,
    wallet_exposure: f64,
) -> Option<Order> {
    if bot_params.risk_wel_enforcer_threshold <= 0.0 {
        return None;
    }
    if state_params.balance <= 0.0 || position.price <= 0.0 {
        return None;
    }
    let allowed_limit = wallet_exposure_limit_with_allowance(bot_params);
    if allowed_limit <= 0.0 {
        return None;
    }
    // Strict target: must enforce WE < target_exposure
    let target_exposure = allowed_limit * bot_params.risk_wel_enforcer_threshold;
    if wallet_exposure <= target_exposure {
        return None;
    }
    let position_size_abs = position.size.abs();
    if position_size_abs <= f64::EPSILON {
        return None;
    }
    // Compute strict reduction: ensure WE(new) < target_exposure after rounding/min constraints
    let target_psize_strict =
        (target_exposure * state_params.balance) / (position.price * exchange_params.c_mult);
    let market_price = if state_params.order_book.bid > 0.0 {
        state_params.order_book.bid
    } else {
        position.price
    };
    if market_price <= 0.0 {
        return None;
    }
    let min_qty = calc_min_entry_qty(market_price, exchange_params);
    // Iteratively increase reduction until resulting WE is strictly below target or pos is fully closed
    let mut close_qty;
    let mut steps = 0usize;
    let max_steps = 10_000usize;
    let mut reduce_qty = (position_size_abs - target_psize_strict).max(0.0);
    if reduce_qty <= f64::EPSILON {
        reduce_qty = exchange_params.qty_step;
    }
    loop {
        let rq = round_up(reduce_qty, exchange_params.qty_step);
        close_qty = f64::min(
            position_size_abs,
            f64::max(min_qty, rq.min(position_size_abs)),
        );
        if close_qty <= f64::EPSILON {
            return None;
        }
        let new_abs_psize = (position_size_abs - close_qty).max(0.0);
        let new_exposure = calc_wallet_exposure(
            exchange_params.c_mult,
            state_params.balance,
            new_abs_psize,
            position.price,
        );
        if new_exposure < target_exposure - 1e-12 {
            break;
        }
        if new_abs_psize <= f64::EPSILON {
            break;
        }
        reduce_qty += exchange_params.qty_step;
        steps += 1;
        if steps > max_steps {
            break;
        }
    }
    Some(Order {
        qty: close_qty,
        price: market_price,
        order_type: OrderType::CloseAutoReduceWelShort,
    })
}

pub fn calc_grid_close_long(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    position: &Position,
) -> Order {
    if position.size <= 0.0 {
        return Order::default();
    }
    if bot_params.close_grid_qty_pct < 0.0 || bot_params.close_grid_qty_pct >= 1.0 {
        return Order {
            qty: -round_(position.size, exchange_params.qty_step),
            price: f64::max(
                state_params.order_book.ask,
                round_up(
                    position.price * (1.0 + bot_params.close_grid_markup_start),
                    exchange_params.price_step,
                ),
            ),
            order_type: OrderType::CloseGridLong,
        };
    }
    let close_prices_start = round_up(
        position.price * (1.0 + bot_params.close_grid_markup_start),
        exchange_params.price_step,
    );
    let close_prices_end = round_up(
        position.price * (1.0 + bot_params.close_grid_markup_end),
        exchange_params.price_step,
    );
    if close_prices_start == close_prices_end {
        return Order {
            qty: -round_(position.size, exchange_params.qty_step),
            price: f64::max(state_params.order_book.ask, close_prices_start),
            order_type: OrderType::CloseGridLong,
        };
    }
    let n_steps =
        ((close_prices_end - close_prices_start).abs() / exchange_params.price_step).ceil();
    let close_grid_qty_pct_modified = f64::max(bot_params.close_grid_qty_pct, 1.0 / n_steps);
    let wallet_exposure = calc_wallet_exposure(
        exchange_params.c_mult,
        state_params.balance,
        position.size,
        position.price,
    );
    let wallet_exposure_ratio = wallet_exposure / wallet_exposure_limit_with_allowance(bot_params);
    let close_price = if wallet_exposure_ratio > 1.0 {
        f64::max(
            state_params.order_book.ask,
            f64::min(close_prices_start, close_prices_end),
        )
    } else {
        f64::max(
            state_params.order_book.ask,
            round_up(
                close_prices_start
                    + (close_prices_end - close_prices_start)
                        * f64::min(1.0, wallet_exposure_ratio),
                exchange_params.price_step,
            ),
        )
    };
    let close_qty = -calc_close_qty(
        &exchange_params,
        &bot_params,
        &position,
        close_grid_qty_pct_modified,
        state_params.balance,
        close_price,
    );
    Order {
        qty: close_qty,
        price: close_price,
        order_type: OrderType::CloseGridLong,
    }
}

pub fn calc_trailing_close_long(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    position: &Position,
    trailing_price_bundle: &TrailingPriceBundle,
) -> Order {
    if position.size == 0.0 {
        return Order::default();
    }
    if bot_params.close_trailing_threshold_pct <= 0.0 {
        // means trailing close immediately from pos open
        if bot_params.close_trailing_retracement_pct > 0.0
            && trailing_price_bundle.min_since_max
                < trailing_price_bundle.max_since_open
                    * (1.0 - bot_params.close_trailing_retracement_pct)
        {
            Order {
                qty: -calc_close_qty(
                    &exchange_params,
                    &bot_params,
                    &position,
                    bot_params.close_trailing_qty_pct,
                    state_params.balance,
                    state_params.order_book.ask,
                ),
                price: state_params.order_book.ask,
                order_type: OrderType::CloseTrailingLong,
            }
        } else {
            Order {
                qty: 0.0,
                price: 0.0,
                order_type: OrderType::CloseTrailingLong,
            }
        }
    } else {
        // means trailing close will activate only after a threshold
        if bot_params.close_trailing_retracement_pct <= 0.0 {
            // close at threshold
            let close_price = f64::max(
                state_params.order_book.ask,
                round_up(
                    position.price * (1.0 + bot_params.close_trailing_threshold_pct),
                    exchange_params.price_step,
                ),
            );
            Order {
                qty: -calc_close_qty(
                    &exchange_params,
                    &bot_params,
                    &position,
                    bot_params.close_trailing_qty_pct,
                    state_params.balance,
                    close_price,
                ),
                price: close_price,
                order_type: OrderType::CloseTrailingLong,
            }
        } else {
            // close if both conditions are met
            if trailing_price_bundle.max_since_open
                > position.price * (1.0 + bot_params.close_trailing_threshold_pct)
                && trailing_price_bundle.min_since_max
                    < trailing_price_bundle.max_since_open
                        * (1.0 - bot_params.close_trailing_retracement_pct)
            {
                let close_price = f64::max(
                    state_params.order_book.ask,
                    round_up(
                        position.price
                            * (1.0 + bot_params.close_trailing_threshold_pct
                                - bot_params.close_trailing_retracement_pct),
                        exchange_params.price_step,
                    ),
                );
                Order {
                    qty: -calc_close_qty(
                        &exchange_params,
                        &bot_params,
                        &position,
                        bot_params.close_trailing_qty_pct,
                        state_params.balance,
                        close_price,
                    ),
                    price: close_price,
                    order_type: OrderType::CloseTrailingLong,
                }
            } else {
                Order {
                    qty: 0.0,
                    price: 0.0,
                    order_type: OrderType::CloseTrailingLong,
                }
            }
        }
    }
}

pub fn calc_next_close_long(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    position: &Position,
    trailing_price_bundle: &TrailingPriceBundle,
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
    if let Some(order) = calc_wel_auto_reduce_long(
        exchange_params,
        state_params,
        bot_params,
        position,
        wallet_exposure,
    ) {
        return order;
    }
    let wallet_exposure_ratio = if wallet_exposure_limit_with_allowance(bot_params) <= 0.0 {
        10.0
    } else {
        wallet_exposure / wallet_exposure_limit_with_allowance(bot_params)
    };
    if bot_params.close_trailing_grid_ratio >= 1.0 || bot_params.close_trailing_grid_ratio <= -1.0 {
        // return trailing only
        return calc_trailing_close_long(
            &exchange_params,
            &state_params,
            &bot_params,
            &position,
            &trailing_price_bundle,
        );
    }
    if bot_params.close_trailing_grid_ratio == 0.0 {
        // return grid only
        return calc_grid_close_long(&exchange_params, &state_params, &bot_params, &position);
    }
    if bot_params.close_trailing_grid_ratio > 0.0 {
        // trailing first
        if wallet_exposure_ratio < bot_params.close_trailing_grid_ratio {
            // return trailing order
            calc_trailing_close_long(
                &exchange_params,
                &state_params,
                &bot_params,
                &position,
                &trailing_price_bundle,
            )
        } else {
            // return grid order, but leave full_psize * close_trailing_grid_ratio for trailing close
            let mut trailing_allocation = cost_to_qty(
                state_params.balance
                    * wallet_exposure_limit_with_allowance(bot_params)
                    * bot_params.close_trailing_grid_ratio,
                position.price,
                exchange_params.c_mult,
            );
            let min_entry_qty = calc_min_entry_qty(position.price, &exchange_params);
            if trailing_allocation < min_entry_qty {
                trailing_allocation = 0.0;
            }
            let grid_allocation = round_(
                (position.size - trailing_allocation) * 1.01, // add 1% to avoid hitting the threshold exactly
                exchange_params.qty_step,
            );
            let position_mod = Position {
                size: f64::min(position.size, f64::max(grid_allocation, min_entry_qty)),
                price: position.price,
            };
            calc_grid_close_long(&exchange_params, &state_params, &bot_params, &position_mod)
        }
    } else {
        // grid first
        if wallet_exposure_ratio < 1.0 + bot_params.close_trailing_grid_ratio {
            // return grid order, closing whole position
            calc_grid_close_long(&exchange_params, &state_params, &bot_params, &position)
        } else {
            // return trailing order, but leave full_psize * (1.0 + close_trailing_grid_ratio) for grid close
            let mut grid_allocation = cost_to_qty(
                state_params.balance
                    * wallet_exposure_limit_with_allowance(bot_params)
                    * (1.0 + bot_params.close_trailing_grid_ratio),
                position.price,
                exchange_params.c_mult,
            );
            let min_entry_qty = calc_min_entry_qty(position.price, &exchange_params);
            if grid_allocation < min_entry_qty {
                grid_allocation = 0.0;
            }
            let trailing_allocation = round_(
                (position.size - grid_allocation) * 1.01,
                exchange_params.qty_step,
            );
            let position_mod = Position {
                size: f64::min(position.size, f64::max(trailing_allocation, min_entry_qty)),
                price: position.price,
            };
            calc_trailing_close_long(
                &exchange_params,
                &state_params,
                &bot_params,
                &position_mod,
                &trailing_price_bundle,
            )
        }
    }
}

pub fn calc_grid_close_short(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    position: &Position,
) -> Order {
    let position_size_abs = position.size.abs();
    if position_size_abs == 0.0 {
        return Order::default();
    }
    if bot_params.close_grid_qty_pct < 0.0 || bot_params.close_grid_qty_pct >= 1.0 {
        return Order {
            qty: round_(position_size_abs, exchange_params.qty_step),
            price: f64::min(
                state_params.order_book.bid,
                round_dn(
                    position.price * (1.0 - bot_params.close_grid_markup_start),
                    exchange_params.price_step,
                ),
            ),
            order_type: OrderType::CloseGridShort,
        };
    }
    let close_prices_start = round_dn(
        position.price * (1.0 - bot_params.close_grid_markup_start),
        exchange_params.price_step,
    );
    let close_prices_end = round_dn(
        position.price * (1.0 - bot_params.close_grid_markup_end),
        exchange_params.price_step,
    );
    if close_prices_start == close_prices_end {
        return Order {
            qty: round_(position_size_abs, exchange_params.qty_step),
            price: f64::min(state_params.order_book.bid, close_prices_start),
            order_type: OrderType::CloseGridShort,
        };
    }
    let n_steps =
        ((close_prices_start - close_prices_end).abs() / exchange_params.price_step).ceil();
    let close_grid_qty_pct_modified = f64::max(bot_params.close_grid_qty_pct, 1.0 / n_steps);
    let wallet_exposure_ratio = calc_wallet_exposure(
        exchange_params.c_mult,
        state_params.balance,
        position_size_abs,
        position.price,
    ) / wallet_exposure_limit_with_allowance(bot_params);
    let close_price = if wallet_exposure_ratio > 1.0 {
        f64::min(
            state_params.order_book.bid,
            f64::max(close_prices_start, close_prices_end),
        )
    } else {
        f64::min(
            state_params.order_book.bid,
            round_dn(
                close_prices_start
                    + (close_prices_end - close_prices_start)
                        * f64::min(1.0, wallet_exposure_ratio),
                exchange_params.price_step,
            ),
        )
    };
    let close_qty = calc_close_qty(
        &exchange_params,
        &bot_params,
        &position,
        close_grid_qty_pct_modified,
        state_params.balance,
        close_price,
    );
    Order {
        qty: close_qty,
        price: close_price,
        order_type: OrderType::CloseGridShort,
    }
}

pub fn calc_trailing_close_short(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    position: &Position,
    trailing_price_bundle: &TrailingPriceBundle,
) -> Order {
    let position_size_abs = position.size.abs();
    if position_size_abs == 0.0 {
        return Order::default();
    }
    if bot_params.close_trailing_threshold_pct <= 0.0 {
        // means trailing stop immediately from pos open
        if bot_params.close_trailing_retracement_pct > 0.0
            && trailing_price_bundle.max_since_min
                > trailing_price_bundle.min_since_open
                    * (1.0 + bot_params.close_trailing_retracement_pct)
        {
            Order {
                qty: calc_close_qty(
                    &exchange_params,
                    &bot_params,
                    &position,
                    bot_params.close_trailing_qty_pct,
                    state_params.balance,
                    state_params.order_book.bid,
                ),
                price: state_params.order_book.bid,
                order_type: OrderType::CloseTrailingShort,
            }
        } else {
            Order {
                qty: 0.0,
                price: 0.0,
                order_type: OrderType::CloseTrailingShort,
            }
        }
    } else {
        // means trailing stop will activate only after a threshold
        if bot_params.close_trailing_retracement_pct <= 0.0 {
            // close at threshold
            let close_price = f64::min(
                state_params.order_book.bid,
                round_dn(
                    position.price * (1.0 - bot_params.close_trailing_threshold_pct),
                    exchange_params.price_step,
                ),
            );
            Order {
                qty: calc_close_qty(
                    &exchange_params,
                    &bot_params,
                    &position,
                    bot_params.close_trailing_qty_pct,
                    state_params.balance,
                    close_price,
                ),
                price: close_price,
                order_type: OrderType::CloseTrailingShort,
            }
        } else {
            if trailing_price_bundle.min_since_open
                < position.price * (1.0 - bot_params.close_trailing_threshold_pct)
                && trailing_price_bundle.max_since_min
                    > trailing_price_bundle.min_since_open
                        * (1.0 + bot_params.close_trailing_retracement_pct)
            {
                let close_price = f64::min(
                    state_params.order_book.bid,
                    round_dn(
                        position.price
                            * (1.0 - bot_params.close_trailing_threshold_pct
                                + bot_params.close_trailing_retracement_pct),
                        exchange_params.price_step,
                    ),
                );
                Order {
                    qty: calc_close_qty(
                        &exchange_params,
                        &bot_params,
                        &position,
                        bot_params.close_trailing_qty_pct,
                        state_params.balance,
                        close_price,
                    ),
                    price: close_price,
                    order_type: OrderType::CloseTrailingShort,
                }
            } else {
                Order {
                    qty: 0.0,
                    price: 0.0,
                    order_type: OrderType::CloseTrailingShort,
                }
            }
        }
    }
}

pub fn calc_next_close_short(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    position: &Position,
    trailing_price_bundle: &TrailingPriceBundle,
) -> Order {
    let position_size_abs = position.size.abs();
    if position_size_abs == 0.0 {
        // no position
        return Order::default();
    }
    let wallet_exposure = calc_wallet_exposure(
        exchange_params.c_mult,
        state_params.balance,
        position_size_abs,
        position.price,
    );
    if let Some(order) = calc_wel_auto_reduce_short(
        exchange_params,
        state_params,
        bot_params,
        position,
        wallet_exposure,
    ) {
        return order;
    }
    if bot_params.close_trailing_grid_ratio >= 1.0 || bot_params.close_trailing_grid_ratio <= -1.0 {
        // return trailing only
        return calc_trailing_close_short(
            &exchange_params,
            &state_params,
            &bot_params,
            &position,
            &trailing_price_bundle,
        );
    }
    if bot_params.close_trailing_grid_ratio == 0.0 {
        // return grid only
        return calc_grid_close_short(&exchange_params, &state_params, &bot_params, &position);
    }
    let wallet_exposure_ratio = calc_wallet_exposure(
        exchange_params.c_mult,
        state_params.balance,
        position_size_abs,
        position.price,
    ) / wallet_exposure_limit_with_allowance(bot_params);
    if bot_params.close_trailing_grid_ratio > 0.0 {
        // trailing first
        if wallet_exposure_ratio < bot_params.close_trailing_grid_ratio {
            // return trailing order, closing whole pos
            calc_trailing_close_short(
                &exchange_params,
                &state_params,
                &bot_params,
                &position,
                &trailing_price_bundle,
            )
        } else {
            // return grid order, but leave full_psize * close_trailing_grid_ratio for trailing close
            let mut trailing_allocation = cost_to_qty(
                state_params.balance
                    * wallet_exposure_limit_with_allowance(bot_params)
                    * bot_params.close_trailing_grid_ratio,
                position.price,
                exchange_params.c_mult,
            );
            let min_entry_qty = calc_min_entry_qty(position.price, &exchange_params);
            if trailing_allocation < min_entry_qty {
                trailing_allocation = 0.0;
            }
            let grid_allocation = round_(
                (position_size_abs - trailing_allocation) * 1.01,
                exchange_params.qty_step,
            );
            let position_mod = Position {
                size: -f64::min(position_size_abs, f64::max(grid_allocation, min_entry_qty)),
                price: position.price,
            };
            calc_grid_close_short(&exchange_params, &state_params, &bot_params, &position_mod)
        }
    } else {
        if wallet_exposure_ratio < 1.0 + bot_params.close_trailing_grid_ratio {
            // return grid order, closing whole position
            return calc_grid_close_short(&exchange_params, &state_params, &bot_params, &position);
        } else {
            // return trailing order, but leave full_psize * (1.0 + close_trailing_grid_ratio) for grid close
            let mut grid_allocation = cost_to_qty(
                state_params.balance
                    * wallet_exposure_limit_with_allowance(bot_params)
                    * (1.0 + bot_params.close_trailing_grid_ratio),
                position.price,
                exchange_params.c_mult,
            );
            let min_entry_qty = calc_min_entry_qty(position.price, &exchange_params);
            if grid_allocation < min_entry_qty {
                grid_allocation = 0.0;
            }
            let trailing_allocation = round_(
                (position_size_abs - grid_allocation) * 1.01,
                exchange_params.qty_step,
            );
            let position_mod = Position {
                size: -f64::min(
                    position_size_abs,
                    f64::max(trailing_allocation, min_entry_qty),
                ),
                price: position.price,
            };
            calc_trailing_close_short(
                &exchange_params,
                &state_params,
                &bot_params,
                &position_mod,
                &trailing_price_bundle,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_exchange_params() -> ExchangeParams {
        ExchangeParams {
            qty_step: 0.01,
            price_step: 0.01,
            min_qty: 0.0,
            min_cost: 0.0,
            c_mult: 1.0,
        }
    }

    #[test]
    fn test_wel_strict_reduce_long_minimal() {
        let exchange = make_exchange_params();
        // Balance 1000, price 100, psize 10.001 -> WE = 1.0001; target = 1.0
        let state = StateParams {
            balance: 1000.0,
            order_book: crate::types::OrderBook {
                ask: 100.0,
                bid: 100.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let bot = BotParams {
            wallet_exposure_limit: 1.0, // WEL_base
            risk_we_excess_allowance_pct: 0.0,
            risk_wel_enforcer_threshold: 1.0,
            ..Default::default()
        };
        let pos = Position {
            size: 10.001,
            price: 100.0,
        };
        let we = calc_wallet_exposure(exchange.c_mult, state.balance, pos.size, pos.price);
        assert!(we > 1.0);
        let order = super::calc_wel_auto_reduce_long(&exchange, &state, &bot, &pos, we)
            .expect("should emit strict reduce order");
        assert!(order.qty < 0.0 && order.price > 0.0);
        let new_psize = (pos.size - order.qty.abs()).max(0.0);
        let new_we = calc_wallet_exposure(exchange.c_mult, state.balance, new_psize, pos.price);
        assert!(new_we < 1.0, "new_we={} not strictly below target", new_we);
    }

    #[test]
    fn test_wel_strict_reduce_short_with_rounding() {
        let exchange = ExchangeParams {
            qty_step: 0.5,
            price_step: 0.5,
            min_qty: 0.0,
            min_cost: 0.0,
            c_mult: 1.0,
        };
        let state = StateParams {
            balance: 500.0,
            order_book: crate::types::OrderBook {
                ask: 50.0,
                bid: 50.0,
                ..Default::default()
            },
            ..Default::default()
        };
        // price 50, psize 10.1 -> WE = (10.1*50)/500 = 1.01; target=1.0
        let bot = BotParams {
            wallet_exposure_limit: 1.0,
            risk_we_excess_allowance_pct: 0.0,
            risk_wel_enforcer_threshold: 1.0,
            ..Default::default()
        };
        let pos = Position {
            size: -10.1,
            price: 50.0,
        };
        let we = calc_wallet_exposure(exchange.c_mult, state.balance, pos.size.abs(), pos.price);
        assert!(we > 1.0);
        let order = super::calc_wel_auto_reduce_short(&exchange, &state, &bot, &pos, we)
            .expect("should emit strict reduce order");
        assert!(order.qty > 0.0 && order.price > 0.0);
        let new_psize = (pos.size.abs() - order.qty.abs()).max(0.0);
        let new_we = calc_wallet_exposure(exchange.c_mult, state.balance, new_psize, pos.price);
        assert!(new_we < 1.0, "new_we={} not strictly below target", new_we);
    }
}

pub fn calc_closes_long(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    position: &Position,
    trailing_price_bundle: &TrailingPriceBundle,
) -> Vec<Order> {
    let mut closes = Vec::<Order>::new();
    let mut psize = position.size;
    for _ in 0..500 {
        let position_mod = Position {
            size: psize,
            price: position.price,
        };
        let mut close = calc_next_close_long(
            exchange_params,
            &state_params,
            bot_params,
            &position_mod,
            &trailing_price_bundle,
        );
        close.price = quantize_price(
            close.price,
            exchange_params.price_step,
            RoundingMode::Nearest,
            "calc_closes_long::price",
        );
        close.qty = quantize_qty(
            close.qty,
            exchange_params.qty_step,
            RoundingMode::Nearest,
            "calc_closes_long::qty",
        );
        if close.qty == 0.0 {
            break;
        }
        psize = round_(psize + close.qty, exchange_params.qty_step);
        if !closes.is_empty() {
            if close.order_type == OrderType::CloseTrailingLong {
                break;
            }
            if closes[closes.len() - 1].price == close.price {
                let previous_close = closes.pop();
                let merged_close = Order {
                    qty: round_(
                        previous_close.unwrap().qty + close.qty,
                        exchange_params.qty_step,
                    ),
                    price: close.price,
                    order_type: close.order_type,
                };
                let mut merged_close = merged_close;
                merged_close.price = quantize_price(
                    merged_close.price,
                    exchange_params.price_step,
                    RoundingMode::Nearest,
                    "calc_closes_long::merged_price",
                );
                merged_close.qty = quantize_qty(
                    merged_close.qty,
                    exchange_params.qty_step,
                    RoundingMode::Nearest,
                    "calc_closes_long::merged_qty",
                );
                closes.push(merged_close);
                continue;
            }
        }
        closes.push(close);
    }
    closes.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap());
    closes
}

pub fn calc_closes_short(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    position: &Position,
    trailing_price_bundle: &TrailingPriceBundle,
) -> Vec<Order> {
    let mut closes = Vec::<Order>::new();
    let mut psize = position.size;
    for _ in 0..500 {
        let position_mod = Position {
            size: psize,
            price: position.price,
        };
        let mut close = calc_next_close_short(
            exchange_params,
            &state_params,
            bot_params,
            &position_mod,
            &trailing_price_bundle,
        );
        close.price = quantize_price(
            close.price,
            exchange_params.price_step,
            RoundingMode::Nearest,
            "calc_closes_short::price",
        );
        close.qty = quantize_qty(
            close.qty,
            exchange_params.qty_step,
            RoundingMode::Nearest,
            "calc_closes_short::qty",
        );
        if close.qty == 0.0 {
            break;
        }
        psize = round_(psize + close.qty, exchange_params.qty_step);
        if !closes.is_empty() {
            if close.order_type == OrderType::CloseTrailingShort {
                break;
            }
            if closes[closes.len() - 1].price == close.price {
                let previous_close = closes.pop();
                let merged_close = Order {
                    qty: round_(
                        previous_close.unwrap().qty + close.qty,
                        exchange_params.qty_step,
                    ),
                    price: close.price,
                    order_type: close.order_type,
                };
                let mut merged_close = merged_close;
                merged_close.price = quantize_price(
                    merged_close.price,
                    exchange_params.price_step,
                    RoundingMode::Nearest,
                    "calc_closes_short::merged_price",
                );
                merged_close.qty = quantize_qty(
                    merged_close.qty,
                    exchange_params.qty_step,
                    RoundingMode::Nearest,
                    "calc_closes_short::merged_qty",
                );
                closes.push(merged_close);
                continue;
            }
        }
        closes.push(close);
    }
    closes.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap());
    closes
}
