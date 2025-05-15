use crate::entries::calc_min_entry_qty;
use crate::types::{
    BotParams, BotParamsPair, EMABands, ExchangeParams, Order, OrderType, Position, Positions,
    StateParams, TrailingPriceBundle,
};
use crate::utils::{
    calc_pprice_diff_int, calc_wallet_exposure, cost_to_qty, interpolate, round_, round_dn,
    round_up,
};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

pub fn calc_close_qty(
    exchange_params: &ExchangeParams,
    bot_params: &BotParams,
    position: &Position,
    close_qty_pct: f64,
    balance: f64,
    close_price: f64,
) -> f64 {
    let full_psize = cost_to_qty(
        balance * bot_params.wallet_exposure_limit,
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
    let wallet_exposure_ratio = wallet_exposure / bot_params.wallet_exposure_limit;
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
    let wallet_exposure_ratio = if bot_params.wallet_exposure_limit <= 0.0 {
        10.0
    } else {
        wallet_exposure / bot_params.wallet_exposure_limit
    };
    if bot_params.enforce_exposure_limit && wallet_exposure_ratio > 1.01 {
        let position_size_lowered = position.size * 0.9;
        let wallet_exposure_lowered = calc_wallet_exposure(
            exchange_params.c_mult,
            state_params.balance,
            position_size_lowered,
            position.price,
        );
        let ideal_psize = interpolate(
            bot_params.wallet_exposure_limit * 1.01,
            &[wallet_exposure, wallet_exposure_lowered],
            &[position.size, position_size_lowered],
        );
        let auto_reduce_qty = position.size - ideal_psize;
        if auto_reduce_qty > 0.0 {
            let close_qty = f64::min(
                round_(position.size, exchange_params.qty_step),
                f64::max(
                    calc_min_entry_qty(state_params.order_book.ask, &exchange_params),
                    round_(auto_reduce_qty, exchange_params.qty_step),
                ),
            );
            return Order {
                price: state_params.order_book.ask,
                qty: -close_qty,
                order_type: OrderType::CloseAutoReduceLong,
            };
        }
    }
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
                    * bot_params.wallet_exposure_limit
                    * bot_params.close_trailing_grid_ratio,
                position.price,
                exchange_params.c_mult,
            );
            let min_entry_qty = calc_min_entry_qty(position.price, &exchange_params);
            if trailing_allocation < min_entry_qty {
                trailing_allocation = 0.0;
            }
            let grid_allocation = round_(
                position.size - trailing_allocation,
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
                    * bot_params.wallet_exposure_limit
                    * (1.0 + bot_params.close_trailing_grid_ratio),
                position.price,
                exchange_params.c_mult,
            );
            let min_entry_qty = calc_min_entry_qty(position.price, &exchange_params);
            if grid_allocation < min_entry_qty {
                grid_allocation = 0.0;
            }
            let trailing_allocation =
                round_(position.size - grid_allocation, exchange_params.qty_step);
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
    let wallet_exposure = calc_wallet_exposure(
        exchange_params.c_mult,
        state_params.balance,
        position_size_abs,
        position.price,
    );
    let wallet_exposure_ratio = wallet_exposure / bot_params.wallet_exposure_limit;
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
    let wallet_exposure_ratio = if bot_params.wallet_exposure_limit <= 0.0 {
        10.0
    } else {
        wallet_exposure / bot_params.wallet_exposure_limit
    };
    if bot_params.enforce_exposure_limit && wallet_exposure_ratio > 1.01 {
        let position_size_lowered = position_size_abs * 0.9;
        let wallet_exposure_lowered = calc_wallet_exposure(
            exchange_params.c_mult,
            state_params.balance,
            position_size_lowered,
            position.price,
        );
        let ideal_psize = interpolate(
            bot_params.wallet_exposure_limit * 1.01,
            &[wallet_exposure, wallet_exposure_lowered],
            &[position_size_abs, position_size_lowered],
        );
        let auto_reduce_qty = position_size_abs - ideal_psize;
        if auto_reduce_qty > 0.0 {
            let close_qty = f64::min(
                round_(position_size_abs, exchange_params.qty_step),
                f64::max(
                    calc_min_entry_qty(state_params.order_book.bid, &exchange_params),
                    round_(auto_reduce_qty, exchange_params.qty_step),
                ),
            );
            return Order {
                price: state_params.order_book.bid,
                qty: close_qty,
                order_type: OrderType::CloseAutoReduceShort,
            };
        }
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
    ) / bot_params.wallet_exposure_limit;
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
                    * bot_params.wallet_exposure_limit
                    * bot_params.close_trailing_grid_ratio,
                position.price,
                exchange_params.c_mult,
            );
            let min_entry_qty = calc_min_entry_qty(position.price, &exchange_params);
            if trailing_allocation < min_entry_qty {
                trailing_allocation = 0.0;
            }
            let grid_allocation = round_(
                position_size_abs - trailing_allocation,
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
                    * bot_params.wallet_exposure_limit
                    * (1.0 + bot_params.close_trailing_grid_ratio),
                position.price,
                exchange_params.c_mult,
            );
            let min_entry_qty = calc_min_entry_qty(position.price, &exchange_params);
            if grid_allocation < min_entry_qty {
                grid_allocation = 0.0;
            }
            let trailing_allocation = round_(
                position_size_abs - grid_allocation,
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
        let close = calc_next_close_long(
            exchange_params,
            &state_params,
            bot_params,
            &position_mod,
            &trailing_price_bundle,
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
        let close = calc_next_close_short(
            exchange_params,
            &state_params,
            bot_params,
            &position_mod,
            &trailing_price_bundle,
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
                closes.push(merged_close);
                continue;
            }
        }
        closes.push(close);
    }
    closes.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap());
    closes
}
