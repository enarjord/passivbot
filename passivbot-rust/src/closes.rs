use crate::constants::{CLOSE, LONG, NO_POS, SHORT};
use crate::entries::calc_min_entry_qty;
use crate::types::{
    BotParams, BotParamsPair, EMABands, ExchangeParams, Order, OrderType, Position, Positions,
    StateParams, TrailingPriceBundle,
};
use crate::utils::{
    calc_pprice_diff_int, calc_wallet_exposure, cost_to_qty, round_, round_dn, round_up,
};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

pub fn calc_grid_close_long(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    position: &Position,
) -> Order {
    if position.size <= 0.0 {
        return Order::default();
    }
    if bot_params.close_grid_markup_range <= 0.0
        || bot_params.close_grid_qty_pct < 0.0
        || bot_params.close_grid_qty_pct >= 1.0
    {
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
    let close_prices_start = round_up(
        position.price * (1.0 + bot_params.close_grid_min_markup),
        exchange_params.price_step,
    );
    let close_prices_end = round_up(
        position.price
            * (1.0 + bot_params.close_grid_min_markup + bot_params.close_grid_markup_range),
        exchange_params.price_step,
    );
    if close_prices_start == close_prices_end {
        return Order {
            qty: -position.size,
            price: f64::max(state_params.order_book.ask, close_prices_start),
            order_type: OrderType::CloseGridLong,
        };
    }
    let n_steps = ((close_prices_end - close_prices_start) / exchange_params.price_step).ceil();
    let close_grid_qty_pct_modified = f64::max(bot_params.close_grid_qty_pct, 1.0 / n_steps);
    let wallet_exposure = calc_wallet_exposure(
        exchange_params.c_mult,
        state_params.balance,
        position.size,
        position.price,
    );
    let wallet_exposure_ratio = f64::min(1.0, wallet_exposure / bot_params.wallet_exposure_limit);
    let close_price = round_up(
        position.price
            * (1.0
                + bot_params.close_grid_min_markup
                + bot_params.close_grid_markup_range * (1.0 - wallet_exposure_ratio)),
        exchange_params.price_step,
    );
    let full_psize = cost_to_qty(
        state_params.balance * bot_params.wallet_exposure_limit,
        position.price,
        exchange_params.c_mult,
    );
    let leftover = f64::max(0.0, position.size - full_psize);
    let close_qty = -f64::min(
        position.size,
        f64::max(
            calc_min_entry_qty(close_price, &exchange_params),
            round_up(
                full_psize * close_grid_qty_pct_modified + leftover,
                exchange_params.qty_step,
            ),
        ),
    );
    Order {
        qty: round_(close_qty, exchange_params.qty_step),
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
    if bot_params.close_trailing_retracement_pct <= 0.0 {
        return Order {
            qty: -position.size,
            price: f64::max(
                state_params.order_book.ask,
                round_up(
                    position.price * (1.0 + bot_params.close_trailing_threshold_pct),
                    exchange_params.price_step,
                ),
            ),
            order_type: OrderType::CloseTrailingLong,
        };
    }
    if trailing_price_bundle.max_price_since_open
        > position.price * (1.0 + bot_params.close_trailing_threshold_pct)
        && trailing_price_bundle.min_price_since_max
            < trailing_price_bundle.max_price_since_open
                * (1.0 - bot_params.close_trailing_retracement_pct)
    {
        Order {
            qty: -position.size,
            price: f64::max(
                state_params.order_book.ask,
                round_up(
                    position.price
                        * (1.0 + bot_params.close_trailing_threshold_pct
                            - bot_params.close_trailing_retracement_pct),
                    exchange_params.price_step,
                ),
            ),
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
    let wallet_exposure_ratio = calc_wallet_exposure(
        exchange_params.c_mult,
        state_params.balance,
        position.size,
        position.price,
    ) / bot_params.wallet_exposure_limit;
    if bot_params.close_trailing_grid_ratio > 0.0 {
        // trailing first
        if wallet_exposure_ratio < bot_params.close_trailing_grid_ratio {
            // return trailing order, closing whole position
            calc_trailing_close_long(
                &exchange_params,
                &state_params,
                &bot_params,
                &position,
                &trailing_price_bundle,
            )
        } else {
            // return grid order, but leave full_psize * close_trailing_grid_ratio for trailing close
            let trailing_allocation = cost_to_qty(
                state_params.balance
                    * bot_params.wallet_exposure_limit
                    * bot_params.close_trailing_grid_ratio,
                position.price,
                exchange_params.c_mult,
            );
            let grid_allocation = round_(
                position.size - trailing_allocation,
                exchange_params.qty_step,
            );
            let position_mod = Position {
                size: f64::min(
                    position.size,
                    f64::max(
                        grid_allocation,
                        calc_min_entry_qty(position.price, &exchange_params),
                    ),
                ),
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
            let grid_allocation = cost_to_qty(
                state_params.balance
                    * bot_params.wallet_exposure_limit
                    * (1.0 + bot_params.close_trailing_grid_ratio),
                position.price,
                exchange_params.c_mult,
            );
            let trailing_allocation =
                round_(position.size - grid_allocation, exchange_params.qty_step);
            let position_mod = Position {
                size: f64::min(
                    position.size,
                    f64::max(
                        trailing_allocation,
                        calc_min_entry_qty(position.price, &exchange_params),
                    ),
                ),
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
    if bot_params.close_grid_markup_range <= 0.0
        || bot_params.close_grid_qty_pct < 0.0
        || bot_params.close_grid_qty_pct >= 1.0
    {
        return Order {
            qty: round_(position_size_abs, exchange_params.qty_step),
            price: f64::min(
                state_params.order_book.bid,
                round_dn(
                    position.price * (1.0 - bot_params.close_grid_min_markup),
                    exchange_params.price_step,
                ),
            ),
            order_type: OrderType::CloseGridShort,
        };
    }
    let close_prices_start = round_dn(
        position.price * (1.0 - bot_params.close_grid_min_markup),
        exchange_params.price_step,
    );
    let close_prices_end = round_dn(
        position.price
            * (1.0 - bot_params.close_grid_min_markup - bot_params.close_grid_markup_range),
        exchange_params.price_step,
    );
    if close_prices_start == close_prices_end {
        return Order {
            qty: round_(position_size_abs, exchange_params.qty_step),
            price: f64::min(state_params.order_book.bid, close_prices_start),
            order_type: OrderType::CloseGridShort,
        };
    }
    let n_steps = ((close_prices_start - close_prices_end) / exchange_params.price_step).ceil();
    let close_grid_qty_pct_modified = f64::max(bot_params.close_grid_qty_pct, 1.0 / n_steps);
    let wallet_exposure = calc_wallet_exposure(
        exchange_params.c_mult,
        state_params.balance,
        position_size_abs,
        position.price,
    );
    let wallet_exposure_ratio = f64::min(1.0, wallet_exposure / bot_params.wallet_exposure_limit);
    let close_price = round_dn(
        position.price
            * (1.0
                - bot_params.close_grid_min_markup
                - bot_params.close_grid_markup_range * (1.0 - wallet_exposure_ratio)),
        exchange_params.price_step,
    );
    let full_psize = cost_to_qty(
        state_params.balance * bot_params.wallet_exposure_limit,
        position.price,
        exchange_params.c_mult,
    );
    let leftover = f64::max(0.0, position_size_abs - full_psize);
    let close_qty = f64::min(
        position_size_abs,
        f64::max(
            calc_min_entry_qty(close_price, &exchange_params),
            round_up(
                full_psize * close_grid_qty_pct_modified + leftover,
                exchange_params.qty_step,
            ),
        ),
    );
    Order {
        qty: round_(close_qty, exchange_params.qty_step),
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
    if bot_params.close_trailing_retracement_pct <= 0.0 {
        return Order {
            qty: position_size_abs,
            price: f64::min(
                state_params.order_book.bid,
                round_dn(
                    position.price * (1.0 - bot_params.close_trailing_threshold_pct),
                    exchange_params.price_step,
                ),
            ),
            order_type: OrderType::CloseTrailingShort,
        };
    }
    if trailing_price_bundle.min_price_since_open
        < position.price * (1.0 - bot_params.close_trailing_threshold_pct)
        && trailing_price_bundle.max_price_since_min
            > trailing_price_bundle.min_price_since_open
                * (1.0 + bot_params.close_trailing_retracement_pct)
    {
        Order {
            qty: position_size_abs,
            price: f64::min(
                state_params.order_book.bid,
                round_dn(
                    position.price
                        * (1.0 - bot_params.close_trailing_threshold_pct
                            + bot_params.close_trailing_retracement_pct),
                    exchange_params.price_step,
                ),
            ),
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
            let trailing_allocation = cost_to_qty(
                state_params.balance
                    * bot_params.wallet_exposure_limit
                    * bot_params.close_trailing_grid_ratio,
                position.price,
                exchange_params.c_mult,
            );
            let grid_allocation = round_(
                position_size_abs - trailing_allocation,
                exchange_params.qty_step,
            );
            let position_mod = Position {
                size: -f64::min(
                    position_size_abs,
                    f64::max(
                        grid_allocation,
                        calc_min_entry_qty(position.price, &exchange_params),
                    ),
                ),
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
            let grid_allocation = cost_to_qty(
                state_params.balance
                    * bot_params.wallet_exposure_limit
                    * (1.0 + bot_params.close_trailing_grid_ratio),
                position.price,
                exchange_params.c_mult,
            );
            let trailing_allocation = round_(
                position_size_abs - grid_allocation,
                exchange_params.qty_step,
            );
            let position_mod = Position {
                size: -f64::min(
                    position_size_abs,
                    f64::max(
                        trailing_allocation,
                        calc_min_entry_qty(position.price, &exchange_params),
                    ),
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
    let mut ask = state_params.order_book.ask;
    for _ in 0..500 {
        let position_mod = Position {
            size: psize,
            price: position.price,
        };
        let mut state_params_mod = state_params.clone();
        state_params_mod.order_book.ask = ask;
        let close = calc_next_close_long(
            exchange_params,
            &state_params_mod,
            bot_params,
            &position_mod,
            &trailing_price_bundle,
        );
        if close.qty == 0.0 {
            break;
        }
        psize = round_(psize + close.qty, exchange_params.qty_step);
        ask = ask.max(close.price);
        if !closes.is_empty() {
            if close.order_type == OrderType::CloseTrailingLong {
                break;
            }
            if closes[closes.len() - 1].price == close.price {
                let previous_close = closes.pop();
                let merged_close = Order {
                    qty: previous_close.unwrap().qty + close.qty,
                    price: close.price,
                    order_type: close.order_type,
                };
                closes.push(merged_close);
                continue;
            }
        }
        closes.push(close);
    }
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
    let mut bid = state_params.order_book.bid;
    for _ in 0..500 {
        let position_mod = Position {
            size: psize,
            price: position.price,
        };
        let mut state_params_mod = state_params.clone();
        state_params_mod.order_book.bid = bid;
        let close = calc_next_close_short(
            exchange_params,
            &state_params_mod,
            bot_params,
            &position_mod,
            &trailing_price_bundle,
        );
        if close.qty == 0.0 {
            break;
        }
        psize = round_(psize + close.qty, exchange_params.qty_step);
        bid = bid.min(close.price);
        if !closes.is_empty() {
            if close.order_type == OrderType::CloseTrailingShort {
                break;
            }
            if closes[closes.len() - 1].price == close.price {
                let previous_close = closes.pop();
                let merged_close = Order {
                    qty: previous_close.unwrap().qty + close.qty,
                    price: close.price,
                    order_type: close.order_type,
                };
                closes.push(merged_close);
                continue;
            }
        }
        closes.push(close);
    }
    closes
}
