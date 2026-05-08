use crate::dynamic::{calc_dynamic_distance_multiplier, DynamicDistanceInputs};
use crate::entries::{calc_min_entry_qty, wallet_exposure_limit_with_allowance};
use crate::strategies::TrailingMartingaleCloseParams;
use crate::types::{
    BotParams, ExchangeParams, Order, OrderType, Position, RuntimeOrderContext, StateParams,
    TrailingPriceBundle,
};
use crate::utils::{
    calc_wallet_exposure, cost_to_qty, quantize_price, quantize_qty, round_, round_dn, round_up,
    RoundingMode,
};

pub fn calc_close_qty(
    exchange_params: &ExchangeParams,
    bot_params: &BotParams,
    runtime_context: &RuntimeOrderContext,
    position: &Position,
    close_qty_pct: f64,
    balance: f64,
    close_price: f64,
) -> f64 {
    let full_psize = cost_to_qty(
        balance * wallet_exposure_limit_with_allowance(bot_params, runtime_context),
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
    runtime_context: &RuntimeOrderContext,
    position: &Position,
    wallet_exposure: f64,
) -> Option<Order> {
    if !bot_params.risk_wel_enforcer_enabled || bot_params.risk_wel_enforcer_threshold <= 0.0 {
        return None;
    }
    if state_params.balance <= 0.0 || position.price <= 0.0 {
        return None;
    }
    let allowed_limit = wallet_exposure_limit_with_allowance(bot_params, runtime_context);
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
    runtime_context: &RuntimeOrderContext,
    position: &Position,
    wallet_exposure: f64,
) -> Option<Order> {
    if !bot_params.risk_wel_enforcer_enabled || bot_params.risk_wel_enforcer_threshold <= 0.0 {
        return None;
    }
    if state_params.balance <= 0.0 || position.price <= 0.0 {
        return None;
    }
    let allowed_limit = wallet_exposure_limit_with_allowance(bot_params, runtime_context);
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

fn calc_close_retracement_multiplier(
    state_params: &StateParams,
    close_params: &TrailingMartingaleCloseParams,
) -> f64 {
    calc_dynamic_distance_multiplier(DynamicDistanceInputs {
        volatility_ema_1m: state_params.volatility_ema_1m,
        volatility_ema_1h: state_params.volatility_ema_1h,
        weight_volatility_1m: close_params.retracement_volatility_1m_weight,
        weight_volatility_1h: close_params.retracement_volatility_1h_weight,
        wallet_exposure_ratio: None,
        weight_wallet_exposure: 0.0,
        min_multiplier: 1.0,
    })
}

fn calc_close_threshold_pct(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    runtime_context: &RuntimeOrderContext,
    close_params: &TrailingMartingaleCloseParams,
    position: &Position,
) -> f64 {
    let wallet_exposure_limit = wallet_exposure_limit_with_allowance(bot_params, runtime_context);
    let wallet_exposure_ratio = if wallet_exposure_limit > 0.0 {
        calc_wallet_exposure(
            exchange_params.c_mult,
            state_params.balance,
            position.size.abs(),
            position.price,
        ) / wallet_exposure_limit
    } else {
        0.0
    };
    close_params.threshold_base_pct
        + wallet_exposure_ratio * close_params.threshold_we_weight
        + state_params.volatility_ema_1m * close_params.threshold_volatility_1m_weight
        + state_params.volatility_ema_1h * close_params.threshold_volatility_1h_weight
}

pub fn calc_grid_close_long(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    runtime_context: &RuntimeOrderContext,
    close_params: &TrailingMartingaleCloseParams,
    position: &Position,
) -> Order {
    if position.size <= 0.0 {
        return Order::default();
    }
    let threshold_pct = calc_close_threshold_pct(
        exchange_params,
        state_params,
        bot_params,
        runtime_context,
        close_params,
        position,
    );
    let close_price = f64::max(
        state_params.order_book.ask,
        round_up(
            position.price * (1.0 + threshold_pct),
            exchange_params.price_step,
        ),
    );
    let qty_pct = if close_params.threshold_we_weight == 0.0 {
        1.0
    } else {
        close_params.qty_pct
    };
    let close_qty = -calc_close_qty(
        &exchange_params,
        &bot_params,
        runtime_context,
        &position,
        qty_pct,
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
    runtime_context: &RuntimeOrderContext,
    close_params: &TrailingMartingaleCloseParams,
    position: &Position,
    trailing_price_bundle: &TrailingPriceBundle,
) -> Order {
    if position.size == 0.0 {
        return Order::default();
    }
    let close_trailing_threshold_pct = calc_close_threshold_pct(
        exchange_params,
        state_params,
        bot_params,
        runtime_context,
        close_params,
        position,
    );
    let close_trailing_retracement_pct = close_params.retracement_base_pct.max(0.0)
        * calc_close_retracement_multiplier(state_params, close_params);
    if close_trailing_threshold_pct <= 0.0 {
        // means trailing close immediately from pos open
        if close_trailing_retracement_pct > 0.0
            && trailing_price_bundle.min_since_max
                < trailing_price_bundle.max_since_open * (1.0 - close_trailing_retracement_pct)
        {
            Order {
                qty: -calc_close_qty(
                    &exchange_params,
                    &bot_params,
                    runtime_context,
                    &position,
                    close_params.qty_pct,
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
        if close_trailing_retracement_pct <= 0.0 {
            // close at threshold
            let close_price = f64::max(
                state_params.order_book.ask,
                round_up(
                    position.price * (1.0 + close_trailing_threshold_pct),
                    exchange_params.price_step,
                ),
            );
            Order {
                qty: -calc_close_qty(
                    &exchange_params,
                    &bot_params,
                    runtime_context,
                    &position,
                    close_params.qty_pct,
                    state_params.balance,
                    close_price,
                ),
                price: close_price,
                order_type: OrderType::CloseTrailingLong,
            }
        } else {
            // close if both conditions are met
            if trailing_price_bundle.max_since_open
                > position.price * (1.0 + close_trailing_threshold_pct)
                && trailing_price_bundle.min_since_max
                    < trailing_price_bundle.max_since_open * (1.0 - close_trailing_retracement_pct)
            {
                let close_price = f64::max(
                    state_params.order_book.ask,
                    round_up(
                        position.price
                            * (1.0 + close_trailing_threshold_pct - close_trailing_retracement_pct),
                        exchange_params.price_step,
                    ),
                );
                Order {
                    qty: -calc_close_qty(
                        &exchange_params,
                        &bot_params,
                        runtime_context,
                        &position,
                        close_params.qty_pct,
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
    runtime_context: &RuntimeOrderContext,
    close_params: &TrailingMartingaleCloseParams,
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
        runtime_context,
        position,
        wallet_exposure,
    ) {
        return order;
    }
    if close_params.retracement_base_pct > 0.0 {
        calc_trailing_close_long(
            exchange_params,
            state_params,
            bot_params,
            runtime_context,
            close_params,
            position,
            trailing_price_bundle,
        )
    } else {
        calc_grid_close_long(
            exchange_params,
            state_params,
            bot_params,
            runtime_context,
            close_params,
            position,
        )
    }
}

pub fn calc_grid_close_short(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    runtime_context: &RuntimeOrderContext,
    close_params: &TrailingMartingaleCloseParams,
    position: &Position,
) -> Order {
    let position_size_abs = position.size.abs();
    if position_size_abs == 0.0 {
        return Order::default();
    }
    let threshold_pct = calc_close_threshold_pct(
        exchange_params,
        state_params,
        bot_params,
        runtime_context,
        close_params,
        position,
    );
    let close_price = f64::min(
        state_params.order_book.bid,
        round_dn(
            position.price * (1.0 - threshold_pct),
            exchange_params.price_step,
        ),
    );
    let qty_pct = if close_params.threshold_we_weight == 0.0 {
        1.0
    } else {
        close_params.qty_pct
    };
    let close_qty = calc_close_qty(
        &exchange_params,
        &bot_params,
        runtime_context,
        &position,
        qty_pct,
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
    runtime_context: &RuntimeOrderContext,
    close_params: &TrailingMartingaleCloseParams,
    position: &Position,
    trailing_price_bundle: &TrailingPriceBundle,
) -> Order {
    let position_size_abs = position.size.abs();
    if position_size_abs == 0.0 {
        return Order::default();
    }
    let close_trailing_threshold_pct = calc_close_threshold_pct(
        exchange_params,
        state_params,
        bot_params,
        runtime_context,
        close_params,
        position,
    );
    let close_trailing_retracement_pct = close_params.retracement_base_pct.max(0.0)
        * calc_close_retracement_multiplier(state_params, close_params);
    if close_trailing_threshold_pct <= 0.0 {
        // means trailing stop immediately from pos open
        if close_trailing_retracement_pct > 0.0
            && trailing_price_bundle.max_since_min
                > trailing_price_bundle.min_since_open * (1.0 + close_trailing_retracement_pct)
        {
            Order {
                qty: calc_close_qty(
                    &exchange_params,
                    &bot_params,
                    runtime_context,
                    &position,
                    close_params.qty_pct,
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
        if close_trailing_retracement_pct <= 0.0 {
            // close at threshold
            let close_price = f64::min(
                state_params.order_book.bid,
                round_dn(
                    position.price * (1.0 - close_trailing_threshold_pct),
                    exchange_params.price_step,
                ),
            );
            Order {
                qty: calc_close_qty(
                    &exchange_params,
                    &bot_params,
                    runtime_context,
                    &position,
                    close_params.qty_pct,
                    state_params.balance,
                    close_price,
                ),
                price: close_price,
                order_type: OrderType::CloseTrailingShort,
            }
        } else {
            if trailing_price_bundle.min_since_open
                < position.price * (1.0 - close_trailing_threshold_pct)
                && trailing_price_bundle.max_since_min
                    > trailing_price_bundle.min_since_open * (1.0 + close_trailing_retracement_pct)
            {
                let close_price = f64::min(
                    state_params.order_book.bid,
                    round_dn(
                        position.price
                            * (1.0 - close_trailing_threshold_pct + close_trailing_retracement_pct),
                        exchange_params.price_step,
                    ),
                );
                Order {
                    qty: calc_close_qty(
                        &exchange_params,
                        &bot_params,
                        runtime_context,
                        &position,
                        close_params.qty_pct,
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
    runtime_context: &RuntimeOrderContext,
    close_params: &TrailingMartingaleCloseParams,
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
        runtime_context,
        position,
        wallet_exposure,
    ) {
        return order;
    }
    if close_params.retracement_base_pct > 0.0 {
        calc_trailing_close_short(
            exchange_params,
            state_params,
            bot_params,
            runtime_context,
            close_params,
            position,
            trailing_price_bundle,
        )
    } else {
        calc_grid_close_short(
            exchange_params,
            state_params,
            bot_params,
            runtime_context,
            close_params,
            position,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_runtime_context() -> RuntimeOrderContext {
        RuntimeOrderContext {
            effective_wallet_exposure_limit: 1.0,
        }
    }

    fn make_exchange_params() -> ExchangeParams {
        ExchangeParams {
            qty_step: 0.01,
            price_step: 0.01,
            min_qty: 0.0,
            min_cost: 0.0,
            c_mult: 1.0,
            ..Default::default()
        }
    }

    fn make_close_params() -> TrailingMartingaleCloseParams {
        TrailingMartingaleCloseParams {
            qty_pct: 1.0,
            threshold_base_pct: 0.01,
            threshold_we_weight: 0.0,
            threshold_volatility_1h_weight: 0.0,
            threshold_volatility_1m_weight: 0.0,
            retracement_base_pct: 0.0,
            retracement_volatility_1h_weight: 0.0,
            retracement_volatility_1m_weight: 0.0,
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
        let order = super::calc_wel_auto_reduce_long(
            &exchange,
            &state,
            &bot,
            &make_runtime_context(),
            &pos,
            we,
        )
        .expect("should emit strict reduce order");
        assert!(order.qty < 0.0 && order.price > 0.0);
        let new_psize = (pos.size - order.qty.abs()).max(0.0);
        let new_we = calc_wallet_exposure(exchange.c_mult, state.balance, new_psize, pos.price);
        assert!(new_we < 1.0, "new_we={} not strictly below target", new_we);
    }

    #[test]
    fn test_wel_enforcer_disabled_suppresses_reduce() {
        let exchange = make_exchange_params();
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
            wallet_exposure_limit: 1.0,
            risk_wel_enforcer_enabled: false,
            risk_wel_enforcer_threshold: 1.0,
            ..Default::default()
        };
        let pos = Position {
            size: 11.0,
            price: 100.0,
        };
        let we = calc_wallet_exposure(exchange.c_mult, state.balance, pos.size, pos.price);
        assert!(we > 1.0);
        assert!(super::calc_wel_auto_reduce_long(
            &exchange,
            &state,
            &bot,
            &make_runtime_context(),
            &pos,
            we,
        )
        .is_none());
    }

    #[test]
    fn test_wel_strict_reduce_short_with_rounding() {
        let exchange = ExchangeParams {
            qty_step: 0.5,
            price_step: 0.5,
            min_qty: 0.0,
            min_cost: 0.0,
            c_mult: 1.0,
            ..Default::default()
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
        let order = super::calc_wel_auto_reduce_short(
            &exchange,
            &state,
            &bot,
            &make_runtime_context(),
            &pos,
            we,
        )
        .expect("should emit strict reduce order");
        assert!(order.qty > 0.0 && order.price > 0.0);
        let new_psize = (pos.size.abs() - order.qty.abs()).max(0.0);
        let new_we = calc_wallet_exposure(exchange.c_mult, state.balance, new_psize, pos.price);
        assert!(new_we < 1.0, "new_we={} not strictly below target", new_we);
    }

    #[test]
    fn test_grid_close_long_adds_volatility_and_wallet_exposure_threshold() {
        let exchange = make_exchange_params();
        let state = StateParams {
            balance: 10_000.0,
            order_book: crate::types::OrderBook {
                ask: 100.0,
                bid: 100.0,
                ..Default::default()
            },
            volatility_ema_1h: 0.004,
            volatility_ema_1m: 0.002,
            ..Default::default()
        };
        let bot = BotParams {
            wallet_exposure_limit: 1.0,
            risk_we_excess_allowance_pct: 0.0,
            ..Default::default()
        };
        let close = TrailingMartingaleCloseParams {
            threshold_base_pct: 0.008,
            threshold_we_weight: -0.002,
            threshold_volatility_1h_weight: 0.25,
            threshold_volatility_1m_weight: 0.15,
            qty_pct: 0.1,
            ..make_close_params()
        };
        let pos = Position {
            size: 50.0,
            price: 100.0,
        };

        let order = calc_grid_close_long(
            &exchange,
            &state,
            &bot,
            &make_runtime_context(),
            &close,
            &pos,
        );

        assert_eq!(order.order_type, OrderType::CloseGridLong);
        assert_eq!(order.price, 100.83);
        assert_eq!(order.qty, -10.0);
    }

    fn make_recursive_close_state() -> StateParams {
        StateParams {
            balance: 10_000.0,
            order_book: crate::types::OrderBook {
                ask: 100.0,
                bid: 100.0,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn make_recursive_close_bot() -> BotParams {
        BotParams {
            wallet_exposure_limit: 1.0,
            risk_we_excess_allowance_pct: 0.0,
            risk_wel_enforcer_enabled: false,
            risk_wel_enforcer_threshold: 0.0,
            ..Default::default()
        }
    }

    #[test]
    fn test_recursive_grid_close_long_negative_we_weight_ascends_as_exposure_falls() {
        let exchange = make_exchange_params();
        let state = make_recursive_close_state();
        let bot = make_recursive_close_bot();
        let close = TrailingMartingaleCloseParams {
            qty_pct: 0.1,
            threshold_base_pct: 0.02,
            threshold_we_weight: -0.01,
            ..make_close_params()
        };
        let pos = Position {
            size: 100.0,
            price: 100.0,
        };

        let closes = calc_closes_long(
            &exchange,
            &state,
            &bot,
            &make_runtime_context(),
            &close,
            &pos,
            &TrailingPriceBundle::default(),
        );

        assert_eq!(closes.len(), 10);
        assert!(closes
            .iter()
            .all(|order| order.order_type == OrderType::CloseGridLong));
        for (idx, order) in closes.iter().enumerate() {
            let expected_price = 101.0 + idx as f64 * 0.1;
            assert!(
                (order.price - expected_price).abs() < 1e-12,
                "idx={} price={} expected={}",
                idx,
                order.price,
                expected_price
            );
            assert!((order.qty + 10.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_recursive_grid_close_long_positive_we_weight_sorts_returned_orders() {
        let exchange = make_exchange_params();
        let state = make_recursive_close_state();
        let bot = make_recursive_close_bot();
        let close = TrailingMartingaleCloseParams {
            qty_pct: 0.1,
            threshold_base_pct: 0.01,
            threshold_we_weight: 0.01,
            ..make_close_params()
        };
        let pos = Position {
            size: 100.0,
            price: 100.0,
        };

        let closes = calc_closes_long(
            &exchange,
            &state,
            &bot,
            &make_runtime_context(),
            &close,
            &pos,
            &TrailingPriceBundle::default(),
        );

        assert_eq!(closes.len(), 10);
        assert!(closes
            .iter()
            .all(|order| order.order_type == OrderType::CloseGridLong));
        // Recursive generation produces 102.0, 101.9, ..., 101.1 as exposure falls; the public
        // close list is sorted ascending before returning.
        for (idx, order) in closes.iter().enumerate() {
            let expected_price = 101.1 + idx as f64 * 0.1;
            assert!(
                (order.price - expected_price).abs() < 1e-12,
                "idx={} price={} expected={}",
                idx,
                order.price,
                expected_price
            );
            assert!((order.qty + 10.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_grid_close_long_zero_we_weight_ignores_qty_pct_and_closes_full_position() {
        let exchange = make_exchange_params();
        let state = make_recursive_close_state();
        let bot = make_recursive_close_bot();
        let close = TrailingMartingaleCloseParams {
            qty_pct: 0.1,
            threshold_base_pct: 0.01,
            threshold_we_weight: 0.0,
            ..make_close_params()
        };
        let pos = Position {
            size: 100.0,
            price: 100.0,
        };

        let closes = calc_closes_long(
            &exchange,
            &state,
            &bot,
            &make_runtime_context(),
            &close,
            &pos,
            &TrailingPriceBundle::default(),
        );

        assert_eq!(closes.len(), 1);
        assert_eq!(closes[0].order_type, OrderType::CloseGridLong);
        assert_eq!(closes[0].price, 101.0);
        assert_eq!(closes[0].qty, -100.0);
    }

    #[test]
    fn test_trailing_close_long_returns_single_order_not_recursive_grid() {
        let exchange = make_exchange_params();
        let state = make_recursive_close_state();
        let bot = make_recursive_close_bot();
        let close = TrailingMartingaleCloseParams {
            qty_pct: 0.1,
            threshold_base_pct: 0.01,
            threshold_we_weight: -0.005,
            retracement_base_pct: 0.002,
            ..make_close_params()
        };
        let pos = Position {
            size: 100.0,
            price: 100.0,
        };
        let trailing = TrailingPriceBundle {
            max_since_open: 102.0,
            min_since_max: 101.0,
            ..Default::default()
        };

        let closes = calc_closes_long(
            &exchange,
            &state,
            &bot,
            &make_runtime_context(),
            &close,
            &pos,
            &trailing,
        );

        assert_eq!(closes.len(), 1);
        assert_eq!(closes[0].order_type, OrderType::CloseTrailingLong);
        assert_eq!(closes[0].price, 100.3);
        assert_eq!(closes[0].qty, -10.0);
    }
}

pub fn calc_closes_long(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    runtime_context: &RuntimeOrderContext,
    close_params: &TrailingMartingaleCloseParams,
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
            runtime_context,
            close_params,
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
    runtime_context: &RuntimeOrderContext,
    close_params: &TrailingMartingaleCloseParams,
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
            runtime_context,
            close_params,
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
