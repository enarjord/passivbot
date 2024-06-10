use crate::grids::{calc_reentry_qty, BotParams, ExchangeParams, Order, Position, StateParams};
use crate::utils::{
    calc_wallet_exposure, calc_wallet_exposure_if_filled, interpolate, round_, round_dn, round_up,
};

pub fn calc_trailing_close_long(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    position: &Position,
    highest_since_position_open: f64,
    lowest_since_highest: f64,
) -> Order {
    if position.size == 0.0 {
        return Order::default();
    }
    if highest_since_position_open
        < position.price * (1.0 + bot_params.close_trailing_threshold_pct)
    {
        return Order::default();
    }
    if lowest_since_highest
        > highest_since_position_open * (1.0 - bot_params.close_trailing_drawdown_pct)
    {
        return Order::default();
    }
    Order {
        qty: -position.size,
        price: f64::max(
            state_params.order_book.ask,
            round_up(
                position.price * (1.0 + bot_params.close_trailing_threshold_pct),
                exchange_params.price_step,
            ),
        ),
        description: String::from("long_trailing_close"),
    }
}

pub fn calc_trailing_entry_long(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    position: &Position,
    lowest_since_position_open: f64,
    highest_since_lowest: f64,
) -> Order {
    // it is assumed there is a position
    if position.size == 0.0 || bot_params.wallet_exposure_limit <= 0.0 {
        return Order::default();
    }
    if lowest_since_position_open > position.price * (1.0 - bot_params.entry_trailing_threshold_pct)
    {
        return Order::default();
    }
    if highest_since_lowest
        < lowest_since_position_open * (1.0 + bot_params.entry_trailing_drawdown_pct)
    {
        return Order::default();
    }
    let wallet_exposure = calc_wallet_exposure(
        exchange_params.c_mult,
        state_params.balance,
        position.size,
        position.price,
    );
    if wallet_exposure > bot_params.wallet_exposure_limit * 0.999 {
        return Order::default();
    }
    let entry_qty = calc_reentry_qty(
        state_params.order_book.bid,
        state_params.balance,
        position.size,
        &exchange_params,
        &bot_params,
    );
    let wallet_exposure_if_filled = calc_wallet_exposure_if_filled(
        state_params.balance,
        position.size,
        position.price,
        entry_qty,
        state_params.order_book.bid,
        &exchange_params,
    );
    if wallet_exposure_if_filled > bot_params.wallet_exposure_limit * 1.01 {
        // reentry too big. Crop current reentry qty.
        let entry_qty = interpolate(
            bot_params.wallet_exposure_limit,
            &[wallet_exposure, wallet_exposure_if_filled],
            &[position.size, position.size + entry_qty],
        ) - position.size;
        Order {
            qty: round_(entry_qty, exchange_params.qty_step),
            price: f64::min(
                state_params.order_book.bid,
                round_dn(
                    position.price * (1.0 - bot_params.entry_trailing_threshold_pct),
                    exchange_params.price_step,
                ),
            ),
            description: String::from("long_trailing_reentry_cropped"),
        }
    } else {
        Order {
            qty: entry_qty,
            price: f64::min(
                state_params.order_book.bid,
                round_dn(
                    position.price * (1.0 - bot_params.entry_trailing_threshold_pct),
                    exchange_params.price_step,
                ),
            ),
            description: String::from("long_trailing_reentry_normal"),
        }
    }
}
