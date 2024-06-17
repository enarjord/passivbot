use crate::constants::{CLOSE, LONG, NO_POS, SHORT};
use crate::entries::calc_min_entry_qty;
use crate::types::{
    BotParams, BotParamsPair, EMABands, ExchangeParams, Order, OrderType, Position, Positions,
    StateParams,
};
use crate::utils::{
    calc_auto_unstuck_allowance, calc_pprice_diff_int, calc_wallet_exposure, cost_to_qty, round_dn,
    round_up,
};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

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

pub fn calc_trailing_close_long(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    position: &Position,
    max_price_since_open: f64,
    min_price_since_max: f64,
) -> Order {
    if position.size == 0.0 {
        return Order::default();
    }
    if bot_params.close_trailing_drawdown_pct == 0.0 {
        return Order {
            qty: -position.size,
            price: f64::max(
                state_params.order_book.ask,
                round_up(
                    position.price
                        * (1.0
                            + f64::max(
                                0.0,
                                bot_params.close_trailing_threshold_pct
                                    - bot_params.close_trailing_drawdown_pct,
                            )),
                    exchange_params.price_step,
                ),
            ),
            order_type: OrderType::CloseTrailingLong,
        };
    }
    if max_price_since_open < position.price * (1.0 + bot_params.close_trailing_threshold_pct) {
        return Order::default();
    }
    if min_price_since_max > max_price_since_open * (1.0 - bot_params.close_trailing_drawdown_pct) {
        return Order::default();
    }
    Order {
        qty: -position.size,
        price: f64::max(
            state_params.order_book.ask,
            round_up(
                position.price
                    * (1.0
                        + f64::max(
                            0.0,
                            bot_params.close_trailing_threshold_pct
                                - bot_params.close_trailing_drawdown_pct,
                        )),
                exchange_params.price_step,
            ),
        ),
        order_type: OrderType::CloseTrailingLong,
    }
}

pub fn determine_position_for_unstucking(
    positions: &Positions,
    exchange_params_list: &[ExchangeParams],
    balance: f64,
    bot_params_pair: &BotParamsPair,
    hlcs_k: &Array2<f64>,
) -> (usize, usize) {
    let mut stuck_positions = Vec::<(usize, usize, f64)>::new();
    for idx in positions.long.keys() {
        let wallet_exposure = calc_wallet_exposure(
            exchange_params_list[*idx].c_mult,
            balance,
            positions.long[idx].size,
            positions.long[idx].price,
        );
        if wallet_exposure / bot_params_pair.long.wallet_exposure_limit
            > bot_params_pair.long.unstuck_threshold
            && hlcs_k[[*idx, CLOSE]] < positions.long[idx].price
        {
            let pprice_diff =
                calc_pprice_diff_int(LONG, positions.long[idx].price, hlcs_k[[*idx, CLOSE]]);
            stuck_positions.push((*idx, LONG, pprice_diff));
        }
    }
    for idx in positions.short.keys() {
        let wallet_exposure = calc_wallet_exposure(
            exchange_params_list[*idx].c_mult,
            balance,
            positions.short[idx].size,
            positions.short[idx].price,
        );
        if wallet_exposure / bot_params_pair.short.wallet_exposure_limit
            > bot_params_pair.short.unstuck_threshold
        {
            let pprice_diff =
                calc_pprice_diff_int(SHORT, positions.short[idx].price, hlcs_k[[*idx, CLOSE]]);
            stuck_positions.push((*idx, SHORT, pprice_diff));
        }
    }
    if stuck_positions.is_empty() {
        return (NO_POS, NO_POS);
    }
    stuck_positions.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    let (idx, pside, _pprice_diff) = stuck_positions[0];
    (idx as usize, pside as usize)
}

pub fn calc_unstuck_close_long(
    exchange_params: &ExchangeParams,
    bot_params: &BotParams,
    hlcs_k_idx: &Array1<f64>,
    ema_band_upper: f64,
    position: &Position,
    balance: f64,
    pnl_cumsum_max: f64,
    pnl_cumsum_last: f64,
) -> Order {
    let auto_unstuck_allowance = calc_auto_unstuck_allowance(
        balance,
        bot_params.unstuck_loss_allowance_pct,
        pnl_cumsum_max,
        pnl_cumsum_last,
    );
    if auto_unstuck_allowance <= 0.0 {
        return Order::default();
    }
    let close_price = f64::max(
        hlcs_k_idx[CLOSE],
        round_up(
            ema_band_upper * (1.0 + bot_params.unstuck_ema_dist),
            exchange_params.price_step,
        ),
    );
    let close_qty = -f64::min(
        position.size,
        f64::max(
            calc_min_entry_qty(close_price, exchange_params),
            round_dn(
                cost_to_qty(
                    balance * bot_params.wallet_exposure_limit * bot_params.unstuck_close_pct,
                    close_price,
                    exchange_params.c_mult,
                ),
                exchange_params.qty_step,
            ),
        ),
    );
    Order {
        qty: close_qty,
        price: close_price,
        order_type: OrderType::CloseUnstuckLong,
    }
}

pub fn calc_unstuck_close_short() {}
