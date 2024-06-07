use crate::utils::{
    calc_min_entry_qty, calc_new_psize_pprice, calc_wallet_exposure_if_filled, cost_to_qty,
    interpolate, qty_to_cost, round_, round_dn, round_up,
};

pub struct ExchangeParams {
    pub qty_step: f64,
    pub price_step: f64,
    pub min_qty: f64,
    pub min_cost: f64,
    pub c_mult: f64,
}

pub struct Position {
    pub size: f64,
    pub price: f64,
}

#[derive(Debug, Default, Clone)]
pub struct EMABands {
    pub upper: f64,
    pub lower: f64,
}

#[derive(Debug, Default)]
pub struct Order {
    pub qty: f64,
    pub price: f64,
    pub description: String,
}

#[derive(Debug, Default, Clone)]
pub struct OrderBook {
    pub bid: f64,
    pub ask: f64,
}

#[derive(Debug, Default, Clone)]
pub struct StateParams {
    pub balance: f64,
    pub order_book: OrderBook,
    pub ema_bands: EMABands,
}

pub struct BotParams {
    pub close_grid_markup_range: f64,
    pub close_grid_min_markup: f64,
    pub close_grid_n_orders: f64,
    pub close_trailing_drawdown_pct: f64,
    pub close_trailing_grid_ratio: f64,
    pub close_trailing_threshold_pct: f64,
    pub entry_grid_double_down_factor: f64,
    pub entry_grid_spacing_weight: f64,
    pub entry_grid_spacing_pct: f64,
    pub entry_initial_ema_dist: f64,
    pub entry_initial_qty_pct: f64,
    pub entry_trailing_drawdown_pct: f64,
    pub entry_trailing_grid_ratio: f64,
    pub entry_trailing_threshold_pct: f64,
    pub n_positions: usize,
    pub total_wallet_exposure_limit: f64,
    pub wallet_exposure_limit: f64, // is total_wallet_exposure_limit / n_positions
    pub unstuck_close_pct: f64,
    pub unstuck_ema_dist: f64,
    pub unstuck_loss_allowance_pct: f64,
    pub unstuck_threshold: f64,
}

fn calc_initial_entry(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    position: &Position,
) -> Order {
    let x = round_dn(
        state_params.ema_bands.lower * (1.0 - bot_params.entry_initial_ema_dist),
        exchange_params.price_step,
    );

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
            description: String::from("long_ientry_normal"),
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
            description: String::from("long_ientry_partial"),
        }
    } else {
        Order::default()
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

    let wallet_exposure =
        qty_to_cost(position.size, position.price, exchange_params.c_mult) / state_params.balance;
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
    let next_wallet_exposure =
        qty_to_cost(next_psize, next_pprice, exchange_params.c_mult) / state_params.balance;

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
        println!("inflating entry qty; old_entry_qty {}, new_entry_qty {}, effective_double_down_factor {}", reentry_order.qty, new_entry_qty, effective_double_down_factor);
        Order {
            qty: round_(new_entry_qty, exchange_params.qty_step),
            price: reentry_order.price,
            description: String::from("long_reentry_inflated"),
        }
    } else {
        Order {
            qty: reentry_order.qty,
            price: reentry_order.price,
            description: String::from("long_reentry_normal"),
        }
    }
}

fn calc_reentry_qty(
    entry_price: f64,
    balance: f64,
    position_size: f64,
    exchange_params: &ExchangeParams,
    bot_params: &BotParams,
) -> f64 {
    f64::max(
        calc_min_entry_qty(entry_price, &exchange_params),
        round_dn(
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
    // returns (Order, is_cropped)
    let wallet_exposure =
        qty_to_cost(position.size, position.price, exchange_params.c_mult) / state_params.balance;
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
                description: String::new(),
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
                description: String::from("long_reentry_cropped"),
            },
            wallet_exposure_if_filled,
            true,
        )
    } else {
        (
            Order {
                qty: entry_qty,
                price: entry_price,
                description: String::from("long_reentry_normal"),
            },
            wallet_exposure_if_filled,
            false,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calc_next_grid_entry_long() {
        let exchange_params = ExchangeParams {
            qty_step: 0.001,
            price_step: 0.01,
            min_qty: 0.001,
            min_cost: 10.0,
            c_mult: 1.0,
        };
        let state_params = StateParams {
            balance: 1000.0,
            order_book: OrderBook {
                bid: 99.99,
                ask: 100.01,
            },
            ema_bands: EMABands {
                upper: 100.03,
                lower: 99.87,
            },
        };
        let bot_params = BotParams {
            close_grid_markup_range: 0.01,
            close_grid_min_markup: 0.01,
            close_grid_n_orders: 5.0,
            close_trailing_drawdown_pct: 0.02,
            close_trailing_grid_ratio: 0.5,
            close_trailing_threshold_pct: 0.01,
            entry_grid_double_down_factor: 2.0,
            entry_grid_spacing_weight: 0.5,
            entry_grid_spacing_pct: 0.03,
            entry_initial_ema_dist: 0.002,
            entry_initial_qty_pct: 0.015,
            entry_trailing_drawdown_pct: 0.02,
            entry_trailing_grid_ratio: 0.5,
            entry_trailing_threshold_pct: 0.01,
            n_positions: 5,
            total_wallet_exposure_limit: 0.5,
            wallet_exposure_limit: 0.1,
            unstuck_close_pct: 0.01,
            unstuck_ema_dist: 0.02,
            unstuck_loss_allowance_pct: 0.01,
            unstuck_threshold: 0.05,
        };

        // Test case 1: Normal initial entry
        let position = Position {
            size: 0.0,
            price: 0.0,
        };
        let order =
            calc_next_grid_entry_long(&exchange_params, &state_params, &bot_params, &position);
        assert!(order.qty > 0.0);
        let target_ientry_price = round_dn(
            f64::min(
                state_params.order_book.bid,
                state_params.ema_bands.lower * (1.0 - bot_params.entry_initial_ema_dist),
            ),
            exchange_params.price_step,
        );
        assert_eq!(order.price, target_ientry_price);
        assert_eq!(order.description, "long_ientry_normal");

        // Test case 2: Partial initial entry
        let position = Position {
            size: 0.01,
            price: target_ientry_price,
        };
        let order =
            calc_next_grid_entry_long(&exchange_params, &state_params, &bot_params, &position);
        assert!(order.qty > 0.0);
        assert_eq!(order.price, target_ientry_price);
        assert_eq!(order.description, "long_ientry_partial");

        // Test case 3: No initial entry (position size is large enough)
        let position = Position {
            size: 1.0,
            price: 100.0,
        };
        let order =
            calc_next_grid_entry_long(&exchange_params, &state_params, &bot_params, &position);
        assert_eq!(order.qty, 0.0);
        assert_eq!(order.price, 0.0);
        assert_eq!(order.description, "");

        // Test case 4: Normal re-entry
        let position = Position {
            size: 0.2,
            price: 100.0,
        };
        let order =
            calc_next_grid_entry_long(&exchange_params, &state_params, &bot_params, &position);
        assert!(order.qty > 0.0);
        assert!(order.price < position.price);
        assert_eq!(order.description, "long_reentry_normal");

        // Test case 5: Re-entry too big (cropped)
        let position = Position {
            size: 0.98,
            price: 100.0,
        };
        let order =
            calc_next_grid_entry_long(&exchange_params, &state_params, &bot_params, &position);
        assert!(order.qty > 0.0);
        assert!(order.price < position.price);
        assert_eq!(order.description, "long_reentry_cropped");

        // Test case 6: Next re-entry too small (inflated)
        let position = Position {
            size: 0.1,
            price: 100.0,
        };
        let order =
            calc_next_grid_entry_long(&exchange_params, &state_params, &bot_params, &position);
        assert!(order.qty > 0.0);
        assert!(order.price < position.price);
        //assert_eq!(order.description, "long_reentry_inflated");

        // Test case 8: Wallet exposure limit reached
        let mut state_params_updated = state_params.clone();
        state_params_updated.balance = 10.0;
        let position = Position {
            size: 1.0,
            price: 100.0,
        };
        let order = calc_next_grid_entry_long(
            &exchange_params,
            &state_params_updated,
            &bot_params,
            &position,
        );
        assert_eq!(order.qty, 0.0);
        assert_eq!(order.price, 0.0);
        assert_eq!(order.description, "");
    }
}
