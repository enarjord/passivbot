use crate::utils::{
    calc_min_entry_qty, calc_new_psize_pprice, calc_wallet_exposure_if_filled, cost_to_qty,
    qty_to_cost, round_, round_dn, round_up,
};

pub struct ExchangeParams {
    pub qty_step: f64,
    pub price_step: f64,
    pub min_qty: f64,
    pub min_cost: f64,
    pub c_mult: f64,
}

struct Position {
    size: f64,
    price: f64,
}

struct EMABands {
    upper: f64,
    lower: f64,
}

struct Order {
    qty: f64,
    price: f64,
    description: String,
}

struct OrderBook {
    bid: f64,
    ask: f64,
}

struct StateParams {
    balance: f64,
    position: Position,
    order_book: OrderBook,
    ema_bands: EMABands,
}

struct BotParams {
    close_grid_markup_range: f64,
    close_grid_min_markup: f64,
    close_grid_n_orders: f64,
    close_trailing_drawdown_pct: f64,
    close_trailing_grid_ratio: f64,
    close_trailing_threshold_pct: f64,
    entry_grid_double_down_factor: f64,
    entry_grid_spacing_weight: f64,
    entry_grid_spacing_pct: f64,
    entry_initial_ema_dist: f64,
    entry_initial_qty_pct: f64,
    entry_trailing_drawdown_pct: f64,
    entry_trailing_grid_ratio: f64,
    entry_trailing_threshold_pct: f64,
    n_positions: usize,
    total_wallet_exposure_limit: f64,
    wallet_exposure_limit: f64, // is total_wallet_exposure_limit / n_positions
    unstuck_close_pct: f64,
    unstuck_ema_dist: f64,
    unstuck_loss_allowance_pct: f64,
    unstuck_threshold: f64,
}

pub fn calc_entry_grid_lazy_long(
    exchange_params: ExchangeParams,
    state_params: StateParams,
    bot_params: BotParams,
) -> Order {
    // returns next entry order
    if bot_params.total_wallet_exposure_limit == 0.0 || bot_params.n_positions == 0 {
        return Order {
            qty: 0.0,
            price: 0.0,
            description: String::new(),
        };
    }

    let initial_entry_price = f64::min(
        state_params.order_book.bid,
        round_dn(
            state_params.ema_bands.lower * (1.0 - bot_params.entry_initial_ema_dist),
            exchange_params.price_step,
        ),
    );
    if initial_entry_price <= exchange_params.price_step {
        return return Order {
            qty: 0.0,
            price: 0.0,
            description: String::new(),
        };
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

    if state_params.position.size == 0.0 {
        // normal initial entry
        return Order {
            qty: initial_entry_qty,
            price: initial_entry_price,
            description: String::from("long_ientry_normal"),
        };
    } else if state_params.position.size < initial_entry_qty * 0.8 {
        // partial initial entry
        let entry_qty = f64::max(
            min_entry_qty,
            round_dn(
                initial_entry_qty - state_params.position.size,
                exchange_params.qty_step,
            ),
        );
        return Order {
            qty: entry_qty,
            price: initial_entry_price,
            description: String::from("long_ientry_partial"),
        };
    }

    let wallet_exposure = qty_to_cost(
        state_params.position.size,
        state_params.position.price,
        exchange_params.c_mult,
    ) / state_params.balance;
    if wallet_exposure >= bot_params.wallet_exposure_limit * 0.999 {
        return return Order {
            qty: 0.0,
            price: 0.0,
            description: String::new(),
        };
    }
    // normal re-entry
    let multiplier =
        (wallet_exposure / bot_params.wallet_exposure_limit) * bot_params.entry_grid_spacing_weight;
    let entry_price = round_dn(
        state_params.position.price
            * (1.0 - bot_params.entry_grid_spacing_pct * (1.0 + multiplier)),
        exchange_params.price_step,
    );
    let entry_price = f64::min(state_params.order_book.bid, entry_price);
    if entry_price <= exchange_params.price_step {
        return return Order {
            qty: 0.0,
            price: 0.0,
            description: String::new(),
        };
    }
    let entry_qty = calc_reentry_qty(
        entry_price,
        state_params.balance,
        state_params.position.size,
        &exchange_params,
        &bot_params,
    );
    let entry_qty = f64::max(initial_entry_qty, entry_qty);
    let wallet_exposure_if_filled = calc_wallet_exposure_if_filled(
        state_params.balance,
        state_params.position.size,
        state_params.position.price,
        entry_qty,
        entry_price,
        &exchange_params,
    );
    let adjust = false;
    if wallet_exposure_if_filled > bot_params.wallet_exposure_limit * 1.01 {
        // re-entry too big
        let adjust = true;
    } else {
        // preview next reentry

        let (new_psize, new_pprice) = calc_new_psize_pprice(
            state_params.position.size,
            state_params.position.price,
            entry_qty,
            entry_price,
            exchange_params.qty_step,
        );
        let new_wallet_exposure =
            qty_to_cost(new_psize, new_pprice, exchange_params.c_mult) / state_params.balance;
        let new_multiplier = (new_wallet_exposure / bot_params.wallet_exposure_limit)
            * bot_params.entry_grid_spacing_weight;
        let new_entry_price = round_dn(
            new_pprice * (1.0 - bot_params.entry_grid_spacing_pct * (1.0 + new_multiplier)),
            exchange_params.price_step,
        );
        let new_entry_qty = calc_reentry_qty(
            new_entry_price,
            state_params.balance,
            state_params.position.size,
            &exchange_params,
            &bot_params,
        );
        let wallet_exposure_if_next_filled = calc_wallet_exposure_if_filled(
            state_params.balance,
            new_psize,
            new_pprice,
            new_entry_qty,
            new_entry_price,
            &exchange_params,
        );
        /* begin python code

        # preview next reentry
        new_psize, new_pprice = calc_new_psize_pprice(
            psize, pprice, entry_qty, entry_price, qty_step
        )
        new_wallet_exposure = qty_to_cost(new_psize, new_pprice, inverse, c_mult) / balance
        new_multiplier = (
            new_wallet_exposure / wallet_exposure_limit
        ) * rentry_pprice_dist_wallet_exposure_weighting
        new_entry_price = round_dn(
            new_pprice * (1 - rentry_pprice_dist * (1 + new_multiplier)), price_step
        )
        new_entry_qty = calc_recursive_reentry_qty(
            balance,
            new_psize,
            new_entry_price,
            inverse,
            qty_step,
            min_qty,
            min_cost,
            c_mult,
            initial_qty_pct,
            ddown_factor,
            wallet_exposure_limit,
        )
        wallet_exposure_if_next_filled = calc_wallet_exposure_if_filled(
            balance,
            new_psize,
            new_pprice,
            new_entry_qty,
            new_entry_price,
            inverse,
            c_mult,
            qty_step,
        )
        if wallet_exposure_if_next_filled > wallet_exposure_limit * 1.2:
            # reentry too small
            adjust = True
        // end python code
        */
    }

    Order {
        qty: 0.0,
        price: initial_entry_price,
        description: String::from("entry order"),
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
