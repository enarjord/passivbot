use super::{
    GeneratedOrders, StrategyParams, StrategyRequest, StrategySide, TrailingGridV7CloseParams,
    TrailingGridV7EntryParams, TrailingGridV7Params,
};
use crate::closes::{
    calc_close_qty, calc_wel_auto_reduce_long, calc_wel_auto_reduce_short, sort_closes_by_price,
};
use crate::entries::{
    calc_cropped_reentry_qty, calc_min_entry_qty, wallet_exposure_limit_with_allowance,
};
use crate::types::{
    BotParams, ExchangeParams, Order, OrderType, Position, RuntimeOrderContext, StateParams,
    TrailingPriceBundle,
};
use crate::utils::{
    calc_ema_price_ask, calc_ema_price_bid, calc_new_psize_pprice, calc_wallet_exposure,
    cost_to_qty, quantize_price, quantize_qty, round_, round_dn, round_up, RoundingMode,
};

fn push_if_nonzero(out: &mut Vec<Order>, order: Order) {
    if order.qty != 0.0 {
        out.push(order);
    }
}

fn extend_nonzero(out: &mut Vec<Order>, orders: Vec<Order>) {
    for order in orders {
        push_if_nonzero(out, order);
    }
}

#[inline]
fn would_fill_next_candle(low: f64, high: f64, qty: f64, price: f64) -> bool {
    if qty > 0.0 {
        low < price
    } else if qty < 0.0 {
        high > price
    } else {
        false
    }
}

#[inline]
fn any_order_would_fill_next_candle(low: f64, high: f64, orders: &[Order]) -> bool {
    orders
        .iter()
        .any(|order| would_fill_next_candle(low, high, order.qty, order.price))
}

fn calc_initial_entry_qty(
    exchange: &ExchangeParams,
    bot: &BotParams,
    runtime: &RuntimeOrderContext,
    entry: &TrailingGridV7EntryParams,
    balance: f64,
    entry_price: f64,
) -> f64 {
    f64::max(
        calc_min_entry_qty(entry_price, exchange),
        round_(
            cost_to_qty(
                balance
                    * wallet_exposure_limit_with_allowance(bot, runtime)
                    * entry.initial_qty_pct,
                entry_price,
                exchange.c_mult,
            ),
            exchange.qty_step,
        ),
    )
}

fn calc_reentry_qty(
    entry_price: f64,
    balance: f64,
    position_size: f64,
    double_down_factor: f64,
    exchange: &ExchangeParams,
    bot: &BotParams,
    runtime: &RuntimeOrderContext,
    entry: &TrailingGridV7EntryParams,
    wallet_exposure_limit_cap: f64,
) -> f64 {
    let effective_wallet_exposure_limit = f64::min(
        wallet_exposure_limit_cap,
        wallet_exposure_limit_with_allowance(bot, runtime),
    );
    f64::max(
        calc_min_entry_qty(entry_price, exchange),
        round_(
            f64::max(
                position_size.abs() * double_down_factor,
                cost_to_qty(balance, entry_price, exchange.c_mult)
                    * effective_wallet_exposure_limit
                    * entry.initial_qty_pct,
            ),
            exchange.qty_step,
        ),
    )
}

fn calc_reentry_price_bid(
    position_price: f64,
    wallet_exposure: f64,
    order_book_bid: f64,
    exchange: &ExchangeParams,
    bot: &BotParams,
    runtime: &RuntimeOrderContext,
    entry: &TrailingGridV7EntryParams,
    volatility_ema_1h: f64,
    wallet_exposure_limit_cap: f64,
) -> f64 {
    let effective_wallet_exposure_limit = f64::min(
        wallet_exposure_limit_cap,
        wallet_exposure_limit_with_allowance(bot, runtime),
    );
    let we_multiplier = if effective_wallet_exposure_limit > 0.0 {
        (wallet_exposure / effective_wallet_exposure_limit) * entry.grid_spacing_we_weight
    } else {
        0.0
    };
    let log_multiplier = volatility_ema_1h * entry.grid_spacing_volatility_weight;
    let spacing_multiplier = 1.0 + we_multiplier + log_multiplier;
    let reentry_price = f64::min(
        round_dn(
            position_price * (1.0 - entry.grid_spacing_pct * spacing_multiplier.max(0.0)),
            exchange.price_step,
        ),
        order_book_bid,
    );
    if reentry_price <= exchange.price_step {
        0.0
    } else {
        reentry_price
    }
}

fn calc_reentry_price_ask(
    position_price: f64,
    wallet_exposure: f64,
    order_book_ask: f64,
    exchange: &ExchangeParams,
    bot: &BotParams,
    runtime: &RuntimeOrderContext,
    entry: &TrailingGridV7EntryParams,
    volatility_ema_1h: f64,
    wallet_exposure_limit_cap: f64,
) -> f64 {
    let effective_wallet_exposure_limit = f64::min(
        wallet_exposure_limit_cap,
        wallet_exposure_limit_with_allowance(bot, runtime),
    );
    let we_multiplier = if effective_wallet_exposure_limit > 0.0 {
        (wallet_exposure / effective_wallet_exposure_limit) * entry.grid_spacing_we_weight
    } else {
        0.0
    };
    let log_multiplier = volatility_ema_1h * entry.grid_spacing_volatility_weight;
    let spacing_multiplier = 1.0 + we_multiplier + log_multiplier;
    let reentry_price = f64::max(
        round_up(
            position_price * (1.0 + entry.grid_spacing_pct * spacing_multiplier.max(0.0)),
            exchange.price_step,
        ),
        order_book_ask,
    );
    if reentry_price <= exchange.price_step {
        0.0
    } else {
        reentry_price
    }
}

fn calc_grid_entry_long(
    exchange: &ExchangeParams,
    state: &StateParams,
    bot: &BotParams,
    runtime: &RuntimeOrderContext,
    entry: &TrailingGridV7EntryParams,
    position: &Position,
    wallet_exposure_limit_cap: f64,
) -> Order {
    if wallet_exposure_limit_with_allowance(bot, runtime) == 0.0 || state.balance <= 0.0 {
        return Order::default();
    }
    let initial_entry_price = calc_ema_price_bid(
        exchange.price_step,
        state.order_book.bid,
        state.ema_bands.lower,
        entry.initial_ema_dist,
    );
    if initial_entry_price <= exchange.price_step {
        return Order::default();
    }
    let mut initial_entry_qty = calc_initial_entry_qty(
        exchange,
        bot,
        runtime,
        entry,
        state.balance,
        initial_entry_price,
    );
    if position.size == 0.0 {
        return Order {
            qty: initial_entry_qty,
            price: initial_entry_price,
            order_type: OrderType::EntryInitialNormalLong,
        };
    } else if position.size < initial_entry_qty * 0.8 {
        return Order {
            qty: f64::max(
                calc_min_entry_qty(initial_entry_price, exchange),
                round_dn(initial_entry_qty - position.size, exchange.qty_step),
            ),
            price: initial_entry_price,
            order_type: OrderType::EntryInitialPartialLong,
        };
    } else if position.size < initial_entry_qty {
        initial_entry_qty = round_(position.size, exchange.qty_step)
            .max(calc_min_entry_qty(initial_entry_price, exchange));
    }
    let wallet_exposure = calc_wallet_exposure(
        exchange.c_mult,
        state.balance,
        position.size,
        position.price,
    );
    let effective_wallet_exposure_limit = f64::min(
        wallet_exposure_limit_cap,
        wallet_exposure_limit_with_allowance(bot, runtime),
    );
    if wallet_exposure >= effective_wallet_exposure_limit * 0.999 {
        return Order::default();
    }
    let reentry_price = calc_reentry_price_bid(
        position.price,
        wallet_exposure,
        state.order_book.bid,
        exchange,
        bot,
        runtime,
        entry,
        state.volatility_ema_1h,
        effective_wallet_exposure_limit,
    );
    if reentry_price <= 0.0 {
        return Order::default();
    }
    let reentry_qty = f64::max(
        calc_reentry_qty(
            reentry_price,
            state.balance,
            position.size,
            entry.grid_double_down_factor,
            exchange,
            bot,
            runtime,
            entry,
            effective_wallet_exposure_limit,
        ),
        initial_entry_qty,
    );
    let (_, reentry_qty_cropped) = calc_cropped_reentry_qty(
        exchange,
        bot,
        runtime,
        position,
        wallet_exposure,
        state.balance,
        reentry_qty,
        reentry_price,
        effective_wallet_exposure_limit,
    );
    if reentry_qty_cropped < reentry_qty {
        return Order {
            qty: reentry_qty_cropped,
            price: reentry_price,
            order_type: OrderType::EntryGridCroppedLong,
        };
    }
    Order {
        qty: reentry_qty,
        price: reentry_price,
        order_type: OrderType::EntryGridNormalLong,
    }
}

fn calc_trailing_entry_long(
    exchange: &ExchangeParams,
    state: &StateParams,
    bot: &BotParams,
    runtime: &RuntimeOrderContext,
    entry: &TrailingGridV7EntryParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
    wallet_exposure_limit_cap: f64,
) -> Order {
    let initial_entry_price = calc_ema_price_bid(
        exchange.price_step,
        state.order_book.bid,
        state.ema_bands.lower,
        entry.initial_ema_dist,
    );
    if initial_entry_price <= exchange.price_step {
        return Order::default();
    }
    let mut initial_entry_qty = calc_initial_entry_qty(
        exchange,
        bot,
        runtime,
        entry,
        state.balance,
        initial_entry_price,
    );
    if position.size == 0.0 {
        return Order {
            qty: initial_entry_qty,
            price: initial_entry_price,
            order_type: OrderType::EntryInitialNormalLong,
        };
    } else if position.size < initial_entry_qty * 0.8 {
        return Order {
            qty: f64::max(
                calc_min_entry_qty(initial_entry_price, exchange),
                round_dn(initial_entry_qty - position.size, exchange.qty_step),
            ),
            price: initial_entry_price,
            order_type: OrderType::EntryInitialPartialLong,
        };
    } else if position.size < initial_entry_qty {
        initial_entry_qty = round_(position.size, exchange.qty_step)
            .max(calc_min_entry_qty(initial_entry_price, exchange));
    }
    let wallet_exposure = calc_wallet_exposure(
        exchange.c_mult,
        state.balance,
        position.size,
        position.price,
    );
    let effective_wallet_exposure_limit = f64::min(
        wallet_exposure_limit_cap,
        wallet_exposure_limit_with_allowance(bot, runtime),
    );
    if wallet_exposure > effective_wallet_exposure_limit * 0.999 {
        return Order::default();
    }
    let threshold_multiplier = if effective_wallet_exposure_limit > 0.0 {
        (wallet_exposure / effective_wallet_exposure_limit) * entry.trailing_threshold_we_weight
    } else {
        0.0
    };
    let threshold_log_multiplier =
        state.volatility_ema_1h * entry.trailing_threshold_volatility_weight;
    let threshold_pct = entry.trailing_threshold_pct
        * (1.0 + threshold_multiplier + threshold_log_multiplier).max(0.0);
    let retracement_multiplier = if effective_wallet_exposure_limit > 0.0 {
        (wallet_exposure / effective_wallet_exposure_limit) * entry.trailing_retracement_we_weight
    } else {
        0.0
    };
    let retracement_log_multiplier =
        state.volatility_ema_1h * entry.trailing_retracement_volatility_weight;
    let retracement_pct = entry.trailing_retracement_pct
        * (1.0 + retracement_multiplier + retracement_log_multiplier).max(0.0);
    let mut entry_triggered = false;
    let mut reentry_price = 0.0;
    if threshold_pct <= 0.0 {
        if retracement_pct > 0.0
            && trailing.max_since_min > trailing.min_since_open * (1.0 + retracement_pct)
        {
            entry_triggered = true;
            reentry_price = state.order_book.bid;
        }
    } else if retracement_pct <= 0.0 {
        entry_triggered = true;
        reentry_price = f64::min(
            state.order_book.bid,
            round_dn(position.price * (1.0 - threshold_pct), exchange.price_step),
        );
    } else if trailing.min_since_open < position.price * (1.0 - threshold_pct)
        && trailing.max_since_min > trailing.min_since_open * (1.0 + retracement_pct)
    {
        entry_triggered = true;
        reentry_price = f64::min(
            state.order_book.bid,
            round_dn(
                position.price * (1.0 - threshold_pct + retracement_pct),
                exchange.price_step,
            ),
        );
    }
    if !entry_triggered {
        return Order {
            qty: 0.0,
            price: 0.0,
            order_type: OrderType::EntryTrailingNormalLong,
        };
    }
    let reentry_qty = f64::max(
        calc_reentry_qty(
            reentry_price,
            state.balance,
            position.size,
            entry.trailing_double_down_factor,
            exchange,
            bot,
            runtime,
            entry,
            effective_wallet_exposure_limit,
        ),
        initial_entry_qty,
    );
    let (_, reentry_qty_cropped) = calc_cropped_reentry_qty(
        exchange,
        bot,
        runtime,
        position,
        wallet_exposure,
        state.balance,
        reentry_qty,
        reentry_price,
        effective_wallet_exposure_limit,
    );
    if reentry_qty_cropped < reentry_qty {
        Order {
            qty: reentry_qty_cropped,
            price: reentry_price,
            order_type: OrderType::EntryTrailingCroppedLong,
        }
    } else {
        Order {
            qty: reentry_qty,
            price: reentry_price,
            order_type: OrderType::EntryTrailingNormalLong,
        }
    }
}

fn calc_next_entry_long(
    exchange: &ExchangeParams,
    state: &StateParams,
    bot: &BotParams,
    runtime: &RuntimeOrderContext,
    entry: &TrailingGridV7EntryParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
) -> Order {
    let base_wallet_exposure_limit = wallet_exposure_limit_with_allowance(bot, runtime);
    if base_wallet_exposure_limit == 0.0 || state.balance <= 0.0 {
        return Order::default();
    }
    let allowed_wallet_exposure_limit = base_wallet_exposure_limit;
    if entry.trailing_grid_ratio >= 1.0 || entry.trailing_grid_ratio <= -1.0 {
        return calc_trailing_entry_long(
            exchange,
            state,
            bot,
            runtime,
            entry,
            position,
            trailing,
            allowed_wallet_exposure_limit,
        );
    } else if entry.trailing_grid_ratio == 0.0 {
        return calc_grid_entry_long(
            exchange,
            state,
            bot,
            runtime,
            entry,
            position,
            allowed_wallet_exposure_limit,
        );
    }
    let wallet_exposure = calc_wallet_exposure(
        exchange.c_mult,
        state.balance,
        position.size,
        position.price,
    );
    let wallet_exposure_ratio = wallet_exposure / base_wallet_exposure_limit;
    if entry.trailing_grid_ratio > 0.0 {
        if wallet_exposure_ratio < entry.trailing_grid_ratio {
            let cap = if wallet_exposure == 0.0 {
                allowed_wallet_exposure_limit
            } else {
                (base_wallet_exposure_limit * entry.trailing_grid_ratio * 1.01)
                    .min(allowed_wallet_exposure_limit)
            };
            calc_trailing_entry_long(
                exchange, state, bot, runtime, entry, position, trailing, cap,
            )
        } else {
            calc_grid_entry_long(
                exchange,
                state,
                bot,
                runtime,
                entry,
                position,
                allowed_wallet_exposure_limit,
            )
        }
    } else if wallet_exposure_ratio < 1.0 + entry.trailing_grid_ratio {
        let cap = if wallet_exposure == 0.0 {
            allowed_wallet_exposure_limit
        } else {
            (base_wallet_exposure_limit * (1.0 + entry.trailing_grid_ratio) * 1.01)
                .min(allowed_wallet_exposure_limit)
        };
        calc_grid_entry_long(exchange, state, bot, runtime, entry, position, cap)
    } else {
        calc_trailing_entry_long(
            exchange,
            state,
            bot,
            runtime,
            entry,
            position,
            trailing,
            allowed_wallet_exposure_limit,
        )
    }
}

fn calc_grid_entry_short(
    exchange: &ExchangeParams,
    state: &StateParams,
    bot: &BotParams,
    runtime: &RuntimeOrderContext,
    entry: &TrailingGridV7EntryParams,
    position: &Position,
    wallet_exposure_limit_cap: f64,
) -> Order {
    if wallet_exposure_limit_with_allowance(bot, runtime) == 0.0 || state.balance <= 0.0 {
        return Order::default();
    }
    let initial_entry_price = calc_ema_price_ask(
        exchange.price_step,
        state.order_book.ask,
        state.ema_bands.upper,
        entry.initial_ema_dist,
    );
    if initial_entry_price <= exchange.price_step {
        return Order::default();
    }
    let mut initial_entry_qty = calc_initial_entry_qty(
        exchange,
        bot,
        runtime,
        entry,
        state.balance,
        initial_entry_price,
    );
    let position_size_abs = position.size.abs();
    if position_size_abs == 0.0 {
        return Order {
            qty: -initial_entry_qty,
            price: initial_entry_price,
            order_type: OrderType::EntryInitialNormalShort,
        };
    } else if position_size_abs < initial_entry_qty * 0.8 {
        return Order {
            qty: -f64::max(
                calc_min_entry_qty(initial_entry_price, exchange),
                round_dn(initial_entry_qty - position_size_abs, exchange.qty_step),
            ),
            price: initial_entry_price,
            order_type: OrderType::EntryInitialPartialShort,
        };
    } else if position_size_abs < initial_entry_qty {
        initial_entry_qty = round_(position_size_abs, exchange.qty_step)
            .max(calc_min_entry_qty(initial_entry_price, exchange));
    }
    let wallet_exposure = calc_wallet_exposure(
        exchange.c_mult,
        state.balance,
        position_size_abs,
        position.price,
    );
    let effective_wallet_exposure_limit = f64::min(
        wallet_exposure_limit_cap,
        wallet_exposure_limit_with_allowance(bot, runtime),
    );
    if wallet_exposure >= effective_wallet_exposure_limit * 0.999 {
        return Order::default();
    }
    let reentry_price = calc_reentry_price_ask(
        position.price,
        wallet_exposure,
        state.order_book.ask,
        exchange,
        bot,
        runtime,
        entry,
        state.volatility_ema_1h,
        effective_wallet_exposure_limit,
    );
    if reentry_price <= 0.0 {
        return Order::default();
    }
    let reentry_qty = f64::max(
        calc_reentry_qty(
            reentry_price,
            state.balance,
            position_size_abs,
            entry.grid_double_down_factor,
            exchange,
            bot,
            runtime,
            entry,
            effective_wallet_exposure_limit,
        ),
        initial_entry_qty,
    );
    let (_, reentry_qty_cropped) = calc_cropped_reentry_qty(
        exchange,
        bot,
        runtime,
        position,
        wallet_exposure,
        state.balance,
        reentry_qty,
        reentry_price,
        effective_wallet_exposure_limit,
    );
    if reentry_qty_cropped < reentry_qty {
        return Order {
            qty: -reentry_qty_cropped,
            price: reentry_price,
            order_type: OrderType::EntryGridCroppedShort,
        };
    }
    Order {
        qty: -reentry_qty,
        price: reentry_price,
        order_type: OrderType::EntryGridNormalShort,
    }
}

fn calc_trailing_entry_short(
    exchange: &ExchangeParams,
    state: &StateParams,
    bot: &BotParams,
    runtime: &RuntimeOrderContext,
    entry: &TrailingGridV7EntryParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
    wallet_exposure_limit_cap: f64,
) -> Order {
    let initial_entry_price = calc_ema_price_ask(
        exchange.price_step,
        state.order_book.ask,
        state.ema_bands.upper,
        entry.initial_ema_dist,
    );
    if initial_entry_price <= exchange.price_step {
        return Order::default();
    }
    let mut initial_entry_qty = calc_initial_entry_qty(
        exchange,
        bot,
        runtime,
        entry,
        state.balance,
        initial_entry_price,
    );
    let position_size_abs = position.size.abs();
    if position_size_abs == 0.0 {
        return Order {
            qty: -initial_entry_qty,
            price: initial_entry_price,
            order_type: OrderType::EntryInitialNormalShort,
        };
    } else if position_size_abs < initial_entry_qty * 0.8 {
        return Order {
            qty: -f64::max(
                calc_min_entry_qty(initial_entry_price, exchange),
                round_dn(initial_entry_qty - position_size_abs, exchange.qty_step),
            ),
            price: initial_entry_price,
            order_type: OrderType::EntryInitialPartialShort,
        };
    } else if position_size_abs < initial_entry_qty {
        initial_entry_qty = round_(position_size_abs, exchange.qty_step)
            .max(calc_min_entry_qty(initial_entry_price, exchange));
    }
    let wallet_exposure = calc_wallet_exposure(
        exchange.c_mult,
        state.balance,
        position_size_abs,
        position.price,
    );
    let effective_wallet_exposure_limit = f64::min(
        wallet_exposure_limit_cap,
        wallet_exposure_limit_with_allowance(bot, runtime),
    );
    if wallet_exposure > effective_wallet_exposure_limit * 0.999 {
        return Order::default();
    }
    let threshold_multiplier = if effective_wallet_exposure_limit > 0.0 {
        (wallet_exposure / effective_wallet_exposure_limit) * entry.trailing_threshold_we_weight
    } else {
        0.0
    };
    let threshold_log_multiplier =
        state.volatility_ema_1h * entry.trailing_threshold_volatility_weight;
    let threshold_pct = entry.trailing_threshold_pct
        * (1.0 + threshold_multiplier + threshold_log_multiplier).max(0.0);
    let retracement_multiplier = if effective_wallet_exposure_limit > 0.0 {
        (wallet_exposure / effective_wallet_exposure_limit) * entry.trailing_retracement_we_weight
    } else {
        0.0
    };
    let retracement_log_multiplier =
        state.volatility_ema_1h * entry.trailing_retracement_volatility_weight;
    let retracement_pct = entry.trailing_retracement_pct
        * (1.0 + retracement_multiplier + retracement_log_multiplier).max(0.0);
    let mut entry_triggered = false;
    let mut reentry_price = 0.0;
    if threshold_pct <= 0.0 {
        if retracement_pct > 0.0
            && trailing.min_since_max < trailing.max_since_open * (1.0 - retracement_pct)
        {
            entry_triggered = true;
            reentry_price = state.order_book.ask;
        }
    } else if retracement_pct <= 0.0 {
        entry_triggered = true;
        reentry_price = f64::max(
            state.order_book.ask,
            round_up(position.price * (1.0 + threshold_pct), exchange.price_step),
        );
    } else if trailing.max_since_open > position.price * (1.0 + threshold_pct)
        && trailing.min_since_max < trailing.max_since_open * (1.0 - retracement_pct)
    {
        entry_triggered = true;
        reentry_price = f64::max(
            state.order_book.ask,
            round_up(
                position.price * (1.0 + threshold_pct - retracement_pct),
                exchange.price_step,
            ),
        );
    }
    if !entry_triggered {
        return Order {
            qty: 0.0,
            price: 0.0,
            order_type: OrderType::EntryTrailingNormalShort,
        };
    }
    let reentry_qty = f64::max(
        calc_reentry_qty(
            reentry_price,
            state.balance,
            position_size_abs,
            entry.trailing_double_down_factor,
            exchange,
            bot,
            runtime,
            entry,
            effective_wallet_exposure_limit,
        ),
        initial_entry_qty,
    );
    let (_, reentry_qty_cropped) = calc_cropped_reentry_qty(
        exchange,
        bot,
        runtime,
        position,
        wallet_exposure,
        state.balance,
        reentry_qty,
        reentry_price,
        effective_wallet_exposure_limit,
    );
    if reentry_qty_cropped < reentry_qty {
        Order {
            qty: -reentry_qty_cropped,
            price: reentry_price,
            order_type: OrderType::EntryTrailingCroppedShort,
        }
    } else {
        Order {
            qty: -reentry_qty,
            price: reentry_price,
            order_type: OrderType::EntryTrailingNormalShort,
        }
    }
}

fn calc_next_entry_short(
    exchange: &ExchangeParams,
    state: &StateParams,
    bot: &BotParams,
    runtime: &RuntimeOrderContext,
    entry: &TrailingGridV7EntryParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
) -> Order {
    let base_wallet_exposure_limit = wallet_exposure_limit_with_allowance(bot, runtime);
    if base_wallet_exposure_limit == 0.0 || state.balance <= 0.0 {
        return Order::default();
    }
    let allowed_wallet_exposure_limit = base_wallet_exposure_limit;
    if entry.trailing_grid_ratio >= 1.0 || entry.trailing_grid_ratio <= -1.0 {
        return calc_trailing_entry_short(
            exchange,
            state,
            bot,
            runtime,
            entry,
            position,
            trailing,
            allowed_wallet_exposure_limit,
        );
    } else if entry.trailing_grid_ratio == 0.0 {
        return calc_grid_entry_short(
            exchange,
            state,
            bot,
            runtime,
            entry,
            position,
            allowed_wallet_exposure_limit,
        );
    }
    let wallet_exposure = calc_wallet_exposure(
        exchange.c_mult,
        state.balance,
        position.size.abs(),
        position.price,
    );
    let wallet_exposure_ratio = wallet_exposure / base_wallet_exposure_limit;
    if entry.trailing_grid_ratio > 0.0 {
        if wallet_exposure_ratio < entry.trailing_grid_ratio {
            let cap = if wallet_exposure == 0.0 {
                allowed_wallet_exposure_limit
            } else {
                (base_wallet_exposure_limit * entry.trailing_grid_ratio * 1.01)
                    .min(allowed_wallet_exposure_limit)
            };
            calc_trailing_entry_short(
                exchange, state, bot, runtime, entry, position, trailing, cap,
            )
        } else {
            calc_grid_entry_short(
                exchange,
                state,
                bot,
                runtime,
                entry,
                position,
                allowed_wallet_exposure_limit,
            )
        }
    } else if wallet_exposure_ratio < 1.0 + entry.trailing_grid_ratio {
        let cap = if wallet_exposure == 0.0 {
            allowed_wallet_exposure_limit
        } else {
            (base_wallet_exposure_limit * (1.0 + entry.trailing_grid_ratio) * 1.01)
                .min(allowed_wallet_exposure_limit)
        };
        calc_grid_entry_short(exchange, state, bot, runtime, entry, position, cap)
    } else {
        calc_trailing_entry_short(
            exchange,
            state,
            bot,
            runtime,
            entry,
            position,
            trailing,
            allowed_wallet_exposure_limit,
        )
    }
}

fn calc_grid_close_long(
    exchange: &ExchangeParams,
    state: &StateParams,
    bot: &BotParams,
    runtime: &RuntimeOrderContext,
    close: &TrailingGridV7CloseParams,
    position: &Position,
) -> Order {
    if position.size <= 0.0 {
        return Order::default();
    }
    if close.grid_qty_pct < 0.0 || close.grid_qty_pct >= 1.0 {
        return Order {
            qty: -round_(position.size, exchange.qty_step),
            price: f64::max(
                state.order_book.ask,
                round_up(
                    position.price * (1.0 + close.grid_markup_start),
                    exchange.price_step,
                ),
            ),
            order_type: OrderType::CloseGridLong,
        };
    }
    let close_prices_start = round_up(
        position.price * (1.0 + close.grid_markup_start),
        exchange.price_step,
    );
    let close_prices_end = round_up(
        position.price * (1.0 + close.grid_markup_end),
        exchange.price_step,
    );
    if close_prices_start == close_prices_end {
        return Order {
            qty: -round_(position.size, exchange.qty_step),
            price: f64::max(state.order_book.ask, close_prices_start),
            order_type: OrderType::CloseGridLong,
        };
    }
    let n_steps = ((close_prices_end - close_prices_start).abs() / exchange.price_step).ceil();
    let close_grid_qty_pct_modified = f64::max(close.grid_qty_pct, 1.0 / n_steps);
    let wallet_exposure = calc_wallet_exposure(
        exchange.c_mult,
        state.balance,
        position.size,
        position.price,
    );
    let allowed_limit = wallet_exposure_limit_with_allowance(bot, runtime);
    let wallet_exposure_ratio = if allowed_limit > 0.0 {
        wallet_exposure / allowed_limit
    } else {
        10.0
    };
    let close_price = if wallet_exposure_ratio > 1.0 {
        f64::max(
            state.order_book.ask,
            f64::min(close_prices_start, close_prices_end),
        )
    } else {
        f64::max(
            state.order_book.ask,
            round_up(
                close_prices_start
                    + (close_prices_end - close_prices_start)
                        * f64::min(1.0, wallet_exposure_ratio),
                exchange.price_step,
            ),
        )
    };
    let close_qty = -calc_close_qty(
        exchange,
        bot,
        runtime,
        position,
        close_grid_qty_pct_modified,
        state.balance,
        close_price,
    );
    Order {
        qty: close_qty,
        price: close_price,
        order_type: OrderType::CloseGridLong,
    }
}

fn calc_trailing_close_long(
    exchange: &ExchangeParams,
    state: &StateParams,
    bot: &BotParams,
    runtime: &RuntimeOrderContext,
    close: &TrailingGridV7CloseParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
) -> Order {
    if position.size == 0.0 {
        return Order::default();
    }
    if close.trailing_threshold_pct <= 0.0 {
        if close.trailing_retracement_pct > 0.0
            && trailing.min_since_max
                < trailing.max_since_open * (1.0 - close.trailing_retracement_pct)
        {
            Order {
                qty: -calc_close_qty(
                    exchange,
                    bot,
                    runtime,
                    position,
                    close.trailing_qty_pct,
                    state.balance,
                    state.order_book.ask,
                ),
                price: state.order_book.ask,
                order_type: OrderType::CloseTrailingLong,
            }
        } else {
            Order {
                qty: 0.0,
                price: 0.0,
                order_type: OrderType::CloseTrailingLong,
            }
        }
    } else if close.trailing_retracement_pct <= 0.0 {
        let close_price = f64::max(
            state.order_book.ask,
            round_up(
                position.price * (1.0 + close.trailing_threshold_pct),
                exchange.price_step,
            ),
        );
        Order {
            qty: -calc_close_qty(
                exchange,
                bot,
                runtime,
                position,
                close.trailing_qty_pct,
                state.balance,
                close_price,
            ),
            price: close_price,
            order_type: OrderType::CloseTrailingLong,
        }
    } else if trailing.max_since_open > position.price * (1.0 + close.trailing_threshold_pct)
        && trailing.min_since_max < trailing.max_since_open * (1.0 - close.trailing_retracement_pct)
    {
        let close_price = f64::max(
            state.order_book.ask,
            round_up(
                position.price
                    * (1.0 + close.trailing_threshold_pct - close.trailing_retracement_pct),
                exchange.price_step,
            ),
        );
        Order {
            qty: -calc_close_qty(
                exchange,
                bot,
                runtime,
                position,
                close.trailing_qty_pct,
                state.balance,
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

fn calc_next_close_long(
    exchange: &ExchangeParams,
    state: &StateParams,
    bot: &BotParams,
    runtime: &RuntimeOrderContext,
    close: &TrailingGridV7CloseParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
) -> Order {
    if position.size == 0.0 {
        return Order::default();
    }
    let wallet_exposure = calc_wallet_exposure(
        exchange.c_mult,
        state.balance,
        position.size,
        position.price,
    );
    if let Some(order) =
        calc_wel_auto_reduce_long(exchange, state, bot, runtime, position, wallet_exposure)
    {
        return order;
    }
    let allowed_limit = wallet_exposure_limit_with_allowance(bot, runtime);
    let wallet_exposure_ratio = if allowed_limit <= 0.0 {
        10.0
    } else {
        wallet_exposure / allowed_limit
    };
    if close.trailing_grid_ratio >= 1.0 || close.trailing_grid_ratio <= -1.0 {
        return calc_trailing_close_long(exchange, state, bot, runtime, close, position, trailing);
    }
    if close.trailing_grid_ratio == 0.0 {
        return calc_grid_close_long(exchange, state, bot, runtime, close, position);
    }
    if close.trailing_grid_ratio > 0.0 {
        if wallet_exposure_ratio < close.trailing_grid_ratio {
            calc_trailing_close_long(exchange, state, bot, runtime, close, position, trailing)
        } else {
            let mut trailing_allocation = cost_to_qty(
                state.balance * allowed_limit * close.trailing_grid_ratio,
                position.price,
                exchange.c_mult,
            );
            let min_entry_qty = calc_min_entry_qty(position.price, exchange);
            if trailing_allocation < min_entry_qty {
                trailing_allocation = 0.0;
            }
            let grid_allocation = round_(
                (position.size - trailing_allocation) * 1.01,
                exchange.qty_step,
            );
            let position_mod = Position {
                size: f64::min(position.size, f64::max(grid_allocation, min_entry_qty)),
                price: position.price,
            };
            calc_grid_close_long(exchange, state, bot, runtime, close, &position_mod)
        }
    } else if wallet_exposure_ratio < 1.0 + close.trailing_grid_ratio {
        calc_grid_close_long(exchange, state, bot, runtime, close, position)
    } else {
        let mut grid_allocation = cost_to_qty(
            state.balance * allowed_limit * (1.0 + close.trailing_grid_ratio),
            position.price,
            exchange.c_mult,
        );
        let min_entry_qty = calc_min_entry_qty(position.price, exchange);
        if grid_allocation < min_entry_qty {
            grid_allocation = 0.0;
        }
        let trailing_allocation =
            round_((position.size - grid_allocation) * 1.01, exchange.qty_step);
        let position_mod = Position {
            size: f64::min(position.size, f64::max(trailing_allocation, min_entry_qty)),
            price: position.price,
        };
        calc_trailing_close_long(
            exchange,
            state,
            bot,
            runtime,
            close,
            &position_mod,
            trailing,
        )
    }
}

fn calc_grid_close_short(
    exchange: &ExchangeParams,
    state: &StateParams,
    bot: &BotParams,
    runtime: &RuntimeOrderContext,
    close: &TrailingGridV7CloseParams,
    position: &Position,
) -> Order {
    let position_size_abs = position.size.abs();
    if position_size_abs == 0.0 {
        return Order::default();
    }
    if close.grid_qty_pct < 0.0 || close.grid_qty_pct >= 1.0 {
        return Order {
            qty: round_(position_size_abs, exchange.qty_step),
            price: f64::min(
                state.order_book.bid,
                round_dn(
                    position.price * (1.0 - close.grid_markup_start),
                    exchange.price_step,
                ),
            ),
            order_type: OrderType::CloseGridShort,
        };
    }
    let close_prices_start = round_dn(
        position.price * (1.0 - close.grid_markup_start),
        exchange.price_step,
    );
    let close_prices_end = round_dn(
        position.price * (1.0 - close.grid_markup_end),
        exchange.price_step,
    );
    if close_prices_start == close_prices_end {
        return Order {
            qty: round_(position_size_abs, exchange.qty_step),
            price: f64::min(state.order_book.bid, close_prices_start),
            order_type: OrderType::CloseGridShort,
        };
    }
    let n_steps = ((close_prices_start - close_prices_end).abs() / exchange.price_step).ceil();
    let close_grid_qty_pct_modified = f64::max(close.grid_qty_pct, 1.0 / n_steps);
    let allowed_limit = wallet_exposure_limit_with_allowance(bot, runtime);
    let wallet_exposure_ratio = if allowed_limit > 0.0 {
        calc_wallet_exposure(
            exchange.c_mult,
            state.balance,
            position_size_abs,
            position.price,
        ) / allowed_limit
    } else {
        10.0
    };
    let close_price = if wallet_exposure_ratio > 1.0 {
        f64::min(
            state.order_book.bid,
            f64::max(close_prices_start, close_prices_end),
        )
    } else {
        f64::min(
            state.order_book.bid,
            round_dn(
                close_prices_start
                    + (close_prices_end - close_prices_start)
                        * f64::min(1.0, wallet_exposure_ratio),
                exchange.price_step,
            ),
        )
    };
    let close_qty = calc_close_qty(
        exchange,
        bot,
        runtime,
        position,
        close_grid_qty_pct_modified,
        state.balance,
        close_price,
    );
    Order {
        qty: close_qty,
        price: close_price,
        order_type: OrderType::CloseGridShort,
    }
}

fn calc_trailing_close_short(
    exchange: &ExchangeParams,
    state: &StateParams,
    bot: &BotParams,
    runtime: &RuntimeOrderContext,
    close: &TrailingGridV7CloseParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
) -> Order {
    let position_size_abs = position.size.abs();
    if position_size_abs == 0.0 {
        return Order::default();
    }
    if close.trailing_threshold_pct <= 0.0 {
        if close.trailing_retracement_pct > 0.0
            && trailing.max_since_min
                > trailing.min_since_open * (1.0 + close.trailing_retracement_pct)
        {
            Order {
                qty: calc_close_qty(
                    exchange,
                    bot,
                    runtime,
                    position,
                    close.trailing_qty_pct,
                    state.balance,
                    state.order_book.bid,
                ),
                price: state.order_book.bid,
                order_type: OrderType::CloseTrailingShort,
            }
        } else {
            Order {
                qty: 0.0,
                price: 0.0,
                order_type: OrderType::CloseTrailingShort,
            }
        }
    } else if close.trailing_retracement_pct <= 0.0 {
        let close_price = f64::min(
            state.order_book.bid,
            round_dn(
                position.price * (1.0 - close.trailing_threshold_pct),
                exchange.price_step,
            ),
        );
        Order {
            qty: calc_close_qty(
                exchange,
                bot,
                runtime,
                position,
                close.trailing_qty_pct,
                state.balance,
                close_price,
            ),
            price: close_price,
            order_type: OrderType::CloseTrailingShort,
        }
    } else if trailing.min_since_open < position.price * (1.0 - close.trailing_threshold_pct)
        && trailing.max_since_min > trailing.min_since_open * (1.0 + close.trailing_retracement_pct)
    {
        let close_price = f64::min(
            state.order_book.bid,
            round_dn(
                position.price
                    * (1.0 - close.trailing_threshold_pct + close.trailing_retracement_pct),
                exchange.price_step,
            ),
        );
        Order {
            qty: calc_close_qty(
                exchange,
                bot,
                runtime,
                position,
                close.trailing_qty_pct,
                state.balance,
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

fn calc_next_close_short(
    exchange: &ExchangeParams,
    state: &StateParams,
    bot: &BotParams,
    runtime: &RuntimeOrderContext,
    close: &TrailingGridV7CloseParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
) -> Order {
    let position_size_abs = position.size.abs();
    if position_size_abs == 0.0 {
        return Order::default();
    }
    let wallet_exposure = calc_wallet_exposure(
        exchange.c_mult,
        state.balance,
        position_size_abs,
        position.price,
    );
    if let Some(order) =
        calc_wel_auto_reduce_short(exchange, state, bot, runtime, position, wallet_exposure)
    {
        return order;
    }
    let allowed_limit = wallet_exposure_limit_with_allowance(bot, runtime);
    if close.trailing_grid_ratio >= 1.0 || close.trailing_grid_ratio <= -1.0 {
        return calc_trailing_close_short(exchange, state, bot, runtime, close, position, trailing);
    }
    if close.trailing_grid_ratio == 0.0 {
        return calc_grid_close_short(exchange, state, bot, runtime, close, position);
    }
    let wallet_exposure_ratio = if allowed_limit > 0.0 {
        wallet_exposure / allowed_limit
    } else {
        10.0
    };
    if close.trailing_grid_ratio > 0.0 {
        if wallet_exposure_ratio < close.trailing_grid_ratio {
            calc_trailing_close_short(exchange, state, bot, runtime, close, position, trailing)
        } else {
            let mut trailing_allocation = cost_to_qty(
                state.balance * allowed_limit * close.trailing_grid_ratio,
                position.price,
                exchange.c_mult,
            );
            let min_entry_qty = calc_min_entry_qty(position.price, exchange);
            if trailing_allocation < min_entry_qty {
                trailing_allocation = 0.0;
            }
            let grid_allocation = round_(
                (position_size_abs - trailing_allocation) * 1.01,
                exchange.qty_step,
            );
            let position_mod = Position {
                size: -f64::min(position_size_abs, f64::max(grid_allocation, min_entry_qty)),
                price: position.price,
            };
            calc_grid_close_short(exchange, state, bot, runtime, close, &position_mod)
        }
    } else if wallet_exposure_ratio < 1.0 + close.trailing_grid_ratio {
        calc_grid_close_short(exchange, state, bot, runtime, close, position)
    } else {
        let mut grid_allocation = cost_to_qty(
            state.balance * allowed_limit * (1.0 + close.trailing_grid_ratio),
            position.price,
            exchange.c_mult,
        );
        let min_entry_qty = calc_min_entry_qty(position.price, exchange);
        if grid_allocation < min_entry_qty {
            grid_allocation = 0.0;
        }
        let trailing_allocation = round_(
            (position_size_abs - grid_allocation) * 1.01,
            exchange.qty_step,
        );
        let position_mod = Position {
            size: -f64::min(
                position_size_abs,
                f64::max(trailing_allocation, min_entry_qty),
            ),
            price: position.price,
        };
        calc_trailing_close_short(
            exchange,
            state,
            bot,
            runtime,
            close,
            &position_mod,
            trailing,
        )
    }
}

fn calc_entries_long(
    exchange: &ExchangeParams,
    state: &StateParams,
    bot: &BotParams,
    runtime: &RuntimeOrderContext,
    entry_params: &TrailingGridV7EntryParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
) -> Vec<Order> {
    let mut entries = Vec::<Order>::new();
    let mut psize = position.size;
    let mut pprice = position.price;
    let mut bid = state.order_book.bid;
    for _ in 0..500 {
        let position_mod = Position {
            size: psize,
            price: pprice,
        };
        let mut state_mod = *state;
        state_mod.order_book.bid = bid;
        let mut entry = calc_next_entry_long(
            exchange,
            &state_mod,
            bot,
            runtime,
            entry_params,
            &position_mod,
            trailing,
        );
        entry.price = quantize_price(
            entry.price,
            exchange.price_step,
            RoundingMode::Nearest,
            "trailing_grid_v7::entries_long_price",
        );
        entry.qty = quantize_qty(
            entry.qty,
            exchange.qty_step,
            RoundingMode::Nearest,
            "trailing_grid_v7::entries_long_qty",
        );
        if entry.qty == 0.0 {
            break;
        }
        if !entries.is_empty() {
            if entry.order_type == OrderType::EntryTrailingNormalLong
                || entry.order_type == OrderType::EntryTrailingCroppedLong
                || entries[entries.len() - 1].price == entry.price
            {
                break;
            }
        }
        (psize, pprice) =
            calc_new_psize_pprice(psize, pprice, entry.qty, entry.price, exchange.qty_step);
        bid = bid.min(entry.price);
        entries.push(entry);
    }
    entries
}

fn calc_entries_short(
    exchange: &ExchangeParams,
    state: &StateParams,
    bot: &BotParams,
    runtime: &RuntimeOrderContext,
    entry_params: &TrailingGridV7EntryParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
) -> Vec<Order> {
    let mut entries = Vec::<Order>::new();
    let mut psize = position.size;
    let mut pprice = position.price;
    let mut ask = state.order_book.ask;
    for _ in 0..500 {
        let position_mod = Position {
            size: psize,
            price: pprice,
        };
        let mut state_mod = *state;
        state_mod.order_book.ask = ask;
        let mut entry = calc_next_entry_short(
            exchange,
            &state_mod,
            bot,
            runtime,
            entry_params,
            &position_mod,
            trailing,
        );
        entry.price = quantize_price(
            entry.price,
            exchange.price_step,
            RoundingMode::Nearest,
            "trailing_grid_v7::entries_short_price",
        );
        entry.qty = quantize_qty(
            entry.qty,
            exchange.qty_step,
            RoundingMode::Nearest,
            "trailing_grid_v7::entries_short_qty",
        );
        if entry.qty == 0.0 {
            break;
        }
        if !entries.is_empty() {
            if entry.order_type == OrderType::EntryTrailingNormalShort
                || entry.order_type == OrderType::EntryTrailingCroppedShort
                || entries[entries.len() - 1].price == entry.price
            {
                break;
            }
        }
        (psize, pprice) =
            calc_new_psize_pprice(psize, pprice, entry.qty, entry.price, exchange.qty_step);
        ask = ask.max(entry.price);
        entries.push(entry);
    }
    entries
}

fn calc_closes_long(
    exchange: &ExchangeParams,
    state: &StateParams,
    bot: &BotParams,
    runtime: &RuntimeOrderContext,
    close_params: &TrailingGridV7CloseParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
) -> Vec<Order> {
    let mut closes = Vec::<Order>::new();
    let mut psize = position.size;
    for _ in 0..500 {
        let position_mod = Position {
            size: psize,
            price: position.price,
        };
        let mut close = calc_next_close_long(
            exchange,
            state,
            bot,
            runtime,
            close_params,
            &position_mod,
            trailing,
        );
        close.price = quantize_price(
            close.price,
            exchange.price_step,
            RoundingMode::Nearest,
            "trailing_grid_v7::closes_long_price",
        );
        close.qty = quantize_qty(
            close.qty,
            exchange.qty_step,
            RoundingMode::Nearest,
            "trailing_grid_v7::closes_long_qty",
        );
        if close.qty == 0.0 {
            break;
        }
        psize = round_(psize + close.qty, exchange.qty_step);
        if !closes.is_empty() {
            if close.order_type == OrderType::CloseTrailingLong {
                break;
            }
            if closes[closes.len() - 1].price == close.price {
                let previous = closes.pop().expect("non-empty closes");
                closes.push(Order {
                    qty: round_(previous.qty + close.qty, exchange.qty_step),
                    price: close.price,
                    order_type: close.order_type,
                });
                continue;
            }
        }
        closes.push(close);
    }
    sort_closes_by_price(&mut closes, false, "trailing_grid_v7::calc_closes_long");
    closes
}

fn calc_closes_short(
    exchange: &ExchangeParams,
    state: &StateParams,
    bot: &BotParams,
    runtime: &RuntimeOrderContext,
    close_params: &TrailingGridV7CloseParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
) -> Vec<Order> {
    let mut closes = Vec::<Order>::new();
    let mut psize = position.size;
    for _ in 0..500 {
        let position_mod = Position {
            size: psize,
            price: position.price,
        };
        let mut close = calc_next_close_short(
            exchange,
            state,
            bot,
            runtime,
            close_params,
            &position_mod,
            trailing,
        );
        close.price = quantize_price(
            close.price,
            exchange.price_step,
            RoundingMode::Nearest,
            "trailing_grid_v7::closes_short_price",
        );
        close.qty = quantize_qty(
            close.qty,
            exchange.qty_step,
            RoundingMode::Nearest,
            "trailing_grid_v7::closes_short_qty",
        );
        if close.qty == 0.0 {
            break;
        }
        psize = round_(psize + close.qty, exchange.qty_step);
        if !closes.is_empty() {
            if close.order_type == OrderType::CloseTrailingShort {
                break;
            }
            if closes[closes.len() - 1].price == close.price {
                let previous = closes.pop().expect("non-empty closes");
                closes.push(Order {
                    qty: round_(previous.qty + close.qty, exchange.qty_step),
                    price: close.price,
                    order_type: close.order_type,
                });
                continue;
            }
        }
        closes.push(close);
    }
    sort_closes_by_price(&mut closes, true, "trailing_grid_v7::calc_closes_short");
    closes
}

pub fn generate_orders(side: StrategySide, request: StrategyRequest<'_>) -> GeneratedOrders {
    let mut generated = GeneratedOrders::default();
    let params = match request.strategy_params {
        StrategyParams::TrailingGridV7(params) => params,
        _ => panic!("trailing_grid_v7 strategy received non-trailing_grid_v7 params"),
    };
    let runtime_context = RuntimeOrderContext {
        effective_wallet_exposure_limit: request.runtime_budget.effective_wallet_exposure_limit,
    };
    let TrailingGridV7Params { entry, close, .. } = *params;
    match side {
        StrategySide::Long => {
            if request.wants_entries {
                if let Some(peek) = request.peek {
                    if peek.expand_entries {
                        extend_nonzero(
                            &mut generated.entries,
                            calc_entries_long(
                                request.exchange,
                                request.state,
                                request.bot_params,
                                &runtime_context,
                                &entry,
                                request.position,
                                request.trailing,
                            ),
                        );
                    } else {
                        push_if_nonzero(
                            &mut generated.entries,
                            calc_next_entry_long(
                                request.exchange,
                                request.state,
                                request.bot_params,
                                &runtime_context,
                                &entry,
                                request.position,
                                request.trailing,
                            ),
                        );
                    }
                } else if let Some(next) = request.next_candle {
                    let next_entry = calc_next_entry_long(
                        request.exchange,
                        request.state,
                        request.bot_params,
                        &runtime_context,
                        &entry,
                        request.position,
                        request.trailing,
                    );
                    if next.tradable
                        && would_fill_next_candle(
                            next.low,
                            next.high,
                            next_entry.qty,
                            next_entry.price,
                        )
                    {
                        extend_nonzero(
                            &mut generated.entries,
                            calc_entries_long(
                                request.exchange,
                                request.state,
                                request.bot_params,
                                &runtime_context,
                                &entry,
                                request.position,
                                request.trailing,
                            ),
                        );
                    } else {
                        push_if_nonzero(&mut generated.entries, next_entry);
                    }
                } else {
                    extend_nonzero(
                        &mut generated.entries,
                        calc_entries_long(
                            request.exchange,
                            request.state,
                            request.bot_params,
                            &runtime_context,
                            &entry,
                            request.position,
                            request.trailing,
                        ),
                    );
                }
            }
            if request.wants_closes {
                if let Some(peek) = request.peek {
                    if peek.expand_closes {
                        extend_nonzero(
                            &mut generated.closes,
                            calc_closes_long(
                                request.exchange,
                                request.state,
                                request.bot_params,
                                &runtime_context,
                                &close,
                                request.position,
                                request.trailing,
                            ),
                        );
                    } else {
                        push_if_nonzero(
                            &mut generated.closes,
                            calc_next_close_long(
                                request.exchange,
                                request.state,
                                request.bot_params,
                                &runtime_context,
                                &close,
                                request.position,
                                request.trailing,
                            ),
                        );
                    }
                } else if let Some(next) = request.next_candle {
                    let next_close = calc_next_close_long(
                        request.exchange,
                        request.state,
                        request.bot_params,
                        &runtime_context,
                        &close,
                        request.position,
                        request.trailing,
                    );
                    if next.tradable {
                        let closes = calc_closes_long(
                            request.exchange,
                            request.state,
                            request.bot_params,
                            &runtime_context,
                            &close,
                            request.position,
                            request.trailing,
                        );
                        if any_order_would_fill_next_candle(next.low, next.high, &closes) {
                            extend_nonzero(&mut generated.closes, closes);
                        } else {
                            push_if_nonzero(&mut generated.closes, next_close);
                        }
                    } else {
                        push_if_nonzero(&mut generated.closes, next_close);
                    }
                } else {
                    extend_nonzero(
                        &mut generated.closes,
                        calc_closes_long(
                            request.exchange,
                            request.state,
                            request.bot_params,
                            &runtime_context,
                            &close,
                            request.position,
                            request.trailing,
                        ),
                    );
                }
            }
        }
        StrategySide::Short => {
            if request.wants_entries {
                if let Some(peek) = request.peek {
                    if peek.expand_entries {
                        extend_nonzero(
                            &mut generated.entries,
                            calc_entries_short(
                                request.exchange,
                                request.state,
                                request.bot_params,
                                &runtime_context,
                                &entry,
                                request.position,
                                request.trailing,
                            ),
                        );
                    } else {
                        push_if_nonzero(
                            &mut generated.entries,
                            calc_next_entry_short(
                                request.exchange,
                                request.state,
                                request.bot_params,
                                &runtime_context,
                                &entry,
                                request.position,
                                request.trailing,
                            ),
                        );
                    }
                } else if let Some(next) = request.next_candle {
                    let next_entry = calc_next_entry_short(
                        request.exchange,
                        request.state,
                        request.bot_params,
                        &runtime_context,
                        &entry,
                        request.position,
                        request.trailing,
                    );
                    if next.tradable
                        && would_fill_next_candle(
                            next.low,
                            next.high,
                            next_entry.qty,
                            next_entry.price,
                        )
                    {
                        extend_nonzero(
                            &mut generated.entries,
                            calc_entries_short(
                                request.exchange,
                                request.state,
                                request.bot_params,
                                &runtime_context,
                                &entry,
                                request.position,
                                request.trailing,
                            ),
                        );
                    } else {
                        push_if_nonzero(&mut generated.entries, next_entry);
                    }
                } else {
                    extend_nonzero(
                        &mut generated.entries,
                        calc_entries_short(
                            request.exchange,
                            request.state,
                            request.bot_params,
                            &runtime_context,
                            &entry,
                            request.position,
                            request.trailing,
                        ),
                    );
                }
            }
            if request.wants_closes {
                if let Some(peek) = request.peek {
                    if peek.expand_closes {
                        extend_nonzero(
                            &mut generated.closes,
                            calc_closes_short(
                                request.exchange,
                                request.state,
                                request.bot_params,
                                &runtime_context,
                                &close,
                                request.position,
                                request.trailing,
                            ),
                        );
                    } else {
                        push_if_nonzero(
                            &mut generated.closes,
                            calc_next_close_short(
                                request.exchange,
                                request.state,
                                request.bot_params,
                                &runtime_context,
                                &close,
                                request.position,
                                request.trailing,
                            ),
                        );
                    }
                } else if let Some(next) = request.next_candle {
                    let next_close = calc_next_close_short(
                        request.exchange,
                        request.state,
                        request.bot_params,
                        &runtime_context,
                        &close,
                        request.position,
                        request.trailing,
                    );
                    if next.tradable {
                        let closes = calc_closes_short(
                            request.exchange,
                            request.state,
                            request.bot_params,
                            &runtime_context,
                            &close,
                            request.position,
                            request.trailing,
                        );
                        if any_order_would_fill_next_candle(next.low, next.high, &closes) {
                            extend_nonzero(&mut generated.closes, closes);
                        } else {
                            push_if_nonzero(&mut generated.closes, next_close);
                        }
                    } else {
                        push_if_nonzero(&mut generated.closes, next_close);
                    }
                } else {
                    extend_nonzero(
                        &mut generated.closes,
                        calc_closes_short(
                            request.exchange,
                            request.state,
                            request.bot_params,
                            &runtime_context,
                            &close,
                            request.position,
                            request.trailing,
                        ),
                    );
                }
            }
        }
    }
    generated
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EMABands, OrderBook};

    fn exchange() -> ExchangeParams {
        ExchangeParams {
            qty_step: 0.01,
            price_step: 0.01,
            min_qty: 0.0,
            min_cost: 0.0,
            c_mult: 1.0,
            ..Default::default()
        }
    }

    fn state() -> StateParams {
        StateParams {
            balance: 10_000.0,
            order_book: OrderBook {
                bid: 100.0,
                ask: 100.0,
            },
            ema_bands: EMABands {
                lower: 99.0,
                upper: 101.0,
            },
            volatility_ema_1h: 0.0,
            volatility_ema_1m: 0.0,
        }
    }

    fn bot() -> BotParams {
        BotParams {
            wallet_exposure_limit: 1.0,
            total_wallet_exposure_limit: 1.0,
            n_positions: 1,
            risk_we_excess_allowance_pct: 0.0,
            risk_wel_enforcer_enabled: false,
            risk_wel_enforcer_threshold: 0.0,
            ..Default::default()
        }
    }

    fn runtime() -> RuntimeOrderContext {
        RuntimeOrderContext {
            effective_wallet_exposure_limit: 1.0,
        }
    }

    fn entry_params() -> TrailingGridV7EntryParams {
        TrailingGridV7EntryParams {
            grid_double_down_factor: 1.0,
            grid_spacing_pct: 0.01,
            initial_ema_dist: 0.0,
            initial_qty_pct: 0.01,
            trailing_double_down_factor: 1.0,
            trailing_grid_ratio: -0.5,
            trailing_retracement_pct: 0.01,
            trailing_threshold_pct: 0.01,
            ..Default::default()
        }
    }

    #[test]
    fn entry_trailing_grid_ratio_negative_uses_grid_then_trailing_for_long() {
        let exchange = exchange();
        let state = state();
        let bot = bot();
        let runtime = runtime();
        let entry = entry_params();
        let trailing = TrailingPriceBundle::default();

        let grid_first_position = Position {
            size: 40.0,
            price: 100.0,
        };
        let grid_order = calc_next_entry_long(
            &exchange,
            &state,
            &bot,
            &runtime,
            &entry,
            &grid_first_position,
            &trailing,
        );
        assert!(matches!(
            grid_order.order_type,
            OrderType::EntryGridNormalLong | OrderType::EntryGridCroppedLong
        ));
        assert!(grid_order.qty > 0.0);

        let trailing_position = Position {
            size: 60.0,
            price: 100.0,
        };
        let trailing_order = calc_next_entry_long(
            &exchange,
            &state,
            &bot,
            &runtime,
            &entry,
            &trailing_position,
            &trailing,
        );
        assert_eq!(
            trailing_order.order_type,
            OrderType::EntryTrailingNormalLong
        );
        assert_eq!(trailing_order.qty, 0.0);
    }

    #[test]
    fn entry_trailing_grid_ratio_negative_uses_grid_then_trailing_for_short() {
        let exchange = exchange();
        let state = state();
        let bot = bot();
        let runtime = runtime();
        let entry = entry_params();
        let trailing = TrailingPriceBundle::default();

        let grid_first_position = Position {
            size: -40.0,
            price: 100.0,
        };
        let grid_order = calc_next_entry_short(
            &exchange,
            &state,
            &bot,
            &runtime,
            &entry,
            &grid_first_position,
            &trailing,
        );
        assert!(matches!(
            grid_order.order_type,
            OrderType::EntryGridNormalShort | OrderType::EntryGridCroppedShort
        ));
        assert!(grid_order.qty < 0.0);

        let trailing_position = Position {
            size: -60.0,
            price: 100.0,
        };
        let trailing_order = calc_next_entry_short(
            &exchange,
            &state,
            &bot,
            &runtime,
            &entry,
            &trailing_position,
            &trailing,
        );
        assert_eq!(
            trailing_order.order_type,
            OrderType::EntryTrailingNormalShort
        );
        assert_eq!(trailing_order.qty, 0.0);
    }

    #[test]
    fn close_grid_markup_start_end_interpolates_by_wallet_exposure() {
        let exchange = exchange();
        let mut state = state();
        state.balance = 20_000.0;
        let bot = bot();
        let runtime = runtime();
        let close = TrailingGridV7CloseParams {
            grid_markup_start: 0.01,
            grid_markup_end: 0.002,
            grid_qty_pct: 0.1,
            ..Default::default()
        };
        let position = Position {
            size: 100.0,
            price: 100.0,
        };

        let close_order =
            calc_grid_close_long(&exchange, &state, &bot, &runtime, &close, &position);

        assert_eq!(close_order.order_type, OrderType::CloseGridLong);
        assert_eq!(close_order.price, 100.6);
        assert!(close_order.qty < 0.0);
    }

    #[test]
    fn close_grid_markup_start_end_interpolates_by_wallet_exposure_for_short() {
        let exchange = exchange();
        let mut state = state();
        state.balance = 20_000.0;
        let bot = bot();
        let runtime = runtime();
        let close = TrailingGridV7CloseParams {
            grid_markup_start: 0.01,
            grid_markup_end: 0.002,
            grid_qty_pct: 0.1,
            ..Default::default()
        };
        let position = Position {
            size: -100.0,
            price: 100.0,
        };

        let close_order =
            calc_grid_close_short(&exchange, &state, &bot, &runtime, &close, &position);

        assert_eq!(close_order.order_type, OrderType::CloseGridShort);
        assert_eq!(close_order.price, 99.4);
        assert!(close_order.qty > 0.0);
    }

    #[test]
    fn close_trailing_grid_ratio_positive_splits_long_by_wallet_exposure() {
        let exchange = exchange();
        let state = state();
        let bot = bot();
        let runtime = runtime();
        let close = TrailingGridV7CloseParams {
            grid_markup_start: 0.01,
            grid_markup_end: 0.005,
            grid_qty_pct: 0.2,
            trailing_grid_ratio: 0.5,
            trailing_qty_pct: 0.25,
            trailing_retracement_pct: 0.005,
            trailing_threshold_pct: 0.01,
        };
        let trailing = TrailingPriceBundle {
            max_since_open: 102.0,
            min_since_max: 101.0,
            ..Default::default()
        };

        let trailing_first = calc_next_close_long(
            &exchange,
            &state,
            &bot,
            &runtime,
            &close,
            &Position {
                size: 40.0,
                price: 100.0,
            },
            &trailing,
        );
        assert_eq!(trailing_first.order_type, OrderType::CloseTrailingLong);
        assert!(trailing_first.qty < 0.0);

        let grid_after_threshold = calc_next_close_long(
            &exchange,
            &state,
            &bot,
            &runtime,
            &close,
            &Position {
                size: 80.0,
                price: 100.0,
            },
            &trailing,
        );
        assert_eq!(grid_after_threshold.order_type, OrderType::CloseGridLong);
        assert!(grid_after_threshold.qty < 0.0);
    }

    #[test]
    fn close_trailing_grid_ratio_positive_splits_short_by_wallet_exposure() {
        let exchange = exchange();
        let state = state();
        let bot = bot();
        let runtime = runtime();
        let close = TrailingGridV7CloseParams {
            grid_markup_start: 0.01,
            grid_markup_end: 0.005,
            grid_qty_pct: 0.2,
            trailing_grid_ratio: 0.5,
            trailing_qty_pct: 0.25,
            trailing_retracement_pct: 0.005,
            trailing_threshold_pct: 0.01,
        };
        let trailing = TrailingPriceBundle {
            min_since_open: 98.0,
            max_since_min: 99.0,
            ..Default::default()
        };

        let trailing_first = calc_next_close_short(
            &exchange,
            &state,
            &bot,
            &runtime,
            &close,
            &Position {
                size: -40.0,
                price: 100.0,
            },
            &trailing,
        );
        assert_eq!(trailing_first.order_type, OrderType::CloseTrailingShort);
        assert!(trailing_first.qty > 0.0);

        let grid_after_threshold = calc_next_close_short(
            &exchange,
            &state,
            &bot,
            &runtime,
            &close,
            &Position {
                size: -80.0,
                price: 100.0,
            },
            &trailing,
        );
        assert_eq!(grid_after_threshold.order_type, OrderType::CloseGridShort);
        assert!(grid_after_threshold.qty > 0.0);
    }

    #[test]
    #[should_panic(expected = "trailing_grid_v7::calc_closes_long: non-finite close price")]
    fn close_sorting_rejects_non_finite_price_on_v7_path() {
        let mut closes = vec![Order {
            qty: -1.0,
            price: f64::NAN,
            order_type: OrderType::CloseGridLong,
        }];

        sort_closes_by_price(
            &mut closes,
            false,
            "trailing_grid_v7::calc_closes_long",
        );
    }
}
