use crate::entries::calc_initial_entry_qty;
use crate::types::{BotParams, ExchangeParams, Order, OrderType, Position, StateParams};
use crate::utils::{round_dn, round_up};

#[derive(Debug, Default, Clone)]
pub struct StrategyOrders {
    pub entries: Vec<Order>,
    pub closes: Vec<Order>,
}

#[inline]
fn calc_pside_bias(balance: f64, mid: f64, psize: f64) -> f64 {
    if balance > 0.0 { psize * mid / balance } else { 0.0 }
}

#[inline]
fn calc_bid_price(state: &StateParams, exchange: &ExchangeParams, bot: &BotParams, psize: f64) -> f64 {
    let mid = (state.order_book.bid + state.order_book.ask) * 0.5;
    let pside_bias = calc_pside_bias(state.balance, mid, psize);
    let target = state.ema_bands.lower
        * (1.0 - bot.entry_initial_ema_dist - pside_bias * bot.entry_grid_spacing_we_weight);
    f64::min(state.order_book.bid, round_dn(target, exchange.price_step))
}

#[inline]
fn calc_ask_price(state: &StateParams, exchange: &ExchangeParams, bot: &BotParams, psize: f64) -> f64 {
    let mid = (state.order_book.bid + state.order_book.ask) * 0.5;
    let pside_bias = calc_pside_bias(state.balance, mid, psize);
    let target = state.ema_bands.upper
        * (1.0 + bot.entry_initial_ema_dist - pside_bias * bot.entry_grid_spacing_we_weight);
    f64::max(state.order_book.ask, round_up(target, exchange.price_step))
}

pub fn calc_orders_long(
    exchange: &ExchangeParams,
    state: &StateParams,
    bot: &BotParams,
    position: &Position,
) -> StrategyOrders {
    let mut orders = StrategyOrders::default();
    let bid_price = calc_bid_price(state, exchange, bot, position.size);
    if bid_price.is_finite()
        && bid_price > 0.0
        && bot.wallet_exposure_limit > 0.0
        && bot.entry_initial_qty_pct > 0.0
    {
        let qty = calc_initial_entry_qty(exchange, bot, state.balance, bid_price);
        if qty > 0.0 {
            orders.entries.push(Order {
                qty,
                price: bid_price,
                order_type: if position.size > 0.0 {
                    OrderType::EntryGridNormalLong
                } else {
                    OrderType::EntryInitialNormalLong
                },
            });
        }
    }

    if position.size > 0.0 {
        let ask_price = calc_ask_price(state, exchange, bot, position.size);
        if ask_price.is_finite() && ask_price > 0.0 {
            orders.closes.push(Order {
                qty: -position.size.abs(),
                price: ask_price,
                order_type: OrderType::CloseGridLong,
            });
        }
    }

    orders
}

pub fn calc_orders_short(
    exchange: &ExchangeParams,
    state: &StateParams,
    bot: &BotParams,
    position: &Position,
) -> StrategyOrders {
    let mut orders = StrategyOrders::default();
    if position.size < 0.0 {
        let bid_price = calc_bid_price(state, exchange, bot, position.size);
        if bid_price.is_finite() && bid_price > 0.0 {
            orders.closes.push(Order {
                qty: position.size.abs(),
                price: bid_price,
                order_type: OrderType::CloseGridShort,
            });
        }
    }

    let ask_price = calc_ask_price(state, exchange, bot, position.size);
    if ask_price.is_finite()
        && ask_price > 0.0
        && bot.wallet_exposure_limit > 0.0
        && bot.entry_initial_qty_pct > 0.0
    {
        let qty = calc_initial_entry_qty(exchange, bot, state.balance, ask_price);
        if qty > 0.0 {
            orders.entries.push(Order {
                qty: -qty,
                price: ask_price,
                order_type: if position.size < 0.0 {
                    OrderType::EntryGridNormalShort
                } else {
                    OrderType::EntryInitialNormalShort
                },
            });
        }
    }

    orders
}
