use super::{EmaAnchorParams, GeneratedOrders, StrategyParams, StrategyRequest, StrategySide};
use crate::entries::calc_min_entry_qty;
use crate::types::{BotParams, ExchangeParams, Order, OrderType, StateParams};
use crate::utils::{cost_to_qty, round_, round_dn, round_up};

#[inline]
fn calc_pside_bias(balance: f64, mid: f64, psize: f64) -> f64 {
    if balance > 0.0 {
        psize * mid / balance
    } else {
        0.0
    }
}

#[inline]
fn effective_wallet_exposure_limit_with_allowance(bot_params: &BotParams, base_limit: f64) -> f64 {
    if base_limit <= 0.0 {
        base_limit
    } else {
        base_limit * (1.0 + bot_params.risk_we_excess_allowance_pct.max(0.0))
    }
}

#[inline]
pub fn calc_bid_price(
    state: &StateParams,
    exchange: &ExchangeParams,
    params: &EmaAnchorParams,
    psize: f64,
) -> f64 {
    let mid = (state.order_book.bid + state.order_book.ask) * 0.5;
    let inventory_shift = calc_pside_bias(state.balance, mid, psize) * params.offset_psize_weight;
    let vol_term = state.offset_volatility_logrange_ema_1m * params.offset_volatility_1m_weight
        + state.entry_volatility_logrange_ema_1h * params.offset_volatility_1h_weight;
    let effective_offset = params.offset * (1.0 + vol_term).max(1.0);
    let target = state.ema_bands.lower * (1.0 - effective_offset - inventory_shift);
    f64::min(state.order_book.bid, round_dn(target, exchange.price_step))
}

#[inline]
pub fn calc_ask_price(
    state: &StateParams,
    exchange: &ExchangeParams,
    params: &EmaAnchorParams,
    psize: f64,
) -> f64 {
    let mid = (state.order_book.bid + state.order_book.ask) * 0.5;
    let inventory_shift = calc_pside_bias(state.balance, mid, psize) * params.offset_psize_weight;
    let vol_term = state.offset_volatility_logrange_ema_1m * params.offset_volatility_1m_weight
        + state.entry_volatility_logrange_ema_1h * params.offset_volatility_1h_weight;
    let effective_offset = params.offset * (1.0 + vol_term).max(1.0);
    let target = state.ema_bands.upper * (1.0 + effective_offset - inventory_shift);
    f64::max(state.order_book.ask, round_up(target, exchange.price_step))
}

#[inline]
pub fn calc_quote_prices(
    state: &StateParams,
    exchange: &ExchangeParams,
    params: &EmaAnchorParams,
    psize: f64,
) -> (f64, f64) {
    (
        calc_bid_price(state, exchange, params, psize),
        calc_ask_price(state, exchange, params, psize),
    )
}

#[inline]
fn calc_base_clip_qty(
    exchange: &ExchangeParams,
    bot_params: &BotParams,
    params: &EmaAnchorParams,
    balance: f64,
    price: f64,
    effective_wallet_exposure_limit: f64,
) -> f64 {
    if !(price.is_finite() && price > 0.0 && balance > 0.0) {
        return 0.0;
    }
    let effective_limit =
        effective_wallet_exposure_limit_with_allowance(bot_params, effective_wallet_exposure_limit);
    if effective_limit <= 0.0 || params.base_qty_pct <= 0.0 {
        return 0.0;
    }
    f64::max(
        calc_min_entry_qty(price, exchange),
        round_(
            cost_to_qty(
                balance * effective_limit * params.base_qty_pct,
                price,
                exchange.c_mult,
            ),
            exchange.qty_step,
        ),
    )
}

#[inline]
fn calc_entry_qty(
    side: StrategySide,
    exchange: &ExchangeParams,
    bot_params: &BotParams,
    params: &EmaAnchorParams,
    balance: f64,
    price: f64,
    effective_wallet_exposure_limit: f64,
    psize: f64,
    mid: f64,
) -> f64 {
    let base_qty = calc_base_clip_qty(
        exchange,
        bot_params,
        params,
        balance,
        price,
        effective_wallet_exposure_limit,
    );
    if base_qty <= 0.0 {
        return 0.0;
    }
    let pside_bias = calc_pside_bias(balance, mid, psize);
    let same_side_bias = match side {
        StrategySide::Long => pside_bias.max(0.0),
        StrategySide::Short => (-pside_bias).max(0.0),
    };
    let multiplier = (1.0 + same_side_bias * params.entry_double_down_factor).max(1.0);
    round_(base_qty * multiplier, exchange.qty_step)
}

#[inline]
fn calc_close_qty(
    exchange: &ExchangeParams,
    bot_params: &BotParams,
    params: &EmaAnchorParams,
    balance: f64,
    close_price: f64,
    position_size_abs: f64,
    effective_wallet_exposure_limit: f64,
) -> f64 {
    if close_price <= 0.0 || position_size_abs <= 0.0 {
        return 0.0;
    }
    let min_qty = calc_min_entry_qty(close_price, exchange);
    if position_size_abs <= min_qty {
        return position_size_abs;
    }
    let clip_qty = f64::min(
        position_size_abs,
        calc_base_clip_qty(
            exchange,
            bot_params,
            params,
            balance,
            close_price,
            effective_wallet_exposure_limit,
        ),
    );
    if clip_qty <= 0.0 {
        return 0.0;
    }
    if position_size_abs - clip_qty < min_qty {
        position_size_abs
    } else {
        clip_qty
    }
}

fn typed_params<'a>(request: &'a StrategyRequest<'a>) -> &'a EmaAnchorParams {
    match request.strategy_params {
        StrategyParams::EmaAnchor(params) => params,
        _ => panic!("ema_anchor strategy received non-ema_anchor params"),
    }
}

pub fn generate_orders(side: StrategySide, request: StrategyRequest<'_>) -> GeneratedOrders {
    let params = typed_params(&request);
    let mut generated = GeneratedOrders::default();
    let effective_wallet_exposure_limit = request.runtime_budget.effective_wallet_exposure_limit;

    match side {
        StrategySide::Long => {
            if request.wants_entries {
                let bid_price = calc_bid_price(
                    request.state,
                    request.exchange,
                    params,
                    request.position.size,
                );
                if bid_price.is_finite() && bid_price > 0.0 && effective_wallet_exposure_limit > 0.0
                {
                    let mid = (request.state.order_book.bid + request.state.order_book.ask) * 0.5;
                    let qty = calc_entry_qty(
                        StrategySide::Long,
                        request.exchange,
                        request.bot_params,
                        params,
                        request.state.balance,
                        bid_price,
                        effective_wallet_exposure_limit,
                        request.position.size,
                        mid,
                    );
                    if qty > 0.0 {
                        generated.entries.push(Order {
                            qty,
                            price: bid_price,
                            order_type: OrderType::EntryEmaAnchorLong,
                        });
                    }
                }
            }

            if request.wants_closes && request.position.size > 0.0 {
                let ask_price = calc_ask_price(
                    request.state,
                    request.exchange,
                    params,
                    request.position.size,
                );
                if ask_price.is_finite() && ask_price > 0.0 {
                    let close_qty = calc_close_qty(
                        request.exchange,
                        request.bot_params,
                        params,
                        request.state.balance,
                        ask_price,
                        request.position.size.abs(),
                        effective_wallet_exposure_limit,
                    );
                    if close_qty > 0.0 {
                        generated.closes.push(Order {
                            qty: -close_qty,
                            price: ask_price,
                            order_type: OrderType::CloseEmaAnchorLong,
                        });
                    }
                }
            }
        }
        StrategySide::Short => {
            if request.wants_closes && request.position.size < 0.0 {
                let bid_price = calc_bid_price(
                    request.state,
                    request.exchange,
                    params,
                    request.position.size,
                );
                if bid_price.is_finite() && bid_price > 0.0 {
                    let close_qty = calc_close_qty(
                        request.exchange,
                        request.bot_params,
                        params,
                        request.state.balance,
                        bid_price,
                        request.position.size.abs(),
                        effective_wallet_exposure_limit,
                    );
                    if close_qty > 0.0 {
                        generated.closes.push(Order {
                            qty: close_qty,
                            price: bid_price,
                            order_type: OrderType::CloseEmaAnchorShort,
                        });
                    }
                }
            }

            if request.wants_entries {
                let ask_price = calc_ask_price(
                    request.state,
                    request.exchange,
                    params,
                    request.position.size,
                );
                if ask_price.is_finite() && ask_price > 0.0 && effective_wallet_exposure_limit > 0.0
                {
                    let mid = (request.state.order_book.bid + request.state.order_book.ask) * 0.5;
                    let qty = calc_entry_qty(
                        StrategySide::Short,
                        request.exchange,
                        request.bot_params,
                        params,
                        request.state.balance,
                        ask_price,
                        effective_wallet_exposure_limit,
                        request.position.size,
                        mid,
                    );
                    if qty > 0.0 {
                        generated.entries.push(Order {
                            qty: -qty,
                            price: ask_price,
                            order_type: OrderType::EntryEmaAnchorShort,
                        });
                    }
                }
            }
        }
    }
    generated
}
