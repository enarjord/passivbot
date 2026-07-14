use super::{EmaAnchorParams, GeneratedOrders, StrategyParams, StrategyRequest, StrategySide};
use crate::dynamic::{calc_dynamic_distance_multiplier, DynamicDistanceInputs};
use crate::entries::{calc_min_entry_qty, wallet_exposure_limit_with_allowance_from_base};
use crate::types::{BotParams, ExchangeParams, Order, OrderType, StateParams};
use crate::utils::{cost_to_qty, qty_to_cost, round_, round_dn, round_up};

#[inline]
fn calc_signed_wallet_exposure_ratio(
    balance: f64,
    mid: f64,
    psize: f64,
    c_mult: f64,
    effective_wallet_exposure_limit: f64,
) -> f64 {
    if balance > 0.0 && mid > 0.0 && c_mult > 0.0 && effective_wallet_exposure_limit > 0.0 {
        psize.signum() * qty_to_cost(psize, mid, c_mult) / balance / effective_wallet_exposure_limit
    } else {
        0.0
    }
}

#[inline]
fn calc_offset_multiplier(state: &StateParams, params: &EmaAnchorParams) -> f64 {
    calc_dynamic_distance_multiplier(DynamicDistanceInputs {
        volatility_ema_1m: state.volatility_ema_1m,
        volatility_ema_1h: state.volatility_ema_1h,
        weight_volatility_1m: params.offset_volatility_1m_weight,
        weight_volatility_1h: params.offset_volatility_1h_weight,
        wallet_exposure_ratio: None,
        weight_wallet_exposure: 0.0,
        min_multiplier: 1.0,
    })
}

#[inline]
pub fn calc_bid_price(
    state: &StateParams,
    exchange: &ExchangeParams,
    params: &EmaAnchorParams,
    psize: f64,
    effective_wallet_exposure_limit: f64,
) -> f64 {
    let mid = (state.order_book.bid + state.order_book.ask) * 0.5;
    let inventory_shift = calc_signed_wallet_exposure_ratio(
        state.balance,
        mid,
        psize,
        exchange.c_mult,
        effective_wallet_exposure_limit,
    ) * params.offset_psize_weight;
    let effective_offset = params.offset * calc_offset_multiplier(state, params);
    let target = state.ema_bands.lower * (1.0 - effective_offset - inventory_shift);
    f64::min(state.order_book.bid, round_dn(target, exchange.price_step))
}

#[inline]
pub fn calc_ask_price(
    state: &StateParams,
    exchange: &ExchangeParams,
    params: &EmaAnchorParams,
    psize: f64,
    effective_wallet_exposure_limit: f64,
) -> f64 {
    let mid = (state.order_book.bid + state.order_book.ask) * 0.5;
    let inventory_shift = calc_signed_wallet_exposure_ratio(
        state.balance,
        mid,
        psize,
        exchange.c_mult,
        effective_wallet_exposure_limit,
    ) * params.offset_psize_weight;
    let effective_offset = params.offset * calc_offset_multiplier(state, params);
    let target = state.ema_bands.upper * (1.0 + effective_offset - inventory_shift);
    f64::max(state.order_book.ask, round_up(target, exchange.price_step))
}

#[inline]
pub fn calc_quote_prices(
    state: &StateParams,
    exchange: &ExchangeParams,
    params: &EmaAnchorParams,
    psize: f64,
    effective_wallet_exposure_limit: f64,
) -> (f64, f64) {
    (
        calc_bid_price(
            state,
            exchange,
            params,
            psize,
            effective_wallet_exposure_limit,
        ),
        calc_ask_price(
            state,
            exchange,
            params,
            psize,
            effective_wallet_exposure_limit,
        ),
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
        wallet_exposure_limit_with_allowance_from_base(bot_params, effective_wallet_exposure_limit);
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
    let signed_we_ratio = calc_signed_wallet_exposure_ratio(
        balance,
        mid,
        psize,
        exchange.c_mult,
        effective_wallet_exposure_limit,
    );
    let same_side_bias = match side {
        StrategySide::Long => signed_we_ratio.max(0.0),
        StrategySide::Short => (-signed_we_ratio).max(0.0),
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
                    effective_wallet_exposure_limit,
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
                    effective_wallet_exposure_limit,
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
                    effective_wallet_exposure_limit,
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
                    effective_wallet_exposure_limit,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EMABands, OrderBook, Position, RuntimeBudgetState, TrailingPriceBundle};

    fn base_state() -> StateParams {
        StateParams {
            balance: 1000.0,
            order_book: OrderBook {
                bid: 99.5,
                ask: 100.5,
            },
            ema_bands: EMABands {
                lower: 100.0,
                upper: 100.0,
            },
            ..Default::default()
        }
    }

    fn base_exchange() -> ExchangeParams {
        ExchangeParams {
            qty_step: 0.01,
            price_step: 0.01,
            min_qty: 0.0,
            min_cost: 0.0,
            c_mult: 1.0,
            ..Default::default()
        }
    }

    fn base_runtime_budget(effective_wallet_exposure_limit: f64) -> RuntimeBudgetState {
        RuntimeBudgetState {
            configured_wallet_exposure_limit: effective_wallet_exposure_limit,
            effective_wallet_exposure_limit,
            configured_n_positions: 1,
            effective_n_positions: 1,
        }
    }

    fn base_params() -> EmaAnchorParams {
        EmaAnchorParams {
            base_qty_pct: 0.01,
            offset: 0.01,
            offset_psize_weight: 0.1,
            ..Default::default()
        }
    }

    #[test]
    fn bid_inventory_shift_uses_signed_wallet_exposure_ratio() {
        let state = StateParams {
            balance: 1000.0,
            order_book: OrderBook {
                bid: 100.0,
                ask: 100.0,
            },
            ema_bands: EMABands {
                lower: 100.0,
                upper: 100.0,
            },
            ..Default::default()
        };
        let exchange = ExchangeParams {
            price_step: 0.01,
            c_mult: 2.0,
            ..Default::default()
        };
        let params = EmaAnchorParams {
            offset: 0.0,
            offset_psize_weight: 0.01,
            ..Default::default()
        };

        let bid_wel_1 = calc_bid_price(&state, &exchange, &params, 1.0, 1.0);
        let bid_wel_half = calc_bid_price(&state, &exchange, &params, 1.0, 0.5);

        assert_eq!(bid_wel_1, 99.8);
        assert_eq!(bid_wel_half, 99.6);
    }

    #[test]
    fn neutral_quotes_do_not_cross_top_of_book() {
        let state = base_state();
        let exchange = base_exchange();
        let params = base_params();

        let (bid, ask) = calc_quote_prices(&state, &exchange, &params, 0.0, 1.0);

        assert!(bid <= state.order_book.bid);
        assert!(ask >= state.order_book.ask);
        assert_eq!(bid, 99.0);
        assert_eq!(ask, 101.0);
    }

    #[test]
    fn inventory_skew_moves_quotes_in_position_reducing_direction() {
        let bid_state = StateParams {
            order_book: OrderBook {
                bid: 1000.0,
                ask: 1000.0,
            },
            ..base_state()
        };
        let ask_state = StateParams {
            order_book: OrderBook { bid: 1.0, ask: 1.0 },
            ..base_state()
        };
        let exchange = base_exchange();
        let params = EmaAnchorParams {
            offset: 0.0,
            offset_psize_weight: 0.2,
            ..base_params()
        };

        let flat_bid = calc_bid_price(&bid_state, &exchange, &params, 0.0, 1.0);
        let long_bid = calc_bid_price(&bid_state, &exchange, &params, 1.0, 1.0);
        let short_bid = calc_bid_price(&bid_state, &exchange, &params, -1.0, 1.0);
        let flat_ask = calc_ask_price(&ask_state, &exchange, &params, 0.0, 1.0);
        let long_ask = calc_ask_price(&ask_state, &exchange, &params, 1.0, 1.0);
        let short_ask = calc_ask_price(&ask_state, &exchange, &params, -1.0, 1.0);

        assert!(long_bid < flat_bid);
        assert!(long_ask < flat_ask);
        assert!(short_bid > flat_bid);
        assert!(short_ask > flat_ask);
    }

    #[test]
    fn no_entries_when_effective_wallet_exposure_limit_is_zero() {
        let state = base_state();
        let exchange = base_exchange();
        let bot_params = BotParams::default();
        let params = base_params();
        let strategy_params = StrategyParams::EmaAnchor(params);
        let position = Position::default();
        let trailing = TrailingPriceBundle::default();
        let request = StrategyRequest {
            wants_entries: true,
            wants_closes: true,
            exchange: &exchange,
            state: &state,
            bot_params: &bot_params,
            strategy_params: &strategy_params,
            runtime_budget: base_runtime_budget(0.0),
            position: &position,
            trailing: &trailing,
            next_candle: None,
            peek: None,
        };

        let generated = generate_orders(StrategySide::Long, request);

        assert!(generated.entries.is_empty());
        assert!(generated.closes.is_empty());
    }

    #[test]
    fn volatility_weights_widen_bid_and_ask_offsets() {
        let calm_state = base_state();
        let volatile_state = StateParams {
            volatility_ema_1m: 0.05,
            volatility_ema_1h: 0.04,
            ..base_state()
        };
        let exchange = base_exchange();
        let params = EmaAnchorParams {
            offset: 0.01,
            offset_volatility_1m_weight: 2.0,
            offset_volatility_1h_weight: 3.0,
            ..base_params()
        };

        let (calm_bid, calm_ask) = calc_quote_prices(&calm_state, &exchange, &params, 0.0, 1.0);
        let (wide_bid, wide_ask) = calc_quote_prices(&volatile_state, &exchange, &params, 0.0, 1.0);

        assert!(wide_bid < calm_bid);
        assert!(wide_ask > calm_ask);
    }

    #[test]
    fn entry_double_down_factor_only_scales_same_side_entry_qty() {
        let state = StateParams {
            order_book: OrderBook {
                bid: 1000.0,
                ask: 1000.0,
            },
            ..base_state()
        };
        let exchange = base_exchange();
        let bot_params = BotParams::default();
        let params = EmaAnchorParams {
            entry_double_down_factor: 1.0,
            offset_psize_weight: 0.0,
            ..base_params()
        };
        let strategy_params = StrategyParams::EmaAnchor(params);
        let trailing = TrailingPriceBundle::default();
        let flat_position = Position {
            size: 0.0,
            price: 100.0,
        };
        let long_position = Position {
            size: 5.0,
            price: 100.0,
        };
        let short_position = Position {
            size: -5.0,
            price: 100.0,
        };

        let flat_request = StrategyRequest {
            wants_entries: true,
            wants_closes: false,
            exchange: &exchange,
            state: &state,
            bot_params: &bot_params,
            strategy_params: &strategy_params,
            runtime_budget: base_runtime_budget(1.0),
            position: &flat_position,
            trailing: &trailing,
            next_candle: None,
            peek: None,
        };
        let long_request = StrategyRequest {
            wants_entries: true,
            wants_closes: false,
            exchange: &exchange,
            state: &state,
            bot_params: &bot_params,
            strategy_params: &strategy_params,
            runtime_budget: base_runtime_budget(1.0),
            position: &long_position,
            trailing: &trailing,
            next_candle: None,
            peek: None,
        };
        let opposite_request = StrategyRequest {
            wants_entries: true,
            wants_closes: false,
            exchange: &exchange,
            state: &state,
            bot_params: &bot_params,
            strategy_params: &strategy_params,
            runtime_budget: base_runtime_budget(1.0),
            position: &short_position,
            trailing: &trailing,
            next_candle: None,
            peek: None,
        };

        let flat_qty = generate_orders(StrategySide::Long, flat_request).entries[0].qty;
        let same_side_qty = generate_orders(StrategySide::Long, long_request).entries[0].qty;
        let opposite_side_qty =
            generate_orders(StrategySide::Long, opposite_request).entries[0].qty;

        assert!(same_side_qty > flat_qty);
        assert_eq!(opposite_side_qty, flat_qty);
    }
}
