use super::{GeneratedOrders, StrategyParams, StrategyRequest, StrategySide};
use crate::closes::{
    calc_closes_long, calc_closes_short, calc_next_close_long, calc_next_close_short,
};
use crate::entries::{
    calc_entries_long, calc_entries_short, calc_next_entry_long, calc_next_entry_short,
};
use crate::types::{Order, RuntimeOrderContext};

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

pub fn generate_orders(side: StrategySide, request: StrategyRequest<'_>) -> GeneratedOrders {
    let mut generated = GeneratedOrders::default();
    let params = match request.strategy_params {
        StrategyParams::TrailingMartingale(params) => params,
        _ => panic!("trailing_martingale strategy received non-trailing_martingale params"),
    };
    let runtime_context = RuntimeOrderContext {
        effective_wallet_exposure_limit: request.runtime_budget.effective_wallet_exposure_limit,
    };
    let entry_params = params.entry_params();
    let close_params = params.close_params();

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
                                &entry_params,
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
                                &entry_params,
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
                        &entry_params,
                        request.position,
                        request.trailing,
                    );
                    let expand_entries = next.tradable
                        && would_fill_next_candle(
                            next.low,
                            next.high,
                            next_entry.qty,
                            next_entry.price,
                        );
                    if expand_entries {
                        extend_nonzero(
                            &mut generated.entries,
                            calc_entries_long(
                                request.exchange,
                                request.state,
                                request.bot_params,
                                &runtime_context,
                                &entry_params,
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
                            &entry_params,
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
                                &close_params,
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
                                &close_params,
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
                        &close_params,
                        request.position,
                        request.trailing,
                    );
                    if next.tradable {
                        let closes = calc_closes_long(
                            request.exchange,
                            request.state,
                            request.bot_params,
                            &runtime_context,
                            &close_params,
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
                            &close_params,
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
                                &entry_params,
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
                                &entry_params,
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
                        &entry_params,
                        request.position,
                        request.trailing,
                    );
                    let expand_entries = next.tradable
                        && would_fill_next_candle(
                            next.low,
                            next.high,
                            next_entry.qty,
                            next_entry.price,
                        );
                    if expand_entries {
                        extend_nonzero(
                            &mut generated.entries,
                            calc_entries_short(
                                request.exchange,
                                request.state,
                                request.bot_params,
                                &runtime_context,
                                &entry_params,
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
                            &entry_params,
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
                                &close_params,
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
                                &close_params,
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
                        &close_params,
                        request.position,
                        request.trailing,
                    );
                    if next.tradable {
                        let closes = calc_closes_short(
                            request.exchange,
                            request.state,
                            request.bot_params,
                            &runtime_context,
                            &close_params,
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
                            &close_params,
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
    use crate::strategies::{
        NextStepHint, TrailingMartingaleCloseParams, TrailingMartingaleParams,
    };
    use crate::types::{
        BotParams, ExchangeParams, OrderBook, Position, RuntimeBudgetState, StateParams,
        TrailingPriceBundle,
    };

    fn recursive_close_request<'a>(
        exchange: &'a ExchangeParams,
        state: &'a StateParams,
        bot: &'a BotParams,
        params: &'a StrategyParams,
        position: &'a Position,
        trailing: &'a TrailingPriceBundle,
        next_low: f64,
        next_high: f64,
    ) -> StrategyRequest<'a> {
        StrategyRequest {
            wants_entries: false,
            wants_closes: true,
            exchange,
            state,
            bot_params: bot,
            strategy_params: params,
            runtime_budget: RuntimeBudgetState {
                configured_wallet_exposure_limit: 1.0,
                effective_wallet_exposure_limit: 1.0,
                configured_n_positions: 1,
                effective_n_positions: 1,
            },
            position,
            trailing,
            next_candle: Some(NextStepHint {
                low: next_low,
                high: next_high,
                tradable: true,
            }),
            peek: None,
        }
    }

    #[test]
    fn next_candle_close_peek_expands_when_any_long_close_rung_would_fill() {
        let exchange = ExchangeParams {
            qty_step: 0.01,
            price_step: 0.01,
            min_qty: 0.0,
            min_cost: 0.0,
            c_mult: 1.0,
            ..Default::default()
        };
        let state = StateParams {
            balance: 10_000.0,
            order_book: OrderBook {
                ask: 100.0,
                bid: 100.0,
            },
            ..Default::default()
        };
        let bot = BotParams {
            wallet_exposure_limit: 1.0,
            total_wallet_exposure_limit: 1.0,
            risk_we_excess_allowance_pct: 0.0,
            risk_wel_enforcer_enabled: false,
            risk_wel_enforcer_threshold: 0.0,
            ..Default::default()
        };
        let params = StrategyParams::TrailingMartingale(TrailingMartingaleParams {
            close: TrailingMartingaleCloseParams {
                qty_pct: 0.1,
                threshold_base_pct: 0.01,
                threshold_we_weight: 0.01,
                ..Default::default()
            },
            ..Default::default()
        });
        let position = Position {
            size: 100.0,
            price: 100.0,
        };
        let trailing = TrailingPriceBundle::default();

        let generated = generate_orders(
            StrategySide::Long,
            recursive_close_request(
                &exchange, &state, &bot, &params, &position, &trailing, 0.0, 101.55,
            ),
        );

        assert_eq!(generated.closes.len(), 10);
        assert_eq!(generated.closes[0].price, 101.1);
        assert_eq!(generated.closes[9].price, 102.0);
    }

    #[test]
    fn next_candle_close_peek_expands_when_any_short_close_rung_would_fill() {
        let exchange = ExchangeParams {
            qty_step: 0.01,
            price_step: 0.01,
            min_qty: 0.0,
            min_cost: 0.0,
            c_mult: 1.0,
            ..Default::default()
        };
        let state = StateParams {
            balance: 10_000.0,
            order_book: OrderBook {
                ask: 100.0,
                bid: 100.0,
            },
            ..Default::default()
        };
        let bot = BotParams {
            wallet_exposure_limit: 1.0,
            total_wallet_exposure_limit: 1.0,
            risk_we_excess_allowance_pct: 0.0,
            risk_wel_enforcer_enabled: false,
            risk_wel_enforcer_threshold: 0.0,
            ..Default::default()
        };
        let params = StrategyParams::TrailingMartingale(TrailingMartingaleParams {
            close: TrailingMartingaleCloseParams {
                qty_pct: 0.1,
                threshold_base_pct: 0.01,
                threshold_we_weight: 0.01,
                ..Default::default()
            },
            ..Default::default()
        });
        let position = Position {
            size: -100.0,
            price: 100.0,
        };
        let trailing = TrailingPriceBundle::default();

        let generated = generate_orders(
            StrategySide::Short,
            recursive_close_request(
                &exchange, &state, &bot, &params, &position, &trailing, 98.05, 200.0,
            ),
        );

        assert_eq!(generated.closes.len(), 10);
        assert_eq!(generated.closes[0].price, 98.9);
        assert_eq!(generated.closes[9].price, 98.0);
    }
}
