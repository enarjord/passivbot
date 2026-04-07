use super::{GeneratedOrders, StrategyParams, StrategyRequest, StrategySide};
use crate::closes::{
    calc_closes_long, calc_closes_short, calc_next_close_long, calc_next_close_short,
};
use crate::entries::{
    calc_entries_long, calc_entries_short, calc_next_entry_long, calc_next_entry_short,
};
use crate::types::Order;

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

pub fn generate_orders(side: StrategySide, request: StrategyRequest<'_>) -> GeneratedOrders {
    let mut generated = GeneratedOrders::default();
    let runtime_params = match request.strategy_params {
        StrategyParams::TrailingGrid(params) => {
            params.apply_to_bot_params(request.bot_params, request.runtime_budget)
        }
        _ => panic!("trailing_grid strategy received non-trailing_grid params"),
    };

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
                                &runtime_params,
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
                                &runtime_params,
                                request.position,
                                request.trailing,
                            ),
                        );
                    }
                } else if let Some(next) = request.next_candle {
                    let next_entry = calc_next_entry_long(
                        request.exchange,
                        request.state,
                        &runtime_params,
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
                                &runtime_params,
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
                            &runtime_params,
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
                                &runtime_params,
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
                                &runtime_params,
                                request.position,
                                request.trailing,
                            ),
                        );
                    }
                } else if let Some(next) = request.next_candle {
                    let next_close = calc_next_close_long(
                        request.exchange,
                        request.state,
                        &runtime_params,
                        request.position,
                        request.trailing,
                    );
                    let expand_closes = next.tradable
                        && would_fill_next_candle(
                            next.low,
                            next.high,
                            next_close.qty,
                            next_close.price,
                        );
                    if expand_closes {
                        extend_nonzero(
                            &mut generated.closes,
                            calc_closes_long(
                                request.exchange,
                                request.state,
                                &runtime_params,
                                request.position,
                                request.trailing,
                            ),
                        );
                    } else {
                        push_if_nonzero(&mut generated.closes, next_close);
                    }
                } else {
                    extend_nonzero(
                        &mut generated.closes,
                        calc_closes_long(
                            request.exchange,
                            request.state,
                            &runtime_params,
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
                                &runtime_params,
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
                                &runtime_params,
                                request.position,
                                request.trailing,
                            ),
                        );
                    }
                } else if let Some(next) = request.next_candle {
                    let next_entry = calc_next_entry_short(
                        request.exchange,
                        request.state,
                        &runtime_params,
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
                                &runtime_params,
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
                            &runtime_params,
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
                                &runtime_params,
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
                                &runtime_params,
                                request.position,
                                request.trailing,
                            ),
                        );
                    }
                } else if let Some(next) = request.next_candle {
                    let next_close = calc_next_close_short(
                        request.exchange,
                        request.state,
                        &runtime_params,
                        request.position,
                        request.trailing,
                    );
                    let expand_closes = next.tradable
                        && would_fill_next_candle(
                            next.low,
                            next.high,
                            next_close.qty,
                            next_close.price,
                        );
                    if expand_closes {
                        extend_nonzero(
                            &mut generated.closes,
                            calc_closes_short(
                                request.exchange,
                                request.state,
                                &runtime_params,
                                request.position,
                                request.trailing,
                            ),
                        );
                    } else {
                        push_if_nonzero(&mut generated.closes, next_close);
                    }
                } else {
                    extend_nonzero(
                        &mut generated.closes,
                        calc_closes_short(
                            request.exchange,
                            request.state,
                            &runtime_params,
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
