use crate::constants::{CLOSE, LONG, NO_POS, SHORT};
use crate::types::ExchangeParams;
use pyo3::prelude::*;

/// Rounds a number to the specified number of decimal places.
fn round_to_decimal_places(value: f64, decimal_places: usize) -> f64 {
    let multiplier = 10f64.powi(decimal_places as i32);
    (value * multiplier).round() / multiplier
}

/// Rounds up a number to the nearest multiple of the given step.
#[pyfunction]
pub fn round_up(n: f64, step: f64) -> f64 {
    let result = (n / step).ceil() * step;
    round_to_decimal_places(result, 10)
}

/// Rounds a number to the nearest multiple of the given step.
#[pyfunction]
pub fn round_(n: f64, step: f64) -> f64 {
    let result = (n / step).round() * step;
    round_to_decimal_places(result, 10)
}

/// Rounds down a number to the nearest multiple of the given step.
#[pyfunction]
pub fn round_dn(n: f64, step: f64) -> f64 {
    let result = (n / step).floor() * step;
    round_to_decimal_places(result, 10)
}

#[pyfunction]
pub fn round_dynamic(n: f64, d: i32) -> f64 {
    if n == 0.0 {
        return n;
    }
    let shift = d - (n.abs().log10().floor() as i32) - 1;
    let multiplier = 10f64.powi(shift);
    let result = (n * multiplier).round() / multiplier;
    round_to_decimal_places(result, 10)
}

#[pyfunction]
pub fn round_dynamic_up(n: f64, d: i32) -> f64 {
    if n == 0.0 {
        return n;
    }
    let shift = d - (n.abs().log10().floor() as i32) - 1;
    let multiplier = 10f64.powi(shift);
    let result = (n * multiplier).ceil() / multiplier;
    round_to_decimal_places(result, 10)
}

#[pyfunction]
pub fn round_dynamic_dn(n: f64, d: i32) -> f64 {
    if n == 0.0 {
        return n;
    }
    let shift = d - (n.abs().log10().floor() as i32) - 1;
    let multiplier = 10f64.powi(shift);
    let result = (n * multiplier).floor() / multiplier;
    round_to_decimal_places(result, 10)
}

#[pyfunction]
pub fn hysteresis_rounding(
    balance: f64,
    last_rounded_balance: f64,
    percentage: f64,
    h: f64,
) -> f64 {
    let step = last_rounded_balance * percentage;
    let threshold = step * h;
    let rounded_balance = if balance > last_rounded_balance + threshold {
        last_rounded_balance + step
    } else if balance < last_rounded_balance - threshold {
        last_rounded_balance - step
    } else {
        last_rounded_balance
    };
    round_dynamic(rounded_balance, 6)
}

#[pyfunction]
pub fn calc_diff(x: f64, y: f64) -> f64 {
    if y == 0.0 {
        if x == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (x - y).abs() / y.abs()
    }
}

#[pyfunction]
pub fn cost_to_qty(cost: f64, price: f64, c_mult: f64) -> f64 {
    if price > 0.0 {
        (cost.abs() / price) / c_mult
    } else {
        0.0
    }
}

#[pyfunction]
pub fn qty_to_cost(qty: f64, price: f64, c_mult: f64) -> f64 {
    (qty.abs() * price) * c_mult
}

#[pyfunction]
pub fn calc_wallet_exposure(
    c_mult: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
) -> f64 {
    if balance <= 0.0 || position_size == 0.0 {
        return 0.0;
    }
    qty_to_cost(position_size, position_price, c_mult) / balance
}

pub fn calc_wallet_exposure_if_filled(
    balance: f64,
    psize: f64,
    pprice: f64,
    qty: f64,
    price: f64,
    exchange_params: &ExchangeParams,
) -> f64 {
    let psize = round_(psize.abs(), exchange_params.qty_step);
    let qty = round_(qty.abs(), exchange_params.qty_step);
    let (new_psize, new_pprice) =
        calc_new_psize_pprice(psize, pprice, qty, price, exchange_params.qty_step);
    calc_wallet_exposure(exchange_params.c_mult, balance, new_psize, new_pprice)
}

#[pyfunction]
pub fn calc_new_psize_pprice(
    psize: f64,
    pprice: f64,
    qty: f64,
    price: f64,
    qty_step: f64,
) -> (f64, f64) {
    if qty == 0.0 {
        return (psize, pprice);
    }
    if psize == 0.0 {
        return (qty, price);
    }
    let new_psize = round_(psize + qty, qty_step);
    if new_psize == 0.0 {
        return (0.0, 0.0);
    }
    (
        new_psize,
        nan_to_0(pprice) * (psize / new_psize) + price * (qty / new_psize),
    )
}

fn nan_to_0(value: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else {
        value
    }
}

pub fn interpolate(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len(), "xs and ys must have the same length");

    let n = xs.len();
    let mut result = 0.0;

    for i in 0..n {
        let mut term = ys[i];
        for j in 0..n {
            if i != j {
                term *= (x - xs[j]) / (xs[i] - xs[j]);
            }
        }
        result += term;
    }

    result
}

#[pyfunction]
pub fn calc_pnl_long(entry_price: f64, close_price: f64, qty: f64, c_mult: f64) -> f64 {
    qty.abs() * c_mult * (close_price - entry_price)
}

#[pyfunction]
pub fn calc_pnl_short(entry_price: f64, close_price: f64, qty: f64, c_mult: f64) -> f64 {
    qty.abs() * c_mult * (entry_price - close_price)
}

pub fn calc_pprice_diff_int(pside: usize, pprice: f64, price: f64) -> f64 {
    match pside {
        LONG => {
            // long
            if pprice > 0.0 {
                1.0 - price / pprice
            } else {
                0.0
            }
        }
        SHORT => {
            // short
            if pprice > 0.0 {
                price / pprice - 1.0
            } else {
                0.0
            }
        }
        _ => panic!("unknown pside {}", pside),
    }
}

#[pyfunction]
pub fn calc_auto_unstuck_allowance(
    balance: f64,
    loss_allowance_pct: f64,
    pnl_cumsum_max: f64,
    pnl_cumsum_last: f64,
) -> f64 {
    // allow up to x% drop from balance peak for auto unstuck

    let balance_peak = balance + (pnl_cumsum_max - pnl_cumsum_last);
    let drop_since_peak_pct = balance / balance_peak - 1.0;
    (balance_peak * (loss_allowance_pct + drop_since_peak_pct)).max(0.0)
}

pub fn calc_ema_price_bid(
    price_step: f64,
    order_book_bid: f64,
    ema_bands_lower: f64,
    ema_dist: f64,
) -> f64 {
    f64::min(
        order_book_bid,
        round_dn(ema_bands_lower * (1.0 - ema_dist), price_step),
    )
}

pub fn calc_ema_price_ask(
    price_step: f64,
    order_book_ask: f64,
    ema_bands_upper: f64,
    ema_dist: f64,
) -> f64 {
    f64::max(
        order_book_ask,
        round_up(ema_bands_upper * (1.0 + ema_dist), price_step),
    )
}
