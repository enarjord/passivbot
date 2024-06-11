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
    round_to_decimal_places(result, 12)
}

/// Rounds a number to the nearest multiple of the given step.
#[pyfunction]
pub fn round_(n: f64, step: f64) -> f64 {
    let result = (n / step).round() * step;
    round_to_decimal_places(result, 12)
}

/// Rounds down a number to the nearest multiple of the given step.
#[pyfunction]
pub fn round_dn(n: f64, step: f64) -> f64 {
    let result = (n / step).floor() * step;
    round_to_decimal_places(result, 12)
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
        (cost / price) / c_mult
    } else {
        0.0
    }
}

#[pyfunction]
pub fn qty_to_cost(qty: f64, price: f64, c_mult: f64) -> f64 {
    (qty.abs() * price) * c_mult
}

pub fn calc_min_entry_qty(initial_entry_price: f64, exchange_params: &ExchangeParams) -> f64 {
    f64::max(
        exchange_params.min_qty,
        round_up(
            cost_to_qty(
                exchange_params.min_cost,
                initial_entry_price,
                exchange_params.c_mult,
            ),
            exchange_params.qty_step,
        ),
    )
}

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
