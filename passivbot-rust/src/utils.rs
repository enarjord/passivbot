use crate::grids::ExchangeParams;
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
    round_to_decimal_places(result, 14)
}

/// Rounds a number to the nearest multiple of the given step.
#[pyfunction]
pub fn round_(n: f64, step: f64) -> f64 {
    let result = (n / step).round() * step;
    round_to_decimal_places(result, 14)
}

/// Rounds down a number to the nearest multiple of the given step.
#[pyfunction]
pub fn round_dn(n: f64, step: f64) -> f64 {
    let result = (n / step).floor() * step;
    round_to_decimal_places(result, 14)
}

pub fn cost_to_qty(cost: f64, price: f64, c_mult: f64) -> f64 {
    if price > 0.0 {
        (cost / price) / c_mult
    } else {
        0.0
    }
}

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
    qty_to_cost(new_psize, new_pprice, exchange_params.c_mult) / balance
}

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
