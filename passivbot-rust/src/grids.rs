use crate::utils::round_up;
use pyo3::prelude::*;

#[pyfunction]
pub fn calc_min_entry_qty(
    price: f64,
    inverse: bool,
    c_mult: f64,
    qty_step: f64,
    min_qty: f64,
    min_cost: f64,
) -> f64 {
    if inverse {
        min_qty
    } else {
        min_qty.max(round_up(
            cost_to_qty(min_cost, price, inverse, c_mult),
            qty_step,
        ))
    }
}

#[pyfunction]
pub fn cost_to_qty(cost: f64, price: f64, inverse: bool, c_mult: f64) -> f64 {
    (if inverse {
        cost * price
    } else {
        if price > 0.0 {
            cost / price
        } else {
            0.0
        }
    }) / c_mult
}

#[pyfunction]
pub fn qty_to_cost(qty: f64, price: f64, inverse: bool, c_mult: f64) -> f64 {
    (if inverse {
        if price > 0.0 {
            qty.abs() / price
        } else {
            0.0
        }
    } else {
        qty.abs() * price
    }) * c_mult
}
