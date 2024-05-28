use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

/// Rounds a floating-point number to a specified number of decimal places.
fn round_to_decimal_places(n: f64, decimal_places: usize) -> f64 {
    let scale_factor = 10f64.powi(decimal_places as i32);
    (n * scale_factor).round() / scale_factor
}

/// Rounds up a number to the nearest multiple of the given step.
#[pyfunction]
fn round_up(n: f64, step: f64) -> f64 {
    let result = (n / step).ceil() * step;
    round_to_decimal_places(result, 14)
}

/// Rounds down a number to the nearest multiple of the given step.
#[pyfunction]
fn round_dn(n: f64, step: f64) -> f64 {
    let result = (n / step).floor() * step;
    round_to_decimal_places(result, 14)
}

/// Rounds a number to the nearest multiple of the given step.
#[pyfunction]
fn round_(n: f64, step: f64) -> f64 {
    let result = (n / step).round() * step;
    round_to_decimal_places(result, 14)
}

/// Rounds a number dynamically to a specified number of decimal places.
#[pyfunction]
fn round_dynamic(n: f64, d: i32) -> f64 {
    if n == 0.0 {
        return n;
    }
    let shift = d - (n.abs().log10().floor() as i32) - 1;
    let scale_factor = 10f64.powi(shift);
    (n * scale_factor).round() / scale_factor
}

/// Rounds a number up dynamically to a specified number of decimal places.
#[pyfunction]
fn round_dynamic_up(n: f64, d: i32) -> f64 {
    if n == 0.0 {
        return n;
    }
    let shift = d - (n.abs().log10().floor() as i32) - 1;
    let scale_factor = 10f64.powi(shift);
    (n * scale_factor).ceil() / scale_factor
}

/// Rounds a number down dynamically to a specified number of decimal places.
#[pyfunction]
fn round_dynamic_dn(n: f64, d: i32) -> f64 {
    if n == 0.0 {
        return n;
    }
    let shift = d - (n.abs().log10().floor() as i32) - 1;
    let scale_factor = 10f64.powi(shift);
    (n * scale_factor).floor() / scale_factor
}

/// Calculates the relative difference between two numbers.
#[pyfunction]
fn calc_diff(x: f64, y: f64) -> f64 {
    (x - y).abs() / y.abs()
}

/// Converts NaN to 0.0.
#[pyfunction]
fn nan_to_0(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x
    }
}

/// Calculates the minimum entry quantity.
#[pyfunction]
fn calc_min_entry_qty(
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
        let qty = cost_to_qty(min_cost, price, inverse, c_mult);
        let rounded_qty = round_up(qty, qty_step);
        min_qty.max(rounded_qty)
    }
}

/// Converts cost to quantity.
#[pyfunction]
fn cost_to_qty(cost: f64, price: f64, inverse: bool, c_mult: f64) -> f64 {
    if inverse {
        (cost * price) / c_mult
    } else {
        if price > 0.0 {
            cost / price / c_mult
        } else {
            0.0
        }
    }
}

/// Converts quantity to cost.
#[pyfunction]
fn qty_to_cost(qty: f64, price: f64, inverse: bool, c_mult: f64) -> f64 {
    if inverse {
        if price > 0.0 {
            (qty / price).abs() * c_mult
        } else {
            0.0
        }
    } else {
        (qty * price).abs() * c_mult
    }
}

/// Calculates next EMA.
#[pyfunction]
fn calc_ema(alpha: f64, alpha_: f64, prev_ema: f64, new_val: f64) -> f64 {
    prev_ema * alpha_ + new_val * alpha
}

/// A Python module implemented in Rust.
#[pymodule]
fn passivbot_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(round_up, m)?)?;
    m.add_function(wrap_pyfunction!(round_dn, m)?)?;
    m.add_function(wrap_pyfunction!(round_, m)?)?;
    m.add_function(wrap_pyfunction!(round_dynamic, m)?)?;
    m.add_function(wrap_pyfunction!(round_dynamic_up, m)?)?;
    m.add_function(wrap_pyfunction!(round_dynamic_dn, m)?)?;
    m.add_function(wrap_pyfunction!(calc_diff, m)?)?;
    m.add_function(wrap_pyfunction!(nan_to_0, m)?)?;
    m.add_function(wrap_pyfunction!(calc_min_entry_qty, m)?)?;
    m.add_function(wrap_pyfunction!(cost_to_qty, m)?)?;
    m.add_function(wrap_pyfunction!(qty_to_cost, m)?)?;
    m.add_function(wrap_pyfunction!(calc_ema, m)?)?;
    Ok(())
}
