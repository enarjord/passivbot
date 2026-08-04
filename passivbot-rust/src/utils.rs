use crate::constants::{LONG, SHORT};
use crate::types::ExchangeParams;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Rounds a number to the specified number of decimal places.
fn round_to_decimal_places(value: f64, decimal_places: usize) -> f64 {
    let multiplier = 10f64.powi(decimal_places as i32);
    (value * multiplier).round() / multiplier
}

fn step_decimal_places(step: f64) -> Option<usize> {
    let mut multiplier = 1.0;
    for decimal_places in 0..=15 {
        let scaled = step.abs() * multiplier;
        let rounded = scaled.round();
        if rounded >= 1.0 && (scaled - rounded).abs() <= scaled.abs().max(1.0) * 1e-12 {
            return Some(decimal_places.max(10));
        }
        multiplier *= 10.0;
    }
    None
}

fn round_to_step_decimal_places(value: f64, step: f64) -> f64 {
    match step_decimal_places(step) {
        Some(decimal_places) => round_to_decimal_places(value, decimal_places),
        None => value,
    }
}

fn differs_only_by_float_representation(value: f64, aligned_value: f64) -> bool {
    let scale = value.abs().max(aligned_value.abs());
    (value - aligned_value).abs() <= f64::EPSILON * scale * 4.0
}

/// Directionally quantize without truncating valid increments below ten decimals.
pub fn round_dn_preserve_step(value: f64, step: f64) -> f64 {
    round_to_step_decimal_places((value / step).floor() * step, step)
}

/// Directionally quantize without truncating valid increments below ten decimals.
pub fn round_up_preserve_step(value: f64, step: f64) -> f64 {
    round_to_step_decimal_places((value / step).ceil() * step, step)
}

/// Snap ordinary binary noise at an aligned increment before rounding down.
pub fn tolerant_round_dn_preserve_step(value: f64, step: f64) -> f64 {
    let step_count = value / step;
    let nearest_count = step_count.round();
    let nearest_value = nearest_count * step;
    if differs_only_by_float_representation(value, nearest_value) {
        round_to_step_decimal_places(nearest_value, step)
    } else {
        round_dn_preserve_step(value, step)
    }
}

/// Snap ordinary binary noise at an aligned increment before rounding up.
pub fn tolerant_round_up_preserve_step(value: f64, step: f64) -> f64 {
    let step_count = value / step;
    let nearest_count = step_count.round();
    let nearest_value = nearest_count * step;
    if differs_only_by_float_representation(value, nearest_value) {
        round_to_step_decimal_places(nearest_value, step)
    } else {
        round_up_preserve_step(value, step)
    }
}

/// Rounds up a number to the nearest multiple of the given step.
#[pyfunction]
pub fn round_up(n: f64, step: f64) -> f64 {
    round_up_preserve_step(n, step)
}

/// Rounds a number to the nearest multiple of the given step.
#[pyfunction]
pub fn round_(n: f64, step: f64) -> f64 {
    let result = (n / step).round() * step;
    round_to_step_decimal_places(result, step)
}

/// Rounds down a number to the nearest multiple of the given step.
#[pyfunction]
pub fn round_dn(n: f64, step: f64) -> f64 {
    round_dn_preserve_step(n, step)
}

#[derive(Clone, Copy, Debug)]
pub enum RoundingMode {
    Nearest,
    Floor,
    Ceil,
}

fn quantize_value(value: f64, step: f64, mode: RoundingMode, context: &str) -> f64 {
    if step <= 0.0 || !value.is_finite() {
        return value;
    }
    let rounded = match mode {
        RoundingMode::Nearest => round_(value, step),
        RoundingMode::Floor => tolerant_round_dn_preserve_step(value, step),
        RoundingMode::Ceil => tolerant_round_up_preserve_step(value, step),
    };
    let diff = (value - rounded).abs();
    // Allow for typical floating noise (up to 1e-8 of the step).
    let tolerance = step * 1e-8;
    if diff > tolerance {
        log::warn!(
            "quantize_value: large adjustment in {} (step {}): {} -> {} (Δ={})",
            context,
            step,
            value,
            rounded,
            diff
        );
    }
    rounded
}

pub fn quantize_price(price: f64, price_step: f64, mode: RoundingMode, context: &str) -> f64 {
    quantize_value(price, price_step, mode, context)
}

pub fn quantize_qty(qty: f64, qty_step: f64, mode: RoundingMode, context: &str) -> f64 {
    quantize_value(qty, qty_step, mode, context)
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

/// Multiplicative (relative) hysteresis.
/// Triggers a change only if the relative difference exceeds `pct`.
/// Example: pct = 0.01 → require >1% change to update.
///
/// Semantics match the Python reference:
/// - If any input is non-finite ⇒ hold `prev_val`.
/// - If `prev_val == 0.0` ⇒ pass through `val`.
/// - Else update iff |val - prev_val| / |prev_val| > max(0, pct).
#[pyfunction]
pub fn hysteresis(val: f64, prev_val: f64, pct: f64) -> f64 {
    if !(val.is_finite() && prev_val.is_finite() && pct.is_finite()) {
        return prev_val;
    }
    let pct = if pct.is_sign_negative() { 0.0 } else { pct };

    if prev_val == 0.0 {
        return val;
    }

    if ((val - prev_val).abs() / prev_val.abs()) > pct {
        val
    } else {
        prev_val
    }
}

pub fn ema_last_f64(values: &[f64], span: f64) -> f64 {
    let alpha = 2.0 / (span + 1.0);
    let one_minus = 1.0 - alpha;
    let mut num: Option<f64> = None;
    let mut den = 1.0;
    for &value in values {
        if !value.is_finite() {
            continue;
        }
        match num {
            None => num = Some(value),
            Some(previous) => {
                let next = alpha * value + one_minus * previous;
                den = alpha + one_minus * den;
                if den <= f64::MIN_POSITIVE {
                    num = Some(alpha * value);
                    den = alpha;
                } else {
                    num = Some(next);
                }
            }
        }
    }
    num.map(|value| value / den).unwrap_or(f64::NAN)
}

#[pyfunction(name = "ema_last")]
pub fn ema_last_py(values: PyReadonlyArray1<'_, f64>, span: f64) -> PyResult<f64> {
    if !span.is_finite() || span <= 0.0 {
        return Err(PyValueError::new_err(
            "EMA span must be finite and greater than zero",
        ));
    }
    let values = values.as_slice()?;
    Ok(ema_last_f64(values, span))
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

#[pyfunction]
pub fn calc_pprice_diff_int(pside: usize, pprice: f64, price: f64) -> f64 {
    match pside {
        LONG => {
            if pprice > 0.0 {
                1.0 - price / pprice
            } else {
                0.0
            }
        }
        SHORT => {
            if pprice > 0.0 {
                price / pprice - 1.0
            } else {
                0.0
            }
        }
        _ => panic!("unknown pside {}", pside),
    }
}

/// Pside-aware signed price difference helper. Alias of calc_pprice_diff_int with clearer name.
#[pyfunction]
pub fn calc_pside_price_diff_int(pside: usize, pprice: f64, price: f64) -> f64 {
    calc_pprice_diff_int(pside, pprice, price)
}

/// Backwards-compatible alias; prefer calc_pside_price_diff_int.
#[pyfunction]
pub fn calc_price_diff_pside_int(pside: usize, pprice: f64, price: f64) -> f64 {
    calc_pside_price_diff_int(pside, pprice, price)
}

pub fn calc_order_price_diff_bid(order_price: f64, market_price: f64) -> f64 {
    if market_price <= 0.0 {
        0.0
    } else {
        1.0 - order_price / market_price
    }
}

pub fn calc_order_price_diff_ask(order_price: f64, market_price: f64) -> f64 {
    if market_price <= 0.0 {
        0.0
    } else {
        order_price / market_price - 1.0
    }
}

#[pyfunction]
pub fn calc_order_price_diff(side: &str, order_price: f64, market_price: f64) -> PyResult<f64> {
    if !order_price.is_finite() || !market_price.is_finite() || market_price <= 0.0 {
        return Ok(0.0);
    }
    let norm_side = side.trim().to_ascii_lowercase();
    let diff = match norm_side.as_str() {
        "buy" => 1.0 - order_price / market_price,
        "sell" => order_price / market_price - 1.0,
        other => {
            return Err(PyValueError::new_err(format!(
                "invalid order side '{}'; expected 'buy' or 'sell'",
                other
            )))
        }
    };
    Ok(diff)
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

#[cfg(test)]
mod tests {
    use super::{
        ema_last_f64, round_, round_dn, round_up, tolerant_round_dn_preserve_step,
        tolerant_round_up_preserve_step,
    };

    #[test]
    fn public_rounding_cleans_float_noise_without_truncating_tiny_steps() {
        assert_eq!(round_(64.850000000000093, 0.01), 64.85);
        assert_eq!(round_(0.06469999999999954, 0.0001), 0.0647);

        assert_eq!(round_(1e-10, 3e-12), 9.9e-11);
        assert_eq!(round_dn(0.999999999999, 3e-12), 0.999999999999);
        assert_eq!(round_up(1.000000000001, 3e-12), 1.000000000002);
    }

    #[test]
    fn tolerant_directional_rounding_does_not_snap_genuine_tick_offsets() {
        assert_eq!(tolerant_round_up_preserve_step(0.1000000005, 0.1), 0.2);
        assert_eq!(tolerant_round_dn_preserve_step(0.0999999995, 0.1), 0.0);
        assert_eq!(tolerant_round_up_preserve_step(0.1 + 0.2, 0.1), 0.3);
        assert_eq!(tolerant_round_dn_preserve_step(0.29, 0.01), 0.29);
    }

    #[test]
    fn ema_last_matches_sequential_reference_and_skips_nonfinite_values() {
        let values = [f64::NAN, 1.0, 2.0, f64::INFINITY, -3.0, 8.0, 13.0];
        let span = 10.0;
        let alpha = 2.0 / (span + 1.0);
        let mut expected = 1.0;
        for value in [2.0, -3.0, 8.0, 13.0] {
            expected = alpha * value + (1.0 - alpha) * expected;
        }
        assert_eq!(ema_last_f64(&values, span), expected);
        assert!(ema_last_f64(&[f64::NAN, f64::INFINITY], span).is_nan());
    }
}
