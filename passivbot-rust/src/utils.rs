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

// Checked decimal arithmetic proves alignment of the finite input values.
// An overflow only declines that proof; it never widens a snapping tolerance.
#[derive(Clone, Copy, Debug)]
struct DecimalParts {
    coefficient: i128,
    exponent: i32,
}

impl DecimalParts {
    fn normalized(mut self) -> Self {
        if self.coefficient == 0 {
            self.exponent = 0;
        } else {
            while self.coefficient % 10 == 0 {
                self.coefficient /= 10;
                self.exponent += 1;
            }
        }
        self
    }

    fn from_float(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        if value == 0.0 {
            return Some(Self {
                coefficient: 0,
                exponent: 0,
            });
        }
        let text = value.to_string();
        let (mantissa, exponent) = match text.split_once('e') {
            Some((mantissa, exponent)) => (mantissa, exponent.parse::<i32>().ok()?),
            None => (text.as_str(), 0),
        };
        let places = mantissa
            .split_once('.')
            .map_or(0, |(_, tail)| tail.len() as i32);
        let digits = mantissa.replace('.', "");
        let trimmed = digits.trim_end_matches('0');
        Some(
            Self {
                coefficient: trimmed.parse().ok()?,
                exponent: exponent - places + (digits.len() - trimmed.len()) as i32,
            }
            .normalized(),
        )
    }

    fn add(self, other: Self) -> Option<Self> {
        if self.coefficient == 0 {
            return Some(other);
        }
        if other.coefficient == 0 {
            return Some(self);
        }
        let exponent = self.exponent.min(other.exponent);
        let left = self
            .coefficient
            .checked_mul(10_i128.checked_pow((self.exponent - exponent) as u32)?)?;
        let right = other
            .coefficient
            .checked_mul(10_i128.checked_pow((other.exponent - exponent) as u32)?)?;
        Some(
            Self {
                coefficient: left.checked_add(right)?,
                exponent,
            }
            .normalized(),
        )
    }

    fn multiply(self, other: Self) -> Option<Self> {
        Some(
            Self {
                coefficient: self.coefficient.checked_mul(other.coefficient)?,
                exponent: self.exponent.checked_add(other.exponent)?,
            }
            .normalized(),
        )
    }

    fn integer_ratio(self, other: Self) -> Option<i128> {
        if other.coefficient <= 0 {
            return None;
        }
        if self.coefficient == 0 {
            return Some(0);
        }
        let difference = self.exponent - other.exponent;
        let (numerator, denominator) = if difference >= 0 {
            (
                self.coefficient
                    .checked_mul(10_i128.checked_pow(difference as u32)?)?,
                other.coefficient,
            )
        } else {
            (
                self.coefficient,
                other
                    .coefficient
                    .checked_mul(10_i128.checked_pow((-difference) as u32)?)?,
            )
        };
        (numerator % denominator == 0).then_some(numerator / denominator)
    }

    fn rounded_ratio(self, other: Self, mode: RoundingMode) -> Option<i128> {
        if other.coefficient <= 0 {
            return None;
        }
        let difference = self.exponent - other.exponent;
        let (numerator, denominator) = if difference >= 0 {
            (
                self.coefficient
                    .checked_mul(10_i128.checked_pow(difference as u32)?)?,
                other.coefficient,
            )
        } else {
            (
                self.coefficient,
                other
                    .coefficient
                    .checked_mul(10_i128.checked_pow((-difference) as u32)?)?,
            )
        };
        let quotient = numerator / denominator;
        let remainder = numerator % denominator;
        let adjustment = match mode {
            RoundingMode::Floor if remainder < 0 => -1,
            RoundingMode::Ceil if remainder > 0 => 1,
            RoundingMode::Nearest
                if remainder.unsigned_abs() >= (denominator / 2 + denominator % 2) as u128 =>
            {
                numerator.signum()
            }
            _ => 0,
        };
        let count = quotient.checked_add(adjustment)?;
        (count.unsigned_abs() < 1_u128 << 53).then_some(count)
    }

    fn to_float(self) -> Option<f64> {
        let value: f64 = format!("{}e{}", self.coefficient, self.exponent)
            .parse()
            .ok()?;
        value.is_finite().then_some(value)
    }
}

// The smallest exact decimal scale keeps ordinary venue steps on a cheap
// integer path. Equality here is required; a tolerance cannot prove a step.
fn exact_step_scale(step: f64) -> Option<(f64, f64)> {
    let mut multiplier = 1.0;
    for _ in 0..=15 {
        let coefficient = (step * multiplier).round();
        if coefficient >= 1.0
            && coefficient < (1_u64 << 50) as f64
            && step == coefficient / multiplier
        {
            return Some((coefficient, multiplier));
        }
        multiplier *= 10.0;
    }
    None
}

// Below this integer scale, a canonical decimal value and its adjacent
// decimals have distinct f64 representations. This is an equality proof,
// not a numerical band around a venue increment.
fn small_decimal_units(value: f64, scale: (f64, f64)) -> Option<i128> {
    let (coefficient, multiplier) = scale;
    let integer = (value * multiplier).round();
    if integer.abs() < (1_u64 << 50) as f64
        && value == integer / multiplier
        && integer % coefficient == 0.0
    {
        Some((integer / coefficient) as i128)
    } else {
        None
    }
}

// Convert the chosen integer count once. Multiplying a physical value back
// by a decimal scale can round a second time and move it off the venue grid.
fn venue_value(count: f64, step: f64, scale: Option<(f64, f64)>) -> f64 {
    if count.is_finite() && count.fract() == 0.0 && count.abs() < (1_u64 << 53) as f64 {
        if let Some((coefficient, multiplier)) = scale {
            let integer = count * coefficient;
            if integer.abs() < (1_u64 << 53) as f64 {
                return integer / multiplier;
            }
        }
        if let Some(value) = DecimalParts::from_float(step)
            .and_then(|step| {
                step.multiply(DecimalParts {
                    coefficient: count as i128,
                    exponent: 0,
                })
            })
            .and_then(DecimalParts::to_float)
        {
            return value;
        }
    }
    round_to_step_decimal_places(count * step, step)
}

fn round_at_venue_step(value: f64, step: f64, mode: RoundingMode) -> f64 {
    let units = value / step;
    let scale = exact_step_scale(step);
    let mut count = match mode {
        RoundingMode::Nearest => units.round(),
        RoundingMode::Floor => units.floor(),
        RoundingMode::Ceil => units.ceil(),
    };
    let fraction = units - units.floor();
    let boundary_distance = match mode {
        RoundingMode::Nearest => (fraction - 0.5).abs(),
        _ => fraction.min(1.0 - fraction),
    };
    // Only the exact decimal ratio decides a near-boundary result.
    if boundary_distance <= 4.0 * f64::EPSILON * units.abs() {
        if scale
            .and_then(|scale| small_decimal_units(value, scale))
            .is_some()
        {
            return value;
        }
        if let Some(exact) = DecimalParts::from_float(value).and_then(|value| {
            let step = DecimalParts::from_float(step)?;
            match mode {
                RoundingMode::Nearest => value.rounded_ratio(step, mode),
                // Preserve physical directional rounding for non-aligned
                // inputs, including residuals formed by callers.
                _ => value
                    .integer_ratio(step)
                    .filter(|count| count.unsigned_abs() < 1_u128 << 53),
            }
        }) {
            count = exact as f64;
        }
    }
    venue_value(count, step, scale)
}

fn decimal_unit_count(quantity: f64, step: DecimalParts) -> Option<i128> {
    let count = DecimalParts::from_float(quantity)?.integer_ratio(step)?;
    (count >= 0 && count < 1_i128 << 53).then_some(count)
}

fn decimal_aligned_price(base: f64, distance: f64, step: f64) -> Option<f64> {
    let factor = DecimalParts {
        coefficient: 1,
        exponent: 0,
    }
    .add(DecimalParts::from_float(distance)?)?;
    let target = DecimalParts::from_float(base)?.multiply(factor)?;
    let count = target.integer_ratio(DecimalParts::from_float(step)?)?;
    if count.unsigned_abs() >= 1_u128 << 52 {
        return None;
    }
    target.to_float()
}

/// Recover an exact decimal venue count only when both physical neighbors
/// remain distinguishable. Fractional or ambiguous inputs retain physical
/// arithmetic at their caller.
pub(crate) fn canonical_qty_units(quantity: f64, step: f64) -> Option<f64> {
    if !(quantity.next_up() - quantity <= step && quantity - quantity.next_down() <= step) {
        return None;
    }
    if quantity >= 0.0 {
        if let Some(count) =
            exact_step_scale(step).and_then(|scale| small_decimal_units(quantity, scale))
        {
            return Some(count as f64);
        }
    }
    decimal_unit_count(quantity, DecimalParts::from_float(step)?).map(|count| count as f64)
}

pub(crate) fn floor_qty_difference(total: f64, held: f64, step: f64) -> f64 {
    // Decreasing total and increasing held both decrease this difference. At
    // a power-of-two boundary, the opposite neighbors have different spacing.
    if total - total.next_down() <= step && held.next_up() - held <= step {
        if total >= 0.0 && held >= 0.0 {
            if let Some(scale) = exact_step_scale(step) {
                if let (Some(total), Some(held)) = (
                    small_decimal_units(total, scale),
                    small_decimal_units(held, scale),
                ) {
                    return venue_value((total - held).max(0) as f64, step, Some(scale));
                }
            }
        }
        let recover = || {
            let exact_step = DecimalParts::from_float(step)?;
            let total_units = decimal_unit_count(total, exact_step)?;
            let held_units = decimal_unit_count(held, exact_step)?;
            exact_step
                .multiply(DecimalParts {
                    coefficient: (total_units - held_units).max(0),
                    exponent: 0,
                })?
                .to_float()
        };
        if let Some(difference) = recover() {
            return difference;
        }
    }
    round_dn_preserve_step(total - held, step)
}

/// Form a relative price with one arithmetic rounding. A nearby residual may
/// be corrected only when the decimal operands prove exact venue alignment.
pub(crate) fn round_price_distance(base: f64, distance: f64, step: f64, mode: RoundingMode) -> f64 {
    let target = base.mul_add(distance, base);
    let units = target / step;
    let scale = exact_step_scale(step);
    let canonical = |count: f64| venue_value(count, step, scale);
    let nearest = canonical(units.round());
    if target == nearest {
        return nearest;
    }
    // This bound only avoids expensive proofs for non-boundary targets.
    // Exact decimal divisibility, not the bound, decides whether to align.
    let candidate_bound =
        4.0 * f64::EPSILON * base.abs().max(target.abs()) * (1.0 + distance.abs());
    if (target - nearest).abs() <= candidate_bound {
        if let Some(aligned) = decimal_aligned_price(base, distance, step) {
            return aligned;
        }
    }
    let count = match mode {
        RoundingMode::Nearest => units.round(),
        RoundingMode::Floor => units.floor(),
        RoundingMode::Ceil => units.ceil(),
    };
    canonical(count)
}

fn differs_only_by_float_representation(value: f64, aligned_value: f64) -> bool {
    let scale = value.abs().max(aligned_value.abs());
    (value - aligned_value).abs() <= f64::EPSILON * scale * 4.0
}

/// Directionally quantize without truncating valid increments below ten decimals.
pub fn round_dn_preserve_step(value: f64, step: f64) -> f64 {
    round_at_venue_step(value, step, RoundingMode::Floor)
}

/// Directionally quantize without truncating valid increments below ten decimals.
pub fn round_up_preserve_step(value: f64, step: f64) -> f64 {
    round_at_venue_step(value, step, RoundingMode::Ceil)
}

/// Snap ordinary binary noise at an aligned increment before rounding down.
pub fn tolerant_round_dn_preserve_step(value: f64, step: f64) -> f64 {
    let step_count = value / step;
    let nearest_count = step_count.round();
    let nearest_value = nearest_count * step;
    if differs_only_by_float_representation(value, nearest_value) {
        venue_value(nearest_count, step, exact_step_scale(step))
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
        venue_value(nearest_count, step, exact_step_scale(step))
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
    round_at_venue_step(n, step, RoundingMode::Nearest)
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
    fn canonical_quantity_proof_does_not_accept_adjacent_inputs() {
        use super::canonical_qty_units;
        for (value, step, count) in [(0.477_f64, 0.001, 477.0), (1.25, 0.25, 5.0)] {
            assert_eq!(canonical_qty_units(value, step), Some(count));
            assert_eq!(canonical_qty_units(value.next_up(), step), None);
            assert_eq!(canonical_qty_units(value.next_down(), step), None);
        }
        assert_eq!(canonical_qty_units(9045.269825510173, 3e-12), None);
    }

    #[test]
    fn large_venue_counts_keep_the_chosen_decimal_increment() {
        // Decimal cleanup must not add a digit after choosing a venue count.
        for sign in [-1.0, 1.0] {
            assert_eq!(
                round_(sign * (9045.269825510173 * 0.5), 3e-12),
                sign * 4522.634912755086
            );
            assert_eq!(
                round_(sign * 9045.269825510173, 3e-12),
                sign * 9045.269825510171
            );
            assert_eq!(
                round_dn(sign * 4522.634912755086, 3e-12),
                sign * 4522.634912755086
            );
            assert_eq!(
                round_up(sign * 4522.634912755086, 3e-12),
                sign * 4522.634912755086
            );
        }
        // Preserve an authored one-float offset, rather than applying a band.
        assert_eq!(round_dn(0.3_f64.next_down(), 0.1), 0.2);
        assert_eq!(round_up(0.3_f64.next_up(), 0.1), 0.4);
        assert_eq!(round_(0.25_f64.next_down(), 0.1), 0.2);
        assert_eq!(round_(0.25_f64.next_up(), 0.1), 0.3);
        assert_eq!(round_(0.25, 0.1), 0.3);
        assert_eq!(round_(-0.25, 0.1), -0.3);
    }

    #[test]
    fn decimal_target_residuals_preserve_aligned_ticks() {
        use super::{round_price_distance, RoundingMode};
        for mode in [RoundingMode::Floor, RoundingMode::Ceil] {
            assert_eq!(round_price_distance(11.2, -0.00625, 0.01, mode), 11.13);
            assert_eq!(round_price_distance(43.2, 0.00625, 0.01, mode), 43.47);
        }
        // A tiny distance cannot justify snapping a genuine adjacent input.
        assert_eq!(
            round_price_distance(11.13_f64.next_down(), 1e-300, 0.01, RoundingMode::Floor),
            11.12
        );
    }

    #[test]
    fn relative_price_quantization_preserves_exact_venue_alignment() {
        use super::{round_price_distance, RoundingMode};
        for (base, distance, step, expected) in [
            (131.25, -0.0016, 0.01, 131.04),
            (100.0, 0.0011, 0.01, 100.11),
            (2.5, -0.044, 0.01, 2.39),
            (0.29, 0.0, 0.01, 0.29),
            (10.0, 0.025, 0.25, 10.25),
            (1e-10, -0.01, 3e-12, 9.9e-11),
        ] {
            assert_eq!(
                round_price_distance(base, distance, step, RoundingMode::Floor),
                expected
            );
            assert_eq!(
                round_price_distance(base, distance, step, RoundingMode::Ceil),
                expected
            );
        }
        assert_eq!(
            round_price_distance(100.0, 0.00111, 0.01, RoundingMode::Floor),
            100.11
        );
        assert_eq!(
            round_price_distance(100.0, 0.00111, 0.01, RoundingMode::Ceil),
            100.12
        );
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
