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

/// Lossless finite-limb transport for the canonical decimal operands used by
/// GPU venue proofs. The sentinel declines a proof; physical arithmetic remains
/// available for coefficients/exponents outside the checked GPU integer domain.
#[pyfunction]
pub fn gpu_decimal_metadata_py(values: Vec<f64>) -> Vec<(u32, u32, u32, i32)> {
    values
        .into_iter()
        .map(|value| {
            let Some(parts) = DecimalParts::from_float(value) else {
                return (0, 0, 0, 2048);
            };
            let coefficient = parts.coefficient.unsigned_abs();
            if coefficient >= 1_u128 << 60 || !(-18..=18).contains(&parts.exponent) {
                return (0, 0, 0, 2048);
            }
            let mask = (1_u128 << 20) - 1;
            (
                (coefficient & mask) as u32,
                ((coefficient >> 20) & mask) as u32,
                ((coefficient >> 40) & mask) as u32,
                parts.exponent + if parts.coefficient < 0 { 1024 } else { 0 },
            )
        })
        .collect()
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

pub(crate) fn round_qty_unit_product(units: f64, factor: f64, step: f64) -> Option<f64> {
    if !units.is_finite() || units < 0.0 || units.fract() != 0.0 || units >= (1_u64 << 53) as f64 {
        return None;
    }
    // An exact integer numerator below this bound keeps distinct decimal
    // sides of a half-step several floats apart after division.
    if let Some((coefficient, multiplier)) = exact_step_scale(factor) {
        let numerator = units * coefficient;
        if numerator < (1_u64 << 50) as f64 {
            return Some(venue_value(
                (numerator / multiplier).round(),
                step,
                exact_step_scale(step),
            ));
        }
    }
    let product = DecimalParts {
        coefficient: units as i128,
        exponent: 0,
    }
    .multiply(DecimalParts::from_float(factor)?)?;
    let count = product.rounded_ratio(
        DecimalParts {
            coefficient: 1,
            exponent: 0,
        },
        RoundingMode::Nearest,
    )?;
    Some(venue_value(count as f64, step, exact_step_scale(step)))
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

/// Preserve the two effective authored distances through an exact alignment
/// proof. Non-aligned pairs retain the existing composed-f64/FMA decision.
pub(crate) fn round_compound_price_distance(
    base: f64,
    first: f64,
    second: f64,
    step: f64,
    mode: RoundingMode,
) -> f64 {
    let distance = first + second;
    let target = base.mul_add(distance, base);
    let nearest = venue_value((target / step).round(), step, exact_step_scale(step));
    // This bound only selects proof candidates; divisibility decides alignment.
    // Keep both operand magnitudes because the composed distance may cancel.
    let candidate_bound =
        8.0 * f64::EPSILON * base.abs().max(target.abs()) * (1.0 + first.abs() + second.abs());
    if (target - nearest).abs() <= candidate_bound {
        let prove = || {
            let factor = DecimalParts {
                coefficient: 1,
                exponent: 0,
            }
            .add(DecimalParts::from_float(first)?)?
            .add(DecimalParts::from_float(second)?)?;
            let exact_target = DecimalParts::from_float(base)?.multiply(factor)?;
            let count = exact_target.integer_ratio(DecimalParts::from_float(step)?)?;
            if count.unsigned_abs() >= 1_u128 << 52 {
                return None;
            }
            exact_target.to_float()
        };
        if let Some(aligned) = prove() {
            return aligned;
        }
    }
    // The original pair did not prove alignment. Do not re-prove its rounded
    // sum as if that were an independently authored decimal distance.
    // An already canonical physical target remains exact even when binary
    // division by the venue step would land just above an integer.
    if target == nearest {
        return nearest;
    }
    let units = target / step;
    let count = match mode {
        RoundingMode::Nearest => units.round(),
        RoundingMode::Floor => units.floor(),
        RoundingMode::Ceil => units.ceil(),
    };
    venue_value(count, step, exact_step_scale(step))
}

fn differs_only_by_float_representation(value: f64, aligned_value: f64) -> bool {
    let scale = value.abs().max(aligned_value.abs());
    (value - aligned_value).abs() <= f64::EPSILON * scale * 4.0
}

// Convert an integer venue count through the step's canonical decimal value.
// A raw binary product or decimal cleanup can land on an adjacent float at
// large magnitudes; accept the canonical decimal representation as well.
fn canonical_step_value(count: f64, step: f64) -> Option<f64> {
    if count < 0.0 || count.fract() != 0.0 {
        return None;
    }
    let text = step.to_string();
    let (mantissa, exponent) = match text.split_once('e') {
        Some((mantissa, exponent)) => (mantissa, exponent.parse::<i32>().ok()?),
        None => (text.as_str(), 0),
    };
    let places = mantissa
        .split_once('.')
        .map_or(0, |(_, tail)| tail.len() as i32);
    let digits = mantissa.replace('.', "");
    let trimmed = digits.trim_end_matches('0');
    let coefficient = trimmed.parse::<u128>().ok()?.checked_mul(count as u128)?;
    let exponent = exponent - places + (digits.len() - trimmed.len()) as i32;
    format!("{}e{}", coefficient, exponent).parse::<f64>().ok()
}

pub const GPU_MARKET_EXECUTION_SELL_LANE: usize = 0;
pub const GPU_MARKET_EXECUTION_BUY_LANE: usize = 1;

/// Recover only an exact, unique positive int32 venue count. The quotient
/// selects candidates; canonical decimal equality is the acceptance proof.
fn gpu_market_price_tick_count(price: f64, step: f64) -> Result<i32, &'static str> {
    if !price.is_finite() || price <= 0.0 || !step.is_finite() || step <= 0.0 {
        return Err("GPU market price and effective step must be finite and positive");
    }
    let nearest = (price / step).round();
    if !nearest.is_finite() || nearest < 0.0 || nearest > i32::MAX as f64 + 1.0 {
        return Err("GPU market price count exceeds the positive int32 domain");
    }
    let mut proven = None;
    for count in [nearest - 1.0, nearest, nearest + 1.0] {
        if count < 1.0 || count > i32::MAX as f64 {
            continue;
        }
        if canonical_step_value(count, step) != Some(price) {
            continue;
        }
        if proven.is_some()
            || canonical_step_value(count - 1.0, step) == Some(price)
            || canonical_step_value(count + 1.0, step) == Some(price)
        {
            return Err("GPU market price does not have a unique venue count");
        }
        proven = Some(count as i32);
    }
    proven.ok_or("GPU market price does not exactly round-trip through its venue count")
}

/// Prepare SELL then BUY counts using the existing CPU market-fill rounding.
/// Invalid active data or an unprovable count is an error, never a sentinel.
pub fn gpu_market_execution_tick_counts(
    close: f64,
    price_step: f64,
    slippage: f64,
) -> Result<[i32; 2], &'static str> {
    if !close.is_finite() || close <= 0.0 {
        return Err("Active GPU market closes must be finite and positive");
    }
    if !price_step.is_finite() || price_step < 0.0 {
        return Err("GPU market price steps must be finite and nonnegative");
    }
    if !slippage.is_finite() || slippage < 0.0 {
        return Err("GPU market slippage must be finite and nonnegative");
    }
    let close = close.max(f64::EPSILON);
    let step = price_step.max(f64::EPSILON);
    let sell = round_dn(close * (1.0 - slippage), step).max(step);
    let buy = round_up(close * (1.0 + slippage), step).max(step);
    let mut counts = [0; 2];
    counts[GPU_MARKET_EXECUTION_SELL_LANE] = gpu_market_price_tick_count(sell, step)?;
    counts[GPU_MARKET_EXECUTION_BUY_LANE] = gpu_market_price_tick_count(buy, step)?;
    Ok(counts)
}

/// Recover a venue count only when the quantity uniquely round-trips in the
/// direction which could make a remaining position smaller.
fn close_operand_step_count(quantity: f64, step: f64, upward: bool) -> Option<u64> {
    if !quantity.is_finite() || quantity <= 0.0 || !step.is_finite() || step <= 0.0 {
        return None;
    }
    let bits = if upward {
        quantity.to_bits() + 1
    } else {
        quantity.to_bits() - 1
    };
    if (quantity - f64::from_bits(bits)).abs() > step {
        return None;
    }
    let nearest = (quantity / step).round();
    // Division can round before integer rounding at large counts.
    let residual = (-nearest).mul_add(step, quantity);
    let count = if residual >= step * 0.5 {
        nearest + 1.0
    } else if residual < -step * 0.5 {
        nearest - 1.0
    } else {
        nearest
    };
    const MAX_EXACT_STEPS: f64 = 9_007_199_254_740_992.0;
    if !count.is_finite() || count < 0.0 || count > MAX_EXACT_STEPS {
        return None;
    }
    let direction = if upward { 1.0 } else { -1.0 };
    let canonical_matches = |count: f64| canonical_step_value(count, step) == Some(quantity);
    // Canonical decimal quantities have their own authority. Do not invent
    // ambiguity by mixing them with a different count's raw binary product.
    // The binary quotient can select an adjacent count, so inspect both sides.
    let candidates = if upward {
        [count + 1.0, count, count - 1.0]
    } else {
        [count - 1.0, count, count + 1.0]
    };
    let mut canonical_seen = false;
    for candidate in candidates {
        if candidate < 0.0 || candidate > MAX_EXACT_STEPS || !canonical_matches(candidate) {
            continue;
        }
        canonical_seen = true;
        let neighbor = candidate + direction;
        if neighbor != candidate && !canonical_matches(neighbor) {
            return Some(candidate as u64);
        }
    }
    if canonical_seen {
        return None;
    }
    // Existing sizing also emits raw products and decimal-cleaned products.
    // Accept only exact, directionally unique roundtrips in that family.
    let matches = |count: f64| {
        let raw = count * step;
        quantity == raw || quantity == round_to_step_decimal_places(raw, step)
    };
    let neighbor = count + direction;
    (matches(count) && neighbor != count && !matches(neighbor)).then_some(count as u64)
}

#[derive(Debug, PartialEq, Eq)]
pub enum CloseWaveRemainder {
    NoRemainder,
    MeetsMinimum,
    BelowMinimum { proven_steps: Option<u64> },
}

/// Classify a close wave using proven per-leg venue counts when floating
/// arithmetic can affect its minimum or the dust-absorption budget.
/// Keep a proven remaining budget so finalization need not subtract float sums
/// again after deciding to absorb dust.
pub fn classify_close_wave_remainder<I>(
    position_size_abs: f64,
    clips: I,
    min_qty: f64,
    qty_step: f64,
) -> CloseWaveRemainder
where
    I: Iterator<Item = f64> + ExactSizeIterator + Clone,
{
    let total_abs: f64 = clips.clone().sum();
    let remainder = position_size_abs - total_abs;
    // A zero requirement cannot leave subminimum dust. Preserve every positive
    // remainder without requiring venue-grid proof of otherwise valid operands.
    if min_qty == 0.0 {
        return if remainder > 0.0 {
            CloseWaveRemainder::MeetsMinimum
        } else {
            CloseWaveRemainder::NoRemainder
        };
    }
    // This selects the proof path; it never grants a quantity tolerance.
    // Include every addition in the naive close-wave sum's error bound.
    let representation_tolerance = f64::EPSILON
        * position_size_abs
            .abs()
            .max(total_abs.abs())
            .max(min_qty.abs())
        * (clips.len() as f64 + 3.0);
    if clips.len() == 1 && clips.clone().next() == Some(position_size_abs) {
        return CloseWaveRemainder::NoRemainder;
    }
    if remainder >= min_qty && remainder - min_qty > representation_tolerance {
        return CloseWaveRemainder::MeetsMinimum;
    }
    if remainder < -representation_tolerance {
        return CloseWaveRemainder::NoRemainder;
    }
    // Prove positive dust even far below the minimum so absorption keeps its
    // exact count budget. A sum near zero can also conceal positive dust,
    // regardless of how large the minimum is.
    let prove = || {
        let position_steps = close_operand_step_count(position_size_abs, qty_step, false)?;
        let minimum_steps = close_operand_step_count(min_qty, qty_step, true)?;
        // Do not reconstruct cardinality from an already rounded aggregate.
        let mut total_steps = 0_u64;
        for clip in clips {
            total_steps =
                total_steps.checked_add(close_operand_step_count(clip, qty_step, true)?)?;
        }
        Some((position_steps.checked_sub(total_steps)?, minimum_steps))
    };
    match prove() {
        Some((0, _)) => CloseWaveRemainder::NoRemainder,
        Some((steps, minimum)) if steps >= minimum => CloseWaveRemainder::MeetsMinimum,
        Some((steps, _)) => CloseWaveRemainder::BelowMinimum {
            proven_steps: Some(steps),
        },
        // A positive ambiguous remainder stays conservative. A float sum which
        // already covers the position retains its existing handling unless an
        // actual positive subminimum remainder was independently proven.
        None if remainder <= 0.0 => CloseWaveRemainder::NoRemainder,
        None => CloseWaveRemainder::BelowMinimum { proven_steps: None },
    }
}

pub fn close_qty_absorbing_proven_remainder(
    clip_qty: f64,
    remaining_steps: u64,
    qty_step: f64,
) -> Option<f64> {
    let count = close_operand_step_count(clip_qty, qty_step, true)?.checked_add(remaining_steps)?;
    if count > 1_u64 << 53 {
        return None;
    }
    canonical_step_value(count as f64, qty_step).filter(|quantity| quantity.is_finite())
}

/// The strategy's single clip and the final order wave use the same contract.
pub fn close_remainder_meets_minimum(
    position_size_abs: f64,
    clip_qty: f64,
    min_qty: f64,
    qty_step: f64,
) -> bool {
    !matches!(
        classify_close_wave_remainder(
            position_size_abs,
            std::iter::once(clip_qty),
            min_qty,
            qty_step
        ),
        CloseWaveRemainder::BelowMinimum { .. }
    )
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

/// Normalize the available completed-sample bootstrap, then discard its weights.
/// The caller bounds the historical prefix and owns subsequent scalar advances.
pub fn ema_bootstrap_f64(values: impl IntoIterator<Item = f64>, alpha: f64) -> f64 {
    let decay = 1.0 - alpha.clamp(0.0, 1.0);
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for value in values {
        if value.is_finite() {
            numerator = value + decay * numerator;
            denominator = 1.0 + decay * denominator;
        }
    }
    if denominator > 0.0 {
        numerator / denominator
    } else {
        f64::NAN
    }
}

/// Bootstrap from at most two spans of the available completed suffix.
pub fn ema_bootstrap_count(span_samples: f64, available: u64) -> u64 {
    // Float-to-integer saturation also bounds overflow of two very large spans.
    ((2.0 * span_samples).ceil().max(1.0) as u64).min(available)
}

pub fn ema_last_f64(values: &[f64], span: f64) -> f64 {
    let count = ema_bootstrap_count(span, values.len() as u64) as usize;
    let start = values.len().saturating_sub(count);
    ema_bootstrap_f64(
        values[start..].iter().copied(),
        (2.0 / (span + 1.0)).min(1.0),
    )
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
    #[test]
    fn zero_close_minimum_needs_no_operand_grid_proof() {
        use super::{classify_close_wave_remainder, CloseWaveRemainder};
        for (position, clips, step) in [
            (8192.0, vec![8191.999999999999], 1e-12),
            (
                f64::from_bits(32768.0_f64.to_bits() - 1),
                vec![32767.99999999999],
                1e-12,
            ),
            (8192.0, vec![4096.0, 4095.999999999999], 1e-12),
        ] {
            assert_eq!(
                classify_close_wave_remainder(position, clips.into_iter(), 0.0, step),
                CloseWaveRemainder::MeetsMinimum
            );
        }
        assert_eq!(
            classify_close_wave_remainder(8192.0, std::iter::once(8192.0), 0.0, 1e-12),
            CloseWaveRemainder::NoRemainder
        );
    }

    #[test]
    fn close_remainder_requires_unique_operand_grid_roundtrips() {
        let step = 0.001;
        let minimum = 9.0 * step;
        let large_clip = (10_000_000_000_000.0 - 9.0) * step;
        assert!(super::close_remainder_meets_minimum(
            1e10, large_clip, minimum, step
        ));
        for position in [f64::from_bits(1e10_f64.to_bits() - 1), 1e10 - 5e-6] {
            assert!(!super::close_remainder_meets_minimum(
                position, large_clip, minimum, step
            ));
        }
        let clip = 32767.991_f64;
        assert!(super::close_remainder_meets_minimum(
            32768.0, clip, minimum, step
        ));
        for (position, clip, minimum) in [
            (f64::from_bits(32768.0_f64.to_bits() - 1), clip, minimum),
            (32768.0, f64::from_bits(clip.to_bits() + 1), minimum),
            (32768.0, clip, f64::from_bits(minimum.to_bits() + 1)),
        ] {
            assert!(!super::close_remainder_meets_minimum(
                position, clip, minimum, step
            ));
        }
        // A raw product and its next count's canonical decimal value can
        // alias while physical float neighbors remain within one venue step.
        let count = 8_000_000_000_000_014.0;
        let quantity = count * step;
        assert_eq!(
            super::canonical_step_value(count + 1.0, step),
            Some(quantity)
        );
        assert_eq!(
            super::close_operand_step_count(quantity, step, true),
            Some(count as u64 + 1)
        );
        // Large count resolution does not justify an off-grid minimum.
        let position = (1_u64 << 53) as f64;
        assert!(!super::close_remainder_meets_minimum(
            position,
            position - 1.0,
            1.5,
            1.0
        ));
    }

    use super::{
        ema_last_f64, round_, round_dn, round_up, tolerant_round_dn_preserve_step,
        tolerant_round_up_preserve_step,
    };

    #[test]
    fn gpu_market_ticks_preserve_the_native_f64_market_boundary() {
        let close = 167.84391568013112_f64;
        let step = 0.01_f64;
        let slippage = 0.0005_f64;
        let ticks = super::gpu_market_execution_tick_counts(close, step, slippage).unwrap();
        assert_eq!(ticks, [16775, 16793]);
        let old_sell = ((close as f32 * (1.0_f32 - slippage as f32)) / step as f32).floor() as i32;
        assert_eq!(old_sell, 16776);
        assert_eq!(
            super::canonical_step_value(ticks[0] as f64, step),
            Some(round_dn(close * (1.0 - slippage), step).max(step))
        );
        assert_eq!(
            super::canonical_step_value(ticks[1] as f64, step),
            Some(round_up(close * (1.0 + slippage), step).max(step))
        );
    }

    #[test]
    fn gpu_market_ticks_match_existing_rounding_at_neighboring_boundaries() {
        for (close, expected) in [
            (0.3_f64.next_down(), [2, 3]),
            (0.3, [3, 3]),
            (0.3_f64.next_up(), [3, 4]),
        ] {
            assert_eq!(
                super::gpu_market_execution_tick_counts(close, 0.1, 0.0).unwrap(),
                expected
            );
        }
        for (close, authored_step, slippage) in [
            (1e-30_f64, 0.0_f64, 0.0_f64),
            (1e-30, 1e-300, 0.0),
            (1e-10, 3e-12, 0.0),
            (3e200, 1e200, 0.0),
            (1e300, 1e300, 0.0),
            (0.1, 0.01, 1.0),
            (0.1, 0.01, 2.0),
        ] {
            let step = authored_step.max(f64::EPSILON);
            let close = close.max(f64::EPSILON);
            let actual =
                super::gpu_market_execution_tick_counts(close, authored_step, slippage).unwrap();
            assert_eq!(
                super::canonical_step_value(actual[0] as f64, step),
                Some(round_dn(close * (1.0 - slippage), step).max(step))
            );
            assert_eq!(
                super::canonical_step_value(actual[1] as f64, step),
                Some(round_up(close * (1.0 + slippage), step).max(step))
            );
        }
    }

    #[test]
    fn gpu_market_tick_proof_rejects_off_grid_values_and_int32_overflow() {
        assert_eq!(super::gpu_market_price_tick_count(0.01, 0.01), Ok(1));
        let maximum = super::canonical_step_value(i32::MAX as f64, 0.01).unwrap();
        assert_eq!(
            super::gpu_market_price_tick_count(maximum, 0.01),
            Ok(i32::MAX)
        );
        let overflow = super::canonical_step_value(i32::MAX as f64 + 1.0, 0.01).unwrap();
        for price in [
            0.0,
            -0.01,
            f64::NAN,
            f64::INFINITY,
            0.3_f64.next_up(),
            overflow,
        ] {
            assert!(super::gpu_market_price_tick_count(price, 0.01).is_err());
        }
        assert!(super::gpu_market_price_tick_count(0.3_f64.next_up(), 0.1).is_err());
        assert!(super::gpu_market_execution_tick_counts(overflow, 0.01, 0.0).is_err());
        assert!(super::gpu_market_execution_tick_counts(f64::MAX, 1.0, 1.0).is_err());
    }

    #[test]
    fn gpu_market_ticks_reject_invalid_active_operands() {
        for close in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(super::gpu_market_execution_tick_counts(close, 0.01, 0.0).is_err());
        }
        for step in [-0.01, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(super::gpu_market_execution_tick_counts(1.0, step, 0.0).is_err());
        }
        for slippage in [-0.001, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(super::gpu_market_execution_tick_counts(1.0, 0.01, slippage).is_err());
        }
    }

    #[test]
    fn public_rounding_cleans_float_noise_without_truncating_tiny_steps() {
        assert_eq!(round_(64.850000000000093, 0.01), 64.85);
        assert_eq!(round_(0.06469999999999954, 0.0001), 0.0647);

        assert_eq!(round_(1e-10, 3e-12), 9.9e-11);
        assert_eq!(round_dn(0.999999999999, 3e-12), 0.999999999999);
        assert_eq!(round_up(1.000000000001, 3e-12), 1.000000000002);
    }

    #[test]
    fn gpu_decimal_transport_preserves_authored_sides_of_boundaries() {
        use super::gpu_decimal_metadata_py;
        let input = [1.15, 1.15 - 1e-12, 1.15 + 1e-12, -0.044, 0.0];
        let metadata = gpu_decimal_metadata_py(input.to_vec());
        assert_ne!(metadata[0], metadata[1]);
        assert_ne!(metadata[0], metadata[2]);
        for (value, (lo, mid, hi, encoded)) in input.into_iter().zip(metadata) {
            assert!(lo < 1 << 20 && mid < 1 << 20 && hi < 1 << 20);
            let magnitude = lo as u64 | (mid as u64) << 20 | (hi as u64) << 40;
            let negative = encoded >= 512;
            let exponent = encoded - if negative { 1024 } else { 0 };
            let restored: f64 = format!(
                "{}{}e{}",
                if negative { "-" } else { "" },
                magnitude,
                exponent
            )
            .parse()
            .unwrap();
            assert_eq!(restored, value);
        }
        assert_eq!(
            gpu_decimal_metadata_py(vec![f64::NAN, 1e-100, 1e100]),
            vec![(0, 0, 0, 2048); 3]
        );
    }

    #[test]
    fn compound_price_alignment_preserves_operands_without_snapping_offsets() {
        use super::{round_compound_price_distance, RoundingMode};
        for (first, second, floor, ceil) in [
            (0.1, -0.09375, 43.47, 43.47),
            (-0.1, 0.09375, 42.93, 42.93),
            (0.1 + 1e-12, -0.09375, 43.47, 43.48),
            (0.1 - 1e-12, -0.09375, 43.46, 43.47),
            (0.1001, -0.09375, 43.47, 43.48),
        ] {
            assert_eq!(
                round_compound_price_distance(43.2, first, second, 0.01, RoundingMode::Floor),
                floor
            );
            assert_eq!(
                round_compound_price_distance(43.2, first, second, 0.01, RoundingMode::Ceil),
                ceil
            );
        }
        // Nonaligned decimals can legitimately round onto a tick in the fused
        // f64 path. An exact-rational directional result must not replace it.
        assert_eq!(
            round_compound_price_distance(
                6532.59,
                -0.025626895304925,
                0.0,
                0.01,
                RoundingMode::Floor
            ),
            6365.18
        );
    }

    #[test]
    fn failed_compound_proof_keeps_the_fused_target_without_reproving_the_sum() {
        use super::{exact_step_scale, round_compound_price_distance, venue_value, RoundingMode};
        for (base, first, second) in [
            (43.2_f64, 0.01_f64, -0.0037499999999999994_f64),
            (43.2, 0.006927290310531164, -0.0006772903105311638),
            (43.2, 0.05788483791847438, -0.007884837918474379),
            (11.2, -0.01, 0.0037499999999999994),
        ] {
            let target = base.mul_add(first + second, base);
            let units = target / 0.01;
            for (mode, count) in [
                (RoundingMode::Floor, units.floor()),
                (RoundingMode::Ceil, units.ceil()),
                (RoundingMode::Nearest, units.round()),
            ] {
                let expected = venue_value(count, 0.01, exact_step_scale(0.01));
                assert_eq!(
                    round_compound_price_distance(base, first, second, 0.01, mode),
                    expected
                );
            }
        }
        assert_eq!(
            round_compound_price_distance(
                43.2,
                0.01,
                -0.0037499999999999994,
                0.01,
                RoundingMode::Ceil
            ),
            43.48
        );
        assert_eq!(
            round_compound_price_distance(
                11.2,
                -0.01,
                0.0037499999999999994,
                0.01,
                RoundingMode::Floor
            ),
            11.12
        );
        // A physical target already equal to the canonical venue value needs
        // no distance proof. Binary division alone must not promote .07 to .08.
        let base = 0.07_f64;
        let distance = 0.01_f64 - 0.009999999999999998_f64;
        assert_eq!(base.mul_add(distance, base), base);
        assert!((base / 0.01).ceil() * 0.01 > base);
        for mode in [
            RoundingMode::Floor,
            RoundingMode::Ceil,
            RoundingMode::Nearest,
        ] {
            assert_eq!(
                round_compound_price_distance(base, 0.01, -0.009999999999999998, 0.01, mode),
                base
            );
        }
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
    fn ema_bootstrap_normalizes_then_uses_scalar_forward_recurrence() {
        let seed = ema_last_f64(&[2.0, 4.0], 3.0);
        assert!((seed - 10.0 / 3.0).abs() < 1e-12);
        let advanced = 0.5 * seed + 0.5 * 8.0;
        assert!((advanced - 17.0 / 3.0).abs() < 1e-12);
        assert!((advanced - ema_last_f64(&[2.0, 4.0, 8.0], 3.0)).abs() > 0.1);
        assert_eq!(ema_last_f64(&[2.0], 3.25), 2.0);
        assert_eq!(
            ema_last_f64(&[f64::NAN, 2.0, f64::INFINITY, 4.0], 3.0),
            seed
        );
        assert!(ema_last_f64(&[f64::NAN, f64::INFINITY], 3.0).is_nan());
        let values = [99.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(
            ema_last_f64(&values, 2.25),
            ema_last_f64(&values[1..], 2.25)
        );
        assert_eq!(ema_last_f64(&[2.0, 4.0], 0.5), 4.0);
    }
}
