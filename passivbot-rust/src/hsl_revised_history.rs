//! Best-effort pair history, independent of legacy episode readiness.
//! This component does not certify lifecycle boundaries or authorize orders.

use crate::hsl_revised_sum::CurrencySum;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum PositionSide {
    Long,
    Short,
}

impl PositionSide {
    fn direction(self) -> f64 {
        if self == Self::Long {
            1.0
        } else {
            -1.0
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Position {
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
    pub size: f64,
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
    pub basis: f64,
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
    pub mark: f64,
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
    pub multiplier: f64,
    /// Optional exchange quantity quantum for judging historical roundoff.
    #[serde(default, deserialize_with = "crate::hsl_revised_json::optional")]
    pub quantity_step: Option<f64>,
    pub inverse: bool,
    pub pside: PositionSide,
}

impl Position {
    fn validate(&self) -> Result<(), String> {
        if [self.size, self.basis, self.mark, self.multiplier]
            .iter()
            .any(|v| !v.is_finite())
            || self
                .quantity_step
                .is_some_and(|q| !q.is_finite() || q <= 0.0)
            || self.mark <= 0.0
            || self.multiplier <= 0.0
            || (self.size != 0.0 && self.basis <= 0.0)
            || self.size * self.pside.direction() < 0.0
        {
            return Err("invalid current revised HSL position".into());
        }
        Ok(())
    }

    fn pnl(&self, size: f64, basis: f64, price: f64) -> f64 {
        if size == 0.0 || price == basis {
            return 0.0;
        }
        let change = if self.inverse {
            // Equivalent to reciprocal subtraction, without inf-inf for tiny
            // positive prices. Ordinary representable cases retain their units.
            (price - basis) / basis / price
        } else {
            price - basis
        };
        let value = size * self.multiplier * change;
        if value.is_nan() {
            // Extreme overflow/underflow can create inf*0. Keep the known PnL
            // direction; the caller bounds and discloses the range approximation.
            f64::INFINITY.copysign(size.signum() * (price - basis).signum())
        } else {
            value
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Fill {
    pub identity: String,
    pub timestamp: i64,
    #[serde(default, deserialize_with = "crate::hsl_revised_json::optional")]
    pub delta: Option<f64>,
    #[serde(default, deserialize_with = "crate::hsl_revised_json::optional")]
    pub price: Option<f64>,
    #[serde(default, deserialize_with = "crate::hsl_revised_json::optional")]
    pub realized: Option<f64>,
    #[serde(default, deserialize_with = "crate::hsl_revised_json::optional")]
    pub fee: Option<f64>,
    pub sequence: Option<i64>,
    pub revision: u64,
}

pub fn default_fills_before_price() -> bool {
    true
}

pub fn fill_precedes_price(fill: i64, price: i64, end: i64, before: bool) -> bool {
    fill < price || (fill == price && (before || price == end))
}

fn usable(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite())
}
fn positive(value: Option<f64>) -> Option<f64> {
    usable(value).filter(|v| *v > 0.0)
}

pub fn canonical_fills(
    fills: &[Fill],
    start: i64,
    end: i64,
    pside: PositionSide,
    reasons: &mut BTreeSet<String>,
) -> Vec<Fill> {
    let mut identities: BTreeMap<&str, Vec<&Fill>> = BTreeMap::new();
    for fill in fills {
        identities.entry(&fill.identity).or_default().push(fill);
    }
    let mut selected = Vec::new();
    for versions in identities.values() {
        let revision = versions.iter().map(|f| f.revision).max().unwrap();
        let latest: Vec<_> = versions.iter().filter(|f| f.revision == revision).collect();
        let first = latest[0];
        if latest.iter().any(|f| *f != first) {
            reasons.insert("conflicting_identity".into());
            continue;
        }
        // Revisions are selected before clipping: an out-of-window correction
        // supersedes its earlier in-window observation.
        if first.timestamp < start || first.timestamp > end {
            continue;
        }
        if usable(first.delta).is_none_or(|v| v == 0.0) {
            reasons.insert("invalid_quantity".into());
        }
        selected.push((**first).clone());
    }
    selected.sort_by(|a, b| {
        a.timestamp
            .cmp(&b.timestamp)
            .then_with(|| a.sequence.is_none().cmp(&b.sequence.is_none()))
            .then_with(|| a.sequence.cmp(&b.sequence))
            .then_with(|| {
                (usable(a.delta).unwrap_or(0.0) * pside.direction() < 0.0)
                    .cmp(&(usable(b.delta).unwrap_or(0.0) * pside.direction() < 0.0))
            })
            .then_with(|| a.identity.cmp(&b.identity))
    });
    let mut index = 0;
    while index < selected.len() {
        let end = selected[index..]
            .iter()
            .position(|f| f.timestamp != selected[index].timestamp)
            .map_or(selected.len(), |n| index + n);
        let cohort = &selected[index..end];
        if cohort.len() > 1
            && (cohort.iter().any(|f| f.sequence.is_none())
                || cohort
                    .iter()
                    .map(|f| f.sequence)
                    .collect::<BTreeSet<_>>()
                    .len()
                    != cohort.len())
        {
            reasons.insert("estimated_fill_order".into());
        }
        index = end;
    }
    selected
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    /// Historical candle samples normally observe all fills at their timestamp.
    /// The simulator explicitly phases a bar's fills after its opening boundary.
    #[serde(default = "default_fills_before_price")]
    pub fills_before_same_time_price: bool,
    pub start: i64,
    pub end: i64,
    pub position: Position,
    pub fills: Vec<Fill>,
    /// Already normalized historical close samples. Gap filling/resampling is a
    /// separate component; this primitive also accepts an empty history grid.
    #[serde(deserialize_with = "crate::hsl_revised_json::prices")]
    pub prices: BTreeMap<i64, f64>,
}

#[derive(Debug, Serialize)]
pub struct Sample {
    pub timestamp: i64,
    pub pnl: f64,
    pub upnl: f64,
    pub size: f64,
    pub basis: f64,
}

#[derive(Debug, Serialize)]
pub struct Event {
    pub fill: Fill,
    pub before: f64,
    pub after: f64,
    pub basis: f64,
    pub gross_realized: f64,
    pub fee: f64,
    pub realized_cumsum: f64,
    pub quantity_estimated: bool,
}

#[derive(Debug, Serialize)]
pub struct History {
    pub samples: Vec<Sample>,
    pub events: Vec<Event>,
    pub reasons: BTreeSet<String>,
}

fn finite(value: f64, reasons: &mut BTreeSet<String>) -> f64 {
    if value.is_finite() {
        value
    } else {
        reasons.insert("numeric_range_approximation".into());
        if value.is_sign_negative() {
            -f64::MAX
        } else {
            f64::MAX
        }
    }
}

pub fn reconstruct(input: &Input) -> Result<History, String> {
    input.position.validate()?;
    if input.start > input.end {
        return Err("reversed HSL history interval".into());
    }
    let p = &input.position;
    let direction = p.pside.direction();
    let mut reasons = BTreeSet::new();
    let fills = canonical_fills(&input.fills, input.start, input.end, p.pside, &mut reasons);
    let mut steps = Vec::with_capacity(fills.len());
    let mut after = p.size.abs();
    let mut quantity_compensation = 0.0;
    let mut quantity_scale = after;
    for f in fills.iter().rev() {
        let delta = usable(f.delta).unwrap_or(0.0) * direction;
        // Compensate repeated decimal lot additions so many partial fills do not
        // accumulate a fictitious opening discrepancy.
        let increment = -delta - quantity_compensation;
        let sum = after + increment;
        quantity_compensation = if sum.is_finite() {
            (sum - after) - increment
        } else {
            0.0
        };
        let mut raw_before = finite(sum, &mut reasons);
        // Cancellation of decimal lot quantities can leave a few binary ulps.
        // This is arithmetic roundoff, not evidence of a contradictory episode.
        // Residual error can originate in a larger later position in this suffix.
        quantity_scale = quantity_scale.max(after.abs()).max(delta.abs());
        let tolerance = 8.0 * f64::EPSILON * quantity_scale;
        let mut rounded_without_precision = false;
        if raw_before != 0.0 && raw_before.abs() <= tolerance {
            rounded_without_precision = p.quantity_step.is_none_or(|step| tolerance >= step / 2.0);
            if rounded_without_precision {
                reasons.insert("quantity_precision_unavailable".into());
            }
            reasons.insert("quantity_roundoff".into());
            raw_before = 0.0;
        }
        if raw_before < 0.0 {
            reasons.insert("clamped_quantity".into());
        }
        let before = raw_before.max(0.0);
        if before == 0.0 {
            quantity_compensation = 0.0;
            quantity_scale = 0.0;
        }
        steps.push((
            f,
            before,
            after,
            delta,
            rounded_without_precision
                || raw_before < 0.0
                || usable(f.delta).is_none_or(|v| v == 0.0),
        ));
        after = before;
    }
    steps.reverse();
    let mut basis = fills
        .iter()
        .find_map(|f| positive(f.price))
        .unwrap_or(if p.basis > 0.0 { p.basis } else { p.mark });
    if after != 0.0 {
        reasons.insert("estimated_opening_basis".into());
    } else {
        basis = 0.0;
    }
    let opening_quantity = after;
    let opening_basis = basis;
    let mut cumulative = CurrencySum::new();
    let mut events = Vec::with_capacity(steps.len());
    for (f, before, after, delta, quantity_estimated) in steps {
        let price = positive(f.price).unwrap_or_else(|| {
            reasons.insert("estimated_fill_price".into());
            if basis > 0.0 {
                basis
            } else if p.basis > 0.0 {
                p.basis
            } else {
                p.mark
            }
        });
        if delta > 0.0 {
            // Weighted forms avoid overflowing quantity*price intermediates.
            let total = before + delta;
            let weight = if total.is_finite() {
                delta / total
            } else {
                reasons.insert("numeric_range_approximation".into());
                (delta / 2.0) / (before / 2.0 + delta / 2.0)
            };
            basis = if p.inverse && before > 0.0 && basis > 0.0 {
                1.0 / ((1.0 - weight) / basis + weight / price)
            } else {
                (1.0 - weight) * basis + weight * price
            };
            basis = finite(basis, &mut reasons);
            if basis <= 0.0 {
                reasons.insert("numeric_range_approximation".into());
                basis = price;
            }
        }
        let gross = usable(f.realized).unwrap_or_else(|| {
            reasons.insert("estimated_realized_pnl".into());
            if delta < 0.0 {
                p.pnl(direction * before.min(-delta), basis, price)
            } else {
                0.0
            }
        });
        let fee = usable(f.fee).unwrap_or_else(|| {
            reasons.insert("unknown_fee".into());
            0.0
        });
        let gross = finite(gross, &mut reasons);
        cumulative.add(gross);
        cumulative.add(fee);
        if after == 0.0 {
            basis = 0.0;
        }
        events.push(Event {
            fill: f.clone(),
            before,
            after,
            basis,
            gross_realized: gross,
            fee,
            realized_cumsum: cumulative.value(&mut reasons),
            quantity_estimated,
        });
    }
    if p.size != 0.0 && basis != p.basis {
        reasons.insert("current_basis_reconciliation".into());
    }
    let mut prices = BTreeMap::new();
    for (&t, &price) in &input.prices {
        if t < input.start || t > input.end {
            continue;
        }
        if price.is_finite() && price > 0.0 {
            prices.insert(t, price);
        } else {
            reasons.insert("invalid_historical_price".into());
        }
    }
    prices.insert(input.end, p.mark);
    let mut samples = Vec::with_capacity(prices.len());
    let mut consumed = 0;
    for (timestamp, price) in prices {
        while consumed < events.len()
            && fill_precedes_price(
                events[consumed].fill.timestamp,
                timestamp,
                input.end,
                input.fills_before_same_time_price,
            )
        {
            consumed += 1;
        }
        let (mut size, mut basis, realized) = if consumed == 0 {
            (opening_quantity, opening_basis, 0.0)
        } else {
            let e = &events[consumed - 1];
            (e.after, e.basis, e.realized_cumsum)
        };
        if timestamp == input.end {
            size = p.size.abs();
            basis = p.basis;
        }
        samples.push(Sample {
            timestamp,
            pnl: realized,
            upnl: finite(p.pnl(direction * size, basis, price), &mut reasons),
            size: direction * size,
            basis,
        });
    }
    Ok(History {
        samples,
        events,
        reasons,
    })
}

#[pyfunction]
pub fn hsl_revised_history(input_json: &str) -> PyResult<String> {
    let input: Input =
        serde_json::from_str(input_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let output = reconstruct(&input).map_err(PyValueError::new_err)?;
    serde_json::to_string(&output).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_history_keeps_current_position_and_mark() {
        let input = Input {
            fills_before_same_time_price: true,
            start: 0,
            end: 60_000,
            position: Position {
                size: 2.0,
                basis: 100.0,
                mark: 80.0,
                multiplier: 1.0,
                quantity_step: None,
                inverse: false,
                pside: PositionSide::Long,
            },
            fills: vec![],
            prices: BTreeMap::new(),
        };
        let h = reconstruct(&input).unwrap();
        assert_eq!(h.samples.len(), 1);
        assert_eq!(h.samples[0].upnl, -40.0);
        assert_eq!(h.samples[0].basis, 100.0);
    }
}
