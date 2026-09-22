//! Best-effort pair history, independent of legacy episode readiness.
//! The estimated path is authoritative for revised HSL only, never an exchange ledger repair.

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
    pub basis_before: f64,
    pub basis: f64,
    pub gross_realized: f64,
    pub fee: f64,
    pub realized_cumsum: f64,
    pub quantity_estimated: bool,
    pub reasons: BTreeSet<String>,
}

/// A current-position correction, not a fabricated execution. There is no
/// invented execution timestamp, price, fee or realized PnL.
#[derive(Debug, Serialize)]
pub struct Reconciliation {
    /// Estimated application time, not an exchange execution timestamp.
    pub applied_at: i64,
    pub before: f64,
    pub after: f64,
    pub basis_before: f64,
    pub basis_after: f64,
    pub delta: f64,
}

#[derive(Debug, Serialize)]
pub struct History {
    pub opening_size: f64,
    pub opening_basis: f64,
    pub reconciliation: Option<Reconciliation>,
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

/// Suppress binary lot roundoff only below half a known exchange quantum.
/// Missing precision remains visible; it is not a lifecycle veto.
fn round_quantity(
    value: f64,
    scale: f64,
    step: Option<f64>,
    reasons: &mut BTreeSet<String>,
) -> f64 {
    let tolerance = 8.0 * f64::EPSILON * scale;
    if value != 0.0 && value.abs() <= tolerance && step.is_none_or(|q| value.abs() < q / 2.0) {
        reasons.insert("quantity_roundoff".into());
        if step.is_none() {
            reasons.insert("quantity_precision_unavailable".into());
        }
        0.0
    } else {
        value
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
    // Minimum feasible inventory at the window's opening. Known quantities are
    // preserved, rather than clamping a reverse walk into impossible transitions.
    // An unexplained larger endpoint is carried from the beginning; we do not
    // assert a recent missing add or erase earlier losses without evidence.
    let mut prefix = CurrencySum::new();
    let mut minimum: f64 = 0.0;
    let mut scale = p.size.abs();
    for f in &fills {
        let delta = usable(f.delta).unwrap_or(0.0) * direction;
        prefix.add(delta);
        let value = prefix.value(&mut reasons);
        scale = scale.max(delta.abs()).max(value.abs());
        minimum = minimum.min(value);
    }
    let mut opening_difference = CurrencySum::new();
    opening_difference.add(p.size.abs());
    opening_difference.subtract(&prefix);
    let mut after = (-minimum)
        .max(opening_difference.value(&mut reasons))
        .max(0.0);
    // Without a quantity quantum, a huge later lot must not round away a real
    // small opening residual. Use the first transition's scale for this estimate.
    let opening_scale = if p.quantity_step.is_some() {
        scale
    } else {
        fills
            .first()
            .and_then(|f| usable(f.delta))
            .map_or(after, f64::abs)
            .max(after)
    };
    after = round_quantity(after, opening_scale, p.quantity_step, &mut reasons);
    if after > 0.0 {
        reasons.insert("estimated_opening_quantity".into());
    }
    let opening_quantity = after;
    let mut inventory = CurrencySum::new();
    inventory.add(after);
    let mut steps = Vec::with_capacity(fills.len());
    let mut episode_scale = after;
    for f in &fills {
        let before = after;
        let delta = usable(f.delta).unwrap_or(0.0) * direction;
        inventory.add(delta);
        episode_scale = episode_scale.max(before).max(delta.abs());
        let mut event_reasons = BTreeSet::new();
        after = round_quantity(
            inventory.value(&mut event_reasons),
            episode_scale,
            p.quantity_step,
            &mut event_reasons,
        )
        .max(0.0);
        if usable(f.delta).is_none_or(|q| q == 0.0) {
            event_reasons.insert("invalid_quantity".into());
        }
        let quantity_estimated = !event_reasons.is_empty();
        steps.push((f, before, after, delta, quantity_estimated, event_reasons));
        if after == 0.0 {
            inventory = CurrencySum::new();
            episode_scale = 0.0;
        }
    }
    let mut basis = fills
        .iter()
        .find_map(|f| positive(f.price))
        .unwrap_or(if p.basis > 0.0 { p.basis } else { p.mark });
    if opening_quantity != 0.0 {
        reasons.insert("estimated_opening_basis".into());
    } else {
        basis = 0.0;
    }
    let opening_basis = basis;
    let mut cumulative = CurrencySum::new();
    let mut events = Vec::with_capacity(steps.len());
    for (f, before, after, delta, quantity_estimated, mut event_reasons) in steps {
        let basis_before = basis;
        let price = positive(f.price).unwrap_or_else(|| {
            event_reasons.insert("estimated_fill_price".into());
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
                event_reasons.insert("numeric_range_approximation".into());
                (delta / 2.0) / (before / 2.0 + delta / 2.0)
            };
            basis = if p.inverse && before > 0.0 && basis > 0.0 {
                1.0 / ((1.0 - weight) / basis + weight / price)
            } else {
                (1.0 - weight) * basis + weight * price
            };
            basis = finite(basis, &mut event_reasons);
            if basis <= 0.0 {
                event_reasons.insert("numeric_range_approximation".into());
                basis = price;
            }
        }
        let gross = usable(f.realized).unwrap_or_else(|| {
            event_reasons.insert("estimated_realized_pnl".into());
            if delta < 0.0 {
                p.pnl(direction * before.min(-delta), basis, price)
            } else {
                0.0
            }
        });
        let fee = usable(f.fee).unwrap_or_else(|| {
            event_reasons.insert("unknown_fee".into());
            0.0
        });
        let gross = finite(gross, &mut event_reasons);
        cumulative.add(gross);
        cumulative.add(fee);
        if after == 0.0 {
            basis = 0.0;
        }
        reasons.extend(event_reasons.iter().cloned());
        events.push(Event {
            fill: f.clone(),
            before,
            after,
            basis_before,
            basis,
            gross_realized: gross,
            fee,
            realized_cumsum: cumulative.value(&mut reasons),
            quantity_estimated,
            reasons: event_reasons,
        });
    }
    if p.size != 0.0 && basis != p.basis {
        reasons.insert("current_basis_reconciliation".into());
    }
    let tail = events.last().map_or(opening_quantity, |e| e.after);
    let delta = round_quantity(
        p.size.abs() - tail,
        episode_scale,
        p.quantity_step,
        &mut reasons,
    );
    let reconciliation = if delta != 0.0 || (p.size != 0.0 && basis != p.basis) {
        if delta != 0.0 {
            reasons.insert("current_quantity_reconciliation".into());
        }
        let applied_at = if p.size == 0.0 && delta != 0.0 && !events.is_empty() {
            reasons.insert("current_flat_timestamp_estimate".into());
            events.last().unwrap().fill.timestamp
        } else {
            input.end
        };
        Some(Reconciliation {
            applied_at,
            before: direction * tail,
            after: p.size,
            basis_before: basis,
            basis_after: p.basis,
            delta: direction * delta,
        })
    } else {
        None
    };
    // Input keys are already ordered and unique. Preserve that order without
    // allocating a second search tree for a sequential reconstruction pass.
    let mut prices = Vec::with_capacity(input.prices.len() + 1);
    for (&t, &price) in &input.prices {
        if t < input.start || t > input.end {
            continue;
        }
        if price.is_finite() && price > 0.0 {
            prices.push((t, price));
        } else {
            reasons.insert("invalid_historical_price".into());
        }
    }
    if prices.last().is_some_and(|(t, _)| *t == input.end) {
        prices.last_mut().unwrap().1 = p.mark;
    } else {
        prices.push((input.end, p.mark));
    }
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
        if timestamp == input.end
            || reconciliation.as_ref().is_some_and(|r| {
                p.size == 0.0
                    && fill_precedes_price(
                        r.applied_at,
                        timestamp,
                        input.end,
                        input.fills_before_same_time_price,
                    )
            })
        {
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
        opening_size: direction * opening_quantity,
        opening_basis,
        reconciliation,
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
