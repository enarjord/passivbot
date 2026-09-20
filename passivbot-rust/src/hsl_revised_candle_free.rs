//! All-candles-absent scope estimate. Known cashflows enrich the single current
//! observation; cashflow peaks are references, never invented past EMA samples.
use crate::hsl_revised::{signal, Observation, Signal};
use crate::hsl_revised_snapshot::{prepare, select, Input as Snapshot, Mode};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub snapshot: Snapshot,
    pub slots: u64,
    pub span: f64,
    pub threshold: f64,
}
#[derive(Debug, Serialize)]
pub struct Output {
    pub signal: Option<Signal>,
    pub realized: f64,
    pub realized_peak: f64,
    pub upnl: f64,
    pub reasons: BTreeSet<String>,
}
fn sum(values: &[f64], reasons: &mut BTreeSet<String>) -> f64 {
    let mut value = values.iter().sum::<f64>();
    if !value.is_finite() {
        reasons.insert("numeric_range_approximation".into());
        let scale = values.iter().map(|v| v.abs()).fold(0.0, f64::max);
        value = values.iter().map(|v| v / scale).sum::<f64>() * scale;
        if !value.is_finite() {
            value = f64::MAX.copysign(value);
        }
    }
    value
}
pub fn estimate(input: &Input) -> Result<Output, String> {
    let snapshot = &input.snapshot;
    let selected = select(snapshot)?;
    if !snapshot.balance.is_finite() || snapshot.balance <= 0.0 {
        return Err("invalid current balance".into());
    }
    if matches!(snapshot.mode, Mode::Coin) && input.slots == 0 {
        return Ok(Output {
            signal: None,
            realized: 0.0,
            realized_peak: 0.0,
            upnl: 0.0,
            reasons: BTreeSet::from(["inactive_scope".into()]),
        });
    }
    for pair in &selected {
        if pair.prices.iter().any(|(t, p)| {
            snapshot.start <= *t
                && *t <= snapshot.now.min(pair.prices_at)
                && p.is_finite()
                && *p > 0.0
        }) {
            return Err("candle-free estimate cannot discard historical prices".into());
        }
    }
    let prepared = prepare(snapshot)?;
    let mut reasons = prepared.reasons;
    reasons.insert("candle_free_reference".into());
    let budget = if matches!(snapshot.mode, Mode::Coin) {
        snapshot.balance / input.slots as f64
    } else {
        snapshot.balance
    };
    let mut timeline = Vec::new();
    for (pi, p) in prepared.pairs.iter().enumerate() {
        timeline.extend(
            p.history
                .events
                .iter()
                .enumerate()
                .map(|(fi, e)| (e.fill.timestamp, pi, fi)),
        );
    }
    timeline.sort_unstable();
    let mut cumulative = vec![0.0; prepared.pairs.len()];
    let mut realized = 0.0;
    let mut peak: f64 = 0.0;
    let mut cursor = 0;
    while cursor < timeline.len() {
        let t = timeline[cursor].0;
        let end = cursor + timeline[cursor..].partition_point(|v| v.0 == t);
        let cohort = &timeline[cursor..end];
        let one_pair = cohort.iter().all(|v| v.1 == cohort[0].1);
        let sequences: BTreeSet<_> = cohort
            .iter()
            .filter_map(|v| prepared.pairs[v.1].history.events[v.2].fill.sequence)
            .collect();
        let exact = one_pair && sequences.len() == cohort.len();
        let mut ordered = cohort.to_vec();
        if exact {
            ordered.sort_by_key(|v| prepared.pairs[v.1].history.events[v.2].fill.sequence);
        }
        let groups: Vec<&[(i64, usize, usize)]> = if exact {
            ordered.chunks(1).collect()
        } else {
            vec![&ordered]
        };
        if !exact && cohort.len() > 1 {
            reasons.insert("cohort_cashflow_peak".into());
        }
        for group in groups {
            for &(_, pi, fi) in group {
                cumulative[pi] = prepared.pairs[pi].history.events[fi].realized_cumsum;
            }
            realized = sum(&cumulative, &mut reasons);
            peak = peak.max(realized);
        }
        cursor = end;
    }
    let upnl = sum(
        &prepared
            .pairs
            .iter()
            .map(|p| p.history.samples.last().unwrap().upnl)
            .collect::<Vec<_>>(),
        &mut reasons,
    );
    let current = Observation {
        timestamp_ms: snapshot.now,
        realized,
        unrealized: upnl,
    };
    // Reuse the shared range-safe currency rebasing. Only its peak is retained;
    // this reference computation does not contribute an observation to the EMA.
    let reference = signal(
        &[
            Observation {
                timestamp_ms: snapshot.now,
                realized: peak,
                unrealized: 0.0,
            },
            current,
        ],
        budget,
        input.span,
        input.threshold,
        None,
    )?;
    let mut result = signal(
        &[current],
        budget,
        input.span,
        input.threshold,
        Some(reference.peaks[1]),
    )?;
    result.numeric_range_approximation |= reference.numeric_range_approximation;
    if result.numeric_range_approximation {
        reasons.insert("numeric_range_approximation".into());
    }
    Ok(Output {
        signal: Some(result),
        realized,
        realized_peak: peak,
        upnl,
        reasons,
    })
}
#[pyfunction]
pub fn hsl_revised_candle_free(input_json: &str) -> PyResult<String> {
    let input: Input =
        serde_json::from_str(input_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = estimate(&input).map_err(PyValueError::new_err)?;
    serde_json::to_string(&result).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overflowing_intermediate_sum_keeps_representable_residual() {
        let mut reasons = BTreeSet::new();
        assert_eq!(sum(&[1e308, 1e308, -1e308], &mut reasons), 1e308);
        assert!(reasons.contains("numeric_range_approximation"));
    }
}
