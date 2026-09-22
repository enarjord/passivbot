//! All-candles-absent scope estimate. Known cashflows enrich the single current
//! observation; cashflow peaks are references, never invented past EMA samples.
use crate::hsl_revised::{singleton_from_loss, validate_settings, Signal};
use crate::hsl_revised_snapshot::{prepare, select, Input as Snapshot, Mode};
use crate::hsl_revised_sum::{currency_sum as sum, CurrencySum};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub snapshot: Snapshot,
    pub slots: u64,
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
    pub span: f64,
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
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
pub fn estimate(input: &Input) -> Result<Output, String> {
    let snapshot = &input.snapshot;
    // Inactivity removes the budget division, not required-current-input validation.
    validate_settings(input.span, input.threshold)?;
    let prepared = prepare(snapshot)?;
    let selected = select(snapshot)?;
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
    let mut cumulative = CurrencySum::new();
    let mut realized = 0.0;
    let mut peak = CurrencySum::new();
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
                let event = &prepared.pairs[pi].history.events[fi];
                cumulative.add(event.gross_realized);
                cumulative.add(event.fee);
            }
            realized = cumulative.value(&mut reasons);
            if cumulative.difference(&peak, &mut reasons) > 0.0 {
                peak = cumulative.clone();
            }
        }
        cursor = end;
    }
    let upnl_terms: Vec<_> = prepared
        .pairs
        .iter()
        .map(|p| p.history.samples.last().unwrap().upnl)
        .collect();
    let upnl = sum(&upnl_terms, &mut reasons);
    // Subtract exact cashflow prefixes before rounding. A large realized baseline
    // must not erase a small loss since its peak, including after netting UPNL.
    let realized_peak = peak.value(&mut reasons);
    peak.subtract(&cumulative);
    for value in upnl_terms {
        peak.add(-value);
    }
    let loss = peak.value(&mut reasons);
    let mut result = singleton_from_loss(budget, upnl, loss, input.span, input.threshold)?;
    result.numeric_range_approximation |= reasons.contains("numeric_range_approximation");
    if result.numeric_range_approximation {
        reasons.insert("numeric_range_approximation".into());
    }
    Ok(Output {
        signal: Some(result),
        realized,
        realized_peak,
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
