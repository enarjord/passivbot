//! Compose prepared, aligned scope observations into the stateless controller trace.
//! Price normalization belongs to the caller. Missing initial exposure seeds one
//! scoped entry-value peak, never an invented price or EMA observation.
use crate::hsl_revised_controller::{Episode, Point};
use crate::hsl_revised_snapshot::{prepare, Input, Output as Prepared};
use crate::hsl_revised_sum::CurrencySum;
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Debug, Serialize)]
pub struct Trace {
    pub episodes: Vec<Episode>,
    pub reasons: BTreeSet<String>,
}

fn centered(cashflows: &CurrencySum, anchor: &CurrencySum, reasons: &mut BTreeSet<String>) -> f64 {
    cashflows.difference(anchor, reasons)
}

fn consume(
    prepared: &Prepared,
    counts: &mut [usize],
    targets: &[usize],
    cashflows: &mut CurrencySum,
    peak: &mut CurrencySum,
    track_peak: bool,
    global_sequence: bool,
    reasons: &mut BTreeSet<String>,
) -> Option<i64> {
    let mut opening = None;
    let mut timeline = Vec::new();
    for (index, target) in targets.iter().copied().enumerate() {
        for event in &prepared.pairs[index].history.events[counts[index]..target] {
            if !event.quantity_estimated && event.before == 0.0 && event.after > 0.0 {
                opening = Some(
                    opening.map_or(event.fill.timestamp, |t: i64| t.min(event.fill.timestamp)),
                );
            }
            if track_peak {
                timeline.push((index, event));
            } else {
                cashflows.add(event.gross_realized);
                cashflows.add(event.fee);
            }
        }
        counts[index] = target;
    }
    timeline.sort_by_key(|(_, e)| e.fill.timestamp);
    let mut cursor = 0;
    while cursor < timeline.len() {
        let end = cursor
            + timeline[cursor..]
                .partition_point(|(_, e)| e.fill.timestamp == timeline[cursor].1.fill.timestamp);
        let cohort = &mut timeline[cursor..end];
        let one_pair = cohort.iter().all(|(i, _)| *i == cohort[0].0);
        let sequences: BTreeSet<_> = cohort.iter().filter_map(|(_, e)| e.fill.sequence).collect();
        let exact = (one_pair || global_sequence) && sequences.len() == cohort.len();
        if exact {
            cohort.sort_by_key(|(_, e)| e.fill.sequence);
        } else if cohort.len() > 1 {
            reasons.insert("cohort_cashflow_peak".into());
        }
        for (index, (_, event)) in cohort.iter().enumerate() {
            cashflows.add(event.gross_realized);
            cashflows.add(event.fee);
            if (exact || index + 1 == cohort.len()) && cashflows.difference(peak, reasons) > 0.0 {
                *peak = cashflows.clone();
            }
        }
        cursor = end;
    }
    opening
}

/// The input price grids must already have identical timestamps for selected
/// pairs, after causal clipping. A mismatch is a composer input-contract error,
/// not a decision that a historical outage should block protection. The later
/// snapshot dispatcher must normalize prices and choose the approved estimator.
pub fn compose(input: &Input) -> Result<Trace, String> {
    compose_with_cashflow_peaks(input, false)
}

pub(crate) fn compose_with_cashflow_peaks(
    input: &Input,
    candle_free: bool,
) -> Result<Trace, String> {
    let prepared = prepare(input)?;
    let mut reasons = prepared.reasons.clone();
    if prepared.pairs.is_empty() {
        return Ok(Trace {
            episodes: vec![Episode {
                points: vec![Point {
                    timestamp: input.now,
                    pnl: 0.0,
                    upnl: 0.0,
                    exposed: false,
                    flatten: false,
                    cashflow_reference_delta: None,
                }],
                entry_reference: None,
                entry_reference_delta: None,
                opened_at: None,
            }],
            reasons,
        });
    }
    let timestamps: Vec<_> = prepared.pairs[0]
        .history
        .samples
        .iter()
        .map(|s| s.timestamp)
        .collect();
    for pair in &prepared.pairs {
        if !pair
            .history
            .samples
            .iter()
            .map(|s| s.timestamp)
            .eq(timestamps.iter().copied())
        {
            return Err("revised HSL trace requires aligned prepared price grids".into());
        }
    }
    // Center every realized cashflow prefix against the common current prefix
    // before float readout. An enormous common baseline must not erase a fee.
    let mut anchor = CurrencySum::new();
    for pair in &prepared.pairs {
        for event in &pair.history.events {
            anchor.add(event.gross_realized);
            anchor.add(event.fee);
        }
    }
    // The reconstructed prefix before retained fills has zero realized cashflow
    // and zero UPNL at its estimated entry value. Rebase that one scope value
    // against the same current endpoint as every ordinary observation. Never
    // add independent per-pair peaks or discard available candle observations.
    let mut entry_reference_delta = if prepared
        .pairs
        .iter()
        .any(|p| p.history.reasons.contains("estimated_opening_basis"))
    {
        let mut reference = CurrencySum::new();
        reference.subtract(&anchor);
        for pair in &prepared.pairs {
            reference.add(-pair.history.samples.last().unwrap().upnl);
        }
        reasons.insert("estimated_entry_peak".into());
        Some(reference.value(&mut reasons))
    } else {
        None
    };
    let mut cashflows = CurrencySum::new();
    let mut peak = CurrencySum::new();
    let mut endpoint = anchor.clone();
    for pair in &prepared.pairs {
        endpoint.add(pair.history.samples.last().unwrap().upnl);
    }
    let mut counts = vec![0; prepared.pairs.len()];
    let mut episodes = Vec::new();
    let mut points = Vec::new();
    let mut opened_at = None;
    let mut boundaries = prepared
        .boundaries
        .iter()
        .filter(|b| b.lifecycle_eligible)
        .peekable();
    for (sample_index, &timestamp) in timestamps.iter().enumerate() {
        // A flatten's exact consumed prefix precedes any same-time reopen.
        // A producer's explicit pre-fill candle phase stays before that prefix.
        // Distinct proven flats in one timestamp remain distinct.
        while boundaries.peek().is_some_and(|b| {
            crate::hsl_revised_history::fill_precedes_price(
                b.timestamp,
                timestamp,
                input.now,
                input.fills_before_same_time_price,
            )
        }) {
            let boundary = boundaries.next().unwrap();
            let targets: Vec<_> = boundary.consumed.iter().map(|c| c.count).collect();
            let opening = consume(
                &prepared,
                &mut counts,
                &targets,
                &mut cashflows,
                &mut peak,
                candle_free,
                input.global_fill_sequence,
                &mut reasons,
            );
            if !episodes.is_empty() {
                opened_at = opened_at.or(opening);
            }
            points.push(Point {
                timestamp: boundary.timestamp,
                pnl: centered(&cashflows, &anchor, &mut reasons),
                upnl: 0.0,
                exposed: false,
                flatten: true,
                cashflow_reference_delta: candle_free
                    .then(|| peak.difference(&endpoint, &mut reasons)),
            });
            episodes.push(Episode {
                points: std::mem::take(&mut points),
                entry_reference: None,
                entry_reference_delta: entry_reference_delta.take(),
                opened_at: opened_at.take(),
            });
            // The actual flat prefix also seeds the next interval. Starting only
            // after a reopen would erase its opening fee from the new peak.
            points.push(Point {
                timestamp: boundary.timestamp,
                pnl: centered(&cashflows, &anchor, &mut reasons),
                upnl: 0.0,
                exposed: false,
                flatten: false,
                cashflow_reference_delta: None,
            });
            peak = cashflows.clone();
        }
        let targets: Vec<_> = prepared
            .pairs
            .iter()
            .map(|p| {
                p.history.events.partition_point(|e| {
                    crate::hsl_revised_history::fill_precedes_price(
                        e.fill.timestamp,
                        timestamp,
                        input.now,
                        input.fills_before_same_time_price,
                    )
                })
            })
            .collect();
        let opening = consume(
            &prepared,
            &mut counts,
            &targets,
            &mut cashflows,
            &mut peak,
            candle_free,
            input.global_fill_sequence,
            &mut reasons,
        );
        if !episodes.is_empty() {
            opened_at = opened_at.or(opening);
        }
        let mut upnl = CurrencySum::new();
        let mut exposed = false;
        for pair in &prepared.pairs {
            let sample = &pair.history.samples[sample_index];
            upnl.add(sample.upnl);
            exposed |= sample.size != 0.0;
        }
        points.push(Point {
            timestamp,
            pnl: centered(&cashflows, &anchor, &mut reasons),
            upnl: upnl.value(&mut reasons),
            exposed,
            flatten: false,
            cashflow_reference_delta: candle_free.then(|| peak.difference(&endpoint, &mut reasons)),
        });
    }
    if !points.is_empty() {
        episodes.push(Episode {
            points,
            entry_reference: None,
            entry_reference_delta,
            opened_at,
        });
    }
    Ok(Trace { episodes, reasons })
}

#[pyfunction]
pub fn hsl_revised_trace(input_json: &str) -> PyResult<String> {
    let input =
        serde_json::from_str(input_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = compose(&input).map_err(PyValueError::new_err)?;
    serde_json::to_string(&result).map_err(|e| PyValueError::new_err(e.to_string()))
}
