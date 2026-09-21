//! Snapshot-to-permission composition shared by revised runtime adapters.
//! No prior decision, saved EMA, journal or cached permission is an input.
use crate::hsl_revised::validate_settings;
use crate::hsl_revised_controller::{self as controller, Decision, Intervention, Restart};
use crate::hsl_revised_snapshot::{self as snapshot, Input as Snapshot, Mode};
use crate::hsl_revised_trace::compose_with_cashflow_peaks;
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub snapshot: Snapshot,
    pub slots: u64,
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
    pub span: f64,
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
    pub threshold: f64,
    pub cooldown_ms: i64,
    pub restart: Restart,
    pub intervention: Intervention,
}

#[derive(Debug, Serialize)]
pub struct Output {
    pub decision: Option<Decision>,
    pub reasons: BTreeSet<String>,
    pub observations: usize,
    pub episodes: usize,
}

/// Normalize only selected pairs to one bounded minute grid. Source-resolution
/// projection is already Rust-owned; these observations retain that causal cut.
/// Whole-scope absence remains sparse: never make a fictitious flat minute tape.
fn normalize(snapshot: &mut Snapshot, reasons: &mut BTreeSet<String>) -> Result<bool, String> {
    let window = snapshot
        .now
        .checked_sub(snapshot.start)
        .filter(|w| (0..=90 * 86_400_000).contains(w))
        .ok_or("invalid revised HSL evaluation interval")?;
    let selected = snapshot::select(snapshot)?;
    let keys: BTreeSet<_> = selected
        .iter()
        .map(|p| {
            (
                p.symbol.clone(),
                p.position.pside == crate::hsl_revised_history::PositionSide::Long,
            )
        })
        .collect();
    let mut histories = Vec::new();
    for (index, pair) in snapshot.pairs.iter().enumerate() {
        if !keys.contains(&(
            pair.symbol.clone(),
            pair.position.pside == crate::hsl_revised_history::PositionSide::Long,
        )) {
            continue;
        }
        let prices: BTreeMap<_, _> = pair
            .prices
            .iter()
            .filter(|(t, p)| {
                snapshot.start <= **t
                    && **t <= snapshot.now.min(pair.prices_at)
                    && p.is_finite()
                    && **p > 0.0
            })
            .map(|(t, p)| (*t, *p))
            .collect();
        histories.push((index, prices));
    }
    let candle_free = histories.iter().all(|(_, p)| p.is_empty());
    if candle_free {
        reasons.insert("candle_free_reference".into());
        for (index, _) in histories {
            snapshot.pairs[index].prices.clear();
        }
        return Ok(true);
    }
    let mut grid = Vec::with_capacity((window / 60_000 + 2) as usize);
    let offset = (60_000 - snapshot.start.rem_euclid(60_000)) % 60_000;
    let mut next = snapshot.start.checked_add(offset);
    while let Some(t) = next.filter(|t| *t <= snapshot.now) {
        grid.push(t);
        next = t.checked_add(60_000);
    }
    // prepare() replaces this endpoint with observed current position and mark.
    if grid.last().copied() != Some(snapshot.now) {
        grid.push(snapshot.now);
    }
    for (index, prices) in histories {
        let pair = &mut snapshot.pairs[index];
        let mut aligned = BTreeMap::new();
        if let Some((_, first)) = prices.first_key_value() {
            for &t in &grid {
                let observed = prices.range(..=t).next_back();
                let value = observed.map_or(*first, |(_, p)| *p);
                if !prices.contains_key(&t) {
                    reasons.insert(
                        if observed.is_some() {
                            "forward_filled_price"
                        } else {
                            "backfilled_price"
                        }
                        .into(),
                    );
                }
                aligned.insert(t, value);
            }
        } else {
            // Mixed scope: preserve other pairs' history. Current mark is an
            // explicit estimator-only approximation for this absent price tape.
            reasons.insert("current_mark_history_estimate".into());
            aligned.extend(grid.iter().map(|t| (*t, pair.position.mark)));
        }
        pair.prices = aligned;
        // Rows have now been computed at evaluation time from causally eligible
        // observations. Original source skew remains in the collected reasons.
        pair.prices_at = snapshot.now;
    }
    Ok(false)
}

pub fn evaluate(mut input: Input) -> Result<Output, String> {
    validate_settings(input.span, input.threshold)?;
    input
        .snapshot
        .now
        .checked_sub(input.snapshot.start)
        .filter(|w| (0..=90 * 86_400_000).contains(w))
        .ok_or("invalid revised HSL evaluation interval")?;
    if input.cooldown_ms < 0 {
        return Err("negative revised HSL cooldown".into());
    }
    // Validate current observations before inactivity or price approximation.
    // Historical defects remain diagnostic, never an old readiness certificate.
    let mut reasons = snapshot::prepare(&input.snapshot)?.reasons;
    if matches!(input.snapshot.mode, Mode::Coin) && input.slots == 0 {
        reasons.insert("inactive_scope".into());
        return Ok(Output {
            decision: None,
            reasons,
            observations: 0,
            episodes: 0,
        });
    }
    let budget = if matches!(input.snapshot.mode, Mode::Coin) {
        input.snapshot.balance / input.slots as f64
    } else {
        input.snapshot.balance
    };
    let candle_free = normalize(&mut input.snapshot, &mut reasons)?;
    let trace = compose_with_cashflow_peaks(&input.snapshot, candle_free)?;
    reasons.extend(trace.reasons);
    let episodes = trace.episodes.len();
    let decisions = controller::replay(&controller::Input {
        episodes: trace.episodes,
        now: input.snapshot.now,
        start: input.snapshot.start,
        budget,
        span: input.span,
        threshold: input.threshold,
        cooldown_ms: input.cooldown_ms,
        restart: input.restart,
        intervention: input.intervention,
    })?;
    if decisions.iter().any(|d| d.numeric_range_approximation) {
        reasons.insert("numeric_range_approximation".into());
    }
    let observations = decisions.len();
    let decision = decisions
        .into_iter()
        .last()
        .ok_or("empty revised HSL evaluation")?;
    Ok(Output {
        decision: Some(decision),
        reasons,
        observations,
        episodes,
    })
}

#[pyfunction]
pub fn hsl_revised_evaluate(input_json: &str) -> PyResult<String> {
    let input =
        serde_json::from_str(input_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let output = evaluate(input).map_err(PyValueError::new_err)?;
    serde_json::to_string(&output).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn selected_sparse_prices_reach_current_permission() {
        let input: Input = serde_json::from_value(json!({
            "snapshot": {
                "now": 180000, "start": 0, "balance": 1000,
                "balance_at": 180000, "config_at": 0, "max_current_age_ms": 120000,
                "mode": "unified", "pairs": [{
                    "symbol": "A", "position": {"size": 10, "basis": 100,
                        "mark": 100, "multiplier": 1, "inverse": false, "pside": "long"},
                    "position_at": 180000, "mark_at": 180000,
                    "fills_started_at": 180000, "fills_at": 180000,
                    "prices_at": 180000, "fills": [], "prices": {"60000": 110},
                    "revisions": [0,0,0,0]
                }]
            },
            "slots": 1, "span": 1, "threshold": 0.05, "cooldown_ms": 0,
            "restart": "always", "intervention": "panic"
        }))
        .unwrap();
        let output = evaluate(input).unwrap();
        assert_eq!(output.observations, 4);
        assert!(output.reasons.contains("backfilled_price"));
        assert!(output.reasons.contains("forward_filled_price"));
        let decision = output.decision.unwrap();
        assert_eq!(decision.action, controller::Action::Panic);
        assert!((decision.raw - 100.0 / 1100.0).abs() < 1e-14);
    }
}
